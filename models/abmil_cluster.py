import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from einops import rearrange, repeat
from conch.open_clip_custom import create_model_from_pretrained, tokenize, get_tokenizer
from models.infonce import InfoNCE

def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            # ref from huggingface
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m,nn.Linear):
            # ref from clam
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m,nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)



class DAttention(nn.Module):
    def __init__(self, input_dim=512, n_classes=2, dropout=0.25, act='relu', 
                 concept_path=None,device='cuda'):
        super(DAttention, self).__init__()
        
        # --- PHẦN 1: CODE GỐC (GIỮ NGUYÊN) ---
        self.L = 512 
        self.D = 128 
        self.K = 1
        
        self.feature = [nn.Linear(input_dim, 512)]
        if act.lower() == 'gelu':
            self.feature += [nn.GELU()]
        else:
            self.feature += [nn.ReLU()]

        if dropout:
            self.feature += [nn.Dropout(0.25)]
        self.feature = nn.Sequential(*self.feature)

        self.linear = nn.Sequential(
            nn.Linear(self.L + 256, self.L),
            nn.ReLU(),
            nn.Linear(self.L, self.L),
            nn.ReLU(),
        )

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(self.L * self.K, n_classes),
        )

        # Khởi tạo weights cho mạng chính -> Trạng thái Random chuẩn đạt 94.21%
        self.apply(initialize_weights)

        # ===========================================================
        # === [CRITICAL FIX] RNG STATE MANAGEMENT ===
        # Lưu lại trạng thái Random hiện tại (của CPU và GPU)
        # Để đảm bảo việc init module mới KHÔNG làm trôi dòng Random của Dropout/DataLoader sau này
        rng_state = torch.get_rng_state()
        if torch.cuda.is_available():
            cuda_rng_state = torch.cuda.get_rng_state()
        # ===========================================================

        # --- PHẦN 2: MODULE MỚI (Làm gì ở đây cũng được) ---
        self.n_classes = n_classes
        self.concepts = None
        self.sim_projector = None 
        self.device = device
        self.info_nce_loss = InfoNCE(temperature=0.1, reduction='mean', negative_mode='unpaired')

        if concept_path is not None:
            concept_dim = self._load_concepts_and_get_dim(concept_path)
            
            # Việc khai báo nn.Linear() và init weights sẽ tiêu tốn Random numbers
            self.sim_projector = nn.Sequential(
                nn.Linear(self.L, self.L),      
                nn.ReLU(),
                nn.Linear(self.L, concept_dim)  
            )
            self.sim_projector.apply(initialize_weights) 
            print(f"Initialized Projection Head: {self.L} -> {concept_dim}")

        # ===========================================================
        # === [CRITICAL FIX] RESTORE RNG STATE ===
        # Trả lại trạng thái Random như lúc vừa init xong mạng chính
        torch.set_rng_state(rng_state)
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(cuda_rng_state)
        # ===========================================================
        
    def reparameterise(self, mu, logvar):
        epsilon = torch.randn_like(mu)
        return mu + epsilon * torch.exp(logvar / 2)

    def kl_loss(self, mu, logvar):
        kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        kl_mean = torch.mean(kl_div)
        return kl_mean

    def _load_concepts_and_get_dim(self, path):
        """
        Load concepts từ file, lưu vào self.concepts và trả về dimension
        để khởi tạo Projection Head cho khớp.
        """
        try:
            print(f"Loading concepts from: {path}")
            # Load raw data (thường là List các Tensor hoặc 1 Tensor lớn)
            raw_concepts = torch.load(path, map_location=self.device) 
            
            # --- BƯỚC 1: Lấy Concept Dimension ---
            if isinstance(raw_concepts, list):
                # Nếu là list [Tensor_Class0, Tensor_Class1...]
                sample_feat = raw_concepts[0]
            else:
                # Nếu là Tensor (C, D) hoặc (C, K, D)
                sample_feat = raw_concepts
            
            # Dimension là chiều cuối cùng (ví dụ 768)
            concept_dim = sample_feat.shape[-1]

            # --- BƯỚC 2: Xử lý và lưu vào self.concepts ---
            # Chỉ lấy đủ số lượng class (tránh lỗi nếu file concept có nhiều class hơn dataset)
            if isinstance(raw_concepts, list):
                raw_concepts = raw_concepts[:self.n_classes]
            
            processed_concepts = []
            for c in raw_concepts:
                # Đảm bảo tensor nằm trên đúng device (cuda/cpu)
                if isinstance(c, torch.Tensor):
                    c = c.to(self.device)
                
                # Xử lý trường hợp 1 class có nhiều vector mẫu (Shape: [K, D])
                # Ta lấy trung bình để ra 1 vector đại diện duy nhất (Prototype) -> Shape: [D]
                if c.dim() > 1:
                    c = c.mean(dim=0) # Mean pooling
                
                processed_concepts.append(c)
            
            # Stack lại thành 1 Tensor duy nhất: (N_classes, Concept_Dim)
            # .detach() cực kỳ quan trọng: Concept là hằng số (Neo), không được update gradient
            self.concepts = torch.stack(processed_concepts, dim=0).detach()
            
            print(f"Concepts loaded successfully. Shape: {self.concepts.shape}. Dim: {concept_dim}")
            
            return concept_dim

        except Exception as e:
            print(f"WARNING: Could not load concepts from {path}.")
            print(f"Error details: {e}")
            print("Using default dimension 512 (fallback) and disabling Sim Loss.")
            
            self.concepts = None
            # Trả về 512 (hoặc giá trị input_dim) để code không bị crash khi init layer
            # Tuy nhiên sim_loss sẽ không chạy do self.concepts = None
            return 512

    def forward(self, x, label=None, lambda_sym=0.0, lambda_cross=1.0, topk_pos=10, topk_neg=10):
        x = x.unsqueeze(0)
        results_dict = {}

        feature = self.feature(x)

        feature = feature.squeeze(0)
        A = self.attention(feature)
        A = torch.transpose(A, -1, -2)  # KxN
        A = F.softmax(A, dim=-1)  # softmax over N
        M = torch.mm(A, feature)  # KxL
        x = self.classifier(M)

        Y_hat = torch.argmax(x)
        Y_prob = F.softmax(x)

        # --- SỬA LẠI DÒNG IF NÀY ---
        # Thêm điều kiện: and self.concepts is not None
        if self.training and label is not None and self.sim_projector is not None and self.concepts is not None:
            feature_proj = self.sim_projector(feature)
            loss_sim = self._calculate_sim_loss(feature_proj, label, lambda_sym, lambda_cross, topk_pos, topk_neg)
            results_dict['sim_loss'] = loss_sim
        else:
            results_dict['sim_loss'] = torch.tensor(0.0, device=feature.device)

        return x, Y_prob, Y_hat, results_dict, feature

    def _calculate_sim_loss(self, X, label, lambda_sym, lambda_cross, topk_pos, topk_neg):
        T = self.concepts
        N, D = X.shape
        C_T = T.shape[0]
        y = int(label.item())
        
        if N == 0 or C_T < self.n_classes:
            return torch.tensor(0.0, device=X.device)

        # Dimension Check (Debug nếu cần)
        # print(f"X shape: {X.shape}, T shape: {T.shape}") 
        # X: (N, 768), T: (C, 768) -> OK

        X_norm = F.normalize(X, dim=-1)
        T_norm = F.normalize(T, dim=-1)
        # X_norm = X
        # T_norm = T
        
        # Matrix Multiplication: (N, D) @ (D, C) -> (N, C)
        S = X_norm @ T_norm.t() 

        # ... (Phần còn lại giữ nguyên như cũ) ...
        k_pos = max(1, min(topk_pos, N))
        k_neg = max(1, min(topk_neg, N))

        pos_idx = torch.topk(S, k_pos, dim=0, largest=True).indices
        neg_idx = torch.topk(S, k_neg, dim=0, largest=False).indices

        Ly = torch.tensor(0.0, device=X.device)
        q_y = None 

        if 0 <= y < C_T:
            idx_pos_y = pos_idx[:, y].unique() 
            idx_neg_y = neg_idx[:, y].unique() 

            if idx_pos_y.numel() > 0 and idx_neg_y.numel() > 0:
                q_y = X[idx_pos_y]      
                neg_y = X[idx_neg_y]    
                t_y = T[y].unsqueeze(0).expand(q_y.size(0), -1) 
                Ly = self.info_nce_loss(q_y, t_y, neg_y)

        L_sym = torch.tensor(0.0, device=X.device)
        if lambda_sym > 0.0 and self.n_classes > 1:
            Lc_list = []
            for c in range(self.n_classes):
                if c == y: continue
                idx_pos_c = pos_idx[:, c].unique()
                idx_neg_c = neg_idx[:, c].unique()
                if idx_pos_c.numel() > 0 and idx_neg_c.numel() > 0:
                    q_c = X[idx_pos_c]
                    neg_c = X[idx_neg_c]
                    t_c = T[c].unsqueeze(0).expand(q_c.size(0), -1)
                    Lc_list.append(self.info_nce_loss(q_c, t_c, neg_c))
            if len(Lc_list) > 0:
                L_sym = torch.stack(Lc_list).mean()

        L_cross = torch.tensor(0.0, device=X.device)
        if lambda_cross > 0.0 and q_y is not None and self.n_classes > 1:
            L_yc_list = []
            for c in range(self.n_classes):
                if c == y: continue
                idx_pos_c = pos_idx[:, c].unique()
                idx_neg_c = neg_idx[:, c].unique()
                if idx_pos_c.numel() > 0 and idx_neg_c.numel() > 0:
                    pos_c = X[idx_pos_c]
                    neg_c = X[idx_neg_c] 
                    m_neg_c = neg_c.mean(dim=0, keepdim=True)
                    p_yc = m_neg_c.expand(q_y.size(0), -1)
                    L_yc_list.append(self.info_nce_loss(q_y, p_yc, pos_c))
            if len(L_yc_list) > 0:
                L_cross = torch.stack(L_yc_list).mean()

        return Ly + lambda_sym * L_sym + lambda_cross * L_cross
    