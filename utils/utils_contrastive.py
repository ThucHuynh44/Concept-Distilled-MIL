import torch
import torch.nn.functional as F
from torch import nn
from typing import Dict, Any, Tuple, List, Optional


# ==============================
# Helpers & Loss Intra-class tính ở ngoài
# ==============================
def load_concepts(path: str, device=None, dtype=None) -> torch.Tensor:
    obj = torch.load(path, map_location=device or "cpu")
    if isinstance(obj, torch.Tensor):
        T = obj
    elif isinstance(obj, dict):
        for k in ["T", "text_embeds", "text_features", "embeddings", "feats", "prompt_feats"]:
            if k in obj and isinstance(obj[k], torch.Tensor):
                T = obj[k]; break
        else:
            raise KeyError("Không tìm thấy tensor embeddings trong file .pt.")
    else:
        raise TypeError("File không phải Tensor hoặc dict chứa Tensor.")
    if dtype is not None: T = T.to(dtype=dtype)
    if device is not None: T = T.to(device)
    return T

@torch.no_grad()
def _sim_matrix(X: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    Xn = F.normalize(X, dim=-1)
    Tn = F.normalize(T, dim=-1)
    return Xn @ Tn.t()  # (N, C)

def _topk_pos_neg_indices(s_col: torch.Tensor, k: int):
    pos = torch.topk(s_col, k=k, largest=True).indices
    neg = torch.topk(s_col, k=k, largest=False).indices
    return pos, neg

def intra_class_infonce_loss(
    X: torch.Tensor,           # (N, D_X) input patch features
    T: torch.Tensor,           # (C, D_T) concept/text embeddings
    y: int | torch.Tensor,     # label đúng của WSI
    *,
    k: int = 8,
    lambda_sym: float = 0.2,
    temperature: float = 0.1,
    reduction: str = "mean",
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    assert X.dim() == 2 and T.dim() == 2, "X (N,Dx), T (C,Dt)"
    N, Dx = X.shape
    C, Dt = T.shape
    if Dx != Dt:
        raise ValueError(f"Dim mismatch: X.D={Dx} vs T.D={Dt}. Hãy thêm projection cho X trước khi tính loss.")
    if isinstance(y, torch.Tensor): y = int(y.item())
    if N < 2: raise ValueError("Cần ít nhất 2 patches để tạo pos/neg.")

    k_eff = max(1, min(k, N // 2))
    S = _sim_matrix(X, T)  # (N, C)

    pos_idx: List[torch.Tensor] = []
    neg_idx: List[torch.Tensor] = []
    for c in range(C):
        p, n = _topk_pos_neg_indices(S[:, c], k=k_eff)
        pos_idx.append(p); neg_idx.append(n)

    def _class_infonce(c: int) -> torch.Tensor:
        q = X[pos_idx[c]]                          # (k_eff, Dx)
        p = T[c].unsqueeze(0).expand_as(q)         # (k_eff, Dx)
        n = X[neg_idx[c]]                          # (k_eff, Dx)
        return info_nce(q, p, n, temperature=temperature, reduction=reduction, negative_mode="unpaired")

    loss_y = _class_infonce(y)
    loss_sym = X.new_tensor(0.0)
    if C > 1 and lambda_sym > 0:
        others = [ _class_infonce(c) for c in range(C) if c != y ]
        if len(others): loss_sym = torch.stack(others).mean()
    loss = loss_y + lambda_sym * loss_sym

    details = {"S": S, "k_eff": k_eff, "pos_idx": pos_idx, "neg_idx": neg_idx,
               "loss_y": loss_y.detach(), "loss_sym": loss_sym.detach()}
    return loss, details