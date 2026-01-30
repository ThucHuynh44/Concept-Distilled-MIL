import numpy as np
import torch
from utils.utils import *
import os
from dataset_modules.dataset_generic import save_splits
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import auc as calc_auc
import torch.nn.functional as F
from models.infonce import InfoNCE


device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Accuracy_Logger(object):
    """Accuracy logger"""
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
    
    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)
    
    def log_batch(self, Y_hat, Y):
        Y_hat = np.array(Y_hat).astype(int)
        Y = np.array(Y).astype(int)
        for label_class in np.unique(Y):
            cls_mask = Y == label_class
            self.data[label_class]["count"] += cls_mask.sum()
            self.data[label_class]["correct"] += (Y_hat[cls_mask] == Y[cls_mask]).sum()
    
    def get_summary(self, c):
        count = self.data[c]["count"] 
        correct = self.data[c]["correct"]
        
        if count == 0: 
            acc = None
        else:
            acc = float(correct) / count
        
        return acc, correct, count

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, stop_epoch=50, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss

def _safe_add_scalar(writer, tag, value, step):
    import numpy as np, torch, math
    if value is None:
        print(f"[TBX] Skip {tag}: value=None (no samples)")
        return
    if torch.is_tensor(value):
        if value.numel() == 0:
            print(f"[TBX] Skip {tag}: empty tensor")
            return
        value = value.detach().float().mean().item()
    else:
        try:
            value = float(np.asarray(value).mean())
        except Exception as e:
            print(f"[TBX] Skip {tag}: cannot convert ({e})")
            return
    if math.isnan(value):
        print(f"[TBX] Skip {tag}: NaN")
        return
    writer.add_scalar(tag, value, step)


def train(datasets, cur, args):
    """   
        train for a single fold
    """
    print('\nTraining Fold {}!'.format(cur))
    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    if args.log_data:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=15)

    else:
        writer = None

    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split, test_split = datasets
    save_splits(datasets, ['train', 'val', 'test'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))
    print("Testing on {} samples".format(len(test_split)))

    print('\nInit loss function...', end=' ')
    if args.bag_loss == 'svm':
        from topk.svm import SmoothTop1SVM
        loss_fn = SmoothTop1SVM(n_classes = args.n_classes)
        if device.type == 'cuda':
            loss_fn = loss_fn.cuda()
    else:
        loss_fn = nn.CrossEntropyLoss()
    print('Done!')
    
    print('\nInit Model...', end=' ')
    model_dict = {"dropout": args.drop_out, 
                  'n_classes': args.n_classes, 
                  "embed_dim": args.embed_dim}
    
    if args.model_size is not None and args.model_type != 'mil':
        model_dict.update({"size_arg": args.model_size})
    
    if args.model_type in ['clam_sb', 'clam_mb','cate_abmil','cate_dl',"abmil",'mean_mil','max_mil','trans_mil','mamba_mil','dsmil']:
        if args.subtyping:
            model_dict.update({'subtyping': True})
        
        if args.B > 0:
            model_dict.update({'k_sample': args.B})
        
        if args.inst_loss == 'svm' and args.model_type in ['clam_sb', 'clam_mb']:
            from topk.svm import SmoothTop1SVM
            instance_loss_fn = SmoothTop1SVM(n_classes = 2)
            if device.type == 'cuda':
                instance_loss_fn = instance_loss_fn.cuda()
        else:
            instance_loss_fn = nn.CrossEntropyLoss()
        
        model_dictionary = {
                "input_dim": args.embed_dim,
                "n_classes": args.n_classes,
                "dropout": args.drop_out}

        if args.model_type == 'mean_mil':
            from models.Mean_Max_MIL import MeanMIL
            model = MeanMIL(args.embed_dim, args.n_classes)
        
        elif args.model_type == 'dsmil':
            from models.dsmil import FCLayer , IClassifier, BClassifier,MILNet
            i_classifier = FCLayer(in_size=args.embed_dim, out_size=args.n_classes)
            b_classifier = BClassifier(input_size=args.embed_dim, output_class=args.n_classes, dropout_v=0.0)
            model = MILNet(i_classifier, b_classifier)

        elif args.model_type == 'max_mil':
            from models.Mean_Max_MIL import MaxMIL
            model = MaxMIL(args.embed_dim, args.n_classes)
        
        elif args.model_type == 'trans_mil':
            from models.TransMIL import TransMIL
            model = TransMIL(args.embed_dim, args.n_classes, dropout=args.drop_out, act='relu')
        
        elif args.model_type == 'mamba_mil':
            from models.MambaMIL import MambaMIL
            model = MambaMIL(in_dim = args.embed_dim, n_classes=args.n_classes, dropout=args.drop_out, act='gelu', layer = args.mambamil_layer, rate = args.mambamil_rate, type = args.mambamil_type)
        
        elif args.model_type =='clam_sb':
            from models.model_clam import CLAM_MB, CLAM_SB
            model = CLAM_SB(**model_dict, instance_loss_fn=instance_loss_fn)

        elif args.model_type == 'clam_mb':
            from models.model_clam import CLAM_MB, CLAM_SB
            model = CLAM_MB(**model_dict, instance_loss_fn=instance_loss_fn)

        elif args.model_type == 'cate_abmil':
            from models.catemil import DAttention as DAttention_CateABMIL
            model = DAttention_CateABMIL(**model_dictionary,concept_path=args.concept_path)

        elif args.model_type == 'abmil':
            from models.abmil import DAttention
            model = DAttention(**model_dictionary)

        elif args.model_type == 'cate_dl':
            from models.cate import CATE
            from models.abmil_cluster import DAttention
            # 1. Khởi tạo Student (DAttention)
            # QUAN TRỌNG: Phải truyền concept_path và device vào đây để Student tự tính Sim Loss
            model = DAttention(**model_dictionary, 
                               concept_path=args.concept_path,device=device
                               )
            cate = CATE(input_dim=args.embed_dim, ib_dim=args.embed_dim, interv_dim=args.embed_dim, concept_path=args.concept_path,n_classes=args.n_classes).to(device)
            cate.eval()
            for p in cate.parameters():
                p.requires_grad_(False)
        else:
            raise NotImplementedError
    
    # else: # args.model_type == 'mil'
    #     if args.n_classes > 2:
    #         model = MIL_fc_mc(**model_dict)
    #     else:
    #         model = MIL_fc(**model_dict)
    _ = model.to(device)
    print('Done!')
    print_network(model)

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    print('Done!')
    
    print('\nInit Loaders...', end=' ')
    train_loader = get_split_loader(train_split, training=True, testing = args.testing, weighted = args.weighted_sample)
    val_loader = get_split_loader(val_split,  testing = args.testing)
    test_loader = get_split_loader(test_split, testing = args.testing)
    print('Done!')

    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping:
        early_stopping = EarlyStopping(patience = 20, stop_epoch=50, verbose = True)

    else:
        early_stopping = None
    print('Done!')

    for epoch in range(args.max_epochs):
        if args.model_type in ['clam_sb', 'clam_mb'] and not args.no_inst_cluster:     
            train_loop_clam(epoch, model, train_loader, optimizer, args.n_classes, args.bag_weight, writer, loss_fn)
            stop = validate_clam(cur, epoch, model, val_loader, args.n_classes, 
                early_stopping, writer, loss_fn, args.results_dir)
        
        elif args.model_type == 'cate_dl':  
            #train_loop_with_catemil(epoch, cate ,model, train_loader, optimizer, args.n_classes, writer, loss_fn)
            train_loop_with_catemil(epoch, cate ,model, train_loader, optimizer, args.n_classes, writer, loss_fn,args.concept_path)
            stop = validate_cate_mil(cur, epoch, cate,model, val_loader, args.n_classes, 
                early_stopping, writer, loss_fn, args.results_dir)

        else:
            train_loop(epoch, model, train_loader, optimizer, args.n_classes, writer, loss_fn)
            stop = validate(cur, epoch, model, val_loader, args.n_classes, 
                early_stopping, writer, loss_fn, args.results_dir)
        
        if stop: 
            break

    if args.early_stopping:
        model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))
    else:
        torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))

    # if args.model_type == 'cate_dl':
    #     _, val_error, val_auc, _= summary_catemil(model, val_loader, args.n_classes)
    #     print('Val error: {:.4f}, ROC AUC: {:.4f}'.format(val_error, val_auc))

    #     results_dict, test_error, test_auc, acc_logger = summary_catemil(model, test_loader, args.n_classes)
    #     print('Test error: {:.4f}, ROC AUC: {:.4f}'.format(test_error, test_auc))
    _, val_error, val_auc, _= summary(model, val_loader, args.n_classes)
    print('Val error: {:.4f}, ROC AUC: {:.4f}'.format(val_error, val_auc))
    results_dict, test_error, test_auc, acc_logger = summary(model, test_loader, args.n_classes)
    print('Test error: {:.4f}, ROC AUC: {:.4f}'.format(test_error, test_auc))

    for i in range(args.n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))

        if writer:
            _safe_add_scalar(writer, f'final/test_class_{i}_acc', acc, 0)

    if writer:
        writer.add_scalar('final/val_error', val_error, 0)
        writer.add_scalar('final/val_auc', val_auc, 0)
        writer.add_scalar('final/test_error', test_error, 0)
        writer.add_scalar('final/test_auc', test_auc, 0)
        writer.close()
    return results_dict, test_auc, val_auc, 1-test_error, 1-val_error 


def train_loop_clam(epoch, model, loader, optimizer, n_classes, bag_weight, writer = None, loss_fn = None):
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)
    
    train_loss = 0.
    train_error = 0.
    train_inst_loss = 0.
    inst_count = 0

    print('\n')
    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        logits, Y_prob, Y_hat, _, instance_dict = model(data, label=label, instance_eval=True)

        acc_logger.log(Y_hat, label)
        loss = loss_fn(logits, label)
        loss_value = loss.item()

        instance_loss = instance_dict['instance_loss']
        inst_count+=1
        instance_loss_value = instance_loss.item()
        train_inst_loss += instance_loss_value
        
        total_loss = bag_weight * loss + (1-bag_weight) * instance_loss 

        inst_preds = instance_dict['inst_preds']
        inst_labels = instance_dict['inst_labels']
        inst_logger.log_batch(inst_preds, inst_labels)

        train_loss += loss_value
        if (batch_idx + 1) % 20 == 0:
            print('batch {}, loss: {:.4f}, instance_loss: {:.4f}, weighted_loss: {:.4f}, '.format(batch_idx, loss_value, instance_loss_value, total_loss.item()) + 
                'label: {}, bag_size: {}'.format(label.item(), data.size(0)))

        error = calculate_error(Y_hat, label)
        train_error += error
        
        # backward pass
        total_loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)
    
    if inst_count > 0:
        train_inst_loss /= inst_count
        print('\n')
        for i in range(2):
            acc, correct, count = inst_logger.get_summary(i)
            print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))

    print('Epoch: {}, train_loss: {:.4f}, train_clustering_loss:  {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_inst_loss,  train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer and acc is not None:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)
        writer.add_scalar('train/clustering_loss', train_inst_loss, epoch)

def train_loop(epoch, model, loader, optimizer, n_classes, writer = None, loss_fn = None):   
    #device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    train_loss = 0.
    train_error = 0.

    print('\n')
    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)

        logits, Y_prob, Y_hat, _, _ = model(data)
        
        acc_logger.log(Y_hat, label)
        loss = loss_fn(logits, label)
        loss_value = loss.item()
        
        train_loss += loss_value
        if (batch_idx + 1) % 20 == 0:
            print('batch {}, loss: {:.4f}, label: {}, bag_size: {}'.format(batch_idx, loss_value, label.item(), data.size(0)))
           
        error = calculate_error(Y_hat, label)
        train_error += error
        
        # backward pass
        loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)

    print('Epoch: {}, train_loss: {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)

def train_loop_with_catemil(
        epoch,
        cate,
        model,
        loader,
        optimizer,
        n_classes,
        writer=None,
        loss_fn=None,
        concept_path=None,
        *,
        kd_T: float = 1.0,
        kd_alpha: float = 0.5,
        lambda_sim: float = 0.01,
        lambda_sym: float = 1.0,
        lambda_cross: float = 1.0,
        topk_pos: int = 20,
        topk_neg: int = 20
        ):

    model.train()
    cate.eval()

    acc_logger = Accuracy_Logger(n_classes=n_classes)
    total_loss = 0.0
    total_ce   = 0.0
    total_kd   = 0.0
    total_err  = 0.0
    total_sim  = 0.0

    print('\n')


    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)

        # --- BƯỚC 1: Teacher (CATE) Forward ---
        with torch.no_grad():
            # Vẫn cần chạy dòng này để lấy x_c cho KD Loss
            # Nhưng không cần quan tâm đến loss trả về của nó nữa
            _, x_c, _ = cate.ib(data, label, results_dict={}) 

        # --- BƯỚC 2: Student (DAttention) Forward ---
        logits, Y_prob, Y_hat, results_dict, _ = model(
            data, 
            label=label, 
            lambda_sym=lambda_sym, 
            lambda_cross=lambda_cross,
            topk_pos=topk_pos,
            topk_neg=topk_neg
        )
        
        loss_sim = results_dict.get('sim_loss', torch.tensor(0.0, device=device))

        # Teacher features forward (để tính KD)
        logits_c, _, _, _, _ = model(x_c)

        # --- BƯỚC 3: Tính toán Loss ---
        
        # 3.1 Cross Entropy
        loss_ce = loss_fn(logits, label)

        # 3.2 Knowledge Distillation (KD)
        T_kd = float(kd_T)
        loss_kd = F.kl_div(
            F.log_softmax(logits / T_kd, dim=1),
            F.softmax(logits_c.detach() / T_kd, dim=1),
            reduction='batchmean'
        ) * (T_kd * T_kd)

        # 3.3 Tổng Loss (ĐÃ BỎ PIM)
        # Công thức gọn hơn: (1-alpha)*CE + alpha*KD + lambda_sim*SIM
        loss = (1.0 - kd_alpha) * loss_ce + \
               kd_alpha * loss_kd + \
               float(lambda_sim) * loss_sim

        # --- BƯỚC 4: Backward ---
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        optimizer.step()

        # --- Logging ---
        acc_logger.log(Y_hat, label)
        total_loss += loss.item()
        total_ce   += loss_ce.item()
        total_kd   += loss_kd.item()
        total_err  += calculate_error(Y_hat, label)
        total_sim  += float(loss_sim)

        if (batch_idx + 1) % 20 == 0:
            print(
                f"[Epoch {epoch}] Batch {batch_idx+1}/{len(loader)} "
                f"CE={loss_ce.item():.4f} KD={loss_kd.item():.4f} "
                f"SIM={float(loss_sim):.4f} " # Bỏ log PIM cho đỡ rối
                f"Total={loss.item():.4f}  |  "
                f"label={label.item()} bag_size={data.size(0)}"
            )

    n_batches = max(1, len(loader))
    avg_loss = total_loss / n_batches
    avg_ce   = total_ce   / n_batches
    avg_kd   = total_kd   / n_batches
    avg_err  = total_err  / n_batches
    avg_sim  = total_sim  / n_batches

    print(
        f"Epoch {epoch}: "
        f"train_loss={avg_loss:.4f} "
        f"(ce={avg_ce:.4f}, kd={avg_kd:.4f}, sim={avg_sim:.4f}), "
        f"train_error={avg_err:.4f}"
    )

    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print(f"class {i}: acc {acc}, correct {correct}/{count}")
        if writer and acc is not None:
            writer.add_scalar(f"train/class_{i}_acc", acc, epoch)

    if writer:
        writer.add_scalar("train/loss", avg_loss, epoch)
        writer.add_scalar("train/loss_ce", avg_ce, epoch)
        writer.add_scalar("train/loss_kd", avg_kd, epoch)
        writer.add_scalar("train/loss_sim", avg_sim, epoch)
        writer.add_scalar("train/error", avg_err, epoch)

# def train_loop_with_catemil(epoch, cate, model, loader, optimizer,
#                          n_classes, writer=None, loss_fn=None,
#                          *,
#                         kd_T: float = 1.0,
#                         kd_alpha: float = 0.5,
#                         lambda_pim: float = 0.0,
#                         lambda_sim: float = 0.0,
#                         max_grad_norm: float | None = None):
#     model.train()
#     cate.eval()
#     acc_logger = Accuracy_Logger(n_classes=n_classes)
#     total_loss = 0.0
#     total_ce   = 0.0
#     total_kd   = 0.0
#     total_err  = 0.0

#     for batch_idx, (data, label) in enumerate(loader):
#         data, label = data.to(device), label.to(device)
#         results_dict = {}
#         with torch.no_grad():
#             results_dict = {}
#             _, x_c, results_dict = cate.ib(data, label, results_dict)

#         logits,  Y_prob, Y_hat, _, _ = model(data)   # [1, C]
#         logits_c, _,     _,     _, _ = model(x_c)

#         loss_ce = loss_fn(logits, label)
#         T = float(kd_T)
#         kd = F.kl_div(
#                 F.log_softmax(logits / T, dim=1),
#                 F.softmax(logits_c.detach() / T, dim=1),
#                 reduction='batchmean'
#             ) * (T * T)

#         loss_pim = results_dict.get('kl_loss', torch.tensor(0., device=device)).detach()
#         loss_sim = results_dict.get('infonce_loss', torch.tensor(0., device=device)).detach()

#         loss = (1.0 - kd_alpha) * loss_ce + kd_alpha * kd \
#                + float(lambda_pim) * loss_pim + float(lambda_sim) * loss_sim


#         optimizer.zero_grad(set_to_none=True)
#         loss.backward()
#         if max_grad_norm is not None:
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
#         optimizer.step()

#         acc_logger.log(Y_hat, label)
#         total_loss += loss.item()
#         total_ce   += loss_ce.item()
#         total_kd   += kd.item()
#         total_err  += calculate_error(Y_hat, label)

#         if (batch_idx + 1) % 20 == 0:
#             print(f"[Epoch {epoch}] Batch {batch_idx+1}/{len(loader)} "
#                   f"CE={loss_ce.item():.4f} KD={kd.item():.4f} "
#                   f"PIM={float(loss_pim):.4f} SIM={float(loss_sim):.4f} "
#                   f"Total={loss.item():.4f}  |  label={label.item()} bag_size={data.size(0)}")


#     n_batches = max(1, len(loader))
#     avg_loss = total_loss / n_batches
#     avg_ce   = total_ce   / n_batches
#     avg_kd   = total_kd   / n_batches
#     avg_err  = total_err  / n_batches

#     print(f"Epoch {epoch}: train_loss={avg_loss:.4f} (ce={avg_ce:.4f}, kd={avg_kd:.4f}), train_error={avg_err:.4f}")
#     for i in range(n_classes):
#         acc, correct, count = acc_logger.get_summary(i)
#         print(f"class {i}: acc {acc}, correct {correct}/{count}")
#         if writer and acc is not None:
#             writer.add_scalar(f"train/class_{i}_acc", acc, epoch)

#     if writer:
#         writer.add_scalar("train/loss", avg_loss, epoch)
#         writer.add_scalar("train/loss_ce", avg_ce, epoch)
#         writer.add_scalar("train/loss_kd", avg_kd, epoch)
#         writer.add_scalar("train/error", avg_err, epoch)

def validate_cate_mil(cur, epoch, cate,model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir=None,*,
        kd_T: float = 1.0,
        kd_alpha: float = 0.5,
        lambda_pim: float = 0.0,
        lambda_sim: float = 0.0):
    model.eval()
    cate.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    # loader.dataset.update_mode(True)
    val_loss = 0.0
    val_ce   = 0.0
    val_kd   = 0.0
    val_error = 0.0
    
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))

    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(loader):
            data, label = data.to(device, non_blocking=True), label.to(device, non_blocking=True)

            # teacher tạo x_c
            results_dict = {}
            _, x_c, results_dict = cate.ib(data, label, results_dict)

            # student dự đoán
            logits, Y_prob, Y_hat, _, _ = model(data)
            logits_c, _, _, _, _ = model(x_c)

            # CE & KD
            T = float(kd_T)
            loss_ce = loss_fn(logits, label)
            loss_kd = F.kl_div(
                F.log_softmax(logits / T, dim=1),
                F.softmax(logits_c / T, dim=1),
                reduction='batchmean'
            ) * (T * T)

            # tổng val loss (khuyến nghị chỉ CE+KD cho early-stopping)
            loss = (1.0 - kd_alpha) * loss_ce + kd_alpha * loss_kd

            # log AUC/acc
            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()
            acc_logger.log(Y_hat, label)

            # tích luỹ
            val_loss  += loss.item()
            val_ce    += loss_ce.item()
            val_kd    += loss_kd.item()
            val_error += calculate_error(Y_hat, label)
            

    n_batches = max(1, len(loader))
    val_loss  /= n_batches
    val_ce    /= n_batches
    val_kd    /= n_batches
    val_error /= n_batches

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
    else:
        auc = roc_auc_score(labels, prob, multi_class='ovr')

    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/loss_ce', val_ce, epoch)
        writer.add_scalar('val/loss_kd', val_kd, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)

    print(f"\nVal Set, val_loss: {val_loss:.4f} (ce={val_ce:.4f}, kd={val_kd:.4f}), "
          f"val_error: {val_error:.4f}, auc: {auc:.4f}")

    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print(f"class {i}: acc {acc}, correct {correct}/{count}")
        if writer and acc is not None:
            writer.add_scalar(f'val/class_{i}_acc', acc, epoch)

    # early stopping theo val_loss (CE+KD)
    if early_stopping:
        assert results_dir is not None
        early_stopping(epoch, val_loss, model,
                       ckpt_name=os.path.join(results_dir, f"s_{cur}_checkpoint.pt"))
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False


# Phiên bản bỏ KD luôn
# def validate_cate_mil(cur, epoch, cate, model, loader, n_classes, early_stopping=None, writer=None, loss_fn=None, results_dir=None, *,
#                       # BỎ HẾT CÁC THAM SỐ THỪA
#                       kd_T: float = 1.0,      # Giữ lại để hứng tham số nếu gọi từ main, nhưng không dùng
#                       kd_alpha: float = 0.3): # Giữ lại để hứng tham số nếu gọi từ main, nhưng không dùng
    
#     model.eval()
#     # cate.eval() --> KHÔNG CẦN CATE NỮA VÌ KHÔNG TÍNH KD
    
#     acc_logger = Accuracy_Logger(n_classes=n_classes)
    
#     val_loss = 0.0
#     val_error = 0.0
    
#     all_probs = []                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
#     all_labels = []                                                                                                                                                                                       

#     with torch.no_grad():
#         for batch_idx, (data, label) in enumerate(loader):
#             data, label = data.to(device, non_blocking=True), label.to(device, non_blocking=True)

#             # 1. Student dự đoán
#             # Không cần chạy cate.ib() hay tính x_c nữa -> Nhanh hơn
#             logits, Y_prob, Y_hat, _, _ = model(data)
            
#             # 2. Tính Loss (CHỈ DÙNG CE)
#             loss = loss_fn(logits, label) 

#             # 3. Logging
#             acc_logger.log(Y_hat, label)
#             val_error += calculate_error(Y_hat, label)
#             val_loss += loss.item()

#             # Lưu xác suất
#             probs = Y_prob.detach().cpu().numpy()
#             all_probs.append(probs)
#             all_labels.append(label.item())

#     # Tính trung bình epoch
#     n_batches = max(1, len(loader))
#     val_loss  /= n_batches
#     val_error /= n_batches

#     # Xử lý AUC
#     if len(all_probs) > 0:
#         if all_probs[0].ndim > 1: all_probs = np.concatenate(all_probs)
#         else: all_probs = np.array(all_probs)
#     if all_probs.ndim == 1: all_probs = all_probs.reshape(-1, n_classes)
#     all_labels = np.array(all_labels)

#     if n_classes == 2:
#         auc = roc_auc_score(all_labels, all_probs[:, 1])
#     else:
#         auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')

#     # Tensorboard (Chỉ còn loss thuần và metric)
#     if writer:
#         writer.add_scalar('val/loss', val_loss, epoch)
#         writer.add_scalar('val/auc', auc, epoch)
#         writer.add_scalar('val/error', val_error, epoch)

#     print(f"\nVal Set, val_loss: {val_loss:.4f}, val_error: {val_error:.4f}, auc: {auc:.4f}")

#     for i in range(n_classes):
#         acc, correct, count = acc_logger.get_summary(i)
#         print(f"class {i}: acc {acc}, correct {correct}/{count}")
#         if writer and acc is not None:
#             writer.add_scalar(f'val/class_{i}_acc', acc, epoch)

#     # Early Stopping dựa trên CE Loss thuần
#     if early_stopping:
#         assert results_dir is not None
#         early_stopping(epoch, val_loss, model,
#                        ckpt_name=os.path.join(results_dir, f"s_{cur}_checkpoint.pt"))
#         if early_stopping.early_stop:
#             print("Early stopping")
#             return True

#     return False

def validate(cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir=None):
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    # loader.dataset.update_mode(True)
    val_loss = 0.
    val_error = 0.
    
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))

    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(loader):
            data, label = data.to(device, non_blocking=True), label.to(device, non_blocking=True)

            logits, Y_prob, Y_hat, _, _ = model(data)

            acc_logger.log(Y_hat, label)
            
            loss = loss_fn(logits, label)

            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()
            
            val_loss += loss.item()
            error = calculate_error(Y_hat, label)
            val_error += error
            

    val_error /= len(loader)
    val_loss /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
    
    else:
        auc = roc_auc_score(labels, prob, multi_class='ovr')
    
    
    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))     

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False

def validate_clam(cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir = None):
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)
    val_loss = 0.
    val_error = 0.

    val_inst_loss = 0.
    val_inst_acc = 0.
    inst_count=0
    
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))
    sample_size = model.k_sample
    with torch.inference_mode():
        for batch_idx, (data, label) in enumerate(loader):
            data, label = data.to(device), label.to(device)      
            logits, Y_prob, Y_hat, _, instance_dict = model(data, label=label, instance_eval=True)
            acc_logger.log(Y_hat, label)
            
            loss = loss_fn(logits, label)

            val_loss += loss.item()

            instance_loss = instance_dict['instance_loss']
            
            inst_count+=1
            instance_loss_value = instance_loss.item()
            val_inst_loss += instance_loss_value

            inst_preds = instance_dict['inst_preds']
            inst_labels = instance_dict['inst_labels']
            inst_logger.log_batch(inst_preds, inst_labels)

            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()
            
            error = calculate_error(Y_hat, label)
            val_error += error

    val_error /= len(loader)
    val_loss /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], prob[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))
    if inst_count > 0:
        val_inst_loss /= inst_count
        for i in range(2):
            acc, correct, count = inst_logger.get_summary(i)
            print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))
    
    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)
        writer.add_scalar('val/inst_loss', val_inst_loss, epoch)


    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        
        if writer and acc is not None:
            writer.add_scalar('val/class_{}_acc'.format(i), acc, epoch)
     

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False

def summary(model, loader, n_classes):
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        slide_id = slide_ids.iloc[batch_idx]
        with torch.inference_mode():
            logits, Y_prob, Y_hat, _, _ = model(data)

        acc_logger.log(Y_hat, label)
        probs = Y_prob.cpu().numpy()
        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()
        
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
        error = calculate_error(Y_hat, label)
        test_error += error

    test_error /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(all_labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in all_labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))


    return patient_results, test_error, auc, acc_logger

# def summary(model, loader, n_classes):
#     #device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     acc_logger = Accuracy_Logger(n_classes=n_classes)
#     model.eval()
#     test_loss = 0.
#     test_error = 0.

#     all_probs = np.zeros((len(loader), n_classes))
#     all_labels = np.zeros(len(loader))
#     all_preds = []

#     slide_ids = loader.dataset.slide_data['slide_id']
#     patient_results = {}

#     all_Y_hat = []
#     all_label = []
#     for batch_idx, (data, label) in enumerate(loader):
#         data, label = data.to(device), label.to(device)
#         slide_id = slide_ids.iloc[batch_idx]
#         with torch.no_grad():
#             logits, Y_prob, Y_hat, _, _ = model(data)

#         acc_logger.log(Y_hat, label)
#         probs = Y_prob.cpu().numpy()
#         all_probs[batch_idx] = probs
#         all_labels[batch_idx] = label.item()
#         all_preds.extend(Y_hat.cpu().numpy())
        
#         patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
#         error = calculate_error(Y_hat, label)
#         test_error += error

#         all_Y_hat.append(Y_hat.cpu().numpy())
#         all_label.append(label.cpu().numpy())

#     test_error /= len(loader)
#     all_Y_hat = np.concatenate(all_Y_hat)
#     all_label = np.concatenate(all_label)

#     if n_classes == 2:
#         auc = roc_auc_score(all_labels, all_probs[:, 1])
#         aucs = []
#     else:
#         aucs = []
#         binary_labels = label_binarize(all_labels, classes=[i for i in range(n_classes)])
#         for class_idx in range(n_classes):
#             if class_idx in all_labels:
#                 fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
#                 aucs.append(calc_auc(fpr, tpr))
#             else:
#                 aucs.append(float('nan'))

#         auc = np.nanmean(np.array(aucs))


#     return patient_results, test_error, auc, acc_logger

def summary_catemil(model, loader, n_classes):
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        slide_id = slide_ids.iloc[batch_idx]
        with torch.inference_mode():
            logits, Y_prob, Y_hat, _,_= model(data)

        acc_logger.log(Y_hat, label)
        probs = Y_prob.cpu().numpy()
        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()
        
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
        error = calculate_error(Y_hat, label)
        test_error += error

    test_error /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(all_labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in all_labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))


    return patient_results, test_error, auc, acc_logger
