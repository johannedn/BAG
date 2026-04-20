import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score


class EdgeScorer(nn.Module):
    def __init__(self, input_dim:int, hidden_dim: int, output_dim:int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.act = nn.ReLU()

    def forward(self, src, dst, z):
        assert src.max() < z.shape[0]
        assert dst.max() < z.shape[0]

        emb_src = z[src]
        emb_dst = z[dst]
        edge_embs = emb_src * emb_dst
        h = self.fc2(self.act(self.fc1(edge_embs)))
        return h
    

class BaseDetector(object):
    def __init__(self, train_config, model_config):
        super().__init__()
        self.model_config = model_config
        self.train_config = train_config
        self.hidden_dim = model_config['hidden_dim']
        self.device = train_config['device']        
        self.best_score = -1
        self.best_test_score = -1
        self.patience_knt = 0
        self.scorer = EdgeScorer(input_dim=self.hidden_dim, hidden_dim=self.hidden_dim, output_dim=1)
                
    def pairwise_loss(self, norm_score, anom_score, margin=0.6):
        return F.relu(margin + norm_score - anom_score).mean()
    
    @torch.no_grad()
    def eval(self, y_true, y_pred, f_1):
        result = {}
        
        y_pred_labels = (y_pred > f_1).float()
        k = int(y_true.sum().item())
        top_k_preds = y_pred.view(-1).cpu().argsort(descending=True)[:k] 

        rec = y_true.cpu()[top_k_preds].sum().float() / k 
        auc = roc_auc_score(y_true.cpu(), y_pred.cpu())
        prc = average_precision_score(y_true.cpu(), y_pred.cpu())
        f1 = f1_score(y_true.cpu().numpy(), y_pred_labels.cpu().numpy(), average='binary')
        
        result['AUROC'] = auc
        result['AUPRC'] = prc
        result['RecK'] = rec
        result['F1'] = f1
                 
        best_threshold = 0
        best_f1 = 0
        for threshold in [i * 0.1 for i in range(2, 8)]:  
            y_pred_labels = (y_pred > threshold).float()
            current_f1 = f1_score(y_true.cpu().numpy(), y_pred_labels.cpu().numpy(), average='binary')
            if current_f1 > best_f1:
                best_f1 = current_f1
                best_threshold = threshold
                
        return result, best_threshold  

    def get_result(self, val_y, val_probs, test_y, test_probs):
        val_score, f_1 = self.eval(val_y, val_probs, f_1=0.5)          
        test_score, _ = self.eval(test_y, test_probs, f_1=f_1) 
    
        return val_score, test_score
