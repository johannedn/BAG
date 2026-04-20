import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from torch_geometric.utils import negative_sampling
from detectors.utils import *
from models.static.gnn import *
from load_config import args   
      
class staticGNNDetector(BaseDetector):
    def __init__(self, train_config, model_config, graph):
        super().__init__(train_config, model_config)
        
        self.graph = graph.to(self.device)                 
        self.train_edge = graph.edge_index
        self.train_x = graph.x
        self.eval_edge = graph.edge_index 
        self.eval_x = graph.x  

        model_config['input_dim'] = self.graph.x.shape[1]
        gnn = globals()[model_config['model']]
        backbone = gnn(**model_config).to(self.device)        
        self.model = nn.Sequential(backbone, self.scorer).to(self.device)

        print("\n========== TRAIN CONFIG ==========")
        for k, v in self.train_config.items():
            print(f"{k}: {v}")
        print("========== MODEL CONFIG ==========")
        for k, v in self.model_config.items():
            print(f"{k}: {v}")             


    def train(self):
        optimizer = torch.optim.Adam(self.model.parameters(),  lr=self.model_config['lr'], weight_decay=args.weight_decay) 
        
        for e in range(self.model_config['epochs']):
            self.model.train()
            optimizer.zero_grad()
                        
            if self.train_config['label'] == False:
                train_norm = self.graph.edge_index[:, self.graph.train_mask]
                train_anom = negative_sampling(
                    edge_index = train_norm,
                    num_nodes = self.graph.num_nodes,
                    num_neg_samples = train_norm.size(1)
                ).to(dtype=torch.int64)  
            else:
                norm_mask = self.graph.train_mask & (self.graph.y == 0) 
                anom_mask = self.graph.train_mask & (self.graph.y == 1) 
                
                train_norm = self.graph.edge_index[:, norm_mask]
                train_anom = self.graph.edge_index[:, anom_mask]
                
            z = self.model[0](self.train_x, self.train_edge) 
            norm_logits = self.model[1](train_norm[0], train_norm[1], z)
            anom_logits = self.model[1](train_anom[0], train_anom[1], z)
            
            if self.train_config['loss'] == 'bce':
                norm_loss = F.binary_cross_entropy_with_logits(norm_logits, torch.zeros_like(norm_logits).to(self.device)) 
                anom_loss = F.binary_cross_entropy_with_logits(anom_logits, torch.ones_like(anom_logits).to(self.device)) 
                loss = norm_loss + anom_loss  
            else:
                loss = self.pairwise_loss(norm_logits, anom_logits)
                
            loss.backward()  
            optimizer.step()  

            self.model.eval()
            
            with torch.no_grad():
                z = self.model[0](self.eval_x, self.eval_edge) 
                logits = self.model[1](self.eval_edge[0], self.eval_edge[1], z)     
                probs = torch.sigmoid(logits)
                
                val_mask, test_mask = self.graph.val_mask, self.graph.test_mask
                val_y, val_probs = self.graph.y[val_mask], probs[val_mask]
                test_y, test_probs = self.graph.y[test_mask], probs[test_mask]
                
                val_score, test_score = self.get_result(val_y, val_probs, test_y, test_probs)
                
                if val_score[self.train_config['metric']] > self.best_score:
                    print('Epoch {}, Loss {:.4f}, Val AUC {:.4f}, PRC {:.4f}, RecK {:.4f}, F1 {:.4f}, test AUC {:.4f}, PRC {:.4f}, RecK {:.4f}, F1 {:.4f}'.format(e, loss, 
                        val_score['AUROC'], val_score['AUPRC'], val_score['RecK'], val_score['F1'], test_score['AUROC'], test_score['AUPRC'], test_score['RecK'], test_score['F1']))  
                    self.patience_knt = 0
                    self.best_score = val_score[self.train_config['metric']]
                    self.best_test_score = test_score
                    
                else:
                    self.patience_knt += 1
                    if self.patience_knt > self.model_config['patience']:
                        break         

        return self.best_test_score

    

class RFGraphDetector(BaseDetector):   
    def __init__(self, train_config, model_config, graph):
        super().__init__(train_config, model_config)
        self.graph = graph.to(self.device) 

        self.train_edge = graph.edge_index[:, graph.train_mask]
        self.train_x = graph.x[torch.unique(self.train_edge)]
        self.eval_edge = graph.edge_index[:, graph.val_mask | graph.test_mask]
        self.eval_x = graph.x[torch.unique(self.eval_edge)] 
            
        model_config['input_dim'] = self.graph.x.shape[1]           

        n_estimators = 100 if 'n_estimators' not in model_config else model_config['n_estimators']
        criterion = 'gini' if 'criterion' not in model_config else model_config['criterion']
        max_samples = None if 'max_samples' not in model_config else model_config['max_samples']
        max_features = 'sqrt' if 'max_features' not in model_config else model_config['max_features']
        self.model = RandomForestClassifier(n_jobs=32, n_estimators=n_estimators, criterion=criterion,
                                            max_samples=max_samples, max_features=max_features)
        self.gnn = GCN(**model_config).to(self.device)

    def train(self):
        test_score = {}
        if self.train_config['label'] == False:
            train_pos = self.graph.edge_index[:, self.graph.train_mask]
            train_neg = negative_sampling(
                edge_index = train_pos,
                num_nodes = self.graph.num_nodes,
                num_neg_samples = train_pos.size(1)
            ).to(dtype=torch.int64)  
        else:
            pos_mask = self.graph.train_mask & (self.graph.y == 0)  # Ensure mask is boolean
            neg_mask = self.graph.train_mask & (self.graph.y == 1)  # Ensure mask is boolean
            train_pos = self.graph.edge_index[:, pos_mask]
            train_neg = self.graph.edge_index[:, neg_mask]  
        
        z = self.gnn(self.graph.x, self.graph.edge_index)  # Get node embeddings
        
        pos_edge_x = z[train_pos[0]] * z[train_pos[1]]
        neg_edge_x = z[train_neg[0]] * z[train_neg[1]]
        train_x = torch.cat([pos_edge_x, neg_edge_x], dim=0) 
        train_y = torch.cat([torch.zeros(pos_edge_x.size(0)), torch.ones(neg_edge_x.size(0))], dim=0) 
        
        self.model.fit(train_x.detach().cpu().numpy(), train_y.detach().cpu().numpy())
        
        val_edge = self.graph.edge_index[:, self.graph.val_mask]        
        val_x =  z[val_edge[0]] * z[val_edge[1]]
        val_x_np = val_x.detach().cpu().numpy()
        y_true = self.graph.y[self.graph.val_mask]
        y_pred = self.model.predict_proba(val_x_np)[:, 1] 
        y_pred = torch.tensor(y_pred, dtype=torch.float32) 

        test_edge = self.graph.edge_index[:, self.graph.test_mask]        
        test_x =  z[test_edge[0]] * z[test_edge[1]]
        test_x_np = test_x.detach().cpu().numpy()
        y_true = self.graph.y[self.graph.test_mask]
        y_pred = self.model.predict_proba(test_x_np)[:, 1] 
        y_pred = torch.tensor(y_pred, dtype=torch.float32) 
               
        _, f_1 = self.eval(y_true, y_pred, 0.5)                           
        test_score, _ = self.eval(y_true, y_pred, f_1)
        
        return test_score
    
    
class XGBGraphDetector(RFGraphDetector):
    def __init__(self, train_config, model_config, graph):
        super().__init__(train_config, model_config, graph)
        self.model = xgb.XGBClassifier(tree_method='gpu_hist', eval_metric=average_precision_score, verbose=2, **model_config)