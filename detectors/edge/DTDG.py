import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling
from detectors.utils import *
from models.dtdg.dtdg import *
from load_config import args

class AddGraphDetector(BaseDetector):
    def __init__(self, train_config, model_config, train_snapshots, valid_snapshots, test_snapshots):
        super().__init__(train_config, model_config)

        self.train_snapshots = [snapshot.to(train_config['device']) for snapshot in train_snapshots]
        if not isinstance(valid_snapshots, list):
            self.valid_snapshots = valid_snapshots.to(train_config['device'])
        else:
            self.valid_snapshots = [snapshot.to(train_config['device']) for snapshot in valid_snapshots]
        if not isinstance(test_snapshots, list):    
            self.test_snapshots = test_snapshots.to(train_config['device'])
        else:
            self.test_snapshots = [snapshot.to(train_config['device']) for snapshot in test_snapshots]
        
        model_config['input_dim'] = self.train_snapshots[0].x.shape[1]
        model_config['num_nodes'] = self.train_snapshots[0].x.shape[0]
        gnn = globals()[model_config['model']]
        backbone = gnn(**model_config).to(train_config['device'])
        self.model = nn.Sequential(backbone, self.scorer).to(train_config['device'])
        
        print("\n========== TRAIN CONFIG ==========")
        for k, v in self.train_config.items():
            print(f"{k}: {v}")
        print("========== MODEL CONFIG ==========")
        for k, v in self.model_config.items():
            print(f"{k}: {v}")
        print("")
                
    def train(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.model_config['lr'], weight_decay=args.weight_decay)
        
        for e in range(self.model_config['epochs']):
            total_loss = 0
            hist_embs = [torch.ones(self.model_config['num_nodes'], self.model_config['hidden_dim']).to(self.device)]

            self.model.train() 
            
            for i, data in enumerate(self.train_snapshots):
                if self.train_config['label'] == False:
                    train_pos = data.edge_index
                    train_neg = negative_sampling(
                        edge_index = train_pos,
                        num_nodes = data.num_nodes,
                        num_neg_samples = train_pos.size(1)
                    ).to(dtype=torch.int64)  
                else:
                    train_pos = data.edge_index[:, (data.y == 0).nonzero(as_tuple=True)[0]]
                    train_neg = data.edge_index[:, (data.y == 1).nonzero(as_tuple=True)[0]]
                
                z = self.model[0](hist_embs, data.edge_index)
                hist_embs.append(z.detach())
                    
                pos_logits = self.model[1](train_pos[0], train_pos[1], z)
                neg_logits = self.model[1](train_neg[0], train_neg[1], z)
                
                pos_loss = F.binary_cross_entropy_with_logits(pos_logits, torch.zeros_like(pos_logits).to(self.device)) 
                neg_loss = F.binary_cross_entropy_with_logits(neg_logits, torch.ones_like(neg_logits).to(self.device)) 
                
                if self.train_config['loss'] == 'bce':
                    loss = pos_loss + neg_loss  
                else:
                    loss = self.pairwise_loss(pos_logits, neg_logits)
  
                loss.backward()
                optimizer.step()                
                optimizer.zero_grad() 
                                         
                total_loss += loss.item()
                   
            total_loss = total_loss / (i+1) 
            
            self.model.eval()
            
            val_y, val_probs = self.prob(self.model, self.valid_snapshots)
            test_y, test_probs = self.prob(self.model, self.test_snapshots)
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

    @torch.no_grad()
    def prob(self, model, snapshots):
        y_ls, probs_ls = [], [],
        hist_embs = [torch.ones(self.model_config['num_nodes'], self.model_config['hidden_dim']).to(self.device)]
    
        for i, data in enumerate(snapshots):

            z = model[0](hist_embs, data.edge_index)
            logits = model[1](data.edge_index[0], data.edge_index[1], z)
            probs = torch.sigmoid(logits)
            y = data.y
            
            hist_embs.append(z)
            y_ls.append(y)
            probs_ls.append(probs)

        y = torch.cat(y_ls, dim=0) 
        probs = torch.cat(probs_ls, dim=0)  
     
        return y, probs
    


class DTDGDetector(BaseDetector):
    def __init__(self, train_config, model_config, train_snapshots, valid_snapshots, test_snapshots):
        super().__init__(train_config, model_config)

        self.train_snapshots = [snapshot.to(train_config['device']) for snapshot in train_snapshots]
        if not isinstance(valid_snapshots, list):
            self.valid_snapshots = valid_snapshots.to(train_config['device'])
        else:
            self.valid_snapshots = [snapshot.to(train_config['device']) for snapshot in valid_snapshots]
        if not isinstance(test_snapshots, list):    
            self.test_snapshots = test_snapshots.to(train_config['device'])
        else:
            self.test_snapshots = [snapshot.to(train_config['device']) for snapshot in test_snapshots]
        
        gnn = globals()[model_config['model']]
        model_config['input_dim'] = self.train_snapshots[0].x.shape[1]
        model_config['num_nodes'] = self.train_snapshots[0].x.shape[0]
        backbone = gnn(**model_config).to(train_config['device'])
        self.model = nn.Sequential(backbone, self.scorer).to(train_config['device'])

        print("========== TRAIN CONFIG ==========")
        for k, v in self.train_config.items():
            print(f"{k}: {v}")
        print("========== MODEL CONFIG ==========")
        for k, v in self.model_config.items():
            print(f"{k}: {v}")              

    def train(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.model_config['lr'], weight_decay=args.weight_decay)
        if self.model_config['model'] == 'EvolveGCN_O':
            for param in self.model[0].parameters():
                param.retain_grad()
            
        for e in range(self.model_config['epochs']):
            total_loss = 0
            if self.model_config['model'] == 'Dy_GrAE':
                h, c = None, None
            else:
                pass
            
            self.model.train()
            for i, data in enumerate(self.train_snapshots):
                if self.train_config['label'] == False:
                    train_pos = data.edge_index
                    train_neg = negative_sampling(
                        edge_index = train_pos,
                        num_nodes = data.num_nodes,
                        num_neg_samples = train_pos.size(1)
                    ).to(dtype=torch.int64)  
                else:
                    train_pos = data.edge_index[:, (data.y == 0).nonzero(as_tuple=True)[0]]
                    train_neg = data.edge_index[:, (data.y == 1).nonzero(as_tuple=True)[0]]
                    
                if self.model_config['model'] == 'Dy_GrAE':
                    z, h, c = self.model[0](data.x, data.edge_index, h, c)
                else:
                    z = self.model[0](data.x, data.edge_index)
                    
                pos_logits = self.model[1](train_pos[0], train_pos[1], z)
                neg_logits = self.model[1](train_neg[0], train_neg[1], z)
                
                pos_loss = F.binary_cross_entropy_with_logits(pos_logits, torch.zeros_like(pos_logits).to(self.device)) 
                neg_loss = F.binary_cross_entropy_with_logits(neg_logits, torch.ones_like(neg_logits).to(self.device)) 
               
                if self.train_config['loss'] == 'bce':
                    loss = pos_loss + neg_loss  
                else:
                    loss = self.pairwise_loss(pos_logits, neg_logits)

                if self.model_config['model'] in ['EvolveGCN_O']:
                    loss.backward(retain_graph=True)
                else:
                    loss.backward()
                    
                optimizer.step()                
                optimizer.zero_grad()   

                if self.model_config['model'] == 'Dy_GrAE':
                    h = h.detach()
                    c = c.detach()
                else:
                    pass
                
                total_loss += loss.item()
                   
            total_loss = total_loss / (i+1) 
                           
            self.model.eval()

            val_y, val_probs = self.prob(self.model, self.valid_snapshots)
            test_y, test_probs = self.prob(self.model, self.test_snapshots)
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

    @torch.no_grad()
    def prob(self, model, snapshots):
        y_ls, probs_ls = [], []
        
        if self.model_config['model'] == 'Dy_GrAE':
            h, c = None, None
        else:
            pass
    
        for i, data in enumerate(snapshots):
            if self.model_config['model'] == 'Dy_GrAE':
                z, h, c = model[0](data.x, data.edge_index, h, c)
            else:
                z = model[0](data.x, data.edge_index)

            y = data.y
            logits = model[1](data.edge_index[0], data.edge_index[1], z)
            probs = torch.sigmoid(logits)
            
            y_ls.append(y)
            probs_ls.append(probs)

        y = torch.cat(y_ls, dim=0) 
        probs = torch.cat(probs_ls, dim=0) 

        return y, probs