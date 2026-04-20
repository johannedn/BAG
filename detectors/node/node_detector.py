from models.static.gnn import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import numpy as np
from models.ctdg.models.TGAT import TGAT
from models.ctdg.models.MemoryModel import MemoryModel, compute_src_dst_node_time_shifts
from models.ctdg.models.CAWN import CAWN
from models.ctdg.models.TCL import TCL
from models.ctdg.models.GraphMixer import GraphMixer
from models.ctdg.models.DyGFormer import DyGFormer
from models.ctdg.models.FreeDyG import FreeDyG
from models.ctdg.models.SAD import GDN
from models.ctdg.utils.utils import get_neighbor_sampler, NegativeEdgeSampler
from load_config import args
from models.ctdg.utils.DataLoader import get_idx_data_loader
from models.ctdg.models.SLADE import SLADE
from tqdm import tqdm

class NodeScorer(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim:int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        x = self.dropout(self.act(self.fc1(x)))
        return self.fc2(x)

class BaseDetector_Node(object):
    def __init__(self, train_config, model_config):
        super().__init__()
        self.model_config = model_config
        self.train_config = train_config
        self.device = train_config['device']        
        self.best_score = -1
        self.patience_knt = 0
                
    def pairwise_loss(self, pos_score, neg_score, margin=0.6):
        return F.relu(margin + pos_score - neg_score).mean()
    
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
  
    
class CTDGDetector_Node(BaseDetector_Node):
    def __init__(self, train_config, model_config, node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data):
        super().__init__(train_config, model_config)
        self.model_name = model_config['model']
        self.node_raw_features = node_raw_features
        self.edge_raw_features = edge_raw_features
        self.full_data = full_data
        self.train_data = train_data
        self.val_data = val_data 
        self.test_data = test_data 

        self.train_neighbor_sampler = get_neighbor_sampler(data = self.train_data, sample_neighbor_strategy = args.sample_neighbor_strategy,
                                                  time_scaling_factor = args.time_scaling_factor, seed=0)
        self.full_neighbor_sampler = get_neighbor_sampler(data = self.full_data, sample_neighbor_strategy = args.sample_neighbor_strategy,
                                                 time_scaling_factor = args.time_scaling_factor, seed=1)
        
        if self.model_name in ['TGAT', 'SAD']:
            backbone = TGAT(node_raw_features=self.node_raw_features, edge_raw_features=self.edge_raw_features, neighbor_sampler=self.train_neighbor_sampler,
                            time_feat_dim=args.time_feat_dim, num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout, device=args.device)
        elif self.model_name == 'CAWN':
            backbone = CAWN(node_raw_features=self.node_raw_features, edge_raw_features=self.edge_raw_features, neighbor_sampler=self.train_neighbor_sampler,
                                    time_feat_dim=args.time_feat_dim, position_feat_dim=args.position_feat_dim, walk_length=args.walk_length,
                                    num_walk_heads=args.num_walk_heads, dropout=args.dropout, device=args.device)
        elif self.model_name == 'TCL':
            backbone = TCL(node_raw_features=self.node_raw_features, edge_raw_features=self.edge_raw_features, neighbor_sampler=self.train_neighbor_sampler,
                                   time_feat_dim=args.time_feat_dim, num_layers=args.num_layers, num_heads=args.num_heads,
                                   num_depths=args.num_neighbors + 1, dropout=args.dropout, device=args.device)
        elif self.model_name == 'GraphMixer':
            backbone = GraphMixer(node_raw_features=self.node_raw_features, edge_raw_features=self.edge_raw_features, neighbor_sampler=self.train_neighbor_sampler,
                                          time_feat_dim=args.time_feat_dim, num_tokens=args.num_neighbors, num_layers=args.num_layers, dropout=args.dropout, device=args.device)
        elif self.model_name == 'DyGFormer':
            backbone = DyGFormer(node_raw_features=self.node_raw_features, edge_raw_features=self.edge_raw_features, neighbor_sampler=self.train_neighbor_sampler,
                                         time_feat_dim=args.time_feat_dim, channel_embedding_dim=args.channel_embedding_dim, patch_size=args.patch_size,
                                         num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout,
                                         max_input_sequence_length=args.max_input_sequence_length, device=args.device)
        elif self.model_name == 'FreeDyG':
            backbone = FreeDyG(node_raw_features=self.node_raw_features, edge_raw_features=self.edge_raw_features, neighbor_sampler=self.train_neighbor_sampler,
                                         time_feat_dim=args.time_feat_dim, channel_embedding_dim=args.channel_embedding_dim,
                                         num_layers=args.num_layers, dropout=args.dropout, max_input_sequence_length=args.max_input_sequence_length, device=args.device)
        
        gdn = GDN(self.device)     ##only for SAD
        self.scorer = NodeScorer(input_dim=self.node_raw_features.shape[1], hidden_dim=self.model_config['hidden_dim'], output_dim=1)
        
        
        if self.model_name == 'SAD':
            self.model = nn.Sequential(backbone, self.scorer, gdn).to(self.device)
        else:
            self.model = nn.Sequential(backbone, self.scorer).to(self.device)
            
    def train(self):
        optimizer = torch.optim.Adam(self.model.parameters(),  lr=self.model_config['lr'], weight_decay=args.weight_decay) 
        
        train_neg_edge_sampler = NegativeEdgeSampler(src_node_ids = self.train_data.src_node_ids, dst_node_ids = self.train_data.dst_node_ids)
        val_neg_edge_sampler = NegativeEdgeSampler(src_node_ids = self.full_data.src_node_ids, dst_node_ids = self.full_data.dst_node_ids, seed=0)
        test_neg_edge_sampler = NegativeEdgeSampler(src_node_ids = self.full_data.src_node_ids, dst_node_ids = self.full_data.dst_node_ids, seed=2)

        train_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(self.train_data.src_node_ids))), batch_size=self.train_config['batch_size'], shuffle=False)
        val_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(self.val_data.src_node_ids))), batch_size=self.train_config['batch_size'], shuffle=False)
        test_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(self.test_data.src_node_ids))), batch_size=self.train_config['batch_size'], shuffle=False)
        
        for e in range(self.model_config['epochs']):
            self.model.train()
            
            if self.model_name in ['DyRep', 'TGAT', 'TGN', 'CAWN', 'TCL', 'GraphMixer', 'DyGFormer','FreeDyG', 'SAD']:
                self.model[0].set_neighbor_sampler(self.train_neighbor_sampler)
                
            train_losses = []
            train_idx_data_loader_tqdm = tqdm(train_idx_data_loader, ncols=120)
            
            for batch_idx, train_data_indices in enumerate(train_idx_data_loader_tqdm):
                train_data_indices = train_data_indices.numpy()
                batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids, batch_labels = \
                    self.train_data.src_node_ids[train_data_indices], self.train_data.dst_node_ids[train_data_indices], \
                    self.train_data.node_interact_times[train_data_indices], self.train_data.edge_ids[train_data_indices], self.train_data.labels[train_data_indices]
                    
                if self.train_config['label'] == False:
                    _, batch_neg_dst_node_ids = train_neg_edge_sampler.sample(size=len(batch_src_node_ids))
                    batch_neg_src_node_ids = batch_src_node_ids

                if self.model_name in ['TGAT', 'CAWN', 'TCL', 'SAD']:
                    batch_src_node_embeddings, batch_dst_node_embeddings = \
                        self.model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                          dst_node_ids=batch_dst_node_ids,
                                                                          node_interact_times=batch_node_interact_times,
                                                                          num_neighbors=args.num_neighbors)
                        
                    if self.train_config['label'] == False:
                        batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                            self.model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                            dst_node_ids=batch_neg_dst_node_ids,
                                                                            node_interact_times=batch_node_interact_times,
                                                                            num_neighbors=args.num_neighbors)
                        
                elif self.model_name in ['GraphMixer']:
                    batch_src_node_embeddings, batch_dst_node_embeddings = \
                        self.model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                          dst_node_ids=batch_dst_node_ids,
                                                                          node_interact_times=batch_node_interact_times,
                                                                          num_neighbors=args.num_neighbors,
                                                                          time_gap=args.time_gap)
                    if self.train_config['label'] == False:
                        batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                            self.model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                            dst_node_ids=batch_neg_dst_node_ids,
                                                                            node_interact_times=batch_node_interact_times,
                                                                            num_neighbors=args.num_neighbors,
                                                                            time_gap=args.time_gap)
                            
                elif self.model_name in ['DyGFormer']:
                    batch_src_node_embeddings, batch_dst_node_embeddings = \
                        self.model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                          dst_node_ids=batch_dst_node_ids,
                                                                          node_interact_times=batch_node_interact_times)

                    if self.train_config['label'] == False:
                        batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                            self.model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                            dst_node_ids=batch_neg_dst_node_ids,
                                                                            node_interact_times=batch_node_interact_times)
                            
                elif self.model_name in ['FreeDyG']:
                    batch_src_node_embeddings, batch_dst_node_embeddings = \
                        self.model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                          dst_node_ids=batch_dst_node_ids,
                                                                          node_interact_times=batch_node_interact_times
                                                                            )
                    if self.train_config['label'] == False:
                        batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                            self.model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                            dst_node_ids=batch_neg_dst_node_ids,
                                                                            node_interact_times=batch_node_interact_times
                                                                            )
                else:
                    raise ValueError(f"Wrong value for model_name {args.model_name}!")
                
                
                # get positive and negative probabilities, shape (batch_size, )
                pos_logits = self.model[1](batch_src_node_embeddings)
                pos_logits = pos_logits.squeeze(dim=-1)

                if self.model_name in ['SAD']:
                    labels = torch.tensor(batch_labels, dtype=torch.float32, device=self.device)
                    time = torch.tensor(batch_node_interact_times, dtype=torch.float32, device=self.device)
    
                    gdn_score = self.model[2](batch_src_node_embeddings, time, labels)  
                    dev, group = self.model[2].dev_diff(torch.squeeze(gdn_score), time, labels)
                    loss_anomaly = torch.Tensor(0).to(self.device)
                    loss_supc = torch.Tensor(0).to(self.device)
                    alpha = 1e-1  # 1e-1
                    beta = 1e-3 # 1e-3
                    loss_anomaly = self.model[2].dev_loss(torch.squeeze(labels), torch.squeeze(gdn_score), torch.squeeze(time))
                    loss_supc = self.model[2].suploss(batch_src_node_embeddings, self.device, group, dev)
                    loss = F.binary_cross_entropy_with_logits(pos_logits, torch.from_numpy(batch_labels).float().to(self.device)) + alpha * loss_anomaly
                    loss += alpha * loss_anomaly + beta * loss_supc
                else:
                    loss = F.binary_cross_entropy_with_logits(pos_logits, torch.from_numpy(batch_labels).float().to(self.device)) 
             
             
                loss = F.binary_cross_entropy_with_logits(pos_logits, torch.from_numpy(batch_labels).float().to(self.device)) 
                                       
                train_losses.append(loss.item())

                optimizer.zero_grad()   
                loss.backward()  
                optimizer.step()  
                 
                train_idx_data_loader_tqdm.set_description(f'Epoch: {e}, train for the {batch_idx + 1}-th batch, train loss: {loss.item()}')           
                
            val_score, f_1 = self.eval(model_name=self.model_name,
                                    model=self.model,
                                    neighbor_sampler=self.full_neighbor_sampler,
                                    evaluate_idx_data_loader=val_idx_data_loader,
                                    evaluate_neg_edge_sampler=val_neg_edge_sampler,
                                    evaluate_data=self.val_data,
                                    num_neighbors=args.num_neighbors,
                                    f_1 = 0.5,
                                    time_gap=args.time_gap)             

            test_score, _  = self.eval(model_name=self.model_name,
                                    model=self.model,
                                    neighbor_sampler=self.full_neighbor_sampler,
                                    evaluate_idx_data_loader=test_idx_data_loader,
                                    evaluate_neg_edge_sampler=test_neg_edge_sampler,
                                    evaluate_data=self.test_data,
                                    num_neighbors=args.num_neighbors,
                                    f_1=f_1,
                                    time_gap=args.time_gap)                

            print('Epoch {}, Loss {:.4f}, Val AUC {:.4f}, PRC {:.4f}, RecK {:.4f}, F1 {:.4f}, test AUC {:.4f}, PRC {:.4f}, RecK {:.4f}, F1 {:.4f}'.format(
                e, np.mean(train_losses), val_score['AUROC'], val_score['AUPRC'], val_score['RecK'], val_score['F1'],
                test_score['AUROC'], test_score['AUPRC'], test_score['RecK'], test_score['F1']))     
                                                         
            if val_score[self.train_config['metric']] > self.best_score:
                self.patience_knt = 0
                self.best_score = val_score[self.train_config['metric']] 
              
            else:
                self.patience_knt += 1
                if self.patience_knt > self.model_config['patience']:
                    break  
                
        return test_score              
                
                                         
    def eval(self, model_name, model, neighbor_sampler, evaluate_idx_data_loader, evaluate_neg_edge_sampler, 
             evaluate_data, num_neighbors, f_1, time_gap):
        result = {}
        assert evaluate_neg_edge_sampler.seed is not None
        evaluate_neg_edge_sampler.reset_random_state()
        model[0].set_neighbor_sampler(neighbor_sampler)

        model.eval()
    
        all_predicts = []
        all_labels = []
        with torch.no_grad():
            evaluate_idx_data_loader_tqdm = tqdm(evaluate_idx_data_loader, ncols=120)
            for batch_idx, evaluate_data_indices in enumerate(evaluate_idx_data_loader_tqdm):
                evaluate_data_indices = evaluate_data_indices.numpy()
                batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids, batch_labels = \
                    evaluate_data.src_node_ids[evaluate_data_indices],  evaluate_data.dst_node_ids[evaluate_data_indices], \
                    evaluate_data.node_interact_times[evaluate_data_indices], evaluate_data.edge_ids[evaluate_data_indices],evaluate_data.labels[evaluate_data_indices]

                if model_name in ['TGAT', 'CAWN', 'TCL', 'SAD']:
                    batch_src_node_embeddings, batch_dst_node_embeddings = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                        dst_node_ids=batch_dst_node_ids,
                                                                        node_interact_times=batch_node_interact_times,
                                                                        num_neighbors=num_neighbors)
                elif model_name in ['GraphMixer']:
                    batch_src_node_embeddings, batch_dst_node_embeddings = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                        dst_node_ids=batch_dst_node_ids,
                                                                        node_interact_times=batch_node_interact_times,
                                                                        num_neighbors=num_neighbors,
                                                                        time_gap=time_gap)
                elif model_name in ['DyGFormer','FreeDyG']:
                    batch_src_node_embeddings, batch_dst_node_embeddings = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                        dst_node_ids=batch_dst_node_ids,
                                                                        node_interact_times=batch_node_interact_times)

                else:
                    raise ValueError(f"Wrong value for model_name {model_name}!")
                
                pos_logits = model[1](batch_src_node_embeddings)
                positive_probabilities = pos_logits.squeeze(dim=-1).sigmoid()
                predicts= positive_probabilities
                labels = torch.from_numpy(batch_labels).float().to(predicts.device)

                all_predicts.append(predicts)
                all_labels.append(labels)

        all_predicts = torch.cat(all_predicts)
        all_labels = torch.cat(all_labels)
        y_pred = all_predicts.cpu().detach()
        y_pred_labels = (y_pred > f_1).float()
        y_true = all_labels.cpu()

        k = int(y_true.sum().item())
        top_k_preds = y_pred.view(-1).cpu().argsort(descending=True)[:k] 
        
        rec = y_true[top_k_preds].sum().float() / k 
        auc = roc_auc_score(y_true.numpy(), y_pred.numpy())                
        prc = average_precision_score(y_true.numpy(), y_pred.numpy())
        f1 = f1_score(y_true.numpy(), y_pred_labels.numpy(), average='binary')

        result['AUROC'] = auc
        result['AUPRC'] = prc
        result['RecK'] = rec
        result['F1'] = f1

        best_threshold = 0
        best_f1 = 0

        for threshold in [i * 0.1 for i in range(2, 8)]: 
            y_pred_labels = (y_pred > threshold).float()
            current_f1 = f1_score(y_true.numpy(), y_pred_labels.numpy(), average='binary')
            if current_f1 > best_f1:
                best_f1 = current_f1
                best_threshold = threshold
                           
        return result, best_threshold     
            
            
class CTDGDetector2_Node(BaseDetector_Node):
    def __init__(self, train_config, model_config, node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data):
        super().__init__(train_config, model_config)
        self.model_name = model_config['model']
        self.node_raw_features = node_raw_features
        self.edge_raw_features = edge_raw_features
        self.full_data = full_data
        self.train_data = train_data
        self.val_data = val_data 
        self.test_data = test_data 

        self.train_neighbor_sampler = get_neighbor_sampler(data = self.train_data, sample_neighbor_strategy = args.sample_neighbor_strategy,
                                                  time_scaling_factor = args.time_scaling_factor, seed=0)
        self.full_neighbor_sampler = get_neighbor_sampler(data = self.full_data, sample_neighbor_strategy = args.sample_neighbor_strategy,
                                                 time_scaling_factor = args.time_scaling_factor, seed=1)

        src_node_mean_time_shift, src_node_std_time_shift, dst_node_mean_time_shift_dst, dst_node_std_time_shift = \
            compute_src_dst_node_time_shifts(self.train_data.src_node_ids, self.train_data.dst_node_ids, self.train_data.node_interact_times)

        if self.model_name in ['SLADE']:    
            backbone = SLADE(node_raw_features=self.node_raw_features, 
                                edge_raw_features=self.edge_raw_features, 
                                neighbor_sampler=self.train_neighbor_sampler,
                                time_feat_dim=args.time_feat_dim,
                                num_layers=args.num_layers, 
                                num_heads=args.num_heads,
                                dropout=args.dropout,
                                device=args.device)
        else:  
            backbone = MemoryModel(node_raw_features=self.node_raw_features, 
                                edge_raw_features=self.edge_raw_features, 
                                neighbor_sampler=self.train_neighbor_sampler,
                                time_feat_dim=args.time_feat_dim,
                                model_name= self.model_name, 
                                num_layers=args.num_layers, 
                                num_heads=args.num_heads,
                                dropout=args.dropout, 
                                src_node_mean_time_shift=src_node_mean_time_shift, 
                                src_node_std_time_shift=src_node_std_time_shift,
                                dst_node_mean_time_shift_dst=dst_node_mean_time_shift_dst, 
                                dst_node_std_time_shift=dst_node_std_time_shift, 
                                device=args.device)
                           
        self.scorer = NodeScorer(input_dim=self.node_raw_features.shape[1], hidden_dim=self.model_config['hidden_dim'], output_dim=1)
        
        if self.model_name in ['SLADE']:    
            self.model = nn.Sequential(backbone).to(self.device)
        else:   
            self.model = nn.Sequential(backbone, self.scorer).to(self.device)
    
    def train(self):
        if self.model_name in ['SLADE']:
            optimizer = torch.optim.Adam(self.model.parameters(),  lr=3e-6, weight_decay=0.0001)
        else:
            optimizer = torch.optim.Adam(self.model.parameters(),  lr=self.model_config['lr'])
        train_neg_edge_sampler = NegativeEdgeSampler(src_node_ids = self.train_data.src_node_ids, dst_node_ids = self.train_data.dst_node_ids)
        val_neg_edge_sampler = NegativeEdgeSampler(src_node_ids = self.full_data.src_node_ids, dst_node_ids = self.full_data.dst_node_ids, seed=0)
        test_neg_edge_sampler = NegativeEdgeSampler(src_node_ids = self.full_data.src_node_ids, dst_node_ids = self.full_data.dst_node_ids, seed=2)

        train_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(self.train_data.src_node_ids))), batch_size=self.train_config['batch_size'], shuffle=False)
        val_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(self.val_data.src_node_ids))), batch_size=self.train_config['batch_size'], shuffle=False)
        test_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(self.test_data.src_node_ids))), batch_size=self.train_config['batch_size'], shuffle=False)

        negative_train_nodes = torch.from_numpy(np.array(list(set(self.train_data.src_node_ids)|set(self.train_data.dst_node_ids)))).long().to(self.device) # for SLADE
                    
        for e in range(self.model_config['epochs']):
            self.model.train()
            
            if self.model_name in ['DyRep','TGN', 'SLADE']:
                self.model[0].set_neighbor_sampler(self.train_neighbor_sampler)
            self.model[0].memory_bank.__init_memory_bank__()
                                
            train_losses = []
            train_idx_data_loader_tqdm = tqdm(train_idx_data_loader, ncols=120)
            
            for batch_idx, train_data_indices in enumerate(train_idx_data_loader_tqdm):
                train_data_indices = train_data_indices.numpy()
                batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids, batch_labels = \
                    self.train_data.src_node_ids[train_data_indices], self.train_data.dst_node_ids[train_data_indices], \
                    self.train_data.node_interact_times[train_data_indices], self.train_data.edge_ids[train_data_indices], self.train_data.labels[train_data_indices]
                
                batch_src_neighbors,_,batch_src_neighbours_time = self.train_neighbor_sampler.get_historical_neighbors(batch_src_node_ids,batch_node_interact_times)
                batch_dst_neighbors,_,batch_dst_neighbours_time = self.train_neighbor_sampler.get_historical_neighbors(batch_dst_node_ids,batch_node_interact_times)

                if self.model_name in ['SLADE']:
                    _, _, _, _, loss = self.model[0].compute_node_diff_score(batch_src_node_ids,
                                                                    batch_dst_node_ids,
                                                                    batch_node_interact_times,
                                                                    batch_src_neighbors, 
                                                                    batch_dst_neighbors,
                                                                    batch_src_neighbours_time,
                                                                    batch_dst_neighbours_time,
                                                                    20,
                                                                    negative_train_nodes) 
                else:
                    
                    batch_src_node_embeddings, batch_dst_node_embeddings, _, _ =  \
                        self.model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                            dst_node_ids=batch_dst_node_ids,
                                                                            node_interact_times=batch_node_interact_times,
                                                                            edge_ids=batch_edge_ids,
                                                                            edges_are_positive=True,
                                                                            num_neighbors=args.num_neighbors)
                        
                    pos_logits = self.model[1](batch_src_node_embeddings)
                    pos_logits = pos_logits.squeeze(dim=-1)

                    loss = F.binary_cross_entropy_with_logits(pos_logits, torch.from_numpy(batch_labels).float().to(self.device))          
                
                train_losses.append(loss.item())

                optimizer.zero_grad() 
                loss.backward()  
                optimizer.step()     
                
                train_idx_data_loader_tqdm.set_description(f'Epoch: {e}, train for the {batch_idx + 1}-th batch, train loss: {loss.item()}')           
                self.model[0].memory_bank.detach_memory_bank()
            
            if self.model_name in ['SLADE']:
                val_score, f_1 = self.eval_SLADE(model_name=self.model_name,
                                        model=self.model,
                                        neighbor_sampler=self.full_neighbor_sampler,
                                        evaluate_idx_data_loader=val_idx_data_loader,
                                        evaluate_neg_edge_sampler=val_neg_edge_sampler,
                                        evaluate_data=self.val_data,
                                        num_neighbors=args.num_neighbors,
                                        f_1 = 0.5)      
                                                
                if val_score[self.train_config['metric']] > self.best_score:
                    self.patience_knt = 0
                    self.best_score = val_score[self.train_config['metric']]       
                    test_score, _ = self.eval_SLADE(model_name=self.model_name,
                                            model=self.model,
                                            neighbor_sampler=self.full_neighbor_sampler,
                                            evaluate_idx_data_loader=test_idx_data_loader,
                                            evaluate_neg_edge_sampler=test_neg_edge_sampler,
                                            evaluate_data=self.test_data,
                                            num_neighbors=args.num_neighbors,
                                            f_1 = f_1)                
                    print('Epoch {}, Loss {:.4f}, Val AUC {:.4f}, PRC {:.4f}, RecK {:.4f}, F1 {:.4f}, test AUC {:.4f}, PRC {:.4f}, RecK {:.4f}, F1 {:.4f}'.format(
                        e, np.mean(train_losses), val_score['AUROC'], val_score['AUPRC'], val_score['RecK'], val_score['F1'],
                        test_score['AUROC'], test_score['AUPRC'], test_score['RecK'], test_score['F1']))
                else:
                    self.patience_knt += 1
                    if self.patience_knt > self.model_config['patience']:
                        break  
            else:
                train_backup_memory_bank = self.model[0].memory_bank.backup_memory_bank()    
                
                val_score, f_1 = self.eval(model_name=self.model_name,
                                        model=self.model,
                                        neighbor_sampler=self.full_neighbor_sampler,
                                        evaluate_idx_data_loader=val_idx_data_loader,
                                        evaluate_neg_edge_sampler=val_neg_edge_sampler,
                                        evaluate_data=self.val_data,
                                        num_neighbors=args.num_neighbors,
                                        f_1 = 0.5)
                
                val_backup_memory_bank = self.model[0].memory_bank.backup_memory_bank()
                self.model[0].memory_bank.reload_memory_bank(train_backup_memory_bank)       

                test_score, _ = self.eval(model_name=self.model_name,
                                        model=self.model,
                                        neighbor_sampler=self.full_neighbor_sampler,
                                        evaluate_idx_data_loader=test_idx_data_loader,
                                        evaluate_neg_edge_sampler=test_neg_edge_sampler,
                                        evaluate_data=self.test_data,
                                        num_neighbors=args.num_neighbors,
                                        f_1 = f_1)                
                self.model[0].memory_bank.reload_memory_bank(val_backup_memory_bank)  
                print('Epoch {}, Loss {:.4f}, Val AUC {:.4f}, PRC {:.4f}, RecK {:.4f}, F1 {:.4f}, test AUC {:.4f}, PRC {:.4f}, RecK {:.4f}, F1 {:.4f}'.format(
                    e, np.mean(train_losses), val_score['AUROC'], val_score['AUPRC'], val_score['RecK'], val_score['F1'],
                    test_score['AUROC'], test_score['AUPRC'], test_score['RecK'], test_score['F1']))
                                                                   
                if val_score[self.train_config['metric']] > self.best_score:
                    self.patience_knt = 0
                    self.best_score = val_score[self.train_config['metric']]       
                else:
                    self.patience_knt += 1
                    if self.patience_knt > self.model_config['patience']:
                        break  
                
        return test_score                

    def eval_SLADE(self, model_name, model, neighbor_sampler, evaluate_idx_data_loader, evaluate_neg_edge_sampler, evaluate_data, num_neighbors, f_1):
        result = {}

        model[0].set_neighbor_sampler(neighbor_sampler)

        model.eval()
    
        all_predicts = []
        all_labels = []

        with torch.no_grad():
            evaluate_idx_data_loader_tqdm = tqdm(evaluate_idx_data_loader, ncols=120)
            for batch_idx, evaluate_data_indices in enumerate(evaluate_idx_data_loader_tqdm):
                evaluate_data_indices = evaluate_data_indices.numpy()
                batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids, batch_labels = \
                                        evaluate_data.src_node_ids[evaluate_data_indices],  \
                                        evaluate_data.dst_node_ids[evaluate_data_indices], \
                                        evaluate_data.node_interact_times[evaluate_data_indices], \
                                        evaluate_data.edge_ids[evaluate_data_indices],\
                                        evaluate_data.labels[evaluate_data_indices]
            
                src_neighbors_batch_np, _, src_neighbors_time_batch_np = model[0].neighbor_sampler.get_historical_neighbors(batch_src_node_ids, batch_node_interact_times, num_neighbors=20)
                dst_neighbors_batch_np, _, dst_neighbors_time_batch_np = model[0].neighbor_sampler.get_historical_neighbors(batch_dst_node_ids, batch_node_interact_times, num_neighbors=20)
                src_neighbors_batch = torch.from_numpy(src_neighbors_batch_np).long().to(self.device)
                dst_neighbors_batch = torch.from_numpy(dst_neighbors_batch_np).long().to(self.device)
                src_neighbors_time_batch = torch.from_numpy(src_neighbors_time_batch_np).long().to(self.device)
                dst_neighbors_time_batch = torch.from_numpy(dst_neighbors_time_batch_np).long().to(self.device)

                sources_batch = torch.from_numpy(batch_src_node_ids).long().to(self.device)
                destinations_batch = torch.from_numpy(batch_dst_node_ids).long().to(self.device)
                timestamps_batch = torch.from_numpy(batch_node_interact_times).float().to(self.device)               

                positive_memory_score, drift_score, _, _ = model[0].compute_anomaly_score(sources_batch, destinations_batch,
                                                                timestamps_batch, src_neighbors_batch, dst_neighbors_batch, 
                                                                src_neighbors_time_batch, dst_neighbors_time_batch, n_neighbors=20)
                labels = torch.from_numpy(batch_labels).float().to(self.device)
                predicts = (-(drift_score.reshape(-1)) -(positive_memory_score.reshape(-1)) + 2) / 4
                
                all_predicts.append(predicts)
                all_labels.append(labels)

        all_predicts = torch.cat(all_predicts)
        all_labels = torch.cat(all_labels)
        y_pred = all_predicts.cpu().detach()
        y_pred_labels = (y_pred > f_1).float()
        y_true = all_labels.cpu()

        k = int(y_true.sum().item())
        top_k_preds = y_pred.view(-1).cpu().argsort(descending=True)[:k] 
        
        rec = y_true[top_k_preds].sum().float() / k 
        auc = roc_auc_score(y_true.numpy(), y_pred.numpy())                
        prc = average_precision_score(y_true.numpy(), y_pred.numpy())
        f1 = f1_score(y_true.numpy(), y_pred_labels.numpy(), average='binary')

        result['AUROC'] = auc
        result['AUPRC'] = prc
        result['RecK'] = rec
        result['F1'] = f1
            
        best_threshold = 0
        best_f1 = 0

        for threshold in [i * 0.1 for i in range(2, 8)]: 
            y_pred_labels = (y_pred > threshold).float()
            current_f1 = f1_score(y_true.numpy(), y_pred_labels.numpy(), average='binary')
            if current_f1 > best_f1:
                best_f1 = current_f1
                best_threshold = threshold
                
        return result, best_threshold 
    
                                         
    def eval(self, model_name, model, neighbor_sampler, evaluate_idx_data_loader, evaluate_neg_edge_sampler, evaluate_data, num_neighbors, f_1):
        result = {}
        assert evaluate_neg_edge_sampler.seed is not None
        evaluate_neg_edge_sampler.reset_random_state()

        if model_name in ['DyRep', 'TGN']:
            model[0].set_neighbor_sampler(neighbor_sampler)

        model.eval()
    
        all_predicts = []
        all_labels = []

        with torch.no_grad():
            evaluate_idx_data_loader_tqdm = tqdm(evaluate_idx_data_loader, ncols=120)
            for batch_idx, evaluate_data_indices in enumerate(evaluate_idx_data_loader_tqdm):
                evaluate_data_indices = evaluate_data_indices.numpy()
                batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids, batch_labels = \
                    evaluate_data.src_node_ids[evaluate_data_indices],  evaluate_data.dst_node_ids[evaluate_data_indices], \
                    evaluate_data.node_interact_times[evaluate_data_indices], evaluate_data.edge_ids[evaluate_data_indices],evaluate_data.labels[evaluate_data_indices]

                batch_src_node_embeddings, batch_dst_node_embeddings,_,_ = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                    dst_node_ids=batch_dst_node_ids,
                                                                    node_interact_times=batch_node_interact_times,
                                                                    edge_ids=batch_edge_ids,
                                                                    edges_are_positive=True,
                                                                    num_neighbors=num_neighbors)
                    
                positive_probabilities = model[1](batch_src_node_embeddings)
                predicts= positive_probabilities.squeeze(dim=-1).sigmoid()
                labels = torch.from_numpy(batch_labels).float().to(predicts.device)

                all_predicts.append(predicts)
                all_labels.append(labels)

        all_predicts = torch.cat(all_predicts)
        all_labels = torch.cat(all_labels)
        y_pred = all_predicts.cpu().detach()
        y_pred_labels = (y_pred > f_1).float()
        y_true = all_labels.cpu()

        k = int(y_true.sum().item())
        top_k_preds = y_pred.view(-1).cpu().argsort(descending=True)[:k] 
        
        rec = y_true[top_k_preds].sum().float() / k 
        auc = roc_auc_score(y_true.numpy(), y_pred.numpy())                
        prc = average_precision_score(y_true.numpy(), y_pred.numpy())
        f1 = f1_score(y_true.numpy(), y_pred_labels.numpy(), average='binary')

        result['AUROC'] = auc
        result['AUPRC'] = prc
        result['RecK'] = rec
        result['F1'] = f1
            
        best_threshold = 0
        best_f1 = 0

        for threshold in [i * 0.1 for i in range(2, 8)]: 
            y_pred_labels = (y_pred > threshold).float()
            current_f1 = f1_score(y_true.numpy(), y_pred_labels.numpy(), average='binary')
            if current_f1 > best_f1:
                best_f1 = current_f1
                best_threshold = threshold
                
        return result, best_threshold 
    