import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm
from detectors.utils import *

from models.ctdg.models.TGAT import TGAT
from models.ctdg.models.MemoryModel import MemoryModel, compute_src_dst_node_time_shifts
from models.ctdg.models.CAWN import CAWN
from models.ctdg.models.TCL import TCL
from models.ctdg.models.GraphMixer import GraphMixer
from models.ctdg.models.DyGFormer import DyGFormer
from models.ctdg.models.FreeDyG import FreeDyG
from models.ctdg.models.SAD import GDN
from models.ctdg.models.SLADE import SLADE
from models.ctdg.utils.utils import get_neighbor_sampler, NegativeEdgeSampler
from load_config import args
from models.ctdg.models.modules import Scorer
from models.ctdg.utils.DataLoader import get_idx_data_loader
import wandb


def log_epoch(epoch, train_losses, val_score, test_score=None):
    # only logs when benchmark.py started a run with --wandb
    if wandb.run is None:
        return
    log = {"epoch": epoch, "train/loss": np.mean(train_losses)}
    log.update({f"val/{k}": v for k, v in val_score.items()})
    if test_score is not None:
        log.update({f"test/{k}": v for k, v in test_score.items()})
    wandb.log(log)



class CTDGDetector(BaseDetector):
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

        print("\n========== TRAIN CONFIG ==========")
        for k, v in self.train_config.items():
            print(f"{k}: {v}")
        print("========== MODEL CONFIG ==========")
        for k, v in self.model_config.items():
            print(f"{k}: {v}")
        print("=================================")     
                
        if self.model_name in ['TGAT', 'SAD']:
            backbone = TGAT(node_raw_features=self.node_raw_features, edge_raw_features=self.edge_raw_features, neighbor_sampler=self.train_neighbor_sampler,
                            time_feat_dim=args.time_feat_dim, num_layers=args.num_layers, num_heads=args.num_heads, dropout=model_config['dropout'], device=train_config['device'])
        elif self.model_name == 'CAWN':
            backbone = CAWN(node_raw_features=self.node_raw_features, edge_raw_features=self.edge_raw_features, neighbor_sampler=self.train_neighbor_sampler,
                                    time_feat_dim=args.time_feat_dim, position_feat_dim=args.position_feat_dim, walk_length=args.walk_length,
                                    num_walk_heads=args.num_walk_heads, dropout=model_config['dropout'], device=train_config['device'])
        elif self.model_name == 'TCL':
            backbone = TCL(node_raw_features=self.node_raw_features, edge_raw_features=self.edge_raw_features, neighbor_sampler=self.train_neighbor_sampler,
                                   time_feat_dim=args.time_feat_dim, num_layers=args.num_layers, num_heads=args.num_heads,
                                   num_depths=args.num_neighbors + 1, dropout=model_config['dropout'], device=train_config['device'])
        elif self.model_name == 'GraphMixer':
            backbone = GraphMixer(node_raw_features=self.node_raw_features, edge_raw_features=self.edge_raw_features, neighbor_sampler=self.train_neighbor_sampler,
                                          time_feat_dim=args.time_feat_dim, num_tokens=args.num_neighbors, num_layers=args.num_layers, dropout=model_config['dropout'], 
                                          device=train_config['device'])
        elif self.model_name == 'DyGFormer':
            backbone = DyGFormer(node_raw_features=self.node_raw_features, edge_raw_features=self.edge_raw_features, neighbor_sampler=self.train_neighbor_sampler,
                                         time_feat_dim=args.time_feat_dim, channel_embedding_dim=args.channel_embedding_dim, patch_size=args.patch_size,
                                         num_layers=args.num_layers, num_heads=args.num_heads, dropout=model_config['dropout'],
                                         max_input_sequence_length=args.max_input_sequence_length, device=train_config['device'])
        elif self.model_name == 'FreeDyG':
            backbone = FreeDyG(node_raw_features=self.node_raw_features, edge_raw_features=self.edge_raw_features, neighbor_sampler=self.train_neighbor_sampler,
                                         time_feat_dim=args.time_feat_dim, channel_embedding_dim=args.channel_embedding_dim,
                                         num_layers=args.num_layers, dropout=model_config['dropout'], max_input_sequence_length=args.max_input_sequence_length, 
                                         device=train_config['device'])

        gdn = GDN(self.device)     ##only for SAD
        self.scorer = Scorer(input_dim=self.node_raw_features.shape[1], hidden_dim=self.model_config['hidden_dim'], output_dim=1)
        
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
            train_idx_data_loader_tqdm = tqdm(train_idx_data_loader, ncols=120, leave=False)
            
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
                pos_logits, edge_embs = self.model[1](input_1=batch_src_node_embeddings, input_2=batch_dst_node_embeddings)
                pos_logits = pos_logits.squeeze(dim=-1)

                if self.train_config['label'] == False:
                    neg_logits, _ = self.model[1](input_1=batch_neg_src_node_embeddings, input_2=batch_neg_dst_node_embeddings)
                    pos_logits = pos_logits.squeeze(dim=-1)
                    
                    if self.train_config['loss'] == 'bce':
                        pos_loss = F.binary_cross_entropy_with_logits(pos_logits, torch.zeros_like(pos_logits).cuda().to(self.device)) 
                        neg_loss = F.binary_cross_entropy_with_logits(neg_logits, torch.ones_like(neg_logits).cuda().to(self.device)) 
                        loss = pos_loss + neg_loss
                    else:
                        loss = self.pairwise_loss(pos_logits, neg_logits)

                else:
                    if self.model_name in ['SAD']:
                        labels = torch.tensor(batch_labels, dtype=torch.float32, device=self.device)
                        time = torch.tensor(batch_node_interact_times, dtype=torch.float32, device=self.device)
        
                        gdn_score = self.model[2](edge_embs, time, labels)  
                        dev, group = self.model[2].dev_diff(torch.squeeze(gdn_score), time, labels)
                        loss_anomaly = torch.Tensor(0).to(self.device)
                        loss_supc = torch.Tensor(0).to(self.device)
                        alpha = 1e-1  # 1e-1
                        beta = 1e-3 # 1e-3
                        loss_anomaly = self.model[2].dev_loss(torch.squeeze(labels), torch.squeeze(gdn_score), torch.squeeze(time))
                        loss_supc = self.model[2].suploss(edge_embs, self.device, group, dev)
                        loss = F.binary_cross_entropy_with_logits(pos_logits, torch.from_numpy(batch_labels).float().to(self.device)) + alpha * loss_anomaly
                        loss += alpha * loss_anomaly + beta * loss_supc
                    else:
                        loss = F.binary_cross_entropy_with_logits(pos_logits, torch.from_numpy(batch_labels).float().to(self.device)) 
                                       
                train_losses.append(loss.item())

                optimizer.zero_grad() 
                loss.backward()  
                optimizer.step()  
                 
                train_idx_data_loader_tqdm.set_description(f'Epoch: {e}, train for the {batch_idx + 1}-th batch, train loss: {loss.item()}')           

                            
            val_y, val_probs, = self.prob(model_name=self.model_name,
                                        model=self.model,
                                        neighbor_sampler=self.full_neighbor_sampler,
                                        evaluate_idx_data_loader=val_idx_data_loader,
                                        evaluate_neg_edge_sampler=val_neg_edge_sampler,
                                        evaluate_data=self.val_data,
                                        num_neighbors=args.num_neighbors,
                                        time_gap=args.time_gap)       
            val_score, f_1  = self.eval(val_y, val_probs, f_1=0.5)

            test_y, test_probs = self.prob(model_name=self.model_name,
                                        model=self.model,
                                        neighbor_sampler=self.full_neighbor_sampler,
                                        evaluate_idx_data_loader=test_idx_data_loader,
                                        evaluate_neg_edge_sampler=test_neg_edge_sampler,
                                        evaluate_data=self.test_data,
                                        num_neighbors=args.num_neighbors,
                                        time_gap=args.time_gap)                
            test_score, f_1  = self.eval(test_y, test_probs, f_1=f_1)
            
            print('Epoch {}, Loss {:.4f}, Val AUC {:.4f}, PRC {:.4f}, RecK {:.4f}, F1 {:.4f}, test AUC {:.4f}, PRC {:.4f}, RecK {:.4f}, F1 {:.4f}'.format(
                e, np.mean(train_losses), val_score['AUROC'], val_score['AUPRC'], val_score['RecK'], val_score['F1'],
                test_score['AUROC'], test_score['AUPRC'], test_score['RecK'], test_score['F1']))
            log_epoch(e, train_losses, val_score, test_score)

            if val_score[self.train_config['metric']] > self.best_score:
                self.patience_knt = 0
                self.best_score = val_score[self.train_config['metric']] 
                 
                
            else:
                self.patience_knt += 1
                if self.patience_knt > self.model_config['patience']:
                    break  

        return test_score    

                          
    def prob(self, model_name, model, neighbor_sampler, evaluate_idx_data_loader, evaluate_neg_edge_sampler, 
             evaluate_data, num_neighbors, time_gap):
        
        assert evaluate_neg_edge_sampler.seed is not None
        evaluate_neg_edge_sampler.reset_random_state()
        model[0].set_neighbor_sampler(neighbor_sampler)

        model.eval()
    
        probs_ls, y_ls,  = [], []
        
        with torch.no_grad():
            evaluate_idx_data_loader_tqdm = tqdm(evaluate_idx_data_loader, ncols=120, leave=False)
            for batch_idx, evaluate_data_indices in enumerate(evaluate_idx_data_loader_tqdm):
                evaluate_data_indices = evaluate_data_indices.numpy()
                batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids, batch_labels = \
                                                                evaluate_data.src_node_ids[evaluate_data_indices],  \
                                                                evaluate_data.dst_node_ids[evaluate_data_indices], \
                                                                evaluate_data.node_interact_times[evaluate_data_indices], \
                                                                evaluate_data.edge_ids[evaluate_data_indices], \
                                                                evaluate_data.labels[evaluate_data_indices]

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
                
                logits, _ = model[1](input_1=batch_src_node_embeddings, input_2=batch_dst_node_embeddings)
                logits = logits.squeeze(dim=-1)           
                probs = torch.sigmoid(logits)
                y = torch.from_numpy(batch_labels).float().to(probs.device)

                probs_ls.append(probs)
                y_ls.append(y)
            
        probs = torch.cat(probs_ls, dim=0)  
        y = torch.cat(y_ls, dim=0) 
        
        return y, probs 
    
                        

class CTDGDetector2(BaseDetector):
    def __init__(self, train_config, model_config, node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data):
        super().__init__(train_config, model_config)
        self.model_name = train_config['model']
        self.node_raw_features = node_raw_features
        self.edge_raw_features = edge_raw_features
        self.full_data = full_data
        self.train_data = train_data
        self.val_data = val_data 
        self.test_data = test_data 

        self.train_neighbor_sampler = get_neighbor_sampler(data = self.train_data, 
                                                           sample_neighbor_strategy = args.sample_neighbor_strategy,
                                                           time_scaling_factor = args.time_scaling_factor, seed=0)
        self.full_neighbor_sampler = get_neighbor_sampler(data = self.full_data, 
                                                          sample_neighbor_strategy = args.sample_neighbor_strategy,
                                                          time_scaling_factor = args.time_scaling_factor, seed=1)

        print("\n========== TRAIN CONFIG ==========")
        for k, v in self.train_config.items():
            print(f"{k}: {v}")
        print("========== MODEL CONFIG ==========")
        for k, v in self.model_config.items():
            print(f"{k}: {v}")  
        
        src_node_mean_time_shift, src_node_std_time_shift, dst_node_mean_time_shift_dst, dst_node_std_time_shift = \
            compute_src_dst_node_time_shifts(self.train_data.src_node_ids, self.train_data.dst_node_ids, self.train_data.node_interact_times)
        
        if self.model_name in ['SLADE']:    
            backbone = SLADE(node_raw_features=self.node_raw_features, 
                                edge_raw_features=self.edge_raw_features, 
                                neighbor_sampler=self.train_neighbor_sampler,
                                time_feat_dim=args.time_feat_dim,
                                num_layers=args.num_layers, 
                                num_heads=args.num_heads,
                                dropout=model_config['dropout'],
                                device=train_config['device'])
        else:
            backbone = MemoryModel(node_raw_features=self.node_raw_features, 
                                edge_raw_features=self.edge_raw_features, 
                                neighbor_sampler=self.train_neighbor_sampler,
                                time_feat_dim=args.time_feat_dim,
                                model_name= self.model_name, 
                                num_layers=args.num_layers, 
                                num_heads=args.num_heads,
                                dropout=model_config['dropout'], 
                                src_node_mean_time_shift=src_node_mean_time_shift, 
                                src_node_std_time_shift=src_node_std_time_shift,
                                dst_node_mean_time_shift_dst=dst_node_mean_time_shift_dst, 
                                dst_node_std_time_shift=dst_node_std_time_shift, 
                                device=train_config['device'])
                           
        self.scorer = Scorer(input_dim=self.node_raw_features.shape[1], hidden_dim=self.node_raw_features.shape[1], output_dim=1)
        
        self.model = nn.Sequential(backbone, self.scorer).to(self.device)
    
    def train(self):
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
            
            if self.model_name in ['DyRep','TGN']:
                self.model[0].set_neighbor_sampler(self.train_neighbor_sampler)
            self.model[0].memory_bank.__init_memory_bank__()
                                
            train_losses = []
            train_idx_data_loader_tqdm = tqdm(train_idx_data_loader, ncols=120, leave=False)
            
            for batch_idx, train_data_indices in enumerate(train_idx_data_loader_tqdm):
                train_data_indices = train_data_indices.numpy()
                
                batch_src_node_ids = self.train_data.src_node_ids[train_data_indices]
                batch_dst_node_ids = self.train_data.dst_node_ids[train_data_indices]
                batch_node_interact_times = self.train_data.node_interact_times[train_data_indices]
                batch_edge_ids = self.train_data.edge_ids[train_data_indices]
                batch_labels = self.train_data.labels[train_data_indices]
                
                batch_src_neighbors,_,batch_src_neighbours_time = self.train_neighbor_sampler.get_historical_neighbors(batch_src_node_ids,batch_node_interact_times)
                batch_dst_neighbors,_,batch_dst_neighbours_time = self.train_neighbor_sampler.get_historical_neighbors(batch_dst_node_ids,batch_node_interact_times)

                                   
                if self.train_config['label'] == False:
                    _, batch_neg_dst_node_ids = train_neg_edge_sampler.sample(size=len(batch_src_node_ids))
                    batch_neg_src_node_ids = batch_src_node_ids
                    
                    batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                        self.model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                        dst_node_ids=batch_neg_dst_node_ids,
                                                                        node_interact_times=batch_node_interact_times,
                                                                        edge_ids=None,
                                                                        edges_are_positive=False,
                                                                        num_neighbors=args.num_neighbors)
                    
                batch_src_node_embeddings, batch_dst_node_embeddings,_,_ =  \
                    self.model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                        dst_node_ids=batch_dst_node_ids,
                                                                        node_interact_times=batch_node_interact_times,
                                                                        edge_ids=batch_edge_ids,
                                                                        edges_are_positive=True,
                                                                        num_neighbors=args.num_neighbors)               

                pos_logits, _ = self.model[1](input_1=batch_src_node_embeddings, input_2=batch_dst_node_embeddings)
                pos_logits = pos_logits.squeeze(dim=-1)                
                
                if self.train_config['label'] == False: 
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
                        neg_logits, _ = self.model[1](input_1=batch_neg_src_node_embeddings, input_2=batch_neg_dst_node_embeddings)
                        neg_logits = neg_logits.squeeze(dim=-1)
                        
                        if self.train_config['loss'] == 'bce':
                            pos_loss = F.binary_cross_entropy_with_logits(pos_logits, torch.zeros_like(pos_logits).to(self.device)) 
                            neg_loss = F.binary_cross_entropy_with_logits(neg_logits, torch.ones_like(neg_logits).to(self.device)) 
                            loss = pos_loss + neg_loss
                        else:
                            loss = self.pairwise_loss(pos_logits, neg_logits)
                else:
                    loss = F.binary_cross_entropy_with_logits(pos_logits, torch.from_numpy(batch_labels).float().to(self.device))          
                
                train_losses.append(loss.item())

                optimizer.zero_grad() 
                loss.backward()  
                optimizer.step()     
                
                train_idx_data_loader_tqdm.set_description(f'Epoch: {e}, train for the {batch_idx + 1}-th batch, train loss: {loss.item()}')           
                self.model[0].memory_bank.detach_memory_bank()
                
            train_backup_memory_bank = self.model[0].memory_bank.backup_memory_bank()    
            
            val_y, val_probs = self.prob(model_name=self.model_name,
                                        model=self.model,
                                        neighbor_sampler=self.full_neighbor_sampler,
                                        evaluate_idx_data_loader=val_idx_data_loader,
                                        evaluate_neg_edge_sampler=val_neg_edge_sampler,
                                        evaluate_data=self.val_data,
                                        num_neighbors=args.num_neighbors)
            val_score, f_1  = self.eval(val_y, val_probs, f_1=0.5)
            
            val_backup_memory_bank = self.model[0].memory_bank.backup_memory_bank()
            self.model[0].memory_bank.reload_memory_bank(train_backup_memory_bank)       
                                              
            if val_score[self.train_config['metric']] > self.best_score:
                self.patience_knt = 0
                self.best_score = val_score[self.train_config['metric']]      
                 
                test_y, test_probs  = self.prob(model_name=self.model_name,
                                            model=self.model,
                                            neighbor_sampler=self.full_neighbor_sampler,
                                            evaluate_idx_data_loader=test_idx_data_loader,
                                            evaluate_neg_edge_sampler=test_neg_edge_sampler,
                                            evaluate_data=self.test_data,
                                            num_neighbors=args.num_neighbors)  
                              
                test_score, f_1  = self.eval(test_y, test_probs, f_1=f_1)
                                
                self.model[0].memory_bank.reload_memory_bank(val_backup_memory_bank)  
                print('Epoch {}, Loss {:.4f}, Val AUC {:.4f}, PRC {:.4f}, RecK {:.4f}, F1 {:.4f}, test AUC {:.4f}, PRC {:.4f}, RecK {:.4f}, F1 {:.4f}'.format(
                    e, np.mean(train_losses), val_score['AUROC'], val_score['AUPRC'], val_score['RecK'], val_score['F1'],
                    test_score['AUROC'], test_score['AUPRC'], test_score['RecK'], test_score['F1']))
                log_epoch(e, train_losses, val_score, test_score)
            else:
                # test set is only evaluated when validation improves
                log_epoch(e, train_losses, val_score)
                self.patience_knt += 1
                if self.patience_knt > self.model_config['patience']:
                    break  

        return test_score                
                
                                             
    def prob(self, model_name, model, neighbor_sampler, evaluate_idx_data_loader, evaluate_neg_edge_sampler, 
             evaluate_data, num_neighbors):
        
        assert evaluate_neg_edge_sampler.seed is not None
        evaluate_neg_edge_sampler.reset_random_state()

        if model_name in ['DyRep', 'TGN']:
            model[0].set_neighbor_sampler(neighbor_sampler)

        model.eval()
    
        probs_ls, y_ls, = [], []

        with torch.no_grad():
            evaluate_idx_data_loader_tqdm = tqdm(evaluate_idx_data_loader, ncols=120, leave=False)
            for batch_idx, evaluate_data_indices in enumerate(evaluate_idx_data_loader_tqdm):
                evaluate_data_indices = evaluate_data_indices.numpy()
                batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids, batch_labels = \
                                                            evaluate_data.src_node_ids[evaluate_data_indices],  \
                                                            evaluate_data.dst_node_ids[evaluate_data_indices], \
                                                            evaluate_data.node_interact_times[evaluate_data_indices], \
                                                            evaluate_data.edge_ids[evaluate_data_indices], \
                                                            evaluate_data.labels[evaluate_data_indices]

                batch_src_node_embeddings, batch_dst_node_embeddings, _, _ = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                    dst_node_ids=batch_dst_node_ids,
                                                                    node_interact_times=batch_node_interact_times,
                                                                    edge_ids=batch_edge_ids,
                                                                    edges_are_positive=True,
                                                                    num_neighbors=num_neighbors)
                    
                logits, _ = model[1](input_1=batch_src_node_embeddings, input_2=batch_dst_node_embeddings)
                logits = logits.squeeze(dim=-1)
                probs = torch.sigmoid(logits)
                y = torch.from_numpy(batch_labels).float().to(probs.device)

                probs_ls.append(probs)
                y_ls.append(y)
                
        probs = torch.cat(probs_ls, dim=0)  
        y = torch.cat(y_ls, dim=0) 
     
        return y, probs
    
    
 