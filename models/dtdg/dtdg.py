import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric_temporal.nn.recurrent import MPNNLSTM, DyGrEncoder, EvolveGCNO, EvolveGCNH, TGCN


# Discrete-time Dynamic GNNs
class Dy_GrAE(torch.nn.Module):
    def __init__(self, input_dim:int, hidden_dim:int, num_layers, **kwargs):
        super().__init__()
        self.recurrent = DyGrEncoder(input_dim, num_layers, 'mean', hidden_dim, 1)

    def forward(self, x, edge_index, h_0, c_0 , edge_weight=None):
        h, h_0, c_0  = self.recurrent(x, edge_index, edge_weight, h_0, c_0)     
        
        return h, h_0, c_0
       
    
class EvolveGCN_H(torch.nn.Module):    
    def __init__(self, input_dim: int, hidden_dim:int, num_nodes, **kwargs):
        super().__init__()
        self.recurrent = EvolveGCNH(num_nodes, input_dim)
        self.linear1 = nn.Linear(input_dim, input_dim) 
        self.linear2 = nn.Linear(input_dim, hidden_dim) 
             
    def forward(self, x, edge_index) -> torch.FloatTensor:
        h = self.recurrent(x, edge_index)
        h = F.relu(h)
        h = self.linear1(h)
        h = F.relu(h)
        h = self.linear2(h)
        
        return h
    

class EvolveGCN_O(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim:int, **kwargs):
        super().__init__()
        self.recurrent1 = EvolveGCNO(input_dim)
        self.recurrent2 = EvolveGCNO(input_dim)
        self.linear = nn.Linear(input_dim, hidden_dim) 
        
    def forward(self, x, edge_index) -> torch.FloatTensor:
        h = self.recurrent1(x, edge_index)
        h = F.relu(h)
        h = self.recurrent2(x, edge_index)
        h = F.relu(h)
        h = self.linear(h)

        return h


class MPNN_LSTM(torch.nn.Module):    
    def __init__(self, input_dim: int, num_nodes:int, hidden_dim:int, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        self.recurrent = MPNNLSTM(input_dim, hidden_dim, num_nodes, 1, 0.1)
        self.linear = torch.nn.Linear(self.hidden_dim*2 + self.input_dim, self.hidden_dim)
        
    def forward(self, x, edge_index, edge_weight=None) -> torch.FloatTensor:
        h = self.recurrent(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear(h)
        h = F.relu(h)
        
        return h
     

class T_GCN(torch.nn.Module): 
    def __init__(self, input_dim: int, hidden_dim:int, **kwargs):
        super().__init__()
        self.recurrent1 = TGCN(input_dim, hidden_dim)
        self.recurrent2 = TGCN(hidden_dim, hidden_dim)
        
    def forward(self, x, edge_index) -> torch.FloatTensor:
        h = self.recurrent1(x, edge_index)
        h = F.relu(h)
        h = self.recurrent2(h, edge_index)
        
        return h
            
            
class CAB(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(CAB, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.attn = nn.MultiheadAttention(out_dim, num_heads=2)
    
    def forward(self, history_embeddings):
        """history_embeddings: Tensor of shape (seq_len, batch_size, embed_dim)"""
        transformed = self.fc(history_embeddings)
        attn_output, _ = self.attn(transformed, transformed, transformed)
        output = attn_output.transpose(0, 1)
        return output


class AddGraph(nn.Module):
    def __init__(self, hidden_dim, w=3, **kwargs):
        super(AddGraph, self).__init__()
        self.gcn = GCNConv(hidden_dim, hidden_dim)
        self.cab = CAB(hidden_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.w = w
    
    def forward(self, hist_embs, edge_index):
        curr_emb = self.gcn(hist_embs[-1], edge_index) 
        if len(hist_embs)>= self.w:
            cab_input = hist_embs[-self.w:]
        else:
            seq_len = len(hist_embs)
            cab_input = hist_embs[-min(self.w, seq_len):]
            
        cab_input = torch.stack(cab_input, dim=0)    
        short_emb = self.cab(cab_input)   
        
        _, h_t = self.gru(short_emb, curr_emb.unsqueeze(0)) 
        return h_t.squeeze(0)       
        