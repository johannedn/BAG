import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, GATConv, ChebConv, SGConv, TransformerConv


# Static GNNS
class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout,
                 activation='ReLU', **kwargs):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GCNConv(input_dim, hidden_dim))
        for _ in range(1, num_layers):
            self.layers.append(GCNConv(hidden_dim, hidden_dim))
        self.act = getattr(nn, activation)()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
    def forward(self, x, edge_index):            
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if i < len(self.layers) - 1:  
                x = self.act(x)
                x = self.dropout(x)
        return x


class GCN_noparam(nn.Module):
    def __init__(self, input_dim, num_layers, dropout, activation='ReLU', **kwargs):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(GCNConv(input_dim, input_dim, add_self_loops=True))
        self.act = getattr(nn, activation)() if activation else nn.Identity()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x, edge_index):
        h_final = x.detach().clone() 
        for i, layer in enumerate(self.layers):
            h = layer(x, edge_index)  
            if i < len(self.layers) - 1:
                h = self.act(h)  
                h = self.dropout(h) 
            h_final = torch.cat([h_final, h], dim=-1) 
            x = h  
        return h_final
    
    
class GIN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout, activation='ReLU', **kwargs):
        super(GIN, self).__init__()
        self.layers = nn.ModuleList()
        self.act = getattr(nn, activation)()

        self.layers.append(GINConv(
            torch.nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                getattr(nn, activation)(),
                nn.Linear(hidden_dim, hidden_dim)
            )
        ))

        for _ in range(1, num_layers):
            self.layers.append(GINConv(
                torch.nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    getattr(nn, activation)(),
                    nn.Linear(hidden_dim, hidden_dim)
                )
            ))
            
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x, edge_index):
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if i < len(self.layers) - 1:
                x = self.act(x)
                x = self.dropout(x)  
        return x


class GAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout, heads=4, activation='ReLU', **kwargs):
        super().__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads, dropout=0.6)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=1,
                             concat=False, dropout=0.6)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x
        
    
class ChebNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout, K=3, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(ChebConv(input_dim, hidden_dim, K=K))
        for _ in range(1, num_layers - 1):
            self.layers.append(ChebConv(hidden_dim, hidden_dim, K=K))
        self.layers.append(ChebConv(hidden_dim, hidden_dim, K=K))
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def forward(self, x, edge_index):
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if i != len(self.layers) - 1:
                x = self.relu(x)
                x = self.dropout(x)
        return x
       
    
class SGC(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout, K=2, activation='ReLU', **kwargs):
        super().__init__()
        self.conv1 = SGConv(input_dim, hidden_dim, K=K, cached=True)
        self.act = getattr(nn, activation)()
        self.conv2 = SGConv(hidden_dim, hidden_dim, K=K, cached=True)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.act(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x
  
    
class GT(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout, heads=4, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(TransformerConv(input_dim, hidden_dim, heads=heads, dropout=0.1, concat=True))

        for _ in range(1, num_layers - 1):
            self.layers.append(TransformerConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=0.1, concat=True))
        self.layers.append(TransformerConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=0.1, concat=False))
        
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()
        
    def forward(self, x, edge_index):
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)  
            if i < len(self.layers) - 1:
                x = self.act(x)
                x = self.dropout(x)
                
        return x