import torch
import numpy as np
import torch.nn as nn
from collections import defaultdict
from torch_scatter import scatter
from ..utils.utils import NeighborSampler, cosine_similarity
from .modules import TimeEncoder, MergeLayer, MultiHeadAttention
    

class SLADE(torch.nn.Module):

    def __init__(self, node_raw_features: np.ndarray, 
                 edge_raw_features: np.ndarray, 
                 neighbor_sampler: NeighborSampler,
                 time_feat_dim: int, 
                 num_layers: int = 1,        ### 2
                 num_heads: int = 2, 
                 dropout: float = 0.1,
                 device: str = 'cpu'):
        """
        :param node_raw_features: ndarray, shape (num_nodes + 1, node_feat_dim)
        :param edge_raw_features: ndarray, shape (num_edges + 1, edge_feat_dim)
        :param neighbor_sampler: NeighborSampler, neighbor sampler
        :param time_feat_dim: int, dimension of time features (encodings)
        :param num_layers: int, number of temporal graph convolution layers
        :param num_heads: int, number of attention heads
        :param dropout: float, dropout rate
        :param src_node_mean_time_shift: float, mean of source node time shifts
        :param src_node_std_time_shift: float, standard deviation of source node time shifts
        :param dst_node_mean_time_shift_dst: float, mean of destination node time shifts
        :param dst_node_std_time_shift: float, standard deviation of destination node time shifts
        :param device: str, device
        """
        super(SLADE, self).__init__()

        self.node_raw_features = torch.from_numpy(node_raw_features.astype(np.float32)).to(device)
        print(self.node_raw_features.shape)
        self.edge_raw_features = torch.from_numpy(edge_raw_features.astype(np.float32)).to(device)
        self.neighbor_sampler = neighbor_sampler
        self.node_feat_dim = self.node_raw_features.shape[1]
        self.edge_feat_dim = self.edge_raw_features.shape[1]
        self.time_feat_dim = self.node_feat_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.device = device
        self.memory = None
        
        self.num_nodes = self.node_raw_features.shape[0]   # number of nodes, including the padded node
        self.memory_dim = self.node_feat_dim        # 172
        self.raw_message_dim = self.memory_dim + self.time_feat_dim
        self.message_dimension = self.raw_message_dim
        
        self.time_encoder = TimeEncoder(time_dim=self.time_feat_dim, parameter_requires_grad=False).to(self.device)

        self.memory_bank = MemoryBank(num_nodes=self.num_nodes, 
                                      memory_dim=self.memory_dim,
                                      raw_message_dim = self.raw_message_dim,
                                      device = self.device)

        self.message_aggregator = MessageAggregator(self.raw_message_dim, self.message_dimension)
        
        self.memory_updater = GRUMemoryUpdater(memory_bank=self.memory_bank, 
                                               message_dim=self.message_dimension, 
                                               memory_dim=self.memory_dim)

        self.embedding_module = GraphAttentionEmbedding(memory=self.memory,
                                                        neighbor_finder = neighbor_sampler,
                                                        time_encoder=self.time_encoder,
                                                        n_layers=self.num_layers,
                                                        n_node_features=self.node_feat_dim,
                                                        n_edge_features=self.edge_feat_dim,
                                                        n_time_features=self.time_feat_dim,
                                                        embedding_dimension=self.node_feat_dim,
                                                        device=self.device,
                                                        n_heads=2, dropout=0.1, use_memory=True)
 
    def compute_temporal_embeddings(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, node_interact_times: np.ndarray,
                                src_neighbors, dst_neighbors, src_neighbors_time, dst_neighbors_time, n_neighbors, test=False):
        n_samples = len(src_node_ids)
        
        src_node_ids = torch.as_tensor(src_node_ids, dtype=torch.long, device=self.device)
        dst_node_ids = torch.as_tensor(dst_node_ids, dtype=torch.long, device=self.device)

        src_neighbors = torch.as_tensor(src_neighbors, dtype=torch.long, device=self.device)
        dst_neighbors = torch.as_tensor(dst_neighbors, dtype=torch.long, device=self.device)

        node_interact_times = torch.as_tensor(node_interact_times, dtype=torch.float32, device=self.device)

        src_neighbors_time = torch.as_tensor(src_neighbors_time, dtype=torch.float32, device=self.device)
        dst_neighbors_time = torch.as_tensor(dst_neighbors_time, dtype=torch.float32, device=self.device)
        neighbors_time = torch.cat([src_neighbors_time, dst_neighbors_time], dim=0).to(self.device)


        # concat
        node_ids = torch.cat([src_node_ids, dst_node_ids], dim=0)
        timestamps = torch.cat([node_interact_times, node_interact_times], dim=0)
        neighbors = torch.cat([src_neighbors, dst_neighbors], dim=0)

        memory = None
        if test:
            all_nodes = torch.concat([node_ids,neighbors.reshape(-1)])
            memory, _ = self.get_updated_memories_tensor(all_nodes, self.memory_bank.messages_tensor, self.memory_bank.messages_time) 

        else:
            memory, _ = self.get_updated_memories_tensor(torch.arange(1, self.num_nodes),
                                                        self.memory_bank.messages_tensor, self.memory_bank.messages_time) 
            
        pos_node_embedding, dst_node_embedding = None, None
        node_embedding_TGAT = self.embedding_module.compute_recovery_memory_embedding(memory=memory,   
                                                                source_nodes=node_ids,
                                                                timestamps=timestamps, 
                                                                neighbors=neighbors, 
                                                                neighbors_time=neighbors_time,
                                                                n_layers=self.num_layers,
                                                                n_neighbors=n_neighbors)
        pos_node_embedding = node_embedding_TGAT[:n_samples]
        dst_node_embedding = node_embedding_TGAT[n_samples:]
        
        node_memory = memory[src_node_ids]
        dst_node_memory = memory[dst_node_ids]

        self.update_memories(node_ids, self.memory_bank.messages_tensor, self.memory_bank.messages_time)
        self.memory_bank.clear_node_raw_messages(node_ids=node_ids)

        unique_src_node_ids, source_message, source_message_ts = self.compute_new_node_raw_messages(src_node_ids=src_node_ids,
                                                                                            dst_node_ids=dst_node_ids,
                                                                                            node_interact_times=node_interact_times)
        unique_dst_node_ids, dst_message, dst_message_ts = self.compute_new_node_raw_messages(src_node_ids=dst_node_ids,
                                                                                            dst_node_ids=src_node_ids,
                                                                                            node_interact_times=node_interact_times)

        self.memory_bank.store_node_raw_messages(unique_src_node_ids, source_message, source_message_ts)
        self.memory_bank.store_node_raw_messages(unique_dst_node_ids, dst_message, dst_message_ts)
                   
        return node_memory, pos_node_embedding, dst_node_memory, dst_node_embedding
    
    


    def get_updated_memories_tensor(self, node_ids: np.ndarray,  messages_tensor, messages_ts_tensor):

        unique_node_ids = torch.unique(node_ids).to(self.device)
        mask = (messages_ts_tensor[unique_node_ids] != 0)

        masked_unique_nodes = unique_node_ids[mask]
        unique_node_messages = messages_tensor[unique_node_ids][mask]
        unique_node_ts = messages_ts_tensor[unique_node_ids][mask]

        if len(masked_unique_nodes) > 0:
            unique_messages = self.message_aggregator.compute_message(unique_node_messages)
        else:
            unique_messages = None
            
        updated_node_memories, updated_node_last_updated_times = self.memory_updater.get_updated_memories(unique_node_ids=masked_unique_nodes,
                                                                                                          unique_node_messages=unique_messages,
                                                                                                          unique_node_timestamps=unique_node_ts)

        return updated_node_memories, updated_node_last_updated_times


    def update_memories(self, node_ids, messages_tensor, messages_ts_tensor):
        unique_node_ids = torch.unique(node_ids).to(self.device)
        mask = (messages_ts_tensor[unique_node_ids] != 0)

        masked_unique_nodes = unique_node_ids[mask]
        unique_node_messages = messages_tensor[unique_node_ids][mask]
        unique_node_ts = messages_ts_tensor[unique_node_ids][mask]
        
        if len(masked_unique_nodes) > 0:
            unique_messages = self.message_aggregator.compute_message(unique_node_messages)
        else:
            unique_messages = None

        self.memory_updater.update_memories(masked_unique_nodes, unique_messages, unique_node_ts)


    def compute_new_node_raw_messages(self, src_node_ids, dst_node_ids, node_interact_times):    ## get_raw_messages

        dst_node_memories = self.memory_bank.get_memories(node_ids=dst_node_ids)

        # Tensor, shape (batch_size, )
        src_node_delta_times = node_interact_times - self.memory_bank.node_last_updated_times[src_node_ids]
        # Tensor, shape (batch_size, time_feat_dim)
        src_node_delta_time_features = self.time_encoder(src_node_delta_times.unsqueeze(dim=1)).reshape(len(src_node_ids), -1)

        # Tensor, shape (batch_size, message_dim = memory_dim + memory_dim + time_feat_dim + edge_feat_dim)
        new_src_node_raw_messages = torch.cat([dst_node_memories, src_node_delta_time_features], dim=1)

        source_nodes_torch = src_node_ids
        (nid, idx) = torch.unique(source_nodes_torch, return_inverse=True)
        message = scatter(new_src_node_raw_messages, idx, reduce='mean', dim=0)
        message_ts = scatter(node_interact_times, idx, reduce='max')
        unique_sources = nid

        return unique_sources, message, message_ts

    def compute_node_diff_score(self, src_node_ids, dst_node_ids, edge_times, src_neighbors, dst_neighbors, src_neighbors_time, dst_neighbors_time, n_neighbors, negative_nodes):
        prev_memory = self.memory_bank.get_memories(src_node_ids)
        prev_dst_memory = self.memory_bank.get_memories(dst_node_ids)
       
               
        node_memory, pos_node_embedding, dst_node_memory, dst_node_embedding = self.compute_temporal_embeddings(src_node_ids, dst_node_ids, edge_times, src_neighbors, dst_neighbors, src_neighbors_time, dst_neighbors_time, n_neighbors)
        negative_memory = self.memory_bank.node_memories[negative_nodes]

        positive_recovery_score = torch.exp(torch.diag(cosine_similarity(pos_node_embedding, node_memory)).reshape(-1))
        negative_recovery_score = torch.exp(cosine_similarity(pos_node_embedding, negative_memory)).sum(dim=1)

        positive_drift_score = (torch.exp(torch.diag(cosine_similarity(node_memory, prev_memory)).reshape(-1)))
        negative_drift_score = torch.exp(cosine_similarity(node_memory, negative_memory)).sum(dim=1)

        dst_positive_recovery_score = torch.exp(torch.diag(cosine_similarity(dst_node_embedding, dst_node_memory)).reshape(-1)) 
        dst_negative_recovery_score = torch.exp(cosine_similarity(dst_node_embedding, negative_memory)).sum(dim=1)

        dst_positive_drift_score = (torch.exp(torch.diag(cosine_similarity(dst_node_memory, prev_dst_memory)).reshape(-1)))
        dst_negative_drift_score = torch.exp(cosine_similarity(dst_node_memory, negative_memory)).sum(dim=1)

        
        contrastive_loss = - torch.log(positive_recovery_score/(negative_recovery_score)) *0.1 \
                            - torch.log(dst_positive_recovery_score/(dst_negative_recovery_score)) *0.1 \
                            - torch.log(positive_drift_score/(negative_drift_score)) \
                            - torch.log(dst_positive_drift_score/(dst_negative_drift_score))

        contrastive_loss = contrastive_loss.mean()

        return positive_recovery_score, positive_drift_score, dst_positive_recovery_score, dst_positive_drift_score, contrastive_loss
    

    def set_neighbor_sampler(self, neighbor_sampler: NeighborSampler):
        self.neighbor_sampler  = neighbor_sampler 
        self.embedding_module.neighbor_sampler = neighbor_sampler 
            
    def compute_anomaly_score(self, source_nodes, destination_nodes, edge_times, src_neighbors, dst_neighbors, src_neighbors_time, dst_neighbors_time, n_neighbors):
        prev_memory = self.memory_bank.get_memories(source_nodes)
        prev_dst_memory = self.memory_bank.get_memories(destination_nodes)
        node_memory, pos_node_embedding, dst_node_memory, dst_node_embedding = self.compute_temporal_embeddings(
                        source_nodes, destination_nodes, edge_times, src_neighbors, dst_neighbors, src_neighbors_time, dst_neighbors_time, n_neighbors, test=True)      

        positive_recovery_score = torch.diag(cosine_similarity(pos_node_embedding, node_memory)).reshape(-1)
        dst_positive_recovery_score = torch.diag(cosine_similarity(dst_node_embedding, dst_node_memory)).reshape(-1)
        positive_drift_score = torch.diag(cosine_similarity(node_memory, prev_memory)).reshape(-1)
        dst_positive_drift_score = torch.diag(cosine_similarity(dst_node_memory, prev_dst_memory)).reshape(-1)

        return positive_recovery_score, positive_drift_score, dst_positive_recovery_score, dst_positive_drift_score      
    

# Message-related Modules
class MessageAggregator(nn.Module):

    def __init__(self, raw_message_dimension, message_dimension):
        super(MessageAggregator, self).__init__()
        
        self.mlp = self.layers = nn.Sequential(
        nn.Linear(raw_message_dimension, raw_message_dimension // 2),
        nn.ReLU(),
        nn.Linear(raw_message_dimension // 2, message_dimension),
        )
        
    def compute_message(self, raw_messages):

        messages = self.mlp(raw_messages)

        return messages


# Memory-related Modules
class MemoryBank(nn.Module):

    def __init__(self, num_nodes: int, memory_dim: int, raw_message_dim, device):
        super(MemoryBank, self).__init__()
        self.num_nodes = num_nodes
        self.memory_dim = memory_dim
        self.raw_message_dim = raw_message_dim
        self.device=device
        self.node_memories = nn.Parameter(torch.zeros((self.num_nodes, self.memory_dim)), requires_grad=False)
        self.node_last_updated_times = nn.Parameter(torch.zeros(self.num_nodes), requires_grad=False)

        self.__init_memory_bank__()

    def __init_memory_bank__(self):
        self.node_memories.data.zero_()
        self.node_last_updated_times.data.zero_()
        self.messages_tensor = nn.Parameter(torch.zeros(self.num_nodes, self.raw_message_dim).to(self.device),requires_grad=False)
        self.messages_time = nn.Parameter(torch.zeros(self.num_nodes).to(self.device),requires_grad=False)

    def get_memories(self, node_ids: np.ndarray):  # 7-3  get_memory
        return self.node_memories[node_ids]

    def set_memories(self, node_ids: np.ndarray, updated_node_memories: torch.Tensor):   # 7-4 set_memory
        self.node_memories[node_ids] = updated_node_memories

    def detach_memory_bank(self):                # 7-6 detach_memory
        self.node_memories.detach_()
        self.messages_tensor.detach_()
        self.messages_time.detach_()

    def store_node_raw_messages(self, node_ids, messages, message_ts):      # 7-2 store_raw_messages
        self.messages_tensor[node_ids] = messages
        self.messages_time[node_ids] = message_ts

    def clear_node_raw_messages(self, node_ids: np.ndarray):   # 7-7 clear_messages
        unique_nodes = torch.unique(node_ids)
        self.messages_tensor[unique_nodes].zero_()
        self.messages_time[unique_nodes].zero_()

    def get_node_last_updated_times(self, unique_node_ids: np.ndarray):    # 7-5 get_last_update
        return self.node_last_updated_times[unique_node_ids]



class MemoryUpdater(nn.Module):

    def __init__(self, memory_bank: MemoryBank):
        super(MemoryUpdater, self).__init__()
        self.memory_bank = memory_bank

    def update_memories(self, unique_node_ids: np.ndarray, 
                        unique_node_messages: torch.Tensor,
                        unique_node_timestamps: np.ndarray):
        if len(unique_node_ids) <= 0:
            return

        assert (self.memory_bank.get_node_last_updated_times(unique_node_ids) <=
                unique_node_timestamps.float().to(unique_node_messages.device)).all().item(), "Trying to update memory to time in the past!"

        node_memories = self.memory_bank.get_memories(node_ids=unique_node_ids)
        updated_node_memories = self.memory_updater(unique_node_messages, node_memories)
        self.memory_bank.set_memories(node_ids=unique_node_ids, updated_node_memories=updated_node_memories)

        # update last updated times for nodes in unique_node_ids
        self.memory_bank.node_last_updated_times[unique_node_ids] = unique_node_timestamps.float().to(unique_node_messages.device)


    def get_updated_memories(self, unique_node_ids: np.ndarray, 
                             unique_node_messages: torch.Tensor,
                             unique_node_timestamps: np.ndarray):

        if len(unique_node_ids) <= 0:
            return self.memory_bank.node_memories.data.clone(), self.memory_bank.node_last_updated_times.data.clone()

        assert (self.memory_bank.get_node_last_updated_times(unique_node_ids=unique_node_ids) <=
                unique_node_timestamps.float().to(unique_node_messages.device)).all().item(), "Trying to update memory to time in the past!"

        # Tensor, shape (num_nodes, memory_dim)
        updated_node_memories = self.memory_bank.node_memories.data.clone()
        updated_node_memories[unique_node_ids] = \
        self.memory_updater(unique_node_messages, updated_node_memories[unique_node_ids])

        # Tensor, shape (num_nodes, )
        updated_node_last_updated_times = self.memory_bank.node_last_updated_times.data.clone()
        updated_node_last_updated_times[unique_node_ids] = \
        unique_node_timestamps.float().to(unique_node_messages.device)

        return updated_node_memories, updated_node_last_updated_times


class GRUMemoryUpdater(MemoryUpdater):
    def __init__(self, memory_bank: MemoryBank, message_dim: int, memory_dim: int):
        super(GRUMemoryUpdater, self).__init__(memory_bank)
        self.memory_updater = nn.GRUCell(input_size=message_dim, hidden_size=memory_dim)


class GraphAttentionEmbedding(nn.Module):   ## Embedding Module in the ori paper
    def __init__(self, memory, neighbor_finder, time_encoder, n_layers, n_node_features, 
                 n_edge_features, n_time_features, embedding_dimension, device,
                 n_heads=2, dropout=0.1, use_memory=True):
        super().__init__()
        self.memory = memory
        self.neighbor_finder = neighbor_finder
        self.time_encoder = time_encoder
        self.n_layers = n_layers
        self.n_node_features = n_node_features
        self.n_edge_features = n_edge_features
        self.n_time_features = n_time_features
        self.embedding_dimension = embedding_dimension
        self.device = device
        self.dropout = dropout
        self.memory_dropout = dropout
        self.use_memory = use_memory
        self.attention_models = TemporalAttentionLayer(
                                    n_node_features=n_node_features,
                                    n_neighbors_features=n_node_features,
                                    time_dim=n_node_features,
                                    n_head=n_heads,
                                    dropout=dropout)

    def compute_recovery_memory_embedding(self, memory, source_nodes, timestamps, neighbors,
                                          neighbors_time, n_layers, n_neighbors=20):

        assert (n_layers >= 0)
        timestamps_torch = timestamps.unsqueeze(dim=1)

        # query node always has the start time -> time span == 0
        source_nodes_time_embedding = self.time_encoder(torch.zeros_like(timestamps_torch).to(self.device))


        neighbors_torch = neighbors
        edge_deltas_torch = timestamps.unsqueeze(dim=1) - neighbors_time
        neighbors = neighbors.flatten()

        mem_neighbor_embeddings = memory[neighbors,:]
        effective_n_neighbors = n_neighbors if n_neighbors > 0 else 1
        mem_neighbor_embeddings = mem_neighbor_embeddings.view(len(source_nodes), effective_n_neighbors, -1)

        edge_time_embeddings = self.time_encoder(edge_deltas_torch.to(self.device))
        mask = neighbors_torch == 0

        if self.training:
            memory_mask = (torch.rand(mask.size()) < self.memory_dropout).to(self.device)
            mask = torch.logical_or(mask.to(self.device), memory_mask) 

        mem_recovery_embedding = self.aggregate(source_nodes_time_embedding,
                                      mem_neighbor_embeddings,
                                      edge_time_embeddings,
                                      mask)
        return mem_recovery_embedding
    
    
    def aggregate(self, source_nodes_time_embedding, neighbor_embeddings, edge_time_embeddings, mask):
        
        attention_model = self.attention_models
        source_embedding, _ = attention_model(source_nodes_time_embedding,
                                          neighbor_embeddings,
                                          edge_time_embeddings,
                                          mask)
        return source_embedding
    

class TemporalAttentionLayer(torch.nn.Module):
    """
    Temporal attention layer. Return the temporal embedding of a node given the node itself,
    its neighbors and the edge timestamps.
    """

    def __init__(self, n_node_features, n_neighbors_features, time_dim, n_head=2, dropout=0.1):
        super().__init__()

        self.n_head = n_head

        self.feat_dim = n_node_features
        self.time_dim = time_dim

        self.query_dim = time_dim
        self.key_dim = n_neighbors_features + time_dim
        self.val_dim = n_neighbors_features + time_dim

        self.multi_head_target = nn.MultiheadAttention(embed_dim=self.query_dim,
                                                    kdim=self.key_dim,
                                                    vdim=self.val_dim,
                                                    num_heads=n_head,
                                                    dropout=dropout)

    def forward(self, src_time_features, neighbors_features,
                neighbors_time_features, neighbors_padding_mask):
        """
        "Temporal attention model
        :param src_node_features: float Tensor of shape [batch_size, n_node_features]
        :param src_time_features: float Tensor of shape [batch_size, 1, time_dim]
        :param neighbors_features: float Tensor of shape [batch_size, n_neighbors, n_node_features]
        :param neighbors_time_features: float Tensor of shape [batch_size, n_neighbors,
        time_dim]
        :param edge_features: float Tensor of shape [batch_size, n_neighbors, n_edge_features]
        :param neighbors_padding_mask: float Tensor of shape [batch_size, n_neighbors]
        :return:
        attn_output: float Tensor of shape [1, batch_size, n_node_features]
        attn_output_weights: [batch_size, 1, n_neighbors]
        """


        query = src_time_features
        key = torch.cat([neighbors_features, neighbors_time_features], dim=2)
        val = torch.cat([neighbors_features, neighbors_time_features], dim=2)

        # Reshape tensors so to expected shape by multi head attention
        query = query.permute([1, 0, 2])  # [1, batch_size, num_of_features]
        key = key.permute([1, 0, 2])  # [n_neighbors, batch_size, num_of_features]
        val = val.permute([1, 0, 2])

        # Compute mask of which source nodes have no valid neighbors
        invalid_neighborhood_mask = neighbors_padding_mask.all(dim=1, keepdim=True)
        # If a source node has no valid neighbor, set it's first neighbor to be valid. This will
        # force the attention to just 'attend' on this neighbor (which has the same features as all
        # the others since they are fake neighbors) and will produce an equivalent result to the
        # original tgat paper which was forcing fake neighbors to all have same attention of 1e-10
        neighbors_padding_mask[invalid_neighborhood_mask.squeeze(), 0] = False


        attn_output, attn_output_weights = self.multi_head_target(query=query, key=key, value=val,
                                                                key_padding_mask=neighbors_padding_mask)
      
      

        attn_output = attn_output.squeeze()
        attn_output_weights = attn_output_weights.squeeze()

        # Source nodes with no neighbors have an all zero attention output. The attention output is
        # then added or concatenated to the original source node features and then fed into an MLP.
        # This means that an all zero vector is not used.
        attn_output = attn_output.masked_fill(invalid_neighborhood_mask, 0)
        attn_output_weights = attn_output_weights.masked_fill(invalid_neighborhood_mask, 0)

        # Skip connection with temporal attention over neighborhood and the features of the node itself

        return attn_output, attn_output_weights
