import numpy as np
import pandas as pd
from pathlib import Path
import torch
import os
from torch_geometric.data import Data
from io import StringIO
from sklearn.cluster import SpectralClustering
from sklearn.model_selection import train_test_split
from pathlib import Path
from datetime import datetime
import random
from collections import defaultdict
import random
import argparse
            
            
class Dataset:
    def __init__(self, name, prefix):
        self.name = name
        self.prefix = prefix
        self.ori_path = f"{prefix}{'ori/'}{name}"
        self.static_path = Path(f"{prefix}{'static/'}{name}")
        self.dtdg_path = Path(f"{prefix}{'discrete/'}{name}")
        self.dtdg_path.mkdir(parents=True, exist_ok=True)
        self.ctdg_path = Path(f"{prefix}{'continuous/'}{name}")
        self.data = self.load_data()

    def load_csv_with_utf8sig(self):
        with open(self.ori_path, 'r', encoding='utf-8-sig') as file:
            content = file.read()
        data_io = StringIO(content)
        return np.loadtxt(data_io, delimiter=',', dtype=float)
        
    def load_data(self):
        try:
            loaders = {
                'uci': lambda: np.loadtxt(self.ori_path, dtype=float, comments='%', delimiter=' '),
                'digg': lambda: np.loadtxt(self.ori_path, dtype=float, comments='%', delimiter=' '),
                'bit_alpha': lambda: np.loadtxt(self.ori_path, dtype=float, comments='%', delimiter='\t'),
                'bit_otc': lambda: np.loadtxt(self.ori_path, dtype=float, comments='%', delimiter='\t'),
                'email_dnc': lambda: self.load_csv_with_utf8sig(), 
                'eucore': lambda:np.loadtxt(self.ori_path, dtype=float),
                'reddit': lambda: pd.read_csv(f"{self.ori_path}.csv").iloc[:, 1:].rename(columns={'user_id': 'source', 'item_id': 'target', 'ts': 'timestamp', 'state_label': 'label'}),
                'wiki': lambda: pd.read_csv(f"{self.ori_path}.csv").rename(columns={'user_id': 'source', 'item_id': 'target', 'ts': 'timestamp', 'state_label': 'label'}),
                'mooc': lambda: pd.read_csv(f"{self.ori_path}.csv").rename(columns={'user_id': 'source', 'item_id': 'target', 'ts': 'timestamp', 'state_label': 'label'})
            }
            return loaders.get(self.name, lambda: np.loadtxt(self.ori_path, dtype=float, comments='%', delimiter=' '))()
        
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
        
    def process_data(self, anomaly_percent, feature_dim, val_ratio, test_ratio, train_snap, val_snap, test_snap, random_seed):
        data = self.load_data()
        if self.name in ['yelp']:
            self.data.columns=['source', 'target', 'weight', 'label', 'timestamp']
            data = self.data.loc[:, ['source', 'target', 'label', 'timestamp']]
            data['label'] = data['label'].replace({-1: 1, 1: 0})     
        elif self.name in ['email_dnc']:
            data = pd.DataFrame(self.data, columns=['source', 'target', 'timestamp'])
        elif self.name in ['eucore']:
            data = pd.DataFrame(self.data, columns=['timestamp', 'source', 'target', 'label'])
        elif self.name in ['mooc', 'wiki']:
            data = self.data      
        elif self.name in ['reddit']:
            data = self.data     
        else:
            data = pd.DataFrame(self.data, columns=['source', 'target', 'weight', 'timestamp'])
        
        data = data.copy()    
        data[['source', 'target']] = data[['source', 'target']].astype(int)
        data = data.sort_values(by='timestamp').reset_index(drop=True)  ## sort by timestamp and reset column index
        
        # Re-index nodes and edges (All node IDs are remapped to consecutive integers ranging from 0 to n − 1.)
        edges_array = data[['source', 'target']].values.flatten()
        _, edges_indices = np.unique(edges_array, return_inverse=True)
        edges = np.reshape(edges_indices, [-1, 2])
        nodes, _ = np.unique(edges, return_inverse=True)
        data[['source', 'target']] = edges
                        
        # For unlabeled data
        # mooc has organic labels too; upstream left it out, which zeroed its labels and injected synthetic anomalies
        if self.name not in['wiki', 'reddit', 'mooc', 'yelp']:
            data['label'] = 0             
            data = data.loc[:, ['source', 'target', 'label', 'timestamp']]
            
            train_data, temp_data = train_test_split(data, test_size=val_ratio + test_ratio, shuffle=False)
            val, test = train_test_split(temp_data, test_size=test_ratio / (val_ratio + test_ratio), shuffle=False)
            fake_edges = generate_anomaly(data, nodes, random_seed)
                   
            for anomaly in anomaly_percent:
                val_data, test_data = insert_anomaly(val, test, fake_edges, anomaly)           
                data = pd.concat([train_data, val_data, test_data], ignore_index=True)      
                
                save_static_data(self.static_path, self.name, data, train_data, val_data, test_data, feature_dim, anomaly)  
                save_dtdg_data(self.dtdg_path, self.name, data, train_data, val_data, test_data, train_snap, val_snap, test_snap, feature_dim, anomaly)    
                save_ctdg_data(self.ctdg_path, self.name, data, feature_dim, anomaly)
        
        # For labeled data 
        else:
            edge_feat = data.iloc[:, 4:]
            data = data.loc[:, ['source', 'target', 'label', 'timestamp']]
            # data = group_anomaly(data)
            data = pd.concat([data, edge_feat], axis=1)
            
            data.target += (max(data.source.max(), data.target.max()) + 1)
            
            train_data, temp_data = train_test_split(data, test_size=val_ratio + test_ratio, shuffle=False)
            val_data, test_data = train_test_split(temp_data, test_size=test_ratio / (val_ratio + test_ratio), shuffle=False)
            
            save_static_data(self.static_path, self.name, data, train_data, val_data, test_data, feature_dim, 0)    
            # save_dtdg_data(self.dtdg_path, self.name, data, train_data, val_data, test_data, train_snap, val_snap, test_snap, feature_dim, 0)    
            # save_ctdg_data(self.ctdg_path, self.name, data, feature_dim, 0)        
            
        return data


def generate_anomaly(data, nodes, random_seed):
    """
    Generate fake (anomalous) edges by following the TADDY anomaly injection strategy.
    Note:
        - This implementation has O(n^2) time complexity due to the adjacency matrix 
          construction and spectral clustering step, which can be slow for large graphs.
        - For very large datasets, a more efficient alternative is to randomly select 
          edges across nodes instead of using spectral clustering.
    """
    normal_edges = data[['source', 'target']].values
    n = len(nodes)
    m = len(normal_edges)
    adjacency_matrix = np.zeros((n, n), dtype=int)
    
    for _, row in data.iterrows():
        source = int(row['source'])
        target = int(row['target'])
        adjacency_matrix[source, target] = 1
        adjacency_matrix[target, source] = 1  
        
    sc = SpectralClustering(n_clusters=12, affinity='precomputed', n_init=6, assign_labels='discretize', random_state=random_seed)
    labels = sc.fit_predict(adjacency_matrix)

    idx_1 = np.random.choice(n, m*2)
    idx_2 = np.random.choice(n, m*2)
    generated_edges = np.column_stack((idx_1, idx_2))
    existing_edges = set(map(tuple, data[['source', 'target']].values))
    
    fake_edges = np.array([x for x in generated_edges if (x[0], x[1]) not in existing_edges and labels[x[0]] != labels[x[1]]])

    return fake_edges


def concat_anomalies(snapshot, num_anomalies, fake_edges, random_seed):
    print(f'fake edges num: {len(fake_edges)}')
    fake_subset = fake_edges.sample(n=num_anomalies, replace=False, random_state=random_seed)

    fake_subset['label'] = 1
    fake_subset['timestamp'] = snapshot['timestamp'].sample(n=num_anomalies, replace=False).values

    # To ensure that the total number of edges remains unchanged, the ori edge at the insertion position is removed.
    for ts in fake_subset['timestamp']:
        idx = snapshot.index[snapshot['timestamp'] == ts]
        if len(idx) > 0:
            snapshot = snapshot.drop(idx[0])
            
    return pd.concat([snapshot, fake_subset]).sort_values('timestamp').reset_index(drop=True)


# Only Inject anomalies into validation and test sets
def insert_anomaly(val_data, test_data, fake_edges, anomaly_percent):
    np.random.seed(0) 
    fake_edges = pd.DataFrame(fake_edges, columns=["source", "target"])

    num_val_anomalies = int(len(val_data) * anomaly_percent)
    num_test_anomalies = int(len(test_data) * anomaly_percent)

    val_data = concat_anomalies(val_data, num_val_anomalies, fake_edges, random_seed=0)
    test_data = concat_anomalies(test_data, num_test_anomalies, fake_edges, random_seed=1) 

    return val_data, test_data


# T1: Structural Anomalies; T2:Temporal Anomalies, T3:Persistent Anomalies, defined in the paper
def group_anomaly(df):
    df['edge'] = list(zip(df['source'], df['target']))
    edge_hist = defaultdict(list)
    t1 = t2 = t3 = t4 = 0
    total_anomalies = 0
    ano_type = []  
    
    for _, row in df.iterrows():
        edge = row['edge']
        label = row['label']
        
        if label == 1:  
            total_anomalies += 1
            history = edge_hist[edge]
            
            if not history:
                t1 += 1
                ano_type.append(1) 
            elif all(l == 1 for l in history):
                t3 += 1
                ano_type.append(3)  
            elif all(l == 0 for l in history):
                t2 += 1
                ano_type.append(2)  
            else:
                t4 += 1
                ano_type.append(4)  
        else:
            ano_type.append(0)  
            
        edge_hist[edge].append(label)
    
    df['ano_type'] = ano_type
    
    return df
    
    
def save_static_data(static_path, name, data, train_data, val_data, test_data, feature_dim, anomaly_percent):
    static_path.mkdir(parents=True, exist_ok=True)
    if anomaly_percent != 0:
        path = os.path.join(static_path, name + '_' + str(anomaly_percent) + '.pt')        
    else:
        path = os.path.join(static_path, name + '.pt')
    
    edge_index = torch.tensor(data[['source', 'target']].to_numpy().T, dtype=torch.long)
    
    edge_masks = [torch.zeros(len(data), dtype=torch.bool) for _ in range(3)]
    train_mask, val_mask, test_mask = edge_masks
    train_mask[:len(train_data)] = True
    val_mask[len(train_data):len(train_data) + len(val_data)] = True
    test_mask[len(train_data) + len(val_data):] = True

    num_nodes = max(data['source'].max(), data['target'].max()) + 1
    x = torch.ones((num_nodes, feature_dim))
    y = torch.tensor(data['label'])

    graph = Data(
        x = x,
        y = y, 
        edge_index = edge_index,
        train_mask = train_mask, 
        val_mask = val_mask,
        test_mask = test_mask,
    )
        
    torch.save(graph, path)
    return graph


def save_dtdg_data(dtdg_path, name, data, train_data, val_data, test_data, train_snap, val_snap, test_snap, feature_dim, anomaly_percent):
    dtdg_path.mkdir(parents=True, exist_ok=True)
    nodes = max(data['source'].max(), data['target'].max()) + 1
    
    def split_snapshot(dataframe, num):
        snapshots = []
        snapshot_size = len(dataframe)//num
        for i in range(num):
            start = i * snapshot_size
            if i == num-1:
                snapshot = dataframe[start:]
            else:
                snapshot = dataframe[start:start + snapshot_size]
            snapshots.append(snapshot)
        return snapshots
    
    def generate_graph(nodes, data, feature_dim):
        node_features = torch.ones(nodes, feature_dim)
        edge_index = torch.tensor(data[['source', 'target']].values.T, dtype=torch.long).contiguous()
        y = torch.tensor(data['label'].values, dtype=torch.long)
        
        return Data(x=node_features, edge_index=edge_index, y=y)
    
    train_snapshot = split_snapshot(train_data, train_snap)
    valid_snapshot = split_snapshot(val_data, val_snap)
    test_snapshot = split_snapshot(test_data, test_snap) 

    train_snapshots = [generate_graph(nodes, data, feature_dim) for data in train_snapshot]
    valid_snapshots = [generate_graph(nodes, data, feature_dim) for data in valid_snapshot]
    test_snapshots = [generate_graph(nodes, data, feature_dim) for data in test_snapshot]
    
    if anomaly_percent != 0:
        paths = {
            'train': os.path.join(dtdg_path, f"{name}_{anomaly_percent}_train.pt"),
            'valid': os.path.join(dtdg_path, f"{name}_{anomaly_percent}_valid.pt"),
            'test': os.path.join(dtdg_path, f"{name}_{anomaly_percent}_test.pt")
        }
    else:
        paths = {
            'train': os.path.join(dtdg_path, f"{name}_train.pt"),
            'valid': os.path.join(dtdg_path, f"{name}_valid.pt"),
            'test': os.path.join(dtdg_path, f"{name}_test.pt")
        }

    torch.save(train_snapshots, paths['train'])
    torch.save(valid_snapshots, paths['valid'])
    torch.save(test_snapshots, paths['test'])
                                
    return train_snapshots, valid_snapshots, test_snapshots
        
        
def save_ctdg_data(ctdg_path, name, data, feature_dim, anomaly_percent):
    if anomaly_percent != 0:   
        ctdg_path = Path(f'{ctdg_path}/{anomaly_percent}/')
    else:
        ctdg_path = Path(f'{ctdg_path}/')
    ctdg_path.mkdir(parents=True, exist_ok=True)
    
    OUT_DF = f'{ctdg_path}/ml_{name}.csv'
    OUT_FEAT = f'{ctdg_path}/ml_{name}.npy'
    OUT_NODE_FEAT = f'{ctdg_path}/ml_{name}_node.npy'
    
    u_list, i_list, ts_list, label_list = [], [], [], []
    feat_l = []
    idx_list = []
    
    data[['source', 'target']] = data[['source', 'target']] + 1
    
    for index, row in data.iterrows():
        u = int(row[0])
        i = int(row[1])
        try:
            ts = float(row[3])  
        except ValueError:
            ts = datetime.strptime(row[3], '%Y-%m-%d').timestamp()
        label = int(row[2])
        idx = index 
        feat = [float(row[col]) for col in range(5, len(row))]
    
        u_list.append(u)
        i_list.append(i)
        ts_list.append(ts)
        label_list.append(label)
        idx_list.append(idx)
        feat_l.append([feat])

    df = pd.DataFrame({
        'u': u_list,
        'i': i_list,
        'ts': ts_list,
        'label': label_list,
        'idx': idx_list
    })

    first_timestamp = df.loc[0, 'ts']
    df['ts'] = df['ts'] - first_timestamp
    df.loc[0, 'ts'] = 0
    df['idx'] = range(1, len(df)+ 1) 
     
    edge_feats = np.array(feat_l)
    edge_feats = np.squeeze(edge_feats, axis=1)
    
    # edge feature for zero index, which is not used (since edge id starts from 1)
    empty = np.zeros(edge_feats.shape[1])[np.newaxis, :]  
    edge_feats = np.vstack([empty, edge_feats])

    # node features with one additional feature for zero index (since node id starts from 1)
    max_idx = max(df.u.max(), df.i.max())
    
    # To ensure that discrete methods can also run properly and allow for a fair comparison, we use an all-ones matrix instead of the all-zero matrix adopted by most DGNN papers. 
    node_feats = np.ones((max_idx+1, feature_dim))   ## zeros
    num_nodes =  node_feats.shape[0]-1
    print('number of nodes ', num_nodes)
    print('number of node features ', node_feats.shape[1])
    print('number of edges ', edge_feats.shape[0] - 1)
    print('number of edge features ', edge_feats.shape[1])

    df.to_csv(OUT_DF)  # edge-list
    np.save(OUT_FEAT, edge_feats)  # edge features
    np.save(OUT_NODE_FEAT, node_feats)  # node features
    
    
    

def get_args():
    parser = argparse.ArgumentParser(description="Dataset preprocessing arguments")

    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    parser.add_argument('--names', nargs='+', default=['wiki'], help='Dataset names to process')
    parser.add_argument('--prefix', type=str, default='./data/', help='Dataset root directory')
    parser.add_argument('--feature_dim', type=int, default=172, help='Node feature dimension')
    parser.add_argument('--snap_num', type=int, default=10, help='Total number of snapshots (for DTDG path)')
    parser.add_argument('--train_snap', type=int, default=10,  help='Number of training snapshots')
    parser.add_argument('--val_snap', type=int, default=4, help='Number of validation snapshots')
    parser.add_argument('--test_snap', type=int, default=6, help='Number of test snapshots')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='Validation ratio')
    parser.add_argument('--test_ratio', type=float, default=0.3,  help='Test ratio')

    parser.add_argument('--anomaly_percent', type=float, nargs='+', default=[0.1, 0.05, 0.01], help='Anomaly ratios to inject')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    for name in args.names:
        try:
            dataset = Dataset(
                name=name,
                prefix=args.prefix,
            )

            data = dataset.process_data(
                anomaly_percent=args.anomaly_percent,
                feature_dim=args.feature_dim,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
                train_snap=args.train_snap,
                val_snap=args.val_snap,
                test_snap=args.test_snap,
                random_seed=args.seed
            )

        except Exception as e:
            print(f"Error processing dataset '{name}': {e}")