from detectors.edge.static import *
from detectors.edge.DTDG import *
from detectors.edge.CTDG import *
from detectors.node.node_detector import *
import torch
import os
import random
import numpy as np
import pandas as pd
from load_config import args
import subprocess, datetime


class ContinuousData:
    def __init__(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, node_interact_times: np.ndarray, 
                 edge_ids: np.ndarray, labels: np.ndarray):

        self.src_node_ids = src_node_ids
        self.dst_node_ids = dst_node_ids
        self.node_interact_times = node_interact_times
        self.edge_ids = edge_ids
        self.labels = labels
        self.num_interactions = len(src_node_ids)
        self.unique_node_ids = set(src_node_ids) | set(dst_node_ids)
        self.num_unique_nodes = len(self.unique_node_ids)


class Data:
    def __init__(self, name, anomaly_ratio: float):
        self.name = name
        self.anomaly_ratio = anomaly_ratio 

    def get_ctdg_data(self, val_ratio=0.2, test_ratio=0.3):
        if self.name not in ['yelp', 'reddit', 'mooc', 'wiki', 'dgraphfin', 'elliptic']:
            ctdg_dir = os.path.join(args.data_root, 'continuous', self.name, str(self.anomaly_ratio))
        else:
            ctdg_dir = os.path.join(args.data_root, 'continuous', self.name)
        graph_df = pd.read_csv(os.path.join(ctdg_dir, 'ml_{}.csv'.format(self.name)))
        edge_features = np.load(os.path.join(ctdg_dir, 'ml_{}.npy'.format(self.name)))
        node_features = np.load(os.path.join(ctdg_dir, 'ml_{}_node.npy'.format(self.name)))

        
        NODE_FEAT_DIM = EDGE_FEAT_DIM = n = args.node_dim

        assert NODE_FEAT_DIM >= node_features.shape[1], f'Node feature dimension in dataset {self.name} is bigger than {NODE_FEAT_DIM}!'
        assert EDGE_FEAT_DIM >= edge_features.shape[1], f'Edge feature dimension in dataset {self.name} is bigger than {EDGE_FEAT_DIM}!'
        if node_features.shape[1] < NODE_FEAT_DIM:
            node_zero_padding = np.zeros((node_features.shape[0], n - node_features.shape[1]))   # can also pad with ones to match the input setting used by DTDG models for fair comparison
            node_features = np.concatenate([node_features, node_zero_padding], axis=1)
        else:
            node_features = node_features
            
        if edge_features.shape[1] < EDGE_FEAT_DIM:
            edge_zero_padding = np.zeros((edge_features.shape[0], n - edge_features.shape[1]))
            edge_features = np.concatenate([edge_features, edge_zero_padding], axis=1)
        else:
            edge_features = edge_features
            
        # get the timestamp of validate and test set
        val_time, test_time = list(np.quantile(graph_df.ts, [(1 - val_ratio - test_ratio), (1 - test_ratio)]))
        print(val_time, test_time)
        
        src_node_ids = graph_df.u.values.astype(np.longlong)
        dst_node_ids = graph_df.i.values.astype(np.longlong)
        node_interact_times = graph_df.ts.values.astype(np.float64)
        edge_ids = graph_df.idx.values.astype(np.longlong)
        labels = graph_df.label.values
        full_data = ContinuousData(src_node_ids=src_node_ids, dst_node_ids=dst_node_ids, 
                                   node_interact_times=node_interact_times, edge_ids=edge_ids, labels=labels)

        random.seed(2020)

        node_set = set(src_node_ids).union(set(dst_node_ids))
        num_total_unique_node_ids = len(node_set)

        # compute nodes which appear at test time
        test_node_set = set(src_node_ids[node_interact_times > val_time]).union(set(dst_node_ids[node_interact_times > val_time]))
        new_test_node_set = set(random.sample(list(test_node_set), int(0.1 * num_total_unique_node_ids)))

        # mask for each source and destination to denote whether they are new test nodes
        new_test_source_mask = graph_df.u.map(lambda x: x in new_test_node_set).values
        new_test_destination_mask = graph_df.i.map(lambda x: x in new_test_node_set).values

        # mask, which is true for edges with both destination and source not being new test nodes (because we want to remove all edges involving any new test node)
        observed_edges_mask = np.logical_and(~new_test_source_mask, ~new_test_destination_mask)

        # for train data, we keep edges happening before the validation time which do not involve any new node, used for inductiveness
        # train_mask = np.logical_and(node_interact_times <= val_time, observed_edges_mask)
        train_mask = node_interact_times <= val_time

        train_data = ContinuousData(src_node_ids=src_node_ids[train_mask], 
                                    dst_node_ids=dst_node_ids[train_mask],
                                    node_interact_times=node_interact_times[train_mask],
                                    edge_ids=edge_ids[train_mask], 
                                    labels=labels[train_mask])

        # define the new nodes sets for testing inductiveness of the model
        train_node_set = set(train_data.src_node_ids).union(train_data.dst_node_ids)
        new_node_set = node_set - train_node_set

        val_mask = np.logical_and(node_interact_times <= test_time, node_interact_times > val_time)
        test_mask = node_interact_times > test_time

        # new edges with new nodes in the val and test set (for inductive evaluation)
        edge_contains_new_node_mask = np.array([(src_node_id in new_node_set or dst_node_id in new_node_set)
                                                for src_node_id, dst_node_id in zip(src_node_ids, dst_node_ids)])
        new_node_val_mask = np.logical_and(val_mask, edge_contains_new_node_mask)
        new_node_test_mask = np.logical_and(test_mask, edge_contains_new_node_mask)

        # validation and test data
        val_data = ContinuousData(src_node_ids=src_node_ids[val_mask], 
                                  dst_node_ids=dst_node_ids[val_mask],
                                  node_interact_times=node_interact_times[val_mask], 
                                  edge_ids=edge_ids[val_mask], 
                                  labels=labels[val_mask])

        test_data = ContinuousData(src_node_ids=src_node_ids[test_mask], 
                                   dst_node_ids=dst_node_ids[test_mask],
                                   node_interact_times=node_interact_times[test_mask], 
                                   edge_ids=edge_ids[test_mask], 
                                   labels=labels[test_mask])


        # validation and test with edges that at least has one new node (not in training set)
        new_node_val_data = ContinuousData(src_node_ids=src_node_ids[new_node_val_mask], 
                                           dst_node_ids=dst_node_ids[new_node_val_mask],
                                           node_interact_times=node_interact_times[new_node_val_mask],
                                           edge_ids=edge_ids[new_node_val_mask], 
                                           labels=labels[new_node_val_mask])

        new_node_test_data = ContinuousData(src_node_ids=src_node_ids[new_node_test_mask], 
                                            dst_node_ids=dst_node_ids[new_node_test_mask],
                                            node_interact_times=node_interact_times[new_node_test_mask],
                                            edge_ids=edge_ids[new_node_test_mask], 
                                            labels=labels[new_node_test_mask])

        print("The dataset has {} interactions, involving {} different nodes".format(full_data.num_interactions, full_data.num_unique_nodes))
        print("The training dataset has {} interactions, involving {} different nodes".format(
            train_data.num_interactions, train_data.num_unique_nodes))
        print("The validation dataset has {} interactions, involving {} different nodes".format(
            val_data.num_interactions, val_data.num_unique_nodes))
        print("The test dataset has {} interactions, involving {} different nodes".format(
            test_data.num_interactions, test_data.num_unique_nodes))
        print("The new node validation dataset has {} interactions, involving {} different nodes".format(
            new_node_val_data.num_interactions, new_node_val_data.num_unique_nodes))
        print("The new node test dataset has {} interactions, involving {} different nodes".format(
            new_node_test_data.num_interactions, new_node_test_data.num_unique_nodes))
        print("{} nodes were used for the inductive testing, i.e. are never seen during training".format(len(new_test_node_set)))

        return node_features, edge_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data


model_detector_dict = {
    # Static GNNs
    'GCN': staticGNNDetector,
    'ChebNet': staticGNNDetector,
    'GIN': staticGNNDetector,
    'SGC': staticGNNDetector,
    'GT': staticGNNDetector,    
    'GAT': staticGNNDetector,
    'RFGraph': RFGraphDetector, 
    'XGBGraph': XGBGraphDetector,
    
    # Discrete-time dynamic GNNs
    'Dy_GrAE': DTDGDetector,
    'T_GCN': DTDGDetector,
    'EvolveGCN_O': DTDGDetector,   
    'EvolveGCN_H': DTDGDetector,
    'MPNN_LSTM': DTDGDetector,
    'AddGraph': AddGraphDetector,

    # Continuous-time dynamic GNNs
    'JODIE': CTDGDetector2,
    'DyRep': CTDGDetector2,
    'TGN': CTDGDetector2,
    'TGAT': CTDGDetector,
    'TCL': CTDGDetector,
    'CAWN': CTDGDetector,
    'GraphMixer': CTDGDetector,
    'DyGFormer': CTDGDetector,
    'FreeDyG': CTDGDetector,

}

node_detector_dict = {
    # Continuous-time dynamic GNNs
    'JODIE': CTDGDetector2_Node,
    'DyRep': CTDGDetector2_Node,
    'TGN': CTDGDetector2_Node,
    'TGAT': CTDGDetector_Node,
    'TCL': CTDGDetector_Node,
    'CAWN': CTDGDetector_Node,
    'GraphMixer': CTDGDetector_Node,
    'DyGFormer': CTDGDetector_Node,
    'FreeDyG': CTDGDetector_Node,
    'SAD': CTDGDetector_Node,
    'SLADE': CTDGDetector2_Node,
}


def save_results(results, file_id):
    if not os.path.exists('results/'):
        os.mkdir('results/')
    if file_id is None:
        file_id = 0
        while os.path.exists('results/{}.csv'.format(file_id)):
        while os.path.exists('results/{}.csv'.format(file_id)):
            file_id += 1
    results = results.copy()
    try:
        results['commit'] = subprocess.check_output(
            ['git', 'describe', '--always', '--dirty'], stderr=subprocess.DEVNULL).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        results['commit'] = 'unknown'
    results['date'] = datetime.datetime.now().isoformat(timespec='seconds')
    results.transpose().to_csv('results/{}.csv'.format(file_id))
    print('save to file ID: {}'.format(file_id))
    return file_id
