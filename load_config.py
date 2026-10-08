import numpy as np
import argparse
import torch

args = None

def parse_args():
    global args
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--trials', type=int, default=1)
    parser.add_argument('--models', type=str, default=None)
    parser.add_argument('--datasets', type=str, default=None)
    parser.add_argument('--anomaly_percents', type=str, default="0.1", choices=["0.01", "0.05", "0.1", "0.01,0.05,0.1"])
    parser.add_argument('--task', type=str, default='Edge', choices=['Node', 'Edge'])
    parser.add_argument('--gpu', type=int, default=0, help='number of gpu to use')
    parser.add_argument('--train_ratio', type=float, default=0.5, help='ratio of test set')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio of validation set')
    parser.add_argument('--test_ratio', type=float, default=0.3, help='ratio of test set')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--hidden_dim', type=int, default=32, help='hidden')  
        
    ## CTDG parameters
    parser.add_argument('--num_neighbors', type=int, default=20, help='number of neighbors to sample for each node')
    parser.add_argument('--sample_neighbor_strategy', type=str, default='recent', choices=['uniform', 'recent', 'time_interval_aware'], help='how to sample historical neighbors')
    parser.add_argument('--time_scaling_factor', default=1e-6, type=float, help='the hyperparameter that controls the sampling preference with time interval, '
                        'a large time_scaling_factor tends to sample more on recent links, 0.0 corresponds to uniform sampling, '
                        'it works when sample_neighbor_strategy == time_interval_aware')
    parser.add_argument('--num_walk_heads', type=int, default=8, help='number of heads used for the attention in walk encoder')
    parser.add_argument('--num_heads', type=int, default=2, help='number of heads used in attention layer')
    parser.add_argument('--num_layers', type=int, default=2, help='number of model layers')
    parser.add_argument('--walk_length', type=int, default=1, help='length of each random walk')
    parser.add_argument('--time_gap', type=int, default=2000, help='time gap for neighbors to compute node features')
    parser.add_argument('--time_feat_dim', type=int, default=100, help='dimension of the time embedding')
    parser.add_argument('--position_feat_dim', type=int, default=172, help='dimension of the position embedding')
    parser.add_argument('--edge_bank_memory_mode', type=str, default='unlimited_memory', help='how memory of EdgeBank works',
                        choices=['unlimited_memory', 'time_window_memory', 'repeat_threshold_memory'])
    parser.add_argument('--time_window_mode', type=str, default='fixed_proportion', help='how to select the time window size for time window memory',
                        choices=['fixed_proportion', 'repeat_interval'])
    parser.add_argument('--patch_size', type=int, default=1, help='patch size')
    parser.add_argument('--channel_embedding_dim', type=int, default=50, help='dimension of each channel embedding')
    parser.add_argument('--max_input_sequence_length', type=int, default=32, help='maximal length of the input sequence of each node')
    parser.add_argument('--node_dim', type=int, default=172, help='maximal length of the input sequence of each node') 
    parser.add_argument('--negative_sample_strategy', type=str, default='random', choices=['random', 'historical', 'inductive'],
                        help='strategy for the negative edge sampling')
    parser.add_argument('--epochs', type=int, default=None, help='override the number of training epochs for every model')
    parser.add_argument('--wandb', action='store_true', help='log runs to Weights & Biases')
    parser.add_argument('--wandb_project', type=str, default='BAG')
    parser.add_argument('--data_root', type=str, default='/data/dygraph/BAG', help='directory holding the preprocessed static/, discrete/ and continuous/ data')

        
    args = parser.parse_args()

    args.device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'

parse_args()


# train_config & model_config
# train_config is determined by the dataset, while model_config is determined by the model. 
default_train_config = {
    'device': args.device,
    'metric': 'AUPRC',
    'loss': 'bce', 
    'label': False,
}

static_model_config = {
    'lr': 0.01,   
    'epochs': 200,
    'patience': 50,
    'dropout': 0.0,
    'hidden_dim': args.hidden_dim,
    'num_layers': 2,
    'device': args.device,
}

dtdg_model_config = {
    'lr': 0.01,   
    'epochs': 200,
    'patience': 50,
    'dropout': 0.0,
    'hidden_dim': args.hidden_dim,
    'num_layers': 2,
    'device': args.device,
}

ctdg_model_config = {
    'lr': 0.0001,  
    'epochs': 100,
    'patience': 15,
    'dropout': 0.1,
    'hidden_dim': args.hidden_dim,
}


def get_train_config(dataset_name):
    if dataset_name in ['reddit', 'mooc','wiki']:
        train_config = default_train_config.copy()
        train_config['loss'] = 'bce'    
        train_config['label'] = True   
        train_config['model'] = None  
        train_config['dataset'] = None   
        train_config['anomaly_percent'] = None   
    else:
        train_config = default_train_config.copy()
        train_config['loss'] ='pairwise'
        train_config['label'] = False   
        train_config['model'] = None  
        train_config['dataset'] = None   
        train_config['anomaly_percent'] = None  

    batch_size_mapping = {
        'eucore': 2000, 'reddit': 2000, 'mooc': 2000, 'as_topology': 1000
    }
    
    train_config['batch_size'] = batch_size_mapping.get(dataset_name, 200)
    train_config['task'] = args.task

    return train_config


def get_model_config(model):
    if model in ['GCN','GIN','ChebNet','SGC','GAT', 'GT', 'RFGraph', 'XGBGraph']:
       model_config = static_model_config.copy()
       
    elif model in ['Dy_GrAE','T_GCN','EvolveGCN_O','EvolveGCN_H','MPNN_LSTM','AddGraph']:
       model_config = dtdg_model_config.copy()

       if model in ['EvolveGCN_H']:
            model_config['lr'] = 0.005
            model_config['epochs'] = 200

       if model in ['AddGraph']:
            model_config['lr'] = 0.001
            model_config['epochs'] = 200
              
    elif model in ['JODIE', 'DyRep', 'TGN', 'TGAT', 'TCL', 'CAWN', 'GraphMixer', 'DyGFormer', 'FreeDyG', 'SAD', 'SLADE']:
        model_config = ctdg_model_config.copy()
            
    model_config['model'] = model
    if args.epochs is not None:
        model_config['epochs'] = args.epochs

    return model_config


