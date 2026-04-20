import time
from utils import *
import pandas
import os
from load_config import get_train_config, get_model_config, args
import warnings
warnings.filterwarnings("ignore")

seed_list = list(range(3407, 10000, 10))

def set_seed(seed=3407):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

columns = ['name']

datasets = ['mooc', 'wiki', 'reddit','email_dnc', 'uci', 'bit_alpha', 'bit_otc', 'digg', 'as_topology', 'eucore']                                                                    # Labeled   7-9
   
models = ['GCN','GIN','ChebNet','SGC','GAT', 'GT', 'RFGraph', 'XGBGraph',                                # Static GNN 0-7
          'Dy_GrAE','T_GCN','EvolveGCN_O','EvolveGCN_H','MPNN_LSTM','AddGraph',                          # DTDG 8-13
          'JODIE','DyRep','TGN','TGAT','TCL','CAWN','GraphMixer','DyGFormer','FreeDyG','SLADE','SAD']    # CTDG 14-24

anomaly_percents = [float(x) for x in args.anomaly_percents.split(',')]

if args.datasets is not None:
    if '-' in args.datasets:
        st, ed = args.datasets.split('-')
        datasets = datasets[int(st):int(ed)+1]
    else:
        datasets = [datasets[int(t)] for t in args.datasets.split(',')]
    print('Evaluated Datasets: ', datasets)

if args.models is not None:
    if '-' in args.models:
        st, ed = args.models.split('-')
        models = models[int(st):int(ed)+1] 
    else:
        models = [models[int(t)] for t in args.models.split(',')]
    print('Evaluated Baselines: ', models)

def get_detector(train_config, model_config, detector_type):
    dataset_name = train_config['dataset']
    anomaly_percent = train_config['anomaly_percent']

    if dataset_name in ('reddit', 'wiki', 'mooc'):
        data_path = os.path.join('data/static', dataset_name, f"{dataset_name}.pt") 
        snap_path = os.path.join('data/discrete', dataset_name) 
    else:
        data_path = os.path.join('data/static', dataset_name, f"{dataset_name}_{anomaly_percent}.pt") 
        snap_path = os.path.join('data/discrete', dataset_name)  
    
    if detector_type in (staticGNNDetector, RFGraphDetector, XGBGraphDetector):
        static_graph = torch.load(data_path)
        detector = detector_type(train_config, model_config, static_graph)
    
    elif detector_type in (AddGraphDetector, DTDGDetector):
        if dataset_name in ('reddit','wiki', 'mooc'):
            train_snapshots = torch.load(os.path.join(snap_path, f"{dataset_name}_train.pt"))
            valid_snapshots = torch.load(os.path.join(snap_path, f"{dataset_name}_valid.pt"))
            test_snapshots = torch.load(os.path.join(snap_path, f"{dataset_name}_test.pt"))
        else:
            train_snapshots = torch.load(os.path.join(snap_path, f"{dataset_name}_{anomaly_percent}_train.pt"))
            valid_snapshots = torch.load(os.path.join(snap_path, f"{dataset_name}_{anomaly_percent}_valid.pt"))
            test_snapshots = torch.load(os.path.join(snap_path, f"{dataset_name}_{anomaly_percent}_test.pt"))

        detector = detector_type(train_config, model_config, train_snapshots, valid_snapshots, test_snapshots)
        
    elif detector_type in (CTDGDetector, CTDGDetector2, CTDGDetector_Node, CTDGDetector2_Node):
        data = Data(dataset_name, anomaly_percent)
        node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data = \
                                                                                                        data.get_ctdg_data()
        detector = detector_type(train_config, model_config, node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data)

    return detector

results = pandas.DataFrame(columns=columns)
file_id = None
for model in models:
    model_result = {'name': model}
    for dataset_name in datasets:
        time_cost = 0

        model_config = get_model_config(model)  
        train_config = get_train_config(dataset_name)
        train_config['dataset'] = dataset_name
        train_config['model'] = model      

        auc_list, pre_list, rec_list, f1_list = [], [], [], []
                
        if dataset_name in ('reddit', 'wiki', 'mooc'):
            
            for t in range(args.trials):
                torch.cuda.empty_cache()
                print("Dataset {}, Model {}, Trial {}".format(dataset_name, model, t))
                if train_config['task']=='Edge':
                    detector_type = model_detector_dict[model]
                else:
                    detector_type = node_detector_dict[model]
                    
                print(detector_type)
                
                seed = seed_list[t]
                set_seed(seed)
                train_config['seed'] = seed
                
                detector = get_detector(train_config, model_config, detector_type )                    
                                    
                st = time.time()
                test_score = detector.train()
                ed = time.time()
                time_cost += ed - st
                
                auc_list.append(test_score['AUROC'])
                pre_list.append(test_score['AUPRC'])
                rec_list.append(test_score['RecK'])
                f1_list.append(test_score['F1'])
                
                del detector
                
                model_result.update({
                    f"{dataset_name}-AUROC mean": np.mean(auc_list),
                    f"{dataset_name}-AUROC std": np.std(auc_list),
                    f"{dataset_name}-AUPRC mean": np.mean(pre_list),
                    f"{dataset_name}-AUPRC std": np.std(pre_list),
                    f"{dataset_name}-RecK mean": np.mean(rec_list),
                    f"{dataset_name}-RecK std": np.std(rec_list),
                    f"{dataset_name}-F1 mean": np.mean(f1_list),
                    f"{dataset_name}-F1 std": np.std(f1_list),
                    f"{dataset_name}-Time": time_cost / args.trials
                })
        else:
            for anomaly_percent in anomaly_percents:
                anomaly_percent_str = str(anomaly_percent)         
                for t in range(args.trials):
                    torch.cuda.empty_cache()
                    
                    train_config['anomaly_percent'] = anomaly_percent_str 

                    seed = seed_list[t]
                    set_seed(seed)
                    train_config['seed'] = seed                              
            
                    detector = get_detector(train_config, model_config, model_detector_dict[model])
                                        
                    st = time.time()
                    test_score = detector.train()
                    auc_list.append(test_score['AUROC'])
                    pre_list.append(test_score['AUPRC'])
                    rec_list.append(test_score['RecK'])
                    f1_list.append(test_score['F1'])
                    
                    ed = time.time()
                    time_cost += ed - st            
                    del detector
                    
                model_result.update({
                    f"{dataset_name}-{anomaly_percent_str}-AUROC mean": np.mean(auc_list),
                    f"{dataset_name}-{anomaly_percent_str}-AUROC std": np.std(auc_list),
                    f"{dataset_name}-{anomaly_percent_str}-AUPRC mean": np.mean(pre_list),
                    f"{dataset_name}-{anomaly_percent_str}-AUPRC std": np.std(pre_list),
                    f"{dataset_name}-{anomaly_percent_str}-RecK mean": np.mean(rec_list),
                    f"{dataset_name}-{anomaly_percent_str}-RecK std": np.std(rec_list),
                    f"{dataset_name}-{anomaly_percent_str}-F1 mean": np.mean(f1_list),
                    f"{dataset_name}-{anomaly_percent_str}-F1 std": np.std(f1_list),
                    f"{dataset_name}-{anomaly_percent_str}-Time": time_cost / args.trials
                })
                
                auc_list, pre_list, rec_list, f1_list = [], [], [], []
                time_cost = 0

    model_result = pandas.DataFrame(model_result, index=[0])
    results = pandas.concat([results, model_result])
    print(results)
    file_id = save_results(results, file_id)