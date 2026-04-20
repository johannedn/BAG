
import torch
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve, f1_score



def get_link_prediction_metrics(predicts: torch.Tensor, labels: torch.Tensor):
    """
    get metrics for the link prediction task
    :param predicts: Tensor, shape (num_samples, )
    :param labels: Tensor, shape (num_samples, )
    :param k_values: list of int, top k values for Rec@k
    :return:
        dictionary of metrics {'metric_name_1': metric_1, ...}
    """
    predicts = predicts.cpu().detach().numpy()
    labels = labels.cpu().numpy()

    metrics = {}
    k = 50
    # Calculate average precision
    if len(np.unique(labels)) > 1:
        average_precision = average_precision_score(y_true=labels, y_score=predicts)
    else:
        average_precision = np.nan
    metrics['average_precision'] = average_precision

    # Calculate ROC AUC
    if len(np.unique(labels)) > 1:
        roc_auc = roc_auc_score(y_true=labels, y_score=predicts)
    else:
        roc_auc = np.nan
    metrics['roc_auc'] = roc_auc

    # Calculate F1 score
    f1 = f1_score(y_true=labels, y_pred=(predicts > 0.5).astype(int))
    metrics['f1'] = f1

    # Calculate Rec@k for each k in k_values
    sorted_indices = np.argsort(predicts)[::-1]
    top_k_indices = sorted_indices[:k]
    rec_at_k = np.sum(labels[top_k_indices]) / np.sum(labels) if np.sum(labels) != 0 else 0
    # metrics[f'rec_at_{k}'] = rec_at_k

    return {'average_precision': average_precision, 'roc_auc': roc_auc, 'rec_at_k': rec_at_k, 'f1': f1}




def get_node_classification_metrics(predicts: torch.Tensor, labels: torch.Tensor):
    """
    get metrics for the node classification task
    :param predicts: Tensor, shape (num_samples, )
    :param labels: Tensor, shape (num_samples, )
    :return:
        dictionary of metrics {'metric_name_1': metric_1, ...}
    """
    predicts = predicts.cpu().detach().numpy()
    labels = labels.cpu().numpy()

    roc_auc = roc_auc_score(y_true=labels, y_score=predicts)

    return {'roc_auc': roc_auc}