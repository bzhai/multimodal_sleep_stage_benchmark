import pandas as pd
import numpy as np
from imblearn.metrics import specificity_score
from sklearn import metrics


def eval_precision(gt, pred, average='macro'):
    if type(gt) is pd.core.frame.DataFrame or type(gt) is pd.core.frame.Series:
        return metrics.precision_score(gt.fillna(0.0), pred.fillna(0.0), average=average)
    else:
        return metrics.precision_score(gt, pred, average=average)


def eval_acc(gt, pred, average='binary'):
    if type(gt) is pd.core.frame.DataFrame or type(gt) is pd.core.frame.Series:
        return metrics.accuracy_score(gt.fillna(0.0), pred.fillna(0.0))
    else:
        return metrics.accuracy_score(gt, pred)

def eval_cohen(gt, pred, average='quadratic'):

    if average != 'binary':
        weight_method = "quadratic"
    else:
        weight_method = None
    if type(gt) is pd.core.frame.DataFrame or type(gt) is pd.core.frame.Series:
        return metrics.cohen_kappa_score(gt.fillna(0.0), pred.fillna(0.0), weights=weight_method)
    else:
        return metrics.cohen_kappa_score(gt, pred, weights=weight_method)

def eval_acc_multiple_classes(gt, pred, label=0):
    tmp_pd = pd.DataFrame()
    tmp_pd['gt'] = gt
    tmp_pd['pred'] = pred
    tmp_pd = tmp_pd[(tmp_pd['gt'] == label) | (tmp_pd['pred'] == label)]
    tmp_pd['pred'] = tmp_pd['pred'].apply(lambda x: 1 if x == label else 0)
    tmp_pd['gt'] = tmp_pd['gt'].apply(lambda x: 1 if x == label else 0)
    return metrics.accuracy_score(tmp_pd['gt'], tmp_pd['pred'])

def eval_recall(gt, pred, average='macro'):
    if type(gt) is pd.core.frame.DataFrame or type(gt) is pd.core.frame.Series:
        return metrics.recall_score(gt.fillna(0.0), pred.fillna(0.0), average=average)
    else:
        return metrics.recall_score(gt, pred, average=average)


def eval_specificity(gt, pred, average='macro'):
    if type(gt) is pd.core.frame.DataFrame or type(gt) is pd.core.frame.Series:
        return specificity_score(gt.fillna(0.0), pred.fillna(0.0), average=average)
    else:
        return specificity_score(gt, pred, average=average)
# def eval_specificity(gt, pred, average='macro'):
#     if type(gt) is pd.core.frame.DataFrame or type(gt) is pd.core.frame.Series:
#         return metrics.recall_score(gt.fillna(0.0).astype(bool) == False, pred.fillna(0.0).astype(bool) == False,
#                                     average=average)
#     else:
#         return metrics.recall_score(gt == False, pred == False, average=average)


def eval_f1(gt, pred, average='macro'):
    if type(gt) is pd.core.frame.DataFrame or type(gt) is pd.core.frame.Series:
        return metrics.f1_score(gt.fillna(0.0), pred.fillna(0.0), average=average)
    else:
        return metrics.f1_score(gt, pred, average='macro')


# def eval_f1_awake(gt, pred, average='macro'):
#     if type(gt) is pd.core.frame.DataFrame or type(gt) is pd.core.frame.Series:
#         return metrics.f1_score(gt.fillna(0.0).astype(bool) == False, pred.fillna(0.0).astype(bool) == False, average=average)
#     else:
#         return metrics.f1_score(gt == False, pred == False, average=average)