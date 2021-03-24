from glob import glob
from scipy import interpolate
from scipy.special import softmax
from scipy.stats import ttest_ind
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
import itertools
import pickle
import time
from sklearn.metrics import confusion_matrix, f1_score, precision_score, roc_auc_score, recall_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score, roc_curve, precision_recall_curve
from datetime import datetime
from mlxtend.plotting import plot_confusion_matrix as mlxtend_plot_confusion_matrix
from mlxtend.evaluate import confusion_matrix as mlxtend_confusion_matrix
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import csv
from tqdm import tqdm
from colorama import Fore
from inspect import signature
from sklearn.preprocessing import StandardScaler
# import seaborn as sns


# def plot_bar_group_paper(total_df, metrics, win_len, output_path):
#     """
#     Plot a
#     """
#     # https://seaborn.pydata.org/tutorial/color_palettes.html
#     for metric in metrics:
#         c_palette = sns.color_palette("colorblind", len(total_df.Algorithms.unique()))
#         rank = total_df[metric].argsort().argsort()
#         sns.set_context("paper")
#         sns.set_style("whitegrid")
#         g = sns.catplot(x="tasks", y=metric, hue="Algorithms", kind="bar", height=8, legend_out=False
#                     , data=total_df[['Algorithms', 'tasks', metric]], palette=c_palette)
#         g.set(xlabel='Tasks and classifiers')
#         g.set(ylim=(15, 100))
#         plt.savefig(os.path.join(output_path, metric + "_%ds_" % win_len + ".png"), dpi=200)


def split_and_extract(x, index):
    t1 = x.split('+-')
    return float(t1[0])

def load_tf_model(model_path=''):
    import tensorflow as tf
    with tf.Session() as sess:
        loaded_saver = tf.train.import_meta_graph(model_path)
        loaded_saver.restore(sess, tf.train.latest_checkpoint('./'))
        print(sess.run('w1:0'))
        return sess


def get_all_files_include_sub(path, file_type):
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if file_type in file:
                files.append(os.path.join(os.path.abspath(r), file))
    return files


def plot_train_history(history, title):
  loss = history.history['loss']
  val_loss = history.history['val_loss']

  epochs = range(len(loss))

  plt.figure()

  plt.plot(epochs, loss, 'b', label='Training loss')
  plt.plot(epochs, val_loss, 'r', label='Validation loss')
  plt.title(title)
  plt.legend()

  plt.show()


def standardize_df_given_feature(df, features=[], scaler=None, df_name="", simple_method=True):
    assert len(features) > 0, print("feature length must greater than 0")
    # check if the df contains nan or inf
    if simple_method:
        print("pre-processing dataset frame using simple method")
        df[features] = df[features].replace([np.inf, -np.inf], np.nan)
        df[features] = df[features].fillna(df[features].mean())
        # df[~np.isfinite(df)] = np.nan
        nan = df[df.isnull().any(axis=1)]
        if nan.shape[0] > 0:
            print("df contains nan")
        inf = df[df.eq(np.inf).any(axis=1)]
        if inf.shape[0] > 0:
            print("df contains inf")
    else:
        print("pre-processing dataset frame using comprehensive method")
        for feature in features:
            print("quality check on %s for column name: % s" % (df_name, feature))
            if df[feature].isnull().values.any():
                df[feature] = df[feature].replace(np.nan, df[~df[feature].isin([np.nan, np.inf, -np.inf])][feature].mean())
            if df[feature].isin([np.inf]).values.any():
                df[feature] = df[feature].replace(np.inf, df[~df[feature].isin([np.nan, np.inf, -np.inf])][feature].max())
            if df[feature].isin([-np.inf]).values.any():
                df[feature] = df[feature].replace(-np.inf, df[~df[feature].isin([np.nan, np.inf, -np.inf])][feature].min())

            df[feature] = df[feature].replace([np.nan, np.inf, -np.inf], 0.0)
    if scaler is None:
        scaler = StandardScaler()
        print(' Not given scaler start training scaler now!')
        scaler.fit(df[features])
    print('start transform dataset :%s' % df_name)
    df[features] = scaler.transform(df[features])
    return scaler


def extract_x_y(df, seq_len, mesaid, label_posi='mid', feature=""):
    df_x = df[df["mesaid"] == mesaid][[feature, "stages"]].copy()
    y = df_x["stages"].astype(int).values  # get the ground truth for y
    del df_x["stages"]
    if label_posi == 'mid':
        for s in range(1, round(seq_len/2) + 1):
            df_x["shift_%d" % s] = df_x[feature].shift(s)
        # as half of the sliding window has reversed order (these df columns)
        columns = df_x.columns.tolist()
        columns = columns[::-1]  # or data_frame = data_frame.sort_index(ascending=True, axis=0)
        df_x = df_x[columns]
        for s in range(1, round(seq_len/2) + 1):
            df_x["shift_-%d" % s] = df_x[feature].shift(-s)
    else:
        for s in range(1, seq_len+1):
            df_x["shift_%d" % s] = df_x["activity"].shift(s)
    x = df_x.fillna(-1).values
    return x, y


def get_data(df, seq_len, feature_list):
    # build dataset by participant ID, extract dataset using sliding window method.
    final_x = []
    # loop all mesa_ids
    for feature in feature_list:
        mesaids = df.mesaid.unique()
        print("Processing feature : %s" % feature)
        x, y = extract_x_y(df, seq_len, mesaids[0], label_posi='mid', feature=feature)

        for mid in tqdm(mesaids[1:], position=0, leave=True, bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.WHITE, Fore.RESET),
                   desc='Process..'):
            x_tmp, y_tmp = extract_x_y(df, seq_len, mid, label_posi='mid', feature=feature)
            x = np.concatenate((x, x_tmp))
            y = np.concatenate((y, y_tmp))
        x = np.expand_dims(x, -1)
        final_x.append(x)
    combined_x = np.concatenate(final_x, axis=-1)
    return combined_x, y


def standardize_features_to_array(df, scalers = None):
    """
    This function will scale the dataset set use SK learn scaler function however we recommend do not pass a feature list
    to the function as it may difficult to save the scaler list into H5py file
    :param df:
    :param features:
    :param scaler:
    :return:
    """
    df = df.apply(lambda x: x.replace([np.nan], x[~x.isin([np.nan, np.inf, -np.inf])].mean()), axis=0)
    df = df.apply(lambda x: x.replace([np.inf], x[~x.isin([np.nan, np.inf, -np.inf])].max()), axis=0)
    df = df.apply(lambda x: x.replace([-np.inf], x[~x.isin([np.nan, np.inf, -np.inf])].min()), axis=0)
    df = df.apply(lambda x: x.replace([np.nan, np.inf, -np.inf], 0.0), axis=0)
    if scalers is not None:
        df = scalers.transform(df)
    else:
        scaler = StandardScaler()
        scaler.fit(df)
        df = scaler.transform(df)
    # the final check to replace any abnormal values
    return df, scaler


def load_scaler(path, file_type=".pkl"):
    scaler = None
    if file_type == ".pkl":
        with open(path, "rb") as f:
            scaler = pickle.load(f)
    return scaler


def get_csv_files(data_path):
    # Remove non-mat files, and perform ascending sort
    print("searching csv files ...")
    allfiles = os.listdir(data_path)
    csv_files = []
    for idx, f in enumerate(allfiles):
        if ".csv" in f:
            csv_files.append(os.path.join(data_path, f))
    print("total found {} files".format(len(csv_files)))
    csv_files.sort()
    return csv_files


def get_statistic_feature(df, column_name, windows_size=20):
    """
    the function will directly change input argument dataset frame, so the argument isn't immutable
    :param df:
    :param column_name: the column name we want to extract its statistic features.
    :param windows_size:
    :return: feature_names : contains the features that extracted from the given window size.
    """
    feature_names = []
    for win_size in np.arange(1, windows_size):
        df["_mean_%d" % win_size] = df[column_name].rolling(window=win_size, center=False,
                                                            min_periods=1).mean().fillna(0.0)
        df["_mean_centered_%d" % win_size] = df[column_name].rolling(window=win_size, center=True,
                                                                     min_periods=1).mean().fillna(0.0)

        df["_median_%d" % win_size] = df[column_name].rolling(window=win_size, center=False,
                                                              min_periods=1).median().fillna(0.0)
        df["_median_centered_%d" % win_size] = df[column_name].rolling(window=win_size, center=True,
                                                                       min_periods=1).median().fillna(0.0)

        df["_std_%d" % win_size] = df[column_name].rolling(window=win_size, center=False, min_periods=1).std().fillna(
            0.0)
        df["_std_centered_%d" % win_size] = df[column_name].rolling(window=win_size, center=True,
                                                                    min_periods=1).std().fillna(0.0)

        df["_max_%d" % win_size] = df[column_name].rolling(window=win_size, center=False, min_periods=1).max().fillna(
            0.0)
        df["_max_centered_%d" % win_size] = df[column_name].rolling(window=win_size, center=True,
                                                                    min_periods=1).max().fillna(0.0)

        df["_min_%d" % win_size] = df[column_name].rolling(window=win_size, center=False, min_periods=1).min().fillna(
            0.0)
        df["_min_centered_%d" % win_size] = df[column_name].rolling(window=win_size, center=True,
                                                                    min_periods=1).min().fillna(0.0)

        df["_var_%d" % win_size] = df[column_name].rolling(window=win_size, center=False, min_periods=1).var().fillna(
            0.0)
        df["_var_centered_%d" % win_size] = df[column_name].rolling(window=win_size, center=True,
                                                                    min_periods=1).var().fillna(0.0)

        df["_nat_%d" % win_size] = ((df[column_name] >= 50) & (df[column_name] < 100)).rolling(window=win_size,
                                                                                               center=False,
                                                                                               min_periods=1).sum().fillna(
            0.0)
        df["_nat_centered_%d" % win_size] = ((df[column_name] >= 50) & (df[column_name] < 100)).rolling(window=win_size,
                                                                                                        center=True,
                                                                                                        min_periods=1).sum().fillna(
            0.0)

        df["_anyact_%d" % win_size] = (df[column_name] > 0).rolling(window=win_size, center=False,
                                                                    min_periods=1).sum().fillna(0.0)
        df["_anyact_centered_%d" % win_size] = (df[column_name] > 0).rolling(window=win_size, center=True,
                                                                             min_periods=1).sum().fillna(0.0)

        if win_size > 3:
            df["_skew_%d" % win_size] = df[column_name].rolling(window=win_size, center=False,
                                                                min_periods=1).skew().fillna(0.0)
            df["_skew_centered_%d" % win_size] = df[column_name].rolling(window=win_size, center=True,
                                                                         min_periods=1).skew().fillna(0.0)
            #
            df["_kurt_%d" % win_size] = df[column_name].rolling(window=win_size, center=False,
                                                                min_periods=1).kurt().fillna(0.0)
            df["_kurt_centered_%d" % win_size] = df[column_name].rolling(window=win_size, center=True,
                                                                         min_periods=1).kurt().fillna(0.0)

        # build up the
        for variant in ["centered_", ""]:
            feature_names.append("_mean_%s%d" % (variant, win_size))
            feature_names.append("_median_%s%d" % (variant, win_size))
            feature_names.append("_max_%s%d" % (variant, win_size))
            feature_names.append("_min_%s%d" % (variant, win_size))
            feature_names.append("_std_%s%d" % (variant, win_size))
            feature_names.append("_var_%s%d" % (variant, win_size))
            feature_names.append("_nat_%s%d" % (variant, win_size))
            feature_names.append("_anyact_%s%d" % (variant, win_size))
            if win_size > 3:
                feature_names.append("_skew_%s%d" % (variant, win_size))
                feature_names.append("_kurt_%s%d" % (variant, win_size))
    df["_Act"] = (df[column_name]).fillna(0.0)
    df["_LocAct"] = (df[column_name] + 1.).apply(np.log).fillna(0.0)  # build up the n log transformation

    feature_names.append("_LocAct")  # add logarithm transformation
    feature_names.append("_Act")
    return feature_names


def pvalue(results, alg1, alg2, metric):
    """
    get the t statistic p-value from two algorithm
    :param results:
    :param alg1:
    :param alg2:
    :param metric:
    :return:
    """
    return ttest_ind(results[alg1][metric], results[alg2][metric])[1]


def make_one_block(source_df, start_idx, end_idx):
    # create a new df from the source df index and fill zeros
    result = pd.Series(data=0, index=source_df.index)
    # set a block in dataset frame with value 1
    result.loc[start_idx:end_idx] = 1
    return result


def get_files_given_type(data_path, file_type):
    print("searching csv files ...")
    allfiles = os.listdir(data_path)
    files = []
    for idx, f in enumerate(allfiles):
        if file_type in f:
            files.append(os.path.basename(f))
    print("total found {} files".format(len(files)))
    files.sort()
    return files


def plot_multiple_classifier_roc(files_path=None):
    """
    it can generate a diagram contains of roc curve for multiple classifiers to show the performance
    :param files_path:
    :return:
    """
    files = get_files_given_type(files_path, file_type='npz')
    # plot roc curve
    plt.figure(0).clf()
    for npz_file in files:
        with np.load(npz_file) as data:
            label = data['experiment']
            y_true = data['y_true']
            y_pred = data['y_pred']
            # label = np.random.randint(2, size=1000)
            fpr, tpr, thresh = roc_curve(y_true, y_pred)
            auc = roc_auc_score(y_true, y_pred)
            plt.plot(fpr, tpr, label=label + " auc=%0.2f" % auc)
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")

    plt.legend(loc=0)


def save_validation_logits(y_true, y_pred, classifier_name=None, file_path=None,):
    if file_path != None:
        save_dict = {"experiment": classifier_name, 'y_true': y_true, 'y_pred': y_pred}
        np.savez(file_path, **save_dict)


# we should first to check if the file existed if not create the file

def log_print_inference(y_test, yhat, label_value, target_names, epochs=0, tensor_board_path='', file_title=""
                        , avg_method='macro', args=None):
    """
    Log inference results to tensor board path, we can track each experiment prediction result include accuracy, recall,
    precision, F1 score, F1 report, confusion matrix and confusion matrix in picture format.
    :param args:
    :param file_title:
    :param y_test:
    :param yhat:
    :param label_value:
    :param target_names:
    :param epochs:
    :param tensor_board_path:
    :return:
    """
    if args is not None:
        write_arguments_to_file(args, os.path.join(tensor_board_path, file_title + "_args.csv"))
    if len(y_test.shape) > 2:
        y_test = np.reshape(y_test, -1)
    accuracy = accuracy_score(y_test, yhat)
    print('Accuracy: %f' % accuracy)
    precision = precision_score(y_test, yhat, average=avg_method)
    print('Precision: %f' % precision)
    recall = recall_score(y_test, yhat, average=avg_method)
    print('Recall: %f' % recall)
    f1_result = f1_score(y_test, yhat, average=avg_method)
    print('F1 score: %f' % f1_result)
    matrix = confusion_matrix(y_test, yhat, label_value)
    report = classification_report(y_test, yhat, target_names=target_names)
    print("Classification report: \n")
    print(report)
    to_json = {'epoch_num': epochs, 'accuracy': accuracy, 'precision_weighted': precision, 'recall': recall,
               'f1_result': f1_result}
    result = pd.DataFrame.from_dict(to_json, orient='index')
    result.to_csv(os.path.join(tensor_board_path, file_title + "metrics_summary.csv"), index=False)
    np.savetxt(os.path.join(tensor_board_path, file_title + 'confusion_matrix.txt'), matrix, fmt='%d', delimiter=',')
    with open(os.path.join(tensor_board_path, file_title + "classification_report.txt"), "w") as text_file:
        text_file.write(report)
    # for binary classification we produce the ROC curve
    if len(target_names) == 2:
        ratio = sum(y_test) / len(y_test)
        print("The ratio between negative and positive case are {}".format(str(ratio)))

    # save the best trained model as well.
    normal_path = plot_save_confusion_matrix(y_test, yhat, normalize=True, class_names=target_names,
                                             location=tensor_board_path, title=file_title)
    return [normal_path]


def plot_pr_re_curve(y_true, y_prob, save_path=None):
    # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    average_precision = average_precision_score(y_true, y_prob)
    step_kwargs = ({'step': 'post'}
                   if 'step' in signature(plt.fill_between).parameters
                   else {})
    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
        average_precision))


def plot_roc_curve(y_true, y_prob, save_path=None):
    if max(y_true) != 1:
        return
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    ####################################
    # The optimal cut off would be where tpr is high and fpr is low
    # tpr - (1-fpr) is zero or near to zero is the optimal cut off point
    ####################################
    i = np.arange(len(tpr))  # index for df
    roc = pd.DataFrame(
        {'fpr': pd.Series(fpr, index=i), 'tpr': pd.Series(tpr, index=i), '1-fpr': pd.Series(1 - fpr, index=i),
         'tf': pd.Series(tpr - (1 - fpr), index=i), 'thresholds': pd.Series(thresholds, index=i)})
    roc_t = roc.loc[(roc.tf - 0).abs().argsort()[:1]]

    if auc > 0.0:
        # when we plot we have to make sure the x and y values are given
        plt.plot(fpr, tpr, color='orange', label='ROC curve (AUC = %0.2f)' % auc)
    else:
        plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot(1 - fpr, tpr, color='red', label='1 - fpr, opt cut point = %0.2f' % roc_t['thresholds'])
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    if len(save_path) > 0:
        _save_path = os.path.join(save_path, "ROC_Curve.png")
        plt.savefig(_save_path, bbox_inches='tight', dpi=300)
        return _save_path
    plt.show()
    f = plt.figure()
    f.clear()
    plt.close(f)
    return ''



def calculate_roc(thresholds, embeddings1, embeddings2, actual_issame, nrof_folds=10):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds))

    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff), 1)
    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):

        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(threshold, dist[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = calculate_accuracy(threshold,
                                                                                                 dist[test_set],
                                                                                                 actual_issame[
                                                                                                     test_set])
        _, _, accuracy[fold_idx] = calculate_accuracy(thresholds[best_threshold_index], dist[test_set],
                                                      actual_issame[test_set])

        tpr = np.mean(tprs, 0)
        fpr = np.mean(fprs, 0)
    return tpr, fpr, accuracy


def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size
    return tpr, fpr, acc


def calculate_val(thresholds, embeddings1, embeddings2, actual_issame, far_target, nrof_folds=10):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    val = np.zeros(nrof_folds)
    far = np.zeros(nrof_folds)

    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff), 1)
    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):

        # Find the threshold that gives FAR = far_target
        far_train = np.zeros(nrof_thresholds)
        for threshold_idx, threshold in enumerate(thresholds):
            _, far_train[threshold_idx] = calculate_val_far(threshold, dist[train_set], actual_issame[train_set])
        if np.max(far_train) >= far_target:
            f = interpolate.interp1d(far_train, thresholds, kind='slinear')
            threshold = f(far_target)
        else:
            threshold = 0.0

        val[fold_idx], far[fold_idx] = calculate_val_far(threshold, dist[test_set], actual_issame[test_set])

    val_mean = np.mean(val)
    far_mean = np.mean(far)
    val_std = np.std(val)
    return val_mean, val_std, far_mean


def calculate_val_far(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))
    val = float(true_accept) / float(n_same)
    far = float(false_accept) / float(n_diff)
    return val, far


def write_arguments_to_file(args, filename):
    with open(filename, 'w') as f:
        for key, value in vars(args).items():
            f.write('%s: %s\n' % (str(key), str(value)))


def parse_args_to_dict(args):
    arg_dict = {}
    for key, value in vars(args).items():
        if type(value) == list:
            temp = ' '.join(str(e) for e in value)
            arg_dict.update({key: temp})
        else:
            arg_dict.update({key: value})
    return arg_dict


def convert_argumments_to_dict(args):
    hyper_params = {}
    for key, value in vars(args).items():
        hyper_params.update({key: str(value)})
    return hyper_params


def convert_abseil_args_dict(abseil_args, module_name=None):
    """
    convert an abseil object to a dictionary
    :param absl_args:
    :return: a dictionary contains user defined args
    """
    d = abseil_args.FLAGS.flags_by_module_dict()
    d = {k: {v.name: v.value for v in vs} for k, vs in d.items() if module_name in k}
    result = None
    if d.__len__() > 0:
        result = dict(list(d.values())[0])
    return result


def convert_args_to_str(args):
    string = ""
    if type(args) is str:
        args = vars(args)
    for key, value in args.items():
        string += str(key.name) + "=" + str(value) + ","
    return string


def zero_leading_int(digits, length):
    assert isinstance(digits, int)
    to_fill = length - len(str(digits))
    if to_fill > 0:
        return str(digits).zfill(length)
    else:
        return str(digits)


def write_arguments_to_file(args, filename):
    """
    this function will write args
    :param args:
    :param filename:
    :return:
    """
    if type(args) is not dict:
        args = vars(args)
    with open(filename, 'w+') as f:
        for key, value in args.items():
            f.write('%s: %s\n' % (key, str(value)))


def plot_save_confusion_matrix(y_true, y_pred, class_names=[], location="",
                               normalize=True,
                               title="",
                               cmap=plt.cm.Blues):
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    if (len(set(y_true))> 2):
        cm = mlxtend_confusion_matrix(y_target=y_true, y_predicted=y_pred, binary=False)
    else:
        cm = mlxtend_confusion_matrix(y_target=y_true, y_predicted=y_pred, binary=True)
    fig, ax = mlxtend_plot_confusion_matrix(conf_mat=cm,
                                    colorbar=True,
                                    show_absolute=True,
                                    show_normed=normalize,
                                    class_names=class_names)

    time_now = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    confusion_png = os.path.join(location, title + "_" + time_now+ "cm.png")
    ax.set_title(title)
    # plt.show()
    fig.savefig(confusion_png,  bbox_inches='tight')

    return confusion_png


def cast_sleep_stages(data, classes=5):
    if type(data) is np.ndarray:
        if classes == 3:
            data[data == 2] = 1  # non-REM
            data[data == 3] = 1  # non-REM
            data[data == 4] = 2  # REM
        elif classes == 4:
            data[data == 2] = 1  # light sleep
            data[data == 3] = 2  # deep sleep
            data[data == 4] = 3  # REM
        elif classes == 2:
            data[data > 0] = 1
        return data
    else:
        # this is for a scalar
        stages_dict = {}
        if classes == 3:
            # dataset=0 wake, dataset=1:non-REM, dataset=2:non-REM, dataset=3:non-REM, dataset=4:REM
            stages_dict = {0: 0, 1: 1, 2: 1, 3: 1, 4: 2}
            return stages_dict[data]  # non-REM
        elif classes == 4:
            # dataset=0:wake, dataset=1, 2:light sleep, dataset=3:deep sleep, dataset =4: REM sleep
            stages_dict = {0: 0, 1: 1, 2: 1, 3: 2, 4: 3}
            return stages_dict[data]
        elif classes == 2:
            stages_dict = {0: 0, 1: 1, 2: 1, 3: 1, 4: 1}
            return stages_dict[data]
        else:
            return data


def convert_int_to_label(num_classes):
    if num_classes == 2:
        return {'0': "Wake", '1': "Sleep"}
    elif num_classes == 3:
        return {'0': "Wake", '1': "Non-REM sleep", '2': "REM sleep"}
    elif num_classes == 4:
        return {'0': "Wake", '1': "Light sleep", '2': "Deep sleep", '3': "REM sleep"}
    elif num_classes == 5:
        return {'0': "Wake", '1': "N1 sleep", '2': "N2 sleep", '3': "N3 sleep", '4': "REM sleep"}


def convert_int_to_label_list(num_classes):
    if num_classes == 2:
        return ["Wake", "Sleep"]
    elif num_classes == 3:
        return ["Wake", "Non-REM sleep", "REM sleep"]
    elif num_classes == 4:
        return ["Wake", "Light sleep", "Deep sleep", "REM sleep"]
    elif num_classes == 5:
        return ["Wake", "N1 sleep", "N2 sleep", "N3 sleep", "REM sleep"]


def get_pid_from_file(file_list, start_pos, end_pos):
    pid = []
    for file in file_list:
        pid.append(file[start_pos:end_pos])
    return pid


def make_eight_classes(df):
    # we have eight combinations 000, 010,100, 110, 001, 011, 111, 101,
    if type(df) is not pd.core.frame.DataFrame or type(df) is pd.core.frame.Series:
        df = pd.DataFrame.from_dict({'gt': df})
    eight_classes = {'000': 0, '010': 1, '100': 2, '110': 3, '001': 4, '011': 5, '111': 6, '101': 7}
    df['gt'] = df['gt'].astype(int)
    df['y+1'] = df["gt"].shift(1).fillna(0).astype(int)
    df['y-1'] = df["gt"].shift(-1).fillna(0).astype(int)
    df['eight_gt'] = df['y-1'].astype(str) + df['gt'].astype(str) + df['y+1'].astype(str)
    df['eight_gt'] = df['eight_gt'].map(eight_classes).fillna(0)
    return df['eight_gt'].values.tolist()


def convert_eight_classes_to_label(pred):
    eight_classes = {'000': 0, '010': 1, '100': 2, '110': 3, '001': 4, '011': 5, '111': 6, '101': 7}
    convert_dict = dict((v,k) for k,v in eight_classes.items())
    results = [convert_dict[x] for x in pred]
    return results


def print_args(args):
    print("args used for this experiment \n")
    print(args)


def get_nndf(folder, nn_type, feature_type, num_classes, modality, hrv_win_len, base_columns=['mesaid', 'linetime', 'activity', 'stages', 'gt_sleep_block']):
    """
    Get the dataframe correpo
    :param task:
    :param nn_type:
    :param feature_type:
    :return: task1_LSTM_raw_100_pred
    """
    # as long as the document type can be distinguished from ML methods
    nn_pred_columns = []
    files = glob(folder + "/" + "%d_stages_%ds_%s_*.csv" % (num_classes, hrv_win_len, nn_type))
    result = []
    for file in files:
        if modality == os.path.basename(file).split('.')[0].split('_')[-1]:
            df = pd.read_csv(file)
            df= df.reset_index(drop=True)
            nn_keys = []
            for k in df.keys():
                if nn_type in k:
                    nn_keys.append(k)
            for k in nn_keys:
                df[k + "_" + feature_type] = df[k]
                nn_pred_columns.append(k + "_" + feature_type)
                del df[k]
            result.append(df)
    if len(result) == 1:
        return result[0], nn_pred_columns
    else:
        merged = pd.merge(result[0], result[1], left_index=True, right_index=True) #left_index=True, right_index=True
        for i in range(2, len(result)):
            merged = pd.merge(merged, result[i], left_index=True, right_index=True)
        all_merged_columns = base_columns + nn_pred_columns
        merged = merged[all_merged_columns]
        return merged, nn_pred_columns


def get_nns(folder, num_classes, modality, feature_type, hrv_win_len, nns=['LSTM', 'CNN'], base_columns=['mesaid', 'linetime', 'activity', 'gt_sleep_block', 'stages']):
    """

    :param task:
    :return:
    """
    # change this code to merge all nn prediction
    if (len(nns) == 1) and (nns[0]) == "LSTM":
        merged, _ = get_nndf(folder, "LSTM", feature_type, num_classes, modality, hrv_win_len, base_columns)
    else:
        lstm_raw, lstm_columns = get_nndf(folder, "LSTM", feature_type, num_classes, modality, hrv_win_len,
                                          base_columns)
        cnn_raw, cnn_columns = get_nndf(folder, "CNN", feature_type, num_classes, modality, hrv_win_len, base_columns)
        merged = pd.merge(lstm_raw, cnn_raw, left_index=True, right_index=True)
        base_columns_merged = [x + '_x' for x in base_columns]
        column_map = dict(zip(base_columns_merged, base_columns))
        merged = merged.rename(columns=column_map)
        all_columns = base_columns + lstm_columns+ cnn_columns
        merged = merged[all_columns]
    if len(merged['stages'].unique()) != num_classes:
        merged['stages'] = merged['stages'].apply(lambda x: cast_sleep_stages(x, classes=num_classes))
    return merged


def calc_transition_probability(df, num_classes):
    df['time_minus_1'] = df.groupby('mesaid')['stages'].shift(-1).fillna(0)
    df['stages'] = df['stages'].apply(lambda x: cast_sleep_stages(x, num_classes))
    df['time_minus_1'] = df['time_minus_1'].apply(lambda x: cast_sleep_stages(x, num_classes))
    df['stages'] = df['stages'].astype(int)
    df['time_minus_1'] = df['time_minus_1'].astype(int)
    transition = pd.crosstab(pd.Series(df['time_minus_1'], name='future'),
                             pd.Series(df['stages'], name='current'), normalize=1)
    transition = transition.reset_index(drop=True)
    return transition


def re_calc_posterior(likelihood, prior_prob):
    vec_prob = np.array(prior_prob)
    posterior = np.multiply(np.array(likelihood), vec_prob)
    posterior = posterior / np.linalg.norm(posterior)
    return posterior


def exp_decay(df, beta=0.2, num_classes=3):
    print("beta value is :%2f" % beta)
    new_df = pd.DataFrame()
    ini_prob = np.array([0.89, 0.1, 0.01])
    for index, row in df.iterrows():
        p_last = ini_prob * beta + (1-beta) * np.array([row['CNN_20_0'], row['CNN_20_1'], row['CNN_20_2']])
        p_current = p_last / np.linalg.norm(p_last)
        current_state = np.argmax(np.asarray(p_current))
        ini_prob = p_current
        for i in np.arange(0, num_classes):
            row.set_value(label='CNN_20_post_%d' % i, value=p_current[i])
        row.set_value(label='CNN_20_post', value=current_state)
        new_df = new_df.append(row)
    return new_df


def weight_likelihood(likelihood, prior_prob):
    '''
    This function is used to weight the likelihood P(stage|x) with prior P(stage), P(stage|x)*P(stage) and normalise the
    weighted probability
    :param likelihood:
    :param prior_prob:
    :return:
    '''
    vec_prob = np.array(prior_prob)
    posterior = np.multiply(np.array(likelihood), vec_prob)
    #posterior = posterior / np.linalg.norm(posterior)
    posterior = softmax(posterior)
    return posterior


def load_h5_dataset(path):
    start = time.time()
    store = pd.HDFStore(path, 'r')
    df_train = store["train"]
    df_test = store["test"]
    feature_name = store["featnames"].values.tolist()
    if type(feature_name[0]) is list:
        feature_name = list(itertools.chain.from_iterable(feature_name))
    store.close()
    print("loading dataset spend : %s" % time.strftime("%H:%M:%S", time.gmtime(time.time() - start)))
    return df_train, df_test, feature_name


def load_h5_df_dataset(path):
    """
    This function needs to be removed, it's a part of data loader
    """
    feature_name = []
    start = time.time()
    store = pd.HDFStore(path, 'r')
    dftrain = store["train"]
    dftest = store["test"]

    feature_name = store["featnames"].values.tolist()
    if type(feature_name[0]) is list:
        feature_name = list(itertools.chain.from_iterable(feature_name))
    store.close()
    print("loading dataset spend : %s" % time.strftime("%H:%M:%S", time.gmtime(time.time() - start)))
    return dftrain, dftest, feature_name


def split_df_to_individual_file(df, folder_path):
    for pid in df.mesaid.unique().tolist():
        tmp_df = df[df['mesaid']==pid]
        tmp_df.to_csv(os.path.join(folder_path, "%d_post_processing_prob_sample.csv" % pid))


def arg_max_pred(df):
    tmp = df.to_numpy()
    pred = np.argmax(tmp)
    return pred


def plot_pid(tmp, path_to_save, title_content, nntype, show):
    if not show:
        plt.ioff()
    # fig, axes = plt.subplots(1, 1, figsize= (20, 15))
    total_stages = len(set(tmp['stages']))
    plt.figure(figsize=(18, 5))
    plt.rcParams['font.size'] = 14
    plt.rcParams['image.cmap'] = 'plasma'
    plt.title(title_content, loc='center', fontsize=20, fontweight=0, color='red')
    # axes[0].tick_params(axis='x', which='both',bottom=False,top=False, labelbottom=False)
    # axes[1].tick_params(axis='x', which='both',bottom=False,top=False, labelbottom=False)
    # axes[2].tick_params(axis='x', which='both',bottom=True,top=False, labelbottom=True, rotation=45)
    plt.ylim([0, total_stages - 0.8])

    plt.plot(tmp.index, 'stages', data=tmp, linewidth=2)
    plt.plot(tmp.index, nntype, data=tmp, color='goldenrod', linewidth=2)
    plt.plot(tmp.index, 'gt_sleep_block', data=tmp, color='green', linewidth=2, linestyle='dashed')
    plt.legend()

    plt.legend(bbox_to_anchor=(0, 1), loc=2, borderaxespad=0.)

    plt.tight_layout()
    plt.savefig(path_to_save)
    if not show:
        plt.close()


def datetime_arange(start_datetime, end_datetime, step):
    """

    :param start_datetime: datetime.datetime(2010, 12, 1)
    :param end_datetime: datetime.datetime(2010, 12, 30, 23, 59, 59
    :param steps: datetime.timedelta(seconds=5)
    :return:
    """
    dt = start_datetime
    result = []
    while dt < end_datetime:
        result.append(dt.strftime('%Y-%m-%d %H:%M:%S'))
        dt += step
    return result


class CsvFileMerger(object):
    """
    file merger class is designed to concat all files in a sub folder
    """
    def __init__(self, dest_path, source_path,  exc_file_type=[], exc_file_names=[], file_type='.csv'):
        assert type(source_path) is list, print("input source path should be a list")
        assert type(exc_file_type) is list, print("exceptional file should be a list")
        assert type(exc_file_names) is list, print("exceptional file name should be a list")
        self.source_path = source_path
        self.dest_path = dest_path
        self.file_type = file_type
        self.all_files = []
        self.unique_list = []
        self.exc_file_type = exc_file_type
        self.exc_file_names = exc_file_names
        self.process_all_files()
    def process_all_files(self):
        for path in self.source_path:
            self.all_files += (get_all_files_include_sub(os.path.abspath(path), file_type=self.file_type))

        if len(self.exc_file_type) > 0:
            remove_file_types = []
            for file in self.all_files:
                for exc in self.exc_file_type:
                    if exc in file:
                        remove_file_types.append(file)
            _tmp_list = [x for x in self.all_files if x not in remove_file_types]
            self.all_files = _tmp_list

        if len(self.exc_file_names) > 0:
            remove_file_list = []
            for file in self.all_files:
                for exc in self.exc_file_names:
                    if exc in file:
                        remove_file_list.append(file)
            _tmp_list = [x for x in self.all_files if x not in remove_file_list]
            self.all_files = _tmp_list
        tmp_all_files = []
        counter = 0
        for file in self.all_files:
            tmp_all_files.append(os.path.basename(file))
        tmp_all_files = set(tmp_all_files)
        print("found unique files in folders")
        for file in list(tmp_all_files):
            print(file)
        for unique_file in tmp_all_files:
            print("Start merge file : %s " % unique_file)
            tmp_files = [x for x in self.all_files if unique_file in x]
            tmp_pd_list = []
            shape_chk = []
            for file in tmp_files:
                _pd = pd.read_csv(file)
                print("shape : %s " % str(_pd.shape))
                tmp_pd_list.append(_pd)
                shape_chk.append(_pd.shape[1])
            # assert len(set(shape_chk)) == 1
            tmp_pd = pd.concat(tmp_pd_list, axis=0, ignore_index=True)
            print("Concat PD shape is %s " % str(tmp_pd.shape))
            tmp_pd.to_csv(os.path.join(self.dest_path, unique_file), index=False)
            counter += 1
            print("file %s is merged" % unique_file)
            print("__________________________________________________________")
        print("total %d files were merged" % counter)


def save_GridSearchCV_csv(file_path, gridcv, sleep_stages):
    result_dic = zip(gridcv.cv_results_['mean_test_score'], gridcv.cv_results_['params'])
    cv_param_scores = []
    for cv_mean, params in zip(gridcv.cv_results_['mean_test_score'], gridcv.cv_results_['params']):
        params = dict(params)
        params.update({'mean_test_score': cv_mean})
        cv_param_scores.append(params)
    cv_param_df = pd.DataFrame.from_dict(cv_param_scores)
    cv_param_df.to_csv(os.path.join(file_path, ("%d_stages_cv_result.csv" % sleep_stages)), index=False)


def ensemble_max_clfs(pred_val, num_clfs, num_classes):
    max_matrix = pred_val.reshape(pred_val.shape[0], num_clfs, num_classes)
    max_matrix = np.amax(max_matrix, axis=1)
    max_matrix = np.argmax(max_matrix, axis=1)
    return max_matrix


def ensemble_mean_clfs(pred_val, num_clfs, num_classes):
    max_matrix = pred_val.reshape(pred_val.shape[0], num_clfs, num_classes)
    max_matrix = np.mean(max_matrix, axis=1)
    max_matrix = np.argmax(max_matrix, axis=1)
    return max_matrix


def save_prediction_results(df_test, pred, pred_path, nn_type, seq_len):
    df_test["%s_%d" % (nn_type, seq_len)] = np.argmax(pred, axis=1)
    df_test["gt_sleep_block"] = df_test["gt_sleep_block"].astype(int)
    df_test["stages"] = df_test["stages"].astype(int)
    df_test["activity"] = df_test["activity"].fillna(0.0)
    df_test[["mesaid", "linetime", "activity", "stages", "gt_sleep_block", "%s_%d" % (nn_type, seq_len)]] \
        .to_csv(pred_path, index=False)


def log_print_metrics(y_pred, y_test, epochs, nn_type, num_classes, tensorboard_path, ag):
    yhat_classes = np.argmax(y_pred, axis=-1)
    # Y_test_classes = np.reshape(y_test, (-1, 2))
    if len(y_test.shape) > 1:
        Y_test_classes = np.argmax(y_test, axis=-1)
    else:
        Y_test_classes = y_test
    if num_classes == 5:
        label_values = [0, 1, 2, 3, 4]
        target_names = ['Wake', 'N1', 'N2', 'N3', 'REM']
    elif num_classes == 4:
        label_values = [0, 1, 2, 3]
        target_names = ['Wake', 'Light', 'Deep', 'REM']
    elif num_classes == 3:
        label_values = [0, 1, 2]
        target_names = ['Wake', 'NREM', 'REM']
    else:
        label_values = [0, 1]
        target_names = ['Wake', 'Sleep']
    log_print_inference(Y_test_classes, yhat_classes, label_value=label_values, target_names=target_names,
                        epochs=epochs, tensor_board_path=tensorboard_path, file_title="dl_exp_%s" % nn_type, args=ag)


def load_pre_splited_train_test_ids(path):
    df = pd.read_csv(path)
    uid_train = df[df['segment'] == "train"]["uids"].values.tolist()
    uid_test = df[df['segment'] == "test"]["uids"].values.tolist()
    return uid_train, uid_test


def optimize_parameters(df_train, featnames, pipe, params, scoring, experiment, num_classes, output_path):
    gs = GridSearchCV(pipe, param_grid=params, n_jobs=4, cv=5, scoring=scoring, verbose=1, pre_dispatch=10)
    gs.fit(df_train[featnames], df_train["stages"])
    with open(os.path.join(output_path, "%d_stages_" % num_classes + "_%s_" % experiment + datetime.now().strftime("%Y%m%d-%H%M%S") + ".txt"), "w+") as f:
        f.writelines("Best parameter for experiment : %s" % experiment)
        f.writelines(str(gs.best_estimator_))
    print(gs.best_estimator_)
    return gs


def display_parameters(gs):
    gsres = pd.DataFrame(gs.cv_results_['params'])
    try:
        gsres["train"] = gs.cv_results_["mean_train_score"]
    except Exception as ee:
        print(str(ee))
    gsres["test"] = gs.cv_results_["mean_test_score"]
    return gsres.sort_values("test", ascending=False)
