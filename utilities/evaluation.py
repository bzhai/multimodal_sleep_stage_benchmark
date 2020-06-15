
from utilities.evaluation_metrics import *
from utilities.utils import *

EVAL_METRICS = ["accuracy", "precision", "recall", "specificity", "f1-score", "cohen-kappa"]
LABEL_EVAL_METRIC = ["accuracy", "precision", "recall", "specificity", "f1-score"]


def evaluate_scoring_algorithm(df, alg, eval_method):
    # This function calculates all metrics per subject per classifier.
    # We first build up sleep block for the algorithm we have to input the dataset frame and the start and end index of the
    # sleep block. The calculation of sleep efficiency will based on four different annotation.

    print('Evaluating model %s...' % alg)
    r = []

    for func in [eval_acc, eval_precision, eval_recall, eval_specificity, eval_f1, eval_cohen]:
        if alg != "stages":
            # we calculate the metrics for each subject
            r.append(df.groupby("mesaid")[[alg, "stages"]].apply(lambda x: func(x["stages"], x[alg], average=eval_method)))
        else:
            v = df.groupby("mesaid")[["stages"]].apply(lambda x: func(x["stages"], x["stages"]))
            r.append(v)

    res = pd.concat(r, axis=1)
    res.columns = EVAL_METRICS
    return res

def evaluate_scoring_algorithm_epoch_wise(df, alg, eval_method):
    # This function calculates all metrics per subject per classifier.
    # We first build up sleep block for the algorithm we have to input the dataset frame and the start and end index of the
    # sleep block. The calculation of sleep efficiency will based on four different annotation.

    print('Evaluating model %s...' % alg)
    r = []
    for func in [eval_acc, eval_precision, eval_recall, eval_specificity, eval_f1, eval_cohen]:
        # we calculate the metrics for each subject
        r.append(func(df["stages"], df[alg], average=eval_method))

    res = pd.Series(r)
    res.index = EVAL_METRICS
    res.name = alg
    return res


def evaluate_scoring_algorithm_by_label(df, alg, num_classes, label_values=[]):
    print('Evaluating label level metrics %s...' % alg)
    # if alg == "stages" we have two columns both named stages will break metric evaluation, so we only pass "stages"
    # column
    if alg == "stages":
        results = df.groupby('mesaid')[["stages"]].apply(
            lambda x: get_all_classes_prsf_metrics(x['stages'], x[alg], label_values=label_values,
                                                   num_classes=num_classes))
    else:
        results = df.groupby('mesaid')[alg, "stages"].apply(
            lambda x: get_all_classes_prsf_metrics(x['stages'], x[alg], label_values=label_values,
                                                   num_classes=num_classes))
    results = results.reset_index()

    return results

def get_all_classes_prsf_metrics(y_true, y_pred, label_values=[], num_classes=0):
    """
    get the precision, recall, specificity, f1 score for each label
    :param y_true:
    :param y_pred:
    :param label_values:
    :param num_classes:
    :return: a dataset frame that can contain all metrics with supports for each class
    """
    results = pd.DataFrame()
    # get the classification result for each label
    evl_result = classification_report(y_true, y_pred, labels=label_values, output_dict=True)
    specificity = specificity_score(y_true, y_pred, labels=label_values, average=None)
    for i in np.arange(num_classes):
        results_dict = evl_result[str(i)]
        results_dict['label'] = i
        results_dict['specificity'] = specificity[i]
        results_dict['accuracy'] = eval_acc_multiple_classes(y_true, y_pred, i)
        results = results.append(results_dict, ignore_index=True)

    return results


def label_level_evaluation_summary(df, scoring_algorithm, num_classes):
    mlresults = []  # each member is a series that contains a metrics' value for each classifier such as acc, precision.
    results = {}  # a dictionary contains all subjects' evaluation results based on the evaluation metrics
    label_values = np.arange(num_classes)
    num_pids = len(df.mesaid.unique())
    for alg in scoring_algorithm:
        res = evaluate_scoring_algorithm_by_label(df, alg, num_classes, label_values=label_values)
        del res['level_1']  # remove the columns after reset the index
        results[alg] = res.copy(deep=True)
        for label in label_values:
            tmp_df = res[res['label'] == label][LABEL_EVAL_METRIC]
            # std and mean will be a Series that calculate the mean and std of each columns of tmp_df
            stds = tmp_df.std(axis=0).rename('Std')
            means = tmp_df.mean(axis=0).rename('Mean')
            se_df = pd.concat([means, stds], axis=1)
            se_df['SE'] =  se_df['Std'] / np.sqrt(num_pids)
            a = se_df.loc[set(LABEL_EVAL_METRIC)].apply(
                lambda x: "%.1f +- %.1f" % ((100. * x["Mean"]), (100. * 1.96 * x["SE"])), axis=1).rename(alg + '_' + str(label))
            mlresults.append(a)
    return mlresults, results


def classifier_level_evaluation_summary_epoch_wise(df, scoring_algorithms, eval_method, num_classes, precomputed_dict=None):
    mlresults = []
    results = {}

    # for each sleep algorithm we calculate its metrics
    for alg in scoring_algorithms:
        if precomputed_dict is not None:
            res = precomputed_dict[alg]
        else:
            if num_classes == 2:
                res = evaluate_scoring_algorithm_epoch_wise(df, alg, 'binary')
            else:
                res = evaluate_scoring_algorithm_epoch_wise(df, alg, eval_method)
            results[alg] = res.copy(deep=True)
        n = len(df.mesaid.unique())

        mlresults.append(res)
    return mlresults, results


def classifier_level_evaluation_summary(df, scoring_algorithms, eval_method, num_classes, precomputed_dict=None ):
    mlresults = []
    results = {}

    # for each sleep algorithm we calculate its metrics
    for alg in scoring_algorithms:
        if precomputed_dict is not None:
            res = precomputed_dict[alg]
        else:
            if num_classes == 2:
                res = evaluate_scoring_algorithm(df, alg, 'binary')
            else:
                res = evaluate_scoring_algorithm(df, alg, eval_method)
            results[alg] = res.copy(deep=True)
        n = len(df.mesaid.unique())
        stds = res.std(axis=0).rename("Std")  # this line will return a vector represent (Series) each column's std
        means = res.mean(axis=0).rename("Mean")  # this line will return a vector represent (Series) each column's mean
        res = pd.concat([means, stds], axis=1)
        res['SE'] = res['Std']/ np.sqrt(n)
        a = res.loc[set(EVAL_METRICS)].apply(
            lambda x: "%.1f +- %.1f" % (100. * x["Mean"], 100. * 1.96 * x["SE"]), axis=1).rename(alg)

        mlresults.append(a)
    return mlresults, results


def evaluate_whole_period_time_ml(df, scoring_algorithm, num_classes):
    """
    the function will evaluate per participants based sleep minutes for each stages
    :return:
    """
    mlresults=[]
    results = {}
    for alg in scoring_algorithm:
        if alg == 'stages':
            res = calc_gt_mins(df['stages'], num_classes=num_classes)
        else:
            res = calc_mins_per_alg_ml(df, alg, num_classes)
        results[alg] = res.copy(deep=True)
        columns = convert_int_to_label(num_classes)
        res = res.rename(columns=columns)
        res = res.rename(index={0:alg})
        mlresults.append(res)
    mlresults = pd.concat(mlresults, ignore_index=True)

    return mlresults, results


def evaluate_whole_period_time(df, scoring_algorithm, num_classes):
    """
    the function will evaluate per participants based sleep minutes for each stages
    :return:
    """
    mlresults=[]
    results = {}
    for alg in scoring_algorithm:
        if alg == 'stages':
            res = df.groupby('mesaid')[["stages"]].apply(lambda x: calc_gt_mins(x['stages'], num_classes=num_classes))
            res = res.reset_index()  # reset index only happens when merge the aggregated results by groupby function-
            del res['level_1']
        else:
            res = calc_mins_per_alg(df, alg, num_classes)
        results[alg] = res.copy(deep=True)
        # res = pd.concat(results, axis=1)
        n = len(df.mesaid.unique())
        #TODO res need convert to minutes
        stds = res.std(axis=0).rename("Std")  # this line will return a vector represent (Series) each column's std
        means = res.mean(axis=0).rename("Mean")  # this line will return a vector represent (Series) each column's mean
        res = pd.concat([means, stds], axis=1)
        res['SE'] = (res['Std'] / np.sqrt(n))
        conf_intl_pred = res.loc[set(np.arange(num_classes).astype(str))].apply(
            lambda x: "%.1f +- %.1f" % (x["Mean"], 1.96 * x["SE"]), axis=1).rename(alg)
        mlresults.append(conf_intl_pred)

    return mlresults, results


def calc_gt_mins(df, num_classes):
    results = pd.DataFrame()
    per_class_mins = {}
    for label in range(num_classes):
        gt_minutes = df[df == label].shape[0]*0.5 # per sleep epoch is 30 seconds
        per_class_mins[str(label)] = gt_minutes
    results = results.append(per_class_mins, ignore_index=True)

    return results


def calc_mins_per_alg_ml(df, alg, num_classes):
    print("claculate minutes for %s" % alg)
    results = calc_mins_per_label(df["stages"], df[alg], num_classes=num_classes)
    return results


def calc_mins_per_alg(df, alg, num_classes):
    print("claculate minutes for %s" % alg )
    if alg == "stages":
        results = df.groupby('mesaid')[["stages"]].apply(
            lambda x: calc_mins_per_label(x['stages'], x[alg], num_classes=num_classes))
    else:
        results = df.groupby('mesaid')[alg, "stages"].apply(
            lambda x: calc_mins_per_label(x['stages'], x[alg], num_classes=num_classes))
    results = results.reset_index()
    del results['level_1']
    return results


def calc_mins_per_label(y_true, y_pred, num_classes=0):
    result = pd.DataFrame()
    per_class_min = {}
    for label in range(num_classes):
        tmp_df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
        gt_minutes = tmp_df[tmp_df['y_true'] == label].shape[0]*0.5  # one epoch is 0.5 minutes
        pred_minutes = tmp_df[tmp_df['y_pred'] == label].shape[0]*0.5
        per_class_min[str(label)] = pred_minutes - gt_minutes
    result = result.append(per_class_min, ignore_index=True)
    return result
