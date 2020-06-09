from glob import glob
import sys
import argparse
# from utilities.utils import get_nndf
from utilities.evaluation import *
from dataset_builder_loader.make_h5py_file import *
from sleep_stage_config import Config


def get_ensemble_nndf(folder, nn_type, feature_type, num_classes, modality, hrv_win_len,
                      base_columns=['mesaid', 'linetime', 'activity', 'stages', 'gt_sleep_block']):
    """
    Get the dataframe correpo
    :param task:
    :param nn_type:
    :param feature_type:
    :return: task1_LSTM_raw_100_pred
    """
    # as long as the document type can be distinguished from ML methods
    nn_pred_columns = []
    files = glob(folder + "/" + "%d_stages_dftest_%ds_ensemble_prob_*.csv" % (num_classes, hrv_win_len))
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
            merged = pd.merge(merged, result[i],left_index=True, right_index=True)

        all_merged_columns = base_columns + nn_pred_columns
        merged = merged[all_merged_columns]

        return merged, nn_pred_columns

def load_ensemble_results(folder, num_classes, modality, feature_type, hrv_win_len):
    """
    Load results from machine learning based methods and combine with  deep learning model based results
    """
    df_nn, _ = get_ensemble_nndf(folder, nn_type='ensemble', feature_type=feature_type, num_classes =num_classes, modality=modality, hrv_win_len=hrv_win_len)
    df_nn = df_nn.rename(columns={"actValue": "activity"})

    for cl in ['activity_y', 'stages_y', 'gt_sleep_block_y']:
        if cl in df_nn.columns:
            del df_nn[cl]
    df_nn["always_1"] = 1
    df_nn["always_0"] = 0
    df_nn["always_2"] = 2
    df_nn["always_3"] = 3
    df_nn["always_4"] = 4
    return df_nn


def load_results(folder, num_classes, modality, feature_type, hrv_win_len):
    """
    Load results from machine learning based methods and combine with  deep learning model based results
    """
    # ALGRESULTS = os.path.join(os.getcwd(), "results", "task%d_formulas.csv" % task)
    ml_results = os.path.join(folder, "%d_stages_%ds_ml_%s.csv" % (num_classes, hrv_win_len, modality))

    # dfalg = pd.read_csv(ALGRESULTS)
    df_ml = pd.read_csv(ml_results)
    df_nn = get_nns(folder, num_classes, modality, feature_type, hrv_win_len)
    df_ml = df_ml.rename(columns={"Unnamed: 0":"algs"})
    df_nn = df_nn.rename(columns={"actValue": "activity"})
    merged = pd.merge(df_ml, df_nn, on=["mesaid", "linetime", "activity", "stages", "gt_sleep_block"])
    for cl in ['activity_y', 'stages_y', 'gt_sleep_block_y']:
        if cl in merged.columns:
            del merged[cl]
    merged["always_1"] = 1
    merged["always_0"] = 0
    merged["always_2"] = 2
    merged["always_3"] = 3
    merged["always_4"] = 4
    return merged

def main(args):
    """
        generate a summary dataset frame with all the evaluation results for different predictive models
    """
    print("Start evaluating the following args: \n")
    print_args(args)
    cfg = Config()
    assert args.modality in cfg.FEATURE_TYPE_DICT.keys()
    feature_type = cfg.FEATURE_TYPE_DICT[args.modality]
    gt = ["stages"]
    default_ml = ["SGD_hinge", "SGD_log", "SGD_perceptron",  "Random_forest_300"]
    default_nn = ["CNN_20", "CNN_50", "CNN_100", "LSTM_20", "LSTM_50", "LSTM_100"]
    default_nn = [x + "_" + feature_type for x in default_nn]
    if args.strategy == "ensemble":
        algs = ["max_ensemble_%s" %feature_type, "mean_ensemble_%s" % feature_type] + gt
    else:
        algs = default_nn + default_ml + gt

    result_folder = cfg.STAGE_OUTPUT_FOLDER_HRV30s[args.num_classes]

    # the pickle result file save all classifier's per subject prediction result for t-test
    if not os.path.exists(os.path.join(result_folder,  "ensemble_data_science")):
        os.mkdir(os.path.join(result_folder, "ensemble_data_science"))
    pickle_result_file = os.path.join(result_folder,  "ensemble_data_science", "%d_stages_results_%s.pkl"
                                      % (args.num_classes, feature_type))
    # the summary file used to save the evaluation summary
    summary_file = os.path.join(result_folder,  "ensemble_data_science",  "%d_stages_summary_%s.csv"
                                % (args.num_classes, feature_type))

    print('loading prediction results from %s' % result_folder)
    if args.strategy == 'ensemble':
        df = load_ensemble_results(os.path.join(result_folder, "ensemble_all"), args.num_classes, args.modality, feature_type, args.hrv_win_len)
    else:
        df = load_results(result_folder, args.num_classes, args.modality, feature_type, args.hrv_win_len)
    print("Expanding algorithms...")

    baselines = []
    for i in np.arange(args.num_classes):
        baselines.append("always_%d" %i)
    algs = baselines + algs
    summary, results = classifier_level_evaluation_summary_epoch_wise(df, algs, eval_method=args.eval_method
                                                                      , num_classes=args.num_classes)  # doesn't do rescore
    summary = pd.DataFrame(summary)
    summary = summary.reindex(sorted(summary.columns), axis=1)
    summary.to_csv(summary_file)
    print("Classifier level summary saved to '%s'" % summary_file)
    with open(pickle_result_file, "wb") as f:
        pickle.dump(results, f)
    print("Created classifier level result file '%s'" % pickle_result_file)
    print("All done!")


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--modality', type=str, default='all', help='the raw modality to use')
    parser.add_argument('--num_classes', type=int, default=3, help='number of classes or labels')
    parser.add_argument('--eval_method', type=str, default='macro',
                        help='The weighting method for classifier level metrics')
    parser.add_argument('--hrv_win_len', type=int, default=30,
                        help='window length to decide which h5 file to use, units=secs')
    parser.add_argument('--strategy', type=str, default='non-ensemble', help='the experiment to plot')
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))