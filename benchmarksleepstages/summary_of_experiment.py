import sys
import argparse
from utilities.evaluation import *
import os
from sleep_stage_config import Config


def load_results(folder, num_classes, modality, feature_type, hrv_win_len):
    """
    Load results from machine learning based methods and combine with  deep learning model based results
    """
    ml_results = os.path.join(folder, "%d_stages_%ds_ml_%s.csv" % (num_classes, hrv_win_len, modality))
    df_ml = pd.read_csv(ml_results)
    df_nn = get_nns(folder, num_classes, modality, feature_type, hrv_win_len)
    df_ml = df_ml.rename(columns={"Unnamed: 0": "algs"})
    df_nn = df_nn.rename(columns={"actValue": "activity"})
    merged = pd.merge(df_ml, df_nn, on=["mesaid", "linetime", "activity", "stages", "gt_sleep_block"])
    assert len(merged.stages.unique()) == num_classes

    for cl in ['activity_y', 'stages_y', 'gt_sleep_block_y']:
        if cl in merged.columns:
            del merged[cl]

    merged["always_0"] = 0
    merged["always_1"] = 1
    merged["always_2"] = 2
    merged["always_3"] = 3
    merged["always_4"] = 4

    return merged


def main(args):
    """
        generate a summary dataset frame with all the evaluation results for different predictive models
    """
    cfg = Config()
    print("Start evaluating the following args: \n")
    print_args(args)

    assert args.modality in cfg.FEATURE_TYPE_DICT.keys()
    feature_type = cfg.FEATURE_TYPE_DICT[args.modality]
    groundtruth = ["stages"]
    defaultml = ["SGD_hinge", "SGD_log", "SGD_perceptron",  "Random_forest_300"]
    defaultnn = ["CNN_20", "CNN_50", "CNN_100", "LSTM_20", "LSTM_50", "LSTM_100"]
    defaultnn = [x + "_" + feature_type for x in defaultnn]
    algs = defaultnn + defaultml + groundtruth
    result_folder = cfg.STAGE_OUTPUT_FOLDER_HRV30s[args.num_classes]
    summary_folder = cfg.SUMMARY_FOLDER[args.period]
    # the pickle result file saves all classifier's per subject prediction result for t-test purpose
    if not os.path.exists(os.path.join(result_folder, summary_folder)):
        os.mkdir(os.path.join(result_folder, summary_folder))
    classifier_eval_result_file = os.path.join(result_folder, summary_folder, "%d_stages_results_%s.pkl"
                                      % (args.num_classes, feature_type))
    # the summary file used to save the evaluation summary
    classifier_eval_summary_file = os.path.join(result_folder, summary_folder, "%d_stages_summary_%s.csv"
                                % (args.num_classes, feature_type))

    label_eval_result_file = os.path.join(result_folder, summary_folder
                                            , "%d_stages_label_level_results_%s.pkl"
                                            % (args.num_classes, feature_type))
    label_eval_summary_file = os.path.join(result_folder, summary_folder, "%d_stages_label_level_summary_%s.csv"
                                      % (args.num_classes, feature_type))
    minutes_prediction_result_file = os.path.join(result_folder, summary_folder
                                                  , "%d_stages_minutes_results_%s.pkl"
                                           % (args.num_classes, feature_type))
    minutes_prediction_summary_file = os.path.join(result_folder, summary_folder
                                                   , "%d_stages_minutes_summary_%s.csv"
                                                   % (args.num_classes, feature_type))
    print('loading prediction results from %s' % result_folder)
    df = load_results(result_folder, args.num_classes, args.modality, feature_type, args.hrv_win_len)
    if args.period == "s":
        df = df[df["gt_sleep_block"] == 1]

    print("Expanding algorithms...")
    baselines = []
    for i in np.arange(args.num_classes):
        baselines.append("always_%d" %i)
    algs = baselines + algs

    print("start %d stages level evaluation" % args.num_classes)
    min_summary, mins_pred_results = evaluate_whole_period_time(df, algs, args.num_classes)
    min_summary = pd.DataFrame(min_summary)
    min_summary = min_summary.rename(columns=convert_int_to_label(args.num_classes))
    min_summary = min_summary.reindex(sorted(min_summary.columns), axis=1)
    min_summary.to_csv(minutes_prediction_summary_file)
    print("Minutes summary saved to %s" % minutes_prediction_summary_file)
    with open(minutes_prediction_result_file, "wb") as f:
        pickle.dump(mins_pred_results, f)
    print("Created minutes prediction result file to '%s'" % minutes_prediction_result_file)

    label_level_summary, label_level_results = label_level_evaluation_summary(df, algs, args.num_classes)
    label_level_summary = pd.DataFrame(label_level_summary)
    label_level_summary = label_level_summary.reindex(sorted(label_level_summary.columns), axis=1)
    label_level_summary.to_csv(label_eval_summary_file)
    print("Label level summary saved to %s" % label_eval_summary_file)
    with open(label_eval_result_file, "wb") as f:
        pickle.dump(label_level_results, f)
    print("Created label level metrics result file '%s'" % label_eval_result_file)

    summary, results = classifier_level_evaluation_summary(df, algs, eval_method=args.eval_method,
                                                           num_classes=args.num_classes)  # doesn't do rescore
    summary = pd.DataFrame(summary)
    summary = summary.reindex(sorted(summary.columns), axis=1)
    summary.to_csv(classifier_eval_summary_file)
    print("Classifier level summary saved to '%s'" % classifier_eval_summary_file)
    with open(classifier_eval_result_file, "wb") as f:
        pickle.dump(results, f)
    print("Created classifier level result file '%s'" % classifier_eval_result_file)
    print("All done!")

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--modality', type=str, default='all', help='the raw modality to use')
    parser.add_argument('--num_classes', type=int, default=2, help='number of classes or labels')
    parser.add_argument('--eval_method', type=str, default='macro',
                        help='The weighting method for classifier level metrics')
    parser.add_argument('--hrv_win_len', type=int, default=30,
                        help='window length to decide which h5 file to use, units=secs')
    parser.add_argument('--period', type=str, default='r', help='the default is r = recording period'
                                                                ', or s = sleep period')
    return parser.parse_args(argv)

if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))