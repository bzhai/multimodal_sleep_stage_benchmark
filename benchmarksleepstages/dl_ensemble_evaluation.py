import argparse
from keras.callbacks import TensorBoard
import sys
from tensorflow.keras.backend import set_session
import tensorflow as tf
from utilities.evaluation import *
import os
from sleep_stage_config import Config
np.set_printoptions(suppress=True)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = False  # to log device placement (on which device the operation ran)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

# #################### Ben track settings######################################################
# EXPERIMENT = "Use raw dataset of HR and accelerometer for sleep stage classification "
# The code in this program is to
# EXPERIMENT = "Use Activity alone with CNN model "
# SETTINGS = "1 CNN network with sleep-wake raw cache"

time_of_run = datetime.now().strftime("%Y%m%d-%H%M%S")
logdir_root = os.path.join(os.getcwd(), "experiment_results", os.path.basename(__file__))
tensorboard_path = os.path.join(logdir_root, time_of_run)
tensorboard_callback = TensorBoard(log_dir=tensorboard_path)
SUMMARY_FILE = os.path.join(logdir_root, "summary_log.csv")
DETAILS_FILE = os.path.join(tensorboard_path, "details_log.csv")


def main(args):
    print_args(args)
    np.random.seed(42)
    cfg = Config()
    # h5_file = globals()["HRV%d_ACC_STD_PATH" % args.hrv_win_len]
    # print("Loading test dataset from %s" % h5_file)
    #dftrain, dftest, featnames = load_h5_df_dataset(h5_file, useCache=True)
    feature_type = cfg.FEATURE_TYPE_DICT[args.modality]
    stage_output_folder = globals()["STAGE_OUTPUT_FOLDER_HRV%ds" % args.hrv_win_len]
    predicted_prob = "%d_stages_%s_%ds_pred_probability_%s.csv" % (args.num_classes, args.prediction_type,
                                                                    args.hrv_win_len, args.modality)
    result_folder = cfg.STAGE_OUTPUT_FOLDER_HRV30s[args.num_classes]

    prediction_prob = os.path.join(os.path.abspath(stage_output_folder[args.num_classes]), "ensemble_%s" % args.modality, predicted_prob)
    clfs_df = pd.read_csv(prediction_prob)
    base_algs = ['stages']
    alg_prob_columns = []
    for model in args.models:
        base_algs.append(model)
        for label in np.arange(args.num_classes):
            alg_prob_columns.append(model + "_" + str(label))
    print("ensemble start!")
    pred_values = clfs_df[alg_prob_columns].values
    prob_matrix = pred_values.reshape(pred_values.shape[0], 6, args.num_classes)

    max_matrix = np.amax(prob_matrix, axis=1)
    clfs_df["max_ensemble"] = np.argmax(max_matrix, axis=1)
    mean_matrix = np.mean(prob_matrix, axis=1)
    clfs_df["mean_ensemble"] = np.argmax(mean_matrix, axis=1)
    # additive_matrix = np.sum(prob_matrix, axis=1)
    # clfs_df["additive_ensemble"] = np.argmax(additive_matrix, axis=1)
    # multiplicative_matrix = np.prod(prob_matrix, axis=1)
    # clfs_df["multiplicative_ensemble"] = np.argmax(multiplicative_matrix, axis=1)
    # ##########these are the old slow method #############
    # clfs_df['mean_ensemble'] = clfs_df[alg_prob_columns].apply(lambda x: cross_clf_mean(x, args.num_classes), axis=1)
    # clfs_df['max_ensemble'] = clfs_df[alg_prob_columns].apply(lambda x: cross_clf_max(x, args.num_classes), axis=1)
    clfs_ensemble_results = os.path.join(result_folder, "ensemble_%s" % args.modality, "%d_stages_ensemble_results_%s.pkl"
                                      % (args.num_classes, feature_type))
    print('post processing is completed!')
    # algs = base_algs + ['max_ensemble', 'mean_ensemble', 'additive_ensemble', 'multiplicative_ensemble']
    algs = base_algs + ['max_ensemble', 'mean_ensemble']
    print('start evaluation ....')
    summary, results = classifier_level_evaluation_summary(clfs_df, algs, eval_method='macro',
                                                           num_classes=args.num_classes)
    summary = pd.DataFrame(summary)
    summary = summary.reindex(sorted(summary.columns), axis=1)
    summary.to_csv(os.path.join(stage_output_folder[args.num_classes], "ensemble_%s" % args.modality,
                                '%d_stages_ensemble_%s_results_%s.csv' %
                                (args.num_classes, args.prediction_type, feature_type)))
    clfs_df.to_csv(os.path.join(stage_output_folder[args.num_classes], "ensemble_%s" % args.modality,
                                "%d_stages_%s_%ds_ensemble_prob_%s.csv" %
                                (args.num_classes, args.prediction_type, args.hrv_win_len, args.modality)))
    print('results and raw prediction are saved!')
    with open(clfs_ensemble_results, "wb") as f:
        pickle.dump(results, f)
    print("Created classifier level result file '%s'" % clfs_ensemble_results)

    ensemble_min_summary_pickle = os.path.join(result_folder, "ensemble_%s" % args.modality, "%d_stages_minutes_results_%s.pkl"
                                                  % (args.num_classes, feature_type))
    ensemble_min_summary = os.path.join(result_folder, "ensemble_%s" % args.modality, "%d_stages_ensemble_minutes_summary_%s.csv"
                                                   % (args.num_classes, feature_type))
    print("start %d stages level evaluation" % args.num_classes)
    min_summary, min_pred_results = evaluate_whole_period_time(clfs_df, algs, args.num_classes)
    min_summary = pd.DataFrame(min_summary)
    min_summary = min_summary.rename(columns=convert_int_to_label(args.num_classes))
    min_summary = min_summary.reindex(sorted(min_summary.columns), axis=1)
    min_summary.to_csv(ensemble_min_summary)
    print("Minutes summary saved to %s" % ensemble_min_summary)
    with open(ensemble_min_summary_pickle, "wb") as f:
        pickle.dump(min_pred_results, f)
    print("Created minutes prediction result file to '%s'" % ensemble_min_summary_pickle)
    sys.exit()


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--modality', type=str, default="all", help='the modality to use.')
    parser.add_argument('--num_classes', type=int, default=5, help='number of classes or labels')
    parser.add_argument('--hrv_win_len', type=int, default=30,
                        help='window length to decide which h5 file to use, units=secs')
    parser.add_argument('--models', type=list, default=['CNN_20', 'CNN_50', 'CNN_100', 'LSTM_20', 'LSTM_50', 'LSTM_100']
                        , help='save prediction probability')
    parser.add_argument('--prediction_type', type=str, default='dftest', help='dataset to predict')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))