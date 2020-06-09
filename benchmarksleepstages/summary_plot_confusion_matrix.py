import sys
import argparse
from sleep_stage_config import Config
#from scipy.stats import ttest_ind
from utilities.utils import *
import matplotlib.pyplot as plt
import seaborn as sns

def main(args):
    cfg = Config()
    default_ml = ["Random_forest_300"]
    default_nn = ["CNN_20_ENMO_HRV", "CNN_50_ENMO_HRV", "CNN_100_ENMO_HRV", "LSTM_20_ENMO_HRV", "LSTM_50_ENMO_HRV",
                 "LSTM_100_ENMO_HRV"]
    # defaultnn = [x + "_" + args.feature_type for x in defaultnn]
    algs = default_nn + default_ml
    feature_to_modality = {'ENMO_HRV':'all', 'ENMO':'acc', 'HRV':'hrv'}
    total_df = [] # will load the prediction summary file
    ml_clfs_file = cfg.STAGE_OUTPUT_FOLDER_HRV30s
    for num_classes in np.arange(2, 6):
        result_folder = ml_clfs_file[num_classes]
        minutes_prediction_summary_file = os.path.join(result_folder, "summary", "%d_stages_summary_%s.csv"
                                                       % (num_classes, args.feature_type))
        tmp_df = pd.read_csv(minutes_prediction_summary_file)
        tmp_df = tmp_df.rename(columns={'Unnamed: 0': 'Algorithms'})
        tmp_df['tasks'] = 'task_' + str(num_classes)
        total_df.append(tmp_df)

        results_outfile = os.path.join(result_folder, "%d_stages_%ds_%s_%s.csv" %
                                       (num_classes, args.hrv_win_len, args.best_algs[num_classes],
                                        feature_to_modality[args.feature_type]))
        pred_df = pd.read_csv(results_outfile)
        if len(pred_df['stages'].unique()) != num_classes:
            pred_df['stages'] = pred_df['stages'].apply(lambda x: cast_sleep_stages(x, classes=num_classes))
        plot_save_confusion_matrix(y_true=pred_df['stages'].values, y_pred=pred_df[args.best_algs[num_classes]].values,
                                   class_names=convert_int_to_label_list(num_classes), location=cfg.EXPERIMENT_RESULTS_ROOT_FOLDER,
                                   title="%d_stages_%s" % (num_classes, args.best_algs[num_classes]))

    column_to_vis = []
    total_df = pd.concat(total_df)
    metrics = ["accuracy", "f1-score", "precision", "recall", "specificity"]
    for metric in metrics:
        total_df[metric] = total_df[metric].apply(lambda x: split_and_extract(x, 0))
        column_to_vis.append("mean_%s" % metric)
    print("extraction is complete!")
    # filter out base line
    total_df = total_df[~total_df['Algorithms'].isin(['always_0', 'always_1', 'always_2', 'always_3', 'always_4', 'stages'])]
    total_df = total_df[total_df['Algorithms'].isin(algs)]
    task_readable = {"task_2": "Sleep, wake", "task_3": "Wake, Non REM sleep, REM sleep",
                   "task_4":  "Wake, deep, light, REM sleep",
                   "task_5": "Wake, N1, N2, N3, REM sleep"}
    total_df['tasks'] = total_df['tasks'].apply(lambda x: task_readable[x])
    alg_readable = pd.read_csv(cfg.ALG_READABLE_FILE)
    alg_readable = dict(zip(alg_readable.Old_Name, alg_readable.New_Name))
    total_df['Algorithms'] = total_df['Algorithms'].apply(lambda x: alg_readable[x])
    plot_bar_group_paper(total_df, metrics, args.hrv_win_len, cfg.EXPERIMENT_RESULTS_ROOT_FOLDER)


def plot_bar_group_paper(total_df, metrics, win_len, output_path):
    # https://seaborn.pydata.org/tutorial/color_palettes.html
    for metric in metrics:
        c_palette = sns.color_palette("colorblind", len(total_df.Algorithms.unique()))
        rank = total_df[metric].argsort().argsort()
        sns.set_context("paper")
        sns.set_style("whitegrid")
        g = sns.catplot(x="tasks", y=metric, hue="Algorithms", kind="bar", height=8, legend_out=False
                    , data=total_df[['Algorithms', 'tasks', metric]], palette=c_palette)
        g.set(xlabel='Tasks and classifiers')
        g.set(ylim=(15, 100))
        plt.savefig(os.path.join(output_path, metric + "_%ds_" % win_len + ".png"), dpi=200)



def plot_scatter_plot(hp_df, title=""):
    print("print is completed.")


def split_and_extract(x, index):
    t1 = x.split('+-')
    return float(t1[0])


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=4, help='number of classes or labels')
    parser.add_argument('--feature_type', type=str, default='ENMO_HRV', help='The feature type for DNN')
    parser.add_argument('--hrv_win_len', type=int, default=30,
                        help='window length to decide which h5 file to use, units=secs')
    parser.add_argument('--best_algs', type=dict, default={2:'CNN_100', 3:'LSTM_50',4: 'LSTM_50', 5:'LSTM_50'}, help='The best algorithms for plot')
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))