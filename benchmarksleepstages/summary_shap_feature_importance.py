from dataset_builder_loader.make_h5py_file import *
from dataset_builder_loader.data_loader import DataLoader
from sleep_stage_config import Config
import os
import argparse
import sys
import pickle
import seaborn as sns
import shap
import matplotlib.colors as mcolors

# matplotlib.use('Agg')

def main(args):
    cfg = Config()
    trained_clfs_file = os.path.join(cfg.STAGE_OUTPUT_FOLDER_HRV30s[args.num_classes],
                                     "%d_stages_%ds_ml_%s.pkl" % (
                                         args.num_classes, args.hrv_win_len, args.modality))

    with open(trained_clfs_file, 'rb') as p:
        trained_clfs = pickle.load(p)
        print(trained_clfs.keys())
    print("load pickle files are completed")
    assert args.rf_trees in [300], print("only 20, 100 random forest estimator size accepted")
    random_forest = trained_clfs['models'][3][1]
    print("Complete loading train classifier :%s" % random_forest)

    stage_output_folder = cfg.STAGE_OUTPUT_FOLDER_HRV30s[args.num_classes]
    if args.load_pretrain == 1:
        with open(os.path.join(stage_output_folder, "shap_value_feature_importance_%d_classes_%s_%ds"
                                                                      % (args.num_classes, args.modality,
                                                                         args.hrv_win_len) + ".pkl"), "rb") as f:
            pre_train_dict = pickle.load(f)
            X = pre_train_dict['X']
            shap_values = pre_train_dict['shape_value']
        #explainer = shap.TreeExplainer(random_forest)
    else:
        print("loading H5 dataset from %s" % cfg.HRV30_ACC_STD_PATH)
        data_loader = DataLoader(cfg, args.modality, args.num_classes, args.seq_len)
        df_train, df_test, featurename = data_loader.load_df_dataset()
        print("loading H5 dataset is completed!")
        X = df_train[featurename].iloc[np.random.choice(df_train.shape[0], args.shap_samples, replace=False)]
        del trained_clfs
        del df_train
        del df_test
        # explain the model's predictions using SHAP values
        explainer = shap.TreeExplainer(random_forest)
        shap_values = explainer.shap_values(X)
        shap_dump = {"shape_value": shap_values, "X": X}
        with open(os.path.join(stage_output_folder[args.num_classes], "shap_value_feature_importance_%d_classes_%s_%ds"
                                                                              % (args.num_classes, args.modality, args.hrv_win_len) + ".pkl"), "wb") as f:
            pickle.dump(shap_dump, f)

    # summarize the effects of all the features
    labels = convert_int_to_label_list(args.num_classes)

    if args.num_classes == 2:
        def class_color(idx):
            class_colors = {0: mcolors.CSS4_COLORS['gold'], 1: mcolors.CSS4_COLORS['midnightblue']}
            return class_colors[idx]
        shap.summary_plot(shap_values, X, plot_type='bar', class_names=labels, color=class_color, max_display=40)

    elif args.num_classes == 3:
        def class_color(idx):
            class_colors = {0: mcolors.CSS4_COLORS['gold'], 1: mcolors.CSS4_COLORS['mediumblue'], 2: mcolors.CSS4_COLORS['skyblue']}
            return class_colors[idx]
        shap.summary_plot(shap_values, X, plot_type='bar', class_names=labels, color=class_color)

    elif args.num_classes == 4:
        def class_color(idx):
            class_colors = {0: mcolors.CSS4_COLORS['gold'], 1: mcolors.CSS4_COLORS['dodgerblue'],
                            2: mcolors.CSS4_COLORS['darkblue'], 3: mcolors.CSS4_COLORS['skyblue']}
            return class_colors[idx]
        shap.summary_plot(shap_values, X, plot_type='bar', class_names=labels, color=class_color,max_display=40)
    else:
        def class_color(idx):
            class_colors = {0: mcolors.CSS4_COLORS['gold'], 1: mcolors.CSS4_COLORS['royalblue'], 2: mcolors.CSS4_COLORS['blue'], 3: mcolors.CSS4_COLORS['navy'],
                            4: mcolors.CSS4_COLORS['skyblue']}
            return class_colors[idx]
        shap.summary_plot(shap_values, X, plot_type='bar', class_names=labels, color=class_color, max_display=40)
    # shap.force_plot(explainer.expected_value, shap_values[0, :], X.iloc[0, :])

    print("Feature importance ranking is completed")
    print("")

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--modality', type=str, default='all', help='the raw modality to use.')
    parser.add_argument('--num_classes', type=int, default=2, help='number of classes or labels')
    parser.add_argument('--load_pretrain', type=int, default=1, help='loading pre-trained SHAP')
    parser.add_argument('--shap_samples', type=int, default=500, help='training examples to calculate shap value')
    parser.add_argument('--hrv_win_len', type=int, default=30,
                        help='window length to decide which h5 file to use, units=secs')
    parser.add_argument('--rf_trees', type=int, default=300, help='training examples to calculate shap value')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))