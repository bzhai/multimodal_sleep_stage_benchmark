from sklearn import metrics
from dataset_builder_loader.data_loader import *

from utilities.utils import *
from sleep_stage_config import Config
import os
import argparse
import sys

from benchmarksleepstages.models.ml_models import *


def run_statistic_ml_test(model, Xtrain, Xtest, Ytrain, Ytest, model_name):
    print("start fitting model %s \n" % model)
    model.fit(Xtrain, Ytrain)

    pred = model.predict(Xtest)
    print(" - Final Acuracy: %.4f" % (metrics.accuracy_score(Ytest, pred)))
    print(" - Final F1: %.4f" % (metrics.f1_score(Ytest, pred, average="macro")))
    print(classification_report(Ytest, pred))
    return model, pred

def build_feature_list(feature_type, full_feature):
    hrv_feature = ["Modified_csi", "csi", "cvi", "cvnni", "cvsd", "hf", "hfnu", "lf", "lf_hf_ratio", "lfnu", "max_hr",
                    "mean_hr", "mean_nni", "median_nni", "min_hr", "nni_20", "nni_50", "pnni_20", "pnni_50",
                    "range_nni", "ratio_sd2_sd1", "rmssd", "sd1", "sd2", "sdnn", "sdsd", "std_hr", "total_power",
                    "triangular_index", "vlf"]
    if feature_type == 'all':
        return full_feature
    elif feature_type == 'hrv':
        return hrv_feature
    elif feature_type == 'acc':
        full_feature = [ele for ele in full_feature if ele not in hrv_feature]
        if "line" in full_feature:
            full_feature.remove('line')
        if 'activity' in full_feature:
            full_feature.remove('activity')
        return full_feature
    elif feature_type == 'hr':
        return ['mean_hr']

# here I did the standardisation dataset to speed up the build of pipeline

def main(args):
    cfg = Config()
    data_loader = DataLoader(cfg, args.modality, args.num_classes, 20)
    print_args(args)
    print("loading H5 dataset from %s" % cfg.HRV30_ACC_STD_PATH)

    data_loader.load_ml_data()
    df_test = data_loader.load_df_dataset()[1]
    scaler = load_scaler(cfg.STANDARDISER[args.hrv_win_len])
    print('-' * 100)
    print("H5 dataset loading is completed !")
    print("Data loading is Done...")
    # Run models:

    saved_models = []
    print("...Runing models...")
    models = build_ml_models()
    for (model_name, model) in models:
        model, df_test[model_name] = run_statistic_ml_test(model, data_loader.x_train, data_loader.x_test,
                                                          data_loader.y_train, data_loader.y_test, model_name)

        saved_models.append([model_name, model])
        print("...Done with %s..." % model_name)
    print("...Done...")

    print("...Saving %d_stages_%d_ml_%s.csv..." % (args.num_classes, args.hrv_win_len, args.modality))
    #Creating a table with output predictions for all the different ML methods
    stage_output_folder = cfg.STAGE_OUTPUT_FOLDER_HRV30s[args.num_classes]
    df_test[[m[0] for m in models]] = df_test[[m[0] for m in models]].astype(float)
    df_test["gt_sleep_block"] = df_test["gt_sleep_block"].astype(int)
    df_test["stages"] = df_test["stages"].astype(int)
    df_test["activity"] = df_test["activity"].fillna(0.0)
    df_test[["mesaid", "linetime", "activity", "stages", "gt_sleep_block"] + [m[0] for m in models]].to_csv(
        os.path.join(stage_output_folder, "%d_stages_%ds_ml_%s.csv" % (args.num_classes, args.hrv_win_len, args.modality)), index=False)
    print("...Done...")

    dict_model = {"scaler": scaler, "models": saved_models}
    print("...Saving trained models for %d_stages_ml_%s.pkl..." % (args.num_classes, args.modality))
    with open(os.path.join(stage_output_folder, "%d_stages_%ds_ml_%s.pkl" % (args.num_classes, args.hrv_win_len, args.modality)), "wb") as f:
        pickle.dump(dict_model, f)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--modality', type=str, default='all',help='the raw modality to use.')
    parser.add_argument('--num_classes', type=int, default=2, help='number of classes or labels')
    parser.add_argument('--use_cache', type=int, default=1, help='loading dataset from cache')
    parser.add_argument('--training', type=int, default=0, help='training and predicting')
    parser.add_argument('--hrv_win_len', type=int, default=30,
                        help='window length to decide which h5 file to use, units=secs')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

