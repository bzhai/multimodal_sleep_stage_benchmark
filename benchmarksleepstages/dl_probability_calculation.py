from dataset_builder_loader.data_loader import *
from utilities.utils import *
import argparse

import os
import sys
import pandas as pd
from tensorflow.keras.backend import set_session
from keras.models import *
import tensorflow as tf
from sleep_stage_config import Config

np.set_printoptions(suppress=True)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = False  # to log device placement (on which device the operation ran)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras


def main(args):

    print_args(args)
    np.random.seed(42)
    cfg = Config()
    data_loader = DataLoader(cfg, args.modality, args.num_classes, 20)
    print("Loading test dataset from %s" % cfg.HRV30_ACC_STD_PATH)
    df_train, df_test, featnames = data_loader.load_df_dataset()
    if args.prediction_type == "dftest":
        del df_train
        df_pred = df_test
    else:
        del df_test
        df_pred = df_train
    print("...The scratch dataset loading is Done.    ..")
    stage_output_folder = cfg.STAGE_OUTPUT_FOLDER_HRV30s
    df_pred = df_pred[['mesaid', 'linetime', 'activity', 'gt_sleep_block', 'stages']]
    mat = {}
    for model in tqdm(args.models):
        dl_len = int(model.split('_')[1])
        data_loader = DataLoader(cfg, args.modality, args.num_classes, dl_len)
        data_loader.load_windowed_data()
        Xtest = data_loader.x_test
        trained_model = "model_%d_stages_%ds_%s_seq_%s.pkl" % (args.num_classes, args.hrv_win_len, model, args.modality)

        assert Xtest.shape[0] == df_pred.shape[0], print("dftest shape: %s is different to Xtest shape: %s" %
                                                        (str(df_pred.shape), str(Xtest.shape)))
        assert Xtest.shape[0] == df_pred.shape[0], exit()
        print("Loading trained model: %s" % trained_model)
        tf_model = load_model(os.path.join(stage_output_folder[args.num_classes], trained_model))
        print("Model loaded from disk!")
        predictions = tf_model.predict(Xtest)
        mat[model] = predictions

        df_pred = df_pred.reset_index(drop=True)
        clfs_labels_dic = {}
        tmp_label_list = []
        for label in np.arange(args.num_classes):  # add the label for each class prediction
            tmp_label_list.append(model + "_" + str(label))
        clfs_labels_dic[model] = tmp_label_list
        tmp_df = pd.DataFrame(mat[model], columns=tmp_label_list)
        df_pred = pd.concat([df_pred, tmp_df], axis=1)
        df_pred[model] = df_pred[clfs_labels_dic[model]].apply(lambda x: arg_max_pred(x), axis=1)

    path_save_prediction = os.path.join(stage_output_folder[args.num_classes], "ensemble_%s" % args.modality,
                                        "%d_stages_%s_%ds_pred_probability_%s.csv" %
                                        (args.num_classes, args.prediction_type, args.hrv_win_len, args.modality))
    df_pred["stages"] = df_pred["stages"].astype(int)
    df_pred.to_csv(path_save_prediction, index=False)



def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--modality', type=str, default="all", help='the modality to use.')
    parser.add_argument('--num_classes', type=int, default=3, help='number of classes or labels')
    parser.add_argument('--hrv_win_len', type=int, default=30,
                        help='window length to decide which h5 file to use, units=secs')
    parser.add_argument('--models', type=list, default=['CNN_20', 'CNN_50', 'CNN_100', 'LSTM_20', 'LSTM_50', 'LSTM_100'], help='save prediction probability')
    parser.add_argument('--prediction_type', type=str, default='dftest', help=' dataset to predict',
                        choices={"dftest", "dftrain"})
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))