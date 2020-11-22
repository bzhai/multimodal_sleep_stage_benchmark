import os
import sys
import argparse
from utilities.utils import *
from sleep_stage_config import Config
from dataset_builder_loader.data_loader import DataLoader
from benchmarksleepstages.models.dl_models import *
import tensorflow as tf


def main(args):
    print_args(args)
    config = Config()
    data_loader = DataLoader(config, args.modality, args.num_classes, args.seq_len)
    stage_output_folder = config.STAGE_OUTPUT_FOLDER_HRV30s[args.num_classes]
    model_output_name = "model_%d_stages_30s_%s_%d_seq_%s.pkl" % (args.num_classes, args.nn_type,
                                                                  args.seq_len, args.modality)
    pred_output_name = "%d_stages_30s_%s_%d_%s.csv" % (args.num_classes, args.nn_type, args.seq_len, args.modality)
    if not os.path.isdir(stage_output_folder):
        os.mkdir(stage_output_folder)
    saved_model = os.path.join(stage_output_folder, model_output_name)
    data_loader.load_windowed_data()
    input_dim = data_loader.x_train.shape[1:]
    # setup the tensorboard
    time_of_run = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_root = os.path.join(os.getcwd(), "experiment_results", os.path.basename(__file__))
    tensorboard_path = os.path.join(log_root, time_of_run)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_path)
    if args.training == 1:
        if args.nn_type == "CNN":
            model = build_cnn_tf_v1(input_dim, args.num_classes)
        elif args.nn_type == "LSTM":
            model = build_lstm_tf_v1(input_dim, args.num_classes)
        else:
            raise Exception("model is not found")
        # prepare tensor board logging
        print("Training Shape: ", data_loader.x_train.shape[1:])
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)
        # fit the model
        model.fit(data_loader.x_train, data_loader.y_train, epochs=args.epochs, batch_size=args.batch_size, verbose=1,
                  validation_data=(data_loader.x_val, data_loader.y_val),
                  callbacks=[tensorboard_callback])
        print("Model trained!")
        model.save(saved_model)
        print("model is saved at %s" % saved_model)
    else:
        model = tf.keras.models.load_model(saved_model)
        print("Model loaded from disk!")

    # forward propagation
    pred = model.predict(data_loader.x_test)
    pred_results_path = os.path.join(stage_output_folder, pred_output_name)
    df_test = data_loader.load_df_dataset()[1]
    save_prediction_results(df_test, pred, pred_results_path, args.nn_type, args.seq_len)
    print("Predictions made. Result saved to %s." % pred_results_path)
    log_print_metrics(pred, data_loader.y_test, args.epochs, args.nn_type, args.num_classes, tensorboard_path, args)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--nn_type', type=str, default="LSTM", help='define the neural network type')
    parser.add_argument('--seq_len', type=int, default=100, help='the window length unit is sleep epochs')
    parser.add_argument('--modality', type=str, default="acc", help='the modality to use.')
    parser.add_argument('--num_classes', type=int, default=3, help='number of classes or labels')
    parser.add_argument('--training', type=int, default=1, help='training or predicting')
    parser.add_argument('--epochs', type=int, default=20, help='training epochs')
    parser.add_argument('--batch_size', type=int, default=512, help='training batch size')
    parser.add_argument('--hrv_win_len', type=int, default=30,
                        help='hrv window length to decide which h5 file to use, unit is secs')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
