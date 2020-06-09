import logging
from utilities.utils import log_print_inference
from sklearn.model_selection import train_test_split
from utilities.utils import *
from datetime import datetime
import random
import tensorflow as tf
import sys
from tensorboard.plugins.hparams import api as hp
from six.moves import xrange  # pylint: disable=redefined-builtin
from absl import app
from absl import flags
import os
from pathlib import Path
import numpy as np
from sleep_stage_config import Config
from dataset_builder_loader.data_loader import DataLoader
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel(logging.ERROR)
# set this TensorFlow session as the default session for Keras

# #################### Ben track settings######################################################
# EXPERIMENT = "Use raw dataset of HR and accelerometer for sleep stage classification "
# The code in this program is to
EXPERIMENT = "Use Activity alone with CNN model "
SETTINGS = "1 CNN network with sleep-wake raw cache"
flags.DEFINE_integer("num_session_groups", 30, "The approximate number of session groups to create.", )
flags.DEFINE_integer("epochs", 1, "Number of epochs per trial.", )
flags.DEFINE_integer("batch_size", 128, "training batch size", )
flags.DEFINE_string("nn_type", "LSTM", "define the neural network type'")
flags.DEFINE_integer("seq_len", 100, "the window length")
flags.DEFINE_integer("num_classes", 3, "number of classes or labels")
flags.DEFINE_string("modality", "all", "the modality to use.")
flags.DEFINE_integer("hrv_win_len", 30, "hrv window length")
flags.DEFINE_integer("gpu_index", 0, "set the index of visible GPU you prefer to use")
flags.DEFINE_integer("rand_seed", 42, "number of random seed")
flags.DEFINE_integer("summary_freq", 6400, "Summaries will be written every n steps, where n is the value of this flag.",
                     )
flags.DEFINE_string("search_method", "grid", "searching method for HP tuning, grid: is for grid search, "
                                             "random: is for random search ")
GPU_INDEX = 0


def init_gpu_setting(data_parallelism=False):
    #tf.config.set_soft_device_placement(True)
    #tf.debugging.set_log_device_placement(True)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only use the first GPU
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.experimental.set_visible_devices(gpus[flags.FLAGS.gpu_index], 'GPU')
            # logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            # print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)


# HP_CON1D_LAYER_1_FILTER = hp.HParam('conv1d_filter', hp.Discrete([32, 64, 128]))

# HP_DROPOUT = hp.HParam('dropout', hp.Discrete([0.5, 0.75, 0.9]))
# HP_LEARNING_RATE = hp.HParam('l_rate', hp.Discrete([0.005, 0.05, 0.5]))
HP_LSTM_LAYERS = hp.HParam("lstm_layers", hp.Discrete([1, 2, 3]))
# HP_LSTM_HIDDEN = hp.HParam('num_hidden', hp.Discrete([32]))
HP_LSTM_HIDDEN = hp.HParam('num_hidden', hp.Discrete([64, 128, 256]))
# HP_DENSE_LAYERS = hp.HParam("dense_layers", hp.IntInterval(1, 3))
# HP_POOLING = hp.HParam('max_polling', hp.Discrete([2, 4, 6]))
# HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd', 'rmsprop']))
METRIC_ACCURACY = 'accuracy'
# we may need F1 here as well.
HPARAMS = [
    # HP_DROPOUT,
    # HP_LEARNING_RATE,
    HP_LSTM_LAYERS,
    HP_LSTM_HIDDEN
]

METRICS = [
    hp.Metric(
        "epoch_accuracy", group="validation", display_name="accuracy (val.)",
    ),
    hp.Metric("epoch_loss", group="validation", display_name="loss (val.)",),
    hp.Metric(
        "batch_accuracy", group="train", display_name="accuracy (train)",
    ),
    hp.Metric("batch_loss", group="train", display_name="loss (train)",),
]


def build_multilayer_lstm_hp(input_dim, hparams, seed, num_classes):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(input_dim))
    for i in xrange(hparams[HP_LSTM_LAYERS]):
        if i > 1:
            model.add(
                tf.keras.layers.LSTM(units=hparams[HP_LSTM_HIDDEN], return_sequences=True)
            )
    model.add(tf.keras.layers.LSTM(units=hparams[HP_LSTM_HIDDEN]))
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    opt = tf.keras.optimizers.RMSprop()
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model


def prepare_data(cfg, modality, num_classes, seq_len):
    data_loader = DataLoader(cfg, modality, num_classes, seq_len)
    data_loader.load_windowed_data()
    return ((data_loader.x_train, data_loader.y_train), (data_loader.x_val, data_loader.y_val))


def run(data, base_path, session_id, hparams):
    """ Run a training/validation session
    Args:
        data: the train and validation data
        base_path: The top-level logdir to which to write summary data.
        session_id: A unique string ID for this session.
        hparams: A dict mapping hyperparameters in `HPARAMS` to values.
    """
    logdir = os.path.join(base_path, session_id)
    ((x_train, y_train), (x_val, y_val)) = data
    model = build_multilayer_lstm_hp(input_dim=x_train.shape[1:], hparams=hparams, seed=session_id,
                                     num_classes=flags.FLAGS.num_classes)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=logdir,
        histogram_freq=2,
        update_freq=flags.FLAGS.summary_freq,
        profile_batch=0
    )
    hparams_callback = hp.KerasCallback(logdir, hparams)
    model.fit(
        x=x_train,
        epochs=flags.FLAGS.epochs,
        y=y_train,
        batch_size=flags.FLAGS.batch_size,
        shuffle=False,
        validation_data=(x_val, y_val),
        callbacks=[tensorboard_callback, hparams_callback], )
    model_file = "model_%d_stages_%ds_%s_%d_seq_%s_%s.h5" % (flags.FLAGS.num_classes, flags.FLAGS.hrv_win_len,
                                                             flags.FLAGS.nn_type, flags.FLAGS.seq_len,
                                                             flags.FLAGS.modality,
                                                             convert_args_to_str(hparams)
                                                             )
    model_save_path = os.path.join(logdir, model_file)
    model.save(model_save_path, save_format='h5')


def run_all_grid_search(cfg, verbose=True):
    """Perform random search over the hyperparameter space.
    Arguments:
        cfg: configuration file
      verbose: If true, print out each run's name as it begins.
    """
    print("current module name: ")
    print(os.path.abspath(__file__))
    args = convert_abseil_args_dict(flags, os.path.basename(__file__))
    print("args used in this experiment")
    print(args)
    data = prepare_data(cfg, flags.FLAGS.modality, flags.FLAGS.num_classes, flags.FLAGS.seq_len)
    time_of_run = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir_root = os.path.join(
        cfg.LSTM_FOLDER, "experiment_results", os.path.basename(__file__))
    log_path = os.path.join(log_dir_root, time_of_run)
    if not os.path.isdir(os.path.abspath(log_path)):
        Path(log_path).mkdir(parents=True, exist_ok=True)

    write_arguments_to_file(args, os.path.join(log_path, 'args.txt'))
    with tf.summary.create_file_writer(log_path).as_default():
        hp.hparams_config(hparams=HPARAMS, metrics=METRICS)
    session_index = 0  # across all session groups
    for num_lstm_layers in HP_LSTM_LAYERS.domain.values:
        for num_lstm_hidden in HP_LSTM_HIDDEN.domain.values:
                hparams = {
                    HP_LSTM_LAYERS: num_lstm_layers,
                    HP_LSTM_HIDDEN: num_lstm_hidden
                }
                hparams_string = str(hparams)
                session_id = str(session_index)
                session_index += 1
                if verbose:
                    print(
                        "--- Running training session %d"
                        % session_index
                    )
                    print(hparams_string)
                run(
                    data=data,
                    base_path=log_path,
                    session_id=session_id,
                    hparams=hparams,
                )


def run_all_random_search(cfg, verbose=True):
    """Perform random search over the hyperparameter space.
    Arguments:
        cfg: configuration file
      verbose: If true, print out each run's name as it begins.
    """
    print("current module name: ")
    print(os.path.abspath(__file__))
    args = convert_abseil_args_dict(flags, os.path.basename(__file__))
    print("args used in this experiment")
    print(args)
    data = prepare_data(cfg, flags.FLAGS.modality, flags.FLAGS.num_classes, flags.FLAGS.seq_len)
    rng = random.Random(flags.FLAGS.rand_seed)
    time_of_run = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir_root = os.path.join(
        cfg.LSTM_FOLDER, "experiment_results", os.path.basename(__file__))
    log_path = os.path.join(log_dir_root, time_of_run)
    if not os.path.isdir(os.path.abspath(log_path)):
        os.makedirs(os.path.abspath(log_path))

    write_arguments_to_file(args, os.path.join(log_path, 'args.txt'))
    with tf.summary.create_file_writer(log_path).as_default():
        hp.hparams_config(hparams=HPARAMS, metrics=METRICS)
    sessions_per_group = 2
    num_sessions = flags.FLAGS.num_session_groups * sessions_per_group
    session_index = 0  # across all session groups
    for group_index in xrange(flags.FLAGS.num_session_groups):
        hparams = {h: h.domain.sample_uniform(rng) for h in HPARAMS}
        hparams_string = str(hparams)
        for repeat_index in xrange(sessions_per_group):
            session_id = str(session_index)
            session_index += 1
            if verbose:
                print(
                    "--- Running training session %d/%d"
                    % (session_index, num_sessions)
                )
                print(hparams_string)
                print("--- repeat #: %d" % (repeat_index + 1))
            run(
                data=data,
                base_path=log_path,
                session_id=session_id,
                hparams=hparams,
            )


def main(unused_argv):
    # shutil.rmtree()
    cfg = Config()
    init_gpu_setting()
    print("Tuning parameters for multiple layers CNN")
    run_all_grid_search(cfg, verbose=True)
    print("Tuning is completed!")


if __name__ == '__main__':
    app.run(main)
