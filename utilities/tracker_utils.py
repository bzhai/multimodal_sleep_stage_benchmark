import csv
import keras
from datetime import datetime
import pandas as pd
import os
import tensorflow as tf
import matplotlib.pyplot as plt

# ######################################################################################################################
# ####################################### classes for tracking start from here  ########################################
# ######################################################################################################################

class KerasLossHistory(keras.callbacks.Callback):
    """
    This class will log all experiments details into csv file, the records include experiment setting,
    key metrics after each epoch and batch training, this class only responsible for writing.
    !TODO do we need to record batch results as a picture?
    """

    def __init__(self, experiment="Default Experiment", settings="Default Setting"
                 , timestamp=datetime.now().strftime("%Y%m%d-%H%M%S")
                 , sum_log_path='summary_log.csv', details_log_path='details_log.csv'):
        super(KerasLossHistory, self).__init__()
        self.sum_log_path = sum_log_path
        self.details_log_path = details_log_path
        self.timestamp = timestamp
        self.details_df = pd.DataFrame(
            columns=['Timestamp', 'Experiment', 'Settings', 'Epoch_num', 'batch_num', 'train_acc', 'loss'])
        self.summary_df = pd.DataFrame(columns=['Timestamp', 'Experiment', 'Settings', 'Epoch_num'
            , 'train_acc', 'loss', 'val_acc', 'val_loss'])
        self.experiment = experiment
        self.settings = settings
        self.epoch_counter = 0
        self.batch_counter = 0
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}

    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}

        # check if the log path is exist
        if not os.path.isdir(os.path.dirname(self.sum_log_path)):
            os.mkdir(os.path.dirname(self.sum_log_path))
        if not os.path.isdir(os.path.dirname(self.details_log_path)):
            os.mkdir(os.path.dirname(self.details_log_path))
        # check if the log file existed
        if not os.path.exists(self.sum_log_path):
            self.summary_df.to_csv(self.sum_log_path, header=True, index=False)
        if not os.path.exists(self.details_log_path):
            self.details_df.to_csv(self.details_log_path, header=True, index=False)

    def on_train_end(self, logs=None):
        self.loss_plot('epoch')

    def on_batch_end(self, batch, logs={}):
        batch_summary = [self.timestamp, self.experiment, self.settings, self.epoch_counter, self.batch_counter,
                         logs.get('acc'), logs.get('loss')]
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))
        with open(self.details_log_path, 'a', newline='') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(batch_summary)
        self.batch_counter = self.batch_counter + 1

    def on_epoch_end(self, epoch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

        epoch_summary = [self.timestamp, self.experiment, self.settings, self.epoch_counter,
                         logs.get('acc'), logs.get('loss'), logs.get('val_acc'), logs.get('val_loss')]
        with open(self.sum_log_path, 'a', newline='') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(epoch_summary)
        self.epoch_counter = self.epoch_counter + 1
        self.batch_counter = 0

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="center")
        # plt.show()
        plt.savefig(os.path.join(os.path.dirname(self.details_log_path), loss_type + '.png'))

    def save_model(self, model):
        model_json = model.to_json()
        with open(os.path.join(os.path.dirname(self.details_log_path), "model.json"), "w") as json_file:
            json_file.write(model_json)
        model.save_weights(os.path.join(os.path.dirname(self.details_log_path), "model.h5"))
        print("Saved model to disk")


class TFTrialTracker(object):
    def __init__(self, experiment="Default Experiment", settings="Default Setting", result_path="",
                 run_ID=datetime.now().strftime("%Y%m%d-%H%M%S"), ml_flow_uid='', args=None, label_value=None,
                 targets_name=None):
        """
        Tensor flow trail logger will track each run results, includes training, val and test. which can be
        useful for hyper-parameter tuning. However this tracker class is only tracking per trail train, validation and
        test results, it doesn't track performance metrics. To use performance metrics, please see function
        log_print_inference for binary classification plot_roc_curve
        :param experiment: the name of the study, purpose
        :param settings: optional the settings of the parameters
        :param result_path:  result path should start from mlruns then organised by python file name
        :param run_ID: the timestamp and ML flow uid
        :param args: the hyper parameters we want to record
        :param ml_flow_uid: the ml flow uid
        """

        self.sum_log_directory = result_path
        self.sum_log_path = os.path.join(result_path, 'summary_log.csv')
        self.run_ID = run_ID + "-" + ml_flow_uid
        self.tensor_board_path = os.path.join(result_path, self.run_ID)
        self.details_log_path = os.path.join(self.tensor_board_path, 'details_log.csv')
        self.details_df = pd.DataFrame(
            columns=['Timestamp', 'run_ID', 'Experiment', 'Settings', 'Epoch_num', 'batch_num', 'train_acc',
                     'train_loss'])
        self.summary_df = pd.DataFrame(columns=['run_ID', 'experiment', 'Settings', 'Timestamp', 'Epoch_num',
                                                'train_acc', 'train_loss', 'val_acc', 'val_loss', 'test_acc',
                                                'test_loss'])
        self.experiment = experiment
        self.settings = settings
        self.epoch_counter = 0
        self.batch_counter = 0
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}
        self.test_loss = {'batch': [], 'epoch': []}
        self.test_acc = {'batch': [], 'epoch': []}
        self.args = args
        self.saver = tf.train.Saver()
        self._create_folder()
        self.on_train_begin()
        self.sess = None
        self.label_value = label_value
        self.targets_name = targets_name

    def _create_folder(self):
        # check if the log path is exist
        if not os.path.isdir(self.sum_log_directory):
            os.mkdir(self.sum_log_directory)
        if not os.path.isdir(self.tensor_board_path):
            os.mkdir(self.tensor_board_path)
        # check if the log file existed
        if not os.path.exists(self.sum_log_path):
            self.summary_df.to_csv(self.sum_log_path, header=True, index=False)
        if not os.path.exists(self.details_log_path):
            self.details_df.to_csv(self.details_log_path, header=True, index=False)

    def on_train_begin(self, sess=tf.Session()):
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}
        self.sess = sess
        self.save_graph_def(sess.graph)

    def log_inference(self, y_test, yhat, epochs=0):
        self.save_tf_model(self.sess)
        return log_print_inference(y_test, yhat, self.label_value, self.targets_name, epochs, self.tensor_board_path)

    def on_batch_end(self, logs={}):
        """
        please pass in a dict with acc, loss, val_acc, val_loss
        we need pass a dictionary for tracking
        :param batch:
        :param logs:
        :return:
        """
        batch_summary = [datetime.now().strftime("%Y%m%d-%H%M%S-%f"), self.run_ID, self.experiment, self.settings,
                         self.epoch_counter, self.batch_counter, logs.get('acc'), logs.get('loss')]
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))
        with open(self.details_log_path, 'a', newline='') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(batch_summary)
        self.batch_counter = self.batch_counter + 1

    def on_epoch_end(self, logs={}):
        """
        pass in last batch train and val performance
        :param epoch: current epoch number
        :param logs: pass in dict: {loss: train loss, acc: train acc, val_loss: x, val_acc: x}
        :return:
        """
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

        epoch_summary = [self.run_ID, self.experiment, self.settings, datetime.now().strftime("%Y%m%d-%H%M%S-%f"),
                         self.epoch_counter, logs.get('acc'), logs.get('loss'), logs.get('val_acc'),
                         logs.get('val_loss'), logs.get('test_acc'), logs.get('test_loss')]
        with open(self.sum_log_path, 'a', newline='') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(epoch_summary)
        self.epoch_counter = self.epoch_counter + 1
        self.batch_counter = 0
        self.loss_plot('epoch')

    def on_train_end(self, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

        epoch_summary = [self.run_ID, self.experiment, self.settings, datetime.now().strftime("%Y%m%d-%H%M%S-%f"),
                         self.epoch_counter, logs.get('acc'), logs.get('loss'), logs.get('val_acc'),
                         logs.get('val_loss'), logs.get('test_acc'), logs.get('test_loss')]
        with open(self.sum_log_path, 'a', newline='') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(epoch_summary)
        self.epoch_counter = self.epoch_counter + 1
        self.batch_counter = 0

    def loss_plot(self, loss_type):
        """
        we will plot train and validation on one chart
        :param loss_type:
        :return:
        """
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        # if loss_type == 'epoch':
        # val_acc
        plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
        # val_loss
        plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="center")
        conf_path = os.path.join(os.path.dirname(self.details_log_path),
                                 str(np.random.randint(0, 100)) + loss_type + '.png')
        plt.savefig(conf_path)
        return conf_path

    def save_tf_model(self, sess=None, step=0):
        if sess is not None:
            self.saver.save(sess, save_path=os.path.join(self.tensor_board_path, 'saved_model'), global_step=step)

    def save_graph_def(self, graph=tf.Graph()):
        from google.protobuf import json_format
        if graph is not None:
            graph_def = graph.as_graph_def()
            json_string = json_format.MessageToJson(graph_def)
            with open(os.path.join(self.tensor_board_path, "model_definition.json"), 'w') as outfile:
                json.dump(json_string, outfile)