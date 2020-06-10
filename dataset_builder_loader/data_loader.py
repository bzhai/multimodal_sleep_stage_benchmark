
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
import os
from sklearn import preprocessing
from random import seed
from random import shuffle
import h5py

from sleep_stage_config import Config
from utilities.utils import *
from sklearn.model_selection import train_test_split


class DataLoader(object):
    """
    a dataset loader for actigraphy
    """

    def __init__(self, cfg, modality, num_classes, seq_len):
        self.config = cfg
        self.modality = modality
        self.num_classes = num_classes
        self.seq_len = seq_len
        self.x_train = []
        self.y_train = []
        self.x_val = []
        self.y_val = []
        self.x_test = []
        self.y_test = []
        self.dl_feature_list = []
        self.ml_feature_list = []
        self.__prepare_feature_list__()

    def __prepare_feature_list__(self):
        if self.modality == "all":
            self.dl_feature_list = ["activity", "mean_nni", "sdnn", "sdsd", "vlf", "lf", "hf", "lf_hf_ratio",
                                 "total_power"]
        elif self.modality == "hrv":
            self.dl_feature_list = ["mean_nni", "sdnn", "sdsd", "vlf", "lf", "hf", "lf_hf_ratio", "total_power"]
        elif self.modality == "acc":
            self.dl_feature_list = ["activity"]
        elif self.modality == "hr":
            self.dl_feature_list = ["mean_nni"]
        self.ml_feature_list = pd.read_csv(self.config.FEATURE_LIST)['feature_list'].values
        self.ml_feature_list = self.__build_feature_list__(self.modality, self.ml_feature_list)

    @staticmethod
    def __check_seq_len__(seq_len):
        if seq_len not in [100, 50, 20]:
            raise Exception("seq_len i error")

    @staticmethod
    def __build_feature_list__(feature_type, full_feature):
        hrv_feature = ["Modified_csi", "csi", "cvi", "cvnni", "cvsd", "hf", "hfnu", "lf", "lf_hf_ratio", "lfnu",
                       "max_hr",
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

    @staticmethod
    def cast_sleep_stages_and_onehot_encode(dataset, num_classes):
        if len(dataset.shape) < 2 and len(set(dataset)) != num_classes:
            dataset = cast_sleep_stages(dataset.astype(int), num_classes)
            if len(dataset.shape) < 2:
                dataset = np.expand_dims(dataset, -1)
            enc = OneHotEncoder(handle_unknown='ignore')
            dataset = enc.fit_transform(dataset).toarray()
            # dataset = tf.keras.utils.to_categorical(dataset, num_classes)
            return dataset
        else:
            return dataset

    def load_windowed_data(self):
        # h5_file = self.config.HRV30_ACC_STD_PATH
        print("Loading h5 dataset from %s" % self.config.HRV30_ACC_STD_PATH)
        # _, dftest, featnames = load_h5_df_dataset(h5_file, useCache=True)
        print("The h5 dataset is loaded from %s" % self.config.HRV30_ACC_STD_PATH)
        cache_path = self.config.NN_ACC_HRV % self.seq_len
        print("...Loading windowed cache dataset from %s" % cache_path)
        with h5py.File(cache_path, 'r') as data:
            if self.modality == "all":
                self.x_train = data["x_train"][:]
                self.y_train = data["y_train"][:]
                self.x_val = data["x_val"][:]
                self.y_val = data["y_val"][:]
                self.x_test = data["x_test"][:]
                self.y_test = data["y_test"][:]
            elif self.modality == "hrv":
                self.x_train = data["x_train"][:, :, 1:]
                self.y_train = data["y_train"][:]
                self.x_val = data["x_val"][:, :, 1:]
                self.y_val = data["y_val"][:]
                self.x_test = data["x_test"][:, :, 1:]
                self.y_test = data["y_test"][:]
            elif self.modality == "acc":
                self.x_train = np.expand_dims(data["x_train"][:, :, 0], -1)
                self.y_train = np.expand_dims(data["y_train"][:], -1)
                self.x_val = np.expand_dims(data["x_val"][:, :, 0], -1)
                self.y_val = np.expand_dims(data["y_val"][:], -1)
                self.x_test = np.expand_dims(data["x_test"][:, :, 0], -1)
                self.y_test = np.expand_dims(data["y_test"][:], -1)
            elif self.modality == "hr":
                self.x_train = np.expand_dims(data["x_train"][:, :, 1], -1)
                self.y_train = np.expand_dims(data["y_train"][:], -1)
                self.x_val = np.expand_dims(data["x_val"][:, :, 1], -1)
                self.y_val = np.expand_dims(data["y_val"][:], -1)
                self.x_test = np.expand_dims(data["x_test"][:, :, 1], -1)
                self.y_test = np.expand_dims(data["y_test"][:], -1)
            data.close()
        if len(self.y_train.shape) < 2 or len(set(self.y_train)) != self.num_classes:
            self.y_train = self.cast_sleep_stages_and_onehot_encode(self.y_train, self.num_classes)
        if len(self.y_test.shape) < 2 or len(set(self.y_test)) != self.num_classes:
            self.y_test = self.cast_sleep_stages_and_onehot_encode(self.y_test, self.num_classes)
        if len(self.y_val.shape) < 2 or len(set(self.y_val)) != self.num_classes:
            self.y_val = self.cast_sleep_stages_and_onehot_encode(self.y_val, self.num_classes)
        return self.y_train, self.y_test, self.y_val

    def load_df_dataset(self):
        df_train, df_test, feature_name = load_h5_dataset(self.config.HRV30_ACC_STD_PATH)
        df_train['stages'] = df_train['stages'].apply(lambda x: cast_sleep_stages(x, classes=self.num_classes))
        df_test['stages'] = df_test['stages'].apply(lambda x: cast_sleep_stages(x, classes=self.num_classes))
        return df_train, df_test, feature_name

    def build_windowed_cache_data(self, win_len):
        self.__check_seq_len__(win_len)
        assert self.modality == "all", "building up cache only works when modality is 'all', " \
                                       "as other modalities included in the cache data file"
        print("Loading H5 dataset....")
        df_train, df_test, feat_names = load_h5_df_dataset(self.config.HRV30_ACC_STD_PATH)
        cache_path = self.config.NN_ACC_HRV % win_len
        print("building cached dataset for window length: %s ....." % win_len)
        x_train, y_train = get_data(df_train, win_len, self.dl_feature_list)
        x_test, y_test = get_data(df_test, win_len, self.dl_feature_list)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.20, random_state=42, shuffle=False)
        with h5py.File(cache_path, 'w') as data:
            data["x_train"] = x_train
            data["y_train"] = y_train
            data["x_val"] = x_val
            data["y_val"] = y_val
            data["x_test"] = x_test
            data["y_test"] = y_test
            data.close()

    def load_ml_data(self):
        df_train, df_test, _ = self.load_df_dataset()
        self.x_train = df_train[self.ml_feature_list]
        self.y_train = df_train["stages"]
        self.x_test = df_test[self.ml_feature_list]
        self.y_test = df_test["stages"]


