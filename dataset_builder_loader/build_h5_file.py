from sleep_stage_config import Config
from utilities.utils import *


class H5DatasetBuilder(object):
    def __init__(self, csv_path, output_path_name, feat_list_path="", pre_split=""):
        self.csv_path = csv_path
        self.output_path_name = output_path_name
        self.pre_split = pre_split
        self.feat_list_path = feat_list_path

    def build_mesa_h5file(self):
        """
        This function is designed to build a H5 file to speed up IO for experiment
        :param data_path:
        :param feat_list_path:
        :param saveCache:
        :param cacheName:
        :return:
        """
        tmp = []
        all_files = os.listdir(self.csv_path)
        csv_raw_files = []
        for idx, f in enumerate(all_files):
            if ".csv" in f:
                csv_raw_files.append(os.path.join(self.csv_path, f))
        csv_raw_files.sort()
        feature_list = pd.read_csv(self.feat_list_path)['feature_list'].values.tolist()
        # for file_name in glob(os.path.join(path, "*"))[:]: # this is the Joao's code
        for file_name in csv_raw_files:
            print(file_name)
            df_tmp = pd.read_csv(file_name)
            # the sleep block is actually the start of sleep onset and end of sleep onset
            # creates a gt_block
            gt_true = df_tmp[df_tmp["stages"] > 0]
            if gt_true.empty:
                print("Ignoring file %s" % file_name)
                continue
            start_block = df_tmp.index.get_loc(gt_true.index[0])
            end_block = df_tmp.index.get_loc(gt_true.index[-1])  # the
            df_tmp["gt_sleep_block"] = make_one_block(df_tmp["stages"], start_block, end_block)
            tmp.append(df_tmp)
        whole_df = pd.concat(tmp)
        del tmp
        whole_df = whole_df.reset_index(drop=True)
        whole_df["binterval"] = whole_df["interval"].replace("ACTIVE", 0).replace("REST", 1).replace("REST-S", 1)
        test_proportion = 0.2
        uids = whole_df.mesaid.unique().astype(int)
        np.random.seed(9)
        np.random.shuffle(uids)
        test_idx = int(uids.shape[0] * test_proportion)
        uids_test, uids_train = uids[:test_idx], uids[test_idx:]
        if len(self.pre_split) > 0:  # load pre-calculated train test PID list
            uids_train, uids_test = load_pre_splited_train_test_ids(self.pre_split)
        train_idx = whole_df[whole_df["mesaid"].apply(lambda x: x in uids_train)].index
        dftrain = whole_df.iloc[train_idx].copy()
        test_idx = whole_df[whole_df["mesaid"].apply(lambda x: x in uids_test)].index
        dftest = whole_df.iloc[test_idx].copy()
        print("start standardisation on df_train....")
        # standardises the whole training df is too time consuming, so 2019-09-27 using simple method
        scaler = standardize_df_given_feature(dftrain, feature_list, df_name="dftrain", simple_method=True)
        print("start standardisation on df_test....")
        standardize_df_given_feature(dftest, feature_list, scaler, df_name="dftest", simple_method=True)

        store = pd.HDFStore(os.path.join(self.output_path_name), 'w')
        store["train"] = dftrain
        store["test"] = dftest
        store["featnames"] = pd.Series(feature_list)
        store.close()
        print('h5 dataset is saved to %s' % os.path.join(self.output_path_name))
        with open(os.path.join(self.output_path_name + '_std_transformer'), "wb") as f:
            pickle.dump(scaler, f)
        return dftrain, dftest, feature_list


if __name__ == '__main__':
    config = Config()
    builder = H5DatasetBuilder(csv_path=config.CSV30_DATA_PATH, output_path_name=config.HRV30_ACC_STD_PATH,
                               feat_list_path=config.FEATURE_LIST, pre_split=config.TRAIN_TEST_SPLIT)
    builder.build_mesa_h5file()

