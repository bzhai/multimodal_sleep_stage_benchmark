import h5py
import itertools
from utilities.utils import *
import time
import pickle
from benchmarksleepstages.stages_classification_settings import *

# def load_mesa_psg(csv_file, ground_truth):
#     df = pd.read_csv(csv_file)
#     df["actValue"] = df["activity"]
#     df['time'] = pd.to_datetime(df['linetime'])
#     if ground_truth == "stage":
#         df["gt"] = df["stage"] > 0
#     elif ground_truth == "interval":
#         df["gt"] = (df["interval"] != "ACTIVE").astype(int)
#     df = df[df["interval"] != "EXCLUDE"]
#     df['active'] = (df['interval'] == "ACTIVE").astype(int)
#     return df


# def build_dl_train_test(h5_file, seq_len, feature_list, output_path):
#     dftrain, dftest, featnames = load_h5_df_dataset(h5_file, useCache=True)
#     Xtrain, Ytrain = get_data(dftrain, seq_len, feature_list)
#     Xtest, Ytest = get_data(dftest, seq_len, feature_list)
#     data = h5py.File(os.path.join(output_path, 'nn_acc_hrv30s_%d.h5' % seq_len), 'w')
#     data["x_train"] = Xtrain
#     data["y_train"] = Ytrain
#     data["x_test"] = Xtest
#     data["y_test"] = Ytest
#     data.close()

def build_mesa_hrv_acc_csv(data_path, feat_list_path, saveCache, cacheName, output_path, pre_split):
    """
    This function is designed to build a H5 file to speed up IO for experiment
    :param data_path:
    :param feat_list_path:
    :param saveCache:
    :param cacheName:
    :return:
    """
    tmp = []
    allfiles = os.listdir(data_path)
    csv_raw_files = []
    for idx, f in enumerate(allfiles):
        if ".csv" in f:
            csv_raw_files.append(os.path.join(data_path, f))
    csv_raw_files.sort()
    feature_list = pd.read_csv(feat_list_path)['feature_list'].values.tolist()
    # for filename in glob(os.path.join(path, "*"))[:]: # this is the Joao's code
    for filename in csv_raw_files:
        print(filename)
        dftmp = pd.read_csv(filename)
        # the sleep block is actually the start of sleep onset and end of sleep onset
        # creates a gt_block
        gtTrue = dftmp[dftmp["stages"] > 0]
        if gtTrue.empty:
            print("Ignoring file %s" % filename)
            continue
        start_block = dftmp.index.get_loc(gtTrue.index[0])
        end_block = dftmp.index.get_loc(gtTrue.index[-1])  # the
        dftmp["gt_sleep_block"] = make_one_block(dftmp["stages"], start_block, end_block)
        tmp.append(dftmp)
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
    if len(pre_split) > 0:  # load pre-calculated train test PID list
        uids_train, uids_test = load_pre_splited_train_test_ids(pre_split)
    train_idx = whole_df[whole_df["mesaid"].apply(lambda x:x in uids_train)].index
    dftrain = whole_df.iloc[train_idx].copy()
    test_idx = whole_df[whole_df["mesaid"].apply(lambda x:x in uids_test)].index
    dftest = whole_df.iloc[test_idx].copy()
    print("start scaling on df train....")
    # standardises the whole training df is too time consuming, so 2019-09-27 simple method
    scaler = standardize_df_given_feature(dftrain, feature_list, df_name="dftrain", simple_method=True)
    standardize_df_given_feature(dftest, feature_list, scaler, df_name="dftest", simple_method=True)


    if saveCache:
        store = pd.HDFStore(os.path.join(output_path, cacheName), 'w')
        store["train"] = dftrain
        store["test"] = dftest
        store["featnames"] = pd.Series(feature_list)
        store.close()
        print('h5 dataset is saved to %s' % os.path.join(output_path, cacheName))
        with open(os.path.join(output_path, cacheName + '.pkl'), "wb") as f:
            pickle.dump(scaler, f)
    return dftrain, dftest, feature_list



# def load_h5np_dataset(path, file_type='.h5'):
#     if ".h5" in path :
#         store = h5py.File(path, 'r')
#         Xtrain = store["Xtrain"]
#         Ytrain = store["Ytrain"]
#         Xtest = store["Xtest"]
#         Ytest = store["Ytest"]
#         return Xtrain, Ytrain, Xtest, Ytest
#     return None, None, None, None


def load_h5_dataset(path):
    start = time.time()
    store = pd.HDFStore(path, 'r')
    df_train = store["train"]
    df_test = store["test"]
    feature_name = store["featnames"].values.tolist()
    if type(feature_name[0]) is list:
        feature_name = list(itertools.chain.from_iterable(feature_name))
    store.close()
    print("loading dataset spend : %s" % time.strftime("%H:%M:%S", time.gmtime(time.time() - start)))
    return df_train, df_test, feature_name

def load_h5_df_dataset(path):
    """
    This function needs to be removed, it's a part of data loader
    """
    feature_name = []
    start = time.time()
    store = pd.HDFStore(path,'r')
    dftrain = store["train"]
    dftest = store["test"]

    feature_name = store["featnames"].values.tolist()
    if type(feature_name[0]) is list:
        feature_name = list(itertools.chain.from_iterable(feature_name))
    store.close()
    print("loading dataset spend : %s" % time.strftime("%H:%M:%S", time.gmtime(time.time() -start)))
    return dftrain, dftest, feature_name
    # if file_type == "h5":
    #     store = pd.HDFStore(path, 'r')
    #     act_feature = store["featnames"]
    #     hrv_feature_names = store["hvr_featnames"]
    #     dftrain = store["train"]
    #     feature_name.append(act_feature)
    #     feature_name.append(hrv_feature_names)
    #     dftest = store["test"]
    #     #feature_names = list(store["hrvfeatnames"].values)
    # else:
    #     # #######################################################################
    #     # ########  split train and test based on the  ##########################
    #     # #######################################################################
    #     # split the train test by the user ID
    #     test_proportion = 0.2
    #     whole_df= pd.DataFrame()
    #     uids = whole_df.mesaid.unique()
    #     np.random.seed(9)
    #     np.random.shuffle(uids)
    #     test_position = int(uids.shape[0] * test_proportion)
    #     uids_test, uids_train = uids[:test_position], uids[test_position:]
    #     train_idx = whole_df[whole_df["mesaid"].apply(lambda x: x in uids_train)].index
    #     dftrain = whole_df.iloc[train_idx].copy()
    #     test_idx = whole_df[whole_df["mesaid"].apply(lambda x: x in uids_test)].index
    #     dftest = whole_df.iloc[test_idx].copy()
    # return dftrain, dftest, feature_name


def save_to_h5(dftrain, dftest, featnames, output_path, cacheName):
    store = pd.HDFStore(os.path.join(output_path, cacheName), 'w')
    store["train"] = dftrain
    store["test"] = dftest
    store["featnames"] = featnames
    store.close()
    print('h5 dataset is saved to %s' % os.path.join(output_path, cacheName))


def subgroup_REM_epochs_transition(h5_file, group_file, output_path, cacheName):
    dftrain, dftest, featnames = load_h5_df_dataset(h5_file, useCache=True)
    dfall = pd.concat([dftrain, dftest], ignore_index=True)
    dfall = dfall.reset_index(drop=True)
    del dftrain
    del dftest
    group_df = pd.read_csv(group_file)
    group_pids = group_df['mesaid'].values.tolist()
    dfall = dfall[dfall['mesaid'].isin(group_pids)]
    dfall = dfall.reset_index(drop=True)
    num_pids = len(group_pids)
    np.random.seed(42)
    uids = np.random.choice(range(num_pids), num_pids, replace=False)
    test_proportion = 0.2
    test_position = int(uids.shape[0] * test_proportion)
    test_idx, train_idx = uids[:test_position], uids[test_position:]
    uids_test = [group_pids[x] for x in test_idx]
    uids_train = [group_pids[x] for x in train_idx]
    assert set(uids_test + uids_train) == set(group_pids)
    pd.DataFrame(uids_train).rename({"0":"pid"}).to_csv(os.path.join(os.path.abspath(output_path), 'train_pid.csv'))
    pd.DataFrame(uids_test).rename({"0": "pid"}).to_csv(
        os.path.join(os.path.abspath(output_path), 'test_pid.csv'))
    train_idx = dfall[dfall["mesaid"].apply(lambda x: x in uids_train)].index
    dftrain = dfall.iloc[train_idx].copy(deep=True)
    test_idx = dfall[dfall["mesaid"].apply(lambda x: x in uids_test)].index
    dftest = dfall.iloc[test_idx].copy(deep=True)
    save_to_h5(dftrain, dftest, pd.Series(featnames), output_path=output_path, cacheName=cacheName)


if __name__ == "__main__":
    #group1_file = "C:/tmp/3stages_freq_trans_post_processing/split_freq_0_0.1_rem_0.05_0.25/group1_freq_0_0.1_rem_0.05_0.25.csv"
    # group1_cache_name = "group1_0_0.1_rem_0.05_0.25.h5"
    # group1_output_file_path = "C:/tmp/3stages_freq_trans_post_processing/split_freq_0_0.1_rem_0.05_0.25"
    #
    # group2_file = "C:/tmp/3stages_freq_trans_post_processing/split_freq_0.1_0.25_rem_0_0.15/group2_freq_0.1_0.25_rem_0_0.15.csv"
    # group2_cache_name = "group2_freq_0.1_0.25_rem_0_0.15.h5"
    # group2_output_file_path = "C:/tmp/3stages_freq_trans_post_processing/split_freq_0.1_0.25_rem_0_0.15"
    # hrv_win_len = 30
    # h5_file = globals()["HRV%d_ACC_STD_PATH" % hrv_win_len]
    # print("Loading test dataset from %s" % h5_file)
    # subgroup_REM_epochs_transition(h5_file, group1_file, group1_output_file_path, group1_cache_name)
    # subgroup_REM_epochs_transition(h5_file, group2_file, group2_output_file_path, group2_cache_name)
    # load_h5_df_dataset(os.path.join(os.path.abspath(output_file_path), cache_name), useCache=True)

    build_mesa_hrv_acc_csv(data_path=CSV30_DATA_PATH, feat_list_path=FEATURE_LIST, saveCache=True,
                           cacheName="hrv30s_acc30s_full_feat_stand.h5", output_path=OUTPUT_PATH
                           , pre_split=TRAIN_TEST_SPLIT)
    # with Pool(5) as p:
    #     print(p.map(f, [(), 2, 3]))
    # print("dataset processing is completed!")

    # p = Process(target=f, args=('bob',))
    # p.start()
    # p.join()
    # ############################# post processing method 1 ###############################################
    # seq_len_list = [20, 50, 100]
    # feature_list = ["activity", "mean_nni", "sdnn", "sdsd", "vlf", "lf", "hf", "lf_hf_ratio", "total_power"]
    # linux_path = '/home/campus.ncl.ac.uk/b3057108/dataset/mesa/Results'
    # windows_path = 'C:/tmp/3stages_freq_trans_post_processing'
    # for seq_len in seq_len_list:
    #     p = Process(target=build_dl_train_test, args=(HRV30_ACC_STD_PATH, seq_len, feature_list, linux_path))
    #     p.start()

