
class Config(object):
    def __init__(self):
        # setup the modality, notice in MESA activity measurement is "activity counts" not ENMO
        self.FEATURE_TYPE_DICT = {'hr': 'HR', 'hrv': 'HRV', 'acc': 'ENMO', 'all': 'ENMO_HRV'}
        # set up the child folder directory with respect to sleep period and sleep recording period
        self.ANALYSIS_SUB_FOLDER = {'sleep_period': "sp_summary", "recording_period": "summary"}
        # download HRV data from MESA website and store it to here
        self.HR_PATH = r"C:\tmp\sleep\opensource\annotations-rpoints"
        # download actigraphy data from MESA website and store it to here
        self.ACC_PATH = r"C:\tmp\sleep\opensource\actigraphy"
        #
        self.CSV30_DATA_PATH = r"C:\tmp\sleep\opensource\HRV30s_ACC_CSV\Aligned_final"
        self.OVERLAP_PATH = r"C:\tmp\sleep\opensource\mesa-actigraphy-psg-overlap.csv"

        # #################
        self.H5_OUTPUT_PATH = r"C:\tmp\sleep\opensource\HRV30s_ACC30s_H5"
        self.HRV30_ACC_STD_PATH = r"C:\tmp\sleep\opensource\HRV30s_ACC30s_H5\hrv30s_acc30s_full_feat_stand.h5"
        self.STANDARDISER = {30: r"C:\tmp\sleep\opensource\HRV30s_ACC30s_H5\HRV30s_ACC30s_full_feat_stand.h5_std_transformer",
                             }
        # Deep learning H5 cache file for windowed data.
        self.NN_ACC_HRV = r"C:\tmp\sleep\opensource\HRV30s_ACC30s_H5\nn_acc_hrv30s_%d.h5"
        self.FEATURE_LIST = "./assets/feature_list.csv"
        self.EXPERIMENT_RESULTS_ROOT_FOLDER = r"C:\tmp\sleep\opensource\Results"


        self.HP_CV_OUTPUT_FOLDER = r"C:\tmp\sleep\opensource\Results\HRV30s_ACC\HP_CV_TUNING"
        # The following setting is for deep learning
        # this is the place to save all experiments' outputs
        self.STAGE_OUTPUT_FOLDER_HRV30s = {2: r"C:\tmp\sleep\opensource\Results\HRV30s_ACC\2stages",
                                      3: r"C:\tmp\sleep\opensource\Results\HRV30s_ACC\3stages",
                                      4: r"C:\tmp\sleep\opensource\Results\HRV30s_ACC\4stages",
                                      5: r"C:\tmp\sleep\opensource\Results\HRV30s_ACC\5stages"}
        self.CNN_FOLDER = r"C:\tmp\sleep\opensource\Results\HRV30s_ACC\HP_CV_TUNING\CNN"
        self.LSTM_FOLDER = r"C:\tmp\sleep\opensource\Results\HRV30s_ACC\HP_CV_TUNING\LSTM"
        self.TRAIN_TEST_SPLIT = "./assets/train_test_pid_split.csv"
        # A readable table for all algorithms
        self.ALG_READABLE_FILE = "./assets/alg_readable.csv"
        self.SUMMARY_FOLDER = {'r': 'summary', 's': 'sp_summary'}