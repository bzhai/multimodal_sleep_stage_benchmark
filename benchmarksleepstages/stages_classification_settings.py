# OS_VERSION = platform.system()
# this is used to generate summary for each modality in a human readable file. only use it in summary_of_experiment
FEATURE_TYPE_DICT = {'hr': 'HR', 'hrv': 'HRV', 'acc': 'ENMO', 'all': 'ENMO_HRV'}
ANALYSIS_SUB_FOLDER = {'sleep_period': "sp_summary", "recording_period": "summary"}

HR_PATH = "F:\\OneDrive - Newcastle University\\Dataset\\MESA\\annotations-rpoints\\"
ACC_PATH = "F:\\OneDrive - Newcastle University\\Dataset\\MESA\\actigraphy\\"
OUTPUT_PATH = "C:/tmp/sleep/opensource/HRV30s_ACC30s_H5"
OVERLAP_PATH = "F:\\OneDrive - Newcastle University\\Dataset\\MESA\\mesa-actigraphy-psg-overlap%d.csv"
##################
CSV30_DATA_PATH = "C:/tmp/sleep/opensource/HRV30s_ACC_CSV"
CSV270_DATA_PATH = "F:/mesa/Input/HRV270s_ACC_CSV"
HRV30_ACC_STD_PATH = "C:\\tmp\\sleep\\opensource\\HRV30s_ACC30s_H5\\hrv30s_acc30s_full_feat_stand.h5"
# HRV270_ACC_STD_PATH = "F:\\mesa\\Input\\HRV270s_ACC_CSV\\hrv270s_acc30s_full_feat_stand.h5"
# HRV60_ACC60_STD_PATH = "F:\\mesa\\Input\\HRV60s_ACC60s_H5\\hrv60s_acc60s_full_feat_stand.h5"

CSV60_DATA_PATH = "F:/mesa/Input/HRV60s_ACC60s_CSV"
SCALER = {30: "F:\\mesa\\Input\\HRV30s_ACC_CSV\\HRV30s_ACC30s_full_feat_stand.h5.pkl",
          270: "F:\\mesa\\Input\\HRV270s_ACC_CSV\\HRV270s_ACC30s_full_feat_stand.h5.pkl"}
FEATURE_LIST = "F:\\mesa\\Input\\feature_list.csv"
OUTPUT_FOLDER = "F:\\mesa\\Results\\"
# The following setting is for deep learning
# NN_ACC_HRV = {
#     "NN_ACC_HRV_100": "C:\\tmp\\sleep\\HRV30s_ACC30s_H5\\nn_acc_hrv30s_100.h5",
#     "NN_ACC_HRV_20": "C:\\tmp\\sleep\\HRV30s_ACC30s_H5\\nn_acc_hrv30s_20.h5",
#     "NN_ACC_HRV_50": "C:\\tmp\\sleep\\HRV30s_ACC30s_H5\\nn_acc_hrv30s_50.h5"
# }
NN_ACC_HRV = "C:\\tmp\\sleep\\HRV30s_ACC30s_H5\\nn_acc_hrv30s_%d.h5"
NN_ACC60_HR60_100 = "F:\\mesa\\Input\\HRV60s_ACC60s_H5\\nn_acc60_hr60s_100.h5"
NN_ACC60_HR60_20 = "F:\\mesa\\Input\\HRV60s_ACC60s_H5\\nn_acc60_hr60s_20.h5"
NN_ACC60_HR60_50 = "F:\\mesa\\Input\\HRV60s_ACC60s_H5\\nn_acc60_hr60s_50.h5"
# only have train and test no validation dataset
NN_TRAIN_TEST_ACC_HRV30_100 = "C:/tmp/3stages_freq_trans_post_processing/nn_acc_hrv30s_100.h5"
NN_TRAIN_TEST_ACC_HRV30_20 = "C:/mesa/Input/HRV270s_ACC_CSV/nn_acc_hrv30s_20.h5"
NN_TRAIN_TEST_ACC_HRV30_50 = "C:/mesa/Input/HRV270s_ACC_CSV/nn_acc_hrv30s_50.h5"
# this is the place to save three stages outputs
STAGE_OUTPUT_FOLDER_HRV270s = {2: "F:\\mesa\\Results\\HRV270s_ACC\\2stages",
                               3: "F:\\mesa\\Results\\HRV270s_ACC\\3stages",
                               4: "F:\\mesa\\Results\\HRV270s_ACC\\4stages",
                               5: "F:\\mesa\\Results\\HRV270s_ACC\\5stages"}
STAGE_OUTPUT_FOLDER_HRV30s = {2: "C:\\tmp\\sleep\\opensource\\Results\\HRV30s_ACC\\2stages",
                               3: "C:\\tmp\\sleep\\opensource\\Results\\HRV30s_ACC\\3stages",
                               4: "C:\\tmp\\sleep\\opensource\\Results\\HRV30s_ACC\\4stages",
                               5: "C:\\tmp\\sleep\\opensource\\Results\\HRV30s_ACC\\5stages"}
STAGE_OUTPUT_FOLDER_HRV60s = {2: "F:\\mesa\\Results\\HRV60s_ACC60s\\2stages",
                              3: "F:\\mesa\\Results\\HRV60s_ACC60s\\3stages",
                              4: "F:\\mesa\\Results\\HRV60s_ACC60s\\4stages",
                              5: "F:\\mesa\\Results\\HRV60s_ACC60s\\5stages",
                              14: "F:\\mesa\\Results\\HRV60s_ACC60s\\14stages"}
STAGE_ML_OUTPUT_FOLDER_HRV30s = "D:\\mesa_ml_trained"
TRAIN_TEST_SPLIT = "F:/mesa/Input/train_test_pid_split.csv"
ALG_READABLE_FILE = "F:\\mesa\\Input\\alg_readable.csv"
CNN_FOLDER = "G:\\Ubicomp_HP_models\\CNN"
LSTM_FOLDER = "G:\\Ubicomp_HP_models\\LSTM"
