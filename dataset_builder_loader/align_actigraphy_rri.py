#-*-coding:utf-8-*-
import pandas as pd
#import seaborn as sns
import platform
from sleep_stage_config import Config
from utilities.utils import *
from sklearn.preprocessing import Normalizer
#sns.set(style='whitegrid', rc={'axes.facecolor': '#EFF2F7'})
import hrvanalysis as hrvana
from datetime import datetime
from pathlib import Path


class MesaHrvFeatureBuilder(object):
    def __init__(self, cfg, hrv_win, standarize):
        self.cfg = cfg
        self.hr_path = cfg.HR_PATH
        self.acc_path = cfg.ACC_PATH
        self.overlap_path = cfg.OVERLAP_PATH
        self.output_path = cfg.CSV30_DATA_PATH
        if os.path.exists(self.output_path):
            Path(self.output_path).mkdir(parents=True, exist_ok=True)
        self.processed_records = os.path.join(self.output_path, os.pardir,
                                              'processed_'+datetime.now().strftime("%Y%m%d-%H%M%S")+'.csv')
        self.rr_act_overlap_df = []  # "rr_ecg_act_overlap.csv"
        self.hrv_win = hrv_win
        self.standarize = standarize

    def process_all_files(self, is_test=False):
        '''
        This function will go through every subject overlapped data and extract the intersect set between hr and acc.
        the dataset quality control will filter out the RRI dataset with lower bound= 300, upper bound with 1000
        the output will be in either test output path or the actual output path.
        :param is_test: true is for test dataset
        :param output15s: true is to ouput the 15s epoch of hr dataset
        :return:
        '''
        # load Acc, HR and overlap files
        if is_test:
            all_acc_files = []
            all_hr_files = []
        else:
            all_acc_files = os.listdir(self.acc_path)
            all_hr_files = os.listdir(self.hr_path)
        overlap_df = pd.read_csv(self.overlap_path)  # only do experiment if they have overlapped ECG and Actigraphy
        total_subjects_list = overlap_df['mesaid'].unique()
        valid_pids = pd.read_csv(self.cfg.TRAIN_TEST_SPLIT)['uids'].values.tolist()
        # here we set the valid subject IDs according to a snapshot of MESA data on 2019-05-01. In this
        # snapshot, we manually checked the aligned data making sure the pre-processing yield satisfied quality of data.
        # ##### The num of total valid subjects should be 1743
        total_subjects_list = list(set(total_subjects_list).intersection(set(valid_pids)))
        total_processed = []
        if not os.path.exists(self.processed_records):
            with open(self.processed_records, "w") as output:
                writer = csv.writer(output, lineterminator='\n')
                writer.writerows(total_processed)
        # tag = datetime.now().strftime("%Y%m%d-%H%M%S")
        for PID in total_subjects_list:
            mesa_id = "%04d" % PID
            # filter Acc and HR based on the overlap records
            print('*' * 100)
            print("Processing subject %s dataset" % mesa_id)
            acc_inlist_idx = [s for s in all_acc_files if mesa_id in s]
            hr_inlist_idx = [s for s in all_hr_files if mesa_id in s]
            feature_list = []
            if len(acc_inlist_idx) > 0 and len(hr_inlist_idx) > 0:
                # get the raw dataset file index
                acc_file_idx = all_acc_files.index(''.join(acc_inlist_idx))
                hr_file_idx = all_hr_files.index(''.join(hr_inlist_idx))
                # load Acc and HR into Pandas
                acc_df = pd.read_csv(os.path.join(self.acc_path, all_acc_files[acc_file_idx]))
                hr_df = pd.read_csv(os.path.join(self.hr_path, all_hr_files[hr_file_idx]))
                featnames = get_statistic_feature(acc_df, column_name="activity", windows_size=20)
                acc_start_idx = overlap_df[overlap_df['mesaid'] == PID]['line'].values[0].astype(int)
                acc_epochs = hr_df['epoch'].max()
                # cut the dataset frame from the overlapped start index to the HR end index
                acc_df = acc_df[acc_start_idx - 1: acc_start_idx + acc_epochs - 1]
                # recalculate the line to the correct index
                acc_df['line'] = acc_df['line']-acc_start_idx+1
                acc_df = acc_df.reset_index(drop=True)
                # calculate the intersect set between HR and acc and cut HR to align the sequence
                # ################ Data quality control for Acc ########################
                # use marker and activity as the indicator column if the shape is different to 2-dim then drop

                list_size_chk = np.array(acc_df[['marker', 'activity']].values.tolist())
                # check whether the activity is empty
                if len(list_size_chk.shape) < 2:
                    print(
                        "File {f_name} doesn't meet dimension requirement, it's size is {wrong_dim}".format(
                            f_name=all_acc_files[acc_file_idx], wrong_dim=list_size_chk.shape)
                    )
                    continue

                # Cut HRV dataset based on length of Actigraphy dataset
                if (int(hr_df['epoch'].tail(1)) > acc_df.shape[0]):
                    hr_df = hr_df[hr_df['epoch'] <= acc_df.shape[0]]
                # take out the noise dataset if two peak overlap or not wear
                hr_df = hr_df[hr_df['TPoint'] > 0]
                # Define RR intervals by taking the difference between each one of the measurements in seconds (*1k)
                hr_df['RR Intervals'] = hr_df['seconds'].diff() * 1000
                hr_df['RR Intervals'].fillna(hr_df['RR Intervals'].mean(), inplace=True)  # fill mean for first sample

                # old method for processing of RR intervals which is inappropriate
                # sampling_df = pd.concat([sampling_df, t1], axis =0 )
                # outlier_low = np.mean(hr_df['HR']) - 6 * np.std(hr_df['HR'])
                # outlier_high = np.mean(hr_df['HR']) + 6 * np.std(hr_df['HR'])
                # hr_df = hr_df[hr_df['HR'] >= outlier_low]
                # hr_df = hr_df[hr_df['HR'] <= outlier_high]

                # apply HRV-Analysis package
                # filter any hear rate over 60000/300 = 200, 60000/2000 = 30
                clean_rri = hr_df['RR Intervals'].values
                clean_rri = hrvana.remove_outliers(rr_intervals=clean_rri, low_rri=300, high_rri=2000)
                clean_rri = hrvana.interpolate_nan_values(rr_intervals=clean_rri, interpolation_method="linear")
                clean_rri = hrvana.remove_ectopic_beats(rr_intervals=clean_rri, method="malik")
                clean_rri = hrvana.interpolate_nan_values(rr_intervals=clean_rri)

                hr_df["RR Intervals"] = clean_rri
                # calculate the Heart Rate
                hr_df['HR'] = np.round((60000.0 / hr_df['RR Intervals']), 0)
                # filter ACC
                acc_df = acc_df[acc_df['interval'] != 'EXCLUDED']
                # filter RRI
                t1 = hr_df.epoch.value_counts().reset_index().rename({'index': 'epoch_idx', 'epoch': 'count'}, axis=1)
                invalid_idx = set(t1[t1['count'] < 3]['epoch_idx'].values)
                del t1
                hr_df = hr_df[~hr_df['epoch'].isin(list(invalid_idx))]
                # get intersect epochs
                hr_epoch_set = set(hr_df['epoch'].values)
                acc_epoch_set = set(acc_df['line'])  # get acc epochs
                # only keep intersect dataset
                diff_epoch_set_a = acc_epoch_set.difference(hr_epoch_set)
                diff_epoch_set_b = hr_epoch_set.difference(acc_epoch_set)
                acc_df = acc_df[~acc_df['line'].isin(diff_epoch_set_a)]
                hr_df = hr_df[~hr_df['epoch'].isin(diff_epoch_set_b)]
                # check see if their epochs are equal
                assert acc_df.shape[0] == len(hr_df['epoch'].unique())
                # filter out any epochs with rri less than 3
                hr_epoch_set = set(hr_df['epoch'].values)
                hr_epoch_set = hr_epoch_set.difference(invalid_idx)
                for _, hr_epoch_idx in enumerate(list(hr_epoch_set)):
                    # sliding window
                    gt_label = hr_df[hr_df['epoch'] == hr_epoch_idx]["stage"].values[0]
                    if self.hrv_win != 0:
                        offset = int(np.floor(self.hrv_win/2))
                        tmp_hr_df = hr_df[hr_df['epoch'].isin(np.arange(hr_epoch_idx-offset, hr_epoch_idx+offset))]
                    else:
                        tmp_hr_df = hr_df[hr_df['epoch'] == hr_epoch_idx]
                    try:  # check to see if the first time stamp is empty
                        start_sec = float(tmp_hr_df['seconds'].head(1) * 1.0)
                    except Exception as ee:
                        print("Exception %s, source dataset: %s" % (ee, tmp_hr_df['seconds'].head(1)))
                    # calculate each epochs' HRV features
                    rr_epoch = tmp_hr_df['RR Intervals'].values
                    all_hr_features = {}
                    try:
                        all_hr_features.update(hrvana.get_time_domain_features(rr_epoch))
                    except Exception as ee:
                        self.log_process(ee, PID, hr_epoch_idx)
                        print("processed time domain features: {}".format(str(ee)))
                    try:
                        all_hr_features.update(hrvana.get_frequency_domain_features(rr_epoch))
                    except Exception as ee:
                        self.log_process(ee, PID, hr_epoch_idx)
                        print("processed frequency domain features: {}".format(str(ee)))
                    try:
                        all_hr_features.update(hrvana.get_poincare_plot_features(rr_epoch))
                    except Exception as ee:
                        self.log_process(ee, PID, hr_epoch_idx)
                        print("processed poincare features: {}".format(str(ee)))
                    try:
                        all_hr_features.update(hrvana.get_csi_cvi_features(rr_epoch))
                    except Exception as ee:
                        self.log_process(ee, PID, hr_epoch_idx)
                        print("processed csi cvi domain features: {}".format(str(ee)))
                    try:
                        all_hr_features.update(hrvana.get_geometrical_features(rr_epoch))
                    except Exception as ee:
                        self.log_process(ee, PID, hr_epoch_idx)
                        print("processed geometrical features: {}".format(str(ee)))

                    all_hr_features.update({'stages': gt_label
                                            , 'mesaid': acc_df[acc_df['line']==hr_epoch_idx]['mesaid'].values[0]
                                            , 'linetime': acc_df[acc_df['line']==hr_epoch_idx]['linetime'].values[0]
                                            , 'line': acc_df[acc_df['line'] == hr_epoch_idx]['line'].values[0]
                                            , 'wake': acc_df[acc_df['line'] == hr_epoch_idx]['wake'].values[0]
                                            , 'interval': acc_df[acc_df['line'] == hr_epoch_idx]['interval'].values[0]
                                            , 'activity': acc_df[acc_df['line'] == hr_epoch_idx]['activity'].values[0]
                                            })
                    feature_list.append(all_hr_features)

            #  If feature list is not empty
            if len(feature_list) > 0:
                hrv_acc_df = pd.DataFrame(feature_list)
                hrv_acc_df = hrv_acc_df.reset_index(drop=True)
                del hrv_acc_df['tinn']  # tinn is empty
                featnames = featnames + ["line"]
                combined_pd = pd.merge(acc_df[featnames], hrv_acc_df, on='line', how='inner')
                #combined_pd = combined_pd.reset_index(drop=True)
                combined_pd['timestamp'] = pd.to_datetime(combined_pd['linetime'])
                combined_pd['base_time'] = pd.to_datetime('00:00:00')
                combined_pd['seconds'] = (combined_pd['timestamp'] - combined_pd['base_time'])
                combined_pd['seconds'] = combined_pd['seconds'].dt.seconds
                combined_pd.drop(['timestamp', 'base_time'], axis=1, inplace=True)
                combined_pd['two_stages'] = combined_pd["stages"].apply(lambda x: 1.0 if x >= 1.0 else 0.0)
                combined_pd.loc[combined_pd['stages'] > 4, 'stages'] = 4  # make sure rem sleep label is 4
                combined_pd = combined_pd.fillna(combined_pd.median())
                combined_pd = combined_pd[combined_pd['interval'] != 'EXCLUDED']
                aligned_data = os.path.join(self.output_path, "Aligned_final")
                if not os.path.exists(aligned_data):
                    os.mkdir(aligned_data)

                # standardise and normalise the df
                feature_list = combined_pd.columns.to_list()
                std_feature = [x for x in feature_list if x not in ['two_stages', 'seconds', 'activity', 'interval',
                                                                    'wake', 'linetime', 'mesaid', 'stages', 'line']]
                if self.standarize:
                    standardize_df_given_feature(combined_pd, std_feature, df_name='combined_df', simple_method=False)
                combined_pd.to_csv(os.path.join(aligned_data, (mesa_id + '_combined.csv')), index=False)
                print("ID: {}, successed process".format(mesa_id))
                with open(self.processed_records, "w") as text_file:
                    text_file.write("ID: {}, successed process \n".format(mesa_id))
                total_processed.append("ID: {}, successed process".format(mesa_id))
            else:
                print("Acc is empty or HRV is empty!")
                total_processed.append("ID: {}, failed process".format(mesa_id))

    def log_process(self, error, mesaid, epoch_idx):
        with open(os.path.join(self.output_path, os.pardir, "process_log" + ".txt"), "a") as file:
            file.write("mesaid, %04d, epoch idx, %d, processed with issues, %s" % (mesaid, epoch_idx, error)
                       + "\n")



if __name__ == "__main__":
    config = Config()
    builder = MesaHrvFeatureBuilder(config, hrv_win=0, standarize=False)
    builder.process_all_files(is_test=False)

