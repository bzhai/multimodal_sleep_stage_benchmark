Movement and Cardiac Sensing Based Sleep Stages Classification
=========================

Despite rapid growth in research that applies machine learning to sleep data, progress in the field appears far less dramatic than in other applications of machine learning.
Sleep data sets are inherently expensive and difficult to obtain and are usually relatively small compared to consumer industry data sets. Often, these data are not open to researchers outside the data owner’s institution.
Here we present four public benchmarks for machine learning researchers interested in sleep medicine research, built using data from the publicly available Multi-Ethnic Study of Atherosclerosis ([paper 1](https://www.ncbi.nlm.nih.gov/pubmed/27070134), [paper 2](https://www.ncbi.nlm.nih.gov/pubmed/29860441) [MESA Dataset]( https://sleepdata.org/datasets/mesa)). Our four clinical prediction tasks are binary sleep-wake classification and multistage classification that were described in [Bing et al., 2020]():

## Environment Requirement

We have two environments for our paper.

1. The first environment is used to build data sets and perform all deep learning experiments. TensorFlow version is 1.14 and python version is 3.7.3
Please build all data sets based on this environment
2. The second environment is only used for DL hyperparameter search of deep learning models. We use TensorFlow 2.0 version to complete the super parameter search。

The dataset can be obtained from the ([RR Interval data](https://sleepdata.org/datasets/mesa/files/polysomnography/annotations-rpoints])) and ([actigraphy data](https://sleepdata.org/datasets/mesa/files/actigraphy)) from ([MESA](https://sleepdata.org/datasets/mesa))
Note: The RR interval is the time between QRS complexes which is also known as Inter-beat interval (IBI).
Setup the experiment directory and update sleep_stage_config.py with these settings.
Here I present an example of the experiment directory structure and their values that should be mapped to Config:
```bash
└─experiment_root
    ├─actigraphy # It’s MESA Actigraphy data directory which maps to ACC_PATH
    ├─Aligned_final # this folder stores the aligned CSV files for raw activity counts and RR intervals.
    ├─annotations-rpoints  # It is MESA RRI data directory which maps to HR_PATH
    ├─HRV30s_ACC30s_H5 # the processed H5 file that contains the aligned HRV features and actigraphy features which maps to H5_OUTPUT_PATH
    └─Results # It is the root folder to store the experiment results. The folder path that saves the results of each task will be a member of python dictionary STAGE_OUTPUT_FOLDER_HRV30s. Its value should assign to EXPERIMENT_RESULTS_ROOT_FOLDER
        └─HRV30s_ACC
            ├─2stages # The experiment results of binary sleep-wake classification will be stored in this folder
            ├─3stages # The experiment results of 3 sleep stages (Wake, REM and NREM) classification task will be stored in this folder
            ├─4stages # The experiment results of 4 sleep stages (Wake, Light, Deep and REM Sleep) classification will be stored in this folder
            ├─5stages # The experiment results of 5 sleep stages (Wake, N1, N2, N3 and REM) classification will be stored in this folder
            └─HP_CV_TUNING 
                ├─CNN # The experiment results of hyperparameter tuning for convolutional neural networks will be stored in this folder
                └─LSTM # The experiment results of hyperparameter tuning for LSTMs will be stored in this folder

```


## Build Dataset

To simplify the reading of benchmark data, we wrote special classes. The following command takes Actigraphy and RRI data to build the dataset that used for the benchmarking experiments. It may take several hours to create the dataset.

Align RRI and Actigraphy data

    python -m dataset_builder_loader.align_actigraphy_rri

Build up the H5py file that contains training and testing dataset

    python -m dataset_builder_loader.build_h5_file
    
Build up the cache files for deep learning sliding window-based dataset per task per sequence length.
    
    python -m dataset_builder_loader.build_cached_sliding_window_dataset –seq_len 20 –modality  all –num_classes 3
 
## Hyper-parameter Tuning
* Hyper-parameter tuning for traditional ML models. 
    
    The following command will explore the best hyper-parameter settings for Random Forest models. An example code is shown below:
        
        python -m benchmarksleepstages.statistic_ml_hyper_randomforest --modality all --num_classes 3

    The following command will explore the best hyper-parameter settings for Random traditional ML models. An example code is shown below:
        
        python -m benchmarksleepstages.statistic_ml_hyperparam_sgd --modality all --num_classes 3
* Hyper-parameter tuning for deep learning models. 
    
    The following command will explore multi-layers CNNs.An example code is shown below:

        python -m benchmarksleepstages.dl_cnn_hp_tuning_multi_layers --epoch 10 batch_size 128 –modality all
    
    The following command will explore multi-LSTMs. an example code is shown below:
        
        python -m benchmarksleepstages.dl_lstm_hp_tuning_multi_layers --epoch 10 batch_size 128 –modality all

## Training and Testing
* ML models
    
    For traditional ML models, we should run experiments for each modality and task. Before running the experiment, make sure that the optimal hyperparameters are set. An example code is shown below:
    
        python -m benchmarksleepstages.statistic_ml_experiment --modality all --num_classes 3
* DL models
    
    For deep learning models, we should run experiments for each modality and task. An example code is shown below:
        
        python -m benchmarksleepstages.dl_experiment --modality all --num_classes 3 --nn_type LSTM --seq_len 20 --epochs 20 --batch_size 128
 
* Ensemble methods
    
    The ensemble method requires that all DL models are trained, and it will load each trained model and generate the prediction of test data set. The ensemble results shown in the paper are calculated based on the combined modalities (--modality all) of each task.
An example code is shown below for the sleep classification task of three-stage using combined modalities:
        
        python -m benchmarksleepstages.dl_probability_calculation --modality all --num_classes 3
 
## Evaluation
The script `summary_of_experiment.py` will calculate minutes level metrics such as the minute deviation and also classifier level metrics includes F1, accuracy, precision and recall for each classifier. Two types evaluation completed in our paper, they are epoch-wise and subject level evaluation.
         
* Epoch-wise evaluation

    The epoch-wised evaluation should be conducted for each task, modality and evaluation period (sleep recording period and sleep period). An example code is shown below for the 3-stage sleep classification task of three-stage that adopted combined modalities based on deep learning methods.
 
        python -m benchmarksleepstages.summary_of_epoch_wise --num_classes 3 --modality all
 
* Subject-level evaluation 

    An example code is shown below for the 3-stage sleep classification task of three-stage that adopted combined modalities based on deep learning methods.
        
        python -m benchmarksleepstages.summary_of_experiment --modality all --num_classes 3    --period r
              
* T-test

    The t-testis conducted based on subjects level. An example jupyter notebook is in `results_analysis.ipynb`

## SHAP Feature Importance and Confusion Matrix
An example code is shown below 
        
        python -m benchmarksleepstages.summary_shap_feature_importance --modality all --load_pretrain 0 --num_class 3


