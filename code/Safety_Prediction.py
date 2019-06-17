# -*- coding: utf-8 -*-
"""
Name: Safety_Prediction.py
Description:
    This script is used to load, transform and make the prediction of the given booking ID.

@author: n.satsawat@gmail.com
"""
# Core python library
import os, sys
import pandas as pd
import numpy as np
import scipy.stats
# Some pandas result display, no impact to what are being stored
pd.set_option('display.max_columns', 500)
pd.set_option('display.float_format', lambda x: '%.4f' % x)
# ignore the warning message
import warnings
warnings.filterwarnings('ignore')
# visualize related
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import xgboost
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score, f1_score
import pickle


def list_all_files_with_ext(path: str, 
                            suffix: str='.csv'):
    """
    Function to list all the files with specific extension in the given path
    @Args:
      path (str): the path to files
      suffix (str): the extension of the files to get; default: .csv
      
    Return: 
      List of files with given extensions in the path
    """
    filenames = os.listdir(path)
    return [path + filename for filename in filenames if filename.endswith(suffix)]


def read_files_(file_to_read: list, 
                show_sample_ind: bool=False):
    """
    Function to read multiple files onto one pandas dataframe
    @Args:
      file_to_read (list): List of path and filename to read
      show_sample_ind (bool): Show the head sample; default is False
      
    Return:
      full pandas dataframe
    """
    _df = pd.concat([pd.read_csv(f) for f in file_to_read], ignore_index=True)
    if show_sample_ind:
        print('\n')
        print('Printing top 5 observations')
        print(_df.head(5))
        
    return _df


def percentile(n: float):
    """
    Function to compute the percentile based on given n, this will use in `agg()` method in pandas
    @Args:
      n (float): the percentile to compute (i.e. 0.5 for median)
      
    Return:
      the percentile value
    """
    def percentile_(x):
        return x.quantile(n)
    
    percentile_.__name__ = 'percentile_{:2.0f}'.format(n*100)
    return percentile_


def mean_abs_dev():
    """
    Function to compute the mean absolute deviation, which is the average distance between
      each data point and the mean. This helps to understand about the variability in the data.
    This should be 
    @Args:
      None
      
    Return:
      MAD value
    """
    def _mean_abs_dev(x):
        return x.mad()
    
    _mean_abs_dev.__name__ = 'mean_abs_dev'
    return _mean_abs_dev


def agg_sensor_per_booking(df: pd.DataFrame, key_col: str='bookingID'):
    """
    Function to create new features based on the accelerometer and gyroscope sensor data
    @Args:
      df: pandas dataframe contain the acceleration and gyro
      key_col: name of the booking ID
    
    Return:
      aggregrated with new features pandas
    """
    _tmp_df = df.groupby(key_col).agg({'acceleration_x': [np.min, np.max, np.mean, 
                                                     np.median, np.var, np.std,
                                                     scipy.stats.skew, scipy.stats.kurtosis,
                                                     mean_abs_dev(), percentile(.25), percentile(.75)
                                                    ],
                                       'acceleration_y': [np.min, np.max, np.mean, 
                                                         np.median, np.var, np.std, 
                                                         scipy.stats.skew, scipy.stats.kurtosis,
                                                         mean_abs_dev(), percentile(.25), percentile(.75)
                                                         ],
                                       'acceleration_z': [np.min, np.max, np.mean, 
                                                          np.median, np.var, np.std, 
                                                          scipy.stats.skew, scipy.stats.kurtosis,
                                                          mean_abs_dev(), percentile(.25), percentile(.75)
                                                          ],
                                       'gyro_x': [np.min, np.max, np.mean, 
                                                  np.median, np.var, np.std,
                                                  scipy.stats.skew, scipy.stats.kurtosis,
                                                  mean_abs_dev(), percentile(.25), percentile(.75)
                                                  ],
                                       'gyro_y': [np.min, np.max, np.mean, 
                                                  np.median, np.var, np.std, 
                                                  scipy.stats.skew, scipy.stats.kurtosis,
                                                  mean_abs_dev(), percentile(.25), percentile(.75)
                                                  ],
                                       'gyro_z': [np.min, np.max, np.mean, 
                                                  np.median, np.var, np.std, 
                                                  scipy.stats.skew, scipy.stats.kurtosis,
                                                  mean_abs_dev(), percentile(.25), percentile(.75)
                                                  ],
                                       'Speed': [np.min, np.max, np.mean, 
                                                 np.median, np.var, np.std, 
                                                 scipy.stats.skew, scipy.stats.kurtosis,
                                                 mean_abs_dev(), percentile(.25), percentile(.75)
                                                 ]
                                       })
    cols = {'acceleration_x_amin': 'min_acc_x', 'acceleration_x_amax': 'max_acc_x',
            'acceleration_x_mean': 'mean_acc_x', 'acceleration_x_median': 'p50_acc_x',
            'acceleration_x_var': 'var_acc_x', 'acceleration_x_std': 'std_acc_x',
            'acceleration_x_skew': 'skew_acc_x', 'acceleration_x_kurtosis': 'kurt_acc_x',
            'acceleration_x_mean_abs_dev': 'mad_acc_x', 'acceleration_x_percentile_25': 'p25_acc_x',
            'acceleration_x_percentile_75': 'p75_acc_x',

            'acceleration_y_amin': 'min_acc_y', 'acceleration_y_amax': 'max_acc_y',
            'acceleration_y_mean': 'mean_acc_y', 'acceleration_y_median': 'p50_acc_y',
            'acceleration_y_var': 'var_acc_y', 'acceleration_y_std': 'std_acc_y',
            'acceleration_y_skew': 'skew_acc_y', 'acceleration_y_kurtosis': 'kurt_acc_y',
            'acceleration_y_mean_abs_dev': 'mad_acc_y', 'acceleration_y_percentile_25': 'p25_acc_y',
            'acceleration_y_percentile_75': 'p75_acc_y',

            'acceleration_z_amin': 'min_acc_z', 'acceleration_z_amax': 'max_acc_z',
            'acceleration_z_mean': 'mean_acc_z', 'acceleration_z_median': 'p50_acc_z',
            'acceleration_z_var': 'var_acc_z', 'acceleration_z_std': 'std_acc_z',
            'acceleration_z_skew': 'skew_acc_z', 'acceleration_z_kurtosis': 'kurt_acc_z',
            'acceleration_z_mean_abs_dev': 'mad_acc_z', 'acceleration_z_percentile_25': 'p25_acc_z',
            'acceleration_z_percentile_75': 'p75_acc_z',

            'gyro_x_amin': 'min_gyro_x', 'gyro_x_amax': 'max_gyro_x',
            'gyro_x_mean': 'mean_gyro_x', 'gyro_x_median': 'p50_gyro_x',
            'gyro_x_var': 'var_gyro_x', 'gyro_x_std': 'std_gyro_x',
            'gyro_x_skew': 'skew_gyro_x', 'gyro_x_kurtosis': 'kurt_gyro_x',
            'gyro_x_mean_abs_dev': 'mad_gyro_x', 'gyro_x_percentile_25': 'p25_gyro_x',
            'gyro_x_percentile_75': 'p75_gyro_x',

            'gyro_y_amin': 'min_gyro_y', 'gyro_y_amax': 'max_gyro_y',
            'gyro_y_mean': 'mean_gyro_y', 'gyro_y_median': 'p50_gyro_y',
            'gyro_y_var': 'var_gyro_y', 'gyro_y_std': 'std_gyro_y',
            'gyro_y_skew': 'skew_gyro_y', 'gyro_y_kurtosis': 'kurt_gyro_y',
            'gyro_y_mean_abs_dev': 'mad_gyro_y', 'gyro_y_percentile_25': 'p25_gyro_y',
            'gyro_y_percentile_75': 'p75_gyro_y',

            'gyro_z_amin': 'min_gyro_z', 'gyro_z_amax': 'max_gyro_z',
            'gyro_z_mean': 'mean_gyro_z', 'gyro_z_median': 'p50_gyro_z',
            'gyro_z_var': 'var_gyro_z', 'gyro_z_std': 'std_gyro_z',
            'gyro_z_skew': 'skew_gyro_z', 'gyro_z_kurtosis': 'kurt_gyro_z',
            'gyro_z_mean_abs_dev': 'mad_gyro_z', 'gyro_z_percentile_25': 'p25_gyro_z',
            'gyro_z_percentile_75': 'p75_gyro_z',

            'Speed_amin': 'min_Speed', 'Speed_amax': 'max_Speed',
            'Speed_mean': 'mean_Speed', 'Speed_median': 'p50_Speed',
            'Speed_var': 'var_Speed', 'Speed_std': 'std_Speed',
            'Speed_skew': 'skew_Speed', 'Speed_kurtosis': 'kurt_Speed',
            'Speed_mean_abs_dev': 'mad_Speed', 'Speed_percentile_25': 'p25_Speed',
            'Speed_percentile_75': 'p75_Speed'
           }
    _tmp_df.columns = _tmp_df.columns.map('_'.join).to_series().map(cols)
    _tmp_df.reset_index(drop=False, inplace=True)
    return _tmp_df


def agg_other_per_booking(df: pd.DataFrame, key_col: str='bookingID'):
    """
    Function to create new features based on other supporting data
    @Args:
      df: pandas dataframe contain the data
      key_col: name of the booking ID
    
    Return:
      aggregrated with new features pandas
    """
    _out_df = df.groupby(key_col).agg({'Accuracy': [np.min, np.median, percentile(.75), np.max],
                                       'second': ['count', 'max']})
    cols = {'Accuracy_amin': 'min_accuracy', 'Accuracy_amax': 'max_accuracy',
            'Accuracy_median': 'median_accuracy', 'Accuracy_percentile_75': 'p75_accuracy',
            'second_count': 'rec_cnt', 'second_max': 'max_sec'
           }
    _out_df.columns = _out_df.columns.map('_'.join).to_series().map(cols)
    _out_df.reset_index(inplace=True)
    _out_df['max_sec'] = _out_df['max_sec'] + 1  # add one as second starts from 0.
    return _out_df


def combine_data(sensor_df, other_df, 
                 key_id: str='bookingID', show_sample_ind: bool=False):
    """
    Function to combine the sensor features and other features onto single data frame
    @Args:
      sensor_df: the pandas data frame containing sensor data
      other_df: the pandas data frame containing other supporting data
      key_id: the key for merge pandas data frame; default is 'bookingID'
      show_sample_ind: boolean to print the head observations; default is False
      
    Return:
      pandas data frame with all features
    """
    _all_df = pd.merge(sensor_df, other_df, how='left', on=key_id)
    if show_sample_ind:
        print(_all_df.head(5))
    
    return _all_df


def load_safety_model(model_filename: str='../model/grab_safety_solution.model'):
    """
    Function to load the XGBClassifier model
    @Args:
      model_filename: the path and model filename
      
    Return:
      XGBClassifier object
    """
    try:
        _xgb = pickle.load(open(model_filename, 'rb'))
        if type(_xgb) == xgboost.XGBClassifier:
            print('Successfully load the model with details:')
            print(_xgb)
        
        else:
            print('Error: This is not XGBClassifier object!!')
            
    except FileNotFoundError:
        print('Error: %s occured!!, please recheck the path and file name.' % sys.exc_info()[0])
        
    return _xgb


def print_evaluation_metrics(y_true, y_pred):
    """
    Function to compute and print the evaluation metric for the prediction
    @Args:
      y_true: the true labels of y
      y_pred: the prediction of y
      
    Return:
      tuble of AUC, accuracy score and F1 score
    """
    print('The AUC score: %.4f, accuracy score: %.4f, f1 score: %.4f' % \
          (roc_auc_score(y_true, y_pred), accuracy_score(y_true, y_pred), f1_score(y_true, y_pred)))
    return roc_auc_score(y_true, y_pred), accuracy_score(y_true, y_pred), f1_score(y_true, y_pred)


if __name__ == '__main__':
    print('Program starts...')
    SEED = 1234
    PATH = os.path.dirname(__file__) if "__file__" in locals() else os.getcwd()
    os.chdir(PATH)
    np.random.seed(SEED)
    DATA_DIR = '../data/safety/features/'
    LABEL_DIR = '../data/safety/labels/'
    OUT_DIR = '../output/'
    LABEL_IND = True  # If there is label file for evaluation, set True; otherwise False.
    
    xgb_model = load_safety_model()
    print('Reading data from directory: %s' % DATA_DIR)
    file_to_read = list_all_files_with_ext(DATA_DIR)
    data_df = read_files_(file_to_read)
    data_df.sort_values(by=['bookingID', 'second'], ascending=True, 
                        inplace=True)
    data_df.reset_index(inplace=True, drop=True)
    print('Completed reading data with total of %s' % str(data_df.shape))
    
    print('Beginning feature transformation...')
    sensor_df = agg_sensor_per_booking(data_df, 'bookingID')
    other_df = agg_other_per_booking(data_df, 'bookingID')
    feature_df = combine_data(sensor_df, other_df, key_id='bookingID', 
                              show_sample_ind=True)
    feature_df.sort_values(by='bookingID', ascending=True, inplace=True)
    print('Done feature transformation!!!')
    
    print('Running prediction to the given dataset...')
    booking_id = feature_df['bookingID']
    X = feature_df.drop('bookingID', axis=1, inplace=False)
    y_pred = xgb_model.predict(X)
    y_prob = xgb_model.predict_proba(X)[:, 1]
    
    prediction_ = pd.DataFrame({'bookingID': booking_id,
                                'prediction': y_pred,
                                'probability': y_prob})
    print('Writing prediction to file...')
    prediction_.to_csv(OUT_DIR + 'all_prediction.csv', index=False)
    print('Saving prediction completed...')
    
    if LABEL_IND:
        print('\nEvaluation mode is set to True')
        print('Reading Target label...')
        file_to_read = list_all_files_with_ext(LABEL_DIR)
        target_df = read_files_(file_to_read)
        target_df.sort_values(by='bookingID', ascending=True, inplace=True)
        target_df.loc[target_df.duplicated(['bookingID'], keep=False), 'label'] = 1
        target_df.drop_duplicates(subset=['bookingID', 'label'], keep='first', inplace=True)        
        target_df.reset_index(drop=True, inplace=True)
        print('The target data shape: %s' % str(target_df.shape))
        print('Running the evaluation...')
        _ = print_evaluation_metrics(target_df.loc[:, 'label'], 
                                     prediction_.loc[:, 'prediction'])
        
    print('Program completed...')
