from catboost import CatBoostClassifier
import numpy as np
import pandas as pd
import catboost
import gc
from sklearn import *

def load_data()
    print('Loading data ...')

    #data_root = '/opt/shared-data/kkbox-churn-prediction-challenge/'
    data_root = '~/churn-prediction/kkbox-churn-prediction-challenge/'
    train = pd.read_csv( data_root+'train.csv')
    members  = pd.read_csv(data_root+'members_v3.csv')
    num_mean = pd.read_csv(data_root+'num_mean.csv')
    transaction = pd.read_csv(data_root+'transac_processed.csv')
    transaction.drop('idx',1)
    transaction_given = pd.read_csv(data_root+'transac_given_processed.csv')


    print('Merging data ...')

    df_train = train.merge(members, how='left', on='msno', copy=False)
    df_train = df_train.merge(num_mean, how='left', on='msno', copy=False)
    df_train = df_train.merge(transaction, how='left', on='msno', copy=False)
    df_train = df_train.merge(transaction_given, how='left', on='msno', copy=False)

    del transaction, transaction_given, members, num_mean
    gc.collect()
