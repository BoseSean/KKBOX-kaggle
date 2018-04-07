import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import xgboost as xgb
import gc
from sklearn import *

print('Loading data ...')

data_root = '~/churn-prediction/kkbox-churn-prediction-challenge/'
train = pd.read_csv( data_root+'train.csv')
train = train.merge(pd.read_csv( data_root+'train_v2.csv'))

members  = pd.read_csv(data_root+'members_v3.csv')
members.drop_duplicates(subset=['msno'], keep='first', inplace=True)

num_mean = pd.read_csv(data_root+'num_mean.csv')
num_mean = num_mean.append(pd.read_csv(data_root+'num_mean_2.csv'))
num_mean.drop_duplicates(subset=['msno'], keep='first', inplace=True)

transaction = pd.read_csv(data_root+'transac_processed.csv')
transaction.drop_duplicates(subset=['msno'], keep='first', inplace=True)
transaction.drop('idx',1)

test = pd.read_csv(data_root+'sample_submission_v2.csv')
test.drop(['is_churn'],axis=1)


print('Merging data ...')

df_train = train.merge(members, how='left', on='msno')
df_train = df_train.merge(num_mean, how='left', on='msno')
df_train = df_train.merge(transaction, how='left', on='msno')

df_test = test.merge(members, how='left', on='msno')
df_test = df_test.merge(num_mean, how='left', on='msno')
df_test = df_test.merge(transaction, how='left', on='msno')