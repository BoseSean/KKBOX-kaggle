import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

print('Loading data ...')
data_root = '/opt/shared-data/kkbox-churn-prediction-challenge/'
train = pd.read_csv( data_root+'train.csv')
members  = pd.read_csv(data_root+'user_logs.csv')

print('Finished loading data ...')