import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


# Any results you write to the current directory are saved as output.
import lightgbm as lgb
import gc
from sklearn import *


print('Loading data ...')

#data_root = '/opt/shared-data/kkbox-churn-prediction-challenge/'
data_root = '~/churn-prediction/kkbox-churn-prediction-challenge/'
train = pd.read_csv( data_root+'train.csv')
members  = pd.read_csv(data_root+'members_v3.csv')
num_mean = pd.read_csv(data_root+'num_mean.csv')
transaction = pd.read_csv(data_root+'transac_processed.csv')
transaction.drop('idx',1)
# test = pd.read_csv(data_root+'sample_submission_zero.csv')


# for c, dtype in zip(info.columns, info.dtypes):
#     if dtype == np.float64:
#         info[c] = info[c].astype(np.float32)
print('Merging data ...')
df_train = train.merge(members, how='left', on='msno')
df_train = df_train.merge(num_mean, how='left', on='msno')
df_train = df_train.merge(transaction, how='left', on='msno')

print('Converting data type ...')
df_train["is_churn"] = df_train["is_churn"].astype('category')
df_train["city"] = df_train["city"].astype('category')
df_train["gender"] = df_train["gender"].astype('category')
df_train["registered_via"] = df_train["registered_via"].astype('category')
df_train["registration_init_time"] = df_train["registration_init_time"].astype('category')

df_train['city'].fillna(method='ffill', inplace=True)
df_train['bd'].fillna(method='ffill', inplace=True)
df_train['gender'].fillna(method='ffill', inplace=True)
df_train['registered_via'].fillna(method='ffill', inplace=True)
df_train["registration_init_time"].fillna(method='ffill', inplace=True)

df_train['total_list_price'] = df_train['total_list_price'].astype(np.int16)
df_train['transaction_span'] = df_train['transaction_span'].astype(np.int16)
df_train['is_auto_renew'] = df_train['is_auto_renew'].astype('category')
df_train['is_cancel_sum'] = df_train['is_cancel_sum'].astype('category')
df_train['trans_count'] = df_train['trans_count'].astype(np.int16)
df_train['total_amount_paid'] = df_train['total_amount_paid'].astype(np.int16)
df_train['difference_in_price_paid'] = df_train['difference_in_price_paid'].astype(np.int16)
df_train['amount_paid_perday'] = df_train['amount_paid_perday'].astype(np.float32)


df_train['avg(num_25)']  = df_train['avg(num_25)'].astype(np.float32)
df_train['avg(num_50)']  = df_train['avg(num_50)'].astype(np.float32)
df_train['avg(num_75)']  = df_train['avg(num_75)'].astype(np.float32)
df_train['avg(num_985)']  = df_train['avg(num_985)'].astype(np.float32)
df_train['avg(num_100)']  = df_train['avg(num_100)'].astype(np.float32)
df_train['avg(num_unq)']  = df_train['avg(num_unq)'].astype(np.float32)
df_train['avg(total_secs)']  = df_train['avg(total_secs)'].astype(np.float32)
df_train['sum(num_25)']  = df_train['sum(num_25)'].astype(np.float32)
df_train['sum(num_50)']  = df_train['sum(num_50)'].astype(np.float32)
df_train['sum(num_75)']  = df_train['sum(num_75)'].astype(np.float32)
df_train['sum(num_985)']  = df_train['sum(num_985)'].astype(np.float32)
df_train['sum(num_100)']  = df_train['sum(num_100)'].astype(np.float32)
df_train['sum(num_unq)']  = df_train['sum(num_unq)'].astype(np.float32)
df_train['sum(total_secs)']  = df_train['sum(total_secs)'].astype(np.float32)

print(df_train.dtypes)
# df_train.fillna(-1)

features = [c for c in df_train.columns if c not in ['is_churn','msno']]
print('Using features')
print(features)



print('Split data ...')
x1, x2, y1, y2 = model_selection.train_test_split(df_train[features], 
    df_train['is_churn'], test_size=0.2, random_state=0)

# lgb
d_train = lgb.Dataset(x1, label=y1)
d_valid = lgb.Dataset(x2, label=y2)
watchlist = [d_train, d_valid]

# lgb_params = {
#     'learning_rate': 0.05,
#     'application': 'binary',
#     'max_depth': 5,
#     'num_leaves': 512,
#     'verbosity': -1,
#     'metric': 'binary_logloss'
# }

print('Training ...')
n_round=500
lgb_params = {
        'learning_rate': 0.05,
        'application': 'binary',
        'max_depth': 7,
        'num_leaves': 256,
        'verbosity': -1,
        'metric': 'binary_logloss'
    }

model = lgb.train(lgb_params, train_set=d_train, num_boost_round=240, 
    valid_sets=watchlist, early_stopping_rounds=50, verbose_eval=100) 
    
print('Saving ...')
model.save_model('model.txt')

# print(df_train)


