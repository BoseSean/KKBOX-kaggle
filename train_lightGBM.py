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
transaction_given = pd.read_csv(data_root+'transac_given_processed.csv')


print('Merging data ...')

df_train = train.merge(members, how='left', on='msno', copy=False)
df_train = df_train.merge(num_mean, how='left', on='msno', copy=False)
df_train = df_train.merge(transaction, how='left', on='msno', copy=False)
df_train = df_train.merge(transaction_given, how='left', on='msno', copy=False)

del transaction, transaction_given, members, num_mean
gc.collect()

# Drop duplicates first
#df_test = df_test.drop_duplicates('msno')

print('Converting data type ...')


df_train['city'].fillna(method='ffill', inplace=True)
df_train['bd'].fillna(method='ffill', inplace=True)
df_train['gender'].fillna(method='ffill', inplace=True)
df_train['registered_via'].fillna(method='ffill', inplace=True)
df_train["registration_init_time"].fillna(method='ffill', inplace=True)


gender = {'male':1, 'female':2}
df_train['gender'] = df_train['gender'].map(gender)
df_train["is_churn"] = df_train["is_churn"].astype(np.int8,copy=False)
df_train["city"] = df_train["city"].astype(np.int8,copy=False)
df_train["registered_via"] = df_train["registered_via"].astype(np.int8,copy=False)
df_train["registration_init_time"] = df_train["registration_init_time"].astype(np.int8,copy=False)


df_train['total_list_price'] = df_train['total_list_price'].astype(np.int16,copy=False)
df_train['transaction_span'] = df_train['transaction_span'].astype(np.int16,copy=False)
df_train['is_auto_renew'] = df_train['is_auto_renew'].astype(np.int8,copy=False)
df_train['is_cancel_sum'] = df_train['is_cancel_sum'].astype(np.int8,copy=False)
df_train['trans_count'] = df_train['trans_count'].astype(np.int16,copy=False)
df_train['total_amount_paid'] = df_train['total_amount_paid'].astype(np.int16,copy=False)
df_train['difference_in_price_paid'] = df_train['difference_in_price_paid'].astype(np.int16,copy=False)
df_train['amount_paid_perday'] = df_train['amount_paid_perday'].astype(np.float32,copy=False)

df_train['payment_method_id'] = df_train['payment_method_id'].astype(np.int16,copy=False)
df_train['payment_plan_days'] = df_train['payment_plan_days'].astype(np.int16,copy=False)
df_train['plan_list_price'] = df_train['plan_list_price'].astype(np.int16,copy=False)
df_train['actual_amount_paid'] = df_train['actual_amount_paid'].astype(np.int16,copy=False)
df_train['is_cancel'] = df_train['is_cancel'].astype(np.int16,copy=False)
df_train['discount'] = df_train['discount'].astype(np.int16,copy=False)
df_train['is_discount'] = df_train['is_discount'].astype(np.int16,copy=False)

print(df_train.dtypes)
# df_train.fillna(-1)

features = [c for c in df_train.columns if c not in ['is_churn','msno','sum(num_25)','sum(num_50)','sum(num_75)','sum(num_985)','sum(num_100)','sum(num_unq)','sum(total_secs)','idx','idx_g']]
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

# print('Training ...')
# n_round=500
# lgb_params = {
#     'objective': 'binary',
#     'metric': 'binary_logloss',
#     'boosting': 'gbdt',
#     'learning_rate': 0.002,  # small learn rate, large number of iterations
#     'verbose': 0,
#     'num_leaves': 108,
#     'bagging_fraction': 0.95,
#     'bagging_freq': 1,
#     'bagging_seed': 1,
#     'feature_fraction': 0.9,
#     'feature_fraction_seed': 1,
#     'max_bin': 128,
#     'max_depth': 7,
#     'reg_alpha': 1,
#     'reg_lambda': 0,
#     'min_split_gain': 0.5,
#     'min_child_weight': 1,
#     'min_child_samples': 10,
#     'scale_pos_weight': 1,
#     'verbosity': -1
# }



# model1 = lgb.train(lgb_params, train_set=d_train, num_boost_round=2500,
#     valid_sets=watchlist, early_stopping_rounds=50, verbose_eval=100)
# model1.save_model('lgb_model/lgb_model1.txt')

lgb_params = {
        'learning_rate': 0.05,
        'application': 'binary',
        'max_depth': 7,
        'num_leaves': 256,
        'verbosity': -1,
        'metric': 'binary_logloss'
    }

model = lgb.train(lgb_params, train_set=d_train, num_boost_round=500,
    valid_sets=watchlist, early_stopping_rounds=66, verbose_eval=100)

print('Saving ...')

model.save_model('lgb_model/lgb_model2.txt')
# print(df_train)
