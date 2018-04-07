import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


# Any results you write to the current directory are saved as output.
import xgboost as xgb
import gc
from sklearn import *

print('Loading data ...')

#data_root = '/opt/shared-data/kkbox-churn-prediction-challenge/'
data_root = '~/churn-prediction/kkbox-churn-prediction-challenge/'
train = pd.read_csv( data_root+'train.csv')
train = train.merge(pd.read_csv( data_root+'train_v2.csv'))
members  = pd.read_csv(data_root+'members_v3.csv')
num_mean = pd.read_csv(data_root+'num_mean.csv')
num_mean = num_mean.append(pd.read_csv(data_root+'num_mean_2.csv'))

transaction = pd.read_csv(data_root+'transac_processed.csv')
transaction.drop('idx',1)

test = pd.read_csv(data_root+'sample_submission_v2.csv')
test.drop(['is_churn'],axis=1)
# test = pd.read_csv(data_root+'sample_submission_zero.csv')


# for c, dtype in zip(info.columns, info.dtypes):
#     if dtype == np.float64:
#         info[c] = info[c].astype(np.float32)
print('Merging data ...')
df_train = train.merge(members, how='left', on='msno')
df_train = df_train.merge(num_mean, how='left', on='msno')
df_train = df_train.merge(transaction, how='left', on='msno')

df_test = test.merge(members, how='left', on='msno')
df_test = df_test.merge(num_mean, how='left', on='msno')
df_test = df_test.merge(transaction, how='left', on='msno')
print('Converting data type ...')
# df_train["is_churn"] = df_train["is_churn"].astype('category')
# df_train["city"] = df_train["city"].astype('category')
# df_train["gender"] = df_train["gender"].astype('category')
# df_train["registered_via"] = df_train["registered_via"].astype('category')
# df_train["registration_init_time"] = df_train["registration_init_time"].astype('category')


# df_train['city'].fillna(method='ffill', inplace=True)
# df_train['bd'].fillna(method='ffill', inplace=True)
# df_train['gender'].fillna(method='ffill', inplace=True)
# df_train['registered_via'].fillna(method='ffill', inplace=True)
# df_train["registration_init_time"].fillna(method='ffill', inplace=True)
gender = {'male':1, 'female':2}
df_train['gender'] = members['gender'].map(gender)
df_train['total_list_price'] = df_train['total_list_price'].astype(np.int16)
df_train['transaction_span'] = df_train['transaction_span'].astype(np.int16)
df_train['is_auto_renew'] = df_train['is_auto_renew'].astype(np.int16)
df_train['is_cancel_sum'] = df_train['is_cancel_sum'].astype(np.int16)
df_train['trans_count'] = df_train['trans_count'].astype(np.int16)
df_train['total_amount_paid'] = df_train['total_amount_paid'].astype(np.int16)
df_train['difference_in_price_paid'] = df_train['difference_in_price_paid'].astype(np.int16)
df_train['amount_paid_perday'] = df_train['amount_paid_perday'].astype(np.float32)

df_train.corr()
print(df_train.dtypes)

df_test['gender'] = members['gender'].map(gender)
df_test['total_list_price'] = df_test['total_list_price'].astype(np.int16)
df_test['transaction_span'] = df_test['transaction_span'].astype(np.int16)
df_test['is_auto_renew'] = df_test['is_auto_renew'].astype(np.int16)
df_test['is_cancel_sum'] = df_test['is_cancel_sum'].astype(np.int16)
df_test['trans_count'] = df_test['trans_count'].astype(np.int16)
df_test['total_amount_paid'] = df_test['total_amount_paid'].astype(np.int16)
df_test['difference_in_price_paid'] = df_test['difference_in_price_paid'].astype(np.int16)
df_test['amount_paid_perday'] = df_test['amount_paid_perday'].astype(np.float32)

print(df_test.dtypes)
# df_train.fillna(-1)

features = [c for c in df_train.columns if c not in ['is_churn','msno']]
print('Using features')
print(features)



print('Split data ...')
x1, x2, y1, y2 = model_selection.train_test_split(df_train[features], 
    df_train['is_churn'], test_size=0.2, random_state=0)

print('Training ...')

params = {
    'eta': 0.07,
    'max_depth': 7,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'seed': 3228,
    'silent': True,
    'tree_method': 'exact'
    }
watchlist = [(xgb.DMatrix(x1, y1), 'train'), (xgb.DMatrix(x2, y2), 'valid')]
model = xgb.train(params, xgb.DMatrix(x1, y1), 200,  watchlist, 
    maximize=False, verbose_eval=100, early_stopping_rounds=10)

print('Saving ...')
model.save_model('xgb_model.model')

print('Predicting ...')
prediction = model.predict(df_test[features])
prediction_df = pd.DataFrame(OrderedDict([ ("msno", test["msno"]),("is_churn", prediction) ]))
prediction_df.to_csv("xgboost_prediction.csv",index=False)
# xgb_pred = model.predict(xgb.DMatrix(test[cols]), ntree_limit=model.best_ntree_limit)
#xgb_pred = model.predict(xgb.DMatrix(test[cols]), ntree_limit=ntree_limit[i])
# xgb_valid = model.predict(xgb.DMatrix(x2))
# print('xgb valid log loss = {}'.format(log_loss(y2,xgb_valid)))
# print(df_train)