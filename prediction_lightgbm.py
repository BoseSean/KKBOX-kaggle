import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import lightgbm as lgb
import gc
from sklearn import *


print('Loading data ...')

data_root = '~/churn-prediction/kkbox-churn-prediction-challenge/'
train = pd.read_csv( data_root+'train.csv')
members  = pd.read_csv(data_root+'members_v3.csv')
members.drop_duplicates(subset=['msno'], keep='first', inplace=True)

num_mean = pd.read_csv(data_root+'num_mean.csv')
num_mean = num_mean.append(pd.read_csv(data_root+'num_mean_2.csv'))
num_mean.drop_duplicates(subset=['msno'], keep='first', inplace=True)

transaction = pd.read_csv(data_root+'transac_processed.csv')
transaction.drop('idx',1)
transaction.drop_duplicates(subset=['msno'], keep='first', inplace=True)

transaction_given = pd.read_csv(data_root+'transac_given_processed.csv')
transaction_given.drop_duplicates(subset=['msno'], keep='first', inplace=True)

test = pd.read_csv(data_root+'sample_submission_v2.csv')
test.drop(['is_churn'],axis=1)
print(test.shape)

print('Merging data ...')

df_test = test.merge(members, how='left', on='msno', copy=False)
df_test = df_test.merge(num_mean, how='left', on='msno', copy=False)
df_test = df_test.merge(transaction, how='left', on='msno', copy=False)
df_test = df_test.merge(transaction_given, how='left', on='msno', copy=False)

print(df_test)
del transaction, transaction_given, members, num_mean
gc.collect()

# Drop duplicates first
#df_test = df_test.drop_duplicates('msno')

print('Converting data type ...')


df_test['city'].fillna(method='ffill', inplace=True)
df_test['bd'].fillna(method='ffill', inplace=True)
df_test['gender'].fillna(method='ffill', inplace=True)
df_test['registered_via'].fillna(method='ffill', inplace=True)
df_test["registration_init_time"].fillna(method='ffill', inplace=True)


gender = {'male':1, 'female':2}
df_test['gender'] = df_test['gender'].map(gender)
df_test["is_churn"] = df_test["is_churn"].astype(np.int8,copy=False)
df_test["city"] = df_test["city"].astype(np.int8,copy=False)
df_test["registered_via"] = df_test["registered_via"].astype(np.int8,copy=False)
df_test["registration_init_time"] = df_test["registration_init_time"].astype(np.int8,copy=False)


df_test['total_list_price'] = df_test['total_list_price'].astype(np.int16,copy=False)
df_test['transaction_span'] = df_test['transaction_span'].astype(np.int16,copy=False)
df_test['is_auto_renew'] = df_test['is_auto_renew'].astype(np.int8,copy=False)
df_test['is_cancel_sum'] = df_test['is_cancel_sum'].astype(np.int8,copy=False)
df_test['trans_count'] = df_test['trans_count'].astype(np.int16,copy=False)
df_test['total_amount_paid'] = df_test['total_amount_paid'].astype(np.int16,copy=False)
df_test['difference_in_price_paid'] = df_test['difference_in_price_paid'].astype(np.int16,copy=False)
df_test['amount_paid_perday'] = df_test['amount_paid_perday'].astype(np.float32,copy=False)

df_test['payment_method_id'] = df_test['payment_method_id'].astype(np.int16,copy=False)
df_test['payment_plan_days'] = df_test['payment_plan_days'].astype(np.int16,copy=False)
df_test['plan_list_price'] = df_test['plan_list_price'].astype(np.int16,copy=False)
df_test['actual_amount_paid'] = df_test['actual_amount_paid'].astype(np.int16,copy=False)
df_test['is_cancel'] = df_test['is_cancel'].astype(np.int16,copy=False)
df_test['discount'] = df_test['discount'].astype(np.int16,copy=False)
df_test['is_discount'] = df_test['is_discount'].astype(np.int16,copy=False)

print(df_test.dtypes)
# df_test.fillna(-1)
features = [c for c in df_test.columns if c not in ['is_churn','msno','sum(num_25)','sum(num_50)','sum(num_75)','sum(num_985)','sum(num_100)','sum(num_unq)','sum(total_secs)','idx','idx_g']]


model = lgb.Booster(model_file="lgb_model/lgb_model1.txt")
prediction = model.predict(df_test[features])
test['is_churn'] = prediction.clip(0.0000001, 0.999999)
test[['msno', 'is_churn']].to_csv("ligthgbm_prediction_1.csv", index=False)

combined = prediction

model = lgb.Booster(model_file="lgb_model/lgb_model2.txt")
prediction = model.predict(df_test[features])
test['is_churn'] = prediction.clip(0.0000001, 0.999999)
test[['msno', 'is_churn']].to_csv("ligthgbm_prediction_2.csv", index=False)

combined+=prediction

combined/=2
test['is_churn'] = combined.clip(0.0000001, 0.999999)
test[['msno', 'is_churn']].to_csv("ligthgbm_prediction_c.csv", index=False)
