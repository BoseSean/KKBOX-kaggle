import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import lightgbm as lgb
import gc
from sklearn import *


print('Loading data ...')

#data_root = '/opt/shared-data/kkbox-churn-prediction-challenge/'
data_root = '~/churn-prediction/kkbox-churn-prediction-challenge/'
members  = pd.read_csv(data_root+'members_v3.csv')
members.drop_duplicates(subset=['msno'], keep='first', inplace=True)

num_mean = pd.read_csv(data_root+'num_mean.csv')
num_mean = num_mean.append(pd.read_csv(data_root+'num_mean_2.csv'))
num_mean.drop_duplicates(subset=['msno'], keep='first', inplace=True)

transaction = pd.read_csv(data_root+'transac_processed.csv')
transaction.drop('idx',1)
transaction_given = pd.read_csv(data_root+'transac_given_processed.csv')
test = pd.read_csv(data_root+'sample_submission_v2.csv')
test.drop(['is_churn'],axis=1)
print(test.shape)
# test = pd.read_csv(data_root+'sample_submission_zero.csv')


# for c, dtype in zip(info.columns, info.dtypes):
#     if dtype == np.float64:
#         info[c] = info[c].astype(np.float32)
print('Merging data ...')
df_test = test.merge(members, how='left', on='msno')
df_test = df_test.merge(num_mean, how='left', on='msno')
df_test = df_test.merge(transaction, how='left', on='msno')
df_test = df_test.merge(transaction_given, how='left', on='msno')

print('Converting data type ...')
df_test["is_churn"] = df_test["is_churn"].astype('category')
df_test["city"] = df_test["city"].astype('category')
df_test["gender"] = df_test["gender"].astype('category')
df_test["registered_via"] = df_test["registered_via"].astype('category')
df_test["registration_init_time"] = df_test["registration_init_time"].astype('category')

df_test['city'].fillna(method='ffill', inplace=True)
df_test['bd'].fillna(method='ffill', inplace=True)
df_test['gender'].fillna(method='ffill', inplace=True)
df_test['registered_via'].fillna(method='ffill', inplace=True)
df_test["registration_init_time"].fillna(method='ffill', inplace=True)

df_test['total_list_price'] = df_test['total_list_price'].astype(np.int16)
df_test['transaction_span'] = df_test['transaction_span'].astype(np.int16)
df_test['is_auto_renew'] = df_test['is_auto_renew'].astype('category')
df_test['is_cancel_sum'] = df_test['is_cancel_sum'].astype('category')
df_test['trans_count'] = df_test['trans_count'].astype(np.int16)
df_test['total_amount_paid'] = df_test['total_amount_paid'].astype(np.int16)
df_test['difference_in_price_paid'] = df_test['difference_in_price_paid'].astype(np.int16)
df_test['amount_paid_perday'] = df_test['amount_paid_perday'].astype(np.float32)


df_test['avg(num_25)']  = df_test['avg(num_25)'].astype(np.float32)
df_test['avg(num_50)']  = df_test['avg(num_50)'].astype(np.float32)
df_test['avg(num_75)']  = df_test['avg(num_75)'].astype(np.float32)
df_test['avg(num_985)']  = df_test['avg(num_985)'].astype(np.float32)
df_test['avg(num_100)']  = df_test['avg(num_100)'].astype(np.float32)
df_test['avg(num_unq)']  = df_test['avg(num_unq)'].astype(np.float32)
df_test['avg(total_secs)']  = df_test['avg(total_secs)'].astype(np.float32)
df_test['sum(num_25)']  = df_test['sum(num_25)'].astype(np.float32)
df_test['sum(num_50)']  = df_test['sum(num_50)'].astype(np.float32)
df_test['sum(num_75)']  = df_test['sum(num_75)'].astype(np.float32)
df_test['sum(num_985)']  = df_test['sum(num_985)'].astype(np.float32)
df_test['sum(num_100)']  = df_test['sum(num_100)'].astype(np.float32)
df_test['sum(num_unq)']  = df_test['sum(num_unq)'].astype(np.float32)
df_test['sum(total_secs)']  = df_test['sum(total_secs)'].astype(np.float32)
df_test['payment_method_id'] = df_test['payment_method_id'].astype(np.int16)
df_test['payment_plan_days'] = df_test['payment_plan_days'].astype(np.int16)
df_test['plan_list_price'] = df_test['plan_list_price'].astype(np.int16)
df_test['actual_amount_paid'] = df_test['actual_amount_paid'].astype(np.int16)
df_test['is_cancel'] = df_test['is_cancel'].astype(np.int16)

features = [c for c in df_test.columns if c not in ['is_churn','msno']]



model = lgb.Booster(model_file="model1.txt")

prediction = model.predict(df_test[features])

# vfunc = np.vectorize(lambda x: 1 if x>0.5 else 0)
# test['is_churn'] = prediction.map()
test[['msno', 'is_churn']].to_csv("ligthgbm_prediction.csv", index=False)



