import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


# Any results you write to the current directory are saved as output.
import xgboost as xgb
import gc
from sklearn import *
from collections import OrderedDict

print("loading...")

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


df_train['avg(num_25)'].fillna(0,inplace = True)
df_train['avg(num_50)'].fillna(0,inplace = True)
df_train['avg(num_75)'].fillna(0,inplace = True)
df_train['avg(num_985)'].fillna(0,inplace = True)
df_train['avg(num_100)'].fillna(0,inplace = True)
df_train['avg(num_unq)'].fillna(0,inplace = True)
df_train['avg(total_secs)'].fillna(0,inplace = True)


print(df_test.dtypes)
# df_test.fillna(-1)
features = [c for c in df_train.columns if c not in ['is_churn','msno','sum(num_25)','sum(num_50)','sum(num_75)','sum(num_985)','sum(num_100)','sum(num_unq)','sum(total_secs)','idx','idx_g']]

print("predicting...")
prediction = 0
fold = 4
for i in range(1,fold):
    model = xgb.Booster()
    model.load_model('xgb_model/xgb_model_'+str(i)+'.model')
    if i==0:
        prediction = model.predict(xgb.DMatrix(df_test[features]))
    else:
        prediction += model.predict(xgb.DMatrix(df_test[features]))
    print(prediction.shape)

prediction /= fold

test['is_churn'] = prediction.clip(0.0000001, 0.999999)
test[['msno', 'is_churn']].to_csv("xgboost_prediction_3.csv", index=False)
# # prediction_df = pd.DataFrame(OrderedDict([ ("msno", test["msno"]),("is_churn", prediction) ]))
# # prediction_df.to_csv("xgboost_prediction.csv",index=False)

