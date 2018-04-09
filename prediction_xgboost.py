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
members  = pd.read_csv(data_root+'members_v3.csv')
# members.drop_duplicates(subset=['msno'], keep='first', inplace=True)

num_mean = pd.read_csv(data_root+'num_mean.csv')
num_mean = num_mean.append(pd.read_csv(data_root+'num_mean_2.csv'))
# num_mean.drop_duplicates(subset=['msno'], keep='first', inplace=True)

transaction = pd.read_csv(data_root+'transac_processed.csv')
# transaction.drop_duplicates(subset=['msno'], keep='first', inplace=True)
transaction.drop('idx',1)

test = pd.read_csv(data_root+'sample_submission_v2.csv')
test.drop(['is_churn'],axis=1)
print(test.shape)

print("merging...")
df_test = test.merge(members, how='left', on='msno')
print(df_test.shape)
df_test = df_test.merge(num_mean, how='left', on='msno')
print(df_test.shape)
df_test = df_test.merge(transaction, how='left', on='msno')
print(df_test.shape)
df_test.drop_duplicates(subset=['msno'], keep='first', inplace=True)
print(df_test.shape)

gender = {'male':1, 'female':2}

df_test['gender'] = members['gender'].map(gender)
df_test['total_list_price'] = df_test['total_list_price'].astype(np.int16)
df_test['transaction_span'] = df_test['transaction_span'].astype(np.int16)
df_test['is_auto_renew'] = df_test['is_auto_renew'].astype(np.int16)
df_test['is_cancel_sum'] = df_test['is_cancel_sum'].astype(np.int16)
df_test['trans_count'] = df_test['trans_count'].astype(np.int16)
df_test['total_amount_paid'] = df_test['total_amount_paid'].astype(np.int16)
df_test['difference_in_price_paid'] = df_test['difference_in_price_paid'].astype(np.int16)
df_test['amount_paid_perday'] = df_test['amount_paid_perday'].astype(np.float32)

print(df_test.shape)


features = [c for c in df_test.columns if c not in ['is_churn','msno']]

print("predicting...")
prediction = 0
fold = 5
for i in range(0,fold):
    model = xgb.Booster()
    model.load_model('xgb_model/xgb_model_'+str(i)+'.model')
    if i==0:
        prediction = model.predict(xgb.DMatrix(df_test[features]))
    else:
        prediction += model.predict(xgb.DMatrix(df_test[features]))
    print(prediction.shape)

prediction /= fold

test['is_churn'] = prediction
test[['msno', 'is_churn']].to_csv("xgboost_prediction1.csv", index=False)
# # prediction_df = pd.DataFrame(OrderedDict([ ("msno", test["msno"]),("is_churn", prediction) ]))
# # prediction_df.to_csv("xgboost_prediction.csv",index=False)
907472
1585762

