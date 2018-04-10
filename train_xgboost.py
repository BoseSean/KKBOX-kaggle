import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


# Any results you write to the current directory are saved as output.
import xgboost as xgb
import gc
from sklearn import *


gc.enable()

print('Loading data ...')

#data_root = '/opt/shared-data/kkbox-churn-prediction-challenge/'
data_root = '~/churn-prediction/kkbox-churn-prediction-challenge/'
train = pd.read_csv( data_root+'train.csv')
members  = pd.read_csv(data_root+'members_v3.csv')
num_mean = pd.read_csv(data_root+'num_mean.csv')
num_mean = num_mean.append(pd.read_csv(data_root+'num_mean_2.csv'))
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


df_train['avg(num_25)'].fillna(0,inplace = True)
df_train['avg(num_50)'].fillna(0,inplace = True)
df_train['avg(num_75)'].fillna(0,inplace = True)
df_train['avg(num_985)'].fillna(0,inplace = True)
df_train['avg(num_100)'].fillna(0,inplace = True)
df_train['avg(num_unq)'].fillna(0,inplace = True)
df_train['avg(total_secs)'].fillna(0,inplace = True)


print(df_train.dtypes)
# df_train.fillna(-1)

features = [c for c in df_train.columns if c not in ['is_churn','msno','sum(num_25)','sum(num_50)','sum(num_75)','sum(num_985)','sum(num_100)','sum(num_unq)','sum(total_secs)','idx','idx_g']]
print('Using features')
print(features)



# from imblearn.over_sampling import SMOTE 

# sm = SMOTE(random_state=42)
# X_res, y_res = sm.fit_sample(df_train[features], df_train['is_churn'])

fold = 5
for i in range(1,fold):
    print('Training ...')
    params = {
        'eta': 0.07,
        'max_depth': 7,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'seed': 3228+i,
        'silent': True,
        # 'tree_method': 'exact',
        'predictor’:’gpu_predictor'   
    }


    # params = {
    # 'base_score': 0.5,
    # 'eta': 0.002,  # use 0.002
    # 'max_depth': 6,
    # 'booster': 'gbtree',
    # 'colsample_bylevel': 1,
    # 'colsample_bytree': 1.0,
    # 'gamma': 1,
    # 'max_child_weight': 5,
    # 'n_estimators': 600,
    # 'reg_alpha': '0',
    # 'reg_lambda': '1',
    # 'scale_pos_weight': 1,
    # 'objective': 'binary:logistic',
    # 'eval_metric': 'logloss',
    # 'seed': 2017+i,
    # 'silent': True
    # }
    x1, x2, y1, y2 = model_selection.train_test_split(df_train[features], 
    df_train['is_churn'], test_size=0.1, random_state=i)
    watchlist = [(xgb.DMatrix(x1, y1), 'train'), (xgb.DMatrix(x2, y2), 'valid')]
    model = xgb.train(params, xgb.DMatrix(x1, y1), 200,  watchlist, 
        maximize=False, verbose_eval=100, early_stopping_rounds=50)

    print('Saving ...'+str(i))
    model.save_model('xgb_model/xgb_model_'+str(i)+'.model')
    del model, x1, x2, y1, y2, watchlist
    gc.collect()
# print('Predicting ...')
# prediction = model.predict(df_test[features])
# prediction_df = pd.DataFrame(OrderedDict([ ("msno", test["msno"]),("is_churn", prediction) ]))
# prediction_df.to_csv("xgboost_prediction.csv",index=False)

# xgb_pred = model.predict(xgb.DMatrix(test[cols]), ntree_limit=model.best_ntree_limit)
#xgb_pred = model.predict(xgb.DMatrix(test[cols]), ntree_limit=ntree_limit[i])
# xgb_valid = model.predict(xgb.DMatrix(x2))
# print('xgb valid log loss = {}'.format(log_loss(y2,xgb_valid)))
# print(df_train)