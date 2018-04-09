import numpy as np
import pandas as pd
import catboost
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
df_test = pd.read_csv(data_root+'sample_submission_v2.csv')

print('Merging data ...')
#df_test = train.merge(members, how='left', on='msno')
df_test = df_test.merge(members, how='left', on='msno')

#df_test = df_test.merge(num_mean, how='left', on='msno')
df_test = df_test.merge(num_mean, how='left', on='msno')

#df_test = df_test.merge(transaction, how='left', on='msno')
df_test = df_test.merge(transaction, how='left', on='msno')

#df_test = df_test.merge(transaction_given, how='left', on='msno')
df_test = df_test.merge(transaction_given, how='left', on='msno')

del transaction, transaction_given, members, num_mean
gc.collect()

# Drop duplicates first
df_test = df_test.drop_duplicates('msno')

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

df_test['payment_method_id'] = df_test['payment_method_id'].astype(np.int16)
df_test['payment_plan_days'] = df_test['payment_plan_days'].astype(np.int16)
df_test['plan_list_price'] = df_test['plan_list_price'].astype(np.int16)
df_test['actual_amount_paid'] = df_test['actual_amount_paid'].astype(np.int16)
df_test['is_cancel'] = df_test['is_cancel'].astype(np.int16)
df_test['discount'] = df_test['discount'].astype(np.int16)
df_test['is_discount'] = df_test['is_discount'].astype('category')

# print(df_test.dtypes)
# # df_test.fillna(-1)
#
# features = [c for c in df_test.columns if c not in ['is_churn','msno']]
# print('Using features')
# print(features)
#
# print('Split data ...')
# x_train, x_validation, y_train, y_validation = model_selection.train_test_split(df_test[features],
#     df_test['is_churn'], test_size=0.2, random_state=0)
# x_test = df_test
#
# model = CatBoostClassifier(
#     #TODO: Train our own parameters?
#     iterations = 200,
#     learning_rate = 0.12,
#     depth = 7,
#     l2_leaf_reg = 3,
#     loss_function = 'Logloss',
#     eval_matric = 'Logloss',
#     randon_seed = 0
# )
#
# categorical_features_indices = np.where(df_test[features].dtypes != (np.float32 or np.int16))[0]
#
# model.fit(
#     x_train, y_train,
#     cat_features = categorical_features_indices,
#     eval_set=(x_validation,y_validation)
# )
#
# #TODO: Parameter tuning if possible. iterations,learning_rate,depth,12_leaf_reg
#
# cat_valid = model.predict_proba(x_validation)[:,1]
# print('Log loss: {}'.format(log_loss(y_validation,cat_valid)))
#
# #retrain model on all data
# model.fit(
#     df_test[features], df_test['is_churn'],
#     cat_features = categorical_features_indices,
#     eval_set=(x_validation,y_validation)
#     )
# print('Saving ...')
# model.save_model('CatBoost_model')

model = catboost.load_model(data_root+'CatBoost.cbm')

df_test['is_churn'] = model.predict(df_test[features])
df_test = df_test[['msno','is_churn']]
df_test.to_csv('catBoost_submission.csv',index=False)
