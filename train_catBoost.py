from catboost import CatBoostClassifier
import numpy as np
import pandas as pd
import catboost
import gc 

gc.enable()

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
df_train["is_churn"] = df_train["is_churn"].astype('category',copy=False)
df_train["city"] = df_train["city"].astype('category',copy=False)
df_train["gender"] = df_train["gender"].astype('category',copy=False)
df_train["registered_via"] = df_train["registered_via"].astype('category',copy=False)
df_train["registration_init_time"] = df_train["registration_init_time"].astype('category',copy=False)

df_train['city'].fillna(method='ffill', inplace=True)
df_train['bd'].fillna(method='ffill', inplace=True)
df_train['gender'].fillna(method='ffill', inplace=True)
df_train['registered_via'].fillna(method='ffill', inplace=True)
df_train["registration_init_time"].fillna(method='ffill', inplace=True)

df_train['total_list_price'] = df_train['total_list_price'].astype(np.int16,copy=False)
df_train['transaction_span'] = df_train['transaction_span'].astype(np.int16,copy=False)
df_train['is_auto_renew'] = df_train['is_auto_renew'].astype('category',copy=False)
df_train['is_cancel_sum'] = df_train['is_cancel_sum'].astype('category',copy=False)
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
df_train['is_discount'] = df_train['is_discount'].astype('category',copy=False)

print(df_train.dtypes)
# df_train.fillna(-1)

features = [c for c in df_train.columns if c not in ['is_churn','msno']]
print('Using features')
print(features)

# print('Split data ...')
# x_train, x_validation, y_train, y_validation = model_selection.train_test_split(df_train[features],
#     df_train['is_churn'], test_size=0.2, random_state=0)
#x_test = df_test

model = CatBoostClassifier(
    #TODO: Train our own parameters?
    iterations = 200,
    learning_rate = 0.12,
    depth = 7,
    l2_leaf_reg = 3,
    loss_function = 'Logloss',
    eval_metric = 'Logloss',
    random_seed = 0
)

categorical_features_indices = np.where(df_train[features].dtypes != (np.float32 or np.int16))[0]

# model.fit(
#     x_train, y_train,
#     cat_features = categorical_features_indices,
#     eval_set=(x_validation,y_validation)
# )
print('training...')
#TODO: Parameter tuning if possible. iterations,learning_rate,depth,l2_leaf_reg
#retrain model on all data
model.fit(
    df_train[features], df_train['is_churn'],
    cat_features = categorical_features_indices
    # ,eval_set=(x_validation,y_validation)
    )
print('prediction...')
cat_valid = model.predict_proba(x_validation)[:,1]
print('Log loss: {}'.format(log_loss(y_validation,cat_valid)))

print('Saving ...')
model.save_model("CatBoost_model",format="cbm")

#df_test['is_churn'] = model.predict(df_test[features])
#df_test = df_test[['msno','is_churn']]
#df_test.to_csv('catBoost_submission.csv',index=False)
