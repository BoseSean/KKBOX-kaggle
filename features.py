import numpy as np
import pandas as pd
import gc

def fix_time(df, time_cols):
    for time_col in time_cols:
        df[time_col] = pd.to_datetime(df[time_col], errors = 'coerce', format = '%Y%m%d')
    return df

gc.enable()

df_train = pd.read_csv('~/train.csv')
df_train = pd.append((df_train, pd.read_csv('~/train_v2.csv')), axis=0, ignore_index=True).reset_index(drop=True)
df_test = pd.read_csv('~/sample_submission_v2.csv')

df_members = pd.read_csv('~/members_v3.csv')

df_transactions = pd.read_csv('~/transactions.csv')
df_transactions = pd.append((df_transactions, pd.read_csv('~/transactions_v2.csv')), axis=0, ignore_index=True).reset_index(drop=True)
df_transactions = df_transactions.sort_values(by=['transaction_date'], ascending=[False]).reset_index(drop=True)
df_transactions = df_transactions.drop_duplicates(subset=['msno'], keep='first')

# Feature Engineering
df_transactions['discount'] = df_transactions['plan_list_price'] - df_transactions['actual_amount_paid']
df_transactions['is_discount'] = df_transactions.discount.apply(lambda x: 1 if x > 0 else 0)
df_transactions['membership_days'] = pd.to_datetime(df_transactions['membership_expire_date']).subtract(pd.to_datetime(df_transactions['transaction_date'])).dt.days.astype(np.int16)
df_transactions['amt_per_day'] = df_transactions['actual_amount_paid'] / df_transactions['payment_plan_days']

df_train['marker'] = 1
df_test['marker'] = 0
df_combined = pd.concat([df_train, df_test], axis=0)

df_combined = pd.merge(df_combined, df_members, how='left', on='msno')
print('merging members')
df_members = [];


gender = {'male':1, 'female':2}
df_combined['gender'] = df_combined['gender'].map(gender)

df_combined = pd.merge(combined, df_transactions, how='left', on='msno')
df_transactions =[]
print('merging transactions')

df_train = df_combined[df_combined['marker'] == 1]
df_test = df_combined[df_combined['marker']  == 0]

df_train.drop(['marker'], axis=1, inplace = True)
df_test.drop(['marker'], axis=1, inplace = True)

del df_combined
gc.collect()
gc.enable()

df_userlogs = pd.read_csv('~/user_logs.csv')
df_userlogs = df_userlogs.drop_duplicates(subset=['msno'], keep='first')
df_userlogs.append(pd.read_csv('~/user_logs_v2.csv'))
df_userlogs = df_userlogs.drop_duplicates(subset=['msno'], keep='first')

df_train = pd.merge(df_train, df_userlogs, how='left', on='msno')
df_test = pd.merge(df_test, df_userlogs, how='left', on='msno')
del df_userlogs
gc.collect()

df_train = df_train.rename(columns = {'date':'user_log_date'})
df_test = df_test.rename(columns = {'date':'user_log_date'})

df_train['user_log_date'] = df_train['user_log_date'].fillna(20170101)
df_train = fix_time(df_train, time_cols = ['transaction_date', 'membership_expire_date', 'registration_init_time', 'user_log_date'])

df_test['user_log_date'] = df_test['user_log_date'].fillna(20170101)
df_test = fix_time(df_test, time_cols = ['transaction_date', 'membership_expire_date', 'registration_init_time', 'user_log_date'])

# Splitting of Dates into Days and Months Respectively

date_dict = {'t_':'transaction_date', 'm_':'membership_expire_date', \
             'r_':'registration_init_time', 'l_':'user_log_date'}

for m in date_dict:
    df_train[m+'month'] = [d.month for d in df_train[date_dict[m]]]
    df_train[m+'day'] = [d.day for d in df_train[date_dict[m]]]

df_train['transaction_date'] = [d.year  for d in df_train['transaction_date']]
df_train['membership_expire_date'] = [d.year for d in df_train['membership_expire_date']]
df_train['registration_init_time'] = [d.year for d in df_train['registration_init_time']]
df_train['last_user_log_date'] = [d.year for d in df_train['last_user_log_date']]

for m in date_dict:
        df_test[m+'month'] = [d.month for d in df_test[date_dict[m]]]
        df_test[m+'day'] = [d.day for d in df_test[date_dict[m]]]

df_test['transaction_date'] = [d.year for d in df_test['transaction_date']]
df_test['membership_expire_date'] = [d.year for d in df_test['membership_expire_date']]
df_test['registration_init_time'] = [d.year for d in df_test['registration_init_time']]
df_test['last_user_log_date'] = [d.year for d in df_test['last_user_log_date']]

#  whether the user automatically renew and not cancel and vice versa

df_train['autorenew_&_not_cancel'] = ((df_train.is_auto_renew == 1) == (df_train.is_cancel == 0)).astype(np.int8)
df_test['autorenew_&_not_cancel'] = ((df_test.is_auto_renew == 1) == (df_test.is_cancel == 0)).astype(np.int8)
df_train['notAutorenew_&_cancel'] = ((df_train.is_auto_renew == 0) == (df_train.is_cancel == 1)).astype(np.int8)
df_test['notAutorenew_&_cancel'] = ((df_test.is_auto_renew == 0) == (df_test.is_cancel == 1)).astype(np.int8)

df_train = df_train.fillna(0)
df_test = df_test.fillna(0)

df_train.to_csv('sorted_train.csv', encoding='utf-8', index= True)
df_test.to_csv('sorted_test.csv', encoding='utf-8', index= True)
