import pandas as pd
import numpy as np

#data_root = '/opt/shared-data/kkbox-churn-prediction-challenge/'
data_root = '~/churn-prediction/kkbox-churn-prediction-challenge/'

df_trans = pd.read_csv(data_root+'transactions.csv')
df_trans = df_trans.append(pd.read_csv(data_root+'transactions_v2.csv'))

df_trans['transaction_date_year'] = df_trans['transaction_date'].apply(lambda x: int(str(x)[:4]))
df_trans['transaction_date_month'] = df_trans['transaction_date'].apply(lambda x: int(str(x)[4:6]))
df_trans['transaction_date_date'] = df_trans['transaction_date'].apply(lambda x: int(str(x)[-2:]))

df_trans['membership_expire_date_year'] = df_trans['membership_expire_date'].apply(lambda x: int(str(x)[:4]))
df_trans['membership_expire_date_month'] = df_trans['membership_expire_date'].apply(lambda x: int(str(x)[4:6]))
df_trans['membership_expire_date_date'] = df_trans['membership_expire_date'].apply(lambda x: int(str(x)[-2:]))

df_trans['transaction_date_year'] = df_trans['transaction_date_year'].astype(np.int16)
df_trans['transaction_date_month'] = df_trans['transaction_date_month'].astype(np.int8)
df_trans['transaction_date_date'] = df_trans['transaction_date_date'].astype(np.int8)

df_trans['membership_expire_date_year'] = df_trans['membership_expire_date_year'].astype(np.int16)
df_trans['membership_expire_date_month'] = df_trans['membership_expire_date_month'].astype(np.int8)
df_trans['membership_expire_date_date'] = df_trans['membership_expire_date_date'].astype(np.int8)

df_trans['payment_method_id'] = df_trans['payment_method_id'].astype(np.int8)
df_trans['payment_plan_days'] = df_trans['payment_plan_days'].astype(np.int16)
df_trans['plan_list_price'] = df_trans['plan_list_price'].astype(np.int16)
df_trans['actual_amount_paid'] = df_trans['actual_amount_paid'].astype(np.int16)
df_trans['is_auto_renew'] = df_trans['is_auto_renew'].astype(np.int8)
df_trans['is_cancel'] = df_trans['is_cancel'].astype(np.int8)

df_trans['discount'] = df_trans['plan_list_price'] - df_transactions['actual_amount_paid']
df_trans['is_discount'] = df_trans.discount.apply(lambda x: 1 if x > 0 else 0)

df_trans = df_trans.drop('transaction_date', 1)
df_trans = df_trans.drop('membership_expire_date', 1)

grouped = df_trans.copy().groupby('msno')
y1 = grouped.agg({'msno' :{'trans_count': 'count'},
                       'payment_plan_days' :{'transaction_span': 'sum'},
                       'plan_list_price' :{'total_list_price':'sum'},
                       'actual_amount_paid' : {'total_amount_paid' : 'sum'},
                       'is_auto_renew' :{'is_auto_renew': 'max'},
                       'is_cancel' : {'is_cancel_sum': 'sum'}})

y1.columns = y1.columns.droplevel(0)
y1.reset_index(inplace = True)
y1['difference_in_price_paid'] = y1['total_list_price'] - y1['total_amount_paid']
y1['amount_paid_perday'] = y1['total_amount_paid'] / y1['transaction_span']


for m in df_trans.columns:
    df_trans[m].fillna(method='ffill', inplace=True)

y1.to_csv('transac_processed.csv')
