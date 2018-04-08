import pandas as pd
import numpy as np

#data_root = '/opt/shared-data/kkbox-churn-prediction-challenge/'
data_root = '~/churn-prediction/kkbox-churn-prediction-challenge/'

df_trans = pd.read_csv(data_root+'transactions.csv')
df_trans = df_trans.append(pd.read_csv(data_root+'transactions_v2.csv'))
df_trans.drop_duplicates(subset = ['msno'], keep = 'first')
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

for m in y1.columns:
    y1[m].fillna(0)

y1.to_csv('transac_processed.csv')
