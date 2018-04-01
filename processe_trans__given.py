import pandas as pd
import numpy as np

#data_root = '/opt/shared-data/kkbox-churn-prediction-challenge/'
data_root = '~/churn-prediction/kkbox-churn-prediction-challenge/'

df_trans = pd.read_csv(data_root+'transactions.csv')
df_trans = df_trans.append(pd.read_csv(data_root+'transactions_v2.csv'))
df_trans.drop_duplicates(subset = ['msno'], keep = 'first')
df_trans = df_trans.drop(['is_auto_renew', 'membership_expire_date','transaction_date'], 1)

for m in df_trans.columns:
    df_trans[m].fillna(method='ffill', inplace=True)

df_trans.to_csv('transac_given_processed.csv')
