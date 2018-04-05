'''
last_1_is_churn', 'last_2_is_churn', 'last_3_is_churn', 'last_4_is_churn', 'last_5_is_churn',
'churn_rate', 'churn_count',
If one member had membership expire five times in the past,
for example 201603, 201604, 201605, 201606, and 201607,
"last_1_is_churn" means did this member churn in 201607 or not;
"last_2_is_churn" means did this member churn in 201606 or not; and so on.
'''

import numpy as np
import pandas as pd
import datetime
from tqdm import tqdm

# transactions = pd.read_csv('trans_test.csv').reset_index(drop=True)
transactions_v1 = pd.read_csv('transactions.csv')
transactions_v2 = pd.read_csv('transactions_v2.csv')
transactions = transactions_v1.append(transactions_v2).reset_index(drop=True)

transactions = transactions.drop(columns=['payment_method_id', 'payment_plan_days',
                                          'plan_list_price', 'actual_amount_paid', 'is_auto_renew', 'is_cancel'])
transactions = transactions.sort_values(
    ['msno', 'transaction_date'])
# result = pd.DataFrame(columns=['msno','last_1_is_churn','last_2_is_churn','last_3_is_churn','last_4_is_churn','last_5_is_churn','churn_rate','churn_count'])
result = pd.DataFrame()


transaction_dates = []
membership_expire_dates = []
prev_msno = ''
total_rows = len(transactions['msno'])


def calc_churn(t_dates, e_dates, msno, result):
    churns = []
    for i, e_date in enumerate(e_dates):
        if int(e_dates[i]) / 100 >= 201703:  # if expiration data is 201703 onwards, treat as no churn
            churns.insert(0, 0)
            continue
        expired_date = datetime.datetime.strptime(str(e_dates[i]), "%Y%m%d")
        churn = 1
        for j, t_date in enumerate(t_dates):
            trans_date = datetime.datetime.strptime(str(t_dates[j]), "%Y%m%d")
            dif_d = (trans_date - expired_date).days
            if (0 <= dif_d < 30):  # if some trans resubscribe
                churn = 0
        churns.insert(0, churn)

    churn_rate = 0
    if len(churns) > 0:
        churn_rate = (sum(churns) / len(churns))
    churn_count = sum(churns)

    while len(churns) < 5:
        churns.append(0)

    df = {'msno': [msno], 'last_1_is_churn': [churns[0]], 'last_2_is_churn': [churns[1]], 'last_3_is_churn': [
        churns[2]], 'last_4_is_churn': [churns[3]], 'last_5_is_churn': [churns[4]], 'churn_rate': [churn_rate], 'churn_count': [churn_count]}
    #df = pd.DataFrame(data=np.array([[1, 2, 3]]), columns=['msno','last_1_is_churn','last_2_is_churn','last_3_is_churn','last_4_is_churn','last_5_is_churn'])
    return result.append(pd.DataFrame(data=df))
    # print(pd.DataFrame(data=df))


for i, row in tqdm(transactions.iterrows(), total=total_rows):
    msno = row['msno']
    transaction_date = row['transaction_date']
    membership_expire_date = row['membership_expire_date']

    if (msno != prev_msno):
        result = calc_churn(transaction_dates,
                            membership_expire_dates, prev_msno, result)
        transaction_dates = []
        membership_expire_dates = []  # clear lists, start recording this new user id

    transaction_dates.append(transaction_date)
    membership_expire_dates.append(membership_expire_date)
    prev_msno = msno

result = result[['msno', 'last_1_is_churn',
                 'last_2_is_churn', 'last_3_is_churn', 'last_4_is_churn', 'last_5_is_churn', 'churn_rate', 'churn_count']]
result.to_csv('last_is_churns.csv', index=False)
