import pandas as pd
from tqdm import tqdm
transactions_v1 = pd.read_csv('transactions.csv')
transactions_v2 = pd.read_csv('transactions_v2.csv')
transactions = transactions_v1.append(transactions_v2).reset_index(drop=True)
transactions = transactions.sort_values(['msno','transaction_date','membership_expire_date'])

total_rows = len(transactions['msno'])
for i, row in tqdm(transactions.iterrows(), total=total_rows):
    pass
