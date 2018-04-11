import pandas as pd
transactions_v1 = pd.read_csv('transactions.csv')
transactions_v2 = pd.read_csv('transactions_v2.csv')
transactions = transactions_v1.append(transactions_v2).reset_index(drop=True)

transactions = transactions.drop(columns=['payment_method_id', 'payment_plan_days',
                                          'plan_list_price', 'actual_amount_paid', 'is_auto_renew', 'is_cancel'])
transactions = transactions.sort_values(
    ['msno', 'transaction_date','membership_expire_date'])
transactions.to_csv('sorted_trans_for_last.csv', index=False)
