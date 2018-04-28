import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


files = [
('xgboost_prediction_3.csv',  0.73*100),
('catboost_prediction_1.csv', 0.18*100),
('ligthgbm_prediction_1.csv', 0.09*100)
]

cnt = 0
for f,p in files:
    cnt += p


data_root = './'

final_result = 0

for f_name,p in files:
    file = pd.read_csv( data_root+f_name)
    final_result += file['is_churn'] * (p / cnt)

file['is_churn'] = final_result.clip(0.0000001, 0.999999)
file[['msno', 'is_churn']].to_csv("combine_submission.csv", index=False)

# for i in [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
#     a = f[(f['is_churn']>i) & (f['is_churn']<=i+0.1)]
#     print(i,a.size)
