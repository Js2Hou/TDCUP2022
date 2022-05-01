import os
import pandas as pd
import numpy as np

csv_path = '/opt/data/private/projects/TDCUP2022/results/base_thred60/base_60_all.csv'
save_csv_path = '/opt/data/private/projects/TDCUP2022/results/base_thred60/answer3.csv'

res = {}

df = pd.read_csv(csv_path, encoding='gbk')

for i in range(len(df)):
    item_ = df.iloc[i].values
    name_, cid = item_[1:3]

    if name_ not in res.keys():
        res[name_] = {}
    
    if cid not in res[name_].keys():
        res[name_][cid] = 0

    if cid != 0:
        res[name_][cid] += 1


res2 = []
for name_, v in res.items():
    for kk, vv in v.items():
        res2.append([name_, kk, vv])

idx = (np.arange(len(res2)) + 1).reshape(-1, 1)

data_new = np.hstack((idx, np.array(res2)))

df_new = pd.DataFrame(data_new)

header = ['序号','文件名','虫子编号','数量']

df_new.to_csv(save_csv_path, header=header,
                        index=False, encoding='gbk')
