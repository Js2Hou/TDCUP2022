"""
根据预测结果，将图像中预测的目标框裁剪下来，保存为同名文件。便于查看检测分类结果
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time


imgs_dir = '/opt/data/private/projects/TDCUP2022/datasets/tdcup/images2'
save_dir = '/opt/data/private/projects/TDCUP2022/results/final_test/base_thred60_iter7w_data_dealed/crop2'
csv_path = '/opt/data/private/projects/TDCUP2022/results/final_test/base_thred60_iter7w_data_dealed/result2.csv'

os.makedirs(save_dir, exist_ok=True)

df = pd.read_csv(csv_path, encoding='gbk')
for i in range(len(df)):
    item_ = df.iloc[i].values
    idx = item_[0]
    class_id = item_[2]

    if class_id == 0:
        continue

    name_ = item_[1]
    x1, y1, x2, y2 = item_[5:].astype(int)

    img_path = os.path.join(imgs_dir, name_)
    img_save_dir = os.path.join(save_dir, str(class_id))
    if not os.path.exists(img_save_dir):
        os.makedirs(img_save_dir)

    img = plt.imread(img_path)
    img_crop = img[y1:y2, x1:x2]

    save_name = str(idx) + '_' + name_
    img_save_path = os.path.join(img_save_dir, save_name)
    plt.imsave(img_save_path, img_crop)

