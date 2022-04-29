# -*- coding: utf-8 -*-
import os
import json
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from cv2 import cv2
import os
import shutil

from sklearn.model_selection import train_test_split
np.random.seed(41)

def map_classname_id(classname_path):
    """
    Generate a mapping of a category name to a category number

    Return:
        classname_to_id (dict): category name and category number 
            are the keys and values respectively
    """
    classname_to_id = {}
    i = 0
    with open(classname_path, 'r') as f:
        for classname in f:
            classname = classname.rstrip()
            classname_to_id.setdefault(classname, i)
            i += 1
    return classname_to_id  


class Csv2CoCo:

    def __init__(self, total_annos, classname2id):
        self.images = []
        self.annotations = []
        self.categories = []
        self.img_id = 0
        self.ann_id = 0
        self.total_annos = total_annos
        self.classname2id = classname2id

        self._init_categories()

    def save_coco_json(self, instance, save_path):
        json.dump(instance, open(save_path, 'w'),
                  ensure_ascii=False, indent=2)  # indent=2 更加美观显示

    # 由txt文件构建COCO
    def to_coco(self, keys):
        for key in keys:
            image_item = self._image(key)
            if type(image_item) == type(-1) and image_item == -1:
                continue
            self.images.append(image_item)
            shapes = self.total_annos[key]
            for shape in shapes:
                label = str(shape[-1])
                bboxi = []
                for cor in shape[:-1]:
                    bboxi.append(int(cor))

                annotation = self._annotation(bboxi, label)
                self.annotations.append(annotation)

                print('label', label)
                print('annotation', annotation)
                self.ann_id += 1
            self.img_id += 1
        instance = {}
        instance['info'] = 'spytensor created'
        instance['license'] = ['license']
        instance['images'] = self.images
        instance['annotations'] = self.annotations
        instance['categories'] = self.categories
        return instance

    # 构建类别
    def _init_categories(self):
        for k, v in self.classname2id.items():
            category = {}
            category['id'] = v
            category['name'] = k
            self.categories.append(category)

    # 构建COCO的image字段
    def _image(self, path):
        image = {}
        print(path)
        img = plt.imread(path)
        if not img.shape:
            return -1
        image['height'] = img.shape[0]
        image['width'] = img.shape[1]
        image['id'] = self.img_id
        image['file_name'] = os.path.basename(path)
        return image

    # 构建COCO的annotation字段
    def _annotation(self, shape, label):
        points = shape[:4]
        annotation = {}
        annotation['id'] = self.ann_id
        annotation['image_id'] = self.img_id

        print(self.classname2id.keys())
        annotation['category_id'] = int(self.classname2id[label])
        #annotation['segmentation'] = self._get_seg(points)
        annotation['bbox'] = self._get_box(points)
        annotation['iscrowd'] = 0
        annotation['area'] = 1.0
        return annotation

    # COCO的格式： [x1,y1,w,h] 对应COCO的bbox格式
    def _get_box(self, points):
        min_x = points[0]
        min_y = points[1]
        max_x = points[2]
        max_y = points[3]
        return [min_x, min_y, max_x - min_x, max_y - min_y]

    # segmentation
    def _get_seg(self, points):
        min_x = points[0]
        min_y = points[1]
        max_x = points[2]
        max_y = points[3]
        h = max_y - min_y
        w = max_x - min_x
        a = []
        a.append([min_x, min_y, min_x, min_y+0.5*h, min_x, max_y, min_x+0.5*w, max_y,
                 max_x, max_y, max_x, max_y-0.5*h, max_x, min_y, max_x-0.5*w, min_y])
        return a


if __name__ == '__main__':
    annos_path = "datasets/annotations_mix.csv"
    saved_coco_path = "datasets/coco_mix/"
    classname_path = 'datasets/category_names.txt'
    test_size = 0.2
    # detect2_list = [8, 222, 280]  # 需要检测的类别编号列表

    classname2id = map_classname_id(classname_path)


    # 整合csv格式标注文件: path xmin ymin xmax ymax id
    total_csv_annotations = {}
    annotations = pd.read_csv(annos_path, header=None).values
    for annotation in annotations:
        if os.path.exists(annotation[0]):
            key = annotation[0]  # image's path
            # key = os.path.basename(annotation[0])

            # (tlpx, tlpy, brpx, brpy, class_id)
            value = np.array([annotation[1:]])

            # remove images without ojects when training
            # if value[0, -1] not in detect2_list:
            #     continue

            if key in total_csv_annotations.keys():
                total_csv_annotations[key] = np.concatenate(
                    (total_csv_annotations[key], value), axis=0)
            else:
                total_csv_annotations[key] = value
    for k, v in total_csv_annotations.items():
        print(k, v)

    # 创建必须的文件夹
    annotations_dir = os.path.join(saved_coco_path, 'annotations')
    train2017_dir = os.path.join(saved_coco_path, 'train2017')
    val2017_dir = os.path.join(saved_coco_path, 'val2017')
    if not os.path.exists(annotations_dir):
        os.makedirs(annotations_dir)
    if not os.path.exists(train2017_dir):
        os.makedirs(train2017_dir)
    if not os.path.exists(val2017_dir):
        os.makedirs(val2017_dir)

    # 按照键值划分数据
    total_keys = list(total_csv_annotations.keys())
    train_keys, val_keys = train_test_split(total_keys, test_size=test_size)
    print(f'train samples: {len(train_keys)}\tval samples: {len(val_keys)}')

    # 把训练集转化为COCO的json格式
    # 复制图片
    for file in train_keys:
        shutil.copy(file, (os.path.join(train2017_dir, os.path.basename(file))))


    c2c = Csv2CoCo(total_annos=total_csv_annotations, classname2id=classname2id)
    train_instance = c2c.to_coco(train_keys)
    c2c.save_coco_json(train_instance, os.path.join(
        annotations_dir, 'instances_train2017.json'))

    # 把验证集转化为COCO的json格式
    # 复制图片
    for file in val_keys:
        shutil.copy(file, (os.path.join(val2017_dir, os.path.basename(file))))

    c2c = Csv2CoCo(total_annos=total_csv_annotations, classname2id=classname2id)
    val_instance = c2c.to_coco(val_keys)
    c2c.save_coco_json(val_instance, os.path.join(
        annotations_dir, 'instances_val2017.json'))
