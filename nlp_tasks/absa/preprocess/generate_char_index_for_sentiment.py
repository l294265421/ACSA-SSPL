# -*- coding: utf-8 -*-
"""

Date:    2018/9/28 15:14
"""

import os
import warnings
import json

import numpy as np

from nlp_tasks.absa.conf import data_path
from nlp_tasks.absa.utils import data_utils

np.random.seed(42)
warnings.filterwarnings('ignore')
os.environ['OMP_NUM_THREADS'] = '8'

train_file_path = data_path.train_subject_char_file_path
test_file_path = data_path.test_public_char_file_path

X_train = data_utils.read_features(train_file_path)

X_test = data_utils.read_features(test_file_path)

X = X_train + X_test

word_count = {}
for sample in X:
    words = sample.split()
    for word in words:
        if word not in word_count:
            word_count[word] = 0
        word_count[word] += 1

word_count_list = list(word_count.items())
word_count_list.sort(key=lambda x: x[1], reverse=True)
count_threshold = 0
word_count_list_bigger_threshold = [item[0] for item in word_count_list if item[1] > count_threshold]

subject_labels = ['价格', '配置', '操控', '舒适性', '油耗', '动力', '内饰', '安全性', '空间', '外观']

word_count_list_bigger_threshold += subject_labels

word_index = {}
for i in range(len(word_count_list_bigger_threshold)):
    word_index[word_count_list_bigger_threshold[i]] = i
json.dump(word_index, open(data_path.data_base_dir + data_path.char_index_sentiment_file_path,
                           mode='w', encoding='utf-8'),
          ensure_ascii=False)
