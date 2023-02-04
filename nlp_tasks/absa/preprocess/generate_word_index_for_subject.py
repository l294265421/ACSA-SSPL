# -*- coding: utf-8 -*-
"""

Date:    2018/9/28 15:14
"""

import os
import warnings
import json

import numpy as np

from nlp_tasks.absa.conf import data_path, datasets
from nlp_tasks.absa.utils import data_utils
from nlp_tasks.absa.utils import file_utils

np.random.seed(42)
warnings.filterwarnings('ignore')
os.environ['OMP_NUM_THREADS'] = '8'

langs = ['']
X = []
for lang in langs:
    train_file_path = data_path.train_subject_word_file_path + lang
    val_file_path = data_path.val_subject_word_file_path + lang

    X_train = data_utils.read_field(train_file_path, 1, separator=datasets.delimeter)[1:]
    X.extend(X_train)
    X_val = data_utils.read_field(val_file_path, 1, separator=datasets.delimeter)[1:]
    X.extend(X_val)

word_count = {}
for sample in X:
    words = sample.split()
    for word in words:
        if word not in word_count:
            word_count[word] = 0
        word_count[word] += 1

word_count_list = list(word_count.items())
word_count_list.sort(key=lambda x: x[1], reverse=True)
count_threshold = 3
word_count_list_bigger_threshold = [item for item in word_count_list if item[1] >= count_threshold]
word_index = {}
for i in range(len(word_count_list_bigger_threshold)):
    word_index[word_count_list_bigger_threshold[i][0]] = i + 1
json.dump(word_index, open(data_path.word_index_subject_file_path, mode='w', encoding='utf-8'), ensure_ascii=False)
