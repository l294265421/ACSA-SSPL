# -*- coding: utf-8 -*-
"""

Date:    2018/9/28 15:14
"""

import os
import warnings
import json
import codecs

import numpy as np

from nlp_tasks.absa.conf import data_path
from nlp_tasks.absa.utils import data_utils
from nlp_tasks.absa.utils import file_utils

word_index = {}
with codecs.open(data_path.bert_dict_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        word_index[token] = len(word_index)

json.dump(word_index, open(data_path.word_index_subject_file_path, mode='w', encoding='utf-8'), ensure_ascii=False)
