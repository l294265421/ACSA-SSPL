# -*- coding: utf-8 -*-
"""
把训练集拆分为训练集和验证集，验证集用于线下效果评估
Date:    2018/9/28 15:14
"""

import random
import os

from nlp_tasks.absa.conf import data_path
from nlp_tasks.absa.utils import file_utils
from nlp_tasks.absa.conf import task_conf


def split():
    train_lines = file_utils.read_all_lines(data_path.train_original_file_path)
    head = train_lines.pop(0)
    train_train_lines = [head]
    train_val_lines = [head]
    id_samples = {}
    aspect_total_count = {}
    for line in train_lines:
        parts = line.split(task_conf.delimeter)
        id = parts[0]
        if id not in id_samples:
            id_samples[id] = []
        id_samples[id].append(line)

        aspect = parts[task_conf.aspect_index]
        if aspect not in aspect_total_count:
            aspect_total_count[aspect] = 0
        aspect_total_count[aspect] += 1

    print('train aspect_total_count: %s' % str(aspect_total_count))

    val_ratio = 0.1
    aspect_count = {}
    values = list(id_samples.values())
    random.shuffle(values)
    for samples in values:
        val = True
        aspects = []
        for example in samples:
            parts = example.split(task_conf.delimeter)
            aspect = parts[task_conf.aspect_index]
            aspects.append(aspect)
            if aspect not in aspect_count:
                aspect_count[aspect] = 0
            if aspect_count[aspect] > (aspect_total_count[aspect] * val_ratio):
                val = False
        if val:
            train_val_lines.extend(samples)
            for aspect in aspects:
                if aspect not in aspect_count:
                    aspect_count[aspect_count] = 0
                aspect_count[aspect] += 1
        else:
            train_train_lines.extend(samples)

    print('val aspect_count: %s' % str(aspect_count))
    file_utils.write_lines(train_train_lines, data_path.train_file_path)
    file_utils.write_lines(train_val_lines, data_path.val_file_path)

    # train aspect_total_count: {'service': 597, 'food': 1232, 'miscellaneous': 1132, 'price': 321, 'ambience': 431}
    # val aspect_count: {'miscellaneous': 114, 'ambience': 44, 'service': 60, 'food': 124, 'price': 33}