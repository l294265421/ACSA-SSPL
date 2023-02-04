# -*- coding: utf-8 -*-
"""

Date:    2018/9/28 15:14
"""

import json


def generate_word_index(texts, aspect_categories=None):
    """

    :param texts:
    :param aspect_categories:
    :return:
    """
    word_count = {}
    for sample in texts:
        words = sample.split(' ')
        for word in words:
            if word not in word_count:
                word_count[word] = 0
            word_count[word] += 1

    if aspect_categories is None:
        aspect_categories = []
    for aspect_category in aspect_categories:
        word_count[aspect_category] = 10000000
    word_count_list = list(word_count.items())
    word_count_list.sort(key=lambda x: x[1], reverse=True)
    count_threshold = 1
    word_count_list_bigger_threshold = [item for item in word_count_list if item[1] >= count_threshold]
    word_index = {}
    for i in range(len(word_count_list_bigger_threshold)):
        word_index[word_count_list_bigger_threshold[i][0]] = i + 1
    return word_index
