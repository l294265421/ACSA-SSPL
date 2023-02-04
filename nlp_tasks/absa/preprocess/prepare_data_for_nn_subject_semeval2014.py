# -*- coding: utf-8 -*-
"""
把原始数据转化为神经网络模型需要的形式:
1. 完成分词的步骤，把词用空格分开
2. 把类别转化为one_hot形式
Date:    2018/9/28 15:14
"""

import logging
import re

import jieba
import nltk

from nlp_tasks.absa.utils import file_utils
from nlp_tasks.absa.conf import data_path, datasets
from nlp_tasks.absa.preprocess import label_mapping_semeval2014 as label_mapping
from nlp_tasks.absa.preprocess import segmentor


def convert_nn_format(lines, label_mapping, sentence_convert_func, class_num):
    """

    :param lines: list of str, str示例：
    vUXizsqexyZVRdFH,因为森林人即将换代，这套系统没必要装在一款即将换代的车型上，因为肯定会影响价格。
    ,价格,0,影响
    :param label_mapping:
    :param sentence_convert_func:
    :param class_num:
    :return: list of str, str示例：
    4QroPd9hNfnCHVt7,四 驱 价格 貌似 挺 高 的 ， 高 的 可以 看齐 XC60 了 ， 看 实 车 前 脸 有点 违
    和 感   不过 大众 的 车 应该 不会 差  ,1 0 0 0 0 0 0 0 0 0,-1,高
    """
    id_parts_label = {}
    for line in lines:
        parts = line.split(datasets.delimeter)
        label = int(label_mapping[parts[2].strip()])
        parts[1] = sentence_convert_func(parts[1])
        if parts[0] not in id_parts_label:
            id_parts_label[parts[0]] = [parts, label]
        else:
            id_parts_label[parts[0]].append(label)
    result = []
    for id, parts_label in id_parts_label.items():
        labels = ['0' for i in range(class_num)]
        for i in parts_label[1:]:
            labels[i] = '1'
        parts = parts_label[0]
        parts[2] = ' '.join(labels)
        result.append(datasets.delimeter.join(parts))

    return result


def sentence_to_word_seq(text=''):
    """
    对文本进行分词，词用空格连接
    Args:
        text: 一段文本
    Returns:
        用空格连接的文本中的词
    """
    return ' '.join(nltk.word_tokenize(text))


def sentence_to_char_seq(text=''):
    """
    文本中的字用空格分开
    Args:
        text: 一段文本
    Returns:
        文本中的字用空格分开
    """
    sentence = re.sub('\s', '', text)
    chars = [c for c in sentence]
    return ' '.join(chars)


def sentence_to_bigram_seq(text=''):
    """
    用空格连接的文本中的bigram字符串
    Args:
        text: 一段文本
    Returns:
        用空格连接的文本中的bigram字符串
    """
    sentence = re.sub('\s', '', text)
    result = []
    for i in range(len(sentence) - 1):
        word = sentence[i: i + 2]
        if '。' in word or '？' in word or '!' in word or '，' in word or '、' in word or \
                '：' in word or '‘ ' in word or '’' in word or '“' in word or '”' in word:
            continue
        result.append(word)
    return ' '.join(result)


if __name__ == '__main__':
    train_file_path = data_path.train_file_path
    val_file_path = data_path.val_file_path
    test_file_path = data_path.test_public_file_path
    input_file_path = [train_file_path, val_file_path, test_file_path]
    output_file_path = [data_path.train_subject_word_file_path, data_path.val_subject_word_file_path,
                        data_path.test_subject_word_file_path]
    class_num = datasets.subject_class_num
    for i, file_path in enumerate(input_file_path):
        lines = file_utils.read_all_lines(file_path)
        head = [lines.pop(0)]
        file_utils.write_lines(head + convert_nn_format(lines, label_mapping.subject_mapping,
                                                        sentence_to_word_seq, class_num),
                               output_file_path[i])
