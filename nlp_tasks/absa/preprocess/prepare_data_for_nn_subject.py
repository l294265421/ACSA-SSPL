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

from nlp_tasks.absa.utils import file_utils
from nlp_tasks.absa.conf import data_path
from nlp_tasks.absa.preprocess import label_mapping
from nlp_tasks.absa.preprocess import segmentor

jieba.load_userdict(open(data_path.my_dict, encoding='utf-8'))


def convert_nn_format(lines, label_mapping, sentence_convert_func, class_num):
    id_parts_label = {}
    for line in lines:
        parts = line.split(',')
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
        result.append(','.join())

    return result


def sentence_to_word_seq(text=''):
    """
    对文本进行分词，词用空格连接
    Args:
        text: 一段文本
    Returns:
        用空格连接的文本中的词
    """
    return segmentor.jieba_segmentor(text)


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
    train_subject_lines = file_utils.read_all_lines(data_path.train_file_path)
    head = [train_subject_lines.pop(0)]
    file_utils.write_lines(head + convert_nn_format(train_subject_lines, label_mapping.subject_mapping,
                                                    sentence_to_word_seq, 10),
                           data_path.train_subject_word_file_path)
    # file_utils.write_lines(convert_nn_format(train_subject_lines, label_mapping.subject_mapping,
    #                                          sentence_to_char_seq, 10),
    #                        data_path.train_subject_char_file_path)
    # file_utils.write_lines(convert_nn_format(train_subject_lines, label_mapping.subject_mapping,
    #                                          sentence_to_bigram_seq, 10),
    #                        data_path.train_subject_bigram_file_path)

    val_subject_lines = file_utils.read_all_lines(data_path.val_file_path)[1:]
    file_utils.write_lines(head + convert_nn_format(val_subject_lines, label_mapping.subject_mapping,
                                                    sentence_to_word_seq, 10),
                           data_path.val_subject_word_file_path)
    # file_utils.write_lines(convert_nn_format(val_subject_lines, label_mapping.subject_mapping,
    #                                          sentence_to_char_seq, 10),
    #                        data_path.val_subject_char_file_path)
    # file_utils.write_lines(convert_nn_format(val_subject_lines, label_mapping.subject_mapping,
    #                                          sentence_to_bigram_seq, 10),
    #                        data_path.val_subject_bigram_file_path)

    # train_subject_sentiment_value_lines = file_utils.read_all_lines(
    #     data_path.train_subject_sentiment_value_file_path)[1:]
    # file_utils.write_lines(convert_nn_format(
    #     train_subject_sentiment_value_lines, label_mapping.subject_sentiment_value_mapping,
    #     sentence_to_word_seq, 30),
    #                        data_path.train_subject_sentiment_value_word_file_path)
    # file_utils.write_lines(convert_nn_format(
    #     train_subject_sentiment_value_lines, label_mapping.subject_sentiment_value_mapping,
    #     sentence_to_char_seq, 30),
    #                        data_path.train_subject_sentiment_value_char_file_path)
    # file_utils.write_lines(convert_nn_format(
    #     train_subject_sentiment_value_lines, label_mapping.subject_sentiment_value_mapping,
    #     sentence_to_bigram_seq, 30),
    #                         data_path.train_subject_sentiment_value_bigram_file_path)

    test_public_lines = file_utils.read_all_lines(data_path.test_public_with_label_file_path)[1:]
    file_utils.write_lines(head + convert_nn_format(test_public_lines, label_mapping.subject_mapping,
                                                    sentence_to_word_seq, 10),
                           data_path.test_subject_word_file_path)
    # file_utils.write_lines(convert_nn_format(test_lines, {}, sentence_to_char_seq, 0, True),
    #                        data_path.test_public_char_file_path)
    # file_utils.write_lines(convert_nn_format(test_lines, {}, sentence_to_bigram_seq, 0, True),
    #                        data_path.test_public_bigram_file_path)
