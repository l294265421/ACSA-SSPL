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
from nlp_tasks.absa.utils import result_utils
from nlp_tasks.absa.conf import data_path
from nlp_tasks.absa.preprocess import label_mapping
from nlp_tasks.absa.preprocess import segmentor
from nlp_tasks.absa.utils import data_utils

jieba.load_userdict(open(data_path.my_dict, encoding='utf-8'))


def convert_sentiment_value_to_nn_format(lines, label_mapping, sentence_convert_func, class_num):
    """

    Args:
        lines: 样本行，示例:
        label_mapping: key: 实际类别 value: 对应的序号，参考 preprocess.label_mapping.py模块
        sentence_convert_func: 把句子转化为由空格连接的词的形式的函数
        class_num: 样本类别总数
    Returns:
        示例:
    """
    result = []
    for line in lines:
        parts = line.split(',')

        label_index = int(label_mapping[parts[3].strip()])
        label_vector = ['0' for i in range(class_num)]
        label_vector[label_index] = '1'
        parts[3] = ' '.join(label_vector)

        sentence = parts[1]
        sentence_segment_result = sentence_convert_func(sentence)
        parts[1] = sentence_segment_result.strip()
        result.append(','.join(parts))
    return result


def sentence_to_word_seq_sentiment(text=''):
    """
    对文本进行分词，词用空格连接
    Args:
        text: 一段文本
    Returns:
        用空格连接的文本中的词
    """
    text = text.strip()
    if len(re.findall('[^，。？！；…]+[，。？！；…]$', text)) == 0:
        text += '。'
    # words = [word for word in jieba.cut(text)]
    # return ' '.join(words)
    return segmentor.jieba_segmentor(text, cut_all=False)


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
    # train_file_path = data_path.train_sentiment_value_exact_file_path
    # val_file_path = data_path.val_sentiment_value_exact_file_path
    # test_file_path = data_path.test_public_for_sentiment_value_exact_file_path
    train_file_path = data_path.train_file_path
    val_file_path = data_path.val_file_path
    test_file_path = data_path.test_public_gold_file_path
    train_sentiment_value_lines = file_utils.read_all_lines(train_file_path)
    head = [train_sentiment_value_lines.pop(0)]
    file_utils.write_lines(head + convert_sentiment_value_to_nn_format(
        train_sentiment_value_lines, label_mapping.sentiment_value_mapping,
        sentence_to_word_seq_sentiment, 3), train_file_path + '.word')
    # file_utils.write_lines(convert_sentiment_value_to_nn_format(
    #     train_sentiment_value_lines, label_mapping.sentiment_value_mapping,
    #     sentence_to_char_seq, 3),
    #                         data_path.train_sentiment_value_char_file_path)
    # file_utils.write_lines(convert_sentiment_value_to_nn_format(
    #     train_sentiment_value_lines, label_mapping.sentiment_value_mapping,
    #     sentence_to_bigram_seq, 3),
    #                         data_path.train_sentiment_value_bigram_file_path)

    val_sentiment_value_lines = file_utils.read_all_lines(val_file_path)[1:]
    file_utils.write_lines(head + convert_sentiment_value_to_nn_format(
        val_sentiment_value_lines, label_mapping.sentiment_value_mapping,
        sentence_to_word_seq_sentiment, 3), val_file_path + '.word')
    # file_utils.write_lines(convert_sentiment_value_to_nn_format(
    #     val_sentiment_value_lines, label_mapping.sentiment_value_mapping,
    #     sentence_to_char_seq, 3),
    #     data_path.val_sentiment_value_char_file_path)
    # file_utils.write_lines(convert_sentiment_value_to_nn_format(
    #     val_sentiment_value_lines, label_mapping.sentiment_value_mapping,
    #     sentence_to_bigram_seq, 3),
    #     data_path.val_sentiment_value_bigram_file_path)

    test_lines = file_utils.read_all_lines(test_file_path)[1:]
    file_utils.write_lines(head + convert_sentiment_value_to_nn_format(
        test_lines, label_mapping.sentiment_value_mapping,
        sentence_to_word_seq_sentiment, 3), test_file_path + '.word')
