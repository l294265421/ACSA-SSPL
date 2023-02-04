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
from nlp_tasks.absa.conf import data_path, task_conf, model_output_type, datasets
from nlp_tasks.absa.preprocess import segmentor


def convert_nn_format(lines: list, sentence_convert_func, label_mapping):
    """

    :param lines: 元素示例：
    content	subject	sentiment_value
    I lOVE THIS PLACE!	miscellaneous	positive
    :param sentence_convert_func:
    :return:
    """
    # key: id value: list,list的第一个元素是parts, 后面的元素是(aspect_label, sentiment_label)
    id_parts_label = {}
    # 记录句子最大长度
    max_words_num = 0
    for line in lines:
        parts = line.split('\t')
        aspect_label = int(label_mapping.subject_mapping[parts[2].strip()])
        sentiment_label = int(label_mapping.sentiment_value_mapping[parts[3].strip()])
        parts[1] = sentence_convert_func(parts[1])
        if len(parts[1].split(' ')) > max_words_num:
            max_words_num = len(parts[1].split(' '))
        if parts[0] not in id_parts_label:
            id_parts_label[parts[0]] = [parts, (aspect_label, sentiment_label)]
        else:
            id_parts_label[parts[0]].append((aspect_label, sentiment_label))
    print('max_words_num: %d' % max_words_num)
    result = []
    for id, parts_label in id_parts_label.items():
        if nlp_tasks.conf.model_output_type == model_output_type.ac_aoa:
            aspect_labels = ['0' for i in range(task_conf.subject_class_num)]
            sentiment_labels = [['0'] * task_conf.sentiment_class_num for i in range(task_conf.subject_class_num)]
            for aspect_label, sentiment_label in parts_label[1:]:
                aspect_labels[aspect_label] = '1'
                sentiment_labels[aspect_label][sentiment_label] = '1'
            sentiment_labels = [' '.join(e) for e in sentiment_labels]
            parts = parts_label[0]
            parts[2] = ' '.join(aspect_labels)
            parts[3] = task_conf.delimeter.join(sentiment_labels)
            result.append(task_conf.delimeter.join(parts))
        elif nlp_tasks.conf.model_output_type == model_output_type.end_to_end_lstm:
            aspect_labels = ['0' for i in range(task_conf.subject_class_num)]
            sentiment_labels = [['0'] * (task_conf.sentiment_class_num + 1) for i in
                                range(task_conf.subject_class_num)]
            for aspect_label, sentiment_label in parts_label[1:]:
                aspect_labels[aspect_label] = '1'
                sentiment_labels[aspect_label][sentiment_label] = '1'
            for sentiment_label in sentiment_labels:
                for label in sentiment_label:
                    if label == '1':
                        break
                else:
                    sentiment_label[task_conf.sentiment_class_num] = '1'
            sentiment_labels = [' '.join(e) for e in sentiment_labels]
            parts = parts_label[0]
            parts[2] = ' '.join(aspect_labels)
            parts[3] = task_conf.delimeter.join(sentiment_labels)
            result.append(task_conf.delimeter.join(parts))
        else:
            raise Exception('不支持的model_output_type: %s' % str(nlp_tasks.conf.model_output_type))

    return result


def sentence_to_word_seq(text=''):
    """
    对文本进行分词，词用空格连接
    Args:
        text: 一段文本
    Returns:
        用空格连接的文本中的词
    """
    return datasets.dataset_name_and_segmentor[task_conf.current_dataset](text)


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


def prepare_data(test_filename=''):
    train_file_path = data_path.train_file_path
    val_file_path = data_path.val_file_path
    test_file_path = data_path.test_public_gold_file_path
    input_file_path = [train_file_path, val_file_path]
    if len(test_filename) == 0:
        input_file_path.append(test_file_path)
    else:
        input_file_path.append(data_path.data_base_dir + test_filename)
    for i, file_path in enumerate(input_file_path):
        lines = file_utils.read_all_lines(file_path)
        head = [lines.pop(0)]
        file_utils.write_lines(head + convert_nn_format(lines, sentence_to_word_seq), file_path + '.word.' + str(
            nlp_tasks.conf.model_output_type))
