# -*- coding: utf-8 -*-

import requests
import json
import re
import time

import jieba
import nltk
import nltk.stem
from nltk.corpus import stopwords

from nlp_tasks.absa.bert_text_classification import tokenization
from nlp_tasks.absa.conf import data_path

english_stop_words = stopwords.words('english')
english_keep_words = set(['not', 'no', 'and', 'but'])
stemmer = nltk.stem.SnowballStemmer('english')
english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%','-']


def jieba_segmentor(text='', cut_all=True):
    """
    对文本进行分词，词用空格连接
    Args:
        text: 一段文本
    Returns:
        用空格连接的文本中的词
    """
    words = [word for word in jieba.cut(text)]
    if not cut_all:
        return ' '.join(words)
    result = []
    punctuation = '！？｡＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏.'
    for word in words:
        if word in punctuation:
            result.append(word)
        else:
            result += jieba.cut(word, cut_all=cut_all)
    return ' '.join(result)


class NltkSegmentor:
    def __init__(self, remove_stop_words=False, remove_punctuations=False, is_stem=False, lower=True):
        self.remove_stop_words = remove_stop_words
        self.remove_punctuations = remove_punctuations
        self.is_stem = is_stem
        self.lower = lower

    def __call__(self, text):
        if self.lower:
            text = text.lower()
        words = nltk.word_tokenize(text)
        if self.is_stem:
            words = [stemmer.stem(word) for word in words]
        if self.remove_stop_words:
            words = [word for word in words if (word not in english_stop_words or word in english_keep_words)]
        if self.remove_punctuations:
            words = [word for word in words if word not in english_punctuations]
        return ' '.join(words)


def nltk_segmentor(text='', remove_stop_words=False, remove_punctuations=False, is_stem=False):
    """
    对文本进行分词，词用空格连接
    Args:
        text: 一段文本
    Returns:
        用空格连接的文本中的词
    """
    text = text.lower()
    words = nltk.word_tokenize(text)
    if is_stem:
        words = [stemmer.stem(word) for word in words]
    if remove_stop_words:
        words = [word for word in words if word not in english_stop_words]
    if remove_punctuations:
        words = [word for word in words if word not in english_punctuations]
    return ' '.join(words)


def bert_segmentor(text=''):
    bert_tokenizer = tokenization.FullTokenizer(vocab_file=data_path.bert_dict_path)
    result = bert_tokenizer.tokenize(text)
    return ' '.join(result)


if __name__ == '__main__':
    print(bert_segmentor('But the staff was so horrible to us.'))