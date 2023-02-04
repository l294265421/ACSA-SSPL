# -*- coding: utf-8 -*-
"""
把原始训练数据分割为主题分类训练样本、情感分类训练样本和30分类训练样本
Date:    2018/9/28 15:14
"""

import pandas as pd

from nlp_tasks.absa.conf import data_path

train_data = pd.read_csv(data_path.train_file_path)

train_subject_data = train_data[['content_id', 'content', 'subject']]
train_subject_data.to_csv(data_path.train_subject_file_path, index=False, encoding='utf-8')

train_sentiment_value_data = train_data[['content_id', 'content', 'subject',
                                         'sentiment_value']]
train_sentiment_value_data.to_csv(data_path.train_sentiment_value_file_path, index=False,
                                  encoding='utf-8')

# train_subject_sentiment_value_data = train_data[['content_id', 'content']]
# train_subject_sentiment_value_data['subject_sentiment_value'] = train_data\
#     .apply(lambda row: row['subject'] + '_' + str(row['sentiment_value']), axis=1)
# train_subject_sentiment_value_data.to_csv(data_path.train_subject_sentiment_value_file_path,
#                                           index=False, encoding='utf-8')
