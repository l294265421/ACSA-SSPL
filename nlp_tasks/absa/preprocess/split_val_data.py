# -*- coding: utf-8 -*-
"""
把原始验证数据分割为主题分类训练样本、情感分类训练样本和30分类训练样本
Date:    2018/9/28 15:14
"""

import pandas as pd

from nlp_tasks.absa.conf import data_path

val_data = pd.read_csv(data_path.val_file_path)

val_subject_data = val_data[['content_id', 'content', 'subject']]
val_subject_data.to_csv(data_path.val_subject_file_path, index=False, encoding='utf-8')

val_sentiment_value_data = val_data[['content_id', 'content', 'subject',
                                         'sentiment_value']]
val_sentiment_value_data.to_csv(data_path.val_sentiment_value_file_path, index=False,
                                encoding='utf-8')

# val_subject_sentiment_value_data = val_data[['content_id', 'content']]
# val_subject_sentiment_value_data['subject_sentiment_value'] = val_data\
#     .apply(lambda row: row['subject'] + '_' + str(row['sentiment_value']), axis=1)
# val_subject_sentiment_value_data.to_csv(data_path.val_subject_sentiment_value_file_path,
#                                           index=False, encoding='utf-8')
