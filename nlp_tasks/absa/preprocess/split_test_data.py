# -*- coding: utf-8 -*-
"""
把原始测试数据分割为主题分类训练样本、情感分类训练样本和30分类训练样本
Date:    2018/9/28 15:14
"""

import pandas as pd

from nlp_tasks.absa.conf import data_path

test_data = pd.read_csv(data_path.test_public_with_label_file_path)

test_subject_data = test_data[['content_id', 'content', 'subject']]
test_subject_data.to_csv(data_path.test_subject_file_path, index=False, encoding='utf-8')

test_sentiment_testue_data = test_data[['content_id', 'content', 'subject',
                                         'sentiment_value']]
test_sentiment_testue_data.to_csv(data_path.test_sentiment_value_file_path, index=False,
                                  encoding='utf-8')

# test_subject_sentiment_testue_data = test_data[['content_id', 'content']]
# test_subject_sentiment_testue_data['subject_sentiment_testue'] = test_data\
#     .apply(lambda row: row['subject'] + '_' + str(row['sentiment_testue']), axis=1)
# test_subject_sentiment_testue_data.to_csv(data_path.test_subject_sentiment_testue_file_path,
#                                           index=False, encoding='utf-8')
