# -*- coding: utf-8 -*-
"""
给测试集添加label，统一训练集和测试集数据格式
Date:    2018/9/28 15:14
"""

import pandas as pd

from nlp_tasks.absa.conf import data_path

test_data = pd.read_csv(data_path.test_public_file_path)
test_data['subject'] = '价格'
test_data['sentiment_value'] = 0
test_data['sentiment_word'] = '无'
test_data.to_csv(data_path.test_public_with_label_file_path, index=False, encoding='utf-8')
