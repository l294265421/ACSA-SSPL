# -*- coding: utf-8 -*-
"""

Date:    2018/11/11 13:51
"""

import json

from nlp_tasks.absa.conf import data_path
from nlp_tasks.absa.utils import file_utils

result = []
train_lines = file_utils.read_all_lines(data_path.train_original_file_path)[1:]
for line in train_lines:
    parts = line.split(',')
    line_json = {}
    line_json['id'] = parts[0]
    line_json['_id'] = parts[0]
    line_json['title'] = '的'
    line_json['type'] = 'news'
    line_json['content'] = parts[1]
    line_json['insert_time'] = '20160713122001'
    result.append(json.dumps(line_json, ensure_ascii=False))

test_lines = file_utils.read_all_lines(data_path.test_public_file_path)[1:]
for line in test_lines:
    parts = line.split(',')
    line_json = {}
    line_json['id'] = parts[0]
    line_json['_id'] = parts[0]
    line_json['title'] = '的'
    line_json['type'] = 'news'
    line_json['content'] = parts[1]
    line_json['insert_time'] = '20180714122000'
    result.append(json.dumps(line_json, ensure_ascii=False))

file_utils.write_lines(result, data_path.data_base_dir + 'cluster')