# -*- coding: utf-8 -*-
"""

Date:    2018/11/5 11:33
"""

from nlp_tasks.absa.conf import data_path
from nlp_tasks.absa.utils import file_utils

subject_result_file_path = data_path.test_subject_result_file_path
subject_result_lines = file_utils.read_all_lines(subject_result_file_path)
result = ['content_id,subject,sentiment_value,sentiment_word']
for line in subject_result_lines:
    parts = line.split(',')
    for subject in parts[1].split('|'):
        result.append(parts[0] + ',' + subject + ',' + '0,')
file_utils.write_lines(result, subject_result_file_path + '.submit')