# -*- coding: utf-8 -*-


from nlp_tasks.absa.conf import data_path
from nlp_tasks.absa.utils import file_utils

chusai_test_file_path = data_path.data_base_dir + 'bdci2018/test_public.csv'
fusai_train_file_path = data_path.data_base_dir + 'bdci2018/train_2.csv'

chusai_test_lines = file_utils.read_all_lines(chusai_test_file_path)[1:]
fusai_train_lines = file_utils.read_all_lines(fusai_train_file_path)
head = fusai_train_lines.pop(0)
fusai_content_line_dict = {}
for line in fusai_train_lines:
    parts = line.split(',')
    if parts[1].strip() in fusai_content_line_dict:
        fusai_content_line_dict[parts[1].strip()].append(parts)
    else:
        fusai_content_line_dict[parts[1].strip()] = [parts]
result = [head]
result_set = set()
for line in chusai_test_lines:
    parts = line.split(',')
    answers = fusai_content_line_dict[parts[1].strip()]
    for answer in answers:
        answer[0] = parts[0]
        answer[1] = parts[1]
        if ','.join(answer) not in result_set:
            result.append(','.join(answer))
            result_set.add(','.join(answer))

file_utils.write_lines(result, data_path.data_base_dir + 'bdci2018/test_public_gold.csv')
