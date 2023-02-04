# -*- coding: utf-8 -*-


from nlp_tasks.absa.conf import data_path
from nlp_tasks.absa.utils import file_utils

chusai_train_file_path = data_path.data_base_dir + 'bdci2018/train.csv'
chusai_test_file_path = data_path.data_base_dir + 'bdci2018/test_public_gold.csv'
fusai_train_file_path = data_path.data_base_dir + 'bdci2018/train_2.csv'

chusai_train_lines = file_utils.read_all_lines(chusai_train_file_path)[1:]
chusai_test_lines = file_utils.read_all_lines(chusai_test_file_path)[1:]
chusai_train_lines.extend(chusai_test_lines)
chusai_train_lines.sort(key=lambda line: line.split(',')[1])
chusai_train_lines = [','.join(line.split(',')[1:]) for line in chusai_train_lines]
file_utils.write_lines(chusai_train_lines, data_path.data_base_dir + 'bdci2018/chusai_data.csv')

fusai_train_lines = file_utils.read_all_lines(fusai_train_file_path)[1:]
fusai_train_lines.sort(key=lambda line: line.split(',')[1])
fusai_train_lines = [','.join(line.split(',')[1:]) for line in fusai_train_lines]
file_utils.write_lines(fusai_train_lines, data_path.data_base_dir + 'bdci2018/fusai_data.csv')

