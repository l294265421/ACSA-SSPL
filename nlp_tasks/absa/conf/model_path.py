import os
from nlp_tasks.absa.conf import data_path

# model_base_dir = os.getcwd() + '/'
model_base_dir = data_path.data_base_dir

model_file_dir = model_base_dir + r'model_files/'
model_log_dir = model_base_dir + r'model_logs/'

bagging_multi_label_topic_file_path = model_file_dir + r'bagging_multi_label_topic'
