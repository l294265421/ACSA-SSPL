import os
import sys

project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
common_data_dir = project_dir + '/data/'
common_code_dir = project_dir + '/nlp_tasks/'

# original_data_dir = r'D:\Users\liyuncong\PycharmProjects\liyuncong-data\nlp\\'
original_data_dir = os.path.join(project_dir, 'original_data/')
original_data_dir_big = r'D:\Users\liyuncong\PycharmProjects\liyuncong-data-big\\'

stopwords_filepath = original_data_dir + 'common/stopwords.txt'


def get_task_data_dir(task_name: str, is_original=False):
    """

    :param task_name: 子任务名
    :return: 保存子任务的数据的目录
    """
    if not is_original:
        return '%s%s/' % (common_data_dir, task_name)
    else:
        return '%s%s/' % (original_data_dir, task_name)
