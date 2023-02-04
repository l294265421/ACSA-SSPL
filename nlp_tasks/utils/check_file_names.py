import os

from nlp_tasks.utils import file_utils

str = 'zhong'

base_dir = r'D:\Users\liyuncong\PycharmProjects\SCAN\nlp_tasks'

for root, dirs, files in os.walk(base_dir, topdown=False):
    for name in files:
        if not name.endswith('py'):
            print(os.path.join(root, name))