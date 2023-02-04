import os

from nlp_tasks.utils import file_utils

str = 'zhong'

base_dir = r'C:\Users\liyuncong\Desktop\EACL2021\119_file_Supplementary'

for root, dirs, files in os.walk(base_dir, topdown=False):
    for name in files:
        filepath = os.path.join(root, name)
        try:
            content = file_utils.read_all_content(filepath)
            if str in content:
                print(filepath)
        except:
            print('error: %s' % filepath)