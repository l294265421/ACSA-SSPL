from nlp_tasks.absa.utils import file_utils
from nlp_tasks.absa.conf import data_path

lines = file_utils.read_all_lines(data_path.sentiment_dict_file_path)[1:]
sentiment_dict = {}
for line in lines:
    parts = line.split(',')
    sentiment_dict[parts[0]] = parts[6]