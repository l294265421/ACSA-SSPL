# -*- coding: utf-8 -*-
"""

Date:    2018/10/12 15:32
"""

from nlp_tasks.absa.conf import data_path
from nlp_tasks.absa.preprocess import label_mapping
from nlp_tasks.absa.utils import file_utils


def split_data_by_topic(file_path):
    lines = file_utils.read_all_lines(file_path)
    topics = label_mapping.subject_mapping.keys()
    for topic in topics:
        topic_lines = []
        for line in lines:
            parts = line.split(',')
            if topic == parts[2]:
                topic_lines.append(line)
        file_utils.write_lines(topic_lines, file_path + '.' + topic)


if __name__ == '__main__':
    split_data_by_topic(data_path.train_sentiment_value_exact_word_file_path)
    split_data_by_topic(data_path.val_sentiment_value_exact_word_file_path)
    split_data_by_topic(data_path.test_public_for_sentiment_value_exact_word_file_path)
    split_data_by_topic(data_path.external_sentiment_data_qiche)

