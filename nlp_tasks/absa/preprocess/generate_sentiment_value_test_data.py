import pandas as pd

from nlp_tasks.absa.conf import data_path
from nlp_tasks.absa.utils import file_utils

if __name__ == '__main__':
    test_public_data = file_utils.read_all_lines(data_path.test_public_file_path)[1:]
    test_public_subject_result = file_utils.read_all_lines(data_path
                                                           .test_subject_result_file_path)
    merge_data = []
    for i in range(len(test_public_data)):
        one_test_public_data = test_public_data[i]
        one_test_public_subject_result = test_public_subject_result[i]
        subjects = one_test_public_subject_result.split(',')[1].split('|')
        for subject in subjects:
            merge_data.append(one_test_public_data + ',' + subject)
    file_utils.write_lines(merge_data, data_path.test_public_for_sentiment_value_file_path)