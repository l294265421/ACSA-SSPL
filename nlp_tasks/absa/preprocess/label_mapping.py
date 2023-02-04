# 情感
from nlp_tasks.absa.conf import datasets, task_conf

sentiment_value_mapping = datasets.dataset_name_and_label_mapping[task_conf.current_dataset]['sentiment_value_mapping']

sentiment_value_mapping_reverse = {item[1]: item[0] for item in sentiment_value_mapping.items()}

sentiment_value_mapping_list = [sentiment_value_mapping_reverse[str(i)] for i in range(len(sentiment_value_mapping_reverse))]

subject_mapping = datasets.dataset_name_and_label_mapping[task_conf.current_dataset]['subject_mapping']

subject_mapping_reverse = {item[1]: item[0] for item in subject_mapping.items()}

subject_mapping_list = [subject_mapping_reverse[str(i)] for i in range(len(subject_mapping_reverse))]

subject_sentiment_value_mapping = {}
subject_sentiment_value_mapping_reverse = {}
subject_sentiment_value_mapping_list = []
for subject_index in subject_mapping_reverse.keys():
    for sentiment_value_index in sentiment_value_mapping_reverse.keys():
        subject_sentiment_value_index = str(int(subject_index) * len(sentiment_value_mapping_reverse) + int(sentiment_value_index))
        subject_sentiment_value = subject_mapping_reverse[subject_index] + '_' + sentiment_value_mapping_reverse[sentiment_value_index]
        subject_sentiment_value_mapping_reverse[subject_sentiment_value_index] = subject_sentiment_value
        subject_sentiment_value_mapping[subject_sentiment_value] = subject_sentiment_value_index
        subject_sentiment_value_mapping_list.append(str(subject_sentiment_value_index) + '_' + subject_sentiment_value)
