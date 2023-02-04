from nlp_tasks.absa.conf import data_path
from nlp_tasks.absa.preprocess import label_mapping
from nlp_tasks.absa.utils import file_utils

test_file_path = data_path.test_public_for_sentiment_value_file_path
test_lines = file_utils.read_all_lines(test_file_path)
id_subjects = {}
id_commnet = {}
ids = []
for line in test_lines:
    parts = line.split(',')
    if parts[0] not in id_subjects:
        id_subjects[parts[0]] = set()
    id_subjects[parts[0]].add(parts[2])
    if parts[0] not in id_commnet:
        id_commnet[parts[0]] = parts[1]
        ids.append(parts[0])

for comment_id in ids:
    comment = id_commnet[comment_id]
    for subject in label_mapping.subject_mapping:
        if subject[:2] in comment and subject not in id_subjects[comment_id]:
            print(comment_id + ',' + comment + ',' + subject)
            file_utils.write_lines([comment_id + ',' + comment + ',' + subject],
                                   data_path.test_public_for_sentiment_value_exact_file_path, mode='a')
            # id_subjects[comment_id].add(subject)

# subject_predict_dir = data_path.data_base_dir + 'subject_predict/'
# subject_predict_filepath = subject_predict_dir + 'test_subject.result.keywords'
# for comment_id in ids:
#     subject_predict = '|'.join(id_subjects[comment_id])
#     file_utils.write_lines([comment_id + ',' + subject_predict], file_path=subject_predict_filepath,  mode='a')
