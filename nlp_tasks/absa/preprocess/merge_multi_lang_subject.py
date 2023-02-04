from nlp_tasks.absa.conf import data_path
from nlp_tasks.absa.utils import file_utils

train_file_path = data_path.train_subject_word_file_path
val_file_path = data_path.val_subject_word_file_path
test_file_path = data_path.test_subject_word_file_path

for file_path in [train_file_path, val_file_path, test_file_path]:
    lines = file_utils.read_all_lines(file_path)
    lines_lang = file_utils.read_all_lines(file_path + '.en')
    lines_merge = []
    for i in range(len(lines)):
        line = lines[i]
        line_parts = line.split(',')
        line_lang = lines_lang[i]
        line_lang_parts = line_lang.split(',')
        line_parts[1] = line_lang_parts[1] + ' ' + line_parts[1]
        line_merge = ','.join(line_parts)
        lines_merge.append(line_merge)
    file_utils.write_lines(lines_merge, file_path + '.merge')
