# -*- coding: utf-8 -*-
"""
把训练集拆分为训练集和验证集，验证集用于线下效果评估
Date:    2018/9/28 15:14
"""

import pandas as pd

from nlp_tasks.absa.conf import data_path
from nlp_tasks.absa.utils import file_utils


def merge_multi_label(lines):
    head = lines.pop(0)
    id_examples = {}
    for line in lines:
        parts = line.split('\t')
        if parts[2] not in id_examples:
            id_examples[parts[2]] = []
        id_examples[parts[2]].append(parts)
    examples = [head]
    for key, example in id_examples.items():
        topics = []
        for e in example:
            topics.append(e[1])
        examples.append(example[0][0] + '\t' + ','.join(topics) + '\t' + example[0][2])
    return examples


train_data = pd.read_csv(data_path.train_file_path)
train_subject_data = train_data[['content', 'subject', 'content_id']]
train_subject_data.to_csv(data_path.tf_bert_data + 'train.tsv', index=False, encoding='utf-8', sep='\t')

lines = file_utils.read_all_lines(data_path.tf_bert_data + 'train.tsv')
train_examples = merge_multi_label(lines)
file_utils.write_lines(train_examples, data_path.tf_bert_data + 'train.tsv')

val_data = pd.read_csv(data_path.val_file_path)
val_subject_data = val_data[['content', 'subject', 'content_id']]
val_subject_data.to_csv(data_path.tf_bert_data + 'dev.tsv', index=False, encoding='utf-8', sep='\t')

lines = file_utils.read_all_lines(data_path.tf_bert_data + 'dev.tsv')
train_examples = merge_multi_label(lines)
file_utils.write_lines(train_examples, data_path.tf_bert_data + 'dev.tsv')

test_data = pd.read_csv(data_path.test_public_file_path)
test_subject_data = test_data[['content', 'content_id']]
test_subject_data.to_csv(data_path.tf_bert_data + 'test.tsv', index=False, encoding='utf-8', sep='\t')
