"""
16年的训练集就是由15年的训练集和测试集加在一起组成的，所以不考虑15年的数据
"""

import json

from nlp_tasks.absa.conf import data_path
from nlp_tasks.absa.utils import file_utils


def generate_samples(json_ojb: list):
    """

    :param json_ojb:
    :return:
    """
    result = []
    sentiment_count = {'positive': 0,
                       'negative': 0,
                       'neutral': 0}
    category_count = {}
    for i, element in enumerate(json_ojb):
        sentence_id = str(i)
        text = element['sentence']
        category = element['aspect']
        sentiment = element['sentiment']
        if category in category_count:
            category_count[category] += 1
        else:
            category_count[category] = 1
        sentiment_count[sentiment] += 1
        example = '\t'.join([sentence_id, text, category, sentiment])
        result.append(example)
    print(sentiment_count)
    print(category_count)
    return result

head = ['content_id\tcontent\tsubject\tsentiment_value']

train_data = json.load(open(data_path.data_base_dir + 'acsa_train.json'))
train_example = generate_samples(train_data)
file_utils.write_lines(head + train_example, data_path.data_base_dir + '/train.csv')

test_data = json.load(open(data_path.data_base_dir + 'acsa_test.json'))
test_examples = generate_samples(test_data)
file_utils.write_lines(head + test_examples, data_path.data_base_dir + '/test_public.csv')
file_utils.write_lines(head + test_examples, data_path.data_base_dir + '/test_public_gold.csv')

