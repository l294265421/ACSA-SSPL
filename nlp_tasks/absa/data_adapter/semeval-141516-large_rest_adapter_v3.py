"""
16年的训练集就是由15年的训练集和测试集加在一起组成的，所以不考虑15年的数据
"""

import json

from nlp_tasks.absa.conf import data_path
from nlp_tasks.absa.utils import file_utils
from nlp_tasks.absa.gcae import getsemeval


def generate_examples(data: list):
    """

    :param data:
    :return:
    """
    result = []
    i = 0
    sentiment_count = {'positive': 0,
                       'negative': 0,
                       'neutral': 0}
    category_count = {}
    for example in data:
        sentence = example['sentence']
        for aspect, sentiment in example['aspect_sentiment']:
            sentiment_count[sentiment] += 1
            if aspect in category_count:
                category_count[aspect] += 1
            else:
                category_count[aspect] = 1
            result.append('\t'.join([str(i), sentence, aspect, sentiment]))
            i += 1
    print(sentiment_count)
    print(category_count)
    return result

years = [14, 15, 16]
semeval_train, semeval_test = getsemeval.get_semeval(years, None, 'r', False)

head = ['content_id\tcontent\tsubject\tsentiment_value']

train_example = generate_examples(semeval_train)
file_utils.write_lines(head + train_example, data_path.data_base_dir + '/train.csv')

test_examples = generate_examples(semeval_test)
file_utils.write_lines(head + test_examples, data_path.data_base_dir + '/test_public.csv')
file_utils.write_lines(head + test_examples, data_path.data_base_dir + '/test_public_gold.csv')

