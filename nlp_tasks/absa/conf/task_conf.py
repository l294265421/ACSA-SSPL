import os

from nlp_tasks.absa.conf import model_output_type

dateset_name_and_data_featues = {
    'SemEval-2016-Task-5-LAPT-SB2': {
        'delimeter': '\t',
        'sentiment_class_num': 4,
        'subject_class_num': 88,
        'aspect_index': 2,
    },
    'SemEval-2016-Task-5-REST-SB2': {
        'delimeter': '\t',
        'sentiment_class_num': 4,
        'subject_class_num': 12,
        'aspect_index': 2,
    },
    'SemEval-2016-Task-5-REST-SB2-three': {
        'delimeter': '\t',
        'sentiment_class_num': 3,
        'subject_class_num': 12,
        'aspect_index': 2,
    },
    'SemEval-2016-Task-5-CH-CAME-SB1': {
        'delimeter': '\t',
        'sentiment_class_num': 2,
        'subject_class_num': 75,
        'aspect_index': 2,
    },
    'SemEval-2016-Task-5-CH-PHNS-SB1': {
        'delimeter': '\t',
        'sentiment_class_num': 2,
        'subject_class_num': 81,
        'aspect_index': 2,
    },
    'SemEval-2014-Task-4-REST': {
        'delimeter': '\t',
        'sentiment_class_num': 4,
        'subject_class_num': 5,
        'aspect_index': 2,
    },
    'bdci2018': {
        'delimeter': ',',
        'sentiment_class_num': 3,
        'subject_class_num': 10,
        'aspect_index': 2,
    },
}

current_dataset = 'SemEval-2016-Task-5-REST-SB2-three'

delimeter = dateset_name_and_data_featues[current_dataset]['delimeter']
sentiment_class_num = dateset_name_and_data_featues[current_dataset]['sentiment_class_num']
subject_class_num = dateset_name_and_data_featues[current_dataset]['subject_class_num']
aspect_index = dateset_name_and_data_featues[current_dataset]['aspect_index']

embed_size = 300
log_attention = False
# 0 ac-aoa
# 1 end_to_end_lstm
model_output_type = model_output_type.ac_aoa


def init(args):
    global current_dataset
    global delimeter
    global sentiment_class_num
    global subject_class_num
    global aspect_index
    global embed_size
    global log_attention
    global model_output_type
    current_dataset = args.current_dataset

    delimeter = dateset_name_and_data_featues[current_dataset]['delimeter']
    sentiment_class_num = dateset_name_and_data_featues[current_dataset]['sentiment_class_num']
    subject_class_num = dateset_name_and_data_featues[current_dataset]['subject_class_num']
    aspect_index = dateset_name_and_data_featues[current_dataset]['aspect_index']

    embed_size = args.embed_size
    log_attention = args.log_attention
    # 0 ac-aoa
    # 1 end_to_end_lstm
    model_output_type = args.model_output_type
