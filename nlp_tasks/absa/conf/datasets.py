from nlp_tasks.absa.preprocess import segmentor
from nlp_tasks.absa.conf import task_conf


embedding_base_dir = 'D:\program\word-vector\\'
# embedding_base_dir = '/home/work/absa/data/word_vector/'
dataset_name_and_embedding_file_path = {
    'SemEval-2014-Task-4-REST': embedding_base_dir + 'glove.840B.300d.txt',
    'SemEval-141516-LARGE-REST': embedding_base_dir + 'glove.840B.300d.txt',
    'bdci2018': embedding_base_dir + '',
    'SemEval-2016-Task-5-REST-SB1': embedding_base_dir + 'glove.840B.300d.txt',
    'SemEval-2016-Task-5-REST-SB2': embedding_base_dir + 'glove.840B.300d.txt',
    'SemEval-2016-Task-5-REST-SB2-three': embedding_base_dir + 'glove.840B.300d.txt',
    'SemEval-2016-Task-5-LAPT-SB1': embedding_base_dir + 'glove.840B.300d.txt',
    'SemEval-2016-Task-5-LAPT-SB2': embedding_base_dir + 'glove.840B.300d.txt',
    'SemEval-2016-Task-5-CH-CAME-SB1': embedding_base_dir + '',
    'SemEval-2016-Task-5-CH-PHNS-SB1': embedding_base_dir + '',
}

dataset_name_and_label_mapping = {
    'SemEval-2014-Task-4-REST': {
        'sentiment_value_mapping': {'negative': '0',
                                    'neutral': '1',
                                    'positive': '2',
                                    'conflict': '3'
                                    },
        'subject_mapping': {'food': '0',
                            'service': '1',
                            'price': '2',
                            'ambience': '3',
                            'miscellaneous': '4',
                            },
    },
    'SemEval-141516-LARGE-REST': {
        'sentiment_value_mapping': {'negative': '0',
                                    'neutral': '1',
                                    'positive': '2',
                                    },
        'subject_mapping': {'food': '0',
                            'service': '1',
                            'misc': '2',
                            'restaurant': '3',
                            'ambience': '4',
                            'price': '5',
                            'drinks': '6',
                            'location': '7',
                            },
    },
    'SemEval-2016-Task-5-REST-SB1': {
        'sentiment_value_mapping': {'negative': '0',
                                    'neutral': '1',
                                    'positive': '2',
                                    # 'conflict': '3'
                                    },
        'subject_mapping': {'FOOD#QUALITY': '0',
                            'FOOD#STYLE_OPTIONS': '1',
                            'RESTAURANT#GENERAL': '2',
                            'SERVICE#GENERAL': '3',
                            'AMBIENCE#GENERAL': '4',
                            'DRINKS#STYLE_OPTIONS': '5',
                            'FOOD#PRICES': '6',
                            'RESTAURANT#PRICES': '7',
                            'LOCATION#GENERAL': '8',
                            'DRINKS#QUALITY': '9',
                            'RESTAURANT#MISCELLANEOUS': '10',
                            'DRINKS#PRICES': '11',
                            },
    },
    'SemEval-2016-Task-5-REST-SB2': {
        'sentiment_value_mapping': {'negative': '0',
                                    'neutral': '1',
                                    'positive': '2',
                                    'conflict': '3'
                                    },
        'subject_mapping': {'FOOD#QUALITY': '0',
                            'FOOD#STYLE_OPTIONS': '1',
                            'RESTAURANT#GENERAL': '2',
                            'SERVICE#GENERAL': '3',
                            'AMBIENCE#GENERAL': '4',
                            'DRINKS#STYLE_OPTIONS': '5',
                            'FOOD#PRICES': '6',
                            'RESTAURANT#PRICES': '7',
                            'LOCATION#GENERAL': '8',
                            'DRINKS#QUALITY': '9',
                            'RESTAURANT#MISCELLANEOUS': '10',
                            'DRINKS#PRICES': '11',
                            },
    },
    'SemEval-2016-Task-5-REST-SB2-three': {
        'sentiment_value_mapping': {'negative': '0',
                                    'neutral': '1',
                                    'positive': '2',
                                    # 'conflict': '3'
                                    },
        'subject_mapping': {'FOOD#QUALITY': '0',
                            'FOOD#STYLE_OPTIONS': '1',
                            'RESTAURANT#GENERAL': '2',
                            'SERVICE#GENERAL': '3',
                            'AMBIENCE#GENERAL': '4',
                            'DRINKS#STYLE_OPTIONS': '5',
                            'FOOD#PRICES': '6',
                            'RESTAURANT#PRICES': '7',
                            'LOCATION#GENERAL': '8',
                            'DRINKS#QUALITY': '9',
                            'RESTAURANT#MISCELLANEOUS': '10',
                            'DRINKS#PRICES': '11',
                            },
    },
    'SemEval-2016-Task-5-LAPT-SB1': {
        'sentiment_value_mapping': {'negative': '0',
                                    'neutral': '1',
                                    'positive': '2',
                                    # 'conflict': '3'
                                    },
        'subject_mapping': {'LAPTOP#GENERAL': '0', 'BATTERY#OPERATION_PERFORMANCE': '1', 'CPU#OPERATION_PERFORMANCE': '2', 'GRAPHICS#GENERAL': '3', 'HARD_DISC#DESIGN_FEATURES': '4', 'LAPTOP#OPERATION_PERFORMANCE': '5', 'LAPTOP#USABILITY': '6', 'LAPTOP#PORTABILITY': '7', 'LAPTOP#PRICE': '8', 'LAPTOP#QUALITY': '9', 'LAPTOP#DESIGN_FEATURES': '10', 'LAPTOP#MISCELLANEOUS': '11', 'OS#DESIGN_FEATURES': '12', 'BATTERY#QUALITY': '13', 'SUPPORT#QUALITY': '14', 'COMPANY#GENERAL': '15', 'DISPLAY#USABILITY': '16', 'DISPLAY#DESIGN_FEATURES': '17', 'OS#GENERAL': '18', 'LAPTOP#CONNECTIVITY': '19', 'DISPLAY#QUALITY': '20', 'OS#USABILITY': '21', 'MOUSE#DESIGN_FEATURES': '22', 'SOFTWARE#MISCELLANEOUS': '23', 'KEYBOARD#DESIGN_FEATURES': '24', 'KEYBOARD#GENERAL': '25', 'SOFTWARE#GENERAL': '26', 'OPTICAL_DRIVES#QUALITY': '27', 'GRAPHICS#QUALITY': '28', 'HARD_DISC#QUALITY': '29', 'WARRANTY#GENERAL': '30', 'MOUSE#QUALITY': '31', 'MULTIMEDIA_DEVICES#USABILITY': '32', 'MULTIMEDIA_DEVICES#QUALITY': '33', 'PORTS#QUALITY': '34', 'DISPLAY#GENERAL': '35', 'POWER_SUPPLY#QUALITY': '36', 'POWER_SUPPLY#OPERATION_PERFORMANCE': '37', 'HARDWARE#QUALITY': '38', 'DISPLAY#OPERATION_PERFORMANCE': '39', 'MULTIMEDIA_DEVICES#GENERAL': '40', 'MULTIMEDIA_DEVICES#DESIGN_FEATURES': '41', 'KEYBOARD#USABILITY': '42', 'KEYBOARD#QUALITY': '43', 'POWER_SUPPLY#DESIGN_FEATURES': '44', 'SHIPPING#QUALITY': '45', 'MOUSE#OPERATION_PERFORMANCE': '46', 'MOUSE#USABILITY': '47', 'OS#MISCELLANEOUS': '48', 'SHIPPING#PRICE': '49', 'KEYBOARD#OPERATION_PERFORMANCE': '50', 'SUPPORT#MISCELLANEOUS': '51', 'CPU#QUALITY': '52', 'GRAPHICS#DESIGN_FEATURES': '53', 'OS#OPERATION_PERFORMANCE': '54', 'MEMORY#DESIGN_FEATURES': '55', 'SOFTWARE#QUALITY': '56', 'SOFTWARE#USABILITY': '57', 'SOFTWARE#DESIGN_FEATURES': '58', 'OS#QUALITY': '59', 'SOFTWARE#OPERATION_PERFORMANCE': '60', 'PORTS#DESIGN_FEATURES': '61', 'CPU#MISCELLANEOUS': '62', 'MOTHERBOARD#QUALITY': '63', 'SOFTWARE#PRICE': '64', 'MOUSE#GENERAL': '65', 'MULTIMEDIA_DEVICES#OPERATION_PERFORMANCE': '66', 'SUPPORT#PRICE': '67', 'WARRANTY#PRICE': '68', 'FANS_COOLING#DESIGN_FEATURES': '69', 'MULTIMEDIA_DEVICES#MISCELLANEOUS': '70', 'FANS_COOLING#QUALITY': '71', 'POWER_SUPPLY#MISCELLANEOUS': '72', 'PORTS#OPERATION_PERFORMANCE': '73', 'GRAPHICS#MISCELLANEOUS': '74', 'FANS_COOLING#OPERATION_PERFORMANCE': '75', 'BATTERY#MISCELLANEOUS': '76', 'HARDWARE#GENERAL': '77', 'OPTICAL_DRIVES#OPERATION_PERFORMANCE': '78', 'HARDWARE#OPERATION_PERFORMANCE': '79', 'CPU#DESIGN_FEATURES': '80', 'POWER_SUPPLY#GENERAL': '81', 'OPTICAL_DRIVES#DESIGN_FEATURES': '82', 'HARD_DISC#GENERAL': '83', 'OPTICAL_DRIVES#GENERAL': '84', 'HARD_DISC#OPERATION_PERFORMANCE': '85', 'BATTERY#DESIGN_FEATURES': '86', 'CPU#GENERAL': '87'},
    },
    'SemEval-2016-Task-5-LAPT-SB2': {
        'sentiment_value_mapping': {'negative': '0',
                                    'neutral': '1',
                                    'positive': '2',
                                    'conflict': '3'
                                    },
        'subject_mapping': {'LAPTOP#GENERAL': '0', 'BATTERY#OPERATION_PERFORMANCE': '1', 'CPU#OPERATION_PERFORMANCE': '2', 'GRAPHICS#GENERAL': '3', 'HARD_DISC#DESIGN_FEATURES': '4', 'LAPTOP#OPERATION_PERFORMANCE': '5', 'LAPTOP#USABILITY': '6', 'LAPTOP#PORTABILITY': '7', 'LAPTOP#PRICE': '8', 'LAPTOP#QUALITY': '9', 'LAPTOP#DESIGN_FEATURES': '10', 'LAPTOP#MISCELLANEOUS': '11', 'OS#DESIGN_FEATURES': '12', 'BATTERY#QUALITY': '13', 'SUPPORT#QUALITY': '14', 'COMPANY#GENERAL': '15', 'DISPLAY#USABILITY': '16', 'DISPLAY#DESIGN_FEATURES': '17', 'OS#GENERAL': '18', 'LAPTOP#CONNECTIVITY': '19', 'DISPLAY#QUALITY': '20', 'OS#USABILITY': '21', 'MOUSE#DESIGN_FEATURES': '22', 'SOFTWARE#MISCELLANEOUS': '23', 'KEYBOARD#DESIGN_FEATURES': '24', 'KEYBOARD#GENERAL': '25', 'SOFTWARE#GENERAL': '26', 'OPTICAL_DRIVES#QUALITY': '27', 'GRAPHICS#QUALITY': '28', 'HARD_DISC#QUALITY': '29', 'WARRANTY#GENERAL': '30', 'MOUSE#QUALITY': '31', 'MULTIMEDIA_DEVICES#USABILITY': '32', 'MULTIMEDIA_DEVICES#QUALITY': '33', 'PORTS#QUALITY': '34', 'DISPLAY#GENERAL': '35', 'POWER_SUPPLY#QUALITY': '36', 'POWER_SUPPLY#OPERATION_PERFORMANCE': '37', 'HARDWARE#QUALITY': '38', 'DISPLAY#OPERATION_PERFORMANCE': '39', 'MULTIMEDIA_DEVICES#GENERAL': '40', 'MULTIMEDIA_DEVICES#DESIGN_FEATURES': '41', 'KEYBOARD#USABILITY': '42', 'KEYBOARD#QUALITY': '43', 'POWER_SUPPLY#DESIGN_FEATURES': '44', 'SHIPPING#QUALITY': '45', 'MOUSE#OPERATION_PERFORMANCE': '46', 'MOUSE#USABILITY': '47', 'OS#MISCELLANEOUS': '48', 'SHIPPING#PRICE': '49', 'KEYBOARD#OPERATION_PERFORMANCE': '50', 'SUPPORT#MISCELLANEOUS': '51', 'CPU#QUALITY': '52', 'GRAPHICS#DESIGN_FEATURES': '53', 'OS#OPERATION_PERFORMANCE': '54', 'MEMORY#DESIGN_FEATURES': '55', 'SOFTWARE#QUALITY': '56', 'SOFTWARE#USABILITY': '57', 'SOFTWARE#DESIGN_FEATURES': '58', 'OS#QUALITY': '59', 'SOFTWARE#OPERATION_PERFORMANCE': '60', 'PORTS#DESIGN_FEATURES': '61', 'CPU#MISCELLANEOUS': '62', 'MOTHERBOARD#QUALITY': '63', 'SOFTWARE#PRICE': '64', 'MOUSE#GENERAL': '65', 'MULTIMEDIA_DEVICES#OPERATION_PERFORMANCE': '66', 'SUPPORT#PRICE': '67', 'WARRANTY#PRICE': '68', 'FANS_COOLING#DESIGN_FEATURES': '69', 'MULTIMEDIA_DEVICES#MISCELLANEOUS': '70', 'FANS_COOLING#QUALITY': '71', 'POWER_SUPPLY#MISCELLANEOUS': '72', 'PORTS#OPERATION_PERFORMANCE': '73', 'GRAPHICS#MISCELLANEOUS': '74', 'FANS_COOLING#OPERATION_PERFORMANCE': '75', 'BATTERY#MISCELLANEOUS': '76', 'HARDWARE#GENERAL': '77', 'OPTICAL_DRIVES#OPERATION_PERFORMANCE': '78', 'HARDWARE#OPERATION_PERFORMANCE': '79', 'CPU#DESIGN_FEATURES': '80', 'POWER_SUPPLY#GENERAL': '81', 'OPTICAL_DRIVES#DESIGN_FEATURES': '82', 'HARD_DISC#GENERAL': '83', 'OPTICAL_DRIVES#GENERAL': '84', 'HARD_DISC#OPERATION_PERFORMANCE': '85', 'BATTERY#DESIGN_FEATURES': '86', 'CPU#GENERAL': '87'},
    },
    'bdci2018': {
        'sentiment_value_mapping': {'-1': '0',
                                    '0': '1',
                                    '1': '2'
                                    },
        'subject_mapping': {'价格': '0',
                            '配置': '1',
                            '操控': '2',
                            '舒适性': '3',
                            '油耗': '4',
                            '动力': '5',
                            '内饰': '6',
                            '安全性': '7',
                            '空间': '8',
                            '外观': '9'
                            },
    },
    'SemEval-2016-Task-5-CH-CAME-SB1': {
        'sentiment_value_mapping': {'negative': '0',
                                    'positive': '1',
                                    },
        'subject_mapping': {'CAMERA#DESIGN_FEATURES': '0', 'CAMERA#OPERATION_PERFORMANCE': '1', 'HARDWARE#USABILITY': '2', 'SOFTWARE#DESIGN_FEATURES': '3', 'LENS#GENERAL': '4', 'LENS#OPERATION_PERFORMANCE': '5', 'BATTERY#QUALITY': '6', 'CAMERA#PORTABILITY': '7', 'PHOTO#QUALITY': '8', 'CAMERA#QUALITY': '9', 'FOCUS#OPERATION_PERFORMANCE': '10', 'HARDWARE#DESIGN_FEATURES': '11', 'SOFTWARE#OPERATION_PERFORMANCE': '12', 'LENS#DESIGN_FEATURES': '13', 'CAMERA#GENERAL': '14', 'HARDWARE#GENERAL': '15', 'HARDWARE#OPERATION_PERFORMANCE': '16', 'DISPLAY#DESIGN_FEATURES': '17', 'DISPLAY#OPERATION_PERFORMANCE': '18', 'CAMERA#PRICE': '19', 'DISPLAY#USABILITY': '20', 'PHOTO#GENERAL': '21', 'HARDWARE#QUALITY': '22', 'KEYBOARD#DESIGN_FEATURES': '23', 'HARDWARE#PRICE': '24', 'FOCUS#GENERAL': '25', 'SOFTWARE#QUALITY': '26', 'FOCUS#QUALITY': '27', 'FOCUS#DESIGN_FEATURES': '28', 'LENS#QUALITY': '29', 'FOCUS#USABILITY': '30', 'BATTERY#DESIGN_FEATURES': '31', 'CAMERA#USABILITY': '32', 'OS#OPERATION_PERFORMANCE': '33', 'MEMORY#OPERATION_PERFORMANCE': '34', 'PORTS#OPERATION_PERFORMANCE': '35', 'BATTERY#USABILITY': '36', 'KEYBOARD#OPERATION_PERFORMANCE': '37', 'DISPLAY#QUALITY': '38', 'LENS#PORTABILITY': '39', 'MULTIMEDIA_DEVICES#CONNECTIVITY': '40', 'MULTIMEDIA_DEVICES#OPERATION_PERFORMANCE': '41', 'MULTIMEDIA_DEVICES#DESIGN_FEATURES': '42', 'MULTIMEDIA_DEVICES#USABILITY': '43', 'SOFTWARE#USABILITY': '44', 'OS#GENERAL': '45', 'PORTS#CONNECTIVITY': '46', 'DISPLAY#GENERAL': '47', 'PORTS#QUALITY': '48', 'CAMERA#MISCELLANEOUS': '49', 'MULTIMEDIA_DEVICES#QUALITY': '50', 'POWER_SUPPLY#USABILITY': '51', 'SOFTWARE#GENERAL': '52', 'POWER_SUPPLY#OPERATION_PERFORMANCE': '53', 'LENS#MISCELLANEOUS': '54', 'PHOTO#MISCELLANEOUS': '55', 'SOFTWARE#MISCELLANEOUS': '56', 'HARDWARE#MISCELLANEOUS': '57', 'FOCUS#MISCELLANEOUS': '58', 'BATTERY#OPERATION_PERFORMANCE': '59', 'MULTIMEDIA_DEVICES#GENERAL': '60', 'OS#QUALITY': '61', 'SOFTWARE#CONNECTIVITY': '62', 'PORTS#DESIGN_FEATURES': '63', 'LENS#PRICE': '64', 'LENS#USABILITY': '65', 'KEYBOARD#QUALITY': '66', 'CPU#OPERATION_PERFORMANCE': '67', 'DISPLAY#MISCELLANEOUS': '68', 'MEMORY#QUALITY': '69', 'HARDWARE#CONNECTIVITY': '70', 'OS#USABILITY': '71', 'MEMORY#DESIGN_FEATURES': '72', 'SUPPORT#GENERAL': '73', 'OS#DESIGN_FEATURES': '74'}
    },
    'SemEval-2016-Task-5-CH-PHNS-SB1': {
        'sentiment_value_mapping': {'negative': '0',
                                    'positive': '1',
                                    },
        'subject_mapping': {'PHONE#DESIGN_FEATURES': '0', 'DISPLAY#QUALITY': '1', 'PORTS#CONNECTIVITY': '2', 'CPU#OPERATION_PERFORMANCE': '3', 'MULTIMEDIA_DEVICES#DESIGN_FEATURES': '4', 'MULTIMEDIA_DEVICES#QUALITY': '5', 'POWER_SUPPLY#QUALITY': '6', 'SOFTWARE#DESIGN_FEATURES': '7', 'HARDWARE#DESIGN_FEATURES': '8', 'PHONE#QUALITY': '9', 'OS#OPERATION_PERFORMANCE': '10', 'PHONE#OPERATION_PERFORMANCE': '11', 'SOFTWARE#OPERATION_PERFORMANCE': '12', 'OS#DESIGN_FEATURES': '13', 'MULTIMEDIA_DEVICES#CONNECTIVITY': '14', 'BATTERY#QUALITY': '15', 'PHONE#GENERAL': '16', 'MULTIMEDIA_DEVICES#OPERATION_PERFORMANCE': '17', 'DISPLAY#DESIGN_FEATURES': '18', 'SUPPORT#USABILITY': '19', 'WARRANTY#OPERATION_PERFORMANCE': '20', 'BATTERY#DESIGN_FEATURES': '21', 'HARDWARE#QUALITY': '22', 'PHONE#CONNECTIVITY': '23', 'KEYBOARD#DESIGN_FEATURES': '24', 'PHONE#PRICE': '25', 'OS#GENERAL': '26', 'HARDWARE#OPERATION_PERFORMANCE': '27', 'CPU#QUALITY': '28', 'MEMORY#QUALITY': '29', 'SOFTWARE#MISCELLANEOUS': '30', 'POWER_SUPPLY#OPERATION_PERFORMANCE': '31', 'BATTERY#OPERATION_PERFORMANCE': '32', 'PORTS#QUALITY': '33', 'DISPLAY#OPERATION_PERFORMANCE': '34', 'OS#USABILITY': '35', 'KEYBOARD#OPERATION_PERFORMANCE': '36', 'SOFTWARE#QUALITY': '37', 'HARD_DISC#QUALITY': '38', 'MULTIMEDIA_DEVICES#GENERAL': '39', 'KEYBOARD#USABILITY': '40', 'OS#CONNECTIVITY': '41', 'SOFTWARE#CONNECTIVITY': '42', 'HARD_DISC#OPERATION_PERFORMANCE': '43', 'HARDWARE#CONNECTIVITY': '44', 'POWER_SUPPLY#CONNECTIVITY': '45', 'PHONE#MISCELLANEOUS': '46', 'MULTIMEDIA_DEVICES#USABILITY': '47', 'DISPLAY#GENERAL': '48', 'SUPPORT#MISCELLANEOUS': '49', 'BATTERY#USABILITY': '50', 'HARDWARE#GENERAL': '51', 'HARDWARE#USABILITY': '52', 'POWER_SUPPLY#GENERAL': '53', 'PHONE#USABILITY': '54', 'DISPLAY#USABILITY': '55', 'MEMORY#USABILITY': '56', 'SOFTWARE#USABILITY': '57', 'PORTS#DESIGN_FEATURES': '58', 'POWER_SUPPLY#USABILITY': '59', 'KEYBOARD#QUALITY': '60', 'OS#QUALITY': '61', 'PORTS#USABILITY': '62', 'CPU#USABILITY': '63', 'PORTS#OPERATION_PERFORMANCE': '64', 'MEMORY#GENERAL': '65', 'CPU#GENERAL': '66', 'KEYBOARD#GENERAL': '67', 'MEMORY#DESIGN_FEATURES': '68', 'OS#PRICE': '69', 'SOFTWARE#GENERAL': '70', 'SUPPORT#QUALITY': '71', 'DISPLAY#MISCELLANEOUS': '72', 'PHONE#PORTABILITY': '73', 'POWER_SUPPLY#PRICE': '74', 'MULTIMEDIA_DEVICES#MISCELLANEOUS': '75', 'HARDWARE#MISCELLANEOUS': '76', 'HARDWARE#PRICE': '77', 'OS#MISCELLANEOUS': '78', 'BATTERY#GENERAL': '79', 'DISPLAY#PORTABILITY': '80'}
    }
}

dataset_name_and_segmentor = {
    'SemEval-2014-Task-4-REST': segmentor.nltk_segmentor,
    'SemEval-141516-LARGE-REST': segmentor.nltk_segmentor,
    'SemEval-2016-Task-5-REST-SB1': segmentor.nltk_segmentor,
    'SemEval-2016-Task-5-REST-SB2': segmentor.NltkSegmentor(remove_stop_words=True, remove_punctuations=True,
                                                            is_stem=False, lower=True),
    'SemEval-2016-Task-5-REST-SB2-three': segmentor.NltkSegmentor(remove_stop_words=True, remove_punctuations=True,
                                                                  is_stem=False, lower=True),
    'SemEval-2016-Task-5-LAPT-SB1': segmentor.nltk_segmentor,
    'SemEval-2016-Task-5-LAPT-SB2': segmentor.NltkSegmentor(remove_stop_words=True, remove_punctuations=True,
                                                            is_stem=False, lower=True),
    'bdci2018': segmentor.jieba_segmentor,
    'SemEval-2016-Task-5-CH-CAME-SB1': segmentor.jieba_segmentor,
    'SemEval-2016-Task-5-CH-PHNS-SB1': segmentor.jieba_segmentor,
}

dataset_name_and_thresholds = {
    'SemEval-2014-Task-4-REST': [0.5] * task_conf.subject_class_num,
    'SemEval-141516-LARGE-REST': [0.5] * task_conf.subject_class_num,
    'SemEval-2016-Task-5-REST-SB1': [0.5] * task_conf.subject_class_num,
    'SemEval-2016-Task-5-REST-SB2': [0.25] * task_conf.subject_class_num,
    'SemEval-2016-Task-5-REST-SB2-three': [0.25] * task_conf.subject_class_num,
    'SemEval-2016-Task-5-LAPT-SB1': [0.5] * task_conf.subject_class_num,
    'SemEval-2016-Task-5-LAPT-SB2': [0.25] * task_conf.subject_class_num,
    'bdci2018': [0.5] * task_conf.subject_class_num,
    'SemEval-2016-Task-5-CH-CAME-SB1': [0.25] * task_conf.subject_class_num,
    'SemEval-2016-Task-5-CH-PHNS-SB1': [0.25] * task_conf.subject_class_num,
}

dataset_name_and_maxlen = {
    'SemEval-2014-Task-4-REST': 79,
    'SemEval-141516-LARGE-REST': 1,
    'SemEval-2016-Task-5-REST-SB1': 39,
    'SemEval-2016-Task-5-REST-SB2': 307,
    'SemEval-2016-Task-5-REST-SB2-three': 307,
    'SemEval-2016-Task-5-LAPT-SB1': 37,
    'SemEval-2016-Task-5-LAPT-SB2': 516,
    'bdci2018': 171,
    'SemEval-2016-Task-5-CH-CAME-SB1': 44,
    'SemEval-2016-Task-5-CH-PHNS-SB1': 59,
}

dataset_name_and_batch_size = {
    'SemEval-2014-Task-4-REST': 32,
    'SemEval-141516-LARGE-REST': 1,
    'SemEval-2016-Task-5-REST-SB1': 32,
    'SemEval-2016-Task-5-REST-SB2': 10,
    'SemEval-2016-Task-5-REST-SB2-three': 10,
    'SemEval-2016-Task-5-LAPT-SB1': 32,
    'SemEval-2016-Task-5-LAPT-SB2': 10,
    'bdci2018': 32,
    'SemEval-2016-Task-5-CH-CAME-SB1': 32,
    'SemEval-2016-Task-5-CH-PHNS-SB1': 32,
}

dataset_name_and_loss_weight = {
    'SemEval-2014-Task-4-REST': [1.0, 1.0],
    'SemEval-141516-LARGE-REST': [1.0, 1.0],
    'SemEval-2016-Task-5-REST-SB1': [1.0, 1.0],
    'SemEval-2016-Task-5-REST-SB2': [1.0, 0.6],
    'SemEval-2016-Task-5-REST-SB2-three': [1.0, 0.6],
    'SemEval-2016-Task-5-LAPT-SB1': [1.0, 1.0],
    'SemEval-2016-Task-5-LAPT-SB2': [1.0, 1.0],
    'bdci2018': [1.0, 1.0],
    'SemEval-2016-Task-5-CH-CAME-SB1': [1.0, 1.0],
    'SemEval-2016-Task-5-CH-PHNS-SB1': [1.0, 1.0],
}

