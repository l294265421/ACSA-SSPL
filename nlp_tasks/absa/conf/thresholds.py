# -*- coding: utf-8 -*-
"""

Date:    2018/9/28 15:14
"""

from nlp_tasks.absa.conf import task_conf
from nlp_tasks.absa.conf import datasets

# topic_positive_threshold = [0.5] * task_conf.subject_class_num
# acaoa_semeval_2016_task_5_rest_sb2
# topic_positive_threshold = [0.25] * task_conf.subject_class_num
# acaoa_semeval_2016_task_5_lapt_sb2
# topic_positive_threshold = [0.25] * task_conf.subject_class_num
# # acaoa_semeval_2016_task_5_rest_sb1
# topic_positive_threshold = [0.4] * task_conf.subject_class_num
topic_positive_threshold = datasets.dataset_name_and_thresholds[task_conf.current_dataset]