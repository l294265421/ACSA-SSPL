"""
我们的联合模型
1. 解决的问题
(1) 现有的联合模型，都是通过共享底层表示来在acd和acsc之间交换信息，我们认为这是不够的，于是提出cae
(2) 现有的联合模型，不同属性不共享情感分类器，这在aspect category数据较少时，非常无效，于是我们共享情感分类器
"""
import argparse
import sys
import random

import torch
import numpy

from nlp_tasks.absa.utils import argument_utils
from nlp_tasks.absa.aspect_category_detection_and_sentiment_classification import acd_and_sc_train_templates_pytorch \
    as templates

# 接收参数
parser = argparse.ArgumentParser()
parser.add_argument('--current_dataset', help='dataset name', default='SemEval-2016-Task-5-LAPT-SB2', type=str)
# parser.add_argument('--hard_dataset', help='hard dataset name', default='', type=str)
parser.add_argument('--hard_test', help='extract hard test sets from test sets', default=True,
                    type=argument_utils.my_bool)
parser.add_argument('--task_name', help='task name', default='acd_and_sc', type=str)
parser.add_argument('--data_type', help='different dataset readers correspond to different data types', default='common', type=str)
parser.add_argument('--model_name', help='model name', default='CAE', type=str)
parser.add_argument('--timestamp', help='timestamp', default=int(1571400646), type=int)
parser.add_argument('--train', help='if train a new model', default=True, type=argument_utils.my_bool)
parser.add_argument('--evaluate', help='if evaluate the new model', default=True, type=argument_utils.my_bool)
parser.add_argument('--error_analysis', help='error analysis', default=False, type=argument_utils.my_bool)
parser.add_argument('--predict', default=True, type=argument_utils.my_bool)
parser.add_argument('--repeat', default=str(1), type=str)
parser.add_argument('--epochs', help='epochs', default=100, type=int)
parser.add_argument('--batch_size', help='batch_size', default=64, type=int)
parser.add_argument('--patience', help='patience', default=10, type=int)
parser.add_argument('--visualize_attention', help='visualize attention', default=False, type=argument_utils.my_bool)
parser.add_argument('--embedding_filepath', help='embedding filepath',
                    default='D:\program\word-vector\glove.840B.300d.txt', type=str)
parser.add_argument('--embed_size', help='embedding dimension', default=300, type=int)
parser.add_argument('--seed', default=776, type=int)
parser.add_argument('--device', default=None, type=str)
parser.add_argument('--gpu_id', default='0', type=str)
parser.add_argument('--only_acd', default=False, type=argument_utils.my_bool)
parser.add_argument('--token_min_padding_length', default=5, type=int)
parser.add_argument('--debug', default=False, type=argument_utils.my_bool)
parser.add_argument('--early_stopping_by_batch', default=False, type=argument_utils.my_bool)
parser.add_argument('--early_stopping', default=True, type=argument_utils.my_bool)
parser.add_argument('--early_stopping_metric', default='accuracy', type=str,
                    choices=['accuracy', 'category_f1', 'merge_micro_f1'])
parser.add_argument('--sparse_reg', default=False, type=argument_utils.my_bool)
parser.add_argument('--orthogonal_reg', default=False, type=argument_utils.my_bool)
parser.add_argument('--position', default=True, type=argument_utils.my_bool)
parser.add_argument('--position_embeddings_dim', help='position embeddings dim', default=64, type=int)
# varients
parser.add_argument('--not_sum', default=True, type=argument_utils.my_bool)
parser.add_argument('--lstm_or_fc_after_embedding_layer', default='lstm', type=str)
parser.add_argument('--acd_sc_mode', default='multi-multi', type=str, help='the acd task mode and the sc task '
                                                                           'mode in joint model',
                    choices=['multi-single', 'multi-multi'])
parser.add_argument('--joint_type', default='joint', type=str)
parser.add_argument('--pipeline', default=True, type=argument_utils.my_bool)
parser.add_argument('--acd_warmup_epochs', help='acd warmup epochs', default=10, type=int)
parser.add_argument('--acd_warmup_patience', help='acd warmup patience', default=4, type=int)
parser.add_argument('--acd_patience', help='acd patience', default=4, type=int)
parser.add_argument('--acd_warmup', help='acd warmup', default=False, type=argument_utils.my_bool)
parser.add_argument('--share_sentiment_classifier', help='share_sentiment_classifier', default=True,
                    type=argument_utils.my_bool)
# because the poor performance of cav
parser.add_argument('--merge_ae', help='merge ae and cav', default=True, type=argument_utils.my_bool)
parser.add_argument('--acd_init_weight', default=1, type=float)

parser.add_argument('--savefig_dir', help='dir to save pictures of visualizing model',
                    default='', type=str)

parser.add_argument('--max_word_len', help='max word length', default=512, type=int)

parser.add_argument('--frozen_all_acsc_parameter_while_pretrain_acd', default=False, type=argument_utils.my_bool)

args = parser.parse_args()

if args.joint_type == 'pipeline':
    args.acd_warmup = True
    args.pipeline = True
elif args.joint_type == 'warmup':
    args.acd_warmup = True
    args.pipeline = False
else:
    args.acd_warmup = False
    args.pipeline = False

args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if args.device is None else torch.device(args.device)
gpu_ids = args.gpu_id.split(',')
if len(gpu_ids) == 1:
    args.gpu_id = -1 if int(gpu_ids[0]) == -1 else 0
else:
    args.gpu_id = list(range(len(gpu_ids)))

configuration = args.__dict__

data_name = args.current_dataset

if configuration['seed'] is not None:
    random.seed(configuration['seed'])
    numpy.random.seed(configuration['seed'])
    torch.manual_seed(configuration['seed'])
    torch.cuda.manual_seed(configuration['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

configuration['model_name_complete'] = '%s-%s' % (args.model_name, args.repeat)

model_name = configuration['model_name']
if model_name == 'HeatCaeM':  # joint model
    template = templates.HeatCaeM(configuration)
elif model_name == 'HeatCae':
    template = templates.HeatCae(configuration)
elif model_name in ['CAE', 'CAE-att-only-in-lstm', 'CAE-att-only-in-embedding',
                    'CAE-add', 'CAE-average-of-two-layer', 'CAE-without-cae',
                    'CaeSupportingPipeline']:
    template = templates.Cae(configuration)
elif model_name in ['AOA']:
    template = templates.AOA(configuration)
elif model_name in ['ae-lstm-cae', 'at-lstm-cae', 'atae-lstm-cae']:  # joint model
    template = templates.AtaeLstmCae(configuration)
elif model_name in ['ae-lstm-m', 'at-lstm-m', 'atae-lstm-m']:  # joint model
    template = templates.AtaeLstmM(configuration)
elif model_name == 'CAE-pretrained-position':
    template = templates.CaePretrainedPosition(configuration)
elif model_name in ['CapsNet', 'CapsNetCav', 'CapsNetCam']:
    template = templates.CapsNetCae(configuration)
elif model_name == 'AS-Capsules':  # joint model
    template = templates.AsCapsules(configuration)
elif model_name == 'Can':  # joint model
    template = templates.Can(configuration)
elif model_name in ['End-to-end-LSTM', 'End-to-end-CNN']:  # joint model
    template = templates.EndToEnd(configuration)
# A Novel Aspect-Guided Deep Transition Model for Aspect Based Sentiment Analysis https://github.com/XL2248/AGDT
# attention监督，当句子里没有提及某个aspect category时，对应的attention应该关注没有意义的停用词；
# 提及某个aspect category时，对应的attention不要关注停用词

if configuration['train']:
    template.train()
if configuration['evaluate']:
    template.evaluate()
if configuration['error_analysis']:
    template.error_analysis()
if configuration['predict']:
    sentences = [['Great taste bad service.', [[2, 0], [4, 0]]]]
    result = template.predict(sentences)
    print(result)
