import json
import numpy as np
import sys
import os
import collections
from typing import List

import torch
from sklearn.preprocessing import MultiLabelBinarizer
from allennlp.data.token_indexers import WordpieceIndexer
from allennlp.data.iterators import BucketIterator
from allennlp.data.iterators import BasicIterator
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders.bert_token_embedder import BertEmbedder, PretrainedBertEmbedder
from allennlp.modules.text_field_embedders import TextFieldEmbedder
import torch.optim as optim
from torch.optim import adagrad
# from allennlp.training.trainer import Trainer
from nlp_tasks.absa.aspect_category_detection_and_sentiment_classification.my_allennlp_trainer import Trainer
from allennlp.data.vocabulary import Vocabulary
from allennlp.predictors import text_classifier
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.dataset_readers import DatasetReader
import spacy
from pytorch_pretrained_bert.tokenization import BertTokenizer
from tqdm import tqdm
from allennlp.nn import util as nn_util
from allennlp.modules.token_embedders.bert_token_embedder import BertModel, PretrainedBertModel
from allennlp.data.instance import Instance
from nlp_tasks.absa.entities import ModelTrainTemplate
from nlp_tasks.utils import word_processor
from nlp_tasks.utils import tokenizers
from nlp_tasks.absa.preprocess import generate_word_index
from nlp_tasks.absa.utils import embedding_utils
from nlp_tasks.utils import attention_visualizer
from nlp_tasks.absa.aspect_category_detection_and_sentiment_classification import acd_and_sc_data_reader
from nlp_tasks.absa.aspect_category_detection_and_sentiment_classification import pytorch_models
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.token_embedders import embedding
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from nlp_tasks.utils import file_utils
from nlp_tasks.common import common_path
from nlp_tasks.bert_keras import tokenizer as bert_tokenizer
from nlp_tasks.absa.aspect_category_detection_and_sentiment_classification import allennlp_callback
from nlp_tasks.absa.data_adapter import data_object
from nlp_tasks.absa.data_adapter import mil_data


class TextInAllAspectSentimentOutTrainTemplate(ModelTrainTemplate.ModelTrainTemplate):
    """

    """

    def __init__(self, configuration):
        super().__init__(configuration)
        self.data_reader: DatasetReader = None
        self.train_data = None
        self.dev_data = None
        self.test_data = None
        self.hard_test_data = None
        self.distinct_categories: List[str] = None
        self.distinct_polarities: List[str] = None
        self._load_data()
        self._get_max_sentence_len()
        if self.configuration['debug']:
            self.train_data = self.train_data[: 128]
            self.dev_data = self.dev_data[: 128]
            self.test_data = self.test_data[: 128]

        self.vocab = None
        self._build_vocab()

        self.iterator = None
        self.val_iterator = None
        self._build_iterator()

        self.acd_model_dir = self.model_dir + 'acd/'

    def _get_max_sentence_len(self):
        len_count = collections.defaultdict(int)
        for data in [self.train_data, self.test_data, self.dev_data]:
            if data is None:
                continue
            for sample in data:
                tokens = sample.fields['tokens'].tokens
                # tokens = sample.fields['sample'].metadata[4]
                # if len(tokens) > self.configuration['max_len']:
                #     print(tokens)
                len_count[len(tokens)] += 1
        len_count_list = [[items[0], items[1]] for items in len_count.items()]
        len_count_list.sort(key=lambda x: x[0])
        self.logger.info('len_count_list: %s' % str(len_count_list))

    def _get_data_reader(self):
        token_indexer = SingleIdTokenIndexer(namespace="tokens")
        position_indexer = SingleIdTokenIndexer(namespace='position')
        aspect_indexer = SingleIdTokenIndexer(namespace='aspect')
        reader = acd_and_sc_data_reader.TextInAllAspectSentimentOut(
            self.distinct_categories, self.distinct_polarities,
            tokenizer=self._get_word_segmenter(),
            token_indexers={"tokens": token_indexer},
            position_indexers={'position': position_indexer},
            aspect_indexers={'aspect': aspect_indexer},
            configuration=self.configuration
        )
        return reader

    def _load_data(self):
        data_filepath = self.base_data_dir + 'data'
        if os.path.exists(data_filepath):
            self.train_data, self.dev_data, self.test_data, self.distinct_categories, self.distinct_polarities, \
            self.hard_test_data = super()._load_object(data_filepath)
            reader = self._get_data_reader()
            self.data_reader = reader
        else:
            train_dev_test_data, distinct_categories, distinct_polarities = self.dataset. \
                generate_acd_and_sc_data(dev_size=0.2)

            if self.configuration['hard_test']:
                train_dev_test_data['hard_test'] = []
                for sample in train_dev_test_data['test']:
                    polarities = set([e[1] for e in sample[1]])
                    if len(polarities) >= 2:
                        train_dev_test_data['hard_test'].append(sample)

            distinct_polarities_new = []
            for polarity in distinct_polarities:
                if polarity != 'conflict':
                    distinct_polarities_new.append(polarity)
            self.distinct_categories = distinct_categories
            self.distinct_polarities = distinct_polarities_new

            train_dev_test_data_label_indexed = {}
            for data_type, data in train_dev_test_data.items():
                if data is None:
                    continue
                data_new = []
                for sample in data:
                    sample_new = [sample[0]]
                    labels_new = []
                    for label in sample[1]:
                        aspect = label[0]
                        polarity = label[1]
                        aspect_index = distinct_categories.index(aspect)
                        if polarity == 'conflict':
                            polarity_index = -100
                        else:
                            polarity_index = distinct_polarities_new.index(polarity)
                        labels_new.append((aspect_index, polarity_index))
                    if len(labels_new) != 0:
                        sample_new.append(labels_new)
                        data_new.append(sample_new)
                train_dev_test_data_label_indexed[data_type] = data_new

            reader = self._get_data_reader()
            self.data_reader = reader
            self.train_data = reader.read(train_dev_test_data_label_indexed['train'])
            self.dev_data = reader.read(train_dev_test_data_label_indexed['dev'])
            self.test_data = reader.read(train_dev_test_data_label_indexed['test'])
            if self.configuration['hard_test']:
                self.hard_test_data = reader.read(train_dev_test_data_label_indexed['hard_test'])
            data = [self.train_data, self.dev_data, self.test_data, self.distinct_categories,
                    self.distinct_polarities, self.hard_test_data]
            super()._save_object(data_filepath, data)

    def _build_vocab(self):
        if self.configuration['train']:
            vocab_file_path = self.base_data_dir + 'vocab'
            if os.path.exists(vocab_file_path):
                self.vocab = super()._load_object(vocab_file_path)
            else:
                data = self.train_data + self.dev_data + self.test_data
                self.vocab = Vocabulary.from_instances(data, max_vocab_size=sys.maxsize)
                super()._save_object(vocab_file_path, self.vocab)
            self.model_meta_data['vocab'] = self.vocab
        else:
            self.vocab = self.model_meta_data['vocab']

    def _build_iterator(self):
        self.iterator = BucketIterator(batch_size=self.configuration['batch_size'],
                                       sorting_keys=[("tokens", "num_tokens")],
                                       )
        self.iterator.index_with(self.vocab)
        self.val_iterator = BasicIterator(batch_size=self.configuration['batch_size'])
        self.val_iterator.index_with(self.vocab)

    def _print_args(self, model):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in model.parameters():
            n_params = torch.prod(torch.tensor(p.shape)).item()
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        self.logger.info('n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        self.logger.info('> training arguments:')
        for arg in self.configuration.keys():
            self.logger.info('>>> {0}: {1}'.format(arg, self.configuration[arg]))

    def _find_model_function_pure(self):
        raise NotImplementedError('_find_model_function_pure')

    def _get_aspect_embeddings_dim(self):
        return 300

    def _get_position_embeddings_dim(self):
        return 300

    def _is_train_token_embeddings(self):
        return False

    def _find_model_function(self):
        embedding_dim = self.configuration['embed_size']
        embedding_matrix_filepath = self.base_data_dir + 'embedding_matrix'
        if os.path.exists(embedding_matrix_filepath):
            embedding_matrix = super()._load_object(embedding_matrix_filepath)
        else:
            embedding_filepath = self.configuration['embedding_filepath']
            embedding_matrix = embedding._read_embeddings_from_text_file(embedding_filepath, embedding_dim,
                                                                         self.vocab, namespace='tokens')
            super()._save_object(embedding_matrix_filepath, embedding_matrix)
        embedding_matrix = embedding_matrix.to(self.configuration['device'])
        token_embedding = Embedding(num_embeddings=self.vocab.get_vocab_size(namespace='tokens'),
                                    embedding_dim=embedding_dim, padding_index=0, vocab_namespace='tokens',
                                    trainable=self._is_train_token_embeddings(), weight=embedding_matrix)
        # the embedder maps the input tokens to the appropriate embedding matrix
        word_embedder: TextFieldEmbedder = BasicTextFieldEmbedder({"tokens": token_embedding})

        position_embedding = Embedding(num_embeddings=self.vocab.get_vocab_size(namespace='position'),
                                    embedding_dim=self._get_position_embeddings_dim(), padding_index=0)
        position_embedder: TextFieldEmbedder = BasicTextFieldEmbedder({"position": position_embedding},
                                                                    # we'll be ignoring masks so we'll need to set this to True
                                                                    allow_unmatched_keys=True)

        aspect_embedding = Embedding(num_embeddings=self.vocab.get_vocab_size(namespace='aspect'),
                                     embedding_dim=self._get_aspect_embeddings_dim(), padding_index=0)
        aspect_embedder: TextFieldEmbedder = BasicTextFieldEmbedder({"aspect": aspect_embedding},
                                                                    # we'll be ignoring masks so we'll need to set this to True
                                                                    allow_unmatched_keys=True)
        model_function: pytorch_models.TextInAllAspectSentimentOutModel = self._find_model_function_pure()
        model = model_function(
            word_embedder,
            position_embedder,
            aspect_embedder,
            self.distinct_categories,
            self.distinct_polarities,
            self.vocab,
            self.configuration,
        )
        self._print_args(model)
        model = model.to(self.configuration['device'])
        return model

    def _get_optimizer(self, model):
        _params = filter(lambda p: p.requires_grad, model.parameters())
        return optim.Adam(_params, lr=0.001, weight_decay=0.00001)

    def _get_acd_optimizer(self, model):
        _params = filter(lambda p: p.requires_grad, model.parameters())
        return optim.Adam(_params, lr=0.001, weight_decay=0.00001)

    def _get_acd_warmup_epoch_num(self):
        return 3

    def _get_estimator(self, model):
        USE_GPU = torch.cuda.is_available()
        if USE_GPU:
            gpu_id = self.configuration['gpu_id']
        else:
            gpu_id = -1
        estimator = pytorch_models.TextInAllAspectSentimentOutEstimator(model, self.val_iterator,
                                                                        self.distinct_categories,
                                                                        self.distinct_polarities,
                                                                        configuration=self.configuration,
                                                                        cuda_device=gpu_id)
        return estimator

    def _get_estimate_callback(self, model):
        result = []
        data_type_and_data = {
            'train': self.train_data,
            'dev': self.dev_data,
            'test': self.test_data
        }
        if self.hard_test_data:
            data_type_and_data['hard_test'] = self.hard_test_data
        estimator = self._get_estimator(model)
        estimate_callback = allennlp_callback.EstimateCallback(data_type_and_data, estimator, self.logger)
        result.append(estimate_callback)
        return result

    def _get_loss_weight_callback(self):
        result = []
        set_loss_weight_callback = allennlp_callback.SetLossWeightCallback(self.model, self.logger,
                                                                           acd_warmup_epoch_num=self._get_acd_warmup_epoch_num())
        result.append(set_loss_weight_callback)
        return result

    def _get_fixed_loss_weight_callback(self, model, category_loss_weight=1, sentiment_loss_weight=1):
        result = []
        fixed_loss_weight_callback = allennlp_callback.FixedLossWeightCallback(model, self.logger,
                                                                             category_loss_weight=category_loss_weight,
                                                                             sentiment_loss_weight=sentiment_loss_weight)
        result.append(fixed_loss_weight_callback)
        return result

    def _get_bert_word_embedder(self):
        return None

    def _inner_train(self):
        USE_GPU = torch.cuda.is_available()
        if USE_GPU:
            gpu_id = self.configuration['gpu_id']
        else:
            gpu_id = -1

        self.model: pytorch_models.TextInAllAspectSentimentOutModel = self._find_model_function()

        estimator = self._get_estimator(self.model)
        if self.configuration['acd_warmup']:
            if self.configuration['frozen_all_acsc_parameter_while_pretrain_acd']:
                self.model.set_grad_for_acsc_parameter(requires_grad=False)

            optimizer = self._get_acd_optimizer(self.model)
            self.logger.info('acd warmup')
            validation_metric = '+category_f1'
            callbacks = self._get_estimate_callback(self.model)
            callbacks.extend(self._get_fixed_loss_weight_callback(self.model, category_loss_weight=1, sentiment_loss_weight=0))
            self._print_args(self.model)
            trainer = Trainer(
                model=self.model,
                optimizer=optimizer,
                iterator=self.iterator,
                train_dataset=self.train_data,
                validation_dataset=self.dev_data,
                cuda_device=gpu_id,
                num_epochs=self.configuration['acd_warmup_epochs'],
                validation_metric=validation_metric,
                validation_iterator=self.val_iterator,
                serialization_dir=self.acd_model_dir,
                patience=None if self.configuration['acd_warmup_patience'] == -1 else self.configuration['acd_warmup_patience'],
                callbacks=callbacks,
                num_serialized_models_to_keep=2,
                early_stopping_by_batch=self.configuration['early_stopping_by_batch'],
                estimator=estimator,
                grad_clipping=5
            )
            metrics = trainer.train()
            self.logger.info('acd metrics: %s' % str(metrics))

            if self.configuration['frozen_all_acsc_parameter_while_pretrain_acd']:
                self.model.set_grad_for_acsc_parameter(requires_grad=True)
            # 恢复bert到初始状态
            if 'bert' in self.configuration and self.configuration['bert']:
                self.model.set_bert_word_embedder()
                bert_word_embedder = self._get_bert_word_embedder()
                self.model.set_bert_word_embedder(bert_word_embedder)

        if self.configuration['only_acd']:
            return None
        validation_metric = '+accuracy'
        if 'early_stopping_metric' in self.configuration:
            validation_metric = '+%s' % self.configuration['early_stopping_metric']
        callbacks = self._get_estimate_callback(self.model)
        if self.configuration['acd_warmup'] and self.configuration['pipeline']:
            callbacks.extend(self._get_fixed_loss_weight_callback(self.model, category_loss_weight=0, sentiment_loss_weight=1))
            # acd 相关的参数不更新
            self.model.no_grad_for_acd_parameter()
        else:
            callbacks.extend(self._get_fixed_loss_weight_callback(self.model,
                                                                  category_loss_weight=self.configuration['acd_init_weight'],
                                                                  sentiment_loss_weight=1))
        self.logger.info('validation_metric: %s' % validation_metric)
        optimizer = self._get_optimizer(self.model)
        self._print_args(self.model)
        trainer = Trainer(
            model=self.model,
            optimizer=optimizer,
            iterator=self.iterator,
            train_dataset=self.train_data,
            validation_dataset=self.dev_data if self.configuration['early_stopping'] else None,
            cuda_device=gpu_id,
            num_epochs=self.configuration['epochs'],
            validation_metric=validation_metric,
            validation_iterator=self.val_iterator,
            serialization_dir=self.model_dir,
            patience=self.configuration['patience'],
            callbacks=callbacks,
            num_serialized_models_to_keep=2,
            early_stopping_by_batch=self.configuration['early_stopping_by_batch'],
            estimator=estimator,
            grad_clipping=5
        )
        metrics = trainer.train()
        self.logger.info('metrics: %s' % str(metrics))

    def _save_model(self):
        torch.save(self.model, self.best_model_filepath)

    def _load_model(self):
        if torch.cuda.is_available():
            self.model = torch.load(self.best_model_filepath)
        else:
            self.model = torch.load(self.best_model_filepath, map_location=torch.device('cpu'))
        self.model.configuration = self.configuration

    def evaluate(self):
        USE_GPU = torch.cuda.is_available()
        if USE_GPU:
            gpu_id = self.configuration['gpu_id']
        else:
            gpu_id = -1
        estimator = pytorch_models.TextInAllAspectSentimentOutEstimator(self.model, self.val_iterator,
                                                                        self.distinct_categories,
                                                                        self.distinct_polarities,
                                                                        configuration=self.configuration,
                                                                        cuda_device=gpu_id)

        data_type_and_data = {
            # 'train': self.train_data,
            'dev': self.dev_data,
            'test': self.test_data
        }
        if self.hard_test_data:
            data_type_and_data['hard_test'] = self.hard_test_data
        if 'performance_of_different_lengths' in self.configuration:
            lengths = self.configuration['performance_of_different_lengths'].split(',')
            if len(lengths) > 1:
                data_of_different_lengths = {int(length): [] for length in lengths}
                for sample in data_type_and_data['test']:
                    tokens = sample.fields['tokens'].tokens
                    for length in data_of_different_lengths:
                        if len(tokens) <= length:
                            data_of_different_lengths[length].append(sample)
                for length, data in data_of_different_lengths.items():
                    if len(data) > 0:
                        data_type_and_data['test_%d' % length] = data
        for data_type, data in data_type_and_data.items():
            result = estimator.estimate(data)
            self.logger.info('data_type: %s result: %s' % (data_type, result))

    def evaluation_on_instance_level(self):
        USE_GPU = torch.cuda.is_available()
        if USE_GPU:
            gpu_id = self.configuration['gpu_id']
        else:
            gpu_id = -1

        dataset_name = self.configuration['current_dataset']
        if dataset_name == 'SemEval-2014-Task-4-REST-DevSplits':
            # mil = mil_data.SemEval2014Task4RESTMil()
            mil = mil_data.SemEval2014Task4RESTHardMil()
        elif dataset_name == 'MAMSACSA':
            mil = mil_data.MAMSACSAMil()
        else:
            raise NotImplementedError('don\'t support evaluate performance on instance for the dataset %s'
                                      % dataset_name)
        samples = mil.load_samples()
        texts = []
        for sample in samples:
            text = sample.text
            labels = [[self.distinct_categories.index(category), 0] for category in sample.categories]
            texts.append([text, labels])
        predictor = pytorch_models.TextInAllAspectSentimentOutPredictorOnInstanceLevel(self.model, self.val_iterator,
                                                                        self.distinct_categories,
                                                                        self.distinct_polarities,
                                                                        configuration=self.configuration,
                                                                        cuda_device=gpu_id)

        data = self.data_reader.read(texts)
        result = predictor.predict(data)
        # 1. 关键实例发现
        # (1) 模型发现的关键实例，即attention weight大于指定阈值的词，
        # (2) 怎么匹配实际的关键实例和预测的关键实例
        # 2. 关键实例情感分类
        # (1) 当关键实例存在多个词时，它的预测的情感怎么计算？
        tokenizer = self._get_word_segmenter()
        correct_sentiment_num = 0
        total_sentiment_num = 0
        tp_for_key_instance = 0
        fp_for_key_instance = 0
        fn_for_key_instance = 0
        for i in range(len(samples)):
            sample = samples[i]
            words = data[i].fields['tokens'].tokens
            key_instances_true = collections.defaultdict(list)
            for key_instance in sample.key_instances:
                text_before_key_instance = sample.text[: key_instance.from_index]
                words_before_key_instance = tokenizer(text_before_key_instance)
                words_of_key_instance = tokenizer(key_instance.text)
                for j in range(len(words_of_key_instance)):
                    key_instances_true[key_instance.category].append({'word': words_of_key_instance[j],
                                                                  'index': len(words_before_key_instance) + j,
                                                                      'polarity': key_instance.polarity})
            # 将key instance中的词赋上在句子中词index
            attention_weights = result[i]['attention_weights']
            categories = sample.categories
            key_instances_pred = collections.defaultdict(list)
            for category in categories:
                category_index = self.distinct_categories.index(category)
                attention_weights_of_this_category = attention_weights[category_index][: len(words)]
                for j in range(len(words)):
                    word = words[j]
                    weight = attention_weights_of_this_category[j]
                    if weight >= 0.1:
                        key_instances_pred[category].append({'word': word, 'index': j})

            key_instances_true_str = set(['%s-%d' % (e['word'], e['index']) for key_instances_of_a_category in key_instances_true.values() for e in key_instances_of_a_category])
            key_instances_pred_str = set(['%s-%d' % (e['word'], e['index']) for key_instances_of_a_category in key_instances_pred.values() for e in key_instances_of_a_category])
            tp_instances = key_instances_true_str & key_instances_pred_str
            fp_instances = key_instances_pred_str.difference(key_instances_true_str)
            fn_instances = key_instances_true_str.difference(key_instances_pred_str)

            tp_for_key_instance += len(tp_instances)
            fp_for_key_instance += len(fp_instances)
            fn_for_key_instance += len(fn_instances)

            word_sentiments = result[i]['word_sentiments']
            for category, key_instances in key_instances_true.items():
                category_index = self.distinct_categories.index(category)
                word_sentiments_of_this_category = word_sentiments[category_index]
                for key_instance in key_instances:
                    total_sentiment_num += 1
                    word, index = key_instance['word'], key_instance['index']
                    word_sentiment = word_sentiments_of_this_category[index]
                    polarity_index = np.argmax(word_sentiment)
                    word_polarity = self.distinct_polarities[polarity_index]
                    if key_instance['polarity'] == word_polarity:
                        correct_sentiment_num += 1

        self.logger.info('tp_for_key_instance: %d fp_for_key_instance: %d fn_for_key_instance: %d' %
                         (tp_for_key_instance, fp_for_key_instance, fn_for_key_instance))
        precision = tp_for_key_instance / (tp_for_key_instance + fp_for_key_instance)
        recall = tp_for_key_instance / (tp_for_key_instance + fn_for_key_instance)
        f1 = 2 * (precision * recall) / (precision + recall)
        self.logger.info('precision: %.5f recall: %.5f f1: %.5f' % (precision, recall, f1))
        self.logger.info('correct_sentiment_num: %d total_sentiment_num: %d sentiment_acc: %.5f' %
                         (correct_sentiment_num, total_sentiment_num, correct_sentiment_num / total_sentiment_num))
        print()

    def predict(self, texts: List[str]=None):
        """

        :param texts: 如果texts为None，就是用训练时的测试集
        :return:
        """
        USE_GPU = torch.cuda.is_available()
        if USE_GPU:
            gpu_id = self.configuration['gpu_id']
        else:
            gpu_id = -1
        predictor = pytorch_models.TextInAllAspectSentimentOutPredictor(self.model, self.val_iterator,
                                                                        self.distinct_categories,
                                                                        self.distinct_polarities,
                                                                        configuration=self.configuration,
                                                                        cuda_device=gpu_id)

        data = self.data_reader.read(texts)
        result = predictor.predict(data)
        return result

    def error_analysis_backup(self):
        """

        :return:
        """
        USE_GPU = torch.cuda.is_available()
        if USE_GPU:
            gpu_id = self.configuration['gpu_id']
        else:
            gpu_id = -1
        predictor = pytorch_models.TextInAllAspectSentimentOutPredictor(self.model, self.val_iterator,
                                                                        self.distinct_categories,
                                                                        self.distinct_polarities,
                                                                        configuration=self.configuration,
                                                                        cuda_device=gpu_id)

        data = self.test_data
        result = predictor.predict(data)
        result_final = []
        for i in range(len(data)):
            instance: Instance = data[i]
            metadata = instance.fields['sample'].metadata
            sentence = metadata[0]
            labels_true = {self.distinct_categories[e[0]]: self.distinct_polarities[e[1]] for e in metadata[1]}
            labels_pred = result[i]
            for label_pred in labels_pred:
                label_true = labels_true[label_pred[0]]
                if label_true == label_pred[1]:
                    continue
                result_final.append((sentence, label_pred[0], label_pred[1], label_true))
        result_str = ['\t'.join(e) for e in result_final]
        output_filepath = os.path.join(self.model_dir, 'error_analysis.csv')
        file_utils.write_lines(result_str, output_filepath)
        return result_final

    def error_analysis(self):
        """

        :return:
        """
        USE_GPU = torch.cuda.is_available()
        if USE_GPU:
            gpu_id = self.configuration['gpu_id']
        else:
            gpu_id = -1
        predictor = pytorch_models.TextInAllAspectSentimentOutPredictor(self.model, self.val_iterator,
                                                                        self.distinct_categories,
                                                                        self.distinct_polarities,
                                                                        configuration=self.configuration,
                                                                        cuda_device=gpu_id)

        data = self.test_data
        result = predictor.predict(data)
        result_final = []
        for i in range(len(data)):
            instance: Instance = data[i]
            metadata = instance.fields['sample'].metadata
            sentence = metadata['text']
            labels_true = {self.distinct_categories[e[0]]: self.distinct_polarities[e[1]] for e in metadata['labels']}
            labels_pred = result[i]
            for label_pred in labels_pred:
                label_true = labels_true[label_pred[0]]
                if label_true == label_pred[1]:
                    continue
                result_final.append((sentence, label_pred[0], label_pred[1], label_true))
        result_str = ['\t'.join(e) for e in result_final]
        output_filepath = os.path.join(self.model_dir, 'error_analysis.csv')
        file_utils.write_lines(result_str, output_filepath)
        return result_final


class AsMil(TextInAllAspectSentimentOutTrainTemplate):
    """

    """

    def __init__(self, configuration):
        super().__init__(configuration)

    def _get_optimizer(self, model):
        # _params = filter(lambda p: p.requires_grad, model.parameters())
        # return optim.Adam(_params, lr=0.001)

        _params = filter(lambda p: p.requires_grad, model.parameters())
        return optim.Adam(_params, lr=0.001, weight_decay=0.00001)

    def _get_position_embeddings_dim(self):
        return self.configuration['position_embeddings_dim']

    def _find_model_function_pure(self):
        return pytorch_models.AsMilSimultaneouslyV5


class BaseSentenceConsituencyAwareModel(TextInAllAspectSentimentOutTrainTemplate):
    """

    """

    def __init__(self, configuration):
        super().__init__(configuration)

    def _inner_train(self):
        USE_GPU = torch.cuda.is_available()
        if USE_GPU:
            gpu_id = self.configuration['gpu_id']
        else:
            gpu_id = -1

        self.model: pytorch_models.TextInAllAspectSentimentOutModel = self._find_model_function()

        estimator = self._get_estimator(self.model)
        if self.configuration['acd_warmup']:
            if self.configuration['frozen_all_acsc_parameter_while_pretrain_acd']:
                self.model.set_grad_for_acsc_parameter(requires_grad=False)

            optimizer = self._get_acd_optimizer(self.model)
            self.logger.info('acd warmup')
            validation_metric = '+accuracy'
            callbacks = self._get_estimate_callback(self.model)
            # warmup这一步也分两种，1，只是预训练属性分类，2，同时训练属性分类和情感分类，但是情感分类和属性分类用一样的
            # 表示，情感分类任务只用于调整attention向更上层转移
            if self.configuration['attention_warmup_init']:
                self.configuration['attention_warmup'] = True
                callbacks.extend(self._get_fixed_loss_weight_callback(self.model, category_loss_weight=1,
                                                                      sentiment_loss_weight=1))
            else:
                callbacks.extend(self._get_fixed_loss_weight_callback(self.model, category_loss_weight=1,
                                                                      sentiment_loss_weight=0))
            self._print_args(self.model)
            trainer = Trainer(
                model=self.model,
                optimizer=optimizer,
                iterator=self.iterator,
                train_dataset=self.train_data,
                validation_dataset=self.dev_data,
                cuda_device=gpu_id,
                num_epochs=self.configuration['acd_warmup_epochs'],
                validation_metric=validation_metric,
                validation_iterator=self.val_iterator,
                serialization_dir=self.acd_model_dir,
                patience=None if self.configuration['acd_warmup_patience'] == -1 else self.configuration['acd_warmup_patience'],
                callbacks=callbacks,
                num_serialized_models_to_keep=2,
                early_stopping_by_batch=self.configuration['early_stopping_by_batch'],
                estimator=estimator,
                grad_clipping=5
            )
            metrics = trainer.train()
            self.logger.info('acd metrics: %s' % str(metrics))
            if self.configuration['attention_warmup_init']:
                self.configuration['attention_warmup'] = False
            if self.configuration['frozen_all_acsc_parameter_while_pretrain_acd']:
                self.model.set_grad_for_acsc_parameter(requires_grad=True)
            # 恢复bert到初始状态
            if 'bert' in self.configuration and self.configuration['bert']:
                self.model.set_bert_word_embedder()
                bert_word_embedder = self._get_bert_word_embedder()
                self.model.set_bert_word_embedder(bert_word_embedder)

        if self.configuration['only_acd']:
            return None
        validation_metric = '+accuracy'
        if 'early_stopping_metric' in self.configuration:
            validation_metric = self.configuration['early_stopping_metric']
        callbacks = self._get_estimate_callback(self.model)
        if self.configuration['acd_warmup'] and self.configuration['pipeline']:
            # acd 相关的参数不更新
            self.model.no_grad_for_acd_parameter()
            if self.configuration['pipeline_with_acd']:
                callbacks.extend(self._get_fixed_loss_weight_callback(self.model, category_loss_weight=1,
                                                                      sentiment_loss_weight=1))
            else:
                callbacks.extend(self._get_fixed_loss_weight_callback(self.model, category_loss_weight=0,
                                                                      sentiment_loss_weight=1))
        else:
            callbacks.extend(self._get_fixed_loss_weight_callback(self.model,
                                                                  category_loss_weight=self.configuration['acd_init_weight'],
                                                                  sentiment_loss_weight=1))
        self.logger.info('validation_metric: %s' % validation_metric)
        optimizer = self._get_optimizer(self.model)
        self._print_args(self.model)
        trainer = Trainer(
            model=self.model,
            optimizer=optimizer,
            iterator=self.iterator,
            train_dataset=self.train_data,
            validation_dataset=self.dev_data if self.configuration['early_stopping'] else None,
            cuda_device=gpu_id,
            num_epochs=self.configuration['epochs'],
            validation_metric=validation_metric,
            validation_iterator=self.val_iterator,
            serialization_dir=self.model_dir,
            patience=self.configuration['patience'],
            callbacks=callbacks,
            num_serialized_models_to_keep=2,
            early_stopping_by_batch=self.configuration['early_stopping_by_batch'],
            estimator=estimator,
            grad_clipping=5
        )
        metrics = trainer.train()
        self.logger.info('metrics: %s' % str(metrics))


class SentenceConsituencyAwareModel(BaseSentenceConsituencyAwareModel):
    """

    """

    def __init__(self, configuration):
        super().__init__(configuration)

    def _get_data_reader(self):
        token_indexer = SingleIdTokenIndexer(namespace="tokens")
        position_indexer = SingleIdTokenIndexer(namespace='position')
        aspect_indexer = SingleIdTokenIndexer(namespace='aspect')
        sentence_constituency_indexer = SingleIdTokenIndexer(namespace='sentence_constituency')
        reader = acd_and_sc_data_reader.TextInAllAspectSentimentOutSentenceConstituency(
            self.distinct_categories, self.distinct_polarities,
            tokenizer=self._get_word_segmenter(),
            token_indexers={"tokens": token_indexer},
            position_indexers={'position': position_indexer},
            aspect_indexers={'aspect': aspect_indexer},
            sentence_constituency_indexer={'sentence_constituency': sentence_constituency_indexer},
            configuration=self.configuration
        )
        return reader

    def _get_optimizer(self, model):
        # _params = filter(lambda p: p.requires_grad, model.parameters())
        # return optim.Adam(_params, lr=0.001)

        _params = filter(lambda p: p.requires_grad, model.parameters())
        return optim.Adam(_params, lr=0.001, weight_decay=0.00001)

    def _get_position_embeddings_dim(self):
        return self.configuration['position_embeddings_dim']

    def _find_model_function_pure(self):
        return pytorch_models.SentenceConsituencyAwareModelV8

    def error_analysis(self):
        """

        :return:
        """
        USE_GPU = torch.cuda.is_available()
        if USE_GPU:
            gpu_id = self.configuration['gpu_id']
        else:
            gpu_id = -1
        predictor = pytorch_models.TextInAllAspectSentimentOutPredictor(self.model, self.val_iterator,
                                                                        self.distinct_categories,
                                                                        self.distinct_polarities,
                                                                        configuration=self.configuration,
                                                                        cuda_device=gpu_id)

        data = self.test_data
        result = predictor.predict(data)
        result_final = [('sentence', 'aspect', 'predict', 'true')]
        for i in range(len(data)):
            instance: Instance = data[i]
            metadata = instance.fields['sample'].metadata
            sentence = metadata['text']
            labels_true = {self.distinct_categories[e[0]]: self.distinct_polarities[e[1]] for e in metadata['labels']}
            labels_pred = result[i]
            for label_pred in labels_pred:
                label_true = labels_true[label_pred[0]]
                if label_true == label_pred[1]:
                    continue
                result_final.append((sentence, label_pred[0], label_pred[1], label_true))
        result_str = ['\t'.join(e) for e in result_final]
        output_filepath = os.path.join(self.model_dir, 'error_analysis.csv')
        file_utils.write_lines(result_str, output_filepath)
        return result_final


class SentenceConsituencyAwareModelBert(BaseSentenceConsituencyAwareModel):
    """

    """

    def __init__(self, configuration):
        self.bert_file_path = configuration['bert_file_path']
        self.bert_vocab_file_path = configuration['bert_vocab_file_path']
        self.max_len = configuration['max_len']
        super().__init__(configuration)

    def _get_bert_word_segmenter(self):
        token_dict = {}
        for line in file_utils.read_all_lines(self.bert_vocab_file_path):
            token = line.strip()
            token_dict[token] = len(token_dict)

        result = bert_tokenizer.EnglishTokenizer(token_dict)
        return result

    def _get_data_reader(self):
        token_indexer = SingleIdTokenIndexer(namespace="tokens")
        bert_tokenizer = BertTokenizer.from_pretrained(self.bert_vocab_file_path, do_lower_case=True)
        bert_token_indexer = WordpieceIndexer(vocab=bert_tokenizer.vocab,
                                              wordpiece_tokenizer=bert_tokenizer.wordpiece_tokenizer.tokenize,
                                              namespace="bert",
                                              use_starting_offsets=False,
                                              max_pieces=self.max_len,
                                              do_lowercase=True,
                                              never_lowercase=None,
                                              start_tokens=None,
                                              end_tokens=None,
                                              separator_token="[SEP]",
                                              truncate_long_sequences=True)
        position_indexer = SingleIdTokenIndexer(namespace='position')
        sentence_constituency_indexer = SingleIdTokenIndexer(namespace='sentence_constituency')
        reader = acd_and_sc_data_reader.AcdAndScDatasetReaderConstituencyBert(
            self.distinct_categories, self.distinct_polarities,
            tokenizer=self._get_word_segmenter(),
            token_indexers={"tokens": token_indexer},
            position_indexers={'position': position_indexer},
            configuration=self.configuration,
            # bert_tokenizer=self._get_bert_word_segmenter(),
            bert_tokenizer=bert_tokenizer,
            bert_token_indexers={"bert": bert_token_indexer},
            sentence_constituency_indexer={'sentence_constituency': sentence_constituency_indexer},
        )
        return reader

    def _get_bert_word_embedder(self):
        pretrained_model = self.bert_file_path
        bert_model = PretrainedBertModel.load(pretrained_model, cache_model=False)
        for param in bert_model.parameters():
            param.requires_grad = (not self.configuration['fixed'])
        bert_embedder = BertEmbedder(bert_model=bert_model, top_layer_only=True)

        bert_word_embedder: TextFieldEmbedder = BasicTextFieldEmbedder({"bert": bert_embedder},
                                                                       # we'll be ignoring masks so we'll need to set this to True
                                                                       allow_unmatched_keys=True)
        bert_word_embedder.to(self.configuration['device'])
        return bert_word_embedder

    def _find_model_function(self):
        embedding_dim = self.configuration['embed_size']
        embedding_matrix_filepath = self.base_data_dir + 'embedding_matrix'
        if os.path.exists(embedding_matrix_filepath):
            embedding_matrix = super()._load_object(embedding_matrix_filepath)
        else:
            embedding_filepath = self.configuration['embedding_filepath']
            embedding_matrix = embedding._read_embeddings_from_text_file(embedding_filepath, embedding_dim,
                                                                         self.vocab, namespace='tokens')
            super()._save_object(embedding_matrix_filepath, embedding_matrix)
        token_embedding = Embedding(num_embeddings=self.vocab.get_vocab_size(namespace='tokens'),
                                    embedding_dim=embedding_dim, padding_index=0, vocab_namespace='tokens',
                                    trainable=False, weight=embedding_matrix)
        # the embedder maps the input tokens to the appropriate embedding matrix
        word_embedder: TextFieldEmbedder = BasicTextFieldEmbedder({"tokens": token_embedding})

        position_embedding = Embedding(num_embeddings=self.vocab.get_vocab_size(namespace='position'),
                                    embedding_dim=25, padding_index=0)
        position_embedder: TextFieldEmbedder = BasicTextFieldEmbedder({"position": position_embedding},
                                                                    # we'll be ignoring masks so we'll need to set this to True
                                                                    allow_unmatched_keys=True)

        bert_word_embedder = self._get_bert_word_embedder()

        model = pytorch_models.ConstituencyBert(
            word_embedder,
            position_embedder,
            self.distinct_categories,
            self.distinct_polarities,
            self.vocab,
            self.configuration,
            bert_word_embedder=bert_word_embedder
        )
        self._print_args(model)
        model = model.to(self.configuration['device'])
        return model

    def _get_optimizer(self, model):
        _params = filter(lambda p: p.requires_grad, model.parameters())
        if self.configuration['fixed']:
            return optim.Adam(_params, lr=0.001, weight_decay=0.00001)
        else:
            return optim.Adam(_params, lr=self.configuration['learning_rate_in_bert'],
                              weight_decay=self.configuration['l2_in_bert'])


class SentenceConsituencyAwareModelBertSingle(BaseSentenceConsituencyAwareModel):
    """

    """

    def __init__(self, configuration):
        self.bert_file_path = configuration['bert_file_path']
        self.bert_vocab_file_path = configuration['bert_vocab_file_path']
        self.max_len = configuration['max_len']
        super().__init__(configuration)

    def _get_bert_word_segmenter(self):
        token_dict = {}
        for line in file_utils.read_all_lines(self.bert_vocab_file_path):
            token = line.strip()
            token_dict[token] = len(token_dict)

        result = bert_tokenizer.EnglishTokenizer(token_dict)
        return result

    def _get_data_reader(self):
        token_indexer = SingleIdTokenIndexer(namespace="tokens")
        bert_tokenizer = BertTokenizer.from_pretrained(self.bert_vocab_file_path, do_lower_case=True)
        bert_token_indexer = WordpieceIndexer(vocab=bert_tokenizer.vocab,
                                              wordpiece_tokenizer=bert_tokenizer.wordpiece_tokenizer.tokenize,
                                              namespace="bert",
                                              use_starting_offsets=False,
                                              max_pieces=self.max_len,
                                              do_lowercase=True,
                                              never_lowercase=None,
                                              start_tokens=None,
                                              end_tokens=None,
                                              separator_token="[SEP]",
                                              truncate_long_sequences=True)
        position_indexer = SingleIdTokenIndexer(namespace='position')
        sentence_constituency_indexer = SingleIdTokenIndexer(namespace='sentence_constituency')
        reader = acd_and_sc_data_reader.AcdAndScDatasetReaderConstituencyBertSingle(
            self.distinct_categories, self.distinct_polarities,
            tokenizer=self._get_word_segmenter(),
            token_indexers={"tokens": token_indexer},
            position_indexers={'position': position_indexer},
            configuration=self.configuration,
            # bert_tokenizer=self._get_bert_word_segmenter(),
            bert_tokenizer=bert_tokenizer,
            bert_token_indexers={"bert": bert_token_indexer},
            sentence_constituency_indexer={'sentence_constituency': sentence_constituency_indexer}
        )
        return reader

    def _get_bert_word_embedder(self):
        pretrained_model = self.bert_file_path
        bert_model = PretrainedBertModel.load(pretrained_model, cache_model=False)
        for param in bert_model.parameters():
            param.requires_grad = (not self.configuration['fixed'])
        bert_embedder = BertEmbedder(bert_model=bert_model, top_layer_only=True)

        bert_word_embedder: TextFieldEmbedder = BasicTextFieldEmbedder({"bert": bert_embedder},
                                                                       # we'll be ignoring masks so we'll need to set this to True
                                                                       allow_unmatched_keys=True)
        bert_word_embedder.to(self.configuration['device'])
        return bert_word_embedder

    def _find_model_function(self):
        embedding_dim = self.configuration['embed_size']
        embedding_matrix_filepath = self.base_data_dir + 'embedding_matrix'
        if os.path.exists(embedding_matrix_filepath):
            embedding_matrix = super()._load_object(embedding_matrix_filepath)
        else:
            embedding_filepath = self.configuration['embedding_filepath']
            embedding_matrix = embedding._read_embeddings_from_text_file(embedding_filepath, embedding_dim,
                                                                         self.vocab, namespace='tokens')
            super()._save_object(embedding_matrix_filepath, embedding_matrix)
        token_embedding = Embedding(num_embeddings=self.vocab.get_vocab_size(namespace='tokens'),
                                    embedding_dim=embedding_dim, padding_index=0, vocab_namespace='tokens',
                                    trainable=False, weight=embedding_matrix)
        # the embedder maps the input tokens to the appropriate embedding matrix
        word_embedder: TextFieldEmbedder = BasicTextFieldEmbedder({"tokens": token_embedding})

        position_embedding = Embedding(num_embeddings=self.vocab.get_vocab_size(namespace='position'),
                                    embedding_dim=25, padding_index=0)
        position_embedder: TextFieldEmbedder = BasicTextFieldEmbedder({"position": position_embedding},
                                                                    # we'll be ignoring masks so we'll need to set this to True
                                                                    allow_unmatched_keys=True)

        bert_word_embedder: TextFieldEmbedder = self._get_bert_word_embedder()
        model = pytorch_models.ConstituencyBertSingle(
            word_embedder,
            position_embedder,
            self.distinct_categories,
            self.distinct_polarities,
            self.vocab,
            self.configuration,
            bert_word_embedder=bert_word_embedder
        )
        self._print_args(model)
        model = model.to(self.configuration['device'])
        return model

    def _get_optimizer(self, model):
        _params = filter(lambda p: p.requires_grad, model.parameters())
        if self.configuration['fixed']:
            return optim.Adam(_params, lr=0.001, weight_decay=0.00001)
        else:
            return optim.Adam(_params, lr=self.configuration['learning_rate_in_bert'],
                              weight_decay=self.configuration['l2_in_bert'])


class Can(TextInAllAspectSentimentOutTrainTemplate):
    """
    2019-emnlp-CAN Constrained Attention Networks for Multi-Aspect Sentiment Analysis
    """

    def __init__(self, configuration):
        configuration['lamda'] = 1
        super().__init__(configuration)

    def _get_optimizer(self, model):
        _params = filter(lambda p: p.requires_grad, model.parameters())
        return adagrad.Adagrad(_params, lr=0.01)

    def _is_train_token_embeddings(self):
        return True

    def _find_model_function_pure(self):
        return pytorch_models.Can


class EndToEnd(TextInAllAspectSentimentOutTrainTemplate):
    """
    2018-emnlp-Joint Aspect and Polarity Classification for Adpect-based Sentiment Analysis with End-to-End Neural Networks
    """

    def __init__(self, configuration):
        super().__init__(configuration)

    def _get_data_reader(self):
        if self.configuration['model_name'] == 'End-to-end-CNN':
            token_indexer = SingleIdTokenIndexer(namespace="tokens",
                                                 token_min_padding_length=self.configuration['token_min_padding_length'])
            position_indexer = SingleIdTokenIndexer(namespace='position')
            aspect_indexer = SingleIdTokenIndexer(namespace='aspect')
            reader = acd_and_sc_data_reader.TextInAllAspectSentimentOut(
                self.distinct_categories, self.distinct_polarities,
                tokenizer=self._get_word_segmenter(),
                token_indexers={"tokens": token_indexer},
                position_indexers={'position': position_indexer},
                aspect_indexers={'aspect': aspect_indexer},
                configuration=self.configuration
            )
            return reader
        else:
            return super()._get_data_reader()

    def _is_train_token_embeddings(self):
        return self.configuration['model_name'] == 'End-to-end-CNN'

    def _find_model_function_pure(self):
        return pytorch_models.EndToEnd


class AsCapsules(TextInAllAspectSentimentOutTrainTemplate):
    """
    2019-Aspect-level Sentiment Analysis using AS-Capsules
    """

    def __init__(self, configuration):
        super().__init__(configuration)

    def _get_aspect_embeddings_dim(self):
        return 256

    def _find_model_function_pure(self):
        return pytorch_models.AsCapsules


class AsMilBert(TextInAllAspectSentimentOutTrainTemplate):
    """

    """

    def __init__(self, configuration):
        self.bert_file_path = configuration['bert_file_path']
        self.bert_vocab_file_path = configuration['bert_vocab_file_path']
        self.max_len = configuration['max_len']
        super().__init__(configuration)

    def _get_bert_word_segmenter(self):
        token_dict = {}
        for line in file_utils.read_all_lines(self.bert_vocab_file_path):
            token = line.strip()
            token_dict[token] = len(token_dict)

        result = bert_tokenizer.EnglishTokenizer(token_dict)
        return result

    def _get_data_reader(self):
        token_indexer = SingleIdTokenIndexer(namespace="tokens")
        bert_tokenizer = BertTokenizer.from_pretrained(self.bert_vocab_file_path, do_lower_case=True)
        bert_token_indexer = WordpieceIndexer(vocab=bert_tokenizer.vocab,
                                              wordpiece_tokenizer=bert_tokenizer.wordpiece_tokenizer.tokenize,
                                              namespace="bert",
                                              use_starting_offsets=False,
                                              max_pieces=self.max_len,
                                              do_lowercase=True,
                                              never_lowercase=None,
                                              start_tokens=None,
                                              end_tokens=None,
                                              separator_token="[SEP]",
                                              truncate_long_sequences=True)
        position_indexer = SingleIdTokenIndexer(namespace='position')
        reader = acd_and_sc_data_reader.AcdAndScDatasetReaderMilSimultaneouslyBert(
            self.distinct_categories, self.distinct_polarities,
            tokenizer=self._get_word_segmenter(),
            token_indexers={"tokens": token_indexer},
            position_indexers={'position': position_indexer},
            configuration=self.configuration,
            # bert_tokenizer=self._get_bert_word_segmenter(),
            bert_tokenizer=bert_tokenizer,
            bert_token_indexers={"bert": bert_token_indexer}
        )
        return reader

    def _load_data(self):
        data_filepath = self.base_data_dir + 'data'
        if os.path.exists(data_filepath):
            self.train_data, self.dev_data, self.test_data, self.distinct_categories, self.distinct_polarities, \
                self.hard_test_data = super()._load_object(data_filepath)
            reader = self._get_data_reader()
            self.data_reader = reader
        else:
            train_dev_test_data, distinct_categories, distinct_polarities = self.dataset. \
                generate_acd_and_sc_data()

            # train_dev_test_data['hard_test'] = None
            # if self.hard_dataset:
            #     train_dev_test_data_hard, _, _ = self.hard_dataset.generate_acd_and_sc_data()
            #     train_dev_test_data['hard_test'] = train_dev_test_data_hard['test']

            if self.configuration['hard_test']:
                train_dev_test_data['hard_test'] = []
                for sample in train_dev_test_data['test']:
                    polarities = set([e[1] for e in sample[1]])
                    if len(polarities) >= 2:
                        train_dev_test_data['hard_test'].append(sample)

            distinct_polarities_new = []
            for polarity in distinct_polarities:
                if polarity != 'conflict':
                    distinct_polarities_new.append(polarity)
            self.distinct_categories = distinct_categories
            self.distinct_polarities = distinct_polarities_new

            # token_indexer = SingleIdTokenIndexer(namespace="tokens")
            # bert_tokenizer = BertTokenizer.from_pretrained(self.bert_vocab_file_path, do_lower_case=True)
            # bert_token_indexer = WordpieceIndexer(vocab=bert_tokenizer.vocab,
            #                                         wordpiece_tokenizer=bert_tokenizer.wordpiece_tokenizer.tokenize,
            #                                          namespace="bert",
            #                                          use_starting_offsets=False,
            #                                          max_pieces=self.max_len,
            #                                          do_lowercase=True,
            #                                          never_lowercase=None,
            #                                          start_tokens=None,
            #                                          end_tokens=None,
            #                                          separator_token="[SEP]",
            #                                          truncate_long_sequences=True)
            # position_indexer = SingleIdTokenIndexer(namespace='position')
            # reader = acd_and_sc_data_reader.AcdAndScDatasetReaderMilSimultaneouslyBert(
            #     self.distinct_categories, self.distinct_polarities,
            #     tokenizer=self._get_word_segmenter(),
            #     token_indexers={"tokens": token_indexer},
            #     position_indexers={'position': position_indexer},
            #     configuration=self.configuration,
            #     # bert_tokenizer=self._get_bert_word_segmenter(),
            #     bert_tokenizer=bert_tokenizer,
            #     bert_token_indexers={"bert": bert_token_indexer}
            # )
            reader = self._get_data_reader()
            self.data_reader = reader

            train_dev_test_data_label_indexed = {}
            for data_type, data in train_dev_test_data.items():
                if data is None:
                    continue
                data_new = []
                for sample in data:
                    sample_new = [sample[0]]
                    labels_new = []
                    for label in sample[1]:
                        aspect = label[0]
                        polarity = label[1]
                        aspect_index = distinct_categories.index(aspect)
                        if polarity == 'conflict':
                            polarity_index = -100
                        else:
                            polarity_index = distinct_polarities_new.index(polarity)
                        labels_new.append((aspect_index, polarity_index))
                    if len(labels_new) != 0:
                        sample_new.append(labels_new)
                        data_new.append(sample_new)
                train_dev_test_data_label_indexed[data_type] = data_new
            self.train_data = reader.read(train_dev_test_data_label_indexed['train'])
            self.dev_data = reader.read(train_dev_test_data_label_indexed['dev'])
            self.test_data = reader.read(train_dev_test_data_label_indexed['test'])
            if self.configuration['hard_test']:
                self.hard_test_data = reader.read(train_dev_test_data_label_indexed['hard_test'])
            data = [self.train_data, self.dev_data, self.test_data, self.distinct_categories,
                    self.distinct_polarities, self.hard_test_data]
            super()._save_object(data_filepath, data)

    def _get_bert_word_embedder(self):
        # bert_embedder = PretrainedBertEmbedder(
        #     pretrained_model=self.bert_file_path,
        #     top_layer_only=True,  # conserve memory
        #     requires_grad=(not self.configuration['fixed'])
        # )

        pretrained_model = self.bert_file_path
        bert_model = PretrainedBertModel.load(pretrained_model, cache_model=False)
        for param in bert_model.parameters():
            param.requires_grad = (not self.configuration['fixed'])
        bert_embedder = BertEmbedder(bert_model=bert_model, top_layer_only=True)

        bert_word_embedder: TextFieldEmbedder = BasicTextFieldEmbedder({"bert": bert_embedder},
                                                                       # we'll be ignoring masks so we'll need to set this to True
                                                                       allow_unmatched_keys=True)
        bert_word_embedder.to(self.configuration['device'])
        return bert_word_embedder

    def _find_model_function(self):
        embedding_dim = self.configuration['embed_size']
        embedding_matrix_filepath = self.base_data_dir + 'embedding_matrix'
        if os.path.exists(embedding_matrix_filepath):
            embedding_matrix = super()._load_object(embedding_matrix_filepath)
        else:
            embedding_filepath = self.configuration['embedding_filepath']
            embedding_matrix = embedding._read_embeddings_from_text_file(embedding_filepath, embedding_dim,
                                                                         self.vocab, namespace='tokens')
            super()._save_object(embedding_matrix_filepath, embedding_matrix)
        token_embedding = Embedding(num_embeddings=self.vocab.get_vocab_size(namespace='tokens'),
                                    embedding_dim=embedding_dim, padding_index=0, vocab_namespace='tokens',
                                    trainable=False, weight=embedding_matrix)
        # the embedder maps the input tokens to the appropriate embedding matrix
        word_embedder: TextFieldEmbedder = BasicTextFieldEmbedder({"tokens": token_embedding})

        position_embedding = Embedding(num_embeddings=self.vocab.get_vocab_size(namespace='position'),
                                    embedding_dim=25, padding_index=0)
        position_embedder: TextFieldEmbedder = BasicTextFieldEmbedder({"position": position_embedding},
                                                                    # we'll be ignoring masks so we'll need to set this to True
                                                                    allow_unmatched_keys=True)

        # bert_embedder = PretrainedBertEmbedder(
        #     pretrained_model=self.bert_file_path,
        #     top_layer_only=True,  # conserve memory
        #     requires_grad=True
        # )
        # bert_word_embedder: TextFieldEmbedder = BasicTextFieldEmbedder({"bert": bert_embedder},
        #                                                                  # we'll be ignoring masks so we'll need to set this to True
        #                                                                  allow_unmatched_keys=True)
        bert_word_embedder = self._get_bert_word_embedder()

        model = pytorch_models.AsMilSimultaneouslyBert(
            word_embedder,
            position_embedder,
            self.distinct_categories,
            self.distinct_polarities,
            self.vocab,
            self.configuration,
            bert_word_embedder=bert_word_embedder
        )
        self._print_args(model)
        model = model.to(self.configuration['device'])
        return model

    def _get_optimizer(self, model):
        _params = filter(lambda p: p.requires_grad, model.parameters())
        if self.configuration['fixed']:
            return optim.Adam(_params, lr=0.001, weight_decay=0.00001)
        else:
            return optim.Adam(_params, lr=self.configuration['learning_rate_in_bert'],
                              weight_decay=self.configuration['l2_in_bert'])


class AsMilBertSingle(TextInAllAspectSentimentOutTrainTemplate):
    """

    """

    def __init__(self, configuration):
        self.bert_file_path = configuration['bert_file_path']
        self.bert_vocab_file_path = configuration['bert_vocab_file_path']
        self.max_len = configuration['max_len']
        super().__init__(configuration)

    def _get_bert_word_segmenter(self):
        token_dict = {}
        for line in file_utils.read_all_lines(self.bert_vocab_file_path):
            token = line.strip()
            token_dict[token] = len(token_dict)

        result = bert_tokenizer.EnglishTokenizer(token_dict)
        return result

    def _load_data(self):
        data_filepath = self.base_data_dir + 'data'
        if os.path.exists(data_filepath):
            self.train_data, self.dev_data, self.test_data, self.distinct_categories, self.distinct_polarities, \
                self.hard_test_data = super()._load_object(data_filepath)
        else:
            train_dev_test_data, distinct_categories, distinct_polarities = self.dataset. \
                generate_acd_and_sc_data()

            train_dev_test_data['hard_test'] = None
            if self.hard_dataset:
                train_dev_test_data_hard, _, _ = self.hard_dataset.generate_acd_and_sc_data()
                train_dev_test_data['hard_test'] = train_dev_test_data_hard['test']

            distinct_polarities_new = []
            for polarity in distinct_polarities:
                if polarity != 'conflict':
                    distinct_polarities_new.append(polarity)
            self.distinct_categories = distinct_categories
            self.distinct_polarities = distinct_polarities_new

            token_indexer = SingleIdTokenIndexer(namespace="tokens")
            bert_tokenizer = BertTokenizer.from_pretrained(self.bert_vocab_file_path, do_lower_case=True)
            bert_token_indexer = WordpieceIndexer(vocab=bert_tokenizer.vocab,
                                                    wordpiece_tokenizer=bert_tokenizer.wordpiece_tokenizer.tokenize,
                                                     namespace="bert",
                                                     use_starting_offsets=False,
                                                     max_pieces=self.max_len,
                                                     do_lowercase=True,
                                                     never_lowercase=None,
                                                     start_tokens=None,
                                                     end_tokens=None,
                                                     separator_token="[SEP]",
                                                     truncate_long_sequences=True)
            position_indexer = SingleIdTokenIndexer(namespace='position')
            reader = acd_and_sc_data_reader.AcdAndScDatasetReaderMilSimultaneouslyBertSingle(
                self.distinct_categories, self.distinct_polarities,
                tokenizer=self._get_word_segmenter(),
                token_indexers={"tokens": token_indexer},
                position_indexers={'position': position_indexer},
                configuration=self.configuration,
                # bert_tokenizer=self._get_bert_word_segmenter(),
                bert_tokenizer=bert_tokenizer,
                bert_token_indexers={"bert": bert_token_indexer}
            )
            self.data_reader = reader

            train_dev_test_data_label_indexed = {}
            for data_type, data in train_dev_test_data.items():
                if data is None:
                    continue
                data_new = []
                for sample in data:
                    sample_new = [sample[0]]
                    labels_new = []
                    for label in sample[1]:
                        aspect = label[0]
                        polarity = label[1]
                        aspect_index = distinct_categories.index(aspect)
                        if polarity == 'conflict':
                            polarity_index = -100
                        else:
                            polarity_index = distinct_polarities_new.index(polarity)
                        labels_new.append((aspect_index, polarity_index))
                    if len(labels_new) != 0:
                        sample_new.append(labels_new)
                        data_new.append(sample_new)
                train_dev_test_data_label_indexed[data_type] = data_new
            self.train_data = reader.read(train_dev_test_data_label_indexed['train'])
            self.dev_data = reader.read(train_dev_test_data_label_indexed['dev'])
            self.test_data = reader.read(train_dev_test_data_label_indexed['test'])
            if self.hard_dataset:
                self.hard_test_data = reader.read(train_dev_test_data_label_indexed['hard_test'])
            data = [self.train_data, self.dev_data, self.test_data, self.distinct_categories,
                    self.distinct_polarities, self.hard_test_data ]
            super()._save_object(data_filepath, data)

    def _get_bert_word_embedder(self):
        # bert_embedder = PretrainedBertEmbedder(
        #     pretrained_model=self.bert_file_path,
        #     top_layer_only=True,  # conserve memory
        #     requires_grad=(not self.configuration['fixed'])
        # )

        pretrained_model = self.bert_file_path
        bert_model = PretrainedBertModel.load(pretrained_model, cache_model=False)
        for param in bert_model.parameters():
            param.requires_grad = (not self.configuration['fixed'])
        bert_embedder = BertEmbedder(bert_model=bert_model, top_layer_only=True)

        bert_word_embedder: TextFieldEmbedder = BasicTextFieldEmbedder({"bert": bert_embedder},
                                                                       # we'll be ignoring masks so we'll need to set this to True
                                                                       allow_unmatched_keys=True)
        bert_word_embedder.to(self.configuration['device'])
        return bert_word_embedder

    def _find_model_function(self):
        embedding_dim = self.configuration['embed_size']
        embedding_matrix_filepath = self.base_data_dir + 'embedding_matrix'
        if os.path.exists(embedding_matrix_filepath):
            embedding_matrix = super()._load_object(embedding_matrix_filepath)
        else:
            embedding_filepath = self.configuration['embedding_filepath']
            embedding_matrix = embedding._read_embeddings_from_text_file(embedding_filepath, embedding_dim,
                                                                         self.vocab, namespace='tokens')
            super()._save_object(embedding_matrix_filepath, embedding_matrix)
        token_embedding = Embedding(num_embeddings=self.vocab.get_vocab_size(namespace='tokens'),
                                    embedding_dim=embedding_dim, padding_index=0, vocab_namespace='tokens',
                                    trainable=False, weight=embedding_matrix)
        # the embedder maps the input tokens to the appropriate embedding matrix
        word_embedder: TextFieldEmbedder = BasicTextFieldEmbedder({"tokens": token_embedding})

        position_embedding = Embedding(num_embeddings=self.vocab.get_vocab_size(namespace='position'),
                                    embedding_dim=25, padding_index=0)
        position_embedder: TextFieldEmbedder = BasicTextFieldEmbedder({"position": position_embedding},
                                                                    # we'll be ignoring masks so we'll need to set this to True
                                                                    allow_unmatched_keys=True)

        bert_word_embedder: TextFieldEmbedder = self._get_bert_word_embedder()
        model = pytorch_models.AsMilSimultaneouslyBertSingle(
            word_embedder,
            position_embedder,
            self.distinct_categories,
            self.distinct_polarities,
            self.vocab,
            self.configuration,
            bert_word_embedder=bert_word_embedder
        )
        self._print_args(model)
        model = model.to(self.configuration['device'])
        return model

    def _get_optimizer(self, model):
        _params = filter(lambda p: p.requires_grad, model.parameters())
        if self.configuration['fixed']:
            return optim.Adam(_params, lr=0.001, weight_decay=0.00001)
        else:
            return optim.Adam(_params, lr=self.configuration['learning_rate_in_bert'],
                              weight_decay=self.configuration['l2_in_bert'])


class Cae(TextInAllAspectSentimentOutTrainTemplate):
    """

    """

    def __init__(self, configuration):
        super().__init__(configuration)

    def _find_model_function_pure(self):
        # ['CAE', 'CAE-att-only-in-lstm', 'CAE-att-only-in-embedding',
        #  'CAE-add', 'CAE-average-of-two-layer']
        model_name = self.configuration['model_name']
        if model_name == 'CAE':
            return pytorch_models.Cae
        elif model_name == 'CaeSupportingPipeline':
            return pytorch_models.CaeSupportingPipeline
        elif model_name == 'CAE-att-only-in-lstm':
            return pytorch_models.CaeAttOnlyInLstm
        elif model_name == 'CAE-att-only-in-embedding':
            return pytorch_models.CaeAttOnlyInEmbedding
        elif model_name == 'CAE-add':
            return pytorch_models.CaeAdd
        elif model_name == 'CAE-average-of-two-layer':
            return pytorch_models.CaeAverageOfTwoLayer
        elif model_name == 'CAE-without-cae':
            return pytorch_models.CaeWithoutCAE


class AOA(TextInAllAspectSentimentOutTrainTemplate):
    """

    """

    def __init__(self, configuration):
        super().__init__(configuration)

    def _get_position_embeddings_dim(self):
        return self.configuration['position_embeddings_dim']

    def _find_model_function_pure(self):
        return pytorch_models.AOA


class AtaeLstmM(TextInAllAspectSentimentOutTrainTemplate):
    """

    """

    def __init__(self, configuration):
        super().__init__(configuration)

    def _find_model_function_pure(self):
        return pytorch_models.AtaeLstmM


class AtaeLstmCae(TextInAllAspectSentimentOutTrainTemplate):
    """

    """

    def __init__(self, configuration):
        super().__init__(configuration)

    def _get_acd_warmup_epoch_num(self):
        return self.configuration['acd_warmup_epoch_num']

    def _find_model_function_pure(self):
        return pytorch_models.AtaeLstmCae


class CapsNetCae(TextInAllAspectSentimentOutTrainTemplate):
    """

    """

    def __init__(self, configuration):
        super().__init__(configuration)

    def _get_acd_warmup_epoch_num(self):
        return self.configuration['acd_warmup_epoch_num']

    def load_sentiment_matrix(self, glove_path, sentiment_path):
        npy_sentiment_path = sentiment_path + '.npy'
        if not os.path.exists(npy_sentiment_path):
            sentiment_matrix = np.zeros((3, 300), dtype=np.float32)
            sd = json.load(open(sentiment_path, 'r', encoding='utf-8'))
            sd['positive'] = set(sd['positive'])
            sd['negative'] = set(sd['negative'])
            sd['neutral'] = set(sd['neutral'])
            with open(glove_path, 'r', encoding='utf-8') as f:
                for line in f:
                    content = line.split(' ')
                    word = content[0]
                    vec = np.array(list(map(float, content[1:])))
                    if word in sd['positive']:
                        sentiment_matrix[0] += vec
                    elif word in sd['negative']:
                        sentiment_matrix[1] += vec
                    elif word in sd['neutral']:
                        sentiment_matrix[2] += vec
            sentiment_matrix -= sentiment_matrix.mean()
            sentiment_matrix = sentiment_matrix / sentiment_matrix.std() * np.sqrt(2.0 / (300.0 + 3.0))
            np.save(npy_sentiment_path, sentiment_matrix)
        return npy_sentiment_path

    def _find_model_function(self):
        embedding_dim = self.configuration['embed_size']
        embedding_matrix_filepath = self.base_data_dir + 'embedding_matrix'
        if os.path.exists(embedding_matrix_filepath):
            embedding_matrix = super()._load_object(embedding_matrix_filepath)
        else:
            embedding_filepath = self.configuration['embedding_filepath']
            embedding_matrix = embedding._read_embeddings_from_text_file(embedding_filepath, embedding_dim,
                                                                         self.vocab, namespace='tokens')
            super()._save_object(embedding_matrix_filepath, embedding_matrix)
        embedding_matrix = embedding_matrix.to(self.configuration['device'])
        token_embedding = Embedding(num_embeddings=self.vocab.get_vocab_size(namespace='tokens'),
                                    embedding_dim=embedding_dim, padding_index=0, vocab_namespace='tokens',
                                    trainable=self._is_train_token_embeddings(), weight=embedding_matrix)
        # the embedder maps the input tokens to the appropriate embedding matrix
        word_embedder: TextFieldEmbedder = BasicTextFieldEmbedder({"tokens": token_embedding})

        position_embedding = Embedding(num_embeddings=self.vocab.get_vocab_size(namespace='position'),
                                    embedding_dim=self._get_position_embeddings_dim(), padding_index=0)
        position_embedder: TextFieldEmbedder = BasicTextFieldEmbedder({"position": position_embedding},
                                                                    # we'll be ignoring masks so we'll need to set this to True
                                                                    allow_unmatched_keys=True)

        aspect_embedding = Embedding(num_embeddings=self.vocab.get_vocab_size(namespace='aspect'),
                                     embedding_dim=self._get_aspect_embeddings_dim(), padding_index=0)
        aspect_embedder: TextFieldEmbedder = BasicTextFieldEmbedder({"aspect": aspect_embedding},
                                                                    # we'll be ignoring masks so we'll need to set this to True
                                                                    allow_unmatched_keys=True)

        embedding_for_capsnet = torch.nn.Embedding(num_embeddings=self.vocab.get_vocab_size(namespace='tokens'),
                                                   embedding_dim=embedding_dim)
        embedding_for_capsnet.weight.data.copy_(embedding_matrix)
        self.configuration['embedding_for_capsnet'] = embedding_for_capsnet
        aspect_embedding_for_capsnet = torch.nn.Embedding(num_embeddings=self.vocab.get_vocab_size(namespace='aspect'),
                                                          embedding_dim=self._get_aspect_embeddings_dim())
        self.configuration['aspect_embedding_for_capsnet'] = aspect_embedding_for_capsnet

        sentiment_path = data_object.MAMSACSA.sentiment_path
        self.configuration['sentiment_matrix'] = self.load_sentiment_matrix(self.configuration['embedding_filepath'],
                                                                            sentiment_path)
        model = pytorch_models.CapsNetCae(
            word_embedder,
            position_embedder,
            aspect_embedder,
            self.distinct_categories,
            self.distinct_polarities,
            self.vocab,
            self.configuration
        )
        self._print_args(model)
        model = model.to(self.configuration['device'])
        return model


class TextInAllAspectOutTrainTemplate(ModelTrainTemplate.ModelTrainTemplate):
    """

    """

    def __init__(self, configuration):
        super().__init__(configuration)
        self.data_reader = None
        self.train_data = None
        self.dev_data = None
        self.test_data = None
        self.hard_test_data = None
        self.distinct_categories = None
        self.distinct_polarities = None
        self._load_data()
        if self.configuration['debug']:
            self.train_data = self.train_data[: 128]
            self.dev_data = self.dev_data[: 128]
            self.test_data = self.test_data[: 128]

        self.vocab = None
        self._build_vocab()

        self.iterator = None
        self.val_iterator = None
        self._build_iterator()

    def _get_data_reader(self):
        token_indexer = SingleIdTokenIndexer(namespace="tokens")
        position_indexer = SingleIdTokenIndexer(namespace='position')
        aspect_indexer = SingleIdTokenIndexer(namespace='aspect')
        reader = acd_and_sc_data_reader.TextInAllAspectOut(
            self.distinct_categories, self.distinct_polarities,
            tokenizer=self._get_word_segmenter(),
            token_indexers={"tokens": token_indexer},
            position_indexers={'position': position_indexer},
            aspect_indexers={'aspect': aspect_indexer},
            configuration=self.configuration
        )
        return reader

    def _load_data(self):
        data_filepath = self.base_data_dir + 'data'
        if os.path.exists(data_filepath):
            self.train_data, self.dev_data, self.test_data, self.distinct_categories, self.distinct_polarities, \
            self.hard_test_data = super()._load_object(data_filepath)
        else:
            train_dev_test_data, distinct_categories, distinct_polarities = self.dataset. \
                generate_acd_and_sc_data()

            if self.configuration['hard_test']:
                train_dev_test_data['hard_test'] = []
                for sample in train_dev_test_data['test']:
                    polarities = set([e[1] for e in sample[1]])
                    if len(polarities) >= 2:
                        train_dev_test_data['hard_test'].append(sample)

            self.distinct_categories = distinct_categories
            self.distinct_polarities = distinct_polarities

            reader = self._get_data_reader()
            self.data_reader = reader

            train_dev_test_data_label_indexed = {}
            for data_type, data in train_dev_test_data.items():
                if data is None:
                    continue
                data_new = []
                for sample in data:
                    sample_new = [sample[0]]
                    labels_new = []
                    for label in sample[1]:
                        aspect = label[0]
                        polarity = label[1]
                        aspect_index = distinct_categories.index(aspect)
                        polarity_index = distinct_polarities.index(polarity)
                        labels_new.append((aspect_index, polarity_index))
                    if len(labels_new) != 0:
                        sample_new.append(labels_new)
                        data_new.append(sample_new)
                train_dev_test_data_label_indexed[data_type] = data_new
            self.train_data = reader.read(train_dev_test_data_label_indexed['train'])
            self.dev_data = reader.read(train_dev_test_data_label_indexed['dev'])
            self.test_data = reader.read(train_dev_test_data_label_indexed['test'])
            if self.configuration['hard_test']:
                self.hard_test_data = reader.read(train_dev_test_data_label_indexed['hard_test'])
            data = [self.train_data, self.dev_data, self.test_data, self.distinct_categories,
                    self.distinct_polarities, self.hard_test_data]
            super()._save_object(data_filepath, data)

    def _build_vocab(self):
        if self.configuration['train']:
            vocab_file_path = self.base_data_dir + 'vocab'
            if os.path.exists(vocab_file_path):
                self.vocab = super()._load_object(vocab_file_path)
            else:
                data = self.train_data + self.dev_data + self.test_data
                self.vocab = Vocabulary.from_instances(data, max_vocab_size=sys.maxsize)
                super()._save_object(vocab_file_path, self.vocab)
            self.model_meta_data['vocab'] = self.vocab
        else:
            self.vocab = self.model_meta_data['vocab']

    def _build_iterator(self):
        self.iterator = BucketIterator(batch_size=self.configuration['batch_size'],
                                       sorting_keys=[("tokens", "num_tokens")],
                                       )
        self.iterator.index_with(self.vocab)
        self.val_iterator = BasicIterator(batch_size=self.configuration['batch_size'])
        self.val_iterator.index_with(self.vocab)

    def _print_args(self, model):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in model.parameters():
            n_params = torch.prod(torch.tensor(p.shape)).item()
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        self.logger.info('n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        self.logger.info('> training arguments:')
        for arg in self.configuration.keys():
            self.logger.info('>>> {0}: {1}'.format(arg, self.configuration[arg]))

    def _find_model_function_pure(self):
        raise NotImplementedError('_find_model_function_pure')

    def _get_aspect_embeddings_dim(self):
        return 300

    def _get_position_embeddings_dim(self):
        return 300

    def _is_train_token_embeddings(self):
        return False

    def _find_model_function(self):
        embedding_dim = self.configuration['embed_size']
        embedding_matrix_filepath = self.base_data_dir + 'embedding_matrix'
        if os.path.exists(embedding_matrix_filepath):
            embedding_matrix = super()._load_object(embedding_matrix_filepath)
        else:
            embedding_filepath = self.configuration['embedding_filepath']
            embedding_matrix = embedding._read_embeddings_from_text_file(embedding_filepath, embedding_dim,
                                                                         self.vocab, namespace='tokens')
            super()._save_object(embedding_matrix_filepath, embedding_matrix)
        embedding_matrix = embedding_matrix.to(self.configuration['device'])
        token_embedding = Embedding(num_embeddings=self.vocab.get_vocab_size(namespace='tokens'),
                                    embedding_dim=embedding_dim, padding_index=0, vocab_namespace='tokens',
                                    trainable=self._is_train_token_embeddings(), weight=embedding_matrix)
        # the embedder maps the input tokens to the appropriate embedding matrix
        word_embedder: TextFieldEmbedder = BasicTextFieldEmbedder({"tokens": token_embedding})

        position_embedding = Embedding(num_embeddings=self.vocab.get_vocab_size(namespace='position'),
                                    embedding_dim=self._get_position_embeddings_dim(), padding_index=0)
        position_embedder: TextFieldEmbedder = BasicTextFieldEmbedder({"position": position_embedding},
                                                                    # we'll be ignoring masks so we'll need to set this to True
                                                                    allow_unmatched_keys=True)

        aspect_embedding = Embedding(num_embeddings=self.vocab.get_vocab_size(namespace='aspect'),
                                     embedding_dim=self._get_aspect_embeddings_dim(), padding_index=0)
        aspect_embedder: TextFieldEmbedder = BasicTextFieldEmbedder({"aspect": aspect_embedding},
                                                                    # we'll be ignoring masks so we'll need to set this to True
                                                                    allow_unmatched_keys=True)
        model_function: pytorch_models.TextInAllAspectSentimentOutModel = self._find_model_function_pure()
        model = model_function(
            word_embedder,
            position_embedder,
            aspect_embedder,
            self.distinct_categories,
            self.distinct_polarities,
            self.vocab,
            self.configuration,
        )
        self._print_args(model)
        model = model.to(self.configuration['device'])
        return model

    def _get_optimizer(self, model):
        _params = filter(lambda p: p.requires_grad, model.parameters())
        return optim.Adam(_params, lr=0.001, weight_decay=0.00001)

    def _get_acd_optimizer(self, model):
        _params = filter(lambda p: p.requires_grad, model.parameters())
        return optim.Adam(_params, lr=0.001, weight_decay=0.00001)

    def _get_acd_warmup_epoch_num(self):
        return 3

    def _get_estimator(self, model):
        USE_GPU = torch.cuda.is_available()
        if USE_GPU:
            gpu_id = self.configuration['gpu_id']
        else:
            gpu_id = -1
        estimator = pytorch_models.TextInAllAspectOutEstimator(model, self.val_iterator,
                                                                        self.distinct_categories,
                                                                        self.distinct_polarities,
                                                                        configuration=self.configuration,
                                                                        cuda_device=gpu_id)
        return estimator

    def _get_estimate_callback(self, model):
        result = []
        data_type_and_data = {
            'dev': self.dev_data,
            'test': self.test_data
        }
        if self.hard_test_data:
            data_type_and_data['hard_test'] = self.hard_test_data
        estimator = self._get_estimator(model)
        estimate_callback = allennlp_callback.EstimateCallback(data_type_and_data, estimator, self.logger)
        result.append(estimate_callback)
        return result

    def _get_loss_weight_callback(self):
        result = []
        set_loss_weight_callback = allennlp_callback.SetLossWeightCallback(self.model, self.logger,
                                                                           acd_warmup_epoch_num=self._get_acd_warmup_epoch_num())
        result.append(set_loss_weight_callback)
        return result

    def _get_fixed_loss_weight_callback(self, model, category_loss_weight=1, sentiment_loss_weight=1):
        result = []
        fixed_loss_weight_callback = allennlp_callback.FixedLossWeightCallback(model, self.logger,
                                                                             category_loss_weight=category_loss_weight,
                                                                             sentiment_loss_weight=sentiment_loss_weight)
        result.append(fixed_loss_weight_callback)
        return result

    def _get_bert_word_embedder(self):
        return None

    def _inner_train(self):
        USE_GPU = torch.cuda.is_available()
        if USE_GPU:
            gpu_id = self.configuration['gpu_id']
        else:
            gpu_id = -1

        self.model: pytorch_models.TextInAllAspectOutModel = self._find_model_function()

        estimator = self._get_estimator(self.model)
        optimizer = self._get_acd_optimizer(self.model)
        self.logger.info('acd warmup')
        validation_metric = '+category_f1'
        callbacks = self._get_estimate_callback(self.model)

        trainer = Trainer(
            model=self.model,
            optimizer=optimizer,
            iterator=self.iterator,
            train_dataset=self.train_data,
            validation_dataset=self.dev_data,
            cuda_device=gpu_id,
            num_epochs=self.configuration['epochs'],
            validation_metric=validation_metric,
            validation_iterator=self.val_iterator,
            serialization_dir=self.model_dir,
            patience=self.configuration['patience'],
            callbacks=callbacks,
            num_serialized_models_to_keep=2,
            early_stopping_by_batch=self.configuration['early_stopping_by_batch'],
            estimator=estimator,
            grad_clipping=5
        )
        metrics = trainer.train()
        self.logger.info('acd metrics: %s' % str(metrics))


    def _save_model(self):
        torch.save(self.model, self.best_model_filepath)

    def _load_model(self):
        self.model = torch.load(self.best_model_filepath)
        self.model.configuration = self.configuration

    def evaluate(self):
        USE_GPU = torch.cuda.is_available()
        if USE_GPU:
            gpu_id = self.configuration['gpu_id']
        else:
            gpu_id = -1
        estimator = self._get_estimator(self.model)

        data_type_and_data = {
            # 'train': self.train_data,
            # 'dev': self.dev_data,
            'test': self.test_data
        }
        if self.hard_test_data:
            data_type_and_data['hard_test'] = self.hard_test_data
        for data_type, data in data_type_and_data.items():
            result = estimator.estimate(data)
            self.logger.info('data_type: %s result: %s' % (data_type, result))


class Acd(TextInAllAspectOutTrainTemplate):
    """
    用途：
    1. 评估属性分类效果
    2. 产生<句子, 属性>的attention权重
    """

    def __init__(self, configuration):
        # configuration['fc_after_embeddings_layer'] = True
        # configuration['only_embeddings_layer'] = True
        super().__init__(configuration)

    def _find_model_function_pure(self):
        return pytorch_models.Acd


class AcdWithInteractiveLoss(TextInAllAspectOutTrainTemplate):
    """
    用途：
    1. 评估属性分类效果
    2. 产生<句子, 属性>的attention权重
    """

    def __init__(self, configuration):
        # configuration['fc_after_embeddings_layer'] = True
        # configuration['only_embeddings_layer'] = True
        super().__init__(configuration)

    def _get_data_reader(self):
        token_indexer = SingleIdTokenIndexer(namespace="tokens")
        position_indexer = SingleIdTokenIndexer(namespace='position')
        aspect_indexer = SingleIdTokenIndexer(namespace='aspect')
        reader = acd_and_sc_data_reader.TextInAllAspectOut(
            self.distinct_categories, self.distinct_polarities,
            tokenizer=self._get_word_segmenter(),
            token_indexers={"tokens": token_indexer},
            position_indexers={'position': position_indexer},
            aspect_indexers={'aspect': aspect_indexer},
            configuration=self.configuration
        )
        return reader

    def _find_model_function_pure(self):
        return pytorch_models.AcdWithInteractiveLoss


class CaePretrainedPosition(TextInAllAspectSentimentOutTrainTemplate):
    """

    """

    def __init__(self, configuration):
        super().__init__(configuration)

    def _find_model_function_pure(self):
        return pytorch_models.CaePretrainedPosition


class TextAspectInSentimentOutTrainTemplate(ModelTrainTemplate.ModelTrainTemplate):
    """
    2016-emnlp-Attention-based LSTM for Aspect-level Sentiment Classification
    """

    def __init__(self, configuration):
        super().__init__(configuration)
        self.data_reader = None
        self.train_data = None
        self.dev_data = None
        self.test_data = None
        self.hard_test_data = None
        self.distinct_categories = None
        self.distinct_polarities = None
        self._load_data()

        self.vocab = None
        self._build_vocab()

        self.iterator = None
        self.val_iterator = None
        self._build_iterator()

    def _get_data_reader(self):
        token_indexer = SingleIdTokenIndexer(namespace="tokens",
                                             token_min_padding_length=self.configuration['token_min_padding_length'])
        aspect_indexer = SingleIdTokenIndexer(namespace='aspect')
        reader = acd_and_sc_data_reader.TextAspectInSentimentOut(
            self.distinct_categories, self.distinct_polarities,
            tokenizer=self._get_word_segmenter(),
            token_indexers={"tokens": token_indexer},
            aspect_indexers={'aspect': aspect_indexer},
            configuration=self.configuration
        )
        return reader

    def _load_data(self):
        data_filepath = self.base_data_dir + 'data'
        if os.path.exists(data_filepath):
            self.train_data, self.dev_data, self.test_data, self.distinct_categories, self.distinct_polarities, \
                self.hard_test_data = super()._load_object(data_filepath)
        else:
            train_dev_test_data, distinct_categories, distinct_polarities = self.dataset. \
                generate_acd_and_sc_data()

            if self.configuration['hard_test']:
                train_dev_test_data['hard_test'] = []
                for sample in train_dev_test_data['test']:
                    polarities = set([e[1] for e in sample[1]])
                    if len(polarities) >= 2:
                        train_dev_test_data['hard_test'].append(sample)

            distinct_polarities_new = []
            for polarity in distinct_polarities:
                if polarity != 'conflict':
                    distinct_polarities_new.append(polarity)
            self.distinct_categories = distinct_categories
            self.distinct_polarities = distinct_polarities_new

            # token_indexer = SingleIdTokenIndexer(namespace="tokens",
            #                                      token_min_padding_length=self.configuration['token_min_padding_length'])
            # aspect_indexer = SingleIdTokenIndexer(namespace='aspect')
            # reader = acd_and_sc_data_reader.TextAspectInSentimentOut(
            #     self.distinct_categories, self.distinct_polarities,
            #     tokenizer=self._get_word_segmenter(),
            #     token_indexers={"tokens": token_indexer},
            #     aspect_indexers={'aspect': aspect_indexer},
            #     configuration=self.configuration
            # )
            reader = self._get_data_reader()
            self.data_reader = reader

            train_dev_test_data_label_indexed = {}
            for data_type, data in train_dev_test_data.items():
                if data is None:
                    continue
                data_new = []
                for sample in data:
                    sample_new = [sample[0]]
                    labels_new = []
                    for label in sample[1]:
                        aspect = label[0]
                        polarity = label[1]
                        aspect_index = distinct_categories.index(aspect)
                        if polarity == 'conflict':
                            polarity_index = -100
                        else:
                            polarity_index = distinct_polarities_new.index(polarity)
                        labels_new.append((aspect_index, polarity_index))
                    if len(labels_new) != 0:
                        sample_new.append(labels_new)
                        data_new.append(sample_new)
                train_dev_test_data_label_indexed[data_type] = data_new
            self.train_data = reader.read(train_dev_test_data_label_indexed['train'])
            self.dev_data = reader.read(train_dev_test_data_label_indexed['dev'])
            self.test_data = reader.read(train_dev_test_data_label_indexed['test'])
            if self.configuration['hard_test']:
                self.hard_test_data = reader.read(train_dev_test_data_label_indexed['hard_test'])
            data = [self.train_data, self.dev_data, self.test_data, self.distinct_categories,
                    self.distinct_polarities, self.hard_test_data]
            super()._save_object(data_filepath, data)

    def _build_vocab(self):
        if self.configuration['train']:
            vocab_file_path = self.base_data_dir + 'vocab'
            if os.path.exists(vocab_file_path):
                self.vocab = super()._load_object(vocab_file_path)
            else:
                data = self.train_data + self.dev_data + self.test_data
                self.vocab = Vocabulary.from_instances(data, max_vocab_size=sys.maxsize)
                super()._save_object(vocab_file_path, self.vocab)
            self.model_meta_data['vocab'] = self.vocab
        else:
            self.vocab = self.model_meta_data['vocab']

    def _build_iterator(self):
        self.iterator = BucketIterator(batch_size=self.configuration['batch_size'],
                                       sorting_keys=[("tokens", "num_tokens")],
                                       )
        self.iterator.index_with(self.vocab)
        self.val_iterator = BasicIterator(batch_size=self.configuration['batch_size'])
        self.val_iterator.index_with(self.vocab)

    def _print_args(self, model):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in model.parameters():
            n_params = torch.prod(torch.tensor(p.shape)).item()
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        self.logger.info('n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        self.logger.info('> training arguments:')
        for arg in self.configuration.keys():
            self.logger.info('>>> {0}: {1}'.format(arg, self.configuration[arg]))

    def _find_model_function_pure(self):
        raise NotImplementedError('_find_model_function_pure')

    def _get_aspect_embeddings_dim(self):
        return 300

    def _init_aspect_embeddings_from_word_embeddings(self):
        return False

    def _find_model_function(self):
        embedding_dim = self.configuration['embed_size']
        embedding_matrix_filepath = self.base_data_dir + 'embedding_matrix'
        if os.path.exists(embedding_matrix_filepath):
            embedding_matrix = super()._load_object(embedding_matrix_filepath)
        else:
            embedding_filepath = self.configuration['embedding_filepath']
            embedding_matrix = embedding._read_embeddings_from_text_file(embedding_filepath, embedding_dim,
                                                                         self.vocab, namespace='tokens')
            super()._save_object(embedding_matrix_filepath, embedding_matrix)
        token_embedding = Embedding(num_embeddings=self.vocab.get_vocab_size(namespace='tokens'),
                                    embedding_dim=embedding_dim, padding_index=0, vocab_namespace='tokens',
                                    trainable=False, weight=embedding_matrix)
        # the embedder maps the input tokens to the appropriate embedding matrix
        word_embedder: TextFieldEmbedder = BasicTextFieldEmbedder({"tokens": token_embedding})

        aspect_embedding_matrix = None
        if self._init_aspect_embeddings_from_word_embeddings():
            embedding_filepath = self.configuration['embedding_filepath']
            aspect_embedding_matrix = embedding._read_embeddings_from_text_file(embedding_filepath, embedding_dim,
                                                                                self.vocab, namespace='aspect')
        aspect_embedding = Embedding(num_embeddings=self.vocab.get_vocab_size(namespace='aspect'),
                                    embedding_dim=self._get_aspect_embeddings_dim(), padding_index=0,
                                     trainable=True, weight=aspect_embedding_matrix)
        aspect_embedder: TextFieldEmbedder = BasicTextFieldEmbedder({"aspect": aspect_embedding},
                                                                    # we'll be ignoring masks so we'll need to set this to True
                                                                    allow_unmatched_keys=True)
        model_function = self._find_model_function_pure()
        model = model_function(
            word_embedder,
            aspect_embedder,
            self.distinct_categories,
            self.distinct_polarities,
            self.vocab,
            self.configuration
        )
        self._print_args(model)
        model = model.to(self.configuration['device'])
        return model

    def _get_estimator(self, model):
        USE_GPU = torch.cuda.is_available()
        if USE_GPU:
            gpu_id = self.configuration['gpu_id']
        else:
            gpu_id = -1
        estimator = pytorch_models.TextAspectInSentimentOutEstimator(model, self.val_iterator,
                                                                     self.distinct_categories,
                                                                     self.distinct_polarities,
                                                                     cuda_device=gpu_id)
        return estimator

    def _get_estimate_callback(self, model):
        result = []
        data_type_and_data = {
            'train': self.train_data,
            'dev': self.dev_data,
            'test': self.test_data
        }
        if self.hard_test_data:
            data_type_and_data['hard_test'] = self.hard_test_data
        estimator = self._get_estimator(model)
        estimate_callback = allennlp_callback.EstimateCallback(data_type_and_data, estimator, self.logger)
        result.append(estimate_callback)
        return result

    def _get_optimizer(self, model):
        _params = filter(lambda p: p.requires_grad, model.parameters())
        return optim.Adam(_params, lr=0.001, weight_decay=0.00001)

    def _inner_train(self):
        USE_GPU = torch.cuda.is_available()
        if USE_GPU:
            gpu_id = self.configuration['gpu_id']
        else:
            gpu_id = -1
        self.model = self._find_model_function()
        # optimizer = adagrad.Adagrad(self.model.parameters(), lr=0.01, weight_decay=0.001)
        # optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=0.00001)
        optimizer = self._get_optimizer(self.model)
        self._print_args(self.model)

        callbacks = self._get_estimate_callback(self.model)
        early_stopping_by_batch: bool = False
        estimator = self._get_estimator(self.model)

        trainer = Trainer(
            model=self.model,
            optimizer=optimizer,
            iterator=self.iterator,
            train_dataset=self.train_data,
            validation_dataset=self.dev_data if self.configuration['early_stopping'] else None,
            cuda_device=gpu_id,
            num_epochs=self.configuration['epochs'],
            validation_metric='+accuracy',
            validation_iterator=self.val_iterator,
            serialization_dir=self.model_dir,
            patience=self.configuration['patience'],
            callbacks =callbacks,
            early_stopping_by_batch=early_stopping_by_batch,
            estimator=estimator
        )
        metrics = trainer.train()
        self.logger.info('metrics: %s' % str(metrics))

    def _save_model(self):
        torch.save(self.model, self.best_model_filepath)

    def _load_model(self):
        self.model = torch.load(self.best_model_filepath)
        self.model.configuration = self.configuration

    def evaluate(self):
        estimator = self._get_estimator(self.model)

        data_type_and_data = {
            # 'train': self.train_data,
            # 'dev': self.dev_data,
            'test': self.test_data
        }
        if self.hard_test_data:
            data_type_and_data['hard_test'] = self.hard_test_data
        for data_type, data in data_type_and_data.items():
            result = estimator.estimate(data)
            self.logger.info('data_type: %s result: %s' % (data_type, result))

    def predict(self, texts: List[str]=None):
        """

        :param texts: 如果texts为None，就是用训练时的测试集
        :return:
        """
        USE_GPU = torch.cuda.is_available()
        if USE_GPU:
            gpu_id = self.configuration['gpu_id']
        else:
            gpu_id = -1
        predictor = pytorch_models.TextAspectInSentimentOutPredictor(self.model, self.val_iterator,
                                                                     self.distinct_categories,
                                                                     self.distinct_polarities,
                                                                     cuda_device=gpu_id)

        data = self.test_data
        result = predictor.predict(data)
        result_final = []
        for i in range(len(result)):
            instance: Instance = data[i]
            sentiment = result[i]
            metadata = instance.fields['sample'].metadata
            label_true = metadata[1]
            category = self.distinct_categories[label_true[0]]
            sentence = metadata[0]

            result_final.append((sentence, category, sentiment))
        return result_final

    def error_analysis(self):
        """

        :return:
        """
        USE_GPU = torch.cuda.is_available()
        if USE_GPU:
            gpu_id = self.configuration['gpu_id']
        else:
            gpu_id = -1
        predictor = pytorch_models.TextAspectInSentimentOutPredictor(self.model, self.val_iterator,
                                                                     self.distinct_categories,
                                                                     self.distinct_polarities,
                                                                     cuda_device=gpu_id)

        data = self.test_data
        result = predictor.predict(data)
        result_final = []
        for i in range(len(result)):
            instance: Instance = data[i]
            sentiment = result[i]
            metadata = instance.fields['sample'].metadata
            label_true = metadata[1]
            category = self.distinct_categories[label_true[0]]
            sentence = metadata[0]

            sentiment_true = self.distinct_polarities[instance.fields['label'].label]
            if sentiment_true == sentiment:
                continue
            result_final.append((sentence, category, sentiment, sentiment_true))
        result_str = ['\t'.join(e) for e in result_final]
        output_filepath = os.path.join(self.model_dir, 'error_analysis.csv')
        file_utils.write_lines(result_str, output_filepath)
        return result_final


class Heat(TextAspectInSentimentOutTrainTemplate):
    """
    2017-CIKM-Aspect-level Sentiment Classification with HEAT (HiErarchical ATtention) Network
    """

    def __init__(self, configuration):
        super().__init__(configuration)

    def _get_aspect_embeddings_dim(self):
        return 32

    def _find_model_function_pure(self):
        return pytorch_models.Heat


class HeatCae(ModelTrainTemplate.ModelTrainTemplate):
    """
    2017-CIKM-Aspect-level Sentiment Classification with HEAT (HiErarchical ATtention) Network
    """

    def __init__(self, configuration):
        super().__init__(configuration)
        self.data_reader = None
        self.train_data = None
        self.dev_data = None
        self.test_data = None
        self.distinct_categories = None
        self.distinct_polarities = None
        self._load_data()

        self.vocab = None
        self._build_vocab()

        self.iterator = None
        self.val_iterator = None
        self._build_iterator()

    def _load_data(self):
        data_filepath = self.base_data_dir + 'data'
        if os.path.exists(data_filepath):
            self.train_data, self.dev_data, self.test_data, self.distinct_categories, self.distinct_polarities \
                = super()._load_object(data_filepath)
        else:
            train_dev_test_data, distinct_categories, distinct_polarities = self.dataset. \
                generate_acd_and_sc_data()
            distinct_polarities_new = []
            for polarity in distinct_polarities:
                if polarity != 'conflict':
                    distinct_polarities_new.append(polarity)
            self.distinct_categories = distinct_categories
            self.distinct_polarities = distinct_polarities_new

            token_indexer = SingleIdTokenIndexer(namespace="tokens")
            aspect_indexer = SingleIdTokenIndexer(namespace='aspect')
            reader = acd_and_sc_data_reader.AcdAndScDatasetReaderHeatCae(
                self.distinct_categories, self.distinct_polarities,
                tokenizer=self._get_word_segmenter(),
                token_indexers={"tokens": token_indexer},
                aspect_indexers={'aspect': aspect_indexer},
                configuration=self.configuration
            )
            self.data_reader = reader

            train_dev_test_data_label_indexed = {}
            for data_type, data in train_dev_test_data.items():
                if data is None:
                    continue
                data_new = []
                for sample in data:
                    sample_new = [sample[0]]
                    labels_new = []
                    for label in sample[1]:
                        aspect = label[0]
                        polarity = label[1]
                        aspect_index = distinct_categories.index(aspect)
                        if polarity == 'conflict':
                            polarity_index = -100
                        else:
                            polarity_index = distinct_polarities_new.index(polarity)
                        labels_new.append((aspect_index, polarity_index))
                    if len(labels_new) != 0:
                        sample_new.append(labels_new)
                        data_new.append(sample_new)
                train_dev_test_data_label_indexed[data_type] = data_new
            self.train_data = reader.read(train_dev_test_data_label_indexed['train'])
            self.dev_data = reader.read(train_dev_test_data_label_indexed['dev'])
            self.test_data = reader.read(train_dev_test_data_label_indexed['test'])
            data = [self.train_data, self.dev_data, self.test_data, self.distinct_categories,
                    self.distinct_polarities]
            super()._save_object(data_filepath, data)

    def _build_vocab(self):
        if self.configuration['train']:
            vocab_file_path = self.base_data_dir + 'vocab'
            if os.path.exists(vocab_file_path):
                self.vocab = super()._load_object(vocab_file_path)
            else:
                data = self.train_data + self.dev_data + self.test_data
                self.vocab = Vocabulary.from_instances(data, max_vocab_size=sys.maxsize)
                super()._save_object(vocab_file_path, self.vocab)
            self.model_meta_data['vocab'] = self.vocab
        else:
            self.vocab = self.model_meta_data['vocab']

    def _build_iterator(self):
        self.iterator = BucketIterator(batch_size=self.configuration['batch_size'],
                                       sorting_keys=[("tokens", "num_tokens")],
                                       )
        self.iterator.index_with(self.vocab)
        self.val_iterator = BasicIterator(batch_size=sys.maxsize)
        self.val_iterator.index_with(self.vocab)

    def _print_args(self, model):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in model.parameters():
            n_params = torch.prod(torch.tensor(p.shape)).item()
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        self.logger.info('n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        self.logger.info('> training arguments:')
        for arg in self.configuration.keys():
            self.logger.info('>>> {0}: {1}'.format(arg, self.configuration[arg]))

    def _find_model_function(self):
        embedding_dim = self.configuration['embed_size']
        embedding_matrix_filepath = self.base_data_dir + 'embedding_matrix'
        if os.path.exists(embedding_matrix_filepath):
            embedding_matrix = super()._load_object(embedding_matrix_filepath)
        else:
            embedding_filepath = self.configuration['embedding_filepath']
            embedding_matrix = embedding._read_embeddings_from_text_file(embedding_filepath, embedding_dim,
                                                                         self.vocab, namespace='tokens')
            super()._save_object(embedding_matrix_filepath, embedding_matrix)
        token_embedding = Embedding(num_embeddings=self.vocab.get_vocab_size(namespace='tokens'),
                                    embedding_dim=embedding_dim, padding_index=0, vocab_namespace='tokens',
                                    trainable=False, weight=embedding_matrix)
        # the embedder maps the input tokens to the appropriate embedding matrix
        word_embedder: TextFieldEmbedder = BasicTextFieldEmbedder({"tokens": token_embedding})

        aspect_embedding = Embedding(num_embeddings=self.vocab.get_vocab_size(namespace='aspect'),
                                    embedding_dim=32, padding_index=0)
        aspect_embedder: TextFieldEmbedder = BasicTextFieldEmbedder({"aspect": aspect_embedding},
                                                                    # we'll be ignoring masks so we'll need to set this to True
                                                                    allow_unmatched_keys=True)

        model = pytorch_models.HeatCae(
            word_embedder,
            aspect_embedder,
            self.distinct_categories,
            self.distinct_polarities,
            self.vocab,
            self.configuration
        )
        self._print_args(model)
        return model

    def _inner_train(self):
        self.model = self._find_model_function()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=0.00001)
        USE_GPU = torch.cuda.is_available()
        trainer = Trainer(
            model=self.model,
            optimizer=optimizer,
            iterator=self.iterator,
            train_dataset=self.train_data,
            validation_dataset=self.dev_data,
            cuda_device=0 if USE_GPU else -1,
            num_epochs=100,
            validation_metric='+accuracy',
            validation_iterator=self.val_iterator,
            serialization_dir=self.model_dir,
            patience=10
        )
        metrics = trainer.train()
        self.logger.info('metrics: %s' % str(metrics))

    def _save_model(self):
        torch.save(self.model, self.best_model_filepath)

    def _load_model(self):
        self.model = torch.load(self.best_model_filepath)
        self.model.configuration = self.configuration

    def evaluate(self):
        estimator = pytorch_models.HeatEstimator(self.model, self.val_iterator,
                                                self.distinct_categories,
                                                self.distinct_polarities)

        result = estimator.estimate(self.test_data)
        self.logger.info(result)


class HeatCaeM(TextInAllAspectSentimentOutTrainTemplate):
    """
    2017-CIKM-Aspect-level Sentiment Classification with HEAT (HiErarchical ATtention) Network
    """

    def __init__(self, configuration):
        super().__init__(configuration)

    def _get_aspect_embeddings_dim(self):
        return 32

    def _find_model_function_pure(self):
        return pytorch_models.HeatCaeM


class HeatCae2(ModelTrainTemplate.ModelTrainTemplate):
    """
    2017-CIKM-Aspect-level Sentiment Classification with HEAT (HiErarchical ATtention) Network
    """

    def __init__(self, configuration):
        super().__init__(configuration)
        self.data_reader = None
        self.train_data = None
        self.dev_data = None
        self.test_data = None
        self.distinct_categories = None
        self.distinct_polarities = None
        self._load_data()

        self.vocab = None
        self._build_vocab()

        self.iterator = None
        self.val_iterator = None
        self._build_iterator()

    def _load_data(self):
        data_filepath = self.base_data_dir + 'data'
        if os.path.exists(data_filepath):
            self.train_data, self.dev_data, self.test_data, self.distinct_categories, self.distinct_polarities \
                = super()._load_object(data_filepath)
        else:
            train_dev_test_data, distinct_categories, distinct_polarities = self.dataset. \
                generate_acd_and_sc_data()
            distinct_polarities_new = []
            for polarity in distinct_polarities:
                if polarity != 'conflict':
                    distinct_polarities_new.append(polarity)
            self.distinct_categories = distinct_categories
            self.distinct_polarities = distinct_polarities_new

            token_indexer = SingleIdTokenIndexer(namespace="tokens")
            position_indexer = SingleIdTokenIndexer(namespace='position')
            aspect_indexer = SingleIdTokenIndexer(namespace='aspect')
            reader = acd_and_sc_data_reader.TextTargetAspectInAllAspectTargetSentimentOut(
                self.distinct_categories, self.distinct_polarities,
                tokenizer=self._get_word_segmenter(),
                token_indexers={"tokens": token_indexer},
                position_indexers={'position': position_indexer},
                aspect_indexer={'aspect': aspect_indexer},
                configuration=self.configuration
            )
            self.data_reader = reader

            train_dev_test_data_label_indexed = {}
            for data_type, data in train_dev_test_data.items():
                if data is None:
                    continue
                data_new = []
                for sample in data:
                    sample_new = [sample[0]]
                    labels_new = []
                    for label in sample[1]:
                        aspect = label[0]
                        polarity = label[1]
                        aspect_index = distinct_categories.index(aspect)
                        if polarity == 'conflict':
                            polarity_index = -100
                        else:
                            polarity_index = distinct_polarities_new.index(polarity)
                        labels_new.append((aspect_index, polarity_index))
                    if len(labels_new) != 0:
                        sample_new.append(labels_new)
                        data_new.append(sample_new)
                train_dev_test_data_label_indexed[data_type] = data_new
            self.train_data = reader.read(train_dev_test_data_label_indexed['train'])
            self.dev_data = reader.read(train_dev_test_data_label_indexed['dev'])
            self.test_data = reader.read(train_dev_test_data_label_indexed['test'])
            data = [self.train_data, self.dev_data, self.test_data, self.distinct_categories,
                    self.distinct_polarities]
            super()._save_object(data_filepath, data)

    def _build_vocab(self):
        if self.configuration['train']:
            vocab_file_path = self.base_data_dir + 'vocab'
            if os.path.exists(vocab_file_path):
                self.vocab = super()._load_object(vocab_file_path)
            else:
                data = self.train_data + self.dev_data + self.test_data
                self.vocab = Vocabulary.from_instances(data, max_vocab_size=sys.maxsize)
                super()._save_object(vocab_file_path, self.vocab)
            self.model_meta_data['vocab'] = self.vocab
        else:
            self.vocab = self.model_meta_data['vocab']

    def _build_iterator(self):
        self.iterator = BucketIterator(batch_size=self.configuration['batch_size'],
                                       sorting_keys=[("tokens", "num_tokens")],
                                       )
        self.iterator.index_with(self.vocab)
        self.val_iterator = BasicIterator(batch_size=sys.maxsize)
        self.val_iterator.index_with(self.vocab)

    def _print_args(self, model):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in model.parameters():
            n_params = torch.prod(torch.tensor(p.shape)).item()
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        self.logger.info('n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        self.logger.info('> training arguments:')
        for arg in self.configuration.keys():
            self.logger.info('>>> {0}: {1}'.format(arg, self.configuration[arg]))

    def _find_model_function(self):
        embedding_dim = self.configuration['embed_size']
        embedding_matrix_filepath = self.base_data_dir + 'embedding_matrix'
        if os.path.exists(embedding_matrix_filepath):
            embedding_matrix = super()._load_object(embedding_matrix_filepath)
        else:
            embedding_filepath = self.configuration['embedding_filepath']
            embedding_matrix = embedding._read_embeddings_from_text_file(embedding_filepath, embedding_dim,
                                                                         self.vocab, namespace='tokens')
            super()._save_object(embedding_matrix_filepath, embedding_matrix)
        token_embedding = Embedding(num_embeddings=self.vocab.get_vocab_size(namespace='tokens'),
                                    embedding_dim=embedding_dim, padding_index=0, vocab_namespace='tokens',
                                    trainable=False, weight=embedding_matrix)
        # the embedder maps the input tokens to the appropriate embedding matrix
        word_embedder: TextFieldEmbedder = BasicTextFieldEmbedder({"tokens": token_embedding})

        aspect_embedding = Embedding(num_embeddings=self.vocab.get_vocab_size(namespace='aspect'),
                                    embedding_dim=32, padding_index=0)
        aspect_embedder: TextFieldEmbedder = BasicTextFieldEmbedder({"aspect": aspect_embedding},
                                                                    # we'll be ignoring masks so we'll need to set this to True
                                                                    allow_unmatched_keys=True)

        model = pytorch_models.HeatCaeM(
            word_embedder,
            aspect_embedder,
            self.distinct_categories,
            self.distinct_polarities,
            self.vocab,
            self.configuration
        )
        self._print_args(model)
        return model

    def _inner_train(self):
        self.model = self._find_model_function()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=0.00001)
        USE_GPU = torch.cuda.is_available()
        trainer = Trainer(
            model=self.model,
            optimizer=optimizer,
            iterator=self.iterator,
            train_dataset=self.train_data,
            validation_dataset=self.dev_data,
            cuda_device=0 if USE_GPU else -1,
            num_epochs=100,
            validation_metric='+accuracy',
            validation_iterator=self.val_iterator,
            serialization_dir=self.model_dir,
            patience=10
        )
        metrics = trainer.train()
        self.logger.info('metrics: %s' % str(metrics))

    def _save_model(self):
        torch.save(self.model, self.best_model_filepath)

    def _load_model(self):
        self.model = torch.load(self.best_model_filepath)
        self.model.configuration = self.configuration

    def evaluate(self):
        estimator = pytorch_models.CaeEstimator(self.model, self.val_iterator,
                                                self.distinct_categories,
                                                self.distinct_polarities)

        result = estimator.estimate(self.test_data)
        self.logger.info(result)


class AtaeLstm(TextAspectInSentimentOutTrainTemplate):
    """
    2016-emnlp-Attention-based LSTM for Aspect-level Sentiment Classification
    """

    def __init__(self, configuration):
        super().__init__(configuration)

    def _find_model_function_pure(self):
        return pytorch_models.AtaeLstm


class TextAspectInSentimentOutTrainTemplateBert(TextAspectInSentimentOutTrainTemplate):
    """
    2016-emnlp-Attention-based LSTM for Aspect-level Sentiment Classification
    """

    def __init__(self, configuration):
        self.bert_file_path = configuration['bert_file_path']
        self.bert_vocab_file_path = configuration['bert_vocab_file_path']
        self.max_len = configuration['max_len']
        super().__init__(configuration)

    def _get_bert_word_segmenter(self):
        token_dict = {}
        for line in file_utils.read_all_lines(self.bert_vocab_file_path):
            token = line.strip()
            token_dict[token] = len(token_dict)

        result = bert_tokenizer.EnglishTokenizer(token_dict)
        return result

    def _get_data_reader(self):
        bert_tokenizer = BertTokenizer.from_pretrained(self.bert_vocab_file_path, do_lower_case=True)
        bert_token_indexer = WordpieceIndexer(vocab=bert_tokenizer.vocab,
                                              wordpiece_tokenizer=bert_tokenizer.wordpiece_tokenizer.tokenize,
                                              namespace="tokens",
                                              use_starting_offsets=False,
                                              max_pieces=self.max_len,
                                              do_lowercase=True,
                                              never_lowercase=None,
                                              start_tokens=None,
                                              end_tokens=None,
                                              separator_token="[SEP]",
                                              truncate_long_sequences=True)

        aspect_indexer = SingleIdTokenIndexer(namespace='aspect')
        reader = acd_and_sc_data_reader.TextAspectInSentimentOutDatasetReaderBert(
            self.distinct_categories, self.distinct_polarities,
            tokenizer=bert_tokenizer,
            token_indexers={"tokens": bert_token_indexer},
            aspect_indexers={'aspect': aspect_indexer},
            configuration=self.configuration
        )
        return reader

    def _find_model_function_pure(self):
        return pytorch_models.TextAspectInSentimentOutModelBert

    def _get_bert_word_embedder(self):
        pretrained_model = self.bert_file_path
        bert_model = PretrainedBertModel.load(pretrained_model, cache_model=False)
        for param in bert_model.parameters():
            param.requires_grad = self.configuration['train_bert']
        bert_embedder = BertEmbedder(bert_model=bert_model, top_layer_only=True)

        bert_word_embedder: TextFieldEmbedder = BasicTextFieldEmbedder({"tokens": bert_embedder},
                                                                       # we'll be ignoring masks so we'll need to set this to True
                                                                       allow_unmatched_keys=True)
        return bert_word_embedder

    def _find_model_function(self):
        position_embedding = Embedding(num_embeddings=self.vocab.get_vocab_size(namespace='position'),
                                    embedding_dim=25, padding_index=0)
        position_embedder: TextFieldEmbedder = BasicTextFieldEmbedder({"position": position_embedding},
                                                                    # we'll be ignoring masks so we'll need to set this to True
                                                                    allow_unmatched_keys=True)

        bert_word_embedder = self._get_bert_word_embedder()

        aspect_embedding_matrix = None
        if self._init_aspect_embeddings_from_word_embeddings():
            embedding_filepath = self.configuration['embedding_filepath']
            aspect_embedding_matrix = embedding._read_embeddings_from_text_file(embedding_filepath, embedding_dim,
                                                                                self.vocab, namespace='aspect')
        aspect_embedding = Embedding(num_embeddings=self.vocab.get_vocab_size(namespace='aspect'),
                                     embedding_dim=self._get_aspect_embeddings_dim(), padding_index=0,
                                     trainable=True, weight=aspect_embedding_matrix)
        aspect_embedder: TextFieldEmbedder = BasicTextFieldEmbedder({"aspect": aspect_embedding},
                                                                    # we'll be ignoring masks so we'll need to set this to True
                                                                    allow_unmatched_keys=True)

        model_function = self._find_model_function_pure()

        model = model_function(
            bert_word_embedder,
            aspect_embedder,
            self.distinct_categories,
            self.distinct_polarities,
            self.vocab,
            self.configuration
        )

        self._print_args(model)
        model = model.to(self.configuration['device'])
        return model

    def _get_optimizer(self, model):
        _params = filter(lambda p: p.requires_grad, model.parameters())
        return optim.Adam(_params, lr=self.configuration['learning_rate_in_bert'], weight_decay=0.00001)


class Lstm(TextAspectInSentimentOutTrainTemplate):
    """
    2016-emnlp-Attention-based LSTM for Aspect-level Sentiment Classification
    """

    def __init__(self, configuration):
        super().__init__(configuration)

    def _find_model_function_pure(self):
        return pytorch_models.Lstm


class BilstmAttn(TextAspectInSentimentOutTrainTemplate):
    """
    2016-emnlp-Attention-based LSTM for Aspect-level Sentiment Classification
    """

    def __init__(self, configuration):
        super().__init__(configuration)

    def _find_model_function_pure(self):
        return pytorch_models.BilstmAttn


class TextCnn(TextAspectInSentimentOutTrainTemplate):
    """
    2016-emnlp-Attention-based LSTM for Aspect-level Sentiment Classification
    """

    def __init__(self, configuration):
        super().__init__(configuration)

    def _find_model_function_pure(self):
        return pytorch_models.TextCNN


class GCAE(TextAspectInSentimentOutTrainTemplate):
    """
    2018-Aspect Based Sentiment Analysis with Gated Convolutional Networks
    """

    def __init__(self, configuration):
        super().__init__(configuration)

    def _init_aspect_embeddings_from_word_embeddings(self):
        return True

    def _find_model_function_pure(self):
        return pytorch_models.GCAE
