import json
import numpy as np

from sklearn.preprocessing import MultiLabelBinarizer
from keras.preprocessing import text as keras_text
from keras.preprocessing import sequence as keras_seq

from nlp_tasks.absa.entities import ModelTrainTemplate
from nlp_tasks.utils import word_processor
from nlp_tasks.utils import tokenizers
from nlp_tasks.absa.preprocess import generate_word_index
from nlp_tasks.absa.utils import embedding_utils
from nlp_tasks.absa.utils import cv_utils
from nlp_tasks.utils import visualizer
from nlp_tasks.absa.models import pytorch_models
from nlp_tasks.absa.models import keras_models


class ContextualizedAspectEmbedding(ModelTrainTemplate.ModelTrainTemplate):
    """

    """

    def __init__(self, configuration):
        super().__init__(configuration)
        self.model_data = self._transform_data_for_model()

    def _find_model_function(self):
        model_name = self.configuration['model_name']
        if model_name == 'cae':
            model_fun = keras_models.cae
        elif model_name == 'cae2':
            model_fun = keras_models.cae2
        elif model_name == 'aae_without_share':
            model_fun = keras_models.aae_without_share
        elif model_name == 'aae_without_aae':
            model_fun = keras_models.aae_without_aae
        elif model_name == 'aae_without_aae_share':
            model_fun = keras_models.aae_without_aae_share
        else:
            return Exception('unsuported model')
        return model_fun

    def _transform_data_for_model(self):
        train_dev_test_data, distinct_categories, distinct_polarities = self.dataset.\
            generate_acd_and_sc_data()
        self.model_meta_data['distinct_categories'] = distinct_categories
        self.model_meta_data['distinct_polarities'] = distinct_polarities
        if 'dev' not in train_dev_test_data or train_dev_test_data['dev'] is None:
            train_dev_test_data['dev'] = train_dev_test_data['test']
        train_dev_test_data_for_nn = {}

        category_mlb = MultiLabelBinarizer(classes=distinct_categories)
        category_mlb.fit(None)
        sentiment_mlb = MultiLabelBinarizer(classes=distinct_polarities)
        sentiment_mlb.fit(None)
        for data_type, data in train_dev_test_data.items():
            if data is None:
                continue
            category_label = [[category_polarity_pair[0] for category_polarity_pair in sample[1]] for sample in data]
            category_label_binary = category_mlb.transform(category_label)
            train_dev_test_data_for_nn['%s_%s_category' % (data_type, 'y')] = category_label_binary
            category_label_binary_multi_output = [category_label_binary[:, i] for i in range(len(distinct_categories))]
            train_dev_test_data_for_nn[
                '%s_%s_category' % (data_type, 'y_multi_output')] = category_label_binary_multi_output

            sentiment_label_binary = []
            for i in range(len(distinct_categories)):
                one_category_sentiment_label_binary = []
                target_category = distinct_categories[i]
                for j in range(len(category_label)):
                    categories = category_label[j]
                    if target_category in categories:
                        for one_label in data[j][1]:
                            if target_category in one_label:
                                target_polarity = one_label[1]
                        polarity_binary = sentiment_mlb.transform([[target_polarity]])[0]
                        one_category_sentiment_label_binary.append(polarity_binary)
                    else:
                        polarity_binary = np.array([0] * len(distinct_polarities))
                        one_category_sentiment_label_binary.append(polarity_binary)
                sentiment_label_binary.append(np.array(one_category_sentiment_label_binary))
            train_dev_test_data_for_nn['%s_%s_sentiment' % (data_type, 'y_multi_output')] = \
                sentiment_label_binary
            train_dev_test_data_for_nn['%s_%s' % (data_type, 'y_multi_output')] = \
                train_dev_test_data_for_nn['%s_%s_category' % (data_type, 'y_multi_output')] \
                + train_dev_test_data_for_nn['%s_%s_sentiment' % (data_type, 'y_multi_output')]

        word_segmenter = super()._get_word_segmenter()
        texts = []
        texts.extend([sample[0] for sample in train_dev_test_data['train']])
        texts.extend([sample[0] for sample in train_dev_test_data['dev']])
        texts.extend([sample[0] for sample in train_dev_test_data['test']])
        texts_word = [word_segmenter(text) for text in texts]
        max_len = max([len(text) for text in texts_word])
        self.logger.info('max_len: %d' % max_len)
        max_len = min(max_len, 307)
        self.model_meta_data['max_len'] = max_len
        self.logger.info('real max_len: %d' % max_len)

        texts_englike = [' '.join(text) for text in texts_word]
        tokenizer = super()._get_keras_tokenizer(texts_englike)
        for data_type, data in train_dev_test_data.items():
            if data is None:
                continue
            contents_englike = [' '.join(word_segmenter(sample[0])) for sample in data]
            x = tokenizer.texts_to_sequences(contents_englike)
            x = keras_seq.pad_sequences(x, maxlen=max_len)
            train_dev_test_data_for_nn['%s_%s' % (data_type, 'x')] = x

        train_x, train_y_mo = train_dev_test_data_for_nn['train_x'], train_dev_test_data_for_nn['train_y_multi_output']
        dev_x, dev_y_mo = train_dev_test_data_for_nn['dev_x'], train_dev_test_data_for_nn['dev_y_multi_output']
        test_x, test_y_mo = train_dev_test_data_for_nn['test_x'], train_dev_test_data_for_nn['test_y_multi_output']
        result = {
            'train': (train_x, train_y_mo),
            'dev': (dev_x, dev_y_mo),
            'test': (test_x, test_y_mo),
        }
        return result

    def _inner_train(self):
        keras_tokenizer = super()._get_keras_tokenizer(None)
        word_index = keras_tokenizer.word_index
        embed_size = self.configuration['embed_size']
        embedding_matrix = super()._build_embedding_matrix(self.configuration['embedding_filepath'], word_index,
                                                           embed_size)
        model_fun = self._find_model_function()
        max_len = self.model_meta_data['max_len']
        distinct_categories = self.model_meta_data['distinct_categories']
        distinct_polarities = self.model_meta_data['distinct_polarities']
        model = model_fun(max_len, len(word_index) + 1, embed_size, embedding_matrix, len(distinct_categories),
                          len(distinct_polarities))

        epochs = self.configuration['epochs']
        monitor = 'acc_sc_visual'
        pretrain_model_path = None
        batch_size = self.configuration['batch_size']
        patience = self.configuration['patience']
        train_x, train_y = self.model_data['train']
        dev_x, dev_y = self.model_data['dev']
        test_x, test_y = self.model_data['test']
        callback_data = {
            'train': (train_x, train_y),
            'val': (dev_x, dev_y),
            'test': (test_x, test_y),
        }
        threshold = self.configuration['threshold']
        customized_callbacks = [cv_utils.AcdAndScMetrics(callback_data, self.logger,
                                                         category_num=len(distinct_categories),
                                                         polarity_num=len(distinct_polarities),
                                                         threshold=threshold)]
        best_model_filepath = self.best_model_filepath
        model_log_dir = self.model_log_dir
        cv_utils.cv_provide_train_val(train_x, train_y, dev_x, dev_y, epochs, monitor,
                                      pretrain_model_path, batch_size, best_model_filepath, None, model,
                                      model_log_dir, customized_callbacks, patience)


class GcnForAspectCategorySentimentAnalysis(ModelTrainTemplate.ModelTrainTemplate):
    """
    针对Semeval2014Task4Rest/MAMSACSA，
    <sentence>
		<text>Though the service might be a little slow, the waitresses are very friendly.</text>
		<aspectCategories>
			<aspectCategory category="service" polarity="negative"/>
			<aspectCategory category="staff" polarity="positive"/>
		</aspectCategories>
	</sentence>
    """

    def __init__(self, configuration):
        super().__init__(configuration)
        self.model_data = self._transform_data_for_model()

    def _find_model_function(self):
        model_name = self.configuration['model_name']
        if model_name == 'cae':
            model_fun = keras_models.cae
        elif model_name == 'cae2':
            model_fun = keras_models.cae2
        elif model_name == 'aae_without_share':
            model_fun = keras_models.aae_without_share
        elif model_name == 'aae_without_aae':
            model_fun = keras_models.aae_without_aae
        elif model_name == 'aae_without_aae_share':
            model_fun = keras_models.aae_without_aae_share
        else:
            return Exception('unsuported model')
        return model_fun

    def _transform_data_for_model(self):
        train_dev_test_data, distinct_categories, distinct_polarities = self.dataset.\
            generate_acd_and_sc_data()
        self.model_meta_data['distinct_categories'] = distinct_categories
        self.model_meta_data['distinct_polarities'] = distinct_polarities
        if 'dev' not in train_dev_test_data or train_dev_test_data['dev'] is None:
            train_dev_test_data['dev'] = train_dev_test_data['test']
        train_dev_test_data_for_nn = {}

        category_mlb = MultiLabelBinarizer(classes=distinct_categories)
        category_mlb.fit(None)
        sentiment_mlb = MultiLabelBinarizer(classes=distinct_polarities)
        sentiment_mlb.fit(None)
        for data_type, data in train_dev_test_data.items():
            if data is None:
                continue
            category_label = [[category_polarity_pair[0] for category_polarity_pair in sample[1]] for sample in data]
            category_label_binary = category_mlb.transform(category_label)
            train_dev_test_data_for_nn['%s_%s_category' % (data_type, 'y')] = category_label_binary
            category_label_binary_multi_output = [category_label_binary[:, i] for i in range(len(distinct_categories))]
            train_dev_test_data_for_nn[
                '%s_%s_category' % (data_type, 'y_multi_output')] = category_label_binary_multi_output

            sentiment_label_binary = []
            for i in range(len(distinct_categories)):
                one_category_sentiment_label_binary = []
                target_category = distinct_categories[i]
                for j in range(len(category_label)):
                    categories = category_label[j]
                    if target_category in categories:
                        for one_label in data[j][1]:
                            if target_category in one_label:
                                target_polarity = one_label[1]
                        polarity_binary = sentiment_mlb.transform([[target_polarity]])[0]
                        one_category_sentiment_label_binary.append(polarity_binary)
                    else:
                        polarity_binary = np.array([0] * len(distinct_polarities))
                        one_category_sentiment_label_binary.append(polarity_binary)
                sentiment_label_binary.append(np.array(one_category_sentiment_label_binary))
            train_dev_test_data_for_nn['%s_%s_sentiment' % (data_type, 'y_multi_output')] = \
                sentiment_label_binary
            train_dev_test_data_for_nn['%s_%s' % (data_type, 'y_multi_output')] = \
                train_dev_test_data_for_nn['%s_%s_category' % (data_type, 'y_multi_output')] \
                + train_dev_test_data_for_nn['%s_%s_sentiment' % (data_type, 'y_multi_output')]

        word_segmenter = super()._get_word_segmenter()
        texts = []
        texts.extend([sample[0] for sample in train_dev_test_data['train']])
        texts.extend([sample[0] for sample in train_dev_test_data['dev']])
        texts.extend([sample[0] for sample in train_dev_test_data['test']])
        texts_word = [word_segmenter(text) for text in texts]
        max_len = max([len(text) for text in texts_word])
        self.logger.info('max_len: %d' % max_len)
        max_len = min(max_len, 307)
        self.model_meta_data['max_len'] = max_len
        self.logger.info('real max_len: %d' % max_len)

        texts_englike = [' '.join(text) for text in texts_word]
        tokenizer = super()._get_keras_tokenizer(texts_englike)
        for data_type, data in train_dev_test_data.items():
            if data is None:
                continue
            contents_englike = [' '.join(word_segmenter(sample[0])) for sample in data]
            x = tokenizer.texts_to_sequences(contents_englike)
            x = keras_seq.pad_sequences(x, maxlen=max_len)
            train_dev_test_data_for_nn['%s_%s' % (data_type, 'x')] = x

        train_x, train_y_mo = train_dev_test_data_for_nn['train_x'], train_dev_test_data_for_nn['train_y_multi_output']
        dev_x, dev_y_mo = train_dev_test_data_for_nn['dev_x'], train_dev_test_data_for_nn['dev_y_multi_output']
        test_x, test_y_mo = train_dev_test_data_for_nn['test_x'], train_dev_test_data_for_nn['test_y_multi_output']
        result = {
            'train': (train_x, train_y_mo),
            'dev': (dev_x, dev_y_mo),
            'test': (test_x, test_y_mo),
        }
        return result