# -*- coding: utf-8 -*-


import copy
import re
from typing import *
from overrides import overrides
import pickle
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder, PytorchSeq2VecWrapper
from allennlp.nn.util import get_text_field_mask
from allennlp.models import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.data.fields import TextField, MetadataField, ArrayField, ListField, LabelField, MultiLabelField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data import Instance
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.nn import util as nn_util
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.dataset_readers import DatasetReader
import torch.nn.functional as F
from allennlp.training import metrics
from allennlp.models import BasicClassifier
from allennlp.data.iterators import DataIterator
from tqdm import tqdm
import spacy
from nltk.corpus import stopwords
from benepar.spacy_plugin import BeneparComponent

english_stop_words = stopwords.words('english')
english_stop_words.extend([',', '.', '?', ';', '-', ':', '\'', '"', '(', ')', '!'])

from nlp_tasks.utils import corenlp_factory
from nlp_tasks.utils import create_graph
from nlp_tasks.utils import my_corenlp
from nlp_tasks.absa.sentence_analysis.constituency_parser import ConstituencyTreeNode


class AcdAndScDatasetReaderCae(DatasetReader):
    def __init__(self, categories: List[str], polarities: List[str],
                 tokenizer: Callable[[str], List[str]] = lambda x: x.split(),
                 token_indexers: Dict[str, TokenIndexer] = None,
                 position_indexers: Dict[str, TokenIndexer] = None,
                 core_nlp: my_corenlp.StanfordCoreNLP=None,
                 configuration=None) -> None:
        super().__init__(lazy=False)
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer(namespace="tokens")}
        self.position_indexers = position_indexers or {"position": SingleIdTokenIndexer(namespace='position')}
        self.categories = categories
        self.polarities = polarities
        self.spacy_nlp = spacy.load("en_core_web_sm")
        self.core_nlp = core_nlp
        self.configuration = configuration

    def _build_graph(self, text):
        graph = create_graph.create_dependency_graph_for_dgl(text, self.spacy_nlp, None)
        return graph

    @overrides
    def text_to_instance(self, sample: list) -> Instance:
        fields = {}

        text: str = sample[0].strip()

        words = self.tokenizer(text)
        sample.append(words)

        graph = self._build_graph(text)
        sample.append(graph)

        tokens = [Token(word) for word in words]

        sentence_field = TextField(tokens, self.token_indexers)
        fields['tokens'] = sentence_field

        position = [Token(str(i)) for i in range(len(tokens))]
        position_field = TextField(position, self.position_indexers)
        fields['position'] = position_field

        category_labels = [0] * len(self.categories)
        polarity_labels = [-100] * len(self.categories)
        if len(sample) > 1:
            labels: list = sample[1]
            for label in labels:
                category_labels[label[0]] = 1
                polarity_labels[label[0]] = label[1]
        label_field = ArrayField(np.array(category_labels + polarity_labels))
        fields["label"] = label_field
        polarity_mask = [1 if polarity_labels[i] != -100 else 0 for i in range(len(self.categories))]
        polarity_mask_field = ArrayField(np.array(polarity_mask))
        fields['polarity_mask'] = polarity_mask_field
        sample_field = MetadataField(sample)
        fields["sample"] = sample_field
        return Instance(fields)

    @overrides
    def _read(self, samples: list) -> Iterator[Instance]:
        for sample in samples:
            yield self.text_to_instance(sample)


class TextInAllAspectSentimentOut(DatasetReader):
    def __init__(self, categories: List[str], polarities: List[str],
                 tokenizer: Callable[[str], List[str]] = lambda x: x.split(),
                 token_indexers: Dict[str, TokenIndexer] = None,
                 position_indexers: Dict[str, TokenIndexer] = None,
                 aspect_indexers: Dict[str, TokenIndexer] = None,
                 core_nlp: my_corenlp.StanfordCoreNLP=None,
                 configuration=None) -> None:
        super().__init__(lazy=False)
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer(namespace="tokens")}
        self.position_indexers = position_indexers or {"position": SingleIdTokenIndexer(namespace='position')}
        self.aspect_indexers = aspect_indexers or {"aspect": SingleIdTokenIndexer(namespace='aspect')}
        self.categories = categories
        self.polarities = polarities
        self.spacy_nlp = spacy.load("en_core_web_sm")
        self.core_nlp = core_nlp
        self.configuration = configuration

    def _build_graph(self, text):
        graph = create_graph.create_dependency_graph_for_dgl(text, self.spacy_nlp, None)
        return graph

    @overrides
    def text_to_instance(self, sample: list) -> Instance:
        fields = {}

        text: str = sample[0].strip()

        words = self.tokenizer(text)
        if 'max_word_len' in self.configuration:
            words = words[: self.configuration['max_word_len']]
        sample.append(words)

        graph = self._build_graph(text)
        sample.append(graph)

        tokens = [Token(word) for word in words]

        sentence_field = TextField(tokens, self.token_indexers)
        fields['tokens'] = sentence_field

        position = [Token(str(i)) for i in range(len(tokens))]
        position_field = TextField(position, self.position_indexers)
        fields['position'] = position_field

        aspects = [Token(category) for category in self.categories]
        aspect_field = TextField(aspects, self.aspect_indexers)
        fields['aspects'] = aspect_field

        category_labels = [0] * len(self.categories)
        polarity_labels = [-100] * len(self.categories)
        total_labels = []
        if len(sample) > 1:
            labels: list = sample[1]
            for label in labels:
                category_labels[label[0]] = 1
                polarity_labels[label[0]] = label[1]
        for i in range(len(self.categories)):
            if polarity_labels[i] == -100:
                total_labels.append(0)
            else:
                total_labels.append(polarity_labels[i] + category_labels[i])

        label_field = ArrayField(np.array(category_labels + polarity_labels + total_labels))
        fields["label"] = label_field
        polarity_mask = [1 if polarity_labels[i] != -100 else 0 for i in range(len(self.categories))]
        polarity_mask_field = ArrayField(np.array(polarity_mask))
        fields['polarity_mask'] = polarity_mask_field

        # stop_word_labels = [1 if word in english_stop_words else 0 for word in words]
        # stop_word_num = sum(stop_word_labels)
        # stop_word_labels = [label / stop_word_num for label in stop_word_labels]
        # sample.append(stop_word_labels)

        sample_field = MetadataField(sample)
        fields["sample"] = sample_field
        return Instance(fields)

    @overrides
    def _read(self, samples: list) -> Iterator[Instance]:
        acd_sc_mode = self.configuration['acd_sc_mode']
        if acd_sc_mode == 'multi-multi':
            for sample in samples:
                yield self.text_to_instance(sample)
        elif acd_sc_mode == 'multi-single':
            for sample in samples:
                text = sample[0]
                labels = sample[1]
                for i in range(len(labels)):
                    labels_copy = [list(e) for e in copy.deepcopy(labels)]
                    for j, label in enumerate(labels_copy):
                        if j != i:
                            labels_copy[j][1] = -100
                    yield self.text_to_instance([text, labels_copy])
        elif acd_sc_mode == 'single-single':
            raise NotImplementedError('single-single')


class TextInAllAspectSentimentOutSentenceConstituency(DatasetReader):
    def __init__(self, categories: List[str], polarities: List[str],
                 tokenizer: Callable[[str], List[str]]=lambda x: x.split(),
                 token_indexers: Dict[str, TokenIndexer]=None,
                 position_indexers: Dict[str, TokenIndexer]=None,
                 aspect_indexers: Dict[str, TokenIndexer]=None,
                 sentence_constituency_indexer: Dict[str, TokenIndexer]=None,
                 core_nlp: my_corenlp.StanfordCoreNLP=None,
                 configuration=None) -> None:
        super().__init__(lazy=False)
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer(namespace="tokens")}
        self.position_indexers = position_indexers or {"position": SingleIdTokenIndexer(namespace='position')}
        self.aspect_indexers = aspect_indexers or {"aspect": SingleIdTokenIndexer(namespace='aspect')}
        self.sentence_constituency_indexer = sentence_constituency_indexer
        self.categories = categories
        self.polarities = polarities
        self.spacy_nlp = spacy.load("en_core_web_sm")
        self.spacy_nlp.add_pipe(BeneparComponent('benepar_en'))
        self.core_nlp = core_nlp
        self.configuration = configuration

    def _build_graph(self, tree: ConstituencyTreeNode):
        graph = create_graph.create_sentence_constituency_graph_for_dgl(tree)
        return graph

    def _build_graph_with_dotted_line(self, tree: ConstituencyTreeNode):
        graph = create_graph.create_sentence_constituency_graph_for_dgl_with_dotted_line(tree)
        return graph

    @overrides
    def text_to_instance(self, sample: list) -> Instance:
        fields = {}

        text: str = sample['text'].strip()
        text = re.sub('\\s+', ' ', text)
        sample['text_clean'] = text

        tree = ConstituencyTreeNode.parse_using_spacy(self.spacy_nlp, text)
        inner_nodes = ConstituencyTreeNode.get_all_inner_nodes(tree)
        inner_nodes.sort(key=lambda x: x.node_id)
        nodes_labels = [e.labels[0] for e in inner_nodes]
        sample['tree'] = tree
        sample['inner_nodes'] = inner_nodes
        sample['nodes_labels'] = nodes_labels

        graph = self._build_graph(tree)
        sample['graph'] = graph

        graph_with_dotted_line = self._build_graph_with_dotted_line(tree)
        sample['graph_with_dotted_line'] = graph_with_dotted_line

        words = self.tokenizer(text)
        sample['words'] = words
        tokens = [Token(word) for word in words]
        sentence_field = TextField(tokens, self.token_indexers)
        fields['tokens'] = sentence_field

        words_for_graph = words + nodes_labels
        sample['words_for_graph'] = ['%d-%s' % (e[0], e[1]) for e in enumerate(words_for_graph)]
        tokens_for_graph = [Token(word) for word in words_for_graph]
        sentence_field_for_graph = TextField(tokens_for_graph, self.token_indexers)
        fields['tokens_for_graph'] = sentence_field_for_graph

        position = [Token(str(i)) for i in range(len(tokens))]
        position_field = TextField(position, self.position_indexers)
        fields['position'] = position_field

        aspects = [Token(category) for category in self.categories]
        aspect_field = TextField(aspects, self.aspect_indexers)
        fields['aspects'] = aspect_field

        category_labels = [0] * len(self.categories)
        polarity_labels = [-100] * len(self.categories)
        total_labels = []
        if len(sample) > 1:
            labels: list = sample['labels']
            for label in labels:
                category_labels[label[0]] = 1
                polarity_labels[label[0]] = label[1]
        for i in range(len(self.categories)):
            if polarity_labels[i] == -100:
                total_labels.append(0)
            else:
                total_labels.append(polarity_labels[i] + category_labels[i])

        label_field = ArrayField(np.array(category_labels + polarity_labels + total_labels))
        fields["label"] = label_field
        polarity_mask = [1 if polarity_labels[i] != -100 else 0 for i in range(len(self.categories))]
        polarity_mask_field = ArrayField(np.array(polarity_mask))
        fields['polarity_mask'] = polarity_mask_field

        # stop_word_labels = [1 if word in english_stop_words else 0 for word in words]
        # stop_word_num = sum(stop_word_labels)
        # stop_word_labels = [label / stop_word_num for label in stop_word_labels]
        # sample.append(stop_word_labels)

        sample_field = MetadataField(sample)
        fields["sample"] = sample_field
        return Instance(fields)

    @overrides
    def _read(self, samples: list) -> Iterator[Instance]:
        acd_sc_mode = self.configuration['acd_sc_mode']
        if acd_sc_mode == 'multi-multi':
            for sample in samples:
                yield self.text_to_instance({'text': sample[0], 'labels': sample[1]})
        elif acd_sc_mode == 'multi-single':
            for sample in samples:
                text = sample[0]
                labels = sample[1]
                for i in range(len(labels)):
                    labels_copy = [list(e) for e in copy.deepcopy(labels)]
                    for j, label in enumerate(labels_copy):
                        if j != i:
                            labels_copy[j][1] = -100
                    yield self.text_to_instance({'text': text, 'labels': labels_copy})
        elif acd_sc_mode == 'single-single':
            raise NotImplementedError('single-single')


class TextInAllAspectOut(DatasetReader):
    def __init__(self, categories: List[str], polarities: List[str],
                 tokenizer: Callable[[str], List[str]] = lambda x: x.split(),
                 token_indexers: Dict[str, TokenIndexer] = None,
                 position_indexers: Dict[str, TokenIndexer] = None,
                 aspect_indexers: Dict[str, TokenIndexer] = None,
                 core_nlp: my_corenlp.StanfordCoreNLP=None,
                 configuration=None) -> None:
        super().__init__(lazy=False)
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer(namespace="tokens")}
        self.position_indexers = position_indexers or {"position": SingleIdTokenIndexer(namespace='position')}
        self.aspect_indexers = aspect_indexers or {"aspect": SingleIdTokenIndexer(namespace='aspect')}
        self.categories = categories
        self.polarities = polarities
        self.spacy_nlp = spacy.load("en_core_web_sm")
        self.spacy_nlp.add_pipe(BeneparComponent('benepar_en'))
        self.core_nlp = core_nlp
        self.configuration = configuration

    def _build_graph(self, tree: ConstituencyTreeNode):
        graph = create_graph.create_sentence_constituency_graph_for_dgl(tree)
        return graph

    def _build_graph_with_dotted_line(self, tree: ConstituencyTreeNode):
        graph = create_graph.create_sentence_constituency_graph_for_dgl_with_dotted_line(tree)
        return graph

    @overrides
    def text_to_instance(self, sample: dict) -> Instance:
        fields = {}

        text: str = sample['text'].strip()
        text = re.sub('\\s+', ' ', text)
        sample['text_clean'] = text

        tree = ConstituencyTreeNode.parse_using_spacy(self.spacy_nlp, text)
        inner_nodes = ConstituencyTreeNode.get_all_inner_nodes(tree)
        inner_nodes.sort(key=lambda x: x.node_id)
        nodes_labels = [e.labels[0] for e in inner_nodes]
        sample['tree'] = tree
        sample['inner_nodes'] = inner_nodes
        sample['nodes_labels'] = nodes_labels

        graph = self._build_graph(tree)
        sample['graph'] = graph

        graph_with_dotted_line = self._build_graph_with_dotted_line(tree)
        sample['graph_with_dotted_line'] = graph_with_dotted_line

        words = self.tokenizer(text)
        sample['words'] = words
        tokens = [Token(word) for word in words]
        sentence_field = TextField(tokens, self.token_indexers)
        fields['tokens'] = sentence_field

        words_for_graph = words + nodes_labels
        sample['words_for_graph'] = ['%d-%s' % (e[0], e[1]) for e in enumerate(words_for_graph)]
        tokens_for_graph = [Token(word) for word in words_for_graph]
        sentence_field_for_graph = TextField(tokens_for_graph, self.token_indexers)
        fields['tokens_for_graph'] = sentence_field_for_graph

        position = [Token(str(i)) for i in range(len(tokens))]
        position_field = TextField(position, self.position_indexers)
        fields['position'] = position_field

        aspects = [Token(category) for category in self.categories]
        aspect_field = TextField(aspects, self.aspect_indexers)
        fields['aspects'] = aspect_field

        category_labels = [0] * len(self.categories)
        if len(sample) > 1:
            labels: list = sample['original_label']
            for label in labels:
                category_labels[label[0]] = 1
        label_field = ArrayField(np.array(category_labels))
        fields["label"] = label_field

        sample_field = MetadataField(sample)
        fields["sample"] = sample_field
        return Instance(fields)

    @overrides
    def _read(self, samples: list) -> Iterator[Instance]:
        for sample in samples:
            yield self.text_to_instance({'text': sample[0], 'original_label': sample[1]})


class AcdAndScDatasetReaderMilSimultaneouslyBert(DatasetReader):
    def __init__(self, categories: List[str], polarities: List[str],
                 tokenizer: Callable[[str], List[str]] = lambda x: x.split(),
                 token_indexers: Dict[str, TokenIndexer] = None,
                 position_indexers: Dict[str, TokenIndexer] = None,
                 core_nlp: my_corenlp.StanfordCoreNLP=None,
                 configuration=None,
                 bert_tokenizer=None,
                 bert_token_indexers=None) -> None:
        super().__init__(lazy=False)
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer(namespace="tokens")}
        self.bert_tokenizer = bert_tokenizer
        self.bert_token_indexers = bert_token_indexers or {"bert": SingleIdTokenIndexer(namespace="bert")}
        self.position_indexers = position_indexers or {"position": SingleIdTokenIndexer(namespace='position')}
        self.categories = categories
        self.polarities = polarities
        self.spacy_nlp = spacy.load("en_core_web_sm")
        self.core_nlp = core_nlp
        self.configuration = configuration

    def _build_graph(self, text):
        graph = create_graph.create_dependency_graph_for_dgl(text, self.spacy_nlp, None)
        return graph

    @overrides
    def text_to_instance(self, sample: list) -> Instance:
        fields = {}

        text: str = sample[0].strip()

        words = []
        for word in self.tokenizer(text):
            if word.strip() == '':
                continue
            words.append(word)
        sample.append(words)

        graph = self._build_graph(text)
        sample.append(graph)

        tokens = [Token(word) for word in words]

        sentence_field = TextField(tokens, self.token_indexers)
        fields['tokens'] = sentence_field

        bert_words = ['[CLS]']
        word_index_and_bert_indices = {}
        for i, word in enumerate(words):
            bert_ws = self.bert_tokenizer.tokenize(word)
            word_index_and_bert_indices[i] = []
            for j in range(len(bert_ws)):
                word_index_and_bert_indices[i].append(len(bert_words) + j)
            bert_words.extend(bert_ws)
        bert_words.append('[SEP]')
        # for i in range(len(words)):
        #     print('%s-%s' % (words[i], str([bert_words[j] for j in word_index_and_bert_indices[i]])))
        bert_text_fileds = []
        bert_words_of_all_aspect = []
        for aspect in self.categories:
            aspect_words = self.bert_tokenizer.tokenize(aspect)
            bert_words_of_aspect = bert_words + aspect_words + ['[SEP]']
            bert_words_of_all_aspect.append(bert_words_of_aspect)
            bert_tokens_of_aspect = [Token(word) for word in bert_words_of_aspect]
            bert_text_field = TextField(bert_tokens_of_aspect, self.bert_token_indexers)
            bert_text_fileds.append(bert_text_field)
        bert_field = ListField(bert_text_fileds)
        fields['bert'] = bert_field
        sample.append(bert_words)
        sample.append(bert_words_of_all_aspect)
        sample.append(word_index_and_bert_indices)

        position = [Token(str(i)) for i in range(len(tokens))]
        position_field = TextField(position, self.position_indexers)
        fields['position'] = position_field

        category_labels = [0] * len(self.categories)
        polarity_labels = [-100] * len(self.categories)
        total_labels = []
        if len(sample) > 1:
            labels: list = sample[1]
            for label in labels:
                category_labels[label[0]] = 1
                polarity_labels[label[0]] = label[1]
        for i in range(len(self.categories)):
            if polarity_labels[i] == -100:
                total_labels.append(0)
            else:
                total_labels.append(polarity_labels[i] + category_labels[i])

        label_field = ArrayField(np.array(category_labels + polarity_labels + total_labels))
        fields["label"] = label_field
        polarity_mask = [1 if polarity_labels[i] != -100 else 0 for i in range(len(self.categories))]
        polarity_mask_field = ArrayField(np.array(polarity_mask))
        fields['polarity_mask'] = polarity_mask_field
        sample_field = MetadataField(sample)
        fields["sample"] = sample_field
        return Instance(fields)

    @overrides
    def _read(self, samples: list) -> Iterator[Instance]:
        acd_sc_mode = self.configuration['acd_sc_mode']
        if acd_sc_mode == 'multi-multi':
            for sample in samples:
                yield self.text_to_instance(sample)
        elif acd_sc_mode == 'multi-single':
            for sample in samples:
                text = sample[0]
                labels = sample[1]
                for i in range(len(labels)):
                    labels_copy = [list(e) for e in copy.deepcopy(labels)]
                    for j, label in enumerate(labels_copy):
                        if j != i:
                            labels_copy[j][1] = -100
                    yield self.text_to_instance([text, labels_copy])
        elif acd_sc_mode == 'single-single':
            raise NotImplementedError('single-single')


class AcdAndScDatasetReaderConstituencyBert(DatasetReader):
    def __init__(self, categories: List[str], polarities: List[str],
                 tokenizer: Callable[[str], List[str]] = lambda x: x.split(),
                 token_indexers: Dict[str, TokenIndexer] = None,
                 position_indexers: Dict[str, TokenIndexer] = None,
                 core_nlp: my_corenlp.StanfordCoreNLP=None,
                 configuration=None,
                 bert_tokenizer=None,
                 bert_token_indexers=None,
                 sentence_constituency_indexer: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=False)
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer(namespace="tokens")}
        self.bert_tokenizer = bert_tokenizer
        self.bert_token_indexers = bert_token_indexers or {"bert": SingleIdTokenIndexer(namespace="bert")}
        self.position_indexers = position_indexers or {"position": SingleIdTokenIndexer(namespace='position')}
        self.sentence_constituency_indexer = sentence_constituency_indexer
        self.categories = categories
        self.polarities = polarities
        self.spacy_nlp = spacy.load("en_core_web_sm")
        self.spacy_nlp.add_pipe(BeneparComponent('benepar_en'))
        self.core_nlp = core_nlp
        self.configuration = configuration

    def _build_graph(self, tree: ConstituencyTreeNode):
        graph = create_graph.create_sentence_constituency_graph_for_dgl(tree)
        return graph

    def _build_graph_with_dotted_line(self, tree: ConstituencyTreeNode):
        graph = create_graph.create_sentence_constituency_graph_for_dgl_with_dotted_line(tree)
        return graph

    @overrides
    def text_to_instance(self, sample: list) -> Instance:
        fields = {}

        text: str = sample['text'].strip()
        text = re.sub('\\s+', ' ', text)
        sample['text_clean'] = text

        tree = ConstituencyTreeNode.parse_using_spacy(self.spacy_nlp, text)
        inner_nodes = ConstituencyTreeNode.get_all_inner_nodes(tree)
        inner_nodes.sort(key=lambda x: x.node_id)
        nodes_labels = [e.labels[0] for e in inner_nodes]
        sample['tree'] = tree
        sample['inner_nodes'] = inner_nodes
        sample['nodes_labels'] = nodes_labels

        graph = self._build_graph(tree)
        sample['graph'] = graph

        graph_with_dotted_line = self._build_graph_with_dotted_line(tree)
        sample['graph_with_dotted_line'] = graph_with_dotted_line

        words = self.tokenizer(text)
        sample['words'] = words
        tokens = [Token(word) for word in words]
        sentence_field = TextField(tokens, self.token_indexers)
        fields['tokens'] = sentence_field

        words_for_graph = words + nodes_labels
        sample['words_for_graph'] = ['%d-%s' % (e[0], e[1]) for e in enumerate(words_for_graph)]
        tokens_for_graph = [Token(word) for word in words_for_graph]
        sentence_field_for_graph = TextField(tokens_for_graph, self.token_indexers)
        fields['tokens_for_graph'] = sentence_field_for_graph

        bert_words = ['[CLS]']
        word_index_and_bert_indices = {}
        for i, word in enumerate(words):
            bert_ws = self.bert_tokenizer.tokenize(word)
            word_index_and_bert_indices[i] = []
            for j in range(len(bert_ws)):
                word_index_and_bert_indices[i].append(len(bert_words) + j)
            bert_words.extend(bert_ws)
        bert_words.append('[SEP]')
        # for i in range(len(words)):
        #     print('%s-%s' % (words[i], str([bert_words[j] for j in word_index_and_bert_indices[i]])))
        bert_text_fileds = []
        bert_words_of_all_aspect = []
        for aspect in self.categories:
            if self.configuration['pair']:
                aspect_words = self.bert_tokenizer.tokenize(aspect)
                bert_words_of_aspect = bert_words + aspect_words + ['[SEP]']
            else:
                bert_words_of_aspect = bert_words
            bert_words_of_all_aspect.append(bert_words_of_aspect)
            bert_tokens_of_aspect = [Token(word) for word in bert_words_of_aspect]
            bert_text_field = TextField(bert_tokens_of_aspect, self.bert_token_indexers)
            bert_text_fileds.append(bert_text_field)
        bert_field = ListField(bert_text_fileds)
        fields['bert'] = bert_field
        sample['bert_words'] = bert_words
        sample['bert_words_of_all_aspect'] = bert_words_of_all_aspect
        sample['word_index_and_bert_indices'] = word_index_and_bert_indices

        position = [Token(str(i)) for i in range(len(tokens))]
        position_field = TextField(position, self.position_indexers)
        fields['position'] = position_field

        category_labels = [0] * len(self.categories)
        polarity_labels = [-100] * len(self.categories)
        total_labels = []
        if len(sample) > 1:
            labels: list = sample['labels']
            for label in labels:
                category_labels[label[0]] = 1
                polarity_labels[label[0]] = label[1]
        for i in range(len(self.categories)):
            if polarity_labels[i] == -100:
                total_labels.append(0)
            else:
                total_labels.append(polarity_labels[i] + category_labels[i])

        label_field = ArrayField(np.array(category_labels + polarity_labels + total_labels))
        fields["label"] = label_field
        polarity_mask = [1 if polarity_labels[i] != -100 else 0 for i in range(len(self.categories))]
        polarity_mask_field = ArrayField(np.array(polarity_mask))
        fields['polarity_mask'] = polarity_mask_field
        sample_field = MetadataField(sample)
        fields["sample"] = sample_field
        return Instance(fields)

    @overrides
    def _read(self, samples: list) -> Iterator[Instance]:
        acd_sc_mode = self.configuration['acd_sc_mode']
        if acd_sc_mode == 'multi-multi':
            for sample in samples:
                yield self.text_to_instance({'text': sample[0], 'labels': sample[1]})
        elif acd_sc_mode == 'multi-single':
            for sample in samples:
                text = sample[0]
                labels = sample[1]
                for i in range(len(labels)):
                    labels_copy = [list(e) for e in copy.deepcopy(labels)]
                    for j, label in enumerate(labels_copy):
                        if j != i:
                            labels_copy[j][1] = -100
                    yield self.text_to_instance({'text': text, 'labels': labels_copy})
        elif acd_sc_mode == 'single-single':
            raise NotImplementedError('single-single')


class TextAspectInSentimentOutDatasetReaderBert(DatasetReader):
    def __init__(self, categories: List[str], polarities: List[str],
                 tokenizer: Callable[[str], List[str]] = lambda x: x.split(),
                 token_indexers: Dict[str, TokenIndexer] = None,
                 aspect_indexers: Dict[str, TokenIndexer] = None,
                 core_nlp: my_corenlp.StanfordCoreNLP=None,
                 configuration=None) -> None:
        super().__init__(lazy=False)
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer(namespace="tokens")}
        self.aspect_indexers = aspect_indexers or {"aspect": SingleIdTokenIndexer(namespace='aspect')}
        self.categories = categories
        self.polarities = polarities
        self.spacy_nlp = spacy.load("en_core_web_sm")
        self.core_nlp = core_nlp
        self.configuration = configuration

    @overrides
    def text_to_instance(self, sample: list) -> Instance:
        fields = {}

        category_index = sample['aspect'][0]
        sentiment_index = sample['aspect'][1]
        aspect = [Token(self.categories[category_index])]
        aspect_field = TextField(aspect, self.aspect_indexers)
        fields['aspect'] = aspect_field

        text: str = sample['text'].strip()

        words = ['[CLS]']
        words.extend(self.tokenizer.tokenize(text))
        words.append('[SEP]')
        if self.configuration['model_name'] == 'BERT-pair':
            words.extend(self.tokenizer.tokenize(self.categories[category_index]))
            words.append('[SEP]')
        sample['words'] = words

        tokens = [Token(word) for word in words]
        sentence_field = TextField(tokens, self.token_indexers)
        fields['tokens'] = sentence_field

        fields["label"] = LabelField(sentiment_index, skip_indexing=True)
        sample_field = MetadataField(sample)
        fields["sample"] = sample_field
        return Instance(fields)

    @overrides
    def _read(self, samples: list) -> Iterator[Instance]:
        for sample in samples:
            multi_aspect = len(sample[1]) > 1
            for aspect in sample[1]:
                yield self.text_to_instance({'text': sample[0], 'aspect': aspect, 'multi_aspect': multi_aspect})


class AcdAndScDatasetReaderMilSimultaneouslyBertSingle(DatasetReader):
    def __init__(self, categories: List[str], polarities: List[str],
                 tokenizer: Callable[[str], List[str]] = lambda x: x.split(),
                 token_indexers: Dict[str, TokenIndexer] = None,
                 position_indexers: Dict[str, TokenIndexer] = None,
                 core_nlp: my_corenlp.StanfordCoreNLP=None,
                 configuration=None,
                 bert_tokenizer=None,
                 bert_token_indexers=None) -> None:
        super().__init__(lazy=False)
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer(namespace="tokens")}
        self.bert_tokenizer = bert_tokenizer
        self.bert_token_indexers = bert_token_indexers or {"bert": SingleIdTokenIndexer(namespace="bert")}
        self.position_indexers = position_indexers or {"position": SingleIdTokenIndexer(namespace='position')}
        self.categories = categories
        self.polarities = polarities
        self.spacy_nlp = spacy.load("en_core_web_sm")
        self.core_nlp = core_nlp
        self.configuration = configuration

    def _build_graph(self, text):
        graph = create_graph.create_dependency_graph_for_dgl(text, self.spacy_nlp, None)
        return graph

    @overrides
    def text_to_instance(self, sample: list) -> Instance:
        fields = {}

        text: str = sample[0].strip()

        words = []
        for word in self.tokenizer(text):
            if word.strip() == '':
                continue
            words.append(word)
        sample.append(words)

        graph = self._build_graph(text)
        sample.append(graph)

        tokens = [Token(word) for word in words]

        sentence_field = TextField(tokens, self.token_indexers)
        fields['tokens'] = sentence_field

        bert_words = ['[CLS]']
        word_index_and_bert_indices = {}
        for i, word in enumerate(words):
            bert_ws = self.bert_tokenizer.tokenize(word)
            word_index_and_bert_indices[i] = []
            for j in range(len(bert_ws)):
                word_index_and_bert_indices[i].append(len(bert_words) + j)
            bert_words.extend(bert_ws)
        bert_words.append('[SEP]')
        # for i in range(len(words)):
        #     print('%s-%s' % (words[i], str([bert_words[j] for j in word_index_and_bert_indices[i]])))
        bert_text_fileds = []
        bert_words_of_all_aspect = []

        bert_words_of_all_aspect.append(bert_words)
        bert_tokens = [Token(word) for word in bert_words]
        bert_text_field = TextField(bert_tokens, self.bert_token_indexers)
        bert_text_fileds.append(bert_text_field)
        bert_field = ListField(bert_text_fileds)
        fields['bert'] = bert_field
        sample.append(bert_words)
        sample.append(bert_words_of_all_aspect)
        sample.append(word_index_and_bert_indices)

        position = [Token(str(i)) for i in range(len(tokens))]
        position_field = TextField(position, self.position_indexers)
        fields['position'] = position_field

        category_labels = [0] * len(self.categories)
        polarity_labels = [-100] * len(self.categories)
        total_labels = []
        if len(sample) > 1:
            labels: list = sample[1]
            for label in labels:
                category_labels[label[0]] = 1
                polarity_labels[label[0]] = label[1]
        for i in range(len(self.categories)):
            if polarity_labels[i] == -100:
                total_labels.append(0)
            else:
                total_labels.append(polarity_labels[i] + category_labels[i])

        label_field = ArrayField(np.array(category_labels + polarity_labels + total_labels))
        fields["label"] = label_field
        polarity_mask = [1 if polarity_labels[i] != -100 else 0 for i in range(len(self.categories))]
        polarity_mask_field = ArrayField(np.array(polarity_mask))
        fields['polarity_mask'] = polarity_mask_field
        sample_field = MetadataField(sample)
        fields["sample"] = sample_field
        return Instance(fields)

    @overrides
    def _read(self, samples: list) -> Iterator[Instance]:
        acd_sc_mode = self.configuration['acd_sc_mode']
        if acd_sc_mode == 'multi-multi':
            for sample in samples:
                yield self.text_to_instance(sample)
        elif acd_sc_mode == 'multi-single':
            for sample in samples:
                text = sample[0]
                labels = sample[1]
                for i in range(len(labels)):
                    labels_copy = [list(e) for e in copy.deepcopy(labels)]
                    for j, label in enumerate(labels_copy):
                        if j != i:
                            labels_copy[j][1] = -100
                    yield self.text_to_instance([text, labels_copy])
        elif acd_sc_mode == 'single-single':
            raise NotImplementedError('single-single')


class AcdAndScDatasetReaderConstituencyBertSingle(DatasetReader):
    def __init__(self, categories: List[str], polarities: List[str],
                 tokenizer: Callable[[str], List[str]] = lambda x: x.split(),
                 token_indexers: Dict[str, TokenIndexer] = None,
                 position_indexers: Dict[str, TokenIndexer] = None,
                 sentence_constituency_indexer: Dict[str, TokenIndexer] = None,
                 core_nlp: my_corenlp.StanfordCoreNLP=None,
                 configuration=None,
                 bert_tokenizer=None,
                 bert_token_indexers=None) -> None:
        super().__init__(lazy=False)
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer(namespace="tokens")}
        self.bert_tokenizer = bert_tokenizer
        self.bert_token_indexers = bert_token_indexers or {"bert": SingleIdTokenIndexer(namespace="bert")}
        self.position_indexers = position_indexers or {"position": SingleIdTokenIndexer(namespace='position')}
        self.sentence_constituency_indexer = sentence_constituency_indexer
        self.categories = categories
        self.polarities = polarities
        self.spacy_nlp = spacy.load("en_core_web_sm")
        self.core_nlp = core_nlp
        self.configuration = configuration

    def _build_graph(self, tree: ConstituencyTreeNode):
        graph = create_graph.create_sentence_constituency_graph_for_dgl(tree)
        return graph

    @overrides
    def text_to_instance(self, sample: list) -> Instance:
        fields = {}

        text: str = sample['text'].strip()

        tree = ConstituencyTreeNode.parse_using_spacy(self.spacy_nlp, text)
        inner_nodes = ConstituencyTreeNode.get_all_inner_nodes(tree)
        inner_nodes.sort(key=lambda x: x.node_id)
        nodes_labels = [e.labels[0] for e in inner_nodes]
        sample['tree'] = tree
        sample['inner_nodes'] = inner_nodes
        sample['nodes_labels'] = nodes_labels

        graph = self._build_graph(tree)
        sample['graph'] = graph

        words = self.tokenizer(text)
        sample['words'] = words
        tokens = [Token(word) for word in words]
        sentence_field = TextField(tokens, self.token_indexers)
        fields['tokens'] = sentence_field

        words_for_graph = words + nodes_labels
        sample['words_for_graph'] = ['%d-%s' % (e[0], e[1]) for e in enumerate(words_for_graph)]
        tokens_for_graph = [Token(word) for word in words_for_graph]
        sentence_field_for_graph = TextField(tokens_for_graph, self.token_indexers)
        fields['tokens_for_graph'] = sentence_field_for_graph

        bert_words = ['[CLS]']
        word_index_and_bert_indices = {}
        for i, word in enumerate(words):
            bert_ws = self.bert_tokenizer.tokenize(word)
            word_index_and_bert_indices[i] = []
            for j in range(len(bert_ws)):
                word_index_and_bert_indices[i].append(len(bert_words) + j)
            bert_words.extend(bert_ws)
        bert_words.append('[SEP]')
        # for i in range(len(words)):
        #     print('%s-%s' % (words[i], str([bert_words[j] for j in word_index_and_bert_indices[i]])))
        bert_text_fileds = []
        bert_words_of_all_aspect = []

        bert_words_of_all_aspect.append(bert_words)
        bert_tokens = [Token(word) for word in bert_words]
        bert_text_field = TextField(bert_tokens, self.bert_token_indexers)
        bert_text_fileds.append(bert_text_field)
        bert_field = ListField(bert_text_fileds)
        fields['bert'] = bert_field
        sample['bert_words'] = bert_words
        sample['bert_words_of_all_aspect'] = bert_words_of_all_aspect
        sample['word_index_and_bert_indices'] = word_index_and_bert_indices

        position = [Token(str(i)) for i in range(len(tokens))]
        position_field = TextField(position, self.position_indexers)
        fields['position'] = position_field

        category_labels = [0] * len(self.categories)
        polarity_labels = [-100] * len(self.categories)
        total_labels = []
        if len(sample) > 1:
            labels: list = sample['labels']
            for label in labels:
                category_labels[label[0]] = 1
                polarity_labels[label[0]] = label[1]
        for i in range(len(self.categories)):
            if polarity_labels[i] == -100:
                total_labels.append(0)
            else:
                total_labels.append(polarity_labels[i] + category_labels[i])

        label_field = ArrayField(np.array(category_labels + polarity_labels + total_labels))
        fields["label"] = label_field
        polarity_mask = [1 if polarity_labels[i] != -100 else 0 for i in range(len(self.categories))]
        polarity_mask_field = ArrayField(np.array(polarity_mask))
        fields['polarity_mask'] = polarity_mask_field
        sample_field = MetadataField(sample)
        fields["sample"] = sample_field
        return Instance(fields)

    @overrides
    def _read(self, samples: list) -> Iterator[Instance]:
        acd_sc_mode = self.configuration['acd_sc_mode']
        if acd_sc_mode == 'multi-multi':
            for sample in samples:
                yield self.text_to_instance({'text': sample[0], 'labels': sample[1]})
        elif acd_sc_mode == 'multi-single':
            for sample in samples:
                text = sample[0]
                labels = sample[1]
                for i in range(len(labels)):
                    labels_copy = [list(e) for e in copy.deepcopy(labels)]
                    for j, label in enumerate(labels_copy):
                        if j != i:
                            labels_copy[j][1] = -100
                    yield self.text_to_instance({'text': text, 'labels': labels_copy})
        elif acd_sc_mode == 'single-single':
            raise NotImplementedError('single-single')


class AcdAndScDatasetReaderForTextAspectInSentimentOutBert(DatasetReader):
    def __init__(self, categories: List[str], polarities: List[str],
                 tokenizer: Callable[[str], List[str]] = lambda x: x.split(),
                 token_indexers: Dict[str, TokenIndexer] = None,
                 position_indexers: Dict[str, TokenIndexer] = None,
                 core_nlp: my_corenlp.StanfordCoreNLP=None,
                 configuration=None,
                 bert_tokenizer=None,
                 bert_token_indexers=None) -> None:
        super().__init__(lazy=False)
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer(namespace="tokens")}
        self.bert_tokenizer = bert_tokenizer
        self.bert_token_indexers = bert_token_indexers or {"bert": SingleIdTokenIndexer(namespace="bert")}
        self.position_indexers = position_indexers or {"position": SingleIdTokenIndexer(namespace='position')}
        self.categories = categories
        self.polarities = polarities
        self.spacy_nlp = spacy.load("en_core_web_sm")
        self.core_nlp = core_nlp
        self.configuration = configuration

    def _build_graph(self, text):
        graph = create_graph.create_dependency_graph_for_dgl(text, self.spacy_nlp, None)
        return graph

    @overrides
    def text_to_instance(self, sample: list) -> Instance:
        fields = {}

        text: str = sample[0].strip()

        words = []
        for word in self.tokenizer(text):
            if word.strip() == '':
                continue
            words.append(word)
        sample.append(words)

        graph = self._build_graph(text)
        sample.append(graph)

        tokens = [Token(word) for word in words]

        sentence_field = TextField(tokens, self.token_indexers)
        fields['tokens'] = sentence_field

        bert_words = ['[CLS]']
        word_index_and_bert_indices = {}
        for i, word in enumerate(words):
            bert_ws_temp = self.bert_tokenizer.tokenize(word)
            bert_ws = bert_ws_temp[1: -1]
            word_index_and_bert_indices[i] = []
            for j in range(len(bert_ws)):
                word_index_and_bert_indices[i].append(len(bert_words) + j)
            bert_words.extend(bert_ws)
        # for i in range(len(words)):
        #     print('%s-%s' % (words[i], str([bert_words[j] for j in word_index_and_bert_indices[i]])))
        bert_text_fileds = []
        bert_words_of_all_aspect = []
        for aspect in self.categories:
            aspect_words = self.bert_tokenizer.tokenize(aspect)
            bert_words_of_aspect = bert_words + aspect_words
            bert_words_of_all_aspect.append(bert_words_of_aspect)
            bert_tokens_of_aspect = [Token(word) for word in bert_words_of_aspect]
            bert_text_field = TextField(bert_tokens_of_aspect, self.bert_token_indexers)
            bert_text_fileds.append(bert_text_field)
        bert_field = ListField(bert_text_fileds)
        fields['bert'] = bert_field
        sample.append(bert_words)
        sample.append(bert_words_of_all_aspect)
        sample.append(word_index_and_bert_indices)

        position = [Token(str(i)) for i in range(len(tokens))]
        position_field = TextField(position, self.position_indexers)
        fields['position'] = position_field

        category_labels = [0] * len(self.categories)
        polarity_labels = [-100] * len(self.categories)
        if len(sample) > 1:
            labels: list = sample[1]
            for label in labels:
                category_labels[label[0]] = 1
                polarity_labels[label[0]] = label[1]
        label_field = ArrayField(np.array(category_labels + polarity_labels))
        fields["label"] = label_field
        polarity_mask = [1 if polarity_labels[i] != -100 else 0 for i in range(len(self.categories))]
        polarity_mask_field = ArrayField(np.array(polarity_mask))
        fields['polarity_mask'] = polarity_mask_field
        sample_field = MetadataField(sample)
        fields["sample"] = sample_field
        return Instance(fields)

    @overrides
    def _read(self, samples: list) -> Iterator[Instance]:
        for sample in samples:
            for aspect in sample[1]:
                yield self.text_to_instance([sample[0], aspect])


class TextAspectInSentimentOut(DatasetReader):
    def __init__(self, categories: List[str], polarities: List[str],
                 tokenizer: Callable[[str], List[str]] = lambda x: x.split(),
                 token_indexers: Dict[str, TokenIndexer] = None,
                 aspect_indexers: Dict[str, TokenIndexer] = None,
                 core_nlp: my_corenlp.StanfordCoreNLP=None,
                 configuration=None) -> None:
        super().__init__(lazy=False)
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer(namespace="tokens")}
        self.aspect_indexers = aspect_indexers or {"aspect": SingleIdTokenIndexer(namespace='aspect')}
        self.categories = categories
        self.polarities = polarities
        self.spacy_nlp = spacy.load("en_core_web_sm")
        self.core_nlp = core_nlp
        self.configuration = configuration

    @overrides
    def text_to_instance(self, sample: list) -> Instance:
        fields = {}

        text: str = sample[0].strip()

        words = self.tokenizer(text)
        sample.insert(2, words)
        tokens = [Token(word) for word in words]
        sentence_field = TextField(tokens, self.token_indexers)
        fields['tokens'] = sentence_field

        aspect = [Token(self.categories[sample[1][0]])]
        aspect_field = TextField(aspect, self.aspect_indexers)
        fields['aspect'] = aspect_field

        fields["label"] = LabelField(sample[1][1], skip_indexing=True)
        sample_field = MetadataField(sample)
        fields["sample"] = sample_field
        return Instance(fields)

    @overrides
    def _read(self, samples: list) -> Iterator[Instance]:
        for sample in samples:
            multi_aspect = len(sample[1]) > 1
            for aspect in sample[1]:
                yield self.text_to_instance([sample[0], aspect, multi_aspect])


class AcdAndScDatasetReaderHeat(DatasetReader):
    def __init__(self, categories: List[str], polarities: List[str],
                 tokenizer: Callable[[str], List[str]] = lambda x: x.split(),
                 token_indexers: Dict[str, TokenIndexer] = None,
                 aspect_indexers: Dict[str, TokenIndexer] = None,
                 core_nlp: my_corenlp.StanfordCoreNLP=None,
                 configuration=None) -> None:
        super().__init__(lazy=False)
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer(namespace="tokens")}
        self.aspect_indexers = aspect_indexers or {"aspect": SingleIdTokenIndexer(namespace='aspect')}
        self.categories = categories
        self.polarities = polarities
        self.spacy_nlp = spacy.load("en_core_web_sm")
        self.core_nlp = core_nlp
        self.configuration = configuration

    @overrides
    def text_to_instance(self, sample: list) -> Instance:
        fields = {}

        text: str = sample[0].strip()

        words = self.tokenizer(text)
        sample.append(words)
        tokens = [Token(word) for word in words]
        sentence_field = TextField(tokens, self.token_indexers)
        fields['tokens'] = sentence_field

        aspect = [Token(self.categories[sample[1][0]])]
        aspect_field = TextField(aspect, self.aspect_indexers)
        fields['aspect'] = aspect_field

        fields["label"] = LabelField(sample[1][1], skip_indexing=True)
        sample_field = MetadataField(sample)
        fields["sample"] = sample_field
        return Instance(fields)

    @overrides
    def _read(self, samples: list) -> Iterator[Instance]:
        for sample in samples:
            for aspect in sample[1]:
                yield self.text_to_instance([sample[0], aspect])


class AcdAndScDatasetReaderHeatCae(DatasetReader):
    def __init__(self, categories: List[str], polarities: List[str],
                 tokenizer: Callable[[str], List[str]] = lambda x: x.split(),
                 token_indexers: Dict[str, TokenIndexer] = None,
                 aspect_indexers: Dict[str, TokenIndexer] = None,
                 core_nlp: my_corenlp.StanfordCoreNLP=None,
                 configuration=None) -> None:
        super().__init__(lazy=False)
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer(namespace="tokens")}
        self.aspect_indexers = aspect_indexers or {"aspect": SingleIdTokenIndexer(namespace='aspect')}
        self.categories = categories
        self.polarities = polarities
        self.spacy_nlp = spacy.load("en_core_web_sm")
        self.core_nlp = core_nlp
        self.configuration = configuration

    @overrides
    def text_to_instance(self, sample: list) -> Instance:
        fields = {}

        text: str = sample['text']
        aspect_index = sample['aspect_index']

        words = self.tokenizer(text)
        sample['words'] = words
        tokens = [Token(word) for word in words]
        sentence_field = TextField(tokens, self.token_indexers)
        fields['tokens'] = sentence_field

        aspects = [Token(category) for category in self.categories]
        aspect_field = TextField(aspects, self.aspect_indexers)
        fields['aspects'] = aspect_field

        if 'sentiment_label' in sample:
            aspect_label = sample['aspect_label']
            sentiment_label = sample['sentiment_label']

            category_labels = [0] * len(self.categories)
            for label in aspect_label:
                category_labels[label] = 1
            fields["aspect_label"] = ArrayField(array=np.array(category_labels))
            # fields["aspect_label"] = MultiLabelField(category_labels, label_namespace='aspect_labels',
            #                                          skip_indexing=True, num_labels=len(self.categories))
            fields["label"] = LabelField(sentiment_label, skip_indexing=True,
                                                   label_namespace='sentiment_labels')
        sample_field = MetadataField(sample)
        fields["sample"] = sample_field
        return Instance(fields)

    @overrides
    def _read(self, samples: list) -> Iterator[Instance]:
        for sample in samples:
            text = sample[0]
            all_aspects = sample[1]
            for i in range(len(all_aspects)):
                aspect = copy.deepcopy(all_aspects)
                aspect_label = [e[0] for e in aspect]
                sentiment_label = aspect[i][1]
                aspect_index = aspect[i][0]
                yield self.text_to_instance({'text': text,
                                             'aspect_label': aspect_label,
                                             'sentiment_label': sentiment_label,
                                             'aspect_index': aspect_index})


class AcdAndScDatasetReaderHeatCaeM(DatasetReader):
    def __init__(self, categories: List[str], polarities: List[str],
                 tokenizer: Callable[[str], List[str]] = lambda x: x.split(),
                 token_indexers: Dict[str, TokenIndexer] = None,
                 aspect_indexers: Dict[str, TokenIndexer] = None,
                 core_nlp: my_corenlp.StanfordCoreNLP=None,
                 configuration=None) -> None:
        super().__init__(lazy=False)
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer(namespace="tokens")}
        self.aspect_indexers = aspect_indexers or {"aspect": SingleIdTokenIndexer(namespace='aspect')}
        self.categories = categories
        self.polarities = polarities
        self.spacy_nlp = spacy.load("en_core_web_sm")
        self.core_nlp = core_nlp
        self.configuration = configuration

    @overrides
    def text_to_instance(self, sample: list) -> Instance:
        fields = {}

        text: str = sample['text']

        words = self.tokenizer(text)
        sample['words'] = words
        tokens = [Token(word) for word in words]
        sentence_field = TextField(tokens, self.token_indexers)
        fields['tokens'] = sentence_field

        aspects = [Token(category) for category in self.categories]
        aspect_field = TextField(aspects, self.aspect_indexers)
        fields['aspects'] = aspect_field

        if 'label' in sample:
            category_labels = [0] * len(self.categories)
            polarity_labels = [-100] * len(self.categories)
            labels: list = sample['label']
            for label in labels:
                category_labels[label[0]] = 1
                polarity_labels[label[0]] = label[1]
            label_field = ArrayField(np.array(category_labels + polarity_labels))
            fields["label"] = label_field
            polarity_mask = [1 if polarity_labels[i] != -100 else 0 for i in range(len(self.categories))]
            polarity_mask_field = ArrayField(np.array(polarity_mask))
            fields['polarity_mask'] = polarity_mask_field

        sample_field = MetadataField(sample)
        fields["sample"] = sample_field
        return Instance(fields)

    @overrides
    def _read(self, samples: list) -> Iterator[Instance]:
        for sample in samples:
            yield self.text_to_instance({'text': sample[0],
                                         'label': sample[1]})

