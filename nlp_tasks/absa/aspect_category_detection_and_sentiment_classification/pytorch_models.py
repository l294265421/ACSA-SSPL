# -*- coding: utf-8 -*-


from typing import *
from overrides import overrides
import time
import copy
import os
import re

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from allennlp.modules.seq2vec_encoders import CnnEncoder as VectorCnnEncoder
from allennlp.nn.util import get_text_field_mask
from allennlp.models import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.data.fields import TextField, MetadataField, ArrayField, LabelField
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
from allennlp.modules import attention
from allennlp.data.iterators import DataIterator
from tqdm import tqdm
from scipy.special import expit
from allennlp.nn import util as allennlp_util
import dgl
from dgl import function as dgl_fn
from sklearn.metrics import f1_score, precision_score, recall_score

from nlp_tasks.utils import attention_visualizer
from nlp_tasks.absa.aspect_category_detection_and_sentiment_classification import allennlp_metrics
from nlp_tasks.absa.aspect_category_detection_and_sentiment_classification.cnn_encoder_seq2seq import CnnEncoder


class AttentionInHtt(nn.Module):
    """
    2016-Hierarchical Attention Networks for Document Classification
    """

    def __init__(self, in_features, out_features, bias=True, softmax=True):
        super().__init__()
        self.W = nn.Linear(in_features, out_features, bias)
        self.uw = nn.Linear(out_features, 1, bias=False)
        self.softmax = softmax

    def forward(self, h: torch.Tensor, mask: torch.Tensor):
        u = self.W(h)
        u = torch.tanh(u)
        similarities = self.uw(u)
        similarities = similarities.squeeze(dim=-1)
        if self.softmax:
            alpha = allennlp_util.masked_softmax(similarities, mask)
            return alpha
        else:
            return similarities


class DotProductAttentionInHtt(nn.Module):
    """
    2016-Hierarchical Attention Networks for Document Classification
    """

    def __init__(self, in_features, out_features, bias=True, softmax=True):
        super().__init__()
        self.uw = nn.Linear(in_features, 1, bias=False)
        self.softmax = softmax

    def forward(self, h: torch.Tensor, mask: torch.Tensor):
        similarities = self.uw(h)
        similarities = similarities.squeeze(dim=-1)
        if self.softmax:
            alpha = allennlp_util.masked_softmax(similarities, mask)
            return alpha
        else:
            return similarities


class AverageAttention(nn.Module):
    """
    2019-emnlp-Attention is not not Explanation
    """

    def __init__(self):
        super().__init__()

    def forward(self, h: torch.Tensor, mask: torch.Tensor):
        alpha = allennlp_util.masked_softmax(mask, mask)
        return alpha
        
        
class BernoulliAttentionInHtt(nn.Module):
    """
    2016-Hierarchical Attention Networks for Document Classification
    """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.W = nn.Linear(in_features, out_features, bias)
        self.uw = nn.Linear(out_features, 1, bias=False)

    def forward(self, h: torch.Tensor, mask: torch.Tensor):
        u = self.W(h)
        u = torch.tanh(u)
        similarities = self.uw(u)
        similarities = similarities.squeeze(dim=-1)
        alpha = torch.sigmoid(similarities)
        return alpha


class AttentionInCan(nn.Module):
    """
    2019-emnlp-CAN Constrained Attention Networks for Multi-Aspect Sentiment Analysis
    """

    def __init__(self, in_features, bias=True, softmax=True):
        super().__init__()
        self.W1 = nn.Linear(in_features, in_features, bias)
        self.W2 = nn.Linear(in_features, in_features, bias)
        self.uw = nn.Linear(in_features, 1, bias=False)
        self.softmax = softmax

    def forward(self, h1: torch.Tensor, h2: torch.Tensor, mask: torch.Tensor):
        u1 = self.W1(h1)
        u2 = self.W2(h2)
        u = u1 + u2
        u = torch.tanh(u)
        similarities = self.uw(u)
        similarities = similarities.squeeze(dim=-1)
        if self.softmax:
            alpha = allennlp_util.masked_softmax(similarities, mask)
            return alpha
        else:
            return similarities


class LocationMaskLayer(nn.Module):
    """
    2017-CIKM-Aspect-level Sentiment Classification with HEAT (HiErarchical ATtention) Network
    """

    def __init__(self, location_num, configuration):
        super().__init__()
        self.location_num = location_num
        self.configuration = configuration

    def forward(self, alpha: torch.Tensor):
        location_num = self.location_num
        location_matrix = torch.zeros([location_num, location_num], dtype=torch.float,
                                      device=self.configuration['device'],
                                      requires_grad=False)
        for i in range(location_num):
            for j in range(location_num):
                location_matrix[i, j] = 1 - (abs(i - j) / location_num)
        result = alpha.mm(location_matrix)
        return result


class TextInAllAspectSentimentOutModel(Model):

    def __init__(self, vocab: Vocabulary, category_loss_weight=1, sentiment_loss_weight=1):
        super().__init__(vocab)
        self.category_loss_weight = category_loss_weight
        self.sentiment_loss_weight = sentiment_loss_weight

    def matrix_mul(self, input, weight, bias=False):
        feature_list = []
        for feature in input:
            feature = torch.mm(feature, weight)
            if isinstance(bias, torch.nn.parameter.Parameter):
                feature = feature + bias.expand(feature.size()[0], bias.size()[1])
            feature = torch.tanh(feature).unsqueeze(0)
            feature_list.append(feature)

        return torch.cat(feature_list, 0).squeeze()

    def element_wise_mul(self, input1, input2, return_not_sum_result=False):
        feature_list = []
        for feature_1, feature_2 in zip(input1, input2):
            feature_2 = feature_2.unsqueeze(1)
            feature_2 = feature_2.expand_as(feature_1)
            feature = feature_1 * feature_2
            feature = feature.unsqueeze(0)
            feature_list.append(feature)
        output = torch.cat(feature_list, 0)

        result = torch.sum(output, 1)
        if return_not_sum_result:
            return result, output
        else:
            return result

    def reduce(self, nodes):
        """Take an average over all neighbor node features hu and use it to
        overwrite the original node feature."""
        m = nodes.mailbox['m']
        accum = torch.sum(m, 1)
        return {'h': accum}

    def pad_dgl_graph(self, graphs, max_node_num):
        graphs_padded = []
        for graph in graphs:
            graph_padded = copy.deepcopy(graph)
            node_num = graph.number_of_nodes()
            graph_padded.add_nodes(max_node_num - node_num)
            graphs_padded.append(graph_padded)
        return graphs_padded

    def no_grad_for_acd_parameter(self):
        self.set_grad_for_acd_parameter(requires_grad=False)

    def set_grad_for_acd_parameter(self, requires_grad=True):
        pass

    def set_grad_for_acsc_parameter(self, requires_grad=True):
        pass

    def _get_model_visualization_picture_filepath(self, configuration: dict, words: list):
        savefig_dir = configuration['savefig_dir']
        if not savefig_dir:
            return None
        filename = '%s-%s.svg' % ('-'.join(words[:3]), str(time.time()))
        filename = re.sub('/', '', filename)
        return os.path.join(savefig_dir, filename)


class TextAspectInSentimentOutModel(Model):

    def __init__(self, vocab: Vocabulary, category_loss_weight=1, sentiment_loss_weight=1):
        super().__init__(vocab)
        self.category_loss_weight = category_loss_weight
        self.sentiment_loss_weight = sentiment_loss_weight

    def matrix_mul(self, input, weight, bias=False):
        feature_list = []
        for feature in input:
            feature = torch.mm(feature, weight)
            if isinstance(bias, torch.nn.parameter.Parameter):
                feature = feature + bias.expand(feature.size()[0], bias.size()[1])
            feature = torch.tanh(feature).unsqueeze(0)
            feature_list.append(feature)

        return torch.cat(feature_list, 0).squeeze()

    def element_wise_mul(self, input1, input2, return_not_sum_result=False):
        feature_list = []
        for feature_1, feature_2 in zip(input1, input2):
            feature_2 = feature_2.unsqueeze(1)
            feature_2 = feature_2.expand_as(feature_1)
            feature = feature_1 * feature_2
            feature = feature.unsqueeze(0)
            feature_list.append(feature)
        output = torch.cat(feature_list, 0)

        result = torch.sum(output, 1)
        if return_not_sum_result:
            return result, output
        else:
            return result

    def reduce(self, nodes):
        """Take an average over all neighbor node features hu and use it to
        overwrite the original node feature."""
        m = nodes.mailbox['m']
        accum = torch.sum(m, 1)
        return {'h': accum}

    def pad_dgl_graph(self, graphs, max_node_num):
        graphs_padded = []
        for graph in graphs:
            graph_padded = copy.deepcopy(graph)
            node_num = graph.number_of_nodes()
            graph_padded.add_nodes(max_node_num - node_num)
            graphs_padded.append(graph_padded)
        return graphs_padded

    def no_grad_for_acd_parameter(self):
        pass

    def _get_model_visualization_picture_filepath(self, configuration: dict, words: list):
        savefig_dir = configuration['savefig_dir']
        if not savefig_dir:
            return None
        filename = '%s-%s.svg' % ('-'.join(words[:3]), str(time.time()))
        filename = re.sub('/', '', filename)
        return os.path.join(savefig_dir, filename)


class Cae(TextInAllAspectSentimentOutModel):
    def __init__(self, word_embedder: TextFieldEmbedder, position_embedder: TextFieldEmbedder,
                 aspect_embedder: TextFieldEmbedder,
                 categories: list, polarities: list, vocab: Vocabulary, configuration: dict,
                 category_loss_weight=1, sentiment_loss_weight=1):
        super().__init__(vocab, category_loss_weight=category_loss_weight, sentiment_loss_weight=sentiment_loss_weight)
        self.configuration = configuration
        self.word_embedder = word_embedder
        self.position_embedder = position_embedder
        self.categories = categories
        self.polarites = polarities
        self.category_num = len(categories)
        self.polarity_num = len(polarities)
        self.category_loss = nn.BCEWithLogitsLoss()
        self.sentiment_loss = nn.CrossEntropyLoss()
        self._accuracy = metrics.CategoricalAccuracy()
        self._f1 = allennlp_metrics.BinaryF1(0.5)

        self.embedding_layer_fc = nn.Linear(word_embedder.get_output_dim(), word_embedder.get_output_dim(), bias=True)
        self.embedding_layer_aspect_attentions = [AttentionInHtt(word_embedder.get_output_dim(),
                                                                 word_embedder.get_output_dim())
                                                  for _ in range(self.category_num)]
        self.embedding_layer_aspect_attentions = nn.ModuleList(self.embedding_layer_aspect_attentions)
        self.lstm_layer_aspect_attentions = [AttentionInHtt(word_embedder.get_output_dim(),
                                                                 word_embedder.get_output_dim())
                                                  for _ in range(self.category_num)]
        self.lstm_layer_aspect_attentions = nn.ModuleList(self.lstm_layer_aspect_attentions)
        self.lstm = torch.nn.LSTM(word_embedder.get_output_dim(), int(word_embedder.get_output_dim() / 2), batch_first=True,
                                  bidirectional=True)
        self.dropout_after_embedding = nn.Dropout(0.5)
        self.dropout_after_lstm = nn.Dropout(0.5)
        self.category_fcs = [nn.Sequential(nn.Linear(word_embedder.get_output_dim() * 2, 32),
                                          nn.ReLU(),
                                          nn.Linear(32, 1)) for _ in range(self.category_num)]
        self.category_fcs = nn.ModuleList(self.category_fcs)
        self.sentiment_fc = nn.Sequential(nn.Linear(word_embedder.get_output_dim() * 2, 32),
                                          nn.ReLU(),
                                          nn.Linear(32, self.polarity_num))

    def forward(self, tokens: Dict[str, torch.Tensor], label: torch.Tensor, position: torch.Tensor,
                polarity_mask: torch.Tensor, sample: list, aspects: torch.Tensor) -> torch.Tensor:
        mask = get_text_field_mask(tokens)
        word_embeddings = self.word_embedder(tokens)
        word_embeddings = self.dropout_after_embedding(word_embeddings)

        embeddings = word_embeddings
        embedding_layer_category_outputs = []
        embedding_layer_category_alphas = []
        embedding_layer_sentiment_outputs = []
        embedding_layer_sentiment_alphas = []
        for i in range(self.category_num):
            embedding_layer_aspect_attention = self.embedding_layer_aspect_attentions[i]
            alpha = embedding_layer_aspect_attention(embeddings, mask)
            embedding_layer_category_alphas.append(alpha)

            category_output = self.element_wise_mul(embeddings, alpha, return_not_sum_result=False)
            embedding_layer_category_outputs.append(category_output)

            # sentiment
            category_output = category_output.unsqueeze(1)
            sentiment_alpha = torch.matmul(category_output, embeddings.transpose(1, 2))
            sentiment_alpha = sentiment_alpha.squeeze(1)
            sentiment_alpha = allennlp_util.masked_softmax(sentiment_alpha, mask)
            embedding_layer_sentiment_alphas.append(sentiment_alpha)
            sentiment_alpha = sentiment_alpha.unsqueeze(1)
            sentiment_output = torch.matmul(sentiment_alpha, word_embeddings).squeeze(1)  # batch_size x 2*hidden_dim
            embedding_layer_sentiment_outputs.append(sentiment_output)

        lstm_result, _ = self.lstm(word_embeddings)
        lstm_result = self.dropout_after_lstm(lstm_result)
        lstm_layer_category_outputs = []
        lstm_layer_category_alphas = []
        lstm_layer_sentiment_outputs = []
        lstm_layer_sentiment_alphas = []
        for i in range(self.category_num):
            lstm_layer_aspect_attention = self.lstm_layer_aspect_attentions[i]
            alpha = lstm_layer_aspect_attention(lstm_result, mask)
            lstm_layer_category_alphas.append(alpha)
            category_output = self.element_wise_mul(lstm_result, alpha, return_not_sum_result=False)
            lstm_layer_category_outputs.append(category_output)

            # sentiment
            category_output = category_output.unsqueeze(1)
            sentiment_alpha = torch.matmul(category_output, lstm_result.transpose(1, 2))
            sentiment_alpha = sentiment_alpha.squeeze(1)
            sentiment_alpha = allennlp_util.masked_softmax(sentiment_alpha, mask)
            lstm_layer_sentiment_alphas.append(sentiment_alpha)
            sentiment_alpha = sentiment_alpha.unsqueeze(1)
            sentiment_output = torch.matmul(sentiment_alpha, lstm_result).squeeze(1)  # batch_size x 2*hidden_dim
            lstm_layer_sentiment_outputs.append(sentiment_output)

        final_category_outputs = []
        final_sentiment_outputs = []
        for i in range(self.category_num):
            fc = self.category_fcs[i]
            category_output = torch.cat([embedding_layer_category_outputs[i], lstm_layer_category_outputs[i]], dim=-1)
            final_category_output = fc(category_output)
            final_category_outputs.append(final_category_output)

            sentiment_output = torch.cat([embedding_layer_sentiment_outputs[i], lstm_layer_sentiment_outputs[i]],
                                         dim=-1)
            final_sentiment_output = self.sentiment_fc(sentiment_output)
            final_sentiment_outputs.append(final_sentiment_output)

        output = {}
        if label is not None:
            category_labels = []
            polarity_labels = []
            polarity_masks = []
            for i in range(self.category_num):
                category_labels.append(label[:, i])
                polarity_labels.append(label[:, i + self.category_num])
                polarity_masks.append(polarity_mask[:, i])
            loss = 0
            for i in range(self.category_num):
                category_temp_loss = self.category_loss(final_category_outputs[i].squeeze(dim=-1), category_labels[i])
                sentiment_temp_loss = self.sentiment_loss(final_sentiment_outputs[i], polarity_labels[i].long())
                loss += category_temp_loss
                loss += sentiment_temp_loss

            # sentiment accuracy
            sentiment_logit = torch.cat(final_sentiment_outputs)
            sentiment_label = torch.cat(polarity_labels)
            sentiment_mask = torch.cat(polarity_masks)
            self._accuracy(sentiment_logit, sentiment_label, sentiment_mask)

            # category f1
            final_category_outputs_prob = [torch.sigmoid(e) for e in final_category_outputs]
            category_prob = torch.cat(final_category_outputs_prob).squeeze()
            category_label = torch.cat(category_labels)
            self._f1(category_prob, category_label)

            output['loss'] = loss

        # visualize attention
        pred_category = [torch.sigmoid(e) for e in final_category_outputs]
        pred_sentiment = [torch.nn.functional.softmax(e, dim=-1) for e in final_sentiment_outputs]
        output['pred_category'] = pred_category
        output['pred_sentiment'] = pred_sentiment
        if self.configuration['visualize_attention']:
            for i in range(len(sample)):
                words = sample[i][2]
                attention_labels = [e.split('/')[0] for e in self.categories]

                # category embedding layer
                visual_attentions = [embedding_layer_category_alphas[j][i][: len(words)].detach().cpu().numpy()
                                     for j in range(self.category_num)]
                titles = ['ce-true: %s - pred: %s' % (str(label[i][j].detach().cpu().numpy()),
                                                   str(pred_category[j][i].detach().cpu().numpy()))
                          for j in range(self.category_num)]
                attention_visualizer.plot_multi_attentions_of_sentence(words, visual_attentions, attention_labels,
                                                                       titles)

                # category lstm layer
                visual_attentions = [lstm_layer_category_alphas[j][i][: len(words)].detach().cpu().numpy()
                                     for j in range(self.category_num)]
                titles = ['cl-true: %s - pred: %s' % (str(label[i][j].detach().cpu().numpy()),
                                                   str(pred_category[j][i].detach().cpu().numpy()))
                          for j in range(self.category_num)]
                attention_visualizer.plot_multi_attentions_of_sentence(words, visual_attentions, attention_labels,
                                                                       titles)

                # sentiment embedding layer
                visual_attentions = [embedding_layer_sentiment_alphas[j][i][: len(words)].detach().cpu().numpy()
                                     for j in range(self.category_num)]
                titles = ['se-true: %s - pred: %s - %s' % (str(label[i][j + self.category_num].detach().cpu().numpy()),
                                                        str(pred_sentiment[j][i].detach().cpu().numpy()),
                                                        str(self.polarites))
                          for j in range(self.category_num)]
                attention_visualizer.plot_multi_attentions_of_sentence(words, visual_attentions, attention_labels,
                                                                       titles)

                # sentiment lstm layer
                visual_attentions = [lstm_layer_sentiment_alphas[j][i][: len(words)].detach().cpu().numpy()
                                     for j in range(self.category_num)]
                titles = ['sl-true: %s - pred: %s - %s' % (str(label[i][j + self.category_num].detach().cpu().numpy()),
                                                        str(pred_sentiment[j][i].detach().cpu().numpy()),
                                                        str(self.polarites))
                          for j in range(self.category_num)]
                attention_visualizer.plot_multi_attentions_of_sentence(words, visual_attentions, attention_labels,
                                                                       titles)

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {
            'accuracy': self._accuracy.get_metric(reset),
            'category_f1': self._f1.get_metric(reset)['fscore']
        }
        return metrics


class AOA(TextInAllAspectSentimentOutModel):
    def __init__(self, word_embedder: TextFieldEmbedder, position_embedder: TextFieldEmbedder,
                 aspect_embedder: TextFieldEmbedder,
                 categories: list, polarities: list, vocab: Vocabulary, configuration: dict,
                 category_loss_weight=1, sentiment_loss_weight=1):
        super().__init__(vocab, category_loss_weight=category_loss_weight, sentiment_loss_weight=sentiment_loss_weight)
        self.configuration = configuration
        self.word_embedder = word_embedder
        self.position_embedder = position_embedder
        self.categories = categories
        self.polarites = polarities
        self.category_num = len(categories)
        self.polarity_num = len(polarities)
        self.category_loss = nn.BCEWithLogitsLoss()
        self.sentiment_loss = nn.CrossEntropyLoss()
        self._accuracy = metrics.CategoricalAccuracy()
        self._f1 = allennlp_metrics.BinaryF1(0.5)

        word_embedding_dim = word_embedder.get_output_dim()
        word_embedding_with_position_dim = word_embedding_dim
        if self.configuration['position']:
            word_embedding_with_position_dim += position_embedder.get_output_dim()

        self.embedding_layer_fc = nn.Linear(word_embedding_with_position_dim, word_embedding_dim, bias=True)

        self.embedding_layer_aspect_attentions = [AttentionInHtt(word_embedding_dim,
                                                                 word_embedding_dim)
                                                  for _ in range(self.category_num)]
        self.embedding_layer_aspect_attentions = nn.ModuleList(self.embedding_layer_aspect_attentions)

        self.lstm = torch.nn.LSTM(word_embedding_with_position_dim, int(word_embedding_dim / 2), batch_first=True,
                                  bidirectional=True)
        self.lstm_layer_aspect_attentions = [AttentionInHtt(word_embedding_dim, word_embedding_dim)
                                                  for _ in range(self.category_num)]
        self.lstm_layer_aspect_attentions = nn.ModuleList(self.lstm_layer_aspect_attentions)

        self.dropout_after_embedding = nn.Dropout(0.5)
        self.dropout_after_lstm = nn.Dropout(0.5)
        self.category_fcs = [nn.Sequential(nn.Linear(word_embedding_dim * 2, word_embedding_dim),
                                          nn.ReLU(),
                                          nn.Linear(word_embedding_dim, 1)) for _ in range(self.category_num)]
        self.category_fcs = nn.ModuleList(self.category_fcs)
        self.sentiment_fc = nn.Sequential(nn.Linear(word_embedding_dim * 2, word_embedding_dim),
                                          nn.ReLU(),
                                          nn.Linear(word_embedding_dim, self.polarity_num))

    def aoa(self, context_tensor, aspect_tensor):
        """

        :param context_tensor: batch_size * length * dim
        :param aspect_tensor: batch_size * length * dim
        :return:
        """
        interaction_mat = torch.matmul(context_tensor,
                                       torch.transpose(aspect_tensor, 1, 2))  # batch_size x (ctx) seq_len x (asp) seq_len
        alpha = F.softmax(interaction_mat, dim=1)  # col-wise, batch_size x (ctx) seq_len x (asp) seq_len
        beta = F.softmax(interaction_mat, dim=2)  # row-wise, batch_size x (ctx) seq_len x (asp) seq_len
        beta_avg = beta.mean(dim=1, keepdim=True)  # batch_size x 1 x (asp) seq_len
        gamma = torch.matmul(alpha, beta_avg.transpose(1, 2))  # batch_size x (ctx) seq_len x 1
        weighted_sum = torch.matmul(torch.transpose(context_tensor, 1, 2), gamma).squeeze(-1)
        return weighted_sum, gamma.squeeze(dim=-1)

    def forward(self, tokens: Dict[str, torch.Tensor], label: torch.Tensor, position: torch.Tensor,
                polarity_mask: torch.Tensor, sample: list, aspects: torch.Tensor) -> torch.Tensor:
        mask = get_text_field_mask(tokens)
        word_embeddings = self.word_embedder(tokens)
        if self.configuration['position']:
            position_embeddings = self.position_embedder(position)
            word_embeddings = torch.cat([word_embeddings, position_embeddings], dim=-1)
        word_embeddings = self.dropout_after_embedding(word_embeddings)

        embeddings = self.embedding_layer_fc(word_embeddings)
        embedding_layer_category_outputs = []
        embedding_layer_category_alphas = []
        embedding_layer_sentiment_outputs = []
        embedding_layer_sentiment_alphas = []
        for i in range(self.category_num):
            embedding_layer_aspect_attention = self.embedding_layer_aspect_attentions[i]
            alpha = embedding_layer_aspect_attention(embeddings, mask)
            embedding_layer_category_alphas.append(alpha)

            category_output, category_output_out_sum = self.element_wise_mul(embeddings, alpha, return_not_sum_result=True)
            embedding_layer_category_outputs.append(category_output)

            # sentiment
            sentiment_output, sentiment_alpha = self.aoa(embeddings, category_output_out_sum)
            # cae
            # category_output = category_output.unsqueeze(1)
            # sentiment_alpha = torch.matmul(category_output, embeddings.transpose(1, 2))
            # sentiment_alpha = sentiment_alpha.squeeze(1)
            # sentiment_alpha = allennlp_util.masked_softmax(sentiment_alpha, mask)
            embedding_layer_sentiment_alphas.append(sentiment_alpha)
            # sentiment_alpha = sentiment_alpha.unsqueeze(1)
            # sentiment_output = torch.matmul(sentiment_alpha, word_embeddings).squeeze(1)  # batch_size x 2*hidden_dim
            embedding_layer_sentiment_outputs.append(sentiment_output)

        lstm_input = word_embeddings
        lstm_result, _ = self.lstm(lstm_input)
        lstm_result = self.dropout_after_lstm(lstm_result)
        lstm_layer_category_outputs = []
        lstm_layer_category_alphas = []
        lstm_layer_sentiment_outputs = []
        lstm_layer_sentiment_alphas = []
        for i in range(self.category_num):
            lstm_layer_aspect_attention = self.lstm_layer_aspect_attentions[i]
            alpha = lstm_layer_aspect_attention(lstm_result, mask)
            lstm_layer_category_alphas.append(alpha)
            category_output, category_output_not_sum = self.element_wise_mul(lstm_result, alpha, return_not_sum_result=True)
            lstm_layer_category_outputs.append(category_output)

            # sentiment
            sentiment_output, sentiment_alpha = self.aoa(lstm_result, category_output_out_sum)
            # cae
            # category_output = category_output.unsqueeze(1)
            # sentiment_alpha = torch.matmul(category_output, lstm_result.transpose(1, 2))
            # sentiment_alpha = sentiment_alpha.squeeze(1)
            # sentiment_alpha = allennlp_util.masked_softmax(sentiment_alpha, mask)
            lstm_layer_sentiment_alphas.append(sentiment_alpha)
            # sentiment_alpha = sentiment_alpha.unsqueeze(1)
            # sentiment_output = torch.matmul(sentiment_alpha, lstm_result).squeeze(1)  # batch_size x 2*hidden_dim
            lstm_layer_sentiment_outputs.append(sentiment_output)

        final_category_outputs = []
        final_sentiment_outputs = []
        for i in range(self.category_num):
            fc = self.category_fcs[i]
            category_output = torch.cat([embedding_layer_category_outputs[i], lstm_layer_category_outputs[i]], dim=-1)
            final_category_output = fc(category_output)
            final_category_outputs.append(final_category_output)

            sentiment_output = torch.cat([embedding_layer_sentiment_outputs[i], lstm_layer_sentiment_outputs[i]],
                                         dim=-1)
            final_sentiment_output = self.sentiment_fc(sentiment_output)
            final_sentiment_outputs.append(final_sentiment_output)

        output = {}
        if label is not None:
            category_labels = []
            polarity_labels = []
            polarity_masks = []
            for i in range(self.category_num):
                category_labels.append(label[:, i])
                polarity_labels.append(label[:, i + self.category_num])
                polarity_masks.append(polarity_mask[:, i])
            loss = 0
            for i in range(self.category_num):
                category_temp_loss = self.category_loss(final_category_outputs[i].squeeze(dim=-1), category_labels[i])
                sentiment_temp_loss = self.sentiment_loss(final_sentiment_outputs[i], polarity_labels[i].long())
                loss += category_temp_loss
                loss += sentiment_temp_loss

            # sentiment accuracy
            sentiment_logit = torch.cat(final_sentiment_outputs)
            sentiment_label = torch.cat(polarity_labels)
            sentiment_mask = torch.cat(polarity_masks)
            self._accuracy(sentiment_logit, sentiment_label, sentiment_mask)

            # category f1
            final_category_outputs_prob = [torch.sigmoid(e) for e in final_category_outputs]
            category_prob = torch.cat(final_category_outputs_prob).squeeze()
            category_label = torch.cat(category_labels)
            self._f1(category_prob, category_label)

            output['loss'] = loss

        # visualize attention
        pred_category = [torch.sigmoid(e) for e in final_category_outputs]
        pred_sentiment = [torch.nn.functional.softmax(e, dim=-1) for e in final_sentiment_outputs]
        output['pred_category'] = pred_category
        output['pred_sentiment'] = pred_sentiment
        if self.configuration['visualize_attention']:
            for i in range(len(sample)):
                words = sample[i][2]
                attention_labels = [e.split('/')[0] for e in self.categories]

                # category embedding layer
                visual_attentions = [embedding_layer_category_alphas[j][i][: len(words)].detach().cpu().numpy()
                                     for j in range(self.category_num)]
                titles = ['ce-true: %s - pred: %s' % (str(label[i][j].detach().cpu().numpy()),
                                                   str(pred_category[j][i].detach().cpu().numpy()))
                          for j in range(self.category_num)]
                attention_visualizer.plot_multi_attentions_of_sentence(words, visual_attentions, attention_labels,
                                                                       titles)

                # category lstm layer
                visual_attentions = [lstm_layer_category_alphas[j][i][: len(words)].detach().cpu().numpy()
                                     for j in range(self.category_num)]
                titles = ['cl-true: %s - pred: %s' % (str(label[i][j].detach().cpu().numpy()),
                                                   str(pred_category[j][i].detach().cpu().numpy()))
                          for j in range(self.category_num)]
                attention_visualizer.plot_multi_attentions_of_sentence(words, visual_attentions, attention_labels,
                                                                       titles)

                # sentiment embedding layer
                visual_attentions = [embedding_layer_sentiment_alphas[j][i][: len(words)].detach().cpu().numpy()
                                     for j in range(self.category_num)]
                titles = ['se-true: %s - pred: %s - %s' % (str(label[i][j + self.category_num].detach().cpu().numpy()),
                                                        str(pred_sentiment[j][i].detach().cpu().numpy()),
                                                        str(self.polarites))
                          for j in range(self.category_num)]
                attention_visualizer.plot_multi_attentions_of_sentence(words, visual_attentions, attention_labels,
                                                                       titles)

                # sentiment lstm layer
                visual_attentions = [lstm_layer_sentiment_alphas[j][i][: len(words)].detach().cpu().numpy()
                                     for j in range(self.category_num)]
                titles = ['sl-true: %s - pred: %s - %s' % (str(label[i][j + self.category_num].detach().cpu().numpy()),
                                                        str(pred_sentiment[j][i].detach().cpu().numpy()),
                                                        str(self.polarites))
                          for j in range(self.category_num)]
                attention_visualizer.plot_multi_attentions_of_sentence(words, visual_attentions, attention_labels,
                                                                       titles)

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {
            'accuracy': self._accuracy.get_metric(reset),
            'category_f1': self._f1.get_metric(reset)['fscore']
        }
        return metrics


class CaeAverageOfTwoLayer(TextInAllAspectSentimentOutModel):
    def __init__(self, word_embedder: TextFieldEmbedder, position_embedder: TextFieldEmbedder,
                 aspect_embedder: TextFieldEmbedder,
                 categories: list, polarities: list, vocab: Vocabulary, configuration: dict,
                 category_loss_weight=1, sentiment_loss_weight=1):
        super().__init__(vocab, category_loss_weight=category_loss_weight, sentiment_loss_weight=sentiment_loss_weight)
        self.configuration = configuration
        self.word_embedder = word_embedder
        self.position_embedder = position_embedder
        self.categories = categories
        self.polarites = polarities
        self.category_num = len(categories)
        self.polarity_num = len(polarities)
        self.category_loss = nn.BCEWithLogitsLoss()
        self.sentiment_loss = nn.CrossEntropyLoss()
        self._accuracy = metrics.CategoricalAccuracy()
        self._f1 = allennlp_metrics.BinaryF1(0.5)

        self.embedding_layer_fc = nn.Linear(word_embedder.get_output_dim(), word_embedder.get_output_dim(), bias=True)
        self.embedding_layer_aspect_attentions = [AttentionInHtt(word_embedder.get_output_dim(),
                                                                 word_embedder.get_output_dim())
                                                  for _ in range(self.category_num)]
        self.embedding_layer_aspect_attentions = nn.ModuleList(self.embedding_layer_aspect_attentions)
        self.lstm_layer_aspect_attentions = [AttentionInHtt(word_embedder.get_output_dim(),
                                                                 word_embedder.get_output_dim())
                                                  for _ in range(self.category_num)]
        self.lstm_layer_aspect_attentions = nn.ModuleList(self.lstm_layer_aspect_attentions)
        self.lstm = torch.nn.LSTM(word_embedder.get_output_dim(), int(word_embedder.get_output_dim() / 2), batch_first=True,
                                  bidirectional=True)
        self.dropout_after_embedding = nn.Dropout(0.5)
        self.dropout_after_lstm = nn.Dropout(0.5)
        self.category_fcs = [nn.Sequential(nn.Linear(word_embedder.get_output_dim(), 32),
                                          nn.ReLU(),
                                          nn.Linear(32, 1)) for _ in range(self.category_num)]
        self.category_fcs = nn.ModuleList(self.category_fcs)
        self.sentiment_fc = nn.Sequential(nn.Linear(word_embedder.get_output_dim(), 32),
                                          nn.ReLU(),
                                          nn.Linear(32, self.polarity_num))

    def forward(self, tokens: Dict[str, torch.Tensor], label: torch.Tensor, position: torch.Tensor,
                polarity_mask: torch.Tensor, sample: list, aspects: torch.Tensor) -> torch.Tensor:
        mask = get_text_field_mask(tokens)
        word_embeddings = self.word_embedder(tokens)
        word_embeddings = self.dropout_after_embedding(word_embeddings)

        embeddings = word_embeddings
        embedding_layer_category_outputs = []
        embedding_layer_category_alphas = []
        embedding_layer_sentiment_outputs = []
        embedding_layer_sentiment_alphas = []
        for i in range(self.category_num):
            embedding_layer_aspect_attention = self.embedding_layer_aspect_attentions[i]
            alpha = embedding_layer_aspect_attention(embeddings, mask)
            embedding_layer_category_alphas.append(alpha)

            category_output = self.element_wise_mul(embeddings, alpha, return_not_sum_result=False)
            embedding_layer_category_outputs.append(category_output)

            # sentiment
            category_output = category_output.unsqueeze(1)
            sentiment_alpha = torch.matmul(category_output, embeddings.transpose(1, 2))
            sentiment_alpha = sentiment_alpha.squeeze(1)
            sentiment_alpha = allennlp_util.masked_softmax(sentiment_alpha, mask)
            embedding_layer_sentiment_alphas.append(sentiment_alpha)
            sentiment_alpha = sentiment_alpha.unsqueeze(1)
            sentiment_output = torch.matmul(sentiment_alpha, word_embeddings).squeeze(1)  # batch_size x 2*hidden_dim
            embedding_layer_sentiment_outputs.append(sentiment_output)

        lstm_result, _ = self.lstm(word_embeddings)
        lstm_result = self.dropout_after_lstm(lstm_result)
        lstm_layer_category_outputs = []
        lstm_layer_category_alphas = []
        lstm_layer_sentiment_outputs = []
        lstm_layer_sentiment_alphas = []
        for i in range(self.category_num):
            lstm_layer_aspect_attention = self.lstm_layer_aspect_attentions[i]
            alpha = lstm_layer_aspect_attention(lstm_result, mask)
            lstm_layer_category_alphas.append(alpha)
            category_output = self.element_wise_mul(lstm_result, alpha, return_not_sum_result=False)
            lstm_layer_category_outputs.append(category_output)

            # sentiment
            category_output = category_output.unsqueeze(1)
            sentiment_alpha = torch.matmul(category_output, lstm_result.transpose(1, 2))
            sentiment_alpha = sentiment_alpha.squeeze(1)
            sentiment_alpha = allennlp_util.masked_softmax(sentiment_alpha, mask)
            lstm_layer_sentiment_alphas.append(sentiment_alpha)
            sentiment_alpha = sentiment_alpha.unsqueeze(1)
            sentiment_output = torch.matmul(sentiment_alpha, lstm_result).squeeze(1)  # batch_size x 2*hidden_dim
            lstm_layer_sentiment_outputs.append(sentiment_output)

        final_category_outputs = []
        final_sentiment_outputs = []
        for i in range(self.category_num):
            fc = self.category_fcs[i]
            final_category_output = (fc(embedding_layer_category_outputs[i]) + fc(lstm_layer_category_outputs[i])) / 2
            final_category_outputs.append(final_category_output)

            final_sentiment_output = (self.sentiment_fc(embedding_layer_sentiment_outputs[i]) +
                                      self.sentiment_fc(lstm_layer_sentiment_outputs[i])) / 2
            final_sentiment_outputs.append(final_sentiment_output)

        output = {}
        if label is not None:
            category_labels = []
            polarity_labels = []
            polarity_masks = []
            for i in range(self.category_num):
                category_labels.append(label[:, i])
                polarity_labels.append(label[:, i + self.category_num])
                polarity_masks.append(polarity_mask[:, i])
            loss = 0
            for i in range(self.category_num):
                category_temp_loss = self.category_loss(final_category_outputs[i].squeeze(dim=-1), category_labels[i])
                sentiment_temp_loss = self.sentiment_loss(final_sentiment_outputs[i], polarity_labels[i].long())
                loss += category_temp_loss
                loss += sentiment_temp_loss

            # sentiment accuracy
            sentiment_logit = torch.cat(final_sentiment_outputs)
            sentiment_label = torch.cat(polarity_labels)
            sentiment_mask = torch.cat(polarity_masks)
            self._accuracy(sentiment_logit, sentiment_label, sentiment_mask)

            # category f1
            final_category_outputs_prob = [torch.sigmoid(e) for e in final_category_outputs]
            category_prob = torch.cat(final_category_outputs_prob).squeeze()
            category_label = torch.cat(category_labels)
            self._f1(category_prob, category_label)

            output['loss'] = loss

        # visualize attention
        pred_category = [torch.sigmoid(e) for e in final_category_outputs]
        pred_sentiment = [torch.nn.functional.softmax(e, dim=-1) for e in final_sentiment_outputs]
        output['pred_category'] = pred_category
        output['pred_sentiment'] = pred_sentiment
        if self.configuration['visualize_attention']:
            for i in range(len(sample)):
                words = sample[i][2]
                attention_labels = [e.split('/')[0] for e in self.categories]

                # category embedding layer
                visual_attentions = [embedding_layer_category_alphas[j][i][: len(words)].detach().cpu().numpy()
                                     for j in range(self.category_num)]
                titles = ['ce-true: %s - pred: %s' % (str(label[i][j].detach().cpu().numpy()),
                                                   str(pred_category[j][i].detach().cpu().numpy()))
                          for j in range(self.category_num)]
                attention_visualizer.plot_multi_attentions_of_sentence(words, visual_attentions, attention_labels,
                                                                       titles)

                # category lstm layer
                visual_attentions = [lstm_layer_category_alphas[j][i][: len(words)].detach().cpu().numpy()
                                     for j in range(self.category_num)]
                titles = ['cl-true: %s - pred: %s' % (str(label[i][j].detach().cpu().numpy()),
                                                   str(pred_category[j][i].detach().cpu().numpy()))
                          for j in range(self.category_num)]
                attention_visualizer.plot_multi_attentions_of_sentence(words, visual_attentions, attention_labels,
                                                                       titles)

                # sentiment embedding layer
                visual_attentions = [embedding_layer_sentiment_alphas[j][i][: len(words)].detach().cpu().numpy()
                                     for j in range(self.category_num)]
                titles = ['se-true: %s - pred: %s - %s' % (str(label[i][j + self.category_num].detach().cpu().numpy()),
                                                        str(pred_sentiment[j][i].detach().cpu().numpy()),
                                                        str(self.polarites))
                          for j in range(self.category_num)]
                attention_visualizer.plot_multi_attentions_of_sentence(words, visual_attentions, attention_labels,
                                                                       titles)

                # sentiment lstm layer
                visual_attentions = [lstm_layer_sentiment_alphas[j][i][: len(words)].detach().cpu().numpy()
                                     for j in range(self.category_num)]
                titles = ['sl-true: %s - pred: %s - %s' % (str(label[i][j + self.category_num].detach().cpu().numpy()),
                                                        str(pred_sentiment[j][i].detach().cpu().numpy()),
                                                        str(self.polarites))
                          for j in range(self.category_num)]
                attention_visualizer.plot_multi_attentions_of_sentence(words, visual_attentions, attention_labels,
                                                                       titles)

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {
            'accuracy': self._accuracy.get_metric(reset),
            'category_f1': self._f1.get_metric(reset)['fscore']
        }
        return metrics


class CaeAdd(TextInAllAspectSentimentOutModel):
    def __init__(self, word_embedder: TextFieldEmbedder, position_embedder: TextFieldEmbedder,
                 aspect_embedder: TextFieldEmbedder,
                 categories: list, polarities: list, vocab: Vocabulary, configuration: dict,
                 category_loss_weight=1, sentiment_loss_weight=1):
        super().__init__(vocab, category_loss_weight=category_loss_weight, sentiment_loss_weight=sentiment_loss_weight)
        self.configuration = configuration
        self.word_embedder = word_embedder
        self.position_embedder = position_embedder
        self.categories = categories
        self.polarites = polarities
        self.category_num = len(categories)
        self.polarity_num = len(polarities)
        self.category_loss = nn.BCEWithLogitsLoss()
        self.sentiment_loss = nn.CrossEntropyLoss()
        self._accuracy = metrics.CategoricalAccuracy()
        self._f1 = allennlp_metrics.BinaryF1(0.5)

        self.embedding_layer_fc = nn.Linear(word_embedder.get_output_dim(), word_embedder.get_output_dim(), bias=True)
        self.embedding_layer_aspect_attentions = [AttentionInHtt(word_embedder.get_output_dim(),
                                                                 word_embedder.get_output_dim())
                                                  for _ in range(self.category_num)]
        self.embedding_layer_aspect_attentions = nn.ModuleList(self.embedding_layer_aspect_attentions)
        self.lstm_layer_aspect_attentions = [AttentionInHtt(word_embedder.get_output_dim(),
                                                                 word_embedder.get_output_dim())
                                                  for _ in range(self.category_num)]
        self.lstm_layer_aspect_attentions = nn.ModuleList(self.lstm_layer_aspect_attentions)
        self.lstm = torch.nn.LSTM(word_embedder.get_output_dim(), int(word_embedder.get_output_dim() / 2), batch_first=True,
                                  bidirectional=True)
        self.dropout_after_embedding = nn.Dropout(0.5)
        self.dropout_after_lstm = nn.Dropout(0.5)
        self.category_fcs = [nn.Sequential(nn.Linear(word_embedder.get_output_dim(), 32),
                                          nn.ReLU(),
                                          nn.Linear(32, 1)) for _ in range(self.category_num)]
        self.category_fcs = nn.ModuleList(self.category_fcs)
        self.sentiment_fc = nn.Sequential(nn.Linear(word_embedder.get_output_dim(), 32),
                                          nn.ReLU(),
                                          nn.Linear(32, self.polarity_num))

    def forward(self, tokens: Dict[str, torch.Tensor], label: torch.Tensor, position: torch.Tensor,
                polarity_mask: torch.Tensor, sample: list, aspects: torch.Tensor) -> torch.Tensor:
        mask = get_text_field_mask(tokens)
        word_embeddings = self.word_embedder(tokens)
        word_embeddings = self.dropout_after_embedding(word_embeddings)

        embeddings = word_embeddings
        embedding_layer_category_outputs = []
        embedding_layer_category_alphas = []
        embedding_layer_sentiment_outputs = []
        embedding_layer_sentiment_alphas = []
        for i in range(self.category_num):
            embedding_layer_aspect_attention = self.embedding_layer_aspect_attentions[i]
            alpha = embedding_layer_aspect_attention(embeddings, mask)
            embedding_layer_category_alphas.append(alpha)

            category_output = self.element_wise_mul(embeddings, alpha, return_not_sum_result=False)
            embedding_layer_category_outputs.append(category_output)

            # sentiment
            category_output = category_output.unsqueeze(1)
            sentiment_alpha = torch.matmul(category_output, embeddings.transpose(1, 2))
            sentiment_alpha = sentiment_alpha.squeeze(1)
            sentiment_alpha = allennlp_util.masked_softmax(sentiment_alpha, mask)
            embedding_layer_sentiment_alphas.append(sentiment_alpha)
            sentiment_alpha = sentiment_alpha.unsqueeze(1)
            sentiment_output = torch.matmul(sentiment_alpha, word_embeddings).squeeze(1)  # batch_size x 2*hidden_dim
            embedding_layer_sentiment_outputs.append(sentiment_output)

        lstm_result, _ = self.lstm(word_embeddings)
        lstm_result = self.dropout_after_lstm(lstm_result)
        lstm_layer_category_outputs = []
        lstm_layer_category_alphas = []
        lstm_layer_sentiment_outputs = []
        lstm_layer_sentiment_alphas = []
        for i in range(self.category_num):
            lstm_layer_aspect_attention = self.lstm_layer_aspect_attentions[i]
            alpha = lstm_layer_aspect_attention(lstm_result, mask)
            lstm_layer_category_alphas.append(alpha)
            category_output = self.element_wise_mul(lstm_result, alpha, return_not_sum_result=False)
            lstm_layer_category_outputs.append(category_output)

            # sentiment
            category_output = category_output.unsqueeze(1)
            sentiment_alpha = torch.matmul(category_output, lstm_result.transpose(1, 2))
            sentiment_alpha = sentiment_alpha.squeeze(1)
            sentiment_alpha = allennlp_util.masked_softmax(sentiment_alpha, mask)
            lstm_layer_sentiment_alphas.append(sentiment_alpha)
            sentiment_alpha = sentiment_alpha.unsqueeze(1)
            sentiment_output = torch.matmul(sentiment_alpha, lstm_result).squeeze(1)  # batch_size x 2*hidden_dim
            lstm_layer_sentiment_outputs.append(sentiment_output)

        final_category_outputs = []
        final_sentiment_outputs = []
        for i in range(self.category_num):
            fc = self.category_fcs[i]
            category_output = embedding_layer_category_outputs[i] + lstm_layer_category_outputs[i]
            final_category_output = fc(category_output)
            final_category_outputs.append(final_category_output)

            sentiment_output = embedding_layer_sentiment_outputs[i] + lstm_layer_sentiment_outputs[i]
            final_sentiment_output = self.sentiment_fc(sentiment_output)
            final_sentiment_outputs.append(final_sentiment_output)

        output = {}
        if label is not None:
            category_labels = []
            polarity_labels = []
            polarity_masks = []
            for i in range(self.category_num):
                category_labels.append(label[:, i])
                polarity_labels.append(label[:, i + self.category_num])
                polarity_masks.append(polarity_mask[:, i])
            loss = 0
            for i in range(self.category_num):
                category_temp_loss = self.category_loss(final_category_outputs[i].squeeze(dim=-1), category_labels[i])
                sentiment_temp_loss = self.sentiment_loss(final_sentiment_outputs[i], polarity_labels[i].long())
                loss += category_temp_loss
                loss += sentiment_temp_loss

            # sentiment accuracy
            sentiment_logit = torch.cat(final_sentiment_outputs)
            sentiment_label = torch.cat(polarity_labels)
            sentiment_mask = torch.cat(polarity_masks)
            self._accuracy(sentiment_logit, sentiment_label, sentiment_mask)

            # category f1
            final_category_outputs_prob = [torch.sigmoid(e) for e in final_category_outputs]
            category_prob = torch.cat(final_category_outputs_prob).squeeze()
            category_label = torch.cat(category_labels)
            self._f1(category_prob, category_label)

            output['loss'] = loss

        # visualize attention
        pred_category = [torch.sigmoid(e) for e in final_category_outputs]
        pred_sentiment = [torch.nn.functional.softmax(e, dim=-1) for e in final_sentiment_outputs]
        output['pred_category'] = pred_category
        output['pred_sentiment'] = pred_sentiment
        if self.configuration['visualize_attention']:
            for i in range(len(sample)):
                words = sample[i][2]
                attention_labels = [e.split('/')[0] for e in self.categories]

                # category embedding layer
                visual_attentions = [embedding_layer_category_alphas[j][i][: len(words)].detach().cpu().numpy()
                                     for j in range(self.category_num)]
                titles = ['ce-true: %s - pred: %s' % (str(label[i][j].detach().cpu().numpy()),
                                                   str(pred_category[j][i].detach().cpu().numpy()))
                          for j in range(self.category_num)]
                attention_visualizer.plot_multi_attentions_of_sentence(words, visual_attentions, attention_labels,
                                                                       titles)

                # category lstm layer
                visual_attentions = [lstm_layer_category_alphas[j][i][: len(words)].detach().cpu().numpy()
                                     for j in range(self.category_num)]
                titles = ['cl-true: %s - pred: %s' % (str(label[i][j].detach().cpu().numpy()),
                                                   str(pred_category[j][i].detach().cpu().numpy()))
                          for j in range(self.category_num)]
                attention_visualizer.plot_multi_attentions_of_sentence(words, visual_attentions, attention_labels,
                                                                       titles)

                # sentiment embedding layer
                visual_attentions = [embedding_layer_sentiment_alphas[j][i][: len(words)].detach().cpu().numpy()
                                     for j in range(self.category_num)]
                titles = ['se-true: %s - pred: %s - %s' % (str(label[i][j + self.category_num].detach().cpu().numpy()),
                                                        str(pred_sentiment[j][i].detach().cpu().numpy()),
                                                        str(self.polarites))
                          for j in range(self.category_num)]
                attention_visualizer.plot_multi_attentions_of_sentence(words, visual_attentions, attention_labels,
                                                                       titles)

                # sentiment lstm layer
                visual_attentions = [lstm_layer_sentiment_alphas[j][i][: len(words)].detach().cpu().numpy()
                                     for j in range(self.category_num)]
                titles = ['sl-true: %s - pred: %s - %s' % (str(label[i][j + self.category_num].detach().cpu().numpy()),
                                                        str(pred_sentiment[j][i].detach().cpu().numpy()),
                                                        str(self.polarites))
                          for j in range(self.category_num)]
                attention_visualizer.plot_multi_attentions_of_sentence(words, visual_attentions, attention_labels,
                                                                       titles)

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {
            'accuracy': self._accuracy.get_metric(reset),
            'category_f1': self._f1.get_metric(reset)['fscore']
        }
        return metrics


class CaeAttOnlyInEmbedding(TextInAllAspectSentimentOutModel):
    def __init__(self, word_embedder: TextFieldEmbedder, position_embedder: TextFieldEmbedder,
                 aspect_embedder: TextFieldEmbedder,
                 categories: list, polarities: list, vocab: Vocabulary, configuration: dict,
                 category_loss_weight=1, sentiment_loss_weight=1):
        super().__init__(vocab, category_loss_weight=category_loss_weight, sentiment_loss_weight=sentiment_loss_weight)
        self.configuration = configuration
        self.word_embedder = word_embedder
        self.position_embedder = position_embedder
        self.categories = categories
        self.polarites = polarities
        self.category_num = len(categories)
        self.polarity_num = len(polarities)
        self.category_loss = nn.BCEWithLogitsLoss()
        self.sentiment_loss = nn.CrossEntropyLoss()
        self._accuracy = metrics.CategoricalAccuracy()
        self._f1 = allennlp_metrics.BinaryF1(0.5)

        self.embedding_layer_fc = nn.Linear(word_embedder.get_output_dim(), word_embedder.get_output_dim(), bias=True)
        self.embedding_layer_aspect_attentions = [AttentionInHtt(word_embedder.get_output_dim(),
                                                                 word_embedder.get_output_dim())
                                                  for _ in range(self.category_num)]
        self.embedding_layer_aspect_attentions = nn.ModuleList(self.embedding_layer_aspect_attentions)
        self.lstm = torch.nn.LSTM(word_embedder.get_output_dim(), int(word_embedder.get_output_dim() / 2), batch_first=True,
                                  bidirectional=True)
        self.dropout_after_embedding = nn.Dropout(0.5)
        self.dropout_after_lstm = nn.Dropout(0.5)
        self.category_fcs = [nn.Sequential(nn.Linear(word_embedder.get_output_dim() * 2, 32),
                                          nn.ReLU(),
                                          nn.Linear(32, 1)) for _ in range(self.category_num)]
        self.category_fcs = nn.ModuleList(self.category_fcs)
        self.sentiment_fc = nn.Sequential(nn.Linear(word_embedder.get_output_dim() * 2, 32),
                                          nn.ReLU(),
                                          nn.Linear(32, self.polarity_num))

    def forward(self, tokens: Dict[str, torch.Tensor], label: torch.Tensor, position: torch.Tensor,
                polarity_mask: torch.Tensor, sample: list, aspects: torch.Tensor) -> torch.Tensor:
        mask = get_text_field_mask(tokens)
        word_embeddings = self.word_embedder(tokens)
        word_embeddings = self.dropout_after_embedding(word_embeddings)

        embeddings = word_embeddings
        embedding_layer_category_outputs = []
        embedding_layer_category_alphas = []
        embedding_layer_sentiment_outputs = []
        embedding_layer_sentiment_alphas = []
        for i in range(self.category_num):
            embedding_layer_aspect_attention = self.embedding_layer_aspect_attentions[i]
            alpha = embedding_layer_aspect_attention(embeddings, mask)
            embedding_layer_category_alphas.append(alpha)

            category_output = self.element_wise_mul(embeddings, alpha, return_not_sum_result=False)
            embedding_layer_category_outputs.append(category_output)

            # sentiment
            category_output = category_output.unsqueeze(1)
            sentiment_alpha = torch.matmul(category_output, embeddings.transpose(1, 2))
            sentiment_alpha = sentiment_alpha.squeeze(1)
            sentiment_alpha = allennlp_util.masked_softmax(sentiment_alpha, mask)
            embedding_layer_sentiment_alphas.append(sentiment_alpha)
            sentiment_alpha = sentiment_alpha.unsqueeze(1)
            sentiment_output = torch.matmul(sentiment_alpha, word_embeddings).squeeze(1)  # batch_size x 2*hidden_dim
            embedding_layer_sentiment_outputs.append(sentiment_output)

        lstm_result, _ = self.lstm(word_embeddings)
        lstm_result = self.dropout_after_lstm(lstm_result)
        lstm_layer_category_outputs = []
        lstm_layer_category_alphas = []
        lstm_layer_sentiment_outputs = []
        lstm_layer_sentiment_alphas = []
        for i in range(self.category_num):
            alpha = embedding_layer_category_alphas[i]
            lstm_layer_category_alphas.append(alpha)
            category_output = self.element_wise_mul(lstm_result, alpha, return_not_sum_result=False)
            lstm_layer_category_outputs.append(category_output)

            # sentiment
            category_output = category_output.unsqueeze(1)
            sentiment_alpha = torch.matmul(category_output, lstm_result.transpose(1, 2))
            sentiment_alpha = sentiment_alpha.squeeze(1)
            sentiment_alpha = allennlp_util.masked_softmax(sentiment_alpha, mask)
            lstm_layer_sentiment_alphas.append(sentiment_alpha)
            sentiment_alpha = sentiment_alpha.unsqueeze(1)
            sentiment_output = torch.matmul(sentiment_alpha, lstm_result).squeeze(1)  # batch_size x 2*hidden_dim
            lstm_layer_sentiment_outputs.append(sentiment_output)

        final_category_outputs = []
        final_sentiment_outputs = []
        for i in range(self.category_num):
            fc = self.category_fcs[i]
            category_output = torch.cat([embedding_layer_category_outputs[i], lstm_layer_category_outputs[i]], dim=-1)
            final_category_output = fc(category_output)
            final_category_outputs.append(final_category_output)

            sentiment_output = torch.cat([embedding_layer_sentiment_outputs[i], lstm_layer_sentiment_outputs[i]],
                                         dim=-1)
            final_sentiment_output = self.sentiment_fc(sentiment_output)
            final_sentiment_outputs.append(final_sentiment_output)

        output = {}
        if label is not None:
            category_labels = []
            polarity_labels = []
            polarity_masks = []
            for i in range(self.category_num):
                category_labels.append(label[:, i])
                polarity_labels.append(label[:, i + self.category_num])
                polarity_masks.append(polarity_mask[:, i])
            loss = 0
            for i in range(self.category_num):
                category_temp_loss = self.category_loss(final_category_outputs[i].squeeze(dim=-1), category_labels[i])
                sentiment_temp_loss = self.sentiment_loss(final_sentiment_outputs[i], polarity_labels[i].long())
                loss += category_temp_loss
                loss += sentiment_temp_loss

            # sentiment accuracy
            sentiment_logit = torch.cat(final_sentiment_outputs)
            sentiment_label = torch.cat(polarity_labels)
            sentiment_mask = torch.cat(polarity_masks)
            self._accuracy(sentiment_logit, sentiment_label, sentiment_mask)

            # category f1
            final_category_outputs_prob = [torch.sigmoid(e) for e in final_category_outputs]
            category_prob = torch.cat(final_category_outputs_prob).squeeze()
            category_label = torch.cat(category_labels)
            self._f1(category_prob, category_label)

            output['loss'] = loss

        # visualize attention
        pred_category = [torch.sigmoid(e) for e in final_category_outputs]
        pred_sentiment = [torch.nn.functional.softmax(e, dim=-1) for e in final_sentiment_outputs]
        output['pred_category'] = pred_category
        output['pred_sentiment'] = pred_sentiment
        if self.configuration['visualize_attention']:
            for i in range(len(sample)):
                words = sample[i][2]
                attention_labels = [e.split('/')[0] for e in self.categories]

                # category embedding layer
                visual_attentions = [embedding_layer_category_alphas[j][i][: len(words)].detach().cpu().numpy()
                                     for j in range(self.category_num)]
                titles = ['ce-true: %s - pred: %s' % (str(label[i][j].detach().cpu().numpy()),
                                                   str(pred_category[j][i].detach().cpu().numpy()))
                          for j in range(self.category_num)]
                attention_visualizer.plot_multi_attentions_of_sentence(words, visual_attentions, attention_labels,
                                                                       titles)

                # category lstm layer
                visual_attentions = [lstm_layer_category_alphas[j][i][: len(words)].detach().cpu().numpy()
                                     for j in range(self.category_num)]
                titles = ['cl-true: %s - pred: %s' % (str(label[i][j].detach().cpu().numpy()),
                                                   str(pred_category[j][i].detach().cpu().numpy()))
                          for j in range(self.category_num)]
                attention_visualizer.plot_multi_attentions_of_sentence(words, visual_attentions, attention_labels,
                                                                       titles)

                # sentiment embedding layer
                visual_attentions = [embedding_layer_sentiment_alphas[j][i][: len(words)].detach().cpu().numpy()
                                     for j in range(self.category_num)]
                titles = ['se-true: %s - pred: %s - %s' % (str(label[i][j + self.category_num].detach().cpu().numpy()),
                                                        str(pred_sentiment[j][i].detach().cpu().numpy()),
                                                        str(self.polarites))
                          for j in range(self.category_num)]
                attention_visualizer.plot_multi_attentions_of_sentence(words, visual_attentions, attention_labels,
                                                                       titles)

                # sentiment lstm layer
                visual_attentions = [lstm_layer_sentiment_alphas[j][i][: len(words)].detach().cpu().numpy()
                                     for j in range(self.category_num)]
                titles = ['sl-true: %s - pred: %s - %s' % (str(label[i][j + self.category_num].detach().cpu().numpy()),
                                                        str(pred_sentiment[j][i].detach().cpu().numpy()),
                                                        str(self.polarites))
                          for j in range(self.category_num)]
                attention_visualizer.plot_multi_attentions_of_sentence(words, visual_attentions, attention_labels,
                                                                       titles)

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {
            'accuracy': self._accuracy.get_metric(reset),
            'category_f1': self._f1.get_metric(reset)['fscore']
        }
        return metrics


class CaeAttOnlyInLstm(TextInAllAspectSentimentOutModel):
    def __init__(self, word_embedder: TextFieldEmbedder, position_embedder: TextFieldEmbedder,
                 aspect_embedder: TextFieldEmbedder,
                 categories: list, polarities: list, vocab: Vocabulary, configuration: dict,
                 category_loss_weight=1, sentiment_loss_weight=1):
        super().__init__(vocab, category_loss_weight=category_loss_weight, sentiment_loss_weight=sentiment_loss_weight)
        self.configuration = configuration
        self.word_embedder = word_embedder
        self.position_embedder = position_embedder
        self.categories = categories
        self.polarites = polarities
        self.category_num = len(categories)
        self.polarity_num = len(polarities)
        self.category_loss = nn.BCEWithLogitsLoss()
        self.sentiment_loss = nn.CrossEntropyLoss()
        self._accuracy = metrics.CategoricalAccuracy()
        self._f1 = allennlp_metrics.BinaryF1(0.5)

        self.lstm_layer_aspect_attentions = [AttentionInHtt(word_embedder.get_output_dim(),
                                                                 word_embedder.get_output_dim())
                                                  for _ in range(self.category_num)]
        self.lstm_layer_aspect_attentions = nn.ModuleList(self.lstm_layer_aspect_attentions)
        self.lstm = torch.nn.LSTM(word_embedder.get_output_dim(), int(word_embedder.get_output_dim() / 2), batch_first=True,
                                  bidirectional=True)
        self.dropout_after_embedding = nn.Dropout(0.5)
        self.dropout_after_lstm = nn.Dropout(0.5)
        self.category_fcs = [nn.Sequential(nn.Linear(word_embedder.get_output_dim(), 32),
                                          nn.ReLU(),
                                          nn.Linear(32, 1)) for _ in range(self.category_num)]
        self.category_fcs = nn.ModuleList(self.category_fcs)
        self.sentiment_fc = nn.Sequential(nn.Linear(word_embedder.get_output_dim(), 32),
                                          nn.ReLU(),
                                          nn.Linear(32, self.polarity_num))

    def forward(self, tokens: Dict[str, torch.Tensor], label: torch.Tensor, position: torch.Tensor,
                polarity_mask: torch.Tensor, sample: list, aspects: torch.Tensor) -> torch.Tensor:
        mask = get_text_field_mask(tokens)
        word_embeddings = self.word_embedder(tokens)
        word_embeddings = self.dropout_after_embedding(word_embeddings)

        lstm_result, _ = self.lstm(word_embeddings)
        lstm_result = self.dropout_after_lstm(lstm_result)
        lstm_layer_category_outputs = []
        lstm_layer_category_alphas = []
        lstm_layer_sentiment_outputs = []
        lstm_layer_sentiment_alphas = []
        for i in range(self.category_num):
            lstm_layer_aspect_attention = self.lstm_layer_aspect_attentions[i]
            alpha = lstm_layer_aspect_attention(lstm_result, mask)
            lstm_layer_category_alphas.append(alpha)
            category_output = self.element_wise_mul(lstm_result, alpha, return_not_sum_result=False)
            lstm_layer_category_outputs.append(category_output)

            # sentiment
            category_output = category_output.unsqueeze(1)
            sentiment_alpha = torch.matmul(category_output, lstm_result.transpose(1, 2))
            sentiment_alpha = sentiment_alpha.squeeze(1)
            sentiment_alpha = allennlp_util.masked_softmax(sentiment_alpha, mask)
            lstm_layer_sentiment_alphas.append(sentiment_alpha)
            sentiment_alpha = sentiment_alpha.unsqueeze(1)
            sentiment_output = torch.matmul(sentiment_alpha, lstm_result).squeeze(1)  # batch_size x 2*hidden_dim
            lstm_layer_sentiment_outputs.append(sentiment_output)

        final_category_outputs = []
        final_sentiment_outputs = []
        for i in range(self.category_num):
            fc = self.category_fcs[i]
            category_output = lstm_layer_category_outputs[i]
            final_category_output = fc(category_output)
            final_category_outputs.append(final_category_output)

            sentiment_output = lstm_layer_sentiment_outputs[i]
            final_sentiment_output = self.sentiment_fc(sentiment_output)
            final_sentiment_outputs.append(final_sentiment_output)

        output = {}
        if label is not None:
            category_labels = []
            polarity_labels = []
            polarity_masks = []
            for i in range(self.category_num):
                category_labels.append(label[:, i])
                polarity_labels.append(label[:, i + self.category_num])
                polarity_masks.append(polarity_mask[:, i])
            loss = 0
            for i in range(self.category_num):
                category_temp_loss = self.category_loss(final_category_outputs[i].squeeze(dim=-1), category_labels[i])
                sentiment_temp_loss = self.sentiment_loss(final_sentiment_outputs[i], polarity_labels[i].long())
                loss += category_temp_loss
                loss += sentiment_temp_loss

            # sentiment accuracy
            sentiment_logit = torch.cat(final_sentiment_outputs)
            sentiment_label = torch.cat(polarity_labels)
            sentiment_mask = torch.cat(polarity_masks)
            self._accuracy(sentiment_logit, sentiment_label, sentiment_mask)

            # category f1
            final_category_outputs_prob = [torch.sigmoid(e) for e in final_category_outputs]
            category_prob = torch.cat(final_category_outputs_prob).squeeze()
            category_label = torch.cat(category_labels)
            self._f1(category_prob, category_label)

            output['loss'] = loss

        # visualize attention
        pred_category = [torch.sigmoid(e) for e in final_category_outputs]
        pred_sentiment = [torch.nn.functional.softmax(e, dim=-1) for e in final_sentiment_outputs]
        output['pred_category'] = pred_category
        output['pred_sentiment'] = pred_sentiment
        if self.configuration['visualize_attention']:
            for i in range(len(sample)):
                words = sample[i][2]
                attention_labels = [e.split('/')[0] for e in self.categories]

                # category lstm layer
                visual_attentions = [lstm_layer_category_alphas[j][i][: len(words)].detach().cpu().numpy()
                                     for j in range(self.category_num)]
                titles = ['cl-true: %s - pred: %s' % (str(label[i][j].detach().cpu().numpy()),
                                                   str(pred_category[j][i].detach().cpu().numpy()))
                          for j in range(self.category_num)]
                attention_visualizer.plot_multi_attentions_of_sentence(words, visual_attentions, attention_labels,
                                                                       titles)

                # sentiment lstm layer
                visual_attentions = [lstm_layer_sentiment_alphas[j][i][: len(words)].detach().cpu().numpy()
                                     for j in range(self.category_num)]
                titles = ['sl-true: %s - pred: %s - %s' % (str(label[i][j + self.category_num].detach().cpu().numpy()),
                                                        str(pred_sentiment[j][i].detach().cpu().numpy()),
                                                        str(self.polarites))
                          for j in range(self.category_num)]
                attention_visualizer.plot_multi_attentions_of_sentence(words, visual_attentions, attention_labels,
                                                                       titles)

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {
            'accuracy': self._accuracy.get_metric(reset),
            'category_f1': self._f1.get_metric(reset)['fscore']
        }
        return metrics


class CaeSupportingPipeline(TextInAllAspectSentimentOutModel):
    def __init__(self, word_embedder: TextFieldEmbedder, position_embedder: TextFieldEmbedder,
                 aspect_embedder: TextFieldEmbedder,
                 categories: list, polarities: list, vocab: Vocabulary, configuration: dict,
                 category_loss_weight=1, sentiment_loss_weight=1):
        super().__init__(vocab, category_loss_weight=category_loss_weight, sentiment_loss_weight=sentiment_loss_weight)
        self.configuration = configuration
        self.word_embedder = word_embedder
        self.position_embedder = position_embedder
        self.categories = categories
        self.polarites = polarities
        self.category_num = len(categories)
        self.polarity_num = len(polarities)
        self.category_loss = nn.BCEWithLogitsLoss()
        self.sentiment_loss = nn.CrossEntropyLoss()
        self._accuracy = metrics.CategoricalAccuracy()
        self._f1 = allennlp_metrics.BinaryF1(0.5)

        self.lstm_layer_aspect_attentions = [AttentionInHtt(word_embedder.get_output_dim(),
                                                                 word_embedder.get_output_dim())
                                                  for _ in range(self.category_num)]
        self.lstm_layer_aspect_attentions = nn.ModuleList(self.lstm_layer_aspect_attentions)
        self.aspect_lstm = torch.nn.LSTM(word_embedder.get_output_dim(), int(word_embedder.get_output_dim() / 2), batch_first=True,
                                  bidirectional=True)
        self.lstm = torch.nn.LSTM(word_embedder.get_output_dim(), int(word_embedder.get_output_dim() / 2),
                                  batch_first=True,
                                  bidirectional=True)
        self.dropout_after_embedding = nn.Dropout(0.5)
        self.dropout_after_lstm = nn.Dropout(0.5)
        self.category_fcs = [nn.Sequential(nn.Linear(word_embedder.get_output_dim(), 32),
                                          nn.ReLU(),
                                          nn.Linear(32, 1)) for _ in range(self.category_num)]
        self.category_fcs = nn.ModuleList(self.category_fcs)
        self.sentiment_fc = nn.Sequential(nn.Linear(word_embedder.get_output_dim(), 32),
                                          nn.ReLU(),
                                          nn.Linear(32, self.polarity_num))

    def no_grad_for_acd_parameter(self):
        acd_layers = []
        acd_layers.append(self.aspect_lstm)
        acd_layers.append(self.lstm_layer_aspect_attentions)
        acd_layers.append(self.category_fcs)
        for layer in acd_layers:
            for name, value in layer.named_parameters():
                value.requires_grad = False

    def forward(self, tokens: Dict[str, torch.Tensor], label: torch.Tensor, position: torch.Tensor,
                polarity_mask: torch.Tensor, sample: list, aspects: torch.Tensor) -> torch.Tensor:
        mask = get_text_field_mask(tokens)
        word_embeddings = self.word_embedder(tokens)
        word_embeddings = self.dropout_after_embedding(word_embeddings)

        aspect_lstm_result, _ = self.aspect_lstm(word_embeddings)
        aspect_lstm_result = self.dropout_after_lstm(aspect_lstm_result)

        lstm_result, _ = self.lstm(word_embeddings)
        lstm_result = self.dropout_after_lstm(lstm_result)

        lstm_layer_category_outputs = []
        lstm_layer_category_alphas = []
        lstm_layer_sentiment_outputs = []
        lstm_layer_sentiment_alphas = []
        for i in range(self.category_num):
            lstm_layer_aspect_attention = self.lstm_layer_aspect_attentions[i]
            alpha = lstm_layer_aspect_attention(aspect_lstm_result, mask)
            lstm_layer_category_alphas.append(alpha)
            category_output = self.element_wise_mul(aspect_lstm_result, alpha, return_not_sum_result=False)
            lstm_layer_category_outputs.append(category_output)

            # sentiment
            category_output = category_output.unsqueeze(1)
            sentiment_alpha = torch.matmul(category_output, lstm_result.transpose(1, 2))
            sentiment_alpha = sentiment_alpha.squeeze(1)
            sentiment_alpha = allennlp_util.masked_softmax(sentiment_alpha, mask)
            lstm_layer_sentiment_alphas.append(sentiment_alpha)
            sentiment_alpha = sentiment_alpha.unsqueeze(1)
            sentiment_output = torch.matmul(sentiment_alpha, lstm_result).squeeze(1)  # batch_size x 2*hidden_dim
            lstm_layer_sentiment_outputs.append(sentiment_output)

        final_category_outputs = []
        final_sentiment_outputs = []
        for i in range(self.category_num):
            fc = self.category_fcs[i]
            category_output = lstm_layer_category_outputs[i]
            final_category_output = fc(category_output)
            final_category_outputs.append(final_category_output)

            sentiment_output = lstm_layer_sentiment_outputs[i]
            final_sentiment_output = self.sentiment_fc(sentiment_output)
            final_sentiment_outputs.append(final_sentiment_output)

        output = {}
        if label is not None:
            category_labels = []
            polarity_labels = []
            polarity_masks = []
            for i in range(self.category_num):
                category_labels.append(label[:, i])
                polarity_labels.append(label[:, i + self.category_num])
                polarity_masks.append(polarity_mask[:, i])
            loss = 0
            total_category_loss = 0
            total_sentiment_loss = 0
            for i in range(self.category_num):
                category_temp_loss = self.category_loss(final_category_outputs[i].squeeze(dim=-1), category_labels[i])
                sentiment_temp_loss = self.sentiment_loss(final_sentiment_outputs[i], polarity_labels[i].long())
                total_category_loss += category_temp_loss
                total_sentiment_loss += sentiment_temp_loss
                loss = self.category_loss_weight * total_category_loss + \
                       self.sentiment_loss_weight * total_sentiment_loss

            # sentiment accuracy
            sentiment_logit = torch.cat(final_sentiment_outputs)
            sentiment_label = torch.cat(polarity_labels)
            sentiment_mask = torch.cat(polarity_masks)
            self._accuracy(sentiment_logit, sentiment_label, sentiment_mask)

            # category f1
            final_category_outputs_prob = [torch.sigmoid(e) for e in final_category_outputs]
            category_prob = torch.cat(final_category_outputs_prob).squeeze()
            category_label = torch.cat(category_labels)
            self._f1(category_prob, category_label)

            output['loss'] = loss

        # visualize attention
        pred_category = [torch.sigmoid(e) for e in final_category_outputs]
        pred_sentiment = [torch.nn.functional.softmax(e, dim=-1) for e in final_sentiment_outputs]
        output['pred_category'] = pred_category
        output['pred_sentiment'] = pred_sentiment
        if self.configuration['visualize_attention']:
            for i in range(len(sample)):
                words = sample[i][2]
                attention_labels = [e.split('/')[0] for e in self.categories]

                # category lstm layer
                visual_attentions = [lstm_layer_category_alphas[j][i][: len(words)].detach().cpu().numpy()
                                     for j in range(self.category_num)]
                titles = ['cl-true: %s - pred: %s' % (str(label[i][j].detach().cpu().numpy()),
                                                   str(pred_category[j][i].detach().cpu().numpy()))
                          for j in range(self.category_num)]
                attention_visualizer.plot_multi_attentions_of_sentence(words, visual_attentions, attention_labels,
                                                                       titles)

                # sentiment lstm layer
                visual_attentions = [lstm_layer_sentiment_alphas[j][i][: len(words)].detach().cpu().numpy()
                                     for j in range(self.category_num)]
                titles = ['sl-true: %s - pred: %s - %s' % (str(label[i][j + self.category_num].detach().cpu().numpy()),
                                                        str(pred_sentiment[j][i].detach().cpu().numpy()),
                                                        str(self.polarites))
                          for j in range(self.category_num)]
                attention_visualizer.plot_multi_attentions_of_sentence(words, visual_attentions, attention_labels,
                                                                       titles)

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {
            'accuracy': self._accuracy.get_metric(reset),
            'category_f1': self._f1.get_metric(reset)['fscore']
        }
        return metrics


class CaeWithoutCAE(TextInAllAspectSentimentOutModel):
    def __init__(self, word_embedder: TextFieldEmbedder, position_embedder: TextFieldEmbedder,
                 aspect_embedder: TextFieldEmbedder,
                 categories: list, polarities: list, vocab: Vocabulary, configuration: dict,
                 category_loss_weight=1, sentiment_loss_weight=1):
        super().__init__(vocab, category_loss_weight=category_loss_weight, sentiment_loss_weight=sentiment_loss_weight)
        self.configuration = configuration
        self.word_embedder = word_embedder
        self.position_embedder = position_embedder
        self.categories = categories
        self.polarites = polarities
        self.category_num = len(categories)
        self.polarity_num = len(polarities)
        self.category_loss = nn.BCEWithLogitsLoss()
        self.sentiment_loss = nn.CrossEntropyLoss()
        self._accuracy = metrics.CategoricalAccuracy()
        self._f1 = allennlp_metrics.BinaryF1(0.5)

        self.lstm_layer_aspect_attentions = [AttentionInHtt(word_embedder.get_output_dim(),
                                                                 word_embedder.get_output_dim())
                                                  for _ in range(self.category_num)]
        self.lstm_layer_aspect_attentions = nn.ModuleList(self.lstm_layer_aspect_attentions)
        self.lstm_layer_sentiment_attentions = [DotProductAttentionInHtt(word_embedder.get_output_dim(),
                                                                         word_embedder.get_output_dim())
                                             for _ in range(self.category_num)]
        self.lstm_layer_sentiment_attentions = nn.ModuleList(self.lstm_layer_sentiment_attentions)
        self.lstm = torch.nn.LSTM(word_embedder.get_output_dim(), int(word_embedder.get_output_dim() / 2), batch_first=True,
                                  bidirectional=True)
        self.dropout_after_embedding = nn.Dropout(0.5)
        self.dropout_after_lstm = nn.Dropout(0.5)
        self.category_fcs = [nn.Sequential(nn.Linear(word_embedder.get_output_dim(), 32),
                                          nn.ReLU(),
                                          nn.Linear(32, 1)) for _ in range(self.category_num)]
        self.category_fcs = nn.ModuleList(self.category_fcs)
        self.sentiment_fcs = [nn.Sequential(nn.Linear(word_embedder.get_output_dim(), 32),
                                          nn.ReLU(),
                                          nn.Linear(32, self.polarity_num)) for _ in range(self.category_num)]
        self.sentiment_fcs = nn.ModuleList(self.sentiment_fcs)

    def forward(self, tokens: Dict[str, torch.Tensor], label: torch.Tensor, position: torch.Tensor,
                polarity_mask: torch.Tensor, sample: list, aspects: torch.Tensor) -> torch.Tensor:
        mask = get_text_field_mask(tokens)
        word_embeddings = self.word_embedder(tokens)
        word_embeddings = self.dropout_after_embedding(word_embeddings)

        lstm_result, _ = self.lstm(word_embeddings)
        lstm_result = self.dropout_after_lstm(lstm_result)
        lstm_layer_category_outputs = []
        lstm_layer_category_alphas = []
        lstm_layer_sentiment_outputs = []
        lstm_layer_sentiment_alphas = []
        for i in range(self.category_num):
            lstm_layer_aspect_attention = self.lstm_layer_aspect_attentions[i]
            alpha = lstm_layer_aspect_attention(lstm_result, mask)
            lstm_layer_category_alphas.append(alpha)
            category_output = self.element_wise_mul(lstm_result, alpha, return_not_sum_result=False)
            lstm_layer_category_outputs.append(category_output)

            # sentiment
            lstm_layer_sentiment_attention = self.lstm_layer_sentiment_attentions[i]
            sentiment_alpha = lstm_layer_sentiment_attention(lstm_result, mask)
            lstm_layer_sentiment_alphas.append(sentiment_alpha)
            sentiment_output = self.element_wise_mul(lstm_result, sentiment_alpha, return_not_sum_result=False)
            lstm_layer_sentiment_outputs.append(sentiment_output)

        final_category_outputs = []
        final_sentiment_outputs = []
        for i in range(self.category_num):
            fc = self.category_fcs[i]
            category_output = lstm_layer_category_outputs[i]
            final_category_output = fc(category_output)
            final_category_outputs.append(final_category_output)

            sentiment_output = lstm_layer_sentiment_outputs[i]
            final_sentiment_output = self.sentiment_fcs[i](sentiment_output)
            final_sentiment_outputs.append(final_sentiment_output)

        output = {}
        if label is not None:
            category_labels = []
            polarity_labels = []
            polarity_masks = []
            for i in range(self.category_num):
                category_labels.append(label[:, i])
                polarity_labels.append(label[:, i + self.category_num])
                polarity_masks.append(polarity_mask[:, i])
            loss = 0
            for i in range(self.category_num):
                category_temp_loss = self.category_loss(final_category_outputs[i].squeeze(dim=-1), category_labels[i])
                sentiment_temp_loss = self.sentiment_loss(final_sentiment_outputs[i], polarity_labels[i].long())
                loss += category_temp_loss
                loss += sentiment_temp_loss

            # sentiment accuracy
            sentiment_logit = torch.cat(final_sentiment_outputs)
            sentiment_label = torch.cat(polarity_labels)
            sentiment_mask = torch.cat(polarity_masks)
            self._accuracy(sentiment_logit, sentiment_label, sentiment_mask)

            # category f1
            final_category_outputs_prob = [torch.sigmoid(e) for e in final_category_outputs]
            category_prob = torch.cat(final_category_outputs_prob).squeeze()
            category_label = torch.cat(category_labels)
            self._f1(category_prob, category_label)

            output['loss'] = loss

        # visualize attention
        pred_category = [torch.sigmoid(e) for e in final_category_outputs]
        pred_sentiment = [torch.nn.functional.softmax(e, dim=-1) for e in final_sentiment_outputs]
        output['pred_category'] = pred_category
        output['pred_sentiment'] = pred_sentiment
        if self.configuration['visualize_attention']:
            for i in range(len(sample)):
                words = sample[i][2]
                attention_labels = [e.split('/')[0] for e in self.categories]

                # category lstm layer
                visual_attentions = [lstm_layer_category_alphas[j][i][: len(words)].detach().cpu().numpy()
                                     for j in range(self.category_num)]
                titles = ['cl-true: %s - pred: %s' % (str(label[i][j].detach().cpu().numpy()),
                                                   str(pred_category[j][i].detach().cpu().numpy()))
                          for j in range(self.category_num)]
                attention_visualizer.plot_multi_attentions_of_sentence(words, visual_attentions, attention_labels,
                                                                       titles)

                # sentiment lstm layer
                visual_attentions = [lstm_layer_sentiment_alphas[j][i][: len(words)].detach().cpu().numpy()
                                     for j in range(self.category_num)]
                titles = ['sl-true: %s - pred: %s - %s' % (str(label[i][j + self.category_num].detach().cpu().numpy()),
                                                        str(pred_sentiment[j][i].detach().cpu().numpy()),
                                                        str(self.polarites))
                          for j in range(self.category_num)]
                attention_visualizer.plot_multi_attentions_of_sentence(words, visual_attentions, attention_labels,
                                                                       titles)

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {
            'accuracy': self._accuracy.get_metric(reset),
            'category_f1': self._f1.get_metric(reset)['fscore']
        }
        return metrics


class AtaeLstmCae(TextInAllAspectSentimentOutModel):
    def __init__(self, word_embedder: TextFieldEmbedder, position_embedder: TextFieldEmbedder,
                 aspect_embedder: TextFieldEmbedder,
                 categories: list, polarities: list, vocab: Vocabulary, configuration: dict,
                 category_loss_weight=1, sentiment_loss_weight=1):
        super().__init__(vocab, category_loss_weight=category_loss_weight, sentiment_loss_weight=sentiment_loss_weight)
        self.configuration = configuration
        self.word_embedder = word_embedder
        self.position_embedder = position_embedder
        self.aspect_embedder = aspect_embedder
        self.categories = categories
        self.polarites = polarities
        self.category_num = len(categories)
        self.polarity_num = len(polarities)
        self.category_loss = nn.BCEWithLogitsLoss()
        self.sentiment_loss = nn.CrossEntropyLoss()
        self._accuracy = metrics.CategoricalAccuracy()
        self._f1 = allennlp_metrics.BinaryF1(0.5)

        word_embedding_dim = word_embedder.get_output_dim()
        if self.configuration['lstm_or_fc_after_embedding_layer'] == 'fc':
            self.embedding_layer_fc = nn.Linear(word_embedding_dim, word_embedding_dim, bias=True)
        else:
            self.embedding_layer_lstm = torch.nn.LSTM(word_embedding_dim, word_embedding_dim, batch_first=True,
                                                      bidirectional=False, num_layers=1)

        aspect_word_embedding_dim = aspect_embedder.get_output_dim()
        if self.configuration['model_name'] in ['ae-lstm-cae', 'atae-lstm-cae']:
            lstm_input_size = word_embedding_dim + aspect_word_embedding_dim
            if self.configuration['merge_ae']:
                lstm_input_size += aspect_word_embedding_dim
        else:
            lstm_input_size = word_embedding_dim
        num_layers = 1
        hidden_size = 300
        self.lstm = torch.nn.LSTM(lstm_input_size, hidden_size, batch_first=True,
                                  bidirectional=False, num_layers=num_layers)
        if self.configuration['model_name'] in ['at-lstm-cae', 'atae-lstm-cae']:
            # self.category_attentions = [AttentionInHtt(word_embedding_dim, aspect_word_embedding_dim)
            #                                               for _ in range(self.category_num)]
            # self.category_attentions = nn.ModuleList(self.category_attentions)
            sentiment_attention_size = word_embedding_dim + aspect_word_embedding_dim
            if self.configuration['merge_ae']:
                sentiment_attention_size += aspect_word_embedding_dim
            sentiment_fc_input_size = 2 * hidden_size
            if self.configuration['share_sentiment_classifier']:
                self.sentiment_attention = AttentionInHtt(sentiment_attention_size, sentiment_attention_size)
                self.sentiment_fc = nn.Sequential(nn.Linear(sentiment_fc_input_size, self.polarity_num))
            else:
                self.sentiment_attentions = [AttentionInHtt(sentiment_attention_size, sentiment_attention_size) for _ in range(self.category_num)]
                self.sentiment_attentions = nn.ModuleList(self.sentiment_attentions)
                self.sentiment_fcs = [nn.Sequential(nn.Linear(sentiment_fc_input_size, self.polarity_num)) for _ in range(self.category_num)]
                self.sentiment_fcs = nn.ModuleList(self.sentiment_fcs)
        else:
            self.sentiment_attention = None
            if self.configuration['share_sentiment_classifier']:
                self.sentiment_fc = nn.Sequential(nn.Linear(hidden_size, self.polarity_num))
            else:
                self.sentiment_fcs = [nn.Sequential(nn.Linear(hidden_size, self.polarity_num)) for _ in range(self.category_num)]
                self.sentiment_fcs = nn.ModuleList(self.sentiment_fcs)

        self.category_attentions_of_embedding_layer = [AttentionInHtt(word_embedding_dim, word_embedding_dim)
                                                       for _ in range(self.category_num)]
        self.category_attentions_of_embedding_layer = nn.ModuleList(self.category_attentions_of_embedding_layer)

        if self.configuration['model_name'] == 'atae-lstm-cae':
            self.category_fcs = [nn.Linear(word_embedding_dim, 1) for _ in range(self.category_num)]
        elif self.configuration['model_name'] == 'at-lstm-cae':
            self.category_fcs = [nn.Linear(word_embedding_dim, 1) for _ in range(self.category_num)]
        else:
            self.category_fcs = [nn.Linear(word_embedding_dim, 1) for _ in range(self.category_num)]
        self.category_fcs = nn.ModuleList(self.category_fcs)

        self.dropout_after_embedding_layer = nn.Dropout(0.5)
        self.dropout_after_lstm_layer = nn.Dropout(0.5)
        self.dropout_after_before_sentiment_fc = nn.Dropout(0.5)

    def no_grad_for_acd_parameter(self):
        acd_layers = []
        if self.configuration['lstm_or_fc_after_embedding_layer'] == 'fc':
            acd_layers.append(self.embedding_layer_fc)
        else:
            acd_layers.append(self.embedding_layer_lstm)
        acd_layers.append(self.category_attentions_of_embedding_layer)
        acd_layers.append(self.category_fcs)
        for layer in acd_layers:
            for name, value in layer.named_parameters():
                value.requires_grad = False

    def forward(self, tokens: Dict[str, torch.Tensor], label: torch.Tensor, position: torch.Tensor,
                polarity_mask: torch.Tensor, sample: list, aspects: torch.Tensor) -> torch.Tensor:
        mask = get_text_field_mask(tokens)
        word_embeddings = self.word_embedder(tokens)
        if self.configuration['lstm_or_fc_after_embedding_layer'] == 'fc':
            embedding_layer_fc = self.embedding_layer_fc(word_embeddings)
        else:
            embedding_layer_fc, (_, _) = self.embedding_layer_lstm(word_embeddings)

        word_embeddings = self.dropout_after_embedding_layer(word_embeddings)

        aspects_separate = [{'aspect': aspects['aspect'][:, i].unsqueeze(1)} for i in range(self.category_num)]
        aspect_embeddings_singles = [self.aspect_embedder(aspects_separate[i]).squeeze(1) for i in
                                     range(self.category_num)]
        aspects_seprate_repeat = [{'aspect': aspects_separate[i]['aspect'].expand_as(tokens['tokens'])} for i in
                                  range(self.category_num)]
        aspect_embeddings_separate = [self.aspect_embedder(aspects_seprate_repeat[i]) for i in range(self.category_num)]

        lstm_inputs = []
        embedding_layer_category_outputs = []
        embedding_layer_category_alphas = []
        for i in range(self.category_num):
            category_attention_of_embedding_layer = self.category_attentions_of_embedding_layer[i]
            category_alpha = category_attention_of_embedding_layer(embedding_layer_fc, mask)
            embedding_layer_category_alphas.append(category_alpha)

        for i in range(self.category_num):
            category_alpha = embedding_layer_category_alphas[i]
            category_output = self.element_wise_mul(embedding_layer_fc, category_alpha, return_not_sum_result=False)
            embedding_layer_category_outputs.append(category_output)

        embedding_layer_sentiment_alphas = []
        for i in range(self.category_num):
            if self.configuration['model_name'] in ['ae-lstm-cae', 'atae-lstm-cae']:
                aspect_embeddings = aspect_embeddings_separate[i]

                category_alpha = embedding_layer_category_alphas[i]
                caee, caee_not_sum = self.element_wise_mul(word_embeddings, category_alpha, return_not_sum_result=True)
                caee_u = caee.unsqueeze(1)
                caee_r = caee_u.repeat(1, aspect_embeddings.size(1), 1)
                if self.configuration['not_sum']:
                    aspect_embeddings_temp = caee_not_sum
                else:
                    aspect_embeddings_temp = caee_r

                if self.configuration['merge_ae']:
                    aspect_embeddings = torch.cat([aspect_embeddings_temp, aspect_embeddings], dim=-1)
                else:
                    aspect_embeddings = aspect_embeddings_temp

                lstm_input = torch.cat([aspect_embeddings, word_embeddings], dim=-1)
            else:
                lstm_input = word_embeddings
            lstm_inputs.append(lstm_input)

        lstm_layer_category_alphas = []
        lstm_layer_category_outputs = []
        final_sentiment_outputs = []
        final_sentiment_output_probs = []
        lstm_layer_sentiment_alphas = []
        for i in range(self.category_num):
            lstm_input = lstm_inputs[i]
            lstm_output, (lstm_hn, lstm_cn) = self.lstm(lstm_input)
            lstm_output = self.dropout_after_lstm_layer(lstm_output)
            lstm_hn = lstm_hn.squeeze(dim=0)

            if self.configuration['model_name'] in ['at-lstm-cae', 'atae-lstm-cae']:
                aspect_embeddings = aspect_embeddings_separate[i]
                # 生成的attention变化太大，导致模型的输入变化太大，使得需要更大的模型才能学到这些
                # category_attention = self.category_attentions[i]
                # category_alpha = category_attention(lstm_output, mask)
                category_alpha = embedding_layer_category_alphas[i]
                lstm_layer_category_alphas.append(category_alpha)
                caee, caee_not_sum = self.element_wise_mul(lstm_output, category_alpha, return_not_sum_result=True)
                lstm_layer_category_outputs.append(caee)
                caee_u = caee.unsqueeze(1)
                caee_r = caee_u.repeat(1, aspect_embeddings.size(1), 1)
                if self.configuration['not_sum']:
                    aspect_embeddings_temp = caee_not_sum
                else:
                    aspect_embeddings_temp = caee_r
                if self.configuration['merge_ae']:
                    aspect_embeddings = torch.cat([aspect_embeddings_temp, aspect_embeddings], dim=-1)
                else:
                    aspect_embeddings = aspect_embeddings_temp

                input_for_attention = torch.cat([aspect_embeddings, lstm_output], dim=-1)
                if self.configuration['share_sentiment_classifier']:
                    alpha = self.sentiment_attention(input_for_attention, mask)
                else:
                    alpha = self.sentiment_attentions[i](input_for_attention, mask)
                lstm_layer_sentiment_alphas.append(alpha)
                sentiment_output = self.element_wise_mul(lstm_output, alpha, return_not_sum_result=False)
                sentiment_output = torch.cat([sentiment_output, lstm_hn], dim=-1)
            else:
                sentiment_output = lstm_hn
            sentiment_output = self.dropout_after_before_sentiment_fc(sentiment_output)
            if self.configuration['share_sentiment_classifier']:
                final_sentiment_output = self.sentiment_fc(sentiment_output)
            else:
                final_sentiment_output = self.sentiment_fcs[i](sentiment_output)
            final_sentiment_outputs.append(final_sentiment_output)
            final_sentiment_output_prob = torch.softmax(final_sentiment_output, dim=-1)
            final_sentiment_output_probs.append(final_sentiment_output_prob)

        final_category_outputs = []
        for i in range(self.category_num):
            category_output1 = embedding_layer_category_outputs[i]
            final_category_output = self.category_fcs[i](category_output1)
            final_category_outputs.append(final_category_output)

        output = {}
        output['alpha'] = embedding_layer_category_alphas
        if label is not None:
            category_labels = []
            polarity_labels = []
            polarity_masks = []
            for i in range(self.category_num):
                category_labels.append(label[:, i])
                polarity_labels.append(label[:, i + self.category_num])
                polarity_masks.append(polarity_mask[:, i])
            loss = 0
            total_category_loss = 0
            total_sentiment_loss = 0
            for i in range(self.category_num):
                category_temp_loss = self.category_loss(final_category_outputs[i].squeeze(dim=-1), category_labels[i])
                sentiment_temp_loss = self.sentiment_loss(final_sentiment_outputs[i], polarity_labels[i].long())
                total_category_loss += category_temp_loss
                if not self.configuration['only_acd']:
                    total_sentiment_loss += sentiment_temp_loss
            loss = self.category_loss_weight * total_category_loss + self.sentiment_loss_weight * total_sentiment_loss

            # sentiment accuracy
            sentiment_logit = torch.cat(final_sentiment_outputs)
            sentiment_label = torch.cat(polarity_labels)
            sentiment_mask = torch.cat(polarity_masks)
            self._accuracy(sentiment_logit, sentiment_label, sentiment_mask)

            # category f1
            final_category_outputs_prob = [torch.sigmoid(e) for e in final_category_outputs]
            category_prob = torch.cat(final_category_outputs_prob).squeeze()
            category_label = torch.cat(category_labels)
            self._f1(category_prob, category_label)

            output['loss'] = loss

        # visualize attention
        pred_category = [torch.sigmoid(e) for e in final_category_outputs]
        pred_sentiment = [torch.nn.functional.softmax(e, dim=-1) for e in final_sentiment_outputs]
        output['pred_category'] = pred_category
        output['pred_sentiment'] = pred_sentiment
        if self.configuration['visualize_attention']:
            for i in range(len(sample)):
                words = sample[i][2]
                if not ('food' in words and 'dreadful' in words):
                    continue
                attention_labels = [e.split('/')[0] for e in self.categories]

                # category
                visual_attentions = [embedding_layer_category_alphas[j][i][: len(words)].detach().cpu().numpy()
                                     for j in range(self.category_num)]
                titles = [' category true: %s - pred: %s' % (str(label[i][j].detach().cpu().numpy()),
                                                   str(pred_category[j][i].detach().cpu().numpy()))
                          for j in range(self.category_num)]
                attention_visualizer.plot_multi_attentions_of_sentence_backup(words, visual_attentions, attention_labels,
                                                                       titles)

                # visual_attentions = [lstm_layer_category_alphas[j][i][: len(words)].detach().cpu().numpy()
                #                      for j in range(self.category_num)]
                # titles = ['true: %s - pred: %s' % (str(label[i][j].detach().cpu().numpy()),
                #                                    str(pred_category[j][i].detach().cpu().numpy()))
                #           for j in range(self.category_num)]
                # attention_visualizer.plot_multi_attentions_of_sentence(words, visual_attentions, attention_labels,
                #                                                        titles)

                # sentiment embedding layer
                # visual_attentions = [embedding_layer_sentiment_alphas[j][i][: len(words)].detach().cpu().numpy()
                #                      for j in range(self.category_num)]
                # titles = ['true: %s - pred: %s - %s' % (str(label[i + self.category_num][j].detach().cpu().numpy()),
                #                                         str(pred_sentiment[j][i].detach().cpu().numpy()),
                #                                         str(self.polarites))
                #           for j in range(self.category_num)]
                # attention_visualizer.plot_multi_attentions_of_sentence(words, visual_attentions, attention_labels,
                #                                                        titles)

                # sentiment lstm layer
                visual_attentions = [lstm_layer_sentiment_alphas[j][i][: len(words)].detach().cpu().numpy()
                                     for j in range(self.category_num)]
                titles = ['sentiment true: %s - pred: %s - %s' % (str(label[i + self.category_num][j].detach().cpu().numpy()),
                                                        str(pred_sentiment[j][i].detach().cpu().numpy()),
                                                        str(self.polarites))
                          for j in range(self.category_num)]
                attention_visualizer.plot_multi_attentions_of_sentence_backup(words, visual_attentions, attention_labels,
                                                                       titles)

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {
            'accuracy': self._accuracy.get_metric(reset),
            'category_f1': self._f1.get_metric(reset)['fscore']
        }
        return metrics


from nlp_tasks.absa.aspect_category_detection_and_sentiment_classification.mams.aspect_category_model.recurrent_capsnet \
    import RecurrentCapsuleNetwork
from nlp_tasks.absa.aspect_category_detection_and_sentiment_classification.mams.module.utils.loss import CapsuleLoss


class CapsNetCae(TextInAllAspectSentimentOutModel):
    def __init__(self, word_embedder: TextFieldEmbedder, position_embedder: TextFieldEmbedder,
                 aspect_embedder: TextFieldEmbedder,
                 categories: list, polarities: list, vocab: Vocabulary, configuration: dict,
                 category_loss_weight=1, sentiment_loss_weight=1):
        super().__init__(vocab, category_loss_weight=category_loss_weight, sentiment_loss_weight=sentiment_loss_weight)
        self.configuration = configuration
        self.word_embedder = word_embedder
        self.position_embedder = position_embedder
        self.aspect_embedder = aspect_embedder
        self.categories = categories
        self.polarites = polarities
        self.category_num = len(categories)
        self.polarity_num = len(polarities)
        self.category_loss = nn.BCEWithLogitsLoss()
        # self.sentiment_loss = nn.CrossEntropyLoss()
        self._accuracy = metrics.CategoricalAccuracy()
        self._f1 = allennlp_metrics.BinaryF1(0.5)

        word_embedding_dim = word_embedder.get_output_dim()
        if self.configuration['lstm_or_fc_after_embedding_layer'] == 'fc':
            self.embedding_layer_fc = nn.Linear(word_embedding_dim, word_embedding_dim, bias=True)
        else:
            self.embedding_layer_lstm = torch.nn.LSTM(word_embedding_dim, word_embedding_dim, batch_first=True,
                                                      bidirectional=False, num_layers=1)

        self.category_attentions_of_embedding_layer = [AttentionInHtt(word_embedding_dim, word_embedding_dim)
                                                       for _ in range(self.category_num)]
        self.category_attentions_of_embedding_layer = nn.ModuleList(self.category_attentions_of_embedding_layer)

        self.category_fcs = [nn.Linear(word_embedding_dim, 1) for _ in range(self.category_num)]
        self.category_fcs = nn.ModuleList(self.category_fcs)

        self.sentiment_loss = CapsuleLoss()
        self.capsnet = RecurrentCapsuleNetwork(
            embedding=self.configuration['embedding_for_capsnet'],
            aspect_embedding=self.configuration['aspect_embedding_for_capsnet'],
            num_layers=2,
            bidirectional=True,
            capsule_size=300,
            dropout=0.5,
            num_categories=self.polarity_num,
            configuration=self.configuration
        )
        self.capsnet.load_sentiment(self.configuration['sentiment_matrix'])

    def no_grad_for_acd_parameter(self):
        acd_layers = []
        if self.configuration['lstm_or_fc_after_embedding_layer'] == 'fc':
            acd_layers.append(self.embedding_layer_fc)
        else:
            acd_layers.append(self.embedding_layer_lstm)
        acd_layers.append(self.category_attentions_of_embedding_layer)
        acd_layers.append(self.category_fcs)
        for layer in acd_layers:
            for name, value in layer.named_parameters():
                value.requires_grad = False

    def forward(self, tokens: Dict[str, torch.Tensor], label: torch.Tensor, position: torch.Tensor,
                polarity_mask: torch.Tensor, sample: list, aspects: torch.Tensor) -> torch.Tensor:
        mask = get_text_field_mask(tokens)
        word_embeddings = self.word_embedder(tokens)
        if self.configuration['lstm_or_fc_after_embedding_layer'] == 'fc':
            embedding_layer_fc = self.embedding_layer_fc(word_embeddings)
        else:
            embedding_layer_fc, (_, _) = self.embedding_layer_lstm(word_embeddings)

        sentence_for_capsnet = tokens['tokens']

        aspects_separate = [{'aspect': aspects['aspect'][:, i].unsqueeze(1)} for i in range(self.category_num)]
        aspect_embeddings_singles = [self.aspect_embedder(aspects_separate[i]).squeeze(1) for i in
                                     range(self.category_num)]
        aspects_seprate_repeat = [{'aspect': aspects_separate[i]['aspect'].expand_as(tokens['tokens'])} for i in
                                  range(self.category_num)]
        aspect_embeddings_separate = [self.aspect_embedder(aspects_seprate_repeat[i]) for i in range(self.category_num)]

        embedding_layer_category_outputs = []

        embedding_layer_category_alphas = []
        for i in range(self.category_num):
            category_attention_of_embedding_layer = self.category_attentions_of_embedding_layer[i]
            category_alpha = category_attention_of_embedding_layer(embedding_layer_fc, mask)
            embedding_layer_category_alphas.append(category_alpha)

        for i in range(self.category_num):
            category_alpha = embedding_layer_category_alphas[i]
            category_output = self.element_wise_mul(embedding_layer_fc, category_alpha, return_not_sum_result=False)
            embedding_layer_category_outputs.append(category_output)

        final_sentiment_outputs = []
        final_sentiment_output_probs = []
        for i in range(self.category_num):
            aspect_for_capsnet = aspects_separate[i]['aspect'].squeeze(dim=-1)
            category_alpha = embedding_layer_category_alphas[i]
            final_sentiment_output = self.capsnet(sentence_for_capsnet, aspect_for_capsnet,
                                                  category_alpha=category_alpha)

            final_sentiment_outputs.append(final_sentiment_output)
            final_sentiment_output_prob = torch.softmax(final_sentiment_output, dim=-1)
            final_sentiment_output_probs.append(final_sentiment_output_prob)

        final_category_outputs = []
        for i in range(self.category_num):
            category_output1 = embedding_layer_category_outputs[i]
            final_category_output = self.category_fcs[i](category_output1)
            final_category_outputs.append(final_category_output)

        output = {}
        output['alpha'] = embedding_layer_category_alphas
        if label is not None:
            category_labels = []
            polarity_labels = []
            polarity_masks = []
            for i in range(self.category_num):
                category_labels.append(label[:, i])
                polarity_labels.append(label[:, i + self.category_num])
                polarity_masks.append(polarity_mask[:, i])
            loss = 0
            total_category_loss = 0
            total_sentiment_loss = 0
            for i in range(self.category_num):
                category_temp_loss = self.category_loss(final_category_outputs[i].squeeze(dim=-1), category_labels[i])
                sentiment_temp_loss = self.sentiment_loss(final_sentiment_outputs[i], polarity_labels[i].long())
                total_category_loss += category_temp_loss
                total_sentiment_loss += sentiment_temp_loss
            if self.category_loss_weight != 0:
                loss += self.category_loss_weight * total_category_loss
            if not self.configuration['only_acd'] and self.sentiment_loss_weight != 0:
                loss += self.sentiment_loss_weight * total_sentiment_loss
            # if torch.isnan(loss):
            #     raise ValueError("nan loss encountered")

            # sentiment accuracy
            sentiment_logit = torch.cat(final_sentiment_outputs)
            sentiment_label = torch.cat(polarity_labels)
            sentiment_mask = torch.cat(polarity_masks)
            self._accuracy(sentiment_logit, sentiment_label, sentiment_mask)

            # category f1
            final_category_outputs_prob = [torch.sigmoid(e) for e in final_category_outputs]
            category_prob = torch.cat(final_category_outputs_prob).squeeze()
            category_label = torch.cat(category_labels)
            self._f1(category_prob, category_label)

            output['loss'] = loss

        # visualize attention
        pred_category = [torch.sigmoid(e) for e in final_category_outputs]
        pred_sentiment = [torch.nn.functional.softmax(e, dim=-1) for e in final_sentiment_outputs]
        output['pred_category'] = pred_category
        output['pred_sentiment'] = pred_sentiment
        if self.configuration['visualize_attention']:
            for i in range(len(sample)):
                words = sample[i][2]
                attention_labels = [e.split('/')[0] for e in self.categories]

                # category
                visual_attentions = [embedding_layer_category_alphas[j][i][: len(words)].detach().cpu().numpy()
                                     for j in range(self.category_num)]
                titles = ['true: %s - pred: %s' % (str(label[i][j].detach().cpu().numpy()),
                                                   str(pred_category[j][i].detach().cpu().numpy()))
                          for j in range(self.category_num)]
                attention_visualizer.plot_multi_attentions_of_sentence(words, visual_attentions, attention_labels,
                                                                       titles)

                # visual_attentions = [lstm_layer_category_alphas[j][i][: len(words)].detach().cpu().numpy()
                #                      for j in range(self.category_num)]
                # titles = ['true: %s - pred: %s' % (str(label[i][j].detach().cpu().numpy()),
                #                                    str(pred_category[j][i].detach().cpu().numpy()))
                #           for j in range(self.category_num)]
                # attention_visualizer.plot_multi_attentions_of_sentence(words, visual_attentions, attention_labels,
                #                                                        titles)

                # sentiment embedding layer
                # visual_attentions = [embedding_layer_sentiment_alphas[j][i][: len(words)].detach().cpu().numpy()
                #                      for j in range(self.category_num)]
                # titles = ['true: %s - pred: %s - %s' % (str(label[i + self.category_num][j].detach().cpu().numpy()),
                #                                         str(pred_sentiment[j][i].detach().cpu().numpy()),
                #                                         str(self.polarites))
                #           for j in range(self.category_num)]
                # attention_visualizer.plot_multi_attentions_of_sentence(words, visual_attentions, attention_labels,
                #                                                        titles)

                # sentiment lstm layer
                # visual_attentions = [lstm_layer_sentiment_alphas[j][i][: len(words)].detach().cpu().numpy()
                #                      for j in range(self.category_num)]
                # titles = ['true: %s - pred: %s - %s' % (str(label[i + self.category_num][j].detach().cpu().numpy()),
                #                                         str(pred_sentiment[j][i].detach().cpu().numpy()),
                #                                         str(self.polarites))
                #           for j in range(self.category_num)]
                # attention_visualizer.plot_multi_attentions_of_sentence(words, visual_attentions, attention_labels,
                #                                                        titles)

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {
            'accuracy': self._accuracy.get_metric(reset),
            'category_f1': self._f1.get_metric(reset)['fscore']
        }
        return metrics


class TextInAllAspectOutModel(Model):

    def __init__(self, vocab: Vocabulary, category_loss_weight=1, sentiment_loss_weight=1):
        super().__init__(vocab)
        self.category_loss_weight = category_loss_weight
        self.sentiment_loss_weight = sentiment_loss_weight

    def matrix_mul(self, input, weight, bias=False):
        feature_list = []
        for feature in input:
            feature = torch.mm(feature, weight)
            if isinstance(bias, torch.nn.parameter.Parameter):
                feature = feature + bias.expand(feature.size()[0], bias.size()[1])
            feature = torch.tanh(feature).unsqueeze(0)
            feature_list.append(feature)

        return torch.cat(feature_list, 0).squeeze()

    def element_wise_mul(self, input1, input2, return_not_sum_result=False):
        feature_list = []
        for feature_1, feature_2 in zip(input1, input2):
            feature_2 = feature_2.unsqueeze(1)
            feature_2 = feature_2.expand_as(feature_1)
            feature = feature_1 * feature_2
            feature = feature.unsqueeze(0)
            feature_list.append(feature)
        output = torch.cat(feature_list, 0)

        result = torch.sum(output, 1)
        if return_not_sum_result:
            return result, output
        else:
            return result

    def reduce(self, nodes):
        """Take an average over all neighbor node features hu and use it to
        overwrite the original node feature."""
        m = nodes.mailbox['m']
        accum = torch.sum(m, 1)
        return {'h': accum}

    def pad_dgl_graph(self, graphs, max_node_num):
        graphs_padded = []
        for graph in graphs:
            graph_padded = copy.deepcopy(graph)
            node_num = graph.number_of_nodes()
            graph_padded.add_nodes(max_node_num - node_num)
            graphs_padded.append(graph_padded)
        return graphs_padded

    def no_grad_for_acd_parameter(self):
        pass

    def _get_model_visualization_picture_filepath(self, configuration: dict, words: list):
        savefig_dir = configuration['savefig_dir']
        if not savefig_dir:
            return None
        filename = '%s-%s.svg' % ('-'.join(words[:3]), str(time.time()))
        filename = re.sub('/', '', filename)
        return os.path.join(savefig_dir, filename)


class Acd(TextInAllAspectOutModel):
    def __init__(self, word_embedder: TextFieldEmbedder, position_embedder: TextFieldEmbedder,
                 aspect_embedder: TextFieldEmbedder,
                 categories: list, polarities: list, vocab: Vocabulary, configuration: dict):
        super().__init__(vocab)
        self.configuration = configuration
        self.word_embedder = word_embedder
        self.position_embedder = position_embedder
        self.aspect_embedder = aspect_embedder
        self.categories = categories
        self.polarites = polarities
        self.category_num = len(categories)
        self.polarity_num = len(polarities)
        self.category_loss = nn.BCEWithLogitsLoss()
        self.sentiment_loss = nn.CrossEntropyLoss()
        self._f1 = allennlp_metrics.BinaryF1(0.5)
        self._accuracy = metrics.CategoricalAccuracy()

        word_embedding_dim = word_embedder.get_output_dim()
        model_name = self.configuration['model_name']
        if model_name == 'acd-affine':
            self.embedding_layer_fc = nn.Linear(word_embedding_dim, word_embedding_dim, bias=True)
        elif model_name == 'acd-bilstm':
            self.embedding_layer_lstm = torch.nn.LSTM(word_embedding_dim, int(word_embedding_dim / 2), batch_first=True,
                                                      bidirectional=True, num_layers=1)
        else:
            self.embedding_layer_lstm = torch.nn.LSTM(word_embedding_dim, word_embedding_dim, batch_first=True,
                                                      bidirectional=False, num_layers=1)

        self.category_attentions_of_embedding_layer = [AttentionInHtt(word_embedding_dim, word_embedding_dim)
                                                       for _ in range(self.category_num)]
        self.category_attentions_of_embedding_layer = nn.ModuleList(self.category_attentions_of_embedding_layer)

        self.category_fcs = [nn.Linear(word_embedding_dim, 1) for _ in range(self.category_num)]
        self.category_fcs = nn.ModuleList(self.category_fcs)

    def forward(self, tokens: Dict[str, torch.Tensor], label: torch.Tensor, position: torch.Tensor,
                sample: list, aspects: torch.Tensor, tokens_for_graph: Dict[str, torch.Tensor]) -> torch.Tensor:
        mask = get_text_field_mask(tokens)
        word_embeddings = self.word_embedder(tokens)
        model_name = self.configuration['model_name']
        if model_name == 'acd-affine':
            word_embeddings_fc = self.embedding_layer_fc(word_embeddings)
        else:
            word_embeddings_fc, (_, _) = self.embedding_layer_lstm(word_embeddings)

        aspects_separate = [{'aspect': aspects['aspect'][:, i].unsqueeze(1)} for i in range(self.category_num)]
        aspect_embeddings_singles = [self.aspect_embedder(aspects_separate[i]).squeeze(1) for i in
                                     range(self.category_num)]
        aspects_seprate_repeat = [{'aspect': aspects_separate[i]['aspect'].expand_as(tokens['tokens'])} for i in
                                  range(self.category_num)]
        aspect_embeddings_separate = [self.aspect_embedder(aspects_seprate_repeat[i]) for i in range(self.category_num)]

        embedding_layer_category_outputs = []
        embedding_layer_category_alphas = []
        embedding_layer_attention_input = word_embeddings_fc
        for i in range(self.category_num):
            category_attention_of_embedding_layer = self.category_attentions_of_embedding_layer[i]
            category_alpha = category_attention_of_embedding_layer(embedding_layer_attention_input, mask)
            embedding_layer_category_alphas.append(category_alpha)
            category_output = self.element_wise_mul(embedding_layer_attention_input, category_alpha,
                                                    return_not_sum_result=False)
            embedding_layer_category_outputs.append(category_output)

        final_category_outputs = []
        for i in range(self.category_num):
            category_output = embedding_layer_category_outputs[i]
            final_category_output = self.category_fcs[i](category_output)
            final_category_outputs.append(final_category_output)

        output = {}
        if label is not None:
            category_labels = []
            for i in range(self.category_num):
                category_labels.append(label[:, i])
            loss = 0
            for i in range(self.category_num):
                category_temp_loss = self.category_loss(final_category_outputs[i].squeeze(dim=-1), category_labels[i])
                loss += category_temp_loss

            # Sparse Regularization Orthogonal Regularization
            # 通过mil可知，一句话的lstm表示里，临近的词有相似的表示，仅仅通过attention权重进行控制是不够的，
            # food的attention权重依旧可以去关注price关注的词的旁边的词，实际上，不同aspect不仅关注的词不仅需要不同，
            # 而且应该相距尽可能远
            if self.configuration['sparse_reg'] or self.configuration['orthogonal_reg']:
                reg_loss = 0
                for j in range(len(sample)):
                    polarity_mask_of_one_sample = label[j]
                    category_alpha_of_one_sample = [embedding_layer_category_alphas[k][j] for k in
                                                    range(self.category_num)]
                    category_alpha_of_mentioned = []
                    category_alpha_of_not_mentioned = []
                    for k in range(self.category_num):
                        if polarity_mask_of_one_sample[k] == 1:
                            category_alpha_of_mentioned.append(category_alpha_of_one_sample[k].unsqueeze(0))
                        else:
                            category_alpha_of_not_mentioned.append(category_alpha_of_one_sample[k].unsqueeze(0))
                    if len(category_alpha_of_not_mentioned) != 0:
                        category_alpha_of_not_mentioned = torch.cat(category_alpha_of_not_mentioned, dim=0)
                        category_alpha_of_not_mentioned = torch.mean(category_alpha_of_not_mentioned, dim=0,
                                                                     keepdim=True)
                        category_alpha_of_mentioned.append(category_alpha_of_not_mentioned)

                    category_eye = torch.eye(len(category_alpha_of_mentioned))
                    category_alpha_of_mentioned = torch.cat(category_alpha_of_mentioned, dim=0)
                    category_alpha_similarity = torch.mm(category_alpha_of_mentioned,
                                                         category_alpha_of_mentioned.t())

                    if self.configuration['sparse_reg'] and self.configuration['orthogonal_reg']:
                        pass
                    elif self.configuration['sparse_reg']:
                        for m in range(len(category_alpha_of_mentioned)):
                            for n in range(len(category_alpha_of_mentioned)):
                                if m != n:
                                    category_eye[m][n] = category_alpha_similarity[m][n]
                    else:
                        # orthogonal_reg
                        for m in range(len(category_alpha_of_mentioned)):
                            category_eye[m][m] = category_alpha_similarity[m][m]
                    category_reg_loss = category_alpha_similarity - category_eye
                    category_reg_loss = torch.norm(category_reg_loss)
                    reg_loss += category_reg_loss
                loss += (reg_loss * self.configuration['attention_lamda'] / len(sample))

            # category f1
            final_category_outputs_prob = [torch.sigmoid(e) for e in final_category_outputs]
            category_prob = torch.cat(final_category_outputs_prob).squeeze()
            category_label = torch.cat(category_labels)
            self._f1(category_prob, category_label)

            output['loss'] = loss

        output['alpha'] = embedding_layer_category_alphas

        # visualize attention
        pred_category = [torch.sigmoid(e) for e in final_category_outputs]
        output['pred_category'] = pred_category
        if self.configuration['visualize_attention']:
            for i in range(len(sample)):
                words = sample[i]['words']
                sentence = ' '.join(words)

                attention_labels = [e.split('/')[0] for e in self.categories]

                # category
                visual_attentions = [embedding_layer_category_alphas[j][i][: len(words)].detach().cpu().numpy()
                                     for j in range(self.category_num)]
                label_true = label[i].detach().cpu().numpy()[: self.category_num]
                # if sum(label_true) <= 1:
                #     continue
                titles = ['true: %s - pred: %s' % (str(label[i][j].detach().cpu().numpy()),
                                                   str(pred_category[j][i].detach().cpu().numpy()))
                          for j in range(self.category_num)]
                attention_visualizer.plot_multi_attentions_of_sentence_backup(words, visual_attentions, attention_labels,
                                                                              titles)

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {
            'category_f1': self._f1.get_metric(reset)['fscore']
        }
        return metrics


class AcdWithInteractiveLoss(TextInAllAspectOutModel):
    def __init__(self, word_embedder: TextFieldEmbedder, position_embedder: TextFieldEmbedder,
                 aspect_embedder: TextFieldEmbedder,
                 categories: list, polarities: list, vocab: Vocabulary, configuration: dict):
        super().__init__(vocab)
        self.configuration = configuration
        self.word_embedder = word_embedder
        self.position_embedder = position_embedder
        self.aspect_embedder = aspect_embedder
        self.categories = categories
        self.polarites = polarities
        self.category_num = len(categories)
        self.polarity_num = len(polarities)
        self.category_loss = nn.BCEWithLogitsLoss()
        self.sentiment_loss = nn.CrossEntropyLoss()
        self._f1 = allennlp_metrics.BinaryF1(0.5)
        self._accuracy = metrics.CategoricalAccuracy()

        word_embedding_dim = word_embedder.get_output_dim()
        model_name = self.configuration['model_name']
        if model_name == 'AcdWithInteractiveLoss-affine':
            self.embedding_layer_fc = nn.Linear(word_embedding_dim, word_embedding_dim, bias=True)
        elif model_name == 'AcdWithInteractiveLoss-bilstm':
            self.embedding_layer_lstm = torch.nn.LSTM(word_embedding_dim, int(word_embedding_dim / 2), batch_first=True,
                                                      bidirectional=True, num_layers=1)
        else:
            self.embedding_layer_lstm = torch.nn.LSTM(word_embedding_dim, word_embedding_dim, batch_first=True,
                                                      bidirectional=False, num_layers=1)

        self.category_attentions_of_embedding_layer = [AttentionInHtt(word_embedding_dim, word_embedding_dim)
                                                       for _ in range(self.category_num)]
        self.category_attentions_of_embedding_layer = nn.ModuleList(self.category_attentions_of_embedding_layer)

        self.category_fcs = [nn.Linear(word_embedding_dim, 1) for _ in range(self.category_num)]
        self.category_fcs = nn.ModuleList(self.category_fcs)

        # 只负责生成针对aspect category的节点表示
        aspect_graph = self.configuration['aspect_graph']
        if aspect_graph == 'attention':
            # 1. 假设：两个词被attention关注，这两个词的平均也被attention关注。todo 需要验证这个假设是否成立？
            # 2. 对于给定属性，应该关注哪些内部节点？
            # (1) 关注子节点中包含与该属性相关的节点，
            # (2) 不关注子节点中包含其它属性相关的节点
            # 3. 之前的方法为什么会关注到其它属性的情感词？
            # (1) 因为很多情感词对不同属性通用的，只是基于情感分词本身很难判断这个情感词是否是描述该属性的
            # (2) 我们的解决方案，借助于情感词的父节点的其它子节点的信息来帮助判断这个情感分词是否是描述该属性的，
            # 具体的，如果情感词的父节点的其它子节点包含其它属性的信息，那么关注这个情感词，将会受到惩罚
            self.gnn_for_aspect = DglGraphAttentionForAspectCategory(word_embedding_dim, word_embedding_dim,
                                                                     self.category_attentions_of_embedding_layer,
                                                                     configuration)
        elif aspect_graph == 'attention_with_dotted_lines':
            self.gnn_for_aspect = DglGraphAttentionForAspectCategoryWithDottedLines(word_embedding_dim,
                                                                                    word_embedding_dim,
                                                                                    self.category_attentions_of_embedding_layer,
                                                                                    configuration)
        elif aspect_graph == 'average':
            self.gnn_for_aspect = DglGraphAverageForAspectCategory(word_embedding_dim, word_embedding_dim,
                                                                   self.configuration)
        elif aspect_graph == 'gcn':
            self.gnn_for_aspect = DglGraphConvolutionForAspectCategory(word_embedding_dim, word_embedding_dim,
                                                                       self.configuration)
        elif aspect_graph == 'gat':
            self.gnn_for_aspect = GATForAspectCategory(word_embedding_dim, word_embedding_dim, opt=self.configuration)
        else:
            raise NotImplementedError(aspect_graph)

    def forward(self, tokens: Dict[str, torch.Tensor], tokens_for_graph: Dict[str, torch.Tensor], label: torch.Tensor,
                position: torch.Tensor, sample: list, aspects: torch.Tensor) -> torch.Tensor:
        mask = get_text_field_mask(tokens)

        mask_for_graph = get_text_field_mask(tokens_for_graph)

        max_len = mask.size()[1]
        max_len_for_graph = mask_for_graph.size()[1]
        padder = nn.ConstantPad2d((0, 0, 0, max_len_for_graph - max_len), 0)

        graphs = [e['graph'] for e in sample]
        graphs_padded = self.pad_dgl_graph(graphs, max_len_for_graph)

        graphs_with_dotted_line = [e['graph_with_dotted_line'] for e in sample]
        graphs_with_dotted_line_padded = self.pad_dgl_graph(graphs_with_dotted_line, max_len_for_graph)
        if not self.configuration['aspect_graph_with_dotted_line']:
            graphs_with_dotted_line_padded = graphs_padded

        word_embeddings = self.word_embedder(tokens)
        model_name = self.configuration['model_name']
        if model_name == 'AcdWithInteractiveLoss-affine':
            word_embeddings_fc = self.embedding_layer_fc(word_embeddings)
        else:
            word_embeddings_fc, (_, _) = self.embedding_layer_lstm(word_embeddings)

        aspects_separate = [{'aspect': aspects['aspect'][:, i].unsqueeze(1)} for i in range(self.category_num)]
        aspect_embeddings_singles = [self.aspect_embedder(aspects_separate[i]).squeeze(1) for i in
                                     range(self.category_num)]
        aspects_seprate_repeat = [{'aspect': aspects_separate[i]['aspect'].expand_as(tokens['tokens'])} for i in
                                  range(self.category_num)]
        aspect_embeddings_separate = [self.aspect_embedder(aspects_seprate_repeat[i]) for i in range(self.category_num)]

        # 在word_embeddings_fc上做图卷积，attention是基于图卷积的输出
        word_embeddings_fc_for_graph = padder(word_embeddings_fc)

        graph_output_for_aspect = self.gnn_for_aspect(word_embeddings_fc_for_graph,
                                                      graphs_with_dotted_line_padded)

        embedding_layer_category_outputs = []
        embedding_layer_category_alphas = []
        if self.configuration['with_constituency']:
            embedding_layer_attention_input = graph_output_for_aspect
            mask = mask_for_graph
        else:
            embedding_layer_attention_input = word_embeddings_fc
        for i in range(self.category_num):
            category_attention_of_embedding_layer = self.category_attentions_of_embedding_layer[i]
            category_alpha = category_attention_of_embedding_layer(embedding_layer_attention_input, mask)
            embedding_layer_category_alphas.append(category_alpha)
            category_output = self.element_wise_mul(embedding_layer_attention_input, category_alpha,
                                                    return_not_sum_result=False)
            embedding_layer_category_outputs.append(category_output)

        final_category_outputs = []
        final_category_outputs_auxillary = []
        for i in range(self.category_num):
            category_output = embedding_layer_category_outputs[i]
            for j in range(self.category_num):
                final_category_output = self.category_fcs[j](category_output)
                if i == j:
                    final_category_outputs.append(final_category_output)
                else:
                    final_category_outputs_auxillary.append(final_category_output)

        output = {}
        if label is not None:
            category_labels = []
            for i in range(self.category_num):
                category_labels.append(label[:, i])
            loss = 0
            for i in range(self.category_num):
                category_temp_loss = self.category_loss(final_category_outputs[i].squeeze(dim=-1), category_labels[i])
                loss += category_temp_loss

            interactive_loss = 0
            for i in range(len(final_category_outputs_auxillary)):
                label_temp = torch.zeros_like(category_labels[0])
                interactive_loss_temp = self.category_loss(final_category_outputs_auxillary[i].squeeze(dim=-1),
                                                        label_temp)
                interactive_loss += interactive_loss_temp
            loss += (interactive_loss * self.configuration['interactive_loss_lamda'] / (self.category_num - 1))

            # Sparse Regularization Orthogonal Regularization
            # 通过mil可知，一句话的lstm表示里，临近的词有相似的表示，仅仅通过attention权重进行控制是不够的，
            # food的attention权重依旧可以去关注price关注的词的旁边的词，实际上，不同aspect不仅关注的词不仅需要不同，
            # 而且应该相距尽可能远
            if self.configuration['sparse_reg'] or self.configuration['orthogonal_reg']:
                reg_loss = 0
                for j in range(len(sample)):
                    polarity_mask_of_one_sample = label[j]
                    category_alpha_of_one_sample = [embedding_layer_category_alphas[k][j] for k in
                                                    range(self.category_num)]
                    category_alpha_of_mentioned = []
                    category_alpha_of_not_mentioned = []
                    for k in range(self.category_num):
                        if polarity_mask_of_one_sample[k] == 1:
                            category_alpha_of_mentioned.append(category_alpha_of_one_sample[k].unsqueeze(0))
                        else:
                            category_alpha_of_not_mentioned.append(category_alpha_of_one_sample[k].unsqueeze(0))
                    if len(category_alpha_of_not_mentioned) != 0:
                        category_alpha_of_not_mentioned = torch.cat(category_alpha_of_not_mentioned, dim=0)
                        category_alpha_of_not_mentioned = torch.mean(category_alpha_of_not_mentioned, dim=0,
                                                                     keepdim=True)
                        category_alpha_of_mentioned.append(category_alpha_of_not_mentioned)

                    category_eye = torch.eye(len(category_alpha_of_mentioned))
                    category_alpha_of_mentioned = torch.cat(category_alpha_of_mentioned, dim=0)
                    category_alpha_similarity = torch.mm(category_alpha_of_mentioned,
                                                         category_alpha_of_mentioned.t())

                    if self.configuration['sparse_reg'] and self.configuration['orthogonal_reg']:
                        pass
                    elif self.configuration['sparse_reg']:
                        for m in range(len(category_alpha_of_mentioned)):
                            for n in range(len(category_alpha_of_mentioned)):
                                if m != n:
                                    category_eye[m][n] = category_alpha_similarity[m][n]
                    else:
                        # orthogonal_reg
                        for m in range(len(category_alpha_of_mentioned)):
                            category_eye[m][m] = category_alpha_similarity[m][m]
                    category_reg_loss = category_alpha_similarity - category_eye
                    category_reg_loss = torch.norm(category_reg_loss)
                    reg_loss += category_reg_loss
                loss += (reg_loss * self.configuration['attention_lamda'] / len(sample))

            # category f1
            final_category_outputs_prob = [torch.sigmoid(e) for e in final_category_outputs]
            category_prob = torch.cat(final_category_outputs_prob).squeeze()
            category_label = torch.cat(category_labels)
            self._f1(category_prob, category_label)

            output['loss'] = loss

        output['alpha'] = embedding_layer_category_alphas

        # visualize attention
        pred_category = [torch.sigmoid(e) for e in final_category_outputs]
        output['pred_category'] = pred_category
        if self.configuration['visualize_attention']:
            for i in range(len(sample)):
                if not self.configuration['with_constituency']:
                    words = sample[i]['words']
                else:
                    words = sample[i]['words_for_graph']
                    # if not ('0-great' in words and '6-dreadful' in words):
                    #     continue
                    print('inner_nodes:-------------------------------------------')
                    for inner_node_index, inner_node in enumerate(sample[i]['inner_nodes']):
                        print('%d-%s-%s' % (
                        len(sample[i]['words']) + inner_node_index, inner_node.labels[0], inner_node.text))

                attention_labels = [e.split('/')[0] for e in self.categories]

                # category
                visual_attentions = [embedding_layer_category_alphas[j][i][: len(words)].detach().cpu().numpy()
                                     for j in range(self.category_num)]
                label_true = label[i].detach().cpu().numpy()[: self.category_num]
                # if sum(label_true) <= 1:
                #     continue
                titles = ['true: %s - pred: %s' % (str(label[i][j].detach().cpu().numpy()),
                                                   str(pred_category[j][i].detach().cpu().numpy()))
                          for j in range(self.category_num)]
                savefig_filepath = super()._get_model_visualization_picture_filepath(self.configuration, words)
                if not savefig_filepath:
                    attention_visualizer.plot_multi_attentions_of_sentence_backup(words, visual_attentions, attention_labels,
                                                                           titles, savefig_filepath=savefig_filepath)
                else:
                    attention_visualizer.plot_multi_attentions_of_sentence(words, visual_attentions, attention_labels,
                                                                           titles, savefig_filepath=savefig_filepath)

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {
            'category_f1': self._f1.get_metric(reset)['fscore']
        }
        return metrics


class AtaeLstmM(Model):
    def __init__(self, word_embedder: TextFieldEmbedder, position_embedder: TextFieldEmbedder,
                 aspect_embedder: TextFieldEmbedder,
                 categories: list, polarities: list, vocab: Vocabulary, configuration: dict):
        super().__init__(vocab)
        self.configuration = configuration
        self.word_embedder = word_embedder
        self.position_embedder = position_embedder
        self.aspect_embedder = aspect_embedder
        self.categories = categories
        self.polarites = polarities
        self.category_num = len(categories)
        self.polarity_num = len(polarities)
        self.category_loss = nn.BCEWithLogitsLoss()
        self.sentiment_loss = nn.CrossEntropyLoss()
        self._accuracy = metrics.CategoricalAccuracy()
        self._f1 = allennlp_metrics.BinaryF1(0.5)

        word_embedding_dim = word_embedder.get_output_dim()
        aspect_word_embedding_dim = aspect_embedder.get_output_dim()
        if self.configuration['model_name'] in ['ae-lstm-cae', 'atae-lstm-cae']:
            lstm_input_size = word_embedding_dim + aspect_word_embedding_dim
        else:
            lstm_input_size = word_embedding_dim
        num_layers = 1
        hidden_size = 300
        self.lstm = torch.nn.LSTM(lstm_input_size, hidden_size, batch_first=True,
                                  bidirectional=False, num_layers=num_layers)
        if self.configuration['model_name'] in ['at-lstm-m', 'atae-lstm-m']:
            self.category_attentions = [AttentionInHtt(word_embedding_dim, aspect_word_embedding_dim)
                                                          for _ in range(self.category_num)]
            self.sentiment_attention = AttentionInHtt(lstm_input_size, lstm_input_size)
            self.sentiment_fc = nn.Sequential(nn.Linear(hidden_size * 2, self.polarity_num))
            if self.configuration['model_name'] == 'atae-lstm-m':
                self.category_attentions_of_embedding_layer = [AttentionInHtt(hidden_size, hidden_size)
                                                               for _ in range(self.category_num)]
        else:
            self.sentiment_attention = None
            self.sentiment_fc = nn.Sequential(nn.Linear(hidden_size, self.polarity_num))

        if self.configuration['model_name'] == 'atae-lstm-m':
            self.category_fcs = [nn.Linear(word_embedding_dim + hidden_size, 1) for _ in range(self.category_num)]
        elif self.configuration['model_name'] == 'at-lstm-m':
            self.category_fcs = [nn.Linear(hidden_size, 1) for _ in range(self.category_num)]
        else:
            self.category_fcs = [nn.Linear(word_embedding_dim, 1) for _ in range(self.category_num)]

    def matrix_mul(self, input, weight, bias=False):
        feature_list = []
        for feature in input:
            feature = torch.mm(feature, weight)
            if isinstance(bias, torch.nn.parameter.Parameter):
                feature = feature + bias.expand(feature.size()[0], bias.size()[1])
            feature = torch.tanh(feature).unsqueeze(0)
            feature_list.append(feature)

        return torch.cat(feature_list, 0).squeeze()

    def element_wise_mul(self, input1, input2, return_not_sum_result=False):
        feature_list = []
        for feature_1, feature_2 in zip(input1, input2):
            feature_2 = feature_2.unsqueeze(1)
            feature_2 = feature_2.expand_as(feature_1)
            feature = feature_1 * feature_2
            feature = feature.unsqueeze(0)
            feature_list.append(feature)
        output = torch.cat(feature_list, 0)

        result = torch.sum(output, 1)
        if return_not_sum_result:
            return result, output
        else:
            return result

    def forward(self, tokens: Dict[str, torch.Tensor], label: torch.Tensor, position: torch.Tensor,
                polarity_mask: torch.Tensor, sample: list, aspects: torch.Tensor) -> torch.Tensor:
        mask = get_text_field_mask(tokens)
        word_embeddings = self.word_embedder(tokens)

        aspects_separate = [{'aspect': aspects['aspect'][:, i].unsqueeze(1)} for i in range(self.category_num)]
        aspect_embeddings_singles = [self.aspect_embedder(aspects_separate[i]).squeeze(1) for i in
                                     range(self.category_num)]
        aspects_seprate_repeat = [{'aspect': aspects_separate[i]['aspect'].expand_as(tokens['tokens'])} for i in
                                  range(self.category_num)]
        aspect_embeddings_separate = [self.aspect_embedder(aspects_seprate_repeat[i]) for i in range(self.category_num)]

        lstm_inputs = []
        embedding_layer_caees = []
        embedding_layer_category_alphas = []
        for i in range(self.category_num):
            if self.configuration['model_name'] in ['ae-lstm-m', 'atae-lstm-m']:
                aspect_embeddings = aspect_embeddings_separate[i]
                lstm_input = torch.cat([aspect_embeddings, word_embeddings], dim=-1)
            else:
                lstm_input = word_embeddings
            lstm_inputs.append(lstm_input)

        lstm_layer_category_alphas = []
        lstm_layer_caees = []
        final_sentiment_outputs = []
        final_sentiment_output_probs = []
        for i in range(self.category_num):
            lstm_input = lstm_inputs[i]
            lstm_output, (lstm_hn, lstm_cn) = self.lstm(lstm_input)
            lstm_hn = lstm_hn.squeeze(dim=0)

            if self.configuration['model_name'] in ['at-lstm-m', 'atae-lstm-m']:
                aspect_embeddings = aspect_embeddings_separate[i]
                input_for_attention = torch.cat([aspect_embeddings, lstm_output], dim=-1)
                alpha = self.sentiment_attention(input_for_attention, mask)
                sentiment_output = self.element_wise_mul(lstm_output, alpha, return_not_sum_result=False)
                sentiment_output = torch.cat([sentiment_output, lstm_hn], dim=-1)
            else:
                sentiment_output = lstm_hn
            final_sentiment_output = self.sentiment_fc(sentiment_output)
            final_sentiment_outputs.append(final_sentiment_output)
            final_sentiment_output_prob = torch.softmax(final_sentiment_output, dim=-1)
            final_sentiment_output_probs.append(final_sentiment_output_prob)

        final_category_outputs = []
        for i in range(self.category_num):
            category_output1 = embedding_layer_caees[i]
            category_output2 = lstm_layer_caees[i]
            category_output = torch.cat([category_output1, category_output2], dim=-1)
            final_category_output = self.category_fcs[i](category_output)
            final_category_outputs.append(final_category_output)

        output = {}
        if label is not None:
            category_labels = []
            polarity_labels = []
            polarity_masks = []
            for i in range(self.category_num):
                category_labels.append(label[:, i])
                polarity_labels.append(label[:, i + self.category_num])
                polarity_masks.append(polarity_mask[:, i])
            loss = 0
            for i in range(self.category_num):
                category_temp_loss = self.category_loss(final_category_outputs[i].squeeze(dim=-1), category_labels[i])
                sentiment_temp_loss = self.sentiment_loss(final_sentiment_outputs[i], polarity_labels[i].long())
                loss += category_temp_loss
                loss += sentiment_temp_loss

            # sentiment accuracy
            sentiment_logit = torch.cat(final_sentiment_outputs)
            sentiment_label = torch.cat(polarity_labels)
            sentiment_mask = torch.cat(polarity_masks)
            self._accuracy(sentiment_logit, sentiment_label, sentiment_mask)

            # category f1
            final_category_outputs_prob = [torch.sigmoid(e) for e in final_category_outputs]
            category_prob = torch.cat(final_category_outputs_prob).squeeze()
            category_label = torch.cat(category_labels)
            self._f1(category_prob, category_label)

            output['loss'] = loss

        # visualize attention
        pred_category = [torch.sigmoid(e) for e in final_category_outputs]
        pred_sentiment = [torch.nn.functional.softmax(e, dim=-1) for e in final_sentiment_outputs]
        output['pred_category'] = pred_category
        output['pred_sentiment'] = pred_sentiment
        if self.configuration['visualize_attention']:
            for i in range(len(sample)):
                words = sample[i][2]
                attention_labels = [e.split('/')[0] for e in self.categories]

                # category
                visual_attentions = [embedding_layer_category_alphas[j][i][: len(words)].detach().cpu().numpy()
                                     for j in range(self.category_num)]
                titles = ['true: %s - pred: %s' % (str(label[i][j].detach().cpu().numpy()),
                                                   str(pred_category[j][i].detach().cpu().numpy()))
                          for j in range(self.category_num)]
                attention_visualizer.plot_multi_attentions_of_sentence(words, visual_attentions, attention_labels,
                                                                       titles)

                # sentiment embedding layer
                visual_attentions = [embedding_layer_sentiment_alphas[j][i][: len(words)].detach().cpu().numpy()
                                     for j in range(self.category_num)]
                titles = ['true: %s - pred: %s - %s' % (str(label[i + self.category_num][j].detach().cpu().numpy()),
                                                        str(pred_sentiment[j][i].detach().cpu().numpy()),
                                                        str(self.polarites))
                          for j in range(self.category_num)]
                attention_visualizer.plot_multi_attentions_of_sentence(words, visual_attentions, attention_labels,
                                                                       titles)

                # sentiment lstm layer
                visual_attentions = [lstm_layer_sentiment_alphas[j][i][: len(words)].detach().cpu().numpy()
                                     for j in range(self.category_num)]
                titles = ['true: %s - pred: %s - %s' % (str(label[i + self.category_num][j].detach().cpu().numpy()),
                                                        str(pred_sentiment[j][i].detach().cpu().numpy()),
                                                        str(self.polarites))
                          for j in range(self.category_num)]
                attention_visualizer.plot_multi_attentions_of_sentence(words, visual_attentions, attention_labels,
                                                                       titles)

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {
            'accuracy': self._accuracy.get_metric(reset),
            'category_f1': self._f1.get_metric(reset)['fscore']
        }
        return metrics


class CaePretrainedPosition(Model):
    def __init__(self, word_embedder: TextFieldEmbedder, position_embedder: TextFieldEmbedder,
                 categories: list, polarities: list, vocab: Vocabulary, configuration: dict):
        super().__init__(vocab)
        self.configuration = configuration
        self.word_embedder = word_embedder
        self.position_embedder = position_embedder
        self.categories = categories
        self.polarites = polarities
        self.category_num = len(categories)
        self.polarity_num = len(polarities)
        self.category_loss = nn.BCEWithLogitsLoss()
        self.sentiment_loss = nn.CrossEntropyLoss()
        self._accuracy = metrics.CategoricalAccuracy()
        self._f1 = allennlp_metrics.BinaryF1(0.5)

        self.embedding_layer_fc = nn.Linear(word_embedder.get_output_dim(), word_embedder.get_output_dim(), bias=True)
        self.embedding_layer_aspect_attentions = [AttentionInHtt(word_embedder.get_output_dim(),
                                                                 word_embedder.get_output_dim())
                                                  for _ in range(self.category_num)]
        self.lstm = torch.nn.LSTM(word_embedder.get_output_dim(), int(word_embedder.get_output_dim() / 2), batch_first=True,
                                  bidirectional=True)
        self.category_fcs = [nn.Linear(word_embedder.get_output_dim(), 1) for _ in range(self.category_num)]
        self.sentiment_fc = nn.Linear(word_embedder.get_output_dim(), self.polarity_num)

        self._accuracy = metrics.CategoricalAccuracy()

    def matrix_mul(self, input, weight, bias=False):
        feature_list = []
        for feature in input:
            feature = torch.mm(feature, weight)
            if isinstance(bias, torch.nn.parameter.Parameter):
                feature = feature + bias.expand(feature.size()[0], bias.size()[1])
            feature = torch.tanh(feature).unsqueeze(0)
            feature_list.append(feature)

        return torch.cat(feature_list, 0).squeeze()

    def element_wise_mul(self, input1, input2, return_not_sum_result=False):
        feature_list = []
        for feature_1, feature_2 in zip(input1, input2):
            feature_2 = feature_2.unsqueeze(1)
            feature_2 = feature_2.expand_as(feature_1)
            feature = feature_1 * feature_2
            feature = feature.unsqueeze(0)
            feature_list.append(feature)
        output = torch.cat(feature_list, 0)

        result = torch.sum(output, 1)
        if return_not_sum_result:
            return result, output
        else:
            return result

    def forward(self, tokens: Dict[str, torch.Tensor], label: torch.Tensor, position: torch.Tensor,
                polarity_mask: torch.Tensor, sample: list) -> torch.Tensor:
        mask = get_text_field_mask(tokens)
        word_embeddings = self.word_embedder(tokens)
        word_embeddings_fc = self.embedding_layer_fc(word_embeddings)

        position_embeddings = self.position_embedder(position)

        embeddings = torch.cat([word_embeddings, position_embeddings], dim=-1)
        batch_size = embeddings.size()[0]
        embedding_layer_category_outputs = []
        embedding_layer_category_alphas = []
        embedding_layer_sentiment_outputs = []
        embedding_layer_sentiment_alphas = []
        for i in range(self.category_num):
            embedding_layer_aspect_attention = self.embedding_layer_aspect_attentions[i]
            alpha = embedding_layer_aspect_attention(word_embeddings_fc, mask)
            embedding_layer_category_alphas.append(alpha)

            category_output = self.element_wise_mul(word_embeddings_fc, alpha, return_not_sum_result=False)
            _, not_sum_result = self.element_wise_mul(embeddings, alpha, return_not_sum_result=True)
            embedding_layer_category_outputs.append(category_output)

            # sentiment
            sentiment_alpha_mat = torch.matmul(not_sum_result, embeddings.transpose(1, 2))
            sentiment_alpha_vector = sentiment_alpha_mat.sum(1)
            sentiment_alpha = allennlp_util.masked_softmax(sentiment_alpha_vector, mask)
            embedding_layer_sentiment_alphas.append(sentiment_alpha)
            sentiment_alpha = sentiment_alpha.unsqueeze(1)
            sentiment_output = torch.matmul(sentiment_alpha, word_embeddings).squeeze(1)  # batch_size x 2*hidden_dim
            embedding_layer_sentiment_outputs.append(sentiment_output)

        lstm_result, _ = self.lstm(word_embeddings)
        lstm_result_with_position = torch.cat([lstm_result, position_embeddings], dim=-1)
        lstm_layer_category_outputs = []
        lstm_layer_sentiment_outputs = []
        lstm_layer_sentiment_alphas = []
        for i in range(self.category_num):
            alpha = embedding_layer_category_alphas[i]
            category_output = self.element_wise_mul(lstm_result, alpha, return_not_sum_result=False)
            _, not_sum_result = self.element_wise_mul(lstm_result_with_position, alpha, return_not_sum_result=True)
            lstm_layer_category_outputs.append(category_output)

            # sentiment
            sentiment_alpha_mat = torch.matmul(not_sum_result, lstm_result_with_position.transpose(1, 2))
            sentiment_alpha_vector = sentiment_alpha_mat.sum(1)
            sentiment_alpha = allennlp_util.masked_softmax(sentiment_alpha_vector, mask)
            lstm_layer_sentiment_alphas.append(sentiment_alpha)
            sentiment_alpha = sentiment_alpha.unsqueeze(1)
            sentiment_output = torch.matmul(sentiment_alpha, lstm_result).squeeze(1)  # batch_size x 2*hidden_dim
            lstm_layer_sentiment_outputs.append(sentiment_output)

        final_category_outputs = []
        final_sentiment_outputs = []
        for i in range(self.category_num):
            fc = self.category_fcs[i]
            category_output = embedding_layer_category_outputs[i]
            final_category_output = fc(category_output)
            final_category_outputs.append(final_category_output)

            sentiment_output = embedding_layer_sentiment_outputs[i]
            final_sentiment_output = self.sentiment_fc(sentiment_output)
            final_sentiment_outputs.append(final_sentiment_output)

        output = {}
        if label is not None:
            category_labels = []
            polarity_labels = []
            polarity_masks = []
            for i in range(self.category_num):
                category_labels.append(label[:, i])
                polarity_labels.append(label[:, i + self.category_num])
                polarity_masks.append(polarity_mask[:, i])
            loss = 0
            for i in range(self.category_num):
                category_temp_loss = self.category_loss(final_category_outputs[i].squeeze(dim=-1), category_labels[i])
                sentiment_temp_loss = self.sentiment_loss(final_sentiment_outputs[i], polarity_labels[i].long())
                loss += category_temp_loss
                loss += sentiment_temp_loss

            # sentiment accuracy
            sentiment_logit = torch.cat(final_sentiment_outputs)
            sentiment_label = torch.cat(polarity_labels)
            sentiment_mask = torch.cat(polarity_masks)
            self._accuracy(sentiment_logit, sentiment_label, sentiment_mask)

            # category f1
            final_category_outputs_prob = [torch.sigmoid(e) for e in final_category_outputs]
            category_prob = torch.cat(final_category_outputs_prob).squeeze()
            category_label = torch.cat(category_labels)
            self._f1(category_prob, category_label)

            output['loss'] = loss

        # visualize attention
        pred_category = [torch.sigmoid(e) for e in final_category_outputs]
        pred_sentiment = [torch.nn.functional.softmax(e, dim=-1) for e in final_sentiment_outputs]
        output['pred_category'] = pred_category
        output['pred_sentiment'] = pred_sentiment
        if self.configuration['visualize_attention']:
            for i in range(len(sample)):
                words = sample[i][2]
                attention_labels = [e.split('/')[0] for e in self.categories]

                # category
                visual_attentions = [embedding_layer_category_alphas[j][i][: len(words)].detach().cpu().numpy()
                                     for j in range(self.category_num)]
                titles = ['true: %s - pred: %s' % (str(label[i][j].detach().cpu().numpy()),
                                                   str(pred_category[j][i].detach().cpu().numpy()))
                          for j in range(self.category_num)]
                attention_visualizer.plot_multi_attentions_of_sentence(words, visual_attentions, attention_labels,
                                                                       titles)

                # sentiment embedding layer
                visual_attentions = [embedding_layer_sentiment_alphas[j][i][: len(words)].detach().cpu().numpy()
                                     for j in range(self.category_num)]
                titles = ['true: %s - pred: %s - %s' % (str(label[i + self.category_num][j].detach().cpu().numpy()),
                                                        str(pred_sentiment[j][i].detach().cpu().numpy()),
                                                        str(self.polarites))
                          for j in range(self.category_num)]
                attention_visualizer.plot_multi_attentions_of_sentence(words, visual_attentions, attention_labels,
                                                                       titles)

                # sentiment lstm layer
                visual_attentions = [lstm_layer_sentiment_alphas[j][i][: len(words)].detach().cpu().numpy()
                                     for j in range(self.category_num)]
                titles = ['true: %s - pred: %s - %s' % (str(label[i + self.category_num][j].detach().cpu().numpy()),
                                                        str(pred_sentiment[j][i].detach().cpu().numpy()),
                                                        str(self.polarites))
                          for j in range(self.category_num)]
                attention_visualizer.plot_multi_attentions_of_sentence(words, visual_attentions, attention_labels,
                                                                       titles)

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {
            'accuracy': self._accuracy.get_metric(reset),
            'category_f1': self._f1.get_metric(reset)['fscore']
        }
        return metrics


class Mil(Model):
    def __init__(self, word_embedder: TextFieldEmbedder, position_embedder: TextFieldEmbedder,
                 categories: list, polarities: list, vocab: Vocabulary, configuration: dict):
        super().__init__(vocab)
        self.configuration = configuration
        self.word_embedder = word_embedder
        self.position_embedder = position_embedder
        self.categories = categories
        self.polarites = polarities
        self.category_num = len(categories)
        self.polarity_num = len(polarities)
        self.category_loss = nn.BCEWithLogitsLoss()
        self.sentiment_loss = nn.CrossEntropyLoss()
        self._accuracy = metrics.CategoricalAccuracy()
        self._f1 = allennlp_metrics.BinaryF1(0.5)

        self.embedding_layer_fc = nn.Linear(word_embedder.get_output_dim(), word_embedder.get_output_dim(), bias=True)
        self.embedding_layer_aspect_attentions = [AttentionInHtt(word_embedder.get_output_dim(),
                                                                 word_embedder.get_output_dim())
                                                  for _ in range(self.category_num)]
        self.lstm = torch.nn.LSTM(word_embedder.get_output_dim(), int(word_embedder.get_output_dim() / 2), batch_first=True,
                                  bidirectional=True)
        self.category_fcs = [nn.Linear(word_embedder.get_output_dim(), 1) for _ in range(self.category_num)]
        self.sentiment_fc = nn.Linear(word_embedder.get_output_dim(), self.polarity_num)

        self._accuracy = metrics.CategoricalAccuracy()

    def matrix_mul(self, input, weight, bias=False):
        feature_list = []
        for feature in input:
            feature = torch.mm(feature, weight)
            if isinstance(bias, torch.nn.parameter.Parameter):
                feature = feature + bias.expand(feature.size()[0], bias.size()[1])
            feature = torch.tanh(feature).unsqueeze(0)
            feature_list.append(feature)

        return torch.cat(feature_list, 0).squeeze()

    def element_wise_mul(self, input1, input2, return_not_sum_result=False):
        feature_list = []
        for feature_1, feature_2 in zip(input1, input2):
            feature_2 = feature_2.unsqueeze(1)
            feature_2 = feature_2.expand_as(feature_1)
            feature = feature_1 * feature_2
            feature = feature.unsqueeze(0)
            feature_list.append(feature)
        output = torch.cat(feature_list, 0)

        result = torch.sum(output, 1)
        if return_not_sum_result:
            return result, output
        else:
            return result

    def forward(self, tokens: Dict[str, torch.Tensor], label: torch.Tensor, position: torch.Tensor,
                polarity_mask: torch.Tensor, sample: list) -> torch.Tensor:
        mask = get_text_field_mask(tokens)
        word_embeddings = self.word_embedder(tokens)
        word_embeddings_fc = self.embedding_layer_fc(word_embeddings)

        position_embeddings = self.position_embedder(position)

        embeddings = torch.cat([word_embeddings, position_embeddings], dim=-1)
        embedding_layer_category_outputs = []
        embedding_layer_category_alphas = []
        embedding_layer_sentiment_outputs = []
        embedding_layer_sentiment_alphas = []
        for i in range(self.category_num):
            embedding_layer_aspect_attention = self.embedding_layer_aspect_attentions[i]
            alpha = embedding_layer_aspect_attention(word_embeddings_fc, mask)
            embedding_layer_category_alphas.append(alpha)

            category_output = self.element_wise_mul(word_embeddings_fc, alpha, return_not_sum_result=False)
            embedding_layer_category_outputs.append(category_output)

        lstm_result, _ = self.lstm(word_embeddings)
        lstm_result_with_position = torch.cat([lstm_result, position_embeddings], dim=-1)
        lstm_layer_category_outputs = []
        lstm_layer_sentiment_outputs = []
        lstm_layer_words_sentiment_soft = []
        for i in range(self.category_num):
            alpha = embedding_layer_category_alphas[i]
            category_output = self.element_wise_mul(lstm_result, alpha, return_not_sum_result=False)
            lstm_layer_category_outputs.append(category_output)

            # sentiment
            sentiment_alpha = embedding_layer_category_alphas[i]
            sentiment_alpha = sentiment_alpha.unsqueeze(1)
            words_sentiment = self.sentiment_fc(lstm_result)
            words_sentiment_soft = torch.softmax(words_sentiment, dim=-1)
            lstm_layer_words_sentiment_soft.append(words_sentiment_soft)
            sentiment_output = torch.matmul(sentiment_alpha, words_sentiment_soft).squeeze(1)  # batch_size x 2*hidden_dim
            lstm_layer_sentiment_outputs.append(sentiment_output)

        final_category_outputs = []
        final_sentiment_outputs = []
        for i in range(self.category_num):
            fc = self.category_fcs[i]
            category_output = embedding_layer_category_outputs[i]
            final_category_output = fc(category_output)
            final_category_outputs.append(final_category_output)

            sentiment_output = lstm_layer_sentiment_outputs[i]
            final_sentiment_output = sentiment_output
            final_sentiment_outputs.append(final_sentiment_output)

        output = {}
        if label is not None:
            category_labels = []
            polarity_labels = []
            polarity_masks = []
            for i in range(self.category_num):
                category_labels.append(label[:, i])
                polarity_labels.append(label[:, i + self.category_num])
                polarity_masks.append(polarity_mask[:, i])
            loss = 0
            for i in range(self.category_num):
                category_temp_loss = self.category_loss(final_category_outputs[i].squeeze(dim=-1), category_labels[i])
                sentiment_temp_loss = self.sentiment_loss(final_sentiment_outputs[i], polarity_labels[i].long())
                loss += category_temp_loss
                loss += sentiment_temp_loss

            # sentiment accuracy
            sentiment_logit = torch.cat(final_sentiment_outputs)
            sentiment_label = torch.cat(polarity_labels)
            sentiment_mask = torch.cat(polarity_masks)
            self._accuracy(sentiment_logit, sentiment_label, sentiment_mask)

            # category f1
            final_category_outputs_prob = [torch.sigmoid(e) for e in final_category_outputs]
            category_prob = torch.cat(final_category_outputs_prob).squeeze()
            category_label = torch.cat(category_labels)
            self._f1(category_prob, category_label)

            output['loss'] = loss

        # visualize attention
        pred_category = [torch.sigmoid(e) for e in final_category_outputs]
        pred_sentiment = [torch.nn.functional.softmax(e, dim=-1) for e in final_sentiment_outputs]
        output['pred_category'] = pred_category
        output['pred_sentiment'] = pred_sentiment
        if self.configuration['visualize_attention']:
            for i in range(len(sample)):
                words = sample[i][2]
                attention_labels = [e.split('/')[0] for e in self.categories]

                # category
                visual_attentions = [embedding_layer_category_alphas[j][i][: len(words)].detach().cpu().numpy()
                                     for j in range(self.category_num)]
                titles = ['true: %s - pred: %s' % (str(label[i][j].detach().cpu().numpy()),
                                                   str(pred_category[j][i].detach().cpu().numpy()))
                          for j in range(self.category_num)]
                attention_visualizer.plot_multi_attentions_of_sentence(words, visual_attentions, attention_labels,
                                                                       titles)

                # sentiment embedding layer
                # visual_attentions = [embedding_layer_sentiment_alphas[j][i][: len(words)].detach().cpu().numpy()
                #                      for j in range(self.category_num)]
                # titles = ['true: %s - pred: %s - %s' % (str(label[i + self.category_num][j].detach().cpu().numpy()),
                #                                         str(pred_sentiment[j][i].detach().cpu().numpy()),
                #                                         str(self.polarites))
                #           for j in range(self.category_num)]
                # attention_visualizer.plot_multi_attentions_of_sentence(words, visual_attentions, attention_labels,
                #                                                        titles)

                # sentiment lstm layer
                # visual_attentions = [lstm_layer_sentiment_alphas[j][i][: len(words)].detach().cpu().numpy()
                #                      for j in range(self.category_num)]
                # titles = ['true: %s - pred: %s - %s' % (str(label[i + self.category_num][j].detach().cpu().numpy()),
                #                                         str(pred_sentiment[j][i].detach().cpu().numpy()),
                #                                         str(self.polarites))
                #           for j in range(self.category_num)]
                # attention_visualizer.plot_multi_attentions_of_sentence(words, visual_attentions, attention_labels,
                #                                                        titles)

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {
            'accuracy': self._accuracy.get_metric(reset),
            'category_f1': self._f1.get_metric(reset)['fscore']
        }
        return metrics

from dgl import DGLGraph


class GatEdge:
    """
    story information of a edge in gat for visualization
    """

    def __init__(self, src_ids: List[int], dst_id: int, alphas: List[float]):
        self.dst_id = dst_id
        self.src_ids = src_ids
        self.alphas = alphas

    def add(self, other_edge: 'GatEdge'):
        """

        :param other_edge:  同样的边
        :return:
        """
        if self.src_ids != other_edge.src_ids or self.dst_id != other_edge.dst_id:
            print('add error')
            return
        for i in range(len(self.alphas)):
            self.alphas[i] += other_edge.alphas[i]

    def divide(self, number: int):
        self.alphas = [alpha / number for alpha in self.alphas]

    def __str__(self):
        return '%s-%s-%s' % (str(self.dst_id), str(self.src_ids), str(self.alphas))


class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, opt: dict={}):
        super().__init__()
        self.opt = opt
        # equation (1)
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        # equation (2)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)

    def edge_attention(self, edges: dgl.EdgeBatch):
        # edge UDF for equation (2)
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges: dgl.EdgeBatch):
        # message UDF for equation (3) & (4)
        result = {'z': edges.src['z'], 'node_ids': edges.src['node_ids'], 'e': edges.data['e']}
        return result

    def reduce_func(self, nodes: dgl.NodeBatch):
        # for visualization
        src_node_ids = nodes.mailbox['node_ids']
        dst_node_ids = nodes.data['node_ids']

        # reduce UDF for equation (3) & (4)
        # equation (3)

        e = nodes.mailbox['e']
        alpha = F.softmax(e, dim=1)
        # equation (4)
        z = nodes.mailbox['z']
        h = torch.sum(alpha * z, dim=1)

        self.for_visualization.append({'src_node_ids': src_node_ids, 'dst_node_ids': dst_node_ids,
                                       'alpha': alpha})

        return {'h': h}

    def forward(self, h, g: List[DGLGraph]):
        self.for_visualization = []

        batched_graph = dgl.batch(g)
        feature = h.view([-1, h.size()[-1]])

        # equation (1)
        z = self.fc(feature)
        batched_graph.ndata['z'] = z

        # 加入node id，用于attention可视化
        node_ids = batched_graph.nodes()
        batched_graph.ndata['node_ids'] = node_ids

        # equation (2)
        batched_graph.apply_edges(self.edge_attention)
        # equation (3) & (4)
        batched_graph.update_all(self.message_func, self.reduce_func)

        ug = dgl.unbatch(batched_graph)
        # output = [torch.unsqueeze(g.ndata.pop('h'), 0).to(self.opt['device']) for g in ug]
        # for visualization in local machine
        output = [torch.unsqueeze(g.ndata.pop('h'), 0) for g in ug]
        output = torch.cat(output, 0)

        # 对for_visualization按照sample进行拆分
        sample_num = len(g)
        node_num_per_sample = h.shape[1]
        edges_of_samples = [[] for _ in range(sample_num)]
        if 'gat_visualization' not in self.opt or self.opt['gat_visualization']:
            for edges in self.for_visualization:
                edge_num = edges['dst_node_ids'].shape[0]
                for i in range(edge_num):
                    src_ids = edges['src_node_ids'][i].cpu().numpy().tolist()
                    src_ids_real = [e % node_num_per_sample for e in src_ids]
                    dst_id = edges['dst_node_ids'][i].cpu().numpy().tolist()
                    dst_id_real = dst_id % node_num_per_sample
                    sample_index = dst_id // node_num_per_sample
                    alphas = edges['alpha'][i].squeeze(dim=-1).cpu().numpy().tolist()
                    edge = GatEdge(src_ids_real, dst_id_real, alphas)
                    edges_of_samples[sample_index].append(edge)

        return output, edges_of_samples


class MultiHeadGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, merge='cat', opt: dict={}):
        super().__init__()
        self.opt = opt
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(in_dim, out_dim, opt))
        self.merge = merge

    def forward(self, h, g: List[DGLGraph]):
        head_outs = [attn_head(h, g) for attn_head in self.heads]
        head_outs_feature = [e[0] for e in head_outs]
        head_outs_attention = [e[1] for e in head_outs]
        if self.merge == 'cat':
            # concat on the output feature dimension (dim=1)
            return torch.cat(head_outs_feature, dim=2), head_outs_attention
        else:
            # merge using average
            return torch.mean(torch.stack(head_outs_feature)), head_outs_attention


class GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads, opt: dict={}):
        super().__init__()
        self.opt = opt
        self.layer1 = MultiHeadGATLayer(in_dim, int(out_dim / num_heads), num_heads, opt=opt)
        # Be aware that the input dimension is hidden_dim*num_heads since
        # multiple head outputs are concatenated together. Also, only
        # one attention head in the output layer.
        # self.layer2 = MultiHeadGATLayer(hidden_dim * num_heads, out_dim, 1)

    def forward(self, h, g: List[DGLGraph]):
        h, attention = self.layer1(h, g)
        # h = F.elu(h)
        # h = self.layer2(h, g)
        # 兼容之前保存的模型
        return h, attention
        # if 'gat_visualization' not in self.opt or self.opt['gat_visualization']:
        #     return h, attention
        # else:
        #     return h


class GATForAspectCategory(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=-1, num_heads=4, opt: dict={}):
        super().__init__()
        self.opt = opt
        self.layer1 = MultiHeadGATLayer(in_dim, int(out_dim / num_heads), num_heads, opt=opt)
        # Be aware that the input dimension is hidden_dim*num_heads since
        # multiple head outputs are concatenated together. Also, only
        # one attention head in the output layer.
        # self.layer2 = MultiHeadGATLayer(hidden_dim * num_heads, out_dim, 1)

    def forward(self, h, g: List[DGLGraph]):
        h, attention = self.layer1(h, g)
        # h = F.elu(h)
        # h = self.layer2(h, g)
        return h, attention
        # if 'gat_visualization' not in self.opt or self.opt['gat_visualization']:
        #     return h, attention
        # else:
        #     return h


class NodeApplyModule(nn.Module):
    def __init__(self, in_feats, out_feats, activation, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_feats, out_feats, bias=bias)
        self.activation = activation

    def forward(self, node):
        h = self.linear(node.data['h'])
        if self.activation is not None:
            h = self.activation(h)
        return {'h': h}


class DglGraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, opt, bias=True, activation=F.relu):
        super().__init__()
        self.opt = opt
        self.in_features = in_features
        self.out_features = out_features
        self.apply_mod = NodeApplyModule(in_features, out_features, activation, bias=bias)

    def _sum(self, nodes: dgl.NodeBatch):
        """Take an average over all neighbor node features hu and use it to
        overwrite the original node feature."""
        m = nodes.mailbox['m']
        accum = torch.sum(m, 1)
        return {'h': accum}

    def _mean(self, nodes: dgl.NodeBatch):
        """Take an average over all neighbor node features hu and use it to
        overwrite the original node feature."""
        m = nodes.mailbox['m']
        mean = torch.mean(m, 1)
        return {'h': mean}

    def forward(self, text, graphs):
        hidden = text
        batched_graph = dgl.batch(graphs)
        feature = hidden.view([-1, hidden.size()[-1]])
        if feature.size()[0] != batched_graph.number_of_nodes():
            print('error')
        batched_graph.ndata['h'] = feature

        gcn_msg = dgl_fn.copy_src(src='h', out='m')
        gcn_reduce = self._mean

        batched_graph.update_all(gcn_msg, gcn_reduce)
        batched_graph.apply_nodes(func=self.apply_mod)

        ug = dgl.unbatch(batched_graph)
        output = [torch.unsqueeze(g.ndata.pop('h'), 0).to(self.opt['device']) for g in ug]
        output = torch.cat(output, 0)
        return output


class DglGraphConvolutionForAspectCategory(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, opt, bias=True, activation=F.relu):
        super().__init__()
        self.opt = opt
        self.in_features = in_features
        self.out_features = out_features
        self.apply_mod = NodeApplyModule(in_features, out_features, activation, bias=bias)

    def _sum(self, nodes: dgl.NodeBatch):
        """Take an average over all neighbor node features hu and use it to
        overwrite the original node feature."""
        m = nodes.mailbox['m']
        accum = torch.sum(m, 1)
        return {'h': accum}

    def _mean(self, nodes: dgl.NodeBatch):
        """Take an average over all neighbor node features hu and use it to
        overwrite the original node feature."""
        m = nodes.mailbox['m']
        mean = torch.mean(m, 1)
        return {'h': mean}

    def forward(self, text, graphs):
        hidden = text
        batched_graph = dgl.batch(graphs)
        feature = hidden.view([-1, hidden.size()[-1]])
        if feature.size()[0] != batched_graph.number_of_nodes():
            print('error')
        batched_graph.ndata['h'] = feature

        gcn_msg = dgl_fn.copy_src(src='h', out='m')
        gcn_reduce = self._mean

        batched_graph.update_all(gcn_msg, gcn_reduce)
        batched_graph.apply_nodes(func=self.apply_mod)

        ug = dgl.unbatch(batched_graph)
        output = [torch.unsqueeze(g.ndata.pop('h'), 0).to(self.opt['device']) for g in ug]
        output = torch.cat(output, 0)
        return output


class DglGraphAverage(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, opt, bias=True, activation=F.relu):
        super().__init__()
        self.opt = opt
        self.in_features = in_features
        self.out_features = out_features

    def _sum(self, nodes: dgl.NodeBatch):
        """Take an average over all neighbor node features hu and use it to
        overwrite the original node feature."""
        m = nodes.mailbox['m']
        accum = torch.sum(m, 1)
        return {'h': accum}

    def _mean(self, nodes: dgl.NodeBatch):
        """Take an average over all neighbor node features hu and use it to
        overwrite the original node feature."""
        m = nodes.mailbox['m']
        mean = torch.mean(m, 1)
        return {'h': mean}

    def forward(self, text, graphs):
        hidden = text
        batched_graph = dgl.batch(graphs)
        feature = hidden.view([-1, hidden.size()[-1]])
        if feature.size()[0] != batched_graph.number_of_nodes():
            print('error')
        batched_graph.ndata['h'] = feature

        gcn_msg = dgl_fn.copy_src(src='h', out='m')
        gcn_reduce = self._mean

        batched_graph.update_all(gcn_msg, gcn_reduce)

        ug = dgl.unbatch(batched_graph)
        output = [torch.unsqueeze(g.ndata.pop('h'), 0).to(self.opt['device']) for g in ug]
        output = torch.cat(output, 0)
        return output


class DglGraphAverageForAspectCategory(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, opt, bias=True, activation=F.relu):
        super().__init__()
        self.opt = opt
        self.in_features = in_features
        self.out_features = out_features

    def _sum(self, nodes: dgl.NodeBatch):
        """Take an average over all neighbor node features hu and use it to
        overwrite the original node feature."""
        m = nodes.mailbox['m']
        accum = torch.sum(m, 1)
        return {'h': accum}

    def _mean(self, nodes: dgl.NodeBatch):
        """Take an average over all neighbor node features hu and use it to
        overwrite the original node feature."""
        m = nodes.mailbox['m']
        mean = torch.mean(m, 1)
        return {'h': mean}

    def forward(self, text, graphs):
        hidden = text
        batched_graph = dgl.batch(graphs)
        feature = hidden.view([-1, hidden.size()[-1]])
        if feature.size()[0] != batched_graph.number_of_nodes():
            print('error')
        batched_graph.ndata['h'] = feature

        gcn_msg = dgl_fn.copy_src(src='h', out='m')
        gcn_reduce = self._mean

        batched_graph.update_all(gcn_msg, gcn_reduce)

        ug = dgl.unbatch(batched_graph)
        output = [torch.unsqueeze(g.ndata.pop('h'), 0).to(self.opt['device']) for g in ug]
        output = torch.cat(output, 0)
        return output


class DglGraphAttentionForAspectCategory(nn.Module):
    """
    计算在所有aspect的attention下孩子节点的权重，所有权重求和然后归一化，即得到最终的
    孩子节点权重
    """
    def __init__(self, in_features, out_features, aspect_attentions, opt, bias=True, activation=F.relu):
        super().__init__()
        self.opt = opt
        self.aspect_attentions: List[AttentionInHtt] = aspect_attentions
        self.in_features = in_features
        self.out_features = out_features

    def element_wise_mul(self, input1, input2, return_not_sum_result=False):
        feature_list = []
        for feature_1, feature_2 in zip(input1, input2):
            feature_2 = feature_2.unsqueeze(1)
            feature_2 = feature_2.expand_as(feature_1)
            feature = feature_1 * feature_2
            feature = feature.unsqueeze(0)
            feature_list.append(feature)
        output = torch.cat(feature_list, 0)

        result = torch.sum(output, 1)
        if return_not_sum_result:
            return result, output
        else:
            return result

    def _message_func(self, edges: dgl.EdgeBatch):
        result = {'m': edges.src['h'], 'm_sentiment': edges.src['h_sentiment']}
        return result

    def _sum(self, nodes: dgl.NodeBatch):
        """Take an average over all neighbor node features hu and use it to
        overwrite the original node feature."""
        m = nodes.mailbox['m']
        accum = torch.sum(m, 1)
        return {'h': accum}

    def _mean(self, nodes: dgl.NodeBatch):
        """Take an average over all neighbor node features hu and use it to
        overwrite the original node feature."""
        m = nodes.mailbox['m']
        mean = torch.mean(m, 1)
        return {'h': mean}

    def _attention(self, nodes: dgl.NodeBatch):
        """Take an average over all neighbor node features hu and use it to
        overwrite the original node feature."""
        m = nodes.mailbox['m']
        alphas = []
        for i in range(len(self.aspect_attentions)):
            aspect_attention = self.aspect_attentions[i]
            alpha = aspect_attention(m, None)
            alphas.append(alpha.unsqueeze(1))
        alpha_cat = torch.cat(alphas, dim=1)
        alpha_final = torch.mean(alpha_cat, dim=1)
        h = self.element_wise_mul(m, alpha_final, return_not_sum_result=False)
        m_sentiment = nodes.mailbox['m_sentiment']
        h_sentiment = self.element_wise_mul(m_sentiment, alpha_final, return_not_sum_result=False)
        return {'h': h, 'h_sentiment': h_sentiment}

    def forward(self, aspect_representation, graphs, sentiment_representation=None):
        if sentiment_representation is None:
            sentiment_representation_flag = False
            sentiment_representation = aspect_representation
        else:
            sentiment_representation_flag = True
        hidden = aspect_representation
        batched_graph = dgl.batch(graphs)
        feature = hidden.view([-1, hidden.size()[-1]])
        feature_sentiment = sentiment_representation.view([-1, sentiment_representation.size()[-1]])
        if feature.size()[0] != batched_graph.number_of_nodes():
            print('error')
        batched_graph.ndata['h'] = feature
        batched_graph.ndata['h_sentiment'] = feature_sentiment

        # gcn_msg = dgl_fn.copy_src(src='h', out='m')
        gcn_msg = self._message_func
        gcn_reduce = self._attention

        batched_graph.update_all(gcn_msg, gcn_reduce)

        ug = dgl.unbatch(batched_graph)
        output = [torch.unsqueeze(g.ndata.pop('h'), 0).to(self.opt['device']) for g in ug]
        output = torch.cat(output, 0)

        output_sentiment = [torch.unsqueeze(g.ndata.pop('h_sentiment'), 0).to(self.opt['device']) for g in ug]
        output_sentiment = torch.cat(output_sentiment, 0)
        if sentiment_representation_flag:
            return output, output_sentiment
        else:
            return output


class DglGraphAttentionForAspectCategoryWithDottedLines(nn.Module):
    """
    计算在所有aspect的attention下孩子节点的权重，所有权重求和然后归一化，即得到最终的
    孩子节点权重
    """
    def __init__(self, in_features, out_features, aspect_attentions, opt, bias=True, activation=F.relu):
        super().__init__()
        self.opt = opt
        self.aspect_attentions: List[AttentionInHtt] = aspect_attentions
        self.in_features = in_features
        self.out_features = out_features

    def element_wise_mul(self, input1, input2, return_not_sum_result=False):
        feature_list = []
        for feature_1, feature_2 in zip(input1, input2):
            feature_2 = feature_2.unsqueeze(1)
            feature_2 = feature_2.expand_as(feature_1)
            feature = feature_1 * feature_2
            feature = feature.unsqueeze(0)
            feature_list.append(feature)
        output = torch.cat(feature_list, 0)

        result = torch.sum(output, 1)
        if return_not_sum_result:
            return result, output
        else:
            return result

    def _message_func(self, edges: dgl.EdgeBatch):
        result = {'m': edges.src['h'], 'dotted_line_masks': edges.data['dotted_line_masks'],
                  'm_sentiment': edges.src['h_sentiment']}
        return result

    def _sum(self, nodes: dgl.NodeBatch):
        """Take an average over all neighbor node features hu and use it to
        overwrite the original node feature."""
        m = nodes.mailbox['m']
        accum = torch.sum(m, 1)
        return {'h': accum}

    def _mean(self, nodes: dgl.NodeBatch):
        """Take an average over all neighbor node features hu and use it to
        overwrite the original node feature."""
        m = nodes.mailbox['m']
        mean = torch.mean(m, 1)
        return {'h': mean}

    def _attention(self, nodes: dgl.NodeBatch):
        """Take an average over all neighbor node features hu and use it to
        overwrite the original node feature."""
        m = nodes.mailbox['m']
        dotted_line_masks = nodes.mailbox['dotted_line_masks'].to(self.opt['device'])
        # for i in range(dotted_line_masks.size()[0]):
        #     print(dotted_line_masks[i])
        alphas = []
        for i in range(len(self.aspect_attentions)):
            aspect_attention = self.aspect_attentions[i]
            alpha_temp = aspect_attention(m, None)
            alpha = alpha_temp * dotted_line_masks.float()
            alphas.append(alpha.unsqueeze(1))
        alpha_cat = torch.cat(alphas, dim=1)
        alpha_final = torch.mean(alpha_cat, dim=1)
        h = self.element_wise_mul(m, alpha_final, return_not_sum_result=False)
        m_sentiment = nodes.mailbox['m_sentiment']
        h_sentiment = self.element_wise_mul(m_sentiment, alpha_final, return_not_sum_result=False)
        return {'h': h, 'h_sentiment': h_sentiment}

    def forward(self, aspect_representation, graphs, sentiment_representation=None):
        if sentiment_representation is None:
            sentiment_representation_flag = False
            # 方便其它逻辑顺利执行
            sentiment_representation = aspect_representation
        else:
            sentiment_representation_flag = True
        hidden = aspect_representation
        batched_graph = dgl.batch(graphs)
        feature = hidden.view([-1, hidden.size()[-1]])
        feature_sentiment = sentiment_representation.view([-1, sentiment_representation.size()[-1]])
        if feature.size()[0] != batched_graph.number_of_nodes():
            print('error')
        batched_graph.ndata['h'] = feature
        batched_graph.ndata['h_sentiment'] = feature_sentiment

        # gcn_msg = dgl_fn.copy_src(src='h', out='m')
        gcn_msg = self._message_func
        gcn_reduce = self._attention

        batched_graph.update_all(gcn_msg, gcn_reduce)

        ug = dgl.unbatch(batched_graph)
        output = [torch.unsqueeze(g.ndata.pop('h'), 0).to(self.opt['device']) for g in ug]
        output = torch.cat(output, 0)

        output_sentiment = [torch.unsqueeze(g.ndata.pop('h_sentiment'), 0).to(self.opt['device']) for g in ug]
        output_sentiment = torch.cat(output_sentiment, 0)
        if sentiment_representation_flag:
            return output, output_sentiment
        else:
            return output


class AsMilSimultaneously(TextInAllAspectSentimentOutModel):
    def __init__(self, word_embedder: TextFieldEmbedder, position_embedder: TextFieldEmbedder,
                 aspect_embedder: TextFieldEmbedder, categories: list, polarities: list, vocab: Vocabulary,
                 configuration: dict, category_loss_weight=1, sentiment_loss_weight=1,
                 acd_model: TextInAllAspectSentimentOutModel=None):
        super().__init__(vocab, category_loss_weight=category_loss_weight, sentiment_loss_weight=sentiment_loss_weight)
        self.configuration = configuration
        self.word_embedder = word_embedder
        self.position_embedder = position_embedder
        self.aspect_embedder = aspect_embedder
        self.categories = categories
        self.polarites = polarities
        self.category_num = len(categories)
        self.polarity_num = len(polarities)
        self.category_loss = nn.BCEWithLogitsLoss()
        self.sentiment_loss = nn.CrossEntropyLoss()
        self._accuracy = metrics.CategoricalAccuracy()
        self._f1 = allennlp_metrics.BinaryF1(0.5)
        self.acd_model = acd_model

        word_embedding_dim = word_embedder.get_output_dim()
        if self.configuration['position']:
            word_embedding_dim += position_embedder.get_output_dim()
        self.embedding_layer_fc = nn.Linear(word_embedding_dim, word_embedding_dim, bias=True)
        self.embedding_layer_aspect_attentions = [AttentionInHtt(word_embedding_dim,
                                                                 word_embedding_dim)
                                                  for _ in range(self.category_num)]
        self.embedding_layer_aspect_attentions = nn.ModuleList(self.embedding_layer_aspect_attentions)

        lstm_input_size = word_embedding_dim
        num_layers = 2
        self.lstm = torch.nn.LSTM(lstm_input_size, int(word_embedding_dim / 2), batch_first=True,
                                  bidirectional=True, num_layers=num_layers, dropout=0.5)

        self.dropout_after_embedding_layer = nn.Dropout(0.5)
        self.dropout_after_lstm_layer = nn.Dropout(0.5)

        self.category_fcs = [nn.Linear(word_embedding_dim, 1) for _ in range(self.category_num)]
        self.category_fcs = nn.ModuleList(self.category_fcs)

        if self.configuration['lstm_layer_category_classifier']:
            self.lstm_category_fcs = [nn.Linear(word_embedding_dim, 1) for _ in range(self.category_num)]
            self.lstm_category_fcs = nn.ModuleList(self.lstm_category_fcs)

        sentiment_fc_input_size = word_embedding_dim
        if not self.configuration['share_sentiment_classifier']:
            self.sentiment_fcs = [nn.Sequential(nn.Linear(sentiment_fc_input_size, sentiment_fc_input_size),
                                                nn.ReLU(),
                                                nn.Linear(sentiment_fc_input_size, self.polarity_num))
                                  for _ in range(self.category_num)]
            self.sentiment_fcs = nn.ModuleList(self.sentiment_fcs)
        else:
            self.sentiment_fc = nn.Sequential(nn.Linear(sentiment_fc_input_size, sentiment_fc_input_size),
                                              nn.ReLU(),
                                              nn.Linear(sentiment_fc_input_size, self.polarity_num))

        # self.gc1 = DglGraphConvolution(word_embedding_dim, word_embedding_dim, configuration)
        # self.gc2 = DglGraphConvolution(word_embedding_dim, word_embedding_dim, configuration)

    def forward(self, tokens: Dict[str, torch.Tensor], label: torch.Tensor, position: torch.Tensor,
                polarity_mask: torch.Tensor, sample: list, aspects: torch.Tensor=None) -> torch.Tensor:
        mask = get_text_field_mask(tokens)
        word_embeddings = self.word_embedder(tokens)
        if self.configuration['position']:
            position_embeddings = self.position_embedder(position)
            word_embeddings = torch.cat([word_embeddings, position_embeddings], dim=-1)
        word_embeddings_fc = self.embedding_layer_fc(word_embeddings)

        word_embeddings = self.dropout_after_embedding_layer(word_embeddings)

        aspects_separate = [{'aspect': aspects['aspect'][:, i].unsqueeze(1)} for i in range(self.category_num)]
        aspect_embeddings_singles = [self.aspect_embedder(aspects_separate[i]).squeeze(1) for i in
                                     range(self.category_num)]
        aspects_seprate_repeat = [{'aspect': aspects_separate[i]['aspect'].expand_as(tokens['tokens'])} for i in
                                  range(self.category_num)]
        aspect_embeddings_separate = [self.aspect_embedder(aspects_seprate_repeat[i]) for i in range(self.category_num)]

        embedding_layer_category_outputs = []
        embedding_layer_category_alphas = []
        embedding_layer_sentiment_outputs = []
        embedding_layer_sentiment_alphas = []

        if self.configuration['pipeline'] and self.acd_model is not None:
            acd_input = {
                'tokens': tokens,
                'label': label,
                'position': position,
                'polarity_mask': polarity_mask,
                'sample': sample,
                'aspects': aspects
            }
            self.acd_model.eval()
            alpha_from_acd_model = self.acd_model(**acd_input)
            embedding_layer_category_alphas = alpha_from_acd_model['alpha']
        else:
            embedding_layer_category_alphas = []
            for i in range(self.category_num):
                embedding_layer_aspect_attention = self.embedding_layer_aspect_attentions[i]
                alpha = embedding_layer_aspect_attention(word_embeddings_fc, mask)
                embedding_layer_category_alphas.append(alpha)

        for i in range(self.category_num):
            alpha = embedding_layer_category_alphas[i]

            category_output = self.element_wise_mul(word_embeddings_fc, alpha, return_not_sum_result=False)
            embedding_layer_category_outputs.append(category_output)

        lstm_result, _ = self.lstm(word_embeddings)
        lstm_result = self.dropout_after_lstm_layer(lstm_result)
        # lstm_result_with_position = torch.cat([lstm_result, position_embeddings], dim=-1)
        lstm_layer_category_outputs = []
        lstm_layer_sentiment_outputs = []
        lstm_layer_words_sentiment_soft = []

        # max_len = tokens['tokens'].size()[1]
        # graphs = [e[3] for e in sample]
        # graphs_padded = self.pad_dgl_graph(graphs, max_len)
        # graph_output1 = F.relu(self.gc1(word_embeddings, graphs_padded))
        # graph_output2 = F.relu(self.gc2(graph_output1, graphs_padded))
        for i in range(self.category_num):
            alpha = embedding_layer_category_alphas[i]
            category_output = self.element_wise_mul(lstm_result, alpha, return_not_sum_result=False)
            lstm_layer_category_outputs.append(category_output)

            # sentiment
            # word_representation_for_sentiment = torch.cat([graph_output2, lstm_result], dim=-1)
            word_representation_for_sentiment = lstm_result
            sentiment_alpha = embedding_layer_category_alphas[i]
            if self.configuration['mil']:
                sentiment_alpha = sentiment_alpha.unsqueeze(1)
                if not self.configuration['share_sentiment_classifier']:
                    words_sentiment = self.sentiment_fcs[i](word_representation_for_sentiment)
                else:
                    words_sentiment = self.sentiment_fc(word_representation_for_sentiment)
                if self.configuration['mil_softmax']:
                    words_sentiment_soft = torch.softmax(words_sentiment, dim=-1)
                    lstm_layer_words_sentiment_soft.append(words_sentiment_soft)
                else:
                    words_sentiment_soft = words_sentiment
                    lstm_layer_words_sentiment_soft.append(torch.softmax(words_sentiment, dim=-1))
                sentiment_output = torch.matmul(sentiment_alpha, words_sentiment_soft).squeeze(1)  # batch_size x 2*hidden_dim
                lstm_layer_sentiment_outputs.append(sentiment_output)
            else:
                sentiment_output_temp = self.element_wise_mul(word_representation_for_sentiment, sentiment_alpha,
                                                         return_not_sum_result=False)
                if not self.configuration['share_sentiment_classifier']:
                    sentiment_output = self.sentiment_fcs[i](sentiment_output_temp)
                else:
                    sentiment_output = self.sentiment_fc(sentiment_output_temp)
                lstm_layer_sentiment_outputs.append(sentiment_output)

        final_category_outputs = []
        final_lstm_category_outputs = []
        final_sentiment_outputs = []
        for i in range(self.category_num):
            fc = self.category_fcs[i]
            category_output = embedding_layer_category_outputs[i]
            final_category_output = fc(category_output)
            final_category_outputs.append(final_category_output)

            if self.configuration['lstm_layer_category_classifier']:
                fc_lstm = self.lstm_category_fcs[i]
                lstm_category_output = lstm_layer_category_outputs[i]
                final_lstm_category_output = fc_lstm(lstm_category_output)
                final_lstm_category_outputs.append(final_lstm_category_output)

            final_sentiment_output = lstm_layer_sentiment_outputs[i]
            final_sentiment_outputs.append(final_sentiment_output)

        output = {}
        output['alpha'] = embedding_layer_category_alphas
        if label is not None:
            category_labels = []
            polarity_labels = []
            polarity_masks = []
            for i in range(self.category_num):
                category_labels.append(label[:, i])
                polarity_labels.append(label[:, i + self.category_num])
                polarity_masks.append(polarity_mask[:, i])
            loss = 0
            total_category_loss = 0
            total_sentiment_loss = 0
            for i in range(self.category_num):
                category_temp_loss = self.category_loss(final_category_outputs[i].squeeze(dim=-1), category_labels[i])
                sentiment_temp_loss = self.sentiment_loss(final_sentiment_outputs[i], polarity_labels[i].long())
                total_category_loss += category_temp_loss
                if not self.configuration['only_acd']:
                    total_sentiment_loss += sentiment_temp_loss
                if self.configuration['lstm_layer_category_classifier']:
                    lstm_category_temp_loss = self.category_loss(final_lstm_category_outputs[i].squeeze(dim=-1),
                                                                 category_labels[i])
                    total_category_loss += lstm_category_temp_loss
            loss = self.category_loss_weight * total_category_loss + self.sentiment_loss_weight * total_sentiment_loss

            # sentiment accuracy
            sentiment_logit = torch.cat(final_sentiment_outputs)
            sentiment_label = torch.cat(polarity_labels)
            sentiment_mask = torch.cat(polarity_masks)
            self._accuracy(sentiment_logit, sentiment_label, sentiment_mask)

            # category f1
            final_category_outputs_prob = [torch.sigmoid(e) for e in final_category_outputs]
            category_prob = torch.cat(final_category_outputs_prob).squeeze()
            category_label = torch.cat(category_labels)
            self._f1(category_prob, category_label)

            output['loss'] = loss

        # visualize attention
        pred_category = [torch.sigmoid(e) for e in final_category_outputs]
        pred_sentiment = [torch.nn.functional.softmax(e, dim=-1) for e in final_sentiment_outputs]
        output['pred_category'] = pred_category
        output['pred_sentiment'] = pred_sentiment
        if self.configuration['visualize_attention']:
            for i in range(len(sample)):
                words = sample[i][2]
                attention_labels = [e.split('/')[0] for e in self.categories]

                # category
                visual_attentions_category = [embedding_layer_category_alphas[j][i][: len(words)].detach().cpu().numpy()
                                              for j in range(self.category_num)]
                titles = ['true: %s - pred: %s' % (str(label[i][j].detach().cpu().numpy()),
                                                   str(pred_category[j][i].detach().cpu().numpy()))
                          for j in range(self.category_num)]
                attention_visualizer.plot_multi_attentions_of_sentence(words, visual_attentions_category,
                                                                       attention_labels, titles)

                # sentiment embedding layer
                # visual_attentions = [embedding_layer_sentiment_alphas[j][i][: len(words)].detach().cpu().numpy()
                #                      for j in range(self.category_num)]
                # titles = ['true: %s - pred: %s - %s' % (str(label[i + self.category_num][j].detach().cpu().numpy()),
                #                                         str(pred_sentiment[j][i].detach().cpu().numpy()),
                #                                         str(self.polarites))
                #           for j in range(self.category_num)]
                # attention_visualizer.plot_multi_attentions_of_sentence(words, visual_attentions, attention_labels,
                #                                                        titles)

                # sentiment lstm layer
                # visual_attentions = [lstm_layer_sentiment_alphas[j][i][: len(words)].detach().cpu().numpy()
                #                      for j in range(self.category_num)]
                # titles = ['true: %s - pred: %s - %s' % (str(label[i + self.category_num][j].detach().cpu().numpy()),
                #                                         str(pred_sentiment[j][i].detach().cpu().numpy()),
                #                                         str(self.polarites))
                #           for j in range(self.category_num)]
                # attention_visualizer.plot_multi_attentions_of_sentence(words, visual_attentions, attention_labels,
                #                                                        titles)

                # sentiment lstm layer
                visual_attentions_sentiment_temp = [lstm_layer_words_sentiment_soft[j][i][: len(words)].detach().cpu().numpy()
                                                    for j in range(self.category_num)]
                for j in range(self.category_num):
                    c_label = label[i][j].detach().cpu().numpy().tolist()
                    if c_label == 1:
                        visual_attentions_sentiment = []
                        labels_sentiment = []
                        sentiment_true_index = int(label[i][j + self.category_num].detach().cpu().numpy().tolist())
                        if sentiment_true_index == -100:
                            continue
                        titles_sentiment = ['true: %s - pred: %s - %s' % (str(self.polarites[sentiment_true_index]),
                                                        str(pred_sentiment[j][i].detach().cpu().numpy()),
                                                        str(self.polarites))]
                        c_attention = embedding_layer_category_alphas[j][i][: len(words)].detach().cpu().numpy()
                        visual_attentions_sentiment.append(c_attention)
                        labels_sentiment.append(self.categories[j].split('/')[0])

                        s_distributions = visual_attentions_sentiment_temp[j]
                        for k in range(self.polarity_num):
                            labels_sentiment.append(self.polarites[k])
                            visual_attentions_sentiment.append(s_distributions[:, k])
                        titles_sentiment.extend([''] * 3)
                        attention_visualizer.plot_multi_attentions_of_sentence(words, visual_attentions_sentiment,
                                                                               labels_sentiment,
                                                                               titles_sentiment)
                        print()
        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {
            'accuracy': self._accuracy.get_metric(reset),
            'category_f1': self._f1.get_metric(reset)['fscore']
        }
        return metrics


class AsMilSimultaneouslyV2(TextInAllAspectSentimentOutModel):
    def __init__(self, word_embedder: TextFieldEmbedder, position_embedder: TextFieldEmbedder,
                 aspect_embedder: TextFieldEmbedder, categories: list, polarities: list, vocab: Vocabulary,
                 configuration: dict, category_loss_weight=1, sentiment_loss_weight=1):
        super().__init__(vocab, category_loss_weight=category_loss_weight, sentiment_loss_weight=sentiment_loss_weight)
        self.configuration = configuration
        self.word_embedder = word_embedder
        self.position_embedder = position_embedder
        self.aspect_embedder = aspect_embedder
        self.categories = categories
        self.polarites = polarities
        self.category_num = len(categories)
        self.polarity_num = len(polarities)
        self.category_loss = nn.BCEWithLogitsLoss()
        self.sentiment_loss = nn.CrossEntropyLoss()
        self._accuracy = metrics.CategoricalAccuracy()
        self._f1 = allennlp_metrics.BinaryF1(0.5)

        word_embedding_dim = word_embedder.get_output_dim()
        if self.configuration['lstm_or_fc_after_embedding_layer'] == 'fc':
            self.embedding_layer_fc = nn.Linear(word_embedding_dim, word_embedding_dim, bias=True)
        else:
            self.embedding_layer_lstm = torch.nn.LSTM(word_embedding_dim, word_embedding_dim, batch_first=True,
                                                      bidirectional=False, num_layers=1)
        # todo 当一句话里没有提到某个aspect category，attention应该关注什么？
        # (1) 类似于bert，添加一个额外的词，attend nothing
        # (2) 关注停用词
        self.embedding_layer_aspect_attentions = [AttentionInHtt(word_embedding_dim,
                                                                 word_embedding_dim)
                                                  for _ in range(self.category_num)]
        self.embedding_layer_aspect_attentions = nn.ModuleList(self.embedding_layer_aspect_attentions)

        lstm_input_size = word_embedding_dim
        if self.configuration['position']:
            lstm_input_size += position_embedder.get_output_dim()
        num_layers = self.configuration['lstm_layer_num_in_lstm']
        self.lstm = torch.nn.LSTM(lstm_input_size, int(word_embedding_dim / 2), batch_first=True,
                                  bidirectional=True, num_layers=num_layers, dropout=0.5)

        self.category_fcs = [nn.Linear(word_embedding_dim, 1) for _ in range(self.category_num)]
        self.category_fcs = nn.ModuleList(self.category_fcs)

        if self.configuration['lstm_layer_category_classifier']:
            self.lstm_category_fcs = [nn.Linear(word_embedding_dim, 1) for _ in range(self.category_num)]
            self.lstm_category_fcs = nn.ModuleList(self.lstm_category_fcs)

        sentiment_fc_input_size = word_embedding_dim
        if not self.configuration['share_sentiment_classifier']:
            self.sentiment_fcs = [nn.Sequential(nn.Linear(sentiment_fc_input_size, sentiment_fc_input_size),
                                                nn.ReLU(),
                                                nn.Linear(sentiment_fc_input_size, self.polarity_num))
                                  for _ in range(self.category_num)]
            self.sentiment_fcs = nn.ModuleList(self.sentiment_fcs)
        else:
            self.sentiment_fc = nn.Sequential(nn.Linear(sentiment_fc_input_size, sentiment_fc_input_size),
                                              nn.ReLU(),
                                              nn.Linear(sentiment_fc_input_size, self.polarity_num))

        self.dropout_after_embedding_layer = nn.Dropout(0.5)
        self.dropout_after_lstm_layer = nn.Dropout(0.5)
        # self.gc1 = DglGraphConvolution(word_embedding_dim, word_embedding_dim, configuration)
        # self.gc2 = DglGraphConvolution(word_embedding_dim, word_embedding_dim, configuration)

    def no_grad_for_acd_parameter(self):
        acd_layers = []
        if self.configuration['lstm_or_fc_after_embedding_layer'] == 'fc':
            acd_layers.append(self.embedding_layer_fc)
        else:
            acd_layers.append(self.embedding_layer_lstm)
        acd_layers.append(self.embedding_layer_aspect_attentions)
        acd_layers.append(self.category_fcs)
        for layer in acd_layers:
            for name, value in layer.named_parameters():
                value.requires_grad = False

    def forward(self, tokens: Dict[str, torch.Tensor], label: torch.Tensor, position: torch.Tensor,
                polarity_mask: torch.Tensor, sample: list, aspects: torch.Tensor=None) -> torch.Tensor:
        mask = get_text_field_mask(tokens)
        word_embeddings = self.word_embedder(tokens)
        if self.configuration['lstm_or_fc_after_embedding_layer'] == 'fc':
            word_embeddings_fc = self.embedding_layer_fc(word_embeddings)
        else:
            word_embeddings_fc, (_, _) = self.embedding_layer_lstm(word_embeddings)

        aspects_separate = [{'aspect': aspects['aspect'][:, i].unsqueeze(1)} for i in range(self.category_num)]
        aspect_embeddings_singles = [self.aspect_embedder(aspects_separate[i]).squeeze(1) for i in
                                     range(self.category_num)]
        aspects_seprate_repeat = [{'aspect': aspects_separate[i]['aspect'].expand_as(tokens['tokens'])} for i in
                                  range(self.category_num)]
        aspect_embeddings_separate = [self.aspect_embedder(aspects_seprate_repeat[i]) for i in range(self.category_num)]

        embedding_layer_category_outputs = []
        embedding_layer_category_alphas = []
        embedding_layer_sentiment_outputs = []
        embedding_layer_sentiment_alphas = []

        embedding_layer_category_alphas = []
        for i in range(self.category_num):
            embedding_layer_aspect_attention = self.embedding_layer_aspect_attentions[i]
            alpha = embedding_layer_aspect_attention(word_embeddings_fc, mask)
            embedding_layer_category_alphas.append(alpha)

        for i in range(self.category_num):
            alpha = embedding_layer_category_alphas[i]
            category_output = self.element_wise_mul(word_embeddings_fc, alpha, return_not_sum_result=False)
            embedding_layer_category_outputs.append(category_output)

        lstm_input = word_embeddings
        if self.configuration['position']:
            position_embeddings = self.position_embedder(position)
            lstm_input = torch.cat([word_embeddings, position_embeddings], dim=-1)
        lstm_input = self.dropout_after_embedding_layer(lstm_input)

        lstm_result, _ = self.lstm(lstm_input)
        lstm_result = self.dropout_after_lstm_layer(lstm_result)
        # lstm_result_with_position = torch.cat([lstm_result, position_embeddings], dim=-1)
        lstm_layer_category_outputs = []
        lstm_layer_sentiment_outputs = []
        lstm_layer_words_sentiment_soft = []

        # max_len = tokens['tokens'].size()[1]
        # graphs = [e[3] for e in sample]
        # graphs_padded = self.pad_dgl_graph(graphs, max_len)
        # graph_output1 = F.relu(self.gc1(word_embeddings, graphs_padded))
        # graph_output2 = F.relu(self.gc2(graph_output1, graphs_padded))
        for i in range(self.category_num):
            alpha = embedding_layer_category_alphas[i]
            category_output = self.element_wise_mul(lstm_result, alpha, return_not_sum_result=False)
            lstm_layer_category_outputs.append(category_output)

            # sentiment
            # word_representation_for_sentiment = torch.cat([graph_output2, lstm_result], dim=-1)
            word_representation_for_sentiment = lstm_result
            sentiment_alpha = embedding_layer_category_alphas[i]
            if self.configuration['mil']:
                sentiment_alpha = sentiment_alpha.unsqueeze(1)
                if not self.configuration['share_sentiment_classifier']:
                    words_sentiment = self.sentiment_fcs[i](word_representation_for_sentiment)
                else:
                    words_sentiment = self.sentiment_fc(word_representation_for_sentiment)
                if self.configuration['mil_softmax']:
                    words_sentiment_soft = torch.softmax(words_sentiment, dim=-1)
                    lstm_layer_words_sentiment_soft.append(words_sentiment_soft)
                else:
                    words_sentiment_soft = words_sentiment
                    lstm_layer_words_sentiment_soft.append(torch.softmax(words_sentiment, dim=-1))
                sentiment_output = torch.matmul(sentiment_alpha, words_sentiment_soft).squeeze(1)  # batch_size x 2*hidden_dim
                lstm_layer_sentiment_outputs.append(sentiment_output)
            else:
                sentiment_output_temp = self.element_wise_mul(word_representation_for_sentiment, sentiment_alpha,
                                                         return_not_sum_result=False)
                if not self.configuration['share_sentiment_classifier']:
                    sentiment_output = self.sentiment_fcs[i](sentiment_output_temp)
                else:
                    sentiment_output = self.sentiment_fc(sentiment_output_temp)
                lstm_layer_sentiment_outputs.append(sentiment_output)

        final_category_outputs = []
        final_lstm_category_outputs = []
        final_sentiment_outputs = []
        for i in range(self.category_num):
            fc = self.category_fcs[i]
            category_output = embedding_layer_category_outputs[i]
            final_category_output = fc(category_output)
            final_category_outputs.append(final_category_output)

            if self.configuration['lstm_layer_category_classifier']:
                fc_lstm = self.lstm_category_fcs[i]
                lstm_category_output = lstm_layer_category_outputs[i]
                final_lstm_category_output = fc_lstm(lstm_category_output)
                final_lstm_category_outputs.append(final_lstm_category_output)

            final_sentiment_output = lstm_layer_sentiment_outputs[i]
            final_sentiment_outputs.append(final_sentiment_output)

        output = {}
        output['alpha'] = embedding_layer_category_alphas
        if label is not None:
            category_labels = []
            polarity_labels = []
            polarity_masks = []
            for i in range(self.category_num):
                category_labels.append(label[:, i])
                polarity_labels.append(label[:, i + self.category_num])
                polarity_masks.append(polarity_mask[:, i])
            loss = 0
            total_category_loss = 0
            total_sentiment_loss = 0
            for i in range(self.category_num):
                category_temp_loss = self.category_loss(final_category_outputs[i].squeeze(dim=-1), category_labels[i])
                sentiment_temp_loss = self.sentiment_loss(final_sentiment_outputs[i], polarity_labels[i].long())
                total_category_loss += category_temp_loss
                if not self.configuration['only_acd']:
                    total_sentiment_loss += sentiment_temp_loss
                if self.configuration['lstm_layer_category_classifier']:
                    lstm_category_temp_loss = self.category_loss(final_lstm_category_outputs[i].squeeze(dim=-1),
                                                                 category_labels[i])
                    total_category_loss += lstm_category_temp_loss
            loss = self.category_loss_weight * total_category_loss + self.sentiment_loss_weight * total_sentiment_loss

            # sentiment accuracy
            sentiment_logit = torch.cat(final_sentiment_outputs)
            sentiment_label = torch.cat(polarity_labels)
            sentiment_mask = torch.cat(polarity_masks)
            self._accuracy(sentiment_logit, sentiment_label, sentiment_mask)

            # category f1
            final_category_outputs_prob = [torch.sigmoid(e) for e in final_category_outputs]
            category_prob = torch.cat(final_category_outputs_prob).squeeze()
            category_label = torch.cat(category_labels)
            self._f1(category_prob, category_label)

            output['loss'] = loss

        # visualize attention
        pred_category = [torch.sigmoid(e) for e in final_category_outputs]
        pred_sentiment = [torch.nn.functional.softmax(e, dim=-1) for e in final_sentiment_outputs]
        output['pred_category'] = pred_category
        output['pred_sentiment'] = pred_sentiment
        if self.configuration['visualize_attention']:
            for i in range(len(sample)):
                words = sample[i][2]
                attention_labels = [e.split('/')[0] for e in self.categories]

                # 只可视化mams
                label_true = label[i].detach().cpu().numpy()[: self.category_num]
                if sum(label_true) <= 1:
                    continue
                # category
                visual_attentions_category = [embedding_layer_category_alphas[j][i][: len(words)].detach().cpu().numpy()
                                              for j in range(self.category_num)]
                titles = ['true: %s - pred: %s' % (str(label[i][j].detach().cpu().numpy()),
                                                   str(pred_category[j][i].detach().cpu().numpy()))
                          for j in range(self.category_num)]
                attention_visualizer.plot_multi_attentions_of_sentence(words, visual_attentions_category,
                                                                       attention_labels, titles)

                # sentiment embedding layer
                # visual_attentions = [embedding_layer_sentiment_alphas[j][i][: len(words)].detach().cpu().numpy()
                #                      for j in range(self.category_num)]
                # titles = ['true: %s - pred: %s - %s' % (str(label[i + self.category_num][j].detach().cpu().numpy()),
                #                                         str(pred_sentiment[j][i].detach().cpu().numpy()),
                #                                         str(self.polarites))
                #           for j in range(self.category_num)]
                # attention_visualizer.plot_multi_attentions_of_sentence(words, visual_attentions, attention_labels,
                #                                                        titles)

                # sentiment lstm layer
                # visual_attentions = [lstm_layer_sentiment_alphas[j][i][: len(words)].detach().cpu().numpy()
                #                      for j in range(self.category_num)]
                # titles = ['true: %s - pred: %s - %s' % (str(label[i + self.category_num][j].detach().cpu().numpy()),
                #                                         str(pred_sentiment[j][i].detach().cpu().numpy()),
                #                                         str(self.polarites))
                #           for j in range(self.category_num)]
                # attention_visualizer.plot_multi_attentions_of_sentence(words, visual_attentions, attention_labels,
                #                                                        titles)

                # sentiment lstm layer
                visual_attentions_sentiment_temp = [lstm_layer_words_sentiment_soft[j][i][: len(words)].detach().cpu().numpy()
                                                    for j in range(self.category_num)]
                for j in range(self.category_num):
                    c_label = label[i][j].detach().cpu().numpy().tolist()
                    if c_label == 1:
                        visual_attentions_sentiment = []
                        labels_sentiment = []
                        sentiment_true_index = int(label[i][j + self.category_num].detach().cpu().numpy().tolist())
                        if sentiment_true_index == -100:
                            continue
                        titles_sentiment = ['true: %s - pred: %s - %s' % (str(self.polarites[sentiment_true_index]),
                                                        str(pred_sentiment[j][i].detach().cpu().numpy()),
                                                        str(self.polarites))]
                        c_attention = embedding_layer_category_alphas[j][i][: len(words)].detach().cpu().numpy()
                        visual_attentions_sentiment.append(c_attention)
                        labels_sentiment.append(self.categories[j].split('/')[0])

                        s_distributions = visual_attentions_sentiment_temp[j]
                        for k in range(self.polarity_num):
                            labels_sentiment.append(self.polarites[k])
                            visual_attentions_sentiment.append(s_distributions[:, k])
                        titles_sentiment.extend([''] * 3)
                        attention_visualizer.plot_multi_attentions_of_sentence(words, visual_attentions_sentiment,
                                                                               labels_sentiment,
                                                                               titles_sentiment)
                        print()
        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {
            'accuracy': self._accuracy.get_metric(reset),
            'category_f1': self._f1.get_metric(reset)['fscore']
        }
        return metrics


class AsMilSimultaneouslyV3(TextInAllAspectSentimentOutModel):
    def __init__(self, word_embedder: TextFieldEmbedder, position_embedder: TextFieldEmbedder,
                 aspect_embedder: TextFieldEmbedder, categories: list, polarities: list, vocab: Vocabulary,
                 configuration: dict, category_loss_weight=1, sentiment_loss_weight=1):
        super().__init__(vocab, category_loss_weight=category_loss_weight, sentiment_loss_weight=sentiment_loss_weight)
        self.configuration = configuration
        self.word_embedder = word_embedder
        self.position_embedder = position_embedder
        self.aspect_embedder = aspect_embedder
        self.categories = categories
        self.polarites = polarities
        self.category_num = len(categories)
        self.polarity_num = len(polarities)
        self.category_loss = nn.BCEWithLogitsLoss()
        self.sentiment_loss = nn.CrossEntropyLoss()
        self._accuracy = metrics.CategoricalAccuracy()
        self._f1 = allennlp_metrics.BinaryF1(0.5)

        word_embedding_dim = word_embedder.get_output_dim()
        if self.configuration['lstm_or_fc_after_embedding_layer'] == 'fc':
            self.embedding_layer_fc = nn.Linear(word_embedding_dim, word_embedding_dim, bias=True)
        else:
            self.embedding_layer_lstm = torch.nn.LSTM(word_embedding_dim, word_embedding_dim, batch_first=True,
                                                      bidirectional=False, num_layers=1)
        # todo 当一句话里没有提到某个aspect category，attention应该关注什么？
        # (1) 类似于bert，添加一个额外的词，attend nothing
        # (2) 关注停用词
        self.embedding_layer_aspect_attentions = [AttentionInHtt(word_embedding_dim,
                                                                 word_embedding_dim)
                                                  for _ in range(self.category_num)]
        self.embedding_layer_aspect_attentions = nn.ModuleList(self.embedding_layer_aspect_attentions)

        lstm_input_size = word_embedding_dim
        if self.configuration['position']:
            lstm_input_size += position_embedder.get_output_dim()
        num_layers = self.configuration['lstm_layer_num_in_lstm']
        self.lstm = torch.nn.LSTM(lstm_input_size, int(word_embedding_dim / 2), batch_first=True,
                                  bidirectional=True, num_layers=num_layers, dropout=0.5)

        self.category_fcs = [nn.Linear(word_embedding_dim, 1) for _ in range(self.category_num)]
        self.category_fcs = nn.ModuleList(self.category_fcs)

        if self.configuration['lstm_layer_category_classifier']:
            self.lstm_category_fcs = [nn.Linear(word_embedding_dim, 1) for _ in range(self.category_num)]
            self.lstm_category_fcs = nn.ModuleList(self.lstm_category_fcs)

        sentiment_fc_input_size = word_embedding_dim
        if not self.configuration['share_sentiment_classifier']:
            self.sentiment_fcs = [nn.Sequential(nn.Linear(sentiment_fc_input_size, sentiment_fc_input_size),
                                                nn.ReLU(),
                                                nn.Linear(sentiment_fc_input_size, self.polarity_num))
                                  for _ in range(self.category_num)]
            self.sentiment_fcs = nn.ModuleList(self.sentiment_fcs)
        else:
            self.sentiment_fc = nn.Sequential(nn.Linear(sentiment_fc_input_size, sentiment_fc_input_size),
                                              nn.ReLU(),
                                              nn.Linear(sentiment_fc_input_size, self.polarity_num))

        self.dropout_after_embedding_layer = nn.Dropout(0.5)
        self.dropout_after_lstm_layer = nn.Dropout(0.5)
        # self.gc1 = DglGraphConvolution(word_embedding_dim, word_embedding_dim, configuration)
        # self.gc2 = DglGraphConvolution(word_embedding_dim, word_embedding_dim, configuration)

    def no_grad_for_acd_parameter(self):
        acd_layers = []
        if self.configuration['lstm_or_fc_after_embedding_layer'] == 'fc':
            acd_layers.append(self.embedding_layer_fc)
        else:
            acd_layers.append(self.embedding_layer_lstm)
        acd_layers.append(self.embedding_layer_aspect_attentions)
        acd_layers.append(self.category_fcs)
        for layer in acd_layers:
            for name, value in layer.named_parameters():
                value.requires_grad = False

    def forward(self, tokens: Dict[str, torch.Tensor], label: torch.Tensor, position: torch.Tensor,
                polarity_mask: torch.Tensor, sample: list, aspects: torch.Tensor=None) -> torch.Tensor:
        mask = get_text_field_mask(tokens)
        word_embeddings = self.word_embedder(tokens)
        if self.configuration['lstm_or_fc_after_embedding_layer'] == 'fc':
            word_embeddings_fc = self.embedding_layer_fc(word_embeddings)
        else:
            word_embeddings_fc, (_, _) = self.embedding_layer_lstm(word_embeddings)

        aspects_separate = [{'aspect': aspects['aspect'][:, i].unsqueeze(1)} for i in range(self.category_num)]
        aspect_embeddings_singles = [self.aspect_embedder(aspects_separate[i]).squeeze(1) for i in
                                     range(self.category_num)]
        aspects_seprate_repeat = [{'aspect': aspects_separate[i]['aspect'].expand_as(tokens['tokens'])} for i in
                                  range(self.category_num)]
        aspect_embeddings_separate = [self.aspect_embedder(aspects_seprate_repeat[i]) for i in range(self.category_num)]

        embedding_layer_category_outputs = []
        embedding_layer_category_alphas = []
        embedding_layer_sentiment_outputs = []
        embedding_layer_sentiment_alphas = []

        embedding_layer_category_alphas = []
        for i in range(self.category_num):
            embedding_layer_aspect_attention = self.embedding_layer_aspect_attentions[i]
            alpha = embedding_layer_aspect_attention(word_embeddings_fc, mask)
            embedding_layer_category_alphas.append(alpha)

        for i in range(self.category_num):
            alpha = embedding_layer_category_alphas[i]
            category_output = self.element_wise_mul(word_embeddings_fc, alpha, return_not_sum_result=False)
            embedding_layer_category_outputs.append(category_output)

        lstm_input = word_embeddings
        if self.configuration['position']:
            position_embeddings = self.position_embedder(position)
            lstm_input = torch.cat([word_embeddings, position_embeddings], dim=-1)
        lstm_input = self.dropout_after_embedding_layer(lstm_input)

        lstm_result, _ = self.lstm(lstm_input)
        lstm_result = self.dropout_after_lstm_layer(lstm_result)
        # lstm_result_with_position = torch.cat([lstm_result, position_embeddings], dim=-1)
        lstm_layer_category_outputs = []
        lstm_layer_sentiment_outputs = []
        lstm_layer_words_sentiment_soft = []

        # max_len = tokens['tokens'].size()[1]
        # graphs = [e[3] for e in sample]
        # graphs_padded = self.pad_dgl_graph(graphs, max_len)
        # graph_output1 = F.relu(self.gc1(word_embeddings, graphs_padded))
        # graph_output2 = F.relu(self.gc2(graph_output1, graphs_padded))
        for i in range(self.category_num):
            alpha = embedding_layer_category_alphas[i]
            category_output = self.element_wise_mul(lstm_result, alpha, return_not_sum_result=False)
            lstm_layer_category_outputs.append(category_output)

            # sentiment
            # word_representation_for_sentiment = torch.cat([graph_output2, lstm_result], dim=-1)
            word_representation_for_sentiment = lstm_result
            sentiment_alpha = embedding_layer_category_alphas[i]
            if self.configuration['mil']:
                sentiment_alpha = sentiment_alpha.unsqueeze(1)
                if not self.configuration['share_sentiment_classifier']:
                    words_sentiment = self.sentiment_fcs[i](word_representation_for_sentiment)
                else:
                    words_sentiment = self.sentiment_fc(word_representation_for_sentiment)
                if self.configuration['mil_softmax']:
                    words_sentiment_soft = torch.softmax(words_sentiment, dim=-1)
                    lstm_layer_words_sentiment_soft.append(words_sentiment_soft)
                else:
                    words_sentiment_soft = words_sentiment
                    lstm_layer_words_sentiment_soft.append(torch.softmax(words_sentiment, dim=-1))
                sentiment_output = torch.matmul(sentiment_alpha, words_sentiment_soft).squeeze(1)  # batch_size x 2*hidden_dim
                lstm_layer_sentiment_outputs.append(sentiment_output)
            else:
                sentiment_output_temp = self.element_wise_mul(word_representation_for_sentiment, sentiment_alpha,
                                                         return_not_sum_result=False)
                if not self.configuration['share_sentiment_classifier']:
                    sentiment_output = self.sentiment_fcs[i](sentiment_output_temp)
                else:
                    sentiment_output = self.sentiment_fc(sentiment_output_temp)
                lstm_layer_sentiment_outputs.append(sentiment_output)

        final_category_outputs = []
        final_lstm_category_outputs = []
        final_sentiment_outputs = []
        for i in range(self.category_num):
            fc = self.category_fcs[i]
            category_output = embedding_layer_category_outputs[i]
            final_category_output = fc(category_output)
            final_category_outputs.append(final_category_output)

            if self.configuration['lstm_layer_category_classifier']:
                fc_lstm = self.lstm_category_fcs[i]
                lstm_category_output = lstm_layer_category_outputs[i]
                final_lstm_category_output = fc_lstm(lstm_category_output)
                final_lstm_category_outputs.append(final_lstm_category_output)

            final_sentiment_output = lstm_layer_sentiment_outputs[i]
            final_sentiment_outputs.append(final_sentiment_output)

        output = {}
        output['alpha'] = embedding_layer_category_alphas
        if label is not None:
            category_labels = []
            polarity_labels = []
            polarity_masks = []
            for i in range(self.category_num):
                category_labels.append(label[:, i])
                polarity_labels.append(label[:, i + self.category_num])
                polarity_masks.append(polarity_mask[:, i])
            loss = 0
            total_category_loss = 0
            total_sentiment_loss = 0
            for i in range(self.category_num):
                category_temp_loss = self.category_loss(final_category_outputs[i].squeeze(dim=-1), category_labels[i])
                sentiment_temp_loss = self.sentiment_loss(final_sentiment_outputs[i], polarity_labels[i].long())
                total_category_loss += category_temp_loss
                if not self.configuration['only_acd']:
                    total_sentiment_loss += sentiment_temp_loss
                if self.configuration['lstm_layer_category_classifier']:
                    lstm_category_temp_loss = self.category_loss(final_lstm_category_outputs[i].squeeze(dim=-1),
                                                                 category_labels[i])
                    total_category_loss += lstm_category_temp_loss
            loss = self.category_loss_weight * total_category_loss + self.sentiment_loss_weight * total_sentiment_loss

            # Sparse Regularization Orthogonal Regularization
            if self.configuration['sparse_reg'] or self.configuration['orthogonal_reg']:
                reg_loss = 0
                for j in range(len(sample)):
                    polarity_mask_of_one_sample = polarity_mask[j]
                    category_alpha_of_one_sample = [embedding_layer_category_alphas[k][j] for k in range(self.category_num)]
                    category_alpha_of_mentioned = []
                    category_alpha_of_not_mentioned = []
                    for k in range(self.category_num):
                        if polarity_mask_of_one_sample[k] == 1:
                            category_alpha_of_mentioned.append(category_alpha_of_one_sample[k].unsqueeze(0))
                        else:
                            category_alpha_of_not_mentioned.append(category_alpha_of_one_sample[k].unsqueeze(0))
                    if len(category_alpha_of_not_mentioned) != 0:
                        category_alpha_of_not_mentioned = torch.cat(category_alpha_of_not_mentioned, dim=0)
                        category_alpha_of_not_mentioned = torch.mean(category_alpha_of_not_mentioned, dim=0, keepdim=True)
                        category_alpha_of_mentioned.append(category_alpha_of_not_mentioned)

                    category_eye = torch.eye(len(category_alpha_of_mentioned))
                    category_alpha_of_mentioned = torch.cat(category_alpha_of_mentioned, dim=0)
                    category_alpha_similarity = torch.mm(category_alpha_of_mentioned, category_alpha_of_mentioned.t())

                    if self.configuration['sparse_reg'] and self.configuration['orthogonal_reg']:
                        pass
                    elif self.configuration['sparse_reg']:
                        for m in range(len(category_alpha_of_mentioned)):
                            for n in range(len(category_alpha_of_mentioned)):
                                if m != n:
                                    category_eye[m][n] = category_alpha_similarity[m][n]
                    else:
                        # orthogonal_reg
                        for m in range(len(category_alpha_of_mentioned)):
                            category_eye[m][m] = category_alpha_similarity[m][m]
                    category_reg_loss = category_alpha_similarity - category_eye
                    category_reg_loss = torch.norm(category_reg_loss)
                    reg_loss += category_reg_loss
                loss += (reg_loss * self.configuration['attention_lamda'] / len(sample))

            # sentiment accuracy
            sentiment_logit = torch.cat(final_sentiment_outputs)
            sentiment_label = torch.cat(polarity_labels)
            sentiment_mask = torch.cat(polarity_masks)
            self._accuracy(sentiment_logit, sentiment_label, sentiment_mask)

            # category f1
            final_category_outputs_prob = [torch.sigmoid(e) for e in final_category_outputs]
            category_prob = torch.cat(final_category_outputs_prob).squeeze()
            category_label = torch.cat(category_labels)
            self._f1(category_prob, category_label)

            output['loss'] = loss

        # visualize attention
        pred_category = [torch.sigmoid(e) for e in final_category_outputs]
        pred_sentiment = [torch.nn.functional.softmax(e, dim=-1) for e in final_sentiment_outputs]
        output['pred_category'] = pred_category
        output['pred_sentiment'] = pred_sentiment
        if self.configuration['visualize_attention']:
            for i in range(len(sample)):
                words = sample[i][2]
                attention_labels = [e.split('/')[0] for e in self.categories]

                # 只可视化mams
                label_true = label[i].detach().cpu().numpy()[: self.category_num]
                if sum(label_true) <= 1:
                    continue
                # category
                visual_attentions_category = [embedding_layer_category_alphas[j][i][: len(words)].detach().cpu().numpy()
                                              for j in range(self.category_num)]
                titles = ['true: %s - pred: %s' % (str(label[i][j].detach().cpu().numpy()),
                                                   str(pred_category[j][i].detach().cpu().numpy()))
                          for j in range(self.category_num)]
                attention_visualizer.plot_multi_attentions_of_sentence(words, visual_attentions_category,
                                                                       attention_labels, titles)

                # sentiment embedding layer
                # visual_attentions = [embedding_layer_sentiment_alphas[j][i][: len(words)].detach().cpu().numpy()
                #                      for j in range(self.category_num)]
                # titles = ['true: %s - pred: %s - %s' % (str(label[i + self.category_num][j].detach().cpu().numpy()),
                #                                         str(pred_sentiment[j][i].detach().cpu().numpy()),
                #                                         str(self.polarites))
                #           for j in range(self.category_num)]
                # attention_visualizer.plot_multi_attentions_of_sentence(words, visual_attentions, attention_labels,
                #                                                        titles)

                # sentiment lstm layer
                # visual_attentions = [lstm_layer_sentiment_alphas[j][i][: len(words)].detach().cpu().numpy()
                #                      for j in range(self.category_num)]
                # titles = ['true: %s - pred: %s - %s' % (str(label[i + self.category_num][j].detach().cpu().numpy()),
                #                                         str(pred_sentiment[j][i].detach().cpu().numpy()),
                #                                         str(self.polarites))
                #           for j in range(self.category_num)]
                # attention_visualizer.plot_multi_attentions_of_sentence(words, visual_attentions, attention_labels,
                #                                                        titles)

                # sentiment lstm layer
                visual_attentions_sentiment_temp = [lstm_layer_words_sentiment_soft[j][i][: len(words)].detach().cpu().numpy()
                                                    for j in range(self.category_num)]
                for j in range(self.category_num):
                    c_label = label[i][j].detach().cpu().numpy().tolist()
                    if c_label == 1:
                        visual_attentions_sentiment = []
                        labels_sentiment = []
                        sentiment_true_index = int(label[i][j + self.category_num].detach().cpu().numpy().tolist())
                        if sentiment_true_index == -100:
                            continue
                        titles_sentiment = ['true: %s - pred: %s - %s' % (str(self.polarites[sentiment_true_index]),
                                                        str(pred_sentiment[j][i].detach().cpu().numpy()),
                                                        str(self.polarites))]
                        c_attention = embedding_layer_category_alphas[j][i][: len(words)].detach().cpu().numpy()
                        visual_attentions_sentiment.append(c_attention)
                        labels_sentiment.append(self.categories[j].split('/')[0])

                        s_distributions = visual_attentions_sentiment_temp[j]
                        for k in range(self.polarity_num):
                            labels_sentiment.append(self.polarites[k])
                            visual_attentions_sentiment.append(s_distributions[:, k])
                        titles_sentiment.extend([''] * 3)
                        attention_visualizer.plot_multi_attentions_of_sentence(words, visual_attentions_sentiment,
                                                                               labels_sentiment,
                                                                               titles_sentiment)
                        print()
        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {
            'accuracy': self._accuracy.get_metric(reset),
            'category_f1': self._f1.get_metric(reset)['fscore']
        }
        return metrics


class AsMilSimultaneouslyV4(TextInAllAspectSentimentOutModel):
    def __init__(self, word_embedder: TextFieldEmbedder, position_embedder: TextFieldEmbedder,
                 aspect_embedder: TextFieldEmbedder, categories: list, polarities: list, vocab: Vocabulary,
                 configuration: dict, category_loss_weight=1, sentiment_loss_weight=1):
        super().__init__(vocab, category_loss_weight=category_loss_weight, sentiment_loss_weight=sentiment_loss_weight)
        self.configuration = configuration
        self.word_embedder = word_embedder
        self.position_embedder = position_embedder
        self.aspect_embedder = aspect_embedder
        self.categories = categories
        self.polarites = polarities
        self.category_num = len(categories)
        self.polarity_num = len(polarities)
        self.category_loss = nn.BCEWithLogitsLoss()
        self.sentiment_loss = nn.CrossEntropyLoss()
        self._accuracy = metrics.CategoricalAccuracy()
        self._f1 = allennlp_metrics.BinaryF1(0.5)

        word_embedding_dim = word_embedder.get_output_dim()
        if self.configuration['lstm_or_fc_after_embedding_layer'] == 'fc':
            self.embedding_layer_fc = nn.Linear(word_embedding_dim, word_embedding_dim, bias=True)
        elif self.configuration['lstm_or_fc_after_embedding_layer'] == 'gcn':
            self.gc_aspect_category = DglGraphConvolution(word_embedding_dim, word_embedding_dim, configuration)
        elif self.configuration['lstm_or_fc_after_embedding_layer'] == 'bilstm':
            self.embedding_layer_lstm = torch.nn.LSTM(word_embedding_dim, int(word_embedding_dim / 2), batch_first=True,
                                                      bidirectional=True, num_layers=1)
        else:
            self.embedding_layer_lstm = torch.nn.LSTM(word_embedding_dim, word_embedding_dim, batch_first=True,
                                                      bidirectional=False, num_layers=1)
        # todo 当一句话里没有提到某个aspect category，attention应该关注什么？
        # (1) 类似于bert，添加一个额外的词，attend nothing
        # (2) 关注停用词
        self.embedding_layer_aspect_attentions = [AttentionInHtt(word_embedding_dim,
                                                                 word_embedding_dim)
                                                  for _ in range(self.category_num)]
        self.embedding_layer_aspect_attentions = nn.ModuleList(self.embedding_layer_aspect_attentions)

        lstm_input_size = word_embedding_dim
        if self.configuration['position']:
            lstm_input_size += position_embedder.get_output_dim()
        num_layers = self.configuration['lstm_layer_num_in_lstm']
        self.lstm = torch.nn.LSTM(lstm_input_size, int(word_embedding_dim / 2), batch_first=True,
                                  bidirectional=True, num_layers=num_layers, dropout=0.5)

        self.category_fcs = [nn.Linear(word_embedding_dim, 1) for _ in range(self.category_num)]
        self.category_fcs = nn.ModuleList(self.category_fcs)

        if self.configuration['lstm_layer_category_classifier']:
            self.lstm_category_fcs = [nn.Linear(word_embedding_dim, 1) for _ in range(self.category_num)]
            self.lstm_category_fcs = nn.ModuleList(self.lstm_category_fcs)

        sentiment_fc_input_size = word_embedding_dim
        if not self.configuration['share_sentiment_classifier']:
            self.sentiment_fcs = [nn.Sequential(nn.Linear(sentiment_fc_input_size, sentiment_fc_input_size),
                                                nn.ReLU(),
                                                nn.Linear(sentiment_fc_input_size, self.polarity_num))
                                  for _ in range(self.category_num)]
            self.sentiment_fcs = nn.ModuleList(self.sentiment_fcs)
        else:
            self.sentiment_fc = nn.Sequential(nn.Linear(sentiment_fc_input_size, sentiment_fc_input_size),
                                              nn.ReLU(),
                                              nn.Linear(sentiment_fc_input_size, self.polarity_num))

        self.dropout_after_embedding_layer = nn.Dropout(0.5)
        self.dropout_after_lstm_layer = nn.Dropout(0.5)
        # self.gc1 = DglGraphConvolution(word_embedding_dim, word_embedding_dim, configuration)
        # self.gc2 = DglGraphConvolution(word_embedding_dim, word_embedding_dim, configuration)

    def no_grad_for_acd_parameter(self):
        acd_layers = []
        if self.configuration['lstm_or_fc_after_embedding_layer'] == 'fc':
            acd_layers.append(self.embedding_layer_fc)
        else:
            acd_layers.append(self.embedding_layer_lstm)
        acd_layers.append(self.embedding_layer_aspect_attentions)
        acd_layers.append(self.category_fcs)
        for layer in acd_layers:
            for name, value in layer.named_parameters():
                value.requires_grad = False

    def forward(self, tokens: Dict[str, torch.Tensor], label: torch.Tensor, position: torch.Tensor,
                polarity_mask: torch.Tensor, sample: list, aspects: torch.Tensor=None) -> torch.Tensor:
        mask = get_text_field_mask(tokens)
        word_embeddings = self.word_embedder(tokens)
        if self.configuration['lstm_or_fc_after_embedding_layer'] == 'fc':
            word_embeddings_fc = self.embedding_layer_fc(word_embeddings)
        elif self.configuration['lstm_or_fc_after_embedding_layer'] == 'gcn':
            max_len = tokens['tokens'].size()[1]
            graphs = [e[3] for e in sample]
            graphs_padded = self.pad_dgl_graph(graphs, max_len)
            word_embeddings_fc = F.relu(self.gc_aspect_category(word_embeddings, graphs_padded))
        else:
            word_embeddings_fc, (_, _) = self.embedding_layer_lstm(word_embeddings)

        aspects_separate = [{'aspect': aspects['aspect'][:, i].unsqueeze(1)} for i in range(self.category_num)]
        aspect_embeddings_singles = [self.aspect_embedder(aspects_separate[i]).squeeze(1) for i in
                                     range(self.category_num)]
        aspects_seprate_repeat = [{'aspect': aspects_separate[i]['aspect'].expand_as(tokens['tokens'])} for i in
                                  range(self.category_num)]
        aspect_embeddings_separate = [self.aspect_embedder(aspects_seprate_repeat[i]) for i in range(self.category_num)]

        embedding_layer_category_outputs = []
        embedding_layer_category_alphas = []
        embedding_layer_sentiment_outputs = []
        embedding_layer_sentiment_alphas = []

        embedding_layer_category_alphas = []
        for i in range(self.category_num):
            embedding_layer_aspect_attention = self.embedding_layer_aspect_attentions[i]
            alpha = embedding_layer_aspect_attention(word_embeddings_fc, mask)
            embedding_layer_category_alphas.append(alpha)

        for i in range(self.category_num):
            alpha = embedding_layer_category_alphas[i]
            category_output = self.element_wise_mul(word_embeddings_fc, alpha, return_not_sum_result=False)
            embedding_layer_category_outputs.append(category_output)

        lstm_input = word_embeddings
        if self.configuration['position']:
            position_embeddings = self.position_embedder(position)
            lstm_input = torch.cat([word_embeddings, position_embeddings], dim=-1)
        lstm_input = self.dropout_after_embedding_layer(lstm_input)

        lstm_result, _ = self.lstm(lstm_input)
        lstm_result = self.dropout_after_lstm_layer(lstm_result)
        # lstm_result_with_position = torch.cat([lstm_result, position_embeddings], dim=-1)
        lstm_layer_category_outputs = []
        lstm_layer_sentiment_outputs = []
        lstm_layer_words_sentiment_soft = []

        # max_len = tokens['tokens'].size()[1]
        # graphs = [e[3] for e in sample]
        # graphs_padded = self.pad_dgl_graph(graphs, max_len)
        # graph_output1 = F.relu(self.gc1(word_embeddings, graphs_padded))
        # graph_output2 = F.relu(self.gc2(graph_output1, graphs_padded))
        for i in range(self.category_num):
            alpha = embedding_layer_category_alphas[i]
            category_output = self.element_wise_mul(lstm_result, alpha, return_not_sum_result=False)
            lstm_layer_category_outputs.append(category_output)

            # sentiment
            # word_representation_for_sentiment = torch.cat([graph_output2, lstm_result], dim=-1)
            word_representation_for_sentiment = lstm_result
            sentiment_alpha = embedding_layer_category_alphas[i]
            if self.configuration['mil']:
                sentiment_alpha = sentiment_alpha.unsqueeze(1)
                if not self.configuration['share_sentiment_classifier']:
                    words_sentiment = self.sentiment_fcs[i](word_representation_for_sentiment)
                else:
                    words_sentiment = self.sentiment_fc(word_representation_for_sentiment)
                if self.configuration['mil_softmax']:
                    words_sentiment_soft = torch.softmax(words_sentiment, dim=-1)
                    lstm_layer_words_sentiment_soft.append(words_sentiment_soft)
                else:
                    words_sentiment_soft = words_sentiment
                    lstm_layer_words_sentiment_soft.append(torch.softmax(words_sentiment, dim=-1))
                sentiment_output = torch.matmul(sentiment_alpha, words_sentiment_soft).squeeze(1)  # batch_size x 2*hidden_dim
                lstm_layer_sentiment_outputs.append(sentiment_output)
            else:
                sentiment_output_temp = self.element_wise_mul(word_representation_for_sentiment, sentiment_alpha,
                                                         return_not_sum_result=False)
                if not self.configuration['share_sentiment_classifier']:
                    sentiment_output = self.sentiment_fcs[i](sentiment_output_temp)
                else:
                    sentiment_output = self.sentiment_fc(sentiment_output_temp)
                lstm_layer_sentiment_outputs.append(sentiment_output)

        final_category_outputs = []
        final_lstm_category_outputs = []
        final_sentiment_outputs = []
        for i in range(self.category_num):
            fc = self.category_fcs[i]
            category_output = embedding_layer_category_outputs[i]
            final_category_output = fc(category_output)
            final_category_outputs.append(final_category_output)

            if self.configuration['lstm_layer_category_classifier']:
                fc_lstm = self.lstm_category_fcs[i]
                lstm_category_output = lstm_layer_category_outputs[i]
                final_lstm_category_output = fc_lstm(lstm_category_output)
                final_lstm_category_outputs.append(final_lstm_category_output)

            final_sentiment_output = lstm_layer_sentiment_outputs[i]
            final_sentiment_outputs.append(final_sentiment_output)

        output = {}
        output['alpha'] = embedding_layer_category_alphas
        if label is not None:
            category_labels = []
            polarity_labels = []
            polarity_masks = []
            for i in range(self.category_num):
                category_labels.append(label[:, i])
                polarity_labels.append(label[:, i + self.category_num])
                polarity_masks.append(polarity_mask[:, i])
            loss = 0
            total_category_loss = 0
            total_sentiment_loss = 0
            for i in range(self.category_num):
                category_temp_loss = self.category_loss(final_category_outputs[i].squeeze(dim=-1), category_labels[i])
                sentiment_temp_loss = self.sentiment_loss(final_sentiment_outputs[i], polarity_labels[i].long())
                total_category_loss += category_temp_loss
                if not self.configuration['only_acd']:
                    total_sentiment_loss += sentiment_temp_loss
                if self.configuration['lstm_layer_category_classifier']:
                    lstm_category_temp_loss = self.category_loss(final_lstm_category_outputs[i].squeeze(dim=-1),
                                                                 category_labels[i])
                    total_category_loss += lstm_category_temp_loss
            loss = self.category_loss_weight * total_category_loss + self.sentiment_loss_weight * total_sentiment_loss

            # Sparse Regularization Orthogonal Regularization
            if self.configuration['sparse_reg'] or self.configuration['orthogonal_reg']:
                reg_loss = 0
                for j in range(len(sample)):
                    polarity_mask_of_one_sample = polarity_mask[j]
                    category_alpha_of_one_sample = [embedding_layer_category_alphas[k][j] for k in range(self.category_num)]
                    category_alpha_of_mentioned = []
                    category_alpha_of_not_mentioned = []
                    for k in range(self.category_num):
                        if polarity_mask_of_one_sample[k] == 1:
                            category_alpha_of_mentioned.append(category_alpha_of_one_sample[k].unsqueeze(0))
                        else:
                            category_alpha_of_not_mentioned.append(category_alpha_of_one_sample[k].unsqueeze(0))
                    if len(category_alpha_of_not_mentioned) != 0:
                        category_alpha_of_not_mentioned = torch.cat(category_alpha_of_not_mentioned, dim=0)
                        category_alpha_of_not_mentioned = torch.mean(category_alpha_of_not_mentioned, dim=0, keepdim=True)
                        category_alpha_of_mentioned.append(category_alpha_of_not_mentioned)

                    category_eye = torch.eye(len(category_alpha_of_mentioned))
                    category_alpha_of_mentioned = torch.cat(category_alpha_of_mentioned, dim=0)
                    category_alpha_similarity = torch.mm(category_alpha_of_mentioned, category_alpha_of_mentioned.t())

                    if self.configuration['sparse_reg'] and self.configuration['orthogonal_reg']:
                        pass
                    elif self.configuration['sparse_reg']:
                        for m in range(len(category_alpha_of_mentioned)):
                            for n in range(len(category_alpha_of_mentioned)):
                                if m != n:
                                    category_eye[m][n] = category_alpha_similarity[m][n]
                    else:
                        # orthogonal_reg
                        for m in range(len(category_alpha_of_mentioned)):
                            category_eye[m][m] = category_alpha_similarity[m][m]
                    # category_eye = nn_util.move_to_device(category_eye, self.configuration['device'])
                    category_eye = category_eye.to(self.configuration['device'])
                    category_alpha_similarity = category_alpha_similarity.to(self.configuration['device'])
                    category_reg_loss = category_alpha_similarity - category_eye
                    category_reg_loss = torch.norm(category_reg_loss)
                    reg_loss += category_reg_loss
                loss += (reg_loss * self.configuration['attention_lamda'] / len(sample))

            # sentiment accuracy
            sentiment_logit = torch.cat(final_sentiment_outputs)
            sentiment_label = torch.cat(polarity_labels)
            sentiment_mask = torch.cat(polarity_masks)
            self._accuracy(sentiment_logit, sentiment_label, sentiment_mask)

            # category f1
            final_category_outputs_prob = [torch.sigmoid(e) for e in final_category_outputs]
            category_prob = torch.cat(final_category_outputs_prob).squeeze()
            category_label = torch.cat(category_labels)
            self._f1(category_prob, category_label)

            output['loss'] = loss

        # visualize attention
        pred_category = [torch.sigmoid(e) for e in final_category_outputs]
        pred_sentiment = [torch.nn.functional.softmax(e, dim=-1) for e in final_sentiment_outputs]
        output['pred_category'] = pred_category
        output['pred_sentiment'] = pred_sentiment
        if self.configuration['visualize_attention']:
            for i in range(len(sample)):
                words = sample[i][2]
                attention_labels = [e.split('/')[0] for e in self.categories]

                # 只可视化mams
                label_true = label[i].detach().cpu().numpy()[: self.category_num]
                if sum(label_true) <= 1:
                    continue
                # category
                visual_attentions_category = [embedding_layer_category_alphas[j][i][: len(words)].detach().cpu().numpy()
                                              for j in range(self.category_num)]
                titles = ['true: %s - pred: %s' % (str(label[i][j].detach().cpu().numpy()),
                                                   str(pred_category[j][i].detach().cpu().numpy()))
                          for j in range(self.category_num)]
                attention_visualizer.plot_multi_attentions_of_sentence(words, visual_attentions_category,
                                                                       attention_labels, titles)

                # sentiment embedding layer
                # visual_attentions = [embedding_layer_sentiment_alphas[j][i][: len(words)].detach().cpu().numpy()
                #                      for j in range(self.category_num)]
                # titles = ['true: %s - pred: %s - %s' % (str(label[i + self.category_num][j].detach().cpu().numpy()),
                #                                         str(pred_sentiment[j][i].detach().cpu().numpy()),
                #                                         str(self.polarites))
                #           for j in range(self.category_num)]
                # attention_visualizer.plot_multi_attentions_of_sentence(words, visual_attentions, attention_labels,
                #                                                        titles)

                # sentiment lstm layer
                # visual_attentions = [lstm_layer_sentiment_alphas[j][i][: len(words)].detach().cpu().numpy()
                #                      for j in range(self.category_num)]
                # titles = ['true: %s - pred: %s - %s' % (str(label[i + self.category_num][j].detach().cpu().numpy()),
                #                                         str(pred_sentiment[j][i].detach().cpu().numpy()),
                #                                         str(self.polarites))
                #           for j in range(self.category_num)]
                # attention_visualizer.plot_multi_attentions_of_sentence(words, visual_attentions, attention_labels,
                #                                                        titles)

                # sentiment lstm layer
                visual_attentions_sentiment_temp = [lstm_layer_words_sentiment_soft[j][i][: len(words)].detach().cpu().numpy()
                                                    for j in range(self.category_num)]
                for j in range(self.category_num):
                    c_label = label[i][j].detach().cpu().numpy().tolist()
                    if c_label == 1:
                        visual_attentions_sentiment = []
                        labels_sentiment = []
                        sentiment_true_index = int(label[i][j + self.category_num].detach().cpu().numpy().tolist())
                        if sentiment_true_index == -100:
                            continue
                        titles_sentiment = ['true: %s - pred: %s - %s' % (str(self.polarites[sentiment_true_index]),
                                                        str(pred_sentiment[j][i].detach().cpu().numpy()),
                                                        str(self.polarites))]
                        c_attention = embedding_layer_category_alphas[j][i][: len(words)].detach().cpu().numpy()
                        visual_attentions_sentiment.append(c_attention)
                        labels_sentiment.append(self.categories[j].split('/')[0])

                        s_distributions = visual_attentions_sentiment_temp[j]
                        for k in range(self.polarity_num):
                            labels_sentiment.append(self.polarites[k])
                            visual_attentions_sentiment.append(s_distributions[:, k])
                        titles_sentiment.extend([''] * 3)
                        attention_visualizer.plot_multi_attentions_of_sentence(words, visual_attentions_sentiment,
                                                                               labels_sentiment,
                                                                               titles_sentiment)
                        print()
        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {
            'accuracy': self._accuracy.get_metric(reset),
            'category_f1': self._f1.get_metric(reset)['fscore']
        }
        return metrics


class AsMilSimultaneouslyV5(TextInAllAspectSentimentOutModel):
    def __init__(self, word_embedder: TextFieldEmbedder, position_embedder: TextFieldEmbedder,
                 aspect_embedder: TextFieldEmbedder, categories: list, polarities: list, vocab: Vocabulary,
                 configuration: dict, category_loss_weight=1, sentiment_loss_weight=1):
        super().__init__(vocab, category_loss_weight=category_loss_weight, sentiment_loss_weight=sentiment_loss_weight)
        self.configuration = configuration
        self.word_embedder = word_embedder
        self.position_embedder = position_embedder
        self.aspect_embedder = aspect_embedder
        self.categories = categories
        self.polarites = polarities
        self.category_num = len(categories)
        self.polarity_num = len(polarities)
        self.category_loss = nn.BCEWithLogitsLoss()
        self.sentiment_loss = nn.CrossEntropyLoss()
        self._accuracy = metrics.CategoricalAccuracy()
        self._f1 = allennlp_metrics.BinaryF1(0.5)

        word_embedding_dim = word_embedder.get_output_dim()
        if self.configuration['lstm_or_fc_after_embedding_layer'] == 'fc':
            self.embedding_layer_fc = nn.Linear(word_embedding_dim, word_embedding_dim, bias=True)
        elif self.configuration['lstm_or_fc_after_embedding_layer'] == 'gcn':
            self.gc_aspect_category = DglGraphConvolution(word_embedding_dim, word_embedding_dim, configuration)
        elif self.configuration['lstm_or_fc_after_embedding_layer'] == 'bilstm':
            self.embedding_layer_lstm = torch.nn.LSTM(word_embedding_dim, int(word_embedding_dim / 2), batch_first=True,
                                                      bidirectional=True, num_layers=1)
        else:
            self.embedding_layer_lstm = torch.nn.LSTM(word_embedding_dim, word_embedding_dim, batch_first=True,
                                                      bidirectional=False, num_layers=1)
        # todo 当一句话里没有提到某个aspect category，attention应该关注什么？
        # (1) 类似于bert，添加一个额外的词，attend nothing
        # (2) 关注停用词
        self.embedding_layer_aspect_attentions = [AttentionInHtt(word_embedding_dim,
                                                                 word_embedding_dim)
                                                  for _ in range(self.category_num)]
        self.embedding_layer_aspect_attentions = nn.ModuleList(self.embedding_layer_aspect_attentions)

        lstm_input_size = word_embedding_dim
        if self.configuration['position']:
            lstm_input_size += position_embedder.get_output_dim()
        if self.configuration['sentence_encoder_for_sentiment'] == 'cnn':
            ngram_filter_sizes = (2, 3, 4)
            self.cnn_encoder = CnnEncoder(lstm_input_size, int(word_embedding_dim / len(ngram_filter_sizes)),
                                          ngram_filter_sizes=ngram_filter_sizes)
        else:
            num_layers = self.configuration['lstm_layer_num_in_lstm']
            self.lstm = torch.nn.LSTM(lstm_input_size, int(word_embedding_dim / 2), batch_first=True,
                                      bidirectional=True, num_layers=num_layers, dropout=0.5)

        self.category_fcs = [nn.Linear(word_embedding_dim, 1) for _ in range(self.category_num)]
        self.category_fcs = nn.ModuleList(self.category_fcs)

        if self.configuration['lstm_layer_category_classifier']:
            self.lstm_category_fcs = [nn.Linear(word_embedding_dim, 1) for _ in range(self.category_num)]
            self.lstm_category_fcs = nn.ModuleList(self.lstm_category_fcs)

        sentiment_fc_input_size = word_embedding_dim
        if not self.configuration['share_sentiment_classifier']:
            self.sentiment_fcs = [nn.Sequential(nn.Linear(sentiment_fc_input_size, sentiment_fc_input_size),
                                                nn.ReLU(),
                                                nn.Linear(sentiment_fc_input_size, self.polarity_num))
                                  for _ in range(self.category_num)]
            self.sentiment_fcs = nn.ModuleList(self.sentiment_fcs)
        else:
            self.sentiment_fc = nn.Sequential(nn.Linear(sentiment_fc_input_size, sentiment_fc_input_size),
                                              nn.ReLU(),
                                              nn.Linear(sentiment_fc_input_size, self.polarity_num))

        self.dropout_after_embedding_layer = nn.Dropout(0.5)
        self.dropout_after_lstm_layer = nn.Dropout(0.5)
        # self.gc1 = DglGraphConvolution(word_embedding_dim, word_embedding_dim, configuration)
        # self.gc2 = DglGraphConvolution(word_embedding_dim, word_embedding_dim, configuration)

    def set_grad_for_acd_parameter(self, requires_grad=True):
        acd_layers = []
        if self.configuration['lstm_or_fc_after_embedding_layer'] == 'fc':
            acd_layers.append(self.embedding_layer_fc)
        else:
            acd_layers.append(self.embedding_layer_lstm)
        acd_layers.append(self.embedding_layer_aspect_attentions)
        acd_layers.append(self.category_fcs)
        for layer in acd_layers:
            for name, value in layer.named_parameters():
                value.requires_grad = requires_grad

    def forward(self, tokens: Dict[str, torch.Tensor], label: torch.Tensor, position: torch.Tensor,
                polarity_mask: torch.Tensor, sample: list, aspects: torch.Tensor=None) -> torch.Tensor:
        mask = get_text_field_mask(tokens)
        word_embeddings = self.word_embedder(tokens)
        if self.configuration['lstm_or_fc_after_embedding_layer'] == 'fc':
            word_embeddings_fc = self.embedding_layer_fc(word_embeddings)
        elif self.configuration['lstm_or_fc_after_embedding_layer'] == 'gcn':
            max_len = tokens['tokens'].size()[1]
            graphs = [e[3] for e in sample]
            graphs_padded = self.pad_dgl_graph(graphs, max_len)
            word_embeddings_fc = F.relu(self.gc_aspect_category(word_embeddings, graphs_padded))
        else:
            word_embeddings_fc, (_, _) = self.embedding_layer_lstm(word_embeddings)

        aspects_separate = [{'aspect': aspects['aspect'][:, i].unsqueeze(1)} for i in range(self.category_num)]
        aspect_embeddings_singles = [self.aspect_embedder(aspects_separate[i]).squeeze(1) for i in
                                     range(self.category_num)]
        aspects_seprate_repeat = [{'aspect': aspects_separate[i]['aspect'].expand_as(tokens['tokens'])} for i in
                                  range(self.category_num)]
        aspect_embeddings_separate = [self.aspect_embedder(aspects_seprate_repeat[i]) for i in range(self.category_num)]

        embedding_layer_category_outputs = []
        embedding_layer_category_alphas = []
        embedding_layer_sentiment_outputs = []
        embedding_layer_sentiment_alphas = []

        embedding_layer_category_alphas = []
        for i in range(self.category_num):
            embedding_layer_aspect_attention = self.embedding_layer_aspect_attentions[i]
            alpha = embedding_layer_aspect_attention(word_embeddings_fc, mask)
            embedding_layer_category_alphas.append(alpha)

        for i in range(self.category_num):
            alpha = embedding_layer_category_alphas[i]
            category_output = self.element_wise_mul(word_embeddings_fc, alpha, return_not_sum_result=False)
            embedding_layer_category_outputs.append(category_output)

        lstm_input = word_embeddings
        if self.configuration['position']:
            position_embeddings = self.position_embedder(position)
            lstm_input = torch.cat([word_embeddings, position_embeddings], dim=-1)
        lstm_input = self.dropout_after_embedding_layer(lstm_input)

        if self.configuration['sentence_encoder_for_sentiment'] == 'cnn':
            lstm_result = self.cnn_encoder(lstm_input, mask)
        else:
            lstm_result, _ = self.lstm(lstm_input)
        lstm_result = self.dropout_after_lstm_layer(lstm_result)
        # lstm_result_with_position = torch.cat([lstm_result, position_embeddings], dim=-1)
        lstm_layer_category_outputs = []
        lstm_layer_sentiment_outputs = []
        lstm_layer_words_sentiment_soft = []

        # max_len = tokens['tokens'].size()[1]
        # graphs = [e[3] for e in sample]
        # graphs_padded = self.pad_dgl_graph(graphs, max_len)
        # graph_output1 = F.relu(self.gc1(word_embeddings, graphs_padded))
        # graph_output2 = F.relu(self.gc2(graph_output1, graphs_padded))
        for i in range(self.category_num):
            alpha = embedding_layer_category_alphas[i]
            category_output = self.element_wise_mul(lstm_result, alpha, return_not_sum_result=False)
            lstm_layer_category_outputs.append(category_output)

            # sentiment
            # word_representation_for_sentiment = torch.cat([graph_output2, lstm_result], dim=-1)
            word_representation_for_sentiment = lstm_result
            sentiment_alpha = embedding_layer_category_alphas[i]
            if self.configuration['mil']:
                sentiment_alpha = sentiment_alpha.unsqueeze(1)
                if not self.configuration['share_sentiment_classifier']:
                    words_sentiment = self.sentiment_fcs[i](word_representation_for_sentiment)
                else:
                    words_sentiment = self.sentiment_fc(word_representation_for_sentiment)
                if self.configuration['mil_softmax']:
                    words_sentiment_soft = torch.softmax(words_sentiment, dim=-1)
                    lstm_layer_words_sentiment_soft.append(words_sentiment_soft)
                else:
                    words_sentiment_soft = words_sentiment
                    lstm_layer_words_sentiment_soft.append(torch.softmax(words_sentiment, dim=-1))
                sentiment_output = torch.matmul(sentiment_alpha, words_sentiment_soft).squeeze(1)  # batch_size x 2*hidden_dim
                lstm_layer_sentiment_outputs.append(sentiment_output)
            else:
                sentiment_output_temp = self.element_wise_mul(word_representation_for_sentiment, sentiment_alpha,
                                                         return_not_sum_result=False)
                if not self.configuration['share_sentiment_classifier']:
                    sentiment_output = self.sentiment_fcs[i](sentiment_output_temp)
                else:
                    sentiment_output = self.sentiment_fc(sentiment_output_temp)
                lstm_layer_sentiment_outputs.append(sentiment_output)

        final_category_outputs = []
        final_lstm_category_outputs = []
        final_sentiment_outputs = []
        for i in range(self.category_num):
            fc = self.category_fcs[i]
            category_output = embedding_layer_category_outputs[i]
            final_category_output = fc(category_output)
            final_category_outputs.append(final_category_output)

            if self.configuration['lstm_layer_category_classifier']:
                fc_lstm = self.lstm_category_fcs[i]
                lstm_category_output = lstm_layer_category_outputs[i]
                final_lstm_category_output = fc_lstm(lstm_category_output)
                final_lstm_category_outputs.append(final_lstm_category_output)

            final_sentiment_output = lstm_layer_sentiment_outputs[i]
            final_sentiment_outputs.append(final_sentiment_output)

        output = {}
        output['alpha'] = embedding_layer_category_alphas
        if label is not None:
            category_labels = []
            polarity_labels = []
            polarity_masks = []
            for i in range(self.category_num):
                category_labels.append(label[:, i])
                polarity_labels.append(label[:, i + self.category_num])
                polarity_masks.append(polarity_mask[:, i])
            loss = 0
            total_category_loss = 0
            total_sentiment_loss = 0
            for i in range(self.category_num):
                category_temp_loss = self.category_loss(final_category_outputs[i].squeeze(dim=-1), category_labels[i])
                sentiment_temp_loss = self.sentiment_loss(final_sentiment_outputs[i], polarity_labels[i].long())
                total_category_loss += category_temp_loss
                if not self.configuration['only_acd']:
                    total_sentiment_loss += sentiment_temp_loss
                if self.configuration['lstm_layer_category_classifier']:
                    lstm_category_temp_loss = self.category_loss(final_lstm_category_outputs[i].squeeze(dim=-1),
                                                                 category_labels[i])
                    total_category_loss += lstm_category_temp_loss
            loss = self.category_loss_weight * total_category_loss + self.sentiment_loss_weight * total_sentiment_loss

            # Sparse Regularization Orthogonal Regularization
            if self.configuration['sparse_reg'] or self.configuration['orthogonal_reg']:
                reg_loss = 0
                for j in range(len(sample)):
                    polarity_mask_of_one_sample = polarity_mask[j]
                    category_alpha_of_one_sample = [embedding_layer_category_alphas[k][j] for k in range(self.category_num)]
                    category_alpha_of_mentioned = []
                    category_alpha_of_not_mentioned = []
                    for k in range(self.category_num):
                        if polarity_mask_of_one_sample[k] == 1:
                            category_alpha_of_mentioned.append(category_alpha_of_one_sample[k].unsqueeze(0))
                        else:
                            category_alpha_of_not_mentioned.append(category_alpha_of_one_sample[k].unsqueeze(0))
                    if len(category_alpha_of_not_mentioned) != 0:
                        category_alpha_of_not_mentioned = torch.cat(category_alpha_of_not_mentioned, dim=0)
                        category_alpha_of_not_mentioned = torch.mean(category_alpha_of_not_mentioned, dim=0, keepdim=True)
                        category_alpha_of_mentioned.append(category_alpha_of_not_mentioned)

                    category_eye = torch.eye(len(category_alpha_of_mentioned))
                    category_alpha_of_mentioned = torch.cat(category_alpha_of_mentioned, dim=0)
                    category_alpha_similarity = torch.mm(category_alpha_of_mentioned, category_alpha_of_mentioned.t())

                    if self.configuration['sparse_reg'] and self.configuration['orthogonal_reg']:
                        pass
                    elif self.configuration['sparse_reg']:
                        for m in range(len(category_alpha_of_mentioned)):
                            for n in range(len(category_alpha_of_mentioned)):
                                if m != n:
                                    category_eye[m][n] = category_alpha_similarity[m][n]
                    else:
                        # orthogonal_reg
                        for m in range(len(category_alpha_of_mentioned)):
                            category_eye[m][m] = category_alpha_similarity[m][m]
                    # category_eye = nn_util.move_to_device(category_eye, self.configuration['device'])
                    category_eye = category_eye.to(self.configuration['device'])
                    category_alpha_similarity = category_alpha_similarity.to(self.configuration['device'])
                    category_reg_loss = category_alpha_similarity - category_eye
                    category_reg_loss = torch.norm(category_reg_loss)
                    reg_loss += category_reg_loss
                loss += (reg_loss * self.configuration['attention_lamda'] / len(sample))

            # sentiment accuracy
            sentiment_logit = torch.cat(final_sentiment_outputs)
            sentiment_label = torch.cat(polarity_labels)
            sentiment_mask = torch.cat(polarity_masks)
            self._accuracy(sentiment_logit, sentiment_label, sentiment_mask)

            # category f1
            final_category_outputs_prob = [torch.sigmoid(e) for e in final_category_outputs]
            category_prob = torch.cat(final_category_outputs_prob).squeeze()
            category_label = torch.cat(category_labels)
            self._f1(category_prob, category_label)

            output['loss'] = loss

        # visualize attention
        pred_category = [torch.sigmoid(e) for e in final_category_outputs]
        pred_sentiment = [torch.nn.functional.softmax(e, dim=-1) for e in final_sentiment_outputs]
        output['pred_category'] = pred_category
        output['pred_sentiment'] = pred_sentiment
        output['embedding_layer_category_alphas'] = embedding_layer_category_alphas
        output['lstm_layer_words_sentiment_soft'] = lstm_layer_words_sentiment_soft
        if self.configuration['visualize_attention']:
            for i in range(len(sample)):
                words = sample[i][2]
                # if not ('while' in words and 'there' in words):
                #     continue

                attention_labels = [e.split('/')[0] for e in self.categories]

                # 只可视化mams
                label_true = label[i].detach().cpu().numpy()[: self.category_num]
                if sum(label_true) <= 1:
                    continue
                # category
                visual_attentions_category = [embedding_layer_category_alphas[j][i][: len(words)].detach().cpu().numpy()
                                              for j in range(self.category_num)]
                titles = ['true: %s - pred: %s' % (str(label[i][j].detach().cpu().numpy()),
                                                   str(pred_category[j][i].detach().cpu().numpy()))
                          for j in range(self.category_num)]
                attention_visualizer.plot_multi_attentions_of_sentence_backup(words, visual_attentions_category,
                                                                       attention_labels, titles)
                # savefig_filepath = super()._get_model_visualization_picture_filepath(self.configuration, words)
                # attention_visualizer.plot_multi_attentions_of_sentence(words, visual_attentions_category,
                #                                                        attention_labels, titles,
                #                                                        savefig_filepath=savefig_filepath)

                # sentiment embedding layer
                # visual_attentions = [embedding_layer_sentiment_alphas[j][i][: len(words)].detach().cpu().numpy()
                #                      for j in range(self.category_num)]
                # titles = ['true: %s - pred: %s - %s' % (str(label[i + self.category_num][j].detach().cpu().numpy()),
                #                                         str(pred_sentiment[j][i].detach().cpu().numpy()),
                #                                         str(self.polarites))
                #           for j in range(self.category_num)]
                # attention_visualizer.plot_multi_attentions_of_sentence(words, visual_attentions, attention_labels,
                #                                                        titles)

                # sentiment lstm layer
                # visual_attentions = [lstm_layer_sentiment_alphas[j][i][: len(words)].detach().cpu().numpy()
                #                      for j in range(self.category_num)]
                # titles = ['true: %s - pred: %s - %s' % (str(label[i + self.category_num][j].detach().cpu().numpy()),
                #                                         str(pred_sentiment[j][i].detach().cpu().numpy()),
                #                                         str(self.polarites))
                #           for j in range(self.category_num)]
                # attention_visualizer.plot_multi_attentions_of_sentence(words, visual_attentions, attention_labels,
                #                                                        titles)

                # sentiment lstm layer
                visual_attentions_sentiment_temp = [lstm_layer_words_sentiment_soft[j][i][: len(words)].detach().cpu().numpy()
                                                    for j in range(self.category_num)]
                for j in range(self.category_num):
                    c_label = label[i][j].detach().cpu().numpy().tolist()
                    if c_label == 1:
                        visual_attentions_sentiment = []
                        labels_sentiment = []
                        sentiment_true_index = int(label[i][j + self.category_num].detach().cpu().numpy().tolist())
                        if sentiment_true_index == -100:
                            continue
                        titles_sentiment = ['true: %s - pred: %s - %s' % (str(self.polarites[sentiment_true_index]),
                                                        str(pred_sentiment[j][i].detach().cpu().numpy()),
                                                        str(self.polarites))]
                        c_attention = embedding_layer_category_alphas[j][i][: len(words)].detach().cpu().numpy()
                        visual_attentions_sentiment.append(c_attention)
                        labels_sentiment.append(self.categories[j].split('/')[0])

                        s_distributions = visual_attentions_sentiment_temp[j]
                        for k in range(self.polarity_num):
                            labels_sentiment.append(self.polarites[k])
                            visual_attentions_sentiment.append(s_distributions[:, k])
                        titles_sentiment.extend([''] * 3)
                        attention_visualizer.plot_multi_attentions_of_sentence_backup(words, visual_attentions_sentiment,
                                                                               labels_sentiment,
                                                                               titles_sentiment)
                        # savefig_filepath = super()._get_model_visualization_picture_filepath(self.configuration, words)
                        # attention_visualizer.plot_multi_attentions_of_sentence(words, visual_attentions_category,
                        #                                                        attention_labels, titles,
                        #                                                        savefig_filepath=savefig_filepath)
                        print()
        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {
            'accuracy': self._accuracy.get_metric(reset),
            'category_f1': self._f1.get_metric(reset)['fscore']
        }
        return metrics


class SentenceConsituencyAwareModel(TextInAllAspectSentimentOutModel):
    def __init__(self, word_embedder: TextFieldEmbedder, position_embedder: TextFieldEmbedder,
                 aspect_embedder: TextFieldEmbedder, categories: list, polarities: list, vocab: Vocabulary,
                 configuration: dict, category_loss_weight=1, sentiment_loss_weight=1):
        super().__init__(vocab, category_loss_weight=category_loss_weight, sentiment_loss_weight=sentiment_loss_weight)
        self.configuration = configuration
        self.word_embedder = word_embedder
        self.position_embedder = position_embedder
        self.aspect_embedder = aspect_embedder
        self.categories = categories
        self.polarites = polarities
        self.category_num = len(categories)
        self.polarity_num = len(polarities)
        self.category_loss = nn.BCEWithLogitsLoss()
        self.sentiment_loss = nn.CrossEntropyLoss()
        self._accuracy = metrics.CategoricalAccuracy()
        self._f1 = allennlp_metrics.BinaryF1(0.5)

        word_embedding_dim = word_embedder.get_output_dim()
        if self.configuration['lstm_or_fc_after_embedding_layer'] == 'fc':
            self.embedding_layer_fc = nn.Linear(word_embedding_dim, word_embedding_dim, bias=True)
        elif self.configuration['lstm_or_fc_after_embedding_layer'] == 'gcn':
            self.gc_aspect_category = DglGraphConvolution(word_embedding_dim, word_embedding_dim, configuration)
        elif self.configuration['lstm_or_fc_after_embedding_layer'] == 'bilstm':
            self.embedding_layer_lstm = torch.nn.LSTM(word_embedding_dim, int(word_embedding_dim / 2), batch_first=True,
                                                      bidirectional=True, num_layers=1)
        else:
            self.embedding_layer_lstm = torch.nn.LSTM(word_embedding_dim, word_embedding_dim, batch_first=True,
                                                      bidirectional=False, num_layers=1)
        # todo 当一句话里没有提到某个aspect category，attention应该关注什么？
        # (1) 类似于bert，添加一个额外的词，attend nothing
        # (2) 关注停用词
        self.embedding_layer_aspect_attentions = [AttentionInHtt(word_embedding_dim,
                                                                 word_embedding_dim)
                                                  for _ in range(self.category_num)]
        self.embedding_layer_aspect_attentions = nn.ModuleList(self.embedding_layer_aspect_attentions)

        lstm_input_size = word_embedding_dim
        if self.configuration['position']:
            lstm_input_size += position_embedder.get_output_dim()
        num_layers = self.configuration['lstm_layer_num_in_lstm']
        self.lstm = torch.nn.LSTM(lstm_input_size, int(word_embedding_dim / 2), batch_first=True,
                                  bidirectional=True, num_layers=num_layers, dropout=0.5)

        self.category_fcs = [nn.Linear(word_embedding_dim, 1) for _ in range(self.category_num)]
        self.category_fcs = nn.ModuleList(self.category_fcs)

        sentiment_fc_input_size = word_embedding_dim
        if not self.configuration['share_sentiment_classifier']:
            self.sentiment_fcs = [nn.Sequential(nn.Linear(sentiment_fc_input_size, sentiment_fc_input_size),
                                                nn.ReLU(),
                                                nn.Linear(sentiment_fc_input_size, self.polarity_num))
                                  for _ in range(self.category_num)]
            self.sentiment_fcs = nn.ModuleList(self.sentiment_fcs)
        else:
            self.sentiment_fc = nn.Sequential(nn.Linear(sentiment_fc_input_size, sentiment_fc_input_size),
                                              nn.ReLU(),
                                              nn.Linear(sentiment_fc_input_size, self.polarity_num))

        self.dropout_after_embedding_layer = nn.Dropout(0.5)
        self.dropout_after_lstm_layer = nn.Dropout(0.5)
        self.gnn_for_aspect = DglGraphAverage(word_embedding_dim, word_embedding_dim, configuration)

        # self.gnn_for_sentiment = DglGraphConvolution(word_embedding_dim, word_embedding_dim, configuration)
        # self.gnn_for_sentiment = GAT(word_embedding_dim, word_embedding_dim, word_embedding_dim, 4, self.configuration)
        self.gnn_for_sentiment = DglGraphAverage(word_embedding_dim, word_embedding_dim, self.configuration)

    def no_grad_for_acd_parameter(self):
        acd_layers = []
        if self.configuration['lstm_or_fc_after_embedding_layer'] == 'fc':
            acd_layers.append(self.embedding_layer_fc)
        else:
            acd_layers.append(self.embedding_layer_lstm)
        acd_layers.append(self.embedding_layer_aspect_attentions)
        acd_layers.append(self.category_fcs)
        acd_layers.append(self.gnn_for_aspect)
        for layer in acd_layers:
            for name, value in layer.named_parameters():
                value.requires_grad = False

    def forward(self, tokens: Dict[str, torch.Tensor], tokens_for_graph: Dict[str, torch.Tensor], label: torch.Tensor, position: torch.Tensor,
                polarity_mask: torch.Tensor, sample: list, aspects: torch.Tensor=None) -> torch.Tensor:
        mask_for_graph = get_text_field_mask(tokens_for_graph)

        mask = get_text_field_mask(tokens)

        max_len_for_graph = mask_for_graph.size()[1]
        graphs = [e['graph'] for e in sample]
        graphs_padded = self.pad_dgl_graph(graphs, max_len_for_graph)
        max_len = mask.size()[1]
        padder = nn.ConstantPad2d((0, 0, 0, max_len_for_graph - max_len), 0)

        word_embeddings = self.word_embedder(tokens)
        if self.configuration['lstm_or_fc_after_embedding_layer'] == 'fc':
            word_embeddings_fc = self.embedding_layer_fc(word_embeddings)
        elif self.configuration['lstm_or_fc_after_embedding_layer'] == 'gcn':
            max_len = tokens['tokens'].size()[1]
            graphs = [e[3] for e in sample]
            graphs_padded = self.pad_dgl_graph(graphs, max_len)
            word_embeddings_fc = F.relu(self.gc_aspect_category(word_embeddings, graphs_padded))
        else:
            word_embeddings_fc, (_, _) = self.embedding_layer_lstm(word_embeddings)

        aspects_separate = [{'aspect': aspects['aspect'][:, i].unsqueeze(1)} for i in range(self.category_num)]
        aspect_embeddings_singles = [self.aspect_embedder(aspects_separate[i]).squeeze(1) for i in
                                     range(self.category_num)]
        aspects_seprate_repeat = [{'aspect': aspects_separate[i]['aspect'].expand_as(tokens['tokens'])} for i in
                                  range(self.category_num)]
        aspect_embeddings_separate = [self.aspect_embedder(aspects_seprate_repeat[i]) for i in range(self.category_num)]

        embedding_layer_category_outputs = []
        embedding_layer_category_alphas = []
        embedding_layer_sentiment_outputs = []
        embedding_layer_sentiment_alphas = []

        embedding_layer_category_alphas = []
        # 在word_embeddings_fc上做图卷积，attention是基于图卷积的输出
        word_embeddings_fc_for_graph = padder(word_embeddings_fc)
        graph_output_for_aspect = self.gnn_for_aspect(word_embeddings_fc_for_graph, graphs_padded)

        for i in range(self.category_num):
            embedding_layer_aspect_attention = self.embedding_layer_aspect_attentions[i]
            alpha = embedding_layer_aspect_attention(graph_output_for_aspect, mask_for_graph)
            embedding_layer_category_alphas.append(alpha)

            category_output = self.element_wise_mul(graph_output_for_aspect, alpha, return_not_sum_result=False)
            embedding_layer_category_outputs.append(category_output)

        lstm_input = word_embeddings
        if self.configuration['position']:
            position_embeddings = self.position_embedder(position)
            lstm_input = torch.cat([word_embeddings, position_embeddings], dim=-1)
        lstm_input = self.dropout_after_embedding_layer(lstm_input)

        if self.configuration['attention_warmup']:
            lstm_result_for_graph = graph_output_for_aspect
        else:
            lstm_result, _ = self.lstm(lstm_input)
            lstm_result = self.dropout_after_lstm_layer(lstm_result)
            # 在lstm_result上做图卷积
            lstm_result_for_graph = padder(lstm_result)
            lstm_result_for_graph = self.gnn_for_sentiment(lstm_result_for_graph, graphs_padded)

        # lstm_result_with_position = torch.cat([lstm_result, position_embeddings], dim=-1)
        lstm_layer_sentiment_outputs = []
        lstm_layer_words_sentiment_soft = []

        for i in range(self.category_num):
            # sentiment
            # word_representation_for_sentiment = torch.cat([graph_output2, lstm_result], dim=-1)
            word_representation_for_sentiment = lstm_result_for_graph
            sentiment_alpha = embedding_layer_category_alphas[i]
            if self.configuration['mil']:
                sentiment_alpha = sentiment_alpha.unsqueeze(1)
                if not self.configuration['share_sentiment_classifier']:
                    words_sentiment = self.sentiment_fcs[i](word_representation_for_sentiment)
                else:
                    words_sentiment = self.sentiment_fc(word_representation_for_sentiment)
                if self.configuration['mil_softmax']:
                    words_sentiment_soft = torch.softmax(words_sentiment, dim=-1)
                    lstm_layer_words_sentiment_soft.append(words_sentiment_soft)
                else:
                    words_sentiment_soft = words_sentiment
                    lstm_layer_words_sentiment_soft.append(torch.softmax(words_sentiment, dim=-1))
                sentiment_output = torch.matmul(sentiment_alpha, words_sentiment_soft).squeeze(1)  # batch_size x 2*hidden_dim
                lstm_layer_sentiment_outputs.append(sentiment_output)
            else:
                sentiment_output_temp = self.element_wise_mul(word_representation_for_sentiment, sentiment_alpha,
                                                         return_not_sum_result=False)
                if not self.configuration['share_sentiment_classifier']:
                    sentiment_output = self.sentiment_fcs[i](sentiment_output_temp)
                else:
                    sentiment_output = self.sentiment_fc(sentiment_output_temp)
                lstm_layer_sentiment_outputs.append(sentiment_output)

        final_category_outputs = []
        final_category_outputs_auxillary = []
        final_sentiment_outputs = []
        for i in range(self.category_num):
            category_output = embedding_layer_category_outputs[i]
            for j in range(self.category_num):
                final_category_output = self.category_fcs[j](category_output)
                if i == j:
                    final_category_outputs.append(final_category_output)
                else:
                    final_category_outputs_auxillary.append(final_category_output)

            final_sentiment_output = lstm_layer_sentiment_outputs[i]
            final_sentiment_outputs.append(final_sentiment_output)

        output = {}
        output['alpha'] = embedding_layer_category_alphas
        if label is not None:
            category_labels = []
            polarity_labels = []
            polarity_masks = []
            for i in range(self.category_num):
                category_labels.append(label[:, i])
                polarity_labels.append(label[:, i + self.category_num])
                polarity_masks.append(polarity_mask[:, i])
            loss = 0
            total_category_loss = 0
            total_sentiment_loss = 0
            for i in range(self.category_num):
                category_temp_loss = self.category_loss(final_category_outputs[i].squeeze(dim=-1), category_labels[i])
                sentiment_temp_loss = self.sentiment_loss(final_sentiment_outputs[i], polarity_labels[i].long())
                total_category_loss += category_temp_loss
                if not self.configuration['only_acd']:
                    total_sentiment_loss += sentiment_temp_loss

            loss = self.category_loss_weight * total_category_loss + self.sentiment_loss_weight * total_sentiment_loss

            interactive_loss = 0
            for i in range(len(final_category_outputs_auxillary)):
                label_temp = torch.zeros_like(category_labels[0])
                interactive_loss_temp = self.category_loss(final_category_outputs_auxillary[i].squeeze(dim=-1),
                                                           label_temp)
                interactive_loss += interactive_loss_temp
            loss += (interactive_loss * self.configuration['interactive_loss_lamda'] / (self.category_num - 1))

            # Sparse Regularization Orthogonal Regularization
            if self.configuration['sparse_reg'] or self.configuration['orthogonal_reg']:
                reg_loss = 0
                for j in range(len(sample)):
                    polarity_mask_of_one_sample = polarity_mask[j]
                    category_alpha_of_one_sample = [embedding_layer_category_alphas[k][j] for k in range(self.category_num)]
                    category_alpha_of_mentioned = []
                    category_alpha_of_not_mentioned = []
                    for k in range(self.category_num):
                        if polarity_mask_of_one_sample[k] == 1:
                            category_alpha_of_mentioned.append(category_alpha_of_one_sample[k].unsqueeze(0))
                        else:
                            category_alpha_of_not_mentioned.append(category_alpha_of_one_sample[k].unsqueeze(0))
                    if len(category_alpha_of_not_mentioned) != 0:
                        category_alpha_of_not_mentioned = torch.cat(category_alpha_of_not_mentioned, dim=0)
                        category_alpha_of_not_mentioned = torch.mean(category_alpha_of_not_mentioned, dim=0, keepdim=True)
                        category_alpha_of_mentioned.append(category_alpha_of_not_mentioned)

                    category_eye = torch.eye(len(category_alpha_of_mentioned))
                    category_alpha_of_mentioned = torch.cat(category_alpha_of_mentioned, dim=0)
                    category_alpha_similarity = torch.mm(category_alpha_of_mentioned, category_alpha_of_mentioned.t())

                    if self.configuration['sparse_reg'] and self.configuration['orthogonal_reg']:
                        pass
                    elif self.configuration['sparse_reg']:
                        for m in range(len(category_alpha_of_mentioned)):
                            for n in range(len(category_alpha_of_mentioned)):
                                if m != n:
                                    category_eye[m][n] = category_alpha_similarity[m][n]
                    else:
                        # orthogonal_reg
                        for m in range(len(category_alpha_of_mentioned)):
                            category_eye[m][m] = category_alpha_similarity[m][m]
                    # category_eye = nn_util.move_to_device(category_eye, self.configuration['device'])
                    category_eye = category_eye.to(self.configuration['device'])
                    category_alpha_similarity = category_alpha_similarity.to(self.configuration['device'])
                    category_reg_loss = category_alpha_similarity - category_eye
                    category_reg_loss = torch.norm(category_reg_loss)
                    reg_loss += category_reg_loss
                loss += (reg_loss * self.configuration['attention_lamda'] / len(sample))

            # sentiment accuracy
            sentiment_logit = torch.cat(final_sentiment_outputs)
            sentiment_label = torch.cat(polarity_labels)
            sentiment_mask = torch.cat(polarity_masks)
            self._accuracy(sentiment_logit, sentiment_label, sentiment_mask)

            # category f1
            final_category_outputs_prob = [torch.sigmoid(e) for e in final_category_outputs]
            category_prob = torch.cat(final_category_outputs_prob).squeeze()
            category_label = torch.cat(category_labels)
            self._f1(category_prob, category_label)

            output['loss'] = loss

        # visualize attention
        pred_category = [torch.sigmoid(e) for e in final_category_outputs]
        pred_sentiment = [torch.nn.functional.softmax(e, dim=-1) for e in final_sentiment_outputs]
        output['pred_category'] = pred_category
        output['pred_sentiment'] = pred_sentiment
        if self.configuration['visualize_attention']:
            for i in range(len(sample)):
                words = sample[i]['words_for_graph']
                print('inner_nodes:-------------------------------------------')
                for inner_node_index, inner_node in enumerate(sample[i]['inner_nodes']):
                    print('%d-%s-%s' % (len(sample[i]['words']) + inner_node_index, inner_node.labels[0], inner_node.text))

                attention_labels = [e.split('/')[0] for e in self.categories]

                # 只可视化mams
                label_true = label[i].detach().cpu().numpy()[: self.category_num]
                # if sum(label_true) <= 1:
                #     continue
                # category
                visual_attentions_category = [embedding_layer_category_alphas[j][i][: len(words)].detach().cpu().numpy()
                                              for j in range(self.category_num)]
                titles = ['true: %s - pred: %s' % (str(label[i][j].detach().cpu().numpy()),
                                                   str(pred_category[j][i].detach().cpu().numpy()))
                          for j in range(self.category_num)]
                attention_visualizer.plot_multi_attentions_of_sentence_backup(words, visual_attentions_category,
                                                                       attention_labels, titles)

                # sentiment embedding layer
                # visual_attentions = [embedding_layer_sentiment_alphas[j][i][: len(words)].detach().cpu().numpy()
                #                      for j in range(self.category_num)]
                # titles = ['true: %s - pred: %s - %s' % (str(label[i + self.category_num][j].detach().cpu().numpy()),
                #                                         str(pred_sentiment[j][i].detach().cpu().numpy()),
                #                                         str(self.polarites))
                #           for j in range(self.category_num)]
                # attention_visualizer.plot_multi_attentions_of_sentence(words, visual_attentions, attention_labels,
                #                                                        titles)

                # sentiment lstm layer
                # visual_attentions = [lstm_layer_sentiment_alphas[j][i][: len(words)].detach().cpu().numpy()
                #                      for j in range(self.category_num)]
                # titles = ['true: %s - pred: %s - %s' % (str(label[i + self.category_num][j].detach().cpu().numpy()),
                #                                         str(pred_sentiment[j][i].detach().cpu().numpy()),
                #                                         str(self.polarites))
                #           for j in range(self.category_num)]
                # attention_visualizer.plot_multi_attentions_of_sentence(words, visual_attentions, attention_labels,
                #                                                        titles)

                # sentiment lstm layer
                if self.configuration['mil']:
                    visual_attentions_sentiment_temp = [lstm_layer_words_sentiment_soft[j][i][: len(words)].detach().cpu().numpy()
                                                        for j in range(self.category_num)]
                    for j in range(self.category_num):
                        c_label = label[i][j].detach().cpu().numpy().tolist()
                        if c_label == 1:
                            visual_attentions_sentiment = []
                            labels_sentiment = []
                            sentiment_true_index = int(label[i][j + self.category_num].detach().cpu().numpy().tolist())
                            if sentiment_true_index == -100:
                                continue
                            titles_sentiment = ['true: %s - pred: %s - %s' % (str(self.polarites[sentiment_true_index]),
                                                            str(pred_sentiment[j][i].detach().cpu().numpy()),
                                                            str(self.polarites))]
                            c_attention = embedding_layer_category_alphas[j][i][: len(words)].detach().cpu().numpy()
                            visual_attentions_sentiment.append(c_attention)
                            labels_sentiment.append(self.categories[j].split('/')[0])

                            s_distributions = visual_attentions_sentiment_temp[j]
                            for k in range(self.polarity_num):
                                labels_sentiment.append(self.polarites[k])
                                visual_attentions_sentiment.append(s_distributions[:, k])
                            titles_sentiment.extend([''] * 3)
                            attention_visualizer.plot_multi_attentions_of_sentence(words, visual_attentions_sentiment,
                                                                                   labels_sentiment,
                                                                                   titles_sentiment)
        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {
            'accuracy': self._accuracy.get_metric(reset),
            'category_f1': self._f1.get_metric(reset)['fscore']
        }
        return metrics


class SentenceConsituencyAwareModelV2(TextInAllAspectSentimentOutModel):
    """
    SentenceConsituencyAwareModel + 内部节点表示不是有平均叶子节点表示得到，而是通过attention
    """
    def __init__(self, word_embedder: TextFieldEmbedder, position_embedder: TextFieldEmbedder,
                 aspect_embedder: TextFieldEmbedder, categories: list, polarities: list, vocab: Vocabulary,
                 configuration: dict, category_loss_weight=1, sentiment_loss_weight=1):
        super().__init__(vocab, category_loss_weight=category_loss_weight, sentiment_loss_weight=sentiment_loss_weight)
        self.configuration = configuration
        self.word_embedder = word_embedder
        self.position_embedder = position_embedder
        self.aspect_embedder = aspect_embedder
        self.categories = categories
        self.polarites = polarities
        self.category_num = len(categories)
        self.polarity_num = len(polarities)
        self.category_loss = nn.BCEWithLogitsLoss()
        self.sentiment_loss = nn.CrossEntropyLoss()
        self._accuracy = metrics.CategoricalAccuracy()
        self._f1 = allennlp_metrics.BinaryF1(0.5)

        word_embedding_dim = word_embedder.get_output_dim()
        if self.configuration['lstm_or_fc_after_embedding_layer'] == 'fc':
            self.embedding_layer_fc = nn.Linear(word_embedding_dim, word_embedding_dim, bias=True)
        elif self.configuration['lstm_or_fc_after_embedding_layer'] == 'gcn':
            self.gc_aspect_category = DglGraphConvolution(word_embedding_dim, word_embedding_dim, configuration)
        elif self.configuration['lstm_or_fc_after_embedding_layer'] == 'bilstm':
            self.embedding_layer_lstm = torch.nn.LSTM(word_embedding_dim, int(word_embedding_dim / 2), batch_first=True,
                                                      bidirectional=True, num_layers=1)
        else:
            self.embedding_layer_lstm = torch.nn.LSTM(word_embedding_dim, word_embedding_dim, batch_first=True,
                                                      bidirectional=False, num_layers=1)
        # todo 当一句话里没有提到某个aspect category，attention应该关注什么？
        # (1) 类似于bert，添加一个额外的词，attend nothing
        # (2) 关注停用词
        self.embedding_layer_aspect_attentions = [AttentionInHtt(word_embedding_dim,
                                                                 word_embedding_dim)
                                                  for _ in range(self.category_num)]
        self.embedding_layer_aspect_attentions = nn.ModuleList(self.embedding_layer_aspect_attentions)

        lstm_input_size = word_embedding_dim
        if self.configuration['position']:
            lstm_input_size += position_embedder.get_output_dim()
        num_layers = self.configuration['lstm_layer_num_in_lstm']
        self.lstm = torch.nn.LSTM(lstm_input_size, int(word_embedding_dim / 2), batch_first=True,
                                  bidirectional=True, num_layers=num_layers, dropout=0.5)

        self.category_fcs = [nn.Linear(word_embedding_dim, 1) for _ in range(self.category_num)]
        self.category_fcs = nn.ModuleList(self.category_fcs)

        sentiment_fc_input_size = word_embedding_dim
        if not self.configuration['share_sentiment_classifier']:
            self.sentiment_fcs = [nn.Sequential(nn.Linear(sentiment_fc_input_size, sentiment_fc_input_size),
                                                nn.ReLU(),
                                                nn.Linear(sentiment_fc_input_size, self.polarity_num))
                                  for _ in range(self.category_num)]
            self.sentiment_fcs = nn.ModuleList(self.sentiment_fcs)
        else:
            self.sentiment_fc = nn.Sequential(nn.Linear(sentiment_fc_input_size, sentiment_fc_input_size),
                                              nn.ReLU(),
                                              nn.Linear(sentiment_fc_input_size, self.polarity_num))

        self.dropout_after_embedding_layer = nn.Dropout(0.5)
        self.dropout_after_lstm_layer = nn.Dropout(0.5)
        self.gnn_for_aspect = DglGraphAttentionForAspectCategory(word_embedding_dim, word_embedding_dim,
                                                                 self.embedding_layer_aspect_attentions, configuration)

        # self.gnn_for_sentiment = DglGraphConvolution(word_embedding_dim, word_embedding_dim, configuration)
        # self.gnn_for_sentiment = GAT(word_embedding_dim, word_embedding_dim, word_embedding_dim, 4, self.configuration)
        # self.gnn_for_sentiment = DglGraphAttentionForAspectCategory(word_embedding_dim, word_embedding_dim, self.configuration)

    def no_grad_for_acd_parameter(self):
        acd_layers = []
        if self.configuration['lstm_or_fc_after_embedding_layer'] == 'fc':
            acd_layers.append(self.embedding_layer_fc)
        else:
            acd_layers.append(self.embedding_layer_lstm)
        acd_layers.append(self.embedding_layer_aspect_attentions)
        acd_layers.append(self.category_fcs)
        acd_layers.append(self.gnn_for_aspect)
        for layer in acd_layers:
            for name, value in layer.named_parameters():
                value.requires_grad = False

    def forward(self, tokens: Dict[str, torch.Tensor], tokens_for_graph: Dict[str, torch.Tensor], label: torch.Tensor, position: torch.Tensor,
                polarity_mask: torch.Tensor, sample: list, aspects: torch.Tensor=None) -> torch.Tensor:
        mask_for_graph = get_text_field_mask(tokens_for_graph)

        mask = get_text_field_mask(tokens)

        max_len_for_graph = mask_for_graph.size()[1]
        graphs = [e['graph'] for e in sample]
        graphs_padded = self.pad_dgl_graph(graphs, max_len_for_graph)
        max_len = mask.size()[1]
        padder = nn.ConstantPad2d((0, 0, 0, max_len_for_graph - max_len), 0)

        word_embeddings = self.word_embedder(tokens)
        if self.configuration['lstm_or_fc_after_embedding_layer'] == 'fc':
            word_embeddings_fc = self.embedding_layer_fc(word_embeddings)
        elif self.configuration['lstm_or_fc_after_embedding_layer'] == 'gcn':
            max_len = tokens['tokens'].size()[1]
            graphs = [e[3] for e in sample]
            graphs_padded = self.pad_dgl_graph(graphs, max_len)
            word_embeddings_fc = F.relu(self.gc_aspect_category(word_embeddings, graphs_padded))
        else:
            word_embeddings_fc, (_, _) = self.embedding_layer_lstm(word_embeddings)

        aspects_separate = [{'aspect': aspects['aspect'][:, i].unsqueeze(1)} for i in range(self.category_num)]
        aspect_embeddings_singles = [self.aspect_embedder(aspects_separate[i]).squeeze(1) for i in
                                     range(self.category_num)]
        aspects_seprate_repeat = [{'aspect': aspects_separate[i]['aspect'].expand_as(tokens['tokens'])} for i in
                                  range(self.category_num)]
        aspect_embeddings_separate = [self.aspect_embedder(aspects_seprate_repeat[i]) for i in range(self.category_num)]

        embedding_layer_category_outputs = []
        embedding_layer_category_alphas = []
        embedding_layer_sentiment_outputs = []
        embedding_layer_sentiment_alphas = []

        embedding_layer_category_alphas = []
        # 在word_embeddings_fc上做图卷积，attention是基于图卷积的输出
        word_embeddings_fc_for_graph = padder(word_embeddings_fc)

        lstm_input = word_embeddings
        if self.configuration['position']:
            position_embeddings = self.position_embedder(position)
            lstm_input = torch.cat([word_embeddings, position_embeddings], dim=-1)
        lstm_input = self.dropout_after_embedding_layer(lstm_input)

        if self.configuration['attention_warmup']:
            graph_output_for_aspect, lstm_result_for_graph = self.gnn_for_aspect(word_embeddings_fc_for_graph,
                                                                                 word_embeddings_fc_for_graph,
                                                                                 graphs_padded)
        else:
            lstm_result, _ = self.lstm(lstm_input)
            lstm_result = self.dropout_after_lstm_layer(lstm_result)
            # 在lstm_result上做图卷积
            lstm_result_for_graph = padder(lstm_result)

            graph_output_for_aspect, lstm_result_for_graph = self.gnn_for_aspect(word_embeddings_fc_for_graph,
                                                                                 lstm_result_for_graph,
                                                                                 graphs_padded)

        for i in range(self.category_num):
            embedding_layer_aspect_attention = self.embedding_layer_aspect_attentions[i]
            alpha = embedding_layer_aspect_attention(graph_output_for_aspect, mask_for_graph)
            embedding_layer_category_alphas.append(alpha)

            category_output = self.element_wise_mul(graph_output_for_aspect, alpha, return_not_sum_result=False)
            embedding_layer_category_outputs.append(category_output)

        # lstm_result_with_position = torch.cat([lstm_result, position_embeddings], dim=-1)
        lstm_layer_sentiment_outputs = []
        lstm_layer_words_sentiment_soft = []

        for i in range(self.category_num):
            # sentiment
            # word_representation_for_sentiment = torch.cat([graph_output2, lstm_result], dim=-1)
            word_representation_for_sentiment = lstm_result_for_graph
            sentiment_alpha = embedding_layer_category_alphas[i]
            if self.configuration['mil']:
                sentiment_alpha = sentiment_alpha.unsqueeze(1)
                if not self.configuration['share_sentiment_classifier']:
                    words_sentiment = self.sentiment_fcs[i](word_representation_for_sentiment)
                else:
                    words_sentiment = self.sentiment_fc(word_representation_for_sentiment)
                if self.configuration['mil_softmax']:
                    words_sentiment_soft = torch.softmax(words_sentiment, dim=-1)
                    lstm_layer_words_sentiment_soft.append(words_sentiment_soft)
                else:
                    words_sentiment_soft = words_sentiment
                    lstm_layer_words_sentiment_soft.append(torch.softmax(words_sentiment, dim=-1))
                sentiment_output = torch.matmul(sentiment_alpha, words_sentiment_soft).squeeze(1)  # batch_size x 2*hidden_dim
                lstm_layer_sentiment_outputs.append(sentiment_output)
            else:
                sentiment_output_temp = self.element_wise_mul(word_representation_for_sentiment, sentiment_alpha,
                                                         return_not_sum_result=False)
                if not self.configuration['share_sentiment_classifier']:
                    sentiment_output = self.sentiment_fcs[i](sentiment_output_temp)
                else:
                    sentiment_output = self.sentiment_fc(sentiment_output_temp)
                lstm_layer_sentiment_outputs.append(sentiment_output)

        final_category_outputs = []
        final_category_outputs_auxillary = []
        final_sentiment_outputs = []
        for i in range(self.category_num):
            category_output = embedding_layer_category_outputs[i]
            for j in range(self.category_num):
                final_category_output = self.category_fcs[j](category_output)
                if i == j:
                    final_category_outputs.append(final_category_output)
                else:
                    final_category_outputs_auxillary.append(final_category_output)

            final_sentiment_output = lstm_layer_sentiment_outputs[i]
            final_sentiment_outputs.append(final_sentiment_output)

        output = {}
        output['alpha'] = embedding_layer_category_alphas
        if label is not None:
            category_labels = []
            polarity_labels = []
            polarity_masks = []
            for i in range(self.category_num):
                category_labels.append(label[:, i])
                polarity_labels.append(label[:, i + self.category_num])
                polarity_masks.append(polarity_mask[:, i])
            loss = 0
            total_category_loss = 0
            total_sentiment_loss = 0
            for i in range(self.category_num):
                category_temp_loss = self.category_loss(final_category_outputs[i].squeeze(dim=-1), category_labels[i])
                sentiment_temp_loss = self.sentiment_loss(final_sentiment_outputs[i], polarity_labels[i].long())
                total_category_loss += category_temp_loss
                if not self.configuration['only_acd']:
                    total_sentiment_loss += sentiment_temp_loss

            loss = self.category_loss_weight * total_category_loss + self.sentiment_loss_weight * total_sentiment_loss

            interactive_loss = 0
            for i in range(len(final_category_outputs_auxillary)):
                label_temp = torch.zeros_like(category_labels[0])
                interactive_loss_temp = self.category_loss(final_category_outputs_auxillary[i].squeeze(dim=-1),
                                                           label_temp)
                interactive_loss += interactive_loss_temp
            loss += (interactive_loss * self.configuration['interactive_loss_lamda'] / (self.category_num - 1))

            # Sparse Regularization Orthogonal Regularization
            if self.configuration['sparse_reg'] or self.configuration['orthogonal_reg']:
                reg_loss = 0
                for j in range(len(sample)):
                    polarity_mask_of_one_sample = polarity_mask[j]
                    category_alpha_of_one_sample = [embedding_layer_category_alphas[k][j] for k in range(self.category_num)]
                    category_alpha_of_mentioned = []
                    category_alpha_of_not_mentioned = []
                    for k in range(self.category_num):
                        if polarity_mask_of_one_sample[k] == 1:
                            category_alpha_of_mentioned.append(category_alpha_of_one_sample[k].unsqueeze(0))
                        else:
                            category_alpha_of_not_mentioned.append(category_alpha_of_one_sample[k].unsqueeze(0))
                    if len(category_alpha_of_not_mentioned) != 0:
                        category_alpha_of_not_mentioned = torch.cat(category_alpha_of_not_mentioned, dim=0)
                        category_alpha_of_not_mentioned = torch.mean(category_alpha_of_not_mentioned, dim=0, keepdim=True)
                        category_alpha_of_mentioned.append(category_alpha_of_not_mentioned)

                    category_eye = torch.eye(len(category_alpha_of_mentioned))
                    category_alpha_of_mentioned = torch.cat(category_alpha_of_mentioned, dim=0)
                    category_alpha_similarity = torch.mm(category_alpha_of_mentioned, category_alpha_of_mentioned.t())

                    if self.configuration['sparse_reg'] and self.configuration['orthogonal_reg']:
                        pass
                    elif self.configuration['sparse_reg']:
                        for m in range(len(category_alpha_of_mentioned)):
                            for n in range(len(category_alpha_of_mentioned)):
                                if m != n:
                                    category_eye[m][n] = category_alpha_similarity[m][n]
                    else:
                        # orthogonal_reg
                        for m in range(len(category_alpha_of_mentioned)):
                            category_eye[m][m] = category_alpha_similarity[m][m]
                    # category_eye = nn_util.move_to_device(category_eye, self.configuration['device'])
                    category_eye = category_eye.to(self.configuration['device'])
                    category_alpha_similarity = category_alpha_similarity.to(self.configuration['device'])
                    category_reg_loss = category_alpha_similarity - category_eye
                    category_reg_loss = torch.norm(category_reg_loss)
                    reg_loss += category_reg_loss
                loss += (reg_loss * self.configuration['attention_lamda'] / len(sample))

            # sentiment accuracy
            sentiment_logit = torch.cat(final_sentiment_outputs)
            sentiment_label = torch.cat(polarity_labels)
            sentiment_mask = torch.cat(polarity_masks)
            self._accuracy(sentiment_logit, sentiment_label, sentiment_mask)

            # category f1
            final_category_outputs_prob = [torch.sigmoid(e) for e in final_category_outputs]
            category_prob = torch.cat(final_category_outputs_prob).squeeze()
            category_label = torch.cat(category_labels)
            self._f1(category_prob, category_label)

            output['loss'] = loss

        # visualize attention
        pred_category = [torch.sigmoid(e) for e in final_category_outputs]
        pred_sentiment = [torch.nn.functional.softmax(e, dim=-1) for e in final_sentiment_outputs]
        output['pred_category'] = pred_category
        output['pred_sentiment'] = pred_sentiment
        if self.configuration['visualize_attention']:
            for i in range(len(sample)):
                words = sample[i]['words_for_graph']
                print('inner_nodes:-------------------------------------------')
                for inner_node_index, inner_node in enumerate(sample[i]['inner_nodes']):
                    print('%d-%s-%s' % (len(sample[i]['words']) + inner_node_index, inner_node.labels[0], inner_node.text))

                attention_labels = [e.split('/')[0] for e in self.categories]

                # 只可视化mams
                label_true = label[i].detach().cpu().numpy()[: self.category_num]
                if sum(label_true) <= 1:
                    continue
                # category
                visual_attentions_category = [embedding_layer_category_alphas[j][i][: len(words)].detach().cpu().numpy()
                                              for j in range(self.category_num)]
                titles = ['true: %s - pred: %s' % (str(label[i][j].detach().cpu().numpy()),
                                                   str(pred_category[j][i].detach().cpu().numpy()))
                          for j in range(self.category_num)]
                attention_visualizer.plot_multi_attentions_of_sentence_backup(words, visual_attentions_category,
                                                                       attention_labels, titles)

                # sentiment embedding layer
                # visual_attentions = [embedding_layer_sentiment_alphas[j][i][: len(words)].detach().cpu().numpy()
                #                      for j in range(self.category_num)]
                # titles = ['true: %s - pred: %s - %s' % (str(label[i + self.category_num][j].detach().cpu().numpy()),
                #                                         str(pred_sentiment[j][i].detach().cpu().numpy()),
                #                                         str(self.polarites))
                #           for j in range(self.category_num)]
                # attention_visualizer.plot_multi_attentions_of_sentence(words, visual_attentions, attention_labels,
                #                                                        titles)

                # sentiment lstm layer
                # visual_attentions = [lstm_layer_sentiment_alphas[j][i][: len(words)].detach().cpu().numpy()
                #                      for j in range(self.category_num)]
                # titles = ['true: %s - pred: %s - %s' % (str(label[i + self.category_num][j].detach().cpu().numpy()),
                #                                         str(pred_sentiment[j][i].detach().cpu().numpy()),
                #                                         str(self.polarites))
                #           for j in range(self.category_num)]
                # attention_visualizer.plot_multi_attentions_of_sentence(words, visual_attentions, attention_labels,
                #                                                        titles)

                # sentiment lstm layer
                if self.configuration['mil']:
                    visual_attentions_sentiment_temp = [lstm_layer_words_sentiment_soft[j][i][: len(words)].detach().cpu().numpy()
                                                        for j in range(self.category_num)]
                    for j in range(self.category_num):
                        c_label = label[i][j].detach().cpu().numpy().tolist()
                        if c_label == 1:
                            visual_attentions_sentiment = []
                            labels_sentiment = []
                            sentiment_true_index = int(label[i][j + self.category_num].detach().cpu().numpy().tolist())
                            if sentiment_true_index == -100:
                                continue
                            titles_sentiment = ['true: %s - pred: %s - %s' % (str(self.polarites[sentiment_true_index]),
                                                            str(pred_sentiment[j][i].detach().cpu().numpy()),
                                                            str(self.polarites))]
                            c_attention = embedding_layer_category_alphas[j][i][: len(words)].detach().cpu().numpy()
                            visual_attentions_sentiment.append(c_attention)
                            labels_sentiment.append(self.categories[j].split('/')[0])

                            s_distributions = visual_attentions_sentiment_temp[j]
                            for k in range(self.polarity_num):
                                labels_sentiment.append(self.polarites[k])
                                visual_attentions_sentiment.append(s_distributions[:, k])
                            titles_sentiment.extend([''] * 3)
                            attention_visualizer.plot_multi_attentions_of_sentence(words, visual_attentions_sentiment,
                                                                                   labels_sentiment,
                                                                                   titles_sentiment)
        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {
            'accuracy': self._accuracy.get_metric(reset),
            'category_f1': self._f1.get_metric(reset)['fscore']
        }
        return metrics


class SentenceConsituencyAwareModelV3(TextInAllAspectSentimentOutModel):
    """
    SentenceConsituencyAwareModelV2 + 针对某个属性，属性分类任务找到能预测这个属性但不能预测其它属性的所有节点
    (内部节点和外部节点，偏向上层节点)，情感分类任务基于找到的节点做属性的情感分类；想证明的是，拥有内部节点的
    知识，能获得的比只通过找到的叶子节点的信息进行情感分类更好的效果；内部节点用作属性分类，比叶子节点有时会有
    更好的效果，因为某些属性，并不是由一个词就能指示的，而是由一小片文本的语义确定的；
    """
    def __init__(self, word_embedder: TextFieldEmbedder, position_embedder: TextFieldEmbedder,
                 aspect_embedder: TextFieldEmbedder, categories: list, polarities: list, vocab: Vocabulary,
                 configuration: dict, category_loss_weight=1, sentiment_loss_weight=1):
        super().__init__(vocab, category_loss_weight=category_loss_weight, sentiment_loss_weight=sentiment_loss_weight)
        self.configuration = configuration
        self.word_embedder = word_embedder
        self.position_embedder = position_embedder
        self.aspect_embedder = aspect_embedder
        self.categories = categories
        self.polarites = polarities
        self.category_num = len(categories)
        self.polarity_num = len(polarities)
        self.category_loss = nn.BCEWithLogitsLoss()
        self.sentiment_loss = nn.CrossEntropyLoss()
        self._accuracy = metrics.CategoricalAccuracy()
        self._f1 = allennlp_metrics.BinaryF1(0.5)

        word_embedding_dim = word_embedder.get_output_dim()
        if self.configuration['lstm_or_fc_after_embedding_layer'] == 'fc':
            self.embedding_layer_fc = nn.Linear(word_embedding_dim, word_embedding_dim, bias=True)
        elif self.configuration['lstm_or_fc_after_embedding_layer'] == 'gcn':
            self.gc_aspect_category = DglGraphConvolution(word_embedding_dim, word_embedding_dim, configuration)
        elif self.configuration['lstm_or_fc_after_embedding_layer'] == 'bilstm':
            self.embedding_layer_lstm = torch.nn.LSTM(word_embedding_dim, int(word_embedding_dim / 2), batch_first=True,
                                                      bidirectional=True, num_layers=1)
        else:
            self.embedding_layer_lstm = torch.nn.LSTM(word_embedding_dim, word_embedding_dim, batch_first=True,
                                                      bidirectional=False, num_layers=1)
        # todo 当一句话里没有提到某个aspect category，attention应该关注什么？
        # (1) 类似于bert，添加一个额外的词，attend nothing
        # (2) 关注停用词
        self.embedding_layer_aspect_attentions = [AttentionInHtt(word_embedding_dim,
                                                                 word_embedding_dim)
                                                  for _ in range(self.category_num)]
        self.embedding_layer_aspect_attentions = nn.ModuleList(self.embedding_layer_aspect_attentions)

        lstm_input_size = word_embedding_dim
        if self.configuration['position']:
            lstm_input_size += position_embedder.get_output_dim()
        num_layers = self.configuration['lstm_layer_num_in_lstm']
        self.lstm = torch.nn.LSTM(lstm_input_size, int(word_embedding_dim / 2), batch_first=True,
                                  bidirectional=True, num_layers=num_layers, dropout=0.5)

        self.category_fcs = [nn.Linear(word_embedding_dim, 1) for _ in range(self.category_num)]
        self.category_fcs = nn.ModuleList(self.category_fcs)

        sentiment_fc_input_size = word_embedding_dim
        if not self.configuration['share_sentiment_classifier']:
            self.sentiment_fcs = [nn.Sequential(nn.Linear(sentiment_fc_input_size, sentiment_fc_input_size),
                                                nn.ReLU(),
                                                nn.Linear(sentiment_fc_input_size, self.polarity_num))
                                  for _ in range(self.category_num)]
            self.sentiment_fcs = nn.ModuleList(self.sentiment_fcs)
        else:
            self.sentiment_fc = nn.Sequential(nn.Linear(sentiment_fc_input_size, sentiment_fc_input_size),
                                              nn.ReLU(),
                                              nn.Linear(sentiment_fc_input_size, self.polarity_num))

        self.dropout_after_embedding_layer = nn.Dropout(0.5)
        self.dropout_after_lstm_layer = nn.Dropout(0.5)
        # 只负责生成针对aspect category的节点表示
        aspect_graph = self.configuration['aspect_graph']
        if aspect_graph == 'attention':
            self.gnn_for_aspect = DglGraphAttentionForAspectCategory(word_embedding_dim, word_embedding_dim,
                                                                     self.embedding_layer_aspect_attentions, configuration)
        else:
            raise NotImplementedError(aspect_graph)

        sentiment_graph = self.configuration['sentiment_graph']
        if sentiment_graph == 'average':
            self.gnn_for_sentiment = DglGraphAverage(word_embedding_dim, word_embedding_dim, self.configuration)
        elif sentiment_graph == 'gcn':
            self.gnn_for_sentiment = DglGraphConvolution(word_embedding_dim, word_embedding_dim, configuration)
        elif sentiment_graph == 'gat':
            self.gnn_for_sentiment = GAT(word_embedding_dim, word_embedding_dim, word_embedding_dim, 4, self.configuration)
        else:
            raise NotImplementedError(sentiment_graph)

    def set_grad_for_acd_parameter(self, requires_grad=True):
        acd_layers = []
        if self.configuration['lstm_or_fc_after_embedding_layer'] == 'fc':
            acd_layers.append(self.embedding_layer_fc)
        elif self.configuration['lstm_or_fc_after_embedding_layer'] == 'gcn':
            acd_layers.append(self.gc_aspect_category)
        else:
            acd_layers.append(self.embedding_layer_lstm)
        acd_layers.append(self.embedding_layer_aspect_attentions)
        acd_layers.append(self.category_fcs)
        acd_layers.append(self.gnn_for_aspect)
        for layer in acd_layers:
            for name, value in layer.named_parameters():
                value.requires_grad = requires_grad

    def forward(self, tokens: Dict[str, torch.Tensor], tokens_for_graph: Dict[str, torch.Tensor], label: torch.Tensor, position: torch.Tensor,
                polarity_mask: torch.Tensor, sample: list, aspects: torch.Tensor=None) -> torch.Tensor:
        mask_for_graph = get_text_field_mask(tokens_for_graph)

        mask = get_text_field_mask(tokens)

        max_len_for_graph = mask_for_graph.size()[1]
        graphs = [e['graph'] for e in sample]
        graphs_padded = self.pad_dgl_graph(graphs, max_len_for_graph)
        max_len = mask.size()[1]
        padder = nn.ConstantPad2d((0, 0, 0, max_len_for_graph - max_len), 0)

        word_embeddings = self.word_embedder(tokens)
        if self.configuration['lstm_or_fc_after_embedding_layer'] == 'fc':
            word_embeddings_fc = self.embedding_layer_fc(word_embeddings)
        elif self.configuration['lstm_or_fc_after_embedding_layer'] == 'gcn':
            max_len = tokens['tokens'].size()[1]
            graphs = [e[3] for e in sample]
            graphs_padded = self.pad_dgl_graph(graphs, max_len)
            word_embeddings_fc = F.relu(self.gc_aspect_category(word_embeddings, graphs_padded))
        else:
            word_embeddings_fc, (_, _) = self.embedding_layer_lstm(word_embeddings)

        aspects_separate = [{'aspect': aspects['aspect'][:, i].unsqueeze(1)} for i in range(self.category_num)]
        aspect_embeddings_singles = [self.aspect_embedder(aspects_separate[i]).squeeze(1) for i in
                                     range(self.category_num)]
        aspects_seprate_repeat = [{'aspect': aspects_separate[i]['aspect'].expand_as(tokens['tokens'])} for i in
                                  range(self.category_num)]
        aspect_embeddings_separate = [self.aspect_embedder(aspects_seprate_repeat[i]) for i in range(self.category_num)]

        embedding_layer_category_outputs = []
        embedding_layer_category_alphas = []
        embedding_layer_sentiment_outputs = []
        embedding_layer_sentiment_alphas = []

        embedding_layer_category_alphas = []
        # 在word_embeddings_fc上做图卷积，attention是基于图卷积的输出
        word_embeddings_fc_for_graph = padder(word_embeddings_fc)

        if self.configuration['attention_warmup']:
            graph_output_for_aspect, lstm_result_for_graph = self.gnn_for_aspect(word_embeddings_fc_for_graph,
                                                                                 word_embeddings_fc_for_graph,
                                                                                 graphs_padded)
        else:
            graph_output_for_aspect, _ = self.gnn_for_aspect(word_embeddings_fc_for_graph,
                                                             word_embeddings_fc_for_graph,
                                                             graphs_padded)

            lstm_input = word_embeddings
            if self.configuration['position']:
                position_embeddings = self.position_embedder(position)
                lstm_input = torch.cat([word_embeddings, position_embeddings], dim=-1)
            lstm_input = self.dropout_after_embedding_layer(lstm_input)
            lstm_result, _ = self.lstm(lstm_input)
            lstm_result = self.dropout_after_lstm_layer(lstm_result)
            # 在lstm_result上做图卷积
            lstm_result_padded_for_graph = padder(lstm_result)

            if self.configuration['acd_sc_encoder_mode'] == 'complex':
                lstm_result_for_graph, _ = self.gnn_for_aspect(lstm_result_padded_for_graph,
                                                               lstm_result_padded_for_graph, graphs_padded)
            else:
                lstm_result_for_graph = self.gnn_for_sentiment(lstm_result_padded_for_graph, graphs_padded)

        acd_sc_encoder_mode = self.configuration['acd_sc_encoder_mode']
        if acd_sc_encoder_mode == 'simple':
            lstm_result_for_graph = graph_output_for_aspect
        elif acd_sc_encoder_mode == 'complex':
            graph_output_for_aspect = lstm_result_for_graph

        for i in range(self.category_num):
            embedding_layer_aspect_attention = self.embedding_layer_aspect_attentions[i]
            # todo 根据节点的深度调整权重，使得偏向上层节点
            alpha = embedding_layer_aspect_attention(graph_output_for_aspect, mask_for_graph)
            embedding_layer_category_alphas.append(alpha)

            category_output = self.element_wise_mul(graph_output_for_aspect, alpha, return_not_sum_result=False)
            embedding_layer_category_outputs.append(category_output)

        # lstm_result_with_position = torch.cat([lstm_result, position_embeddings], dim=-1)
        lstm_layer_sentiment_outputs = []
        lstm_layer_words_sentiment_soft = []

        for i in range(self.category_num):
            # sentiment
            # word_representation_for_sentiment = torch.cat([graph_output2, lstm_result], dim=-1)
            word_representation_for_sentiment = lstm_result_for_graph
            sentiment_alpha = embedding_layer_category_alphas[i]
            if self.configuration['mil']:
                sentiment_alpha = sentiment_alpha.unsqueeze(1)
                if not self.configuration['share_sentiment_classifier']:
                    words_sentiment = self.sentiment_fcs[i](word_representation_for_sentiment)
                else:
                    words_sentiment = self.sentiment_fc(word_representation_for_sentiment)
                if self.configuration['mil_softmax']:
                    words_sentiment_soft = torch.softmax(words_sentiment, dim=-1)
                    lstm_layer_words_sentiment_soft.append(words_sentiment_soft)
                else:
                    words_sentiment_soft = words_sentiment
                    lstm_layer_words_sentiment_soft.append(torch.softmax(words_sentiment, dim=-1))
                sentiment_output = torch.matmul(sentiment_alpha, words_sentiment_soft).squeeze(1)  # batch_size x 2*hidden_dim
                lstm_layer_sentiment_outputs.append(sentiment_output)
            else:
                sentiment_output_temp = self.element_wise_mul(word_representation_for_sentiment, sentiment_alpha,
                                                         return_not_sum_result=False)
                if not self.configuration['share_sentiment_classifier']:
                    sentiment_output = self.sentiment_fcs[i](sentiment_output_temp)
                else:
                    sentiment_output = self.sentiment_fc(sentiment_output_temp)
                lstm_layer_sentiment_outputs.append(sentiment_output)

        final_category_outputs = []
        final_category_outputs_auxillary = []
        final_sentiment_outputs = []
        for i in range(self.category_num):
            category_output = embedding_layer_category_outputs[i]
            for j in range(self.category_num):
                final_category_output = self.category_fcs[j](category_output)
                if i == j:
                    final_category_outputs.append(final_category_output)
                else:
                    final_category_outputs_auxillary.append(final_category_output)

            final_sentiment_output = lstm_layer_sentiment_outputs[i]
            final_sentiment_outputs.append(final_sentiment_output)

        output = {}
        output['alpha'] = embedding_layer_category_alphas
        if label is not None:
            category_labels = []
            polarity_labels = []
            polarity_masks = []
            for i in range(self.category_num):
                category_labels.append(label[:, i])
                polarity_labels.append(label[:, i + self.category_num])
                polarity_masks.append(polarity_mask[:, i])
            loss = 0
            total_category_loss = 0
            total_sentiment_loss = 0
            for i in range(self.category_num):
                category_temp_loss = self.category_loss(final_category_outputs[i].squeeze(dim=-1), category_labels[i])
                sentiment_temp_loss = self.sentiment_loss(final_sentiment_outputs[i], polarity_labels[i].long())
                total_category_loss += category_temp_loss
                if not self.configuration['only_acd']:
                    total_sentiment_loss += sentiment_temp_loss

            loss = self.category_loss_weight * total_category_loss + self.sentiment_loss_weight * total_sentiment_loss

            interactive_loss = 0
            for i in range(len(final_category_outputs_auxillary)):
                label_temp = torch.zeros_like(category_labels[0])
                interactive_loss_temp = self.category_loss(final_category_outputs_auxillary[i].squeeze(dim=-1),
                                                           label_temp)
                interactive_loss += interactive_loss_temp
            loss += (interactive_loss * self.configuration['interactive_loss_lamda'] / (self.category_num - 1))

            # Sparse Regularization Orthogonal Regularization
            if self.configuration['sparse_reg'] or self.configuration['orthogonal_reg']:
                reg_loss = 0
                for j in range(len(sample)):
                    polarity_mask_of_one_sample = polarity_mask[j]
                    category_alpha_of_one_sample = [embedding_layer_category_alphas[k][j] for k in range(self.category_num)]
                    category_alpha_of_mentioned = []
                    category_alpha_of_not_mentioned = []
                    for k in range(self.category_num):
                        if polarity_mask_of_one_sample[k] == 1:
                            category_alpha_of_mentioned.append(category_alpha_of_one_sample[k].unsqueeze(0))
                        else:
                            category_alpha_of_not_mentioned.append(category_alpha_of_one_sample[k].unsqueeze(0))
                    if len(category_alpha_of_not_mentioned) != 0:
                        category_alpha_of_not_mentioned = torch.cat(category_alpha_of_not_mentioned, dim=0)
                        category_alpha_of_not_mentioned = torch.mean(category_alpha_of_not_mentioned, dim=0, keepdim=True)
                        category_alpha_of_mentioned.append(category_alpha_of_not_mentioned)

                    category_eye = torch.eye(len(category_alpha_of_mentioned))
                    category_alpha_of_mentioned = torch.cat(category_alpha_of_mentioned, dim=0)
                    category_alpha_similarity = torch.mm(category_alpha_of_mentioned, category_alpha_of_mentioned.t())

                    if self.configuration['sparse_reg'] and self.configuration['orthogonal_reg']:
                        pass
                    elif self.configuration['sparse_reg']:
                        for m in range(len(category_alpha_of_mentioned)):
                            for n in range(len(category_alpha_of_mentioned)):
                                if m != n:
                                    category_eye[m][n] = category_alpha_similarity[m][n]
                    else:
                        # orthogonal_reg
                        for m in range(len(category_alpha_of_mentioned)):
                            category_eye[m][m] = category_alpha_similarity[m][m]
                    # category_eye = nn_util.move_to_device(category_eye, self.configuration['device'])
                    category_eye = category_eye.to(self.configuration['device'])
                    category_alpha_similarity = category_alpha_similarity.to(self.configuration['device'])
                    category_reg_loss = category_alpha_similarity - category_eye
                    category_reg_loss = torch.norm(category_reg_loss)
                    reg_loss += category_reg_loss
                loss += (reg_loss * self.configuration['attention_lamda'] / len(sample))

            # sentiment accuracy
            sentiment_logit = torch.cat(final_sentiment_outputs)
            sentiment_label = torch.cat(polarity_labels)
            sentiment_mask = torch.cat(polarity_masks)
            self._accuracy(sentiment_logit, sentiment_label, sentiment_mask)

            # category f1
            final_category_outputs_prob = [torch.sigmoid(e) for e in final_category_outputs]
            category_prob = torch.cat(final_category_outputs_prob).squeeze()
            category_label = torch.cat(category_labels)
            self._f1(category_prob, category_label)

            output['loss'] = loss

        # visualize attention
        pred_category = [torch.sigmoid(e) for e in final_category_outputs]
        pred_sentiment = [torch.nn.functional.softmax(e, dim=-1) for e in final_sentiment_outputs]
        output['pred_category'] = pred_category
        output['pred_sentiment'] = pred_sentiment
        if self.configuration['visualize_attention']:
            for i in range(len(sample)):
                words = sample[i]['words_for_graph']
                print('inner_nodes:-------------------------------------------')
                for inner_node_index, inner_node in enumerate(sample[i]['inner_nodes']):
                    print('%d-%s-%s' % (len(sample[i]['words']) + inner_node_index, inner_node.labels[0], inner_node.text))

                attention_labels = [e.split('/')[0] for e in self.categories]

                # 只可视化mams
                label_true = label[i].detach().cpu().numpy()[: self.category_num]
                if sum(label_true) <= 1:
                    continue
                # category
                visual_attentions_category = [embedding_layer_category_alphas[j][i][: len(words)].detach().cpu().numpy()
                                              for j in range(self.category_num)]
                titles = ['true: %s - pred: %s' % (str(label[i][j].detach().cpu().numpy()),
                                                   str(pred_category[j][i].detach().cpu().numpy()))
                          for j in range(self.category_num)]
                savefig_filepath = super()._get_model_visualization_picture_filepath(self.configuration, words)
                attention_visualizer.plot_multi_attentions_of_sentence(words, visual_attentions_category,
                                                                       attention_labels, titles,
                                                                       savefig_filepath=savefig_filepath)

                # sentiment embedding layer
                # visual_attentions = [embedding_layer_sentiment_alphas[j][i][: len(words)].detach().cpu().numpy()
                #                      for j in range(self.category_num)]
                # titles = ['true: %s - pred: %s - %s' % (str(label[i + self.category_num][j].detach().cpu().numpy()),
                #                                         str(pred_sentiment[j][i].detach().cpu().numpy()),
                #                                         str(self.polarites))
                #           for j in range(self.category_num)]
                # attention_visualizer.plot_multi_attentions_of_sentence(words, visual_attentions, attention_labels,
                #                                                        titles)

                # sentiment lstm layer
                # visual_attentions = [lstm_layer_sentiment_alphas[j][i][: len(words)].detach().cpu().numpy()
                #                      for j in range(self.category_num)]
                # titles = ['true: %s - pred: %s - %s' % (str(label[i + self.category_num][j].detach().cpu().numpy()),
                #                                         str(pred_sentiment[j][i].detach().cpu().numpy()),
                #                                         str(self.polarites))
                #           for j in range(self.category_num)]
                # attention_visualizer.plot_multi_attentions_of_sentence(words, visual_attentions, attention_labels,
                #                                                        titles)

                # sentiment lstm layer
                if self.configuration['mil']:
                    visual_attentions_sentiment_temp = [lstm_layer_words_sentiment_soft[j][i][: len(words)].detach().cpu().numpy()
                                                        for j in range(self.category_num)]
                    for j in range(self.category_num):
                        c_label = label[i][j].detach().cpu().numpy().tolist()
                        if c_label == 1:
                            visual_attentions_sentiment = []
                            labels_sentiment = []
                            sentiment_true_index = int(label[i][j + self.category_num].detach().cpu().numpy().tolist())
                            if sentiment_true_index == -100:
                                continue
                            titles_sentiment = ['true: %s - pred: %s - %s' % (str(self.polarites[sentiment_true_index]),
                                                            str(pred_sentiment[j][i].detach().cpu().numpy()),
                                                            str(self.polarites))]
                            c_attention = embedding_layer_category_alphas[j][i][: len(words)].detach().cpu().numpy()
                            visual_attentions_sentiment.append(c_attention)
                            labels_sentiment.append(self.categories[j].split('/')[0])

                            s_distributions = visual_attentions_sentiment_temp[j]
                            for k in range(self.polarity_num):
                                labels_sentiment.append(self.polarites[k])
                                visual_attentions_sentiment.append(s_distributions[:, k])
                            titles_sentiment.extend([''] * 3)
                            attention_visualizer.plot_multi_attentions_of_sentence(words, visual_attentions_sentiment,
                                                                                   labels_sentiment,
                                                                                   titles_sentiment)
        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {
            'accuracy': self._accuracy.get_metric(reset),
            'category_f1': self._f1.get_metric(reset)['fscore']
        }
        return metrics


class SentenceConsituencyAwareModelV4(TextInAllAspectSentimentOutModel):
    """
    SentenceConsituencyAwareModelV2 + 针对某个属性，属性分类任务找到能预测这个属性但不能预测其它属性的所有节点
    (内部节点和外部节点，偏向上层节点)，情感分类任务基于找到的节点做属性的情感分类；想证明的是，拥有内部节点的
    知识，能获得的比只通过找到的叶子节点的信息进行情感分类更好的效果；内部节点用作属性分类，比叶子节点有时会有
    更好的效果，因为某些属性，并不是由一个词就能指示的，而是由一小片文本的语义确定的；
    """
    def __init__(self, word_embedder: TextFieldEmbedder, position_embedder: TextFieldEmbedder,
                 aspect_embedder: TextFieldEmbedder, categories: list, polarities: list, vocab: Vocabulary,
                 configuration: dict, category_loss_weight=1, sentiment_loss_weight=1):
        super().__init__(vocab, category_loss_weight=category_loss_weight, sentiment_loss_weight=sentiment_loss_weight)
        self.configuration = configuration
        self.word_embedder = word_embedder
        self.position_embedder = position_embedder
        self.aspect_embedder = aspect_embedder
        self.categories = categories
        self.polarites = polarities
        self.category_num = len(categories)
        self.polarity_num = len(polarities)
        self.category_loss = nn.BCEWithLogitsLoss()
        self.sentiment_loss = nn.CrossEntropyLoss()
        self._accuracy = metrics.CategoricalAccuracy()
        self._f1 = allennlp_metrics.BinaryF1(0.5)

        word_embedding_dim = word_embedder.get_output_dim()
        if self.configuration['lstm_or_fc_after_embedding_layer'] == 'fc':
            self.embedding_layer_fc = nn.Linear(word_embedding_dim, word_embedding_dim, bias=True)
        elif self.configuration['lstm_or_fc_after_embedding_layer'] == 'gcn':
            self.gc_aspect_category = DglGraphConvolution(word_embedding_dim, word_embedding_dim, configuration)
        elif self.configuration['lstm_or_fc_after_embedding_layer'] == 'bilstm':
            self.embedding_layer_lstm = torch.nn.LSTM(word_embedding_dim, int(word_embedding_dim / 2), batch_first=True,
                                                      bidirectional=True, num_layers=1)
        else:
            self.embedding_layer_lstm = torch.nn.LSTM(word_embedding_dim, word_embedding_dim, batch_first=True,
                                                      bidirectional=False, num_layers=1)
        # todo 当一句话里没有提到某个aspect category，attention应该关注什么？
        # (1) 类似于bert，添加一个额外的词，attend nothing
        # (2) 关注停用词
        self.embedding_layer_aspect_attentions = [AttentionInHtt(word_embedding_dim,
                                                                 word_embedding_dim)
                                                  for _ in range(self.category_num)]
        self.embedding_layer_aspect_attentions = nn.ModuleList(self.embedding_layer_aspect_attentions)

        lstm_input_size = word_embedding_dim
        if self.configuration['position']:
            lstm_input_size += position_embedder.get_output_dim()
        num_layers = self.configuration['lstm_layer_num_in_lstm']
        self.lstm = torch.nn.LSTM(lstm_input_size, int(word_embedding_dim / 2), batch_first=True,
                                  bidirectional=True, num_layers=num_layers, dropout=0.5)

        self.category_fcs = [nn.Linear(word_embedding_dim, 1) for _ in range(self.category_num)]
        self.category_fcs = nn.ModuleList(self.category_fcs)

        sentiment_fc_input_size = word_embedding_dim
        if not self.configuration['share_sentiment_classifier']:
            self.sentiment_fcs = [nn.Sequential(nn.Linear(sentiment_fc_input_size, sentiment_fc_input_size),
                                                nn.ReLU(),
                                                nn.Linear(sentiment_fc_input_size, self.polarity_num))
                                  for _ in range(self.category_num)]
            self.sentiment_fcs = nn.ModuleList(self.sentiment_fcs)
        else:
            self.sentiment_fc = nn.Sequential(nn.Linear(sentiment_fc_input_size, sentiment_fc_input_size),
                                              nn.ReLU(),
                                              nn.Linear(sentiment_fc_input_size, self.polarity_num))

        self.dropout_after_embedding_layer = nn.Dropout(0.5)
        self.dropout_after_lstm_layer = nn.Dropout(0.5)
        # 只负责生成针对aspect category的节点表示
        aspect_graph = self.configuration['aspect_graph']
        if aspect_graph == 'attention':
            self.gnn_for_aspect = DglGraphAttentionForAspectCategoryWithDottedLines(word_embedding_dim, word_embedding_dim,
                                                                     self.embedding_layer_aspect_attentions, configuration)
        else:
            raise NotImplementedError(aspect_graph)

        sentiment_graph = self.configuration['sentiment_graph']
        if sentiment_graph == 'average':
            self.gnn_for_sentiment = DglGraphAverage(word_embedding_dim, word_embedding_dim, self.configuration)
        elif sentiment_graph == 'gcn':
            self.gnn_for_sentiment = DglGraphConvolution(word_embedding_dim, word_embedding_dim, configuration)
        elif sentiment_graph == 'gat':
            self.gnn_for_sentiment = GAT(word_embedding_dim, word_embedding_dim, word_embedding_dim, 4, self.configuration)
        else:
            raise NotImplementedError(sentiment_graph)

    def set_grad_for_acd_parameter(self, requires_grad=True):
        acd_layers = []
        if self.configuration['lstm_or_fc_after_embedding_layer'] == 'fc':
            acd_layers.append(self.embedding_layer_fc)
        elif self.configuration['lstm_or_fc_after_embedding_layer'] == 'gcn':
            acd_layers.append(self.gc_aspect_category)
        else:
            acd_layers.append(self.embedding_layer_lstm)
        acd_layers.append(self.embedding_layer_aspect_attentions)
        acd_layers.append(self.category_fcs)
        acd_layers.append(self.gnn_for_aspect)
        for layer in acd_layers:
            for name, value in layer.named_parameters():
                value.requires_grad = requires_grad

    def forward(self, tokens: Dict[str, torch.Tensor], tokens_for_graph: Dict[str, torch.Tensor], label: torch.Tensor, position: torch.Tensor,
                polarity_mask: torch.Tensor, sample: list, aspects: torch.Tensor=None) -> torch.Tensor:
        mask_for_graph = get_text_field_mask(tokens_for_graph)

        mask = get_text_field_mask(tokens)

        max_len_for_graph = mask_for_graph.size()[1]
        max_len = mask.size()[1]
        padder = nn.ConstantPad2d((0, 0, 0, max_len_for_graph - max_len), 0)

        graphs = [e['graph'] for e in sample]
        graphs_padded = self.pad_dgl_graph(graphs, max_len_for_graph)

        graphs_with_dotted_line = [e['graph_with_dotted_line'] for e in sample]
        graphs_with_dotted_line_padded = self.pad_dgl_graph(graphs_with_dotted_line, max_len_for_graph)

        word_embeddings = self.word_embedder(tokens)
        if self.configuration['lstm_or_fc_after_embedding_layer'] == 'fc':
            word_embeddings_fc = self.embedding_layer_fc(word_embeddings)
        elif self.configuration['lstm_or_fc_after_embedding_layer'] == 'gcn':
            max_len = tokens['tokens'].size()[1]
            graphs = [e[3] for e in sample]
            graphs_padded = self.pad_dgl_graph(graphs, max_len)
            word_embeddings_fc = F.relu(self.gc_aspect_category(word_embeddings, graphs_padded))
        else:
            word_embeddings_fc, (_, _) = self.embedding_layer_lstm(word_embeddings)

        aspects_separate = [{'aspect': aspects['aspect'][:, i].unsqueeze(1)} for i in range(self.category_num)]
        aspect_embeddings_singles = [self.aspect_embedder(aspects_separate[i]).squeeze(1) for i in
                                     range(self.category_num)]
        aspects_seprate_repeat = [{'aspect': aspects_separate[i]['aspect'].expand_as(tokens['tokens'])} for i in
                                  range(self.category_num)]
        aspect_embeddings_separate = [self.aspect_embedder(aspects_seprate_repeat[i]) for i in range(self.category_num)]

        embedding_layer_category_outputs = []
        embedding_layer_category_alphas = []
        embedding_layer_sentiment_outputs = []
        embedding_layer_sentiment_alphas = []

        embedding_layer_category_alphas = []
        # 在word_embeddings_fc上做图卷积，attention是基于图卷积的输出
        word_embeddings_fc_for_graph = padder(word_embeddings_fc)

        if self.configuration['attention_warmup']:
            graph_output_for_aspect = self.gnn_for_aspect(word_embeddings_fc_for_graph, graphs_with_dotted_line_padded)
            lstm_result_for_graph = self.gnn_for_sentiment(word_embeddings_fc_for_graph, graphs_padded)
        else:
            lstm_input = word_embeddings
            if self.configuration['position']:
                position_embeddings = self.position_embedder(position)
                lstm_input = torch.cat([word_embeddings, position_embeddings], dim=-1)
            lstm_input = self.dropout_after_embedding_layer(lstm_input)
            lstm_result, _ = self.lstm(lstm_input)
            lstm_result = self.dropout_after_lstm_layer(lstm_result)
            # 在lstm_result上做图卷积
            lstm_result_padded_for_graph = padder(lstm_result)

            acd_sc_encoder_mode = self.configuration['acd_sc_encoder_mode']
            if acd_sc_encoder_mode == 'simple':
                graph_output_for_aspect = self.gnn_for_aspect(word_embeddings_fc_for_graph,
                                                              graphs_with_dotted_line_padded)
                lstm_result_for_graph = self.gnn_for_sentiment(word_embeddings_fc_for_graph, graphs_padded)
            elif acd_sc_encoder_mode == 'complex':
                graph_output_for_aspect = self.gnn_for_aspect(lstm_result_padded_for_graph,
                                                              graphs_with_dotted_line_padded)
                lstm_result_for_graph = self.gnn_for_sentiment(lstm_result_padded_for_graph, graphs_padded)
            else:
                # mixed
                graph_output_for_aspect = self.gnn_for_aspect(word_embeddings_fc_for_graph,
                                                              graphs_with_dotted_line_padded)
                lstm_result_for_graph = self.gnn_for_sentiment(lstm_result_padded_for_graph, graphs_padded)

        for i in range(self.category_num):
            embedding_layer_aspect_attention = self.embedding_layer_aspect_attentions[i]
            # todo 根据节点的深度调整权重，使得偏向上层节点
            alpha = embedding_layer_aspect_attention(graph_output_for_aspect, mask_for_graph)
            embedding_layer_category_alphas.append(alpha)

            category_output = self.element_wise_mul(graph_output_for_aspect, alpha, return_not_sum_result=False)
            embedding_layer_category_outputs.append(category_output)

        # lstm_result_with_position = torch.cat([lstm_result, position_embeddings], dim=-1)
        lstm_layer_sentiment_outputs = []
        lstm_layer_words_sentiment_soft = []

        for i in range(self.category_num):
            # sentiment
            # word_representation_for_sentiment = torch.cat([graph_output2, lstm_result], dim=-1)
            word_representation_for_sentiment = lstm_result_for_graph
            sentiment_alpha = embedding_layer_category_alphas[i]
            if self.configuration['mil']:
                sentiment_alpha = sentiment_alpha.unsqueeze(1)
                if not self.configuration['share_sentiment_classifier']:
                    words_sentiment = self.sentiment_fcs[i](word_representation_for_sentiment)
                else:
                    words_sentiment = self.sentiment_fc(word_representation_for_sentiment)
                if self.configuration['mil_softmax']:
                    words_sentiment_soft = torch.softmax(words_sentiment, dim=-1)
                    lstm_layer_words_sentiment_soft.append(words_sentiment_soft)
                else:
                    words_sentiment_soft = words_sentiment
                    lstm_layer_words_sentiment_soft.append(torch.softmax(words_sentiment, dim=-1))
                sentiment_output = torch.matmul(sentiment_alpha, words_sentiment_soft).squeeze(1)  # batch_size x 2*hidden_dim
                lstm_layer_sentiment_outputs.append(sentiment_output)
            else:
                sentiment_output_temp = self.element_wise_mul(word_representation_for_sentiment, sentiment_alpha,
                                                         return_not_sum_result=False)
                if not self.configuration['share_sentiment_classifier']:
                    sentiment_output = self.sentiment_fcs[i](sentiment_output_temp)
                else:
                    sentiment_output = self.sentiment_fc(sentiment_output_temp)
                lstm_layer_sentiment_outputs.append(sentiment_output)

        final_category_outputs = []
        final_category_outputs_auxillary = []
        final_sentiment_outputs = []
        for i in range(self.category_num):
            category_output = embedding_layer_category_outputs[i]
            for j in range(self.category_num):
                final_category_output = self.category_fcs[j](category_output)
                if i == j:
                    final_category_outputs.append(final_category_output)
                else:
                    final_category_outputs_auxillary.append(final_category_output)

            final_sentiment_output = lstm_layer_sentiment_outputs[i]
            final_sentiment_outputs.append(final_sentiment_output)

        output = {}
        output['alpha'] = embedding_layer_category_alphas
        if label is not None:
            category_labels = []
            polarity_labels = []
            polarity_masks = []
            for i in range(self.category_num):
                category_labels.append(label[:, i])
                polarity_labels.append(label[:, i + self.category_num])
                polarity_masks.append(polarity_mask[:, i])
            loss = 0
            total_category_loss = 0
            total_sentiment_loss = 0
            for i in range(self.category_num):
                category_temp_loss = self.category_loss(final_category_outputs[i].squeeze(dim=-1), category_labels[i])
                sentiment_temp_loss = self.sentiment_loss(final_sentiment_outputs[i], polarity_labels[i].long())
                total_category_loss += category_temp_loss
                if not self.configuration['only_acd']:
                    total_sentiment_loss += sentiment_temp_loss

            loss = self.category_loss_weight * total_category_loss + self.sentiment_loss_weight * total_sentiment_loss

            interactive_loss = 0
            for i in range(len(final_category_outputs_auxillary)):
                label_temp = torch.zeros_like(category_labels[0])
                interactive_loss_temp = self.category_loss(final_category_outputs_auxillary[i].squeeze(dim=-1),
                                                           label_temp)
                interactive_loss += interactive_loss_temp
            loss += (interactive_loss * self.configuration['interactive_loss_lamda'] / (self.category_num - 1))

            # Sparse Regularization Orthogonal Regularization
            if self.configuration['sparse_reg'] or self.configuration['orthogonal_reg']:
                reg_loss = 0
                for j in range(len(sample)):
                    polarity_mask_of_one_sample = polarity_mask[j]
                    category_alpha_of_one_sample = [embedding_layer_category_alphas[k][j] for k in range(self.category_num)]
                    category_alpha_of_mentioned = []
                    category_alpha_of_not_mentioned = []
                    for k in range(self.category_num):
                        if polarity_mask_of_one_sample[k] == 1:
                            category_alpha_of_mentioned.append(category_alpha_of_one_sample[k].unsqueeze(0))
                        else:
                            category_alpha_of_not_mentioned.append(category_alpha_of_one_sample[k].unsqueeze(0))
                    if len(category_alpha_of_not_mentioned) != 0:
                        category_alpha_of_not_mentioned = torch.cat(category_alpha_of_not_mentioned, dim=0)
                        category_alpha_of_not_mentioned = torch.mean(category_alpha_of_not_mentioned, dim=0, keepdim=True)
                        category_alpha_of_mentioned.append(category_alpha_of_not_mentioned)

                    category_eye = torch.eye(len(category_alpha_of_mentioned))
                    category_alpha_of_mentioned = torch.cat(category_alpha_of_mentioned, dim=0)
                    category_alpha_similarity = torch.mm(category_alpha_of_mentioned, category_alpha_of_mentioned.t())

                    if self.configuration['sparse_reg'] and self.configuration['orthogonal_reg']:
                        pass
                    elif self.configuration['sparse_reg']:
                        for m in range(len(category_alpha_of_mentioned)):
                            for n in range(len(category_alpha_of_mentioned)):
                                if m != n:
                                    category_eye[m][n] = category_alpha_similarity[m][n]
                    else:
                        # orthogonal_reg
                        for m in range(len(category_alpha_of_mentioned)):
                            category_eye[m][m] = category_alpha_similarity[m][m]
                    # category_eye = nn_util.move_to_device(category_eye, self.configuration['device'])
                    category_eye = category_eye.to(self.configuration['device'])
                    category_alpha_similarity = category_alpha_similarity.to(self.configuration['device'])
                    category_reg_loss = category_alpha_similarity - category_eye
                    category_reg_loss = torch.norm(category_reg_loss)
                    reg_loss += category_reg_loss
                loss += (reg_loss * self.configuration['attention_lamda'] / len(sample))

            # sentiment accuracy
            sentiment_logit = torch.cat(final_sentiment_outputs)
            sentiment_label = torch.cat(polarity_labels)
            sentiment_mask = torch.cat(polarity_masks)
            self._accuracy(sentiment_logit, sentiment_label, sentiment_mask)

            # category f1
            final_category_outputs_prob = [torch.sigmoid(e) for e in final_category_outputs]
            category_prob = torch.cat(final_category_outputs_prob).squeeze()
            category_label = torch.cat(category_labels)
            self._f1(category_prob, category_label)

            output['loss'] = loss

        # visualize attention
        pred_category = [torch.sigmoid(e) for e in final_category_outputs]
        pred_sentiment = [torch.nn.functional.softmax(e, dim=-1) for e in final_sentiment_outputs]
        output['pred_category'] = pred_category
        output['pred_sentiment'] = pred_sentiment
        if self.configuration['visualize_attention']:
            for i in range(len(sample)):
                words = sample[i]['words_for_graph']
                print('inner_nodes:-------------------------------------------')
                for inner_node_index, inner_node in enumerate(sample[i]['inner_nodes']):
                    print('%d-%s-%s' % (len(sample[i]['words']) + inner_node_index, inner_node.labels[0], inner_node.text))

                attention_labels = [e.split('/')[0] for e in self.categories]

                # 只可视化mams
                label_true = label[i].detach().cpu().numpy()[: self.category_num]
                if sum(label_true) <= 1:
                    continue
                # category
                visual_attentions_category = [embedding_layer_category_alphas[j][i][: len(words)].detach().cpu().numpy()
                                              for j in range(self.category_num)]
                titles = ['true: %s - pred: %s' % (str(label[i][j].detach().cpu().numpy()),
                                                   str(pred_category[j][i].detach().cpu().numpy()))
                          for j in range(self.category_num)]
                savefig_filepath = super()._get_model_visualization_picture_filepath(self.configuration, words)
                attention_visualizer.plot_multi_attentions_of_sentence(words, visual_attentions_category,
                                                                       attention_labels, titles,
                                                                       savefig_filepath=savefig_filepath)

                # sentiment embedding layer
                # visual_attentions = [embedding_layer_sentiment_alphas[j][i][: len(words)].detach().cpu().numpy()
                #                      for j in range(self.category_num)]
                # titles = ['true: %s - pred: %s - %s' % (str(label[i + self.category_num][j].detach().cpu().numpy()),
                #                                         str(pred_sentiment[j][i].detach().cpu().numpy()),
                #                                         str(self.polarites))
                #           for j in range(self.category_num)]
                # attention_visualizer.plot_multi_attentions_of_sentence(words, visual_attentions, attention_labels,
                #                                                        titles)

                # sentiment lstm layer
                # visual_attentions = [lstm_layer_sentiment_alphas[j][i][: len(words)].detach().cpu().numpy()
                #                      for j in range(self.category_num)]
                # titles = ['true: %s - pred: %s - %s' % (str(label[i + self.category_num][j].detach().cpu().numpy()),
                #                                         str(pred_sentiment[j][i].detach().cpu().numpy()),
                #                                         str(self.polarites))
                #           for j in range(self.category_num)]
                # attention_visualizer.plot_multi_attentions_of_sentence(words, visual_attentions, attention_labels,
                #                                                        titles)

                # sentiment lstm layer
                if self.configuration['mil']:
                    visual_attentions_sentiment_temp = [lstm_layer_words_sentiment_soft[j][i][: len(words)].detach().cpu().numpy()
                                                        for j in range(self.category_num)]
                    for j in range(self.category_num):
                        c_label = label[i][j].detach().cpu().numpy().tolist()
                        if c_label == 1:
                            visual_attentions_sentiment = []
                            labels_sentiment = []
                            sentiment_true_index = int(label[i][j + self.category_num].detach().cpu().numpy().tolist())
                            if sentiment_true_index == -100:
                                continue
                            titles_sentiment = ['true: %s - pred: %s - %s' % (str(self.polarites[sentiment_true_index]),
                                                            str(pred_sentiment[j][i].detach().cpu().numpy()),
                                                            str(self.polarites))]
                            c_attention = embedding_layer_category_alphas[j][i][: len(words)].detach().cpu().numpy()
                            visual_attentions_sentiment.append(c_attention)
                            labels_sentiment.append(self.categories[j].split('/')[0])

                            s_distributions = visual_attentions_sentiment_temp[j]
                            for k in range(self.polarity_num):
                                labels_sentiment.append(self.polarites[k])
                                visual_attentions_sentiment.append(s_distributions[:, k])
                            titles_sentiment.extend([''] * 3)
                            attention_visualizer.plot_multi_attentions_of_sentence(words, visual_attentions_sentiment,
                                                                                   labels_sentiment,
                                                                                   titles_sentiment)
        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {
            'accuracy': self._accuracy.get_metric(reset),
            'category_f1': self._f1.get_metric(reset)['fscore']
        }
        return metrics


class SentenceConsituencyAwareModelV5(TextInAllAspectSentimentOutModel):
    """
    SentenceConsituencyAwareModelV2 + 针对某个属性，属性分类任务找到能预测这个属性但不能预测其它属性的所有节点
    (内部节点和外部节点，偏向上层节点)，情感分类任务基于找到的节点做属性的情感分类；想证明的是，拥有内部节点的
    知识，能获得的比只通过找到的叶子节点的信息进行情感分类更好的效果；内部节点用作属性分类，比叶子节点有时会有
    更好的效果，因为某些属性，并不是由一个词就能指示的，而是由一小片文本的语义确定的；
    """
    def __init__(self, word_embedder: TextFieldEmbedder, position_embedder: TextFieldEmbedder,
                 aspect_embedder: TextFieldEmbedder, categories: list, polarities: list, vocab: Vocabulary,
                 configuration: dict, category_loss_weight=1, sentiment_loss_weight=1):
        super().__init__(vocab, category_loss_weight=category_loss_weight, sentiment_loss_weight=sentiment_loss_weight)
        self.configuration = configuration
        self.word_embedder = word_embedder
        self.position_embedder = position_embedder
        self.aspect_embedder = aspect_embedder
        self.categories = categories
        self.polarites = polarities
        self.category_num = len(categories)
        self.polarity_num = len(polarities)
        self.category_loss = nn.BCEWithLogitsLoss()
        self.sentiment_loss = nn.CrossEntropyLoss()
        self._accuracy = metrics.CategoricalAccuracy()
        self._f1 = allennlp_metrics.BinaryF1(0.5)

        word_embedding_dim = word_embedder.get_output_dim()
        if self.configuration['lstm_or_fc_after_embedding_layer'] == 'fc':
            self.embedding_layer_fc = nn.Linear(word_embedding_dim, word_embedding_dim, bias=True)
        elif self.configuration['lstm_or_fc_after_embedding_layer'] == 'gcn':
            self.gc_aspect_category = DglGraphConvolution(word_embedding_dim, word_embedding_dim, configuration)
        elif self.configuration['lstm_or_fc_after_embedding_layer'] == 'bilstm':
            self.embedding_layer_lstm = torch.nn.LSTM(word_embedding_dim, int(word_embedding_dim / 2), batch_first=True,
                                                      bidirectional=True, num_layers=1)
        else:
            self.embedding_layer_lstm = torch.nn.LSTM(word_embedding_dim, word_embedding_dim, batch_first=True,
                                                      bidirectional=False, num_layers=1)
        # todo 当一句话里没有提到某个aspect category，attention应该关注什么？
        # (1) 类似于bert，添加一个额外的词，attend nothing
        # (2) 关注停用词
        self.embedding_layer_aspect_attentions = [AttentionInHtt(word_embedding_dim,
                                                                 word_embedding_dim)
                                                  for _ in range(self.category_num)]
        self.embedding_layer_aspect_attentions = nn.ModuleList(self.embedding_layer_aspect_attentions)

        lstm_input_size = word_embedding_dim
        if self.configuration['position']:
            lstm_input_size += position_embedder.get_output_dim()
        num_layers = self.configuration['lstm_layer_num_in_lstm']
        self.lstm = torch.nn.LSTM(lstm_input_size, int(word_embedding_dim / 2), batch_first=True,
                                  bidirectional=True, num_layers=num_layers, dropout=0.5)

        self.category_fcs = [nn.Linear(word_embedding_dim, 1) for _ in range(self.category_num)]
        self.category_fcs = nn.ModuleList(self.category_fcs)

        sentiment_fc_input_size = word_embedding_dim
        if not self.configuration['share_sentiment_classifier']:
            self.sentiment_fcs = [nn.Sequential(nn.Linear(sentiment_fc_input_size, sentiment_fc_input_size),
                                                nn.ReLU(),
                                                nn.Linear(sentiment_fc_input_size, self.polarity_num))
                                  for _ in range(self.category_num)]
            self.sentiment_fcs = nn.ModuleList(self.sentiment_fcs)
        else:
            self.sentiment_fc = nn.Sequential(nn.Linear(sentiment_fc_input_size, sentiment_fc_input_size),
                                              nn.ReLU(),
                                              nn.Linear(sentiment_fc_input_size, self.polarity_num))

        self.dropout_after_embedding_layer = nn.Dropout(0.5)
        self.dropout_after_lstm_layer = nn.Dropout(0.5)
        # 只负责生成针对aspect category的节点表示
        aspect_graph = self.configuration['aspect_graph']
        if aspect_graph == 'attention':
            self.gnn_for_aspect = DglGraphAttentionForAspectCategoryWithDottedLines(word_embedding_dim, word_embedding_dim,
                                                                     self.embedding_layer_aspect_attentions, configuration)
        elif aspect_graph == 'average':
            self.gnn_for_aspect = DglGraphAverageForAspectCategory(word_embedding_dim, word_embedding_dim, self.configuration)
        elif aspect_graph == 'gcn':
            self.gnn_for_aspect = DglGraphConvolutionForAspectCategory(word_embedding_dim, word_embedding_dim, self.configuration)
        elif aspect_graph == 'gat':
            self.gnn_for_aspect = GATForAspectCategory(word_embedding_dim, word_embedding_dim, self.configuration)
        else:
            raise NotImplementedError(aspect_graph)

        sentiment_graph = self.configuration['sentiment_graph']
        if sentiment_graph == 'average':
            self.gnn_for_sentiment = DglGraphAverage(word_embedding_dim, word_embedding_dim, self.configuration)
        elif sentiment_graph == 'gcn':
            self.gnn_for_sentiment = DglGraphConvolution(word_embedding_dim, word_embedding_dim, configuration)
        elif sentiment_graph == 'gat':
            self.gnn_for_sentiment = GAT(word_embedding_dim, word_embedding_dim, word_embedding_dim, 4, self.configuration)
        else:
            raise NotImplementedError(sentiment_graph)

    def set_grad_for_acd_parameter(self, requires_grad=True):
        acd_layers = []
        if self.configuration['lstm_or_fc_after_embedding_layer'] == 'fc':
            acd_layers.append(self.embedding_layer_fc)
        elif self.configuration['lstm_or_fc_after_embedding_layer'] == 'gcn':
            acd_layers.append(self.gc_aspect_category)
        else:
            acd_layers.append(self.embedding_layer_lstm)
        acd_layers.append(self.embedding_layer_aspect_attentions)
        acd_layers.append(self.category_fcs)
        acd_layers.append(self.gnn_for_aspect)
        for layer in acd_layers:
            for name, value in layer.named_parameters():
                value.requires_grad = requires_grad

    def forward(self, tokens: Dict[str, torch.Tensor], tokens_for_graph: Dict[str, torch.Tensor], label: torch.Tensor, position: torch.Tensor,
                polarity_mask: torch.Tensor, sample: list, aspects: torch.Tensor=None) -> torch.Tensor:
        mask_for_graph = get_text_field_mask(tokens_for_graph)

        mask = get_text_field_mask(tokens)

        max_len_for_graph = mask_for_graph.size()[1]
        max_len = mask.size()[1]
        padder = nn.ConstantPad2d((0, 0, 0, max_len_for_graph - max_len), 0)

        graphs = [e['graph'] for e in sample]
        graphs_padded = self.pad_dgl_graph(graphs, max_len_for_graph)

        graphs_with_dotted_line = [e['graph_with_dotted_line'] for e in sample]
        graphs_with_dotted_line_padded = self.pad_dgl_graph(graphs_with_dotted_line, max_len_for_graph)

        word_embeddings = self.word_embedder(tokens)
        if self.configuration['lstm_or_fc_after_embedding_layer'] == 'fc':
            word_embeddings_fc = self.embedding_layer_fc(word_embeddings)
        elif self.configuration['lstm_or_fc_after_embedding_layer'] == 'gcn':
            max_len = tokens['tokens'].size()[1]
            graphs = [e[3] for e in sample]
            graphs_padded = self.pad_dgl_graph(graphs, max_len)
            word_embeddings_fc = F.relu(self.gc_aspect_category(word_embeddings, graphs_padded))
        else:
            word_embeddings_fc, (_, _) = self.embedding_layer_lstm(word_embeddings)

        aspects_separate = [{'aspect': aspects['aspect'][:, i].unsqueeze(1)} for i in range(self.category_num)]
        aspect_embeddings_singles = [self.aspect_embedder(aspects_separate[i]).squeeze(1) for i in
                                     range(self.category_num)]
        aspects_seprate_repeat = [{'aspect': aspects_separate[i]['aspect'].expand_as(tokens['tokens'])} for i in
                                  range(self.category_num)]
        aspect_embeddings_separate = [self.aspect_embedder(aspects_seprate_repeat[i]) for i in range(self.category_num)]

        embedding_layer_category_outputs = []
        embedding_layer_category_alphas = []
        embedding_layer_sentiment_outputs = []
        embedding_layer_sentiment_alphas = []

        embedding_layer_category_alphas = []
        # 在word_embeddings_fc上做图卷积，attention是基于图卷积的输出
        word_embeddings_fc_for_graph = padder(word_embeddings_fc)

        if self.configuration['attention_warmup']:
            graph_output_for_aspect, lstm_result_for_graph = self.gnn_for_aspect(word_embeddings_fc_for_graph,
                                                                                 word_embeddings_fc_for_graph,
                                                                                 graphs_with_dotted_line_padded)
        else:
            lstm_input = word_embeddings
            if self.configuration['position']:
                position_embeddings = self.position_embedder(position)
                lstm_input = torch.cat([word_embeddings, position_embeddings], dim=-1)
            lstm_input = self.dropout_after_embedding_layer(lstm_input)
            lstm_result, _ = self.lstm(lstm_input)
            lstm_result = self.dropout_after_lstm_layer(lstm_result)
            # 在lstm_result上做图卷积
            lstm_result_padded_for_graph = padder(lstm_result)

            acd_sc_encoder_mode = self.configuration['acd_sc_encoder_mode']
            if acd_sc_encoder_mode == 'simple':
                graph_output_for_aspect, lstm_result_for_graph = self.gnn_for_aspect(word_embeddings_fc_for_graph,
                                                                                     word_embeddings_fc_for_graph,
                                                                                     graphs_with_dotted_line_padded)
            elif acd_sc_encoder_mode == 'complex':
                graph_output_for_aspect, lstm_result_for_graph = self.gnn_for_aspect(lstm_result_padded_for_graph,
                                                                                     lstm_result_padded_for_graph,
                                                                                     graphs_with_dotted_line_padded)
            else:
                # mixed
                graph_output_for_aspect, _ = self.gnn_for_aspect(word_embeddings_fc_for_graph,
                                                              word_embeddings_fc_for_graph,
                                                              graphs_with_dotted_line_padded)
                lstm_result_for_graph = self.gnn_for_sentiment(lstm_result_padded_for_graph, graphs_padded)

        for i in range(self.category_num):
            embedding_layer_aspect_attention = self.embedding_layer_aspect_attentions[i]
            # todo 根据节点的深度调整权重，使得偏向上层节点
            alpha = embedding_layer_aspect_attention(graph_output_for_aspect, mask_for_graph)
            embedding_layer_category_alphas.append(alpha)

            category_output = self.element_wise_mul(graph_output_for_aspect, alpha, return_not_sum_result=False)
            embedding_layer_category_outputs.append(category_output)

        # lstm_result_with_position = torch.cat([lstm_result, position_embeddings], dim=-1)
        lstm_layer_sentiment_outputs = []
        lstm_layer_words_sentiment_soft = []

        for i in range(self.category_num):
            # sentiment
            # word_representation_for_sentiment = torch.cat([graph_output2, lstm_result], dim=-1)
            word_representation_for_sentiment = lstm_result_for_graph
            sentiment_alpha = embedding_layer_category_alphas[i]
            if self.configuration['mil']:
                sentiment_alpha = sentiment_alpha.unsqueeze(1)
                if not self.configuration['share_sentiment_classifier']:
                    words_sentiment = self.sentiment_fcs[i](word_representation_for_sentiment)
                else:
                    words_sentiment = self.sentiment_fc(word_representation_for_sentiment)
                if self.configuration['mil_softmax']:
                    words_sentiment_soft = torch.softmax(words_sentiment, dim=-1)
                    lstm_layer_words_sentiment_soft.append(words_sentiment_soft)
                else:
                    words_sentiment_soft = words_sentiment
                    lstm_layer_words_sentiment_soft.append(torch.softmax(words_sentiment, dim=-1))
                sentiment_output = torch.matmul(sentiment_alpha, words_sentiment_soft).squeeze(1)  # batch_size x 2*hidden_dim
                lstm_layer_sentiment_outputs.append(sentiment_output)
            else:
                sentiment_output_temp = self.element_wise_mul(word_representation_for_sentiment, sentiment_alpha,
                                                         return_not_sum_result=False)
                if not self.configuration['share_sentiment_classifier']:
                    sentiment_output = self.sentiment_fcs[i](sentiment_output_temp)
                else:
                    sentiment_output = self.sentiment_fc(sentiment_output_temp)
                lstm_layer_sentiment_outputs.append(sentiment_output)

        final_category_outputs = []
        final_category_outputs_auxillary = []
        final_sentiment_outputs = []
        for i in range(self.category_num):
            category_output = embedding_layer_category_outputs[i]
            for j in range(self.category_num):
                final_category_output = self.category_fcs[j](category_output)
                if i == j:
                    final_category_outputs.append(final_category_output)
                else:
                    final_category_outputs_auxillary.append(final_category_output)

            final_sentiment_output = lstm_layer_sentiment_outputs[i]
            final_sentiment_outputs.append(final_sentiment_output)

        output = {}
        output['alpha'] = embedding_layer_category_alphas
        if label is not None:
            category_labels = []
            polarity_labels = []
            polarity_masks = []
            for i in range(self.category_num):
                category_labels.append(label[:, i])
                polarity_labels.append(label[:, i + self.category_num])
                polarity_masks.append(polarity_mask[:, i])
            loss = 0
            total_category_loss = 0
            total_sentiment_loss = 0
            for i in range(self.category_num):
                category_temp_loss = self.category_loss(final_category_outputs[i].squeeze(dim=-1), category_labels[i])
                sentiment_temp_loss = self.sentiment_loss(final_sentiment_outputs[i], polarity_labels[i].long())
                total_category_loss += category_temp_loss
                if not self.configuration['only_acd']:
                    total_sentiment_loss += sentiment_temp_loss

            loss = self.category_loss_weight * total_category_loss + self.sentiment_loss_weight * total_sentiment_loss

            interactive_loss = 0
            for i in range(len(final_category_outputs_auxillary)):
                label_temp = torch.zeros_like(category_labels[0])
                interactive_loss_temp = self.category_loss(final_category_outputs_auxillary[i].squeeze(dim=-1),
                                                           label_temp)
                interactive_loss += interactive_loss_temp
            loss += (interactive_loss * self.configuration['interactive_loss_lamda'] / (self.category_num - 1))

            # Sparse Regularization Orthogonal Regularization
            if self.configuration['sparse_reg'] or self.configuration['orthogonal_reg']:
                reg_loss = 0
                for j in range(len(sample)):
                    polarity_mask_of_one_sample = polarity_mask[j]
                    category_alpha_of_one_sample = [embedding_layer_category_alphas[k][j] for k in range(self.category_num)]
                    category_alpha_of_mentioned = []
                    category_alpha_of_not_mentioned = []
                    for k in range(self.category_num):
                        if polarity_mask_of_one_sample[k] == 1:
                            category_alpha_of_mentioned.append(category_alpha_of_one_sample[k].unsqueeze(0))
                        else:
                            category_alpha_of_not_mentioned.append(category_alpha_of_one_sample[k].unsqueeze(0))
                    if len(category_alpha_of_not_mentioned) != 0:
                        category_alpha_of_not_mentioned = torch.cat(category_alpha_of_not_mentioned, dim=0)
                        category_alpha_of_not_mentioned = torch.mean(category_alpha_of_not_mentioned, dim=0, keepdim=True)
                        category_alpha_of_mentioned.append(category_alpha_of_not_mentioned)

                    category_eye = torch.eye(len(category_alpha_of_mentioned))
                    category_alpha_of_mentioned = torch.cat(category_alpha_of_mentioned, dim=0)
                    category_alpha_similarity = torch.mm(category_alpha_of_mentioned, category_alpha_of_mentioned.t())

                    if self.configuration['sparse_reg'] and self.configuration['orthogonal_reg']:
                        pass
                    elif self.configuration['sparse_reg']:
                        for m in range(len(category_alpha_of_mentioned)):
                            for n in range(len(category_alpha_of_mentioned)):
                                if m != n:
                                    category_eye[m][n] = category_alpha_similarity[m][n]
                    else:
                        # orthogonal_reg
                        for m in range(len(category_alpha_of_mentioned)):
                            category_eye[m][m] = category_alpha_similarity[m][m]
                    # category_eye = nn_util.move_to_device(category_eye, self.configuration['device'])
                    category_eye = category_eye.to(self.configuration['device'])
                    category_alpha_similarity = category_alpha_similarity.to(self.configuration['device'])
                    category_reg_loss = category_alpha_similarity - category_eye
                    category_reg_loss = torch.norm(category_reg_loss)
                    reg_loss += category_reg_loss
                loss += (reg_loss * self.configuration['attention_lamda'] / len(sample))

            # sentiment accuracy
            sentiment_logit = torch.cat(final_sentiment_outputs)
            sentiment_label = torch.cat(polarity_labels)
            sentiment_mask = torch.cat(polarity_masks)
            self._accuracy(sentiment_logit, sentiment_label, sentiment_mask)

            # category f1
            final_category_outputs_prob = [torch.sigmoid(e) for e in final_category_outputs]
            category_prob = torch.cat(final_category_outputs_prob).squeeze()
            category_label = torch.cat(category_labels)
            self._f1(category_prob, category_label)

            output['loss'] = loss

        # visualize attention
        pred_category = [torch.sigmoid(e) for e in final_category_outputs]
        pred_sentiment = [torch.nn.functional.softmax(e, dim=-1) for e in final_sentiment_outputs]
        output['pred_category'] = pred_category
        output['pred_sentiment'] = pred_sentiment
        if self.configuration['visualize_attention']:
            for i in range(len(sample)):
                words = sample[i]['words_for_graph']
                print('inner_nodes:-------------------------------------------')
                for inner_node_index, inner_node in enumerate(sample[i]['inner_nodes']):
                    print('%d-%s-%s' % (len(sample[i]['words']) + inner_node_index, inner_node.labels[0], inner_node.text))

                attention_labels = [e.split('/')[0] for e in self.categories]

                # 只可视化mams
                label_true = label[i].detach().cpu().numpy()[: self.category_num]
                if sum(label_true) <= 1:
                    continue
                # category
                visual_attentions_category = [embedding_layer_category_alphas[j][i][: len(words)].detach().cpu().numpy()
                                              for j in range(self.category_num)]
                titles = ['true: %s - pred: %s' % (str(label[i][j].detach().cpu().numpy()),
                                                   str(pred_category[j][i].detach().cpu().numpy()))
                          for j in range(self.category_num)]
                savefig_filepath = super()._get_model_visualization_picture_filepath(self.configuration, words)
                attention_visualizer.plot_multi_attentions_of_sentence(words, visual_attentions_category,
                                                                       attention_labels, titles,
                                                                       savefig_filepath=savefig_filepath)

                # sentiment embedding layer
                # visual_attentions = [embedding_layer_sentiment_alphas[j][i][: len(words)].detach().cpu().numpy()
                #                      for j in range(self.category_num)]
                # titles = ['true: %s - pred: %s - %s' % (str(label[i + self.category_num][j].detach().cpu().numpy()),
                #                                         str(pred_sentiment[j][i].detach().cpu().numpy()),
                #                                         str(self.polarites))
                #           for j in range(self.category_num)]
                # attention_visualizer.plot_multi_attentions_of_sentence(words, visual_attentions, attention_labels,
                #                                                        titles)

                # sentiment lstm layer
                # visual_attentions = [lstm_layer_sentiment_alphas[j][i][: len(words)].detach().cpu().numpy()
                #                      for j in range(self.category_num)]
                # titles = ['true: %s - pred: %s - %s' % (str(label[i + self.category_num][j].detach().cpu().numpy()),
                #                                         str(pred_sentiment[j][i].detach().cpu().numpy()),
                #                                         str(self.polarites))
                #           for j in range(self.category_num)]
                # attention_visualizer.plot_multi_attentions_of_sentence(words, visual_attentions, attention_labels,
                #                                                        titles)

                # sentiment lstm layer
                if self.configuration['mil']:
                    visual_attentions_sentiment_temp = [lstm_layer_words_sentiment_soft[j][i][: len(words)].detach().cpu().numpy()
                                                        for j in range(self.category_num)]
                    for j in range(self.category_num):
                        c_label = label[i][j].detach().cpu().numpy().tolist()
                        if c_label == 1:
                            visual_attentions_sentiment = []
                            labels_sentiment = []
                            sentiment_true_index = int(label[i][j + self.category_num].detach().cpu().numpy().tolist())
                            if sentiment_true_index == -100:
                                continue
                            titles_sentiment = ['true: %s - pred: %s - %s' % (str(self.polarites[sentiment_true_index]),
                                                            str(pred_sentiment[j][i].detach().cpu().numpy()),
                                                            str(self.polarites))]
                            c_attention = embedding_layer_category_alphas[j][i][: len(words)].detach().cpu().numpy()
                            visual_attentions_sentiment.append(c_attention)
                            labels_sentiment.append(self.categories[j].split('/')[0])

                            s_distributions = visual_attentions_sentiment_temp[j]
                            for k in range(self.polarity_num):
                                labels_sentiment.append(self.polarites[k])
                                visual_attentions_sentiment.append(s_distributions[:, k])
                            titles_sentiment.extend([''] * 3)
                            attention_visualizer.plot_multi_attentions_of_sentence(words, visual_attentions_sentiment,
                                                                                   labels_sentiment,
                                                                                   titles_sentiment)
        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {
            'accuracy': self._accuracy.get_metric(reset),
            'category_f1': self._f1.get_metric(reset)['fscore']
        }
        return metrics


class SentenceConsituencyAwareModelV6(TextInAllAspectSentimentOutModel):
    """
    SentenceConsituencyAwareModelV2 + 针对某个属性，属性分类任务找到能预测这个属性但不能预测其它属性的所有节点
    (内部节点和外部节点，偏向上层节点)，情感分类任务基于找到的节点做属性的情感分类；想证明的是，拥有内部节点的
    知识，能获得的比只通过找到的叶子节点的信息进行情感分类更好的效果；内部节点用作属性分类，比叶子节点有时会有
    更好的效果，因为某些属性，并不是由一个词就能指示的，而是由一小片文本的语义确定的；
    """
    def __init__(self, word_embedder: TextFieldEmbedder, position_embedder: TextFieldEmbedder,
                 aspect_embedder: TextFieldEmbedder, categories: list, polarities: list, vocab: Vocabulary,
                 configuration: dict, category_loss_weight=1, sentiment_loss_weight=1):
        super().__init__(vocab, category_loss_weight=category_loss_weight, sentiment_loss_weight=sentiment_loss_weight)
        self.configuration = configuration
        self.word_embedder = word_embedder
        self.position_embedder = position_embedder
        self.aspect_embedder = aspect_embedder
        self.categories = categories
        self.polarites = polarities
        self.category_num = len(categories)
        self.polarity_num = len(polarities)
        self.category_loss = nn.BCEWithLogitsLoss()
        self.sentiment_loss = nn.CrossEntropyLoss()
        self._accuracy = metrics.CategoricalAccuracy()
        self._f1 = allennlp_metrics.BinaryF1(0.5)

        word_embedding_dim = word_embedder.get_output_dim()
        aspect_encoder_input_size = word_embedding_dim
        if self.configuration['position']:
            aspect_encoder_input_size += position_embedder.get_output_dim()
        if self.configuration['lstm_or_fc_after_embedding_layer'] == 'fc':
            self.embedding_layer_fc = nn.Linear(aspect_encoder_input_size, word_embedding_dim, bias=True)
        elif self.configuration['lstm_or_fc_after_embedding_layer'] == 'gcn':
            self.gc_aspect_category = DglGraphConvolution(aspect_encoder_input_size, word_embedding_dim, configuration)
        elif self.configuration['lstm_or_fc_after_embedding_layer'] == 'bilstm':
            self.embedding_layer_lstm = torch.nn.LSTM(aspect_encoder_input_size, int(word_embedding_dim / 2), batch_first=True,
                                                      bidirectional=True, num_layers=1)
        else:
            self.embedding_layer_lstm = torch.nn.LSTM(aspect_encoder_input_size, word_embedding_dim, batch_first=True,
                                                      bidirectional=False, num_layers=1)
        # todo 当一句话里没有提到某个aspect category，attention应该关注什么？
        # (1) 类似于bert，添加一个额外的词，attend nothing
        # (2) 关注停用词
        self.embedding_layer_aspect_attentions = [AttentionInHtt(word_embedding_dim,
                                                                 word_embedding_dim)
                                                  for _ in range(self.category_num)]
        self.embedding_layer_aspect_attentions = nn.ModuleList(self.embedding_layer_aspect_attentions)

        lstm_input_size = word_embedding_dim
        if self.configuration['position']:
            lstm_input_size += position_embedder.get_output_dim()
        num_layers = self.configuration['lstm_layer_num_in_lstm']
        self.lstm = torch.nn.LSTM(lstm_input_size, int(word_embedding_dim / 2), batch_first=True,
                                  bidirectional=True, num_layers=num_layers, dropout=0.5)

        self.category_fcs = [nn.Linear(word_embedding_dim, 1) for _ in range(self.category_num)]
        self.category_fcs = nn.ModuleList(self.category_fcs)

        sentiment_fc_input_size = word_embedding_dim
        if not self.configuration['share_sentiment_classifier']:
            self.sentiment_fcs = [nn.Sequential(nn.Linear(sentiment_fc_input_size, sentiment_fc_input_size),
                                                nn.ReLU(),
                                                nn.Linear(sentiment_fc_input_size, self.polarity_num))
                                  for _ in range(self.category_num)]
            self.sentiment_fcs = nn.ModuleList(self.sentiment_fcs)
        else:
            self.sentiment_fc = nn.Sequential(nn.Linear(sentiment_fc_input_size, sentiment_fc_input_size),
                                              nn.ReLU(),
                                              nn.Linear(sentiment_fc_input_size, self.polarity_num))

        self.dropout_after_embedding_layer = nn.Dropout(0.5)
        self.dropout_after_lstm_layer = nn.Dropout(0.5)
        # 只负责生成针对aspect category的节点表示
        aspect_graph = self.configuration['aspect_graph']
        if aspect_graph == 'attention':
            # 1. 假设：两个词被attention关注，这两个词的平均也被attention关注。todo 需要验证这个假设是否成立？
            # 2. 对于给定属性，应该关注哪些内部节点？
            # (1) 关注子节点中包含与该属性相关的节点，
            # (2) 不关注子节点中包含其它属性相关的节点
            # 3. 之前的方法为什么会关注到其它属性的情感词？
            # (1) 因为很多情感词对不同属性通用的，只是基于情感分词本身很难判断这个情感词是否是描述该属性的
            # (2) 我们的解决方案，借助于情感词的父节点的其它子节点的信息来帮助判断这个情感分词是否是描述该属性的，
            # 具体的，如果情感词的父节点的其它子节点包含其它属性的信息，那么关注这个情感词，将会受到惩罚
            self.gnn_for_aspect = DglGraphAttentionForAspectCategoryWithDottedLines(word_embedding_dim, word_embedding_dim,
                                                                     self.embedding_layer_aspect_attentions, configuration)
        elif aspect_graph == 'average':
            self.gnn_for_aspect = DglGraphAverageForAspectCategory(word_embedding_dim, word_embedding_dim, self.configuration)
        elif aspect_graph == 'gcn':
            self.gnn_for_aspect = DglGraphConvolutionForAspectCategory(word_embedding_dim, word_embedding_dim, self.configuration)
        elif aspect_graph == 'gat':
            self.gnn_for_aspect = GATForAspectCategory(word_embedding_dim, word_embedding_dim, self.configuration)
        else:
            raise NotImplementedError(aspect_graph)

        sentiment_graph = self.configuration['sentiment_graph']
        if sentiment_graph == 'average':
            self.gnn_for_sentiment = DglGraphAverage(word_embedding_dim, word_embedding_dim, self.configuration)
        elif sentiment_graph == 'gcn':
            self.gnn_for_sentiment = DglGraphConvolution(word_embedding_dim, word_embedding_dim, configuration)
        elif sentiment_graph == 'gat':
            self.gnn_for_sentiment = GAT(word_embedding_dim, word_embedding_dim, word_embedding_dim, 4, self.configuration)
        else:
            raise NotImplementedError(sentiment_graph)

    def set_grad_for_acd_parameter(self, requires_grad=True):
        acd_layers = []
        if self.configuration['lstm_or_fc_after_embedding_layer'] == 'fc':
            acd_layers.append(self.embedding_layer_fc)
        elif self.configuration['lstm_or_fc_after_embedding_layer'] == 'gcn':
            acd_layers.append(self.gc_aspect_category)
        else:
            acd_layers.append(self.embedding_layer_lstm)
        acd_layers.append(self.embedding_layer_aspect_attentions)
        acd_layers.append(self.category_fcs)
        acd_layers.append(self.gnn_for_aspect)
        for layer in acd_layers:
            for name, value in layer.named_parameters():
                value.requires_grad = requires_grad

    def forward(self, tokens: Dict[str, torch.Tensor], tokens_for_graph: Dict[str, torch.Tensor], label: torch.Tensor, position: torch.Tensor,
                polarity_mask: torch.Tensor, sample: list, aspects: torch.Tensor=None) -> torch.Tensor:
        mask_for_graph = get_text_field_mask(tokens_for_graph)

        mask = get_text_field_mask(tokens)

        max_len_for_graph = mask_for_graph.size()[1]
        max_len = mask.size()[1]
        padder = nn.ConstantPad2d((0, 0, 0, max_len_for_graph - max_len), 0)

        graphs = [e['graph'] for e in sample]
        graphs_padded = self.pad_dgl_graph(graphs, max_len_for_graph)

        graphs_with_dotted_line = [e['graph_with_dotted_line'] for e in sample]
        graphs_with_dotted_line_padded = self.pad_dgl_graph(graphs_with_dotted_line, max_len_for_graph)
        if not self.configuration['graph_with_dotted_line']:
            graphs_with_dotted_line_padded = graphs_padded

        word_embeddings = self.word_embedder(tokens)
        if self.configuration['position']:
            position_embeddings = self.position_embedder(position)
            word_embeddings = torch.cat([word_embeddings, position_embeddings], dim=-1)
            word_embeddings = self.dropout_after_embedding_layer(word_embeddings)

        if self.configuration['lstm_or_fc_after_embedding_layer'] == 'fc':
            word_embeddings_fc = self.embedding_layer_fc(word_embeddings)
        elif self.configuration['lstm_or_fc_after_embedding_layer'] == 'gcn':
            max_len = tokens['tokens'].size()[1]
            graphs = [e[3] for e in sample]
            graphs_padded = self.pad_dgl_graph(graphs, max_len)
            word_embeddings_fc = F.relu(self.gc_aspect_category(word_embeddings, graphs_padded))
        else:
            word_embeddings_fc, (_, _) = self.embedding_layer_lstm(word_embeddings)

        aspects_separate = [{'aspect': aspects['aspect'][:, i].unsqueeze(1)} for i in range(self.category_num)]
        aspect_embeddings_singles = [self.aspect_embedder(aspects_separate[i]).squeeze(1) for i in
                                     range(self.category_num)]
        aspects_seprate_repeat = [{'aspect': aspects_separate[i]['aspect'].expand_as(tokens['tokens'])} for i in
                                  range(self.category_num)]
        aspect_embeddings_separate = [self.aspect_embedder(aspects_seprate_repeat[i]) for i in range(self.category_num)]

        embedding_layer_category_outputs = []
        embedding_layer_category_alphas = []
        embedding_layer_sentiment_outputs = []
        embedding_layer_sentiment_alphas = []

        embedding_layer_category_alphas = []
        # 在word_embeddings_fc上做图卷积，attention是基于图卷积的输出
        word_embeddings_fc_for_graph = padder(word_embeddings_fc)

        if self.configuration['attention_warmup']:
            graph_output_for_aspect, lstm_result_for_graph = self.gnn_for_aspect(word_embeddings_fc_for_graph,
                                                                                 word_embeddings_fc_for_graph,
                                                                                 graphs_with_dotted_line_padded)
        else:
            lstm_result, _ = self.lstm(word_embeddings)
            lstm_result = self.dropout_after_lstm_layer(lstm_result)
            # 在lstm_result上做图卷积
            lstm_result_padded_for_graph = padder(lstm_result)

            acd_sc_encoder_mode = self.configuration['acd_sc_encoder_mode']
            if acd_sc_encoder_mode == 'simple':
                graph_output_for_aspect, lstm_result_for_graph = self.gnn_for_aspect(word_embeddings_fc_for_graph,
                                                                                     word_embeddings_fc_for_graph,
                                                                                     graphs_with_dotted_line_padded)
            elif acd_sc_encoder_mode == 'complex':
                graph_output_for_aspect, lstm_result_for_graph = self.gnn_for_aspect(lstm_result_padded_for_graph,
                                                                                     lstm_result_padded_for_graph,
                                                                                     graphs_with_dotted_line_padded)
            else:
                # mixed
                graph_output_for_aspect, _ = self.gnn_for_aspect(word_embeddings_fc_for_graph,
                                                              word_embeddings_fc_for_graph,
                                                              graphs_with_dotted_line_padded)
                lstm_result_for_graph = self.gnn_for_sentiment(lstm_result_padded_for_graph, graphs_padded)

        for i in range(self.category_num):
            embedding_layer_aspect_attention = self.embedding_layer_aspect_attentions[i]
            # todo 根据节点的深度调整权重，使得偏向上层节点
            alpha = embedding_layer_aspect_attention(graph_output_for_aspect, mask_for_graph)
            embedding_layer_category_alphas.append(alpha)

            category_output = self.element_wise_mul(graph_output_for_aspect, alpha, return_not_sum_result=False)
            embedding_layer_category_outputs.append(category_output)

        # lstm_result_with_position = torch.cat([lstm_result, position_embeddings], dim=-1)
        lstm_layer_sentiment_outputs = []
        lstm_layer_words_sentiment_soft = []

        for i in range(self.category_num):
            # sentiment
            # word_representation_for_sentiment = torch.cat([graph_output2, lstm_result], dim=-1)
            word_representation_for_sentiment = lstm_result_for_graph
            sentiment_alpha = embedding_layer_category_alphas[i]
            if self.configuration['mil']:
                sentiment_alpha = sentiment_alpha.unsqueeze(1)
                if not self.configuration['share_sentiment_classifier']:
                    words_sentiment = self.sentiment_fcs[i](word_representation_for_sentiment)
                else:
                    words_sentiment = self.sentiment_fc(word_representation_for_sentiment)
                if self.configuration['mil_softmax']:
                    words_sentiment_soft = torch.softmax(words_sentiment, dim=-1)
                    lstm_layer_words_sentiment_soft.append(words_sentiment_soft)
                else:
                    words_sentiment_soft = words_sentiment
                    lstm_layer_words_sentiment_soft.append(torch.softmax(words_sentiment, dim=-1))
                sentiment_output = torch.matmul(sentiment_alpha, words_sentiment_soft).squeeze(1)  # batch_size x 2*hidden_dim
                lstm_layer_sentiment_outputs.append(sentiment_output)
            else:
                sentiment_output_temp = self.element_wise_mul(word_representation_for_sentiment, sentiment_alpha,
                                                         return_not_sum_result=False)
                if not self.configuration['share_sentiment_classifier']:
                    sentiment_output = self.sentiment_fcs[i](sentiment_output_temp)
                else:
                    sentiment_output = self.sentiment_fc(sentiment_output_temp)
                lstm_layer_sentiment_outputs.append(sentiment_output)

        final_category_outputs = []
        final_category_outputs_auxillary = []
        final_sentiment_outputs = []
        for i in range(self.category_num):
            category_output = embedding_layer_category_outputs[i]
            for j in range(self.category_num):
                final_category_output = self.category_fcs[j](category_output)
                if i == j:
                    final_category_outputs.append(final_category_output)
                else:
                    final_category_outputs_auxillary.append(final_category_output)

            final_sentiment_output = lstm_layer_sentiment_outputs[i]
            final_sentiment_outputs.append(final_sentiment_output)

        output = {}
        output['alpha'] = embedding_layer_category_alphas
        if label is not None:
            category_labels = []
            polarity_labels = []
            polarity_masks = []
            for i in range(self.category_num):
                category_labels.append(label[:, i])
                polarity_labels.append(label[:, i + self.category_num])
                polarity_masks.append(polarity_mask[:, i])
            loss = 0
            total_category_loss = 0
            total_sentiment_loss = 0
            for i in range(self.category_num):
                category_temp_loss = self.category_loss(final_category_outputs[i].squeeze(dim=-1), category_labels[i])
                sentiment_temp_loss = self.sentiment_loss(final_sentiment_outputs[i], polarity_labels[i].long())
                total_category_loss += category_temp_loss
                if not self.configuration['only_acd']:
                    total_sentiment_loss += sentiment_temp_loss

            loss = self.category_loss_weight * total_category_loss + self.sentiment_loss_weight * total_sentiment_loss

            interactive_loss = 0
            for i in range(len(final_category_outputs_auxillary)):
                label_temp = torch.zeros_like(category_labels[0])
                interactive_loss_temp = self.category_loss(final_category_outputs_auxillary[i].squeeze(dim=-1),
                                                           label_temp)
                interactive_loss += interactive_loss_temp
            loss += (interactive_loss * self.configuration['interactive_loss_lamda'] / (self.category_num - 1))

            # Sparse Regularization Orthogonal Regularization
            if self.configuration['sparse_reg'] or self.configuration['orthogonal_reg']:
                reg_loss = 0
                for j in range(len(sample)):
                    polarity_mask_of_one_sample = polarity_mask[j]
                    category_alpha_of_one_sample = [embedding_layer_category_alphas[k][j] for k in range(self.category_num)]
                    category_alpha_of_mentioned = []
                    category_alpha_of_not_mentioned = []
                    for k in range(self.category_num):
                        if polarity_mask_of_one_sample[k] == 1:
                            category_alpha_of_mentioned.append(category_alpha_of_one_sample[k].unsqueeze(0))
                        else:
                            category_alpha_of_not_mentioned.append(category_alpha_of_one_sample[k].unsqueeze(0))
                    if len(category_alpha_of_not_mentioned) != 0:
                        category_alpha_of_not_mentioned = torch.cat(category_alpha_of_not_mentioned, dim=0)
                        category_alpha_of_not_mentioned = torch.mean(category_alpha_of_not_mentioned, dim=0, keepdim=True)
                        category_alpha_of_mentioned.append(category_alpha_of_not_mentioned)

                    category_eye = torch.eye(len(category_alpha_of_mentioned))
                    category_alpha_of_mentioned = torch.cat(category_alpha_of_mentioned, dim=0)
                    category_alpha_similarity = torch.mm(category_alpha_of_mentioned, category_alpha_of_mentioned.t())

                    if self.configuration['sparse_reg'] and self.configuration['orthogonal_reg']:
                        pass
                    elif self.configuration['sparse_reg']:
                        for m in range(len(category_alpha_of_mentioned)):
                            for n in range(len(category_alpha_of_mentioned)):
                                if m != n:
                                    category_eye[m][n] = category_alpha_similarity[m][n]
                    else:
                        # orthogonal_reg
                        for m in range(len(category_alpha_of_mentioned)):
                            category_eye[m][m] = category_alpha_similarity[m][m]
                    # category_eye = nn_util.move_to_device(category_eye, self.configuration['device'])
                    category_eye = category_eye.to(self.configuration['device'])
                    category_alpha_similarity = category_alpha_similarity.to(self.configuration['device'])
                    category_reg_loss = category_alpha_similarity - category_eye
                    category_reg_loss = torch.norm(category_reg_loss)
                    reg_loss += category_reg_loss
                loss += (reg_loss * self.configuration['attention_lamda'] / len(sample))

            # sentiment accuracy
            sentiment_logit = torch.cat(final_sentiment_outputs)
            sentiment_label = torch.cat(polarity_labels)
            sentiment_mask = torch.cat(polarity_masks)
            self._accuracy(sentiment_logit, sentiment_label, sentiment_mask)

            # category f1
            final_category_outputs_prob = [torch.sigmoid(e) for e in final_category_outputs]
            category_prob = torch.cat(final_category_outputs_prob).squeeze()
            category_label = torch.cat(category_labels)
            self._f1(category_prob, category_label)

            output['loss'] = loss

        # visualize attention
        pred_category = [torch.sigmoid(e) for e in final_category_outputs]
        pred_sentiment = [torch.nn.functional.softmax(e, dim=-1) for e in final_sentiment_outputs]
        output['pred_category'] = pred_category
        output['pred_sentiment'] = pred_sentiment
        if self.configuration['visualize_attention']:
            for i in range(len(sample)):
                words = sample[i]['words_for_graph']
                print('inner_nodes:-------------------------------------------')
                for inner_node_index, inner_node in enumerate(sample[i]['inner_nodes']):
                    print('%d-%s-%s' % (len(sample[i]['words']) + inner_node_index, inner_node.labels[0], inner_node.text))

                attention_labels = [e.split('/')[0] for e in self.categories]

                # 只可视化mams
                label_true = label[i].detach().cpu().numpy()[: self.category_num]
                if sum(label_true) <= 1:
                    continue
                # category
                visual_attentions_category = [embedding_layer_category_alphas[j][i][: len(words)].detach().cpu().numpy()
                                              for j in range(self.category_num)]
                titles = ['true: %s - pred: %s' % (str(label[i][j].detach().cpu().numpy()),
                                                   str(pred_category[j][i].detach().cpu().numpy()))
                          for j in range(self.category_num)]
                savefig_filepath = super()._get_model_visualization_picture_filepath(self.configuration, words)
                attention_visualizer.plot_multi_attentions_of_sentence(words, visual_attentions_category,
                                                                       attention_labels, titles,
                                                                       savefig_filepath=savefig_filepath)

                # sentiment embedding layer
                # visual_attentions = [embedding_layer_sentiment_alphas[j][i][: len(words)].detach().cpu().numpy()
                #                      for j in range(self.category_num)]
                # titles = ['true: %s - pred: %s - %s' % (str(label[i + self.category_num][j].detach().cpu().numpy()),
                #                                         str(pred_sentiment[j][i].detach().cpu().numpy()),
                #                                         str(self.polarites))
                #           for j in range(self.category_num)]
                # attention_visualizer.plot_multi_attentions_of_sentence(words, visual_attentions, attention_labels,
                #                                                        titles)

                # sentiment lstm layer
                # visual_attentions = [lstm_layer_sentiment_alphas[j][i][: len(words)].detach().cpu().numpy()
                #                      for j in range(self.category_num)]
                # titles = ['true: %s - pred: %s - %s' % (str(label[i + self.category_num][j].detach().cpu().numpy()),
                #                                         str(pred_sentiment[j][i].detach().cpu().numpy()),
                #                                         str(self.polarites))
                #           for j in range(self.category_num)]
                # attention_visualizer.plot_multi_attentions_of_sentence(words, visual_attentions, attention_labels,
                #                                                        titles)

                # sentiment lstm layer
                if self.configuration['mil']:
                    visual_attentions_sentiment_temp = [lstm_layer_words_sentiment_soft[j][i][: len(words)].detach().cpu().numpy()
                                                        for j in range(self.category_num)]
                    for j in range(self.category_num):
                        c_label = label[i][j].detach().cpu().numpy().tolist()
                        if c_label == 1:
                            visual_attentions_sentiment = []
                            labels_sentiment = []
                            sentiment_true_index = int(label[i][j + self.category_num].detach().cpu().numpy().tolist())
                            if sentiment_true_index == -100:
                                continue
                            titles_sentiment = ['true: %s - pred: %s - %s' % (str(self.polarites[sentiment_true_index]),
                                                            str(pred_sentiment[j][i].detach().cpu().numpy()),
                                                            str(self.polarites))]
                            c_attention = embedding_layer_category_alphas[j][i][: len(words)].detach().cpu().numpy()
                            visual_attentions_sentiment.append(c_attention)
                            labels_sentiment.append(self.categories[j].split('/')[0])

                            s_distributions = visual_attentions_sentiment_temp[j]
                            for k in range(self.polarity_num):
                                labels_sentiment.append(self.polarites[k])
                                visual_attentions_sentiment.append(s_distributions[:, k])
                            titles_sentiment.extend([''] * 3)
                            attention_visualizer.plot_multi_attentions_of_sentence(words, visual_attentions_sentiment,
                                                                                   labels_sentiment,
                                                                                   titles_sentiment)
        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {
            'accuracy': self._accuracy.get_metric(reset),
            'category_f1': self._f1.get_metric(reset)['fscore']
        }
        return metrics


class SentenceConsituencyAwareModelV7(TextInAllAspectSentimentOutModel):
    """
    SentenceConsituencyAwareModelV2 + 针对某个属性，属性分类任务找到能预测这个属性但不能预测其它属性的所有节点
    (内部节点和外部节点，偏向上层节点)，情感分类任务基于找到的节点做属性的情感分类；想证明的是，拥有内部节点的
    知识，能获得的比只通过找到的叶子节点的信息进行情感分类更好的效果；内部节点用作属性分类，比叶子节点有时会有
    更好的效果，因为某些属性，并不是由一个词就能指示的，而是由一小片文本的语义确定的；
    """
    def __init__(self, word_embedder: TextFieldEmbedder, position_embedder: TextFieldEmbedder,
                 aspect_embedder: TextFieldEmbedder, categories: list, polarities: list, vocab: Vocabulary,
                 configuration: dict, category_loss_weight=1, sentiment_loss_weight=1):
        super().__init__(vocab, category_loss_weight=category_loss_weight, sentiment_loss_weight=sentiment_loss_weight)
        self.configuration = configuration
        self.word_embedder = word_embedder
        self.position_embedder = position_embedder
        self.aspect_embedder = aspect_embedder
        self.categories = categories
        self.polarites = polarities
        self.category_num = len(categories)
        self.polarity_num = len(polarities)
        self.category_loss = nn.BCEWithLogitsLoss()
        self.sentiment_loss = nn.CrossEntropyLoss()
        self._accuracy = metrics.CategoricalAccuracy()
        self._f1 = allennlp_metrics.BinaryF1(0.5)

        word_embedding_dim = word_embedder.get_output_dim()
        aspect_encoder_input_size = word_embedding_dim
        if self.configuration['aspect_position']:
            aspect_encoder_input_size += position_embedder.get_output_dim()
        if self.configuration['lstm_or_fc_after_embedding_layer'] == 'fc':
            self.embedding_layer_fc = nn.Linear(aspect_encoder_input_size, word_embedding_dim, bias=True)
        elif self.configuration['lstm_or_fc_after_embedding_layer'] == 'bilstm':
            self.embedding_layer_lstm = torch.nn.LSTM(aspect_encoder_input_size, int(word_embedding_dim / 2), batch_first=True,
                                                      bidirectional=True, num_layers=1)
        else:
            self.embedding_layer_lstm = torch.nn.LSTM(aspect_encoder_input_size, word_embedding_dim, batch_first=True,
                                                      bidirectional=False, num_layers=1)
        # todo 当一句话里没有提到某个aspect category，attention应该关注什么？
        # (1) 类似于bert，添加一个额外的词，attend nothing
        # (2) 关注停用词
        self.embedding_layer_aspect_attentions = [AttentionInHtt(word_embedding_dim,
                                                                 word_embedding_dim)
                                                  for _ in range(self.category_num)]
        self.embedding_layer_aspect_attentions = nn.ModuleList(self.embedding_layer_aspect_attentions)

        lstm_input_size = word_embedding_dim
        if self.configuration['position']:
            lstm_input_size += position_embedder.get_output_dim()
        num_layers = self.configuration['lstm_layer_num_in_lstm']
        self.lstm = torch.nn.LSTM(lstm_input_size, int(word_embedding_dim / 2), batch_first=True,
                                  bidirectional=True, num_layers=num_layers, dropout=0.5)

        self.category_fcs = [nn.Linear(word_embedding_dim, 1) for _ in range(self.category_num)]
        self.category_fcs = nn.ModuleList(self.category_fcs)

        sentiment_fc_input_size = word_embedding_dim
        if not self.configuration['share_sentiment_classifier']:
            self.sentiment_fcs = [nn.Sequential(nn.Linear(sentiment_fc_input_size, sentiment_fc_input_size),
                                                nn.ReLU(),
                                                nn.Linear(sentiment_fc_input_size, self.polarity_num))
                                  for _ in range(self.category_num)]
            self.sentiment_fcs = nn.ModuleList(self.sentiment_fcs)
        else:
            self.sentiment_fc = nn.Sequential(nn.Linear(sentiment_fc_input_size, sentiment_fc_input_size),
                                              nn.ReLU(),
                                              nn.Linear(sentiment_fc_input_size, self.polarity_num))

        self.dropout_after_embedding_layer = nn.Dropout(0.5)
        self.dropout_after_lstm_layer = nn.Dropout(0.5)
        # 只负责生成针对aspect category的节点表示
        aspect_graph = self.configuration['aspect_graph']
        if aspect_graph == 'attention':
            # 1. 假设：两个词被attention关注，这两个词的平均也被attention关注。todo 需要验证这个假设是否成立？
            # 2. 对于给定属性，应该关注哪些内部节点？
            # (1) 关注子节点中包含与该属性相关的节点，
            # (2) 不关注子节点中包含其它属性相关的节点
            # 3. 之前的方法为什么会关注到其它属性的情感词？
            # (1) 因为很多情感词对不同属性通用的，只是基于情感分词本身很难判断这个情感词是否是描述该属性的
            # (2) 我们的解决方案，借助于情感词的父节点的其它子节点的信息来帮助判断这个情感分词是否是描述该属性的，
            # 具体的，如果情感词的父节点的其它子节点包含其它属性的信息，那么关注这个情感词，将会受到惩罚
            self.gnn_for_aspect = DglGraphAttentionForAspectCategory(word_embedding_dim, word_embedding_dim,
                                                                     self.embedding_layer_aspect_attentions,
                                                                     configuration)
        elif aspect_graph == 'attention_with_dotted_lines':
            self.gnn_for_aspect = DglGraphAttentionForAspectCategoryWithDottedLines(word_embedding_dim,
                                                                                    word_embedding_dim,
                                                                                    self.embedding_layer_aspect_attentions,
                                                                                    configuration)
        elif aspect_graph == 'average':
            self.gnn_for_aspect = DglGraphAverageForAspectCategory(word_embedding_dim, word_embedding_dim, self.configuration)
        elif aspect_graph == 'gcn':
            self.gnn_for_aspect = DglGraphConvolutionForAspectCategory(word_embedding_dim, word_embedding_dim, self.configuration)
        elif aspect_graph == 'gat':
            self.gnn_for_aspect = GATForAspectCategory(word_embedding_dim, word_embedding_dim, opt=self.configuration)
        else:
            raise NotImplementedError(aspect_graph)

        sentiment_graph = self.configuration['sentiment_graph']
        if sentiment_graph == 'average':
            self.gnn_for_sentiment = DglGraphAverage(word_embedding_dim, word_embedding_dim, self.configuration)
        elif sentiment_graph == 'gcn':
            self.gnn_for_sentiment = DglGraphConvolution(word_embedding_dim, word_embedding_dim, configuration)
        elif sentiment_graph == 'gat':
            self.gnn_for_sentiment = GAT(word_embedding_dim, word_embedding_dim, word_embedding_dim, 4, self.configuration)
        else:
            raise NotImplementedError(sentiment_graph)

    def set_grad_for_acd_parameter(self, requires_grad=True):
        acd_layers = []
        if self.configuration['lstm_or_fc_after_embedding_layer'] == 'fc':
            acd_layers.append(self.embedding_layer_fc)
        elif self.configuration['lstm_or_fc_after_embedding_layer'] == 'gcn':
            acd_layers.append(self.gc_aspect_category)
        else:
            acd_layers.append(self.embedding_layer_lstm)
        acd_layers.append(self.embedding_layer_aspect_attentions)
        acd_layers.append(self.category_fcs)
        acd_layers.append(self.gnn_for_aspect)
        for layer in acd_layers:
            for name, value in layer.named_parameters():
                value.requires_grad = requires_grad

    def forward(self, tokens: Dict[str, torch.Tensor], tokens_for_graph: Dict[str, torch.Tensor], label: torch.Tensor,
                position: torch.Tensor, polarity_mask: torch.Tensor, sample: list, aspects: torch.Tensor=None):
        mask = get_text_field_mask(tokens)

        mask_for_graph = get_text_field_mask(tokens_for_graph)

        max_len = mask.size()[1]
        max_len_for_graph = mask_for_graph.size()[1]
        padder = nn.ConstantPad2d((0, 0, 0, max_len_for_graph - max_len), 0)

        graphs = [e['graph'] for e in sample]
        graphs_padded = self.pad_dgl_graph(graphs, max_len_for_graph)

        graphs_with_dotted_line = [e['graph_with_dotted_line'] for e in sample]
        graphs_with_dotted_line_padded = self.pad_dgl_graph(graphs_with_dotted_line, max_len_for_graph)
        if not self.configuration['aspect_graph_with_dotted_line']:
            graphs_with_dotted_line_padded = graphs_padded

        position_embeddings = self.position_embedder(position)

        word_embeddings = self.word_embedder(tokens)

        word_embeddings_fc_input = word_embeddings
        if self.configuration['aspect_position']:
            word_embeddings_fc_input = torch.cat([word_embeddings, position_embeddings], dim=-1)
        if self.configuration['lstm_or_fc_after_embedding_layer'] == 'fc':
            word_embeddings_fc = self.embedding_layer_fc(word_embeddings_fc_input)
        else:
            word_embeddings_fc, (_, _) = self.embedding_layer_lstm(word_embeddings_fc_input)

        aspects_separate = [{'aspect': aspects['aspect'][:, i].unsqueeze(1)} for i in range(self.category_num)]
        aspect_embeddings_singles = [self.aspect_embedder(aspects_separate[i]).squeeze(1) for i in
                                     range(self.category_num)]
        aspects_seprate_repeat = [{'aspect': aspects_separate[i]['aspect'].expand_as(tokens['tokens'])} for i in
                                  range(self.category_num)]
        aspect_embeddings_separate = [self.aspect_embedder(aspects_seprate_repeat[i]) for i in range(self.category_num)]

        # 在word_embeddings_fc上做图卷积，attention是基于图卷积的输出
        word_embeddings_fc_for_graph = padder(word_embeddings_fc)

        lstm_input = word_embeddings
        if self.configuration['position']:
            lstm_input = torch.cat([lstm_input, position_embeddings], dim=-1)
            lstm_input = self.dropout_after_embedding_layer(lstm_input)
        lstm_result, _ = self.lstm(lstm_input)
        lstm_result = self.dropout_after_lstm_layer(lstm_result)
        # 在lstm_result上做图卷积
        lstm_result_padded_for_graph = padder(lstm_result)

        if self.configuration['attention_warmup']:
            graph_output_for_aspect = self.gnn_for_aspect(word_embeddings_fc_for_graph, graphs_with_dotted_line_padded)
            lstm_result_for_graph = graph_output_for_aspect
        else:
            acd_sc_encoder_mode = self.configuration['acd_sc_encoder_mode']
            if acd_sc_encoder_mode == 'same':
                acd_encoder_mode = self.configuration['acd_encoder_mode']
                if acd_encoder_mode == 'simple':
                    graph_output_for_aspect = self.gnn_for_aspect(word_embeddings_fc_for_graph,
                                                                  graphs_with_dotted_line_padded)
                    # lstm_result_for_graph = self.gnn_for_aspect(word_embeddings_fc_for_graph,
                    #                                             graphs_with_dotted_line_padded)
                elif acd_encoder_mode == 'complex':
                    graph_output_for_aspect = self.gnn_for_aspect(lstm_result_padded_for_graph,
                                                                  graphs_with_dotted_line_padded)
                    # lstm_result_for_graph = self.gnn_for_aspect(lstm_result_padded_for_graph,
                    #                                             graphs_with_dotted_line_padded)
                else:
                    raise NotImplementedError()
                lstm_result_for_graph = graph_output_for_aspect
            else:
                # mixed
                sentiment_encoder_with_own_gnn = self.configuration['sentiment_encoder_with_own_gnn']
                if sentiment_encoder_with_own_gnn:
                    graph_output_for_aspect = self.gnn_for_aspect(word_embeddings_fc_for_graph,
                                                                  graphs_with_dotted_line_padded)
                    lstm_result_for_graph = self.gnn_for_sentiment(lstm_result_padded_for_graph, graphs_padded)
                else:
                    # attention
                    graph_output_for_aspect, lstm_result_for_graph = self.gnn_for_aspect(word_embeddings_fc_for_graph,
                                                                                         lstm_result_padded_for_graph,
                                                                                         graphs_with_dotted_line_padded)
        embedding_layer_category_outputs = []
        embedding_layer_category_alphas = []
        embedding_layer_sentiment_outputs = []
        embedding_layer_sentiment_alphas = []
        for i in range(self.category_num):
            embedding_layer_aspect_attention = self.embedding_layer_aspect_attentions[i]
            # todo 根据节点的深度调整权重，使得偏向上层节点
            alpha = embedding_layer_aspect_attention(graph_output_for_aspect, mask_for_graph)
            embedding_layer_category_alphas.append(alpha)

            category_output = self.element_wise_mul(graph_output_for_aspect, alpha, return_not_sum_result=False)
            embedding_layer_category_outputs.append(category_output)

        # lstm_result_with_position = torch.cat([lstm_result, position_embeddings], dim=-1)
        lstm_layer_sentiment_outputs = []
        lstm_layer_words_sentiment_soft = []

        for i in range(self.category_num):
            # sentiment
            # word_representation_for_sentiment = torch.cat([graph_output2, lstm_result], dim=-1)
            word_representation_for_sentiment = lstm_result_for_graph
            sentiment_alpha = embedding_layer_category_alphas[i]
            if self.configuration['mil']:
                sentiment_alpha = sentiment_alpha.unsqueeze(1)
                if not self.configuration['share_sentiment_classifier']:
                    words_sentiment = self.sentiment_fcs[i](word_representation_for_sentiment)
                else:
                    words_sentiment = self.sentiment_fc(word_representation_for_sentiment)
                if self.configuration['mil_softmax']:
                    words_sentiment_soft = torch.softmax(words_sentiment, dim=-1)
                    lstm_layer_words_sentiment_soft.append(words_sentiment_soft)
                else:
                    words_sentiment_soft = words_sentiment
                    lstm_layer_words_sentiment_soft.append(torch.softmax(words_sentiment, dim=-1))
                sentiment_output = torch.matmul(sentiment_alpha, words_sentiment_soft).squeeze(1)  # batch_size x 2*hidden_dim
                lstm_layer_sentiment_outputs.append(sentiment_output)
            else:
                sentiment_output_temp = self.element_wise_mul(word_representation_for_sentiment, sentiment_alpha,
                                                         return_not_sum_result=False)
                if not self.configuration['share_sentiment_classifier']:
                    sentiment_output = self.sentiment_fcs[i](sentiment_output_temp)
                else:
                    sentiment_output = self.sentiment_fc(sentiment_output_temp)
                lstm_layer_sentiment_outputs.append(sentiment_output)

        final_category_outputs = []
        final_category_outputs_auxillary = []
        final_sentiment_outputs = []
        for i in range(self.category_num):
            category_output = embedding_layer_category_outputs[i]
            for j in range(self.category_num):
                final_category_output = self.category_fcs[j](category_output)
                if i == j:
                    final_category_outputs.append(final_category_output)
                else:
                    final_category_outputs_auxillary.append(final_category_output)

            final_sentiment_output = lstm_layer_sentiment_outputs[i]
            final_sentiment_outputs.append(final_sentiment_output)

        output = {}
        output['alpha'] = embedding_layer_category_alphas
        if label is not None:
            category_labels = []
            polarity_labels = []
            polarity_masks = []
            for i in range(self.category_num):
                category_labels.append(label[:, i])
                polarity_labels.append(label[:, i + self.category_num])
                polarity_masks.append(polarity_mask[:, i])
            loss = 0
            total_category_loss = 0
            total_sentiment_loss = 0
            for i in range(self.category_num):
                category_temp_loss = self.category_loss(final_category_outputs[i].squeeze(dim=-1), category_labels[i])
                sentiment_temp_loss = self.sentiment_loss(final_sentiment_outputs[i], polarity_labels[i].long())
                total_category_loss += category_temp_loss
                if not self.configuration['only_acd']:
                    total_sentiment_loss += sentiment_temp_loss

            loss = self.category_loss_weight * total_category_loss + self.sentiment_loss_weight * total_sentiment_loss

            interactive_loss = 0
            for i in range(len(final_category_outputs_auxillary)):
                label_temp = torch.zeros_like(category_labels[0])
                interactive_loss_temp = self.category_loss(final_category_outputs_auxillary[i].squeeze(dim=-1),
                                                           label_temp)
                interactive_loss += interactive_loss_temp
            loss += (interactive_loss * self.configuration['interactive_loss_lamda'] / (self.category_num - 1))

            # Sparse Regularization Orthogonal Regularization
            if self.configuration['sparse_reg'] or self.configuration['orthogonal_reg']:
                reg_loss = 0
                for j in range(len(sample)):
                    polarity_mask_of_one_sample = polarity_mask[j]
                    category_alpha_of_one_sample = [embedding_layer_category_alphas[k][j] for k in range(self.category_num)]
                    category_alpha_of_mentioned = []
                    category_alpha_of_not_mentioned = []
                    for k in range(self.category_num):
                        if polarity_mask_of_one_sample[k] == 1:
                            category_alpha_of_mentioned.append(category_alpha_of_one_sample[k].unsqueeze(0))
                        else:
                            category_alpha_of_not_mentioned.append(category_alpha_of_one_sample[k].unsqueeze(0))
                    if len(category_alpha_of_not_mentioned) != 0:
                        category_alpha_of_not_mentioned = torch.cat(category_alpha_of_not_mentioned, dim=0)
                        category_alpha_of_not_mentioned = torch.mean(category_alpha_of_not_mentioned, dim=0, keepdim=True)
                        category_alpha_of_mentioned.append(category_alpha_of_not_mentioned)

                    category_eye = torch.eye(len(category_alpha_of_mentioned))
                    category_alpha_of_mentioned = torch.cat(category_alpha_of_mentioned, dim=0)
                    category_alpha_similarity = torch.mm(category_alpha_of_mentioned, category_alpha_of_mentioned.t())

                    if self.configuration['sparse_reg'] and self.configuration['orthogonal_reg']:
                        pass
                    elif self.configuration['sparse_reg']:
                        for m in range(len(category_alpha_of_mentioned)):
                            for n in range(len(category_alpha_of_mentioned)):
                                if m != n:
                                    category_eye[m][n] = category_alpha_similarity[m][n]
                    else:
                        # orthogonal_reg
                        for m in range(len(category_alpha_of_mentioned)):
                            category_eye[m][m] = category_alpha_similarity[m][m]
                    # category_eye = nn_util.move_to_device(category_eye, self.configuration['device'])
                    category_eye = category_eye.to(self.configuration['device'])
                    category_alpha_similarity = category_alpha_similarity.to(self.configuration['device'])
                    category_reg_loss = category_alpha_similarity - category_eye
                    category_reg_loss = torch.norm(category_reg_loss)
                    reg_loss += category_reg_loss
                loss += (reg_loss * self.configuration['attention_lamda'] / len(sample))

            # sentiment accuracy
            sentiment_logit = torch.cat(final_sentiment_outputs)
            sentiment_label = torch.cat(polarity_labels)
            sentiment_mask = torch.cat(polarity_masks)
            self._accuracy(sentiment_logit, sentiment_label, sentiment_mask)

            # category f1
            final_category_outputs_prob = [torch.sigmoid(e) for e in final_category_outputs]
            category_prob = torch.cat(final_category_outputs_prob).squeeze()
            category_label = torch.cat(category_labels)
            self._f1(category_prob, category_label)

            output['loss'] = loss

        # visualize attention
        pred_category = [torch.sigmoid(e) for e in final_category_outputs]
        pred_sentiment = [torch.nn.functional.softmax(e, dim=-1) for e in final_sentiment_outputs]
        output['pred_category'] = pred_category
        output['pred_sentiment'] = pred_sentiment
        if self.configuration['visualize_attention']:
            for i in range(len(sample)):
                words = sample[i]['words_for_graph']
                print('inner_nodes:-------------------------------------------')
                for inner_node_index, inner_node in enumerate(sample[i]['inner_nodes']):
                    print('%d-%s-%s' % (len(sample[i]['words']) + inner_node_index, inner_node.labels[0], inner_node.text))

                attention_labels = [e.split('/')[0] for e in self.categories]

                # 只可视化mams
                label_true = label[i].detach().cpu().numpy()[: self.category_num]
                if sum(label_true) <= 1:
                    continue
                # category
                visual_attentions_category = [embedding_layer_category_alphas[j][i][: len(words)].detach().cpu().numpy()
                                              for j in range(self.category_num)]
                titles = ['true: %s - pred: %s' % (str(label[i][j].detach().cpu().numpy()),
                                                   str(pred_category[j][i].detach().cpu().numpy()))
                          for j in range(self.category_num)]
                savefig_filepath = super()._get_model_visualization_picture_filepath(self.configuration, words)
                attention_visualizer.plot_multi_attentions_of_sentence(words, visual_attentions_category,
                                                                       attention_labels, titles,
                                                                       savefig_filepath=savefig_filepath)

                # sentiment embedding layer
                # visual_attentions = [embedding_layer_sentiment_alphas[j][i][: len(words)].detach().cpu().numpy()
                #                      for j in range(self.category_num)]
                # titles = ['true: %s - pred: %s - %s' % (str(label[i + self.category_num][j].detach().cpu().numpy()),
                #                                         str(pred_sentiment[j][i].detach().cpu().numpy()),
                #                                         str(self.polarites))
                #           for j in range(self.category_num)]
                # attention_visualizer.plot_multi_attentions_of_sentence(words, visual_attentions, attention_labels,
                #                                                        titles)

                # sentiment lstm layer
                # visual_attentions = [lstm_layer_sentiment_alphas[j][i][: len(words)].detach().cpu().numpy()
                #                      for j in range(self.category_num)]
                # titles = ['true: %s - pred: %s - %s' % (str(label[i + self.category_num][j].detach().cpu().numpy()),
                #                                         str(pred_sentiment[j][i].detach().cpu().numpy()),
                #                                         str(self.polarites))
                #           for j in range(self.category_num)]
                # attention_visualizer.plot_multi_attentions_of_sentence(words, visual_attentions, attention_labels,
                #                                                        titles)

                # sentiment lstm layer
                if self.configuration['mil']:
                    visual_attentions_sentiment_temp = [lstm_layer_words_sentiment_soft[j][i][: len(words)].detach().cpu().numpy()
                                                        for j in range(self.category_num)]
                    for j in range(self.category_num):
                        c_label = label[i][j].detach().cpu().numpy().tolist()
                        if c_label == 1:
                            visual_attentions_sentiment = []
                            labels_sentiment = []
                            sentiment_true_index = int(label[i][j + self.category_num].detach().cpu().numpy().tolist())
                            if sentiment_true_index == -100:
                                continue
                            titles_sentiment = ['true: %s - pred: %s - %s' % (str(self.polarites[sentiment_true_index]),
                                                            str(pred_sentiment[j][i].detach().cpu().numpy()),
                                                            str(self.polarites))]
                            c_attention = embedding_layer_category_alphas[j][i][: len(words)].detach().cpu().numpy()
                            visual_attentions_sentiment.append(c_attention)
                            labels_sentiment.append(self.categories[j].split('/')[0])

                            s_distributions = visual_attentions_sentiment_temp[j]
                            for k in range(self.polarity_num):
                                labels_sentiment.append(self.polarites[k])
                                visual_attentions_sentiment.append(s_distributions[:, k])
                            titles_sentiment.extend([''] * 3)
                            attention_visualizer.plot_multi_attentions_of_sentence(words, visual_attentions_sentiment,
                                                                                   labels_sentiment,
                                                                                   titles_sentiment)
        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {
            'accuracy': self._accuracy.get_metric(reset),
            'category_f1': self._f1.get_metric(reset)['fscore']
        }
        return metrics


class SentenceConsituencyAwareModelV8(TextInAllAspectSentimentOutModel):
    """
    SentenceConsituencyAwareModelV2 + 针对某个属性，属性分类任务找到能预测这个属性但不能预测其它属性的所有节点
    (内部节点和外部节点，偏向上层节点)，情感分类任务基于找到的节点做属性的情感分类；想证明的是，拥有内部节点的
    知识，能获得的比只通过找到的叶子节点的信息进行情感分类更好的效果；内部节点用作属性分类，比叶子节点有时会有
    更好的效果，因为某些属性，并不是由一个词就能指示的，而是由一小片文本的语义确定的；
    """
    def __init__(self, word_embedder: TextFieldEmbedder, position_embedder: TextFieldEmbedder,
                 aspect_embedder: TextFieldEmbedder, categories: list, polarities: list, vocab: Vocabulary,
                 configuration: dict, category_loss_weight=1, sentiment_loss_weight=1):
        super().__init__(vocab, category_loss_weight=category_loss_weight, sentiment_loss_weight=sentiment_loss_weight)
        self.configuration = configuration
        self.word_embedder = word_embedder
        self.position_embedder = position_embedder
        self.aspect_embedder = aspect_embedder
        self.categories = categories
        self.polarites = polarities
        self.category_num = len(categories)
        self.polarity_num = len(polarities)
        self.category_loss = nn.BCEWithLogitsLoss()
        self.sentiment_loss = nn.CrossEntropyLoss()
        self._accuracy = metrics.CategoricalAccuracy()
        self._f1 = allennlp_metrics.BinaryF1(0.5)

        word_embedding_dim = word_embedder.get_output_dim()
        aspect_encoder_input_size = word_embedding_dim
        if self.configuration['aspect_position']:
            aspect_encoder_input_size += position_embedder.get_output_dim()
        if self.configuration['lstm_or_fc_after_embedding_layer'] == 'fc':
            self.embedding_layer_fc = nn.Linear(aspect_encoder_input_size, word_embedding_dim, bias=True)
        elif self.configuration['lstm_or_fc_after_embedding_layer'] == 'bilstm':
            self.embedding_layer_lstm = torch.nn.LSTM(aspect_encoder_input_size, int(word_embedding_dim / 2), batch_first=True,
                                                      bidirectional=True, num_layers=1)
        else:
            self.embedding_layer_lstm = torch.nn.LSTM(aspect_encoder_input_size, word_embedding_dim, batch_first=True,
                                                      bidirectional=False, num_layers=1)
        # todo 当一句话里没有提到某个aspect category，attention应该关注什么？
        # (1) 类似于bert，添加一个额外的词，attend nothing
        # (2) 关注停用词
        self.embedding_layer_aspect_attentions = [AttentionInHtt(word_embedding_dim,
                                                                 word_embedding_dim)
                                                  for _ in range(self.category_num)]
        self.embedding_layer_aspect_attentions = nn.ModuleList(self.embedding_layer_aspect_attentions)

        lstm_input_size = word_embedding_dim
        if self.configuration['position']:
            lstm_input_size += position_embedder.get_output_dim()
        num_layers = self.configuration['lstm_layer_num_in_lstm']
        self.lstm = torch.nn.LSTM(lstm_input_size, int(word_embedding_dim / 2), batch_first=True,
                                  bidirectional=True, num_layers=num_layers, dropout=0.5)

        self.category_fcs = [nn.Linear(word_embedding_dim, 1) for _ in range(self.category_num)]
        self.category_fcs = nn.ModuleList(self.category_fcs)

        if self.configuration['pipeline_with_acd']:
            self.category_fcs_pipeline_with_acd = [nn.Linear(word_embedding_dim, 1) for _ in range(self.category_num)]
            self.category_fcs_pipeline_with_acd = nn.ModuleList(self.category_fcs)

        sentiment_fc_input_size = word_embedding_dim
        if not self.configuration['share_sentiment_classifier']:
            self.sentiment_fcs = [nn.Sequential(nn.Linear(sentiment_fc_input_size, sentiment_fc_input_size),
                                                nn.ReLU(),
                                                nn.Linear(sentiment_fc_input_size, self.polarity_num))
                                  for _ in range(self.category_num)]
            self.sentiment_fcs = nn.ModuleList(self.sentiment_fcs)
        else:
            self.sentiment_fc = nn.Sequential(nn.Linear(sentiment_fc_input_size, sentiment_fc_input_size),
                                              nn.ReLU(),
                                              nn.Linear(sentiment_fc_input_size, self.polarity_num))

        self.dropout_after_embedding_layer = nn.Dropout(0.5)
        self.dropout_after_lstm_layer = nn.Dropout(0.5)
        # 只负责生成针对aspect category的节点表示
        aspect_graph = self.configuration['aspect_graph']
        if aspect_graph == 'attention':
            # 1. 假设：两个词被attention关注，这两个词的平均也被attention关注。todo 需要验证这个假设是否成立？
            # 2. 对于给定属性，应该关注哪些内部节点？
            # (1) 关注子节点中包含与该属性相关的节点，
            # (2) 不关注子节点中包含其它属性相关的节点
            # 3. 之前的方法为什么会关注到其它属性的情感词？
            # (1) 因为很多情感词对不同属性通用的，只是基于情感分词本身很难判断这个情感词是否是描述该属性的
            # (2) 我们的解决方案，借助于情感词的父节点的其它子节点的信息来帮助判断这个情感分词是否是描述该属性的，
            # 具体的，如果情感词的父节点的其它子节点包含其它属性的信息，那么关注这个情感词，将会受到惩罚
            self.gnn_for_aspect = DglGraphAttentionForAspectCategory(word_embedding_dim, word_embedding_dim,
                                                                     self.embedding_layer_aspect_attentions,
                                                                     configuration)
        elif aspect_graph == 'attention_with_dotted_lines':
            self.gnn_for_aspect = DglGraphAttentionForAspectCategoryWithDottedLines(word_embedding_dim,
                                                                                    word_embedding_dim,
                                                                                    self.embedding_layer_aspect_attentions,
                                                                                    configuration)
        elif aspect_graph == 'average':
            self.gnn_for_aspect = DglGraphAverageForAspectCategory(word_embedding_dim, word_embedding_dim, self.configuration)
        elif aspect_graph == 'gcn':
            self.gnn_for_aspect = DglGraphConvolutionForAspectCategory(word_embedding_dim, word_embedding_dim, self.configuration)
        elif aspect_graph == 'gat':
            self.gnn_for_aspect = GATForAspectCategory(word_embedding_dim, word_embedding_dim, opt=self.configuration)
        else:
            raise NotImplementedError(aspect_graph)

        sentiment_graph = self.configuration['sentiment_graph']
        if sentiment_graph == 'average':
            self.gnn_for_sentiment = DglGraphAverage(word_embedding_dim, word_embedding_dim, self.configuration)
        elif sentiment_graph == 'gcn':
            self.gnn_for_sentiment = DglGraphConvolution(word_embedding_dim, word_embedding_dim, configuration)
        elif sentiment_graph == 'gat':
            self.gnn_for_sentiment = GAT(word_embedding_dim, word_embedding_dim, word_embedding_dim, 4, self.configuration)
        else:
            raise NotImplementedError(sentiment_graph)

        if self.configuration['acd_encoder_mode_for_sentiment_attention_warmup'] == 'mixed':
            if self.configuration['gnn_for_sentiment_attention_warmup'] == 'average':
                  self.gnn_for_sentiment_attention_warmup = DglGraphAverage(word_embedding_dim, word_embedding_dim, self.configuration)
            else:
                self.gnn_for_sentiment_attention_warmup = GAT(word_embedding_dim, word_embedding_dim, word_embedding_dim, 4, self.configuration)

    def set_grad_for_acd_parameter(self, requires_grad=True):
        acd_layers = []
        if self.configuration['lstm_or_fc_after_embedding_layer'] == 'fc':
            acd_layers.append(self.embedding_layer_fc)
        elif self.configuration['lstm_or_fc_after_embedding_layer'] == 'gcn':
            acd_layers.append(self.gc_aspect_category)
        else:
            acd_layers.append(self.embedding_layer_lstm)
        acd_layers.append(self.embedding_layer_aspect_attentions)
        acd_layers.append(self.category_fcs)
        acd_layers.append(self.gnn_for_aspect)
        for layer in acd_layers:
            for name, value in layer.named_parameters():
                value.requires_grad = requires_grad

    def forward(self, tokens: Dict[str, torch.Tensor], tokens_for_graph: Dict[str, torch.Tensor], label: torch.Tensor,
                position: torch.Tensor, polarity_mask: torch.Tensor, sample: list, aspects: torch.Tensor=None):
        mask = get_text_field_mask(tokens)

        mask_for_graph = get_text_field_mask(tokens_for_graph)

        max_len = mask.size()[1]
        max_len_for_graph = mask_for_graph.size()[1]
        padder = nn.ConstantPad2d((0, 0, 0, max_len_for_graph - max_len), 0)

        graphs = [e['graph'] for e in sample]
        graphs_padded = self.pad_dgl_graph(graphs, max_len_for_graph)

        graphs_with_dotted_line = [e['graph_with_dotted_line'] for e in sample]
        graphs_with_dotted_line_padded = self.pad_dgl_graph(graphs_with_dotted_line, max_len_for_graph)
        if not self.configuration['aspect_graph_with_dotted_line']:
            graphs_with_dotted_line_padded = graphs_padded

        position_embeddings = self.position_embedder(position)

        word_embeddings = self.word_embedder(tokens)

        word_embeddings_fc_input = word_embeddings
        if self.configuration['aspect_position']:
            word_embeddings_fc_input = torch.cat([word_embeddings, position_embeddings], dim=-1)
        if self.configuration['lstm_or_fc_after_embedding_layer'] == 'fc':
            word_embeddings_fc = self.embedding_layer_fc(word_embeddings_fc_input)
        else:
            word_embeddings_fc, (_, _) = self.embedding_layer_lstm(word_embeddings_fc_input)

        aspects_separate = [{'aspect': aspects['aspect'][:, i].unsqueeze(1)} for i in range(self.category_num)]
        aspect_embeddings_singles = [self.aspect_embedder(aspects_separate[i]).squeeze(1) for i in
                                     range(self.category_num)]
        aspects_seprate_repeat = [{'aspect': aspects_separate[i]['aspect'].expand_as(tokens['tokens'])} for i in
                                  range(self.category_num)]
        aspect_embeddings_separate = [self.aspect_embedder(aspects_seprate_repeat[i]) for i in range(self.category_num)]

        # 在word_embeddings_fc上做图卷积，attention是基于图卷积的输出
        word_embeddings_fc_for_graph = padder(word_embeddings_fc)

        lstm_input = word_embeddings
        if self.configuration['position']:
            lstm_input = torch.cat([lstm_input, position_embeddings], dim=-1)
            lstm_input = self.dropout_after_embedding_layer(lstm_input)
        lstm_result, _ = self.lstm(lstm_input)
        lstm_result = self.dropout_after_lstm_layer(lstm_result)
        # 在lstm_result上做图卷积
        lstm_result_padded_for_graph = padder(lstm_result)

        if self.configuration['attention_warmup']:
            graph_output_for_aspect = self.gnn_for_aspect(word_embeddings_fc_for_graph, graphs_with_dotted_line_padded)
            if self.configuration['acd_encoder_mode_for_sentiment_attention_warmup'] == 'same':
                lstm_result_for_graph = graph_output_for_aspect
            else:
                lstm_result_for_graph = self.gnn_for_sentiment_attention_warmup(word_embeddings_fc_for_graph,
                                                                                graphs_with_dotted_line_padded)
        else:
            acd_sc_encoder_mode = self.configuration['acd_sc_encoder_mode']
            if acd_sc_encoder_mode == 'same':
                acd_encoder_mode = self.configuration['acd_encoder_mode']
                if acd_encoder_mode == 'simple':
                    graph_output_for_aspect = self.gnn_for_aspect(word_embeddings_fc_for_graph,
                                                                  graphs_with_dotted_line_padded)
                    # lstm_result_for_graph = self.gnn_for_aspect(word_embeddings_fc_for_graph,
                    #                                             graphs_with_dotted_line_padded)
                    lstm_result_for_graph = graph_output_for_aspect
                elif acd_encoder_mode == 'complex':
                    graph_output_for_aspect = self.gnn_for_aspect(lstm_result_padded_for_graph,
                                                                  graphs_with_dotted_line_padded)
                    # lstm_result_for_graph = self.gnn_for_aspect(lstm_result_padded_for_graph,
                    #                                             graphs_with_dotted_line_padded)
                    lstm_result_for_graph = graph_output_for_aspect
                else:
                    graph_output_for_aspect = self.gnn_for_aspect(word_embeddings_fc_for_graph,
                                                                  graphs_with_dotted_line_padded)
                    lstm_result_for_graph = self.gnn_for_sentiment(word_embeddings_fc_for_graph,
                                                                   graphs_with_dotted_line_padded)
                    if self.configuration['gat_visualization']:
                        graph_output_for_aspect, gat_alpha_for_aspect = graph_output_for_aspect
                        lstm_result_for_graph, gat_alpha_for_sentiment = lstm_result_for_graph

                        # average gat alpha
                        head_num = len(gat_alpha_for_aspect)
                        gat_alpha_for_aspect_average = copy.deepcopy(gat_alpha_for_aspect[0])
                        for gat_alpha_of_one_head in gat_alpha_for_aspect[1:]:
                            for k in range(len(gat_alpha_of_one_head[0])):
                                gat_alpha_for_aspect_average[0][k].add(gat_alpha_of_one_head[0][k])
                        for edge in gat_alpha_for_aspect_average[0]:
                            edge.divide(head_num)
                        gat_alpha_for_sentiment_average = copy.deepcopy(gat_alpha_for_sentiment[0])
                        for gat_alpha_of_one_head in gat_alpha_for_sentiment[1:]:
                            for k in range(len(gat_alpha_of_one_head[0])):
                                gat_alpha_for_sentiment_average[0][k].add(gat_alpha_of_one_head[0][k])
                        for edge in gat_alpha_for_sentiment_average[0]:
                            edge.divide(head_num)


                        print()
            else:
                # mixed
                sentiment_encoder_with_own_gnn = self.configuration['sentiment_encoder_with_own_gnn']
                if sentiment_encoder_with_own_gnn:
                    graph_output_for_aspect = self.gnn_for_aspect(word_embeddings_fc_for_graph,
                                                                  graphs_with_dotted_line_padded)
                    lstm_result_for_graph = self.gnn_for_sentiment(lstm_result_padded_for_graph, graphs_padded)
                else:
                    # attention
                    graph_output_for_aspect, lstm_result_for_graph = self.gnn_for_aspect(word_embeddings_fc_for_graph,
                                                                                         lstm_result_padded_for_graph,
                                                                                         graphs_with_dotted_line_padded)

        if not self.configuration['constituency_tree']:
            graph_output_for_aspect, lstm_result_for_graph = word_embeddings_fc, lstm_result
            mask_for_graph = mask

        embedding_layer_category_outputs = []
        embedding_layer_category_alphas = []
        embedding_layer_sentiment_outputs = []
        embedding_layer_sentiment_alphas = []
        for i in range(self.category_num):
            embedding_layer_aspect_attention = self.embedding_layer_aspect_attentions[i]
            # todo 根据节点的深度调整权重，使得偏向上层节点
            alpha = embedding_layer_aspect_attention(graph_output_for_aspect, mask_for_graph)
            embedding_layer_category_alphas.append(alpha)

            category_output = self.element_wise_mul(graph_output_for_aspect, alpha, return_not_sum_result=False)
            embedding_layer_category_outputs.append(category_output)

        # lstm_result_with_position = torch.cat([lstm_result, position_embeddings], dim=-1)
        lstm_layer_sentiment_outputs = []
        lstm_layer_words_sentiment_soft = []

        for i in range(self.category_num):
            # sentiment
            # word_representation_for_sentiment = torch.cat([graph_output2, lstm_result], dim=-1)
            word_representation_for_sentiment = lstm_result_for_graph
            sentiment_alpha = embedding_layer_category_alphas[i]
            if self.configuration['mil']:
                sentiment_alpha = sentiment_alpha.unsqueeze(1)
                if not self.configuration['share_sentiment_classifier']:
                    words_sentiment = self.sentiment_fcs[i](word_representation_for_sentiment)
                else:
                    words_sentiment = self.sentiment_fc(word_representation_for_sentiment)
                if self.configuration['mil_softmax']:
                    words_sentiment_soft = torch.softmax(words_sentiment, dim=-1)
                    lstm_layer_words_sentiment_soft.append(words_sentiment_soft)
                else:
                    words_sentiment_soft = words_sentiment
                    lstm_layer_words_sentiment_soft.append(torch.softmax(words_sentiment, dim=-1))
                sentiment_output = torch.matmul(sentiment_alpha, words_sentiment_soft).squeeze(1)  # batch_size x 2*hidden_dim
                lstm_layer_sentiment_outputs.append(sentiment_output)
            else:
                sentiment_output_temp = self.element_wise_mul(word_representation_for_sentiment, sentiment_alpha,
                                                         return_not_sum_result=False)
                if not self.configuration['share_sentiment_classifier']:
                    sentiment_output = self.sentiment_fcs[i](sentiment_output_temp)
                else:
                    sentiment_output = self.sentiment_fc(sentiment_output_temp)
                lstm_layer_sentiment_outputs.append(sentiment_output)

        final_category_outputs = []
        final_category_outputs_auxillary = []
        final_sentiment_outputs = []
        for i in range(self.category_num):
            category_output = embedding_layer_category_outputs[i]
            for j in range(self.category_num):
                final_category_output = self.category_fcs[j](category_output)
                if i == j:
                    final_category_outputs.append(final_category_output)
                else:
                    final_category_outputs_auxillary.append(final_category_output)

            final_sentiment_output = lstm_layer_sentiment_outputs[i]
            final_sentiment_outputs.append(final_sentiment_output)

        # --warmup=True --pipeline=True --attention_warmup_init=True --attention_warmup==False --pipeline_with_acd==True
        if self.configuration['acd_warmup'] and self.configuration['pipeline'] \
            and self.configuration['attention_warmup_init'] and not self.configuration['attention_warmup'] \
            and self.configuration['pipeline_with_acd']:
            final_category_outputs = []
            final_category_outputs_auxillary = []
            for i in range(self.category_num):
                word_representation_for_sentiment = lstm_result_for_graph
                sentiment_alpha = embedding_layer_category_alphas[i]
                category_output = self.element_wise_mul(word_representation_for_sentiment, sentiment_alpha,
                                                        return_not_sum_result=False)
                for j in range(self.category_num):
                    final_category_output = self.category_fcs_pipeline_with_acd[j](category_output)
                    if i == j:
                        final_category_outputs.append(final_category_output)
                    else:
                        final_category_outputs_auxillary.append(final_category_output)

        output = {}
        output['alpha'] = embedding_layer_category_alphas
        if label is not None:
            category_labels = []
            polarity_labels = []
            polarity_masks = []
            for i in range(self.category_num):
                category_labels.append(label[:, i])
                polarity_labels.append(label[:, i + self.category_num])
                polarity_masks.append(polarity_mask[:, i])

            total_category_loss = 0
            total_sentiment_loss = 0
            for i in range(self.category_num):
                category_temp_loss = self.category_loss(final_category_outputs[i].squeeze(dim=-1), category_labels[i])
                sentiment_temp_loss = self.sentiment_loss(final_sentiment_outputs[i], polarity_labels[i].long())
                total_category_loss += category_temp_loss
                if not self.configuration['only_acd']:
                    total_sentiment_loss += sentiment_temp_loss

            interactive_loss = 0
            for i in range(len(final_category_outputs_auxillary)):
                label_temp = torch.zeros_like(category_labels[0])
                interactive_loss_temp = self.category_loss(final_category_outputs_auxillary[i].squeeze(dim=-1),
                                                           label_temp)
                interactive_loss += interactive_loss_temp
            total_interactive_loss = interactive_loss * self.configuration['interactive_loss_lamda'] / (self.category_num - 1)

            loss = self.category_loss_weight * (total_category_loss + total_interactive_loss) + \
                   self.sentiment_loss_weight * total_sentiment_loss

            # Sparse Regularization Orthogonal Regularization
            if self.configuration['sparse_reg'] or self.configuration['orthogonal_reg']:
                reg_loss = 0
                for j in range(len(sample)):
                    polarity_mask_of_one_sample = polarity_mask[j]
                    category_alpha_of_one_sample = [embedding_layer_category_alphas[k][j] for k in range(self.category_num)]
                    category_alpha_of_mentioned = []
                    category_alpha_of_not_mentioned = []
                    for k in range(self.category_num):
                        if polarity_mask_of_one_sample[k] == 1:
                            category_alpha_of_mentioned.append(category_alpha_of_one_sample[k].unsqueeze(0))
                        else:
                            category_alpha_of_not_mentioned.append(category_alpha_of_one_sample[k].unsqueeze(0))
                    if len(category_alpha_of_not_mentioned) != 0:
                        category_alpha_of_not_mentioned = torch.cat(category_alpha_of_not_mentioned, dim=0)
                        category_alpha_of_not_mentioned = torch.mean(category_alpha_of_not_mentioned, dim=0, keepdim=True)
                        category_alpha_of_mentioned.append(category_alpha_of_not_mentioned)

                    category_eye = torch.eye(len(category_alpha_of_mentioned))
                    category_alpha_of_mentioned = torch.cat(category_alpha_of_mentioned, dim=0)
                    category_alpha_similarity = torch.mm(category_alpha_of_mentioned, category_alpha_of_mentioned.t())

                    if self.configuration['sparse_reg'] and self.configuration['orthogonal_reg']:
                        pass
                    elif self.configuration['sparse_reg']:
                        for m in range(len(category_alpha_of_mentioned)):
                            for n in range(len(category_alpha_of_mentioned)):
                                if m != n:
                                    category_eye[m][n] = category_alpha_similarity[m][n]
                    else:
                        # orthogonal_reg
                        for m in range(len(category_alpha_of_mentioned)):
                            category_eye[m][m] = category_alpha_similarity[m][m]
                    # category_eye = nn_util.move_to_device(category_eye, self.configuration['device'])
                    category_eye = category_eye.to(self.configuration['device'])
                    category_alpha_similarity = category_alpha_similarity.to(self.configuration['device'])
                    category_reg_loss = category_alpha_similarity - category_eye
                    category_reg_loss = torch.norm(category_reg_loss)
                    reg_loss += category_reg_loss
                loss += (reg_loss * self.configuration['attention_lamda'] / len(sample))

            # sentiment accuracy
            sentiment_logit = torch.cat(final_sentiment_outputs)
            sentiment_label = torch.cat(polarity_labels)
            sentiment_mask = torch.cat(polarity_masks)
            self._accuracy(sentiment_logit, sentiment_label, sentiment_mask)

            # category f1
            final_category_outputs_prob = [torch.sigmoid(e) for e in final_category_outputs]
            category_prob = torch.cat(final_category_outputs_prob).squeeze()
            category_label = torch.cat(category_labels)
            self._f1(category_prob, category_label)

            output['loss'] = loss

        # visualize attention
        pred_category = [torch.sigmoid(e) for e in final_category_outputs]
        pred_sentiment = [torch.nn.functional.softmax(e, dim=-1) for e in final_sentiment_outputs]
        output['pred_category'] = pred_category
        output['pred_sentiment'] = pred_sentiment
        if self.configuration['visualize_attention']:
            for i in range(len(sample)):
                if self.configuration['constituency_tree']:
                    words = sample[i]['words_for_graph']
                    print('inner_nodes:-------------------------------------------')
                    for inner_node_index, inner_node in enumerate(sample[i]['inner_nodes']):
                        print('%d-%s-%s' % (len(sample[i]['words']) + inner_node_index, inner_node.labels[0], inner_node.text))
                else:
                    words = sample[i]['words']

                attention_labels = [e.split('/')[0] for e in self.categories]

                # 只可视化mams
                label_true = label[i].detach().cpu().numpy()[: self.category_num]
                # if sum(label_true) <= 1:
                #     continue
                # category
                visual_attentions_category = [embedding_layer_category_alphas[j][i][: len(words)].detach().cpu().numpy()
                                              for j in range(self.category_num)]

                titles_sentiment_temp = []
                for j in range(self.category_num):
                    c_label = label[i][j].detach().cpu().numpy().tolist()
                    sentiment_true_index = int(label[i][j + self.category_num].detach().cpu().numpy().tolist())
                    if sentiment_true_index == -100:
                        polarity_temp = '-100'
                    else:
                        polarity_temp = self.polarites[sentiment_true_index]
                    titles_sentiment_temp.append('true: %s - pred: %s - %s' % (str(polarity_temp),
                                                                      str(pred_sentiment[j][
                                                                              i].detach().cpu().numpy()),
                                                                      str(self.polarites)))
                titles = ['true: %s - pred: %s - %s' % (str(label[i][j].detach().cpu().numpy()),
                                                          str(pred_category[j][i].detach().cpu().numpy()),
                                                          titles_sentiment_temp[j])
                          for j in range(self.category_num)]
                print(words)
                print(visual_attentions_category)
                print(attention_labels)
                print(titles)
                savefig_filepath = super()._get_model_visualization_picture_filepath(self.configuration, words)
                attention_visualizer.plot_multi_attentions_of_sentence_backup(words, visual_attentions_category,
                                                                       attention_labels, titles,
                                                                       savefig_filepath=savefig_filepath)

                # sentiment embedding layer
                # visual_attentions = [embedding_layer_sentiment_alphas[j][i][: len(words)].detach().cpu().numpy()
                #                      for j in range(self.category_num)]
                # titles = ['true: %s - pred: %s - %s' % (str(label[i + self.category_num][j].detach().cpu().numpy()),
                #                                         str(pred_sentiment[j][i].detach().cpu().numpy()),
                #                                         str(self.polarites))
                #           for j in range(self.category_num)]
                # attention_visualizer.plot_multi_attentions_of_sentence(words, visual_attentions, attention_labels,
                #                                                        titles)

                # sentiment lstm layer
                # visual_attentions = [lstm_layer_sentiment_alphas[j][i][: len(words)].detach().cpu().numpy()
                #                      for j in range(self.category_num)]
                # titles = ['true: %s - pred: %s - %s' % (str(label[i + self.category_num][j].detach().cpu().numpy()),
                #                                         str(pred_sentiment[j][i].detach().cpu().numpy()),
                #                                         str(self.polarites))
                #           for j in range(self.category_num)]
                # attention_visualizer.plot_multi_attentions_of_sentence(words, visual_attentions, attention_labels,
                #                                                        titles)

                # sentiment lstm layer
                if self.configuration['mil']:
                    visual_attentions_sentiment_temp = [lstm_layer_words_sentiment_soft[j][i][: len(words)].detach().cpu().numpy()
                                                        for j in range(self.category_num)]
                    for j in range(self.category_num):
                        c_label = label[i][j].detach().cpu().numpy().tolist()
                        if c_label == 1:
                            visual_attentions_sentiment = []
                            labels_sentiment = []
                            sentiment_true_index = int(label[i][j + self.category_num].detach().cpu().numpy().tolist())
                            if sentiment_true_index == -100:
                                continue
                            titles_sentiment = ['true: %s - pred: %s - %s' % (str(self.polarites[sentiment_true_index]),
                                                            str(pred_sentiment[j][i].detach().cpu().numpy()),
                                                            str(self.polarites))]
                            c_attention = embedding_layer_category_alphas[j][i][: len(words)].detach().cpu().numpy()
                            visual_attentions_sentiment.append(c_attention)
                            labels_sentiment.append(self.categories[j].split('/')[0])

                            s_distributions = visual_attentions_sentiment_temp[j]
                            for k in range(self.polarity_num):
                                labels_sentiment.append(self.polarites[k])
                                visual_attentions_sentiment.append(s_distributions[:, k])
                            titles_sentiment.extend([''] * 3)
                            attention_visualizer.plot_multi_attentions_of_sentence_backup(words, visual_attentions_sentiment,
                                                                                   labels_sentiment,
                                                                                   titles_sentiment)
        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {
            'accuracy': self._accuracy.get_metric(reset),
            'category_f1': self._f1.get_metric(reset)['fscore']
        }
        return metrics


class AsMilBernoulliAttention(TextInAllAspectSentimentOutModel):
    def __init__(self, word_embedder: TextFieldEmbedder, position_embedder: TextFieldEmbedder,
                 aspect_embedder: TextFieldEmbedder, categories: list, polarities: list, vocab: Vocabulary,
                 configuration: dict, category_loss_weight=1, sentiment_loss_weight=1,
                 acd_model: TextInAllAspectSentimentOutModel=None):
        super().__init__(vocab, category_loss_weight=category_loss_weight, sentiment_loss_weight=sentiment_loss_weight)
        self.configuration = configuration
        self.word_embedder = word_embedder
        self.position_embedder = position_embedder
        self.aspect_embedder = aspect_embedder
        self.categories = categories
        self.polarites = polarities
        self.category_num = len(categories)
        self.polarity_num = len(polarities)
        self.category_loss = nn.BCEWithLogitsLoss()
        self.sentiment_loss = nn.CrossEntropyLoss()
        self._accuracy = metrics.CategoricalAccuracy()
        self._f1 = allennlp_metrics.BinaryF1(0.5)
        self.acd_model = acd_model

        word_embedding_dim = word_embedder.get_output_dim()
        self.embedding_layer_fc = nn.Linear(word_embedding_dim, word_embedding_dim, bias=True)
        self.embedding_layer_aspect_attentions = [BernoulliAttentionInHtt(word_embedding_dim,
                                                                          word_embedding_dim)
                                                  for _ in range(self.category_num)]
        self.embedding_layer_aspect_attentions = nn.ModuleList(self.embedding_layer_aspect_attentions)

        lstm_input_size = word_embedding_dim
        if self.configuration['position']:
            lstm_input_size += position_embedder.get_output_dim()
        num_layers = 3
        self.lstm = torch.nn.LSTM(lstm_input_size, int(word_embedding_dim / 2), batch_first=True,
                                  bidirectional=True, num_layers=num_layers, dropout=0.5)

        self.category_fcs = [nn.Linear(word_embedding_dim, 1) for _ in range(self.category_num)]
        self.category_fcs = nn.ModuleList(self.category_fcs)

        if self.configuration['lstm_layer_category_classifier']:
            self.lstm_category_fcs = [nn.Linear(word_embedding_dim, 1) for _ in range(self.category_num)]
            self.lstm_category_fcs = nn.ModuleList(self.lstm_category_fcs)

        sentiment_fc_input_size = word_embedding_dim
        if not self.configuration['share_sentiment_classifier']:
            self.sentiment_fcs = [nn.Sequential(nn.Linear(sentiment_fc_input_size, sentiment_fc_input_size),
                                                nn.ReLU(),
                                                nn.Linear(sentiment_fc_input_size, self.polarity_num))
                                  for _ in range(self.category_num)]
            self.sentiment_fcs = nn.ModuleList(self.sentiment_fcs)
        else:
            self.sentiment_fc = nn.Sequential(nn.Linear(sentiment_fc_input_size, sentiment_fc_input_size),
                                              nn.ReLU(),
                                              nn.Linear(sentiment_fc_input_size, self.polarity_num))

        self.dropout_after_embedding_layer = nn.Dropout(0.5)
        self.dropout_after_lstm_layer = nn.Dropout(0.5)
        # self.gc1 = DglGraphConvolution(word_embedding_dim, word_embedding_dim, configuration)
        # self.gc2 = DglGraphConvolution(word_embedding_dim, word_embedding_dim, configuration)

    def forward(self, tokens: Dict[str, torch.Tensor], label: torch.Tensor, position: torch.Tensor,
                polarity_mask: torch.Tensor, sample: list, aspects: torch.Tensor=None) -> torch.Tensor:
        mask = get_text_field_mask(tokens)
        word_embeddings = self.word_embedder(tokens)
        word_embeddings_fc = self.embedding_layer_fc(word_embeddings)

        aspects_separate = [{'aspect': aspects['aspect'][:, i].unsqueeze(1)} for i in range(self.category_num)]
        aspect_embeddings_singles = [self.aspect_embedder(aspects_separate[i]).squeeze(1) for i in
                                     range(self.category_num)]
        aspects_seprate_repeat = [{'aspect': aspects_separate[i]['aspect'].expand_as(tokens['tokens'])} for i in
                                  range(self.category_num)]
        aspect_embeddings_separate = [self.aspect_embedder(aspects_seprate_repeat[i]) for i in range(self.category_num)]

        embedding_layer_category_outputs = []
        embedding_layer_category_alphas = []
        embedding_layer_sentiment_outputs = []
        embedding_layer_sentiment_alphas = []

        if self.configuration['pipeline'] and self.acd_model is not None:
            acd_input = {
                'tokens': tokens,
                'label': label,
                'position': position,
                'polarity_mask': polarity_mask,
                'sample': sample,
                'aspects': aspects
            }
            self.acd_model.eval()
            alpha_from_acd_model = self.acd_model(**acd_input)
            embedding_layer_category_alphas = alpha_from_acd_model['alpha']
        else:
            embedding_layer_category_alphas = []
            for i in range(self.category_num):
                embedding_layer_aspect_attention = self.embedding_layer_aspect_attentions[i]
                alpha = embedding_layer_aspect_attention(word_embeddings_fc, mask)
                embedding_layer_category_alphas.append(alpha)

        for i in range(self.category_num):
            alpha = embedding_layer_category_alphas[i]
            category_output = self.element_wise_mul(word_embeddings_fc, alpha, return_not_sum_result=False)
            embedding_layer_category_outputs.append(category_output)

        lstm_input = word_embeddings
        if self.configuration['position']:
            position_embeddings = self.position_embedder(position)
            lstm_input = torch.cat([word_embeddings, position_embeddings], dim=-1)
        lstm_input = self.dropout_after_embedding_layer(lstm_input)

        lstm_result, _ = self.lstm(lstm_input)
        lstm_result = self.dropout_after_lstm_layer(lstm_result)
        # lstm_result_with_position = torch.cat([lstm_result, position_embeddings], dim=-1)
        lstm_layer_category_outputs = []
        lstm_layer_sentiment_outputs = []
        lstm_layer_words_sentiment_soft = []

        # max_len = tokens['tokens'].size()[1]
        # graphs = [e[3] for e in sample]
        # graphs_padded = self.pad_dgl_graph(graphs, max_len)
        # graph_output1 = F.relu(self.gc1(word_embeddings, graphs_padded))
        # graph_output2 = F.relu(self.gc2(graph_output1, graphs_padded))
        for i in range(self.category_num):
            alpha = embedding_layer_category_alphas[i]
            category_output = self.element_wise_mul(lstm_result, alpha, return_not_sum_result=False)
            lstm_layer_category_outputs.append(category_output)

            # sentiment
            # word_representation_for_sentiment = torch.cat([graph_output2, lstm_result], dim=-1)
            word_representation_for_sentiment = lstm_result
            sentiment_alpha = embedding_layer_category_alphas[i]
            if self.configuration['mil']:
                sentiment_alpha = sentiment_alpha.unsqueeze(1)
                if not self.configuration['share_sentiment_classifier']:
                    words_sentiment = self.sentiment_fcs[i](word_representation_for_sentiment)
                else:
                    words_sentiment = self.sentiment_fc(word_representation_for_sentiment)
                if self.configuration['mil_softmax']:
                    words_sentiment_soft = torch.softmax(words_sentiment, dim=-1)
                    lstm_layer_words_sentiment_soft.append(words_sentiment_soft)
                else:
                    words_sentiment_soft = words_sentiment
                    lstm_layer_words_sentiment_soft.append(torch.softmax(words_sentiment, dim=-1))
                sentiment_output = torch.matmul(sentiment_alpha, words_sentiment_soft).squeeze(1)  # batch_size x 2*hidden_dim
                lstm_layer_sentiment_outputs.append(sentiment_output)
            else:
                sentiment_output_temp = self.element_wise_mul(word_representation_for_sentiment, sentiment_alpha,
                                                         return_not_sum_result=False)
                if not self.configuration['share_sentiment_classifier']:
                    sentiment_output = self.sentiment_fcs[i](sentiment_output_temp)
                else:
                    sentiment_output = self.sentiment_fc(sentiment_output_temp)
                lstm_layer_sentiment_outputs.append(sentiment_output)

        final_category_outputs = []
        final_lstm_category_outputs = []
        final_sentiment_outputs = []
        for i in range(self.category_num):
            fc = self.category_fcs[i]
            category_output = embedding_layer_category_outputs[i]
            final_category_output = fc(category_output)
            final_category_outputs.append(final_category_output)

            if self.configuration['lstm_layer_category_classifier']:
                fc_lstm = self.lstm_category_fcs[i]
                lstm_category_output = lstm_layer_category_outputs[i]
                final_lstm_category_output = fc_lstm(lstm_category_output)
                final_lstm_category_outputs.append(final_lstm_category_output)

            final_sentiment_output = lstm_layer_sentiment_outputs[i]
            final_sentiment_outputs.append(final_sentiment_output)

        output = {}
        output['alpha'] = embedding_layer_category_alphas
        if label is not None:
            category_labels = []
            polarity_labels = []
            polarity_masks = []
            for i in range(self.category_num):
                category_labels.append(label[:, i])
                polarity_labels.append(label[:, i + self.category_num])
                polarity_masks.append(polarity_mask[:, i])
            loss = 0
            total_category_loss = 0
            total_sentiment_loss = 0
            for i in range(self.category_num):
                category_temp_loss = self.category_loss(final_category_outputs[i].squeeze(dim=-1), category_labels[i])
                sentiment_temp_loss = self.sentiment_loss(final_sentiment_outputs[i], polarity_labels[i].long())
                total_category_loss += category_temp_loss
                if not self.configuration['only_acd']:
                    total_sentiment_loss += sentiment_temp_loss
                if self.configuration['lstm_layer_category_classifier']:
                    lstm_category_temp_loss = self.category_loss(final_lstm_category_outputs[i].squeeze(dim=-1),
                                                                 category_labels[i])
                    total_category_loss += lstm_category_temp_loss
            loss = self.category_loss_weight * total_category_loss + self.sentiment_loss_weight * total_sentiment_loss

            # sentiment accuracy
            sentiment_logit = torch.cat(final_sentiment_outputs)
            sentiment_label = torch.cat(polarity_labels)
            sentiment_mask = torch.cat(polarity_masks)
            self._accuracy(sentiment_logit, sentiment_label, sentiment_mask)

            # category f1
            final_category_outputs_prob = [torch.sigmoid(e) for e in final_category_outputs]
            category_prob = torch.cat(final_category_outputs_prob).squeeze()
            category_label = torch.cat(category_labels)
            self._f1(category_prob, category_label)

            output['loss'] = loss

        # visualize attention
        pred_category = [torch.sigmoid(e) for e in final_category_outputs]
        pred_sentiment = [torch.nn.functional.softmax(e, dim=-1) for e in final_sentiment_outputs]
        output['pred_category'] = pred_category
        output['pred_sentiment'] = pred_sentiment
        if self.configuration['visualize_attention']:
            for i in range(len(sample)):
                words = sample[i][2]
                attention_labels = [e.split('/')[0] for e in self.categories]

                # category
                visual_attentions_category = [embedding_layer_category_alphas[j][i][: len(words)].detach().cpu().numpy()
                                              for j in range(self.category_num)]
                titles = ['true: %s - pred: %s' % (str(label[i][j].detach().cpu().numpy()),
                                                   str(pred_category[j][i].detach().cpu().numpy()))
                          for j in range(self.category_num)]
                attention_visualizer.plot_multi_attentions_of_sentence(words, visual_attentions_category,
                                                                       attention_labels, titles)

                # sentiment embedding layer
                # visual_attentions = [embedding_layer_sentiment_alphas[j][i][: len(words)].detach().cpu().numpy()
                #                      for j in range(self.category_num)]
                # titles = ['true: %s - pred: %s - %s' % (str(label[i + self.category_num][j].detach().cpu().numpy()),
                #                                         str(pred_sentiment[j][i].detach().cpu().numpy()),
                #                                         str(self.polarites))
                #           for j in range(self.category_num)]
                # attention_visualizer.plot_multi_attentions_of_sentence(words, visual_attentions, attention_labels,
                #                                                        titles)

                # sentiment lstm layer
                # visual_attentions = [lstm_layer_sentiment_alphas[j][i][: len(words)].detach().cpu().numpy()
                #                      for j in range(self.category_num)]
                # titles = ['true: %s - pred: %s - %s' % (str(label[i + self.category_num][j].detach().cpu().numpy()),
                #                                         str(pred_sentiment[j][i].detach().cpu().numpy()),
                #                                         str(self.polarites))
                #           for j in range(self.category_num)]
                # attention_visualizer.plot_multi_attentions_of_sentence(words, visual_attentions, attention_labels,
                #                                                        titles)

                # sentiment lstm layer
                visual_attentions_sentiment_temp = [lstm_layer_words_sentiment_soft[j][i][: len(words)].detach().cpu().numpy()
                                                    for j in range(self.category_num)]
                for j in range(self.category_num):
                    c_label = label[i][j].detach().cpu().numpy().tolist()
                    if c_label == 1:
                        visual_attentions_sentiment = []
                        labels_sentiment = []
                        sentiment_true_index = int(label[i][j + self.category_num].detach().cpu().numpy().tolist())
                        if sentiment_true_index == -100:
                            continue
                        titles_sentiment = ['true: %s - pred: %s - %s' % (str(self.polarites[sentiment_true_index]),
                                                        str(pred_sentiment[j][i].detach().cpu().numpy()),
                                                        str(self.polarites))]
                        c_attention = embedding_layer_category_alphas[j][i][: len(words)].detach().cpu().numpy()
                        visual_attentions_sentiment.append(c_attention)
                        labels_sentiment.append(self.categories[j].split('/')[0])

                        s_distributions = visual_attentions_sentiment_temp[j]
                        for k in range(self.polarity_num):
                            labels_sentiment.append(self.polarites[k])
                            visual_attentions_sentiment.append(s_distributions[:, k])
                        titles_sentiment.extend([''] * 3)
                        attention_visualizer.plot_multi_attentions_of_sentence(words, visual_attentions_sentiment,
                                                                               labels_sentiment,
                                                                               titles_sentiment)
                        print()
        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {
            'accuracy': self._accuracy.get_metric(reset),
            'category_f1': self._f1.get_metric(reset)['fscore']
        }
        return metrics


class AsMil(Model):
    def __init__(self, word_embedder: TextFieldEmbedder, position_embedder: TextFieldEmbedder,
                 categories: list, polarities: list, vocab: Vocabulary, configuration: dict):
        super().__init__(vocab)
        self.configuration = configuration
        self.word_embedder = word_embedder
        self.position_embedder = position_embedder
        self.categories = categories
        self.polarites = polarities
        self.category_num = len(categories)
        self.polarity_num = len(polarities)
        self.category_loss = nn.BCEWithLogitsLoss()
        self.sentiment_loss = nn.CrossEntropyLoss()
        self._accuracy = metrics.CategoricalAccuracy()
        self._f1 = allennlp_metrics.BinaryF1(0.5)

        word_embedding_dim = word_embedder.get_output_dim()
        if self.configuration['position']:
            word_embedding_dim += position_embedder.get_output_dim()
        self.embedding_layer_fc = nn.Linear(word_embedding_dim, word_embedding_dim, bias=True)
        self.embedding_layer_aspect_attentions = [AttentionInHtt(word_embedding_dim,
                                                                 word_embedding_dim)
                                                  for _ in range(self.category_num)]

        lstm_input_size = word_embedding_dim
        num_layers = 3
        self.lstm = torch.nn.LSTM(lstm_input_size, int(word_embedding_dim / 2), batch_first=True,
                                  bidirectional=True, num_layers=num_layers, dropout=0.5)

        self.category_fcs = [nn.Linear(word_embedding_dim, 1) for _ in range(self.category_num)]
        if self.configuration['lstm_layer_category_classifier']:
            self.lstm_category_fcs = [nn.Linear(word_embedding_dim, 1) for _ in range(self.category_num)]
        self.sentiment_fc = nn.Sequential(nn.Linear(word_embedding_dim, word_embedding_dim),
                                          nn.ReLU(),
                                          nn.Linear(word_embedding_dim, self.polarity_num))

        # self.gc1 = DglGraphConvolution(word_embedding_dim, word_embedding_dim, configuration)
        # self.gc2 = DglGraphConvolution(word_embedding_dim, word_embedding_dim, configuration)

    def matrix_mul(self, input, weight, bias=False):
        feature_list = []
        for feature in input:
            feature = torch.mm(feature, weight)
            if isinstance(bias, torch.nn.parameter.Parameter):
                feature = feature + bias.expand(feature.size()[0], bias.size()[1])
            feature = torch.tanh(feature).unsqueeze(0)
            feature_list.append(feature)

        return torch.cat(feature_list, 0).squeeze()

    def element_wise_mul(self, input1, input2, return_not_sum_result=False):
        feature_list = []
        for feature_1, feature_2 in zip(input1, input2):
            feature_2 = feature_2.unsqueeze(1)
            feature_2 = feature_2.expand_as(feature_1)
            feature = feature_1 * feature_2
            feature = feature.unsqueeze(0)
            feature_list.append(feature)
        output = torch.cat(feature_list, 0)

        result = torch.sum(output, 1)
        if return_not_sum_result:
            return result, output
        else:
            return result

    def reduce(self, nodes):
        """Take an average over all neighbor node features hu and use it to
        overwrite the original node feature."""
        m = nodes.mailbox['m']
        accum = torch.sum(m, 1)
        return {'h': accum}

    def pad_dgl_graph(self, graphs, max_node_num):
        graphs_padded = []
        for graph in graphs:
            graph_padded = copy.deepcopy(graph)
            node_num = graph.number_of_nodes()
            graph_padded.add_nodes(max_node_num - node_num)
            graphs_padded.append(graph_padded)
        return graphs_padded

    def forward(self, tokens: Dict[str, torch.Tensor], label: torch.Tensor, position: torch.Tensor,
                polarity_mask: torch.Tensor, sample: list) -> torch.Tensor:
        mask = get_text_field_mask(tokens)
        word_embeddings = self.word_embedder(tokens)
        if self.configuration['position']:
            position_embeddings = self.position_embedder(position)
            word_embeddings = torch.cat([word_embeddings, position_embeddings], dim=-1)
        word_embeddings_fc = self.embedding_layer_fc(word_embeddings)

        embedding_layer_category_outputs = []
        embedding_layer_category_alphas = []
        embedding_layer_sentiment_outputs = []
        embedding_layer_sentiment_alphas = []
        for i in range(self.category_num):
            embedding_layer_aspect_attention = self.embedding_layer_aspect_attentions[i]
            alpha = embedding_layer_aspect_attention(word_embeddings_fc, mask)
            embedding_layer_category_alphas.append(alpha)

            category_output = self.element_wise_mul(word_embeddings_fc, alpha, return_not_sum_result=False)
            embedding_layer_category_outputs.append(category_output)

        lstm_result, _ = self.lstm(word_embeddings)
        # lstm_result_with_position = torch.cat([lstm_result, position_embeddings], dim=-1)
        lstm_layer_category_outputs = []
        lstm_layer_sentiment_outputs = []
        lstm_layer_words_sentiment_soft = []

        # max_len = tokens['tokens'].size()[1]
        # graphs = [e[3] for e in sample]
        # graphs_padded = self.pad_dgl_graph(graphs, max_len)
        # graph_output1 = F.relu(self.gc1(word_embeddings, graphs_padded))
        # graph_output2 = F.relu(self.gc2(graph_output1, graphs_padded))
        for i in range(self.category_num):
            alpha = embedding_layer_category_alphas[i]
            category_output = self.element_wise_mul(lstm_result, alpha, return_not_sum_result=False)
            lstm_layer_category_outputs.append(category_output)

            # sentiment
            # word_representation_for_sentiment = torch.cat([graph_output2, lstm_result], dim=-1)
            word_representation_for_sentiment = lstm_result
            sentiment_alpha = embedding_layer_category_alphas[i]
            if self.configuration['mil']:
                sentiment_alpha = sentiment_alpha.unsqueeze(1)
                words_sentiment = self.sentiment_fc(word_representation_for_sentiment)
                if self.configuration['mil_softmax']:
                    words_sentiment_soft = torch.softmax(words_sentiment, dim=-1)
                    lstm_layer_words_sentiment_soft.append(words_sentiment_soft)
                else:
                    words_sentiment_soft = words_sentiment
                    lstm_layer_words_sentiment_soft.append(torch.softmax(words_sentiment, dim=-1))
                sentiment_output = torch.matmul(sentiment_alpha, words_sentiment_soft).squeeze(1)  # batch_size x 2*hidden_dim
                lstm_layer_sentiment_outputs.append(sentiment_output)
            else:
                sentiment_output = self.element_wise_mul(word_representation_for_sentiment, sentiment_alpha,
                                                         return_not_sum_result=False)
                lstm_layer_sentiment_outputs.append(sentiment_output)

        final_category_outputs = []
        final_lstm_category_outputs = []
        final_sentiment_outputs = []
        for i in range(self.category_num):
            fc = self.category_fcs[i]
            category_output = embedding_layer_category_outputs[i]
            final_category_output = fc(category_output)
            final_category_outputs.append(final_category_output)

            if self.configuration['lstm_layer_category_classifier']:
                fc_lstm = self.lstm_category_fcs[i]
                lstm_category_output = lstm_layer_category_outputs[i]
                final_lstm_category_output = fc_lstm(lstm_category_output)
                final_lstm_category_outputs.append(final_lstm_category_output)

            sentiment_output = lstm_layer_sentiment_outputs[i]
            if self.configuration['mil']:
                final_sentiment_output = sentiment_output
            else:
                final_sentiment_output = self.sentiment_fc(sentiment_output)
            final_sentiment_outputs.append(final_sentiment_output)

        output = {}
        if label is not None:
            category_labels = []
            polarity_labels = []
            polarity_masks = []
            for i in range(self.category_num):
                category_labels.append(label[:, i])
                polarity_labels.append(label[:, i + self.category_num])
                polarity_masks.append(polarity_mask[:, i])
            loss = 0
            for i in range(self.category_num):
                category_temp_loss = self.category_loss(final_category_outputs[i].squeeze(dim=-1), category_labels[i])
                sentiment_temp_loss = self.sentiment_loss(final_sentiment_outputs[i], polarity_labels[i].long())
                loss += category_temp_loss
                if not self.configuration['only_acd']:
                    loss += sentiment_temp_loss
                if self.configuration['lstm_layer_category_classifier']:
                    lstm_category_temp_loss = self.category_loss(final_lstm_category_outputs[i].squeeze(dim=-1),
                                                                 category_labels[i])
                    loss += lstm_category_temp_loss

            # sentiment accuracy
            sentiment_logit = torch.cat(final_sentiment_outputs)
            sentiment_label = torch.cat(polarity_labels)
            sentiment_mask = torch.cat(polarity_masks)
            self._accuracy(sentiment_logit, sentiment_label, sentiment_mask)

            # category f1
            final_category_outputs_prob = [torch.sigmoid(e) for e in final_category_outputs]
            category_prob = torch.cat(final_category_outputs_prob).squeeze()
            category_label = torch.cat(category_labels)
            self._f1(category_prob, category_label)

            output['loss'] = loss

        # visualize attention
        pred_category = [torch.sigmoid(e) for e in final_category_outputs]
        pred_sentiment = [torch.nn.functional.softmax(e, dim=-1) for e in final_sentiment_outputs]
        output['pred_category'] = pred_category
        output['pred_sentiment'] = pred_sentiment
        if self.configuration['visualize_attention']:
            for i in range(len(sample)):
                words = sample[i][2]
                attention_labels = [e.split('/')[0] for e in self.categories]

                # category
                visual_attentions_category = [embedding_layer_category_alphas[j][i][: len(words)].detach().cpu().numpy()
                                              for j in range(self.category_num)]
                titles = ['true: %s - pred: %s' % (str(label[i][j].detach().cpu().numpy()),
                                                   str(pred_category[j][i].detach().cpu().numpy()))
                          for j in range(self.category_num)]
                attention_visualizer.plot_multi_attentions_of_sentence(words, visual_attentions_category,
                                                                       attention_labels, titles)

                # sentiment embedding layer
                # visual_attentions = [embedding_layer_sentiment_alphas[j][i][: len(words)].detach().cpu().numpy()
                #                      for j in range(self.category_num)]
                # titles = ['true: %s - pred: %s - %s' % (str(label[i + self.category_num][j].detach().cpu().numpy()),
                #                                         str(pred_sentiment[j][i].detach().cpu().numpy()),
                #                                         str(self.polarites))
                #           for j in range(self.category_num)]
                # attention_visualizer.plot_multi_attentions_of_sentence(words, visual_attentions, attention_labels,
                #                                                        titles)

                # sentiment lstm layer
                # visual_attentions = [lstm_layer_sentiment_alphas[j][i][: len(words)].detach().cpu().numpy()
                #                      for j in range(self.category_num)]
                # titles = ['true: %s - pred: %s - %s' % (str(label[i + self.category_num][j].detach().cpu().numpy()),
                #                                         str(pred_sentiment[j][i].detach().cpu().numpy()),
                #                                         str(self.polarites))
                #           for j in range(self.category_num)]
                # attention_visualizer.plot_multi_attentions_of_sentence(words, visual_attentions, attention_labels,
                #                                                        titles)

                # sentiment lstm layer
                visual_attentions_sentiment_temp = [lstm_layer_words_sentiment_soft[j][i][: len(words)].detach().cpu().numpy()
                                                    for j in range(self.category_num)]
                for j in range(self.category_num):
                    c_label = label[i][j].detach().cpu().numpy().tolist()
                    if c_label == 1:
                        visual_attentions_sentiment = []
                        labels_sentiment = []
                        sentiment_true_index = int(label[i][j + self.category_num].detach().cpu().numpy().tolist())
                        if sentiment_true_index == -100:
                            continue
                        titles_sentiment = ['true: %s - pred: %s - %s' % (str(self.polarites[sentiment_true_index]),
                                                        str(pred_sentiment[j][i].detach().cpu().numpy()),
                                                        str(self.polarites))]
                        c_attention = embedding_layer_category_alphas[j][i][: len(words)].detach().cpu().numpy()
                        visual_attentions_sentiment.append(c_attention)
                        labels_sentiment.append(self.categories[j].split('/')[0])

                        s_distributions = visual_attentions_sentiment_temp[j]
                        for k in range(self.polarity_num):
                            labels_sentiment.append(self.polarites[k])
                            visual_attentions_sentiment.append(s_distributions[:, k])
                        titles_sentiment.extend([''] * 3)
                        attention_visualizer.plot_multi_attentions_of_sentence(words, visual_attentions_sentiment,
                                                                               labels_sentiment,
                                                                               titles_sentiment)
                        print()
        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {
            'accuracy': self._accuracy.get_metric(reset),
            'category_f1': self._f1.get_metric(reset)['fscore']
        }
        return metrics


class AsMilSimultaneouslyBert(TextInAllAspectSentimentOutModel):
    def __init__(self, word_embedder: TextFieldEmbedder, position_embedder: TextFieldEmbedder,
                 categories: list, polarities: list, vocab: Vocabulary, configuration: dict,
                 bert_word_embedder: TextFieldEmbedder=None):
        super().__init__(vocab)
        self.configuration = configuration
        self.word_embedder = word_embedder
        self.position_embedder = position_embedder
        self.categories = categories
        self.polarites = polarities
        self.category_num = len(categories)
        self.polarity_num = len(polarities)
        self.category_loss = nn.BCEWithLogitsLoss()
        self.sentiment_loss = nn.CrossEntropyLoss()
        self._accuracy = metrics.CategoricalAccuracy()
        self._f1 = allennlp_metrics.BinaryF1(0.5)

        word_embedding_dim = word_embedder.get_output_dim()
        if self.configuration['lstm_or_fc_after_embedding_layer'] == 'fc':
            self.embedding_layer_fc = nn.Linear(word_embedding_dim, word_embedding_dim, bias=True)
        elif self.configuration['lstm_or_fc_after_embedding_layer'] == 'bilstm':
            self.embedding_layer_lstm = torch.nn.LSTM(word_embedding_dim, int(word_embedding_dim / 2), batch_first=True,
                                                      bidirectional=True, num_layers=1)
        else:
            self.embedding_layer_lstm = torch.nn.LSTM(word_embedding_dim, word_embedding_dim, batch_first=True,
                                                      bidirectional=False, num_layers=1)

        self.embedding_layer_aspect_attentions = [AttentionInHtt(word_embedding_dim,
                                                                 word_embedding_dim)
                                                  for _ in range(self.category_num)]
        self.embedding_layer_aspect_attentions = nn.ModuleList(self.embedding_layer_aspect_attentions)

        self.category_fcs = [nn.Linear(word_embedding_dim, 1) for _ in range(self.category_num)]
        self.category_fcs = nn.ModuleList(self.category_fcs)

        if self.configuration['lstm_layer_num_in_bert'] != 0:
            num_layers = self.configuration['lstm_layer_num_in_bert']
            bilstm_hidden_size_in_bert = self.configuration['bilstm_hidden_size_in_bert']
            if bilstm_hidden_size_in_bert == 0:
                bilstm_hidden_size_in_bert = int(word_embedding_dim / 2)
            self.lstm = torch.nn.LSTM(768, bilstm_hidden_size_in_bert, batch_first=True,
                                      bidirectional=True, num_layers=num_layers,
                                      dropout=self.configuration['dropout_in_bert'])
            hidden_size = bilstm_hidden_size_in_bert * 2
        else:
            hidden_size = 768
        if self.configuration['only_bert']:
            self.sentiment_fc = nn.Sequential(
                # nn.Linear(768, 768),
                # nn.ReLU(),
                nn.Linear(768, self.polarity_num))
        else:
            self.sentiment_fc = nn.Sequential(
                # nn.Linear(hidden_size, hidden_size),
                # nn.ReLU(),
                nn.Linear(hidden_size, self.polarity_num))
        self.bert_word_embedder = bert_word_embedder

        self.dropout_after_embedding_layer = nn.Dropout(self.configuration['dropout_in_bert'])

    def set_bert_word_embedder(self, bert_word_embedder: TextFieldEmbedder=None):
        self.bert_word_embedder = bert_word_embedder

    def set_grad_for_acd_parameter(self, requires_grad=True):
        acd_layers = []
        if self.configuration['lstm_or_fc_after_embedding_layer'] == 'fc':
            acd_layers.append(self.embedding_layer_fc)
        else:
            acd_layers.append(self.embedding_layer_lstm)
        acd_layers.append(self.embedding_layer_aspect_attentions)
        acd_layers.append(self.category_fcs)
        for layer in acd_layers:
            for name, value in layer.named_parameters():
                value.requires_grad = requires_grad

    def set_grad_for_acsc_parameter(self, requires_grad=True):
        acsc_layers = []
        if self.configuration['lstm_layer_num_in_bert'] != 0:
            acsc_layers.append(self.lstm)
        acsc_layers.append(self.sentiment_fc)
        for layer in acsc_layers:
            for name, value in layer.named_parameters():
                value.requires_grad = requires_grad
        bert_model = self.bert_word_embedder._token_embedders['bert'].bert_model
        for param in bert_model.parameters():
            param.requires_grad = requires_grad

    def forward(self, tokens: Dict[str, torch.Tensor], label: torch.Tensor, position: torch.Tensor,
                polarity_mask: torch.Tensor, sample: list, bert: torch.Tensor) -> torch.Tensor:
        bert_mask = bert['mask']
        # bert_word_embeddings = self.bert_word_embedder(bert)
        token_type_ids = bert['bert-type-ids']
        # token_type_ids_size = token_type_ids.size()
        # for i in range(token_type_ids_size[1]):
        #     print(token_type_ids[0][i])
        offsets = bert['bert-offsets']
        bert_word_embeddings = self.bert_word_embedder(bert, token_type_ids=token_type_ids, offsets=offsets)

        mask = get_text_field_mask(tokens)
        word_embeddings = self.word_embedder(tokens)
        word_embeddings_size = word_embeddings.size()
        if self.configuration['lstm_or_fc_after_embedding_layer'] == 'fc':
            word_embeddings_fc = self.embedding_layer_fc(word_embeddings)
        else:
            word_embeddings_fc, (_, _) = self.embedding_layer_lstm(word_embeddings)

        embedding_layer_category_outputs = []
        embedding_layer_category_alphas = []
        embedding_layer_sentiment_outputs = []
        embedding_layer_sentiment_alphas = []
        bert_clses_of_all_aspect = []
        for i in range(self.category_num):
            embedding_layer_aspect_attention = self.embedding_layer_aspect_attentions[i]
            alpha = embedding_layer_aspect_attention(word_embeddings_fc, mask)
            embedding_layer_category_alphas.append(alpha)

            category_output = self.element_wise_mul(word_embeddings_fc, alpha, return_not_sum_result=False)
            embedding_layer_category_outputs.append(category_output)

            # 16 x 768
            bert_clses_of_aspect = bert_word_embeddings[:, i, 0, :]
            bert_clses_of_all_aspect.append(bert_clses_of_aspect)

            if not self.configuration['only_bert']:
                # 16 x 73 x 768
                bert_word_embeddings_of_aspect = bert_word_embeddings[:, i, :, :]
                aspect_word_embeddings_from_bert = []
                for j in range(len(sample)):
                    aspect_word_embeddings_from_bert_of_one_sample = []
                    all_word_indices_in_bert = sample[j][6]
                    categories_mentioned = [e[0] for e in sample[j][1]]
                    for k in range(word_embeddings_size[1]):
                        is_index_greater_than_max_len = False
                        if k in all_word_indices_in_bert:
                            for index in all_word_indices_in_bert[k]:
                                if index >= self.configuration['max_len']:
                                    is_index_greater_than_max_len = True
                                    break
                        if not is_index_greater_than_max_len and k in all_word_indices_in_bert and i in categories_mentioned:
                            word_indices_in_bert = all_word_indices_in_bert[k]
                            word_bert_embeddings = []
                            for word_index_in_bert in word_indices_in_bert:
                                word_bert_embedding = bert_word_embeddings_of_aspect[j][word_index_in_bert]
                                word_bert_embeddings.append(word_bert_embedding)
                            if len(word_bert_embeddings) == 0:
                                print()
                            if len(word_bert_embeddings) > 1:
                                word_bert_embeddings_unsqueeze = [torch.unsqueeze(e, dim=0) for e in word_bert_embeddings]
                                word_bert_embeddings_cat = torch.cat(word_bert_embeddings_unsqueeze, dim=0)
                                word_bert_embeddings_sum = torch.sum(word_bert_embeddings_cat, dim=0)
                                word_bert_embeddings_ave = word_bert_embeddings_sum / len(word_bert_embeddings)
                            else:
                                word_bert_embeddings_ave = word_bert_embeddings[0]
                            aspect_word_embeddings_from_bert_of_one_sample.append(
                                torch.unsqueeze(word_bert_embeddings_ave, 0))
                        else:
                            zero = torch.zeros_like(torch.unsqueeze(bert_clses_of_aspect[0], 0))
                            aspect_word_embeddings_from_bert_of_one_sample.append(zero)
                    aspect_word_embeddings_from_bert_of_one_sample_cat = torch.cat(aspect_word_embeddings_from_bert_of_one_sample, dim=0)
                    aspect_word_embeddings_from_bert.append(torch.unsqueeze(aspect_word_embeddings_from_bert_of_one_sample_cat, dim=0))
                aspect_word_embeddings_from_bert_cat = torch.cat(aspect_word_embeddings_from_bert, dim=0)
                if self.configuration['lstm_layer_num_in_bert'] != 0:
                    aspect_word_embeddings_from_bert_cat, _ = self.lstm(aspect_word_embeddings_from_bert_cat)
                embedding_layer_sentiment_outputs.append(aspect_word_embeddings_from_bert_cat)

        lstm_layer_category_outputs = []
        lstm_layer_sentiment_outputs = []
        lstm_layer_words_sentiment_soft = []
        sentiment_output_clses_soft = []
        for i in range(self.category_num):
            sentiment_output_temp = bert_clses_of_all_aspect[i]
            if self.configuration['dropout_after_cls']:
                sentiment_output_temp = self.dropout_after_embedding_layer(sentiment_output_temp)
            sentiment_output_cls = self.sentiment_fc(sentiment_output_temp)
            sentiment_output_clses_soft.append(torch.softmax(sentiment_output_cls, dim=-1))
            if self.configuration['only_bert']:
                sentiment_output = sentiment_output_cls
                lstm_layer_sentiment_outputs.append(sentiment_output)
            else:
                # sentiment
                aspect_word_embeddings_from_bert = embedding_layer_sentiment_outputs[i]
                word_representation_for_sentiment = self.dropout_after_embedding_layer(aspect_word_embeddings_from_bert)

                sentiment_alpha = embedding_layer_category_alphas[i]
                if self.configuration['mil']:
                    sentiment_alpha = sentiment_alpha.unsqueeze(1)
                    words_sentiment = self.sentiment_fc(word_representation_for_sentiment)
                    if self.configuration['mil_softmax']:
                        words_sentiment_soft = torch.softmax(words_sentiment, dim=-1)
                        lstm_layer_words_sentiment_soft.append(words_sentiment_soft)
                    else:
                        words_sentiment_soft = words_sentiment
                        lstm_layer_words_sentiment_soft.append(torch.softmax(words_sentiment, dim=-1))
                    sentiment_output_mil = torch.matmul(sentiment_alpha, words_sentiment_soft).squeeze(1)  # batch_size x 2*hidden_dim
                    if self.configuration['concat_cls_vector']:
                        if self.configuration['concat_cls_vector_mode'] == 'average':
                            sentiment_output = (sentiment_output_mil + sentiment_output_cls) / 2
                        else:
                            sentiment_output = sentiment_output_mil + sentiment_output_cls
                    else:
                        sentiment_output = sentiment_output_mil
                    lstm_layer_sentiment_outputs.append(sentiment_output)
                else:
                    sentiment_output_temp = self.element_wise_mul(word_representation_for_sentiment, sentiment_alpha,
                                                                  return_not_sum_result=False)
                    sentiment_output_not_mil = self.sentiment_fc(sentiment_output_temp)
                    if self.configuration['concat_cls_vector']:
                        if self.configuration['concat_cls_vector_mode'] == 'average':
                            sentiment_output = (sentiment_output_not_mil + sentiment_output_cls) / 2
                        else:
                            sentiment_output = sentiment_output_not_mil + sentiment_output_cls
                    else:
                        sentiment_output = sentiment_output_not_mil
                    lstm_layer_sentiment_outputs.append(sentiment_output)

        final_category_outputs = []
        final_lstm_category_outputs = []
        final_sentiment_outputs = []
        for i in range(self.category_num):
            fc = self.category_fcs[i]
            category_output = embedding_layer_category_outputs[i]
            final_category_output = fc(category_output)
            final_category_outputs.append(final_category_output)

            sentiment_output = lstm_layer_sentiment_outputs[i]
            final_sentiment_output = sentiment_output
            final_sentiment_outputs.append(final_sentiment_output)

        output = {}
        if label is not None:
            category_labels = []
            polarity_labels = []
            polarity_masks = []
            for i in range(self.category_num):
                category_labels.append(label[:, i])
                polarity_labels.append(label[:, i + self.category_num])
                polarity_masks.append(polarity_mask[:, i])
            loss = 0
            total_category_loss = 0
            total_sentiment_loss = 0
            for i in range(self.category_num):
                category_temp_loss = self.category_loss(final_category_outputs[i].squeeze(dim=-1), category_labels[i])
                sentiment_temp_loss = self.sentiment_loss(final_sentiment_outputs[i], polarity_labels[i].long())
                if not self.configuration['only_sc']:
                    total_category_loss += category_temp_loss
                if not self.configuration['only_acd']:
                    total_sentiment_loss += sentiment_temp_loss

            loss = self.category_loss_weight * total_category_loss + self.sentiment_loss_weight * total_sentiment_loss

            # sentiment accuracy
            sentiment_logit = torch.cat(final_sentiment_outputs)
            sentiment_label = torch.cat(polarity_labels)
            sentiment_mask = torch.cat(polarity_masks)
            self._accuracy(sentiment_logit, sentiment_label, sentiment_mask)

            # category f1
            final_category_outputs_prob = [torch.sigmoid(e) for e in final_category_outputs]
            category_prob = torch.cat(final_category_outputs_prob).squeeze()
            category_label = torch.cat(category_labels)
            self._f1(category_prob, category_label)

            output['loss'] = loss

        # visualize attention
        pred_category = [torch.sigmoid(e) for e in final_category_outputs]
        pred_sentiment = [torch.nn.functional.softmax(e, dim=-1) for e in final_sentiment_outputs]
        output['pred_category'] = pred_category
        output['pred_sentiment'] = pred_sentiment
        output['embedding_layer_category_alphas'] = embedding_layer_category_alphas
        output['lstm_layer_words_sentiment_soft'] = lstm_layer_words_sentiment_soft
        if self.configuration['visualize_attention']:
            for i in range(len(sample)):
                words: list = sample[i][2]
                # if not ('while' in words and 'it' in words):
                #     continue
                attention_labels = [e.split('/')[0] for e in self.categories]

                # category
                visual_attentions_category = [embedding_layer_category_alphas[j][i][: len(words)].detach().cpu().numpy()
                                              for j in range(self.category_num)]
                titles = ['true: %s - pred: %s' % (str(label[i][j].detach().cpu().numpy()),
                                                   str(pred_category[j][i].detach().cpu().numpy()))
                          for j in range(self.category_num)]
                # savefig_filepath = super()._get_model_visualization_picture_filepath(self.configuration, words)
                # attention_visualizer.plot_multi_attentions_of_sentence(words, visual_attentions_category,
                #                                                        attention_labels, titles, savefig_filepath)
                attention_visualizer.plot_multi_attentions_of_sentence_backup(words, visual_attentions_category,
                                                                       attention_labels, titles)

                # sentiment lstm layer
                visual_attentions_sentiment_temp = [lstm_layer_words_sentiment_soft[j][i][: len(words)]
                                                    for j in range(self.category_num)]
                if self.configuration['concat_cls_vector']:
                    words.insert(0, '[CLS]')
                    clses_sentiment_temp = [e.unsqueeze(dim=1)[i] for e in sentiment_output_clses_soft]
                    visual_attentions_sentiment_temp = [torch.cat([visual_attentions_sentiment_temp[j], clses_sentiment_temp[j]], dim=0) for j in range(len(visual_attentions_sentiment_temp))]
                visual_attentions_sentiment_temp = [e.detach().cpu().numpy() for e in visual_attentions_sentiment_temp]

                for j in range(self.category_num):
                    c_label = label[i][j].detach().cpu().numpy().tolist()
                    if c_label == 1:
                        visual_attentions_sentiment = []
                        labels_sentiment = []
                        sentiment_true_index = int(label[i][j + self.category_num].detach().cpu().numpy().tolist())
                        if sentiment_true_index == -100:
                            continue
                        titles_sentiment = ['true: %s - pred: %s - %s' % (str(self.polarites[sentiment_true_index]),
                                                        str(pred_sentiment[j][i].detach().cpu().numpy()),
                                                        str(self.polarites))]
                        c_attention = embedding_layer_category_alphas[j][i][: len(words)].detach().cpu().numpy()
                        if self.configuration['concat_cls_vector']:
                            c_attention = np.array([1] + c_attention.tolist())
                        visual_attentions_sentiment.append(c_attention)
                        labels_sentiment.append(self.categories[j].split('/')[0])

                        s_distributions = visual_attentions_sentiment_temp[j]
                        for k in range(self.polarity_num):
                            labels_sentiment.append(self.polarites[k])
                            visual_attentions_sentiment.append(s_distributions[:, k])
                        titles_sentiment.extend([''] * 3)
                        # savefig_filepath = super()._get_model_visualization_picture_filepath(self.configuration, words)
                        # attention_visualizer.plot_multi_attentions_of_sentence(words, visual_attentions_sentiment,
                        #                                                        labels_sentiment,
                        #                                                        titles_sentiment, savefig_filepath)
                        attention_visualizer.plot_multi_attentions_of_sentence_backup(words, visual_attentions_sentiment,
                                                                               labels_sentiment,
                                                                               titles_sentiment)
        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {
            'accuracy': self._accuracy.get_metric(reset),
            'category_f1': self._f1.get_metric(reset)['fscore']
        }
        return metrics


class AsMilSimultaneouslyBertBackup(TextInAllAspectSentimentOutModel):
    def __init__(self, word_embedder: TextFieldEmbedder, position_embedder: TextFieldEmbedder,
                 categories: list, polarities: list, vocab: Vocabulary, configuration: dict,
                 bert_word_embedder: TextFieldEmbedder=None):
        super().__init__(vocab)
        self.configuration = configuration
        self.word_embedder = word_embedder
        self.position_embedder = position_embedder
        self.categories = categories
        self.polarites = polarities
        self.category_num = len(categories)
        self.polarity_num = len(polarities)
        self.category_loss = nn.BCEWithLogitsLoss()
        self.sentiment_loss = nn.CrossEntropyLoss()
        self._accuracy = metrics.CategoricalAccuracy()
        self._f1 = allennlp_metrics.BinaryF1(0.5)

        word_embedding_dim = word_embedder.get_output_dim()
        if self.configuration['lstm_or_fc_after_embedding_layer'] == 'fc':
            self.embedding_layer_fc = nn.Linear(word_embedding_dim, word_embedding_dim, bias=True)
        elif self.configuration['lstm_or_fc_after_embedding_layer'] == 'bilstm':
            self.embedding_layer_lstm = torch.nn.LSTM(word_embedding_dim, int(word_embedding_dim / 2), batch_first=True,
                                                      bidirectional=True, num_layers=1)
        else:
            self.embedding_layer_lstm = torch.nn.LSTM(word_embedding_dim, word_embedding_dim, batch_first=True,
                                                      bidirectional=False, num_layers=1)

        self.embedding_layer_aspect_attentions = [AttentionInHtt(word_embedding_dim,
                                                                 word_embedding_dim)
                                                  for _ in range(self.category_num)]
        self.embedding_layer_aspect_attentions = nn.ModuleList(self.embedding_layer_aspect_attentions)

        self.category_fcs = [nn.Linear(word_embedding_dim, 1) for _ in range(self.category_num)]
        self.category_fcs = nn.ModuleList(self.category_fcs)

        if self.configuration['lstm_layer_num_in_bert'] != 0:
            num_layers = self.configuration['lstm_layer_num_in_bert']
            bilstm_hidden_size_in_bert = self.configuration['bilstm_hidden_size_in_bert']
            if bilstm_hidden_size_in_bert == 0:
                bilstm_hidden_size_in_bert = int(word_embedding_dim / 2)
            self.lstm = torch.nn.LSTM(768, bilstm_hidden_size_in_bert, batch_first=True,
                                      bidirectional=True, num_layers=num_layers,
                                      dropout=self.configuration['dropout_in_bert'])
            hidden_size = bilstm_hidden_size_in_bert * 2
        else:
            hidden_size = 768
        if self.configuration['only_bert']:
            self.sentiment_fc = nn.Sequential(
                # nn.Linear(768, 768),
                # nn.ReLU(),
                nn.Linear(768, self.polarity_num))
        else:
            self.sentiment_fc = nn.Sequential(
                # nn.Linear(hidden_size, hidden_size),
                # nn.ReLU(),
                nn.Linear(hidden_size, self.polarity_num))
        self.bert_word_embedder = bert_word_embedder

        self.dropout_after_embedding_layer = nn.Dropout(self.configuration['dropout_in_bert'])

    def set_bert_word_embedder(self, bert_word_embedder: TextFieldEmbedder=None):
        self.bert_word_embedder = bert_word_embedder

    def set_grad_for_acd_parameter(self, requires_grad=True):
        acd_layers = []
        if self.configuration['lstm_or_fc_after_embedding_layer'] == 'fc':
            acd_layers.append(self.embedding_layer_fc)
        else:
            acd_layers.append(self.embedding_layer_lstm)
        acd_layers.append(self.embedding_layer_aspect_attentions)
        acd_layers.append(self.category_fcs)
        for layer in acd_layers:
            for name, value in layer.named_parameters():
                value.requires_grad = requires_grad

    def set_grad_for_acsc_parameter(self, requires_grad=True):
        acsc_layers = []
        if self.configuration['lstm_layer_num_in_bert'] != 0:
            acsc_layers.append(self.lstm)
        acsc_layers.append(self.sentiment_fc)
        for layer in acsc_layers:
            for name, value in layer.named_parameters():
                value.requires_grad = requires_grad
        bert_model = self.bert_word_embedder._token_embedders['bert'].bert_model
        for param in bert_model.parameters():
            param.requires_grad = requires_grad

    def forward(self, tokens: Dict[str, torch.Tensor], label: torch.Tensor, position: torch.Tensor,
                polarity_mask: torch.Tensor, sample: list, bert: torch.Tensor) -> torch.Tensor:
        bert_mask = bert['mask']
        # bert_word_embeddings = self.bert_word_embedder(bert)
        token_type_ids = bert['bert-type-ids']
        # token_type_ids_size = token_type_ids.size()
        # for i in range(token_type_ids_size[1]):
        #     print(token_type_ids[0][i])
        offsets = bert['bert-offsets']
        bert_word_embeddings = self.bert_word_embedder(bert, token_type_ids=token_type_ids, offsets=offsets)

        mask = get_text_field_mask(tokens)
        word_embeddings = self.word_embedder(tokens)
        word_embeddings_size = word_embeddings.size()
        if self.configuration['lstm_or_fc_after_embedding_layer'] == 'fc':
            word_embeddings_fc = self.embedding_layer_fc(word_embeddings)
        else:
            word_embeddings_fc, (_, _) = self.embedding_layer_lstm(word_embeddings)

        embedding_layer_category_outputs = []
        embedding_layer_category_alphas = []
        embedding_layer_sentiment_outputs = []
        embedding_layer_sentiment_alphas = []
        bert_clses_of_all_aspect = []
        for i in range(self.category_num):
            embedding_layer_aspect_attention = self.embedding_layer_aspect_attentions[i]
            alpha = embedding_layer_aspect_attention(word_embeddings_fc, mask)
            embedding_layer_category_alphas.append(alpha)

            category_output = self.element_wise_mul(word_embeddings_fc, alpha, return_not_sum_result=False)
            embedding_layer_category_outputs.append(category_output)

            bert_clses_of_aspect = bert_word_embeddings[:, i, 0, :]
            bert_clses_of_all_aspect.append(bert_clses_of_aspect)

            if not self.configuration['only_bert']:
                bert_word_embeddings_of_aspect = bert_word_embeddings[:, i, :, :]
                aspect_word_embeddings_from_bert = []
                for j in range(len(sample)):
                    aspect_word_embeddings_from_bert_of_one_sample = []
                    all_word_indices_in_bert = sample[j][6]
                    for k in range(word_embeddings_size[1]):
                        if k in all_word_indices_in_bert:
                            word_indices_in_bert = all_word_indices_in_bert[k]
                            word_bert_embeddings = []
                            for word_index_in_bert in word_indices_in_bert:
                                word_bert_embedding = bert_word_embeddings_of_aspect[j][word_index_in_bert]
                                word_bert_embeddings.append(word_bert_embedding)
                            if len(word_bert_embeddings) == 0:
                                print()
                            if len(word_bert_embeddings) > 1:
                                word_bert_embeddings_unsqueeze = [torch.unsqueeze(e, dim=0) for e in word_bert_embeddings]
                                word_bert_embeddings_cat = torch.cat(word_bert_embeddings_unsqueeze, dim=0)
                                word_bert_embeddings_sum = torch.sum(word_bert_embeddings_cat, dim=0)
                                word_bert_embeddings_ave = word_bert_embeddings_sum / len(word_bert_embeddings)
                            else:
                                word_bert_embeddings_ave = word_bert_embeddings[0]
                            aspect_word_embeddings_from_bert_of_one_sample.append(
                                torch.unsqueeze(word_bert_embeddings_ave, 0))
                        else:
                            zero = torch.zeros_like(aspect_word_embeddings_from_bert_of_one_sample[-1])
                            aspect_word_embeddings_from_bert_of_one_sample.append(zero)
                    aspect_word_embeddings_from_bert_of_one_sample_cat = torch.cat(aspect_word_embeddings_from_bert_of_one_sample, dim=0)
                    aspect_word_embeddings_from_bert.append(torch.unsqueeze(aspect_word_embeddings_from_bert_of_one_sample_cat, dim=0))
                aspect_word_embeddings_from_bert_cat = torch.cat(aspect_word_embeddings_from_bert, dim=0)
                if self.configuration['lstm_layer_num_in_bert'] != 0:
                    aspect_word_embeddings_from_bert_cat, _ = self.lstm(aspect_word_embeddings_from_bert_cat)
                embedding_layer_sentiment_outputs.append(aspect_word_embeddings_from_bert_cat)

        lstm_layer_category_outputs = []
        lstm_layer_sentiment_outputs = []
        lstm_layer_words_sentiment_soft = []
        sentiment_output_clses_soft = []
        for i in range(self.category_num):
            sentiment_output_temp = bert_clses_of_all_aspect[i]
            if self.configuration['dropout_after_cls']:
                sentiment_output_temp = self.dropout_after_embedding_layer(sentiment_output_temp)
            sentiment_output_cls = self.sentiment_fc(sentiment_output_temp)
            sentiment_output_clses_soft.append(torch.softmax(sentiment_output_cls, dim=-1))
            if self.configuration['only_bert']:
                sentiment_output = sentiment_output_cls
                lstm_layer_sentiment_outputs.append(sentiment_output)
            else:
                # sentiment
                aspect_word_embeddings_from_bert = embedding_layer_sentiment_outputs[i]
                word_representation_for_sentiment = self.dropout_after_embedding_layer(aspect_word_embeddings_from_bert)

                sentiment_alpha = embedding_layer_category_alphas[i]
                if self.configuration['mil']:
                    sentiment_alpha = sentiment_alpha.unsqueeze(1)
                    words_sentiment = self.sentiment_fc(word_representation_for_sentiment)
                    if self.configuration['mil_softmax']:
                        words_sentiment_soft = torch.softmax(words_sentiment, dim=-1)
                        lstm_layer_words_sentiment_soft.append(words_sentiment_soft)
                    else:
                        words_sentiment_soft = words_sentiment
                        lstm_layer_words_sentiment_soft.append(torch.softmax(words_sentiment, dim=-1))
                    sentiment_output_mil = torch.matmul(sentiment_alpha, words_sentiment_soft).squeeze(1)  # batch_size x 2*hidden_dim
                    if self.configuration['concat_cls_vector']:
                        if self.configuration['concat_cls_vector_mode'] == 'average':
                            sentiment_output = (sentiment_output_mil + sentiment_output_cls) / 2
                        else:
                            sentiment_output = sentiment_output_mil + sentiment_output_cls
                    else:
                        sentiment_output = sentiment_output_mil
                    lstm_layer_sentiment_outputs.append(sentiment_output)
                else:
                    sentiment_output_temp = self.element_wise_mul(word_representation_for_sentiment, sentiment_alpha,
                                                                  return_not_sum_result=False)
                    sentiment_output_not_mil = self.sentiment_fc(sentiment_output_temp)
                    if self.configuration['concat_cls_vector']:
                        if self.configuration['concat_cls_vector_mode'] == 'average':
                            sentiment_output = (sentiment_output_not_mil + sentiment_output_cls) / 2
                        else:
                            sentiment_output = sentiment_output_not_mil + sentiment_output_cls
                    else:
                        sentiment_output = sentiment_output_not_mil
                    lstm_layer_sentiment_outputs.append(sentiment_output)

        final_category_outputs = []
        final_lstm_category_outputs = []
        final_sentiment_outputs = []
        for i in range(self.category_num):
            fc = self.category_fcs[i]
            category_output = embedding_layer_category_outputs[i]
            final_category_output = fc(category_output)
            final_category_outputs.append(final_category_output)

            sentiment_output = lstm_layer_sentiment_outputs[i]
            final_sentiment_output = sentiment_output
            final_sentiment_outputs.append(final_sentiment_output)

        output = {}
        if label is not None:
            category_labels = []
            polarity_labels = []
            polarity_masks = []
            for i in range(self.category_num):
                category_labels.append(label[:, i])
                polarity_labels.append(label[:, i + self.category_num])
                polarity_masks.append(polarity_mask[:, i])
            loss = 0
            total_category_loss = 0
            total_sentiment_loss = 0
            for i in range(self.category_num):
                category_temp_loss = self.category_loss(final_category_outputs[i].squeeze(dim=-1), category_labels[i])
                sentiment_temp_loss = self.sentiment_loss(final_sentiment_outputs[i], polarity_labels[i].long())
                if not self.configuration['only_sc']:
                    total_category_loss += category_temp_loss
                if not self.configuration['only_acd']:
                    total_sentiment_loss += sentiment_temp_loss

            loss = self.category_loss_weight * total_category_loss + self.sentiment_loss_weight * total_sentiment_loss

            # sentiment accuracy
            sentiment_logit = torch.cat(final_sentiment_outputs)
            sentiment_label = torch.cat(polarity_labels)
            sentiment_mask = torch.cat(polarity_masks)
            self._accuracy(sentiment_logit, sentiment_label, sentiment_mask)

            # category f1
            final_category_outputs_prob = [torch.sigmoid(e) for e in final_category_outputs]
            category_prob = torch.cat(final_category_outputs_prob).squeeze()
            category_label = torch.cat(category_labels)
            self._f1(category_prob, category_label)

            output['loss'] = loss

        # visualize attention
        pred_category = [torch.sigmoid(e) for e in final_category_outputs]
        pred_sentiment = [torch.nn.functional.softmax(e, dim=-1) for e in final_sentiment_outputs]
        output['pred_category'] = pred_category
        output['pred_sentiment'] = pred_sentiment
        output['embedding_layer_category_alphas'] = embedding_layer_category_alphas
        output['lstm_layer_words_sentiment_soft'] = lstm_layer_words_sentiment_soft
        if self.configuration['visualize_attention']:
            for i in range(len(sample)):
                words: list = sample[i][2]
                # if not ('while' in words and 'it' in words):
                #     continue
                attention_labels = [e.split('/')[0] for e in self.categories]

                # category
                visual_attentions_category = [embedding_layer_category_alphas[j][i][: len(words)].detach().cpu().numpy()
                                              for j in range(self.category_num)]
                titles = ['true: %s - pred: %s' % (str(label[i][j].detach().cpu().numpy()),
                                                   str(pred_category[j][i].detach().cpu().numpy()))
                          for j in range(self.category_num)]
                # savefig_filepath = super()._get_model_visualization_picture_filepath(self.configuration, words)
                # attention_visualizer.plot_multi_attentions_of_sentence(words, visual_attentions_category,
                #                                                        attention_labels, titles, savefig_filepath)
                attention_visualizer.plot_multi_attentions_of_sentence_backup(words, visual_attentions_category,
                                                                       attention_labels, titles)

                # sentiment lstm layer
                visual_attentions_sentiment_temp = [lstm_layer_words_sentiment_soft[j][i][: len(words)]
                                                    for j in range(self.category_num)]
                if self.configuration['concat_cls_vector']:
                    words.insert(0, '[CLS]')
                    clses_sentiment_temp = [e.unsqueeze(dim=1)[i] for e in sentiment_output_clses_soft]
                    visual_attentions_sentiment_temp = [torch.cat([visual_attentions_sentiment_temp[j], clses_sentiment_temp[j]], dim=0) for j in range(len(visual_attentions_sentiment_temp))]
                visual_attentions_sentiment_temp = [e.detach().cpu().numpy() for e in visual_attentions_sentiment_temp]

                for j in range(self.category_num):
                    c_label = label[i][j].detach().cpu().numpy().tolist()
                    if c_label == 1:
                        visual_attentions_sentiment = []
                        labels_sentiment = []
                        sentiment_true_index = int(label[i][j + self.category_num].detach().cpu().numpy().tolist())
                        if sentiment_true_index == -100:
                            continue
                        titles_sentiment = ['true: %s - pred: %s - %s' % (str(self.polarites[sentiment_true_index]),
                                                        str(pred_sentiment[j][i].detach().cpu().numpy()),
                                                        str(self.polarites))]
                        c_attention = embedding_layer_category_alphas[j][i][: len(words)].detach().cpu().numpy()
                        if self.configuration['concat_cls_vector']:
                            c_attention = np.array([1] + c_attention.tolist())
                        visual_attentions_sentiment.append(c_attention)
                        labels_sentiment.append(self.categories[j].split('/')[0])

                        s_distributions = visual_attentions_sentiment_temp[j]
                        for k in range(self.polarity_num):
                            labels_sentiment.append(self.polarites[k])
                            visual_attentions_sentiment.append(s_distributions[:, k])
                        titles_sentiment.extend([''] * 3)
                        # savefig_filepath = super()._get_model_visualization_picture_filepath(self.configuration, words)
                        # attention_visualizer.plot_multi_attentions_of_sentence(words, visual_attentions_sentiment,
                        #                                                        labels_sentiment,
                        #                                                        titles_sentiment, savefig_filepath)
                        attention_visualizer.plot_multi_attentions_of_sentence_backup(words, visual_attentions_sentiment,
                                                                               labels_sentiment,
                                                                               titles_sentiment)
        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {
            'accuracy': self._accuracy.get_metric(reset),
            'category_f1': self._f1.get_metric(reset)['fscore']
        }
        return metrics


class ConstituencyBert(TextInAllAspectSentimentOutModel):
    def __init__(self, word_embedder: TextFieldEmbedder, position_embedder: TextFieldEmbedder,
                 categories: list, polarities: list, vocab: Vocabulary, configuration: dict,
                 bert_word_embedder: TextFieldEmbedder=None):
        super().__init__(vocab)
        self.configuration = configuration
        self.word_embedder = word_embedder
        self.position_embedder = position_embedder
        self.categories = categories
        self.polarites = polarities
        self.category_num = len(categories)
        self.polarity_num = len(polarities)
        self.category_loss = nn.BCEWithLogitsLoss()
        self.sentiment_loss = nn.CrossEntropyLoss()
        self._accuracy = metrics.CategoricalAccuracy()
        self._f1 = allennlp_metrics.BinaryF1(0.5)

        word_embedding_dim = word_embedder.get_output_dim()
        aspect_encoder_input_size = word_embedding_dim
        if self.configuration['aspect_position']:
            aspect_encoder_input_size += position_embedder.get_output_dim()
        if self.configuration['lstm_or_fc_after_embedding_layer'] == 'fc':
            self.embedding_layer_fc = nn.Linear(aspect_encoder_input_size, word_embedding_dim, bias=True)
        elif self.configuration['lstm_or_fc_after_embedding_layer'] == 'bilstm':
            self.embedding_layer_lstm = torch.nn.LSTM(aspect_encoder_input_size, int(word_embedding_dim / 2), batch_first=True,
                                                      bidirectional=True, num_layers=1)
        else:
            self.embedding_layer_lstm = torch.nn.LSTM(aspect_encoder_input_size, word_embedding_dim, batch_first=True,
                                                      bidirectional=False, num_layers=1)

        self.embedding_layer_aspect_attentions = [AttentionInHtt(word_embedding_dim,
                                                                 word_embedding_dim)
                                                  for _ in range(self.category_num)]
        self.embedding_layer_aspect_attentions = nn.ModuleList(self.embedding_layer_aspect_attentions)

        self.category_fcs = [nn.Linear(word_embedding_dim, 1) for _ in range(self.category_num)]
        self.category_fcs = nn.ModuleList(self.category_fcs)

        if self.configuration['lstm_layer_num_in_bert'] != 0:
            num_layers = self.configuration['lstm_layer_num_in_bert']
            bilstm_hidden_size_in_bert = self.configuration['bilstm_hidden_size_in_bert']
            if bilstm_hidden_size_in_bert == 0:
                bilstm_hidden_size_in_bert = int(word_embedding_dim / 2)
            self.lstm = torch.nn.LSTM(768, bilstm_hidden_size_in_bert, batch_first=True,
                                      bidirectional=True, num_layers=num_layers,
                                      dropout=self.configuration['dropout_in_bert'])
            hidden_size = bilstm_hidden_size_in_bert * 2
        else:
            hidden_size = 768
        if self.configuration['only_bert']:
            self.sentiment_fc = nn.Sequential(
                # nn.Linear(768, 768),
                # nn.ReLU(),
                nn.Linear(768, self.polarity_num))
        else:
            self.sentiment_fc = nn.Sequential(
                # nn.Linear(hidden_size, hidden_size),
                # nn.ReLU(),
                nn.Linear(hidden_size, self.polarity_num))
        self.sentiment_fc_for_attention_warmup = nn.Sequential(
            # nn.Linear(hidden_size, hidden_size),
            # nn.ReLU(),
            nn.Linear(word_embedding_dim, self.polarity_num))
        self.bert_word_embedder = bert_word_embedder

        self.dropout_after_embedding_layer = nn.Dropout(self.configuration['dropout_in_bert'])

        # 只负责生成针对aspect category的节点表示
        aspect_graph = self.configuration['aspect_graph']
        if aspect_graph == 'attention':
            # 1. 假设：两个词被attention关注，这两个词的平均也被attention关注。todo 需要验证这个假设是否成立？
            # 2. 对于给定属性，应该关注哪些内部节点？
            # (1) 关注子节点中包含与该属性相关的节点，
            # (2) 不关注子节点中包含其它属性相关的节点
            # 3. 之前的方法为什么会关注到其它属性的情感词？
            # (1) 因为很多情感词对不同属性通用的，只是基于情感分词本身很难判断这个情感词是否是描述该属性的
            # (2) 我们的解决方案，借助于情感词的父节点的其它子节点的信息来帮助判断这个情感分词是否是描述该属性的，
            # 具体的，如果情感词的父节点的其它子节点包含其它属性的信息，那么关注这个情感词，将会受到惩罚
            self.gnn_for_aspect = DglGraphAttentionForAspectCategory(word_embedding_dim, word_embedding_dim,
                                                                     self.embedding_layer_aspect_attentions,
                                                                     configuration)
        elif aspect_graph == 'attention_with_dotted_lines':
            self.gnn_for_aspect = DglGraphAttentionForAspectCategoryWithDottedLines(word_embedding_dim,
                                                                                    word_embedding_dim,
                                                                                    self.embedding_layer_aspect_attentions,
                                                                                    configuration)
        elif aspect_graph == 'average':
            self.gnn_for_aspect = DglGraphAverageForAspectCategory(word_embedding_dim, word_embedding_dim,
                                                                   self.configuration)
        elif aspect_graph == 'gcn':
            self.gnn_for_aspect = DglGraphConvolutionForAspectCategory(word_embedding_dim, word_embedding_dim,
                                                                       self.configuration)
        elif aspect_graph == 'gat':
            self.gnn_for_aspect = GATForAspectCategory(word_embedding_dim, word_embedding_dim, opt=self.configuration)
        else:
            raise NotImplementedError(aspect_graph)

        sentiment_graph = self.configuration['sentiment_graph']
        if sentiment_graph == 'average':
            self.gnn_for_sentiment = DglGraphAverage(hidden_size, hidden_size, self.configuration)
        elif sentiment_graph == 'gcn':
            self.gnn_for_sentiment = DglGraphConvolution(hidden_size, hidden_size, configuration)
        elif sentiment_graph == 'gat':
            self.gnn_for_sentiment = GAT(hidden_size, hidden_size, hidden_size, 4,
                                         self.configuration)
        else:
            raise NotImplementedError(sentiment_graph)

        if self.configuration['acd_encoder_mode_for_sentiment_attention_warmup'] == 'mixed':
            if self.configuration['gnn_for_sentiment_attention_warmup'] == 'average':
                self.gnn_for_sentiment_attention_warmup = DglGraphAverage(word_embedding_dim, word_embedding_dim,
                                                                          self.configuration)
            else:
                self.gnn_for_sentiment_attention_warmup = GAT(word_embedding_dim, word_embedding_dim,
                                                              word_embedding_dim, 4, self.configuration)

    def set_bert_word_embedder(self, bert_word_embedder: TextFieldEmbedder=None):
        self.bert_word_embedder = bert_word_embedder

    def set_grad_for_acd_parameter(self, requires_grad=True):
        acd_layers = []
        if self.configuration['lstm_or_fc_after_embedding_layer'] == 'fc':
            acd_layers.append(self.embedding_layer_fc)
        else:
            acd_layers.append(self.embedding_layer_lstm)
        acd_layers.append(self.embedding_layer_aspect_attentions)
        acd_layers.append(self.category_fcs)
        acd_layers.append(self.gnn_for_aspect)
        for layer in acd_layers:
            for name, value in layer.named_parameters():
                value.requires_grad = requires_grad

    def set_grad_for_acsc_parameter(self, requires_grad=True):
        acsc_layers = []
        if self.configuration['lstm_layer_num_in_bert'] != 0:
            acsc_layers.append(self.lstm)
        acsc_layers.append(self.sentiment_fc)
        acsc_layers.append(self.gnn_for_sentiment)
        for layer in acsc_layers:
            for name, value in layer.named_parameters():
                value.requires_grad = requires_grad
        bert_model = self.bert_word_embedder._token_embedders['bert'].bert_model
        for param in bert_model.parameters():
            param.requires_grad = requires_grad

    def forward(self, tokens: Dict[str, torch.Tensor], tokens_for_graph: Dict[str, torch.Tensor], label: torch.Tensor,
                position: torch.Tensor, polarity_mask: torch.Tensor, sample: list, bert: torch.Tensor) -> torch.Tensor:
        bert_mask = bert['mask']
        # bert_word_embeddings = self.bert_word_embedder(bert)
        token_type_ids = bert['bert-type-ids']
        offsets = bert['bert-offsets']
        bert_word_embeddings = self.bert_word_embedder(bert, token_type_ids=token_type_ids, offsets=offsets)

        mask = get_text_field_mask(tokens)

        mask_for_graph = get_text_field_mask(tokens_for_graph)

        max_len_for_graph = mask_for_graph.size()[1]
        max_len = mask.size()[1]
        padder = nn.ConstantPad2d((0, 0, 0, max_len_for_graph - max_len), 0)

        graphs = [e['graph'] for e in sample]
        graphs_padded = self.pad_dgl_graph(graphs, max_len_for_graph)

        graphs_with_dotted_line = [e['graph_with_dotted_line'] for e in sample]
        graphs_with_dotted_line_padded = self.pad_dgl_graph(graphs_with_dotted_line, max_len_for_graph)
        if not self.configuration['aspect_graph_with_dotted_line']:
            graphs_with_dotted_line_padded = graphs_padded

        word_embeddings = self.word_embedder(tokens)
        word_embeddings_size = word_embeddings.size()
        if self.configuration['lstm_or_fc_after_embedding_layer'] == 'fc':
            word_embeddings_fc = self.embedding_layer_fc(word_embeddings)
        else:
            word_embeddings_fc, (_, _) = self.embedding_layer_lstm(word_embeddings)

        # 在word_embeddings_fc上做图卷积，attention是基于图卷积的输出
        word_embeddings_fc_for_graph = padder(word_embeddings_fc)
        graph_output_for_aspect = self.gnn_for_aspect(word_embeddings_fc_for_graph, graphs_with_dotted_line_padded)
        graph_output_for_aspect = self.gnn_for_aspect(word_embeddings_fc_for_graph, graphs_with_dotted_line_padded)

        # todo
        if isinstance(graph_output_for_aspect, tuple):
            graph_output_for_aspect = graph_output_for_aspect[0]

        if self.configuration['acd_encoder_mode_for_sentiment_attention_warmup'] == 'same':
            lstm_result_for_graph = graph_output_for_aspect
        else:
            lstm_result_for_graph = self.gnn_for_sentiment_attention_warmup(word_embeddings_fc_for_graph,
                                                                            graphs_with_dotted_line_padded)
        # todo
        if isinstance(lstm_result_for_graph, tuple):
            lstm_result_for_graph = lstm_result_for_graph[0]

        bert_clses_of_all_aspect = []
        embedding_layer_sentiment_outputs = []
        for i in range(self.category_num):
            bert_clses_of_aspect = bert_word_embeddings[:, i, 0, :]
            bert_clses_of_all_aspect.append(bert_clses_of_aspect)

            if not self.configuration['only_bert']:
                bert_word_embeddings_of_aspect = bert_word_embeddings[:, i, :, :]
                aspect_word_embeddings_from_bert = []
                for j in range(len(sample)):
                    aspect_word_embeddings_from_bert_of_one_sample = []
                    all_word_indices_in_bert = sample[j]['word_index_and_bert_indices']
                    for k in range(word_embeddings_size[1]):
                        if k in all_word_indices_in_bert:
                            word_indices_in_bert = all_word_indices_in_bert[k]
                            word_bert_embeddings = []
                            for word_index_in_bert in word_indices_in_bert:
                                word_bert_embedding = bert_word_embeddings_of_aspect[j][word_index_in_bert]
                                word_bert_embeddings.append(word_bert_embedding)
                            if len(word_bert_embeddings) == 0:
                                print()
                            if len(word_bert_embeddings) > 1:
                                word_bert_embeddings_unsqueeze = [torch.unsqueeze(e, dim=0) for e in
                                                                  word_bert_embeddings]
                                word_bert_embeddings_cat = torch.cat(word_bert_embeddings_unsqueeze, dim=0)
                                word_bert_embeddings_sum = torch.sum(word_bert_embeddings_cat, dim=0)
                                word_bert_embeddings_ave = word_bert_embeddings_sum / len(word_bert_embeddings)
                            else:
                                word_bert_embeddings_ave = word_bert_embeddings[0]
                            aspect_word_embeddings_from_bert_of_one_sample.append(
                                torch.unsqueeze(word_bert_embeddings_ave, 0))
                        else:
                            zero = torch.zeros_like(aspect_word_embeddings_from_bert_of_one_sample[-1])
                            aspect_word_embeddings_from_bert_of_one_sample.append(zero)
                    aspect_word_embeddings_from_bert_of_one_sample_cat = torch.cat(
                        aspect_word_embeddings_from_bert_of_one_sample, dim=0)
                    aspect_word_embeddings_from_bert.append(
                        torch.unsqueeze(aspect_word_embeddings_from_bert_of_one_sample_cat, dim=0))
                aspect_word_embeddings_from_bert_cat = torch.cat(aspect_word_embeddings_from_bert, dim=0)
                if self.configuration['lstm_layer_num_in_bert'] != 0:
                    aspect_word_embeddings_from_bert_cat, _ = self.lstm(aspect_word_embeddings_from_bert_cat)

                # 在lstm_result上做图卷积
                aspect_word_embeddings_from_bert_cat_padded_for_graph = padder(aspect_word_embeddings_from_bert_cat)

                sentiment_encoder_with_own_gnn = self.configuration['sentiment_encoder_with_own_gnn']
                if sentiment_encoder_with_own_gnn:
                    aspect_word_embeddings_from_bert_cat_result_for_graph = self.gnn_for_sentiment(
                        aspect_word_embeddings_from_bert_cat_padded_for_graph, graphs_with_dotted_line_padded)
                else:
                    # attention
                    _, aspect_word_embeddings_from_bert_cat_result_for_graph = self.gnn_for_aspect(
                        word_embeddings_fc_for_graph,
                        graphs_with_dotted_line_padded,
                       aspect_word_embeddings_from_bert_cat_padded_for_graph)

                embedding_layer_sentiment_outputs.append(aspect_word_embeddings_from_bert_cat_result_for_graph)

        embedding_layer_category_outputs = []
        embedding_layer_category_alphas = []
        for i in range(self.category_num):
            embedding_layer_aspect_attention = self.embedding_layer_aspect_attentions[i]
            # todo 根据节点的深度调整权重，使得偏向上层节点
            alpha = embedding_layer_aspect_attention(graph_output_for_aspect, mask_for_graph)
            embedding_layer_category_alphas.append(alpha)

            category_output = self.element_wise_mul(graph_output_for_aspect, alpha, return_not_sum_result=False)
            embedding_layer_category_outputs.append(category_output)

        lstm_layer_sentiment_outputs = []
        lstm_layer_words_sentiment_soft = []
        sentiment_output_clses_soft = []
        for i in range(self.category_num):
            sentiment_output_temp = bert_clses_of_all_aspect[i]
            if self.configuration['dropout_after_cls']:
                sentiment_output_temp = self.dropout_after_embedding_layer(sentiment_output_temp)
            sentiment_output_cls = self.sentiment_fc(sentiment_output_temp)
            sentiment_output_clses_soft.append(torch.softmax(sentiment_output_cls, dim=-1))
            if self.configuration['only_bert']:
                sentiment_output = sentiment_output_cls
                lstm_layer_sentiment_outputs.append(sentiment_output)
            else:
                # sentiment
                aspect_word_embeddings_from_bert = embedding_layer_sentiment_outputs[i]
                word_representation_for_sentiment = self.dropout_after_embedding_layer(aspect_word_embeddings_from_bert)

                sentiment_alpha = embedding_layer_category_alphas[i]
                if self.configuration['mil']:
                    sentiment_alpha = sentiment_alpha.unsqueeze(1)
                    words_sentiment = self.sentiment_fc(word_representation_for_sentiment)
                    if self.configuration['mil_softmax']:
                        words_sentiment_soft = torch.softmax(words_sentiment, dim=-1)
                        lstm_layer_words_sentiment_soft.append(words_sentiment_soft)
                    else:
                        words_sentiment_soft = words_sentiment
                        lstm_layer_words_sentiment_soft.append(torch.softmax(words_sentiment, dim=-1))
                    sentiment_output_mil = torch.matmul(sentiment_alpha, words_sentiment_soft).squeeze(1)  # batch_size x 2*hidden_dim
                    if self.configuration['concat_cls_vector']:
                        sentiment_output = sentiment_output_mil + sentiment_output_cls
                    else:
                        sentiment_output = sentiment_output_mil
                    lstm_layer_sentiment_outputs.append(sentiment_output)
                else:
                    if self.configuration['attention_warmup']:
                        word_representation_for_sentiment = lstm_result_for_graph
                    sentiment_output_temp = self.element_wise_mul(word_representation_for_sentiment, sentiment_alpha,
                                                                  return_not_sum_result=False)
                    if self.configuration['attention_warmup']:
                        sentiment_output_not_mil = self.sentiment_fc_for_attention_warmup(sentiment_output_temp)
                    else:
                        sentiment_output_not_mil = self.sentiment_fc(sentiment_output_temp)
                    if self.configuration['concat_cls_vector'] and not self.configuration['attention_warmup']:
                        sentiment_output = sentiment_output_not_mil + sentiment_output_cls
                    else:
                        sentiment_output = sentiment_output_not_mil
                    lstm_layer_sentiment_outputs.append(sentiment_output)

        final_category_outputs = []
        final_category_outputs_auxillary = []
        final_sentiment_outputs = []
        for i in range(self.category_num):
            category_output = embedding_layer_category_outputs[i]
            for j in range(self.category_num):
                final_category_output = self.category_fcs[j](category_output)
                if i == j:
                    final_category_outputs.append(final_category_output)
                else:
                    final_category_outputs_auxillary.append(final_category_output)

            sentiment_output = lstm_layer_sentiment_outputs[i]
            final_sentiment_output = sentiment_output
            final_sentiment_outputs.append(final_sentiment_output)

        output = {}
        if label is not None:
            category_labels = []
            polarity_labels = []
            polarity_masks = []
            for i in range(self.category_num):
                category_labels.append(label[:, i])
                polarity_labels.append(label[:, i + self.category_num])
                polarity_masks.append(polarity_mask[:, i])
            loss = 0
            total_category_loss = 0
            total_sentiment_loss = 0
            for i in range(self.category_num):
                category_temp_loss = self.category_loss(final_category_outputs[i].squeeze(dim=-1), category_labels[i])
                sentiment_temp_loss = self.sentiment_loss(final_sentiment_outputs[i], polarity_labels[i].long())
                if not self.configuration['only_sc']:
                    total_category_loss += category_temp_loss
                if not self.configuration['only_acd']:
                    total_sentiment_loss += sentiment_temp_loss

            loss = self.category_loss_weight * total_category_loss + self.sentiment_loss_weight * total_sentiment_loss

            interactive_loss = 0
            for i in range(len(final_category_outputs_auxillary)):
                label_temp = torch.zeros_like(category_labels[0])
                interactive_loss_temp = self.category_loss(final_category_outputs_auxillary[i].squeeze(dim=-1),
                                                           label_temp)
                interactive_loss += interactive_loss_temp
            loss += (interactive_loss * self.configuration['interactive_loss_lamda'] / (self.category_num - 1))

            # sentiment accuracy
            sentiment_logit = torch.cat(final_sentiment_outputs)
            sentiment_label = torch.cat(polarity_labels)
            sentiment_mask = torch.cat(polarity_masks)
            self._accuracy(sentiment_logit, sentiment_label, sentiment_mask)

            # category f1
            final_category_outputs_prob = [torch.sigmoid(e) for e in final_category_outputs]
            category_prob = torch.cat(final_category_outputs_prob).squeeze()
            category_label = torch.cat(category_labels)
            self._f1(category_prob, category_label)

            output['loss'] = loss

        # visualize attention
        pred_category = [torch.sigmoid(e) for e in final_category_outputs]
        pred_sentiment = [torch.nn.functional.softmax(e, dim=-1) for e in final_sentiment_outputs]
        output['pred_category'] = pred_category
        output['pred_sentiment'] = pred_sentiment
        if self.configuration['visualize_attention']:
            for i in range(len(sample)):
                words = sample[i]['words_for_graph']
                print('inner_nodes:-------------------------------------------')
                for inner_node_index, inner_node in enumerate(sample[i]['inner_nodes']):
                    print('%d-%s-%s' % (
                    len(sample[i]['words']) + inner_node_index, inner_node.labels[0], inner_node.text))

                attention_labels = [e.split('/')[0] for e in self.categories]

                # 只可视化mams
                label_true = label[i].detach().cpu().numpy()[: self.category_num]
                if sum(label_true) <= 1:
                    continue

                # category
                visual_attentions_category = [embedding_layer_category_alphas[j][i][: len(words)].detach().cpu().numpy()
                                              for j in range(self.category_num)]

                titles_sentiment_temp = []
                for j in range(self.category_num):
                    c_label = label[i][j].detach().cpu().numpy().tolist()
                    sentiment_true_index = int(label[i][j + self.category_num].detach().cpu().numpy().tolist())
                    if sentiment_true_index == -100:
                        polarity_temp = '-100'
                    else:
                        polarity_temp = self.polarites[sentiment_true_index]
                    titles_sentiment_temp.append('true: %s - pred: %s - %s' % (str(polarity_temp),
                                                                               str(pred_sentiment[j][
                                                                                       i].detach().cpu().numpy()),
                                                                               str(self.polarites)))
                titles = ['true: %s - pred: %s - %s' % (str(label[i][j].detach().cpu().numpy()),
                                                        str(pred_category[j][i].detach().cpu().numpy()),
                                                        titles_sentiment_temp[j])
                          for j in range(self.category_num)]
                savefig_filepath = super()._get_model_visualization_picture_filepath(self.configuration, words)
                attention_visualizer.plot_multi_attentions_of_sentence(words, visual_attentions_category,
                                                                       attention_labels, titles, savefig_filepath)

                if self.configuration['mil']:
                    # sentiment lstm layer
                    visual_attentions_sentiment_temp = [lstm_layer_words_sentiment_soft[j][i][: len(words)]
                                                        for j in range(self.category_num)]
                    if self.configuration['concat_cls_vector']:
                        words.insert(0, '[CLS]')
                        clses_sentiment_temp = [e.unsqueeze(dim=1)[i] for e in sentiment_output_clses_soft]
                        visual_attentions_sentiment_temp = [torch.cat([visual_attentions_sentiment_temp[j], clses_sentiment_temp[j]], dim=0) for j in range(len(visual_attentions_sentiment_temp))]
                    visual_attentions_sentiment_temp = [e.detach().cpu().numpy() for e in visual_attentions_sentiment_temp]

                    for j in range(self.category_num):
                        c_label = label[i][j].detach().cpu().numpy().tolist()
                        if c_label == 1:
                            visual_attentions_sentiment = []
                            labels_sentiment = []
                            sentiment_true_index = int(label[i][j + self.category_num].detach().cpu().numpy().tolist())
                            if sentiment_true_index == -100:
                                continue
                            titles_sentiment = ['true: %s - pred: %s - %s' % (str(self.polarites[sentiment_true_index]),
                                                            str(pred_sentiment[j][i].detach().cpu().numpy()),
                                                            str(self.polarites))]
                            c_attention = embedding_layer_category_alphas[j][i][: len(words)].detach().cpu().numpy()
                            if self.configuration['concat_cls_vector']:
                                c_attention = np.array([1] + c_attention.tolist())
                            visual_attentions_sentiment.append(c_attention)
                            labels_sentiment.append(self.categories[j].split('/')[0])

                            s_distributions = visual_attentions_sentiment_temp[j]
                            for k in range(self.polarity_num):
                                labels_sentiment.append(self.polarites[k])
                                visual_attentions_sentiment.append(s_distributions[:, k])
                            titles_sentiment.extend([''] * 3)
                            savefig_filepath = super()._get_model_visualization_picture_filepath(self.configuration, words)
                            attention_visualizer.plot_multi_attentions_of_sentence(words, visual_attentions_sentiment,
                                                                                   labels_sentiment,
                                                                                   titles_sentiment, savefig_filepath)
        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {
            'accuracy': self._accuracy.get_metric(reset),
            'category_f1': self._f1.get_metric(reset)['fscore']
        }
        return metrics


class AsMilSimultaneouslyBertSingle(TextInAllAspectSentimentOutModel):
    def __init__(self, word_embedder: TextFieldEmbedder, position_embedder: TextFieldEmbedder,
                 categories: list, polarities: list, vocab: Vocabulary, configuration: dict,
                 bert_word_embedder: TextFieldEmbedder=None):
        super().__init__(vocab)
        self.configuration = configuration
        self.word_embedder = word_embedder
        self.position_embedder = position_embedder
        self.categories = categories
        self.polarites = polarities
        self.category_num = len(categories)
        self.polarity_num = len(polarities)
        self.category_loss = nn.BCEWithLogitsLoss()
        self.sentiment_loss = nn.CrossEntropyLoss()
        self._accuracy = metrics.CategoricalAccuracy()
        self._f1 = allennlp_metrics.BinaryF1(0.5)

        word_embedding_dim = word_embedder.get_output_dim()
        if self.configuration['lstm_or_fc_after_embedding_layer'] == 'fc':
            self.embedding_layer_fc = nn.Linear(word_embedding_dim, word_embedding_dim, bias=True)
        else:
            self.embedding_layer_lstm = torch.nn.LSTM(word_embedding_dim, word_embedding_dim, batch_first=True,
                                                      bidirectional=False, num_layers=1)

        self.embedding_layer_aspect_attentions = [AttentionInHtt(word_embedding_dim,
                                                                 word_embedding_dim)
                                                  for _ in range(self.category_num)]
        self.embedding_layer_aspect_attentions = nn.ModuleList(self.embedding_layer_aspect_attentions)

        self.category_fcs = [nn.Linear(word_embedding_dim, 1) for _ in range(self.category_num)]
        self.category_fcs = nn.ModuleList(self.category_fcs)

        if self.configuration['lstm_layer_num_in_bert'] != 0:
            num_layers = self.configuration['lstm_layer_num_in_bert']
            self.lstm = torch.nn.LSTM(768, int(word_embedding_dim / 2), batch_first=True,
                                      bidirectional=True, num_layers=num_layers,
                                      dropout=self.configuration['dropout_in_bert'])
            hidden_size = word_embedding_dim
        else:
            hidden_size = 768
        self.sentiment_fc = nn.Sequential(nn.Linear(hidden_size, self.polarity_num))

        self.bert_word_embedder = bert_word_embedder

        self.dropout_after_embedding_layer = nn.Dropout(self.configuration['dropout_in_bert'])

    def set_grad_for_acd_parameter(self, requires_grad=True):
        acd_layers = []
        if self.configuration['lstm_or_fc_after_embedding_layer'] == 'fc':
            acd_layers.append(self.embedding_layer_fc)
        else:
            acd_layers.append(self.embedding_layer_lstm)
        acd_layers.append(self.embedding_layer_aspect_attentions)
        acd_layers.append(self.category_fcs)
        for layer in acd_layers:
            for name, value in layer.named_parameters():
                value.requires_grad = requires_grad

    def set_grad_for_acsc_parameter(self, requires_grad=True):
        acsc_layers = []
        if self.configuration['lstm_layer_num_in_bert'] != 0:
            acsc_layers.append(self.lstm)
        acsc_layers.append(self.sentiment_fc)
        for layer in acsc_layers:
            for name, value in layer.named_parameters():
                value.requires_grad = requires_grad
        bert_model = self.bert_word_embedder._token_embedders['bert'].bert_model
        for param in bert_model.parameters():
            param.requires_grad = requires_grad

    def set_bert_word_embedder(self, bert_word_embedder: TextFieldEmbedder=None):
        self.bert_word_embedder = bert_word_embedder

    def forward(self, tokens: Dict[str, torch.Tensor], label: torch.Tensor, position: torch.Tensor,
                polarity_mask: torch.Tensor, sample: list, bert: torch.Tensor) -> torch.Tensor:
        bert_mask = bert['mask']
        bert_word_embeddings = self.bert_word_embedder(bert)

        mask = get_text_field_mask(tokens)
        word_embeddings = self.word_embedder(tokens)
        word_embeddings_size = word_embeddings.size()
        if self.configuration['lstm_or_fc_after_embedding_layer'] == 'fc':
            word_embeddings_fc = self.embedding_layer_fc(word_embeddings)
        else:
            word_embeddings_fc, (_, _) = self.embedding_layer_lstm(word_embeddings)

        embedding_layer_category_outputs = []
        embedding_layer_category_alphas = []
        embedding_layer_sentiment_outputs = []
        embedding_layer_sentiment_alphas = []
        bert_clses_of_all_aspect = []
        for i in range(self.category_num):
            embedding_layer_aspect_attention = self.embedding_layer_aspect_attentions[i]
            alpha = embedding_layer_aspect_attention(word_embeddings_fc, mask)
            embedding_layer_category_alphas.append(alpha)

            category_output = self.element_wise_mul(word_embeddings_fc, alpha, return_not_sum_result=False)
            embedding_layer_category_outputs.append(category_output)

        bert_clses_of_aspect = bert_word_embeddings[:, 0, 0, :]
        bert_clses_of_all_aspect.append(bert_clses_of_aspect)

        if not self.configuration['only_bert']:
            bert_word_embeddings_of_aspect = bert_word_embeddings[:, 0, :, :]
            aspect_word_embeddings_from_bert = []
            for j in range(len(sample)):
                aspect_word_embeddings_from_bert_of_one_sample = []
                all_word_indices_in_bert = sample[j][6]
                for k in range(word_embeddings_size[1]):
                    if k in all_word_indices_in_bert:
                        word_indices_in_bert = all_word_indices_in_bert[k]
                        word_bert_embeddings = []
                        for word_index_in_bert in word_indices_in_bert:
                            word_bert_embedding = bert_word_embeddings_of_aspect[j][word_index_in_bert]
                            word_bert_embeddings.append(word_bert_embedding)
                        if len(word_bert_embeddings) == 0:
                            print()
                        if len(word_bert_embeddings) > 1:
                            word_bert_embeddings_unsqueeze = [torch.unsqueeze(e, dim=0) for e in word_bert_embeddings]
                            word_bert_embeddings_cat = torch.cat(word_bert_embeddings_unsqueeze, dim=0)
                            word_bert_embeddings_sum = torch.sum(word_bert_embeddings_cat, dim=0)
                            word_bert_embeddings_ave = word_bert_embeddings_sum / len(word_bert_embeddings)
                        else:
                            word_bert_embeddings_ave = word_bert_embeddings[0]
                        aspect_word_embeddings_from_bert_of_one_sample.append(
                            torch.unsqueeze(word_bert_embeddings_ave, 0))
                    else:
                        zero = torch.zeros_like(aspect_word_embeddings_from_bert_of_one_sample[-1])
                        aspect_word_embeddings_from_bert_of_one_sample.append(zero)
                aspect_word_embeddings_from_bert_of_one_sample_cat = torch.cat(
                    aspect_word_embeddings_from_bert_of_one_sample, dim=0)
                aspect_word_embeddings_from_bert.append(
                    torch.unsqueeze(aspect_word_embeddings_from_bert_of_one_sample_cat, dim=0))
            aspect_word_embeddings_from_bert_cat = torch.cat(aspect_word_embeddings_from_bert, dim=0)
            if self.configuration['lstm_layer_num_in_bert'] != 0:
                aspect_word_embeddings_from_bert_cat, _ = self.lstm(aspect_word_embeddings_from_bert_cat)
            embedding_layer_sentiment_outputs.append(aspect_word_embeddings_from_bert_cat)

        lstm_layer_category_outputs = []
        lstm_layer_sentiment_outputs = []
        lstm_layer_words_sentiment_soft = []

        for i in range(self.category_num):
            sentiment_output_temp = bert_clses_of_all_aspect[0]
            sentiment_output_cls = self.sentiment_fc(sentiment_output_temp)
            if self.configuration['only_bert']:
                sentiment_output = sentiment_output_cls
                lstm_layer_sentiment_outputs.append(sentiment_output)
            else:
                # sentiment
                aspect_word_embeddings_from_bert = embedding_layer_sentiment_outputs[0]
                word_representation_for_sentiment = self.dropout_after_embedding_layer(aspect_word_embeddings_from_bert)

                sentiment_alpha = embedding_layer_category_alphas[i]
                if self.configuration['mil']:
                    sentiment_alpha = sentiment_alpha.unsqueeze(1)
                    words_sentiment = self.sentiment_fc(word_representation_for_sentiment)
                    if self.configuration['mil_softmax']:
                        words_sentiment_soft = torch.softmax(words_sentiment, dim=-1)
                        lstm_layer_words_sentiment_soft.append(words_sentiment_soft)
                    else:
                        words_sentiment_soft = words_sentiment
                        lstm_layer_words_sentiment_soft.append(torch.softmax(words_sentiment, dim=-1))
                    sentiment_output_mil = torch.matmul(sentiment_alpha, words_sentiment_soft).squeeze(1)  # batch_size x 2*hidden_dim
                    if self.configuration['concat_cls_vector']:
                        if self.configuration['mil_softmax']:
                            sentiment_output_cls_softmax = torch.softmax(sentiment_output_cls, dim=-1)
                            sentiment_output = sentiment_output_mil + sentiment_output_cls_softmax
                        else:
                            sentiment_output = sentiment_output_mil + sentiment_output_cls
                    else:
                        sentiment_output = sentiment_output_mil
                    lstm_layer_sentiment_outputs.append(sentiment_output)
                else:
                    sentiment_output_temp = self.element_wise_mul(word_representation_for_sentiment, sentiment_alpha,
                                                                  return_not_sum_result=False)
                    sentiment_output_not_mil = self.sentiment_fc(sentiment_output_temp)
                    if self.configuration['concat_cls_vector']:
                        sentiment_output = sentiment_output_not_mil + sentiment_output_cls
                    else:
                        sentiment_output = sentiment_output_not_mil
                    lstm_layer_sentiment_outputs.append(sentiment_output)

        final_category_outputs = []
        final_lstm_category_outputs = []
        final_sentiment_outputs = []
        for i in range(self.category_num):
            fc = self.category_fcs[i]
            category_output = embedding_layer_category_outputs[i]
            final_category_output = fc(category_output)
            final_category_outputs.append(final_category_output)

            sentiment_output = lstm_layer_sentiment_outputs[i]
            final_sentiment_output = sentiment_output
            final_sentiment_outputs.append(final_sentiment_output)

        output = {}
        if label is not None:
            category_labels = []
            polarity_labels = []
            polarity_masks = []
            for i in range(self.category_num):
                category_labels.append(label[:, i])
                polarity_labels.append(label[:, i + self.category_num])
                polarity_masks.append(polarity_mask[:, i])
            loss = 0
            total_category_loss = 0
            total_sentiment_loss = 0
            for i in range(self.category_num):
                category_temp_loss = self.category_loss(final_category_outputs[i].squeeze(dim=-1), category_labels[i])
                sentiment_temp_loss = self.sentiment_loss(final_sentiment_outputs[i], polarity_labels[i].long())
                total_category_loss += category_temp_loss
                if not self.configuration['only_acd']:
                    total_sentiment_loss += sentiment_temp_loss

            loss = self.category_loss_weight * total_category_loss + self.sentiment_loss_weight * total_sentiment_loss

            # sentiment accuracy
            sentiment_logit = torch.cat(final_sentiment_outputs)
            sentiment_label = torch.cat(polarity_labels)
            sentiment_mask = torch.cat(polarity_masks)
            self._accuracy(sentiment_logit, sentiment_label, sentiment_mask)

            # category f1
            final_category_outputs_prob = [torch.sigmoid(e) for e in final_category_outputs]
            category_prob = torch.cat(final_category_outputs_prob).squeeze()
            category_label = torch.cat(category_labels)
            self._f1(category_prob, category_label)

            output['loss'] = loss

        # visualize attention
        pred_category = [torch.sigmoid(e) for e in final_category_outputs]
        pred_sentiment = [torch.nn.functional.softmax(e, dim=-1) for e in final_sentiment_outputs]
        output['pred_category'] = pred_category
        output['pred_sentiment'] = pred_sentiment
        if self.configuration['visualize_attention']:
            for i in range(len(sample)):
                words = sample[i][2]
                attention_labels = [e.split('/')[0] for e in self.categories]

                # category
                visual_attentions_category = [embedding_layer_category_alphas[j][i][: len(words)].detach().cpu().numpy()
                                              for j in range(self.category_num)]
                titles = ['true: %s - pred: %s' % (str(label[i][j].detach().cpu().numpy()),
                                                   str(pred_category[j][i].detach().cpu().numpy()))
                          for j in range(self.category_num)]
                savefig_filepath = super()._get_model_visualization_picture_filepath(self.configuration, words)
                attention_visualizer.plot_multi_attentions_of_sentence(words, visual_attentions_category,
                                                                       attention_labels, titles, savefig_filepath)

                # sentiment lstm layer
                visual_attentions_sentiment_temp = [lstm_layer_words_sentiment_soft[j][i][: len(words)].detach().cpu().numpy()
                                                    for j in range(self.category_num)]
                for j in range(self.category_num):
                    c_label = label[i][j].detach().cpu().numpy().tolist()
                    if c_label == 1:
                        visual_attentions_sentiment = []
                        labels_sentiment = []
                        sentiment_true_index = int(label[i][j + self.category_num].detach().cpu().numpy().tolist())
                        if sentiment_true_index == -100:
                            continue
                        titles_sentiment = ['true: %s - pred: %s - %s' % (str(self.polarites[sentiment_true_index]),
                                                        str(pred_sentiment[j][i].detach().cpu().numpy()),
                                                        str(self.polarites))]
                        c_attention = embedding_layer_category_alphas[j][i][: len(words)].detach().cpu().numpy()
                        visual_attentions_sentiment.append(c_attention)
                        labels_sentiment.append(self.categories[j].split('/')[0])

                        s_distributions = visual_attentions_sentiment_temp[j]
                        for k in range(self.polarity_num):
                            labels_sentiment.append(self.polarites[k])
                            visual_attentions_sentiment.append(s_distributions[:, k])
                        titles_sentiment.extend([''] * 3)
                        savefig_filepath = super()._get_model_visualization_picture_filepath(self.configuration, words)
                        attention_visualizer.plot_multi_attentions_of_sentence(words, visual_attentions_sentiment,
                                                                               labels_sentiment,
                                                                               titles_sentiment, savefig_filepath)
        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {
            'accuracy': self._accuracy.get_metric(reset),
            'category_f1': self._f1.get_metric(reset)['fscore']
        }
        return metrics


class ConstituencyBertSingle(TextInAllAspectSentimentOutModel):
    def __init__(self, word_embedder: TextFieldEmbedder, position_embedder: TextFieldEmbedder,
                 categories: list, polarities: list, vocab: Vocabulary, configuration: dict,
                 bert_word_embedder: TextFieldEmbedder=None):
        super().__init__(vocab)
        self.configuration = configuration
        self.word_embedder = word_embedder
        self.position_embedder = position_embedder
        self.categories = categories
        self.polarites = polarities
        self.category_num = len(categories)
        self.polarity_num = len(polarities)
        self.category_loss = nn.BCEWithLogitsLoss()
        self.sentiment_loss = nn.CrossEntropyLoss()
        self._accuracy = metrics.CategoricalAccuracy()
        self._f1 = allennlp_metrics.BinaryF1(0.5)

        word_embedding_dim = word_embedder.get_output_dim()
        if self.configuration['lstm_or_fc_after_embedding_layer'] == 'fc':
            self.embedding_layer_fc = nn.Linear(word_embedding_dim, word_embedding_dim, bias=True)
        else:
            self.embedding_layer_lstm = torch.nn.LSTM(word_embedding_dim, word_embedding_dim, batch_first=True,
                                                      bidirectional=False, num_layers=1)

        self.embedding_layer_aspect_attentions = [AttentionInHtt(word_embedding_dim,
                                                                 word_embedding_dim)
                                                  for _ in range(self.category_num)]
        self.embedding_layer_aspect_attentions = nn.ModuleList(self.embedding_layer_aspect_attentions)

        self.category_fcs = [nn.Linear(word_embedding_dim, 1) for _ in range(self.category_num)]
        self.category_fcs = nn.ModuleList(self.category_fcs)

        if self.configuration['lstm_layer_num_in_bert'] != 0:
            num_layers = self.configuration['lstm_layer_num_in_bert']
            self.lstm = torch.nn.LSTM(768, int(word_embedding_dim / 2), batch_first=True,
                                      bidirectional=True, num_layers=num_layers,
                                      dropout=self.configuration['dropout_in_bert'])
            hidden_size = word_embedding_dim
        else:
            hidden_size = 768
        self.sentiment_fc = nn.Sequential(nn.Linear(hidden_size, self.polarity_num))

        self.sentiment_fc_for_attention_warmup = nn.Sequential(
            # nn.Linear(hidden_size, hidden_size),
            # nn.ReLU(),
            nn.Linear(word_embedding_dim, self.polarity_num))

        self.bert_word_embedder = bert_word_embedder

        self.dropout_after_embedding_layer = nn.Dropout(self.configuration['dropout_in_bert'])

        # 只负责生成针对aspect category的节点表示
        self.gnn_for_aspect = DglGraphAttentionForAspectCategory(word_embedding_dim, word_embedding_dim,
                                                                 self.embedding_layer_aspect_attentions, configuration)

        # self.gnn_for_sentiment = DglGraphConvolution(word_embedding_dim, word_embedding_dim, configuration)
        self.gnn_for_sentiment = DglGraphAverage(word_embedding_dim, word_embedding_dim, self.configuration)
        # self.gnn_for_sentiment = GAT(word_embedding_dim, word_embedding_dim, word_embedding_dim, 4, self.configuration)
        # self.gnn_for_sentiment = DglGraphAttentionForAspectCategory(word_embedding_dim, word_embedding_dim, self.configuration)

    def set_grad_for_acd_parameter(self, requires_grad=True):
        acd_layers = []
        if self.configuration['lstm_or_fc_after_embedding_layer'] == 'fc':
            acd_layers.append(self.embedding_layer_fc)
        else:
            acd_layers.append(self.embedding_layer_lstm)
        acd_layers.append(self.embedding_layer_aspect_attentions)
        acd_layers.append(self.category_fcs)
        acd_layers.append(self.gnn_for_aspect)
        for layer in acd_layers:
            for name, value in layer.named_parameters():
                value.requires_grad = requires_grad

    def set_grad_for_acsc_parameter(self, requires_grad=True):
        acsc_layers = []
        if self.configuration['lstm_layer_num_in_bert'] != 0:
            acsc_layers.append(self.lstm)
        acsc_layers.append(self.sentiment_fc)
        for layer in acsc_layers:
            for name, value in layer.named_parameters():
                value.requires_grad = requires_grad
        bert_model = self.bert_word_embedder._token_embedders['bert'].bert_model
        for param in bert_model.parameters():
            param.requires_grad = requires_grad

    def set_bert_word_embedder(self, bert_word_embedder: TextFieldEmbedder=None):
        self.bert_word_embedder = bert_word_embedder

    def forward(self, tokens: Dict[str, torch.Tensor], tokens_for_graph: Dict[str, torch.Tensor], label: torch.Tensor,
                position: torch.Tensor, polarity_mask: torch.Tensor, sample: list, bert: torch.Tensor) -> torch.Tensor:
        mask_for_graph = get_text_field_mask(tokens_for_graph)

        bert_mask = bert['mask']
        bert_word_embeddings = self.bert_word_embedder(bert)

        mask = get_text_field_mask(tokens)

        max_len_for_graph = mask_for_graph.size()[1]
        graphs = [e['graph'] for e in sample]
        graphs_padded = self.pad_dgl_graph(graphs, max_len_for_graph)
        max_len = mask.size()[1]
        padder = nn.ConstantPad2d((0, 0, 0, max_len_for_graph - max_len), 0)

        word_embeddings = self.word_embedder(tokens)
        word_embeddings_size = word_embeddings.size()
        if self.configuration['lstm_or_fc_after_embedding_layer'] == 'fc':
            word_embeddings_fc = self.embedding_layer_fc(word_embeddings)
        else:
            word_embeddings_fc, (_, _) = self.embedding_layer_lstm(word_embeddings)

        embedding_layer_category_outputs = []
        embedding_layer_category_alphas = []
        embedding_layer_sentiment_outputs = []
        embedding_layer_sentiment_alphas = []

        # 在word_embeddings_fc上做图卷积，attention是基于图卷积的输出
        word_embeddings_fc_for_graph = padder(word_embeddings_fc)
        graph_output_for_aspect, lstm_result_for_graph = self.gnn_for_aspect(word_embeddings_fc_for_graph,
                                                                             word_embeddings_fc_for_graph,
                                                                             graphs_padded)

        bert_clses_of_all_aspect = []
        for i in range(self.category_num):
            embedding_layer_aspect_attention = self.embedding_layer_aspect_attentions[i]
            alpha = embedding_layer_aspect_attention(graph_output_for_aspect, mask_for_graph)
            embedding_layer_category_alphas.append(alpha)

            category_output = self.element_wise_mul(graph_output_for_aspect, alpha, return_not_sum_result=False)
            embedding_layer_category_outputs.append(category_output)

        bert_clses_of_aspect = bert_word_embeddings[:, 0, 0, :]
        bert_clses_of_all_aspect.append(bert_clses_of_aspect)

        if not self.configuration['only_bert']:
            bert_word_embeddings_of_aspect = bert_word_embeddings[:, 0, :, :]
            aspect_word_embeddings_from_bert = []
            for j in range(len(sample)):
                aspect_word_embeddings_from_bert_of_one_sample = []
                all_word_indices_in_bert = sample[j]['word_index_and_bert_indices']
                for k in range(word_embeddings_size[1]):
                    if k in all_word_indices_in_bert:
                        word_indices_in_bert = all_word_indices_in_bert[k]
                        word_bert_embeddings = []
                        for word_index_in_bert in word_indices_in_bert:
                            word_bert_embedding = bert_word_embeddings_of_aspect[j][word_index_in_bert]
                            word_bert_embeddings.append(word_bert_embedding)
                        if len(word_bert_embeddings) == 0:
                            print()
                        if len(word_bert_embeddings) > 1:
                            word_bert_embeddings_unsqueeze = [torch.unsqueeze(e, dim=0) for e in word_bert_embeddings]
                            word_bert_embeddings_cat = torch.cat(word_bert_embeddings_unsqueeze, dim=0)
                            word_bert_embeddings_sum = torch.sum(word_bert_embeddings_cat, dim=0)
                            word_bert_embeddings_ave = word_bert_embeddings_sum / len(word_bert_embeddings)
                        else:
                            word_bert_embeddings_ave = word_bert_embeddings[0]
                        aspect_word_embeddings_from_bert_of_one_sample.append(
                            torch.unsqueeze(word_bert_embeddings_ave, 0))
                    else:
                        zero = torch.zeros_like(aspect_word_embeddings_from_bert_of_one_sample[-1])
                        aspect_word_embeddings_from_bert_of_one_sample.append(zero)
                aspect_word_embeddings_from_bert_of_one_sample_cat = torch.cat(
                    aspect_word_embeddings_from_bert_of_one_sample, dim=0)
                aspect_word_embeddings_from_bert.append(
                    torch.unsqueeze(aspect_word_embeddings_from_bert_of_one_sample_cat, dim=0))
            aspect_word_embeddings_from_bert_cat = torch.cat(aspect_word_embeddings_from_bert, dim=0)
            if self.configuration['lstm_layer_num_in_bert'] != 0:
                aspect_word_embeddings_from_bert_cat, _ = self.lstm(aspect_word_embeddings_from_bert_cat)

            # 在lstm_result上做图卷积
            aspect_word_embeddings_from_bert_cat_padded_for_graph = padder(aspect_word_embeddings_from_bert_cat)
            aspect_word_embeddings_from_bert_cat_result_for_graph = self.gnn_for_sentiment(
                aspect_word_embeddings_from_bert_cat_padded_for_graph, graphs_padded)
            embedding_layer_sentiment_outputs.append(aspect_word_embeddings_from_bert_cat_result_for_graph)

        lstm_layer_category_outputs = []
        lstm_layer_sentiment_outputs = []
        lstm_layer_words_sentiment_soft = []

        for i in range(self.category_num):
            sentiment_output_temp = bert_clses_of_all_aspect[0]
            sentiment_output_cls = self.sentiment_fc(sentiment_output_temp)
            if self.configuration['only_bert']:
                sentiment_output = sentiment_output_cls
                lstm_layer_sentiment_outputs.append(sentiment_output)
            else:
                # sentiment
                aspect_word_embeddings_from_bert = embedding_layer_sentiment_outputs[0]
                word_representation_for_sentiment = self.dropout_after_embedding_layer(aspect_word_embeddings_from_bert)

                sentiment_alpha = embedding_layer_category_alphas[i]
                if self.configuration['mil']:
                    sentiment_alpha = sentiment_alpha.unsqueeze(1)
                    words_sentiment = self.sentiment_fc(word_representation_for_sentiment)
                    if self.configuration['mil_softmax']:
                        words_sentiment_soft = torch.softmax(words_sentiment, dim=-1)
                        lstm_layer_words_sentiment_soft.append(words_sentiment_soft)
                    else:
                        words_sentiment_soft = words_sentiment
                        lstm_layer_words_sentiment_soft.append(torch.softmax(words_sentiment, dim=-1))
                    sentiment_output_mil = torch.matmul(sentiment_alpha, words_sentiment_soft).squeeze(1)  # batch_size x 2*hidden_dim
                    if self.configuration['concat_cls_vector']:
                        if self.configuration['mil_softmax']:
                            sentiment_output_cls_softmax = torch.softmax(sentiment_output_cls, dim=-1)
                            sentiment_output = sentiment_output_mil + sentiment_output_cls_softmax
                        else:
                            sentiment_output = sentiment_output_mil + sentiment_output_cls
                    else:
                        sentiment_output = sentiment_output_mil
                    lstm_layer_sentiment_outputs.append(sentiment_output)
                else:
                    if self.configuration['attention_warmup']:
                        word_representation_for_sentiment = lstm_result_for_graph
                    sentiment_output_temp = self.element_wise_mul(word_representation_for_sentiment, sentiment_alpha,
                                                                  return_not_sum_result=False)
                    if self.configuration['attention_warmup']:
                        sentiment_output_not_mil = self.sentiment_fc_for_attention_warmup(sentiment_output_temp)
                    else:
                        sentiment_output_not_mil = self.sentiment_fc(sentiment_output_temp)
                    if self.configuration['concat_cls_vector']:
                        sentiment_output = sentiment_output_not_mil + sentiment_output_cls
                    else:
                        sentiment_output = sentiment_output_not_mil
                    lstm_layer_sentiment_outputs.append(sentiment_output)

        final_category_outputs = []
        final_category_outputs_auxillary = []
        final_lstm_category_outputs = []
        final_sentiment_outputs = []
        for i in range(self.category_num):
            category_output = embedding_layer_category_outputs[i]
            for j in range(self.category_num):
                final_category_output = self.category_fcs[j](category_output)
                if i == j:
                    final_category_outputs.append(final_category_output)
                else:
                    final_category_outputs_auxillary.append(final_category_output)

            sentiment_output = lstm_layer_sentiment_outputs[i]
            final_sentiment_output = sentiment_output
            final_sentiment_outputs.append(final_sentiment_output)

        output = {}
        if label is not None:
            category_labels = []
            polarity_labels = []
            polarity_masks = []
            for i in range(self.category_num):
                category_labels.append(label[:, i])
                polarity_labels.append(label[:, i + self.category_num])
                polarity_masks.append(polarity_mask[:, i])
            loss = 0
            total_category_loss = 0
            total_sentiment_loss = 0
            for i in range(self.category_num):
                category_temp_loss = self.category_loss(final_category_outputs[i].squeeze(dim=-1), category_labels[i])
                sentiment_temp_loss = self.sentiment_loss(final_sentiment_outputs[i], polarity_labels[i].long())
                total_category_loss += category_temp_loss
                if not self.configuration['only_acd']:
                    total_sentiment_loss += sentiment_temp_loss

            loss = self.category_loss_weight * total_category_loss + self.sentiment_loss_weight * total_sentiment_loss

            interactive_loss = 0
            for i in range(len(final_category_outputs_auxillary)):
                label_temp = torch.zeros_like(category_labels[0])
                interactive_loss_temp = self.category_loss(final_category_outputs_auxillary[i].squeeze(dim=-1),
                                                           label_temp)
                interactive_loss += interactive_loss_temp
            loss += (interactive_loss * self.configuration['interactive_loss_lamda'] / (self.category_num - 1))

            # sentiment accuracy
            sentiment_logit = torch.cat(final_sentiment_outputs)
            sentiment_label = torch.cat(polarity_labels)
            sentiment_mask = torch.cat(polarity_masks)
            self._accuracy(sentiment_logit, sentiment_label, sentiment_mask)

            # category f1
            final_category_outputs_prob = [torch.sigmoid(e) for e in final_category_outputs]
            category_prob = torch.cat(final_category_outputs_prob).squeeze()
            category_label = torch.cat(category_labels)
            self._f1(category_prob, category_label)

            output['loss'] = loss

        # visualize attention
        pred_category = [torch.sigmoid(e) for e in final_category_outputs]
        pred_sentiment = [torch.nn.functional.softmax(e, dim=-1) for e in final_sentiment_outputs]
        output['pred_category'] = pred_category
        output['pred_sentiment'] = pred_sentiment
        if self.configuration['visualize_attention']:
            for i in range(len(sample)):
                words = sample[i]['words_for_graph']
                print('inner_nodes:-------------------------------------------')
                for inner_node_index, inner_node in enumerate(sample[i]['inner_nodes']):
                    print('%d-%s-%s' % (
                        len(sample[i]['words']) + inner_node_index, inner_node.labels[0], inner_node.text))

                attention_labels = [e.split('/')[0] for e in self.categories]

                # 只可视化mams
                label_true = label[i].detach().cpu().numpy()[: self.category_num]
                if sum(label_true) <= 1:
                    continue

                # category
                visual_attentions_category = [embedding_layer_category_alphas[j][i][: len(words)].detach().cpu().numpy()
                                              for j in range(self.category_num)]
                titles = ['true: %s - pred: %s' % (str(label[i][j].detach().cpu().numpy()),
                                                   str(pred_category[j][i].detach().cpu().numpy()))
                          for j in range(self.category_num)]
                savefig_filepath = super()._get_model_visualization_picture_filepath(self.configuration, words)
                attention_visualizer.plot_multi_attentions_of_sentence(words, visual_attentions_category,
                                                                       attention_labels, titles, savefig_filepath)

                if self.configuration['mil']:
                    # sentiment lstm layer
                    visual_attentions_sentiment_temp = [lstm_layer_words_sentiment_soft[j][i][: len(words)].detach().cpu().numpy()
                                                        for j in range(self.category_num)]
                    for j in range(self.category_num):
                        c_label = label[i][j].detach().cpu().numpy().tolist()
                        if c_label == 1:
                            visual_attentions_sentiment = []
                            labels_sentiment = []
                            sentiment_true_index = int(label[i][j + self.category_num].detach().cpu().numpy().tolist())
                            if sentiment_true_index == -100:
                                continue
                            titles_sentiment = ['true: %s - pred: %s - %s' % (str(self.polarites[sentiment_true_index]),
                                                            str(pred_sentiment[j][i].detach().cpu().numpy()),
                                                            str(self.polarites))]
                            c_attention = embedding_layer_category_alphas[j][i][: len(words)].detach().cpu().numpy()
                            visual_attentions_sentiment.append(c_attention)
                            labels_sentiment.append(self.categories[j].split('/')[0])

                            s_distributions = visual_attentions_sentiment_temp[j]
                            for k in range(self.polarity_num):
                                labels_sentiment.append(self.polarites[k])
                                visual_attentions_sentiment.append(s_distributions[:, k])
                            titles_sentiment.extend([''] * 3)
                            savefig_filepath = super()._get_model_visualization_picture_filepath(self.configuration, words)
                            attention_visualizer.plot_multi_attentions_of_sentence(words, visual_attentions_sentiment,
                                                                                   labels_sentiment,
                                                                                   titles_sentiment, savefig_filepath)
        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {
            'accuracy': self._accuracy.get_metric(reset),
            'category_f1': self._f1.get_metric(reset)['fscore']
        }
        return metrics


class Estimator:

    def estimate(self, ds: Iterable[Instance]) -> dict:
        raise NotImplementedError('estimate')


class TextInAllAspectSentimentOutEstimator(Estimator):
    def __init__(self, model: Model, iterator: DataIterator, categories: list, polarities: list,
                 cuda_device: int = -1, configuration: dict=None) -> None:
        super().__init__()
        self.model = model
        self.iterator = iterator
        self.categories = categories
        self.polarities = polarities
        self._sentiment_accuracy = metrics.CategoricalAccuracy()
        self._sentiment_accuracy_temp = metrics.CategoricalAccuracy()
        self._aspect_f1 = allennlp_metrics.BinaryF1(0.5)
        self._aspect_f1_temp = allennlp_metrics.BinaryF1(0.5)
        self.cuda_device = cuda_device
        self.configuration = configuration
        self.other_metrics = {}
        self.debug = False

    def _get_other_metrics(self, reset=True):
        result = self.other_metrics
        if reset:
            self.other_metrics = {}
        return result

    def _print_tensor(self, tensors: List):
        print('------------------------------------------------------')
        list_list = [e.detach().cpu().numpy().tolist() if not isinstance(e, np.ndarray) else e.tolist() for e in tensors]
        for k in range(len(list_list[0])):
            format_str = '-'.join(['%s'] * len(list_list))
            values = tuple(e[k] for e in list_list)
            print(format_str % values)

    def _acd_aspect_and_metrics(self, category_labels, aspect_pred):
        acd_aspect_and_metrics = {}
        for i, aspect in enumerate(self.categories):
            aspect_label_i = category_labels[i].detach().cpu().numpy().astype(int)
            aspect_pred_i = (aspect_pred[i].squeeze(dim=-1).detach().cpu().numpy() > 0.5).astype(int)
            if self.debug:
                self._print_tensor([aspect_pred_i, aspect_label_i])
            aspect_f1 = f1_score(aspect_label_i, aspect_pred_i, average='binary')
            aspect_precision = precision_score(aspect_label_i, aspect_pred_i, average='binary')
            aspect_recall = recall_score(aspect_label_i, aspect_pred_i, average='binary')
            acd_aspect_and_metrics[aspect] = {
                'f1': aspect_f1,
                'precision': aspect_precision,
                'recall': aspect_recall
            }
        return acd_aspect_and_metrics

    def _acsc_aspect_and_metrics(self, polarity_labels, sentiment_pred, polarity_masks):
        acsc_aspect_and_metrics = {}
        for i, aspect in enumerate(self.categories):
            aspect_sentiment_label_i = polarity_labels[i]
            aspect_sentiment_pred_i = sentiment_pred[i]
            aspect_sentiment_mask_i = polarity_masks[i]
            self._sentiment_accuracy_temp(aspect_sentiment_pred_i, aspect_sentiment_label_i,
                                          aspect_sentiment_mask_i)
            aspect_acc_temp = self._sentiment_accuracy_temp.get_metric(reset=True),
            acsc_aspect_and_metrics[aspect] = {
                'acc': aspect_acc_temp[0],
            }
        return acsc_aspect_and_metrics

    def _polarity_metrics(self, sentiment_logit, sentiment_label, sentiment_mask):
        sentiment_label_pred_list = sentiment_logit.argmax(dim=-1).detach().cpu().numpy().tolist()
        sentiment_label_list = sentiment_label.detach().cpu().numpy().tolist()
        sentiment_mask_list = sentiment_mask.detach().cpu().numpy().tolist()
        sentiment_label_pred_final = []
        sentiment_label_final = []
        for i in range(len(sentiment_mask_list)):
            sentiment_label_list_i = sentiment_label_list[i]
            sentiment_label_pred_list_i = sentiment_label_pred_list[i]
            sentiment_mask_list_i = sentiment_mask_list[i]
            if sentiment_mask_list_i == 0:
                continue
            sentiment_label_pred_final.append(sentiment_label_pred_list_i)
            sentiment_label_final.append(sentiment_label_list_i)
        sentiment_f1s = f1_score(np.array(sentiment_label_final), np.array(sentiment_label_pred_final), average=None,
                                 labels=list(range(len(self.polarities))))
        sentiment_precisions = precision_score(np.array(sentiment_label_final), np.array(sentiment_label_pred_final),
                                               average=None, labels=list(range(len(self.polarities))))
        sentiment_recalls = recall_score(np.array(sentiment_label_final), np.array(sentiment_label_pred_final),
                                         average=None, labels=list(range(len(self.polarities))))
        polarity_metrics = {}
        for i, polarity in enumerate(self.polarities):
            polarity_metrics[polarity] = {
                'f1': sentiment_f1s[i],
                'precision': sentiment_precisions[i],
                'recall': sentiment_recalls[i]
            }
        return polarity_metrics

    def _merge_micro_f1(self, merge_label_real, merge_logit_real):
        tp = 0
        pred_total = 0
        true_total = 0
        for i in range(merge_logit_real.shape[0]):
            pred = merge_logit_real[i]
            true = merge_label_real[i]
            if pred != 0:
                pred_total += 1
            if true != 0:
                true_total += 1
            if pred == true != 0:
                tp += 1
        if pred_total == 0:
            pred_total = 0.0000000000000001
        if true_total == 0:
            true_total = 0.0000000000000001
        p = tp / pred_total
        r = tp / true_total
        if p == 0 and r == 0:
            f1 = 0
        else:
            f1 = 2 * (p * r) / (p + r)
        return f1

    def _inner_estimate(self, label, polarity_mask, aspect_pred, sentiment_pred, merge_pred):
        category_labels = []
        polarity_labels = []
        merge_labeles = []
        polarity_masks = []
        category_num = len(self.categories)
        for i in range(category_num):
            category_labels.append(label[:, i])
            polarity_labels.append(label[:, i + category_num])
            polarity_masks.append(polarity_mask[:, i])
            merge_labeles.append(label[:, i + category_num * 2])
        if self.debug:
            self._print_tensor([label] + category_labels + polarity_labels + merge_labeles)
            self._print_tensor([polarity_mask] + polarity_masks)
        acd_aspect_and_metrics = self._acd_aspect_and_metrics(category_labels, aspect_pred)
        self.other_metrics['acd_metrics'] = acd_aspect_and_metrics
        # category f1
        category_prob = torch.cat(aspect_pred).squeeze()
        category_label = torch.cat(category_labels)
        self._aspect_f1(category_prob, category_label)

        if not self.configuration['only_acd']:
            acsc_aspect_and_metrics = self._acsc_aspect_and_metrics(polarity_labels, sentiment_pred,
                                                                    polarity_masks)
            self.other_metrics['acsc_metrics'] = acsc_aspect_and_metrics

            # sentiment accuracy
            sentiment_logit = torch.cat(sentiment_pred)
            sentiment_label = torch.cat(polarity_labels)
            sentiment_mask = torch.cat(polarity_masks)
            self._sentiment_accuracy(sentiment_logit, sentiment_label, sentiment_mask)

            polarity_metrics = self._polarity_metrics(sentiment_logit, sentiment_label,
                                                      sentiment_mask)
            self.other_metrics['polarity_metrics'] = polarity_metrics

            # merge
            merge_logit = torch.cat(merge_pred)
            merge_pred_aspect_indicator = (merge_logit.argmax(dim=-1) != 0)
            merge_pred_aspect_indicator = nn_util.move_to_device(merge_pred_aspect_indicator, self.cuda_device)

            merge_label = torch.cat(merge_labeles)
            merge_label_aspect_indicator = (merge_label != 0)
            merge_label_aspect_indicator = nn_util.move_to_device(merge_label_aspect_indicator, self.cuda_device)

            merge_aspect_indicator = merge_pred_aspect_indicator | merge_label_aspect_indicator
            if self.debug:
                self._print_tensor([merge_logit, merge_pred_aspect_indicator, merge_label, merge_label_aspect_indicator, merge_aspect_indicator])

            merge_logit_real = merge_logit[merge_aspect_indicator].argmax(dim=-1).detach().cpu().numpy()
            merge_label_real = merge_label[merge_aspect_indicator].detach().cpu().numpy()
            if self.debug:
                self._print_tensor([merge_logit_real, merge_label_real])
            # merge_micro_f1 = f1_score(merge_label_real, merge_logit_real, average='micro')
            merge_micro_f1 = self._merge_micro_f1(merge_label_real, merge_logit_real)
            self.other_metrics['merge_micro_f1'] = merge_micro_f1

    def estimate(self, ds: Iterable[Instance]) -> dict:
        self.model.eval()
        pred_generator = self.iterator(ds, num_epochs=1, shuffle=False)
        pred_generator_tqdm = tqdm(pred_generator, total=self.iterator.get_num_batches(ds))
        with torch.no_grad():
            labels = []
            polarity_masks = []
            pred_categorys = []
            pred_sentiments = []
            pred_merges = []
            for batch in pred_generator_tqdm:
                label = batch['label']
                labels.append(label)

                polarity_mask = batch['polarity_mask']
                polarity_masks.append(polarity_mask)

                batch = nn_util.move_to_device(batch, self.cuda_device)
                out_dict = self.model(**batch)
                pred_category = out_dict['pred_category']
                pred_categorys.append(pred_category)
                if not self.configuration['only_acd']:
                    pred_sentiment = out_dict['pred_sentiment']
                    pred_sentiments.append(pred_sentiment)

                    if 'merge_pred' in out_dict:
                        pred_merge = out_dict['merge_pred']
                    else:
                        pred_merge = []
                        for i in range(len(self.categories)):
                            pred_category_i = pred_category[i].detach().clone().squeeze(-1)
                            pred_sentiment_i = torch.softmax(pred_sentiment[i], dim=-1)
                            aspect_threshold = 0.5 if 'aspect_threshold' not in self.configuration else self.configuration['aspect_threshold']
                            pred_category_i_indicator = pred_category_i > aspect_threshold
                            pred_category_i_indicator_not = pred_category_i <= aspect_threshold
                            if self.debug:
                                print(i)
                                self._print_tensor([pred_category_i, pred_sentiment_i, pred_category_i_indicator, pred_category_i_indicator_not])
                            pred_category_i[pred_category_i_indicator] = 0
                            pred_category_i[pred_category_i_indicator_not] = 1.1
                            pred_category_i = pred_category_i.unsqueeze(-1)
                            if self.debug:
                                self._print_tensor([pred_category[i], pred_category_i, torch.cat([pred_category_i, pred_sentiment_i], dim=-1)])
                            pred_merge.append(torch.cat([pred_category_i, pred_sentiment_i], dim=-1))
                    pred_merges.append(pred_merge)
            label_final = torch.cat(labels, dim=0)
            polarity_mask_final = torch.cat(polarity_masks, dim=0)
            pred_category_final = []
            pred_sentiment_final = []
            pred_merge_final = []
            for i in range(len(self.categories)):
                pred_category_i = [e[i] for e in pred_categorys]
                pred_category_i_cat = torch.cat(pred_category_i, dim=0)
                pred_category_final.append(pred_category_i_cat)
                if not self.configuration['only_acd']:
                    pred_sentiment_i = [e[i] for e in pred_sentiments]
                    pred_sentiment_i_cat = torch.cat(pred_sentiment_i, dim=0)
                    pred_sentiment_final.append(pred_sentiment_i_cat)

                    pred_merge_i = [e[i] for e in pred_merges]
                    pred_merge_i_cat = torch.cat(pred_merge_i, dim=0)
                    pred_merge_final.append(pred_merge_i_cat)

            # self._estimate(label_final, polarity_mask_final, pred_category_final, pred_sentiment_final)
            self._inner_estimate(label_final, polarity_mask_final, pred_category_final, pred_sentiment_final,
                                 pred_merge_final)
        return {'sentiment_acc': self._sentiment_accuracy.get_metric(reset=True),
                'category_f1': self._aspect_f1.get_metric(reset=True),
                'other_metrics': self._get_other_metrics()}


class Predictor:

    def predict(self, ds: Iterable[Instance]) -> dict:
        raise NotImplementedError('predict')


class TextInAllAspectSentimentOutPredictor(Predictor):
    def __init__(self, model: Model, iterator: DataIterator, categories: list, polarities: list,
                 cuda_device: int = -1, configuration: dict=None) -> None:
        super().__init__()
        self.model = model
        self.iterator = iterator
        self.categories = categories
        self.polarities = polarities
        self.cuda_device = cuda_device
        self.configuration = configuration
        self.debug = False

    def _print_tensor(self, tensors: List):
        print('------------------------------------------------------')
        list_list = [e.detach().cpu().numpy().tolist() if not isinstance(e, np.ndarray) else e.tolist() for e in tensors]
        for k in range(len(list_list[0])):
            format_str = '-'.join(['%s'] * len(list_list))
            values = tuple(e[k] for e in list_list)
            print(format_str % values)

    def predict(self, ds: Iterable[Instance]) -> dict:
        with torch.no_grad():
            self.model.eval()
            pred_generator = self.iterator(ds, num_epochs=1, shuffle=False)
            pred_generator_tqdm = tqdm(pred_generator, total=self.iterator.get_num_batches(ds))
            labels = []
            polarity_masks = []
            pred_categorys = []
            pred_sentiments = []
            pred_merges = []
            for batch in pred_generator_tqdm:
                label = batch['label']
                labels.append(label)

                polarity_mask = batch['polarity_mask']
                polarity_masks.append(polarity_mask)

                batch = nn_util.move_to_device(batch, self.cuda_device)
                out_dict = self.model(**batch)
                pred_category = out_dict['pred_category']
                pred_categorys.append(pred_category)
                if not self.configuration['only_acd']:
                    pred_sentiment = out_dict['pred_sentiment']
                    pred_sentiments.append(pred_sentiment)

                    if 'merge_pred' in out_dict:
                        pred_merge = out_dict['merge_pred']
                    else:
                        pred_merge = []
                        for i in range(len(self.categories)):
                            pred_category_i = pred_category[i].detach().clone().squeeze(-1)
                            pred_sentiment_i = torch.softmax(pred_sentiment[i], dim=-1)
                            aspect_threshold = 0.5 if 'aspect_threshold' not in self.configuration else self.configuration['aspect_threshold']
                            pred_category_i_indicator = pred_category_i > aspect_threshold
                            pred_category_i_indicator_not = pred_category_i <= aspect_threshold
                            if self.debug:
                                print(i)
                                self._print_tensor([pred_category_i, pred_sentiment_i, pred_category_i_indicator, pred_category_i_indicator_not])
                            pred_category_i[pred_category_i_indicator] = 0
                            pred_category_i[pred_category_i_indicator_not] = 1.1
                            pred_category_i = pred_category_i.unsqueeze(-1)
                            if self.debug:
                                self._print_tensor([pred_category[i], pred_category_i, torch.cat([pred_category_i, pred_sentiment_i], dim=-1)])
                            pred_merge.append(torch.cat([pred_category_i, pred_sentiment_i], dim=-1))
                    pred_merges.append(pred_merge)
            label_final = torch.cat(labels, dim=0)
            polarity_mask_final = torch.cat(polarity_masks, dim=0)
            pred_category_final = []
            pred_sentiment_final = []
            pred_merge_final = []
            for i in range(len(self.categories)):
                pred_category_i = [e[i] for e in pred_categorys]
                pred_category_i_cat = torch.cat(pred_category_i, dim=0)
                pred_category_final.append(pred_category_i_cat)
                if not self.configuration['only_acd']:
                    pred_sentiment_i = [e[i] for e in pred_sentiments]
                    pred_sentiment_i_cat = torch.cat(pred_sentiment_i, dim=0)
                    pred_sentiment_final.append(pred_sentiment_i_cat)

                    pred_merge_i = [e[i] for e in pred_merges]
                    pred_merge_i_cat = torch.cat(pred_merge_i, dim=0)
                    pred_merge_final.append(pred_merge_i_cat)
            result = []
            for i in range(len(ds)):
                sample_label = label_final[i][len(self.categories): len(self.categories) + len(self.categories)]
                sample_predict = [pred_sentiment_final[j][i] for j in range(len(self.categories))]
                sample_result = []
                for j in range(len(self.categories)):
                    if sample_label[j] == -100:
                        continue
                    category = self.categories[j]
                    sentiment_index = sample_predict[j].argmax(dim=-1)
                    sentiment = self.polarities[sentiment_index]
                    sample_result.append((category, sentiment))
                result.append(sample_result)
        return result


class TextInAllAspectSentimentOutPredictorOnInstanceLevel(Predictor):
    def __init__(self, model: Model, iterator: DataIterator, categories: list, polarities: list,
                 cuda_device: int = -1, configuration: dict=None) -> None:
        super().__init__()
        self.model = model
        self.iterator = iterator
        self.categories = categories
        self.polarities = polarities
        self.cuda_device = cuda_device
        self.configuration = configuration
        self.debug = False

    def _print_tensor(self, tensors: List):
        print('------------------------------------------------------')
        list_list = [e.detach().cpu().numpy().tolist() if not isinstance(e, np.ndarray) else e.tolist() for e in tensors]
        for k in range(len(list_list[0])):
            format_str = '-'.join(['%s'] * len(list_list))
            values = tuple(e[k] for e in list_list)
            print(format_str % values)

    def predict(self, ds: Iterable[Instance]) -> dict:
        result = []
        with torch.no_grad():
            self.model.eval()
            pred_generator = self.iterator(ds, num_epochs=1, shuffle=False)
            pred_generator_tqdm = tqdm(pred_generator, total=self.iterator.get_num_batches(ds))
            for batch in pred_generator_tqdm:
                batch = nn_util.move_to_device(batch, self.cuda_device)
                out_dict = self.model(**batch)
                attention_weights = out_dict['embedding_layer_category_alphas']
                word_sentiments = out_dict['lstm_layer_words_sentiment_soft']
                for i in range(word_sentiments[0].shape[0]):
                    attention_weights_of_one_sample = [e[i].detach().cpu().numpy() for e in attention_weights]
                    word_sentiments_of_one_sample = [e[i].detach().cpu().numpy() for e in word_sentiments]
                    result.append({'attention_weights': attention_weights_of_one_sample,
                                   'word_sentiments': word_sentiments_of_one_sample})
        return result


class TextInAllAspectOutEstimator(Estimator):
    def __init__(self, model: Model, iterator: DataIterator, categories: list, polarities: list,
                 cuda_device: int = -1, configuration: dict=None) -> None:
        super().__init__()
        self.model = model
        self.iterator = iterator
        self.categories = categories
        self.polarities = polarities
        self._f1 = allennlp_metrics.BinaryF1(0.5)
        self.cuda_device = cuda_device
        self.configuration = configuration

    def _estimate(self, label, pred_category) -> np.ndarray:
        category_labels = []
        for i in range(len(self.categories)):
            category_labels.append(label[:, i])

        # category f1
        category_prob = torch.cat(pred_category).squeeze()
        category_label = torch.cat(category_labels)
        self._f1(category_prob, category_label)

    def estimate(self, ds: Iterable[Instance]) -> dict:
        self.model.eval()
        pred_generator = self.iterator(ds, num_epochs=1, shuffle=False)
        pred_generator_tqdm = tqdm(pred_generator,
                                   total=self.iterator.get_num_batches(ds))
        with torch.no_grad():
            labels = []
            pred_categorys = []
            for batch in pred_generator_tqdm:
                label = batch['label']
                labels.append(label)

                batch = nn_util.move_to_device(batch, self.cuda_device)
                out_dict = self.model(**batch)
                pred_category = out_dict['pred_category']
                pred_categorys.append(pred_category)
            label_final = torch.cat(labels, dim=0)
            pred_category_final = []
            for i in range(len(self.categories)):
                pred_category_i = [e[i] for e in pred_categorys]
                pred_category_i_cat = torch.cat(pred_category_i, dim=0)
                pred_category_final.append(pred_category_i_cat)
            self._estimate(label_final, pred_category_final)
        return {'category_f1': self._f1.get_metric(reset=True)}


class TextInAllAspectSentimentOutEstimatorAll(Estimator):
    def __init__(self, model: Model, iterator: DataIterator, categories: list, polarities: list,
                 cuda_device: int = -1, configuration: dict=None) -> None:
        super().__init__()
        self.model = model
        self.iterator = iterator
        self.categories = categories
        self.polarities = polarities
        self._accuracy = metrics.CategoricalAccuracy()
        self._f1 = allennlp_metrics.BinaryF1(0.5)
        self.cuda_device = cuda_device
        self.configuration = configuration

    def _estimate(self, batch) -> np.ndarray:
        label = batch['label']
        polarity_mask = batch['polarity_mask']
        category_labels = []
        polarity_labels = []
        polarity_masks = []
        for i in range(len(self.categories)):
            category_labels.append(label[:, i])
            polarity_labels.append(label[:, i + len(self.categories)])
            polarity_masks.append(polarity_mask[:, i])

        out_dict = self.model(**batch)
        pred_category = out_dict['pred_category']

        if not self.configuration['only_acd']:
            pred_sentiment = out_dict['pred_sentiment']

            sentiment_logit = torch.cat(pred_sentiment)
            sentiment_label = torch.cat(polarity_labels)
            sentiment_mask = torch.cat(polarity_masks)
            self._accuracy(sentiment_logit, sentiment_label, sentiment_mask)

        # category f1
        category_prob = torch.cat(pred_category).squeeze()
        category_label = torch.cat(category_labels)
        self._f1(category_prob, category_label)

    def estimate(self, ds: Iterable[Instance]) -> dict:
        self.model.eval()
        pred_generator = self.iterator(ds, num_epochs=1, shuffle=False)
        pred_generator_tqdm = tqdm(pred_generator,
                                   total=self.iterator.get_num_batches(ds))
        with torch.no_grad():
            for batch in pred_generator_tqdm:
                batch = nn_util.move_to_device(batch, self.cuda_device)
                self._estimate(batch)
        return {'sentiment_acc': self._accuracy.get_metric(reset=True),
                'category_f1': self._f1.get_metric(reset=True)}


class CaeEstimator:
    def __init__(self, model: Model, iterator: DataIterator, categories: list, polarities: list,
                 cuda_device: int = -1) -> None:
        self.model = model
        self.iterator = iterator
        self.categories = categories
        self.polarities = polarities
        self._accuracy = metrics.CategoricalAccuracy()
        self._f1 = allennlp_metrics.BinaryF1(0.5)
        self.cuda_device = cuda_device

    def _estimate(self, batch) -> np.ndarray:
        label = batch['label']
        polarity_mask = batch['polarity_mask']
        category_labels = []
        polarity_labels = []
        polarity_masks = []
        for i in range(len(self.categories)):
            category_labels.append(label[:, i])
            polarity_labels.append(label[:, i + len(self.categories)])
            polarity_masks.append(polarity_mask[:, i])

        out_dict = self.model(**batch)
        pred_category = out_dict['pred_category']
        pred_sentiment = out_dict['pred_sentiment']

        sentiment_logit = torch.cat(pred_sentiment)
        sentiment_label = torch.cat(polarity_labels)
        sentiment_mask = torch.cat(polarity_masks)
        self._accuracy(sentiment_logit, sentiment_label, sentiment_mask)

        # category f1
        category_prob = torch.cat(pred_category).squeeze()
        category_label = torch.cat(category_labels)
        self._f1(category_prob, category_label)

    def estimate(self, ds: Iterable[Instance]) -> np.ndarray:
        self.model.eval()
        pred_generator = self.iterator(ds, num_epochs=1, shuffle=False)
        pred_generator_tqdm = tqdm(pred_generator,
                                   total=self.iterator.get_num_batches(ds))
        with torch.no_grad():
            for batch in pred_generator_tqdm:
                batch = nn_util.move_to_device(batch, self.cuda_device)
                self._estimate(batch)
        return {'sentiment_acc': self._accuracy.get_metric(),
                'category_f1': self._f1.get_metric()}


class Heat(TextAspectInSentimentOutModel):
    def __init__(self, word_embedder: TextFieldEmbedder, aspect_embedder: TextFieldEmbedder,
                 categories: list, polarities: list, vocab: Vocabulary, configuration: dict):
        super().__init__(vocab)
        self.configuration = configuration
        self.word_embedder = word_embedder
        self.aspect_embedder = aspect_embedder
        self.categories = categories
        self.polarites = polarities
        self.category_num = len(categories)
        self.polarity_num = len(polarities)
        self.category_loss = nn.BCEWithLogitsLoss()
        self.sentiment_loss = nn.CrossEntropyLoss()
        self._accuracy = metrics.CategoricalAccuracy()

        word_embedding_dim = word_embedder.get_output_dim()
        lstm_input_size = word_embedding_dim
        num_layers = 1
        hidden_size = 32
        self.aspect_gru = torch.nn.GRU(lstm_input_size, hidden_size, batch_first=True,
                                       bidirectional=True, num_layers=num_layers)
        self.sentiment_gru = torch.nn.GRU(lstm_input_size, hidden_size, batch_first=True,
                                          bidirectional=True, num_layers=num_layers)
        self.aspect_attention = AttentionInHtt(hidden_size * 3, hidden_size)
        self.sentiment_attention = AttentionInHtt(hidden_size * 5, hidden_size, softmax=False)

        self.sentiment_fc = nn.Sequential(nn.Linear(hidden_size * 3, self.polarity_num))

    def forward(self, tokens: Dict[str, torch.Tensor], aspect: torch.Tensor, label: torch.Tensor,
                sample: list) -> torch.Tensor:
        mask = get_text_field_mask(tokens)
        word_embeddings = self.word_embedder(tokens)
        aspect_gru_output, _ = self.aspect_gru(word_embeddings)
        sentiment_gru_output, _ = self.sentiment_gru(word_embeddings)

        aspect_embeddings_single = self.aspect_embedder(aspect).squeeze(1)
        aspect_repeat = {'aspect': aspect['aspect'].expand_as(tokens['tokens'])}
        aspect_embeddings = self.aspect_embedder(aspect_repeat)

        input_for_aspect_attention = torch.cat([aspect_embeddings, aspect_gru_output], dim=-1)

        aspect_alpha = self.aspect_attention(input_for_aspect_attention, mask)
        category_output = self.element_wise_mul(aspect_gru_output, aspect_alpha, return_not_sum_result=False)

        category_output_unsqueeze = category_output.unsqueeze(1)
        category_output_repeat = category_output_unsqueeze.repeat(1, sentiment_gru_output.size()[1], 1)
        input_for_sentiment_attention = torch.cat([aspect_embeddings, category_output_repeat, sentiment_gru_output],
                                                  dim=-1)
        similarities = self.sentiment_attention(input_for_sentiment_attention, mask)

        location_mask_layer = LocationMaskLayer(aspect_alpha.size(1), self.configuration)
        location_mask = location_mask_layer(aspect_alpha)

        similarities_with_location = similarities * location_mask
        sentiment_alpha = allennlp_util.masked_softmax(similarities_with_location, mask)

        sentiment_output = self.element_wise_mul(sentiment_gru_output, sentiment_alpha, return_not_sum_result=False)

        sentiment_output_with_aspect_embeddings = torch.cat([aspect_embeddings_single, sentiment_output],
                                                            dim=-1)
        final_sentiment_output = self.sentiment_fc(sentiment_output_with_aspect_embeddings)
        final_sentiment_output_prob = torch.softmax(final_sentiment_output, dim=-1)
        output = {'final_sentiment_output_prob': final_sentiment_output_prob}
        if label is not None:
            sentiment_loss = self.sentiment_loss(final_sentiment_output, label)
            self._accuracy(final_sentiment_output, label)
            output['loss'] = sentiment_loss

        # visualize attention
        if self.configuration['visualize_attention']:
            for i in range(len(sample)):
                words = sample[i][2]
                attention_labels = [self.categories[sample[i][1][0]]] * 2

                visual_attentions = [aspect_alpha[i][: len(words)].detach().cpu().numpy()]
                visual_attentions.extend([sentiment_alpha[i][: len(words)].detach().cpu().numpy()])
                titles = ['aspect-true: %s - pred: %s - %s' % (str(label[i].detach().cpu().numpy()),
                                                           str(final_sentiment_output_prob[i].detach().cpu().numpy()),
                                                           str(self.polarites)
                                                           )
                          ]
                titles.append('sentiment')
                attention_visualizer.plot_multi_attentions_of_sentence(words, visual_attentions, attention_labels,
                                                                       titles)
        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {
            'accuracy': self._accuracy.get_metric(reset),
        }
        return metrics


class HeatCae(Model):
    def __init__(self, word_embedder: TextFieldEmbedder, aspect_embedder: TextFieldEmbedder,
                 categories: list, polarities: list, vocab: Vocabulary, configuration: dict):
        super().__init__(vocab)
        self.configuration = configuration
        self.word_embedder = word_embedder
        self.aspect_embedder = aspect_embedder
        self.categories = categories
        self.polarites = polarities
        self.category_num = len(categories)
        self.polarity_num = len(polarities)
        self.category_loss = nn.BCEWithLogitsLoss()
        self.sentiment_loss = nn.CrossEntropyLoss()
        self._accuracy = metrics.CategoricalAccuracy()
        self._f1 = allennlp_metrics.BinaryF1(0.5)

        word_embedding_dim = word_embedder.get_output_dim()
        lstm_input_size = word_embedding_dim
        num_layers = 1
        hidden_size = 32
        self.aspect_gru = torch.nn.GRU(lstm_input_size, hidden_size, batch_first=True,
                                       bidirectional=True, num_layers=num_layers)
        # self.aspect_fc = nn.Linear(lstm_input_size, hidden_size * 2)
        self.sentiment_gru = torch.nn.GRU(lstm_input_size, hidden_size, batch_first=True,
                                          bidirectional=True, num_layers=num_layers)
        self.aspect_attentions = [AttentionInHtt(hidden_size * 3, hidden_size) for _ in range(self.category_num)]
        self.sentiment_attention = AttentionInHtt(hidden_size * 5, hidden_size, softmax=False)

        self.sentiment_fc = nn.Sequential(nn.Linear(hidden_size * 3, self.polarity_num))
        self.category_fcs = [nn.Linear(hidden_size * 2, 1) for _ in range(self.category_num)]

    def matrix_mul(self, input, weight, bias=False):
        feature_list = []
        for feature in input:
            feature = torch.mm(feature, weight)
            if isinstance(bias, torch.nn.parameter.Parameter):
                feature = feature + bias.expand(feature.size()[0], bias.size()[1])
            feature = torch.tanh(feature).unsqueeze(0)
            feature_list.append(feature)

        return torch.cat(feature_list, 0).squeeze()

    def element_wise_mul(self, input1, input2, return_not_sum_result=False):
        feature_list = []
        for feature_1, feature_2 in zip(input1, input2):
            feature_2 = feature_2.unsqueeze(1)
            feature_2 = feature_2.expand_as(feature_1)
            feature = feature_1 * feature_2
            feature = feature.unsqueeze(0)
            feature_list.append(feature)
        output = torch.cat(feature_list, 0)

        result = torch.sum(output, 1)
        if return_not_sum_result:
            return result, output
        else:
            return result

    def reduce(self, nodes):
        """Take an average over all neighbor node features hu and use it to
        overwrite the original node feature."""
        m = nodes.mailbox['m']
        accum = torch.sum(m, 1)
        return {'h': accum}

    def pad_dgl_graph(self, graphs, max_node_num):
        graphs_padded = []
        for graph in graphs:
            graph_padded = copy.deepcopy(graph)
            node_num = graph.number_of_nodes()
            graph_padded.add_nodes(max_node_num - node_num)
            graphs_padded.append(graph_padded)
        return graphs_padded

    def forward(self, tokens: Dict[str, torch.Tensor], aspects: torch.Tensor, sample: list,
                aspect_label: torch.Tensor = None, label: torch.Tensor=None) -> torch.Tensor:
        mask = get_text_field_mask(tokens)
        word_embeddings = self.word_embedder(tokens)
        aspect_gru_output, _ = self.aspect_gru(word_embeddings)
        # aspect_gru_output = self.aspect_fc(word_embeddings)
        sentiment_gru_output, _ = self.sentiment_gru(word_embeddings)

        aspects_separate = [{'aspect': aspects['aspect'][:, i].unsqueeze(1)} for i in range(self.category_num)]
        aspect_embeddings_singles = [self.aspect_embedder(aspects_separate[i]).squeeze(1) for i in
                                     range(self.category_num)]
        aspects_seprate_repeat = [{'aspect': aspects_separate[i]['aspect'].expand_as(tokens['tokens'])} for i in
                                  range(self.category_num)]
        aspect_embeddings_separate = [self.aspect_embedder(aspects_seprate_repeat[i]) for i in range(self.category_num)]

        aspect_alphas = []
        category_outputs = []
        for i in range(self.category_num):
            aspect_embeddings = aspect_embeddings_separate[i]
            input_for_aspect_attention = torch.cat([aspect_embeddings, aspect_gru_output], dim=-1)
            aspect_alpha_temp = self.aspect_attentions[i](input_for_aspect_attention, mask)
            aspect_alphas.append(aspect_alpha_temp)
            category_output_temp = self.element_wise_mul(aspect_gru_output, aspect_alpha_temp,
                                                         return_not_sum_result=False)
            category_outputs.append(category_output_temp)
        category_output = []
        aspect_alpha = []
        aspect_embeddings_single = []
        for i in range(len(sample)):
            aspect_index = sample[i]['aspect_index']
            category_output.append(category_outputs[aspect_index][i].unsqueeze(0))
            aspect_alpha.append(aspect_alphas[aspect_index][i].unsqueeze(0))
            aspect_embeddings_single.append(aspect_embeddings_singles[aspect_index][i].unsqueeze(0))
        category_output = torch.cat(category_output, dim=0)
        aspect_alpha = torch.cat(aspect_alpha, dim=0)
        aspect_embeddings_single = torch.cat(aspect_embeddings_single, dim=0)
        category_output_unsqueeze = category_output.unsqueeze(1)
        category_output_repeat = category_output_unsqueeze.repeat(1, sentiment_gru_output.size()[1], 1)
        input_for_sentiment_attention = torch.cat([aspect_embeddings, category_output_repeat, sentiment_gru_output],
                                                  dim=-1)
        similarities = self.sentiment_attention(input_for_sentiment_attention, mask)

        location_mask_layer = LocationMaskLayer(aspect_alpha.size(1), self.configuration)
        location_mask = location_mask_layer(aspect_alpha)

        similarities_with_location = similarities * location_mask
        sentiment_alpha = allennlp_util.masked_softmax(similarities_with_location, mask)

        sentiment_output = self.element_wise_mul(sentiment_gru_output, sentiment_alpha, return_not_sum_result=False)

        sentiment_output_with_aspect_embeddings = torch.cat([aspect_embeddings_single, sentiment_output],
                                                            dim=-1)
        final_sentiment_output = self.sentiment_fc(sentiment_output_with_aspect_embeddings)
        final_sentiment_output_prob = torch.softmax(final_sentiment_output, dim=-1)
        output = {'final_sentiment_output_prob': final_sentiment_output_prob}

        final_category_outputs = []
        for i in range(self.category_num):
            fc = self.category_fcs[i]
            category_output = category_outputs[i]
            final_category_output = fc(category_output)
            final_category_outputs.append(final_category_output)

        final_category_output = torch.cat(final_category_outputs, dim=1)
        if label is not None:
            sentiment_loss = self.sentiment_loss(final_sentiment_output, label)
            loss = sentiment_loss
            category_loss = self.category_loss(final_category_output, aspect_label.float())
            loss += category_loss
            output['loss'] = loss

            self._accuracy(final_sentiment_output, label)
            # category f1
            final_category_outputs_prob = [torch.sigmoid(e) for e in final_category_outputs]
            category_prob = torch.cat(final_category_outputs_prob).squeeze()
            aspect_labels = [aspect_label[:, k].squeeze() for k in range(self.category_num)]
            category_label = torch.cat(aspect_labels)
            self._f1(category_prob, category_label)
        pred_category = [torch.sigmoid(e) for e in final_category_outputs]
        output['pred_category'] = pred_category
        if self.configuration['visualize_attention']:
            for i in range(len(sample)):
                words = sample[i]['words']
                attention_labels = [e.split('/')[0] for e in self.categories]

                # category
                visual_attentions_category = [aspect_alphas[j][i][: len(words)].detach().cpu().numpy()
                                              for j in range(self.category_num)]
                titles = ['true: %s - pred: %s' % (str(aspect_label[i][j].detach().cpu().numpy()),
                                                   str(pred_category[j][i].detach().cpu().numpy()))
                          for j in range(self.category_num)]
                attention_visualizer.plot_multi_attentions_of_sentence(words, visual_attentions_category,
                                                                       attention_labels, titles)

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {
            'accuracy': self._accuracy.get_metric(reset),
            'category_f1': self._f1.get_metric(reset)['fscore']
        }
        return metrics


class HeatCaeM(TextInAllAspectSentimentOutModel):
    def __init__(self, word_embedder: TextFieldEmbedder, position_embedder: TextFieldEmbedder, aspect_embedder: TextFieldEmbedder,
                 categories: list, polarities: list, vocab: Vocabulary, configuration: dict):
        super().__init__(vocab)
        self.configuration = configuration
        self.word_embedder = word_embedder
        self.position_embedder = position_embedder
        self.aspect_embedder = aspect_embedder
        self.categories = categories
        self.polarites = polarities
        self.category_num = len(categories)
        self.polarity_num = len(polarities)
        self.category_loss = nn.BCEWithLogitsLoss()
        self.sentiment_loss = nn.CrossEntropyLoss()
        self._accuracy = metrics.CategoricalAccuracy()
        self._f1 = allennlp_metrics.BinaryF1(0.5)

        word_embedding_dim = word_embedder.get_output_dim()
        lstm_input_size = word_embedding_dim
        num_layers = 1
        hidden_size = 32
        self.aspect_gru = torch.nn.GRU(lstm_input_size, hidden_size, batch_first=True,
                                       bidirectional=True, num_layers=num_layers)
        self.sentiment_gru = torch.nn.GRU(lstm_input_size, hidden_size, batch_first=True,
                                          bidirectional=True, num_layers=num_layers)
        self.aspect_attentions = [AttentionInHtt(hidden_size * 3, hidden_size) for _ in range(self.category_num)]
        self.sentiment_attention = AttentionInHtt(hidden_size * 5, hidden_size, softmax=False)

        self.sentiment_fc = nn.Sequential(nn.Linear(hidden_size * 3, self.polarity_num))
        self.category_fcs = [nn.Linear(hidden_size * 2, 1) for _ in range(self.category_num)]

    def forward(self, tokens: Dict[str, torch.Tensor], aspects: torch.Tensor, sample: list,label: torch.Tensor = None,
                polarity_mask: torch.Tensor=None, position: torch.Tensor=None) -> torch.Tensor:
        mask = get_text_field_mask(tokens)
        word_embeddings = self.word_embedder(tokens)
        aspect_gru_output, _ = self.aspect_gru(word_embeddings)
        sentiment_gru_output, _ = self.sentiment_gru(word_embeddings)

        aspects_separate = [{'aspect': aspects['aspect'][:, i].unsqueeze(1)} for i in range(self.category_num)]
        aspect_embeddings_singles = [self.aspect_embedder(aspects_separate[i]).squeeze(1) for i in range(self.category_num)]
        aspects_seprate_repeat = [{'aspect': aspects_separate[i]['aspect'].expand_as(tokens['tokens'])} for i in range(self.category_num)]
        aspect_embeddings_separate = [self.aspect_embedder(aspects_seprate_repeat[i]) for i in range(self.category_num)]

        aspect_alphas = []
        sentiment_alphas = []
        final_category_outputs = []
        final_sentiment_outputs = []
        final_sentiment_output_probs = []
        for i in range(self.category_num):
            aspect_embeddings = aspect_embeddings_separate[i]
            input_for_aspect_attention = torch.cat([aspect_embeddings, aspect_gru_output], dim=-1)
            aspect_alpha = self.aspect_attentions[i](input_for_aspect_attention, mask)
            aspect_alphas.append(aspect_alpha)
            category_output = self.element_wise_mul(aspect_gru_output, aspect_alpha,
                                                         return_not_sum_result=False)
            category_fc = self.category_fcs[i]
            final_category_output = category_fc(category_output)
            final_category_outputs.append(final_category_output)

            category_output_unsqueeze = category_output.unsqueeze(1)
            category_output_repeat = category_output_unsqueeze.repeat(1, sentiment_gru_output.size()[1], 1)
            input_for_sentiment_attention = torch.cat([aspect_embeddings, category_output_repeat, sentiment_gru_output],
                                                      dim=-1)
            similarities = self.sentiment_attention(input_for_sentiment_attention, mask)

            location_mask_layer = LocationMaskLayer(aspect_alpha.size(1), self.configuration)
            location_mask = location_mask_layer(aspect_alpha)

            similarities_with_location = similarities * location_mask
            sentiment_alpha = allennlp_util.masked_softmax(similarities_with_location, mask)
            sentiment_alphas.append(sentiment_alpha)

            sentiment_output = self.element_wise_mul(sentiment_gru_output, sentiment_alpha, return_not_sum_result=False)

            aspect_embeddings_single = aspect_embeddings_singles[i]
            sentiment_output_with_aspect_embeddings = torch.cat([aspect_embeddings_single, sentiment_output],
                                                                dim=-1)
            final_sentiment_output = self.sentiment_fc(sentiment_output_with_aspect_embeddings)
            final_sentiment_outputs.append(final_sentiment_output)
            final_sentiment_output_prob = torch.softmax(final_sentiment_output, dim=-1)
            final_sentiment_output_probs.append(final_sentiment_output_prob)

        output = {}
        if label is not None:
            category_labels = []
            polarity_labels = []
            polarity_masks = []
            for i in range(self.category_num):
                category_labels.append(label[:, i])
                polarity_labels.append(label[:, i + self.category_num])
                polarity_masks.append(polarity_mask[:, i])

            loss = 0
            final_category_output = torch.cat(final_category_outputs, dim=1)
            category_label = torch.cat([e.unsqueeze(1)for e in category_labels], dim=1)
            category_loss = self.category_loss(final_category_output, category_label)
            loss += category_loss
            if not self.configuration['only_acd']:
                for i in range(self.category_num):
                    sentiment_temp_loss = self.sentiment_loss(final_sentiment_outputs[i], polarity_labels[i].long())
                    loss += sentiment_temp_loss

            # sentiment accuracy
            sentiment_logit = torch.cat(final_sentiment_outputs)
            sentiment_label = torch.cat(polarity_labels)
            sentiment_mask = torch.cat(polarity_masks)
            self._accuracy(sentiment_logit, sentiment_label, sentiment_mask)

            # category f1
            final_category_outputs_prob = [torch.sigmoid(e) for e in final_category_outputs]
            category_prob = torch.cat(final_category_outputs_prob).squeeze()
            category_label = torch.cat(category_labels)
            self._f1(category_prob, category_label)

            output['loss'] = loss
        pred_category = [torch.sigmoid(e) for e in final_category_outputs]
        pred_sentiment = [torch.nn.functional.softmax(e, dim=-1) for e in final_sentiment_outputs]
        output['pred_category'] = pred_category
        output['pred_sentiment'] = pred_sentiment

        # visualize attention
        if self.configuration['visualize_attention']:
            for i in range(len(sample)):
                words = sample[i][2]
                attention_labels = [e.split('/')[0] for e in self.categories]

                # category embedding layer
                visual_attentions = [aspect_alphas[j][i][: len(words)].detach().cpu().numpy()
                                     for j in range(self.category_num)]
                titles = ['aspect-true: %s - pred: %s' % (str(label[i][j].detach().cpu().numpy()),
                                                      str(pred_category[j][i].detach().cpu().numpy()))
                          for j in range(self.category_num)]
                attention_visualizer.plot_multi_attentions_of_sentence(words, visual_attentions, attention_labels,
                                                                       titles)
                # sentiment embedding layer
                visual_attentions = [sentiment_alphas[j][i][: len(words)].detach().cpu().numpy()
                                     for j in range(self.category_num)]
                titles = ['sentiment-true: %s - pred: %s - %s' % (str(label[i][j + self.category_num].detach().cpu().numpy()),
                                                           str(pred_sentiment[j][i].detach().cpu().numpy()),
                                                           str(self.polarites))
                          for j in range(self.category_num)]
                attention_visualizer.plot_multi_attentions_of_sentence(words, visual_attentions, attention_labels,
                                                                       titles)

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {
            'accuracy': self._accuracy.get_metric(reset),
            'category_f1': self._f1.get_metric(reset)['fscore']
        }
        return metrics


class AtaeLstm(TextAspectInSentimentOutModel):
    def __init__(self, word_embedder: TextFieldEmbedder, aspect_embedder: TextFieldEmbedder,
                 categories: list, polarities: list, vocab: Vocabulary, configuration: dict):
        super().__init__(vocab)
        self.configuration = configuration
        self.word_embedder = word_embedder
        self.aspect_embedder = aspect_embedder
        self.categories = categories
        self.polarites = polarities
        self.category_num = len(categories)
        self.polarity_num = len(polarities)
        self.sentiment_loss = nn.CrossEntropyLoss()
        self._accuracy = metrics.CategoricalAccuracy()

        word_embedding_dim = word_embedder.get_output_dim()
        aspect_word_embedding_dim = aspect_embedder.get_output_dim()
        if self.configuration['model_name'] in ['ae-lstm', 'atae-lstm']:
            lstm_input_size = word_embedding_dim + aspect_word_embedding_dim
        else:
            lstm_input_size = word_embedding_dim
        num_layers = 1
        hidden_size = 300
        self.lstm = torch.nn.LSTM(lstm_input_size, hidden_size, batch_first=True,
                                          bidirectional=False, num_layers=num_layers)
        if self.configuration['model_name'] in ['at-lstm', 'atae-lstm']:
            attention_input_size = word_embedding_dim + aspect_word_embedding_dim
            self.sentiment_attention = AttentionInHtt(attention_input_size, lstm_input_size)
            self.sentiment_fc = nn.Sequential(nn.Linear(hidden_size * 2, self.polarity_num))
        else:
            self.sentiment_attention = None
            self.sentiment_fc = nn.Sequential(nn.Linear(hidden_size, self.polarity_num))

    def forward(self, tokens: Dict[str, torch.Tensor], aspect: torch.Tensor, label: torch.Tensor,
                sample: list) -> torch.Tensor:
        mask = get_text_field_mask(tokens)
        word_embeddings = self.word_embedder(tokens)

        aspect_embeddings_single = self.aspect_embedder(aspect).squeeze(1)
        aspect_repeat = {'aspect': aspect['aspect'].expand_as(tokens['tokens'])}
        aspect_embeddings = self.aspect_embedder(aspect_repeat)

        if self.configuration['model_name'] in ['ae-lstm', 'atae-lstm']:
            lstm_input = torch.cat([aspect_embeddings, word_embeddings], dim=-1)
        else:
            lstm_input = word_embeddings
        lstm_output, (lstm_hn, lstm_cn) = self.lstm(lstm_input)
        lstm_hn = lstm_hn.squeeze(dim=0)

        if self.configuration['model_name'] in ['at-lstm', 'atae-lstm']:
            input_for_attention = torch.cat([aspect_embeddings, lstm_output], dim=-1)
            alpha = self.sentiment_attention(input_for_attention, mask)
            sentiment_output = self.element_wise_mul(lstm_output, alpha, return_not_sum_result=False)
            sentiment_output = torch.cat([sentiment_output, lstm_hn], dim=-1)
        else:
            sentiment_output = lstm_hn
        final_sentiment_output = self.sentiment_fc(sentiment_output)
        final_sentiment_output_prob = torch.softmax(final_sentiment_output, dim=-1)
        output = {'final_sentiment_output_prob': final_sentiment_output_prob}
        if label is not None:
            sentiment_loss = self.sentiment_loss(final_sentiment_output, label)
            self._accuracy(final_sentiment_output, label)
            output['loss'] = sentiment_loss

        # visualize attention
        if self.configuration['visualize_attention']:
            for i in range(len(sample)):
                words = sample[i][2]
                if not (('great' in words and 'dreadful' in words) or ('surprisingly' in words and 'decor' in words)):
                    continue
                attention_labels = [self.categories[sample[i][1][0]]]

                # 只可视化mams
                if not sample[i][3]:
                    continue

                visual_attentions = [alpha[i][: len(words)].detach().cpu().numpy()]
                titles = ['true: %s - pred: %s - %s' % (str(label[i].detach().cpu().numpy()),
                                                               str(final_sentiment_output_prob[i].detach().cpu().numpy()),
                                                               str(self.polarites)
                                                               )
                          ]
                savefig_filepath = super()._get_model_visualization_picture_filepath(self.configuration, words)
                attention_visualizer.plot_multi_attentions_of_sentence(words, visual_attentions, attention_labels,
                                                                       titles, savefig_filepath=savefig_filepath)

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {
            'accuracy': self._accuracy.get_metric(reset),
        }
        return metrics


class TextAspectInSentimentOutModelBert(TextAspectInSentimentOutModel):
    def __init__(self, word_embedder: TextFieldEmbedder, aspect_embedder: TextFieldEmbedder,
                 categories: list, polarities: list, vocab: Vocabulary, configuration: dict):
        super().__init__(vocab)
        self.configuration = configuration
        self.word_embedder = word_embedder
        self.aspect_embedder = aspect_embedder
        self.categories = categories
        self.polarites = polarities
        self.category_num = len(categories)
        self.polarity_num = len(polarities)
        self.sentiment_loss = nn.CrossEntropyLoss()
        self._accuracy = metrics.CategoricalAccuracy()

        word_embedding_dim = word_embedder.get_output_dim()
        self.sentiment_fc = nn.Sequential(nn.Linear(word_embedding_dim, self.polarity_num))

    def forward(self, tokens: Dict[str, torch.Tensor], aspect: torch.Tensor, label: torch.Tensor,
                sample: list) -> torch.Tensor:
        mask = get_text_field_mask(tokens)
        # bert_mask = tokens['mask']
        # tokens_index = tokens['tokens']
        # tokens_size = tokens_index.size()
        # for i in range(tokens_size[0]):
        #     print(tokens_index[i])
        #     print(mask[i])
        #     print(tokens['tokens-type-ids'][i])
        #     print(sample[i]['words'])
        #     print()
        token_type_ids = tokens['tokens-type-ids']
        offsets = tokens['tokens-offsets']
        word_embeddings = self.word_embedder(tokens, token_type_ids=token_type_ids, offsets=offsets)

        sentiment_output = word_embeddings[:, 0, :]
        final_sentiment_output = self.sentiment_fc(sentiment_output)
        final_sentiment_output_prob = torch.softmax(final_sentiment_output, dim=-1)
        output = {'final_sentiment_output_prob': final_sentiment_output_prob}
        if label is not None:
            sentiment_loss = self.sentiment_loss(final_sentiment_output, label)
            self._accuracy(final_sentiment_output, label)
            output['loss'] = sentiment_loss

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {
            'accuracy': self._accuracy.get_metric(reset),
        }
        return metrics


class Lstm(Model):
    def __init__(self, word_embedder: TextFieldEmbedder, aspect_embedder: TextFieldEmbedder,
                 categories: list, polarities: list, vocab: Vocabulary, configuration: dict):
        super().__init__(vocab)
        self.configuration = configuration
        self.word_embedder = word_embedder
        self.aspect_embedder = aspect_embedder
        self.categories = categories
        self.polarites = polarities
        self.category_num = len(categories)
        self.polarity_num = len(polarities)
        self.sentiment_loss = nn.CrossEntropyLoss()
        self._accuracy = metrics.CategoricalAccuracy()

        word_embedding_dim = word_embedder.get_output_dim()
        lstm_input_size = word_embedding_dim
        num_layers = 1
        hidden_size = 300
        self.lstm = torch.nn.LSTM(lstm_input_size, hidden_size, batch_first=True,
                                          bidirectional=False, num_layers=num_layers)
        self.sentiment_fc = nn.Sequential(nn.Linear(hidden_size, self.polarity_num))

    def matrix_mul(self, input, weight, bias=False):
        feature_list = []
        for feature in input:
            feature = torch.mm(feature, weight)
            if isinstance(bias, torch.nn.parameter.Parameter):
                feature = feature + bias.expand(feature.size()[0], bias.size()[1])
            feature = torch.tanh(feature).unsqueeze(0)
            feature_list.append(feature)

        return torch.cat(feature_list, 0).squeeze()

    def element_wise_mul(self, input1, input2, return_not_sum_result=False):
        feature_list = []
        for feature_1, feature_2 in zip(input1, input2):
            feature_2 = feature_2.unsqueeze(1)
            feature_2 = feature_2.expand_as(feature_1)
            feature = feature_1 * feature_2
            feature = feature.unsqueeze(0)
            feature_list.append(feature)
        output = torch.cat(feature_list, 0)

        result = torch.sum(output, 1)
        if return_not_sum_result:
            return result, output
        else:
            return result

    def reduce(self, nodes):
        """Take an average over all neighbor node features hu and use it to
        overwrite the original node feature."""
        m = nodes.mailbox['m']
        accum = torch.sum(m, 1)
        return {'h': accum}

    def pad_dgl_graph(self, graphs, max_node_num):
        graphs_padded = []
        for graph in graphs:
            graph_padded = copy.deepcopy(graph)
            node_num = graph.number_of_nodes()
            graph_padded.add_nodes(max_node_num - node_num)
            graphs_padded.append(graph_padded)
        return graphs_padded

    def forward(self, tokens: Dict[str, torch.Tensor], aspect: torch.Tensor, label: torch.Tensor,
                sample: list) -> torch.Tensor:
        mask = get_text_field_mask(tokens)
        word_embeddings = self.word_embedder(tokens)

        lstm_input = word_embeddings
        lstm_output, (lstm_hn, lstm_cn) = self.lstm(lstm_input)
        lstm_hn = lstm_hn.squeeze(dim=0)

        sentiment_output = lstm_hn
        final_sentiment_output = self.sentiment_fc(sentiment_output)
        final_sentiment_output_prob = torch.softmax(final_sentiment_output, dim=-1)
        output = {'final_sentiment_output_prob': final_sentiment_output_prob}
        if label is not None:
            sentiment_loss = self.sentiment_loss(final_sentiment_output, label)
            self._accuracy(final_sentiment_output, label)
            output['loss'] = sentiment_loss

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {
            'accuracy': self._accuracy.get_metric(reset),
        }
        return metrics


class BilstmAttn(TextAspectInSentimentOutModel):
    def __init__(self, word_embedder: TextFieldEmbedder, aspect_embedder: TextFieldEmbedder,
                 categories: list, polarities: list, vocab: Vocabulary, configuration: dict):
        super().__init__(vocab)
        self.configuration = configuration
        self.word_embedder = word_embedder
        self.aspect_embedder = aspect_embedder
        self.categories = categories
        self.polarites = polarities
        self.category_num = len(categories)
        self.polarity_num = len(polarities)
        self.sentiment_loss = nn.CrossEntropyLoss()
        self._accuracy = metrics.CategoricalAccuracy()

        word_embedding_dim = word_embedder.get_output_dim()

        lstm_input_size = word_embedding_dim
        num_layers = 1
        hidden_size = 300
        self.lstm = torch.nn.LSTM(lstm_input_size, hidden_size, batch_first=True,
                                          bidirectional=True, num_layers=num_layers)
        self.sentiment_attention = AttentionInHtt(hidden_size * 2, hidden_size * 2)
        self.sentiment_fc = nn.Sequential(nn.Linear(hidden_size * 2, self.polarity_num))

    def forward(self, tokens: Dict[str, torch.Tensor], aspect: torch.Tensor, label: torch.Tensor,
                sample: list) -> torch.Tensor:
        mask = get_text_field_mask(tokens)
        word_embeddings = self.word_embedder(tokens)

        lstm_input = word_embeddings
        lstm_output, (lstm_hn, lstm_cn) = self.lstm(lstm_input)

        input_for_attention = lstm_output
        alpha = self.sentiment_attention(input_for_attention, mask)
        sentiment_output = self.element_wise_mul(lstm_output, alpha, return_not_sum_result=False)

        final_sentiment_output = self.sentiment_fc(sentiment_output)
        final_sentiment_output_prob = torch.softmax(final_sentiment_output, dim=-1)
        output = {'final_sentiment_output_prob': final_sentiment_output_prob}
        if label is not None:
            sentiment_loss = self.sentiment_loss(final_sentiment_output, label)
            self._accuracy(final_sentiment_output, label)
            output['loss'] = sentiment_loss

            # visualize attention
            if self.configuration['visualize_attention']:
                for i in range(len(sample)):
                    words = sample[i][2]
                    # if not (('great' in words and 'dreadful' in words) or (
                    #         'surprisingly' in words and 'decor' in words)):
                    #     continue
                    attention_labels = [self.categories[sample[i][1][0]]]

                    # 只可视化mams
                    # if not sample[i][3]:
                    #     continue

                    visual_attentions = [alpha[i][: len(words)].detach().cpu().numpy()]
                    titles = ['true: %s - pred: %s - %s' % (str(label[i].detach().cpu().numpy()),
                                                            str(final_sentiment_output_prob[i].detach().cpu().numpy()),
                                                            str(self.polarites)
                                                            )
                              ]
                    savefig_filepath = super()._get_model_visualization_picture_filepath(self.configuration, words)
                    attention_visualizer.plot_multi_attentions_of_sentence_backup(words, visual_attentions, attention_labels,
                                                                           titles, savefig_filepath=savefig_filepath)

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {
            'accuracy': self._accuracy.get_metric(reset),
        }
        return metrics


class GCAE(Model):
    def __init__(self, word_embedder: TextFieldEmbedder, aspect_embedder: TextFieldEmbedder,
                 categories: list, polarities: list, vocab: Vocabulary, configuration: dict):
        super().__init__(vocab)
        self.configuration = configuration
        self.word_embedder = word_embedder
        self.aspect_embedder = aspect_embedder
        self.categories = categories
        self.polarites = polarities
        self.category_num = len(categories)
        self.polarity_num = len(polarities)
        self.sentiment_loss = nn.CrossEntropyLoss()
        self._accuracy = metrics.CategoricalAccuracy()

        word_embedding_dim = word_embedder.get_output_dim()

        aspect_embed_dim = 300
        # V = args.embed_num
        D = word_embedding_dim
        C = self.polarity_num
        A = self.category_num

        Co = 100
        Ks = [3, 4, 5]

        self.convs1 = nn.ModuleList([nn.Conv1d(D, Co, K) for K in Ks])
        self.convs2 = nn.ModuleList([nn.Conv1d(D, Co, K) for K in Ks])

        self.fc1 = nn.Linear(len(Ks) * Co, C)
        self.fc_aspect = nn.Linear(aspect_embed_dim, Co)

    def matrix_mul(self, input, weight, bias=False):
        feature_list = []
        for feature in input:
            feature = torch.mm(feature, weight)
            if isinstance(bias, torch.nn.parameter.Parameter):
                feature = feature + bias.expand(feature.size()[0], bias.size()[1])
            feature = torch.tanh(feature).unsqueeze(0)
            feature_list.append(feature)

        return torch.cat(feature_list, 0).squeeze()

    def element_wise_mul(self, input1, input2, return_not_sum_result=False):
        feature_list = []
        for feature_1, feature_2 in zip(input1, input2):
            feature_2 = feature_2.unsqueeze(1)
            feature_2 = feature_2.expand_as(feature_1)
            feature = feature_1 * feature_2
            feature = feature.unsqueeze(0)
            feature_list.append(feature)
        output = torch.cat(feature_list, 0)

        result = torch.sum(output, 1)
        if return_not_sum_result:
            return result, output
        else:
            return result

    def reduce(self, nodes):
        """Take an average over all neighbor node features hu and use it to
        overwrite the original node feature."""
        m = nodes.mailbox['m']
        accum = torch.sum(m, 1)
        return {'h': accum}

    def pad_dgl_graph(self, graphs, max_node_num):
        graphs_padded = []
        for graph in graphs:
            graph_padded = copy.deepcopy(graph)
            node_num = graph.number_of_nodes()
            graph_padded.add_nodes(max_node_num - node_num)
            graphs_padded.append(graph_padded)
        return graphs_padded

    def forward(self, tokens: Dict[str, torch.Tensor], aspect: torch.Tensor, label: torch.Tensor,
                sample: list) -> torch.Tensor:
        mask = get_text_field_mask(tokens)
        feature = self.word_embedder(tokens)


        aspect_v = self.aspect_embedder(aspect)
        aspect_v = aspect_v.squeeze(1)

        x = []
        for conv in self.convs1:
            feature_t = feature.transpose(1, 2)
            xe = F.tanh(conv(feature_t))
            x.append(xe)
        y = []
        for conv in self.convs2:
            ye1 = conv(feature.transpose(1, 2))
            ye2 = self.fc_aspect(aspect_v)
            ye2_unsqueeze = ye2.unsqueeze(2)
            ye2_expand = ye2_unsqueeze.expand_as(ye1)
            ye = F.relu(ye1 + ye2_expand)
            y.append(ye)
        x_temp = []
        for i, j in zip(x, y):
            x_temp_e = i * j
            x_temp.append(x_temp_e)
        x = x_temp
        # pooling method
        x0 = []
        for i in x:
            x0e = F.max_pool1d(i, i.size(2))
            x0e_s = x0e.squeeze(2)
            x0.append(x0e_s)
        x0 = [i.view(i.size(0), -1) for i in x0]

        x0 = torch.cat(x0, 1)
        final_sentiment_output = self.fc1(x0)  # (N,C)

        final_sentiment_output_prob = torch.softmax(final_sentiment_output, dim=-1)
        output = {'final_sentiment_output_prob': final_sentiment_output_prob}
        if label is not None:
            sentiment_loss = self.sentiment_loss(final_sentiment_output, label)
            self._accuracy(final_sentiment_output, label)
            output['loss'] = sentiment_loss

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {
            'accuracy': self._accuracy.get_metric(reset),
        }
        return metrics


class TextCNN(Model):
    def __init__(self, word_embedder: TextFieldEmbedder, aspect_embedder: TextFieldEmbedder,
                 categories: list, polarities: list, vocab: Vocabulary, configuration: dict):
        super().__init__(vocab)
        self.configuration = configuration
        self.word_embedder = word_embedder
        self.aspect_embedder = aspect_embedder
        self.categories = categories
        self.polarites = polarities
        self.category_num = len(categories)
        self.polarity_num = len(polarities)
        self.sentiment_loss = nn.CrossEntropyLoss()
        self._accuracy = metrics.CategoricalAccuracy()

        word_embedding_dim = word_embedder.get_output_dim()
        filter_sizes = (2,)
        num_filters = 256
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, num_filters, (k, word_embedding_dim)) for k in filter_sizes])
        self.sentiment_fc = nn.Linear(num_filters * len(filter_sizes), self.polarity_num)
        self.dropout = nn.Dropout(0.5)

    def matrix_mul(self, input, weight, bias=False):
        feature_list = []
        for feature in input:
            feature = torch.mm(feature, weight)
            if isinstance(bias, torch.nn.parameter.Parameter):
                feature = feature + bias.expand(feature.size()[0], bias.size()[1])
            feature = torch.tanh(feature).unsqueeze(0)
            feature_list.append(feature)

        return torch.cat(feature_list, 0).squeeze()

    def element_wise_mul(self, input1, input2, return_not_sum_result=False):
        feature_list = []
        for feature_1, feature_2 in zip(input1, input2):
            feature_2 = feature_2.unsqueeze(1)
            feature_2 = feature_2.expand_as(feature_1)
            feature = feature_1 * feature_2
            feature = feature.unsqueeze(0)
            feature_list.append(feature)
        output = torch.cat(feature_list, 0)

        result = torch.sum(output, 1)
        if return_not_sum_result:
            return result, output
        else:
            return result

    def reduce(self, nodes):
        """Take an average over all neighbor node features hu and use it to
        overwrite the original node feature."""
        m = nodes.mailbox['m']
        accum = torch.sum(m, 1)
        return {'h': accum}

    def pad_dgl_graph(self, graphs, max_node_num):
        graphs_padded = []
        for graph in graphs:
            graph_padded = copy.deepcopy(graph)
            node_num = graph.number_of_nodes()
            graph_padded.add_nodes(max_node_num - node_num)
            graphs_padded.append(graph_padded)
        return graphs_padded

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x))
        x = x.squeeze(3)
        x = F.max_pool1d(x, x.size(2))
        x = x.squeeze(2)
        return x

    def forward(self, tokens: Dict[str, torch.Tensor], aspect: torch.Tensor, label: torch.Tensor,
                sample: list) -> torch.Tensor:
        mask = get_text_field_mask(tokens)
        word_embeddings = self.word_embedder(tokens)

        word_embeddings = word_embeddings.unsqueeze(1)
        sentiment_output = torch.cat([self.conv_and_pool(word_embeddings, conv) for conv in self.convs], 1)
        sentiment_output = self.dropout(sentiment_output)

        final_sentiment_output = self.sentiment_fc(sentiment_output)
        final_sentiment_output_prob = torch.softmax(final_sentiment_output, dim=-1)
        output = {'final_sentiment_output_prob': final_sentiment_output_prob}
        if label is not None:
            sentiment_loss = self.sentiment_loss(final_sentiment_output, label)
            self._accuracy(final_sentiment_output, label)
            output['loss'] = sentiment_loss

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {
            'accuracy': self._accuracy.get_metric(reset),
        }
        return metrics


class HeatEstimator:
    def __init__(self, model: Model, iterator: DataIterator, categories: list, polarities: list,
                 cuda_device: int = -1) -> None:
        self.model = model
        self.iterator = iterator
        self.categories = categories
        self.polarities = polarities
        self._accuracy = metrics.CategoricalAccuracy()
        self.cuda_device = cuda_device

    def _estimate(self, batch) -> np.ndarray:
        label = batch['label']
        out_dict = self.model(**batch)
        sentiment_prob = out_dict['final_sentiment_output_prob']
        self._accuracy(sentiment_prob, label)

    def estimate(self, ds: Iterable[Instance]) -> np.ndarray:
        self.model.eval()
        pred_generator = self.iterator(ds, num_epochs=1, shuffle=False)
        pred_generator_tqdm = tqdm(pred_generator,
                                   total=self.iterator.get_num_batches(ds))
        with torch.no_grad():
            for batch in pred_generator_tqdm:
                batch = nn_util.move_to_device(batch, self.cuda_device)
                self._estimate(batch)
        return self._accuracy.get_metric()


class TextAspectInSentimentOutEstimator:
    def __init__(self, model: Model, iterator: DataIterator, categories: list, polarities: list,
                 cuda_device: int = -1) -> None:
        self.model = model
        self.iterator = iterator
        self.categories = categories
        self.polarities = polarities
        self._accuracy = metrics.CategoricalAccuracy()
        self._sentiment_accuracy_temp = metrics.CategoricalAccuracy()
        self.cuda_device = cuda_device
        self.other_metrics = {}

    def _get_other_metrics(self, reset=True):
        result = self.other_metrics
        if reset:
            self.other_metrics = {}
        return result

    def _acsc_aspect_and_metrics(self, y_true, y_pred, aspect_index):
        acsc_aspect_and_metrics = {}
        for i, aspect in enumerate(self.categories):
            current_aspect_index = aspect_index == i
            aspect_sentiment_label_i = y_true[current_aspect_index]
            aspect_sentiment_pred_i = y_pred[current_aspect_index]
            self._sentiment_accuracy_temp(aspect_sentiment_pred_i, aspect_sentiment_label_i)
            aspect_acc_temp = self._sentiment_accuracy_temp.get_metric(reset=True),
            acsc_aspect_and_metrics[aspect] = {
                'acc': aspect_acc_temp[0],
            }
        return acsc_aspect_and_metrics

    def estimate(self, ds: Iterable[Instance]) -> np.ndarray:
        self.model.eval()
        pred_generator = self.iterator(ds, num_epochs=1, shuffle=False)
        pred_generator_tqdm = tqdm(pred_generator,
                                   total=self.iterator.get_num_batches(ds))
        with torch.no_grad():
            aspect_indices = []
            labels = []
            preds = []
            for batch in pred_generator_tqdm:
                batch = nn_util.move_to_device(batch, self.cuda_device)
                label = batch['label']
                out_dict = self.model(**batch)
                sentiment_prob = out_dict['final_sentiment_output_prob']
                labels.append(label)
                preds.append(sentiment_prob)

                for e in batch['sample']:
                    aspect_index = e[1][0]
                    aspect_indices.append(aspect_index)
            label_final = torch.cat(labels, dim=0)
            pred_final = torch.cat(preds, dim=0)
            self._accuracy(pred_final, label_final)
            aspect_indices = torch.tensor(aspect_indices)
            acsc_aspect_and_metrics = self._acsc_aspect_and_metrics(label_final, pred_final, aspect_indices)
            self.other_metrics['acsc_metrics'] = acsc_aspect_and_metrics
        return {'sentiment_acc': self._accuracy.get_metric(reset=True),
                'other_metrics': self._get_other_metrics()}


class TextAspectInSentimentOutPredictor(Predictor):
    def __init__(self, model: Model, iterator: DataIterator, categories: list, polarities: list,
                 cuda_device: int = -1) -> None:
        self.model = model
        self.iterator = iterator
        self.categories = categories
        self.polarities = polarities
        self._accuracy = metrics.CategoricalAccuracy()
        self.cuda_device = cuda_device

    def predict(self, ds: Iterable[Instance]) -> np.ndarray:
        with torch.no_grad():
            self.model.eval()
            pred_generator = self.iterator(ds, num_epochs=1, shuffle=False)
            pred_generator_tqdm = tqdm(pred_generator,
                                       total=self.iterator.get_num_batches(ds))
            labels = []
            preds = []
            for batch in pred_generator_tqdm:
                batch = nn_util.move_to_device(batch, self.cuda_device)
                label = batch['label']
                out_dict = self.model(**batch)
                sentiment_prob = out_dict['final_sentiment_output_prob']
                labels.append(label)
                preds.append(sentiment_prob)
            label_final = torch.cat(labels, dim=0)
            pred_final = torch.cat(preds, dim=0)
            result = []
            label_index_final = pred_final.argmax(dim=-1)
            for i in range(len(ds)):
                index_true = label_final[i]
                index_pred = label_index_final[i]
                sentiment = self.polarities[index_pred]
                result.append(sentiment)
        return result


class AsCapsules(TextInAllAspectSentimentOutModel):
    def __init__(self, word_embedder: TextFieldEmbedder, position_embedder: TextFieldEmbedder,
                 aspect_embedder: TextFieldEmbedder, categories: list, polarities: list, vocab: Vocabulary,
                 configuration: dict):
        super().__init__(vocab)
        self.configuration = configuration
        self.word_embedder = word_embedder
        self.position_embedder = position_embedder
        self.aspect_embedder = aspect_embedder
        self.categories = categories
        self.polarites = polarities
        self.category_num = len(categories)
        self.polarity_num = len(polarities)
        self.category_loss = nn.BCEWithLogitsLoss()
        self.sentiment_loss = nn.CrossEntropyLoss()
        self._accuracy = metrics.CategoricalAccuracy()
        self._f1 = allennlp_metrics.BinaryF1(0.5)

        self.aspect_attention = AttentionInHtt(256 * 3, 256)
        self.share_attention = AttentionInHtt(256 * 3, 256)
        self.sentiment_attention = AttentionInHtt(256 * 3, 256)


        lstm_input_size = word_embedder.get_output_dim()
        num_layers = 1
        self.rnne = torch.nn.LSTM(lstm_input_size, 256, batch_first=True,
                                  bidirectional=True, num_layers=num_layers)
        self.rnns = torch.nn.LSTM(256 * 3, 256, batch_first=True,
                                  bidirectional=True, num_layers=num_layers)
        self.dropout_after_embedding = nn.Dropout(0.5)

        self.category_fcs = [nn.Linear(256 * 6, 1) for _ in range(self.category_num)]
        self.category_fcs = nn.ModuleList(self.category_fcs)

        self.dropouts_category_fc = [nn.Dropout(0.5) for _ in range(self.category_num)]
        self.dropouts_category_fc = nn.ModuleList(self.dropouts_category_fc)

        self.sentiment_fcs = [nn.Linear(256 * 6, self.polarity_num) for _ in range(self.category_num)]
        self.sentiment_fcs = nn.ModuleList(self.sentiment_fcs)

        self.dropout_sentiment_fc = [nn.Dropout(0.5) for _ in range(self.category_num)]
        self.dropout_sentiment_fc = nn.ModuleList(self.dropout_sentiment_fc)

    def forward(self, tokens: Dict[str, torch.Tensor], label: torch.Tensor, position: torch.Tensor,
                polarity_mask: torch.Tensor, sample: list, aspects: torch.Tensor=None) -> torch.Tensor:
        mask = get_text_field_mask(tokens)
        word_embeddings = self.word_embedder(tokens)
        word_embeddings = self.dropout_after_embedding(word_embeddings)

        h1, _ = self.rnne(word_embeddings)

        aspects_separate = [{'aspect': aspects['aspect'][:, i].unsqueeze(1)} for i in range(self.category_num)]
        aspect_embeddings_singles = [self.aspect_embedder(aspects_separate[i]).squeeze(1) for i in
                                     range(self.category_num)]
        aspects_seprate_repeat = [{'aspect': aspects_separate[i]['aspect'].expand_as(tokens['tokens'])} for i in
                                  range(self.category_num)]
        aspect_embeddings_separate = [self.aspect_embedder(aspects_seprate_repeat[i]) for i in range(self.category_num)]

        final_category_outputs = []
        final_sentiment_outputs = []
        aspect_alphas = []
        shared_alphas = []
        sentiment_alphas = []
        for i in range(self.category_num):
            ec = aspect_embeddings_separate[i]
            rnns_input = torch.cat([h1, ec], dim=-1)
            h2, _ = self.rnns(rnns_input)

            attention_input = torch.cat([h2, ec], dim=-1)

            aspect_alpha = self.aspect_attention(attention_input, mask)
            shared_alpha = self.share_attention(attention_input, mask)
            sentiment_alpha = self.sentiment_attention(attention_input, mask)

            aspect_alphas.append(aspect_alpha)
            shared_alphas.append(shared_alpha)
            sentiment_alphas.append(sentiment_alpha)

            vah = self.element_wise_mul(h2, aspect_alpha, return_not_sum_result=False)
            val = self.element_wise_mul(h1, aspect_alpha, return_not_sum_result=False)
            vs = self.element_wise_mul(h2, shared_alpha, return_not_sum_result=False)
            voh = self.element_wise_mul(h2, sentiment_alpha, return_not_sum_result=False)
            vol = self.element_wise_mul(h1, sentiment_alpha, return_not_sum_result=False)
            ra = torch.cat([vah, val, vs], dim=-1)
            ra = self.dropouts_category_fc[i](ra)
            ro = torch.cat([voh, vol, vs], dim=-1)
            ro = self.dropout_sentiment_fc[i](ro)

            category_output = self.category_fcs[i](ra)
            final_category_outputs.append(category_output)

            sentiment_output = self.sentiment_fcs[i](ro)
            final_sentiment_outputs.append(sentiment_output)

        output = {}
        if label is not None:
            category_labels = []
            polarity_labels = []
            polarity_masks = []
            for i in range(self.category_num):
                category_labels.append(label[:, i])
                polarity_labels.append(label[:, i + self.category_num])
                polarity_masks.append(polarity_mask[:, i])
            loss = 0
            for i in range(self.category_num):
                category_temp_loss = self.category_loss(final_category_outputs[i].squeeze(dim=-1), category_labels[i])
                sentiment_temp_loss = self.sentiment_loss(final_sentiment_outputs[i], polarity_labels[i].long())
                loss += category_temp_loss
                loss += sentiment_temp_loss

            # sentiment accuracy
            sentiment_logit = torch.cat(final_sentiment_outputs)
            sentiment_label = torch.cat(polarity_labels)
            sentiment_mask = torch.cat(polarity_masks)
            self._accuracy(sentiment_logit, sentiment_label, sentiment_mask)

            # category f1
            final_category_outputs_prob = [torch.sigmoid(e) for e in final_category_outputs]
            category_prob = torch.cat(final_category_outputs_prob).squeeze()
            category_label = torch.cat(category_labels)
            self._f1(category_prob, category_label)

            output['loss'] = loss

        pred_category = [torch.sigmoid(e) for e in final_category_outputs]
        pred_sentiment = [torch.nn.functional.softmax(e, dim=-1) for e in final_sentiment_outputs]
        output['pred_category'] = pred_category
        output['pred_sentiment'] = pred_sentiment
        if self.configuration['visualize_attention']:
            for i in range(len(sample)):
                words = sample[i][2]
                # if not (('tiny' in words and 'restaurant' in words) or ('surprisingly' in words and 'decor' in words)):
                #     continue
                attention_labels = [e.split('/')[0] for e in self.categories]

                # 只可视化mams
                label_true = label[i].detach().cpu().numpy()[: self.category_num]
                if sum(label_true) <= 1:
                    continue

                # category alpha
                visual_attentions = [aspect_alphas[j][i][: len(words)].detach().cpu().numpy()
                                     for j in range(self.category_num)]
                titles = ['category-true: %s - pred: %s' % (str(label[i][j].detach().cpu().numpy()),
                                                   str(pred_category[j][i].detach().cpu().numpy()))
                          for j in range(self.category_num)]
                savefig_filepath = super()._get_model_visualization_picture_filepath(self.configuration, words)
                # attention_visualizer.plot_multi_attentions_of_sentence_backup(words, visual_attentions, attention_labels,
                #                                                        titles, savefig_filepath=savefig_filepath)

                # shared alpha
                visual_attentions = [shared_alphas[j][i][: len(words)].detach().cpu().numpy()
                                     for j in range(self.category_num)]
                titles = ['shared-true: %s - pred: %s' % (str(label[i][j].detach().cpu().numpy()),
                                                   str(pred_category[j][i].detach().cpu().numpy()))
                          for j in range(self.category_num)]
                savefig_filepath = super()._get_model_visualization_picture_filepath(self.configuration, words)
                # attention_visualizer.plot_multi_attentions_of_sentence_backup(words, visual_attentions, attention_labels,
                #                                                        titles, savefig_filepath=savefig_filepath)

                # sentiment alpha
                visual_attentions = [sentiment_alphas[j][i][: len(words)].detach().cpu().numpy()
                                     for j in range(self.category_num)]
                titles = ['sentiment-true: %s - pred: %s - %s' % (str(label[i][j + self.category_num].detach().cpu().numpy()),
                                                        str(pred_sentiment[j][i].detach().cpu().numpy()),
                                                        str(self.polarites))
                          for j in range(self.category_num)]
                savefig_filepath = super()._get_model_visualization_picture_filepath(self.configuration, words)
                attention_visualizer.plot_multi_attentions_of_sentence_backup(words, visual_attentions, attention_labels,
                                                                       titles, savefig_filepath=savefig_filepath)

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {
            'accuracy': self._accuracy.get_metric(reset),
            'category_f1': self._f1.get_metric(reset)['fscore']
        }
        return metrics


class Can(TextInAllAspectSentimentOutModel):
    """
    M-CAN-2Ro
    """
    def __init__(self, word_embedder: TextFieldEmbedder, position_embedder: TextFieldEmbedder,
                 aspect_embedder: TextFieldEmbedder, categories: list, polarities: list, vocab: Vocabulary,
                 configuration: dict):
        super().__init__(vocab)
        self.configuration = configuration
        self.word_embedder = word_embedder
        self.position_embedder = position_embedder
        self.aspect_embedder = aspect_embedder
        self.categories = categories
        self.polarites = polarities
        self.category_num = len(categories)
        self.polarity_num = len(polarities)
        self.category_loss = nn.BCEWithLogitsLoss()
        self.sentiment_loss = nn.CrossEntropyLoss()
        self._accuracy = metrics.CategoricalAccuracy()
        self._f1 = allennlp_metrics.BinaryF1(0.5)

        self.aspect_attentions = nn.ModuleList([AttentionInCan(300) for _ in range(self.category_num)])
        self.sentiment_attentions = nn.ModuleList([AttentionInCan(300) for _ in range(self.category_num)])

        lstm_input_size = word_embedder.get_output_dim()
        num_layers = 1
        self.lstm = torch.nn.LSTM(lstm_input_size, 300, batch_first=True,
                                  bidirectional=False, num_layers=num_layers)
        self.dropout_after_embedding = nn.Dropout(0.7)
        self.dropout_after_lstm = nn.Dropout(0.7)
        self.category_fcs = nn.ModuleList([nn.Linear(300, 1) for _ in range(self.category_num)])
        self.sentiment_fcs = nn.ModuleList([nn.Linear(300, 300) for _ in range(self.category_num)])
        self.sentiment_fcs2 = nn.ModuleList([nn.Linear(300, 300) for _ in range(self.category_num)])
        self.sentiment_fcs3 = nn.ModuleList([nn.Linear(300, self.polarity_num) for _ in range(self.category_num)])

    def forward(self, tokens: Dict[str, torch.Tensor], label: torch.Tensor, position: torch.Tensor,
                polarity_mask: torch.Tensor, sample: list, aspects: torch.Tensor=None) -> torch.Tensor:
        mask = get_text_field_mask(tokens)
        word_embeddings = self.word_embedder(tokens)
        word_embeddings = self.dropout_after_embedding(word_embeddings)

        h1, (h_n, c_n) = self.lstm(word_embeddings)
        h_n = h_n.squeeze(0)
        h1 = self.dropout_after_lstm(h1)

        aspects_separate = [{'aspect': aspects['aspect'][:, i].unsqueeze(1)} for i in range(self.category_num)]
        aspect_embeddings_singles = [self.aspect_embedder(aspects_separate[i]).squeeze(1) for i in
                                     range(self.category_num)]
        aspects_seprate_repeat = [{'aspect': aspects_separate[i]['aspect'].expand_as(tokens['tokens'])} for i in
                                  range(self.category_num)]
        aspect_embeddings_separate = [self.aspect_embedder(aspects_seprate_repeat[i]) for i in range(self.category_num)]

        final_category_outputs = []
        category_alphas = []
        final_sentiment_outputs = []
        sentiment_alphas = []
        for i in range(self.category_num):
            ec = aspect_embeddings_separate[i]

            sentiment_alpha = self.sentiment_attentions[i](h1, ec, mask)
            sentiment_alphas.append(sentiment_alpha)
            sentiment_repr = self.element_wise_mul(h1, sentiment_alpha, return_not_sum_result=False)

            rsk1 = self.sentiment_fcs[i](h_n)
            rsk2 = self.sentiment_fcs2[i](sentiment_repr)
            rsk = rsk1 + rsk2
            rsk = torch.tanh(rsk)
            sentiment_output = self.sentiment_fcs3[i](rsk)

            aspect_alpha = self.aspect_attentions[i](h1, ec, mask)
            category_alphas.append(aspect_alpha)
            aspect_repr = self.element_wise_mul(h1, aspect_alpha, return_not_sum_result=False)
            category_output = self.category_fcs[i](aspect_repr)

            final_category_outputs.append(category_output)
            final_sentiment_outputs.append(sentiment_output)

        output = {}
        if label is not None:
            category_labels = []
            polarity_labels = []
            polarity_masks = []
            for i in range(self.category_num):
                category_labels.append(label[:, i])
                polarity_labels.append(label[:, i + self.category_num])
                polarity_masks.append(polarity_mask[:, i])
            loss = 0
            for i in range(self.category_num):
                category_temp_loss = self.category_loss(final_category_outputs[i].squeeze(dim=-1), category_labels[i])
                category_temp_loss = category_temp_loss / self.category_num
                sentiment_temp_loss = self.sentiment_loss(final_sentiment_outputs[i], polarity_labels[i].long())
                loss += category_temp_loss
                loss += sentiment_temp_loss

            # Sparse Regularization Orthogonal Regularization
            if self.configuration['sparse_reg'] or self.configuration['orthogonal_reg']:
                reg_loss = 0
                for j in range(len(sample)):
                    polarity_mask_of_one_sample = polarity_mask[j]
                    category_alpha_of_one_sample = [category_alphas[k][j] for k in range(self.category_num)]
                    sentiment_alpha_of_one_sample = [category_alphas[k][j] for k in range(self.category_num)]
                    category_alpha_of_mentioned = []
                    category_alpha_of_not_mentioned = []
                    sentiment_alpha_of_mentioned = []
                    sentiment_alpha_of_not_mentioned = []
                    for k in range(self.category_num):
                        if polarity_mask_of_one_sample[k] == 1:
                            category_alpha_of_mentioned.append(category_alpha_of_one_sample[k].unsqueeze(0))
                            sentiment_alpha_of_mentioned.append(sentiment_alpha_of_one_sample[k].unsqueeze(0))
                        else:
                            category_alpha_of_not_mentioned.append(category_alpha_of_one_sample[k].unsqueeze(0))
                            sentiment_alpha_of_not_mentioned.append(sentiment_alpha_of_one_sample[k].unsqueeze(0))
                    if len(sentiment_alpha_of_not_mentioned) != 0:
                        category_alpha_of_not_mentioned = torch.cat(category_alpha_of_not_mentioned, dim=0)
                        sentiment_alpha_of_not_mentioned = torch.cat(sentiment_alpha_of_not_mentioned, dim=0)
                        category_alpha_of_not_mentioned = torch.mean(category_alpha_of_not_mentioned, dim=0, keepdim=True)
                        category_alpha_of_mentioned.append(category_alpha_of_not_mentioned)
                        sentiment_alpha_of_not_mentioned = torch.mean(sentiment_alpha_of_not_mentioned, dim=0, keepdim=True)
                        sentiment_alpha_of_mentioned.append(sentiment_alpha_of_not_mentioned)

                    category_eye = torch.eye(len(category_alpha_of_mentioned))
                    sentiment_eye = torch.eye(len(sentiment_alpha_of_mentioned))
                    category_alpha_of_mentioned = torch.cat(category_alpha_of_mentioned, dim=0)
                    sentiment_alpha_of_mentioned = torch.cat(sentiment_alpha_of_mentioned, dim=0)
                    category_alpha_similarity = torch.mm(category_alpha_of_mentioned, category_alpha_of_mentioned.t())
                    sentiment_alpha_similarity = torch.mm(sentiment_alpha_of_mentioned, sentiment_alpha_of_mentioned.t())
                    category_reg_loss = category_alpha_similarity - category_eye
                    category_reg_loss = torch.norm(category_reg_loss)
                    sentiment_reg_loss = sentiment_alpha_similarity - sentiment_eye
                    sentiment_reg_loss = torch.norm(sentiment_reg_loss)
                    reg_loss += category_reg_loss
                    reg_loss += sentiment_reg_loss
                loss += (reg_loss * self.configuration['lamda'] / len(sample))
            # sentiment accuracy
            sentiment_logit = torch.cat(final_sentiment_outputs)
            sentiment_label = torch.cat(polarity_labels)
            sentiment_mask = torch.cat(polarity_masks)
            self._accuracy(sentiment_logit, sentiment_label, sentiment_mask)

            # category f1
            final_category_outputs_prob = [torch.sigmoid(e) for e in final_category_outputs]
            category_prob = torch.cat(final_category_outputs_prob).squeeze()
            category_label = torch.cat(category_labels)
            self._f1(category_prob, category_label)

            output['loss'] = loss

        pred_category = [torch.sigmoid(e) for e in final_category_outputs]
        pred_sentiment = [torch.nn.functional.softmax(e, dim=-1) for e in final_sentiment_outputs]
        output['pred_category'] = pred_category
        output['pred_sentiment'] = pred_sentiment
        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {
            'accuracy': self._accuracy.get_metric(reset),
            'category_f1': self._f1.get_metric(reset)['fscore']
        }
        return metrics


class EndToEndBackup(TextInAllAspectSentimentOutModel):
    """
    2018-emnlp-Joint Aspect and Polarity Classification for Adpect-based Sentiment Analysis with End-to-End Neural Networks
    """
    def __init__(self, word_embedder: TextFieldEmbedder, position_embedder: TextFieldEmbedder,
                 aspect_embedder: TextFieldEmbedder, categories: list, polarities: list, vocab: Vocabulary,
                 configuration: dict):
        super().__init__(vocab)
        self.configuration = configuration
        self.word_embedder = word_embedder
        self.position_embedder = position_embedder
        self.aspect_embedder = aspect_embedder
        self.categories = categories
        self.polarites = polarities
        self.category_num = len(categories)
        self.polarity_num = len(polarities)
        self.category_loss = nn.BCEWithLogitsLoss()
        self.sentiment_loss = nn.CrossEntropyLoss()
        self._accuracy = metrics.CategoricalAccuracy()
        self._f1 = allennlp_metrics.BinaryF1(0.5)

        word_embedding_dim = word_embedder.get_output_dim()

        model_name = self.configuration['model_name']
        if model_name == 'End-to-end-LSTM':
            num_layers = 1
            self.lstm = torch.nn.LSTM(word_embedding_dim, int(word_embedding_dim / 2), batch_first=True,
                                      bidirectional=True, num_layers=num_layers)
        else:
            raise NotImplementedError(model_name)

        self.dropout_after_embedding = nn.Dropout(0.5)
        self.dropout_after_lstm = nn.Dropout(0.5)

        self.fcs = [nn.Linear(300, self.polarity_num + 1) for _ in range(self.category_num)]

    def forward(self, tokens: Dict[str, torch.Tensor], label: torch.Tensor, position: torch.Tensor,
                polarity_mask: torch.Tensor, sample: list, aspects: torch.Tensor=None) -> torch.Tensor:
        mask = get_text_field_mask(tokens)
        word_embeddings = self.word_embedder(tokens)

        model_name = self.configuration['model_name']
        if model_name == 'End-to-end-LSTM':
            word_embeddings = self.dropout_after_embedding(word_embeddings)
            h1, (h_n, c_n) = self.lstm(word_embeddings)
            v = torch.cat([h_n[0], h_n[1]], dim=-1)
        else:
            raise NotImplementedError(model_name)

        predicts_of_aspects = [self.fcs[i](v) for i in range(self.category_num)]
        sentiment_predicts = [torch.softmax(predicts_of_aspect[:, 1:], dim=-1) for predicts_of_aspect in predicts_of_aspects]
        aspect_predicts = [(torch.argmax(predicts_of_aspect, dim=-1, keepdim=True) == 0).float() for predicts_of_aspect in predicts_of_aspects]

        output = {}
        if label is not None:
            category_labels = []
            polarity_labels = []
            polarity_masks = []
            total_labeles = []
            for i in range(self.category_num):
                category_labels.append(label[:, i])
                polarity_labels.append(label[:, i + self.category_num])
                polarity_masks.append(polarity_mask[:, i])
                total_labeles.append(label[:, i + self.category_num * 2])
            loss = 0
            for i in range(self.category_num):
                temp_loss = self.sentiment_loss(predicts_of_aspects[i], total_labeles[i].long())
                loss += temp_loss

            # sentiment accuracy
            sentiment_logit = torch.cat(sentiment_predicts)
            sentiment_label = torch.cat(polarity_labels)
            sentiment_mask = torch.cat(polarity_masks)
            self._accuracy(sentiment_logit, sentiment_label, sentiment_mask)

            # category f1
            category_prob = torch.cat(aspect_predicts).squeeze()
            category_label = torch.cat(category_labels)
            self._f1(category_prob, category_label)

            output['loss'] = loss

        output['pred_category'] = aspect_predicts
        output['pred_sentiment'] = sentiment_predicts
        output['predicts_of_aspects'] = predicts_of_aspects
        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {
            'accuracy': self._accuracy.get_metric(reset),
            'category_f1': self._f1.get_metric(reset)['fscore']
        }
        return metrics


class EndToEnd(TextInAllAspectSentimentOutModel):
    """
    2018-emnlp-Joint Aspect and Polarity Classification for Adpect-based Sentiment Analysis with End-to-End Neural Networks
    """
    def __init__(self, word_embedder: TextFieldEmbedder, position_embedder: TextFieldEmbedder,
                 aspect_embedder: TextFieldEmbedder, categories: list, polarities: list, vocab: Vocabulary,
                 configuration: dict):
        super().__init__(vocab)
        self.configuration = configuration
        self.word_embedder = word_embedder
        self.position_embedder = position_embedder
        self.aspect_embedder = aspect_embedder
        self.categories = categories
        self.polarites = polarities
        self.category_num = len(categories)
        self.polarity_num = len(polarities)
        self.loss = nn.CrossEntropyLoss()

        word_embedding_dim = word_embedder.get_output_dim()

        model_name = self.configuration['model_name']
        if model_name == 'End-to-end-LSTM':
            num_layers = 1
            self.lstm = torch.nn.LSTM(word_embedding_dim, 200, batch_first=True,
                                      bidirectional=True, num_layers=num_layers)
            self.dropout_after_embedding = nn.Dropout(0.5)
            fc_input_size = 400
        elif model_name == 'End-to-end-CNN':
            ngram_filter_sizes = (3, 4, 5)
            num_filters = 300
            self.cnn = VectorCnnEncoder(word_embedding_dim, num_filters, ngram_filter_sizes=ngram_filter_sizes)
            self.dropout_after_embedding = nn.Dropout(0)
            fc_input_size = num_filters * len(ngram_filter_sizes)
        else:
            raise NotImplementedError(model_name)

        self.dropout_after_lstm = nn.Dropout(0.5)

        self.fcs = [nn.Linear(fc_input_size, self.polarity_num + 1) for _ in range(self.category_num)]
        self.fcs = nn.ModuleList(self.fcs)

    def forward(self, tokens: Dict[str, torch.Tensor], label: torch.Tensor, position: torch.Tensor,
                polarity_mask: torch.Tensor, sample: list, aspects: torch.Tensor=None) -> torch.Tensor:
        mask = get_text_field_mask(tokens)
        word_embeddings = self.word_embedder(tokens)

        model_name = self.configuration['model_name']
        if model_name == 'End-to-end-LSTM':
            word_embeddings = self.dropout_after_embedding(word_embeddings)
            h1, (h_n, c_n) = self.lstm(word_embeddings)
            v = torch.cat([h_n[0], h_n[1]], dim=-1)
        elif model_name == 'End-to-end-CNN':
            v = self.cnn(word_embeddings, mask)
        else:
            raise NotImplementedError(model_name)

        v = self.dropout_after_lstm(v)

        predicts_of_aspects = [self.fcs[i](v) for i in range(self.category_num)]

        output = {}
        if label is not None:
            merge_labeles = []
            for i in range(self.category_num):
                merge_labeles.append(label[:, i + self.category_num * 2])
            loss = 0
            for i in range(self.category_num):
                temp_loss = self.loss(predicts_of_aspects[i], merge_labeles[i].long())
                loss += temp_loss
            output['loss'] = loss

        output['merge_pred'] = predicts_of_aspects
        # [aspect, negative, neutral, positive]
        sentiment_pred = [torch.softmax(predicts_of_aspect[:, 1:], dim=-1) for predicts_of_aspect in
                          predicts_of_aspects]
        aspect_pred = [(torch.argmax(predicts_of_aspect, dim=-1, keepdim=True) != 0).float() for predicts_of_aspect
                       in predicts_of_aspects]
        output['pred_sentiment'] = sentiment_pred
        output['pred_category'] = aspect_pred
        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {
        }
        return metrics