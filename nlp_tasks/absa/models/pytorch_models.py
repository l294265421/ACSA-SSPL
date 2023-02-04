# -*- coding: utf-8 -*-


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
import dgl.function as fn
import dgl
from dgl.nn.pytorch import edge_softmax, GATConv, RelGraphConv

from nlp_tasks.absa.models import pytorch_attention


class DynamicLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=True, dropout=0,
                 bidirectional=False, only_use_last_hidden_state=False, rnn_type='LSTM'):
        """
        LSTM which can hold variable length sequence, use like TensorFlow's RNN(input, length...).

        :param input_size:The number of expected features in the input x
        :param hidden_size:The number of features in the hidden state h
        :param num_layers:Number of recurrent layers.
        :param bias:If False, then the layer does not use bias weights b_ih and b_hh. Default: True
        :param batch_first:If True, then the input and output tensors are provided as (batch, seq, feature)
        :param dropout:If non-zero, introduces a dropout layer on the outputs of each RNN layer except the last layer
        :param bidirectional:If True, becomes a bidirectional RNN. Default: False
        :param rnn_type: {LSTM, GRU, RNN}
        """
        super(DynamicLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.only_use_last_hidden_state = only_use_last_hidden_state
        self.rnn_type = rnn_type

        if self.rnn_type == 'LSTM':
            self.RNN = nn.LSTM(
                input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
        elif self.rnn_type == 'GRU':
            self.RNN = nn.GRU(
                input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
        elif self.rnn_type == 'RNN':
            self.RNN = nn.RNN(
                input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)

    def forward(self, x, x_len, h0=None):
        """
        sequence -> sort -> pad and pack ->process using RNN -> unpack ->unsort

        :param x: sequence embedding vectors
        :param x_len: numpy/tensor list
        :return:
        """
        """sort"""
        x_sort_idx = torch.argsort(-x_len)
        x_unsort_idx = torch.argsort(x_sort_idx).long()
        x_len = x_len[x_sort_idx]
        x = x[x_sort_idx.long()]
        """pack"""
        x_emb_p = torch.nn.utils.rnn.pack_padded_sequence(x, x_len, batch_first=self.batch_first)

        # process using the selected RNN
        if self.rnn_type == 'LSTM':
            if h0 is None:
                out_pack, (ht, ct) = self.RNN(x_emb_p, None)
            else:
                out_pack, (ht, ct) = self.RNN(x_emb_p, (h0, h0))
        else:
            if h0 is None:
                out_pack, ht = self.RNN(x_emb_p, None)
            else:
                out_pack, ht = self.RNN(x_emb_p, h0)
            ct = None
        """unsort: h"""
        ht = torch.transpose(ht, 0, 1)[
            x_unsort_idx]  # (num_layers * num_directions, batch, hidden_size) -> (batch, ...)
        ht = torch.transpose(ht, 0, 1)

        if self.only_use_last_hidden_state:
            return ht
        else:
            """unpack: out"""
            out = torch.nn.utils.rnn.pad_packed_sequence(out_pack, batch_first=self.batch_first)  # (sequence, lengths)
            out = out[0]  #
            out = out[x_unsort_idx]
            """unsort: out c"""
            if self.rnn_type == 'LSTM':
                ct = torch.transpose(ct, 0, 1)[
                    x_unsort_idx]  # (num_layers * num_directions, batch, hidden_size) -> (batch, ...)
                ct = torch.transpose(ct, 0, 1)

            return out, (ht, ct)


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, text, adj):
        hidden = torch.matmul(text, self.weight)
        denom = torch.sum(adj, dim=2, keepdim=True) + 1
        output = torch.matmul(adj, hidden) / denom
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class ASGCN(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(ASGCN, self).__init__()
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.text_lstm = DynamicLSTM(opt.embed_size, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.gc1 = GraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim)
        self.gc2 = GraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim)
        self.fc = nn.Linear(2*opt.hidden_dim, opt.polarities_dim)
        self.text_embed_dropout = nn.Dropout(0.3)

    def position_weight(self, x, aspect_double_idx, text_len, aspect_len):
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        aspect_double_idx = aspect_double_idx.cpu().numpy()
        text_len = text_len.cpu().numpy()
        aspect_len = aspect_len.cpu().numpy()
        weight = [[] for i in range(batch_size)]
        for i in range(batch_size):
            context_len = text_len[i] - aspect_len[i]
            for j in range(aspect_double_idx[i,0]):
                weight[i].append(1-(aspect_double_idx[i,0]-j)/context_len)
            for j in range(aspect_double_idx[i,0], aspect_double_idx[i,1]+1):
                weight[i].append(0)
            for j in range(aspect_double_idx[i,1]+1, text_len[i]):
                weight[i].append(1-(j-aspect_double_idx[i,1])/context_len)
            for j in range(text_len[i], seq_len):
                weight[i].append(0)
        weight = torch.tensor(weight).unsqueeze(2).to(self.opt.device)
        return weight*x

    def mask(self, x, aspect_double_idx):
        batch_size, seq_len = x.shape[0], x.shape[1]
        aspect_double_idx = aspect_double_idx.cpu().numpy()
        mask = [[] for i in range(batch_size)]
        for i in range(batch_size):
            for j in range(aspect_double_idx[i,0]):
                mask[i].append(0)
            for j in range(aspect_double_idx[i,0], aspect_double_idx[i,1]+1):
                mask[i].append(1)
            for j in range(aspect_double_idx[i,1]+1, seq_len):
                mask[i].append(0)
        mask = torch.tensor(mask).unsqueeze(2).float().to(self.opt.device)
        return mask*x

    def forward(self, inputs):
        text_indices, aspect_indices, left_indices, adj = inputs
        text_len = torch.sum(text_indices != 0, dim=-1)
        aspect_len = torch.sum(aspect_indices != 0, dim=-1)
        left_len = torch.sum(left_indices != 0, dim=-1)
        aspect_double_idx = torch.cat([left_len.unsqueeze(1), (left_len+aspect_len-1).unsqueeze(1)], dim=1)
        text = self.embed(text_indices)
        text = self.text_embed_dropout(text)
        text_out, (_, _) = self.text_lstm(text, text_len)
        x = F.relu(self.gc1(self.position_weight(text_out, aspect_double_idx, text_len, aspect_len), adj))
        x = F.relu(self.gc2(self.position_weight(x, aspect_double_idx, text_len, aspect_len), adj))
        x = self.mask(x, aspect_double_idx)
        alpha_mat = torch.matmul(x, text_out.transpose(1, 2))
        alpha = F.softmax(alpha_mat.sum(1, keepdim=True), dim=2)
        x = torch.matmul(alpha, text_out).squeeze(1) # batch_size x 2*hidden_dim
        output = self.fc(x)
        return output


class DglGraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, opt, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.opt = opt
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

    def collate(self, graphs):
        batched_graph = dgl.batch(graphs)
        return batched_graph

    def reduce(self, nodes):
        """Take an average over all neighbor node features hu and use it to
        overwrite the original node feature."""
        m = nodes.mailbox['m']
        accum = torch.sum(m, 1)
        return {'h': accum}

    def forward(self, text, graphs):
        hidden = torch.matmul(text, self.weight)
        batched_graph = dgl.batch(graphs)
        feature = hidden.view([-1, hidden.size()[-1]])
        if feature.size()[0] != batched_graph.number_of_nodes():
            print('error')
        batched_graph.ndata['h'] = feature
        msg = fn.copy_src(src='h', out='m')
        batched_graph.update_all(msg, self.reduce)
        ug = dgl.unbatch(batched_graph)
        output = [torch.unsqueeze(g.ndata.pop('h'), 0).to(self.opt.device) / (g.in_degrees().view(-1, 1).float().to(self.opt.device) + 1) for g in ug]
        output = torch.cat(output, 0)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class DglASGCN(ASGCN):
    def __init__(self, embedding_matrix, opt):
        super().__init__(embedding_matrix, opt)
        self.gc1 = DglGraphConvolution(2 * opt.hidden_dim, 2 * opt.hidden_dim, opt)
        self.gc2 = DglGraphConvolution(2 * opt.hidden_dim, 2 * opt.hidden_dim, opt)

    def forward(self, inputs):
        text_indices, aspect_indices, left_indices, adj = inputs
        text_len = torch.sum(text_indices != 0, dim=-1)
        aspect_len = torch.sum(aspect_indices != 0, dim=-1)
        left_len = torch.sum(left_indices != 0, dim=-1)
        aspect_double_idx = torch.cat([left_len.unsqueeze(1), (left_len+aspect_len-1).unsqueeze(1)], dim=1)
        text = self.embed(text_indices)
        text = self.text_embed_dropout(text)
        text_out, (_, _) = self.text_lstm(text, text_len)
        x = F.relu(self.gc1(self.position_weight(text_out, aspect_double_idx, text_len, aspect_len), adj))
        x = F.relu(self.gc2(self.position_weight(x, aspect_double_idx, text_len, aspect_len), adj))
        x = self.mask(x, aspect_double_idx)
        alpha_mat = torch.matmul(x, text_out.transpose(1, 2))
        alpha = F.softmax(alpha_mat.sum(1, keepdim=True), dim=2)
        x = torch.matmul(alpha, text_out).squeeze(1) # batch_size x 2*hidden_dim
        output = self.fc(x)
        return output


class DglMaskASGCN(ASGCN):
    def __init__(self, embedding_matrix, opt):
        super().__init__(embedding_matrix, opt)
        self.gc1 = DglGraphConvolution(2 * opt.hidden_dim, 2 * opt.hidden_dim, opt)
        self.gc2 = DglGraphConvolution(2 * opt.hidden_dim, 2 * opt.hidden_dim, opt)

        # attention weight
        hidden_size = opt.hidden_dim
        self.word_weight = nn.Parameter(torch.Tensor(2 * hidden_size, 2 * hidden_size))
        self.word_bias = nn.Parameter(torch.Tensor(1, 2 * hidden_size))
        self.context_weight = nn.Parameter(torch.Tensor(2 * hidden_size, 1))
        self._create_weights(mean=0.0, std=0.05)

    def _create_weights(self, mean=0.0, std=0.05):
        self.word_weight.data.normal_(mean, std)
        self.context_weight.data.normal_(mean, std)

    def matrix_mul(self, input, weight, bias=False):
        feature_list = []
        for feature in input:
            feature = torch.mm(feature, weight)
            if isinstance(bias, torch.nn.parameter.Parameter):
                feature = feature + bias.expand(feature.size()[0], bias.size()[1])
            feature = torch.tanh(feature).unsqueeze(0)
            feature_list.append(feature)

        return torch.cat(feature_list, 0).squeeze()

    def element_wise_mul(self, input1, input2):
        feature_list = []
        for feature_1, feature_2 in zip(input1, input2):
            feature_2 = feature_2.unsqueeze(1)
            feature_2 = feature_2.expand_as(feature_1)
            feature = feature_1 * feature_2
            feature = feature.unsqueeze(0)
            feature_list.append(feature)
        output = torch.cat(feature_list, 0)

        result = torch.sum(output, 1)
        return result

    def forward(self, inputs):
        text_indices, aspect_indices, left_indices,   adj = inputs
        text_len = torch.sum(text_indices != 0, dim=-1)
        aspect_len = torch.sum(aspect_indices != 0, dim=-1)
        left_len = torch.sum(left_indices != 0, dim=-1)
        aspect_double_idx = torch.cat([left_len.unsqueeze(1), (left_len+aspect_len-1).unsqueeze(1)], dim=1)
        text = self.embed(text_indices)
        text = self.text_embed_dropout(text)
        text_out, (_, _) = self.text_lstm(text, text_len)
        x = F.relu(self.gc1(self.position_weight(text_out, aspect_double_idx, text_len, aspect_len), adj))
        x = F.relu(self.gc2(self.position_weight(x, aspect_double_idx, text_len, aspect_len), adj))
        x = self.mask(x, aspect_double_idx)

        # 在x上应用attention，得到aspect表示用于情感分类
        alpha_mat = self.matrix_mul(x, self.context_weight)
        alpha = F.softmax(alpha_mat)
        output = self.element_wise_mul(x, alpha) # batch_size x 2*hidden_dim

        output = self.fc(output)
        return output


class DglASGCNModelAspectDependency(ASGCN):
    def __init__(self, embedding_matrix, opt):
        super().__init__(embedding_matrix, opt)
        self.gc1 = DglGraphConvolution(2 * opt.hidden_dim, 2 * opt.hidden_dim, opt)
        self.gc2 = DglGraphConvolution(2 * opt.hidden_dim, 2 * opt.hidden_dim, opt)

        # attention weight
        hidden_size = opt.hidden_dim
        self.word_weight = nn.Parameter(torch.Tensor(2 * hidden_size, 2 * hidden_size))
        self.word_bias = nn.Parameter(torch.Tensor(1, 2 * hidden_size))
        self.context_weight = nn.Parameter(torch.Tensor(2 * hidden_size, 1))
        self._create_weights(mean=0.0, std=0.05)

    def _create_weights(self, mean=0.0, std=0.05):
        self.word_weight.data.normal_(mean, std)
        self.context_weight.data.normal_(mean, std)

    def matrix_mul(self, input, weight, bias=False):
        feature_list = []
        for feature in input:
            feature = torch.mm(feature, weight)
            if isinstance(bias, torch.nn.parameter.Parameter):
                feature = feature + bias.expand(feature.size()[0], bias.size()[1])
            feature = torch.tanh(feature).unsqueeze(0)
            feature_list.append(feature)

        return torch.cat(feature_list, 0).squeeze()

    def element_wise_mul(self, input1, input2):
        feature_list = []
        for feature_1, feature_2 in zip(input1, input2):
            feature_2 = feature_2.unsqueeze(1)
            feature_2 = feature_2.expand_as(feature_1)
            feature = feature_1 * feature_2
            feature = feature.unsqueeze(0)
            feature_list.append(feature)
        output = torch.cat(feature_list, 0)

        result = torch.sum(output, 1)
        return result

    def forward(self, inputs):
        text_indices, adj, aspert_term_indices = inputs
        text_len = torch.sum(text_indices != 0, dim=-1)
        text = self.embed(text_indices)
        text = self.text_embed_dropout(text)
        text_out, (_, _) = self.text_lstm(text, text_len)
        x = F.relu(self.gc1(text_out, adj))
        x = F.relu(self.gc2(x, adj))

        outputs = []
        for i in range(aspert_term_indices.size()[1]):
            aspert_term_index = aspert_term_indices[:, i]
            x_aspect = self.mask(x, aspert_term_index)

            # 在x上应用attention，得到aspect表示用于情感分类
            alpha_mat = self.matrix_mul(x_aspect, self.context_weight)
            alpha = F.softmax(alpha_mat)
            output = self.element_wise_mul(x_aspect, alpha)  # batch_size x 2*hidden_dim

            output = self.fc(output)
            outputs.append(output)

        return outputs


class DglASGCNModelAspectDependency2(ASGCN):
    def __init__(self, embedding_matrix, opt):
        super().__init__(embedding_matrix, opt)
        self.gc1 = DglGraphConvolution(2 * opt.hidden_dim, 2 * opt.hidden_dim, opt)
        self.gc2 = DglGraphConvolution(2 * opt.hidden_dim, 2 * opt.hidden_dim, opt)
        self.gc3 = DglGraphConvolution(2 * opt.hidden_dim, 2 * opt.hidden_dim, opt)

        # attention weight
        hidden_size = opt.hidden_dim
        self.word_weight = nn.Parameter(torch.Tensor(2 * hidden_size, 2 * hidden_size))
        self.word_bias = nn.Parameter(torch.Tensor(1, 2 * hidden_size))
        self.context_weight = nn.Parameter(torch.Tensor(2 * hidden_size, 1))
        self._create_weights(mean=0.0, std=0.05)

    def _create_weights(self, mean=0.0, std=0.05):
        self.word_weight.data.normal_(mean, std)
        self.context_weight.data.normal_(mean, std)

    def matrix_mul(self, input, weight, bias=False):
        feature_list = []
        for feature in input:
            feature = torch.mm(feature, weight)
            if isinstance(bias, torch.nn.parameter.Parameter):
                feature = feature + bias.expand(feature.size()[0], bias.size()[1])
            feature = torch.tanh(feature).unsqueeze(0)
            feature_list.append(feature)

        return torch.cat(feature_list, 0).squeeze()

    def element_wise_mul(self, input1, input2):
        feature_list = []
        for feature_1, feature_2 in zip(input1, input2):
            feature_2 = feature_2.unsqueeze(1)
            feature_2 = feature_2.expand_as(feature_1)
            feature = feature_1 * feature_2
            feature = feature.unsqueeze(0)
            feature_list.append(feature)
        output = torch.cat(feature_list, 0)

        result = torch.sum(output, 1)
        return result

    def forward(self, inputs):
        text_indices, adj, aspert_term_indices, aspect_term_graph = inputs
        text_len = torch.sum(text_indices != 0, dim=-1)
        text = self.embed(text_indices)
        text = self.text_embed_dropout(text)
        text_out, (_, _) = self.text_lstm(text, text_len)
        x = F.relu(self.gc1(text_out, adj))
        x = F.relu(self.gc2(x, adj))
        # x = F.relu(self.gc3(x, aspect_term_graph))

        outputs = []
        for i in range(aspert_term_indices.size()[1]):
            aspert_term_index = aspert_term_indices[:, i]
            x_aspect = self.mask(x, aspert_term_index)

            # 在x上应用attention，得到aspect表示用于情感分类
            alpha_mat = self.matrix_mul(x_aspect, self.context_weight)
            alpha = F.softmax(alpha_mat)
            output = self.element_wise_mul(x_aspect, alpha)  # batch_size x 2*hidden_dim

            output = self.fc(output)
            outputs.append(output)

        return outputs


class DglASGCNModelAspectDependency3(ASGCN):
    def __init__(self, embedding_matrix, opt):
        super().__init__(embedding_matrix, opt)
        self.gc1 = DglRelationGraphConvolutionNetwork(2 * opt.hidden_dim, 2 * opt.hidden_dim, opt)
        self.gc2 = DglRelationGraphConvolutionNetwork(2 * opt.hidden_dim, 2 * opt.hidden_dim, opt)

        # attention weight
        hidden_size = opt.hidden_dim
        self.word_weight = nn.Parameter(torch.Tensor(2 * hidden_size, 2 * hidden_size))
        self.word_bias = nn.Parameter(torch.Tensor(1, 2 * hidden_size))
        self.context_weight = nn.Parameter(torch.Tensor(2 * hidden_size, 1))
        self._create_weights(mean=0.0, std=0.05)

    def _create_weights(self, mean=0.0, std=0.05):
        self.word_weight.data.normal_(mean, std)
        self.context_weight.data.normal_(mean, std)

    def matrix_mul(self, input, weight, bias=False):
        feature_list = []
        for feature in input:
            feature = torch.mm(feature, weight)
            if isinstance(bias, torch.nn.parameter.Parameter):
                feature = feature + bias.expand(feature.size()[0], bias.size()[1])
            feature = torch.tanh(feature).unsqueeze(0)
            feature_list.append(feature)

        return torch.cat(feature_list, 0).squeeze()

    def element_wise_mul(self, input1, input2):
        feature_list = []
        for feature_1, feature_2 in zip(input1, input2):
            feature_2 = feature_2.unsqueeze(1)
            feature_2 = feature_2.expand_as(feature_1)
            feature = feature_1 * feature_2
            feature = feature.unsqueeze(0)
            feature_list.append(feature)
        output = torch.cat(feature_list, 0)

        result = torch.sum(output, 1)
        return result

    def forward(self, inputs):
        text_indices, adj, aspert_term_indices, aspect_term_graph = inputs
        text_len = torch.sum(text_indices != 0, dim=-1)
        text = self.embed(text_indices)
        text = self.text_embed_dropout(text)
        text_out, (_, _) = self.text_lstm(text, text_len)
        x = F.relu(self.gc1(text_out, adj))
        x = F.relu(self.gc2(x, adj))
        # x = F.relu(self.gc3(x, aspect_term_graph))

        outputs = []
        for i in range(aspert_term_indices.size()[1]):
            aspert_term_index = aspert_term_indices[:, i]
            x_aspect = self.mask(x, aspert_term_index)

            # 在x上应用attention，得到aspect表示用于情感分类
            alpha_mat = self.matrix_mul(x_aspect, self.context_weight)
            alpha = F.softmax(alpha_mat)
            output = self.element_wise_mul(x_aspect, alpha)  # batch_size x 2*hidden_dim

            output = self.fc(output)
            outputs.append(output)

        return outputs


class GAT(nn.Module):
    def __init__(self,
                 g,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_classes,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual):
        super(GAT, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        # input projection (no residual)
        self.gat_layers.append(GATConv(
            in_dim, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                num_hidden * heads[l-1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation))
        # output projection
        self.gat_layers.append(GATConv(
            num_hidden * heads[-2], num_classes, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None))

    def forward(self, inputs):
        h = inputs
        for l in range(self.num_layers):
            h = self.gat_layers[l](self.g, h).flatten(1)
        # output projection
        logits = self.gat_layers[-1](self.g, h).mean(1)
        return logits


class DglGraphAttentionNetwork(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        # self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        num_heads = 4
        self.out_features = int((in_features / num_heads))
        self.gat = GATConv(in_features, self.out_features, num_heads)
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

    def collate(self, graphs):
        batched_graph = dgl.batch(graphs)
        return batched_graph

    def reduce(self, nodes):
        """Take an average over all neighbor node features hu and use it to
        overwrite the original node feature."""
        m = nodes.mailbox['m']
        accum = torch.sum(m, 1)
        return {'h': accum}

    def forward(self, text, graphs):
        hidden = torch.matmul(text, self.weight)
        batched_graph = dgl.batch(graphs)
        hidden_size = hidden.size()
        feature = hidden.view([-1, hidden_size[-1]])

        result = self.gat(batched_graph, feature)
        output = result.view([result.size()[0], -1])
        output_size = output.size()
        output = output.view([hidden_size[0], -1, output_size[-1]])
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class DglASGAT(DglASGCN):
    def __init__(self, embedding_matrix, opt):
        super().__init__(embedding_matrix, opt)
        self.gc1 = DglGraphAttentionNetwork(2 * opt.hidden_dim, 2 * opt.hidden_dim)
        self.gc2 = DglGraphAttentionNetwork(2 * opt.hidden_dim, 2 * opt.hidden_dim)

    def forward(self, inputs):
        text_indices, aspect_indices, left_indices, adj = inputs
        text_len = torch.sum(text_indices != 0, dim=-1)
        aspect_len = torch.sum(aspect_indices != 0, dim=-1)
        left_len = torch.sum(left_indices != 0, dim=-1)
        aspect_double_idx = torch.cat([left_len.unsqueeze(1), (left_len+aspect_len-1).unsqueeze(1)], dim=1)
        text = self.embed(text_indices)
        text = self.text_embed_dropout(text)
        text_out, (_, _) = self.text_lstm(text, text_len)
        x = F.relu(self.gc1(self.position_weight(text_out, aspect_double_idx, text_len, aspect_len), adj))
        x = F.relu(self.gc2(self.position_weight(x, aspect_double_idx, text_len, aspect_len), adj))
        x = self.mask(x, aspect_double_idx)
        alpha_mat = torch.matmul(x, text_out.transpose(1, 2))
        alpha = F.softmax(alpha_mat.sum(1, keepdim=True), dim=2)
        x = torch.matmul(alpha, text_out).squeeze(1) # batch_size x 2*hidden_dim
        output = self.fc(x)
        return output


class DglRelationGraphConvolutionNetwork(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, opt, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.opt = opt
        self.rgcn = RelGraphConv(in_features, out_features, opt.num_rels, self_loop=False, activation=F.relu,
                                 regularizer='basis', num_bases=3)

    def collate(self, graphs):
        batched_graph = dgl.batch(graphs)
        return batched_graph

    def reduce(self, nodes):
        """Take an average over all neighbor node features hu and use it to
        overwrite the original node feature."""
        m = nodes.mailbox['m']
        accum = torch.sum(m, 1)
        return {'h': accum}

    def forward(self, text, graphs):
        rel_type = [torch.tensor(g.edata.get('rel_type')) for g in graphs]
        rel_type_all = torch.cat(rel_type).to(self.opt.device)
        hidden = text
        batched_graph = dgl.batch(graphs)
        hidden_size = hidden.size()
        feature = hidden.view([-1, hidden_size[-1]]).to(self.opt.device)
        result = self.rgcn(batched_graph, feature, rel_type_all, norm=None)
        output = result.view([result.size()[0], -1])
        output_size = output.size()
        output = output.view([hidden_size[0], -1, output_size[-1]])
        return output


class DglASRGCN(DglASGCN):
    def __init__(self, embedding_matrix, opt):
        super().__init__(embedding_matrix, opt)
        self.gc1 = DglRelationGraphConvolutionNetwork(2 * opt.hidden_dim, 2 * opt.hidden_dim)
        self.gc2 = DglRelationGraphConvolutionNetwork(2 * opt.hidden_dim, 2 * opt.hidden_dim)

    def forward(self, inputs):
        text_indices, aspect_indices, left_indices, adj = inputs
        text_len = torch.sum(text_indices != 0, dim=-1)
        aspect_len = torch.sum(aspect_indices != 0, dim=-1)
        left_len = torch.sum(left_indices != 0, dim=-1)
        aspect_double_idx = torch.cat([left_len.unsqueeze(1), (left_len+aspect_len-1).unsqueeze(1)], dim=1)
        text = self.embed(text_indices)
        text = self.text_embed_dropout(text)
        text_out, (_, _) = self.text_lstm(text, text_len)
        x = F.relu(self.gc1(self.position_weight(text_out, aspect_double_idx, text_len, aspect_len), adj))
        x = F.relu(self.gc2(self.position_weight(x, aspect_double_idx, text_len, aspect_len), adj))
        x = self.mask(x, aspect_double_idx)
        alpha_mat = torch.matmul(x, text_out.transpose(1, 2))
        alpha = F.softmax(alpha_mat.sum(1, keepdim=True), dim=2)
        x = torch.matmul(alpha, text_out).squeeze(1) # batch_size x 2*hidden_dim
        output = self.fc(x)
        return output


class ThreeStagesModel(nn.Module):
    """

    """
    pass
