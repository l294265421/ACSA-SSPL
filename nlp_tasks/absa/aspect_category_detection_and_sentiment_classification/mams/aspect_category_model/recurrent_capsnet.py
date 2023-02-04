import torch
from torch import nn
import torch.nn.functional as F
from nlp_tasks.absa.aspect_category_detection_and_sentiment_classification.mams.aspect_category_model.capsnet import CapsuleNetwork


class RecurrentCapsuleNetwork(CapsuleNetwork):

    def __init__(self, embedding, aspect_embedding, num_layers, bidirectional, capsule_size, dropout, num_categories,
                 configuration=None):
        super(RecurrentCapsuleNetwork, self).__init__(
            embedding=embedding,
            aspect_embedding=aspect_embedding,
            hidden_size=embedding.embedding_dim * (2 if bidirectional else 1),
            capsule_size=capsule_size,
            dropout=dropout,
            num_categories=num_categories,
            configuration=configuration
        )
        embed_size = embedding.embedding_dim
        if 'Cav' in self.configuration['model_name']:
            rnn_input_size = embed_size * 3
        elif 'Cam' in self.configuration['model_name']:
            rnn_input_size = embed_size * 3
        else:
            rnn_input_size = embed_size * 2
        self.rnn = nn.GRU(
            input_size=rnn_input_size,
            hidden_size=embed_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True
        )
        self.bidirectional = bidirectional

    def _sentence_encode(self, sentence, aspect, mask=None, car=None):
        batch_size, time_step, embed_size = sentence.size()
        if 'Cav' in self.configuration['model_name']:
            aspect_aware_sentence = torch.cat((
                sentence, car[0].unsqueeze(1).expand(batch_size, time_step, embed_size * 2)
            ), dim=-1)
        elif 'Cam' in self.configuration['model_name']:
            aspect_aware_sentence = torch.cat((
                sentence, car[1]
            ), dim=-1)
        else:
            aspect_aware_sentence = torch.cat((
                sentence, aspect.unsqueeze(1).expand(batch_size, time_step, embed_size)
            ), dim=-1)
        output, _ = self.rnn(aspect_aware_sentence)
        if self.bidirectional:
            sentence = sentence.unsqueeze(-1).expand(batch_size, time_step, embed_size, 2)
            sentence = sentence.contiguous().view(batch_size, time_step, embed_size * 2)
        output = output + sentence
        output = F.dropout(output, p=self.dropout, training=self.training)
        return output