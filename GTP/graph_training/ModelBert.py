'''
Concrete MethodModule class for a specific learning MethodModule
'''


import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertPooler
from GTP.graph_training.ModelComponent import BertEmbeddings, BertEncoder, VanBertEmbeddings, PairBertEmbeddings


BertLayerNorm = torch.nn.LayerNorm
class MethodLDABert(BertPreTrainedModel):
    data = None

    def __init__(self, config):
        super(MethodLDABert, self).__init__(config)
        self.config = config

        self.embeddings = PairBertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    # def get_input_embeddings(self):
    #     return self.embeddings.raw_feature_embeddings
    #
    # def set_input_embeddings(self, value):
    #     self.embeddings.raw_feature_embeddings = value

    def _prune_heads(self, heads_to_prune):
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def setting_preparation(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None,
                            head_mask=None, inputs_embeds=None, encoder_hidden_states=None, encoder_attention_mask=None,
                            graph_embedding=None):

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if graph_embedding is not None:
            graph_shape = graph_embedding.shape
            # print(graph_shape)
            graph_attention_mask = torch.ones((graph_shape[0], graph_shape[1]), dtype=torch.long, device=input_ids.device)
            attention_mask = torch.cat((attention_mask, graph_attention_mask), dim=1)
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            if self.config.is_decoder:
                batch_size, seq_length = input_shape
                seq_ids = torch.arange(seq_length, device=device)
                causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
                causal_mask = causal_mask.to(torch.long)  # not converting to long will cause errors with pytorch version < 1.3
                extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
                    input_shape, attention_mask.shape
                )
            )

        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)

            if encoder_attention_mask.dim() == 3:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
            elif encoder_attention_mask.dim() == 2:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
            else:
                raise ValueError(
                    "Wrong shape for encoder_hidden_shape (shape {}) or encoder_attention_mask (shape {})".format(
                        encoder_hidden_shape, encoder_attention_mask.shape
                    )
                )

            encoder_extended_attention_mask = encoder_extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
            encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -10000.0
        else:
            encoder_extended_attention_mask = None

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = (
                    head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                )  # We can specify head_mask for each layer
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype)
        else:
            head_mask = [None] * self.config.num_hidden_layers

        return token_type_ids, extended_attention_mask, encoder_extended_attention_mask, head_mask


    def forward(self, input_ids=None, token_type_ids=None, position_ids=None,attention_mask=None, inputs_embeds=None,
                head_mask=None, labels=None, graph_embedding=None, LDA_weight=None):

        token_type_ids, extended_attention_mask, encoder_extended_attention_mask, head_mask = self.setting_preparation(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            graph_embedding=graph_embedding[0]

        )

        if head_mask is None:
            head_mask = [None] * self.config.num_hidden_layers
        # print('attention mask shape:',extended_attention_mask.shape) # 16,1,1,136
        embedding_output = self.embeddings(input_ids=input_ids, token_type_ids=token_type_ids,position_ids=position_ids,
                                           graph_embedding=graph_embedding)# 16,136,768
        # print('embedding_output.shape ',embedding_output.shape)
        encoder_outputs = self.encoder(embedding_output, head_mask=head_mask, residual_h=None, LDA_weight=LDA_weight, attention_mask=extended_attention_mask,
                                       encoder_attention_mask=encoder_extended_attention_mask)
        # print(encoder_outputs)
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        # outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]
        return logits

    def run(self):
        pass



class MethodVanBert(BertPreTrainedModel):
    data = None

    def __init__(self, config):
        super(MethodVanBert, self).__init__(config)
        self.config = config

        self.embeddings = VanBertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.num_labels = self.config.num_labels
        self.init_weights()

    # def get_input_embeddings(self):
    #     return self.embeddings.raw_feature_embeddings
    #
    # def set_input_embeddings(self, value):
    #     self.embeddings.raw_feature_embeddings = value

    def _prune_heads(self, heads_to_prune):
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def setting_preparation(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, encoder_hidden_states=None, encoder_attention_mask=None,):

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            if self.config.is_decoder:
                batch_size, seq_length = input_shape
                seq_ids = torch.arange(seq_length, device=device)
                causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
                causal_mask = causal_mask.to(torch.long)  # not converting to long will cause errors with pytorch version < 1.3
                extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
                    input_shape, attention_mask.shape
                )
            )

        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)

            if encoder_attention_mask.dim() == 3:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
            elif encoder_attention_mask.dim() == 2:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
            else:
                raise ValueError(
                    "Wrong shape for encoder_hidden_shape (shape {}) or encoder_attention_mask (shape {})".format(
                        encoder_hidden_shape, encoder_attention_mask.shape
                    )
                )

            encoder_extended_attention_mask = encoder_extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
            encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -10000.0
        else:
            encoder_extended_attention_mask = None

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = (
                    head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                )  # We can specify head_mask for each layer
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype)
        else:
            head_mask = [None] * self.config.num_hidden_layers

        return token_type_ids, extended_attention_mask, encoder_extended_attention_mask, head_mask


    def forward(self, input_ids=None, token_type_ids=None, position_ids=None,attention_mask=None, inputs_embeds=None, head_mask=None, labels=None):

        token_type_ids, extended_attention_mask, encoder_extended_attention_mask, head_mask = self.setting_preparation(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,

        )


        if head_mask is None:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(input_ids=input_ids, token_type_ids=token_type_ids, position_ids=position_ids)
        encoder_outputs = self.encoder(embedding_output, head_mask=head_mask, residual_h=None, LDA_weight=None, attention_mask=extended_attention_mask,
                                       encoder_attention_mask=encoder_extended_attention_mask)
        # print(encoder_outputs)
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        # outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]

        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))


        return {
            'logits':logits,
            'loss':loss
        }

    def run(self):
        pass