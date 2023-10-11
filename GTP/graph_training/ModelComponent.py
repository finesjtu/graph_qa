'''
Concrete MethodModule class for a specific learning MethodModule
'''

import math
import torch
import torch.nn as nn
# from transformers.modeling_bert import BertPredictionHeadTransform, BertAttention, BertIntermediate, BertOutput
from transformers.models.bert.modeling_bert import BertPredictionHeadTransform, BertSelfOutput, BertIntermediate, BertOutput, BertConfig
from transformers.configuration_utils import PretrainedConfig

from typing import List, Optional, Tuple, Union, Set
BertLayerNorm = torch.nn.LayerNorm

def find_pruneable_heads_and_indices(
    heads: List[int], n_heads: int, head_size: int, already_pruned_heads: Set[int]
) -> Tuple[Set[int], torch.LongTensor]:
    """
    Finds the heads and their indices taking `already_pruned_heads` into account.

    Args:
        heads (`List[int]`): List of the indices of heads to prune.
        n_heads (`int`): The number of heads in the model.
        head_size (`int`): The size of each head.
        already_pruned_heads (`Set[int]`): A set of already pruned heads.

    Returns:
        `Tuple[Set[int], torch.LongTensor]`: A tuple with the remaining heads and their corresponding indices.
    """
    mask = torch.ones(n_heads, head_size)
    heads = set(heads) - already_pruned_heads  # Convert to set and remove already pruned heads
    for head in heads:
        # Compute how many pruned heads are before the head and move the index accordingly
        head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
        mask[head] = 0
    mask = mask.view(-1).contiguous().eq(1)
    index: torch.LongTensor = torch.arange(len(mask))[mask].long()
    return heads, index

def prune_linear_layer(layer: nn.Linear, index: torch.LongTensor, dim: int = 0) -> nn.Linear:
    """
    Prune a linear layer to keep only entries in index.

    Used to remove heads.

    Args:
        layer (`torch.nn.Linear`): The layer to prune.
        index (`torch.LongTensor`): The indices to keep in the layer.
        dim (`int`, *optional*, defaults to 0): The dimension on which to keep the indices.

    Returns:
        `torch.nn.Linear`: The pruned layer as a new layer with `requires_grad=True`.
    """
    index = index.to(layer.weight.device)
    W = layer.weight.index_select(dim, index).clone().detach()
    if layer.bias is not None:
        if dim == 1:
            b = layer.bias.clone().detach()
        else:
            b = layer.bias[index].clone().detach()
    new_size = list(layer.weight.size())
    new_size[dim] = len(index)
    new_layer = nn.Linear(new_size[1], new_size[0], bias=layer.bias is not None).to(layer.weight.device)
    new_layer.weight.requires_grad = False
    new_layer.weight.copy_(W.contiguous())
    new_layer.weight.requires_grad = True
    if layer.bias is not None:
        new_layer.bias.requires_grad = False
        new_layer.bias.copy_(b.contiguous())
        new_layer.bias.requires_grad = True
    return new_layer

class GraphBertConfig(PretrainedConfig):

    def __init__(
        self,
        residual_type = 'none',
        x_size=3000,
        semantic_dim=1024,
        graph_seq_dim=512,
        graph_syn_dim=512,
        num_labels=7,
        k=5,
        graph_embedding=False,
        # max_wl_role_index = 1500,
        # max_hop_dis_index = 100,
        # max_inti_pos_index = 100,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=8,
        intermediate_size=768,
        hidden_act="gelu",
        hidden_dropout_prob=0.5,
        attention_probs_dropout_prob=0.3,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        is_decoder=False,

        **kwargs
    ):
        super(GraphBertConfig, self).__init__(**kwargs)
        self.semantic_dim = semantic_dim
        self.graph_seq_dim = graph_seq_dim
        self.graph_syn_dim = graph_syn_dim
        # self.max_wl_role_index = max_wl_role_index
        # self.max_hop_dis_index = max_hop_dis_index
        # self.max_inti_pos_index = max_inti_pos_index
        self.residual_type = residual_type
        self.x_size = x_size
        self.num_labels = num_labels
        self.graph_embedding = graph_embedding
        self.k = k
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.is_decoder = is_decoder

class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None,
                encoder_attention_mask=None, residual_h=None, LDA_weight=None):
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(hidden_states, attention_mask, head_mask[i], encoder_hidden_states,
                                         encoder_attention_mask, LDA_weight=LDA_weight)
            hidden_states = layer_outputs[0]

            #---- add residual ----
            if residual_h is not None:
                for index in range(hidden_states.size()[1]):
                    hidden_states[:,index,:] += residual_h

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)

class BertSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        # print(new_x_shape)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
        LDA_weight: Optional[torch.FloatTensor] = None,
    ) -> Tuple[torch.Tensor]:
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            # print(self.all_head_size)
            # print(self.key(hidden_states).shape)
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        # print(query_layer.shape)
        # print(key_layer.shape)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        # print(attention_scores.shape)
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key
        # print(attention_scores.shape)

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        if LDA_weight is None:


            attention_probs = nn.functional.softmax(attention_scores, dim=-1)# *W-LDA_weight
            # print('attention_scores shape:', attention_scores.shape) #16*16*136*136
        elif LDA_weight is not None:
            sequence_length = attention_scores.shape[-1]
            graph_length = LDA_weight.shape[-1]
            batch_size = LDA_weight.shape[0]
            # print(attention_scores.shape)
            # print('LDA shape:',LDA_weight.shape)

            # sequence_weight = torch.cat((torch.ones((batch_size, sequence_length-graph_length, sequence_length-graph_length)),
            #                              torch.ones((batch_size, sequence_length-graph_length, graph_length))), dim=2)
            # graph_weight = torch.cat((torch.ones((batch_size, graph_length, sequence_length - graph_length)),
            #                           LDA_weight), dim=2)
            sequence_weight = torch.cat(
                (torch.ones((batch_size, sequence_length - graph_length, sequence_length - graph_length)),
                 torch.rand((batch_size, sequence_length - graph_length, graph_length)) > 0.7), dim=2)
            # torch.rand((10, 10, 10)) > 0.3

            # 第一个 是 sequence weight 第二个是 seq-graph weight
            graph_weight = torch.cat((torch.rand((batch_size, graph_length, sequence_length - graph_length)) > 0.7,
                                      LDA_weight), dim=2)
            # 第一个是seq-graph weight
            LDA_weight = torch.cat((sequence_weight, graph_weight), dim=1)
            # print('sequence_weight:',sequence_weight.shape)
            # print('graph_weight:',graph_weight.shape   )
            shape = attention_scores.shape[1]
            LDA_weight = LDA_weight.repeat(shape,1,1,1).permute(1,0,2,3).to(attention_scores.device)
            # print('LDA shape reshape:', LDA_weight.shape)
            # print(LDA_weight.shape)
            # print(nn.functional.softmax(LDA_weight, dim=-1))
            attention_probs = nn.functional.softmax(attention_scores * LDA_weight, dim=-1)   # *W-LDA_weight

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)
        # print(value_layer.shape)
        # print(attention_probs.shape)
        # print(context_layer.shape)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        # print(new_context_layer_shape)
        # print(context_layer.shape)
        # print('=')
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


class BertAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        self.self = BertSelfAttention(config, position_embedding_type=position_embedding_type)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
        LDA_weight: Optional[torch.FloatTensor] = None,
    ) -> Tuple[torch.Tensor]:
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
            LDA_weight,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs

class VanBertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))


    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]
        # print(self.position_ids)
        # print(position_ids)
        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings

        # embeddings = inputs_embeds

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class PairBertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.graph_embeddings = nn.Linear(config.graph_embedding_dim, config.hidden_size)
        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))


    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0,
            graph_embedding=None
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]


        seq_length = input_shape[1]

        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if graph_embedding is not None:
            graph_length = graph_embedding[0].shape[1]
            token_type_ids_graph = torch.ones((input_shape[0], graph_length), device=input_ids.device, dtype=torch.long)
            token_type_ids = torch.cat((token_type_ids, token_type_ids_graph), dim=1)
            if len(graph_embedding) > 1:

                g = graph_embedding[0] + graph_embedding[1]
            else:
                g = graph_embedding[0]
            # print(g.shape)

            graph_embeddings = self.graph_embeddings(g)

        if position_ids is None:
            if graph_embedding is not None:
                position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length + graph_length]
                # print(position_ids.shape)
            else:
                position_ids = self.position_ids[:,
                               past_key_values_length: seq_length + past_key_values_length]
        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664


        # print(token_type_ids_graph)

        # print('token_type_ids.shape',token_type_ids.shape)# 16,136


        # print('graph_embeddings.shape',graph_embeddings.shape)# 16,8,768

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
            # print('inputs_embeds.shape',inputs_embeds.shape)# 16,128,768
        if graph_embedding is not None:
            inputs_embeds = torch.cat((inputs_embeds, graph_embeddings), dim=1)

        # print('inputs_embeds.shape',inputs_embeds.shape)# 16.136.768
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        # print('token_type_embeddings.shape',token_type_embeddings.shape)# 16,136,768



        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings

        # embeddings = inputs_embeds

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings # 16,136,768

class BertEmbeddings(nn.Module):
    """Construct the embeddings from features, wl, position and hop vectors.
    """

    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        # self.raw_feature_embeddings = nn.Linear(config.x_size, config.hidden_size)
        # self.wl_role_embeddings = nn.Embedding(config.max_wl_role_index, config.hidden_size)
        self.semantic_embedding = nn.Linear(config.semantic_dim, config.hidden_size)
        if config.graph_embedding:
            self.graph_seq_embedding = nn.Linear(config.graph_seq_dim, config.hidden_size)
            self.graph_syn_embedding = nn.Linear(config.graph_syn_dim, config.hidden_size)
        # self.inti_pos_embeddings = nn.Embedding(config.max_inti_pos_index, config.hidden_size)
        # self.hop_dis_embeddings = nn.Embedding(config.max_hop_dis_index, config.hidden_size)
        self.config = config
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, semantic_features=None, graph_seq_features=None, graph_syn_features=None):
        # print('='*20)
        # print('config size: ', self.config.x_size, self.config.max_wl_role_index)
        # print('input dimension: ', raw_features.shape, wl_role_ids.shape)
        #
        # print('layer: ', self.raw_feature_embeddings, self.wl_role_embeddings)
        semantic_embeddings = self.semantic_embedding(semantic_features)
        if self.config.graph_embedding:
            graph_seq_embeddings = self.graph_seq_embedding(graph_seq_features)
            graph_syn_embeddings = self.graph_syn_embedding(graph_syn_features)
            embeddings = semantic_embeddings + graph_seq_embeddings + graph_syn_embeddings
        else:
            embeddings = semantic_embeddings
        # role_embeddings = self.wl_role_embeddings(wl_role_ids)
        # print('output dimension: ', raw_feature_embeds.shape, role_embeddings.shape)
        # position_embeddings = self.inti_pos_embeddings(init_pos_ids)
        # hop_embeddings = self.hop_dis_embeddings(hop_dis_ids)

        #---- here, we use summation ----
        # for i in range(10):
        #     print('*'*100)
        #     print(raw_feature_embeds[i])
        #     print(role_embeddings[i])


        # + position_embeddings + hop_embeddings raw_feature_embeds +role_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        # print('output embedding: ', embeddings.shape)
        return embeddings



class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention(config)
        self.is_decoder = config.is_decoder
        if self.is_decoder:
            self.crossattention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        LDA_weight=None
    ):
        self_attention_outputs = self.attention(hidden_states, attention_mask, head_mask, LDA_weight=LDA_weight)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        if self.is_decoder and encoder_hidden_states is not None:
            cross_attention_outputs = self.crossattention(
                attention_output, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:]  # add cross attentions if we output attention weights

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + outputs
        return outputs