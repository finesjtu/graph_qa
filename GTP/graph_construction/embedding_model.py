from transformers import BertModel, BertConfig, BertForMaskedLM, BertTokenizer, BertForSequenceClassification
from transformers import __version__ as transformers_version
import torch.nn as nn
import torch
import os
import torch.nn.functional as F

class BertModelEmbedding(nn.Module):
    def __init__(self, opt, continuous_train=False):
        super(BertModelEmbedding, self).__init__()
        self.continuous_train = continuous_train
        self.config = BertConfig.from_pretrained('bert-large-uncased')
        self.bert = BertForMaskedLM.from_pretrained('bert-large-uncased', output_hidden_states=True)
        # self.bert_c = BertForSequenceClassification.from_pretrained('bert-large-uncased')
        self.num_label = opt['num_label']
        self.model_save_path = opt['model_save_path']
        # self.classification = nn.Linear(self.config.hidden_size, self.num_label)
        self.tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
        # print(self.bert)
        # print(self.bert_c)
        print(self.config)
        if self.continuous_train and os.path.exists(opt['model_save_path'] + 'pytorch_model.bin'):
            self.load_model()
        # self._my_init()
    def _my_init(self):
        def init_weights(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # elif isinstance(module, BertLayerNorm):
            #     # Slightly different from the BERT pytorch version, which should be a bug.
            #     # Note that it only affects on training from scratch. For detailed discussions, please contact xiaodl@.
            #     # Layer normalization (https://arxiv.org/abs/1607.06450)
            #     # support both old/latest version
            #     if 'beta' in dir(module) and 'gamma' in dir(module):
            #         module.beta.data.zero_()
            #         module.gamma.data.fill_(1.0)
            #     else:
            #         module.bias.data.zero_()
            #         module.weight.data.fill_(1.0)
            if isinstance(module, nn.Linear):
                module.bias.data.zero_()
        self.apply(init_weights)

    def save_model(self):
        self.bert.save_pretrained(self.model_save_path)

    def load_model(self):

        self.bert.load_state_dict(torch.load(self.model_save_path + 'pytorch_model.bin'))

    def _mask_tokens(self, input_ids):
        """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
        # print(list(input_ids))
        # print(input_ids)
        labels = input_ids.clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability 0.15)
        probability_matrix = torch.full(labels.shape, 0.15)
        special_tokens_mask = [self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in
                               labels.tolist()]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)

        masked_indices = torch.bernoulli(probability_matrix).bool()

        # if a version of transformers < 2.4.0 is used, -1 is the expected value for indices to ignore
        if [int(v) for v in transformers_version.split('.')][:3] >= [2, 4, 0]:
            ignore_value = -100
        else:
            ignore_value = -1

        labels[~masked_indices] = ignore_value  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long).cuda()
        # print(indices_random)
        input_ids[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return input_ids, labels

    def forward(self, encoder_input):
        # encoder_input = self.tokenizer(row_data, return_tensors='pt', padding='max_length', truncation=True,
        #                           max_length=256)
        input_ids, labels = self._mask_tokens(encoder_input['input_ids'].cuda())
        encoder_input['input_ids'] = input_ids
        # print(input_ids, labels)
        # embeddings = self.bert(input_ids, attention_mask).last_hidden_state
        # pooler_output = self.bert(input_ids, attention_mask).pooler_output
        outputs = self.bert(**encoder_input, labels=labels)
        embedding = outputs.hidden_states[-1][:, 0]
        # embedding = outputs.hidden_states[-1]
        # print(outputs)
        # print('*'*100)
        # # print(outputs['hidden_states'])
        # print(len(outputs['hidden_states']))
        # print(outputs['hidden_states'][-1].shape)
        # print('*'*100)
        loss = outputs.loss
        logits = outputs.logits
        # print(embeddings.shape, pooler_output.shape)
        # output = self.classification(pooler_output)
        # print(output)
        return loss, embedding



class SeqEvalModel(nn.Module):
    def __init__(self, opt, input_dim, hidden_dim):
        super(SeqEvalModel, self).__init__()
        self.num_label = opt['num_label']
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, self.num_label)
        # torch.nn.init.xavier_uniform_(self.fc1.weight.data)
        # torch.nn.init.xavier_uniform_(self.fc2.weight.data)



    def forward(self, inputs):
        # inputs = F.dropout(inputs)
        hidden_state = self.fc1(inputs)
        outputs = self.fc2(hidden_state)
        logits = F.softmax(outputs, dim=1)

        return logits
