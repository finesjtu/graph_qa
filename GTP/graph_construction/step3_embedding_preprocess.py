from transformers import RobertaTokenizer, RobertaModel, RobertaForSequenceClassification, BertTokenizer, BertModel, \
    BertForSequenceClassification, BertConfig
import torch
import os
import json
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm, trange
import pickle
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import accuracy_score
from time import time
from embedding_model import BertModelEmbedding, SeqEvalModel
# from embedding_model import BertModelEmbedding, SeqEvalModel
from log_wrapper import create_logger
# argmin semeval2016t6
dataset = 'argmin'
path = '../dataset_roberta/'
train_dataset_path = path + dataset + '_train.json'
test_dataset_path = path + dataset + '_test.json'
dev_dataset_path = path + dataset + '_dev.json'
standford_path = './standford_den/' + dataset + '_stan.pkl'
output = './graph_cache/'
vocab_path = output + dataset + '_vocab.txt'
embedding_save_path = '/mnt/data1/GAS_DATA_WSK/graph_construction_pretrained_model/lm_training/' + dataset + '/embedding.pkl'
label_save_path = './graph_cache/' + dataset + '_label.pkl'
uid_save_path = './graph_cache/' + dataset + '_uid.pkl'
device = torch.device('cuda')
label_path = './graph_cache/' + dataset + '_labels.txt'
model_save_path = '/mnt/data1/GAS_DATA_WSK/graph_construction_pretrained_model/lm_training/' + dataset + '/'
log_path = '/mnt/data1/GAS_DATA_WSK/graph_construction_pretrained_model/lm_training/' + dataset + '/embedding_log.log'
if not os.path.exists(model_save_path):
    os.mkdir(model_save_path)
mode = 'w'
logger = create_logger('log_e', log_path, mode=mode)
continuous_train = True
load_model = False
batch_size = 16
epochs = 10
max_length = 256
with open(label_path, 'r') as f:
    label = f.readlines()

opt = {
    'num_label': len(label),
    'max_length': max_length,
    'model_save_path': model_save_path
}

def pkl_dump(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def get_label(path):
    label = []
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = json.loads(line)
            label.append(line['label'])

    return label

def get_vocab_label(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        label = [10000] * len(lines)
        # print(len(label))
    return label

class row_data(Dataset):
    def __init__(self, train_dataset_path, opt, tokenizer):
        self.train_dataset_path = train_dataset_path
        self.tokenizer = tokenizer
        self.examples = []
        self.doc_train_name_list = []
        self.doc_train_list = []
        self.label = []
        self.uid = []
        self.opt = opt
        with open(self.train_dataset_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                # example_meta = {}
                line = json.loads(line)
                data = line['hypothesis'] + ' ' + line['premise']
                self.examples.append(line)
                self.doc_train_name_list.append(line)
                self.label.append(line['label'])
                self.uid.append(line['uid'])
                self.doc_train_list.append(data)
        # self._generate_dataset(self.examples)


    def __getitem__(self, index):

        return self.doc_train_list[index], self.uid[index], self.label[index]
    def __len__(self):
        return len(self.doc_train_list)


    @staticmethod
    def _seq_length(parts, only_shortenable: bool = False):
        return sum([len(x) for x, shortenable in parts if not only_shortenable or shortenable]) if parts else 0

    @staticmethod
    def _remove_last(parts):
        last_idx = max(idx for idx, (seq, shortenable) in enumerate(parts) if shortenable and seq)
        parts[last_idx] = (parts[last_idx][0][:-1], parts[last_idx][1])

    def truncate(self, parts_a, parts_b, max_length: int):
        """Truncate two sequences of text to a predefined total maximum length"""
        total_len = self._seq_length(parts_a) + self._seq_length(parts_b)
        total_len += self.tokenizer.num_special_tokens_to_add(bool(parts_b))
        num_tokens_to_remove = total_len - max_length

        if num_tokens_to_remove <= 0:
            return parts_a, parts_b

        for _ in range(num_tokens_to_remove):
            if self._seq_length(parts_a, only_shortenable=True) > self._seq_length(parts_b, only_shortenable=True):
                self._remove_last(parts_a)
            else:
                self._remove_last(parts_b)


class all_row_data(Dataset):
    def __init__(self, train_dataset_path, test_dataset_path):
        self.train_dataset_path = train_dataset_path
        self.test_dataset_path = test_dataset_path
        self.doc_train_name_list = []
        self.doc_train_list = []
        self.label = []
        self.uid = []
        with open(self.train_dataset_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = json.loads(line)
                data = line['hypothesis'] + ' ' + line['premise']
                self.doc_train_name_list.append(line)
                self.label.append(line['label'])
                self.uid.append(line['uid'])
                self.doc_train_list.append(data)
        with open(self.test_dataset_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = json.loads(line)
                data = line['hypothesis'] + ' ' + line['premise']
                self.doc_train_name_list.append(line)
                self.label.append(line['label'])
                self.uid.append(line['uid'])
                self.doc_train_list.append(data)

    def __getitem__(self, index):

        return self.doc_train_list[index], self.uid[index], self.label[index]
    def __len__(self):
        return len(self.doc_train_list)


class vocab_data(Dataset):
    def __init__(self, vocab_path):
        self.vocab_path = vocab_path
        self.uid = []
        self.words = []
        self.label = []
        with open(self.vocab_path, 'r') as f:
            lines = f.readlines()
            for line in range(len(lines)):
                self.words.append(lines[line].strip())
                self.uid.append(line)
                self.label.append(10000)
    def __getitem__(self, index):

        return self.words[index], self.uid[index], self.label[index]
    def __len__(self):
        return len(self.words)

tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')

embedding = []
train_label = get_label(train_dataset_path)
test_label = get_label(test_dataset_path)
vocab_label = get_vocab_label(vocab_path)

label_list = train_label + vocab_label + test_label

logger.info(f'Train Label length:{len(train_label)}')
logger.info(f'Test Label length:{len(test_label)}')
logger.info(f'Vocab Label length:{len(vocab_label)}')
if len(train_label) + len(test_label) + len(vocab_label) == len(label_list):
    logger.info('Label length check done!')
logger.info(f'Label length:{len(label_list)}')
# pkl_dump(label_list, label_save_path)

train_test_mask = [True]*len(train_label) + [False] * len(test_label)

label_embedding_list = train_label + test_label
# print(label_embedding_list)
# 如果效果不好 使用LM辅助任务训练全部数据
# train_dataset = DataLoader(row_data(train_dataset_path,opt,BertTokenizer), shuffle=False, batch_size=batch_size)
train_dataset = DataLoader(all_row_data(train_dataset_path, test_dataset_path), shuffle=True, batch_size=batch_size)
embedding_dataset = DataLoader(all_row_data(train_dataset_path, test_dataset_path), shuffle=False, batch_size=batch_size)

acc_all = []
loss_all = []

model = BertModelEmbedding(opt, continuous_train=continuous_train).cuda()

no_decay = ['bias', 'gamma', 'beta', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_parameters = [
    {'params': [p for n, p in model.named_parameters() if n not in no_decay], 'weight_decay_rate': 0.01},
    {'params': [p for n, p in model.named_parameters() if n in no_decay], 'weight_decay_rate': 0.0}
    ]
optimizer = Adam(params=optimizer_parameters, lr=2e-5, betas=(0.9,0.98), eps=1e-9)
best_loss = 1000
if not load_model:
    for epoch in trange(epochs):
        start = time()
        model.train()
        for index, data in enumerate(train_dataset):
            row_data, uid, label = data
            encoder_input = tokenizer(row_data, return_tensors='pt', padding='max_length', truncation=True, max_length=256).to(device)
            loss, _ = model(encoder_input)
            loss_all.append(loss.cpu().detach().numpy())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if index % 20 == 0:
                logger.info('epoch:{}--batch:{}, loss:{}'.format(epoch, index, np.mean(loss.cpu().detach().numpy())))
        logger.info('*'*100)
        logger.info('Epoch:{}, loss:{}, train time:{}'.format(epoch, np.mean(loss_all), time() - start))
        logger.info('*' * 100)
        if np.mean(loss_all) < best_loss:
            best_loss = np.mean(loss_all)
            model.save_model()
            logger.info('model saved!')




model.load_model()
model.eval()
semantic_data = {}
with torch.no_grad():
    with tqdm(total=len(embedding_dataset), desc='(T)') as pbar:
        for index, data in enumerate(embedding_dataset):
            row_data, uid, label = data
            encoder_input = tokenizer(row_data, return_tensors='pt', padding='max_length', truncation=True, max_length=256).to(device)

            loss, embedding_data = model(encoder_input)
            embedding.extend(embedding_data.cpu().detach().numpy())
            pbar.update()

    logger.info(f'Embedding done! The shape of Embedding is:{np.array(embedding).shape}')
    semantic_data['label'] = label_embedding_list
    semantic_data['embedding'] = np.array(embedding)
    semantic_data['mask'] = train_test_mask


# print(semantic_data)
pkl_dump(semantic_data, embedding_save_path)