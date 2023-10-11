import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertConfig
import argparse
import time
import numpy as np
import os
import pickle as pkl
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from transformers import BertForSequenceClassification, BertConfig, BertPreTrainedModel, BertTokenizer
from transformers import get_linear_schedule_with_warmup

from GTP.graph_training.ModelBert import MethodLDABert, MethodVanBert
from GTP.graph_training.ModelComponent import GraphBertConfig

from GTP.graph_training.GraphDataLoader import GraphData, GraphDataTest, GraphDataW2v, GraphDataWordEmbedding





dataset = 'semeval2016t6'
batch_size = 16
epochs = 10
lr = 1e-5
device = 'cuda'
graph_embedding = False
neighbour_save = False
LDA_path = '/mnt/data1/GAS_DATA_WSK/lda_edge/' + dataset + '/similarity.pkl'



train_dataset = GraphDataWordEmbedding(dataset, 'train')
test_dataset = GraphDataWordEmbedding(dataset, 'test')




num_labels = max(train_dataset.label) + 1

train_dataloder = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
test_dataloder = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

pre_name = 'bert-large-uncased'
config = BertConfig.from_pretrained(pre_name)
tokenizer = BertTokenizer.from_pretrained(pre_name)
config.num_labels = num_labels
# config.position_embedding_type = 'relative_key' #relative_key_query
# model = BertForSequenceClassification.from_pretrained(pre_name, config=config).to(device)
model = MethodVanBert.from_pretrained(pre_name, config=config).to(device)

no_decay = ['bias', 'gamma', 'beta', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_parameters = [
    {'params': [p for n, p in model.named_parameters() if n not in no_decay], 'weight_decay_rate': 0.01},
    {'params': [p for n, p in model.named_parameters() if n in no_decay], 'weight_decay_rate': 0.0}
    ]
optimizer = optim.AdamW(optimizer_parameters, lr=lr)
# scheduler = get_linear_schedule_with_warmup(optimizer, 900, epochs*len(train_dataloder))
# print(model)

best_acc = []
best_loss = []
best_f1 = []
for epoch in range(epochs):
    model.train()
    # start = time.time()
    loss_all = []
    acc_all = []
    f1_all = []
    for batch_index, batch_data in enumerate(train_dataloder):
        semantic_batch_data = batch_data['semantic_data']
        semantic_batch = tokenizer(semantic_batch_data, padding='max_length', truncation=True, max_length=256)
        input_ids = torch.tensor(np.array(semantic_batch['input_ids'])).to(device)
        token_type_ids = torch.tensor(np.array(semantic_batch['token_type_ids'])).to(device)
        attention_mask = torch.tensor(np.array(semantic_batch['attention_mask'])).to(device)
        # print(semantic_batch)
        # print(semantic_batch_data)
        label = batch_data['label'].to(device)
        # print(semantic_batch_data.shape)
        # print(label)

        result = model(input_ids=input_ids, token_type_ids=token_type_ids,
                      attention_mask=attention_mask, labels=label)

        logit = result['logits']
        loss = result['loss']
        # logit = result.logits
        # loss = result.loss
        prob = torch.argmax(F.softmax(logit, dim=1), dim=1)

        acc = accuracy_score(label.cpu(), prob.cpu())
        f1 = f1_score(label.cpu(), prob.cpu(), average='macro')

        loss_all.append(loss.cpu().detach().numpy())
        acc_all.append(acc)
        f1_all.append(f1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step()
        if batch_index % 20 == 0:
            print('epoch:{}--batch:{}, loss:{}, acc:{}, f1_score:{}'.format(epoch, batch_index, loss.cpu().detach().numpy(), acc,
                                                                            f1))

    print('Epoch:{}--------loss:{}, acc:{}, f1_score:{}'.format(epoch, np.mean(loss_all), np.mean(acc_all),
                                                                    np.mean(f1_all)))
    # print(loss_all)
    # print(acc_all)
    # print(f1_all)
    print('='* 100)
    test_acc_all = []
    test_f1_all = []
    test_loss_all = []
    model.eval()
    with torch.no_grad():
        for index, batch_data in enumerate(test_dataloder):
            semantic_batch_data = batch_data['semantic_data']
            semantic_batch = tokenizer(semantic_batch_data, padding='max_length', truncation=True, max_length=256)
            input_ids = torch.tensor(np.array(semantic_batch['input_ids'])).to(device)
            token_type_ids = torch.tensor(np.array(semantic_batch['token_type_ids'])).to(device)
            attention_mask = torch.tensor(np.array(semantic_batch['attention_mask'])).to(device)
            # print(semantic_batch)
            # print(semantic_batch_data)
            label = batch_data['label'].to(device)
            # print(semantic_batch_data.shape)
            # print(label)


            result = model(input_ids=input_ids, token_type_ids=token_type_ids,
                           attention_mask=attention_mask, labels=label)
            logit = result['logits']
            loss = result['loss']
            # logit = result.logits
            # loss = result.loss
            prob = torch.argmax(F.softmax(logit, dim=1), dim=1)

            acc = accuracy_score(label.cpu(), prob.cpu())
            f1 = f1_score(label.cpu(), prob.cpu(), average='macro')



            test_loss_all.append(loss.cpu().detach().numpy())
            test_acc_all.append(acc)
            test_f1_all.append(f1)

            loss_all.append(loss.cpu().detach().numpy())
    print('Test_Epoch:{}--------test loss:{}, test acc:{}, test f1_score:{}'.format(epoch, np.mean(test_loss_all), np.mean(test_acc_all),
                                                                np.mean(test_f1_all)))
    # print(test_acc_all)
    # print(test_f1_all)
    print('='*100)
    best_acc.append(np.mean(test_acc_all))
    best_f1.append(np.mean(test_f1_all))

print('Acc:',best_acc)
print('F1:',best_f1)
best_acc = np.max(best_acc)
best_f1 = np.max(best_f1)

print('Best Acc:{}, Best F1:{}'.format(best_acc, best_f1))





#
# config = GraphBertConfig.from_pretrained('bert-base-uncased')
# model = MethodLDABert(config)
#
