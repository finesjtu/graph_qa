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


from GTP.graph_training.ModelBert import MethodLDABert
from GTP.graph_training.ModelComponent import GraphBertConfig

from GTP.graph_training.GraphDataLoader import GraphData, GraphDataTest, GraphDataW2v

def args_setting():
    parser = argparse.ArgumentParser(description='LDABert')
    parser.add_argument("--dataset", type=str, default='ibmcs')
    parser.add_argument("--dropout", type=float, default=0.5,help="dropout probability")
    parser.add_argument("--lr_c", type=float, default=0.01,help="learning rate")
    parser.add_argument("--seed", type=int, default=100,help="random seed")
    parser.add_argument("--n-hidden", type=int, default=128,help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=2,help="number of hidden gcn layers")
    parser.add_argument("--weight-decay", type=float, default=5e-4,help="Weight for L2 loss")
    parser.add_argument("--aggregator-type", type=str, default="gcn")
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--sample-list', type=list, default=[4,4])
    parser.add_argument("--n-epochs", type=int, default=100,help="number of training epochs")
    parser.add_argument("--file-id", type=str, default='128')
    parser.add_argument("--gpu", type=int, default=0,help="gpu")
    parser.add_argument("--lr", type=float, default=2e-3,help="learning rate")
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--half', type=bool, default=False)
    parser.add_argument('--mask_rate', type=float, default=0)
    parser.add_argument('--center_num', type=int, default=7)
    args = parser.parse_args()

    return args



dataset = 'perspectrum'
neighbor_num = 10
batch_size = 16
topic_num = 10
epochs = 20
lr = 1e-5
device = 'cuda'
graph_embedding = False
neighbour_save = False
LDA_path = '/mnt/data1/GAS_DATA_WSK/lda_edge/' + dataset + '/' + str(topic_num) + 'similarity.pkl'
seed = 42
# print(seed)
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)


train_dataset = GraphDataW2v(dataset, 'train',neighbor_num=neighbor_num)
test_dataset = GraphDataW2v(dataset, 'test',neighbor_num=neighbor_num)




num_labels = max(train_dataset.label) + 1

train_dataloder = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
test_dataloder = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

pretrain_model = 'bert-large-uncased'
config = BertConfig.from_pretrained(pretrain_model)
tokenizer = BertTokenizer.from_pretrained(pretrain_model)
print(config)
config.num_labels = num_labels
# config.graph_embedding = graph_embedding
config.graph_embedding_dim = 512
max_length = 256
# config.semantic_dim = 300
model = MethodLDABert.from_pretrained(pretrain_model, config=config).to(device)
no_decay = ['bias', 'gamma', 'beta', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_parameters = [
    {'params': [p for n, p in model.named_parameters() if n not in no_decay], 'weight_decay_rate': 0.01},
    {'params': [p for n, p in model.named_parameters() if n in no_decay], 'weight_decay_rate': 0.0}
    ]
optimizer = optim.AdamW(optimizer_parameters, lr=lr)
# print(model)

best_acc = []
best_loss = []
best_f1 = []
batch_count = 0
for epoch in range(epochs):
    loss_all = []
    acc_all = []
    f1_all = []
    model.train()
    # start = time.time()
    for batch_index, batch_data in enumerate(train_dataloder):
        batch_count += batch_index
        # print(time.time()-start)
        # start = time.time()
        # batch_LDA = torch.FloatTensor
        semantic_batch_data = batch_data['semantic_data']
        graph_syn_data = batch_data['graph_syn_data'].to(device)
        graph_seq_data = batch_data['graph_seq_data'].to(device)
        neighbor_index= batch_data['neighbor_index'].to(device)
        LDA_weight = batch_data['neighbor_ratio']
        # print('neighbor_ratio shape:', LDA_weight.shape)
        semantic_batch_tokens = tokenizer(semantic_batch_data, padding='max_length', truncation=True, max_length=max_length)
        input_ids = torch.tensor(np.array(semantic_batch_tokens['input_ids'])).to(device)
        token_type_ids = torch.tensor(np.array(semantic_batch_tokens['token_type_ids'])).to(device)
        attention_mask = torch.tensor(np.array(semantic_batch_tokens['attention_mask'])).to(device)
        # print(semantic_batch_tokens)
        label = batch_data['label'].to(device)
        # print(semantic_batch_data.shape)
        # print(label)
        # 与BERTVan差一个graph embedding的layer， 如果LDA和graph_Embedding都不输入的话
        logit = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                      LDA_weight=LDA_weight, graph_embedding=[graph_seq_data, graph_syn_data])
        prob = torch.argmax(F.softmax(logit, dim=1), dim=1)

        acc = accuracy_score(label.cpu(), prob.cpu())
        f1 = f1_score(label.cpu(), prob.cpu(), average='macro')

        loss = F.cross_entropy(logit, label)
        loss_all.append(loss.cpu().detach().numpy())

        acc_all.append(acc)
        f1_all.append(f1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_index % 20 == 0:
            print('epoch:{}--batch:{}, loss:{}, acc:{}, f1_score:{}'.format(epoch, batch_index, loss.cpu().detach().numpy(), acc,
                                                                            f1))
            # test_acc_all = []
            # test_f1_all = []
            # test_loss_all = []
            # model.eval()
            # with torch.no_grad():
            #     for index, batch_data in enumerate(test_dataloder):
            #         semantic_batch_data = batch_data['semantic_data']
            #         graph_syn_data = batch_data['graph_syn_data'].to(device)
            #         graph_seq_data = batch_data['graph_seq_data'].to(device)
            #         neighbor_index = batch_data['neighbor_index'].to(device)
            #         LDA_weight = batch_data['neighbor_ratio']
            #         semantic_batch_tokens = tokenizer(semantic_batch_data, padding='max_length', truncation=True,
            #                                           max_length=max_length)
            #         input_ids = torch.tensor(np.array(semantic_batch_tokens['input_ids'])).to(device)
            #         # token_type_ids = semantic_batch_tokens['token_type_ids']
            #         attention_mask = torch.tensor(np.array(semantic_batch_tokens['attention_mask'])).to(device)
            #         # print(semantic_batch_tokens)
            #         label = batch_data['label'].to(device)
            #         # print(semantic_batch_data.shape)
            #         # print(label)
            #
            #         logit = model(input_ids=input_ids, attention_mask=attention_mask, LDA_weight=LDA_weight,
            #                       graph_embedding=[graph_seq_data, graph_syn_data])
            #         prob = torch.argmax(F.softmax(logit, dim=1), dim=1)
            #         # print(F.softmax(logit, dim=1))
            #         # print(logit)
            #         # print(prob)
            #         # print(label)
            #         acc = accuracy_score(label.cpu(), prob.cpu())
            #         f1 = f1_score(label.cpu(), prob.cpu(), average='macro')
            #         loss = F.cross_entropy(logit, label)
            #
            #         test_loss_all.append(loss.cpu().detach().numpy())
            #         test_acc_all.append(acc)
            #         test_f1_all.append(f1)
            #
            #         loss_all.append(loss.cpu().detach().numpy())
            # print(
            #     'Test_Epoch:{}--------test loss:{}, test acc:{}, test f1_score:{}'.format(epoch, np.mean(test_loss_all),
            #                                                                               np.mean(test_acc_all),
            #                                                                               np.mean(test_f1_all)))
            # print('=' * 100)
            # best_acc.append(np.mean(test_acc_all))
            # best_f1.append(np.mean(test_f1_all))
            # model.train()



    print('='* 100)
    test_acc_all = []
    test_f1_all = []
    test_loss_all = []
    model.eval()
    with torch.no_grad():
        for index, batch_data in enumerate(test_dataloder):
            semantic_batch_data = batch_data['semantic_data']
            graph_syn_data = batch_data['graph_syn_data'].to(device)
            graph_seq_data = batch_data['graph_seq_data'].to(device)
            neighbor_index = batch_data['neighbor_index'].to(device)
            LDA_weight = batch_data['neighbor_ratio']
            semantic_batch_tokens = tokenizer(semantic_batch_data, padding='max_length', truncation=True,
                                              max_length=max_length)
            input_ids = torch.tensor(np.array(semantic_batch_tokens['input_ids'])).to(device)
            # token_type_ids = semantic_batch_tokens['token_type_ids']
            attention_mask = torch.tensor(np.array(semantic_batch_tokens['attention_mask'])).to(device)
            # print(semantic_batch_tokens)
            label = batch_data['label'].to(device)
            # print(semantic_batch_data.shape)
            # print(label)

            logit = model(input_ids=input_ids, attention_mask=attention_mask, LDA_weight=LDA_weight,
                          graph_embedding=[graph_seq_data, graph_syn_data])
            prob = torch.argmax(F.softmax(logit, dim=1), dim=1)
            # print(F.softmax(logit, dim=1))
            # print(logit)
            # print(prob)
            # print(label)
            acc = accuracy_score(label.cpu(), prob.cpu())
            f1 = f1_score(label.cpu(), prob.cpu(), average='macro')
            loss = F.cross_entropy(logit, label)

            test_loss_all.append(loss.cpu().detach().numpy())
            test_acc_all.append(acc)
            test_f1_all.append(f1)

            loss_all.append(loss.cpu().detach().numpy())
    print('Test_Epoch:{}--------test loss:{}, test acc:{}, test f1_score:{}'.format(epoch, np.mean(test_loss_all), np.mean(test_acc_all),
                                                                np.mean(test_f1_all)))
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
