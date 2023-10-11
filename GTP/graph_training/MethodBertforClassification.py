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
from transformers import BertForSequenceClassification, BertConfig, BertPreTrainedModel


from GTP.graph_training.ModelBert import MethodLDABert
from GTP.graph_training.ModelComponent import GraphBertConfig

from GTP.graph_training.GraphDataLoader import GraphData, GraphDataTest

def args_setting():
    parser = argparse.ArgumentParser(description='LDABert')
    parser.add_argument("--dataset", type=str, default='argmin')
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


def Batch_LDA(LDA_data, index):

    np_data = LDA_data
    LDA_1 = np_data[index]
    LDA_2 = np.transpose(LDA_1)[index]
    LDA_3 = np.transpose(LDA_2)

    # batch_LDA = np.zeros((len(index), len(index)))
    # for i in range(len(index)):
    #     for j in range(len(index)):
    #         batch_LDA[i][j] = np_data[index[i], index[j]]
    batch_LDA = LDA_3
    return torch.tensor(batch_LDA)

dataset = 'argmin'
batch_size = 1024
epochs = 10
lr = 1e-5
device = 'cuda'
graph_embedding = False
neighbour_save = False
LDA_path = '/mnt/data1/GAS_DATA_WSK/lda_edge/' + dataset + '/similarity.pkl'



with open(LDA_path, 'rb') as f:
    LDA_data = pkl.load(f).todense()
Alldata = GraphData(dataset=dataset, nei_save=True)
# all_len = Alldata.get_len()
# Alldata.save_neighbour()
train_dataset = GraphData(dataset, 'train', nei_save=neighbour_save)
test_dataset = GraphData(dataset, 'test', nei_save=neighbour_save)

# train_len = train_dataset.get_len()
# test_len = test_dataset.get_len()
#
# print(all_len, train_len, test_len)

semantic_data, graph_seq_data, graph_syn_data = Alldata.embedding_data_load()
train_semantic_data, train_graph_seq_data, train_graph_syn_data = train_dataset.embedding_data_load()
test_semantic_data, test_graph_seq_data, test_graph_syn_data = test_dataset.embedding_data_load()
semantic_dim = semantic_data['embedding'].shape[1]
graph_seq_dim = graph_seq_data.shape[1]
graph_syn_dim = graph_syn_data.shape[1]

num_labels = max(semantic_data['label']) + 1

train_dataloder = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
test_dataloder = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)


config = GraphBertConfig.from_pretrained('bert-base-uncased')
config.num_labels = num_labels
config.graph_embedding = graph_embedding
model = MethodLDABert(config).to(device)
no_decay = ['bias', 'gamma', 'beta', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_parameters = [
    {'params': [p for n, p in model.named_parameters() if n not in no_decay], 'weight_decay_rate': 0.01},
    {'params': [p for n, p in model.named_parameters() if n in no_decay], 'weight_decay_rate': 0.0}
    ]
optimizer = optim.Adam(optimizer_parameters, lr=lr)
# print(model)
loss_all = []
acc_all = []
f1_all = []
best_acc = []
best_loss = []
best_f1 = []
for epoch in range(epochs):
    model.train()
    # start = time.time()
    for batch_index, batch_data in enumerate(train_dataloder):
        # print(time.time()-start)
        # start = time.time()
        # batch_LDA = torch.FloatTensor
        semantic_batch_data = batch_data['semantic_data'].to(device)
        graph_seq_batch_data = batch_data['graph_seq_data'].to(device)
        graph_syn_batch_data = batch_data['graph_syn_data'].to(device)
        label = batch_data['label'].to(device)
        neighbour_index = batch_data['neighbour']
        sample_index = batch_data['index']
        # print(semantic_batch_data.shape)
        # print(sample_index)
        # print(time.time()-start)
        # start = time.time()
        batch_LDA = []
        # 根据neighbour取出表征，构建LDA矩阵
        for index in neighbour_index:
            LDA = Batch_LDA(LDA_data, index)
            # print(LDA.shape)
            batch_LDA.append(np.array(LDA))
        #     batch_LDA = np.append(batch_LDA, LDA, axis=1)
        #     print(batch_LDA.shape)
        batch_LDA = torch.FloatTensor(np.array(batch_LDA)).to(device)
        # print(batch_LDA)
        # print(batch_LDA.shape)
        # print(time.time()-start)
        logit = model(semantic_batch_data, LDA_weight=None)
        prob = torch.argmax(F.softmax(logit, dim=1), dim=1)
        # print(F.softmax(logit, dim=1))
        # print(logit)
        # print(prob)
        # print(label)
        acc = accuracy_score(label.cpu(), prob.cpu())
        f1 = f1_score(label.cpu(), prob.cpu())
        loss = F.cross_entropy(logit, label)
        loss_all.append(loss.cpu().detach().numpy())
        acc_all.append(acc)
        f1_all.append(f1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_index % 10 == 0:
            print('epoch:{}--batch:{}, loss:{}, acc:{}, f1_score:{}'.format(epoch, batch_index, loss.cpu().detach().numpy(), acc,
                                                                            f1))

    print('Epoch:{}--------loss:{}, acc:{}, f1_score:{}'.format(epoch, np.mean(loss_all), np.mean(acc_all),
                                                                    np.mean(f1_all)))
    print('='* 100)
    test_acc_all = []
    test_f1_all = []
    test_loss_all = []
    model.eval()
    with torch.no_grad():
        for index, batch_data in enumerate(test_dataloder):


            semantic_batch_data = batch_data['semantic_data'].to(device)
            # graph_seq_batch_data = batch_data['graph_seq_data'].to(device)
            # graph_syn_batch_data = batch_data['graph_syn_data'].to(device)
            neighbour_index = batch_data['neighbour']
            sample_index = batch_data['index']
            label = batch_data['label'].to(device)
            batch_LDA = []
            # 根据neighbour取出表征，构建LDA矩阵
            for index in neighbour_index:
                LDA = Batch_LDA(LDA_data, index)
                # print(LDA.shape)
                batch_LDA.append(np.array(LDA))
            #     batch_LDA = np.append(batch_LDA, LDA, axis=1)
            #     print(batch_LDA.shape)
            batch_LDA = torch.FloatTensor(np.array(batch_LDA)).to(device)
            logit = model(semantic_batch_data, LDA_weight=None)
            prob = torch.argmax(F.softmax(logit, dim=1), dim=1)
            # print(F.softmax(logit, dim=1))
            # print(logit)
            # print(prob)
            # print(label)
            acc = accuracy_score(label.cpu(), prob.cpu())
            f1 = f1_score(label.cpu(), prob.cpu())
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
