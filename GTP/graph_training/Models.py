## coding:utf-8
import dgl.nn as dglnn
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl import data
import torch
import scipy.sparse as sp
import pickle as pkl
import numpy as np
from GTP.graph_training.GraphDataLoader import GraphData
from GTP.graph_construction.log_wrapper import create_logger

class SAGE(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super().__init__()
        self.conv1 = dglnn.SAGEConv(in_feats=in_feats, out_feats=hid_feats, aggregator_type='pool', feat_drop=0.3)
        self.conv2 = dglnn.SAGEConv(in_feats=hid_feats, out_feats=out_feats, aggregator_type='pool', feat_drop=0.3)
        # self.conv1 = dglnn.GatedGraphConv(in_feats=in_feats, out_feats=hid_feats, n_steps=3, n_etypes=1)
        # self.conv2 = dglnn.GatedGraphConv(in_feats=hid_feats, out_feats=out_feats, n_steps=3, n_etypes=1)

    def forward(self, graph, inputs):
        h = self.conv1(graph, inputs)
        h = F.relu(h)
        h = self.conv2(graph, h)
        return h

def evaluate(model, graph, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(graph, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

def pkl_load(path):
    with open(path, 'rb') as f:
        data = pkl.load(f)
    return data

logger = create_logger('test', './testlog.log')
dataset = 'argmin'
DataLoader = GraphData(dataset=dataset,logger=logger)
graph, label_data, n_labels, train_index, test_index = DataLoader.graph_data_process()



node_features = graph.ndata['feat']
node_labels = graph.ndata['label']
train_mask = graph.ndata['train_mask']
test_mask = graph.ndata['test_mask']
n_features = node_features.shape[1]
n_labels = int(node_labels.max().item() + 1)
model = SAGE(in_feats=n_features, hid_feats=128, out_feats=n_labels)
opt = torch.optim.Adam(model.parameters(), lr=0.01)

acc_record = 0.0
for epoch in range(50):
    model.train()
    output = model(graph, node_features)
    loss = F.cross_entropy(output[train_mask], node_labels[train_mask])
    acc_train = evaluate(model, graph, node_features, node_labels, train_mask)
    acc = evaluate(model, graph, node_features, node_labels, test_mask)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if acc > acc_record:
        acc_record = acc
    print('Epoch:{}, Loss:{}, Train_Acc:{}, Test_Acc:{}'.format(epoch, loss.item(), acc_train, acc))

print('best test acc:', acc_record)
