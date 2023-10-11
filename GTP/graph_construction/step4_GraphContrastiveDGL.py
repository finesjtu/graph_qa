import torch
import os.path as osp
import GCL.losses as L
import GCL.augmentors as A
import torch.nn.functional as F
import torch_geometric.transforms as T

from tqdm import tqdm, trange
from torch.optim import Adam
from GCL.eval import get_split, LREvaluator
from GCL.models import DualBranchContrast
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
import torch.nn as nn
import pickle as pkl
import numpy as np
import dgl
import dgl.nn as dglnn
from numpy import random
import logging
from log_wrapper import create_logger
from GTP.graph_training.GraphDataLoader import GraphData



from dgl.data import DGLDataset

DEVICE = torch.device('cuda')

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

class GConv(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, activation, num_layers):
        super(GConv, self).__init__()
        self.activation = activation()
        self.layers = torch.nn.ModuleList()
        self.layers.append(dglnn.GraphConv(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(dglnn.GraphConv(hidden_dim, hidden_dim))

    def forward(self, blocks):
        z = blocks[0].srcdata['feat']
        for i, conv in enumerate(self.layers):
            z = conv(blocks[i], z)
            z = self.activation(z)
        return z


class Encoder(torch.nn.Module):
    def __init__(self, encoder, augmentor, hidden_dim, proj_dim, samples_num, embedding_dim):
        super(Encoder, self).__init__()
        self.encoder = encoder
        self.augmentor = augmentor
        # self.embedding = nn.Embedding(samples_num, embedding_dim)
        self.fc1 = torch.nn.Linear(hidden_dim, proj_dim)
        self.fc2 = torch.nn.Linear(proj_dim, hidden_dim)

    def forward(self, blocks):
        # print(self.augmentor)
        aug1, aug2 = self.augmentor
        # for block in blocks:
        #     block.srcdata['feat'] = self.embedding(block.srcdata['x'])
        #     block.dstdata['feat'] = self.embedding(block.dstdata['x'])
        # x = self.embedding(x)
        # print(x.shape)
        # print(x)
        blocks1 = aug1.augment(blocks)
        blocks2 = aug2.augment(blocks)
        # Z is the embedding of node
        z = self.encoder(blocks)
        z1 = self.encoder(blocks1)
        z2 = self.encoder(blocks2)

        return z, z1, z2

    def project(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

class EdgeRemoving():
    def __init__(self, pe: float):
        super(EdgeRemoving, self).__init__()
        self.pe = pe

    def augment(self, blocks):
        for block in blocks:
            print(block.edata)
            edge_index = block.edata['_ID']
            edge_weight = block.edata['weight']
            mask = torch.rand(edge_index.shape[0], device=DEVICE) >= self.pe
            remove_index = edge_index[mask]
            block.remove_edges(remove_index)



        return blocks

    def edge_adj(self, block):
        pass

class FeatureMasking():
    def __init__(self, pf: float):
        super(FeatureMasking, self).__init__()
        self.pf = pf

    def augment(self, blocks):
        dst_last_data = self.drop_feature(blocks[0].srcdata['feat'])
        for block in blocks:
            src_data = dst_last_data
            dst_data = self.drop_feature(block.dstdata['feat'])

            dst_last_data = dst_data

            block.srcdata['feat'] = src_data
            block.dstdata['feat'] = dst_data

        return blocks

    def drop_feature(self, x):
        drop_mask = torch.empty((x.size(1),), dtype=torch.float32).uniform_(0, 1) < self.pf
        drop_mask = drop_mask.to(DEVICE)
        x = x.clone()
        x[:, drop_mask] = 0
        return x


def train(encoder_model, contrast_model, blocks, optimizer):
    encoder_model.train()
    optimizer.zero_grad()
    # z, z1, z2 = encoder_model(data.x, data.edge_index, data.edge_attr)
    z, z1, z2 = encoder_model(blocks)
    # print(z.shape)
    # print(z.shape)
    h1, h2 = [encoder_model.project(x) for x in [z1, z2]]
    loss = contrast_model(h1, h2)
    loss.backward()
    optimizer.step()
    return loss.item()


def test(encoder_model, blocks):
    encoder_model.eval()
    # z, _, _ = encoder_model(data.x, data.edge_index, data.edge_attr)
    # print(blocks)
    z, _, _ = encoder_model(blocks)
    index = blocks[-1].dstdata['x']
    # print(z.shape, blocks[-1].dstdata['label'].shape)

    split = get_split(num_samples=len(index), train_ratio=0.1, test_ratio=0.8)
    result = LREvaluator(num_epochs=2000)(z, blocks[-1].dstdata['label'], split)
    return result, z

def pkl_load(path):
    with open(path, 'rb') as f:
        data = pkl.load(f)
    return data


def load_graph_data(graph_name, dataset, embedding_dim):
    graph_name = graph_name
    dataset = dataset
    embedding_dim = embedding_dim
    path = '/root/NLPCODE/GAS/GTP/graph_construction/graph_cache/'
    feature_path = path + dataset + '_w2v_feature.pkl'
    edge_path = path + 'ind.' + dataset + '.' + graph_name + '.adj'
    label_path = path + dataset + '_label.pkl'
    edge_data = pkl_load(edge_path)
    feature_data = pkl_load(feature_path)
    # data = GraphData(dataset)
    # print(edge_data.row)
    # print(edge_data.col)
    # print(edge_data)
    graph = dgl.from_scipy(edge_data, eweight_name='weight')
    graph = dgl.add_self_loop(graph)
    # graph_test = graph
    # graph.edata['weight'] = torch.tensor([0.1]*len(graph.edata['weight']))
    logger.info(graph)
    logger.info(graph.num_edges())
    label_data = pkl_load(label_path)

    train_labels = []
    test_labels = []
    train_index = []
    test_index = []
    logger.info(len(label_data))
    for i in range(len(label_data)):
        label = label_data[i]
        if label == 10000:
            break
        else:
            train_labels.append(i)
            train_index.append(i)

    for i in range(len(label_data[len(train_labels):])):
        label = label_data[len(train_labels) + i]
        index = len(train_labels) + i
        if label != 10000:
            # print(label_data[index])
            # print(label)
            test_labels.append(i)
            test_index.append(index)
    # print(label_data[test_index[0]:])
    labels = train_labels + test_labels
    train_mask_index = range(len(train_labels))
    test_mask_index = range(len(label_data) - len(test_labels), len(label_data))
    train_mask = torch.tensor([False] * len(label_data))
    test_mask = torch.tensor([False] * len(label_data))
    # print(test_mask)
    for i in train_mask_index:
        train_mask[i] = True

    for i in test_mask_index:
        test_mask[i] = True

    graph.ndata['label'] = torch.tensor(np.array([0 if x == 10000 else x for x in label_data]))
    graph.ndata['train_mask'] = torch.tensor(np.array(list(train_mask)))
    graph.ndata['test_mask'] = torch.tensor(np.array(list(test_mask)))
    graph.ndata['val_mask'] = torch.tensor(np.array(list(test_mask)))
    edge = edge_data.toarray()
    vocab_label_index = []
    # for i, l in enumerate(label_data):
    #     if l == 10000:
    #         vocab_label_index.append(i)
    # print(vocab_label_index)
    # for i, label in enumerate(label_data):
    #     if label != 10000:
    #         adj = edge[i]
    #         index_p = np.argpartition(adj, -10)[-10:]
    #         print(len(adj[adj>0]))
    #         print(np.argpartition(adj, -10)[-10:])
    #         if len(adj[adj>0]) < 10:
    #             print(adj[index_p[0]])
            # print(adj)

    # feature_data = np.zeros(len(label_data), 1024)
    graph.ndata['feat'] = torch.FloatTensor(feature_data)
    graph.ndata['x'] = torch.tensor(np.array(list(range(len(label_data)))))
    # node_features = graph.ndata['feat']
    node_labels = graph.ndata['label']
    # train_mask = graph.ndata['train_mask']
    # # valid_mask = graph.ndata['val_mask']
    # test_mask = graph.ndata['test_mask']
    # # print(test_mask)
    # n_features = node_features.shape[1]
    n_labels = int(node_labels.max().item() + 1)
    return graph, label_data, n_labels, train_index, test_index


def main(dataset_name, graph_name, logger):
    result = []
    # device = torch.device('cuda')
    dataset_name = dataset_name
    graph_name = graph_name
    embedding_dim = 300
    num_epochs = 100
    batch_size = 2048
    hidden_dim = 512
    proj_dim = 64
    num_layers = 2

    # path = osp.join(osp.expanduser('~'), 'datasets')
    # dataset = Planetoid(path, name='Cora', transform=T.NormalizeFeatures())
    # # print(dataset[0]['edge_attr'])
    # data = dataset[0].to(device)
    path = './graph_cache/'

    logger.info('Dataset name:{}, Graph name:{}'.format(dataset_name, graph_name))

    graph, label_data, n_labels, train_index, test_index = load_graph_data(graph_name, dataset_name, embedding_dim=embedding_dim)
    train_nids = list(range(len(label_data)))
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
    dataloader = dgl.dataloading.NodeDataLoader(
        graph, train_nids, sampler,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=4)




    test_nids = train_index + test_index
    # print(test_nids)
    test_dataloader = dgl.dataloading.NodeDataLoader(
        graph, test_nids, sampler,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=4)
    # print(graph.num_nodes())
    # for i in range(1):
    #     input_nodes, output_nodes, blocks = next(iter(dataloader))
    #     for j in blocks:
    #         print(j.srcdata)
    #         print(j.dstdata)

    # aug1 = EdgeRemoving(pe=0.3)
    aug1 = FeatureMasking(pf=0.3)
    aug2 = FeatureMasking(pf=0.3)

    gconv = GConv(input_dim=embedding_dim, hidden_dim=hidden_dim, activation=torch.nn.ReLU, num_layers=num_layers).to(DEVICE)
    # encoder_model = Encoder(encoder=gconv, augmentor=(aug1, aug2), hidden_dim=256, proj_dim=32, samples_num=data['num_samples'],
    #                         embedding_dim=300).to(DEVICE)
    encoder_model = Encoder(encoder=gconv, augmentor=(aug1, aug2), hidden_dim=hidden_dim, proj_dim=proj_dim,
                            samples_num=graph.num_nodes(),
                            embedding_dim=embedding_dim).to(DEVICE)
    contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=0.2), mode='L2L', intraview_negs=False).to(DEVICE)

    optimizer = Adam(encoder_model.parameters(), lr=0.001)
    # print(encoder_model)
    # print(contrast_model)
    with tqdm(total=num_epochs*len(dataloader), desc='(T)') as pbar:
        for epoch in range(1, num_epochs + 1):
            for input_nodes, output_nodes, blocks in dataloader:
                blocks = [b.to(DEVICE) for b in blocks]
                # print(blocks[0].srcdata)
                # input_features = blocks[0].srcdata['feat']
                # output_labels = blocks[-1].dstdata['label']

                loss = train(encoder_model, contrast_model, blocks, optimizer)
                pbar.set_postfix({'loss': loss})
                pbar.update()
    test_results = {
        'micro_f1':[],
        'macro_f1':[]
    }
    embedding_all = []
    for input_nodes, output_nodes, blocks in test_dataloader:
        blocks = [b.to(DEVICE) for b in blocks]
        test_result, embedding = test(encoder_model, blocks)
        test_results['micro_f1'].append(test_result["micro_f1"])
        test_results['macro_f1'].append(test_result["macro_f1"])
        embedding_all.extend(embedding.cpu().detach().numpy())
    logger.info(f'Embedding shape:{np.array(embedding_all).shape}')

    logger.info(f'(E): Best test F1Mi={np.mean(test_results["micro_f1"]):.4f}, F1Ma={np.mean(test_results["macro_f1"]):.4f}')
    return np.mean(test_results["macro_f1"]), embedding_all


if __name__ == '__main__':

    result = []
    dataset_name = 'arc'
    graph_name = 'syn'
    embedding_all = []
    embedding_save_path = '/mnt/data1/GAS_DATA_WSK/graph_construction_graph_embedding/' \
                          + dataset_name + '_' + graph_name + '_graph_embedding.pkl'
    log_path = '/mnt/data1/GAS_DATA_WSK/graph_construction_graph_embedding/' \
                          + dataset_name + '_' + graph_name + '_graph_embedding_log.log'
    # logging.basicConfig(filename=log_path, level=logging.DEBUG)
    mode = 'w'
    logger = create_logger('log_e', log_path, mode=mode)

    # set_seed(10000)
    for i in range(10):
        logger.info('='*100)
        logger.info('Iteration:{}'.format(i + 1))
        result_test, embedding = main(dataset_name, graph_name, logger)
        # print(torch.seed())
        embedding_all.append(np.array(embedding))
        result.append(result_test)
    index = np.argmax(np.array(result))
    logger.info(f'Max result:{max(result)}')
    logger.info(f'Mean result:{np.mean(result)}')
    logger.info(f'Var of result:{np.var(result)}')
    logger.info(f'Std of result:{np.std(result, ddof=1)}')
    logger.info(f'Max result index:{index}')
    logger.info(f'Result:{result}')
    with open(embedding_save_path, 'wb') as f:
        pkl.dump(embedding_all[index], f)

    # main()

