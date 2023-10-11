import torch
import dgl.nn as dglnn
import dgl
import pickle as pkl
import numpy as np
from GTP.graph_construction.log_wrapper import create_logger
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from time import time
import os
import json
#
#TODO: GraphDataW2v增加graph embedding的数据加载

class GraphData(Dataset):
    def __init__(self, dataset, dataset_split=None, logger=None, nei_save=True):
        super(GraphData, self).__init__()
        self.dataset = dataset
        if logger is None:
            self.logger = create_logger('test', './testlog.log')
        else:
            self.logger = logger
        self.path = '/mnt/data1/GAS_DATA_WSK/graph_construction_pretrained_model/lm_training/' + self.dataset + '/'
        self.semantic_embedding_path = self.path + 'embedding.pkl'
        self.graph_SYN_embedding_path = '/mnt/data1/GAS_DATA_WSK/graph_construction_graph_embedding/' + self.dataset + '_syn_graph_embedding.pkl'
        self.graph_SEQ_embedding_path = '/mnt/data1/GAS_DATA_WSK/graph_construction_graph_embedding/' + self.dataset + '_seq_graph_embedding.pkl'
        self.neighbour_save_path = '/mnt/data1/GAS_DATA_WSK/lda_edge/' + self.dataset + '/neighbour.pkl'
        self.edge_path = '/mnt/data1/GAS_DATA_WSK/lda_edge/' + self.dataset + '/similarity.pkl'
        self.dataset_split = dataset_split
        if nei_save:
            self.save_neighbour()
        self.LDA_data = self.pkl_load(self.edge_path).todense()
        self.semantic_embedding_data, self.graph_SEQ_embedding_data, self.graph_SYN_embedding_data,\
        self.label, self.neighbour_index, self.ratio = self.get_split_data()
        self.all_semantic_embedding_dict, self.all_graph_SEQ_embedding_data, self.all_graph_SYN_embedding_data = self.embedding_data_load()
        self.length = len(self.label)

        # self.semantic_embedding_dict, self.graph_SEQ_embedding_data, self.graph_SYN_embedding_data = self.embedding_data_load()

    def get_split_data(self):
        neighbour_index, ratio = self.load_neighbour()
        semantic_embedding_dict, graph_SEQ_embedding_data, graph_SYN_embedding_data = self.embedding_data_load()
        # start = time()

        # print('neigh data load:{}s'.format(time()-start))
        mask = semantic_embedding_dict['mask']
        self.train_count = sum([1 if x == True else 0 for x in mask])
        if self.dataset_split == 'train':
            mask = mask
            return semantic_embedding_dict['embedding'][mask], graph_SEQ_embedding_data[mask], \
                   graph_SYN_embedding_data[mask], np.array(semantic_embedding_dict['label'])[mask], np.array(neighbour_index)[mask],\
                   np.array(ratio)[mask]
        elif self.dataset_split == 'test':
            mask = np.logical_not(mask)
            return semantic_embedding_dict['embedding'][mask], graph_SEQ_embedding_data[mask], \
                   graph_SYN_embedding_data[mask], np.array(semantic_embedding_dict['label'])[mask], np.array(neighbour_index)[mask],\
                   np.array(ratio)[mask]
        else:
            return semantic_embedding_dict['embedding'], graph_SEQ_embedding_data, \
                   graph_SYN_embedding_data, np.array(semantic_embedding_dict['label']), np.array(neighbour_index),\
                   np.array(ratio)[mask]

    def get_neighbour(self, index):
        neighbour, ratio = self.load_neighbour()
        semantic_embedding_dict, graph_SEQ_embedding_data, graph_SYN_embedding_data = self.embedding_data_load()
        semantic_nei_data = semantic_embedding_dict['embedding'][neighbour[index]]
        graph_SEQ_nei_data = graph_SEQ_embedding_data[neighbour[index]]
        graph_SYN_nei_data = graph_SYN_embedding_data[neighbour[index]]
        return semantic_nei_data, graph_SEQ_nei_data, graph_SYN_nei_data

    def __getitem__(self, index):
        # semantic_embedding_data, graph_SEQ_embedding_data, graph_SYN_embedding_data, label, \
        # semantic_nei_data, graph_SEQ_nei_data, graph_SYN_nei_data = self.get_split_data(index)
        # semantic_nei_data, graph_SEQ_nei_data, graph_SYN_nei_data = self.get_neighbour(index)
        # print(semantic_nei_data.shape)
        # print(graph_SYN_nei_data.shape)
        # print(graph_SEQ_nei_data.shape)
        # print(semantic_embedding_data[index].reshape(1,-1).shape)
        # print(np.concatenate((semantic_embedding_data[index].reshape(1,-1), semantic_nei_data), axis=0).shape)

        neighbour_index = self.neighbour_index[index]
        # if self.dataset_split == 'test':
        #     test_index = index + self.train_count
        #(6, 1024)
        # (6, 512)
        # (6, 512)
        # (1, 1024)
        # (7, 1024)
        # print(index)

        # print(index)



        return {'semantic_data':self.all_semantic_embedding_dict['embedding'][neighbour_index],
                'graph_seq_data':self.all_graph_SEQ_embedding_data[neighbour_index],
                'graph_syn_data':self.all_graph_SYN_embedding_data[neighbour_index],
                'label':self.label[index],
                'index':index,
                'neighbour':self.neighbour_index[index],
                'ratio': self.ratio[index]
                }

    def __len__(self):
        length = self.get_len()
        return length

    def pkl_load(self, path):
        with open(path, 'rb') as f:
            data = pkl.load(f)
        return data

    def embedding_data_load(self):
        semantic_embedding_data = self.pkl_load(self.semantic_embedding_path)
        graph_SEQ_embedding_data = self.pkl_load(self.graph_SEQ_embedding_path)
        graph_SYN_embedding_data = self.pkl_load(self.graph_SYN_embedding_path)
        # print(semantic_embedding_data)
        return semantic_embedding_data, np.array(graph_SEQ_embedding_data), np.array(graph_SYN_embedding_data)

    def get_len(self):
        semantic_embedding_data, _,_ = self.embedding_data_load()
        mask = semantic_embedding_data['mask']
        mask_co = np.zeros(len(mask))
        length = 0
        if self.dataset_split == 'train':
            mask = mask
            length = sum(mask + mask_co)
        elif self.dataset_split == 'test':
            mask = np.logical_not(mask)
            length = sum(mask + mask_co)
        else:
            length = len(mask)
        return int(length)


    def edge_data_load(self):
        edge_data = self.pkl_load(self.edge_path)
        return edge_data

    def get_embedding_data_info(self):
        semantic_embedding_data, graph_SEQ_embedding_data, graph_SYN_embedding_data = self.embedding_data_load()
        self.logger.info(f"Semantic Embedding shape is:{semantic_embedding_data['embedding'].shape}")
        self.logger.info(f"Graph SEQ Embedding shape is:{np.array(graph_SEQ_embedding_data).shape}")
        self.logger.info(f"Graph SYN Embedding shape is:{np.array(graph_SYN_embedding_data).shape}")
        self.logger.info(f"Label length is:{len(semantic_embedding_data['label'])}")

    def embedding_incorprate(self, semantic_embedding, graph_syn_embedding, graph_seq_embedding, incorp_mode='add'):
        self.logger.info(f"The shape of semantic_embedding is {semantic_embedding.shape}")
        self.logger.info(f"The shape of graph_syn_embedding is {graph_syn_embedding.shape}")
        self.logger.info(f"The shape of graph_seq_embedding is {graph_seq_embedding.shape}")
        if incorp_mode == 'add':
            graph_embedding = np.concatenate((graph_seq_embedding, graph_syn_embedding), axis=1)
            if graph_embedding.shape == semantic_embedding.shape:
                embedding = semantic_embedding + graph_embedding
                self.logger.info(f"The shape of Embedding is {embedding.shape}")
            else:
                self.logger.warning(f"The dimensions of two matrix are not matched!")
                return None
        return embedding


    def neighbor_index_get(self, edge_data, neighbor_num=8):
        matrix = edge_data.toarray()
        neighbors = []
        ratios = []
        for i in matrix:
            index = i.argsort()[-neighbor_num:][:]
            neighbors.append(index)
            # ratio = F.softmax(torch.tensor(i[index]), dim=0)
            ratio = torch.tensor(i[index])
            # print(ratio)
            ratios.append(ratio.view(neighbor_num, -1))

        return neighbors, ratios

    def save_neighbour(self):
        edge_data = self.edge_data_load()
        print('Getting Neighbour Information and Save it!')
        neighbours, ratios = self.neighbor_index_get(edge_data)
        neighbour_info = {
            'neighbour': neighbours,
            'ratio': ratios
        }
        with open(self.neighbour_save_path, 'wb') as f:
            pkl.dump(neighbour_info, f)
        # if not os.path.exists(self.neighbour_save_path):
        #     edge_data = self.edge_data_load()
        #     print('Getting Neighbour Information and Save it!')
        #     neighbours, ratios = self.neighbor_index_get(edge_data)
        #     neighbour_info = {
        #         'neighbour': neighbours,
        #         'ratio': ratios
        #     }
        #     with open(self.neighbour_save_path, 'wb') as f:
        #         pkl.dump(neighbour_info, f)
        # else:
        #     Warning('There is already neighbour information. '
        #             'If you wanna process new information, please delete the information first!')

    def load_neighbour(self):
        if not os.path.exists(self.neighbour_save_path):
            Warning('There is no neighbour information, please use save_neighbour() first!')
        else:
            neighbour_info = self.pkl_load(self.neighbour_save_path)
            return neighbour_info['neighbour'], neighbour_info['ratio']

    def graph_data_process(self):
        edge_data = self.edge_data_load()
        graph = dgl.from_scipy(edge_data, eweight_name='weight')
        graph = dgl.add_self_loop(graph)
        self.logger.info(graph)
        self.logger.info(graph.num_edges())
        semantic_embedding_data, graph_SEQ_embedding_data, graph_SYN_embedding_data = self.embedding_data_load()
        label_data = semantic_embedding_data['label']
        embedding = self.embedding_incorprate(semantic_embedding_data['embedding'], graph_SYN_embedding_data,
                                              graph_SEQ_embedding_data)
        train_test_mask = semantic_embedding_data['mask']
        train_mask = train_test_mask
        test_mask = [x==False for x in train_mask]
        test_index, train_index = [], []
        [train_index.append(i) if x==True else test_index.append(i) for i,x in enumerate(train_mask)]
        graph.ndata['label'] = torch.tensor(np.array(label_data))
        graph.ndata['train_mask'] = torch.tensor(np.array(train_mask))
        graph.ndata['test_mask'] = torch.tensor(np.array(test_mask))
        graph.ndata['val_mask'] = torch.tensor(np.array(test_mask))

        graph.ndata['feat'] = torch.FloatTensor(embedding)
        graph.ndata['x'] = torch.tensor(np.arange(len(label_data)))
        node_labels = graph.ndata['label']

        n_labels = int(node_labels.max().item() + 1)
        return graph, label_data, n_labels, train_index, test_index

    def sequence_data_process(self):
        edge_data = self.edge_data_load()
        semantic_embedding_data, graph_SEQ_embedding_data, graph_SYN_embedding_data =self.embedding_data_load()
        embedding = self.embedding_incorprate(semantic_embedding_data['embedding'], graph_SYN_embedding_data,
                                              graph_SEQ_embedding_data)
        incor_embedding = []
        neighbors, ratios = self.neighbor_index_get(edge_data)
        for i, neigh in tqdm(enumerate(neighbors)):
            ratio = np.array(ratios[i])
            embedding_neigh = embedding[neigh]

            neigh_sum = np.sum(embedding_neigh * ratio, axis=0)
            incor_embedding.append(neigh_sum)
        return np.array(incor_embedding)

    def get_semantic_data(self):
        semantic_embedding_data = self.pkl_load(self.semantic_embedding_path)
        return semantic_embedding_data['embedding']

    def get_graph_data(self):
        semantic_embedding_data, graph_SEQ_embedding_data, graph_SYN_embedding_data = self.embedding_data_load()
        graph_embedding = np.concatenate((graph_SEQ_embedding_data, graph_SYN_embedding_data), axis=1)
        return graph_embedding
    def get_graph_seq_data(self):
        graph_SEQ_embedding_data = self.pkl_load(self.graph_SEQ_embedding_path)
        graph_embedding = np.array(graph_SEQ_embedding_data)
        return graph_embedding
    def get_graph_syn_data(self):
        graph_SYN_embedding_data = self.pkl_load(self.graph_SYN_embedding_path)
        graph_embedding = np.array(graph_SYN_embedding_data)
        return graph_embedding

class GraphDataTest(Dataset):
    def __init__(self, dataset, dataset_split=None):
        super(GraphDataTest, self).__init__()
        self.dataset = dataset

        self.path = '/mnt/data1/GAS_DATA_WSK/graph_construction_pretrained_model/lm_training/' + self.dataset + '/'
        self.semantic_embedding_path = self.path + 'embedding.pkl'
        self.graph_SYN_embedding_path = '/mnt/data1/GAS_DATA_WSK/graph_construction_graph_embedding/' + self.dataset + '_syn_graph_embedding.pkl'
        self.graph_SEQ_embedding_path = '/mnt/data1/GAS_DATA_WSK/graph_construction_graph_embedding/' + self.dataset + '_seq_graph_embedding.pkl'
        self.neighbour_save_path = '/mnt/data1/GAS_DATA_WSK/lda_edge/' + self.dataset + '/neighbour.pkl'
        self.edge_path = '/mnt/data1/GAS_DATA_WSK/lda_edge/' + self.dataset + '/similarity.pkl'
        self.dataset_split = dataset_split

        self.semantic_embedding_data, self.label = self.get_split_data()
        self.length = len(self.label)


    def get_split_data(self):
        semantic_embedding_dict = self.embedding_data_load()
        mask = semantic_embedding_dict['mask']
        if self.dataset_split == 'train':
            mask = mask
            return semantic_embedding_dict['embedding'][mask], np.array(semantic_embedding_dict['label'])[mask]
        elif self.dataset_split == 'test':
            mask = np.logical_not(mask)
            return semantic_embedding_dict['embedding'][mask], np.array(semantic_embedding_dict['label'])[mask]
        else:
            return semantic_embedding_dict['embedding'], np.array(semantic_embedding_dict['label'])[mask]

    def get_neighbour(self, index):
        neighbour, ratio = self.load_neighbour()
        semantic_embedding_dict = self.embedding_data_load()
        semantic_nei_data = semantic_embedding_dict['embedding'][neighbour[index]]

        return semantic_nei_data

    def save_neighbour(self):
        if not os.path.exists(self.neighbour_save_path):
            edge_data = self.edge_data_load()
            print('Getting Neighbour Information and Save it!')
            neighbours, ratios = self.neighbor_index_get(edge_data)
            neighbour_info = {
                'neighbour': neighbours,
                'ratio': ratios
            }
            with open(self.neighbour_save_path, 'wb') as f:
                pkl.dump(neighbour_info, f)
        else:
            Warning('There is already neighbour information. '
                    'If you wanna process new information, please delete the information first!')

    def load_neighbour(self):
        if not os.path.exists(self.neighbour_save_path):
            Warning('There is no neighbour information, please use save_neighbour() first!')
        else:
            neighbour_info = self.pkl_load(self.neighbour_save_path)
            return neighbour_info['neighbour'], neighbour_info['ratio']

    def __getitem__(self, index):

        # semantic_nei_data = self.get_neighbour(index)

        return {'semantic_data':self.semantic_embedding_data[index],
                'label':self.label[index],
                'index':index}


    def __len__(self):

        return self.length

    def pkl_load(self, path):
        with open(path, 'rb') as f:
            data = pkl.load(f)
        return data

    def embedding_data_load(self):
        semantic_embedding_data = self.pkl_load(self.semantic_embedding_path)
        return semantic_embedding_data

    def get_len(self):
        semantic_embedding_data = self.embedding_data_load()
        mask = semantic_embedding_data['mask']
        mask_co = np.zeros(len(mask))
        if self.dataset_split == 'train':
            mask = mask
            length = sum(mask + mask_co)
        elif self.dataset_split == 'test':
            mask = np.logical_not(mask)
            length = sum(mask + mask_co)
        else:
            length = len(mask)
        return int(length)

    def edge_data_load(self):
        edge_data = self.pkl_load(self.edge_path)
        return edge_data

    def neighbor_index_get(self, edge_data, neighbor_num=7):
        matrix = edge_data.toarray()
        neighbors = []
        ratios = []
        for i in matrix:
            index = i.argsort()[-neighbor_num:][:-1]
            neighbors.append(index)
            ratio = F.softmax(torch.tensor(i[index]), dim=0)
            # print(ratio)
            ratios.append(ratio.view(neighbor_num - 1, -1))

        return neighbors, ratios

class GraphDataW2v(Dataset):
    def __init__(self, dataset, data_split, neighbor_num=1):
        super(GraphDataW2v, self).__init__()
        self.dataset = dataset
        self.neighbor_num = neighbor_num
        self.path = '../dataset_roberta/'
        self.w2v_path = '../graph_construction/graph_cache/word2vec.pkl'
        self.train_corpus_path = self.path + dataset + '_train.json'
        self.test_corpus_path = self.path + dataset + '_test.json'
        self.w2v_dict = self.pkl_load(self.w2v_path)
        self.train_size = self.get_train_size()
        if data_split == 'train':
            self.corpus_data, self.label = self.get_corpus_data(self.train_corpus_path)
        elif data_split == 'test':
            self.corpus_data, self.label = self.get_corpus_data(self.test_corpus_path)
        self.data_split = data_split

        self.graph_SYN_embedding_path = '/mnt/data1/GAS_DATA_WSK/graph_construction_graph_embedding/' + self.dataset + '_syn_graph_embedding.pkl'
        self.graph_SEQ_embedding_path = '/mnt/data1/GAS_DATA_WSK/graph_construction_graph_embedding/' + self.dataset + '_seq_graph_embedding.pkl'
        self.neighbour_save_path = '/mnt/data1/GAS_DATA_WSK/lda_edge/' + self.dataset + '/neighbour.pkl'
        self.edge_path = '/mnt/data1/GAS_DATA_WSK/lda_edge/' + self.dataset + '/similarity.pkl'
        self.save_neighbour()
        self.data_split = data_split
        self.graph_SYN_data, self.graph_SEQ_data = self.get_graph_data()
        print(len(self.graph_SYN_data))
        self.neighbor_index, self.ratio = self.load_neighbour()


    def pkl_load(self, path):
        with open(path, 'rb') as f:
            data = pkl.load(f)
        return data

    def get_graph_data(self):
        graph_syn_data = self.pkl_load(self.graph_SYN_embedding_path)
        graph_seq_data = self.pkl_load(self.graph_SEQ_embedding_path)

        return graph_syn_data, graph_seq_data

    #
    def neighbor_index_get(self, edge_data):
        matrix = edge_data.toarray()
        neighbor_num = self.neighbor_num
        neighbors = []
        ratios = []
        for i in matrix:
            index = i.argsort()[-neighbor_num:][:][::-1]
            # print(i)

            neighbors.append(index)
            ratios_meta = []
            for j in index:
                # neighbor_neighbor =
                neighbor_ratio = matrix[j][index]

                # ratio = F.softmax(torch.tensor(i[index]), dim=0)
                ratio = neighbor_ratio
                # print(ratio)
                ratios_meta.append(ratio)
            ratios.append(torch.tensor(np.array(ratios_meta)))
            # print(index)
        return neighbors, ratios

    def save_neighbour(self):
        edge_data = self.pkl_load(self.edge_path)
        print('Getting Neighbour Information and Save it!')
        neighbours, ratios = self.neighbor_index_get(edge_data)
        neighbour_info = {
            'neighbour': neighbours,
            'ratio': ratios
        }
        with open(self.neighbour_save_path, 'wb') as f:
            pkl.dump(neighbour_info, f)


    def load_neighbour(self):
        if not os.path.exists(self.neighbour_save_path):
            Warning('There is no neighbour information, please use save_neighbour() first!')
        else:
            neighbour_info = self.pkl_load(self.neighbour_save_path)
            return neighbour_info['neighbour'], neighbour_info['ratio']


    def __getitem__(self, index):
        semantic_text = self.corpus_data[index]
        label = np.array(self.label[index])
        # print(index)
        # print(self.train_size)
        if self.data_split == 'test':
            index = index + self.train_size
        neighbor_index, neighbor_ratio = self.neighbor_index[index], self.ratio[index]
        # print(neighbor_index)
        # print(index)
        # print(type(self.graph_SYN_data))
        # print(self.graph_SYN_data.shape)
        # print(self.graph_SYN_data)


        return {
            'semantic_data':semantic_text,
            'graph_syn_data':self.graph_SYN_data[neighbor_index],
            'graph_seq_data':self.graph_SEQ_data[neighbor_index],
            'label':label,
            'neighbor_index':neighbor_index,
            'neighbor_ratio': neighbor_ratio
        }
    def __len__(self):
        return len(self.label)

    def pkl_load(self, path):
        with open(path, 'rb') as f:
            data = pkl.load(f)
        return data

    def get_train_size(self):
        with open(self.train_corpus_path, 'r') as f:
            lines = f.readlines()
            train_size = len(lines)
        return train_size

    def get_corpus_data(self, path):
        train_corpus = []
        train_label = []
        with open(path, 'r') as f:
            lines = f.readlines()
            for line in tqdm(lines):
                line = json.loads(line)
                data = line['hypothesis'] + ' ' + line['premise']
                # data = data.split(' ')
                # trunc_data = self.get_trunc(data)
                train_corpus.append(data)
                train_label.append(line['label'])
        return train_corpus, train_label


class GraphDataWordEmbedding(Dataset):
    def __init__(self, dataset, data_split):
        super(GraphDataWordEmbedding, self).__init__()
        self.path = '../dataset_roberta/'
        self.w2v_path = '../graph_construction/graph_cache/word2vec.pkl'
        self.train_corpus_path = self.path + dataset + '_train.json'
        self.test_corpus_path = self.path + dataset + '_test.json'
        self.w2v_dict = self.pkl_load(self.w2v_path)
        if data_split == 'train':
            self.corpus_data, self.label = self.get_corpus_data(self.train_corpus_path)
        elif data_split == 'test':
            self.corpus_data, self.label = self.get_corpus_data(self.test_corpus_path)
        self.data_split = data_split

    def __getitem__(self, index):

        label = np.array(self.label[index])
        return {
            'semantic_data':self.corpus_data[index],
            'label':label
        }
    def __len__(self):
        return len(self.label)

    def pkl_load(self, path):
        with open(path, 'rb') as f:
            data = pkl.load(f)
        return data

    def get_corpus_data(self, path):
        train_corpus = []
        train_label = []
        with open(path, 'r') as f:
            lines = f.readlines()
            for line in tqdm(lines):
                line = json.loads(line)
                data = line['hypothesis'] + ' ' + line['premise']

                train_corpus.append(data)
                train_label.append(line['label'])
        return train_corpus, train_label

    def get_trunc(self, data, length=128):
        data_meta = []
        for i in data:
            if i in self.w2v_dict:
                data_meta.append(i)
        if len(data_meta) >= length:
            trunc_data = data_meta[:length]
        elif len(data_meta) < length:
            for i in range(length- len(data_meta)):
                data_meta.append('.')
                trunc_data = data_meta
        return trunc_data

if __name__ == '__main__':
    dataset = GraphDataW2v('semeval2016t6', 'train')
    data = dataset.__getitem__(200)
    print(data)

