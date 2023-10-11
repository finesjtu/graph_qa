#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
import pickle
import pickle as pkl
import random
from math import log
import json
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from stanfordcorenlp import StanfordCoreNLP
import scipy.sparse as sp
import re
import dgl


# argmin semeval2016t6
dataset = 'argmin'
path = '../dataset_roberta/'
train_dataset_path = path + dataset + '_train.json'
test_dataset_path = path + dataset + '_test.json'
dev_dataset_path = path + dataset + '_dev.json'
standford_path = './standford_den/' + dataset + '_stan.pkl'
output = './graph_cache/'

nlp = StanfordCoreNLP(r'./stanford-corenlp-full-2018-10-05', lang='en')
stop_words = set(stopwords.words('english'))
word_embeddings_dim = 300
window_size = 7
word_vector_map = {}

# shulffing
doc_train_name_list = []
doc_test_name_list = []
doc_train_list = []
doc_test_list = []

with open('./vocab.txt', 'r') as f:
    vocab_bert = [x.strip() for x in f.readlines()]

def words_tokenize(sentence):
    words = nlp.word_tokenize(sentence)
    word_list = []
    pattern = re.compile('[0-9’!"#$%&\'()*+,\-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~\s]+')
    for word in words:
        word = word.lower()
        if word not in stop_words:
            if not re.findall(pattern, word):
                if word in vocab_bert:
                    word_list.append(word)
    return word_list

print('*'*20, 'data loading....', '*'*20)
train_label_list = []
test_label_list = []
with open(train_dataset_path, 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = json.loads(line)
        data = line['hypothesis'] + ' ' + line['premise']
        doc_train_name_list.append(line)
        train_label_list.append(line['label'])
        # print(data)
        #
        # break
        doc_train_list.append(data)

with open(test_dataset_path, 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = json.loads(line)
        data = line['hypothesis'] + ' ' + line['premise']
        doc_test_name_list.append(line)
        test_label_list.append(line['label'])
        # print(data)
        #
        # break
        doc_test_list.append(data)


vocab_doc_words_list = doc_train_list + doc_test_list

print('*'*20, 'vocab construction....', '*'*20)
# print(vocab_doc_words_list[130])
# build vocab
word_freq = {}
word_set = set()
for doc_words in vocab_doc_words_list:
    # print(doc_words)
    words = words_tokenize(doc_words)
    # print(words)
    for word in words:
        word_set.add(word)
        if word in word_freq:
            word_freq[word] += 1
        else:
            word_freq[word] = 1

vocab = list(word_set)
vocab_size = len(vocab)
print(vocab)
print(vocab_size)
#
# # print(vocab)
print(vocab_size)
vcoab_label_list = [10000]*len(vocab)
print('train_label_list:', len(train_label_list))
print('vcoab_label_list:', len(vcoab_label_list))
print('test_label_list:', len(test_label_list))
label_list = train_label_list + vcoab_label_list + test_label_list
with open(output + dataset + '_label.pkl', 'wb') as f:
    pkl.dump(label_list, f)
print('label_list:',len(label_list))
print('*'*20, 'vocab list constructing....', '*'*20)
word_doc_list = {}

for i in range(len(vocab_doc_words_list)):
    doc_words = vocab_doc_words_list[i]
    # print(type(doc_words))
    words = words_tokenize(doc_words)
    appeared = set()
    for word in words:
        if word in appeared:
            continue
        if word in word_doc_list:
            doc_list = word_doc_list[word]
            doc_list.append(i)
            word_doc_list[word] = doc_list
        else:
            word_doc_list[word] = [i]
        appeared.add(word)
# print(word_doc_list)


word_doc_freq = {}
for word, doc_list in word_doc_list.items():
    word_doc_freq[word] = len(doc_list)

word_id_map = {}
id_word_map = {}
for i in range(vocab_size):
    word_id_map[vocab[i]] = i
    id_word_map[i] = vocab[i]

vocab_str = '\n'.join(vocab)

f = open(output + dataset + '_vocab.txt', 'w')
f.write(vocab_str)
f.close()

'''
Word definitions begin
'''
'''
definitions = []

for word in vocab:
    word = word.strip()
    synsets = wn.synsets(clean_str(word))
    word_defs = []
    for synset in synsets:
        syn_def = synset.definition()
        word_defs.append(syn_def)
    word_des = ' '.join(word_defs)
    if word_des == '':
        word_des = '<PAD>'
    definitions.append(word_des)

string = '\n'.join(definitions)

#
# f = open('data/corpus/' + dataset + '_vocab_def.txt', 'w')
# f.write(string)
# f.close()
#
# tfidf_vec = TfidfVectorizer(max_features=1000)
# tfidf_matrix = tfidf_vec.fit_transform(definitions)
# tfidf_matrix_array = tfidf_matrix.toarray()
# print(tfidf_matrix_array[0], len(tfidf_matrix_array[0]))
#
# word_vectors = []
#
# for i in range(len(vocab)):
#     word = vocab[i]
#     vector = tfidf_matrix_array[i]
#     str_vector = []
#     for j in range(len(vector)):
#         str_vector.append(str(vector[j]))
#     temp = ' '.join(str_vector)
#     word_vector = word + ' ' + temp
#     word_vectors.append(word_vector)
#
# string = '\n'.join(word_vectors)
#
# f = open('data/corpus/' + dataset + '_word_vectors.txt', 'w')
# f.write(string)
# f.close()
#
# word_vector_file = 'data/corpus/' + dataset + '_word_vectors.txt'
# _, embd, word_vector_map = loadWord2Vec(word_vector_file)
# word_embeddings_dim = len(embd[0])
# '''

'''
Word definitions end
'''
print('*'*20, 'label list constructing....', '*'*20)
# label list
label_set = set()
for doc_meta in doc_train_name_list:
    label_set.add(str(doc_meta['label']))
label_list = list(label_set)

label_list_str = '\n'.join(label_list)
f = open(output + dataset + '_labels.txt', 'w')
f.write(label_list_str)
f.close()

'''
构建features
'''

# x: feature vectors of training docs, no initial features
# slect 90% training set

# word_vector_file = input1 + '_word_vectors.txt'
# _, embd, word_vector_map = loadWord2Vec(word_vector_file)
# word_embeddings_dim = len(embd[0])


print('*'*20, 'train data constructing....', '*'*20)
row_x = []
col_x = []
data_x = []
for i in range(len(doc_train_list)):
    doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])
    doc_words = doc_train_list[i]
    words = words_tokenize(doc_words)
    doc_len = len(words)
    for word in words:
        if word in word_vector_map:
            word_vector = word_vector_map[word]
            # print(doc_vec)
            # print(np.array(word_vector))
            doc_vec = doc_vec + np.array(word_vector)

    for j in range(word_embeddings_dim):
        row_x.append(i)
        col_x.append(j)
        # np.random.uniform(-0.25, 0.25)
        data_x.append(doc_vec[j] / doc_len)  # doc_vec[j]/ doc_len

# x = sp.csr_matrix((real_train_size, word_embeddings_dim), dtype=np.float32)
x = sp.csr_matrix((data_x, (row_x, col_x)), shape=(
    len(doc_train_list), word_embeddings_dim))

y = []
for i in range(len(doc_train_list)):
    doc_meta = doc_train_name_list[i]

    label = str(doc_meta['label'])
    one_hot = [0 for l in range(len(label_list))]
    label_index = label_list.index(label)
    one_hot[label_index] = 1
    y.append(one_hot)
y = np.array(y)
# print(y)

print('*'*20, 'test data constructing....', '*'*20)
# tx: feature vectors of test docs, no initial features
test_size = len(doc_test_list)

row_tx = []
col_tx = []
data_tx = []
for i in range(test_size):
    doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])
    doc_words = doc_test_list[i]
    words = words_tokenize(doc_words)
    doc_len = len(words)
    for word in words:
        if word in word_vector_map:
            word_vector = word_vector_map[word]
            doc_vec = doc_vec + np.array(word_vector)

    for j in range(word_embeddings_dim):
        row_tx.append(i)
        col_tx.append(j)
        # np.random.uniform(-0.25, 0.25)
        data_tx.append(doc_vec[j] / doc_len)  # doc_vec[j] / doc_len

# tx = sp.csr_matrix((test_size, word_embeddings_dim), dtype=np.float32)
tx = sp.csr_matrix((data_tx, (row_tx, col_tx)),
                   shape=(test_size, word_embeddings_dim))

ty = []
for i in range(test_size):
    doc_meta = doc_test_name_list[i]
    label = str(doc_meta['label'])
    one_hot = [0 for l in range(len(label_list))]
    label_index = label_list.index(label)
    one_hot[label_index] = 1
    ty.append(one_hot)
ty = np.array(ty)
# print(ty)

# allx: the the feature vectors of both labeled and unlabeled training instances
# (a superset of x)
# unlabeled training instances -> words
print('*'*20, 'train/test data constructing....', '*'*20)
word_vectors = np.random.uniform(-0.01, 0.01,
                                 (vocab_size, word_embeddings_dim))

for i in range(len(vocab)):
    word = vocab[i]
    if word in word_vector_map:
        vector = word_vector_map[word]
        word_vectors[i] = vector

row_allx = []
col_allx = []
data_allx = []

for i in range(len(vocab_doc_words_list)):
    doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])
    doc_words = vocab_doc_words_list[i]
    words = doc_words.split()
    doc_len = len(words)
    for word in words:
        if word in word_vector_map:
            word_vector = word_vector_map[word]
            doc_vec = doc_vec + np.array(word_vector)

    for j in range(word_embeddings_dim):
        row_allx.append(int(i))
        col_allx.append(j)
        # np.random.uniform(-0.25, 0.25)
        data_allx.append(doc_vec[j] / doc_len)  # doc_vec[j]/doc_len
for i in range(vocab_size):
    for j in range(word_embeddings_dim):
        row_allx.append(int(i + len(vocab_doc_words_list)))
        col_allx.append(j)
        data_allx.append(word_vectors.item((i, j)))

row_allx = np.array(row_allx)
col_allx = np.array(col_allx)
data_allx = np.array(data_allx)

allx = sp.csr_matrix(
    (data_allx, (row_allx, col_allx)), shape=(len(vocab_doc_words_list) + vocab_size, word_embeddings_dim))

ally = []
all_name_list = doc_train_name_list + doc_test_name_list
for i in range(len(vocab_doc_words_list)):
    doc_meta = all_name_list[i]

    label = str(doc_meta['label'])
    one_hot = [0 for l in range(len(label_list))]
    label_index = label_list.index(label)
    one_hot[label_index] = 1
    ally.append(one_hot)

for i in range(vocab_size):
    one_hot = [0 for l in range(len(label_list))]
    ally.append(one_hot)

ally = np.array(ally)

print(x.shape, y.shape, tx.shape, ty.shape, allx.shape, ally.shape)
print('*'*20, 'train/test data dumping....', '*'*20)
# dump objects
f = open(output + "/ind.{}.x".format(dataset), 'wb')
pkl.dump(x, f)
f.close()

f = open(output + "/ind.{}.y".format(dataset), 'wb')
pkl.dump(y, f)
f.close()

f = open(output + "/ind.{}.tx".format(dataset), 'wb')
pkl.dump(tx, f)
f.close()

f = open(output + "/ind.{}.ty".format(dataset), 'wb')
pkl.dump(ty, f)
f.close()

f = open(output + "/ind.{}.allx".format(dataset), 'wb')
pkl.dump(allx, f)
f.close()

f = open(output + "/ind.{}.ally".format(dataset), 'wb')
pkl.dump(ally, f)
f.close()
#
'''
Doc word heterogeneous graph 1
'''
print('*'*20, 'begin constructing graph......','*'*20)
# word co-occurence with context windows
windows = []

for doc_words in vocab_doc_words_list:
    words = words_tokenize(doc_words)
    length = len(words)
    if length <= window_size:
        windows.append(words)
    else:
        # print(length, length - window_size + 1)
        for j in range(length - window_size + 1):
            window = words[j: j + window_size]
            windows.append(window)
            # print(window)

word_window_freq = {}
for window in windows:
    appeared = set()
    for i in range(len(window)):
        if window[i] in appeared:
            continue
        if window[i] in word_window_freq:
            word_window_freq[window[i]] += 1
        else:
            word_window_freq[window[i]] = 1
        appeared.add(window[i])

word_pair_count = {}
for window in windows:
    for i in range(1, len(window)):
        for j in range(0, i):
            word_i = window[i]
            word_i_id = word_id_map[word_i]
            word_j = window[j]
            word_j_id = word_id_map[word_j]
            if word_i_id == word_j_id:
                continue
            word_pair_str = str(word_i_id) + ',' + str(word_j_id)
            if word_pair_str in word_pair_count:
                word_pair_count[word_pair_str] += 1
            else:
                word_pair_count[word_pair_str] = 1
            # two orders
            word_pair_str = str(word_j_id) + ',' + str(word_i_id)
            if word_pair_str in word_pair_count:
                word_pair_count[word_pair_str] += 1
            else:
                word_pair_count[word_pair_str] = 1

row = []
col = []
weight = []
weight1 = []
weight2 = []

# 根据stanford句法依存构建边权重
data1 = pickle.load(open(standford_path, "rb"))
max_count1 = 0.0
min_count1 = 0.0
count1 = []
for key in data1:
    if data1[key] > max_count1:
        max_count1 = data1[key]
    if data1[key] < min_count1:
        min_count1 = data1[key]
    count1.append(data1[key])
count_mean1 = np.mean(count1)
count_var1 = np.var(count1)
count_std1 = np.std(count1, ddof=1)

# # 根据语义依存构建边权重
# data2 = pickle.load(open(input4 + "_semantic_0.05.pkl", "rb"))
# max_count2 = 0.0
# min_count2 = 0.0
# count2 = []
# for key in data2:
#     if data2[key] > max_count2:
#         max_count2 = data2[key]
#     if data2[key] < min_count2:
#         min_count2 = data2[key]
#     count2.append(data2[key])
# count_mean2 = np.mean(count2)
# count_var2 = np.var(count2)
# count_std2 = np.std(count2, ddof=1)
#
# compute weights
# 没有权重的单词对直接用PMI作为权重
num_window = len(windows)
for key in word_pair_count:
    temp = key.split(',')
    i = int(temp[0])
    j = int(temp[1])
    count = word_pair_count[key]
    word_freq_i = word_window_freq[vocab[i]]
    word_freq_j = word_window_freq[vocab[j]]
    pmi = log((1.0 * count / num_window) /
              (1.0 * word_freq_i * word_freq_j / (num_window * num_window)))
    if pmi <= 0:
        continue
    # pmi
    row.append(len(doc_train_name_list) + i)
    col.append(len(doc_train_name_list) + j)
    weight.append(pmi)
    # 句法依存
    if i not in id_word_map or j not in id_word_map:
        continue
    newkey = id_word_map[i] + ',' + id_word_map[j]
    if newkey in data1:
        # min-max标准化
        wei = (data1[newkey] - min_count1) / (max_count1 - min_count1)
        # 0均值标准化
        # wei = (data1[key]-count_mean1)/ count_std1
        # 出现频度比例，出现1的时候比较多
        # wei = data1[key] / data2[key]
        weight1.append(wei)
    else:
        weight1.append(pmi)
    # 语义依存
    # if newkey in data2:
    #     # min-max标准化
    #     wei = (data2[newkey] - min_count2) / (max_count2 - min_count2)
    #     # 0均值标准化
    #     # wei = (data2[key]-count_mean2)/ count_std2
    #     # 出现频度比例，出现1的时候比较多
    #     # wei = data2[key] / data2[key]
    #     weight2.append(wei)
    # else:
    #     weight2.append(pmi)

'''
for i in range(vocab_size):
    for j in range(vocab_size):
        if vocab[i] in word_vector_map and vocab[j] in word_vector_map:
            vector_i = np.array(word_vector_map[vocab[i]])
            vector_j = np.array(word_vector_map[vocab[j]])
            similarity = 1.0 - cosine(vector_i, vector_j)
            if similarity > 0.9:
                print(vocab[i], vocab[j], similarity)
                row.append(train_size + i)
                col.append(train_size + j)
                weight.append(similarity)
'''
# doc word frequency
weight_tfidf = []
doc_word_freq = {}
for doc_id in range(len(vocab_doc_words_list)):
    doc_words = vocab_doc_words_list[doc_id]
    words = words_tokenize(doc_words)
    for word in words:
        word_id = word_id_map[word]
        doc_word_str = str(doc_id) + ',' + str(word_id)
        if doc_word_str in doc_word_freq:
            doc_word_freq[doc_word_str] += 1
        else:
            doc_word_freq[doc_word_str] = 1
train_size = len(doc_train_name_list)
for i in range(len(vocab_doc_words_list)):
    doc_words = vocab_doc_words_list[i]
    words = words_tokenize(doc_words)
    doc_word_set = set()
    for word in words:
        if word in doc_word_set:
            continue
        j = word_id_map[word]
        key = str(i) + ',' + str(j)
        freq = doc_word_freq[key]
        if i < train_size:
            row.append(i)
        else:
            row.append(i + vocab_size)
        col.append(train_size + j)
        idf = log(1.0 * len(vocab_doc_words_list) /
                  word_doc_freq[vocab[j]])
        weight_tfidf.append(freq * idf)
        doc_word_set.add(word)

weight = weight + weight_tfidf
print('train size:{}, vocab size:{}, test size:{}'.format(train_size, vocab_size, test_size))
print('train idx:{}, test idx:{}-{}'. format(train_size, train_size + vocab_size, train_size + vocab_size + test_size))
# print()
node_size = train_size + vocab_size + test_size
adj = sp.csr_matrix(
    (weight, (row, col)), shape=(node_size, node_size))
adj_list = [[row[i], col[i]] for i in range(len(row))]

# dump objects
f = open(output + '/ind.{}.seq.adj'.format(dataset), 'wb')
pkl.dump(adj, f)
f.close()

f = open(output + '/ind.{}.seq.adj.list'.format(dataset), 'wb')
pkl.dump(adj_list, f)
f.close()
print('The max of weight_seq:',max(weight))
print('The max of weighttfidf:',max(weight_tfidf))
print('The min of weight_seq:',min(weight))
print('The min of weighttfidf:',min(weight_tfidf))
print('词频构图1完成')

weight = weight1 + weight_tfidf
node_size = train_size + vocab_size + test_size
adj = sp.csr_matrix(
    (weight, (row, col)), shape=(node_size, node_size))

adj_list = [[row[i], col[i]] for i in range(len(row))]
# dump objects
f = open(output + '/ind.{}.syn.adj'.format(dataset), 'wb')
pkl.dump(adj, f)
f.close()

f = open(output + '/ind.{}.syn.adj.list'.format(dataset), 'wb')
pkl.dump(adj_list, f)
f.close()

print('The max of weight_syn:',max(weight1))
print('The max of weighttfidf:',max(weight_tfidf))
print('The min of weight_syn:',min(weight1))
print('The min of weighttfidf:',min(weight_tfidf))
print('语法构图2完成')
print('Dataset is :{}'.format(dataset))
with open('./matrix.pkl', 'wb') as f:
    pkl.dump([weight, row, col, node_size], f)

# graph = dgl.from_scipy(adj, eweight_name='weight')
# print(graph)
# train_nids = list(range(512))
# sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
# dataloader = dgl.dataloading.NodeDataLoader(
#     graph, train_nids, sampler,
#     batch_size=256,
#     shuffle=True,
#     drop_last=False)
#
# for i in range(3):
#     input_nodes, output_nodes, blocks = next(iter(dataloader))
#     # print(input_nodes)
#     # print(output_nodes)
#     print(blocks)
# weight = weight2 + weight_tfidf
# node_size = train_size + vocab_size + test_size
# adj = sp.csr_matrix(
#     (weight, (row, col)), shape=(node_size, node_size))
#
# # dump objects
# f = open(output2 + '/ind.{}.adj2'.format(dataset), 'wb')
# pkl.dump(adj, f)
# f.close()
#
# print('构图3完成')
