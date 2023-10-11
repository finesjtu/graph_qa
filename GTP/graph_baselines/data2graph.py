import os
import random
import numpy as np
import pickle as pkl
import scipy.sparse as sp
import sys
from tqdm import tqdm
import csv
import re
# 把数据集处理成每个文档一个图

datasets = ['mr', 'ohsumed', 'R8', 'R52', 'TREC', 'ag_news', 'WebKB', 'SST1', 'SST2', 'D4']

dataset = 'mr'
if dataset not in datasets:
    sys.exit("wrong dataset name")

try:
    window_size = int(sys.argv[2])
except:
    window_size = 3
    print('using default window size = 3')

try:
    weighted_graph = bool(sys.argv[3])
except:
    weighted_graph = False
    print('using default unweighted graph')

truncate = False # whether to truncate long document
MAX_TRUNC_LEN = 350


print('loading raw data')
print('*'*100)
# load pre-trained word embeddings
word_embeddings_dim = 300
word_embeddings = {}
if sys.argv[2] == 'CN':
    with open('../data/renmin.char', 'r') as f:
        for line in f.readlines():
            data = line.split()
            word_embeddings[str(data[0])] = list(map(float,data[1:]))
elif sys.argv[2] == 'EN':
    with open('../glove.6B/glove.6B.' + str(word_embeddings_dim) + 'd.txt', 'r') as f:
        for line in f.readlines():
            data = line.split()
            word_embeddings[str(data[0])] = list(map(float,data[1:]))
# print(word_embeddings['我'])

# load document list
doc_name_list = []
doc_train_list = []
doc_test_list = []
print('get corpus labels……')
print('*'*100)
#读取语料的标注信息

if sys.argv[2] == 'EN':
    with open('../data/GNNDATA/' + dataset + '.txt', 'r') as f:
        for line in f.readlines():
            doc_name_list.append(line.strip())
            temp = line.split("\t")
            # print(temp)

            if temp[1].find('test') != -1:
                doc_test_list.append(line.strip())
            elif temp[1].find('train') != -1:
                doc_train_list.append(line.strip())
elif sys.argv[2] == 'CN':
    f = open('../data/civil_difficulty_test.csv', 'r', encoding='utf-8')
    test = csv.reader(f)
    for number, content in enumerate(test):
        doc_test_list.append('{}\ttest\t{}'.format(number, content[0]))
        doc_name_list.append('{}\ttest\t{}'.format(number, content[0]))
    # print(doc_test_list)
    f.close()
    f = open('../data/civil_difficulty_train.csv', 'r', encoding='utf-8')
    train = csv.reader(f)
    len_test = len(doc_test_list)
    for number, content in enumerate(train):
        doc_train_list.append('{}\ttrain\t{}'.format(number + len_test, content[0]))
        doc_name_list.append('{}\ttrain\t{}'.format(number + len_test, content[0]))
    # print(doc_train_list)
    f.close()



# load raw text
doc_content_list = []
if sys.argv[2] == 'EN':
    with open('../data/GNNDATA/corpus/' + dataset + '.clean.txt', 'r') as f:
        for line in f.readlines():
            doc_content_list.append(line.strip())
elif sys.argv[2] == 'CN':
    with open('../data/civil_difficulty_test.csv', 'r') as f:
        csv_reader = csv.reader(f)
        for i in csv_reader:
            doc_content_list.append(i[1])
    f.close()
    with open('../data/civil_difficulty_train.csv', 'r') as f:
        csv_reader = csv.reader(f)
        for i in csv_reader:
            doc_content_list.append(i[1])
    f.close()



# map and shuffle
train_ids = []
for train_name in doc_train_list:

    train_id = doc_train_list.index(train_name)
    train_ids.append(train_id)
    # print(train_name, train_id)
    # break


test_ids = []
for test_name in doc_test_list:
    test_id = doc_test_list.index(test_name)
    test_ids.append(test_id)

# print(test_ids)
ids = train_ids + test_ids


process_doc_name_list = []
process_doc_words_list = []
for i in ids:
    # print(i)
    process_doc_name_list.append(doc_name_list[int(i)])
    process_doc_words_list.append(doc_content_list[int(i)])

print('build corpus vocabulary……')
print('*'*100)
# build corpus vocabulary
word_set = set()
# 借助于set构建单词表
vocab = []
if sys.argv[2] == 'EN':
    for doc_words in process_doc_words_list:
        words = doc_words.split()
        word_set.update(words)

    vocab = list(word_set)
    vocab_size = len(vocab)

    word_id_map = {}
    for i in range(vocab_size):
        word_id_map[vocab[i]] = i
elif sys.argv[2] == 'CN':

    for doc_words in process_doc_words_list:
        # clean_words = []
        words = jieba.cut(doc_words)
        words = list(words)
        isnum = re.compile(r'^(\-|\+)?\d+(\.\d+)?$')

        for i in words:
            if isnum.findall(i):
                # print(i)
                words.remove(i)

        word_set.update(words)
        # print(word_set)
        # print(len(list(word_set)))
    vocab = list(word_set)
    vocab_size = len(vocab)

    word_id_map = {}
    for i in range(vocab_size):
        word_id_map[vocab[i]] = i
print('initialize out-of-vocabulary word embeddings')
print(vocab_size)
# print(word_id_map)
print('*'*100)

# initialize out-of-vocabulary word embeddings
oov = {}
for v in vocab:
    oov[v] = np.random.uniform(-0.01, 0.01, word_embeddings_dim)

print('build label list')
print('*'*100)
# build label list
label_set = set()
for doc_meta in process_doc_name_list:
    temp = doc_meta.split('\t')
    label_set.add(temp[2])
    # print(temp)
    # print(temp[2])
label_list = list(label_set)
label_list.sort()
# print(label_list)

print('select 90% training set')
print('*'*100)
# select 90% training set
train_size = len(train_ids)
val_size = int(0.1 * train_size)
real_train_size = train_size - val_size
test_size = len(test_ids)
print('prepare for the graph construction……')
print('*'*100)

# build graph function
def build_graph(start, end):
    x_adj = []
    x_feature = []
    y = []
    doc_len_list = []
    vocab_set = set()
    global_vocab = {}

    for i in tqdm(range(start, end)):
        #中英文分词不同
        if sys.argv[2] == 'CN':
            doc_words = list(jieba.cut(process_doc_words_list[i]))
            isnum = re.compile(r'^(\-|\+)?\d+(\.\d+)?$')
            for j in doc_words:
                if isnum.findall(j):
                    # print(i)
                    doc_words.remove(j)
        elif sys.argv[2] == 'EN':
            doc_words = process_doc_words_list[i].split()


        if truncate:
            doc_words = doc_words[:MAX_TRUNC_LEN]
        doc_len = len(doc_words)
        # count every document words in vocab, words' number = nodes' number
        doc_vocab = list(set(doc_words))
        doc_nodes = len(doc_vocab)

        doc_len_list.append(doc_nodes)
        vocab_set.update(doc_vocab)
        doc_word_id_map = {}
        for j in range(doc_nodes):
            doc_word_id_map[doc_vocab[j]] = j
        # sliding windows, 先将窗口大小的单词全部取出来
        windows = []
        if doc_len <= window_size:
            windows.append(doc_words)
        else:
            for j in range(doc_len - window_size + 1):
                window = doc_words[j: j + window_size]
                windows.append(window)

        word_pair_count = {}
        for window in windows:
            for p in range(1, len(window)):
                for q in range(0, p):
                    # 记录每个窗口下单词之间是否共现
                    word_p = window[p]
                    word_p_id = word_id_map[word_p]
                    word_q = window[q]
                    word_q_id = word_id_map[word_q]
                    if word_p_id == word_q_id:
                        continue
                    # 记录每个窗口下单词共现的ID对
                    word_pair_key = (word_p_id, word_q_id)
                    # word co-occurrences as weights, 出现次数权重加权
                    if word_pair_key in word_pair_count:
                        word_pair_count[word_pair_key] += 1.
                    else:
                        word_pair_count[word_pair_key] = 1.
                    # bi-direction, 双向
                    word_pair_key = (word_q_id, word_p_id)
                    if word_pair_key in word_pair_count:
                        word_pair_count[word_pair_key] += 1.
                    else:
                        word_pair_count[word_pair_key] = 1.
        # print(word_pair_count)
        row = []
        col = []
        weight = []
        features = []

        for key in word_pair_count:
            p = key[0]
            q = key[1]
            row.append(doc_word_id_map[vocab[p]])
            col.append(doc_word_id_map[vocab[q]])
            weight.append(word_pair_count[key] if weighted_graph else 1.)
        adj = sp.csr_matrix((weight, (row, col)), shape=(doc_nodes, doc_nodes))
        for k, v in sorted(doc_word_id_map.items(), key=lambda x: x[1]):
            features.append(word_embeddings[k] if k in word_embeddings else oov[k])
        # 邻接矩阵和embedding向量
        x_adj.append(adj)
        x_feature.append(features)

    # one-hot labels
    for i in range(start, end):
        doc_meta = process_doc_name_list[i]
        temp = doc_meta.split('\t')
        label = temp[2]
        one_hot = [0 for l in range(len(label_list))]
        label_index = label_list.index(label)
        one_hot[label_index] = 1
        y.append(one_hot)
        # print(one_hot, label)
    y = np.array(y)

    return x_adj, x_feature, y, doc_len_list, vocab_set


print('building graphs for training')
x_adj, x_feature, y, _, _ = build_graph(start=0, end=real_train_size)
print('building graphs for training + validation')
allx_adj, allx_feature, ally, doc_len_list_train, vocab_train = build_graph(start=0, end=train_size)
print('vocab size:',len(list(vocab_train)))
print('building graphs for test')
tx_adj, tx_feature, ty, doc_len_list_test, vocab_test = build_graph(start=train_size, end=train_size+test_size)
doc_len_list = doc_len_list_train + doc_len_list_test


# statistics
print('max_doc_length',max(doc_len_list),'min_doc_length',min(doc_len_list),
      'average {:.2f}'.format(np.mean(doc_len_list)))
print('training_vocab',len(vocab_train),'test_vocab',len(vocab_test),
      'intersection',len(vocab_train & vocab_test))


# dump objects
with open("../data/GNNDATA/ind.{}.x_adj".format(dataset), 'wb') as f:
    pkl.dump(x_adj, f)

with open("../data/GNNDATA/ind.{}.x_embed".format(dataset), 'wb') as f:
    pkl.dump(x_feature, f)

with open("../data/GNNDATA/ind.{}.y".format(dataset), 'wb') as f:
    pkl.dump(y, f)

with open("../data/GNNDATA/ind.{}.tx_adj".format(dataset), 'wb') as f:
    pkl.dump(tx_adj, f)

with open("../data/GNNDATA/ind.{}.tx_embed".format(dataset), 'wb') as f:
    pkl.dump(tx_feature, f)

with open("../data/GNNDATA/ind.{}.ty".format(dataset), 'wb') as f:
    pkl.dump(ty, f)

with open("../data/GNNDATA/ind.{}.allx_adj".format(dataset), 'wb') as f:
    pkl.dump(allx_adj, f)

with open("../data/GNNDATA/ind.{}.allx_embed".format(dataset), 'wb') as f:
    pkl.dump(allx_feature, f)

with open("../data/GNNDATA/ind.{}.ally".format(dataset), 'wb') as f:
    pkl.dump(ally, f)
