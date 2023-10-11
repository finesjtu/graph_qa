from Graph_data_prepare.Dataloader import Dataloder
import pickle as pkl
import numpy as np
#coding:utf-8
import matplotlib.pyplot as plt
import matplotlib
# import numpy as np
import warnings
import json
import os
import sys
from tqdm import tqdm
import gensim
from gensim import corpora
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel
from gensim.models import LdaMulticore
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
# import pickle as pkl
import scipy.sparse as sp
from tqdm import trange

warnings.filterwarnings('ignore')
stopwords = set(stopwords.words('english'))
# ibmcs 0.8 argmin 0.9 semeval2016t6 0.8 perspectrum 0.8
dataset_name = 'semeval2016t6'
word_embeddings_dim = 300
num_topics = 10
threshold = 0.8
neighbor_ratio = 1
graph_type = 'GNN' # GNN
model_type = 'SAT'

dataloder = Dataloder(dataset_name=dataset_name)

raw_data = dataloder.data
raw_label = dataloder.label
raw_uid = dataloder.uid
raw_train_split = dataloder.train_split
raw_test_split = dataloder.test_split
raw_dev_split = dataloder.dev_split

print('The length of corpus:{}'.format(len(raw_label)))
print('The index of train data:{}'.format(raw_train_split))
print('The index of test data:{}'.format(raw_test_split))
print('The index of dev data:{}'.format(raw_dev_split))
# word_embeddings = {}
with open('/root/NLPCODE/GAS/GTP/graph_construction/graph_cache/word2vec.pkl', 'rb') as f:
    w2v_dict = pkl.load(f)
print('Word2Vec load done!')
doc_length = 50
doc_embedding = []
for text in raw_data:
    words = text.split()
    words_embedding = []
    for word in words:
        word = word.lower()
        if word in w2v_dict.keys():
            w2v = w2v_dict[word]
            words_embedding.append(w2v)
    if graph_type == 'CNN':
        if len(words_embedding) < 50:
            pad_embedding = [0] * 300
            for i in range(50 - len(words_embedding)):
                words_embedding.append(pad_embedding)
        if len(words_embedding) > 50:
            words_embedding = words_embedding[:50]
    elif graph_type == 'GNN':
        # print(np.array(words_embedding).shape)
        words_embedding = np.average(np.array(words_embedding), axis=0)
    doc_embedding.append(words_embedding)
print(np.array(doc_embedding).shape)
if graph_type == 'GNN':
    for i in doc_embedding:
        try:
            if len(i) != 300:
                print(i)
        except TypeError:
            print('There is nan value!')

print(len(doc_embedding))


data_set = []
for text in raw_data:
    result = []
    for word in text.split():
        word = word.lower()
        if word not in stopwords:
            result.append(word)
    data_set.append(result)


dictionary = corpora.Dictionary(data_set)
# print(dictionary)
corpus = [dictionary.doc2bow(text.split()) for text in raw_data]
ldamodel = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=30, random_state=1)

result_lda = []
for i in corpus[:]:
    # i = ' '.join(i)
    # print(i)
    vec = ldamodel.get_document_topics(i, minimum_probability=0)
    vec = [x[1] for x in vec]
    result_lda.append(vec)

print(len(result_lda))
adj_list = []
sim = cosine_similarity(result_lda)
shape = sim.shape[0]
row = []
col = []
value = []
for i in trange(shape):
    for j in range(shape):
        if sim[i][j] > threshold:
            label_i = raw_label[i]
            label_j = raw_label[j]
            if label_i == label_j:
                row.append(i)
                col.append(j)
                value.append(sim[i][j])
            elif label_j != label_i:
                if np.random.random(1) < neighbor_ratio:
                    row.append(i)
                    col.append(j)
                    value.append(sim[i][j])

matrix = sp.coo_matrix((value, (row, col)), shape=(shape, shape))
edges = np.array([row,col]).T
print(edges.shape)
# print(edges)
whole_data = {
    'graph_adj': matrix,
    'doc_feature': doc_embedding,
    'edges': edges,
}
#'/mnt/data1/GAS_DATA_WSK/lda_edge/' + self.dataset + '/similarity.pkl'

with open('/mnt/data1/GAS_DATA_WSK/lda_edge/' + dataset_name + '_' + model_type + '_similarity.pkl', 'wb') as f:
    pkl.dump(whole_data, f)

