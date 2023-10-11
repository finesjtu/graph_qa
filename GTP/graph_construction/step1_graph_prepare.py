## coding:utf-8
import os

import string
from stanfordcorenlp import StanfordCoreNLP
from nltk.corpus import stopwords
import json
import pickle
from tqdm import trange
import re

# data = pickle.load(open("./mr_stan.pkl", "rb"))
#
# print(list(data.keys ())[:100])
# perspectrum
dataset = 'semeval2016t6'
path = '../dataset_roberta/'
train_dataset_path = path + dataset + '_train.json'
test_dataset_path = path + dataset + '_test.json'
dev_dataset_path = path + dataset + '_dev.json'
output = './standford_den/'

data = []

with open('./vocab.txt', 'r') as f:
    vocab_bert = [x.strip() for x in f.readlines()]

with open(train_dataset_path, 'r') as f:
    data_str = f.readlines()

    for i in data_str:
        json_data = json.loads(i)
        if dataset == 'semeval2019t7':
            data_meta = json_data['premise']
        else:
            data_meta = json_data['hypothesis'] + ' for the topic \"' + json_data['premise'] + '\"'
        data.append(data_meta)


with open(test_dataset_path, 'r') as f:
    data_str = f.readlines()

    for i in data_str:
        json_data = json.loads(i)
        if dataset == 'semeval2019t7':
            data_meta = json_data['premise']
        else:
            data_meta = json_data['hypothesis'] + ' for the topic \"' +json_data['premise'] + '\"'
        data.append(data_meta)


def stopwords_analysis(word):

    pattern = re.compile('[0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~\s]+')
    word = word.lower()
    if word not in stop_words:
        if not re.findall(pattern, word):
            if word in vocab_bert:
                return True
    return False

nlp = StanfordCoreNLP(r'./stanford-corenlp-full-2018-10-05', lang='en')


#路径设置
os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
os.path.abspath(os.path.dirname(os.getcwd()))
os.path.abspath(os.path.join(os.getcwd(), ".."))

stop_words = set(stopwords.words('english'))



#获取句法依存关系对
rela_pair_count_str = {}
for doc_id in trange(len(data)):
    # print(doc_id)
    words = data[doc_id]
    words = words.split("\n")
    rela=[]
    # print('Words:', words)
    # print(string)
    for window in words:
        if window==' ':
            continue
        #构造rela_pair_count
        window = window.replace(string.punctuation, ' ')

        res = nlp.dependency_parse(window)
        token = nlp.word_tokenize(window)
        for i, begin, end in res:
            # print(i, '-'.join([str(begin), token[begin-1]]), '-'.join([str(end), token[end-1]]))
            rela.append(token[begin-1] + ', ' + token[end-1])
        for pair in rela:
            pair=pair.split(", ")
            # print(pair)
            if pair[0]=='ROOT' or pair[1]=='ROOT':
                continue
            if pair[0] == pair[1]:
                continue
            # if pair[0] in string.punctuation or pair[1] in string.punctuation:
            #     continue
            if pair[0] in stop_words or pair[1] in stop_words:
                continue

            if stopwords_analysis(pair[0]) and stopwords_analysis(pair[1]):
                word_pair_str = pair[0].lower() + ',' + pair[1].lower()
                inverse_key = pair[1].lower() + ',' + pair[0].lower()
                # print(word_pair_str)
                rela_pair_count_str[word_pair_str] = rela_pair_count_str.setdefault(word_pair_str, 0) + 1
                rela_pair_count_str[inverse_key] = rela_pair_count_str.setdefault(inverse_key, 0) + 1
            else:
                continue
                # if word_pair_str in rela_pair_count_str:
                #     rela_pair_count_str[word_pair_str] += 1
                # else:
                #     rela_pair_count_str[word_pair_str] = 1
                # # two orders
                # word_pair_str = pair[1] + ',' + pair[0]
                # if word_pair_str in rela_pair_count_str:
                #     rela_pair_count_str[word_pair_str] += 1
                # else:
                #     rela_pair_count_str[word_pair_str] = 1

# print(rela_pair_count_str.keys())
# print(rela_pair_count_str.values())
# 将rela_pair_count_str存成pkl格式
output1 = open(output + '{}_stan.pkl'.format(dataset),'wb')
pickle.dump(rela_pair_count_str, output1)



