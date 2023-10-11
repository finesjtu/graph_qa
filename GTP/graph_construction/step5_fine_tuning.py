from graph_bert.code.DatasetLoader import DatasetLoader
from graph_bert.code.MethodBertComp import GraphBertConfig
from graph_bert.code.MethodGraphBertNodeClassification import MethodGraphBertNodeClassification
from graph_bert.code.ResultSaving import ResultSaving
from graph_bert.code.Settings import Settings
import numpy as np
import torch


#---- 'cora' , 'citeseer', 'pubmed' ----

dataset_name = 'argmin'
adj_name = 'syn'
np.random.seed(42)
torch.manual_seed(42)

#---- cora-small is for debuging only ----
if dataset_name == 'arc':
    nclass = 4
    nfeature = 1024
    ngraph = 10
elif dataset_name == 'perspectrum':
    nclass = 2
    nfeature = 1024
    ngraph = 10
elif dataset_name == 'argmin':
    nclass = 2
    nfeature = 1024
    ngraph = 10
elif dataset_name == 'ibmcs':
    nclass = 2
    nfeature = 1024
    ngraph = 10
elif dataset_name == 'snopes':
    nclass = 2
    nfeature = 1024
    ngraph = 10
elif dataset_name == 'cora':
    nclass = 7
    nfeature = 1433
    ngraph = 2708
elif dataset_name == 'semeval2016t6':
    nclass = 3
    nfeature = 1024
    ngraph = 19717
elif dataset_name == 'mnli':
    nclass = 2
    nfeature = 1024
    ngraph = 10

#---- Fine-Tuning Task 1: Graph Bert Node Classification (Cora, Citeseer, and Pubmed) ----
if 1:
    #---- hyper-parameters ----
    if dataset_name == 'pubmed':
        lr = 0.001
        k = 30
        max_epoch = 1000 # 500 ---- do an early stop when necessary ----
    elif dataset_name == 'cora':
        lr = 0.01
        k = 8
        batch_size = 100000
        max_epoch = 150 # 150 ---- do an early stop when necessary ----
    elif dataset_name == 'arc':
        lr = 0.0001
        k = 7
        batch_size = 128
        max_epoch = 150  # 150 ---- do an early stop when necessary ----
    elif dataset_name == 'perspectrum':
        lr = 0.00001
        k = 7
        batch_size = 128
        max_epoch = 150  # 150 ---- do an early stop when necessary ----
    elif dataset_name == 'snopes':
        lr = 0.0001
        k = 7
        batch_size = 128
        max_epoch = 150  # 150 ---- do an early stop when necessary ----
    elif dataset_name == 'argmin':
        lr = 0.00001
        k = 7
        batch_size = 128
        max_epoch = 150  # 150 ---- do an early stop when necessary ----
    elif dataset_name == 'mnli':
        lr = 0.00001
        k = 7
        batch_size = 128
        max_epoch = 150

    elif dataset_name == 'semeval2016t6':
        lr = 0.0001
        k = 7
        batch_size = 1024
        max_epoch = 150  # 150 ---- do an early stop when necessary ----
    x_size = nfeature
    hidden_size = intermediate_size = 256
    num_attention_heads = 4
    num_hidden_layers = 4
    y_size = nclass
    graph_size = ngraph
    residual_type = 'none'
    # --------------------------

    print('************ Start ************')
    print('GrapBert, dataset: ' + dataset_name + ', residual: ' + residual_type + ', k: ' + str(k) + ', hidden dimension: ' + str(hidden_size) +', hidden layer: ' + str(num_hidden_layers) + ', attention head: ' + str(num_attention_heads))
    # ---- objection initialization setction ---------------
    data_obj = DatasetLoader()
    if dataset_name == 'cora':
        data_obj.dataset_source_folder_path = './data/' + dataset_name + '/'
    elif dataset_name != 'cora':
        data_obj.dataset_source_folder_path = '/root/NLPCODE/GAS/GTP/graph_construction/graph_cache/'
    data_obj.dataset_name = dataset_name
    data_obj.k = k
    data_obj.load_all_tag = True

    bert_config = GraphBertConfig(residual_type = residual_type, k=k, x_size=nfeature, y_size=y_size, hidden_size=hidden_size,
                                  intermediate_size=intermediate_size, num_attention_heads=num_attention_heads,
                                  num_hidden_layers=num_hidden_layers, dataset_name=dataset_name)
    method_obj = MethodGraphBertNodeClassification(bert_config)
    # print(method_obj)
    #---- set to false to run faster ----
    method_obj.spy_tag = True
    method_obj.max_epoch = max_epoch
    method_obj.lr = lr
    method_obj.batch_size = batch_size

    result_obj = ResultSaving()
    result_obj.result_destination_folder_path = '/root/NLPCODE/GAS/GTP/graph_construction/graph_bert/result/GraphBert/'
    result_obj.result_destination_file_name = dataset_name + '_' + str(num_hidden_layers)

    setting_obj = Settings()

    evaluate_obj = None
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.load_run_save_evaluate(cuda=True, adj_name=adj_name)
    # ------------------------------------------------------


    print('************ Finish ************')
#------------------------------------

