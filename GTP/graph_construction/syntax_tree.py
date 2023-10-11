from stanfordcorenlp import StanfordCoreNLP
import networkx as nx
import matplotlib.pyplot as plt
import json
# 连接到StanfordCoreNLP服务
nlp = StanfordCoreNLP(r'./stanford-corenlp-full-2018-10-05', lang='en')

# 句子
sentence = "Just north of the Shalom Tower is the Yemenite Quarter, its main attractions being the bustling Carmel market and good Oriental restaurants."

# 进行依存句法分析
output = nlp.annotate(sentence, properties={
                       'annotators': 'tokenize,ssplit,pos,lemma,depparse',
                       'outputFormat': 'json'
                   })

# 解析依赖关系
output = json.loads(output)
dependencies = output['sentences'][0]['basicDependencies']

# 创建有向图
dependency_graph = nx.DiGraph()

# 添加词和边到图中
for dep in dependencies:
    word = dep['dependentGloss']
    head = dep['governorGloss']
    dep_label = dep['dep']
    dependency_graph.add_node(word, label=word)
    if head != 'ROOT':
        dependency_graph.add_edge(head, word, label=dep_label)

# 使用fruchterman_reingold_layout布局绘制图
pos = nx.fruchterman_reingold_layout(dependency_graph)

# 绘制节点
node_labels = nx.get_node_attributes(dependency_graph, 'label')
nx.draw_networkx_nodes(dependency_graph, pos, node_size=800, node_color='skyblue')
nx.draw_networkx_labels(dependency_graph, pos, labels=node_labels, font_color='black', font_weight='bold')

# 绘制边
edge_labels = nx.get_edge_attributes(dependency_graph, 'label')
nx.draw_networkx_edges(dependency_graph, pos, width=1.5, arrowsize=12)
nx.draw_networkx_edge_labels(dependency_graph, pos, edge_labels=edge_labels, font_size=8)

# 显示图
plt.axis('off')
plt.show()