import networkx as nx
import networkx.algorithms.community as nx_comm
import matplotlib.pyplot as plt
from networkx.generators.random_graphs import gnm_random_graph
import pandas as pd
import numpy as np
import time
# from SparseShield_NIvsHS.Scripts.SparseShieldSolver import SparseShieldSolver
# from SparseShield_NIvsHS.Scripts.SparseShieldSeedlessSolver import SparseShieldSeedlessSolver
# from SparseShield_NIvsHS.Scripts.NetShieldSolver import NetShieldSolver
import os
import tensorflow as tf
from keras.models import load_model
from keras.optimizers import SGD
from matplotlib.figure import Figure
import random

from flask import Flask, render_template, request
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import base64
from io import BytesIO

app = Flask(__name__, static_url_path='/static')



def read_graph_from_file(tree_dir, filename):
    with open(os.path.join(tree_dir, filename), 'r') as file:
        G = nx.DiGraph()
        for line in file:
            if '->' in line:
                parent_node, child_node = line.strip().split('->')
                G.add_edge(parent_node, child_node)
    return G

def str_to_npa(s):
  data_list = s.split(' ')
  c = 0
  for x in data_list:
    if x == '':
      c += 1
  for _ in range(c):
    data_list.remove('')
  data_array = np.array([float(num) for num in data_list])
  return data_array

tree_dir = 'twitter15/tree'
d1 = pd.read_csv('t15_text_n2v.csv')
d1['n2v'] = d1['n2v'].apply(lambda x: x.replace('[', ''))
d1['n2v'] = d1['n2v'].apply(lambda x: x.replace(']', ''))
d1['n2v'] = d1['n2v'].apply(lambda x: str_to_npa(x))

nodes_list = []

with open('t15_imm_nodes.txt', 'r') as f:
    for line in f:
        row_data = line.strip().split(',')[:-1]     
        priority_nodes = [int(node) for node in row_data]        
        nodes_list.append(priority_nodes)

d1['nodes'] = nodes_list
print("Load BERT embeddings")
embeddings = np.load('bert_embeddings.npy')

d1['bert_embeddings'] = list(embeddings)

# model = load_model("model.h5", compile=False)
# model.compile(loss='binary_crossentropy', optimizer=SGD(), metrics=['accuracy'])
full_model = load_model("bert_n2v_model.h5", compile=False)
full_model.compile(loss='binary_crossentropy', optimizer=SGD(), metrics=['accuracy'])

def draw_graph(row_index):
    tweet_id = d1.loc[row_index, 'tweet_id']
    G = read_graph_from_file(tree_dir, str(tweet_id) + ".txt")
    pos = nx.kamada_kawai_layout(G)
    fig = plt.figure(figsize=(5, 5))
    nx.draw_networkx(G, pos, with_labels=True, node_color='skyblue', node_size=500, edge_color='gray')
    plt.axis('off')
    return fig

def draw_graph2(G):
    fig = plt.figure(figsize=(5, 5))
    pos = nx.kamada_kawai_layout(G)
    node_options = {"node_color": "red", "node_size":30}
    edge_options = {"width": .5, "alpha": .5, "edge_color":"black"}
    nx.draw_networkx_nodes(G, pos, **node_options)
    nx.draw_networkx_edges(G, pos, **edge_options)
    return fig
   

def draw_graph_imm(G, immunized_nodes):
    nodes = G.nodes()
    nodes_list = list(nodes)
    nodes_list[0]
    d = {nodes_list[i]: i for i in range(len(nodes_list))}
    H = nx.relabel_nodes(G, d)
    G = H.copy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    pos = nx.kamada_kawai_layout(G)
    
    
    immunized_node_options = {"node_color": "green", "node_size": 30}
    default_node_options = {"node_color": "red", "node_size": 30}
    
    nx.draw_networkx_nodes(G, pos, nodelist=immunized_nodes, ax=ax2, **immunized_node_options)
    
    non_immunized_nodes = [node for node in G.nodes() if node not in immunized_nodes]
    nx.draw_networkx_nodes(G, pos, nodelist=non_immunized_nodes, ax=ax2, **default_node_options)
    
    nx.draw_networkx_nodes(G, pos, ax=ax1, **default_node_options)

    ax2.set_title("Graph with Immunized nodes highlighted")    
    ax1.set_title("Original Graph")
    
    
    edge_options = {"width": 0.5, "alpha": 0.5, "edge_color": "black"}
    nx.draw_networkx_edges(G, pos, ax=ax1, **edge_options)
    nx.draw_networkx_edges(G, pos, ax=ax2, **edge_options)
    
    return fig

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        number = int(request.form['number'])
        row_index = number
        text = d1.loc[row_index, 'text']
        tweet_id = d1.loc[row_index, 'tweet_id']
        bert_embedding = d1.loc[row_index, 'bert_embeddings']
        n2v_embedding = d1.loc[row_index, 'n2v']
        imm_nodes = d1.loc[row_index, 'nodes']
        input_embedding = bert_embedding.reshape(1, 33, 768)
        input_node = n2v_embedding.reshape(1, 100)
        input_node_tensor = tf.convert_to_tensor(input_node)
        input_tensor = tf.convert_to_tensor(input_embedding)
        prediction = full_model.predict([input_tensor, input_node_tensor])
        if prediction > 0.5:
            prediction = "True News"
            prediction_class = "true-news"
        else:
            prediction = "Fake News"
            prediction_class = "fake-news"
        print(prediction)
        G = read_graph_from_file(tree_dir, str(tweet_id) + ".txt")
        fig = draw_graph_imm(G, imm_nodes)
        # Convert the figure to an image
        output = BytesIO()
        FigureCanvas(fig).print_png(output)
        image_data = base64.b64encode(output.getvalue()).decode('utf-8')
        return render_template('index.html', text=text, prediction=prediction, prediction_class=prediction_class, image_data=image_data)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)