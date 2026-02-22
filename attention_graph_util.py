import networkx as nx
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import tensorflow as tf

def get_adjmat(mat, input_tokens):
    n_layers, length, _ = mat.shape
    adj_mat = np.zeros(((n_layers+1)*length, (n_layers+1)*length))
    labels_to_index = {}
    for k in np.arange(length):
        labels_to_index[str(k)+"_"+input_tokens[k]] = k

    for i in np.arange(1,n_layers+1):
        for k_f in np.arange(length):
            index_from = (i)*length+k_f
            label = "L"+str(i)+"_"+str(k_f)
            labels_to_index[label] = index_from
            for k_t in np.arange(length):
                index_to = (i-1)*length+k_t
                adj_mat[index_from][index_to] = mat[i-1][k_f][k_t]
                
    return adj_mat, labels_to_index 


def draw_attention_graph(adjmat, labels_to_index, n_layers, length, 
                         top_k=3, min_threshold=1e-4, figsize=(16,10),
                         focus_target_idx=None): 

    A = adjmat.copy()
    num_nodes = A.shape[0]

    A[A < min_threshold] = 0.0

    for i in range(num_nodes):
        row = A[i]
        if np.count_nonzero(row) > top_k:
            top_indices = np.argsort(row)[-top_k:]
            mask = np.zeros_like(row)
            mask[top_indices] = 1
            A[i] = row * mask

    G = nx.DiGraph()
    for i in range(num_nodes):
        for j in range(num_nodes):
            if A[i, j] > 0:
                G.add_edge(i, j, weight=A[i, j], capacity=A[i, j])

    if focus_target_idx is not None:
        top_node_id = n_layers * length + focus_target_idx
        
        valid_nodes = nx.descendants(G, top_node_id)
        valid_nodes.add(top_node_id)
        
        G = G.subgraph(valid_nodes).copy()

    pos = {}
    label_pos = {}
    for i in range(n_layers+1):
        for k in range(length):
            pos[i*length+k] = ((i+0.4)*2, length - k)
            label_pos[i*length+k] = (i*2, length - k)

    index_to_labels = {}
    for key in labels_to_index:
        idx = labels_to_index[key]
        token = key.split("_")[-1]
        if idx < length:
            index_to_labels[idx] = token
        else:
            index_to_labels[idx] = ""

    plt.figure(figsize=figsize)

    nodes_to_draw = list(G.nodes())
    pos_filtered = {k: pos[k] for k in nodes_to_draw}
    label_pos_filtered = {k: label_pos[k] for k in nodes_to_draw}

    nx.draw_networkx_nodes(G, pos_filtered, node_color='black', node_size=40)
    
    # Filtra os labels para não dar erro
    labels_filtered = {k: index_to_labels[k] for k in nodes_to_draw if k in index_to_labels}
    nx.draw_networkx_labels(G, pos=label_pos_filtered, labels=labels_filtered, font_size=14)

    weights = [G[u][v]['weight'] for u,v in G.edges()]
    if len(weights) > 0:
        max_w = max(weights)
        min_w = min(weights)
        norm_weights = [
            0.5 + 4 * ((w - min_w) / (max_w - min_w + 1e-8))
            for w in weights
        ]
    else:
        norm_weights = []

    for ((u,v), w, width) in zip(G.edges(), weights, norm_weights):
        nx.draw_networkx_edges(
            G, pos_filtered, edgelist=[(u,v)], width=width,
            edge_color='royalblue', alpha=min(0.9, w*2)
        )

    plt.axis('off')
    plt.tight_layout()

    return G
    
def compute_flows(G, labels_to_index, input_nodes, length):
    number_of_nodes = len(labels_to_index)
    flow_values=np.zeros((number_of_nodes,number_of_nodes))
    for key in labels_to_index:
        if key not in input_nodes:
            current_layer = int(labels_to_index[key] / length)
            pre_layer = current_layer - 1
            u = labels_to_index[key]
            for inp_node_key in input_nodes:
                v = labels_to_index[inp_node_key]
                flow_value = nx.maximum_flow_value(G,u,v, flow_func=nx.algorithms.flow.edmonds_karp)
                flow_values[u][pre_layer*length+v ] = flow_value
            flow_values[u] /= flow_values[u].sum()
            
    return flow_values

def compute_node_flow(G, labels_to_index, input_nodes, output_nodes,length):
    number_of_nodes = len(labels_to_index)
    flow_values=np.zeros((number_of_nodes,number_of_nodes))
    for key in output_nodes:
        if key not in input_nodes:
            current_layer = int(labels_to_index[key] / length)
            pre_layer = current_layer - 1
            u = labels_to_index[key]
            for inp_node_key in input_nodes:
                v = labels_to_index[inp_node_key]
                flow_value = nx.maximum_flow_value(G,u,v, flow_func=nx.algorithms.flow.edmonds_karp)
                flow_values[u][pre_layer*length+v ] = flow_value
            flow_values[u] /= flow_values[u].sum()
            
    return flow_values

def compute_joint_attention(att_mat, add_residual=True):
    if add_residual:
        residual_att = np.eye(att_mat.shape[1])[None,...]
        aug_att_mat = att_mat + residual_att
        aug_att_mat = aug_att_mat / aug_att_mat.sum(axis=-1)[...,None]
    else:
       aug_att_mat =  att_mat
    
    joint_attentions = np.zeros(aug_att_mat.shape)

    layers = joint_attentions.shape[0]
    joint_attentions[0] = aug_att_mat[0]
    for i in np.arange(1,layers):
        joint_attentions[i] = aug_att_mat[i].dot(joint_attentions[i-1])
        
    return joint_attentions