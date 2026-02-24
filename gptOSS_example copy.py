import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from attention_graph_util import *
import seaborn as sns
import itertools 
import matplotlib as mpl
import torch
from transformers import AutoModelForSequenceClassification
from transformers import AutoModelForMaskedLM
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="2"

rc={'font.size': 10, 'axes.labelsize': 10, 'legend.fontsize': 10.0, 
    'axes.titlesize': 32, 'xtick.labelsize': 20, 'ytick.labelsize': 16}
plt.rcParams.update(**rc)
mpl.rcParams['axes.linewidth'] = .5 #set the value globally


def plot_attention_heatmap(att, s_position, t_positions, sentenceax):
    print(att.shape)
    # att = B, L, L
    #   print(att[:,s_position].shape, att[:,s_position])
    #   print(att.shape, att[:,s_position, t_positions])
    # pega todas as matrizes de atenção nas linhas s e colunas t, geralmente s é escalar
    # flipa no eixo do indice da matriz. Antes era (0, ..., L-1) agora (L-1, ..., 0)
    cls_att = np.flip(att[:,s_position, t_positions], axis=0)

    # cls_att = np.flip(att[:,s_position, 1:-1], axis=0)
    #   print(cls_att)
    # x é somente o t
    xticklb = input_tokens= list(itertools.compress(sentenceax, [i in t_positions for i in np.arange(len(sentence)+1)]))
    yticklb = [str(i) if i%2 ==0 else '' for i in np.arange(att.shape[0],0, -1)]
    # xticklb = list(itertools.compress(['<cls>']+sentence.split()+['<sep>'], [True for i in np.arange(len(sentence)+1)]))
    # print(xticklb)
    # print(sentence)


    yticklb = [str(i) if i%2 ==0 else '' for i in np.arange(att.shape[0],0, -1)]
    ax = sns.heatmap(cls_att, xticklabels=xticklb, yticklabels=yticklb, cmap="YlOrRd")
    return ax


def convert_adjmat_tomats(adjmat, n_layers, l):
   mats = np.zeros((n_layers,l,l))
   
   for i in np.arange(n_layers):
       mats[i] = adjmat[(i+1)*l:(i+2)*l,i*l:(i+1)*l]
       
   return mats

def draw_attention_graph(adjmat, labels_to_index, n_layers, length):
    A = adjmat
    G=nx.from_numpy_array(A, create_using=nx.DiGraph())
    for i in np.arange(A.shape[0]):
        for j in np.arange(A.shape[1]):
            nx.set_edge_attributes(G, {(i,j): A[i,j]}, 'capacity')

    pos = {}
    label_pos = {}
    for i in np.arange(n_layers+1):
        for k_f in np.arange(length):
            pos[i*length+k_f] = ((i+0.4)*2, length - k_f)
            label_pos[i*length+k_f] = (i*2, length - k_f)

    index_to_labels = {}
    for key in labels_to_index:
        index_to_labels[labels_to_index[key]] = key.split("_")[-1]
        if labels_to_index[key] >= length:
            index_to_labels[labels_to_index[key]] = ''

    #plt.figure(1,figsize=(20,12))

    nx.draw_networkx_nodes(G,pos,node_color='green', label=index_to_labels, node_size=50)
    nx.draw_networkx_labels(G,pos=label_pos, labels=index_to_labels, font_size=18)

    all_weights = []
    #4 a. Iterate through the graph nodes to gather all the weights
    for (node1,node2,data) in G.edges(data=True):
        all_weights.append(data['weight']) #we'll use this when determining edge thickness

    #4 b. Get unique weights
    unique_weights = list(set(all_weights))

    #4 c. Plot the edges - one by one!
    for weight in unique_weights:
        #4 d. Form a filtered list with just the weight you want to draw
        weighted_edges = [(node1,node2) for (node1,node2,edge_attr) in G.edges(data=True) if edge_attr['weight']==weight]
        #4 e. I think multiplying by [num_nodes/sum(all_weights)] makes the graphs edges look cleaner
        
        w = weight #(weight - min(all_weights))/(max(all_weights) - min(all_weights))
        width = w
        nx.draw_networkx_edges(G,pos,edgelist=weighted_edges,width=width, edge_color='darkblue')
    
    return G
 
def get_adjmat(mat, input_tokens):
    n_layers, length, _ = mat.shape
    adj_mat = np.zeros(((n_layers+1)*length, (n_layers+1)*length))
    print(adj_mat.shape, n_layers, length)
    labels_to_index = {}
    for k in np.arange(length):
        labels_to_index[str(k)+"_"+input_tokens[k]] = k

    #cada camada
    for i in np.arange(1,n_layers+1):
        #cada nó em camada
        for k_f in np.arange(length):
            index_from = (i)*length+k_f
            label = "L"+str(i)+"_"+str(k_f)
            labels_to_index[label] = index_from
            for k_t in np.arange(length):
                # cada nó em camada seguinte
                index_to = (i-1)*length+k_t
                # adj[i,j] é o valor da atenção 
                adj_mat[index_from][index_to] = mat[i-1][k_f][k_t]
    # print(adj_mat[length:2*length,0:length], i, k_f, k_t)
    return adj_mat, labels_to_index 


pretrained_weights = 'openai/gpt-oss-20b'
model = AutoModelForCausalLM.from_pretrained(pretrained_weights,
                                             output_hidden_states=True,
                                             output_attentions=True)
model.zero_grad()
tokenizer = AutoTokenizer.from_pretrained(pretrained_weights, use_fast=True)


sentences = {}

sentences[0] = "He talked to her about his book"

sentences[1] = "She asked the doctor about hers backache"

sentences[2] = "The author talked to Sara about his"

sentences[3] = "John tried to convince Mary of his love and brought flowers for "

sentences[4] = "Mary convinced John of her love"

sentences[5] = "Barack Obama was the president of the"

sentences[6] = "Artificial intelligence is the field of study that"

sentences[7] = "Why is the sky blue?"

sentences[8] = "If Paul's wife is Mary, Mary's husband is"

ex_id = 8
sentence = sentences[ex_id]

tokens =  tokenizer.tokenize(sentences[ex_id])
# tokens =  ['[CLS]'] + tokenizer.tokenize(sentences[ex_id]) + ['[SEP]']
# tokens = [['[cls]']+tokenizer.tokenize(sent)+['[sep]'] for sent in sentences.values()]
print(len(tokens), tokens)


# tokeniza a sentença
tf_input_ids = tokenizer.encode(sentence)
# tf_input_ids = [tokenizer.encode(sentence) for sentence in sentences.values()] # lista de todas as sentenças
print(f"decoding: {tokenizer.decode(tf_input_ids)}")


# transforma em um batch de sentença única
# input_ids =[ torch.tensor([tf_input_id])for tf_input_id in tf_input_ids]
input_ids = torch.tensor([tf_input_ids])


# saída no formato codificado: 1(batch), L(sentença), V (Vocab)
model_outputs = model(input_ids)
# model_outputs = [model(ids) for ids in input_ids]

# pega as atenções e hidden_states 12 Camadas, 1(batch), 12 (cabeças) , L(sentença), L(sentença)
all_hidden_states, all_attentions =  model_outputs['hidden_states'], model_outputs['attentions'] 

# all_hidden_states, all_attentions =  [model_output['hidden_states'] for model_output in model_outputs],  [model_output['attentions'] for model_output in model_outputs]
# print(len(all_attentions), all_attentions[0].shape, all_attentions[1].shape, all_attentions[11].shape)

# _attentions = [[att.detach().numpy() for att in all_attention] for all_attention in all_attentions] # transforma em array

# transforma cada camada em array para poder operar depois
_attentions = [att.float().cpu().detach().numpy() for att in all_attentions] # transforma em array
print(len(_attentions)) # (24,1,64,L,L)
# 24 camadas, 1 batch, 64 cabeças, LXL

# mata a camada de batch
attentions_mat = np.asarray(_attentions)[:,0]
print(attentions_mat.shape)

print(input_ids)
print(tokens)

output_ids = model.generate(input_ids, max_length=30)

print(tokenizer.decode(output_ids[0], skip_special_tokens=True))

output = model(input_ids).logits[0] # pega logits da saída
# print(len(output), output.shape, src[ex_id], output[0, src[ex_id]].shape)
# output = [model(id)[0] for id in input_ids]

#softmask da coluna com mask
predicted_target = torch.nn.Softmax(dim=0)(output[-1,:])

# converte para array e pega o argmax(maior logit) para o último token
previewd = np.argmax(predicted_target.float().cpu().detach().numpy(), axis=-1)

k= 5
topk = np.argsort(predicted_target.float().cpu().detach().numpy())[-k:]

print(np.argsort(predicted_target.float().cpu().detach().numpy())[-k:][::1])
print(tokenizer.decode(previewd), previewd) 


# pega topk mais provaveis para plotar
yax = [float(predicted_target[id].detach()) for id in topk]
xax = [tokenizer.decode(id) for id in topk] 
print(yax, xax)


fig = plt.figure(1,figsize=(6,6))
ax = sns.barplot(x= xax, y=yax, linewidth=0
)
sns.despine(fig=fig, ax=None, top=True, right=True, left=True, bottom=False, offset=None, trim=False)
ax.set_yticks([])
plt.savefig('rat_bert_bar_{}.png'.format(ex_id), format='png', transparent=True, dpi=360, bbox_inches='tight')


plt.figure(2,figsize=(3,6))
#soma no eixo das cabeças e tira a média
# passa onde está o mask
# passa t e sentença
# print(attentions_mat.sum(axis=1)/attentions_mat.shape[1])

# basicamente, quanto o s presta atenção aos outros tokens
# é possível ver o automasking

sentenceax = [tokenizer.decode(id) for id in tf_input_ids]
# print(sentenceax)
t_list=[]
for i in range(len(tokens)):
    t_list.append(i)
t_pos= tuple(t_list)
plot_attention_heatmap(attentions_mat.sum(axis=1)/attentions_mat.shape[1], s_position=len(tokens)-1, t_positions=t_pos, sentenceax=sentenceax)

#raw attention
plt.savefig('rat_bert_att_{}.png'.format(ex_id), format='png', transparent=True, dpi=360, bbox_inches='tight')


res_att_mat = attentions_mat.sum(axis=1)/attentions_mat.shape[1]
print(res_att_mat.shape)
# print(np.eye(res_att_mat.shape[1]).shape)
# print(np.eye(res_att_mat.shape[1])[None, ...].shape)
res_att_mat = res_att_mat + np.eye(res_att_mat.shape[1])[None,...]

# renormaliza, mesma coisa de dividir por dois
res_att_mat = res_att_mat / res_att_mat.sum(axis=-1)[...,None]
# print((res_att_mat/2) [0])
# print(res_att_mat[0]) 

# [[0, 0, 0, ..., 0]]
# [[W1, 0, 0, ..., 0]]
# [[0, W2, 0, ..., 0]]
# [[0, 0, 0, ..., Wn, 0]]
res_adj_mat, res_labels_to_index = get_adjmat(mat=res_att_mat, input_tokens=tokens)

# print(res_labels_to_index)

res_G = draw_attention_graph(res_adj_mat,res_labels_to_index, n_layers=res_att_mat.shape[0], length=res_att_mat.shape[-1])


output_nodes = []
input_nodes = []
for key in res_labels_to_index:
    if 'L24' in key:
        # não entendi
        output_nodes.append(key)
    if res_labels_to_index[key] < attentions_mat.shape[-1]:
        input_nodes.append(key)
print(input_nodes)
print(output_nodes)

import importlib
import attention_graph_util as a
importlib.reload(a)

# attention flow
# basta entender agora o algoritmo de max flow
flow_values = a.compute_flows(res_G, res_labels_to_index, input_nodes, length=attentions_mat.shape[-1])
flow_G = draw_attention_graph(flow_values,res_labels_to_index, n_layers=attentions_mat.shape[0], length=attentions_mat.shape[-1])

# dá para perceber também o masking, dado que os tokens só observam o passado, a concentração é maior nos tokens iniciais


flow_att_mat = convert_adjmat_tomats(flow_values, n_layers=attentions_mat.shape[0], l=attentions_mat.shape[-1])

# print(flow_att_mat)

# figs = plt.subplot()
plt.figure(3,figsize=(3,6))
t_list=[]
for i in range(len(tokens)):
    t_list.append(i)
t_pos= tuple(t_list)
plot_attention_heatmap(flow_att_mat, s_position=len(tokens)-1, t_positions=t_pos, sentenceax=sentenceax)

# attention flow
plt.savefig('res_fat_bert_att_{}.png'.format(ex_id), format='png', transparent=True,dpi=360, bbox_inches='tight')


#attention rollout
joint_attentions = compute_joint_attention(res_att_mat, add_residual=False)


joint_att_adjmat, joint_labels_to_index = get_adjmat(mat=joint_attentions, input_tokens=tokens)

G = draw_attention_graph(joint_att_adjmat,joint_labels_to_index, n_layers=joint_attentions.shape[0], length=joint_attentions.shape[-1])


plt.figure(4,figsize=(3,6))
t_list=[]
for i in range(len(tokens)):
    t_list.append(i)
t_pos= tuple(t_list)
plot_attention_heatmap(joint_attentions, s_position=len(tokens)-1, t_positions=t_pos, sentenceax=sentenceax)
plt.savefig('res_jat_bert_att_{}.png'.format(ex_id), format='png', transparent=True, dpi=360, bbox_inches='tight')


