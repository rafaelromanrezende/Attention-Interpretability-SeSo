import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from attention_graph_util import *
import seaborn as sns
import itertools 
import matplotlib as mpl

rc={'font.size': 10, 'axes.labelsize': 10, 'legend.fontsize': 10.0, 
    'axes.titlesize': 32, 'xtick.labelsize': 20, 'ytick.labelsize': 16}
plt.rcParams.update(**rc)
mpl.rcParams['axes.linewidth'] = .5 #set the value globally

import torch
import matplotlib.pyplot as plt

# Importando apenas o que importa, sem o "*"
from transformers import (
    BertConfig, 
    BertForMaskedLM, 
    BertTokenizer,
    BertModel, 
    OpenAIGPTModel, OpenAIGPTTokenizer,
    GPT2Model, GPT2Tokenizer,
    CTRLModel, CTRLTokenizer,
    TransfoXLModel, TransfoXLTokenizer,
    XLNetModel, XLNetTokenizer,
    XLMModel, XLMTokenizer,
    DistilBertModel, DistilBertTokenizer,
    RobertaModel, RobertaTokenizer,
    BertForPreTraining, BertForNextSentencePrediction,
    BertForSequenceClassification, BertForTokenClassification, 
    BertForQuestionAnswering
)

from pathlib import Path


def plot_attention_heatmap(att, s_position, t_positions, tokens_list):
    cls_att = np.flip(att[:, s_position, t_positions], axis=0)
    
    xticklb = [tokens_list[i] for i in t_positions]
    yticklb = [str(i) if i%2 ==0 else '' for i in np.arange(att.shape[0],0, -1)]
    
    ax = sns.heatmap(cls_att, xticklabels=xticklb, yticklabels=yticklb, cmap="YlOrRd")
    return ax


def convert_adjmat_tomats(adjmat, n_layers, l):
   mats = np.zeros((n_layers,l,l))
   
   for i in np.arange(n_layers):
       mats[i] = adjmat[(i+1)*l:(i+2)*l,i*l:(i+1)*l]
       
   return mats

# Transformers has a unified API
# for 8 transformer architectures and 30 pretrained weights.
#          Model          | Tokenizer          | Pretrained weights shortcut
MODELS = [(BertModel,       BertTokenizer,       'bert-base-uncased'),
          (OpenAIGPTModel,  OpenAIGPTTokenizer,  'openai-gpt'),
          (GPT2Model,       GPT2Tokenizer,       'gpt2'),
          (CTRLModel,       CTRLTokenizer,       'ctrl'),
          (TransfoXLModel,  TransfoXLTokenizer,  'transfo-xl-wt103'),
          (XLNetModel,      XLNetTokenizer,      'xlnet-base-cased'),
          (XLMModel,        XLMTokenizer,        'xlm-mlm-enfr-1024'),
          (DistilBertModel, DistilBertTokenizer, 'distilbert-base-uncased'),
          (RobertaModel,    RobertaTokenizer,    'roberta-base')]

# Each architecture is provided with several class for fine-tuning on down-stream tasks, e.g.
BERT_MODEL_CLASSES = [BertModel, BertForPreTraining, BertForMaskedLM, BertForNextSentencePrediction,
                      BertForSequenceClassification, BertForTokenClassification, BertForQuestionAnswering]

IMAGES_DIR = Path("images/bert")
IMAGES_DIR.mkdir(exist_ok=True)

# All the classes for an architecture can be initiated from pretrained weights for this architecture
# Note that additional weights added for fine-tuning are only initialized
# and need to be trained on the down-stream task
pretrained_weights = 'bert-large-uncased'
tokenizer = BertTokenizer.from_pretrained(pretrained_weights)

model = BertForMaskedLM.from_pretrained(pretrained_weights,
                                  output_hidden_states=True,
                                  output_attentions=True)

sentences = {}
src = {}
targets = {}
candidates = {}
sentences[1] = "She asked the doctor about "+tokenizer.mask_token+" backache"
src[1] = 6
targets[1] = (1,4) # she, doctor
candidates[1] = ('her', 'his')

sentences[0] = "He talked to her about his book"
src[0] = 6 # Aponta para 'his'
targets[0] = (1,4) # he, her
candidates[0] = ('his', 'her')

sentences[2] = "The author talked to Sara about "+tokenizer.mask_token+" book"
src[2] = 7
targets[2] = (2,5) # author, sara
candidates[2] = ('his', 'her')

sentences[3] = "John tried to convince Mary of his love and brought flowers for "+tokenizer.mask_token
src[3] = 13
targets[3] = (1,5) # john, mary
candidates[3] = ('her', 'him')

sentences[4] = "Mary convinced John of "+tokenizer.mask_token+" love"
src[4] = 5
targets[4] = (1,3) # mary, john
candidates[4] = ('her', 'his')


sentences[5] = "Barack Obama was the president of the " + tokenizer.mask_token
src[5] = 8
targets[5] = (1, 2) # Barack, Obama
candidates[5] = ('usa', 'us')

sentences[6] = "Artificial intelligence is the " + tokenizer.mask_token + " of study that"
src[6] = 5
targets[6] = (1, 2) # Artificial, intelligence
candidates[6] = ('field', 'area')

sentences[7] = "Why is the sky " + tokenizer.mask_token + " ?"
src[7] = 5
targets[7] = (1, 4) # Why, sky
candidates[7] = ('blue', 'dark')

sentences[8] = "If Paul's wife is Mary, Mary's husband is " + tokenizer.mask_token
src[8] = 14
targets[8] = (2, 7) # Paul, Mary
candidates[8] = ('paul', 'mary')

for ex_id in range(1 , 9):
    OUTPUT_DIR = IMAGES_DIR / str(ex_id)
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    sentence = sentences[ex_id]

    tokens = tokenizer.tokenize(sentence)
    print(tokens)

    tf_input_ids = tokenizer.encode(sentence)
    input_ids = torch.tensor([tf_input_ids])

    all_hidden_states, all_attentions = model(input_ids)[-2:]

    _attentions = [att.detach().numpy() for att in all_attentions]
    attentions_mat = np.asarray(_attentions)[:,0]

    attentions_mat = attentions_mat[:, :, 1:-1, 1:-1]

    output = model(input_ids)[0]
    predicted_target = torch.nn.Softmax(dim=-1)(output[0,src[ex_id]])

    top_pred_id = np.argmax(output.detach().numpy()[0], axis=-1)
    print("Previsão bruta da frase:", tokenizer.decode(top_pred_id))
    
    print("Palavra na posição analisada (src):", tokenizer.decode([tf_input_ids[src[ex_id]]]))

    t1_id = tf_input_ids[targets[ex_id][0]]
    t2_id = tf_input_ids[targets[ex_id][1]]
    print(f"Target 1 ({tokenizer.decode([t1_id])}) prob no src: {predicted_target[t1_id]:.6f}")
    print(f"Target 2 ({tokenizer.decode([t2_id])}) prob no src: {predicted_target[t2_id]:.6f}")

    k=5
    topk_vals, topk_idx = torch.topk(predicted_target, k)

    print(f"Top {k} value")
    for idx, val in zip(topk_idx, topk_vals):
        print(f"Token: {tokenizer.decode([idx.item()]):15} | prob: {val.item():.6f} | id: {idx.item()}")

    word_a, word_b = candidates[ex_id]
    id_a = tokenizer.encode([word_a])[1] # Pega o ID da palavra
    id_b = tokenizer.encode([word_b])[1]
    
    prob_a = predicted_target[id_a].item()
    prob_b = predicted_target[id_b].item()
    
    print(f"\n-- Decisão do MASK: '{word_a}' vs '{word_b}' --")
    print(f"Probabilidade de '{word_a}': {prob_a:.6f}")
    print(f"Probabilidade de '{word_b}': {prob_b:.6f}")
    print(f"O modelo prefere '{word_a}'?: {prob_a > prob_b}")

    fig = plt.figure(figsize=(2,6))
    ax = sns.barplot(
        x=[f'{id_a}', '{id_b}'], 
        y = [
        predicted_target[id_a].detach().item(),
        predicted_target[id_b].detach().item()
        ],
        linewidth=0, 
        palette='Set1'
    )
    sns.despine(fig=fig, ax=None, top=True, right=True, left=True, bottom=False, offset=None, trim=False)
    ax.set_yticks([])
    plt.savefig(OUTPUT_DIR /'rat_bert_bar_{}.png'.format(ex_id), format='png', transparent=True, dpi=360, bbox_inches='tight')
    plt.close()
    s_pos_corrigida = src[ex_id] - 1
    t_pos_corrigidas = (targets[ex_id][0] - 1, targets[ex_id][1] - 1)

    plt.figure(figsize=(3,6))

    raw_attention_avg = attentions_mat.sum(axis=1) / attentions_mat.shape[1]

    numero_da_cabeca = 5 # Teste números de 0 a 11

    res_att_mat = attentions_mat[:, numero_da_cabeca, :, :]

    plot_attention_heatmap(
        raw_attention_avg, 
        s_pos_corrigida, 
        t_positions=t_pos_corrigidas, 
        tokens_list=tokens 
    )

    plt.savefig(OUTPUT_DIR /'rat_bert_att_{}.png'.format(ex_id), format='png', transparent=True, dpi=360, bbox_inches='tight')
    plt.close()
    #descomente isso se quiser usar a média das cabeças em cada camada
    # res_att_mat = attentions_mat.sum(axis=1)/attentions_mat.shape[1]
    res_att_mat = res_att_mat + (0.3 * np.eye(res_att_mat.shape[1])[None,...])
    res_att_mat = res_att_mat / res_att_mat.sum(axis=-1)[...,None]
 
    res_adj_mat, res_labels_to_index = get_adjmat(mat=res_att_mat, input_tokens=tokens)

    res_G = draw_attention_graph(res_adj_mat,res_labels_to_index, n_layers=res_att_mat.shape[0], length=res_att_mat.shape[-1])

    last_layer_name = f'L{attentions_mat.shape[0]}' # Descobre automaticamente se é L6, L12 ou L24
    output_nodes = []
    input_nodes = []
    for key in res_labels_to_index:
        if last_layer_name in key: # <-- AGORA É DINÂMICO
            output_nodes.append(key)
        if res_labels_to_index[key] < attentions_mat.shape[-1]:
            input_nodes.append(key)

    flow_values = compute_flows(res_G, res_labels_to_index, input_nodes, length=attentions_mat.shape[-1])
    flow_G = draw_attention_graph(flow_values,res_labels_to_index, n_layers=attentions_mat.shape[0], length=attentions_mat.shape[-1])

    flow_att_mat = convert_adjmat_tomats(flow_values, n_layers=attentions_mat.shape[0], l=attentions_mat.shape[-1])

    s_pos_corrigida = src[ex_id] - 1
    t_pos_corrigidas = (targets[ex_id][0] - 1, targets[ex_id][1] - 1)

    plt.figure(figsize=(3,6))

    plot_attention_heatmap(
        flow_att_mat, 
        s_pos_corrigida, 
        t_positions=t_pos_corrigidas, 
        tokens_list=tokens 
    )

    plt.savefig(OUTPUT_DIR /'res_fat_bert_att_{}.png'.format(ex_id), format='png', transparent=True,dpi=300, bbox_inches='tight')
    plt.close()
    joint_attentions = compute_joint_attention(res_att_mat, add_residual=False)
    joint_att_adjmat, joint_labels_to_index = get_adjmat(mat=joint_attentions, input_tokens=tokens)

    G = draw_attention_graph(joint_att_adjmat,joint_labels_to_index, n_layers=joint_attentions.shape[0], length=joint_attentions.shape[-1])

    s_pos_corrigida = src[ex_id] - 1
    t_pos_corrigidas = (targets[ex_id][0] - 1, targets[ex_id][1] - 1)

    plt.figure(figsize=(3,6))

    plot_attention_heatmap(
    joint_attentions, 
    s_pos_corrigida, 
    t_positions=t_pos_corrigidas, 
    tokens_list=tokens 
    )

    plt.savefig(OUTPUT_DIR /'res_jat_bert_att_{}.png'.format(ex_id), format='png', transparent=True, dpi=360, bbox_inches='tight')
    plt.close()

    