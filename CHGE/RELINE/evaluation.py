import  csv
import numpy as np
# import networkx as nx
import scipy.sparse as sp
import random
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import ast
from sklearn.metrics import average_precision_score, roc_auc_score,f1_score,recall_score


def load_test_dataset(true_pair_filename, false_pair_filename):
    true_pairs = []
    with open(true_pair_filename) as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            u, v = row
            u = u
            v = v
            true_pairs.append((u, v))
    false_pairs = []
    with open(false_pair_filename) as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            u, v = row
            u = u
            v = v
            false_pairs.append((u, v))
    return true_pairs, false_pairs

def perf_evaluate(emb_outputs, true_pairs, false_pairs):
    # user_embs = emb_outputs[:num_users, :]
    # pred = sigmoid(user_embs.dot(user_embs.T))
    # pred = sigmoid(np.dot(user_embs,user_embs.T))
    y_score = []
    y_true = []
    y_score1 = []
    for u, v in true_pairs:
        data0 = np.array(emb_outputs[u]).reshape(1, 128)
        data1 = np.array(emb_outputs[v]).reshape(1, 128)
        pred = sigmoid(data0.dot(data1.T))
        y_score.append(pred)
        y_true.append(1)
        if (pred > 0.5):
            y_score1.append(1)
        else:
            y_score1.append(0)
    for u, v in false_pairs:
        data0 = np.array(emb_outputs[u])
        data1 = np.array(emb_outputs[v])
        pred = sigmoid(data0.dot(data1.T))
        y_score.append(pred)
        y_true.append(0)
        if (pred > 0.5):
            y_score1.append(1)
        else:
            y_score1.append(0)
    f1score = f1_score(y_true, y_score1,average='micro')
    recallscore = recall_score(y_true, y_score1,average='micro')
    y_score = np.array(y_score, dtype=np.float32)
    y_true = np.array(y_true, dtype=np.int8)
    y_score = np.array(y_score, dtype=np.float32)
    y_true = np.array(y_true, dtype=np.int8)
    roc_auc = roc_auc_score(y_true, y_score)
    pr_auc = average_precision_score(y_true, y_score)
    return roc_auc, pr_auc, f1score, recallscore

def load_embedding(embedding_filename):
    c_i = open(embedding_filename)
    final_embeddings={}
    for line in c_i:
        tts = line.strip(r'\s+').split()
        user=tts[0]
        data=tts[1:]
        final_embeddings[user]= list(map(float, data))
    return  final_embeddings

def sigmoid(x):
    return 1. / (1. + np.exp(-x))


country='BR'

true_pairs,false_pairs=load_test_dataset('../dataset/graph/' + country + '/user_user_graph_test_true.csv','../dataset/graph/' + country + '/user_user_graph_test_false.csv')
final_embeddings=load_embedding('user_venue_catg_embedding1.txt')
roc_auc, pr_auc,f1score, recallscore = perf_evaluate(final_embeddings, true_pairs, false_pairs)
print("evaluation on test set:")
print("ROCAUC={:.5f}".format(roc_auc))
print("PRAUC={:.5f}".format(pr_auc))
print("f1score={:.5f}".format(f1score))
print("recall={:.5f}".format(recallscore))