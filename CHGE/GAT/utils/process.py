import numpy as np
import networkx as nx
import scipy.sparse as sp
import random
import tensorflow as tf
import sklearn.metrics
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score,recall_score,f1_score
import  csv
import pickle
"""
 Prepare adjacency matrix by expanding up to a given neighbourhood.
 This will insert loops on every node.
 Finally, the matrix is converted to bias vectors.
 Expected shape: [graph, nodes, nodes]
"""
def adj_to_bias(adj, sizes, nhood=1):
    nb_graphs = adj.shape[0]
    mt = np.empty(adj.shape)
    for g in range(nb_graphs):
        mt[g] = np.eye(adj.shape[1])
        for _ in range(nhood):
            mt[g] = adj[g]
    return -1e9 * (1.0 - mt)

###############################################
# This section of code adapted from tkipf/gcn #
###############################################

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def load_graph(graph_file_name):
    graph = sp.load_npz(graph_file_name)
    return graph

def load_features(user_emb_file_name, poi_emb_file_name,catg_emb_file_name,method='concat'):
    user_embeddings = np.load(user_emb_file_name)
    poi_embeddings = np.load(poi_emb_file_name)
    catg_embeddings = np.load(catg_emb_file_name)
    if method == 'concat':
        zero_pad_user = np.zeros((user_embeddings.shape[0],
                                user_embeddings.shape[1]), dtype=np.float32)
        zero_pad_poi = np.zeros((poi_embeddings.shape[0],
                                user_embeddings.shape[1]), dtype=np.float32)
        zero_pad_catg = np.zeros((catg_embeddings.shape[0],
                                 user_embeddings.shape[1]), dtype=np.float32)

        user_embeddings = np.concatenate((user_embeddings,zero_pad_user), axis=1)
        poi_embeddings = np.concatenate((zero_pad_poi,poi_embeddings), axis=1)
        catg_embeddings = np.concatenate((zero_pad_catg,catg_embeddings), axis=1)
        node_features = np.vstack((user_embeddings, poi_embeddings))
        node_features = np.vstack((node_features, catg_embeddings))
    node_features = np.vstack((node_features, np.zeros(node_features.shape[1])))
    scalar = StandardScaler().fit(node_features)
    node_features = scalar.transform(node_features)
    node_features=np.mat(node_features)
    return node_features

def load_test_dataset(true_pair_filename, false_pair_filename):
    true_pairs = []
    with open(true_pair_filename) as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            u, v = row
            u = int(u)
            v = int(v)
            true_pairs.append((u, v))
    false_pairs = []
    with open(false_pair_filename) as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            u, v = row
            u = int(u)
            v = int(v)
            false_pairs.append((u, v))
    return true_pairs, false_pairs


def load_train_dataset(true_pair_filename, false_pair_filename):
    true_pairs = []
    with open(true_pair_filename) as csvfile:
        next(csvfile)
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            u, v = row
            u = int(u)
            v = int(v)
            true_pairs.append((u, v))
    false_pairs = []
    with open(false_pair_filename) as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            u, v = row
            u = int(u)
            v = int(v)
            false_pairs.append((u, v))
            false_pairs.append((v, u))
    return true_pairs, false_pairs


def load_random_data(size):
    adj = sp.random(size, size, density=0.002)  # density similar to cora
    features = sp.random(size, 1000, density=0.015)
    int_labels = np.random.randint(7, size=(size))
    labels = np.zeros((size, 7))  # Nx7
    labels[np.arange(size), int_labels] = 1

    train_mask = np.zeros((size,)).astype(bool)
    train_mask[np.arange(size)[0:int(size / 2)]] = 1

    val_mask = np.zeros((size,)).astype(bool)
    val_mask[np.arange(size)[int(size / 2):]] = 1

    test_mask = np.zeros((size,)).astype(bool)
    test_mask[np.arange(size)[int(size / 2):]] = 1

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    # sparse NxN, sparse NxF, norm NxC, ..., norm Nx1, ...
    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask

def construct_supports(graph, num_users,num_pois,num_catgs):
    rows = graph.row
    cols = graph.col
    values = graph.data
    num_nodes = num_users +num_catgs+num_pois
    graph = sp.lil_matrix((num_nodes + 1, num_nodes + 1), dtype='d')
    for row, col, val in zip(rows, cols, values):
            graph[row, col] = 1
    for i in range(num_users+num_catgs+num_pois):
        graph[i, i] = 1.
    supports=[]
    supports.append(graph)

    return supports

def construct_semi_supervised_supports(graph, partial_social_graph, num_users,num_pois,num_catgs):
    supports = construct_supports(graph, num_users,num_pois,num_catgs)
    num_nodes = num_users +num_catgs+num_pois
    semi_sup_graph = sp.lil_matrix((num_nodes + 1, num_nodes + 1), dtype='d')
    rows = partial_social_graph.row
    cols = partial_social_graph.col
    for row, col in zip(rows, cols):
        semi_sup_graph[row, col] = 1.
    for i in range(num_users):
        semi_sup_graph[i, i] = 1.
    supports.append(semi_sup_graph)
    return supports

def convert_to_adj_list(graph):
    adj_list = {}
    rows = graph.row
    cols = graph.col
    for row, col in zip(rows, cols):
        adj_list.setdefault(row, [])
        adj_list[row].append(col)
    print('number of isolated nodes:', graph.shape[0] - len(adj_list))
    return adj_list

def sample_context(graph, num_users, num_pois,num_catgs, walk_len, num_walks, num_pos_sample):
    graph = convert_to_adj_list(graph)
    num_nodes = num_users + num_catgs+num_pois
    homo_samples = dict()
    heter_samples = dict()

    for node in range(num_nodes):
        if node not in graph:
            continue
        for v in graph[node]:
            if (node < num_users and v < num_users) or \
                    (node >= num_users+ num_pois and v >= num_users +num_pois) or(num_users+ num_pois>node >= num_users and num_users+ num_pois >v >= num_users):
            # if (node < num_users and v < num_users):
                homo_samples.setdefault(node, [])
                homo_samples[node].append(v)
            else:
                heter_samples.setdefault(node, [])
                heter_samples[node].append(v)

    homo_samples_walk = dict()
    heter_samples_walk = dict()
    for node in range(num_nodes):
        if node not in graph:
            continue
        for _ in range(num_walks):
            curr_node = node
            for _ in range(walk_len):
                next_node = random.choice(graph[curr_node])
                if curr_node != node:
                    if (node < num_users and curr_node < num_users) or \
                            (node >= num_users+ num_pois and curr_node >= num_users+ num_pois) or(num_users+ num_pois>node >= num_users and num_users+ num_pois >curr_node >= num_users):
                    # if (node < num_users and curr_node < num_users):
                        homo_samples_walk.setdefault(node, [])
                        homo_samples_walk[node].append(curr_node)
                    else:
                        heter_samples_walk.setdefault(node, [])
                        heter_samples_walk[node].append(curr_node)
                curr_node = next_node
        if node % 1000 == 0:
            print("finish random walk for", node, "nodes")

    # fill blanks in positive sample:
    homo_samples_matrix = num_nodes * np.ones((num_nodes + 1, num_pos_sample))
    heter_samples_matrix = num_nodes * np.ones((num_nodes + 1, num_pos_sample))
    for node in range(num_nodes):
        if node in homo_samples:
            num_samples = len(homo_samples_walk[node])
            samples = np.random.choice(homo_samples[node], num_pos_sample // 2, replace=True)
            if num_samples >= num_pos_sample:
                samples = np.concatenate((
                    samples,
                    np.random.choice(homo_samples_walk[node], num_pos_sample // 2, replace=False)
                ))
                homo_samples_matrix[node] = samples
            else:
                samples = np.concatenate((samples,
                                          np.random.choice(homo_samples_walk[node], num_pos_sample // 2, replace=True)))
                homo_samples_matrix[node] = samples

        if node not in heter_samples:
            continue
        num_samples = len(heter_samples_walk[node])
        samples = np.random.choice(heter_samples[node], num_pos_sample // 2, replace=True)
        if num_samples > num_pos_sample:
            samples = np.concatenate((
                samples,
                np.random.choice(heter_samples_walk[node], num_pos_sample // 2, replace=False)
            ))
            heter_samples_matrix[node] = samples
        else:
            samples = np.concatenate((samples,
                                      np.random.choice(heter_samples_walk[node], num_pos_sample // 2, replace=True)))
            heter_samples_matrix[node] = samples
    del homo_samples
    del heter_samples

    return homo_samples_matrix, heter_samples_matrix

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

def standardize_data(f, train_mask):
    """Standardize feature matrix and convert to tuple representation"""
    # standardize data
    f = f.todense()
    mu = f[train_mask == True, :].mean(axis=0)
    sigma = f[train_mask == True, :].std(axis=0)
    f = f[:, np.squeeze(np.array(sigma > 0))]
    mu = f[train_mask == True, :].mean(axis=0)
    sigma = f[train_mask == True, :].std(axis=0)
    f = (f - mu) / sigma
    return f


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense(), sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def preprocess_adj_bias(adj):
    num_nodes = adj.shape[0]
    adj = adj + sp.eye(num_nodes)  # self-loop

    adj[adj > 0.0] = 1.0
    if not sp.isspmatrix_coo(adj):
        adj = adj.tocoo()
    adj = adj.astype(np.float32)
    # This is where I made a mistake, I used (adj.row, adj.col) instead
    indices = np.vstack((adj.col, adj.row)).transpose()

    return tf.SparseTensor(indices=indices, values=adj.data, dense_shape=adj.shape)

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def perf_evaluate(emb_outputs, true_pairs, false_pairs, num_users):
    user_embs = emb_outputs[:num_users, :]
    pred = sigmoid(user_embs.dot(user_embs.T))
    y_score = []
    y_true = []
    for u, v in true_pairs:
        y_score.append(pred[u, v])
        y_true.append(1)
    for u, v in false_pairs:
        y_score.append(pred[u, v])
        y_true.append(0)

    y_score = np.array(y_score, dtype=np.float32)
    y_true = np.array(y_true, dtype=np.int8)

    roc_auc = roc_auc_score(y_true, y_score)
    pr_auc = average_precision_score(y_true, y_score)
    return roc_auc, pr_auc

def sample_semi_supervised_context(partial_social_graph, num_pos_sample):
    num_users = partial_social_graph.shape[0]
    partial_social_graph = convert_to_adj_list(partial_social_graph)
    samples = np.zeros((num_users, num_pos_sample), dtype=np.int32)
    for u in range(num_users):
        if u not in partial_social_graph:
            samples[u] = u * np.ones(num_pos_sample, dtype=np.int32)
            continue
        context = []
        for v in partial_social_graph[u]:
            context.append(v)
        samples[u] = np.random.choice(context, num_pos_sample, replace=True)
    return samples