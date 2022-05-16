# -*- coding:utf-8 -*-

"""



Author:

    Weichen Shen,wcshen1994@163.com



Reference:

    [1] Tang J, Qu M, Wang M, et al. Line: Large-scale information network embedding[C]//Proceedings of the 24th International Conference on World Wide Web. International World Wide Web Conferences Steering Committee, 2015: 1067-1077.(https://arxiv.org/pdf/1503.03578.pdf)



"""
import math
import random

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Embedding, Input, Lambda
from tensorflow.python.keras.models import Model

from alias import create_alias_table, alias_sample
from utils import preprocess_nxgraph,preprocess_nxgraph_1


def list2dict(node_list):
    idx2node = node_list
    node2idx = dict()
    for idx,node in enumerate(idx2node):
        node2idx[node] = idx
    return idx2node, node2idx


def line_loss(y_true, y_pred):
    return -K.mean(K.log(K.sigmoid(y_true*y_pred)))


def create_model(numNodes, embedding_size, order='second'):

    v_i = Input(shape=(1,))
    v_j = Input(shape=(1,))

    first_emb = Embedding(numNodes, embedding_size, name='first_emb')
    second_emb = Embedding(numNodes, embedding_size, name='second_emb')
    context_emb = Embedding(numNodes, embedding_size, name='context_emb')

    v_i_emb = first_emb(v_i)
    v_j_emb = first_emb(v_j)

    v_i_emb_second = second_emb(v_i)
    v_j_context_emb = context_emb(v_j)

    first = Lambda(lambda x: tf.reduce_sum(
        x[0]*x[1], axis=-1, keepdims=False), name='first_order')([v_i_emb, v_j_emb])
    second = Lambda(lambda x: tf.reduce_sum(
        x[0]*x[1], axis=-1, keepdims=False), name='second_order')([v_i_emb_second, v_j_context_emb])
    # print(first)
    if order == 'first':
        output_list = [first]
    elif order == 'second':
        output_list = [second]
    else:
        output_list = [first, second]

    model = Model(inputs=[v_i, v_j], outputs=output_list)

    return model, {'first': first_emb, 'second': second_emb}


class LINE:
    def __init__(self, graph1,graph2,graph3,graph4, graph5,graph6,number_of_graph,data_list1, data_list2,data_list3,data_list4,embedding_size=8, negative_ratio=5, order='second',):
        """

        :param graph:
        :param embedding_size:
        :param negative_ratio:
        :param order: 'first','second','all'
        """
        if order not in ['first', 'second', 'all']:
            raise ValueError('mode must be fisrt,second,or all')

        self.graph1 = graph1
        self.graph2 = graph2
        self.graph3 = graph3
        self.graph4 = graph4
        self.graph5 = graph5
        self.graph6 = graph6
        # self.graph7 = graph7

        self.data_list1 = data_list1
        self.data_list2 = data_list2
        self.data_list3 = data_list3
        self.data_list4 = data_list4
        self.number_of_graph=number_of_graph
        self.idx2node, self.node2idx = preprocess_nxgraph(self.data_list1,self.data_list2,self.data_list3,self.data_list4)
        self.use_alias = True

        self.rep_size = embedding_size
        self.order = order

        self._embeddings = {}
        self.negative_ratio = negative_ratio
        self.order = order

        self.node_size = len(self.idx2node)
        self.edge_size = graph1.number_of_edges()
        self.samples_per_epoch = self.edge_size*(1+negative_ratio)*self.number_of_graph

        self.graph1_negative_node_list,self.graph1_node_accept,self.graph1_node_alias,self.graph1_edge_accept,self.graph1_edge_alias=\
            self._gen_sampling_table(self.graph1 )
        self.graph2_negative_node_list,self.graph2_node_accept, self.graph2_node_alias, self.graph2_edge_accept, self.graph2_edge_alias = \
            self._gen_sampling_table(self.graph2)
        self.graph3_negative_node_list,self.graph3_node_accept, self.graph3_node_alias, self.graph3_edge_accept, self.graph3_edge_alias = \
            self._gen_sampling_table(self.graph3)
        self.graph4_negative_node_list,self.graph4_node_accept, self.graph4_node_alias, self.graph4_edge_accept, self.graph4_edge_alias = \
            self._gen_sampling_table(self.graph4)

        self.graph5_negative_node_list,self.graph5_node_accept, self.graph5_node_alias, self.graph5_edge_accept, self.graph5_edge_alias = \
            self._gen_sampling_table(self.graph5)

        self.graph6_negative_node_list,self.graph6_node_accept, self.graph6_node_alias, self.graph6_edge_accept, self.graph6_edge_alias = \
        self._gen_sampling_table(self.graph6)
        #
        # self.graph7_negative_node_list, self.graph7_node_accept, self.graph7_node_alias, self.graph7_edge_accept, self.graph7_edge_alias = \
        #     self._gen_sampling_table(self.graph7)
        self.reset_model()

    def reset_training_config(self, batch_size, times):
        self.batch_size = batch_size
        self.steps_per_epoch = (
            (self.samples_per_epoch - 1) // self.batch_size + 1)*times

    def reset_model(self, opt='adam'):

        self.model, self.embedding_dict = create_model(
            self.node_size, self.rep_size, self.order)
        self.model.compile(opt, line_loss)
        self.batch_it = self.batch_iter(self.node2idx)

    def _gen_sampling_table(self,graph):

        # create sampling table for vertex
        power = 0.75
        numNodes = graph.number_of_nodes()
        node_degree = np.zeros(numNodes)  # out degree
        idx2node, node2idx=preprocess_nxgraph_1(graph)
        for edge in graph.edges():
            node_degree[node2idx[edge[1]]
                        ] += graph[edge[0]][edge[1]]['weight']

        total_sum_node = sum([math.pow(node_degree[i], power)
                         for i in range(numNodes)])
        norm_prob_node = [float(math.pow(node_degree[j], power)) /
                     total_sum_node for j in range(numNodes)]

        node_accept, node_alias = create_alias_table(norm_prob_node)

        # create sampling table for edge
        numEdges = graph.number_of_edges()

        total_sum_edge = sum([graph[edge[0]][edge[1]]['weight']
                         for edge in graph.edges()])
        norm_prob_edge = [graph[edge[0]][edge[1]]['weight']*
                     numEdges / total_sum_edge for edge in graph.edges()]
        edge_accept, edge_alias = create_alias_table(norm_prob_edge)

        return idx2node,node_accept,node_alias,edge_accept,edge_alias

    def batch_iter(self, node2idx):
        graph1_edges = [(node2idx[x[0]], node2idx[x[1]]) for x in self.graph1.edges()]
        graph2_edges = [(node2idx[x[0]], node2idx[x[1]]) for x in self.graph2.edges()]
        graph3_edges = [(node2idx[x[0]], node2idx[x[1]]) for x in self.graph3.edges()]
        graph4_edges = [(node2idx[x[0]], node2idx[x[1]]) for x in self.graph4.edges()]
        graph5_edges = [(node2idx[x[0]], node2idx[x[1]]) for x in self.graph5.edges()]
        graph6_edges = [(node2idx[x[0]], node2idx[x[1]]) for x in self.graph6.edges()]
        # graph7_edges = [(node2idx[x[0]], node2idx[x[1]]) for x in self.graph7.edges()]
        graph1_data_size = self.graph1.number_of_edges()
        graph2_data_size = self.graph2.number_of_edges()
        graph3_data_size = self.graph3.number_of_edges()
        graph4_data_size = self.graph4.number_of_edges()
        graph5_data_size = self.graph5.number_of_edges()
        graph6_data_size = self.graph6.number_of_edges()
        # graph7_data_size = self.graph7.number_of_edges()

        graph1_batchSize = self.batch_size
        graph2_batchSize = int(self.batch_size * graph2_data_size / graph1_batchSize)
        graph3_batchSize = int(self.batch_size * graph3_data_size / graph1_batchSize)
        graph4_batchSize = int(self.batch_size * graph4_data_size / graph1_batchSize)
        graph5_batchSize = int(self.batch_size * graph5_data_size / graph1_batchSize)
        graph6_batchSize = int(self.batch_size * graph6_data_size / graph1_batchSize)
        # graph7_batchSize = int(self.batch_size * graph7_data_size / graph1_batchSize)

        graph1_shuffle_indices = np.random.permutation(np.arange(graph1_data_size))
        graph2_shuffle_indices = np.random.permutation(np.arange(graph2_data_size))
        graph3_shuffle_indices = np.random.permutation(np.arange(graph3_data_size))
        graph4_shuffle_indices = np.random.permutation(np.arange(graph4_data_size))
        graph5_shuffle_indices = np.random.permutation(np.arange(graph5_data_size))
        graph6_shuffle_indices = np.random.permutation(np.arange(graph6_data_size))
        # graph7_shuffle_indices = np.random.permutation(np.arange(graph7_data_size))

        graph1_start_index = 0
        graph1_end_index = min(graph1_start_index + graph1_batchSize, graph1_data_size)

        graph2_start_index = 0
        graph2_end_index = min(graph2_start_index + graph2_batchSize, graph2_data_size)

        graph3_start_index = 0
        graph3_end_index = min(graph3_start_index + graph3_batchSize, graph3_data_size)

        graph4_start_index = 0
        graph4_end_index = min(graph4_start_index + graph4_batchSize, graph4_data_size)

        graph5_start_index = 0
        graph5_end_index = min(graph5_start_index + graph5_batchSize, graph5_data_size)
        #
        graph6_start_index = 0
        graph6_end_index = min(graph6_start_index + graph6_batchSize, graph6_data_size)
        #
        # graph7_start_index = 0
        # graph7_end_index = min(graph7_start_index + graph7_batchSize, graph7_data_size)

        # positive or negative mod  graph1
        mod = 0
        mod_size = 1 + self.negative_ratio
        graph1_h = []
        graph2_h = []
        graph3_h = []
        graph4_h = []
        graph5_h = []
        graph6_h = []
        # graph7_h = []
        count = 0
        h = []
        t = []
        while True:
            if mod == 0:
                h = []
                t = []
                graph1_h = []
                graph1_t = []
                for i in range(graph1_start_index, graph1_end_index):
                    if random.random() >= self.graph1_edge_accept[graph1_shuffle_indices[i]]:
                        graph1_shuffle_indices[i] = self.graph1_edge_alias[graph1_shuffle_indices[i]]
                    cur_h = graph1_edges[graph1_shuffle_indices[i]][0]
                    cur_t = graph1_edges[graph1_shuffle_indices[i]][1]
                    graph1_h.append(cur_h)
                    graph1_t.append(cur_t)
                    h.append(cur_h)
                    t.append(cur_t)

                graph2_h = []
                graph2_t = []
                for i in range(graph2_start_index, graph2_end_index):
                    if random.random() >= self.graph2_edge_accept[graph2_shuffle_indices[i]]:
                        graph2_shuffle_indices[i] = self.graph2_edge_alias[graph2_shuffle_indices[i]]
                    cur_h = graph2_edges[graph2_shuffle_indices[i]][0]
                    cur_t = graph2_edges[graph2_shuffle_indices[i]][1]
                    graph2_h.append(cur_h)
                    graph2_t.append(cur_t)
                    h.append(cur_h)
                    t.append(cur_t)

                graph3_h = []
                graph3_t = []
                for i in range(graph3_start_index, graph3_end_index):
                    if random.random() >= self.graph3_edge_accept[graph3_shuffle_indices[i]]:
                        graph3_shuffle_indices[i] = self.graph3_edge_alias[graph3_shuffle_indices[i]]
                    cur_h = graph3_edges[graph3_shuffle_indices[i]][0]
                    cur_t = graph3_edges[graph3_shuffle_indices[i]][1]
                    graph3_h.append(cur_h)
                    graph3_t.append(cur_t)
                    h.append(cur_h)
                    t.append(cur_t)

                graph4_h = []
                graph4_t = []
                for i in range(graph4_start_index, graph4_end_index):
                    if random.random() >= self.graph4_edge_accept[graph4_shuffle_indices[i]]:
                        graph4_shuffle_indices[i] = self.graph4_edge_alias[graph4_shuffle_indices[i]]
                    cur_h = graph4_edges[graph4_shuffle_indices[i]][0]
                    cur_t = graph4_edges[graph4_shuffle_indices[i]][1]
                    graph4_h.append(cur_h)
                    graph4_t.append(cur_t)
                    h.append(cur_h)
                    t.append(cur_t)

                graph5_h = []
                graph5_t = []
                for i in range(graph5_start_index, graph5_end_index):
                    if random.random() >= self.graph5_edge_accept[graph5_shuffle_indices[i]]:
                        graph5_shuffle_indices[i] = self.graph5_edge_alias[graph5_shuffle_indices[i]]
                    cur_h = graph5_edges[graph5_shuffle_indices[i]][0]
                    cur_t = graph5_edges[graph5_shuffle_indices[i]][1]
                    graph5_h.append(cur_h)
                    graph5_t.append(cur_t)
                    h.append(cur_h)
                    t.append(cur_t)
                #
                graph6_h = []
                graph6_t = []
                for i in range(graph6_start_index, graph6_end_index):
                    if random.random() >= self.graph6_edge_accept[graph6_shuffle_indices[i]]:
                        graph6_shuffle_indices[i] = self.graph6_edge_alias[graph6_shuffle_indices[i]]
                    cur_h = graph6_edges[graph6_shuffle_indices[i]][0]
                    cur_t = graph6_edges[graph6_shuffle_indices[i]][1]
                    graph6_h.append(cur_h)
                    graph6_t.append(cur_t)
                    h.append(cur_h)
                    t.append(cur_t)
                #
                # graph7_h = []
                # graph7_t = []
                # for i in range(graph7_start_index, graph7_end_index):
                #     if random.random() >= self.graph7_edge_accept[graph7_shuffle_indices[i]]:
                #         graph7_shuffle_indices[i] = self.graph7_edge_alias[graph7_shuffle_indices[i]]
                #     cur_h = graph7_edges[graph7_shuffle_indices[i]][0]
                #     cur_t = graph7_edges[graph7_shuffle_indices[i]][1]
                #     graph7_h.append(cur_h)
                #     graph7_t.append(cur_t)
                #     h.append(cur_h)
                #     t.append(cur_t)
                # graph_sign = np.ones(len(graph1_h) + len(graph2_h)+ len(graph3_h)+ len(graph4_h) + len(graph5_h)+ len(graph6_h))
                graph_sign = np.ones(len(h))
            else:
                graph_sign = np.ones(len(h)) * -1
                graph1_t = []
                t = []
                for i in range(len(graph1_h)):
                    temp=alias_sample(self.graph1_node_accept, self.graph1_node_alias)
                    graph1_t.append(node2idx[self.graph1_negative_node_list[temp]])
                    t.append(node2idx[self.graph1_negative_node_list[temp]])
                graph2_t = []
                for i in range(len(graph2_h)):
                    temp=alias_sample(
                        self.graph2_node_accept, self.graph2_node_alias)
                    graph2_t.append(node2idx[self.graph2_negative_node_list[temp]])
                    t.append(node2idx[self.graph2_negative_node_list[temp]])

                graph3_t = []
                for i in range(len(graph3_h)):
                    temp=alias_sample(
                        self.graph3_node_accept, self.graph3_node_alias)
                    graph3_t.append(node2idx[self.graph3_negative_node_list[temp]])
                    t.append(node2idx[self.graph3_negative_node_list[temp]])

                graph4_t = []
                for i in range(len(graph4_h)):
                    temp=alias_sample(
                        self.graph4_node_accept, self.graph4_node_alias)
                    graph4_t.append(node2idx[self.graph4_negative_node_list[temp]])
                    t.append(node2idx[self.graph4_negative_node_list[temp]])

                graph5_t = []
                for i in range(len(graph5_h)):
                    temp=alias_sample(
                        self.graph5_node_accept, self.graph5_node_alias)
                    graph5_t.append(node2idx[self.graph5_negative_node_list[temp]])
                    t.append(node2idx[self.graph5_negative_node_list[temp]])

                graph6_t = []
                for i in range(len(graph6_h)):
                    temp=alias_sample(
                        self.graph6_node_accept, self.graph6_node_alias)
                    graph6_t.append(node2idx[self.graph6_negative_node_list[temp]])
                    t.append(node2idx[self.graph6_negative_node_list[temp]])
                #
                # graph7_t = []
                # for i in range(len(graph7_h)):
                #     graph7_t.append(alias_sample(self.graph7_node_accept, self.graph7_node_alias))
                #     graph7_t.append(node2idx[self.graph7_negative_node_list[temp]])
                #     t.append(node2idx[self.graph7_negative_node_list[temp]])

                # graph_sign = np.ones(len(graph1_h) +len(graph2_h) + len(graph3_h) + len(graph4_h)+ len(graph5_h)+ len(graph6_h)) * -1
            if self.order == 'all':
                yield ([np.array(h), np.array(t)], [graph_sign, graph_sign])
            else:
                yield ([np.array(h), np.array(t)], [graph_sign])
            # if self.order == 'all':
            #     yield ([np.array(graph1_h +graph2_h + graph3_h+ graph4_h + graph5_h+ graph6_h), np.array(graph1_t +graph2_t + graph3_t+ graph4_t+ graph5_t+ graph6_t)], [graph_sign, graph_sign])
            #
            # else:
            #     yield ([np.array(graph1_h +graph2_h + graph3_h+ graph4_h+ graph5_h+ graph6_h ), np.array(graph1_t +graph2_t + graph3_t+ graph4_t + graph5_t+ graph6_t)], [graph_sign])

            mod += 1
            mod %= mod_size
            if mod == 0:
                graph1_start_index = graph1_end_index
                graph1_end_index = min(graph1_start_index + graph1_batchSize, graph1_data_size)

                graph2_start_index = graph2_end_index
                graph2_end_index = min(graph2_start_index + graph2_batchSize, graph2_data_size)

                graph3_start_index =  graph3_end_index
                graph3_end_index = min( graph3_start_index +  graph3_batchSize,  graph3_data_size)

                graph4_start_index = graph4_end_index
                graph4_end_index = min(graph4_start_index + graph4_batchSize, graph4_data_size)

                graph5_start_index = graph5_end_index
                graph5_end_index = min(graph5_start_index + graph5_batchSize, graph5_data_size)

                graph6_start_index = graph6_end_index
                graph6_end_index = min(graph6_start_index + graph6_batchSize, graph6_data_size)
                #
                # graph7_start_index = graph7_end_index
                # graph7_end_index = min(graph7_start_index + graph7_batchSize, graph7_data_size)

            if graph1_start_index >= graph1_data_size:
                count += 1
                mod = 0
                graph1_h = []
                graph1_shuffle_indices = np.random.permutation(np.arange(graph1_data_size))
                graph1_start_index = 0
                graph1_end_index = min(graph1_start_index + graph1_batchSize, graph1_data_size)

                graph2_h = []
                graph2_shuffle_indices = np.random.permutation(np.arange(graph2_data_size))
                graph2_start_index = 0
                graph2_end_index = min(graph2_start_index + graph2_batchSize, graph2_data_size)

                graph3_h = []
                graph3_shuffle_indices = np.random.permutation(np.arange(graph3_data_size))
                graph3_start_index = 0
                graph3_end_index = min(graph3_start_index + graph3_batchSize, graph3_data_size)

                graph4_h = []
                graph4_shuffle_indices = np.random.permutation(np.arange(graph4_data_size))
                graph4_start_index = 0
                graph4_end_index = min(graph4_start_index + graph4_batchSize, graph4_data_size)

                graph5_h = []
                graph5_shuffle_indices = np.random.permutation(np.arange(graph5_data_size))
                graph5_start_index = 0
                graph5_end_index = min(graph5_start_index + graph5_batchSize, graph5_data_size)

                graph6_h = []
                graph6_shuffle_indices = np.random.permutation(np.arange(graph6_data_size))
                graph6_start_index = 0
                graph6_end_index = min(graph6_start_index + graph6_batchSize, graph6_data_size)
                #
                # graph7_h = []
                # graph7_shuffle_indices = np.random.permutation(np.arange(graph7_data_size))
                # graph7_start_index = 0
                # graph7_end_index = min(graph7_start_index + graph7_batchSize, graph7_data_size)

    def get_embeddings(self,):
        self._embeddings = {}
        if self.order == 'first':
            embeddings = self.embedding_dict['first'].get_weights()[0]
        elif self.order == 'second':
            embeddings = self.embedding_dict['second'].get_weights()[0]
        else:
            embeddings = np.hstack((self.embedding_dict['first'].get_weights()[
                                   0], self.embedding_dict['second'].get_weights()[0]))
        idx2node = self.idx2node
        for i, embedding in enumerate(embeddings):
            self._embeddings[idx2node[i]] = embedding
        return self._embeddings

    def train(self, batch_size=1024, epochs=1, initial_epoch=0, verbose=1, times=1):
        self.reset_training_config(batch_size, times)
        hist = self.model.fit_generator(self.batch_it, epochs=epochs, initial_epoch=initial_epoch, steps_per_epoch=self.steps_per_epoch,
                                        verbose=verbose)

        return hist
