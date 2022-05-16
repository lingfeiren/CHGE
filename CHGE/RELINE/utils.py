# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 21:39:52 2020

@author: Administrator
"""
import pandas as pd
import networkx as nx
import numpy as np


####################user2venue/venue2venue/venue2catg###############
def graphConstruct_user2venue(path_user2venue):
    # timeshot, poi
    df = pd.read_csv(path_user2venue, sep = ',')
    df['weight'] = df['delta_times']
    df= df['weight'].groupby([df['user'],df['poi']]).sum()
    df = df.reset_index()
    g_user2venue = nx.from_pandas_edgelist(df, source='user',target='poi',edge_attr=['weight'],\
                                         create_using=nx.DiGraph())
    return df,g_user2venue

def graphConstruct_venue2venue(path_venue2venue):
    # timeshot, poi
    df = pd.read_csv(path_venue2venue, sep = ',')
    df['weight'] = df['delta_times']
    df= df['weight'].groupby([df['source_poi'],df['target_poi']]).sum()
    df = df.reset_index()
    g_venue2venue = nx.from_pandas_edgelist(df, source='source_poi',target='target_poi',edge_attr=['weight'],\
                                         create_using=nx.DiGraph())
    return df,g_venue2venue


def graphConstruct_poi2catg(path_venue2catg):
    # timeshot, poi
    df = pd.read_csv(path_venue2catg, sep = ',')
    df['weight'] = df['delta_times']
    df= df['weight'].groupby([df['poi'],df['catg']]).sum()
    df = df.reset_index()
    g_venue2catg = nx.from_pandas_edgelist(df, source='poi',target='catg',edge_attr=['weight'],\
                                         create_using=nx.DiGraph())

    return df,g_venue2catg

#######################user2catg/catg2catg/catg2venue################

def graphConstruct_user2catg(path_user2catg):
    # timeshot, poi
    df = pd.read_csv(path_user2catg, sep = ',')
    df['weight'] = df['delta_times']
    df= df['weight'].groupby([df['user'],df['catg']]).sum()
    df = df.reset_index()
    g_user2catg = nx.from_pandas_edgelist(df, source='user',target='catg',edge_attr=['weight'],\
                                         create_using=nx.DiGraph())
    return df,g_user2catg

def graphConstruct_catg2catg(path_catg2catg):
    # timeshot, poi
    df = pd.read_csv(path_catg2catg, sep = ',')
    df['weight'] = df['delta_times']
    df= df['weight'].groupby([df['source_catg'],df['target_catg']]).sum()
    df = df.reset_index()
    g_catg2catg = nx.from_pandas_edgelist(df, source='source_catg',target='target_catg',edge_attr=['weight'],\
                                         create_using=nx.DiGraph())
    return df,g_catg2catg

def graphConstruct_user2user(path_user2user):
    # timeshot, poi
    df = pd.read_csv(path_user2user, sep = ',')
    df['weight'] = df['delta_times']
    df= df['weight'].groupby([df['source_user'],df['target_user']]).sum()
    df = df.reset_index()
    g_user2user = nx.from_pandas_edgelist(df, source='source_user',target='target_user',edge_attr=['weight'],\
                                         create_using=nx.Graph())
    return df,g_user2user


def graphConstruct_time2catg(path_time2catg):
    # timeshot, poi
    df = pd.read_csv(path_time2catg, sep = ',')
    df['weight'] = df['delta_times']
    df= df['weight'].groupby([df['catg'],df['time']]).sum()
    df = df.reset_index()
    g_time2catg = nx.from_pandas_edgelist(df, source='catg',target='time',edge_attr=['weight'],\
                                         create_using=nx.DiGraph())
    return df, g_time2catg

def preprocess_nxgraph(data_list1,data_list2,data_list3,data_list4):
    data_list=data_list1+data_list2+data_list3+data_list4
    node2idx = {}
    idx2node = []
    node_size = 0
    for node in data_list:
        node2idx[node] = node_size
        idx2node.append(node)
        node_size += 1
    return idx2node, node2idx


# def preprocess_nxgraph_1(aa):
#     node2idx = {}
#     idx2node = []
#     node_size = 0
#     for node in aa:
#         node2idx[node] = node_size
#         idx2node.append(node)
#         node_size += 1
#     return idx2node, node2idx

def preprocess_nxgraph_1(graph):
    node2idx = {}
    idx2node = []
    node_size = 0
    for node in graph.nodes():
        node2idx[node] = node_size
        idx2node.append(node)
        node_size += 1
    return idx2node, node2idx