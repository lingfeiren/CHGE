import pandas as pd
import networkx as nx
import numpy as np
from numpy import random
from scipy import sparse as sp
country='BR'
num_users = 6952
num_pois = 14649
num_catgs=346


user_poi_graph = '../dataset/graph/'+country+'/user_poi_graph.csv'
user_user_meeting_graph = '../dataset/graph/'+country+'/user_user_meeting_graph.csv'
user_catg_graph = '../dataset/graph/'+country+'/user_catg_graph.csv'
poi_poi_graph = '../dataset/graph/'+country+'/poi_poi_graph.csv'
poi_catg_graph = '../dataset/graph/'+country+'/poi_catg_graph.csv'
catg_catg_graph = '../dataset/graph/'+country+'/catg_catg_graph.csv'


def graphConstruct_user2user(user_user_graph_train):
    # timeshot, poi
    df = pd.read_csv(user_user_graph_train, sep = ',')
    df['weight'] = np.ones(len(df),dtype=np.float)
    df= df['weight'].groupby([df['source_user'],df['target_user']]).sum()
    df = df.reset_index()

    g_user2user = nx.from_pandas_edgelist(df, source='source_user',target='target_user',edge_attr=['weight'],\
                                         create_using=nx.Graph())
    return g_user2user

def graphConstruct_user2poi(user_poi_graph):
    # timeshot, poi
    df = pd.read_csv(user_poi_graph, sep = ',')
    df['weight'] = df['delta_times']
    df= df['weight'].groupby([df['user'],df['poi']]).sum()
    df = df.reset_index()

    g_user2poi = nx.from_pandas_edgelist(df, source='user',target='poi',edge_attr=['weight'],\
                                         create_using=nx.Graph())
    return g_user2poi

def graphConstruct_catg2catg(catg_catg_graph):
    # timeshot, poi
    df = pd.read_csv(catg_catg_graph, sep = ',')
    df['weight'] = np.ones(len(df),dtype=np.float)
    df= df['weight'].groupby([df['source_catg'],df['target_catg']]).sum()
    df = df.reset_index()

    g_catg2catg= nx.from_pandas_edgelist(df, source='source_catg',target='target_catg',edge_attr=['weight'],\
                                         create_using=nx.Graph())
    return g_catg2catg

def graphConstruct_poi2poi(poi_poi_graph):
    # timeshot, poi
    df = pd.read_csv(poi_poi_graph, sep = ',')
    df['weight'] = np.ones(len(df),dtype=np.float)
    df= df['weight'].groupby([df['source_poi'],df['target_poi']]).sum()
    df = df.reset_index()

    g_poi2poi= nx.from_pandas_edgelist(df, source='source_poi',target='target_poi',edge_attr=['weight'],\
                                         create_using=nx.Graph())
    return g_poi2poi

def graphConstruct_poi2catg(poi_catg_graph):
    # timeshot, poi
    df = pd.read_csv(poi_catg_graph, sep = ',')
    df['weight'] = np.ones(len(df),dtype=np.float)
    df= df['weight'].groupby([df['poi'],df['catg']]).sum()
    df = df.reset_index()

    g_poi2catg = nx.from_pandas_edgelist(df, source='poi',target='catg',edge_attr=['weight'],\
                                         create_using=nx.Graph())
    return g_poi2catg

def graphConstruct_user2catg(user_catg_graph):
    # timeshot, poi
    df = pd.read_csv(user_catg_graph, sep = ',')
    df['weight'] = np.ones(len(df),dtype=np.float)
    df= df['weight'].groupby([df['user'],df['catg']]).sum()
    df = df.reset_index()

    g_user2catg = nx.from_pandas_edgelist(df, source='user',target='catg',edge_attr=['weight'],\
                                         create_using=nx.Graph())
    return g_user2catg


g_user2poi=graphConstruct_user2poi(user_poi_graph)
g_user2catg=graphConstruct_user2catg(user_catg_graph)
g_poi2catg=graphConstruct_poi2catg(poi_catg_graph)
g_catg2catg=graphConstruct_catg2catg(catg_catg_graph)
g_poi2poi=graphConstruct_poi2poi(poi_poi_graph)


matrix = [[0 for x in range(num_users+num_pois+num_catgs)] for y in range(num_users+num_pois+num_catgs)]

for data in g_user2poi.edges():

    matrix[data[0]][data[1]]=g_user2poi[data[0]][data[1]]['weight']
    matrix[data[1]][data[0]] = g_user2poi[data[0]][data[1]]['weight']

for data in g_user2catg.edges():

    matrix[data[0]][data[1]]=g_user2catg[data[0]][data[1]]['weight']
    matrix[data[1]][data[0]] = g_user2catg[data[0]][data[1]]['weight']

for data in g_poi2poi.edges():
    matrix[data[0]][data[1]]=g_poi2poi[data[0]][data[1]]['weight']
    matrix[data[1]][data[0]] = g_poi2poi[data[0]][data[1]]['weight']

for data in g_poi2catg.edges():
    matrix[data[0]][data[1]]=g_poi2catg[data[0]][data[1]]['weight']
    matrix[data[1]][data[0]] = g_poi2catg[data[0]][data[1]]['weight']

for data in g_catg2catg.edges():
    matrix[data[0]][data[1]]=g_catg2catg[data[0]][data[1]]['weight']
    matrix[data[1]][data[0]] = g_catg2catg[data[0]][data[1]]['weight']

sp.save_npz('../dataset/GAT_dataset/'+country+'/GAT_austin_heter_graph.npz',sp.coo_matrix(matrix))

embedding = '../dataset/graph/'+country+'/user_venue_catg_embedding.txt'
fo=open(embedding)
user_embedding=[]
poi_embedding=[]
catg_embedding=[]
embedding={}
for line in fo:
    tts = line.strip(r'\s+').split()
    id=int(tts[0])
    data=list(map(float, tts[1:]))
    embedding[id]=data


for id in range(num_users):
    user_embedding.append(embedding[id])

for id in range(num_users,num_users+num_pois):
    poi_embedding.append(embedding[id])

for id in range(num_users+num_pois,num_users+num_pois+num_catgs):
    catg_embedding.append(embedding[id])

np.save('../dataset/GAT_dataset/'+country+'/GAT_austin_user_embeddings.npy', np.matrix(user_embedding))
np.save('../dataset/GAT_dataset/'+country+'/GAT_austin_poi_embeddings.npy', np.matrix(poi_embedding))
np.save('../dataset/GAT_dataset/'+country+'/GAT_austin_catg_embeddings.npy', np.matrix(catg_embedding))

