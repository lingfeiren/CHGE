from utils import  graphConstruct_user2catg,graphConstruct_venue2venue,graphConstruct_catg2catg
from utils import  graphConstruct_user2venue,graphConstruct_poi2catg,graphConstruct_time2catg,graphConstruct_user2user
from line import LINE

if __name__ =="__main__":

    country = 'BR'
    path_user2poi = r'../dataset/graph/' + country + '/user_poi_graph.csv'
    df_user2venue, g_user2venue = graphConstruct_user2venue(path_user2poi)
    print('graph user-poi has been contructed completely!')

    path_user2catg = r'../dataset/graph/' + country + '/user_catg_graph.csv'
    df_user2catg, g_user2catg = graphConstruct_user2catg(path_user2catg)
    print('graph user-catg has been contructed completely!')

    path_venue2venue = r'../dataset/graph/' + country + '/poi_poi_graph.csv'
    df_venue2venue, g_venue2venue = graphConstruct_venue2venue(path_venue2venue)
    print('graph venue-venue has been contructed completely!')

    path_catg2catg = r'../dataset/graph/' + country + '/catg_catg_graph.csv'
    df_catg2catg, g_catg2catg = graphConstruct_catg2catg(path_catg2catg)
    print('graph catg-catg has been contructed completely!')

    path_poi2catg = r'../dataset/graph/' + country + '/poi_catg_graph.csv'
    df_poi2catg, g_poi2catg = graphConstruct_poi2catg(path_poi2catg)
    print('graph catg-catg has been contructed completely!')

    path_time2catg = r'../dataset/graph/' + country + '/time_catg_graph.csv'
    df_time2catg, g_time2catg = graphConstruct_time2catg(path_time2catg)
    print('graph time-catg has been contructed completely!')


    user_list = sorted(list(set(df_user2catg['user'])))
    venue_list = sorted(list(set(df_user2venue['poi'])))
    catg_list = sorted(list(set(df_user2catg['catg'])))
    time_list = sorted(list(set(df_time2catg['time'])))

    model = LINE(g_user2venue,g_user2catg,g_venue2venue,g_catg2catg,g_poi2catg,g_time2catg,6,user_list,venue_list,catg_list,time_list,embedding_size=128, order='second')
    model.train(batch_size=1024, epochs=20, verbose=2)
    embeddings= model.get_embeddings()

    node_presention=r'../dataset/graph/' + country + '/user_venue_catg_embedding.txt'
    fo = open(node_presention, 'w')
    for key in embeddings.keys():
        fo.write(str(key) + " ")
        for j in range(len(embeddings[key])):
            fo.write(str(embeddings[key][j]) + " ")
        fo.write("\n")
    fo.close()
