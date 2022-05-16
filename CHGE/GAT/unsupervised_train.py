# import time
import numpy as np
import tensorflow as tf

from models import GAT
from utils import process
import os
checkpt_file = 'pre_trained/cora/mod_cora.ckpt'
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
Sparse = True
num_supports = 3
# change num_users and num_pois accroding to the dataset
num_users = 6952
num_pois = 14649
num_catgs=346



Batch_Size = 1
country='BR'
Epochs = 150
Patience = 100
Learning_Rate = 0.0001
Weight_Decay = 0.0005
ffd_drop = 0.
attn_drop = 0.
num_walks=100
walk_len=5
num_pos_samples=80
neg_sample_size=21
heter_weights=0.9
neg_sample_weights=0.3
Residual = False
# training params
batch_size = Batch_Size
nb_epochs = Epochs
patience = Patience
lr = Learning_Rate
l2_coef = Weight_Decay
residual = Residual
hid_units = [256] # numbers of hidden units per each attention head in each layer
n_heads = [8, 1] # additional entry for the output layer

nonlinearity = tf.nn.elu
optimizer = tf.keras.optimizers.Adam(lr = lr)

graph = process.load_graph('../dataset/GAT_dataset/'+country+'/GAT_austin_heter_graph.npz')

# print(graph.row)
node_features = process.load_features('../dataset/GAT_dataset/'+country+'/GAT_austin_user_embeddings.npy',
                              '../dataset/GAT_dataset/'+country+'/GAT_austin_poi_embeddings.npy','../dataset/GAT_dataset/'+country+'/GAT_austin_catg_embeddings.npy')

true_pairs_val, false_pairs_val = process.load_train_dataset('../dataset/graph/'+country+'/user_user_graph_train_true.csv',
                          '../dataset/graph/'+country+'/user_user_graph_train_false.csv')
true_pairs_test, false_pairs_test = process.load_test_dataset('../dataset/graph/'+country+'/user_user_graph_test_true.csv',
                          '../dataset/graph/'+country+'/user_user_graph_test_false.csv')

nb_nodes = node_features.shape[0]
ft_size = node_features.shape[1]

supports = process.construct_supports(graph, num_users,num_pois,num_catgs)

node_features=node_features[np.newaxis]

homo_samples, heter_samples = process.sample_context(
                                    graph,
                                    num_users,
                                    num_pois,
                                    num_catgs,
                                    walk_len,
                                    num_walks,
                                    num_pos_samples
                              )

print(f'These are the parameters')
print(f'batch_size: {batch_size}')
print(f'nb_nodes: {nb_nodes}')
print(f'ft_size: {ft_size}')
print(f'nb_classes: {hid_units[0]}')

if Sparse:
    support = []
    for i in range(len(supports)):
        biases =process.preprocess_adj_bias(supports[i])
        support.append(biases)
else:
    adj = supports.todense()
    adj = adj[np.newaxis]
    biases = process.adj_to_bias(adj, [nb_nodes], nhood=1)


model = GAT(hid_units, n_heads, 128, nb_nodes ,neg_sample_size,heter_weights,neg_sample_weights,Sparse, ffd_drop=ffd_drop, attn_drop=attn_drop,
            activation=tf.nn.elu, residual=False)


def train(model, inputs, bias_mat, homo_samples, heter_samples,neg_samples, training):

    with tf.GradientTape() as tape:
        logits, loss = model(inputs=inputs,
                            training=training,
                            bias_mat=bias_mat,
                            homo_samples=homo_samples,
                            heter_samples=heter_samples,
                            neg_samples=neg_samples,
                            l2_coef=l2_coef)


    gradients = tape.gradient(loss, model.trainable_variables)
    gradient_variables = zip(gradients, model.trainable_variables)
    optimizer.apply_gradients(gradient_variables)

    return logits, loss


def evaluate(model, inputs, bias_mat, homo_samples, heter_samples,neg_samples, training):
    logits, loss = model(inputs=inputs,
                            training=training,
                            bias_mat=bias_mat,
                            homo_samples=homo_samples,
                            heter_samples=heter_samples,
                            neg_samples=neg_samples,
                            l2_coef=l2_coef)

    return logits, loss

print('model: ' + str('SpGAT' if Sparse else 'GAT'))
vlss_mn = np.inf
vacc_mx = 0.0
curr_step = 0

train_loss_avg = 0
train_acc_avg = 0
val_loss_avg = 0
val_acc_avg = 0

model_number = 0
roc_auc_max=0
for epoch in range(nb_epochs):
    ###Training Segment###
    neg_samples_1 = np.random.randint(0, num_users,
                                      (num_users + num_pois + num_catgs, neg_sample_size // 3), dtype=np.int32)
    neg_samples_2 = np.random.randint(num_users, num_users + num_pois,
                                      (num_users + num_pois + num_catgs, neg_sample_size // 3), dtype=np.int32)
    neg_samples_3 = np.random.randint(num_users + num_pois, num_users + num_pois + num_catgs,
                                      (num_users + num_pois + num_catgs, neg_sample_size // 3), dtype=np.int32)
    neg_samples = np.concatenate((neg_samples_1, neg_samples_2), axis=1)
    neg_samples = np.concatenate((neg_samples, neg_samples_3), axis=1)
    neg_samples = np.vstack((neg_samples,
                             (num_users + num_pois+num_catgs) * np.ones((1, neg_sample_size), dtype=np.int32)))
    tr_step = 0
    tr_size = node_features.shape[0]
    while tr_step * batch_size < tr_size:

        if Sparse:
            bbias = support
        else:
            bbias = support[tr_step * batch_size:(tr_step + 1) * batch_size]

        _, loss_value_tr = train(model,
                                inputs=node_features[tr_step * batch_size:(tr_step + 1) * batch_size],
                                bias_mat=bbias,
                                homo_samples=homo_samples[:, epoch % num_pos_samples],
                                heter_samples=heter_samples[:, epoch % num_pos_samples],
                                neg_samples=neg_samples,
                                training=True)
        train_loss_avg = loss_value_tr
        print('Train epoch =',epoch, 'Training: loss = %.5f' %(train_loss_avg ))

        final_embeddings, _ = evaluate(model,
                                 inputs=node_features[tr_step * batch_size:(tr_step + 1) * batch_size],
                                 bias_mat=bbias,
                                 homo_samples=homo_samples[:, epoch % num_pos_samples],
                                 heter_samples=heter_samples[:, epoch % num_pos_samples],
                                 neg_samples=neg_samples,
                                 training=False)
        tr_step += 1
        final_embeddings=final_embeddings.numpy()
        roc_auc, pr_auc = process.perf_evaluate(final_embeddings, true_pairs_val, false_pairs_val, num_users)
        if roc_auc>roc_auc_max:
            roc_auc_max=roc_auc
            final_embeddings1=final_embeddings

roc_auc, pr_auc= process.perf_evaluate(final_embeddings1, true_pairs_test, false_pairs_test, num_users)
print("evaluation on test set:")
print("ROCAUC={:.5f}".format(roc_auc))
print("AP={:.5f}".format(pr_auc))
