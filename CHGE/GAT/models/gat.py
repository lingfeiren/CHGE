import numpy as np
import tensorflow as tf
from utils.layers import *

def choose_attn_head(Sparse):
    if Sparse:
        chosen_attention = sp_attn_head
    else:
        chosen_attention = attn_head

    return chosen_attention

class inference(tf.keras.layers.Layer):
    def __init__(self, n_heads, hid_units, nb_classes, nb_nodes, Sparse, ffd_drop=0.0, attn_drop=0.0,
                 activation=tf.nn.elu, residual=False):
        super(inference, self).__init__()
        attned_head = choose_attn_head(Sparse)
        self.attns = []
        self.sec_attns = []
        self.final_attns = []
        self.final_sum = n_heads[-1]
        # 构造 n_heads[0] 个 attention
        for i in range(n_heads[0]):
            self.attns.append(attned_head(hidden_dim=hid_units[0], nb_nodes=nb_nodes,
                                          in_drop=ffd_drop, coef_drop=attn_drop,
                                          activation=activation,
                                          residual=residual))

        # hid_units表示每一个attention head中每一层的隐藏单元个数
        # 若给定hid_units = [8], 表示使用单个全连接层
        # 因此，不执行下面的代码
        for i in range(1, len(hid_units)):
            sec_attns = []
            for j in range(n_heads[i]):
                sec_attns.append(attned_head(hidden_dim=hid_units[i], nb_nodes=nb_nodes,
                                             in_drop=ffd_drop, coef_drop=attn_drop,
                                             activation=activation,
                                             residual=residual))
                self.sec_attns.append(sec_attns)

        # 加上输出层
        for i in range(n_heads[-1]):
            self.final_attns.append(attned_head(hidden_dim=nb_classes, nb_nodes=nb_nodes,
                                                in_drop=ffd_drop, coef_drop=attn_drop,
                                                activation=lambda x: x,
                                                residual=residual))

    def __call__(self, inputs, bias_mat, training):
        first_attn = []
        out = []
        # 计算 n_heads[0] 个 attention
        for indiv_attn in self.attns:
            first_attn.append(indiv_attn(seq=inputs, bias_mat=bias_mat, training=training))
        # h_1.shape: (num_graph, num_nodes, hidden_dim*n_heads[0])
        h_1 = tf.concat(first_attn, axis=-1)
        # 如果 attention 使用了多层网络，则依次计算
        for sec_attns in self.sec_attns:
            next_attn = []
            for indiv_attns in sec_attns:
                next_attn.append(indiv_attns(seq=h_1, bias_mat=bias_mat, training=training))
            h_1 = tf.concat(next_attn, axis=-1)
        # 得到最终的预测结果
        for indiv_attn in self.final_attns:
            out.append(indiv_attn(seq=h_1, bias_mat=bias_mat, training=training))
        # 将结果在最后一个维度取均值
        # logits.shape: (num_graph, num_nodes, nb_classes)
        logits = tf.add_n(out) / self.final_sum
        return logits


class GAT(tf.keras.Model):
    def __init__(self, hid_units, n_heads, nb_classes, nb_nodes ,neg_sample_size,heter_weights,neg_sample_weights,Sparse, ffd_drop=0.0, attn_drop=0.0,
                 activation=tf.nn.elu, residual=False):
        super(GAT, self).__init__()
        '''
        hid_units: 隐藏单元个数
        n_heads: 每层使用的注意力头个数
        nb_classes: 类别数，7
        nb_nodes: 节点的个数，2708
        activation: 激活函数
        residual: 是否使用残差连接
        '''
        self.hid_units = hid_units  # [128]
        self.n_heads = n_heads  # [8,1]
        self.nb_classes = nb_classes
        self.nb_nodes = nb_nodes
        self.activation = activation
        self.residual = residual
        self.neg_sample_size=neg_sample_size

        self.inferencing = inference(n_heads, hid_units, nb_classes, self.nb_nodes, Sparse=Sparse, ffd_drop=ffd_drop,
                                     attn_drop=attn_drop, activation=activation, residual=residual)
        self.skip_gram = SkipGramLayer(self.nb_classes,heter_weights=heter_weights, neg_sample_weights=neg_sample_weights)

    def masked_softmax_cross_entropy(self, logits, labels, mask):
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        mask = tf.cast(mask, dtype=tf.float32)
        mask /= tf.reduce_mean(mask)
        loss *= mask
        return tf.reduce_mean(loss)

    def masked_accuracy(self, logits, labels, mask):
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        accuracy_all = tf.cast(correct_prediction, tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        mask /= tf.reduce_mean(mask)
        accuracy_all *= mask
        return tf.reduce_mean(accuracy_all)

    def __call__(self, inputs, training, bias_mat, homo_samples, heter_samples,neg_samples,l2_coef):
        logits0 = self.inferencing(inputs=inputs, bias_mat=bias_mat[0], training=training)
        logits=logits0
        self.outputs = tf.squeeze(logits, 0)
        self.outputs = tf.nn.l2_normalize(self.outputs, 1)
        homo_sampling_idx = homo_samples
        heter_sampling_idx = heter_samples
        neg_sampling_idx = neg_samples

        homo_sampling_idx1=[int(x) for x in homo_sampling_idx.tolist()]
        heter_sampling_idx1=[int(x) for x in heter_sampling_idx.tolist()]

        self.homo_samples = tf.stop_gradient(tf.gather(self.outputs, homo_sampling_idx1))
        self.heter_samples = tf.stop_gradient(tf.gather(self.outputs, heter_sampling_idx1))

        neg_samples_list = list()
        for i in range(self.neg_sample_size):
            neg_samples_list.append(tf.stop_gradient(tf.gather(self.outputs, neg_sampling_idx[:, i])))
        self.neg_samples = tf.stack(neg_samples_list, axis=1)

        loss=self.skip_gram.loss(self.outputs, self.homo_samples, self.heter_samples, self.neg_samples)

        lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in self.trainable_variables if v.name not
                           in ['bias', 'gamma', 'b', 'g', 'beta']]) * l2_coef

        loss = loss + lossL2


        return self.outputs, loss

