import tensorflow as tf
import numpy as np


class attn_head(tf.keras.layers.Layer):
    def __init__(self, hidden_dim, nb_nodes=None, in_drop=0.0, coef_drop=0.0, activation=tf.nn.elu, residual=False):
        super(attn_head, self).__init__()
        self.activation = activation
        self.residual = residual

        self.in_dropout = tf.keras.layers.Dropout(in_drop)
        self.coef_dropout = tf.keras.layers.Dropout(coef_drop)
        self.conv_no_bias = tf.keras.layers.Conv1D(hidden_dim, 1, use_bias=False)
        self.conv_f1 = tf.keras.layers.Conv1D(1, 1)
        self.conv_f2 = tf.keras.layers.Conv1D(1, 1)

        self.conv_residual = tf.keras.layers.Conv1D(hidden_dim, 1)
        self.bias_zero = tf.Variable(tf.zeros(hidden_dim))

    def __call__(self, seq, bias_mat, training):
        # 输入的节点特征
        seq = self.in_dropout(seq, training=training)
        # 使用 hidden_dim=8 个1维卷积，卷积核大小为1
        # 相当于 Wh
        # seq_fts.shape: (num_graph, num_nodes, hidden_dim)
        seq_fts = self.conv_no_bias(seq)
        # 1x1 卷积可以理解为按hidden_dim这个通道进行加权求和，但参数共享
        # 相当于单输出全连接层1
        # f_1.shape: (num_graph, num_nodes, 1)
        f_1 = self.conv_f1(seq_fts)
        # 相当于单输出全连接层2
        f_2 = self.conv_f2(seq_fts)
        # 广播机制计算(num_graph,num_nodes,1)+(num_graph,1,num_nodes)
        # logits.shape: (num_graph, num_nodes, num_nodes)
        # 相当于计算了所有节点的 [e_ij]
        logits = f_1 + tf.transpose(f_2,[0,2,1])
        # 得到邻居节点的注意力系数：[alpha_ij]
        # coefs.shape: (num_graph, num_nodes, num_nodes)
        coefs = tf.nn.softmax(tf.nn.leaky_relu(logits,alpha=0.25)+bias_mat)
        # dropout
        coefs = self.coef_dropout(coefs,training = training)
        seq_fts = self.in_dropout(seq_fts,training = training)
        # 计算：[alpha_ij] x Wh
        # vals.shape: (num_graph, num_nodes, num_nodes)
        vals = tf.matmul(coefs, seq_fts)
        vals = tf.cast(vals, dtype=tf.float32)
        # 最终结果再加上一个 bias
        ret = vals + self.bias_zero
        # 残差
        if self.residual:
            if seq.shape[-1] != ret.shape[-1]:
                ret = ret + self.conv_residual(seq)
            else:
                ret = ret + seq
        # 返回 h' = σ([alpha_ij] x Wh)
        # shape: (num_graph, num_nodes, hidden_dim)
        return self.activation(ret)


class sp_attn_head(tf.keras.layers.Layer):
    def __init__(self, hidden_dim, nb_nodes, in_drop=0.0, coef_drop=0.0, activation=tf.nn.elu, residual=False):
        super(sp_attn_head, self).__init__()
        self.hidden_dim = hidden_dim
        self.nb_nodes = nb_nodes
        self.activation = activation
        self.residual = residual

        self.in_dropout = tf.keras.layers.Dropout(in_drop)
        self.coef_dropout = tf.keras.layers.Dropout(coef_drop)

        self.conv_no_bias = tf.keras.layers.Conv1D(hidden_dim, 1, use_bias=False)
        self.conv_f1 = tf.keras.layers.Conv1D(1, 1)
        self.conv_f2 = tf.keras.layers.Conv1D(1, 1)

        self.conv_residual = tf.keras.layers.Conv1D(hidden_dim, 1)
        self.bias_zero = tf.Variable(tf.zeros(hidden_dim))

    def __call__(self, seq, bias_mat, training):
        adj_mat = bias_mat
        seq = self.in_dropout(seq, training=training)
        seq_fts = self.conv_no_bias(seq)
        f_1 = self.conv_f1(seq_fts)
        f_2 = self.conv_f2(seq_fts)
        f_1 = tf.reshape(f_1, (self.nb_nodes, 1))
        f_1 = adj_mat * f_1
        f_2 = tf.reshape(f_2, (self.nb_nodes, 1))
        f_2 = adj_mat * tf.transpose(f_2, [1, 0])
        logits = tf.compat.v1.sparse_add(f_1, f_2)

        lrelu = tf.SparseTensor(indices=logits.indices,
                                values=tf.nn.leaky_relu(logits.values,alpha=0.3),
                                dense_shape=logits.dense_shape)
        coefs = tf.compat.v2.sparse.softmax(lrelu)

        if training != False:
            coefs = tf.SparseTensor(indices=coefs.indices,
                                    values=self.coef_dropout(coefs.values, training=training),
                                    dense_shape=coefs.dense_shape)
            seq_fts = self.in_dropout(seq_fts, training=training)

        coefs = tf.compat.v2.sparse.reshape(coefs, [self.nb_nodes, self.nb_nodes])

        seq_fts = tf.squeeze(seq_fts)
        vals = tf.sparse.sparse_dense_matmul(coefs, seq_fts)
        vals = tf.expand_dims(vals, axis=0)
        vals.set_shape([1, self.nb_nodes, self.hidden_dim])

        ret = vals + self.bias_zero
        if self.residual:
            if seq.shape[-1] != ret.shape[-1]:
                ret = ret + self.conv_residual(seq)
            else:
                ret = ret + seq
        return self.activation(ret)

class SkipGramLayer():
    def __init__(self, input_dim, heter_weights=1.0, neg_sample_weights=1.0, **kwargs):
        super(SkipGramLayer, self).__init__(**kwargs)

        self.input_dim = input_dim
        self.heter_weights = heter_weights
        self.neg_sample_weights = neg_sample_weights

    def affinity(self, inputs1, inputs2):
        # element-wise production
        # 1-D tensor of shape (batch_size, )
        result = tf.reduce_sum(inputs1 * inputs2, axis=1)
        return result

    def neg_cost(self, inputs1, neg_samples):
        # neg sample size: (batch_size, num_neg_samples, input_dim)
        # (batch_size, 1, input_dim)
        inputs1_reshaped = tf.expand_dims(inputs1, axis=1)
        # tensor of shape (batch_size, 1, num_neg_samples)
        neg_aff = tf.matmul(inputs1_reshaped, tf.transpose(neg_samples, perm=[0, 2, 1]))
        # squeeze
        neg_aff = tf.squeeze(neg_aff, [1])
        return neg_aff

    def loss(self, inputs1, inputs2_u, inputs2_l, neg_samples):
        return self._skipgram_loss(inputs1, inputs2_u, inputs2_l, neg_samples)

    def _skipgram_loss(self, inputs1, inputs2_u, inputs2_l, neg_samples):
        aff_1 = self.affinity(inputs1, inputs2_u)
        aff_2 = self.affinity(inputs1, inputs2_l)
        neg_cost = self.neg_cost(inputs1, neg_samples)
        true_1_xent = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(aff_1), logits=aff_1)
        true_2_xent = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(aff_2), logits=aff_2)
        negative_xent = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(neg_cost), logits=neg_cost)
        loss = tf.reduce_mean(true_1_xent) + self.heter_weights * tf.reduce_mean(true_2_xent) + \
               self.neg_sample_weights * tf.reduce_mean(tf.reduce_sum(negative_xent, axis=1))
        return loss


class SemiSupSkipGramLayer():
    def __init__(self, input_dim,num_users,heter_weights=1.0, neg_sample_weights=1.0,semi_sup_weights=1.0, **kwargs):
        super(SemiSupSkipGramLayer, self).__init__(**kwargs)

        self.input_dim = input_dim
        self.heter_weights = heter_weights
        self.neg_sample_weights = neg_sample_weights
        self.semi_sup_weights = semi_sup_weights
        self.num_users=num_users

    def affinity(self, inputs1, inputs2):
        # element-wise production
        # 1-D tensor of shape (batch_size, )
        result = tf.reduce_sum(inputs1 * inputs2, axis=1)
        return result

    def neg_cost(self, inputs1, neg_samples):
        # neg sample size: (batch_size, num_neg_samples, input_dim)
        # (batch_size, 1, input_dim)
        inputs1_reshaped = tf.expand_dims(inputs1, axis=1)
        # tensor of shape (batch_size, 1, num_neg_samples)
        neg_aff = tf.matmul(inputs1_reshaped, tf.transpose(neg_samples, perm=[0, 2, 1]))
        # squeeze
        neg_aff = tf.squeeze(neg_aff, [1])
        return neg_aff

    def loss(self, inputs1, inputs2_u, inputs2_l, neg_samples,semi_sup_samples):
        return self._skipgram_loss(inputs1, inputs2_u, inputs2_l, neg_samples,semi_sup_samples)

    def _skipgram_loss(self, inputs1, inputs2_u, inputs2_l, neg_samples,semi_sup_samples):
        aff_1 = self.affinity(inputs1, inputs2_u)
        aff_2 = self.affinity(inputs1, inputs2_l)
        aff_semi_sup = self.affinity(inputs1[:self.num_users], semi_sup_samples)
        neg_cost = self.neg_cost(inputs1, neg_samples)
        true_1_xent = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(aff_1), logits=aff_1)
        true_2_xent = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(aff_2), logits=aff_2)
        negative_xent = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(neg_cost), logits=neg_cost)
        semi_sup_xent = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(aff_semi_sup), logits=aff_semi_sup)
        loss = tf.reduce_mean(true_1_xent) + self.heter_weights * tf.reduce_mean(true_2_xent) + \
               self.neg_sample_weights * tf.reduce_mean(tf.reduce_sum(negative_xent, axis=1))+ \
                self.semi_sup_weights * tf.reduce_mean(semi_sup_xent)
        return loss