#coding=utf-8

import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import layers
import numpy as np

def length(sequence):

    used = tf.sign(tf.reduce_max( tf.abs(sequence), reduction_indices=2))
    seq_len = tf.reduce_sum(used,reduction_indices=1) #即每个句子的单词个数

    return tf.cast(seq_len,tf.int32)


class LSTM_CNN_Model():

    def __init__(self,num_classes,init_embedding_mat,max_doc_length,filter_sizes,
                 num_filters,embedding_size=300,hidden_size=50):

        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.init_embedding_mat = init_embedding_mat
        self.max_doc_length = max_doc_length
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters

        with tf.name_scope("placeholder"):

            # x.shape为 [batch_size,文档单词个数]
            self.input_x = tf.placeholder(tf.int32,[None,None],name="input_x")
            self.input_y = tf.placeholder(tf.float32,[None,num_classes],name="input_y")

            self.rnn_input_keep_prob = tf.placeholder(tf.float32,name="rnn_input_keep_prob")
            self.rnn_output_keep_prob = tf.placeholder(tf.float32,name="rnn_output_keep_prob")

            # 句子级别上最大的单词数
            # self.batch_size = tf.placeholder(tf.int32, name="batch_size")

        # create model
        # share embedding_mat
        word_embedded_x = self.word2vec(self.input_x,"embedding")

        # word_encoder
        # word_encode.shape为 [ -1,max_time,hidden_size*2]
        self.word_encoded_x = self.BidirectionalGRUEncoder(word_embedded_x,"word_encoder")

        # 共享attention layer中的所有参数
        # word_encoded.shape [batch , max_doc_lenght , hiddent_size * 2]
        word_context = tf.Variable(tf.truncated_normal(shape=[self.hidden_size * 2], dtype=tf.float32))
        W_word = tf.Variable(
            tf.truncated_normal(shape=[self.hidden_size * 2, self.hidden_size * 2], dtype=tf.float32),
            name="word_context_weight")
        b_word = tf.Variable(tf.truncated_normal(shape=[self.hidden_size*2], dtype=tf.float32),
                                 name="word_context_bias")
        # # word attention [batch , max_doc_length]
        weight_x = self.AttentionLayer(self.word_encoded_x,word_context,W_word,b_word,name="attention_1")

        # # 根据新的embedding,和 weights ，计算两个句子的距离。 # feature one #############
        with tf.name_scope("sentence_encode"): # [batch,hidden_size * 2]
            self.rnn_atten_encode = self.sentence_encoder(self.word_encoded_x,weight_x)

        # 根据gru编码的encode 在进行CNN
        # word_encode_x.shape = [-1,max_doc_lenght,hiddent_size * 2 ]
        # cnn_output.shape = [batch, num_filters * 2 * len(filter_sizes)]
        with tf.name_scope("cnn_layer") :
            # 先将rnn计算到的权重先相乘
            # word_encoded_x.shape = [batch,sequence_length,hidden_size*2]
            # weight_x.shape=[batch,max_doc_length]
            expand_weight_x = tf.expand_dims(weight_x, -1)
            # atten_sens.shape = [batch,hidden_size * 2]
            self.word_encoded_x = tf.multiply(self.word_encoded_x, expand_weight_x)

            cnn_output = self.cnn_layer(self.word_encoded_x)

        # 再接入一个全连接层
        with tf.name_scope("fully_connection_layer") :
            # self.out = self.fully_connection_layer(cnn_output,self.rnn_atten_encode)
            self.out = self.fully_connection_layer(cnn_output,self.rnn_atten_encode)

    # embedding_layer
    def word2vec(self,input,scope_name):

        with tf.name_scope(scope_name):
            # 初始化一个vocab_size * embedding_size 的权重矩阵 ，用作初始化embedding
            # embedding_mat = tf.Variable(self.init_embedding_mat,name="embedding_mat",dtype=tf.float32)
            # embedding_mat = tf.Variable(tf.truncated_normal((self.vocab_size,self.embedding_size)))

            word_embedding = tf.nn.embedding_lookup(self.init_embedding_mat,input,name="word_embedding")

        return word_embedding


    def BidirectionalGRUEncoder(self,word_encoded,name):

        with tf.name_scope(name):
            # 输入的inputs的shape是[batch_size,max_time,embedding_size]
            GRU_cell_fw = rnn.GRUCell(self.hidden_size)
            GRU_cell_bw = rnn.GRUCell(self.hidden_size)

            GRU_cell_fw_dr = tf.nn.rnn_cell.DropoutWrapper(GRU_cell_fw, input_keep_prob=self.rnn_input_keep_prob,
                                                           output_keep_prob=self.rnn_output_keep_prob)
            GRU_cell_bw_dr = tf.nn.rnn_cell.DropoutWrapper(GRU_cell_bw,input_keep_prob=self.rnn_input_keep_prob,
                                                           output_keep_prob=self.rnn_output_keep_prob)

            ((fw_outputs,bw_outputs),(_,_)) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=GRU_cell_fw_dr,
                cell_bw=GRU_cell_bw_dr,
                inputs=word_encoded,
                sequence_length=length(word_encoded),
                dtype=tf.float32
            )
            # outputs的size是[batch_size,max_time,hidden_size*2]
            outputs = tf.concat((fw_outputs,bw_outputs),2,name="hidden_state")
            return outputs

    def AttentionLayer(self,word_encoded,word_context,W_word,b_word,name):

        with tf.name_scope(name) :
            # U_{it} = tanh(W_{w}u_{it} + b_{w})
            # 其实就是将原来的word_encoder进行重新编码。维度不变。
            mat_multi = tf.matmul(W_word,tf.reshape(word_encoded,[-1,self.hidden_size*2]),transpose_a=False
                                  ,transpose_b=True)
            U_w = tf.tanh(tf.reshape( tf.transpose(mat_multi),shape=[-1,self.max_doc_length,self.hidden_size*2]) + b_word,name="U_w")

            # 计算词权重
            # expand_word_context.shape [hidden_size * 2 , 1 ]
            expand_word_context = tf.expand_dims(word_context,-1)
            # word_logits.shape [batch,max_doc_length]
            word_logits = tf.reshape(tf.matmul(tf.reshape(U_w,shape=[-1,self.hidden_size*2]),expand_word_context),shape=[-1,self.max_doc_length])

            # alpha.shape [batch , max_doc_length]
            alpha = tf.nn.softmax(logits=word_logits,name="alpha")

        return alpha

    # 将两个句子表示 之间进行拼接，作为featue_layer,再接一个全连接层，进行分类
    def sentence_encoder(self,word_encoded_x,weight_x):

        # atten_sen1/2.shape 为：[batch_size,hidden_size*2]

        # word_encode.shape [batch, max_doc_length, hidden_size*2]
        # weight.shape [batch, max_doc_length , 1]
        expand_weight_x = tf.expand_dims(weight_x,-1)
        # atten_sens.shape = [batch,hidden_size * 2]
        atten_sen = tf.reduce_sum(tf.multiply(word_encoded_x,expand_weight_x),axis=1,name="atten_sen")

        return atten_sen

    def cnn_layer(self,word_encode):
        """
        :param word_encode: 经过GRU,进行编码, shape = [batch, max_doc_length,hidden_size *2 ]
        :return:h_pool_flat.shape = [batch,num_filters * len(filter_sizes)]
        """
        # expand_word_encode.shape= [batch,max_doc_length, hidden_size*2,1]
        expand_word_encode = tf.expand_dims(word_encode,-1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []  # 对于每一个filter_size ，将每个池化层得到
        for i, filter_size in enumerate(self.filter_sizes):
            filter_size = int(filter_size)
            with tf.name_scope("conv-maxpool-%s" % filter_size):

                # Convolution Layer
                filter_shape = [filter_size,self.hidden_size*2,1,self.num_filters]

                W = tf.Variable(tf.truncated_normal(filter_shape,stddev=0.1),name="W")
                b = tf.Variable(tf.constant(0.1,shape=[self.num_filters]),name="b1")

                # 卷积层 shape: [batch, max_doc_length - filter_szie + 1 , 1 , num_filters]
                conv = tf.nn.conv2d(expand_word_encode,W,strides=[1,1,1,1],padding='VALID',name="conv1")

                # 非线性激活函数
                # h1.shape=[batch,max_doc_length - filter_szie + 1 , 1 , num_filters]
                h = tf.nn.relu( tf.nn.bias_add(conv,b),name="relu1")

                # 采用最大池化层 pooled.shap = [batch,1,1,num_filters]
                max_pooled = tf.nn.max_pool(h,ksize=[1,self.max_doc_length-filter_size+1,1,1],strides=[1,1,1,1],padding='VALID',name="pool1")
                average_pooled = tf.nn.avg_pool(h,ksize=[1,self.max_doc_length-filter_size+1,1,1],strides=[1,1,1,1],padding='VALID',name="pool2")

                # 汇总所有不同filter_size的池化结果
                pooled_outputs.append(max_pooled)
                pooled_outputs.append(average_pooled)

        # 汇总所有的池化后的特征
        num_filters_total = self.num_filters * 2 * len(self.filter_sizes)

        h_pool = tf.concat(pooled_outputs, 3)

        # h_pool_flat.shape = [batch, num_filters_total]
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total],name="cnn_output")

        # 添加 dropout，规则化
        with tf.name_scope("dropout"):
            h_drop = tf.nn.dropout(h_pool_flat, self.rnn_input_keep_prob)

        return h_drop

    # 全连接层
    def fully_connection_layer(self,cnn_output,rnn_atten_encode):
        """
        :param cnn_output: cnn传入的feature . shape = [batch,num_filters * 2 * len(filter_sizes)]
        :param rnn_atten_encode: shape=[batch,hidden_size * 2]
        :return: fc_output
        """
        # 拼接
        sum_input = tf.concat([cnn_output,rnn_atten_encode],1)
        # sum_input = cnn_output

        # 增加多一层全连接层
        fc_output1 = tf.contrib.layers.fully_connected(inputs=sum_input, num_outputs=128,
                                                      activation_fn=None)

        # fc_output.shape = [batch ,num_classes]
        fc_output = tf.contrib.layers.fully_connected(inputs=fc_output1, num_outputs=self.num_classes, activation_fn=None)
        predict = tf.argmax(fc_output, axis=1, name="prediction") # 预测结果

        return fc_output