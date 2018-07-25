#coding=utf-8

import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import layers
import numpy as np

def length(sequence):

    used = tf.sign(tf.reduce_max( tf.abs(sequence), reduction_indices=2))
    seq_len = tf.reduce_sum(used,reduction_indices=1) #即每个句子的单词个数

    return tf.cast(seq_len,tf.int32)

class Model():

    def __init__(self,vocab_size,num_classes,init_embedding_mat,max_doc_length
        ,embedding_size=300,hidden_size=50):

        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.init_embedding_mat = init_embedding_mat
        self.max_doc_length = max_doc_length

        with tf.name_scope("placeholder"):

            #x.shape为 [batch_size,文档单词个数]
            self.input_x = tf.placeholder(tf.int32,[None,None],name="input_x_1")
            self.input_y = tf.placeholder(tf.float32,[None,num_classes],name="input_y")

            self.rnn_input_keep_prob = tf.placeholder(tf.float32,name="rnn_input_keep_prob")
            self.rnn_output_keep_prob = tf.placeholder(tf.float32,name="rnn_output_keep_prob")

            # 句子级别上最大的单词数
            self.batch_size = tf.placeholder(tf.int32, name="batch_size")

        #create model
        #share embedding_mat
        word_embedded_x1 = self.word2vec(self.input_x_1,"embedding_x1")
        word_embedded_x2 = self.word2vec(self.input_x_2,"embedding_x2")

        # word_encoder
        # word_encode.shape为 [ -1,max_time,hidden_size*2]
        with tf.variable_scope("rnn_encoder") as share_encoder_scope :
            word_encoded_x1 = self.BidirectionalGRUEncoder(word_embedded_x1,"word_encoder_x1")

        # 共享attention layer中的所有参数
        # word_encoded.shape [batch , max_doc_lenght , hiddent_size * 2]
        word_context = tf.Variable(tf.truncated_normal(shape=[self.hidden_size * 2], dtype=tf.float32))
        W_word = tf.Variable(
            tf.truncated_normal(shape=[self.hidden_size * 2, self.hidden_size * 2], dtype=tf.float32),
            name="word_context_weight")
        b_word = tf.Variable(tf.truncated_normal(shape=[self.hidden_size*2], dtype=tf.float32),
                                 name="word_context_bias")
        # word attention
        weight_x1 = self.AttentionLayer(word_encoded_x1,word_context,W_word,b_word,name="attention_1")

#根据新的embedding,和 weights ，计算两个句子的距离。
        out = self.classifier(word_encoded_x1,weight_x1)

        self.out = out

    #embedding_layer
    def word2vec(self,input,scope_name):

        with tf.name_scope(scope_name):
            #初始化一个vocab_size * embedding_size 的权重矩阵 ，用作初始化embedding
            # embedding_mat = tf.Variable(self.init_embedding_mat,name="embedding_mat",dtype=tf.float32)
            # embedding_mat = tf.Variable(tf.truncated_normal((self.vocab_size,self.embedding_size)))

            word_embedding = tf.nn.embedding_lookup(self.init_embedding_mat,input,name="word_embedding")

        return word_embedding


    def BidirectionalGRUEncoder(self,word_encoded,name):

        with tf.name_scope(name):
            #输入的inputs的shape是[batch_size,max_time,embedding_size]
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
            #outputs的size是[batch_size,max_time,hidden_size*2]
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


    # 将 两个句子表示 之间进行拼接，作为featue_layer,再接一个全连接层，进行分类
    def classifier(self,word_encoded_x,weight_x):

        #atten_sen1/2.shape 为：[batch_size,hidden_size*2]

        # word_encode.shape [batch, max_doc_length, hidden_size*2]
        # weight.shape [batch, max_doc_length , 1]
        expand_weight_x = tf.expand_dims(weight_x,-1)
        # atten_sens.shape = [batch,]
        atten_sen = tf.reduce_sum(tf.multiply(word_encoded_x,expand_weight_x),axis=1)

        with tf.name_scope("classification"):

            # union = tf.concat((atten_sen1,atten_sen2),axis=1)

            # out1 = layers.fully_connected(inputs=union,num_outputs=30,activation_fn=tf.nn.tanh)
            out = layers.fully_connected(inputs=union, num_outputs=self.num_classes, activation_fn=None)

            predict = tf.argmax(out, axis=1, name="prediction")

            return out
