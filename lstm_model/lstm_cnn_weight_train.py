#coding=utf-8

import tensorflow as tf
import time
import os
from lstm_cnn_weight_model import LSTM_CNN_Model as LSTM_CNN_Model
import numpy as np
import Data_helper
from utils.path_util import from_project_root
from tensorflow.contrib import learn
import pickle as pk
import collections
import gensim
import utils.json_util as ju
from lstm_model.model_tool import get_term_weight,get_index_text
import tqdm
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

#Data loading params
tf.flags.DEFINE_integer("num_classes",19,"number of classes")
tf.flags.DEFINE_integer("embedding_size",64,"Dimensionality of word embedding")
tf.flags.DEFINE_integer("hidden_size",64,"Dimensionality of GRU hidden layer(default 50)") #===============
tf.flags.DEFINE_float("dev_sample_percentage",0.002,"dev_sample_percentage")
tf.flags.DEFINE_integer("batch_size",100,"Batch Size of training data(default 50)")
tf.flags.DEFINE_integer("checkpoint_every",50,"Save model after this many steps (default 100)")
tf.flags.DEFINE_integer("num_checkpoints",10,"Number of checkpoints to store (default 5)")
tf.flags.DEFINE_integer("evaluate_every",100,"evaluate every this many batches")
tf.flags.DEFINE_float("learning_rate",0.01,"learning rate")  #====================
tf.flags.DEFINE_integer("grad_clip",5,"grad clip to prevent gradient explode")
tf.flags.DEFINE_integer("epoch",5,"number of epoch")
tf.flags.DEFINE_integer("max_word_in_sent",800,"max_word_in_sent")
tf.flags.DEFINE_float("regularization_rate",0.001,"regularization rate random") #=======================

# cnn
tf.flags.DEFINE_string("filter_sizes","2,3,4","the size of the filter")
tf.flags.DEFINE_integer("num_filters",64,"the num of channels in per filter")

tf.flags.DEFINE_float("rnn_input_keep_prob",0.9,"rnn_input_keep_prob")
tf.flags.DEFINE_float("rnn_output_keep_prob",0.9,"rnn_output_keep_prob")

tf.flags.DEFINE_string("train_file0","lstm_model/processed_data/one_gram/filter-1gram_phrase_level_data_0.csv","train file url")
tf.flags.DEFINE_string("train_file1","lstm_model/processed_data/one_gram/filter-1gram_phrase_level_data_1.csv","train file url")
tf.flags.DEFINE_string("train_file2","lstm_model/processed_data/one_gram/filter-1gram_phrase_level_data_2.csv","train file url")
tf.flags.DEFINE_string("train_file3","lstm_model/processed_data/one_gram/filter-1gram_phrase_level_data_3.csv","train file url")
tf.flags.DEFINE_string("train_file4","lstm_model/processed_data/one_gram/filter-1gram_phrase_level_data_4.csv","train file url")


tf.flags.DEFINE_string("vocab_file","lstm_model/processed_data/one_gram/filter-1gram_phrase_level_vocab.pk","vocab file url")
tf.flags.DEFINE_string("vocab_file_csv","lstm_model/processed_data/one_gram/filter-1gram_phrase_level_vocab.csv","vocab csv file url")
tf.flags.DEFINE_string("word2vec_file","embedding_model/models/w2v_phrase_64_2_10_15.bin","vocab csv file url")
tf.flags.DEFINE_string("dc_file","lstm_model/processed_data/one_gram/phrase_level_1gram_dc.json","dc file url")

# add
tf.flags.DEFINE_string("dev_file","lstm_model/processed_data/one_gram/filter-1gram_phrase_level_data_dev.csv","dev file url")

FLAGS = tf.flags.FLAGS
# =====================load data========================================================================================
# load the training data
# 准备数据
print("Loading Data...")
train_x_text0,train_y0 = Data_helper.load_data_and_labels(from_project_root(FLAGS.train_file0))
train_x_text1,train_y1 = Data_helper.load_data_and_labels(from_project_root(FLAGS.train_file1))
train_x_text2,train_y2 = Data_helper.load_data_and_labels(from_project_root(FLAGS.train_file2))
train_x_text3,train_y3 = Data_helper.load_data_and_labels(from_project_root(FLAGS.train_file3))

train_x_text = []
train_x_text.extend(train_x_text0)
train_x_text.extend(train_x_text1)
train_x_text.extend(train_x_text2)
train_x_text.extend(train_x_text3)
train_y = []
train_y.extend(train_y0)
train_y.extend(train_y1)
train_y.extend(train_y2)
train_y.extend(train_y3)

dev_x_text,dev_y = Data_helper.get_predict_data(from_project_root(FLAGS.train_file4))

# =====================build vocab =====================================================================================
train_x_vecs = get_index_text(train_x_text,FLAGS.max_word_in_sent,from_project_root(FLAGS.vocab_file))
dev_x_vecs = get_index_text(dev_x_text,FLAGS.max_word_in_sent,from_project_root(FLAGS.vocab_file))

train_term_weights = get_term_weight(train_x_text,FLAGS.max_word_in_sent,from_project_root(FLAGS.dc_file))
dev_term_wegits = get_term_weight(dev_x_text, FLAGS.max_word_in_sent, from_project_root(FLAGS.dc_file))

# 使用预训练的embedding
model = gensim.models.Word2Vec.load(from_project_root(FLAGS.word2vec_file))
init_embedding_mat = []
init_embedding_mat.append([1.0] * FLAGS.embedding_size)
with open(from_project_root(FLAGS.vocab_file_csv),'r',encoding='utf-8') as f:
    for line in f.readlines():
        line_list = line.strip().split(',')
        word = line_list[0]
        if word not in model:
            init_embedding_mat.append([1.0] * FLAGS.embedding_size)
        else:
            init_embedding_mat.append(model[word])

embedding_mat = tf.Variable(init_embedding_mat,name="embedding")
print("加载数据完成.....")

# 格式化输出
print("Train / Dev split: {:d} / {:d}".format(len(train_y),len(dev_y)))

# =====================split dev and text ==============================================================================

print("data load finished!!!")

'''
vocab_size,num_classes,embedding_size=300,hidden_size=50
'''

with tf.Session() as sess:

    new_model = LSTM_CNN_Model(
        num_classes=FLAGS.num_classes,
        embedding_size = FLAGS.embedding_size,
        hidden_size = FLAGS.hidden_size,
        init_embedding_mat = embedding_mat,
        max_doc_length = FLAGS.max_word_in_sent,
        filter_sizes = list(map(int, FLAGS.filter_sizes.split(","))),
        num_filters = FLAGS.num_filters
    )

    with tf.name_scope("loss"):

        # 给GRU加上L2正则化
        tv = tf.trainable_variables()  # 得到所有可以训练的参数，即所有trainable=True的tf.Variable / tf.get_variable
        regularization_cost = FLAGS.regularization_rate * tf.reduce_sum([tf.nn.l2_loss(v) for v in tv]) #0.001是一个lambda超参数

        original_cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=new_model.input_y,
                                                                     logits=new_model.out))
        loss = original_cost + regularization_cost

        # loss =  -tf.reduce_sum(new_model.input_y*tf.log(tf.clip_by_value(new_model.out,1e-10,1.0)))

    with tf.name_scope("accuray") :
        predict = tf.argmax(new_model.out,axis=1,name="predict")
        label = tf.argmax(new_model.input_y,axis=1,name="label")
        acc = tf.reduce_mean(tf.cast(tf.equal(predict,label),tf.float32))

    # create model path
    timestamp = str( int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir,"runs",timestamp))
    print("Wrinting to {} \n".format(out_dir))

    # global step
    global_step = tf.Variable(0,trainable=False)
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)

    # RNN中常用的梯度截断，防止出现梯度过大难以求导的现象
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), FLAGS.grad_clip)
    grads_and_vars = tuple(zip(grads, tvars))
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    # Keep track of gradient values and sparsity(optional)
    grad_summaries = []
    for g, v in grads_and_vars:
        if g is not None:
            grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
            grad_summaries.append(grad_hist_summary)

    grad_summaries_merged = tf.summary.merge(grad_summaries)

    loss_summary = tf.summary.scalar('loss',loss)
    acc_summary = tf.summary.scalar('acc',acc)

    train_summary_op = tf.summary.merge([loss_summary,acc_summary,grad_summaries_merged])
    train_summary_dir = os.path.join(out_dir,"summaries","train")
    train_summary_writer = tf.summary.FileWriter(train_summary_dir,sess.graph)

    dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
    dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
    dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

    checkpoint_dir = os.path.abspath(os.path.join(out_dir,"checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir,"model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    saver = tf.train.Saver(tf.global_variables(),max_to_keep=FLAGS.num_checkpoints)

    sess.run(tf.global_variables_initializer())
    # Write vocabulary

    def train_step(x_batch,y_batch,term_weight_batch):

        feed_dict={
            new_model.input_x:x_batch,
            new_model.input_y:y_batch,
            new_model.term_weight:term_weight_batch,
            new_model.rnn_input_keep_prob:FLAGS.rnn_input_keep_prob,
            new_model.rnn_output_keep_prob:FLAGS.rnn_output_keep_prob
        }
        _,step,summaries,cost,accuracy = sess.run([train_op,global_step,train_summary_op,loss,acc],feed_dict)

        time_str = str( int(time.time()))
        print("{} : step {}, loss {:g} , acc {:g}".format(time_str,step,cost,accuracy))

        return step

    def dev_step(dev_x_vecs,dev_y,dev_term_wegits,per_predict_limit):

        sum_predict = len(dev_y)
        batch_size = int(sum_predict / per_predict_limit)

        batch_prediction_all = []
        # 一个一个进行预测
        for index in range(batch_size):

            start_index = index * per_predict_limit
            if index == batch_size - 1:
                end_index = sum_predict
            else:
                end_index = start_index + per_predict_limit

            dev_x_vecs_batch = dev_x_vecs[start_index:end_index]
            dev_term_wegits_batch = dev_term_wegits[start_index:end_index]

            feed_dict={
                new_model.input_x:dev_x_vecs_batch,
                new_model.term_weight:dev_term_wegits_batch,
                new_model.rnn_input_keep_prob: 1.0,
                new_model.rnn_output_keep_prob: 1.0
            }
            predict_result = sess.run(new_model.predict, feed_dict)
            batch_prediction_all.extend(predict_result)

        reset_prediction_all = []
        for predit in batch_prediction_all:
            reset_prediction_all.append(int(predit) + 1)

        macro_f1 = f1_score(dev_y, reset_prediction_all, average='macro')
        accuracy_score1 = accuracy_score(dev_y, reset_prediction_all, normalize=True)

        print("=====================dev===========================")
        print("macro_f1:{}".format(macro_f1))
        print("accuracy:{}".format(accuracy_score1))
        print("=====================end===========================")

    for epoch in range(FLAGS.epoch):
        print('current epoch %s' % (epoch + 1))

        for i in range(0,len(train_y)-FLAGS.batch_size,FLAGS.batch_size):

            x_batch = train_x_vecs[i:i+FLAGS.batch_size]
            y_batch = train_y[i:i+FLAGS.batch_size]
            term_weight_batch = train_term_weights[i:i+FLAGS.batch_size]
            step = train_step(x_batch,y_batch,term_weight_batch)

            if step % FLAGS.evaluate_every == 0:
                dev_step(dev_x_vecs,dev_y,dev_term_wegits,FLAGS.batch_size)

            if step % FLAGS.checkpoint_every == 0 :
                path = saver.save(sess,checkpoint_prefix,global_step=step)
                print("Saved model checkpoint to {} \n".format(path))