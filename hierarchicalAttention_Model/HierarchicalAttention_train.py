# -*- coding: utf-8 -*-
# training the model.
import tensorflow as tf
import os
import numpy as np
from hierarchicalAttention_Model.HierarchicalAttention_model import HierarchicalAttention
import lstm_model.Data_helper as Data_helper
from utils.path_util import from_project_root
import pickle as pk
from lstm_model.model_tool import get_term_weight,get_index_text
import gensim

# configuration
FLAGS=tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("num_classes",19,"number of label")
tf.app.flags.DEFINE_float("learning_rate",0.01,"learning rate") #TODO 0.01
tf.app.flags.DEFINE_integer("batch_size", 100, "Batch size for training/evaluating.") # 批处理的大小 32-->128 #TODO
tf.app.flags.DEFINE_integer("decay_steps", 6000, "how many steps before decay learning rate.") # 6000批处理的大小 32-->128
tf.app.flags.DEFINE_float("decay_rate", 1.0, "Rate of decay for learning rate.") # 0.87一次衰减多少
tf.app.flags.DEFINE_integer("sequence_length",800,"max sentence length")
tf.app.flags.DEFINE_integer("embed_size",128,"embedding size")
tf.app.flags.DEFINE_boolean("is_training",True,"is traning.true:tranining,false:testing/inference")
tf.app.flags.DEFINE_integer("num_epochs",20,"number of epochs to run.")
tf.app.flags.DEFINE_integer("validate_every", 100, "Validate every validate_every epochs.") # 每10轮做一次验证
tf.app.flags.DEFINE_boolean("use_embedding",True,"whether to use embedding or not.")

tf.app.flags.DEFINE_string("traning_data_path","train-zhihu4-only-title-all.txt","path of traning data.") # O.K.train-zhihu4-only-title-all.txt-->training-data/test-zhihu4-only-title.txt--->'training-data/train-zhihu5-only-title-multilabel.txt'
tf.app.flags.DEFINE_string("word2vec_model_path","zhihu-word2vec-title-desc.bin-100","word2vec's vocabulary and vectors") # zhihu-word2vec.bin-100-->zhihu-word2vec-multilabel-minicount15.bin-100
tf.app.flags.DEFINE_boolean("multi_label_flag",True,"use multi label or single label.")
tf.app.flags.DEFINE_integer("num_sentences", 4, "number of sentences in the document") # 每10轮做一次验证
tf.app.flags.DEFINE_integer("hidden_size",100,"hidden size")

# prepare thr training data
tf.flags.DEFINE_string("train_file0", "lstm_model/processed_data/one_gram/filter-1gram_phrase_level_data_200_0.csv","train file url")
tf.flags.DEFINE_string("train_file1", "lstm_model/processed_data/one_gram/filter-1gram_phrase_level_data_200_1.csv","train file url")
tf.flags.DEFINE_string("train_file2", "lstm_model/processed_data/one_gram/filter-1gram_phrase_level_data_200_2.csv","train file url")
tf.flags.DEFINE_string("train_file3", "lstm_model/processed_data/one_gram/filter-1gram_phrase_level_data_200_3.csv","train file url")
tf.flags.DEFINE_string("train_file4", "lstm_model/processed_data/one_gram/filter-1gram_phrase_level_data_200_4.csv","train file url")

tf.flags.DEFINE_string("vocab_file","lstm_model/processed_data/one_gram/filter-1gram_phrase_level_data_200_vocab.pk","vocab file url")
tf.flags.DEFINE_string("vocab_file_csv","lstm_model/processed_data/one_gram/filter-1gram_phrase_level_data_200_vocab.csv","vocab csv file url")
tf.flags.DEFINE_string("word2vec_file", "embedding_model/models/w2v_phrase_128_2_10_15.bin", "vocab csv file url")
tf.flags.DEFINE_string("dc_file", "lstm_model/processed_data/one_gram/phrase_level_1gram_dc.json", "dc file url")

FLAGS = tf.flags.FLAGS
# =====================load data========================================================================================
# load the training data
# 准备数据
print("Loading Data...")
train_x_text0, train_y0 = Data_helper.load_data_and_labels(from_project_root(FLAGS.train_file0))
train_x_text1, train_y1 = Data_helper.load_data_and_labels(from_project_root(FLAGS.train_file1))
train_x_text3, train_y3 = Data_helper.load_data_and_labels(from_project_root(FLAGS.train_file3))
train_x_text4, train_y4 = Data_helper.load_data_and_labels(from_project_root(FLAGS.train_file4))
train_x_text = []
train_x_text.extend(train_x_text0)
train_x_text.extend(train_x_text1)
train_x_text.extend(train_x_text3)
train_x_text.extend(train_x_text4)
train_y = []
train_y.extend(train_y0)
train_y.extend(train_y1)
train_y.extend(train_y3)
train_y.extend(train_y4)

vocab = pk.load(open(from_project_root(FLAGS.vocab_file)),'rb')

# dev_x_text,dev_y = Data_helper.get_predict_data(from_project_root(FLAGS.train_file2))
dev_x_text,dev_y = Data_helper.load_data_and_labels(from_project_root(FLAGS.train_file2))

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

# 2.create session.
config=tf.ConfigProto()
config.gpu_options.allow_growth=True
with tf.Session(config=config) as sess:
    # Instantia te Model
    # num_classes, learning_rate, batch_size, decay_steps, decay_rate,sequence_length,num_sentences,vocab_size,embed_size,
    # hidden_size,is_training
    model=HierarchicalAttention(FLAGS.num_classes,
                                FLAGS.learning_rate,
                                FLAGS.batch_size,
                                FLAGS.decay_steps,
                                FLAGS.decay_rate,
                                FLAGS.sequence_length,
                                FLAGS.num_sentences,
                                len(vocab),
                                FLAGS.embed_size,
                                FLAGS.hidden_size,
                                FLAGS.is_training,
                                embedding_mat,
                                multi_label_flag=FLAGS.multi_label_flag)
    #




