# -*- coding: utf-8 -*-
# training the model.
import tensorflow as tf
import os
import numpy as np
from hierarchicalAttention_Model.HierarchicalAttention_model import HierarchicalAttention
import hierarchicalAttention_Model.Data_helper as Data_helper
from utils.path_util import from_project_root
import pickle as pk
from lstm_model.model_tool import get_term_weight,get_index_text
import gensim
import time
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

# configuration
FLAGS=tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("num_classes",19,"number of label")
tf.app.flags.DEFINE_float("learning_rate",0.002,"learning rate") # TODO 0.01
tf.app.flags.DEFINE_integer("batch_size", 100, "Batch size for training/evaluating.") # 批处理的大小 32-->128 # TODO
tf.app.flags.DEFINE_integer("decay_steps", 6000, "how many steps before decay learning rate.") # 6000批处理的大小 32-->128
tf.app.flags.DEFINE_float("decay_rate", 1.0, "Rate of decay for learning rate.") # 0.87一次衰减多少
tf.app.flags.DEFINE_integer("sequence_length",800,"max sentence length")
tf.app.flags.DEFINE_integer("embed_size",128,"embedding size")
tf.app.flags.DEFINE_boolean("is_training",True,"is traning.true:tranining,false:testing/inference")
tf.app.flags.DEFINE_integer("validate_every", 100, "Validate every validate_every epochs.") # 每10轮做一次验证
tf.flags.DEFINE_float("l2_lambda",0.0001,"regularization rate random")
tf.flags.DEFINE_integer("grad_clip",5,"grad clip to prevent gradient explode")
tf.flags.DEFINE_integer("checkpoint_every",100,"Save model after this many steps (default 100)")
tf.flags.DEFINE_integer("evaluate_every",100,"evaluate every this many batches")
tf.flags.DEFINE_integer("num_checkpoints",20,"Number of checkpoints to store (default 5)")
tf.flags.DEFINE_integer("epoch",4,"number of epoch")
tf.flags.DEFINE_float("dropout_keep_prob",0.5,"rnn_input_keep_prob")
tf.app.flags.DEFINE_integer("num_sentences", 80, "number of sentences in the document") # 每10轮做一次验证
tf.app.flags.DEFINE_integer("hidden_size",128,"hidden size")

# prepare thr training data
tf.flags.DEFINE_string("train_file0", "lstm_model/processed_data/one_gram/filter-1gram_phrase_level_data_400_0.csv","train file url")
tf.flags.DEFINE_string("train_file1", "lstm_model/processed_data/one_gram/filter-1gram_phrase_level_data_400_1.csv","train file url")
tf.flags.DEFINE_string("train_file2", "lstm_model/processed_data/one_gram/filter-1gram_phrase_level_data_400_2.csv","train file url")
tf.flags.DEFINE_string("train_file3", "lstm_model/processed_data/one_gram/filter-1gram_phrase_level_data_400_3.csv","train file url")
tf.flags.DEFINE_string("train_file4", "lstm_model/processed_data/one_gram/filter-1gram_phrase_level_data_400_4.csv","train file url")

tf.flags.DEFINE_string("vocab_file","lstm_model/processed_data/one_gram/filter-1gram_phrase_level_data_400_vocab.pk","vocab file url")
tf.flags.DEFINE_string("vocab_file_csv","lstm_model/processed_data/one_gram/filter-1gram_phrase_level_data_400_vocab.csv","vocab csv file url")
tf.flags.DEFINE_string("word2vec_file", "embedding_model/models/w2v_phrase_128_3_10_15.bin", "vocab csv file url")

FLAGS = tf.flags.FLAGS
# =====================load data========= ===============================================================================
# load the training data
# 准备数据
print("Loading Data...")
train_x_text2, train_y2= Data_helper.load_data_and_labels(from_project_root(FLAGS.train_file2))
train_x_text1, train_y1 = Data_helper.load_data_and_labels(from_project_root(FLAGS.train_file1))
train_x_text0, train_y0 = Data_helper.load_data_and_labels(from_project_root(FLAGS.train_file0))
train_x_text4, train_y4 = Data_helper.load_data_and_labels(from_project_root(FLAGS.train_file4))
train_x_text = []
train_x_text.extend(train_x_text2)
train_x_text.extend(train_x_text0)
train_x_text.extend(train_x_text1)
train_x_text.extend(train_x_text4)
train_y = []
train_y.extend(train_y2)
train_y.extend(train_y0)
train_y.extend(train_y1)
train_y.extend(train_y4)

vocab = pk.load(open(from_project_root(FLAGS.vocab_file),'rb'))

dev_x_text,dev_y = Data_helper.get_predict_data(from_project_root(FLAGS.train_file3))

# 将x_text进行向量化
train_x_vecs = get_index_text(train_x_text,FLAGS.sequence_length,from_project_root(FLAGS.vocab_file))
dev_x_vecs = get_index_text(dev_x_text,FLAGS.sequence_length,from_project_root(FLAGS.vocab_file))

# 使用预训练的embedding
model = gensim.models.Word2Vec.load(from_project_root(FLAGS.word2vec_file))
init_embedding_mat = []
init_embedding_mat.append([1.0] * FLAGS.embed_size)
with open(from_project_root(FLAGS.vocab_file_csv),'r',encoding='utf-8') as f:
    for line in f.readlines():
        line_list = line.strip().split(',')
        word = line_list[0]
        if word not in model:
            init_embedding_mat.append([1.0] * FLAGS.embed_size)
        else:
            init_embedding_mat.append(model[word])
embedding_mat = tf.Variable(init_embedding_mat,name="embedding")

# x_train,x_dev = x[:dev_sample_index],x[dev_sample_index:]
# y_train,y_dev = y[:dev_sample_index],y[dev_sample_index:]

# 格式化输出
print("Train / embed_sizeDev split: {:d} / {:d}".format(len(train_y),len(dev_y)))

with tf.Session() as sess:
    # 创建对象
    model=HierarchicalAttention(FLAGS.num_classes,
                                FLAGS.learning_rate,
                                FLAGS.decay_steps,
                                FLAGS.decay_rate,
                                FLAGS.sequence_length,
                                FLAGS.num_sentences,
                                FLAGS.embed_size,
                                FLAGS.hidden_size,
                                embedding_mat,
                                FLAGS.is_training)

    with tf.name_scope("loss"):
        # input: `logits`:[batch_size, num_classes], and `labels`:[batch_size]
        # output: A 1-D `Tensor` of length `batch_size` of the same type as `logits`
        #  with the softmax cross entropy loss.
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=model.input_y,
                                                                logits=model.logits)
        # sigmoid_cross_entropy_with_logits.#losses=tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y,logits=self.logits)
        # print("1.sparse_softmax_cross_entropy_with_logits.losses:",losses) # shape=(?,)
        loss = tf.reduce_mean(losses)  # print("2.loss.loss:", loss) #shape=()
        l2_losses = tf.add_n(
            [tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * FLAGS.l2_lambda
        loss = loss + l2_losses

    with tf.name_scope("accuracy"):

        correct_prediction = tf.equal(tf.cast(model.predictions, tf.int32),
                                      model.input_y)  # tf.argmax(self.logits, 1)-->[batch_size]
        acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy")  # shape=()

    # create model path
    timestamp = str(int(time.time()))+"_3"
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    print("Wrinting to {} \n".format(out_dir))

    # global step
    global_step = tf.Variable(0, trainable=False)
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)

    # RNN中常用的梯度截断，防止出现梯度过大难以求导的现象
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), FLAGS.grad_clip)
    grads_and_vars = tuple(zip(grads, tvars))
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    # Keep track of gradient values and sparsity (optional)
    grad_summaries = []
    for g, v in grads_and_vars:
        if g is not None:
            grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
            grad_summaries.append(grad_hist_summary)

    grad_summaries_merged = tf.summary.merge(grad_summaries)

    loss_summary = tf.summary.scalar('loss', loss)
    acc_summary = tf.summary.scalar('acc', acc)

    train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
    train_summary_dir = os.path.join(out_dir, "summaries", "train")
    train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

    dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
    dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
    dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

    sess.run(tf.global_variables_initializer())

    def train_step(x_batch,y_batch):

        feed_dict={
            model.input_x:x_batch,
            model.input_y:y_batch,
            model.batch_size:len(x_batch),
            model.dropout_keep_prob:FLAGS.dropout_keep_prob
        }
        _,step,summaries,cost,accuracy = sess.run([train_op,global_step,train_summary_op,loss,acc],feed_dict)

        time_str = str( int(time.time()))
        print("{} : step {}, loss {:g} , acc {:g}".format(time_str,step,cost,accuracy))

        return step,accuracy

    def dev_step(x_batch,y_batch,writer=None):

        feed_dict={
            model.input_x:x_batch,
            model.input_y:y_batch,
            model.batch_size:len(x_batch),
            model.dropout_keep_prob:1.0
        }

        step, summaries, cost, accuracy = sess.run([global_step, dev_summary_op, loss, acc], feed_dict)

        time_str = str(int(time.time()))
        print("++++++++++++++++++dev++++++++++++++{}: step {}, loss {:g}, acc {:g}".format(time_str, step, cost,
                                                                                           accuracy))

        if writer:
            writer.add_summary(summaries, step)


    def dev_step(dev_x_vecs, dev_y, per_predict_limit):

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

            feed_dict = {
                model.input_x: dev_x_vecs_batch,
                model.batch_size: len(dev_x_vecs_batch),
                model.dropout_keep_prob: 1.0
            }
            predict_result,predict_logit = sess.run([model.predictions,model.logits], feed_dict)
            # change　加入一个loss,
            # step, summaries, cost, accuracy = sess.run([global_step, dev_summary_op, loss, acc], feed_dict)

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


    best_acc = 0
    for epoch in range(FLAGS.epoch):
        print('current epoch %s' % (epoch + 1))

        for i in range(0,len(train_y)-FLAGS.batch_size,FLAGS.batch_size):

            x_batch = train_x_vecs[i:i+FLAGS.batch_size]
            y_batch = train_y[i:i+FLAGS.batch_size]
            step,accuracy = train_step(x_batch,y_batch)

            if step % FLAGS.evaluate_every == 0:
                dev_step(dev_x_vecs,dev_y,per_predict_limit=100)

            if step % FLAGS.checkpoint_every == 0 :
                best_acc = accuracy
                path = saver.save(sess,checkpoint_prefix,global_step=step)
                print("Saved model checkpoint to {} \n".format(path))