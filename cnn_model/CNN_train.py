#coding=utf-8

import tensorflow as tf
import Data_helper
import numpy as np
from tensorflow.contrib import learn
from CNN_Model import CNN_Model
import time
import os
import datetime

#Setting parameters

#==================================================================================================

#dev percent 验证集的大小
tf.flags.DEFINE_float("dev_sample_percentage",0.1,"Percentage of the training data to use for validation")

#Model HypereParameters
tf.flags.DEFINE_integer("embedding_dim",64,"Dimensionality of character embedding")
tf.flags.DEFINE_string("filter_size","3,4,5","the size of the filter")
tf.flags.DEFINE_integer("num_filters",32,"the num of channels in per filter")
tf.flags.DEFINE_float("dropout_keep_prob",0.5,"Dropout keep probability for regularization")
tf.flags.DEFINE_float("l2_reg_lambda",0.0,"l2 regularization lambda fro regularization")

#Training Parameters
tf.flags.DEFINE_integer("batch_size", 50, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_float("learning_rate",1e-3,"Learning rate for the optimizer")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

#Training file path # 训练数据，已经分好词
tf.flags.DEFINE_string("train_file","../init_data/train_set.csv","Training file")

FLAGS = tf.flags.FLAGS
# FLAGS._parse_flags()

#=======================================================================================================================
#参数打印输出
print("\n Parameters:")

for attr,value in sorted(FLAGS.__flags.items()):

    print("{}={}".format(attr.upper(),value))

print("")

#=======================================================================================================================

#准备数据
#x_text : 分好词的字符串数组 , example: ["a b c","d e f g h"]
#y : label example: [[0,1],[1,0],...]

print("Loading Data...")
x_text,y = Data_helper.load_data_and_labels(FLAGS.train_file)

y = np.array(y)

#=======================================================================================================================

#Build Vacabulary  由于卷积神经网络需要固定句子的长度
max_document_length = max(len(x.split(" ")) for x in x_text)
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length) #创建一个字典处理器,并设置句子固定长度
x = np.array( list( vocab_processor.fit_transform(x_text)))   #x就转化为字典的下表表示的数组

#格式化输出
print("Vocabulary size :{:d}".format(len(vocab_processor.vocabulary_)))

#=======================================================================================================================

#由于读进来的数据的label可能是有规律的，因此，使用洗牌
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y))) #使用permutation对有序列进行重新随机排列

x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

#=======================================================================================================================
#数据集划分

dev_sample_index = -1 * int( FLAGS.dev_sample_percentage * len(y))

x_train,x_dev = x_shuffled[:dev_sample_index],x_shuffled[dev_sample_index:]
y_train,y_dev = y_shuffled[:dev_sample_index],y_shuffled[dev_sample_index:]

#情况内存
del x,y,x_shuffled,y_shuffled

#格式化输出
print("Train / Dev split: {:d} / {:d}".format(len(y_train),len(y_dev)))

#=======================================================================================================================
#模型训练

with tf.Graph().as_default():

    #配置session,处理器使用的配置
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement
    )
    sess = tf.Session(config=session_conf)

    #
    with sess.as_default():

        cnn = CNN_Model(
            sequence_length= x_train.shape[1],
            num_classes = y_train.shape[1],
            vocab_size = len(vocab_processor.vocabulary_),
            embedding_size=FLAGS.embedding_dim,
            filter_sizes = list(map(int, FLAGS.filter_size.split(","))),
            num_filters = FLAGS.num_filters,
            l2_reg_lambda=FLAGS.l2_reg_lambda)

        #Define Training procedure
        global_step = tf.Variable(0,name="global_step",trainable=False)
        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
        grads_and_vars = optimizer.compute_gradients(cnn.loss) #计算梯度 加入global_steps，记录步数
        # 以上两步相当于 trian_op = optimizer.minimize(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars,global_step=global_step)

        #Output diretory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir,"runs",timestamp))
        print("Writing to {} \n".format(out_dir))


        #Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss",cnn.loss)
        acc_summary = tf.summary.scalar("accuracy",cnn.accuracy)

        #Train Summary
        train_summary_op = tf.summary.merge([loss_summary,acc_summary])
        train_summary_dir = os.path.join(out_dir,"summaries","train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir,sess.graph) # train_summary_writer.add_summary(summaries,step)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        #模型保存
        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        saver = tf.train.Saver(tf.global_variables(),max_to_keep=FLAGS.num_checkpoints)

        #Write vocabulary
        vocab_processor.save(os.path.join(out_dir,"vocab"))

        #Initialize all variables
        sess.run(tf.global_variables_initializer())

        #
        def train_step(x_batch,y_batch):

            feed_dic = {
                cnn.input_x:x_batch,
                cnn.input_y:y_batch,
                cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _,step,summaries,loss,accuracy = sess.run(
                [train_op,global_step,train_summary_op,cnn.loss,cnn.accuracy],feed_dic
            )

            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)


        def dev_step(x_batch,y_batch,writer=None):
            #Evaluate the model by dev samples
            feed_dic = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.dropout_keep_prob: 1.0
            }

            step,summaries,loss,accuracy = sess.run(
                [global_step,dev_summary_op,cnn.loss,cnn.accuracy],feed_dic)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)

        # Generate batches
        batches = Data_helper.batch_iter(
            list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)


        #Training loop. For each batch
        for batch in batches:
            x_batch, y_batch = zip(*batch)  #这里有个星星的哟~
            train_step(x_batch, y_batch)

            #获取当前训练的步数
            current_step = tf.train.global_step(sess, global_step)

            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                dev_step(x_dev, y_dev, writer=dev_summary_writer)
                print("")

            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))

