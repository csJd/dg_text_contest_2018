#-*-coding:utf-8-*-

import tensorflow as tf
import Data_helper
from utils.data_util import from_project_root
from tensorflow.contrib import learn
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import pickle as pk
import collections


# ===================================================================================
# 参数设置
# 参数设置
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
# 预测文件路径
tf.flags.DEFINE_string("predict_filename","lstm_model/processed_data/filter_phrase_level_data_dev.csv","predict_filename path")

# vocabulary path
tf.flags.DEFINE_string("vocabulary_path","./runs/1533037840/vocab","vocabulary_path")

# model checkpoint path
tf.flags.DEFINE_string("meta_path","./runs/1533037840/checkpoints/model-100.meta","meta_path")
tf.flags.DEFINE_string("model_path","./runs/1533037840/checkpoints/model-100","model_path")

# result output filename
tf.flags.DEFINE_string("result_path","./result/result_predict.csv","result path")
tf.flags.DEFINE_string("vocab_file","lstm_model/processed_data/phrase_level_vocab.pk","vocab file url")
tf.flags.DEFINE_string("dc_file","lstm_model/processed_data/phrase_level_dc.pk","dc file url")
tf.flags.DEFINE_integer("max_word_in_sent",1000,"max_word_in_sent")

FLAGS = tf.flags.FLAGS
# FLAGS._parse_flags()

# ===================================================================================
# 获取预测文本
predict_context,predict_labels = Data_helper.get_predict_data(from_project_root(FLAGS.predict_filename))

# 加载词典
vocab_dict = pk.load(open(from_project_root(FLAGS.vocab_file),'rb'))
x_vecs = []
for x in predict_context:
    x_word_list = x.strip().split()
    x_vec = [0] * FLAGS.max_word_in_sent
    for i in range(min(FLAGS.max_word_in_sent,len(x_word_list))):
        x_vec[i] = vocab_dict[x_word_list[i]]
    x_vecs.append(x_vec)

# 加载term_weight
dc_dict = pk.load(open(from_project_root(FLAGS.dc_file), 'rb'))
term_weights = []
for x in predict_context:
    x_word_list = x.strip().split()
    sen_len = len(x_word_list)
    # 计算文档级别的tf
    tf_dict = collections.defaultdict(int)
    for word in x_word_list:
        tf_dict[word] += 1
    term_weight = [0] * FLAGS.max_word_in_sent
    for i in range(min(FLAGS.max_word_in_sent,len(x_word_list))):
        term_weight[i] = tf_dict[x_word_list[i]] / sen_len * dc_dict[x_word_list[i]]

    term_weights.append(term_weight)

#
print("加载数据完毕。。。")
# predition
print("prediction.......")
# 预测
graph = tf.Graph()

with graph.as_default():

    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)

    sess = tf.Session(config=session_conf)

    with sess.as_default():

        # 加载训练好的模型
        saver = tf.train.import_meta_graph(FLAGS.meta_path)
        saver.restore(sess,FLAGS.model_path)

        # 获取模型输入
        input_x = graph.get_operation_by_name("placeholder/input_x").outputs[0]
        term_weight = graph.get_operation_by_name("placeholder/term_weight")
        rnn_input_keep_prob = graph.get_operation_by_name("placeholder/rnn_input_keep_prob").outputs[0]
        rnn_output_keep_prob = graph.get_operation_by_name("placeholder/rnn_output_keep_prob").outputs[0]

        #
        predictions = graph.get_operation_by_name("fully_connection_layer/prediction").outputs[0]

        #
        per_predict_limit = 200
        sum_predict = len(x_vecs)
        batch_size = int(sum_predict / per_predict_limit)

        batch_prediction_all = []
        # 一个一个进行预测
        for index in tqdm(range(batch_size)):
            start_index = index * per_predict_limit

            if index == batch_size - 1 :
                end_index = sum_predict
            else:
                end_index = start_index + per_predict_limit
            predict_text = x_vecs[start_index:end_index]
            current_term_weight = term_weights[start_index:end_index]
            predict_result = sess.run(predictions,{input_x:predict_text,rnn_input_keep_prob:1.0,
                                                   term_weight:current_term_weight,
                                                   rnn_output_keep_prob:1.0})
            batch_prediction_all.extend(predict_result)

        # 预测结果输出
        # print(batch_prediction_all)
        reset_prediction_all = []
        for predit in batch_prediction_all:
            reset_prediction_all.append(int(predit)+1)

        real_label = np.array(predict_labels).astype(int)
        predict_labels  = reset_prediction_all

        macro_f1 = f1_score(real_label,reset_prediction_all,average='macro')
        accuracy_score1 = accuracy_score(real_label,reset_prediction_all,normalize=True)

        print("macro_f1:{}".format(macro_f1))
        print("accuracy:{}".format(accuracy_score1))

        # ids = np.array(predict_labels).astype(int)
        # predict_labels = reset_prediction_all
        # # 写入文件
        # with open(FLAGS.result_path,'w',encoding='utf-8') as f:
        #     f.write("id,class\n")
        #     for i in range(len(ids)):
        #         f.write("{},{}\n".format(ids[i],predict_labels[i]))
