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
from lstm_model.model_tool import get_term_weight,get_index_text


# ===================================================================================
# 参数设置
# 参数设置
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
# 预测文件路径
tf.flags.DEFINE_string("predict_filename","lstm_model/processed_data/one_gram/filter-1gram_phrase_level_test_data_200.csv","predict_filename path")

# vocabulary path
tf.flags.DEFINE_string("vocab_file","lstm_model/processed_data/one_gram/filter-1gram_phrase_level_data_200_vocab.pk","vocab file url")
tf.flags.DEFINE_integer("max_word_in_sent",800,"max_word_in_sent")

def main():

    # 将保存的预测结果进行预测
    model_path = ['1534321225_0','1534310253_1','1534299886_2','1534289273_3','1534259360_4']
    for cv_num_i in range(5):

        for step_i in [900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000,2100,2200,2300]:

            print("cv_num:{} step_i:{}".format(cv_num_i,step_i))
            # model checkpoint path
            step = str(step_i)
            cv_num = str(cv_num_i)
            tf.flags.DEFINE_string("meta_path", "./runs/"+model_path[cv_num_i]+"/checkpoints/model-" + step + ".meta", "meta_path")
            tf.flags.DEFINE_string("model_path", "./runs/"+model_path[cv_num_i]+"/checkpoints/model-" + step, "model_path")
            tf.flags.DEFINE_string("dc_file", "lstm_model/processed_data/one_gram/phrase_level_1gram_dc.json",
                                   "dc file url")

            tf.flags.DEFINE_string("result_path", "lstm_model/result_test/result_predict" + cv_num + "-" + step + ".csv",
                                   "result path")

            FLAGS = tf.flags.FLAGS
            # FLAGS._parse_flags()

            # ===================================================================================
            # 获取预测文本
            predict_context, predict_labels = Data_helper.get_predict_data(from_project_root(FLAGS.predict_filename))
            x_vecs = get_index_text(predict_context, FLAGS.max_word_in_sent, from_project_root(FLAGS.vocab_file))
            term_wegits_vec = get_term_weight(predict_context, FLAGS.max_word_in_sent, from_project_root(FLAGS.dc_file))

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
                    saver.restore(sess, FLAGS.model_path)

                    # 获取模型输入
                    input_x = graph.get_operation_by_name("placeholder/input_x").outputs[0]
                    term_weight = graph.get_operation_by_name("placeholder/term_weight").outputs[0]
                    rnn_input_keep_prob = graph.get_operation_by_name("placeholder/rnn_input_keep_prob").outputs[0]
                    rnn_output_keep_prob = graph.get_operation_by_name("placeholder/rnn_output_keep_prob").outputs[0]
                    #
                    predictions = graph.get_operation_by_name("fully_connection_layer/prediction").outputs[0]
                    prediction_pro = graph.get_operation_by_name("fully_connection_layer/softmax_prob").outputs[0]

                    per_predict_limit = 100
                    sum_predict = len(x_vecs)
                    batch_size = int(sum_predict / per_predict_limit)

                    batch_prediction_all = []
                    batch_prediction_pro_all = []
                    # 一个一个进行预测
                    for index in tqdm(range(batch_size)):
                        start_index = index * per_predict_limit

                        if index == batch_size - 1:
                            end_index = sum_predict
                        else:
                            end_index = start_index + per_predict_limit
                        predict_text = x_vecs[start_index:end_index]
                        term_weights_batch = term_wegits_vec[start_index:end_index]
                        predict_result, predcit_pros = sess.run([predictions, prediction_pro], {input_x: predict_text,
                                                                                                term_weight: term_weights_batch,
                                                                                                rnn_input_keep_prob: 1.0,
                                                                                                rnn_output_keep_prob: 1.0})
                        batch_prediction_all.extend(predict_result)
                        batch_prediction_pro_all.extend(predcit_pros)

                    # 预测结果输出
                    # print(batch_prediction_all)
                    # reset_prediction_all = []
                    # for predit in batch_prediction_all:
                    #     reset_prediction_all.append(int(predit)+1)
                    #
                    # real_label = np.array(predict_labels).astype(int)
                    # predict_labels  = reset_prediction_all
                    #
                    # macro_f1 = f1_score(real_label,reset_prediction_all,average='macro')
                    # accuracy_score1 = accuracy_score(real_label,reset_prediction_all,normalize=True)
                    #
                    # print("macro_f1:{}".format(macro_f1))
                    # print("accuracy:{}".format(accuracy_score1))

                    # 写入文件
                    pk.dump(batch_prediction_pro_all, open(from_project_root(FLAGS.result_path), 'wb'))

if __name__ == '__main__':
    main()