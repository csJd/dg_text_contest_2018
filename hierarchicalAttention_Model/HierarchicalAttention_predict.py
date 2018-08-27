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
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
# 预测文件路径
tf.flags.DEFINE_string("predict_filename",
                       "lstm_model/processed_data/one_gram/filter-1gram_phrase_level_data_400_dev.csv",
                       "predict_filename path")
# vocabulary path
tf.flags.DEFINE_string("vocab_file",
                       "lstm_model/processed_data/one_gram/filter-1gram_phrase_level_data_400_vocab.pk",
                       "vocab file url")
tf.flags.DEFINE_integer("max_word_in_sent",800,"max_word_in_sent")

FLAGS = tf.flags.FLAGS
def main():

    predict_context, predict_labels = Data_helper.get_predict_data(from_project_root(FLAGS.predict_filename))
    x_vecs = get_index_text(predict_context, FLAGS.max_word_in_sent, from_project_root(FLAGS.vocab_file))

    # 将保存的预测结果进行预测
    model_path = ['1534321225_0','1534310253_1','1534299886_2','1534289273_3','1534259360_4']
    for cv_num_i in range(5):
        for step_i in [900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000,2100,2200,2300]:

            print("cv_num:{} step_i:{}".format(cv_num_i,step_i))
            # model checkpoint path
            step = str(step_i)
            cv_num = str(cv_num_i)

            # tf.flags.DEFINE_string("result_path", "lstm_model/result_test/result_predict" + cv_num + "-" + step + ".csv",
            #                        "result path")
            #
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
                    saver = tf.train.import_meta_graph("./runs/"+model_path[cv_num]+"/checkpoints/model-" + step + ".meta")
                    saver.restore(sess, "./runs/"+model_path[cv_num_i]+"/checkpoints/model-" + step)
                    # 获取模型输入
                    input_x = graph.get_operation_by_name("placeholder/input_x").outputs[0]
                    dropout_keep_prob = graph.get_operation_by_name("placeholder/dropout_keep_prob").outputs[0]
                    batch_size_placeholder = graph.get_operation_by_name("placeholder/batch_size").outputs[0]

                    predictions = graph.get_operation_by_name("predictions").outputs[0]
                    prediction_pro = graph.get_operation_by_name("output/softmax_pro").outputs[0]

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

                        feed_dict = {
                            input_x: predict_text,
                            batch_size_placeholder:len(predict_text),
                            dropout_keep_prob:1.0
                        }
                        predict_result, predcit_pros = sess.run([predictions, prediction_pro], feed_dict)
                        batch_prediction_all.extend(predict_result)
                        batch_prediction_pro_all.extend(predcit_pros)

                    # 预测结果输出
                    # print(batch_prediction_all)
                    reset_prediction_all = []
                    for predit in batch_prediction_all:
                        reset_prediction_all.append(int(predit)+1)
                    #
                    real_label = np.array(predict_labels).astype(int)

                    macro_f1 = f1_score(real_label,reset_prediction_all,average='macro')
                    accuracy_score1 = accuracy_score(real_label,reset_prediction_all,normalize=True)

                    print("=======================================")
                    print("macro_f1:{}".format(macro_f1))
                    print("accuracy:{}".format(accuracy_score1))
                    print("=======================================")

                    # 写入文件
                    pk.dump(batch_prediction_pro_all, open(from_project_root( "hierarchicalAttention_Model/result/result_predict" + cv_num + "-" + step + ".pk"), 'wb'))

if __name__ == '__main__':
    main()