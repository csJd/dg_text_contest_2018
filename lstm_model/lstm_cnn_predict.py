#-*-coding:utf-8-*-

import tensorflow as tf
import Data_helper
from utils.data_util import from_project_root
from tensorflow.contrib import learn
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score



# ===================================================================================
# 参数设置
# 参数设置
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
# 预测文件路径
tf.flags.DEFINE_string("predict_filename","lstm_model/processed_data/filtered_word_seg_dev.csv","predict_filename path")

# vocabulary path
tf.flags.DEFINE_string("vocabulary_path","./runs/1532577530/vocab","vocabulary_path")

# model checkpoint path
tf.flags.DEFINE_string("meta_path","./runs/1532577530/checkpoints/model-2000.meta","meta_path")
tf.flags.DEFINE_string("model_path","./runs/1532577530/checkpoints/model-2000","model_path")

# result output filename
tf.flags.DEFINE_string("result_path","./result/result_predict.txt","result path")

FLAGS = tf.flags.FLAGS
# FLAGS._parse_flags()

# ===================================================================================
# 获取预测文本
predict_labels,predict_context = Data_helper.get_predict_data(from_project_root(FLAGS.predict_filename))

# 加载词典
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(FLAGS.vocabulary_path)
x_text = np.array(list(vocab_processor.transform(predict_context)))

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
        rnn_input_keep_prob = graph.get_operation_by_name("placeholder/rnn_input_keep_prob").outputs[0]
        rnn_output_keep_prob = graph.get_operation_by_name("placeholder/rnn_output_keep_prob").outputs[0]

        #
        predictions = graph.get_operation_by_name("classification/prediction").outputs[0]

        #
        per_predict_limit = 400
        sum_predict = len(x_text)
        batch_size = int(sum_predict / per_predict_limit)

        batch_prediction_all = []
        # 一个一个进行预测
        for index in range(batch_size):
            start_index = index * per_predict_limit

            if index == batch_size - 1 :
                end_index = sum_predict
            else:
                end_index = start_index + per_predict_limit
            predict_text = x_text[start_index:end_index]
            predict_result = sess.run(predictions,{input_x:predict_text,rnn_input_keep_prob:1.0,
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