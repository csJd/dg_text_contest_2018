#coding=utf-8
import tensorflow as tf
import Data_helper
import numpy as np
from tensorflow.contrib import learn
import pandas as pd

#===================================================================================
#参数设置
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

#预测文件路径
tf.flags.DEFINE_string("predict_filename","../processed_data/dev_processed_data_split.csv","predict_filename path")

#vocabulary path
tf.flags.DEFINE_string("vocabulary_path","./runs/1522228533/vocab","vocabulary_path")

#model checkpoint path
tf.flags.DEFINE_string("meta_path","./runs/1522228533/checkpoints/model-1900.meta","meta_path")
tf.flags.DEFINE_string("model_path","./runs/1522228533/checkpoints/model-1900","model_path")

#result output filename
tf.flags.DEFINE_string("result_path","./result/result_predict.txt","result path")

FLAGS = tf.flags.FLAGS
#FLAGS._parse_flags()

#===================================================================================
#获取预测文本数组
predict_context = Data_helper.get_predict_data(FLAGS.predict_filename)

#===================================================================================
#将文本映射到字典下标数组
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(FLAGS.vocabulary_path)
x_text = np.array( list(vocab_processor.transform(predict_context)))

#===================================================================================
#预测 Predict
print("Prediction.....")
graph = tf.Graph()

with graph.as_default():

    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)

    sess = tf.Session(config = session_conf)

    with sess.as_default():

        #加载训练好的模型
        saver = tf.train.import_meta_graph(FLAGS.meta_path)
        saver.restore(sess,FLAGS.model_path)

        #获取模型的输入
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        #Prediction
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        batch_prediction_all = []
        #一个一个进行预测
        for one_text in x_text:

            one_text = [one_text]

            prediction_one_text = sess.run(predictions,{input_x:one_text,dropout_keep_prob:1.0})

            batch_prediction_all.extend(prediction_one_text)


        #预测结果输出
        # print(batch_prediction_all)
        reset_prediction_all = []

        #将结果写入文件
        with open(FLAGS.result_path,'w') as f :

            for i in range(len(predict_context)):

                if batch_prediction_all[i] == 1:
                    label = 0
                    reset_prediction_all.append(label)
                else:
                    label = 1
                    reset_prediction_all.append(label)
                f.write(str(label) + "," + str(predict_context[i]) + "\n")

        print(reset_prediction_all)
