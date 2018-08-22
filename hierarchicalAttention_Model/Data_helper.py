#coding=utf-8
import pandas as pd
import numpy as np
#
def load_data_and_labels(train_file):

    # train_file = "../processed_data/splited_train_data.csv"

    x_text = []
    y = []

    with open(train_file,'r',encoding='utf-8') as f:
        for line in f.readlines():
            row = line.strip().split(',')
            x_text.append(row[1])
            train_label = int(row[0])

            #构造label
            # label = [0] * 19
            # label[train_label-1] = 1
            y.append(train_label-1)

    return x_text,y

'''
    获取分好词的文件
'''
def get_predict_data(predict_file):
     # train_file = "../processed_data/splited_train_data.csv"

    x_text = []
    y = []

    with open(predict_file,'r',encoding='utf-8') as f:
        for line in f.readlines():
            row = line.strip().split(',')
            x_text.append(row[1])

            #垃圾邮件 [1,0]  正常邮件 [0,1]
            real_label = int(row[0])
            y.append(real_label)

    return x_text,y

'''
    根绝x_text样本数据，使用tf_dc_value进行向量化，
    并使用pca降维办法
'''
def load_word_bag_sentence_encode(pca_tfbdc_file):

    #
    pca_tfbdc_vectors = []
    with open(pca_tfbdc_file,'r',encoding='utf-8') as f:
        for line in f.readlines():
            line_list = line.strip().split(",")
            pca_tfbdc_vectors.append(np.array(line_list[1].strip().split()).astype(np.float32))
    return np.array(pca_tfbdc_vectors)
    pass

# test
if __name__ == "__main__":
    load_word_bag_sentence_encode("")

