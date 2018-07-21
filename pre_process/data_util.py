#coding=utf-8

import pandas as pd
import numpy as np
import pickle as pk

"""
    utils:
        get_set_len(filename) : 获取数据集中训练样本个数    
        extract_data(filename,extract_index,extracted_file)：根据列坐标提取对应的列，并成成文件
        split_data(filename,train_percent,train_filename,test_filename):将训练集打乱并根据比例划分数据集 
        load_word_dict(train_file,word_dict_pickle) : 根据训练文件建立词典,train_file格式 <label,word_doc>
        tf_bdc_filter(..) : 根据词频tf和bdc权重过滤大量对分类无用的词（噪音）
"""

# 根据词频 和 bdc权重值 过滤不重要的词
def tf_bdc_filter(tf_pickle,bdc_pickle,current_word_dict_pickle,new_word_dict_pickle):
    """
    :param tf_pickle:
    :param bdc_pickle:
    :param current_word_dict_pickle:
    :param new_word_dict_pickle:
    :return:
    """
    tf_value_dict = pk.load(open(tf_pickle,'rb'))
    bdc_value_dict = pk.load(open(bdc_pickle,'rb'))
    current_word_dict = pk.load(open(current_word_dict_pickle,'rb'))

    temp_count = 0
    for (word,word_id) in current_word_dict.items():

        if tf_value_dict[word] == 1 and bdc_value_dict[word] == 1:
            temp_count+=1

    print(temp_count)

# 建立词典 dict(word:id)
def load_word_dict(train_file,word_dict_pickle):
    """
    :param train_file: 训练文件 ,分为两列， 格式：<label, word_doc>
    :param word_dict_pickle:  保存pickle dict{word:id}
    :return: None
    """
    word_dict = {}
    id_count = 0
    with open(train_file,'r',encoding='utf-8') as f:
        for line in f.readlines():
            line_list = line.strip().split(',')
            word_list = line_list[1].strip().split()

            for word in word_list:

                if word not in word_dict.keys():
                    word_dict[word] = id_count
                    id_count += 1

    f.close()

    pk.dump(word_dict,open(word_dict_pickle,'wb'))

# 获取数据集训练样本个数
def get_set_len(filename):
    # 查询数据集数据条数
    count_line = 0
    with open(filename, 'r', encoding='utf-8') as f:

        line = f.readline()
        while line:

            count_line += 1
            # 排除第一行
            if count_line == 1:
                line = f.readline()
                continue
            line = f.readline()
    f.close()
    return count_line

# 将原文件中 label , "词"级别文档 ，两列提取出来
def extract_data(filename,extract_index,extracted_file):
    """
    :param filename: 原来训练文件
    :param extract_index: 提取的列数数组
    :param extracted_file: 提取好的文件
    :return: None
    """
    with open(extracted_file, 'w', encoding='utf-8') as wf:
        with open(filename,'r',encoding='utf-8') as rf :

            for line in rf.readlines()[1:]:
                line_list = line.strip().split(',')

                # 写入的一行字符串
                str_line = ""
                for i in range(len(extract_index)):
                    str_line += line_list[extract_index[i]]
                    if i < len(extract_index)-1:
                        str_line += ","
                # 写入文件
                wf.write(str_line+"\n")
    rf.close()
    wf.close()

# 将训练集划分为 7 ：3 ， 7份训练 ， 3份测试
def split_data(filename,train_percent,train_filename,test_filename):

    df = pd.read_csv(filename,sep=',')
    data_list = np.array(df)
    del(df)
    sample_len = len(data_list)

    train_len = int(train_percent * sample_len)

    # 打乱样本集
    shuffled_list = np.arange(sample_len)
    np.random.shuffle(shuffled_list)
    shuffled_data_list = data_list[shuffled_list]

    # 划分
    train_list = shuffled_data_list[:train_len]
    test_list = shuffled_data_list[train_len:]

    # 使用pickle保存
    train_filename = open(train_filename, 'wb')
    pk.dump(train_list,train_filename)

    test_filename = open(test_filename,'wb')
    pk.dump(test_list,test_filename)

def main():
    init_train_file = "../data/train_set.csv"
    train_filename = "../processed_data/train_data.pk"
    test_filename = "../processed_data/test_data.pk"
    # split_data(filename,0.7,train_filename,test_filename)

    extracted_file = "../processed_data/phrase_level_data.csv"
    # extract_data(init_train_file,[3,2],extracted_file)

    # load_word_dict(extracted_file,"../pre_process_data/phrase_level_word_dict.pk")
    tf_pickle = "E:\deve-program\pycharm-workplace\daguanbei\processed_data\phrase_level_tf.pk"
    bdc_pickle = "E:\deve-program\pycharm-workplace\daguanbei\processed_data\phrase_level_bdcValue.pk"
    current_word_dict_pickle = "E:\deve-program\pycharm-workplace\daguanbei\processed_data\phrase_level_word_dict.pk"
    new_word_dict_pickle = "../processed_data/phrase_level_new_word_dict.pk"
    tf_bdc_filter(tf_pickle, bdc_pickle, current_word_dict_pickle, new_word_dict_pickle)
    pass


if __name__ == "__main__":

    main()
