# -*- coding: utf-8 -*-
import pre_process.data_util as du
import numpy as np
from tqdm import tqdm
import pandas as pd
import datetime
from scipy.sparse import csr_matrix
from collections import Counter
import collections
import pickle as pk
from utils.path_util import from_project_root

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


def sentence_to_vector(sentence_data, word_list, word_df, is_tf=False):
    """
    将数据集转化成稀疏矩阵
    :param sentence_data:
    :param word_list:
    :param word_df:
    :param is_tf:
    :return:
    """
    row = []
    col = []
    value = []
    i = 0
    not_in_dic = []
    is_none = True
    for sentence in tqdm(sentence_data):
        sentence_split = sentence.split(" ")
        sentence_count = Counter(sentence_split)
        for word, count in sentence_count.items():
            try:
                col_value = word_df.loc[word, "index"]
                weight_value = word_df.loc[word, "weight"]
                if is_tf:
                    weight_value = weight_value * (np.log(count) + 1)
                col.append(col_value)
                value.append(weight_value)
                row.append(i)
                is_none = False
            except KeyError:
                not_in_dic.append(word)
        if is_none:
            print(i)
        i = i + 1
    print(len(Counter(not_in_dic)))
    sentence_vector = csr_matrix((value, (row, col)), shape=(len(sentence_data), len(word_list)))
    return sentence_vector


def weight_vector(train_data, test_data, sentence_type, is_tf, weight_type, min_weight=0, min_df=0, max_df=1):
    """
    根据各种特征权重值将语句转化成向量空间模型
    :param train_data:
    :param test_data:
    :param sentence_type:
    :param is_tf:
    :param weight_type:
    :param min_weight:
    :param min_df:
    :param max_df:
    :return:
    """
    sample_num = len(train_data)
    max_df = max_df * sample_num
    word_weight_df = du.read_weight_csv(weight_type, sentence_type)
    word_select_df = word_weight_df[(word_weight_df["weight"] > min_weight) & (word_weight_df["count"] > min_df) & (word_weight_df["count"] < max_df)].copy()
    word_list = np.array(word_select_df.index, dtype=str)
    len_row = word_select_df.shape[0]
    print("word_length: ", len_row)
    word_select_df["index"] = np.arange(len_row)
    train_vector = sentence_to_vector(train_data, word_list, word_select_df, is_tf=is_tf)
    test_vector = sentence_to_vector(test_data, word_list, word_select_df, is_tf=is_tf)
    train_vector = normalize(train_vector, axis=1, norm="l2")
    test_vector = normalize(test_vector, axis=1, norm="l2")
    du.write_vector(train_vector, weight_type, sentence_type, data_type="train")
    du.write_vector(test_vector, weight_type, sentence_type, data_type="test")
    return train_vector, test_vector


def concat(weight_type1, weight_type2, sentence_type):
    """
    将weight_list中的特征权重组合起来
    :param weight_list: list，元素是weight_type
    :param sentence_type:
    :return: None
    """
    word_weight_df1 = du.read_weight_csv(weight_type1, sentence_type)
    word_weight_df2 = du.read_weight_csv(weight_type2, sentence_type)
    word_list = np.array(word_weight_df1.index, dtype=str)
    word_weight = []
    for word in word_list:
        weight_value = word_weight_df1.loc[word, "weight"] * word_weight_df2.loc[word, "weight"]
        word_weight.append(weight_value)
    word_sum = word_weight_df2["count"].values
    new_weight_type = weight_type1 + "_" + weight_type2
    # 将词的权重值保存到pickle中，格式{"word": idf_value}
    du.write_weight_pickle(word_list, word_weight, weight_type=new_weight_type, sentence_type=sentence_type)
    # 将词的权重值保存到csv中，格式为["word", "weight", "count"]
    du.write_weight_csv(word_list, word_weight, word_sum, weight_type=new_weight_type, sentence_type=sentence_type)



def one_hot(param_data, sentence_type):
    """
    计算每个词对应的one_hot值
    :param param_data:
    :param sentence_type:
    :return:
    """
    word_dictionary = []
    data = None
    if sentence_type == "phrase":
        data = param_data["word_seg"].values
    elif sentence_type == "word":
        data = param_data["article"].values
    for sentence in tqdm(data):
        word_list = sentence.split(" ")
        word_list_only = list(set(word_list))
        word_dictionary.extend(word_list_only)
    word_dictionary_count = Counter(word_dictionary)
    word_dictionary_only = list(word_dictionary_count.items())
    word_value = [1] * len(word_dictionary_only)
    word_df = pd.DataFrame(word_dictionary_only, columns=["word", "count"])
    word_df["weight"] = word_value
    filename = "processed_data/csv_weight/" + sentence_type + "_level_one_hot.csv"
    filename = from_project_root(filename)
    word_df.to_csv(filename, index=False)


def idf(word_filename, raw_filename, sentence_type, smooth_idf=0):
    """
    读取词分布文件word_filename，计算词的idf值
    :param word_filename: 词分布文件，每个词在19个类别的文档数
    :param raw_filename: 源文件路径
    :param sentence_type: 句子级别，取"word"或者"phrase"
    :param smooth_idf: 整数，对idf作平滑处理
    :return: None
    """
    word_label_df = du.read_word_df(word_filename)
    word_list = np.array(word_label_df.index, dtype=str)
    word_count = word_label_df["count"].values
    data_df = du.read_data_df(raw_filename, data_type="train")
    sentence_count = data_df.shape[0]
    word_idf = np.log((sentence_count + 1)/ (word_count+smooth_idf)) + 1
    # 将词的idf保存到pickle中，格式{"word": idf_value}
    du.write_weight_pickle(word_list, word_idf, weight_type="idf", sentence_type=sentence_type)
    # 将词的idf保存到csv中，格式为["word", "weight", "count"]
    du.write_weight_csv(word_list, word_idf, word_count, weight_type="idf", sentence_type=sentence_type)


def tf_idf(train_data, test_data, weight_type, sentence_type):
    """

    :param train_data:
    :param test_data:
    :param weight_type:
    :param sentence_type:
    :return:
    """
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,2),max_df=0.9, min_df=3, use_idf=1, smooth_idf=1, sublinear_tf=1)
    # tfidf_vectorizer = TfidfVectorizer(use_idf=1, smooth_idf=1, sublinear_tf=1)
    train_vector = tfidf_vectorizer.fit_transform(train_data)
    test_vector = tfidf_vectorizer.transform(test_data)
    du.write_vector(train_vector, weight_type, sentence_type, data_type="train")
    du.write_vector(test_vector, weight_type, sentence_type, data_type="test")
    return train_vector, test_vector


def dc_bdc(sentence_type, is_bdc):
    """
    计算词的dc值或者bdc值
    :param sentence_type: str，句子级别
    :param is_bdc: bool
    :return:
    """
    if is_bdc:
        filename = "processed_data/word_distribution/" + sentence_type + "_label_bdc_count.csv"
    else:
        filename = "processed_data/word_distribution/" + sentence_type + "_label_count.csv"
    filename = from_project_root(filename)
    word_label_df = du.read_word_df(filename)
    raw_num, col_num = word_label_df.shape
    C = col_num - 1
    word_sum = word_label_df.pop("count").values
    word_label_value_non = word_label_df.values
    word_label_value = word_label_df.fillna(0).values
    word_label_sum = np.sum(word_label_value, axis=1)
    word_label_value = (word_label_value_non.T / word_label_sum).T
    word_label_value_log = np.nan_to_num(np.log2(word_label_value))
    word_label_value = np.nan_to_num(word_label_value)
    word_weight = np.sum(word_label_value * word_label_value_log, axis=1) / np.log2(C) + 1
    word_list = np.array(word_label_df.index, dtype=str)
    if is_bdc:
        weight_type = "bdc"
    else:
        weight_type = "dc"
    # 将词的idf保存到pickle中，格式{"word": idf_value}
    du.write_weight_pickle(word_list, word_weight, weight_type=weight_type, sentence_type=sentence_type)
    # 将词的idf保存到csv中，格式为["word", "weight", "count"]
    du.write_weight_csv(word_list, word_weight, word_sum, weight_type=weight_type, sentence_type=sentence_type)

# 浩鹏的代码
# 计算类别中词的频率 f(w,c)
def cal_wordInCategory_fre(train_file):
    """
    :param trian_file: 训练文件 ： 分为两列 <label,doc_words>
    :return:
    """
    # 统计每个类别中，每个词的词频
    wordInCategory_dict = {} # key:label value: dict{key:word value:count}

    with open(train_file,'r',encoding='utf-8') as f:

        for line in f.readlines():
            line_list = line.strip().split(',')
            label = line_list[0]
            word_list = line_list[1].strip().split()
            #
            if label not in wordInCategory_dict.keys():
                wordInCategory_dict[label] = {}

            for word in word_list:

                if word not in wordInCategory_dict[label].keys():
                    wordInCategory_dict[label][word] = 1
                else:
                    wordInCategory_dict[label][word] += 1
    f.close()
    return wordInCategory_dict

# 计算类别频率 即 f(c)
def calc_category_fre(train_file):
    """
        :param trian_file: 训练文件 ： 分为两列 <label,doc_words>
        :return:
        """
    # 统计每个类别中，每个词的词频
    category_dict = collections.defaultdict(int)  # key:label value: dict{key:word value:count}

    with open(train_file, 'r', encoding='utf-8') as f:

        for line in f.readlines():
            line_list = line.strip().split(',')
            label = line_list[0]

            category_dict[label] += 1
    f.close()

    return category_dict

# 计算bdc权重的主函数
def cal_bdc_value(train_file,word_dict_url,bdc_pickle):
    """
    :param train_file:
    :param word_dict_url:
    :param bdc_pickle:
    :return:
    """

    wordInCategory_dict = cal_wordInCategory_fre(train_file)
    category_dict = calc_category_fre(train_file)
    word_dict = pk.load(open(word_dict_url,'rb'))

    # 分别计算每个词的bdc_value
    bdc_value_dict = collections.defaultdict(float)

    for (word,word_id) in word_dict.items():

        sum_A = 0
        # 计算该词在所有类别中的频繁度
        for (category,frequence) in category_dict.items():

            sum_A += (wordInCategory_dict[category].get(word,0) / float(category_dict[category]))

        # 计算该词在分子的值
        sum_B = 0
        for (category,frequence) in category_dict.items():

            A =(wordInCategory_dict[category].get(word,0) / float(category_dict[category])) / sum_A

            if A == 0:
                continue


            sum_B += (A * np.log(A))
        #
        bdc_value = 1 + (sum_B / np.log(len(wordInCategory_dict)))

        bdc_value_dict[word] = bdc_value
    #
    pk.dump(bdc_value_dict,open(bdc_pickle,'wb'))


# 结合tf_idf_model中的tf_value 和 bdc_value
def get_tf_bdc_value(tf_pickle,bdc_pickle,word_dict_pickle,tf_bdc_pickle):

    tf_dict = pk.load(open(tf_pickle,'rb'))
    bdc_dict = pk.load(open(bdc_pickle,'rb'))
    word_dict = pk.load(open(word_dict_pickle,'rb'))
    tf_bdc_dict = collections.defaultdict(float)

    for (word,wordid) in word_dict.items():

        tf_bdc_dict[word] = tf_dict[word] * bdc_dict[word]

    #save
    pk.dump(tf_bdc_dict,open(tf_bdc_pickle,'wb'))


# 计算词频
def cal_tf(train_file,tf_dict_pickle):
    """
    :param train_file:训练样本
    :param tf_dict_pickle: tf值字典 key: phrase_id value:count
    :return: None
    """
    tf_dict = collections.defaultdict(int)
    with open(train_file,'r',encoding='utf-8') as f:

        for line in f.readlines():
            line_list = line.strip().split(',')

            word_list = line_list[1].strip().split()
            for w in word_list:
                tf_dict[w] += 1

    f.close()

    # save
    pk.dump(tf_dict,open(tf_dict_pickle,'wb'))

# 计算idf_value
def cal_idf(train_file,idf_dict_pickle):

    word_doc_dict = collections.defaultdict(int)
    sum_doc_count = 0
    idf_dict = collections.defaultdict(float)

    # 其实就是统计一个词出现过的文档数
    with open(train_file,'r',encoding='utf-8') as f:

        for line in f.readlines():
            sum_doc_count += 1
            line_list = line.strip().split(',')
            label = line_list[0]
            word_list = list(set(line_list[1].strip().split()))

            for word in word_list:
                word_doc_dict[word] += 1

        # 计算单词的逆文档向量
        for (word,word_count_in_doc) in word_doc_dict.items():

            idf_dict[word] = np.log((sum_doc_count / (word_count_in_doc + 1)))

    # save
    pk.dump(idf_dict,open(idf_dict_pickle,'wb'))

    f.close()

if __name__ == "__main__":

    # 计算tf
    # train_file = from_project_root("lstm_model/processed_data/phrase_level_data.csv")
    # tf_dict_pickle =from_project_root("lstm_model/processed_data/phrase_tf.pk")
    # cal_tf(train_file,tf_dict_pickle)
    # exit()

    # 计算bdc值
    # train_file = from_project_root("lstm_model/processed_data/phrase_level_data.csv")
    # word_dict_url = from_project_root("lstm_model/processed_data/phrase_tf.pk")
    # bdc_pickle = from_project_root("lstm_model/processed_data/phrase_bdc.pk")
    # cal_bdc_value(train_file,word_dict_url,bdc_pickle)
    # exit()

    # 计算dc
    sentence_type = "phrase"
    is_bdc = False
    dc_bdc(sentence_type, is_bdc)
    exit()

    # validation_train_data_filename = "./data/small_train.csv"
    # validation_train_data = du.read_data_df(validation_train_data_filename, data_type="train")
    print("start")
    start_time = datetime.datetime.now()
    idf("./word_distribution/label_word_count.csv", "./data/small_train.csv",
        "./processed_data/phrase_level_tf.pk", sentence_type="phrase", smooth_idf=0)
    # one_hot(validation_train_data, type="word")
    # dc_bdc(type="word", is_bdc=False)
    end_time = datetime.datetime.now()
    print((end_time-start_time).seconds)

