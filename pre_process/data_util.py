# -*- coding: utf-8 -*
import pandas as pd
import csv
import numpy as np
from tqdm import tqdm
from collections import Counter
import pickle as pk
from utils.path_util import from_project_root
import collections
from smart_open import smart_open

csv.field_size_limit(500*1024*1024)


def read_data_csv(filename):
    """
    以csv库读取csv文件
    :param filename:
    :return:
    """
    dataList = []
    with open(filename, 'r', encoding='utf-8') as fp:
        data = csv.reader(fp)
        for line in data:
            dataList.append(line)
    data_df = pd.DataFrame(dataList[1:], columns=["id", "article", "word_seg", "classify"])
    return data_df


def smart_read_data_df(filename, data_type):
    """
    读取大数据量的csv文件
    :param filename:
    :param data_type:
    :return:
    """
    if data_type == "train":
        data_df = pd.read_csv(smart_open(filename, "r", encoding="utf-8"),
                          dtype={"id": str, "article": str, "word_seg": str, "classify": np.int})
    elif data_type == "test":
        data_df = pd.read_csv(smart_open(filename, "r", encoding="utf-8"),
                              dtype={"id": str, "article": str, "word_seg": str})
    return data_df

def read_data_df(filename, data_type):
    """
    分块读取文件
    :param filename:
    :param data_type:
    :return:
    """
    filename = from_project_root(filename)
    if data_type == "train":
        data_df = pd.read_csv(filename, chunksize=10000,
                              dtype={"id": str, "article": str, "word_seg": str, "class": np.int}, engine="c")
    elif data_type == "test":
        data_df = pd.read_csv(filename, chunksize=10000,
                              dtype={"id": str, "article": str, "word_seg": str}, engine="c")
    tr_list = []
    for tr in data_df:
        tr_list.append(tr)
    data_df = pd.concat(tr_list)
    return data_df


def extract_data(filename, sentence_type, output_filename):
    '''
    根据源文件中的sentence_type提取label，shentence_type两列
    :param filename: 源文件路径名
    :param sentence_type: 提取的句子级别,取值"article"或者"word_seg"
    :param output_filename: 输出文件路径名
    :return: None
    '''
    raw_data_df = read_data_df(filename, data_type="train")
    sentence_data_df = raw_data_df[[sentence_type, "class"]].copy()
    sentence_data_df.to_csv(output_filename, index=False)


def load_word_dict(train_file, sentence_type, word_dict_pickle):
    '''
    根据训练集建立词典
    :param train_file:训练文件，后缀名是csv，内容格式为<sentence, label>
    :param sentence_type:句子级别，取值"article"或者"word_seg"
    :param word_dict_pickle: 保存pickle dict{word：id}
    :return: None
    '''
    data_df = pd.read_csv(train_file)
    sentences = data_df[sentence_type].values
    word_list = []
    for sentence in tqdm(sentences):
        temp_word_list = sentence.split()
        word_list.extend(temp_word_list)
    word_dict = dict(Counter(word_list))
    pk.dump(word_dict, open(word_dict_pickle, "wb"))


def read_word_df(filename):
    """
    读取特殊的DataFrame
    :param filename:
    :return:
    """
    filename = from_project_root(filename)
    word_label_df = pd.read_csv(filename, dtype={"word": str}, engine="c")
    word_label_df.set_index("word", inplace=True)
    return word_label_df


def write_data_df(filename, param_data_df):
    filename = from_project_root(filename)
    param_data_df.to_csv(filename, index=False)


def write_weight_pickle(word_list, word_weight, weight_type, sentence_type):
    """
    将每个词的特征权重写入pickle文件，写入格式为{“word”： word_weight}
    :param word_list: list或者array，元素是词
    :param word_weight: list或者array，元素词对应的权重值
    :param weight_type: str，权重形式
    :param sentence_type: str，句子级别
    :return: None
    """
    pickle_filename = "processed_data/pickle_weight/" + sentence_type + "_level_" + weight_type + ".pk"
    pickle_filename = from_project_root(pickle_filename)
    word_weight_dict = dict(zip(word_list, word_weight))
    pk.dump(word_weight_dict, open(pickle_filename, "wb"))


def write_weight_csv(word_list, word_weight, word_count, weight_type, sentence_type):
    """
    将每个词的特征权重写入csv文件，列索引为["word", "weight", "count"]
    :param word_list: list或者array，元素是词
    :param word_weight: list或者array，元素词对应的权重值
    :param word_count: list或者array，元素是词对应的文档数,count
    :param weight_type: str，权重形式
    :param sentence_type: str，句子级别
    :return: None
    """
    weight_df = pd.DataFrame()
    weight_df["word"] = word_list
    weight_df["weight"] = word_weight
    weight_df["count"] = word_count
    csv_filename = "processed_data/csv_weight/" + sentence_type + "_level_" + weight_type + ".csv"
    csv_filename = from_project_root(csv_filename)
    weight_df.to_csv(csv_filename, index=False)

def read_weight_csv(weight_type, sentence_type):
    """
    读取csv文件的词和特征权重
    :param weight_type: str，特征权重形式
    :param sentence_type: str，句子级别
    :return: DataFrame
    """
    csv_filename = "processed_data/csv_weight/" + sentence_type + "_level_" + weight_type + ".csv"
    csv_filename = from_project_root(csv_filename)
    word_weight_df = pd.read_csv(csv_filename, dtype={"word": str}, engine="c")
    word_weight_df.set_index("word", inplace=True)
    return word_weight_df

from scipy.io import mmwrite
from scipy.io import mmread


def write_vector(sentence_vector, weight_type, sentence_type, data_type):
    """
    将稀疏矩阵保存到本地文件
    :param sentence_vector: csr，稀疏矩阵
    :param weight_type: str，特征权重形式
    :param sentence_type: str，句子级别
    :param data_type: str，数据集形式，取值"test"或者"train"
    :return: None
    """
    vector_filename = "processed_data/vector/" + sentence_type + "_" + weight_type + "_" + data_type + ".mtx"
    vector_filename = from_project_root(vector_filename)
    mmwrite(vector_filename, sentence_vector)


def read_csr(filename):
    """
    读取稀疏矩阵
    :param filename: 稀疏矩阵存储路径
    :return: csr，稀疏矩阵
    """
    csr = mmread(filename)
    return csr


def save_array(filename, param_arr):
    """
    数据的写入与读取
    :param filename:str，本地文件路径
    :param param_arr: array
    :return:
    """
    filename = from_project_root(filename)
    np.save(filename, param_arr)

def load_array(filename):
    label_array = np.load(filename)
    return label_array


def word_in_label(data_df, sentence_type, is_bdc=False):
    """
    计算词在每个类别中出现的文档数
    :param data_df: DataFrame
    :param sentence_type: str, 句子级别
    :param is_bdc: bool，布尔值,默认值是False
    :return: None
    """
    word_dict = collections.defaultdict(dict)
    column = "error"
    if sentence_type == "word":
        column = "article"
    elif sentence_type == "phrase":
        column = "word_seg"
    data_groupby_label = data_df.groupby(["class"])
    label_sum = []
    for name, group in tqdm(data_groupby_label):
        sentence_list = group[column].values
        for sentence in sentence_list:
            word_split = sentence.split(" ")
            word_set = list(set(word_split))
            for word in word_set:
                try:
                    word_dict[word][name] += 1
                except KeyError:
                    word_dict[word][name] = 1
        label_sum.append(group.shape[0])
    df = pd.DataFrame(word_dict).T
    word_sum = df.apply(lambda x: x.sum(), axis=1)
    if is_bdc:
        df = df.apply(lambda x: x/label_sum, axis=1)
        filename = "processed_data/word_distribution/"+ sentence_type + "_label_bdc_count.csv"
    else:
        filename = "processed_data/word_distribution/" + sentence_type + "_label_count.csv"
    filename = from_project_root(filename)
    df["count"] = word_sum
    df = df.reset_index()
    df.rename(columns={"index": "word"}, inplace=True)
    df.to_csv(filename, index=False)


def load_vector_label(weight_type, sentence_type):
    """
    词向量和标签的文件载入
    :param weight_type: 特征权重的形式
    :param sentence_type: str，句子级别
    :return:
    """
    train_label_filename = from_project_root("processed_data/label/train_label.npy")
    test_label_filename = from_project_root("processed_data/label/test_label.npy")
    train_label = load_array(train_label_filename)
    test_label = load_array(test_label_filename)
    train_vector_filename = from_project_root("processed_data/vector/" + sentence_type + "_" + weight_type + "_" + "train.mtx")
    test_vector_filename = from_project_root("processed_data/vector/" + sentence_type + "_" + weight_type + "_" + "test.mtx")
    train_vector = read_csr(train_vector_filename)
    test_vector = read_csr(test_vector_filename)
    return train_vector, train_label, test_vector, test_label

from sklearn.model_selection import train_test_split


def split_data(param_data_df):
    train_df, validation_df = train_test_split(param_data_df, test_size=0.2)
    write_data_df("./data/small_train.csv", train_df)
    write_data_df("./data/small_test.csv", validation_df)


def get_data_label(param_data_df, sentence_type, data_type):
    """
    根据sentence_type返回相应的数据集，如果是训练集，还要label
    :param param_data_df: DataFrame
    :param sentence_type: str，句子级别
    :param data_type: str,取“train”或者“test”
    :return: 如果是“trian”，则返回数据集和label；如果是“test”，则返回数据集
    """
    if sentence_type == "word":
        column = "article"
    elif sentence_type == "phrase":
        column = "word_seg"
    data_list = param_data_df[column].values
    if data_type == "train":
        data_label = param_data_df["class"].values
        return data_list, data_label
    elif data_type == "test":
        return data_list

#计算预测值中各个类别的数目
def cal_compare_label(filename, predict_y, original_y):
    label_dict = {}
    length = original_y.shape[0]
    for i in range(length):
        if original_y[i] not in label_dict:
            label_dict[original_y[i]] = [0, 0, 0]
        if original_y[i] == predict_y[i]:
            label_dict[original_y[i]][0] += 1
        elif original_y[i] != predict_y[i]:
            if predict_y[i] not in label_dict:
                label_dict[predict_y[i]] = [0, 0, 0]
            label_dict[predict_y[i]][1] += 1
        label_dict[original_y[i]][2] += 1
    label_df = pd.DataFrame(label_dict, index=["right", "error", "original"])
    write_data_df(filename, label_df)

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

# 统计一个词出现过多少个类别中
def calc_labelCount_per_words(train_file,labelFrequency_pickle):
    """
    Args:
        train_file: 训练文件， 如： ../data/train_set.csv ，分为两列 <label,doc_words>
        labelFrequency_pickle: 单词的类别频率picdle路径
    Returns:
        None
    """
    lf_label_dict = {}
    lf_dict = collections.defaultdict(int)
    with open(train_file,'r',encoding='utf-8') as f:
        for line in f.readlines():
            line_list = line.strip().split(',')
            label = line_list[0]
            word_list = line_list[1].strip().split()

            for word in word_list:

                if word not in lf_label_dict.keys():
                    lf_label_dict[word] = set()

                lf_label_dict[word].add(label)

        for (word,data_set) in lf_label_dict.items():

            lf_dict[word] = len(data_set)

    f.close()

    # save
    pk.dump(lf_dict,open(labelFrequency_pickle,'wb'))

# 计算每个词的文档频率
def cal_document_frequency(train_file,df_pickle):
    """
    Args:
        train_file:
        df_pickle:

    Returns:
    """
    df_dict = collections.defaultdict(int)
    with open(train_file,'r',encoding='utf-8') as f:

        for line in f.readlines():
            line_list = line.strip().split(',')
            word_list = list(set(line_list[1].strip().split()))
            for word in word_list:
                df_dict[word] += 1
    f.close()

    #save
    pk.dump(df_dict,open(df_pickle,'wb'))

if __name__ == "__main__":
    train_data_filename = "./data/small_train.csv"
    # extract_data(train_data_filename, sentence_type="word_seg", output_filename="./data/word_seg.csv")
    train_data_df = read_data_df(train_data_filename, data_type="train")

    #load_word_dict("./data/word_seg.csv", "word_seg", "./processed_data/phrase_level_word_dict.pk")
    word_in_label(train_data_df, sentence_type="phrase", is_bdc=False)
    #分割数据集
    #split_data(train_data_df)