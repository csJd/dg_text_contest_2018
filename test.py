#coding=utf-8
import collections
import numpy as np
import pickle as pk

# 查询训练集基本信息
# 训练集中 共有102278个训练样本
# 共有19个分类
# 句子的最大长度为：16000以上
# 词典个数：16097

def test1():

    filename = "./processed_data/phrase_level_data.csv"
    line_count = 0
    with open(filename,'r',encoding='utf-8') as f:

        for line in f.readlines():
            line_count+= 1

    f.close()
    print(line_count)
    pass

def test2():

    data = pk.load(open("E:\deve-program\pycharm-workplace\dg_text\processed_data\phrase_level_word_dict.pk",'rb'))

    print(len(data))


if __name__ == "__main__":
    test1()
