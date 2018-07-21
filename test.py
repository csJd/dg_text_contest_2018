#coding=utf-8
import collections
import numpy as np

# 查询训练集基本信息
# 训练集中 共有102278个训练样本
# 共有19个分类
# 句子的最大长度为：16000以上
# 词典个数：16097

def test1():

    filename = "./pre_process_data/phrase_level_data.csv"
    line_count = 0
    with open(filename,'r',encoding='utf-8') as f:

        for line in f.readlines():
            line_count+= 1

    f.close()
    print(line_count)
    pass

def test2():

    dict = {'1':12,"2":34}

    print(dict.get('1',0))


if __name__ == "__main__":
    test2()
