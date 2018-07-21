#coding=utf-8
import collections
import pickle as pk
import numpy as np

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
def main():
    train_file = "../pre_process_data/phrase_level_data.csv"
    tf_dict_pickle = "../pre_process_data/phrase_level_tf.pk"
    # cal_tf(train_file,tf_dict_pickle)
    idf_dict_pickle = "../pre_process_data/phrase_level_idf.pk"
    cal_idf(train_file,idf_dict_pickle)
    pass


if __name__ =="__main__":

    main()



