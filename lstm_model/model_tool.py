# coding: utf-8
# created by Pengpeng on 2018/8/10
import pickle as pk
import utils.json_util as ju
import collections
import numpy as np

# 根据自己的字典将文本下标化
def get_index_text(x_text_arr,max_doc_len,vocab_file):

    vocab_dict = pk.load(open(vocab_file,'rb'))
    x_vecs = []
    for x in x_text_arr:
        word_list = x.strip().split()
        x_vec = [0] * max_doc_len
        for i in range(min(max_doc_len,len(word_list))):
            x_vec[i] = vocab_dict[word_list[i]]
        x_vecs.append(x_vec)
    return x_vecs

# 根据权重值，对每个单词赋予权重
def get_term_weight(x_text_arr, max_doc_len, term_weight_file):

    dc_dict = ju.load(term_weight_file)
    term_weights = []
    for x in x_text_arr:
        x_word_list = x.strip().split()
        sen_length = len(x_word_list)
        # 计算文档级别的tf
        tf_dict = collections.defaultdict(int)
        for word in x_word_list:
            tf_dict[word] += 1
        term_weight = [0] * max_doc_len
        for i in range(min(max_doc_len, len(x_word_list))):
            term_weight[i] = tf_dict[x_word_list[i]] / sen_length * dc_dict[x_word_list[i]]
        # 进行归一化
        term_weight = np.array(term_weight)
        max_value = term_weight.max()
        min_value = term_weight.min()
        mid_value = max_value - min_value
        if mid_value == 0:  # 加一操作，防止遇到0的现象
            term_weight = [1 for value in term_weight]
        else:
            term_weight = [((value - min_value) / mid_value)*2 + 1 for value in term_weight]

        term_weights.append(term_weight)

    return term_weights

# 将数据集划分为n折
def split_filter_data(train_file,n):



    pass

def main():
    pass


if __name__ == '__main__':
    main()