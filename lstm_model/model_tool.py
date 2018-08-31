# coding: utf-8
# created by Pengpeng on 2018/8/10
import pickle as pk
import utils.json_util as ju
import collections
import numpy as np
from utils.path_util import from_project_root

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
# 根据数据文件划分为分数
def split_data_to_parts(train_data_file,n_parts,dev_nums):

    all_datas = []
    with open(train_data_file,'r',encoding='utf-8') as f:
        for line in f.readlines():
            all_datas.append(line.strip())

    #
    dev_data = all_datas[-dev_nums:]
    write_to_csv(dev_data,train_data_file,"dev")

    sample_size = len(all_datas) - dev_nums

    part_size = sample_size // n_parts
    for i in range(n_parts):
        startindex = i * part_size
        if (i+1) * part_size + n_parts > sample_size - 1:
            endindex = sample_size - 1
        else:
            endindex = (i+1) * part_size
        part_data = all_datas[startindex:endindex]
        write_to_csv(part_data,train_data_file,str(i))

def write_to_csv(part_data,train_data_file,i):

    print(len(part_data))
    new_file_name = train_data_file[:-4]+"_"+i+".csv"
    with open(new_file_name,'w',encoding='utf-8') as f:
        for line in part_data:
            f.write(line+"\n")

def main():
    train_data_file= from_project_root("lstm_model/processed_data/one_gram/filter-1gram_phrase_level_data_200.csv")
    n_parts=5
    dev_nums = 5000
    split_data_to_parts(train_data_file,n_parts,dev_nums)
    pass


if __name__ == '__main__':
    main()