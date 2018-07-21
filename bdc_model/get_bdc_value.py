#coding=utf-8
import collections
import pickle as pk
import numpy as np

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


def cal_bdc_value(train_file,word_dict_url,bdc_pickle):

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
        bdc_value = 1 + (sum_B / np.log(len(category)))

        bdc_value_dict[word] = bdc_value
    #
    pk.dump(bdc_value_dict,open(bdc_pickle,'wb'))

def main():

    train_file = "../pre_process_data/phrase_level_data.csv"
    word_dict_url = "../pre_process_data/phrase_level_word_dict.pk"
    bdc_pickle = "../pre_process_data/phrase_level_bdcValue.pk"
    cal_bdc_value(train_file,word_dict_url,bdc_pickle)
    pass

if __name__ == "__main__":

    main()