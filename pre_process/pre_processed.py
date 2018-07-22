#coding=utf-8
import pickle as pk

# 通过权值 tf idf bdc 三个权重将对分类没有用的词过滤掉。
def phrase_filter(tf_pickle,idf_pickle,bdc_pickle,train_file,word_dict_file):

    tf_value_dict = pk.load(open(tf_pickle,'rb'))
    idf_value_dict = pk.load(open(idf_pickle,'rb'))
    bdc_value_dict = pk.load(open(bdc_pickle,'rb'))

    word_dict = pk.load(open(word_dict_file,'rb'))

    filter_count = 0
    for (word,word_id) in word_dict:

        # 条件一
        if tf_value_dict[word] == 1 :

            pass


    #
    # with open(train_file,'r',encoding='utf-8') as f:
    #
    #     for line in f.readlines():
    #         line_list = line.strip().split(',')
    #         label = line_list[0]
    #         word_list = line_list[1].strip().split()
    #     f.close()

    pass

#
def main():

    pass


if __name__ == "__main__":

    phrase_filter()