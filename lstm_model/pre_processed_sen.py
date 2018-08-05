# coding: utf-8
# created by Pengpeng on 2018/7/29
"""
   本文件目的是用来根据tf-bdc压缩句子长度,在句子级别上进行操作.
   基本思想:
          我们认为对一个文档来说,若是词频越高,bdc值越高,那么该词语就越能表达本文档含义.
          去除超高词频的词语,这类词语很有可能是无意义词.[因为在本压缩算法里面这类词语并不能去除]
"""
from utils.path_util import from_project_root
import collections
import pickle as pk
import tqdm

# 建立n-gram
def create_n_gram_sentence(phrase_train_file):
    #
    with open(phrase_train_file,'r',encoding='utf-8') as f:

        pass
    pass

# 根据权重方案进行过滤
def pre_processed_sen(bdc_pickle,tf_pickle,dc_pickle,df_pickle,train_file,processed_data_file,limit_word=400):

    """
    :param bdc_pickle: 根据全局计算出来的bdc权重
    :param train_file: 级别上的数据文件,此文件需要初步的去噪.[]
    :param limit_word: 限定每个样本文档不重复词语的个数[阈值]
    :param processed_data_file: 预处理好的文档路径
    :return:
    """
    # 加载bdc_value
    bdc_dict = pk.load(open(bdc_pickle,'rb'))
    # 加载tf_value
    tf_dict = pk.load(open(tf_pickle,'rb'))
    # 加载dc_value
    dc_dict = pk.load(open(dc_pickle,'rb'))
    # 加载df_value
    df_dict = pk.load(open(df_pickle,'rb'))
    line_count = 0
    # 读取训练文档
    with open(train_file,'r',encoding='utf-8') as f,open(processed_data_file,'w',encoding='utf-8') as wf:

        for line in f.readlines():
            print("filtered_line={}".format(line_count))
            line_count += 1

            line_list = line.strip().split(',')
            # 预处理完的词语列表
            processed_word_list = []
            # 记录词语的权重
            label = line_list[0]
            word_list = line_list[1].strip().split()

            # 过滤超高词频的词语==========================
            filted_word_list = []
            for word in word_list:
                if int(df_dict[word]) <= 3 :
                    continue
                if tf_dict[word] <= 5:
                    continue
                filted_word_list.append(word)

            sen_len = len(filted_word_list) # 作归一化使用,以免句子的长度影响最后句子级别上的权重

            # 计算句子级别上tf ==============================
            word_dict = collections.defaultdict(float)

            for word in filted_word_list:
                word_dict[word] += 1.0
            # 归一化,计算tf-bdc value =========================
            for (word,tf_value) in word_dict.items():
                word_dict[word] = word_dict[word] / sen_len * dc_dict[word]

            # 对word_dict权重进行排序: 从大到小排序 =============================
            sorted_word_tuple = sorted(word_dict.items(), key=lambda item: item[1], reverse=True)

            if len(sorted_word_tuple) < limit_word:  # 如果小于阈值,无需压缩
                processed_word_list = filted_word_list
                wf.write("{},{}\n".format(label, ' '.join(processed_word_list)))
                continue

            # 截取前limit_word阈值的词语,并将tuple转化成list类型=================================
            keep_words = []
            for (word,tf_bdc_value) in sorted_word_tuple[:limit_word]:
                keep_words.append(word)
            #
            for word in filted_word_list:
                if word in keep_words:
                    processed_word_list.append(word)
            wf.write("{},{}\n".format(label,' '.join(processed_word_list)))

def main():
    bdc_pickle= from_project_root("lstm_model/processed_data/phrase_bdc.pk")
    tf_pickle = from_project_root("lstm_model/processed_data/phrase_tf.pk")
    dc_pickle = from_project_root("lstm_model/processed_data/phrase_level_dc.pk")
    df_pickle = from_project_root("lstm_model/processed_data/df_pickle.pk")

    train_file = from_project_root("lstm_model/processed_data/phrase_level_data.csv")
    processed_data_file = from_project_root("lstm_model/processed_data/filter_phrase_level_data_200.csv")
    pre_processed_sen(bdc_pickle, tf_pickle,dc_pickle,df_pickle,train_file, processed_data_file, limit_word=200)
    pass

if __name__ == '__main__':
    main()


