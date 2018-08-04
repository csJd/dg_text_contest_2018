# coding: utf-8
# created by Pengpeng on 2018/7/22
import pickle as pk
from utils.path_util import from_project_root
from utils.data_util import train_dev_split
import collections
from sklearn.externals import joblib
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD

def process_sen(tf_dict,word_list):

    filtered_word_list = []
    for word in word_list:

        if tf_dict[word] <=2 or tf_dict[word] >= 6000:
           continue
        filtered_word_list.append(word)

    return filtered_word_list

# 将pk文件转化为csv文件 【排序】
def  transfer_pk_to_csv(pickle_file,save_csv_file):

    data_dict = pk.load(open(pickle_file,'rb'))

    sorted_tuple = sorted(data_dict.items(), key=lambda x: x[1], reverse=False)

    with open(save_csv_file,'w',encoding='utf-8') as f:

        for (word,value) in sorted_tuple:

            f.write("{},{}\n".format(word,value))

# 将原训练文件，提取对应的列
def extract_data(train_file,extract_index,save_file):
    """
    Args:
        train_file: train_set.csv
        extract_index:  [3,2] : phrase_level and [3,1] : word_level
        save_file:
    Returns:
    """
    with open(train_file,'r',encoding='utf-8') as f,open(save_file,'w',encoding='utf-8') as wf:
        for line in f.readlines()[1:]:
            line_list = line.strip().split(',')

            for i in range(len(extract_index)):
                wf.write(line_list[extract_index[i]])
                if i < len(extract_index)-1:
                    wf.write(",")

            wf.write("\n")

# 查询训练集平均句子长度
def get_average_sen_len(filename):

    sum_len = 0
    sen_count = 0
    limit_count = 0
    with open(filename,'r',encoding='utf-8') as f:
        for line in f.readlines():
            sen_count += 1
            line_list = line.strip().split(',')
            sen_len = len(line_list[1].strip().split())
            if sen_len < 600:
                limit_count += 1
            sum_len += sen_len
    average_len = sum_len / sen_count
    print("句子平均长度：{}".format(average_len))
    print("句子长度大于250: {}".format(limit_count))
    return average_len

# 从phrase_level_dc.csv中提取df——词语的文档数
def get_dc_pickle(dc_csv_file,df_pickle):
    """
    从文件lstm_model/processed_data/phrase_level_dc.csv中提取df值
    Args:
        dc_csv_file: lstm_model/processed_data/phrase_level_dc.csv
        df_pickle:lstm_model/processed_data/phrase_df.pk
    Returns:
    """
    df_dict = collections.defaultdict(float)
    with open(dc_csv_file,'r',encoding='utf-8') as f:
        for line in f.readlines()[1:]:
            line_list = line.strip().split(',')
            word = line_list[0]
            df_value = float(line_list[2])
            df_dict[word] = df_value

    pk.dump(df_dict,open(df_pickle,'wb'))
    pass

# 从数据文件中建立共有的字典
def create_vocab_dict(train_file,vocab_pickle):
    vocab_dict = collections.defaultdict(int)
    word_count = 1
    with open(train_file,'r',encoding='utf-8') as f:
        for line in f.readlines():
            line_list = line.strip().split(',')
            word_list = line_list[1].strip().split()
            for word in word_list:
                if word not in vocab_dict.keys():
                    vocab_dict[word] = word_count
                    word_count += 1
    pk.dump(vocab_dict,open(vocab_pickle,'wb'))
    return vocab_dict

# 使用pca将tf-bdc词袋模型的高维度进行降维
def pca(tfbdc_word_bag_pickle,pca_tfbdc_file):
    """
    Args:
        tfbdc_pickle: 词袋模型的pickle
        pca_tfbdc_pickle: pca降维之后的picklex
    Returns:x
    """
    # 加载tf_bdc权重表示
    x,y = joblib.load(tfbdc_word_bag_pickle)
    svd = TruncatedSVD(1000)
    X_transformed = svd.fit_transform(x)
    # joblib.dump((X_transformed, y), pca_tfbdc_pickle)
    with open(pca_tfbdc_file,'w',encoding='utf-8') as f:
        for i in len(y):
            f.write("{},{}\n".format(y[i],X_transformed[i]))

    print(X_transformed)

def main():

    # 将词袋模型的tf_bdc权重进行降维
    tfbdc_word_bag_pickle = from_project_root("lstm_model/processed_data/vector/tfbdc_1gram_300000_Xy.pk")
    pca_tfbdc_pickle = from_project_root("lstm_model/processed_data/vector/pca_tfbdc_1gram_300000_Xy.pk")
    pca(tfbdc_word_bag_pickle,pca_tfbdc_pickle)
    exit()

    # 根据train_file建立字典
    # train_file = from_project_root("lstm_model/processed_data/filter_phrase_level_data.csv")
    # vocab_pickle = from_project_root("lstm_model/processed_data/filter_phrase_level_vocab.pk")
    # create_vocab_dict(train_file,vocab_pickle)
    # exit()

    # 将pk转化为csv文件
    # pickle_file = from_project_root("lstm_model/processed_data/filter_phrase_level_vocab.pk")
    # save_csv_file = from_project_root("lstm_model/processed_data/filter_phrase_level_vocab.csv")
    # transfer_pk_to_csv(pickle_file, save_csv_file)
    # exit()

    # 计算平均句子长度
    # get_average_sen_len(from_project_root("lstm_model/processed_data/filtered_phrase_data_train.csv"))
    # exit()

    # extract data
    # extract_data(from_project_root("data/test_set.csv"),[0,2],
    #              from_project_root("lstm_model/processed_data/phrase_level_test_data.csv"))
    # exit()

    # 划分数据集
    train_dev_split(from_project_root("lstm_model/processed_data/filter_phrase_level_data.csv"))
    exit()

    # tf_pickle = from_project_root("lstm_model/processed_data/phrase_tf.pk")
    # transfer_tf
    # transfer_pk_to_csv(tf_pickle,from_project_root("lstm_model/processed_data/phrase_tf.csv"))
    # filter_words(tf_pickle)
    # exit()

    # 从phrase_level_dc.csv中提取df——词语的文档数
    dc_csv_file = from_project_root("lstm_model/processed_data/phrase_level_dc.csv")
    df_pickle = from_project_root("lstm_model/processed_data/df_pickle.pk")
    get_dc_pickle(dc_csv_file, df_pickle)


if __name__ == '__main__':
    main()