# -*- coding: utf-8 -*-
import pre_process.data_util as du
import term_weighting_model.text_represent as tr
import term_weighting_model.classifier as cf
#from utils.path_util import from_prohect_root


def creat_processed_data():
    """
    批量生成预处理的数据文件
    :return: None
    """
    raw_data_filename = "data/small_train.csv"
    # raw_train_data = du.read_data_df(raw_data_filename, data_type="train")
    sentence_type = "phrase"
    # # 读取数据并划分验证集
    #
    # # 读取数据并生成word_distribution,不经过bdc的处理
    # du.word_in_label(raw_train_data, sentence_type=sentence_type, is_bdc=False)
    # # 读取数据并生成word_distribution，经过bdc的处理
    # du.word_in_label(raw_train_data, sentence_type=sentence_type, is_bdc=True)

    # 生成词的idf值
    word_filename = "processed_data/word_distribution/" + sentence_type + "_label_count.csv"
    tr.idf(word_filename, raw_data_filename, sentence_type, smooth_idf=0)

    # # 生成词的dc值
    # tr.dc_bdc(sentence_type, is_bdc=False)
    #
    # # 生成词的bdc值
    # tr.dc_bdc(sentence_type, is_bdc=True)

def classifier_model(train_vector, train_label, test_vector, test_label):
    """

    :param train_vector:
    :param train_label:
    :param test_vector:
    :param test_label:
    :return:
    """
    print("train model")
    cf.knn(train_vector, train_label, test_vector, test_label)
    cf.m_nb(train_vector, train_label, test_vector, test_label)
    #cf.rf(train_vector, train_label, test_vector, test_label)
    #cf.svm_svc(train_vector, train_label, test_vector, test_label)
    #cf.ovr(train_vector, train_label, test_vector, test_label)
    #cf.xgb(train_vector, train_label, test_vector, test_label)
    #cf.gbc(train_vector, train_label, test_vector, test_label)
    cf.ls(train_vector, train_label, test_vector, test_label)


def term_model(weight_type, sentence_type, weight_tes, count_tes):
    """
    根据weight_type和sentence_type跑分类模型
    :param weight_type: str，特征权重形式
    :param sentence_type: str，句子级别
    :param weight_tes: 浮点数，特征权重值的阈值，大于该阈值的特征词才被保留
    :param count_tes: 整数，特征词出现的文档数，大于该阈值的特征词才被保留
    :return: None
    """
    validation_train_data_filename = "data/small_train.csv"
    validation_test_data_filename = "data/small_test.csv"
    validation_train_data = du.read_data_df(validation_train_data_filename, data_type="train")
    validation_test_data = du.read_data_df(validation_test_data_filename, data_type="train")
    train_data, train_label = du.get_data_label(validation_train_data, sentence_type="word", data_type="train")
    test_data, test_label = du.get_data_label(validation_test_data, sentence_type="word", data_type="train")
    # train_vector, test_vector = tr.tf_idf(train_data, test_data, "tf_idf", "phrase")
    # du.save_array("processed_data/label/train_label.npy", train_label)
    # du.save_array("processed_data/label/test_label.npy", test_label)
    train_vector, test_vector = tr.weight_vector(train_data, test_data, sentence_type=sentence_type, is_tf=False,
                                                weight_type=weight_type, weight_tes=weight_tes, count_tes=count_tes)
    classifier_model(train_vector, train_label, test_vector, test_label)


def run(weight_type, sentence_type):
    """
    如果只是改变分类模型，而没有改变向量空间模型，则调用该方法
    :param weight_type: str，特征权重形式
    :param sentence_type: str，句子级别
    :return: None
    """
    train_vector, train_label, test_vector, test_label = du.load_vector_label(weight_type, sentence_type)
    classifier_model(train_vector, train_label, test_vector, test_label)


if __name__ == "__main__":

    # creat_processed_data()

    weight_type = "bdc"
    sentence_type = "phrase"
    weight_tes = 0
    count_tes = 0
    term_model(weight_type, sentence_type, weight_tes, count_tes)

    # run(weight_type, sentence_type)

