# -*- coding: utf-8 -*-
import pre_process.data_util as du
import term_weighting_model.text_represent as tr
import term_weighting_model.classifier as cf
#from utils.path_util import from_prohect_root

def run():
    validation_train_data_filename = "data/small_train.csv"
    validation_test_data_filename = "data/small_test.csv"
    validation_train_data = du.read_data_df(validation_train_data_filename, data_type="train")
    validation_test_data = du.read_data_df(validation_test_data_filename, data_type="train")
    train_data, train_label = du.get_data_label(validation_train_data, type="word", data_type="train")
    test_data, test_label = du.get_data_label(validation_test_data, type="word", data_type="train")
    train_vector, test_vector = tr.tf_idf(train_data, test_data, "tf_idf", "phrase")
    # train_vector, test_vector = tr.weihgt_vector(train_data, test_data, sentence_type='phrase',is_tf=False,
    #                                             weight_type="idf", weight_tes=0, count_tes=0)
    cf.knn(train_vector, train_label, test_vector, test_label)
    cf.m_nb(train_vector, train_label, test_vector, test_label)
    print("train model")
    #cf.rf(train_vector, train_label, test_vector, test_label)
    cf.knn(train_vector, train_label, test_vector, test_label)
    cf.m_nb(train_vector, train_label, test_vector, test_label)
    #cf.svm_svc(train_vector, train_label, test_vector, test_label)
    #cf.ovr(train_vector, train_label, test_vector, test_label)
    #cf.xgb(train_vector, train_label, test_vector, test_label)
    #cf.gbc(train_vector, train_label, test_vector, test_label)
    cf.ls(train_vector, train_label, test_vector, test_label)


if __name__  == "__main__":
#     dp.save_array("./label/train_label.npy", train_label)
#     dp.save_array("./label/test_label.npy", test_label)
    # train_vector, test_vector = tr.tf_idf(train_data, test_data, "word")
    # train_vector, train_label, test_vector, test_label = dp.load_vector_label(method_type="tf_idf")
    run()


