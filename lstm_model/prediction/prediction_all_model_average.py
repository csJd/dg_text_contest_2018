# 读取文件夹中所有的文件
import os
from utils.path_util import from_project_root
from lstm_model import Data_helper
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import pickle as pk


def read_all_filenames(dir_url):
    '''
    :param dir_url:存储所有预测结果文件的文件夹路径
    :return:
    '''
    for root,dirs,files in os.walk(dir_url):
        pass

    all_predict_files = []
    for file_name in files:
        all_predict_files.append(dir_url+"/"+file_name)

    return all_predict_files
#
def main(result_dir):

    all_predict_files = read_all_filenames(from_project_root(result_dir))

    all_predict_results = []
    for predict_file in all_predict_files:
        all_predict_results.append(pk.load(open(predict_file,'rb')))

    predict_all = []
    predict_all_pro = []
    for i in range(len(all_predict_results[0])):

        predict_one_merge = np.array([0.0] * 19)

        for j in range(len(all_predict_results)):

            predict_one_merge = predict_one_merge + np.array(all_predict_results[j][i])

        #
        predict_one_merge = predict_one_merge / len(all_predict_results)
        max_index = np.where(predict_one_merge == np.max(predict_one_merge))[0][0]
        predict_all.append(max_index + 1)
        predict_all_pro.append(predict_one_merge)

    # predict_context, predict_labels = Data_helper.get_predict_data(from_project_root(
    #     "lstm_model/processed_data/one_gram/filter_1-gram_phrase_level_data_400_dev.csv"))
    #
    # macro_f1 = f1_score(predict_all, predict_labels, average='macro')
    # accuracy_score1 = accuracy_score(predict_all, predict_labels, normalize=True)
    #
    # print("macro_f1:{}".format(macro_f1))
    # print("accuracy:{}".format(accuracy_score1))

    # save
    pk.dump(predict_all_pro,open(result_dir+"/predict_merge.pk",'wb'))

if __name__ == "__main__":

    result_dir = from_project_root("hierarchicalAttention_Model/result")
    main(result_dir)