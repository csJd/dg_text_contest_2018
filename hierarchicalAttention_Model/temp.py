# coding: utf-8
# created by Pengpeng on 2018/8/27
import pickle as pk
import numpy as np
from hierarchicalAttention_Model import Data_helper
from utils.path_util import from_project_root
#
def predict(pro_file):

    data = pk.load(open(pro_file,'rb'))
    result = []
    for predict_one_merge in data:
        max_index = np.where(predict_one_merge == np.max(predict_one_merge))[0][0]
        result.append(max_index+1)

    predict_context, ids = Data_helper.get_predict_data(
        from_project_root("lstm_model/processed_data/phrase_level_test_data.csv"))

    # 保存结果
    with open(from_project_root("hierarchicalAttention_Model/result/result_rcnn_rcnn_atten_han_5cv.csv"), 'w', encoding='utf-8') as f:
        f.write("id,class\n")
        for i in range(len(ids)):
            f.write("{},{}\n".format(ids[i], result[i]))


def main():

    pro_file = from_project_root("hierarchicalAttention_Model/temp/predict_merge.pk")
    predict(pro_file)

    pass


if __name__ == '__main__':
    main()