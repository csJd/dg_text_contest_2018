# coding: utf-8
# created by Pengpeng on 2018/8/15
import pickle as pk
from utils.path_util import from_project_root
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import lstm_model.Data_helper as Data_helper


def main():

    pros = []  # 保存了75个模型
    for cv_num in range(5):
        for step in [900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300]:
            pros.append(pk.load(open(from_project_root("lstm_model/result_test/result_predict"+str(cv_num)+"-"+str(step)+".csv"),'rb')))
    predict_merge = []
    pro_merge = []
    for i in range(len(pros[0])): # 预测样本的条数

        one_pro_merge = np.array([0.0] * 19 )
        for j in range(75):  # 分别拿出每个模型的第i个样本的预测结果
            one_pro_merge = one_pro_merge + np.array(pros[j][i])

        max_index = np.where(one_pro_merge==np.max(one_pro_merge))[0][0]
        predict_merge.append(max_index%19+1)  # 这就是最终的预测结果pro_merge
        pro_merge.append(one_pro_merge/75)
    #
    predict_context, ids = Data_helper.get_predict_data(from_project_root("lstm_model/processed_data/phrase_level_test_data.csv"))

    # 保存结果
    with open(from_project_root("lstm_model/result/result_rnn_cnn_attention.csv"),'w',encoding='utf-8') as f:
        f.write("id,class\n")
        for i in range(len(ids)):
            f.write("{},{}\n".format(ids[i],predict_merge[i]))

    # 保存概率文件
    pk.dump(pro_merge,open(from_project_root("lstm_model/result/result_rcnn_0.77.pk"),'wb'))

def rcnn_rcnn_attention():

    rcnn_model = from_project_root("lstm_model/result/prob_rcnnon_cv0.789744.csv")
    rnn_cnn_attention = from_project_root("lstm_model/result/result_rcnn_0.775.pk")

    rcnn_pro = []
    with open(rcnn_model,'r',encoding='utf-8') as f:
        for line in f.readlines()[1:]:
            pro_list = np.array(line.strip().split(',')[:-1]).astype(np.float32)
            rcnn_pro.append(pro_list)

    rcnn_attention_pro = pk.load(open(rnn_cnn_attention,'rb'))

    predict_merge = []
    predict_pro_merge = []
    for i in range(len(rcnn_attention_pro)): # 预测样本的条数

        one_predict_merge = (np.array(rcnn_pro[i])/5 + np.array(rcnn_attention_pro[i])) / 2

        predict_pro_merge.append(one_predict_merge)

        max_index = np.where(one_predict_merge==np.max(one_predict_merge))[0][0]
        print(max_index)
        predict_merge.append(max_index+1)  # 这就是最终的预测结果pro_merge

    predict_context, ids = Data_helper.get_predict_data(
        from_project_root("lstm_model/processed_data/phrase_level_test_data.csv"))

    # 保存
    pk.dump(predict_pro_merge,open(from_project_root("lstm_model/result/pro_rcnn_rnn_cnn_attention_0.794.pk"),'wb'))

    # 保存结果
    with open(from_project_root("lstm_model/result/result_rcnn_rnn_cnn_attention.csv"), 'w', encoding='utf-8') as f:
        f.write("id,class\n")
        for i in range(len(ids)):
            f.write("{},{}\n".format(ids[i], predict_merge[i]))


    pass

if __name__ == "__main__":
    rcnn_rcnn_attention()
    exit()
    main()











    pass