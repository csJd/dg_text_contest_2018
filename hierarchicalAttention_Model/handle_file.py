# coding: utf-8
# created by Pengpeng on 2018/9/1
from utils.path_util import from_project_root
import pickle as pk
# 1534979954_4  1535012868_3  1535078354_2  1535118789_1  1535152941_0
# 生成(x,y,x_test)

# 多个五折结果
def create_x_y_x_test(model_paths,save_file):
    X = []
    for path in model_paths:
        data = pk.load(from_project_root(path))
        print(data)
        exit()



def main():

    # 得到X
    model_paths = ["hierarchicalAttention_Model/result/stacking_result/0/predict_merge_0_train.pk",
                   "hierarchicalAttention_Model/result/stacking_result/1/predict_merge_1_train.pk",
                   "hierarchicalAttention_Model/result/stacking_result/2/predict_merge_2_train.pk",
                   "hierarchicalAttention_Model/result/stacking_result/3/predict_merge_3_train.pk",
                   "hierarchicalAttention_Model/result/stacking_result/4/predict_merge_4_train.pk",
                   ""]
    create_x_y_x_test(model_paths,save_file)

    pass


if __name__ == '__main__':
    main()