#coding=utf-8
import pandas as pd


def get_labels(init_file,predict_file):

    init_label = []
    predict_label = []

    pd_init = pd.read_csv(init_file,sep="^",header=None)

    for index,row in pd_init.iterrows():

        init_label.append(row[0])

    pd_predict = pd.read_csv(predict_file,sep=",",header=None)

    for index,row in pd_predict.iterrows():

        predict_label.append(row[0])

    print(predict_label)
    print(init_label)

    correct_count = 0
    error_index = []
    for i in range(len(init_label)):

        if init_label[i] == predict_label[i]:

            correct_count += 1

        else:

            error_index.append(i)

    print("correct_count : "+str(correct_count))
    correct_rate = correct_count / len(pd_predict)

    return correct_rate,error_index

if __name__ == "__main__":

    correct_rate,error_index = get_labels("../processed_data/dev_processed_data_split.csv","./result/result_predict.txt")

    print("correct_rate : "+str(correct_rate))
    print("error_email  : "+str(error_index))
