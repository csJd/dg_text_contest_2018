# -*- coding: utf-8 -*-

import pre_process.data_util as dp
import numpy as np

from sklearn import metrics

def train_model(clf, x_train, y_train, x_test, y_test):

    clf.fit(x_train, y_train)
    predict_test = clf.predict(x_test)
    dp.cal_compare_label("processed_data/com_result/compare_test.csv", predict_test, y_test)
    output_F_value(predict_test, y_test)
    predict_train = clf.predict(x_train)
    dp.cal_compare_label("processed_data/com_result/compare_train.csv", predict_train, y_train)
    output_F_value(predict_train, y_train)

def train_model_proba(clf, x_train, y_train, x_test, y_test):

    clf.fit(x_train, y_train)
    predict_test = clf.predict_proba(x_test)
    predict_test = np.argmax(predict_test, axis=1) + 1
    dp.cal_compare_label("processed_data/com_result/compare_test.csv", predict_test, y_test)
    output_F_value(predict_test, y_test)
    predict_train = clf.predict_proba(x_train)
    predict_train = np.argmax(predict_train, axis=1) + 1
    output_F_value(predict_train, y_train)
    dp.cal_compare_label("processed_data/com_result/compare_train.csv", predict_train, y_train)

def output_F_value(predict_y, original_y):

    print(metrics.f1_score(original_y, predict_y, average="macro"))

def eval_F_value(predict_y, d_test):
    original_y = d_test.get_label()
    return("F1", metrics.f1_score(original_y, predict_y, average="macro"))

from sklearn.neighbors import KNeighborsClassifier
def knn(x_train, y_train, x_test, y_test):
    clf = KNeighborsClassifier(n_neighbors=3)
    train_model(clf, x_train, y_train, x_test, y_test)
    print("knn")

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

def ovr(x_train, y_train, x_test, y_test):

    clf = OneVsRestClassifier(LinearSVC(), n_jobs=7)
    train_model(clf, x_train, y_train, x_test, y_test)
    print("ovr")

def ls(x_train, y_train, x_test, y_test):
    clf = LinearSVC()
    train_model(clf, x_train, y_train, x_test, y_test)
    print("ls")

from sklearn.naive_bayes import MultinomialNB

def m_nb(x_train, y_train, x_test, y_test):
    clf = MultinomialNB()
    train_model_proba(clf, x_train, y_train, x_test, y_test)
    print("nb")

from sklearn.ensemble import RandomForestClassifier

def rf(x_train, y_train, x_test, y_test):
    #clf = RandomForestClassifier(min_samples_leaf=5, n_estimators=200)
    clf = OneVsRestClassifier(RandomForestClassifier(min_samples_leaf=5, n_estimators=200))
    train_model(clf, x_train, y_train, x_test, y_test)
    print("rf")

from sklearn.ensemble import GradientBoostingClassifier

def gbc(x_train, y_train, x_test, y_test):
    clf = GradientBoostingClassifier()
    train_model(clf, x_train, y_train, x_test, y_test)
    print("gbc")

from sklearn.svm import SVC

def svm_svc(x_train, y_train, x_test, y_test):
    clf = SVC(kernel="linear")
    train_model(clf, x_train, y_train, x_test, y_test)
    print("svc")

# from xgboost.sklearn import XGBClassifier
# import  xgboost

# def xgb(x_train, y_train, x_test, y_test):
#     param = {}
#     param['objective'] = 'multi:softmax'
#     param["booster"] = "gbtree"
#     param["early_stopping_rounds"] = 100
#     param["scale_pos_weight"] = 10
#     param['eta'] = 0.3
#     param['min_child_weight'] = 3
#     param["colsample_bytree"] = 0.4
#     param["gamma"] = 2
#     param["lambda"] = 1
#     param["alpha"] = 5
#     param["subsample"] = 0.7
#     param['max_depth'] = 10
#     param['silent'] = 1
#     param["base_score"] = 1
#     #param["eval_metric"] = "mlogloss"
#     param['nthread'] = 7
#     param['num_class'] = 20
#     xg_train = xgboost.DMatrix(x_train, label=y_train)
#     xg_test = xgboost.DMatrix(x_test, label=y_test)
#     watchlist = [(xg_train, "train"), (xg_test, "test")]
#     num_round = 1000
#     clf = xgboost.train(param, xg_train, num_round, watchlist, feval=eval_F_value, maximize=False)
#     pred = clf.predict(xg_test)
#     dp.cal_compare_label("./com_label/compare_test2.csv", pred, y_test)
#     output_F_value(pred, y_test)
#     print("xgb")

# def xgb(x_train, y_train, x_test, y_test):
#     param = {}
#     param['objective'] = 'multi:softmax'
#     param["booster"] = "gblinear"
#     param["early_stopping_rounds"] = 100
#     param["lambda"] = 10
#     param['silent'] = 1
#     param['nthread'] = 7
#     param['num_class'] = 20
#     xg_train = xgboost.DMatrix(x_train, label=y_train)
#     xg_test = xgboost.DMatrix(x_test, label=y_test)
#     watchlist = [(xg_train, "train"), (xg_test, "test")]
#     num_round = 1000
#     clf = xgboost.train(param, xg_train, num_round, watchlist, feval=eval_F_value, maximize=False)
#     pred = clf.predict(xg_test)
#     dp.cal_compare_label("./com_label/compare_test2.csv", pred, y_test)
#     output_F_value(pred, y_test)
#     print("xgb")

# def xgb(x_train, y_train, x_test, y_test):
#     clf = XGBClassifier(booster="dart", objective="multi:softmax", n_jobs=6)
#     train_model(clf, x_train, y_train, x_test, y_test)
#     print("xgb")



