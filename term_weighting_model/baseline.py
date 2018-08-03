# coding: utf-8
# created by deng on 7/31/2018

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.externals import joblib

import term_weighting_model.tw_to_vector as tw2v
from utils.path_util import from_project_root
from term_weighting_model.transformer import TfdcTransformer
import utils.json_util as ju


def tfdc_baseline():
    column = "word_seg"
    train_df = pd.read_csv(from_project_root('data/train_set.csv'))
    test_df = pd.read_csv(from_project_root('data/test_set.csv'))
    vec = CountVectorizer(ngram_range=(1, 3), min_df=3, max_df=0.8,
                          max_features=3000000, token_pattern='\w+')
    trans = TfdcTransformer(sublinear_tf=True, balanced=True)
    X_train = vec.fit_transform(train_df[column])
    y_train = np.array((train_df["class"]).astype(int))
    X_train = trans.fit_transform(X_train, y_train)

    X_test = vec.transform(test_df[column])
    X_test = trans.transform(X_test)

    lin_clf = LinearSVC()
    lin_clf.fit(X_train, y_train)
    preds = lin_clf.predict(X_test)

    result_file = open(from_project_root('processed_data/com_result/baseline_tfbdc_3gram_3000000.csv'), 'w')
    result_file.write("id,class" + "\n")
    for i, label in enumerate(preds):
        result_file.write(str(i) + "," + str(label) + "\n")
    result_file.close()


def tfidf_baseline():
    column = "word_seg"
    train = pd.read_csv(from_project_root('data/train_set.csv'))
    test = pd.read_csv(from_project_root('data/test_set.csv'))
    vec = TfidfVectorizer(ngram_range=(1, 3), min_df=3, max_df=0.8, max_features=2000000, sublinear_tf=1)
    trn_term_doc = vec.fit_transform(train[column])
    test_term_doc = vec.transform(test[column])

    y = (train["class"]).astype(int)
    lin_clf = LinearSVC()
    lin_clf.fit(trn_term_doc, y)
    preds = lin_clf.predict(test_term_doc)

    result_file = open(from_project_root('processed_data/com_result/baseline_tfidf_3gram_2000000.csv'), 'w')
    result_file.write("id,class" + "\n")
    for i, label in enumerate(preds):
        result_file.write(str(i) + "," + str(label) + "\n")
    result_file.close()


def train_and_predict_test(clf, X_train, y_train, X_test):
    """ train clf use all training data and generate result data of test set

    Args:
        clf: classifier
        X_test: X of test data
        X_train: X of all training data
        y_train: y of all training data

    """

    print("classifier is training")
    clf.fit(X_train, y_train)
    print("training finished, predicting")
    y_pred = clf.predict(X_test)

    result_file = open(from_project_root('processed_data/com_result/bdc4000000.csv'), 'w')
    result_file.write("id,class" + "\n")
    for i, label in enumerate(y_pred):
        result_file.write(str(i) + "," + str(label) + "\n")
    result_file.close()


def main():
    column = "word_seg"
    test_df = pd.read_csv(from_project_root('data/test_set.csv'))
    sentences = test_df[column]
    tw_dict = ju.load(from_project_root("processed_data/saved_weight/phrase_level_3gram_bdc.json"))
    X_test = tw2v.to_vector(sentences, tw_dict, max_features=4000000)

    X, y = joblib.load(from_project_root("processed_data/vector/bdc_3gram_4000000_Xy.pk"))
    clf = LinearSVC()
    train_and_predict_test(clf, X, y, X_test)


if __name__ == '__main__':
    # main()
    tfdc_baseline()
