# coding: utf-8
# created by deng on 7/31/2018

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.externals import joblib

import term_weighting_model.tw_to_vector as tw2v
from utils.path_util import from_project_root


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

    result_file = open('processed_data/com_result/baseline_tfidf_3gram_2000000.csv', 'w')
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

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    result_file = open('processed_data/com_result/result.csv', 'w')
    result_file.write("id,class" + "\n")
    for i, label in enumerate(y_pred):
        result_file.write(str(i) + "," + str(label) + "\n")
    result_file.close()


def main():
    column = "word_seg"
    test_df = pd.read_csv(from_project_root('data/test_set.csv'))
    sentences = test_df[column]
    tw_dict = from_project_root("processed_data/saved_weight/phrase_level_3gram_bdc.json")
    X_test = tw2v.to_vector(sentences, tw_dict)

    X, y = joblib.load(from_project_root("processed_data/vector/bdc_3gram_4000000_Xy.pk"))
    clf = LinearSVC()
    train_and_predict_test(clf, X, y, X_test)


if __name__ == '__main__':
    main()