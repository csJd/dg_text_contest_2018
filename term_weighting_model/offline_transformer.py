# coding: utf-8
# created by deng on 7/30/2018

import numpy as np
import scipy as sp

import term_weighting_model.calc_weightings as cw
from utils.data_util import load_to_df
from utils.path_util import from_project_root

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.externals import joblib

MAX_FEATURES = 4000000
MIN_DF = 3
MAX_DF = 0.8
MAX_N = 3
TW_TYPE = 'dc'
COLUMN = 'word_seg'
TRAIN_URL = from_project_root('data/train_set.csv')


def transform(train_url=TRAIN_URL, test_url=None, column='word_seg', tw_type=TW_TYPE):
    """

    Args:
        column: column to use
        train_url: str, url to train data (with header)
        test_url: url to test data
        tw_type: str, term wighting type {idf, dc, bdc}

    Returns:
        X, y, X_test: vertorized data

    """
    data_url = from_project_root("processed_data\phrase_level_data.csv")
    if column == 'article':
        data_url = data_url.replace('phrase', 'word')

    if tw_type == 'idf':
        return tfidf_transform(train_url, test_url, column=column)
    elif tw_type == 'dc':
        dc_dict = cw.calc_dc(data_url, ngram=MAX_N)
        return dict_transform(dc_dict, train_url, test_url, column=column)
    elif tw_type == 'bdc':
        bdc_dict = cw.calc_bdc(data_url, ngram=MAX_N)
        return dict_transform(bdc_dict, train_url, test_url, column=column)


def tfidf_transform(train_url, test_url, column='word_seg', sublinear_tf=True, max_n=MAX_N,
                    min_df=MIN_DF, max_df=MAX_DF, max_features=MAX_FEATURES):
    """ vectorize use TfidfVectorizer

    Args:
        train_url: url to train data
        test_url: url to test data
        column: column to use
        sublinear_tf:
        max_n:
        min_df:
        max_df:
        max_features:

    Returns:
        X, X_test, y: vectorized data

    """
    # set token_pattern(default: (?u)\b\w\w+\b' to keep single char tokens
    vectorizer = TfidfVectorizer(min_df=min_df, max_df=max_df, max_features=max_features, ngram_range=(1, max_n),
                                 sublinear_tf=sublinear_tf, token_pattern='(?u)\w+')
    train_df = load_to_df(train_url)
    X = vectorizer.fit_transform(train_df[column])
    y = np.asarray(train_df['class'])
    X_test = None
    if test_url:
        X_test = vectorizer.transform(load_to_df(test_url)[column])
    return X, y, X_test


def dict_transform(tw_dict, train_url=TRAIN_URL, test_url=None, column='word_seg',
                   max_n=MAX_N, min_df=MIN_DF, max_df=MAX_DF, max_features=MAX_FEATURES,
                   normalize=True, sublinear_tf=True, re_weight=0):
    """ use offline dict to transform data into vector

    Args:
        train_url: url to train data (with header)
        test_url: url to test data (with header)
        sentences: list of sentence to be vectorized
        tw_dict: term weighting dict to use
        max_n: max_n for CountVectorizer
        min_df: min_df for CountVectorizer
        max_df: max_df for CountVectorizer
        normalize: normalize the vector or not
        sublinear_tf: use 1 + log(tf) instead of tf
        max_features: max_features for CountVectorizer
        re_weight: if re_weight > 0, use (1-re_weight) + (re_weight) * weights instead of weight
        column: column to use in dataframe

    Returns:
        X, y, X_test: vectorized data

    """
    print("transforming...")
    train_df = load_to_df(train_url)
    vectorizer = CountVectorizer(min_df=min_df, max_df=max_df, ngram_range=(1, max_n),
                                 token_pattern='(?u)\w+', max_features=max_features)
    X_train = vectorizer.fit_transform(train_df[column])  # use train data to get vocab
    y_train = np.asarray(train_df['class'])
    X_test = vectorizer.transform(load_to_df(test_url)[column]) if test_url else None

    # get words of all columns represent to
    words = vectorizer.get_feature_names()
    # get weights of words
    weights = np.array([tw_dict[word] for word in words])
    if re_weight > 0:
        weights = 1 + re_weight * weights

    for X in (X_train, X_test):
        if X is None:
            continue

        # sublinear_tf like tf-idf
        if sublinear_tf:
            X.data = np.log(X.data) + 1

        X = X.multiply(weights)  # can not use * to multiply
        if normalize:
            norm = sp.sparse.linalg.norm(X, axis=1)
            for i, row in enumerate(X.row):
                X.data[i] /= norm[row]

    return X_train, y_train, X_test


def main():
    print("data generating...")
    xy_url = from_project_root("processed_data/vector/{}_tf{}_{}gram_{}_XyN.pk"
                               .format(COLUMN, TW_TYPE, MAX_N, MAX_FEATURES))
    # test_url = None
    test_url = from_project_root('data/test_set.csv')
    if test_url:
        xy_url.replace('XyN', 'XyX_test')
    print("generated (X, y, X_test) will be saved at", xy_url)
    X, y, X_test = transform(TRAIN_URL, test_url, column=COLUMN, tw_type=TW_TYPE)
    joblib.dump((X, y, X_test), xy_url)
    pass


if __name__ == '__main__':
    main()
