# coding: utf-8
# created by deng on 7/30/2018

import numpy as np
import scipy as sp

import utils.json_util as ju
import term_weighting_model.calc_weightings as cw
from utils.data_util import load_raw_data
from utils.path_util import from_project_root
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.externals import joblib
from tqdm import tqdm

MAX_FEATURES = 5000000
MIN_DF = 3
MAX_DF = 0.8
MAX_N = 3
TW_TYPE = 'bdc'
TRAIN_URL = from_project_root("processed_data/phrase_level_data.csv")


def tw_vectorize(data_url, tw_type='idf'):
    """

    Args:
        data_url: str, url to data file
        tw_type: str, term wighting type {idf, dc, bdc}

    Returns:
        X, y

    """
    labels, sentences = load_raw_data(data_url, ngram=None)
    X = None
    y = np.array(labels)
    if tw_type == 'idf':
        X = tfidf_to_vector(sentences)
    elif tw_type == 'dc':
        dc_dict = cw.calc_dc(data_url, ngram=MAX_N)
        X = to_vector(sentences, dc_dict)
    elif tw_type == 'bdc':
        bdc_dict = cw.calc_bdc(data_url, ngram=MAX_N)
        X = to_vector(sentences, bdc_dict)

    return X, y


def tfidf_to_vector(sentences):
    """ vectorize use TfidfVectorizer

    Args:
        sentences: list of sentence to be vectorized

    Returns:
        X, vectorized data

    """
    # set token_pattern(default: (?u)\b\w\w+\b' to keep single char tokens
    vectorizer = TfidfVectorizer(min_df=MIN_DF, max_df=MAX_DF, max_features=MAX_FEATURES,
                                 ngram_range=(1, MAX_N), sublinear_tf=True, token_pattern='(?u)\w+')
    X = vectorizer.fit_transform(sentences)
    return X


def to_vector(sentences, tw_dict, max_features=MAX_FEATURES, normalize=True, sublinear_tf=True):
    """

    Args:
        sentences: list of sentence to be vectorized
        tw_dict: term weighting dict to use
        normalize: normalize the vector or not
        sublinear_tf: use 1 + log(tf) instead of tf
        max_features: max_features for CountVectorizer

    Returns:
        X, vectorized data

    """
    print("transforming...")
    _, train_sentences = load_raw_data(TRAIN_URL, ngram=None)
    vectorizer = CountVectorizer(min_df=MIN_DF, max_df=MAX_DF, ngram_range=(1, MAX_N),
                                 token_pattern='(?u)\w+', max_features=MAX_FEATURES)
    vectorizer.fit(train_sentences)  # use train data to get vocab
    X = vectorizer.transform(sentences)

    # get words of all columns represent to
    words = vectorizer.get_feature_names()

    # sublinear_tf like tf-idf
    if sublinear_tf:
        X.data = np.log(X.data) + 1

    # get weights of words
    weights = np.array([tw_dict[word] for word in words])
    X = X.multiply(weights)  # can not use * to multiply

    if normalize:
        norm = sp.sparse.linalg.norm(X, axis=1)
        for i, row in enumerate(tqdm(X.row)):
            X.data[i] /= norm[row]

    return X


def main():
    X, y = tw_vectorize(TRAIN_URL, tw_type=TW_TYPE)
    joblib.dump((X, y), from_project_root("processed_data/vector/tf{}_{}gram_{}_Xy.pk"
                                          .format(TW_TYPE, MAX_N, MAX_FEATURES)))
    pass


if __name__ == '__main__':
    main()
