# coding: utf-8
# created by deng on 7/30/2018

import numpy as np
import scipy as sp

import utils.json_util as ju
from utils.data_util import load_raw_data
from utils.path_util import from_project_root
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.externals import joblib
from tqdm import tqdm

MAX_FEATURES = 20000
MIN_DF = 3
MAX_DF = 0.8
DATA_URL = from_project_root("processed_data/phrase_level_data.csv")


def tfidf_to_vector(data_url):
    """ vectorize use TfidfVectorizer

    Args:
        data_url: url to data to be transformed

    Returns:
        X, y

    """
    labels, sentences = load_raw_data(data_url, sentence_to_list=False)
    vectorizer = TfidfVectorizer(min_df=MIN_DF, max_df=MAX_DF, max_features=MAX_FEATURES)
    X = vectorizer.fit_transform(sentences)
    y = np.array(labels)
    return X, y


def to_vector(data_url, tw_dict, normalize=True):
    """

    Args:
        data_url: url to the data to transfer into vector
        tw_dict: term weighting dict to use
        normalize: normalize the vector or not

    Returns:
        X, y

    """
    labels, sentences = load_raw_data(data_url, sentence_to_list=False)
    print("transforming...")
    vectorizer = CountVectorizer(min_df=MIN_DF, max_df=MAX_DF, max_features=MAX_FEATURES)
    X = vectorizer.fit_transform(sentences)
    y = np.array(labels)

    # get words of all columns represent to
    words = vectorizer.get_feature_names()

    # get weights of words
    weights = np.array([tw_dict[word] for word in words])
    X = X.multiply(weights)  # can not use * to multiply

    if normalize:
        norm = sp.sparse.linalg.norm(X, axis=1)
        for i, row in enumerate(tqdm(X.row)):
            X.data[i] /= norm[row]

    return X, y


def main():
    bdc_dict = ju.load(from_project_root("processed_data/saved_weight/phrase_level_bdc.json"))
    X, y = to_vector(DATA_URL, bdc_dict)
    joblib.dump((X, y), from_project_root("processed_data/vector/bdc_Xy.pk"))
    pass


if __name__ == '__main__':
    main()
