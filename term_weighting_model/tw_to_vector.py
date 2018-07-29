# coding: utf-8
# created by deng on 7/30/2018

from utils.data_util import load_raw_data
from sklearn.feature_extraction.text import CountVectorizer


def to_vector(data_url, tw_dict, norm=True):
    """

    Args:
        data_url: the data to transfer into vector
        tw_dict: term weighting dict to use
        norm: norm the vector or not

    Returns:
        X, y

    """
    labels, sentences = load_raw_data(data_url)
    print("transforming...")
    vectorizer = CountVectorizer(min_df=1, max_df=0.8)
    X = vectorizer.fit_transform(sentences)

    # get words of all columns represent to
    words = vectorizer.get_feature_names()
    weights = list()
    for word in words:
        weights.append(tw_dict[word])

    # remain to complete
    pass


def main():
    pass


if __name__ == '__main__':
    main()
