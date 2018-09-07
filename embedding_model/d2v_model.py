# coding: utf-8
# created by deng on 7/25/2018

from utils.path_util import from_project_root, exists
from utils.data_util import load_raw_data, load_to_df

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.externals import joblib
from collections import OrderedDict
from time import time

import numpy as np

# define some val

DATA_URL = from_project_root("processed_data/phrase_level_data.csv")
TRAIN_URL = from_project_root("data/train_set.csv")
TEST_URL = from_project_root("data/test_set.csv")
N_JOBS = 4


def train_d2v_model(data_url=None, kwargs=None):
    """ get or train a new d2v_model

    Args:
        data_url: url to data file
        kwargs: args for d2v model

    Returns:
        w2v_model

    """
    model_url = args_to_url(kwargs)
    if exists(model_url):
        return Doc2Vec.load(model_url)

    if data_url is not None:
        _, sequences = load_raw_data(data_url)

    # use data from all train text and test text
    else:
        train_df = load_to_df(TRAIN_URL)
        test_df = load_to_df(TEST_URL)
        sequences = train_df['word_seg'].append(test_df['word_seg'], ignore_index=True)
        sequences = sequences.apply(str.split)

    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(sequences)]
    print("Doc2Vec model is training...\n trained model will be saved at \n ", model_url)
    # more info here [https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.Word2Vec]
    s_time = time()
    model = Doc2Vec(documents, workers=N_JOBS, **kwargs)
    model.save(model_url)
    e_time = time()
    print("training finished in %.3f seconds" % (e_time - s_time))
    return model


def infer_dvs(model_url, sentences):
    """ refer doc vectors of sentences

    Args:
        model_url: url to d2v model
        sentences: sentences, every sentence is a list of words

    Returns:
        np.ndarray: doc vectors

    """
    dv_model = Doc2Vec.load(model_url)
    dvs = np.array([])
    for sentence in sentences:
        dv = dv_model.infer_vector(sentence)
        dvs = np.append(dvs, dv)
    return dvs.reshape(-1, dv_model.vector_size)


def args_to_url(args, prefix='d2v_word_seg_'):
    """ generate model_url from args

    Args:
        args: args dict
        prefix: filename prefix

    Returns:
        str: model_url for train_d2v_model

    """
    filename = '_'.join([str(x) for x in OrderedDict(args).values()]) + '.bin'
    return from_project_root("embedding_model/models/" + prefix + filename)


def gen_data_for_clf(model_url, save_url):
    train_df = load_to_df(TRAIN_URL)
    test_df = load_to_df(TEST_URL)
    X = infer_dvs(model_url, train_df['word_seg'].apply(str.split))
    y = train_df['class'].values
    X_test = infer_dvs(model_url, test_df['word_seg'].apply(str.split))
    joblib.dump((X, y, X_test), save_url)


def main():
    kwargs = {
        'vector_size': 300,
        'min_count': 5,
        'window': 5,
        'epochs': 5,
        'hs': 0,
    }
    model = train_d2v_model(data_url=None, kwargs=kwargs)
    print('model vocab len = ', len(model.wv.vocab))
    save_url = from_project_root("processed_data/vector/dvs_300.pk")
    gen_data_for_clf(args_to_url(kwargs), save_url=save_url)
    pass


if __name__ == '__main__':
    main()
