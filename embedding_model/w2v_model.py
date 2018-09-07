# coding: utf-8
# created by deng on 7/25/2018

from utils.path_util import from_project_root, exists
from utils.data_util import load_raw_data, load_to_df

from gensim.models.word2vec import Word2Vec, Word2VecKeyedVectors
from sklearn.externals import joblib
from time import time
import numpy as np

DATA_URL = from_project_root("processed_data/phrase_level_data.csv")
TRAIN_URL = from_project_root("data/train_set.csv")
TEST_URL = from_project_root("data/test_set.csv")
N_JOBS = 4


def train_w2v_model(data_url=None, kwargs=None):
    """ get or train a new d2v_model

    Args:
        data_url: url to data file, None to train use
        kwargs: args for d2v model

    Returns:
        w2v_model

    """
    model_url = args_to_url(kwargs)
    if exists(model_url):
        return Word2Vec.load(model_url)

    if data_url is not None:
        _, sequences = load_raw_data(data_url)

    # use data from all train text and test text
    else:
        train_df = load_to_df(TRAIN_URL)
        test_df = load_to_df(TEST_URL)
        sequences = train_df['word_seg'].append(test_df['word_seg'], ignore_index=True)
        sequences = sequences.apply(str.split)

    print("Word2Vec model is training...\n trained model will be saved at \n ", model_url)
    s_time = time()
    # more info here [https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.Word2Vec]
    model = Word2Vec(sequences, workers=N_JOBS, **kwargs)
    e_time = time()
    print("training finished in %.3f seconds" % (e_time - s_time))
    model.save(model_url)
    # save wv of model
    wv_save_url = model_url.replace('.bin', '.txt').replace('w2v', 'wv')
    model.wv.save_word2vec_format(wv_save_url, binary=False)
    return model


def load_wv(url):
    """ load KeyedVectors wv

    Args:
        url: url to wv file

    Returns:
        Word2VecKeyedVectors: wv

    """
    return Word2VecKeyedVectors.load_word2vec_format(url, binary=False)


def args_to_url(args, prefix='w2v_word_seg_'):
    """ generate model_url from args

    Args:
        args: args dict
        prefix: filename prefix to save model

    Returns:
        str: model_url for train_w2v_model

    """
    args = dict(sorted(args.items(), key=lambda x: x[0]))
    filename = '_'.join([str(x) for x in args.values()]) + '.txt'
    return from_project_root("embedding_model/models/" + prefix + filename)


def avg_wv_of_words(wv_url, words):
    """ get avg word vector of words

    Args:
        wv_url: url to wv file
        words: word list

    Returns:
        np.ndarray: averaged word vector

    """
    wv = load_wv(wv_url)
    wvs = np.array([])
    for word in words:
        if word not in wv.vocab:
            continue
        wvs = np.append(wvs, wv[word])
    wvs = wvs.reshape(-1, wv.vector_size)
    avg_wv = np.mean(wvs, axis=0)
    return avg_wv.reshape((wv.vector_size,))


def infer_avg_wvs(wv_url, sentences):
    """ refer avg word vectors of sentences

    Args:
        wv_url: url to wv
        sentences: sentences, every sentence is a list of words

    Returns:
        np.ndarray: averaged word vectors

    """
    dvs = np.array([])
    wv = load_wv(wv_url)
    for sentence in sentences:
        wvs = np.array([])
        for word in sentence:
            if word not in wv.vocab:
                continue
            wvs = np.append(wvs, wv[word])
        wvs = wvs.reshape(-1, wv.vector_size)
        avg_wv = np.mean(wvs, axis=0)
        avg_wv = avg_wv.reshape((wv.vector_size,))

        dvs = np.append(dvs, avg_wv)
    return dvs.reshape(len(sentences), -1)


def gen_data_for_clf(wv_url, save_url):
    train_df = load_to_df(TRAIN_URL)
    test_df = load_to_df(TEST_URL)
    X = infer_avg_wvs(wv_url, train_df['word_seg'].apply(str.split))
    y = train_df['class'].values
    X_test = infer_avg_wvs(wv_url, test_df['word_seg'].apply(str.split))
    joblib.dump((X, y, X_test), save_url)


def main():
    kwargs = {
        'size': 300,
        'min_count': 5,
        'window': 5,
        'iter': 5,
        'sg': 1,
        'hs': 1
    }
    model = train_w2v_model(data_url=None, kwargs=kwargs)
    print(len(model.wv.vocab))
    save_url = from_project_root("processed_data/vector/avg_wvs_300.pk")
    gen_data_for_clf(args_to_url(kwargs), save_url=save_url)
    pass


if __name__ == '__main__':
    main()
