# coding: utf-8
# created by deng on 7/25/2018


from utils.path_util import from_project_root, exists
from utils.data_util import load_raw_data
from gensim.models.word2vec import Word2Vec, Word2VecKeyedVectors
from time import time

import numpy as np

# define some val

DATA_URL = from_project_root("processed_data/phrase_level_data.csv")
N_JOBS = 4


def train_w2v_model(data_url, model_url, args):
    """ get or train a new d2v_model

    Args:
        data_url: url to data file
        model_url: url to save trained model
        args: args for d2v model

    Returns:
        w2v_model

    """
    if exists(model_url):
        return Word2Vec.load(model_url)

    _, documents = load_raw_data(data_url)
    print("Word2Vec model is training...\n trained model will be saved at \n ", model_url)
    s_time = time()
    # more info here [https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.Word2Vec]
    model = Word2Vec(documents, workers=N_JOBS,
                     size=args['dim'],
                     min_count=args['min_count'],
                     window=args['window'],
                     iter=args['epochs'])
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


def args_to_url(args):
    """ generate model_url from args

    Args:
        args: args dict

    Returns:
        str: model_url for train_w2v_model

    """
    filename = '_'.join([str(x) for x in args.values()]) + '.bin'
    return from_project_root("embedding_model/models/w2v_phrase_" + filename)


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
    for sentence in sentences:
        avg_wv = avg_wv_of_words(wv_url, sentence)
        dvs = np.append(dvs, avg_wv)
    return dvs.reshape(len(sentences), -1)


def main():
    args = {
        'dim': 64,
        'min_count': 2,
        'window': 5,
        'epochs': 1
    }
    model = train_w2v_model(DATA_URL, args_to_url(args), args=args)
    print(len(model.wv.vocab))
    pass


if __name__ == '__main__':
    main()
