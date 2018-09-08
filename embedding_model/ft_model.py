# coding: utf-8
# created by deng on 7/25/2018

import fasttext as ft
from time import time
from sklearn.metrics import f1_score, accuracy_score
from collections import OrderedDict
from sklearn.model_selection import StratifiedKFold
from sklearn.externals import joblib
import numpy as np
import tempfile

from utils.data_util import load_raw_data, load_to_df
from utils.path_util import from_project_root, exists, basename

# Define some url
TRAIN_URL = from_project_root("data/train_set.csv")
TEST_URL = from_project_root("data/test_set.csv")
N_CLASSES = 19

# Define some static args for ft model
FT_LABEL_PREFIX = '__label__'
N_JOBS = 4


def ft_process(data_url=None):
    """ process data into what ft model need, and save it into './processed_data' dir

    Args:
        data_url: url to original .csv data

    Returns:
        str: url to saved processed data

    """
    save_filename = basename(data_url).replace('.csv', '_ft.csv')
    save_url = from_project_root("embedding_model/processed_data/" + save_filename)

    # file specified by data_url is already processed
    if exists(save_url):
        return save_url
    if data_url is not None:
        labels, sentences = load_raw_data(data_url)
    else:
        train_df = load_to_df(TRAIN_URL)
        labels = train_df['class'].values
        sentences = train_df['word_seg']

    with open(save_url, "w", encoding='utf-8', newline='\n') as ft_file:
        for i in range(len(labels)):
            label = FT_LABEL_PREFIX + str(labels[i])
            sentence = ' '.join(sentences[i])
            ft_file.write('{} {}\n'.format(label, sentence))
    return save_url


def train_ft_model(args, data_url=None):
    """ load the ft model or train a new one

    Args:
        data_url: train data url
        args: args for model
        validation: do validation or not

    Returns:
        ft model

    """
    if data_url is None or not data_url.endswith('_ft.csv'):
        data_url = ft_process(data_url)

    # model specified by model_url is already trained and saved
    model_url = args_to_url(data_url, args)
    if exists(model_url):
        return ft.load_model(model_url, label_prefix=FT_LABEL_PREFIX)

    print("fasttext model is training, model will be saved at\n ", model_url)
    model_url = model_url[:-4]  # ft will add .bin automatically

    '''
    List of available params and their default value:

        input_file     			training file path (required)
        output         			output file path (required)
        label_prefix   			label prefix ['__label__']
        lr             			learning rate [0.1]
        lr_update_rate 			change the rate of updates for the learning rate [100]
        dim            			size of word vectors [100]
        ws             			size of the context window [5]
        epoch          			number of epochs [5]
        min_count      			minimal number of word occurences [1]
        neg            			number of negatives sampled [5]
        word_ngrams    			max length of word ngram [1]
        loss           			loss function {ns, hs, softmax} [softmax]
        bucket         			number of buckets [0]
        minn           			min length of char ngram [0]
        maxn           			max length of char ngram [0]
        thread         			number of threads [12]
        t              			sampling threshold [0.0001]
        silent         			disable the log output from the C++ extension [1]
        encoding       			specify input_file encoding [utf-8]
        pretrained_vectors		pretrained word vectors (.vec file) for supervised learning []
        
    '''

    s_time = time()
    clf = ft.supervised(data_url, model_url, thread=N_JOBS, label_prefix=FT_LABEL_PREFIX, **args)
    e_time = time()
    print("training finished in %.3f second\n" % (e_time - s_time))
    return clf


def args_to_url(data_url, args):
    """ generate model_url from args

    Args:
        data_url: for ft_model
        args: args dict

    Returns:
        str: model_url for train_ft_model

    """
    level = ['phrase'] if 'phrase' in data_url else ['word']
    filename = '_'.join(level + [str(x) for x in OrderedDict(args).values()]) + '.bin'
    return from_project_root("embedding_model/models/ft_" + filename)


def print_model_details(clf):
    """ print details of ft model

    Args:
        clf: ft model

    """
    print(" labels_cnt  :", len(clf.labels))
    print(" vector dim  :", clf.dim)
    print(" window size :", clf.ws)
    print(" epochs      :", clf.epoch)
    print(" max ngram   :", clf.word_ngrams)

    labels, sentences = load_raw_data()
    sentences = [' '.join(sentence) for sentence in sentences]
    # ft predicted label is a list
    pred_labels = [int(label[0]) for label in clf.predict(sentences)]
    macro_f1 = f1_score(labels, pred_labels, average='macro')
    acc = accuracy_score(labels, pred_labels)

    print(" macro-f1    :", macro_f1)
    print(" accuracy    :", acc)


def gen_data_for_stacking(args, column='word_seg', n_splits=5, random_state=None):
    """

    Args:
        args:
        column:
        n_splits:
        random_state:

    Returns:

    """

    train_df = load_to_df(TRAIN_URL)
    y = train_df['class'].values
    X = train_df[column]
    X_test = load_to_df(TEST_URL)[column]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=bool(random_state), random_state=random_state)
    y_pred = np.zeros((X.shape[0],))  # for printing score of each fold
    y_pred_proba = np.zeros((X.shape[0], N_CLASSES))
    y_test_pred_proba = np.zeros((X_test.shape[0], N_CLASSES))

    with tempfile.NamedTemporaryFile() as t_file:
        for ind, (train_index, cv_index) in enumerate(skf.split(X, y)):  # cv split
            X_train, X_cv = X[train_index], X[cv_index]
            y_train, y_cv = y[train_index], y[cv_index]

            with open(t_file.name, "w", encoding='utf-8', newline='\n') as ft_file:
                for i in range(len(y_train)):
                    label = FT_LABEL_PREFIX + str(y_train[i])
                    ft_file.write('{} {}\n'.format(label, X_train[i]))
            clf = ft.supervised(t_file.name, output=None, thread=N_JOBS, label_prefix=FT_LABEL_PREFIX, **args)
            y_pred[cv_index] = [int(label[0]) for label in clf.predict(X_cv)]
            y_pred_proba[cv_index] = [[t[1] for t in sorted(proba, key=lambda x: int(x[0]))]
                                      for proba in clf.predict_proba(X_cv, N_CLASSES)]

            if ind == 0:
                print(sorted(proba, key=lambda x: int(x[0]))
                      for proba in clf.predict_proba(X_cv[0], N_CLASSES))

            print("%d/%d cv macro f1 :" % (ind + 1, n_splits),
                  f1_score(y_cv, y_pred[cv_index], average='macro'))
            y_test_pred_proba += [[t[1] for t in sorted(proba, key=lambda x: int(x[0]))]
                                  for proba in clf.predict_proba(X_test, N_CLASSES)]

    print("macro f1:", f1_score(y, y_pred, average='macro'))  # calc macro_f1 score
    y_test_pred_proba /= n_splits  # normalize to 1
    return y_pred_proba, y, y_test_pred_proba


def main():
    args = {
        'lr': 0.01,
        'dim': 300,
        'ws': 5,
        'epoch': 10,
        'word_ngrams': 5
    }
    # clf = train_ft_model(args, TRAIN_URL)
    # print_model_details(clf)
    joblib.dump(gen_data_for_stacking(args), from_project_root("ft_300.pk"))
    pass


if __name__ == '__main__':
    main()
