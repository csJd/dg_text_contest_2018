# coding: utf-8
# created by deng on 8/5/2018

from itertools import product
from collections import Counter
from os.path import exists
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from sklearn.base import clone
from tqdm import tqdm
import scipy as sp
import numpy as np
import pandas as pd

from utils.path_util import from_project_root
from term_weighting_model.transformer import generate_vectors
from utils.data_util import load_to_df

N_CLASSES = 19
N_JOBS = 4
CV = 5


def load_params():
    """ load params

    Returns:
        list, list of params dict

    """

    # params_grad = [
    #     {
    #         'column': ['word_seg', 'article'],
    #         'max_n': [1],
    #         'min_df': [2, 3],
    #         'max_df': [0.8, 0.9],
    #         'max_features': [200000, 300000],
    #         'balanced': [False, True],
    #         're_weight': [0, 9]
    #     },
    #     {
    #         'column': ['word_seg', 'article'],
    #         'max_n': [2],
    #         'min_df': [2, 3],
    #         'max_df': [0.8, 0.9],
    #         'max_features': [500000, 1000000, 2000000],
    #         'balanced': [False, True],
    #         're_weight': [0, 9]
    #     },
    #     {
    #         'column': ['word_seg', 'article'],
    #         'max_n': [3],
    #         'min_df': [2, 3],
    #         'max_df': [0.8, 0.9],
    #         'max_features': [1000000, 2000000, 3000000, 4000000],
    #         'balanced': [False, True],
    #         're_weight': [0, 9]
    #     },
    # ]

    params_grad = [
        {
            'column': ['word_seg'],
            'trans_type': ['dc', 'idf'],
            'max_n': [1],
            'min_df': [2, 3],
            'max_df': [0.8],
            'max_features': [200000],
            'balanced': [False],
            're_weight': [0]
        },
        {
            'column': ['word_seg', 'article'],
            'trans_type': ['dc', 'idf'],
            'max_n': [2],
            'min_df': [3],
            'max_df': [0.8, 0.9],
            'max_features': [200000, 500000],
            'balanced': [False, True],
            're_weight': [6]
        },
        {
            'column': ['word_seg', 'article'],
            'trans_type': ['dc', 'idf'],
            'max_n': [3],
            'min_df': [3],
            'max_df': [0.8],
            'max_features': [4000000],
            'balanced': [False],
            're_weight': [9]
        },
    ]

    params_list = list()
    for params_dict in params_grad:
        keys, value_lists = zip(*(params_dict.items()))
        for prod in product(*value_lists):
            params_list.append(dict(zip(keys, prod)))
    print('len(params_list) =', len(params_list))

    return params_list


def run_parallel(index, train_url, test_url, params, clf, cv, random_state, proba=False):
    """ for run cvs parallel

    Args:
        index: index to know which cv it belongs to
        train_url: train data url
        test_url: teat data url
        params: params for generate_vectors
        clf: classifier
        cv: n_splits for KFold
        random_state: random_state for KFold
        proba: True to predict probabilities of labels instead label

    Returns:

    """

    X, y, X_test = generate_vectors(train_url, test_url, **params)
    if not sp.sparse.isspmatrix_csr(X):
        X = sp.sparse.csr_matrix(X)

    skf = StratifiedKFold(n_splits=cv, random_state=random_state)
    y_pred = np.zeros((X.shape[0], 1))
    y_pred_proba = np.zeros((X.shape[0], N_CLASSES))
    y_test_pred_proba = np.zeros((X_test.shape[0], N_CLASSES))
    for ind, (train_index, cv_index) in enumerate(skf.split(X, y)):
        X_train, X_cv = X[train_index], X[cv_index]
        y_train, y_cv = y[train_index], y[cv_index]
        clf.fit(X_train, y_train)
        y_pred[cv_index] = clf.predict(X_cv)
        y_pred_proba[cv_index] = clf._predict_proba_lr(X_cv)
        print("%d/%d cv macro f1 of params set #%d:" % (ind + 1, cv, index),
              f1_score(y_cv, y_pred[cv_index], average='macro'))
        y_test_pred_proba += clf._predict_proba_lr(X_test)
    print("#%d macro f1: " % index, f1_score(y, y_pred, average='macro'))

    y_test_pred = clf.predict(X_test).reshape(X_test.shape[0], 1)
    y_test_pred_proba /= y_test_pred_proba.sum(axis=1)
    if not proba:
        return index, y_pred, y_test_pred
    return index, y_pred_proba, y_test_pred_proba


def feature_stacking(cv=CV, random_state=None, proba=False):
    """

    Args:
        cv: n_splits for KFold
        random_state: random_state for KFlod
        proba: True to predict probabilities of labels instead label

    Returns:
        X, y, X_test

    """

    clf = LinearSVC()
    train_url = from_project_root("data/train_set.csv")
    test_url = from_project_root("data/test_set.csv")
    # test_url = None
    X, y, X_test = generate_vectors(train_url, test_url)  # for X.shape

    params_list = load_params()
    parallel = joblib.Parallel(n_jobs=N_JOBS, verbose=1)
    rets = parallel(joblib.delayed(run_parallel)(
        ind, train_url, test_url, params, clone(clf), cv, random_state, proba
    ) for ind, params in enumerate(params_list))
    rets = sorted(rets, key=lambda x: x[0])

    X_stack_train = np.empty((X.shape[0], 0), float)
    X_stack_test = np.empty((X_test.shape[0], 0), float)
    for ind, y_pred, y_pred_test in rets:
        X_stack_train = np.append(X_stack_train, y_pred, axis=1)
        X_stack_test = np.append(X_stack_test, y_pred_test, axis=1)

    return X_stack_train, y, X_stack_test


def generate_meta_feature(data_url):
    """ generate meta feature

    Args:
        data_url: url to data

    Returns:
        generated meta DataFrame

    """
    save_url = data_url.replace('.csv', '_meta_df.pk')
    if exists(save_url):
        return joblib.load(save_url)

    data_df = load_to_df(data_url)
    meta_df = pd.DataFrame()

    for level in ('word_seg', 'article'):
        # word num
        meta_df[level + '_num'] = data_df[level].apply(lambda x: len(x.split()))
        # different word num
        meta_df[level + '_unique'] = data_df[level].apply(lambda x: len(set(x.split())))
        # most common word num
        meta_df[[level + '_common', level + '_common_num']] = pd.DataFrame(data_df[level].apply(
            lambda x: Counter(x.split()).most_common(1)[0]).tolist())

    # average phrase len
    meta_df['avg_phrase_len'] = meta_df['article_num'] / meta_df['word_seg_num']

    joblib.dump(meta_df, save_url)
    return meta_df


def main():
    # load_params()
    save_url = from_project_root("processed_data/vector/stacked_proba_XyX_test_%d.pk" % len(load_params()))
    joblib.dump(feature_stacking(proba=True), save_url)


if __name__ == '__main__':
    main()
