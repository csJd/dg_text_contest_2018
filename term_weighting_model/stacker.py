# coding: utf-8
# created by deng on 8/5/2018

from itertools import product
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score
from sklearn.svm import LinearSVC
from tqdm import tqdm
import scipy as sp
import numpy as np

from utils.path_util import from_project_root
from term_weighting_model.transformer import generate_vectors


def load_params():
    """ load params

    Returns:
        list, list of params dict

    """

    params_grad = [
        {
            'column': ['word_seg', 'article'],
            'max_n': [1],
            'min_df': [2, 3],
            'max_df': [0.8, 0.9],
            'max_features': [200000, 300000],
            'balanced': [False, True],
            're_weight': [0, 9]
        },
        {
            'column': ['word_seg', 'article'],
            'max_n': [2],
            'min_df': [2, 3],
            'max_df': [0.8, 0.9],
            'max_features': [500000, 1000000, 2000000],
            'balanced': [False, True],
            're_weight': [0, 9]
        },
        {
            'column': ['word_seg', 'article'],
            'max_n': [3],
            'min_df': [2, 3],
            'max_df': [0.8, 0.9],
            'max_features': [1000000, 2000000, 3000000, 4000000],
            'balanced': [False, True],
            're_weight': [0, 9]
        },
    ]

    params_list = list()
    for params_dict in params_grad:
        keys, value_lists = zip(*(params_dict.items()))
        for prod in product(*value_lists):
            params_list.append(dict(zip(keys, prod)))
    print(len(params_list))

    return params_list


def feature_stacking(cv=5, random_state=None):
    # do something

    clf = LinearSVC()
    train_url = from_project_root("data/train_set.csv")
    # test_url = from_project_root("data/test_set.csv")
    test_url = None
    X_stack_train = np.array([])
    X_stack_test = np.array([])
    params_list = load_params()
    for i, params in enumerate(tqdm(params_list)):
        X, y, X_test = generate_vectors(train_url, test_url, **params)
        if not sp.sparse.isspmatrix_csr(X):
            X = sp.sparse.csr_matrix(X)
        skf = StratifiedKFold(n_splits=cv, random_state=random_state)
        y_pred = np.zeros(y.shape)
        for train_index, cv_index in skf.split(X, y):
            X_train, X_cv = X[train_index], X[cv_index]
            y_train, y_cv = y[train_index], y[cv_index]
            clf.fit(X_train, y_train)
            y_pred[cv_index] = clf.predict(X_cv)
            print("cv macro f1: ", f1_score(y_cv, y_pred[cv_index], average='macro'))

        print("macro f1: ", f1_score(y, y_pred, average='macro'))
        # break  # break for test
        X_stack_train = np.append(X_stack_train, y_pred)

        y_pred_test = clf.predict(X_test)
        X_stack_test = np.append(X_stack_test, y_pred_test)


def main():
    # load_params()
    feature_stacking()


if __name__ == '__main__':
    main()
