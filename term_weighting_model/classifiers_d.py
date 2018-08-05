# coding: utf-8
# created by deng on 7/27/2018

from xgboost.sklearn import XGBClassifier
from lightgbm.sklearn import LGBMClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib
from sklearn.metrics import f1_score, accuracy_score
from time import time
import pandas as pd

from utils.path_util import from_project_root
from term_weighting_model.transformer import generate_vectors

N_JOBS = -1
CV = 5


def tune_clf(clf, X, y, param_grid):
    """

    Args:
        clf: clf to be tuned
        param_grid: param_grid for GridSearchCV
        X: X for fit
        y: y for fit

    Returns:
        GridSearchCV: fitted clf

    """
    if param_grid is None:
        print("None as param_grid is invalid, the original clf will be returned")
        return clf
    s_time = time()
    clf = GridSearchCV(clf, param_grid, scoring='f1_macro', n_jobs=N_JOBS, cv=CV,
                       error_score=0, verbose=True)
    clf.fit(X, y)
    e_time = time()
    # print cv results
    print("grid_search_cv is done in %.3f seconds" % (e_time - s_time))
    print("mean_test_macro_f1 = %f\n" % clf.cv_results_['mean_test_score'])
    return clf


def init_param_grid(clf=None, clf_type=None):
    """

    Args:
        clf_type: the type of clf

    Returns:
        param_grid for clf

    """
    if isinstance(clf, SVC) or clf_type == 'svc':
        param_grid = [
            {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
            {'C': [1, 10, 100, 1000], 'gamma': [0.01, 0.001], 'kernel': ['rbf']},
        ]
    elif isinstance(clf, LGBMClassifier) or clf_type == 'lgbm':
        param_grid = [
            {'boosting_type': ['gbdt', 'dart', 'rf'], 'learning_rate': [0.1, 0.01]},
        ]
    elif isinstance(clf, XGBClassifier) or clf_type == 'xgb':
        param_grid = [
            {'learning_rate': [0.3, 0.1, 0.01]}
        ]
    elif isinstance(clf, LinearSVC) or clf_type == 'lsvc':
        param_grid = [
            {'C': [1]}
        ]
    else:
        param_grid = None
    return param_grid


def init_linear_clfs():
    """ init linear classifiers for training

    Returns:
        dict, clfs

    """
    clfs = dict()

    # add linearSVC model
    clfs['lsvc'] = LinearSVC()

    # add SGD model
    # clfs['sgd'] = SGDClassifier()

    # add KNN model
    # clfs['knn'] = KNeighborsClassifier()

    return clfs


def init_clfs():
    """ init classifiers to train
    
    Returns:
        dict, clfs

    """
    clfs = dict()

    # Add lgbm model
    clfs['lgbm'] = LGBMClassifier()

    # Add xgb model
    clfs['xgb'] = XGBClassifier()

    # Add svc model
    clfs['svc'] = SVC()

    return clfs


def train_clfs(clfs, X, y, test_size=0.2, tuning=False, random_state=None):
    """ train clfs

    Args:
        clfs: classifiers
        X: data X of shape (samples_num, feature_num)
        y: target y of shape (samples_num,)
        test_size: test_size for train_test_split
        tuning: whether to tune parameters use param_grid_cv
        random_state: random_state for train_test_split

    """

    # split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    print("train data shape", X_train.shape, y_train.shape)
    print("dev data shape  ", X_test.shape, y_test.shape)
    for clf_name in clfs:
        clf = clfs[clf_name]
        if tuning:
            print("grid search cv on %s is running" % clf_name)
            param_grid = init_param_grid(clf)
            clf = tune_clf(clf, X, y, param_grid)
            print('cv_results\n', clf.cv_results_)
            continue

        print("%s model is training" % clf_name)
        s_time = time()
        clf.fit(X_train, y_train)
        e_time = time()
        print(" training finished in %.3f seconds" % (e_time - s_time))

        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average='macro')
        print(" accuracy = %f\n f1_score = %f\n" % (acc, macro_f1))

    return clfs


def train_and_gen_result(clf, X, y, X_test, save_url='result.csv'):
    """ train and generate result with specific clf

    Args:
        clf: classifier
        X: vectorized data
        y: target
        X_test: test data
        save_url: url to save the result file

    """
    clf.fit(X, y)
    pred = clf.predict(X_test)

    result_file = open(save_url, 'w')
    result_file.write("id,class" + "\n")
    for i, label in enumerate(pred):
        result_file.write(str(i) + "," + str(label) + "\n")
    result_file.close()


def main():

    clfs = init_linear_clfs()

    # load from pickle
    # xy_url = from_project_root("processed_data/vector/dc_3gram_4000000_Xy.pk")
    # print("loading data from", xy_url)
    # X, y = joblib.load(xy_url)

    # generate from original csv
    train_url = from_project_root("data/train_set.csv")
    # test_url = from_project_root("data/test_set.csv")
    test_url = None
    X, y, X_test = generate_vectors(train_url, test_url, max_n=3, min_df=3, max_df=0.8,
                                    max_features=4000000, balanced=False)
    train_clfs(clfs, X, y, tuning=True)
    # save_url = from_project_root("processed_data/com_result/result.csv")
    # clf = LinearSVC(C=1)
    # train_and_gen_result(clf, X, y, X_test, save_url)
    pass


if __name__ == '__main__':
    main()
