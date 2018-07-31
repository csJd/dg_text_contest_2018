# coding: utf-8
# created by deng on 7/31/2018

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from utils.path_util import from_project_root


def main():
    column = "word_seg"
    train = pd.read_csv(from_project_root('data/train_set.csv'))
    test = pd.read_csv(from_project_root('data/test_set.csv'))
    vec = TfidfVectorizer(ngram_range=(1, 3), min_df=3, max_df=0.8, max_features=2000000, sublinear_tf=1)
    trn_term_doc = vec.fit_transform(train[column])
    test_term_doc = vec.transform(test[column])

    y = (train["class"]).astype(int)
    lin_clf = svm.LinearSVC()
    lin_clf.fit(trn_term_doc, y)
    preds = lin_clf.predict(test_term_doc)

    result_file = open('processed_data/com_result/baseline_tfidf_3gram_2000000.csv', 'w')
    result_file.write("id,class" + "\n")
    for i, label in enumerate(preds):
        result_file.write(str(i) + "," + str(label) + "\n")
    result_file.close()
    pass


if __name__ == '__main__':
    main()
