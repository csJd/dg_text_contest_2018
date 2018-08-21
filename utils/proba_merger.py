# coding: utf-8
# created by deng on 8/21/2018

import pandas as pd
import numpy as np
from sklearn.externals import joblib

from utils.path_util import from_project_root

N_CLASSES = 19


def pk_to_csv(pk_url):
    """
    Args:
        pk_url: url to pk proba file

    """
    save_url = pk_url.replace('pk', 'csv')
    arr = joblib.load(pk_url)
    proba_df = pd.DataFrame(arr, columns=['class_prob_' + str(i + 1) for i in range(N_CLASSES)])
    proba_df.to_csv(save_url, index_label='id')
    return proba_df


def merge_probas(proba_dict, save_url):
    """  merge proba results

    Args:
        proba_dict: dict, {url: weight}

    """
    sum_df = None
    for url in proba_dict:
        print(url, proba_dict[url])
        proba_df = pd.read_csv(url, index_col='id') if url.endswith('.csv') else pk_to_csv(url)
        if sum_df is None:
            sum_df = proba_df * proba_dict[url]
        else:
            sum_df += proba_df * proba_dict[url]
    result_df = pd.DataFrame(np.argmax(sum_df.values, axis=1) + 1, columns=['class'])
    result_df.to_csv(save_url, index_label='id')


def main():
    proba_dict = {
        from_project_root('processed_data/com_result/34_xgb_proba_787.csv'): 0.3,
        from_project_root('processed_data/com_result/50_meta_xgb_proba_786.csv'): 0.2,
        from_project_root('processed_data/com_result/prob_rcnnon_cv0.789744.csv'): 0.1,
    }
    save_url = from_project_root('processed_data/result_789_787.csv')
    merge_probas(proba_dict, save_url)
    pass


if __name__ == '__main__':
    main()
