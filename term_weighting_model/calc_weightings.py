# coding: utf-8
# created by deng on 7/26/2018

from utils.path_util import from_project_root
from utils.data_util import load_raw_data
from tqdm import tqdm
from os.path import exists
import utils.json_util as ju
import numpy as np
import collections

DATA_URL = from_project_root("processed_data/phrase_level_data.csv")


def calc_tf(data_url=DATA_URL, update=False):
    """ calc the tf value of all tokens

    Args:
        data_url: url to data file
        update: update dict even it exists

    Returns:
        dict: tf dict {word: tf_value}

    """
    level = 'phrase' if 'phrase' in data_url else 'word'
    tf_url = from_project_root("processed_data/saved_weight/{}_level_tf.json".format(level))
    if not update and exists(tf_url):
        return ju.load(tf_url)

    tf_dict = collections.defaultdict(int)
    _, sentences = load_raw_data(data_url)
    for sentence in tqdm(sentences):
        for word in sentence:
            tf_dict[word] += 1

    ju.dump(ju.sort_dict_by_value(tf_dict, reverse=True), tf_url)


def calc_dc(data_url=DATA_URL, update=False):
    """ calc the dc value of all tokens

    Args:
        data_url: url to data file
        update: update dict even it exists

    Returns:
        dict: dc dict {word: dc_value}

    """
    level = 'phrase' if 'phrase' in data_url else 'word'
    dc_url = from_project_root("processed_data/saved_weight/{}_level_dc.json".format(level))
    if not update and exists(dc_url):
        return ju.load(dc_url)
    calc_bdc(DATA_URL, update=True)
    return ju.load(dc_url)


def calc_bdc(data_url=DATA_URL, update=False):
    """ calc the bdc value of all tokens

    Args:
        data_url: url to data file
        update: update dict even it exists

    Returns:
        dict: bdc dict {word: bdc_value}

    """
    level = 'phrase' if 'phrase' in data_url else 'word'
    bdc_url = from_project_root("processed_data/saved_weight/{}_level_bdc.json".format(level))
    dc_url = from_project_root("processed_data/saved_weight/{}_level_dc.json".format(level))
    if not update and exists(bdc_url):
        return ju.load(bdc_url)

    labels, sentences = load_raw_data(data_url)
    word_label_dict = collections.defaultdict(dict)  # store f(t, c_i)
    label_words_num = collections.defaultdict(int)  # to store all f(c_i)
    for label, sentence in tqdm(zip(labels, sentences), total=len(labels)):
        label_words_num[label] += len(sentence)
        for word in sentence:
            try:
                word_label_dict[word][label] += 1
            except KeyError:
                word_label_dict[word][label] = 1

    bdc_dict = collections.defaultdict(float)
    dc_dict = collections.defaultdict(float)
    for word in tqdm(word_label_dict):

        # for calc dc
        arr = np.array(list(word_label_dict[word].values()))  # f(t, c_i) for all labels
        arr = arr / arr.sum()  # f(t, c_i) / f(t)
        arr = np.log(arr) * arr
        dc_dict[word] = 1 + arr.sum() / np.log(len(label_words_num))  # norm

        # for calc bdc
        for label in word_label_dict[word]:
            word_label_dict[word][label] /= label_words_num[label]  # p(t, c_i) = f(t, c_i) / f(c_i)
        arr = np.array(list(word_label_dict[word].values()))  # p(t, c_i) for all labels
        arr = arr / arr.sum()  # p(t, c_i) / sum(p(t, c_i))
        arr = np.log(arr) * arr
        bdc_dict[word] = 1 + arr.sum() / np.log(len(label_words_num))  # norm

    # to sort save calculated result
    ju.dump(ju.sort_dict_by_value(bdc_dict), bdc_url)
    ju.dump(ju.sort_dict_by_value(dc_dict), dc_url)
    return bdc_dict


def main():
    calc_tf()
    pass


if __name__ == '__main__':
    main()
