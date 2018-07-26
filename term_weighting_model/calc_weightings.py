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


def calc_bdc(data_url=DATA_URL):
    """ calc the bdc value of all tokens

    Args:
        data_url: url to data file

    Returns:
        dict: bdc dict {word: bdc_value}

    """
    bdc_url = from_project_root("processed_data/saved_weight/phrase_level_bdc.json")
    if exists(bdc_url):
        return ju.load(bdc_url)

    labels, sentences = load_raw_data(data_url)
    word_label_dict = collections.defaultdict(dict)  # store f(t, c_i)
    label_words_num = collections.defaultdict(int)  # to store all f(c_i)
    label_set = set()
    for label, sentence in zip(labels, sentences):
        label_set.add(label)
        label_words_num[label] += len(sentence)
        for word in sentence:
            try:
                word_label_dict[word][label] += 1
            except KeyError:
                word_label_dict[word][label] = 1

    bdc_dict = collections.defaultdict(float)
    for word in tqdm(word_label_dict):
        for label in word_label_dict[word]:
            word_label_dict[word][label] /= label_words_num[label]  # p(t, c_i) = f(t, c_i) / f(c_i)
        values_array = np.array(list(word_label_dict[word].values()))  # p(t, c_i) for all labels
        values_array = values_array / values_array.sum()  # p(t, c_i) / sum(p(t, c_i))
        values_array = np.log(values_array) * values_array
        bdc_dict[word] = 1 + values_array.sum() / np.log(len(label_words_num))  # norm

    # to save calculated result
    ju.dump(bdc_dict, bdc_url)
    return bdc_dict


def main():
    pass


if __name__ == '__main__':
    main()
