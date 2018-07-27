# coding: utf-8
# created by deng on 7/27/2018

from utils.path_util import from_project_root
import utils.json_util as ju

DATA_URL = from_project_root("processed_data/phrase_level_data_train.csv")

BDC_DICT = ju.load(from_project_root("processed_data/saved_weight/phrase_level_bdc.json"))
DC_DICT = ju.load(from_project_root("processed_data/saved_weight/phrase_level_dc.json"))
TF_DICT = ju.load(from_project_root("processed_data/saved_weight/phrase_level_tf.json"))


def filtered_by_dict(word, dic=BDC_DICT, lower=5., upper=1.e5):
    """ filtering words according their tf

    Args:
        word: the sentence to process
        dic: the dict to use
        lower: lower bound
        upper: upper bound

    Returns:
        bool: True if the word should be filtered

    """
    return dic[word] < lower or dic[word] > upper


def process_data(data_url=DATA_URL, save_url=None):
    """ process data according to specific rules

    Args:
        data_url: url to data file
        save_url: url to save processed data

    """
    if save_url is None:
        save_url = data_url.replace('.csv', '_tw_precessed.csv')
    print("processing data at %s ... \n processed data will be saved at %s" % (data_url, save_url))
    with open(data_url, 'r', encoding='utf-8') as data_file, \
            open(save_url, 'w', encoding='utf-8', newline='\n') as save_file:
        for line in data_file:
            label, sentence = line.split(',')
            filtered_sentence = list()
            for word in sentence.split():
                # only keep word whose bdc value higher than 0.5
                if filtered_by_dict(word, dic=BDC_DICT, lower=0.5):
                    continue

                # only keep word whose tf value in (5, 1e5)
                if filtered_by_dict(word, dic=TF_DICT, lower=5, upper=1e5):
                    continue
                filtered_sentence.append(word)

                # truncate sentence if it longer than 1500 words
                if len(filtered_sentence) > 1500:
                    break

            filtered_sentence = ' '.join(filtered_sentence)
            save_file.write("{},{}\n".format(label, filtered_sentence))
    print("finish processing")


def main():
    process_data()
    pass


if __name__ == '__main__':
    main()
