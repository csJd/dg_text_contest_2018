# coding: utf-8
# created by deng on 7/24/2018

from sklearn.model_selection import train_test_split
from utils.path_util import from_project_root
import pre_process.data_util as pdu


WORD_LEVEL_DATA_URL = from_project_root("")
PHRASE_LEVEL_DATA_URL = from_project_root("")
WEIGHTINGS = ['tf', 'idf', 'lf', 'df', 'bdc']


def load_raw_data(data_url, sentence_to_list=False):
    """ load data to get labels list and sentences list

    Args:
        data_url: url to data file
        sentence_to_list: whether transform sentence to word list

    Returns:
        (list, list): labels and sentences

    """
    with open(data_url, "r", encoding="utf-8") as data_file:
        labels = list()
        sentences = list()
        print("loading data from \n ", data_url)
        for line in data_file:
            line = line.split(',')
            labels.append(int(line[0]))
            if sentence_to_list:
                sentences.append(line[1].split())
            else:
                sentences.append(line[1])
        print("finished loading\n")
    return labels, sentences


def train_dev_split(data_url, dev_size=0.2, including_header=False):
    """ split .csv data into train data and dev data and save them in the same dir

    Args:
        data_url: url to original data
        dev_size: the ratio of dev_data in data
        including_header: if the data file including header

    """
    print("splitting data file %s" % data_url)
    train_url = data_url[:-4] + '_train.csv'
    dev_url = data_url[:-4] + '_dev.csv'
    with open(data_url, "r", encoding='utf-8') as data_file, \
            open(train_url, "w", encoding='utf-8', newline='\n') as train_file, \
            open(dev_url, "w", encoding='utf-8', newline='\n') as dev_file:
        lines = data_file.readlines()
        if including_header:
            lines = lines[1:]

        # use specific random_state to generate the same data all the time
        train_lines, dev_lines = train_test_split(lines, test_size=dev_size, random_state=233)
        train_file.writelines(train_lines)
        dev_file.writelines(dev_lines)
        print("finished splitting data(%d samples) into train_data(%d samples) and dev_data(%d samples)"
              % (len(lines), len(train_lines), len(dev_lines)))


def gen_phrase_data(data_url):
    """ generate phrase level data (word_seg)

    Args:
        data_url: original data url

    """
    # generate processed_data/phrase_level_data.csv
    phrase_data_url = from_project_root("processed_data/phrase_level_data.csv")
    pdu.extract_data(data_url, 'word_seg', phrase_data_url)


def gen_word_data(data_url):
    """ generate word level data (article)

    Args:
        data_url: original data url

    """
    # generate processed_data/word_level_data.csv
    word_data_url = from_project_root("processed_data/word_level_data.csv")
    pdu.extract_data(data_url, 'article', word_data_url)


def gen_term_weighting(level='phrase', weightings=WEIGHTINGS):
    """ to generate

    Args:
        level: choose which level term weightings to generate
        weightings: which kinds of weightings to generate

    """


def main():
    data_url = from_project_root("data/train_set.csv")
    gen_phrase_data(data_url)
    # train_data_url = from_project_root("processed_data/phrase_level_data_train.csv")
    # labels, sentences = load_raw_data(train_data_url)
    pass


if __name__ == '__main__':
    main()
