# coding: utf-8
# created by deng on 7/24/2018

from sklearn.model_selection import train_test_split
from utils.path_util import from_project_root
import pre_process.data_util as pdu


def load_raw_data(data_url):
    """ load data to get labels list and sentences list

    Args:
        data_url: url to data file

    Returns:
        (list, list): labels and sentences

    """
    with open(data_url, "r", encoding="utf-8") as data_file:
        lines = data_file.readlines()
        labels = list()
        sentences = list()
        for line in lines:
            line = line.split(',')
            labels.append(int(line[0]))
            sentences.append(line[1].split())
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
    """ generate phrase level data

    Args:
        data_url: original data url

    """
    # processed_data/phrase_level_data.csv
    phrase_data_url = from_project_root("processed_data/phrase_level_data.csv")
    pdu.extract_data(data_url, [3, 1], phrase_data_url)


def main():
    data_url = from_project_root("data/train_set.csv")
    gen_phrase_data(data_url)
    # train_data_url = from_project_root("processed_data/phrase_level_data_train.csv")
    # labels, sentences = load_raw_data(train_data_url)
    pass


if __name__ == '__main__':
    main()
