# coding: utf-8
# created by deng on 7/24/2018


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


def main():
    pass


if __name__ == '__main__':
    main()
