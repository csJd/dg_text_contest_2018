# coding: utf-8
# created by deng on 7/25/2018

import fasttext as ft
from utils.data_util import load_raw_data
from utils.path_util import from_project_root, exists, basename
from time import time

from sklearn.metrics import f1_score, accuracy_score

# Define some url
TRAIN_URL = from_project_root("processed_data/phrase_level_data_train.csv")
DEV_URL = from_project_root("processed_data/phrase_level_data_dev.csv")

# Define some static args for ft model
FT_LABEL_PREFIX = '__label__'
THREAD = 4


def ft_process(data_url):
    """ process data into what ft model need, and save it into './processed_data' dir

    Args:
        data_url: url to original .csv data

    Returns:
        str: url to saved processed data

    """
    save_filename = basename(data_url).replace('.csv', '_ft.csv')
    save_url = from_project_root("embedding_model/processed_data/" + save_filename)

    # file specified by data_url is already processed
    if exists(save_url):
        return save_url

    with open(save_url, "w", encoding='utf-8') as ft_file:
        labels, sentences = load_raw_data(data_url)
        for i in range(len(labels)):
            label = FT_LABEL_PREFIX + str(labels[i])
            sentence = ' '.join(sentences[i])
            ft_file.write('{} {}\n'.format(label, sentence))
    return save_url


def ft_train(data_url, model_url, args):
    """ load the ft model or train a new one

    Args:
        data_url: train data url
        model_url: url to save trained model
        args: args for model

    Returns:
        ft model

    """
    if not data_url.endswith('_ft.csv'):
        data_url = ft_process(data_url)

    # model specified by model_url is already trained and saved
    if exists(model_url):
        return ft.load_model(model_url, label_prefix=FT_LABEL_PREFIX)

    print("fasttext model is training, model will be saved at\n ", model_url)
    model_url = model_url[:-4]  # ft will add .bin automatically

    '''
    List of available params and their default value:

        input_file     			training file path (required)
        output         			output file path (required)
        label_prefix   			label prefix ['__label__']
        lr             			learning rate [0.1]
        lr_update_rate 			change the rate of updates for the learning rate [100]
        dim            			size of word vectors [100]
        ws             			size of the context window [5]
        epoch          			number of epochs [5]
        min_count      			minimal number of word occurences [1]
        neg            			number of negatives sampled [5]
        word_ngrams    			max length of word ngram [1]
        loss           			loss function {ns, hs, softmax} [softmax]
        bucket         			number of buckets [0]
        minn           			min length of char ngram [0]
        maxn           			max length of char ngram [0]
        thread         			number of threads [12]
        t              			sampling threshold [0.0001]
        silent         			disable the log output from the C++ extension [1]
        encoding       			specify input_file encoding [utf-8]
        pretrained_vectors		pretrained word vectors (.vec file) for supervised learning []
        
    '''

    s_time = time()
    clf = ft.supervised(data_url, model_url, thread=THREAD, label_prefix=FT_LABEL_PREFIX,
                        lr=args['lr'],
                        dim=args['dim'],
                        ws=args['ws'],
                        epoch=args['epochs'],
                        word_ngrams=args['ngram'])
    e_time = time()
    print("training finished in %.3f second\n" % (e_time - s_time))
    return clf


def args_to_url(args):
    """ generate model_url from args

    Args:
        args: args dict

    Returns:
        str: model_url for tf_train

    """
    filename = '_'.join([str(x) for x in args.values()]) + '.bin'
    return from_project_root("embedding_model/models/ft_phrase_" + filename)


def print_model_details(clf):
    """ print details of ft model

    Args:
        clf: ft model

    """
    print(" labels_cnt  :", len(clf.labels))
    print(" vector dim  :", clf.dim)
    print(" window size :", clf.ws)
    print(" epochs      :", clf.epoch)
    print(" max ngram   :", clf.word_ngrams)

    labels, sentences = load_raw_data(DEV_URL)
    sentences = [' '.join(sentence) for sentence in sentences]
    # ft predicted label is a list
    pred_labels = [int(label[0]) for label in clf.predict(sentences)]
    macro_f1 = f1_score(labels, pred_labels, average='macro')
    acc = accuracy_score(labels, pred_labels)

    print(" macro-f1    :", macro_f1)
    print(" accuracy    :", acc)


def main():
    args = {
        'lr': 0.1,
        'dim': 100,
        'ws': 5,
        'epochs': 10,
        'ngram': 1
    }
    model_url = args_to_url(args)
    clf = ft_train(TRAIN_URL, model_url, args)
    print_model_details(clf)

    pass


if __name__ == '__main__':
    main()
