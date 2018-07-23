#coding=utf-8
import pickle as pk
import collections
# 结合tf_idf_model中的tf_value 和 bdc_value
def get_tf_bdc_value(tf_pickle,bdc_pickle,word_dict_pickle,tf_bdc_pickle):

    tf_dict = pk.load(open(tf_pickle,'rb'))
    bdc_dict = pk.load(open(bdc_pickle,'rb'))
    word_dict = pk.load(open(word_dict_pickle,'rb'))
    tf_bdc_dict = collections.defaultdict(float)

    for (word,wordid) in word_dict.items():

        tf_bdc_dict[word] = tf_dict[word] * bdc_dict[word]

    #save
    pk.dump(tf_bdc_dict,open(tf_bdc_pickle,'wb'))

def main():
    tf_pickle = "E:\deve-program\pycharm-workplace\dg_text\processed_data\phrase_level_tf.pk"
    bdc_pickle="E:\deve-program\pycharm-workplace\dg_text\processed_data\phrase_level_bdc.pk"
    word_dict_pickle="E:\deve-program\pycharm-workplace\dg_text\processed_data\phrase_level_word_dict.pk"
    tf_bdc_pickle= "../processed_data/phrase_level_tfbdc.pk"
    get_tf_bdc_value(tf_pickle, bdc_pickle, word_dict_pickle, tf_bdc_pickle)
    pass


if __name__ == "__main__":

    main()