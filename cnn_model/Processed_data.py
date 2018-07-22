# coding: utf-8
# created by Pengpeng on 2018/7/22
import pickle as pk

# 根据各种权重，过滤语料中的词语
def filter_words(tf_pickle,bdc_pickle,tf_bdc_picle,vocab_pickle):
    """
    Args:
        tf_pickle: tf文件路径
        bdc_pickle:  bdc文件路径
        tf_bdc_picle: tf_bdc文件路径
        vocab_pickle: 词典

    Returns:

    """
    # 加载权重文件
    tf_dict = pk.load(open(tf_pickle,'rb'))
    bdc_dict = pk.load(open(bdc_pickle,'rb'))
    tf_bdc_dict = pk.load(open(tf_bdc_picle,'rb'))
    vocab_dict = pk.load(open(vocab_pickle,'rb'))


    # 过滤条件 = 查询
    del_count = 0
    for (word,word_id) in vocab_dict.items():

        # 出现过一次
        if tf_dict[word] == 3 and bdc_dict[word] == 1 :
            del_count += 1

    print(del_count)

def main():

    tf_pickle = "E:\deve-program\pycharm-workplace\dg_text\processed_data\phrase_level_tf.pk"
    bdc_pickle = "E:\deve-program\pycharm-workplace\dg_text\processed_data\phrase_level_bdcValue.pk"
    tf_bdc_picle = "E:\deve-program\pycharm-workplace\dg_text\processed_data\phrase_level_tfbdc.pk"
    vocab_pickle = "E:\deve-program\pycharm-workplace\dg_text\processed_data\phrase_level_word_dict.pk"
    filter_words(tf_pickle, bdc_pickle, tf_bdc_picle, vocab_pickle)


if __name__ == '__main__':
    main()