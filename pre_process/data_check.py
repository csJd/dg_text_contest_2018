#coding=utf-8
import collections

# 查询数据集的基本信息
def checkdata(filename):
    """
    :param filename: 数据源文件 ./init_data/train_set.csv
            第0列：文章id  第1列：“字”级别上的表示 第2列：“词”级别上的表示  第3列： label
    :return: 数据集的基本情况，基本情况如下：
            1. 训练集中样本个数 ==> count_line
            2. 类别字典,字典说明 key: label 和 value: 该label样本的个数 ==> label_dict
            3. 句子的长度 ： ==>max_len
            4. 单词字典 : ==>word_dict
    """
    #查询数据集数据条数
    count_line = 0
    label_dict = collections.defaultdict(int)
    doc_max_len = 0 # 文章最大长度
    phrase_dict = [] # 保存独立的词
    with open(filename,'r',encoding='utf-8') as f:

        for line in f.readlines()[1:]:

            line_list = line.strip().split(',')

            label = line_list[3]
            words = line_list[2].strip().split()

            phrase_dict.extend(words)
            phrase_dict = list(set(phrase_dict))

            if doc_max_len < len(words):
                doc_max_len = len(words)
                print("part max len is {}".format(str(doc_max_len)))

            label_dict[label] += 1

    f.close()

    print("训练集中共有训练样本：{}".format(count_line))
    print("类别字典为：{}".format(label_dict))
    print("类别个数为：{}".format(len(label_dict.keys())))
    print("sentence max len : {}".format(str(doc_max_len)))
    print("word dictionary size is {} ".format(str(len(phrase_dict))))

def main():
    train_data_file = "../init_data/train_set.csv"

    checkdata(train_data_file)

if __name__ == "__main__":

    main()