#coding=utf-8

'''
    本py文件，数据集MSRP预处理
'''
import os
import pickle
import numpy as np
from nltk.tokenize import WordPunctTokenizer
from collections import defaultdict

#使用nltk分词
word_tokenizer = WordPunctTokenizer()
import nltk
from nltk.corpus import stopwords
import nltk.stem

def pre_process_sent(sen):
    # 分词
    words = word_tokenizer.tokenize(sen)

    # 去除非英文字符
    english_tokens = [w for w in words if (w.isalpha())]  # 开头不是英文字符的就直接去除
    del words

    # 去除停用词
    # filtered_tokens = [w for w in english_tokens if (w not in stopwords.words('english'))]
    # del english_tokens

    # 对词进行词干化 stem
    # stem_words = []
    # stemmer = nltk.stem.SnowballStemmer('english')
    # #
    # for token in filtered_tokens:
    #     new_token = stemmer.stem(token)
    #     stem_words.append(new_token)
    #
    # del filtered_tokens

    return english_tokens


def build_vocab(vocab_path,msr_paths):

    if os.path.exists(vocab_path): #如果加载完
        vocab_file = open(vocab_path,'rb')
        vocab = pickle.load(vocab_file)
        print("load vocab finish!!")
    else:
        #记录单词的频率
        word_freq = defaultdict(int)

        for msrp_path in msr_paths:
            #读取数据集，并进行分词，统计每个单词的频率，save 在 word_freq
            with open(msrp_path,'r',encoding='utf-8') as f:

                line_count = 0
                for line in f:
                    line_count += 1
                    if line_count == 1:
                        continue

                    print("已经处理{}条数据".format(line_count))

                    #读取每一行的两个句子。
                    line_arr = line.strip().split("\t")
                    sent1 = line_arr[3]
                    sent2 = line_arr[4]
                    sen = sent1 + " " + sent2

                    stem_words = pre_process_sent(sen)

                    for stem_word in stem_words:

                        word_freq[stem_word] += 1

            f.close()
        print("load finished")

        #构建vocabuary,并将出现次数小于5的单词全部去除，视为UNKNOW
        vocab = {}
        i = 1
        for word,freq in word_freq.items():
                vocab[word] = i
                i += 1

        #保存词汇表
        with open(vocab_path,"wb") as g:
            pickle.dump(vocab,g)
            print(len(vocab))
            print("vocab save finished")

    return vocab


def load_dataset(msr_paths,msr_data_path,vocab_path,max_word_in_sent):
    '''
    :param msr_paths: msr_paths[0] : train_file_path and msr_paths[1] : test_file_path
    :param msr_data_path:
    :param vocab_path:
    :return:
    '''
    if not os.path.exists(msr_data_path):

        vocab = build_vocab(vocab_path,msr_paths)
        UNKNOWN = 0


        train_data_x1 = []
        train_data_x2 = []
        train_y = []

        test_data_x1 = []
        test_data_x2 = []
        test_y = []

        #构造训练集
        for i in range(len(msr_paths)):
            with open(msr_paths[i],'r',encoding='utf-8') as f:
                line_count = 0
                for line_index,line in enumerate(f):

                    line_count += 1
                    if line_count == 1:
                        continue

                    line_arr = line.strip().split("\t")
                    text1 = line_arr[3]
                    text2 = line_arr[4]
                    y = line_arr[0]

                    #分词
                    text1_arr = pre_process_sent(text1)
                    text2_arr = pre_process_sent(text2)

                    word_to_index_x1 = np.zeros([max_word_in_sent],dtype=int)
                    for j,word in enumerate(text1_arr):
                        word_to_index_x1[j] = vocab.get(word,UNKNOWN)

                    word_to_index_x2 = np.zeros([max_word_in_sent],dtype=int)
                    for j,word in enumerate(text2_arr):
                        word_to_index_x2[j] = vocab.get(word,UNKNOWN)

                    if y == '0':
                        label = [1,0]
                    elif y == '1':
                        label = [0,1]

                    if i == 0 : #train_data
                        train_data_x1.append(word_to_index_x1)
                        train_data_x2.append(word_to_index_x2)
                        train_y.append(label)
                    else:
                        test_data_x1.append(word_to_index_x1)
                        test_data_x2.append(word_to_index_x2)
                        test_y.append(label)


        #save the train/test data
        pickle.dump((train_data_x1,train_data_x2,train_y,test_data_x1,test_data_x2,test_y),open(msr_data_path,'wb'))


    else:
        data_file = open(msr_data_path,'rb')
        train_data_x1,train_data_x2,train_y,test_data_x1,test_data_x2,test_y = pickle.load(data_file)


    return train_data_x1,train_data_x2,train_y,test_data_x1,test_data_x2,test_y

def get_embedding_mat(google_embedding_path,init_embedding_path,vocab_path,msr_paths):

    if os.path.exists(init_embedding_path):

        init_embedding_mat = np.loadtxt(init_embedding_path,dtype=np.float32)

        return init_embedding_mat

    else:
        vocab = build_vocab(vocab_path,msr_paths)
        i = 0
        word_vecs = {}
        with open(google_embedding_path, 'rb') as f:
            header = f.readline()  # 读取头部
            # print('header',header)
            vocab_size, layer_size = map(int, header.split())  #
            print('vocalsize: ', vocab_size, 'layer_size: ', layer_size)

            # 计算一个300维度np.float32的向量占有的字符个数
            binary_len = np.dtype('float32').itemsize * layer_size
            # 一个单词 有两行表示： 第一行为 : word  第二行 ： 300维的vec
            for line in range(vocab_size):
                word = []
                while True:
                    ch = f.read(1)  # 每次读取1个字符。
                    # print ch
                    if ch == b' ':  # 如果是一个空字符，说明此行后面的内容是一个300维度vec
                        word = ''.join(word)
                        break

                    if ch != b'\n':
                        word.append(ch.decode('raw_unicode_escape'))  # 将byte类型j进行反编码
                        # print(word)

                # print word
                if word in vocab:
                    i = i + 1
                    # np.fromstring(f.read)
                    word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')

                    # 若所需要的词向量刚好都在 GoogleNew-vec中查找到，则直接退出vecs的寻找。
                    print('word' + str(i), word)
                    # if i == len(vocab):
                    #     break
                else:
                    f.read(binary_len)  # 如果word不在vocab里面，更新当前指针

            f.close()

            #对于在google_embedding中没有找到的单词,随机赋值
            for word in vocab:

                if word not in word_vecs:
                    word_vecs[word] = np.random.uniform(-0.25,0.25,300)

            #save to local file
            vecs = []
            #UNKNOW word
            vecs.append(np.random.uniform(-0.25,0.25,300))
            for word in vocab:
                vecs.append(word_vecs[word])

            np.savetxt(init_embedding_path,vecs)

            return word_vecs


if __name__ =="__main__":

#===========加载训练集，测试集================================
    # vocab_path = "./pre_process_msrp/msr_paraphrase_vocab.pickle"
    # msr_paths = ["../dataset/MSRP/msr_paraphrase_train.txt","../dataset/MSRP/msr_paraphrase_test.txt"]
    # msr_data_path = "./pre_process_msrp/msr_paraphrase_data.pickle"
    # build_vocab(vocab_path,msr_paths)
    # train_data_x1, train_data_x2, train_y, test_data_x1, test_data_x2, test_y=load_dataset(msr_paths,msr_data_path,vocab_path,35)

#==========根据vocab 加载 google_embedding=========
    # google_embedding_path = "../static_dataset/GoogleNews-vectors-negative300.bin"
    # init_embedding_path = "./pre_process_msrp/init_embedding_mat.txt"
    #
    # embedding_mat = get_embedding_mat(google_embedding_path,init_embedding_path,vocab_path)
    # print(embedding_mat)
# ==========根据vocab 加载 google_embedding=========

#=======================测试分词效果===============================================
    sen1 = "Both studies are published in the Journal of the American Medical Association"
    sen2 = "The study appears in the latest issue of the Journal of the American Medical Association."

    sen1_arr = pre_process_sent(sen1)
    print(sen1_arr)

    sen2_arr = pre_process_sent(sen2)
    print(sen2_arr)

#=======================测试分词效果===============================================