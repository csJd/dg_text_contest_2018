# coding: utf-8
# created by Pengpeng on 2018/7/22
import collections

# 查询各个类别拥有的样本数
def check_label_count(train_data_url):

    label_dict = collections.defaultdict(int)
    with open(train_data_url,'r',encoding='utf-8') as f:
        for line in f.readlines()[1:]:
            line_list = line.strip().split(',')
            label = line_list[3]
            label_dict[label] += 1
    f.close()

    for (label,count) in label_dict.items():

        print("{}:{}".format(label,count))

def main():

    train_data_url = "../data/train_set.csv"
    check_label_count(train_data_url)

if __name__ == '__main__':
    main()