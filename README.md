## 数据集基本情况
    
* 文件基本说明
    * 第0列：文章id  
    * 第1列：“字”级别上的表示 
    * 第2列：“词”级别上的表示  
    * 第3列： label【类别】
       
* 类别统计
    * 类别样本数统计[共19个类别]
        * 1:5375, 2:2901, 3:8313, 4:3824, 5:2369, 6:6888, 7:3038, 8:6972, 9:7675, 10:4963,
        * 11:3571, 12:5326, 13:7907, 14:6740, 15:7511, 16:3220, 17:3094, 18:7066, 19:5524

    * phrase_level训练集
        * 共有 102277个训练样本
        * 训练集中"phrase_leve"上字典维度有 875129 
    
    * word_level训练集
    
## processed_data/数据文件说明
    
* phrase_level_data.csv
    * “phrase”级别的文档表示,分为两列，<label,phrase_doc>  
    * 调用文件../pre_process/data_util.py中方法extract_data
* word_level_data.csv                        
    * "word"级别上的文档表示，分为两列，<label,word_doc>
    * 调用文件../pre_process/data_util.py中方法extract_data(..)
* phrase_level_tf.pk
    * “phrase”级别上词频文件，格式： dict{word:word_frequence}
    * 调用文件../tf_idf_model/tf_idf.py中方法cal_tf(..)
* phrase_level_bdcValue.pk
    * "phrase"级别上词bdc值，格式： dict{word:bdc_value}
    * 调用文件../bdc_model/get_bdc_value.py中方法cal_bdc_value(..)
* phrase_level_idf.pk
    * "phrase"级别上词的逆文档向量 格式：dict{word:idf_value}
    * 调用文件../tf_idf_model/tf_idf.py中方法cal_idf(..)
* phrase_level_tfbdc.pk
    * "phrase"级别上词的tfbdc权重向量，格式: dict{word:tfbdc_value}
    * 调用文件../bdc_model/get_tf_bdc.py中方法get_tf_bdc_value(..)
* phrase_level_lf.pk
    * "phrase"级别上词出现的类别频率，格式： dict{word:lf_value}
    * 调用文件../pre_process/data_util.py中方法 calc_labelCount_per_words(..)
* phrase_level_df.pk
    * "phrase"级别上次出现的文档频率，格式: dict{word:df_value}
    * 调用文件../pre_process/data_util.py中方法cal_document_frequency(..)

      
    
    
## 初步分工

* 分配部分
    * term weight(Important!) +多分类器 【小黑仔，登登】   
    * word_embedding 训练 + doc2vec (LDA)【小黑仔，登登】    
    * deep_learning_model (CNN+RNN) 【鹏鹏，编写各种深度模型】
        【小黑仔，登登，看论文！】
        
        
## Git使用说明
* 登录 Github 添加自己公钥到 Github 账号
  * `git-bash` 运行 `cat ~/.ssh/id_rsa.pub`，复制结果在[此处](https://github.com/settings/keys)选择`New SSH key`粘贴到`Key`输入框，提示文件不存在则先运行`ssh-keygen`一直点回车就行
  
* clone 项目
  * 在保存项目的位置 `git-bash` 运行 `git clone git@github.com:csJd/dg_text.git`，生成的 `dg_text` 为项目文件夹
  
* push 更改
  * 将代码复制到 `dg_text` 文件夹恰当位置，运行以下命令push更改，后续修改了项目文件也是这样push更改：
```sh
git add .
git commit -m "message"
git push
```

* pull 更改
  * pull 其他人对项目文件作更改, 运行 `git pull` 即可
   

## Pycharm 建议设置

* 换行符统一为'\n'
  * File | Settings | Editor | Code Style
    * `Line separator` 选择 `Unix and OS X (\n)`

* 编码统一为`UTF-8`
  * File | Settings | Editor | File Encodings
    * Global 和 Project 都选择 `UTF-8`
    * 下方 Default encoding for properties files也选择 `UTF-8`

* Python Doc 使用 Google 风格
  * File | Settings | Tools | Python Integrated Tools
    * `Docstring format` 选择 `Google`

* 设置Python 默认代码
  * File | Settings | Editor | File and Code Templates
    * 选择 `Python Script` 粘贴以下内容, `${USER}`可换为自己想显示的昵称
    * 可以自己按需修改

```python
# coding: utf-8
# created by ${USER} on ${DATE}


def main():
    pass


if __name__ == '__main__':
    main()

```

* Python代码中的文件路径
  * 建议所有路径都使用 `utils` 包下的 `path_util.from_project_root` 方法得到绝对路径
  * 例如要访问`data/train_set.csv`时，先鼠标右键复制`train_set.csv`的相对路径，然后直接调用方法就好
```python
from utils.path_util import from_project_root
train_data_url = from_project_root('data/train_set.csv')
print(train_data_url)
```

           
