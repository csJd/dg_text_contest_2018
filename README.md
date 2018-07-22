## Pycharm 建议设置

* 换行符统一为'\n'
  * File | Settings | Code Style
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


## 数据集基本情况
    
* 文件基本说明
    * 第0列：文章id  
    * 第1列：“字”级别上的表示 
    * 第2列：“词”级别上的表示  
    * 第3列： label【类别】
       
* 类别统计
    
    * phrase_level训练集
        * 共有 102277个训练样本
        * 训练集中"phrase_leve"上字典维度有 875129 
    
    * word_level训练集
    
    
## 初步分工

* 分配部分
    * term weight(Important!) +多分类器 【小黑仔，登登】   
    * word_embedding 训练 + doc2vec (LDA)【小黑仔，登登】    
    * deep_learning_model (CNN+RNN) 【鹏鹏，编写各种深度模型】
        【小黑仔，登登，看论文！】
            
