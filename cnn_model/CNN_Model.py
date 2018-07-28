import tensorflow as tf

class CNN_Model(object):

    '''
        A CNN Model for sentense classification
        Used an embedding layer ,convolutional , max-pooling and softmax layper , sum = 4 laypers
    '''
    def __init__(self,sequence_length,num_classes,vocab_size,embedding_size,filter_sizes,num_filters,l2_reg_lambda=0.0):

        #PlaceHolder for input , output and dropout
        self.input_x = tf.placeholder(tf.int32,[None,sequence_length],name='input-x')
        self.input_y = tf.placeholder(tf.float32,[None,num_classes],name='input-y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob") # regularization

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        #Embedding layer
        with tf.device('/cpu:0'),tf.name_scope("embedding"):

            self.W = tf.Variable(
                tf.random_uniform([vocab_size,embedding_size],-1.0,1.0),name="W"
            )
            self.embedding_chars = tf.nn.embedding_lookup(self.W,self.input_x) #产生 [None,sequence_length,embedding_size]
            #在现有的基础上增加一个维度,因为 [batch,sequance_len,embeding_dim,channel]
            self.embedding_chars_expanded = tf.expand_dims(self.embedding_chars,-1)

        #Create a convolution + maxpool layer for each filter size
        pooled_outputs = []  #对于每一个filter_size ，将每个池化层得到
        for i, filter_size in enumerate(filter_sizes):

            with tf.name_scope("conv-maxpool-%s" % filter_size):

                #Convolution Layer
                filter_shape1 = [filter_size,embedding_size,1,num_filters]
                W1 = tf.Variable(tf.truncated_normal(filter_shape1,stddev=0.1),name="W1")
                b1 = tf.Variable(tf.constant(0.1,shape=[num_filters]),name="b1")

                #卷积层 shape: [batch, sequence_length - filter_szie + 1 , 1 , num_filters]
                conv1 = tf.nn.conv2d(self.embedding_chars_expanded,W1,strides=[1,1,1,1],padding='VALID',name="conv1")
                #非线性激活函数
                h1  = tf.nn.relu( tf.nn.bias_add(conv1,b1),name="relu1")
                # 采用最大池化层 pooled.shap = [batch,]
                pooled1 = tf.nn.max_pool(h1,ksize=[1,sequence_length - filter_size + 1,1,1],strides=[1,1,1,1],padding='VALID',name="pool1")
                #汇总所有不同filter_size的池化结果
                pooled_outputs.append(pooled1)

                # 由于需要考虑更长的词语
                # transpose trans_conv1.shape = [batch,sen_len-filter_size+1,num_filters,1]
                # trans_conv1 = tf.transpose(conv1,[0,1,3,2])
                # filter_shape2 = [2,num_filters,1,num_filters]
                # W2 = tf.Variable(tf.truncated_normal(filter_shape2,stddev=0.1),name="W2")
                # b2 = tf.Variable(tf.constant(0.1,shape=[num_filters],name="b2"))
                # # conv2.shape = [batch, sen_len-filter_size+1-2+1,1,num_filters]
                # conv2 = tf.nn.conv2d(trans_conv1,W2,strides=[1,1,1,1],padding='VALID',name="conv2")
                # h2 = tf.nn.relu(tf.nn.bias_add(conv2,b2),name='relu2')
                # # max_pooling pooled shape
                # pooled2 = tf.nn.max_pool(h2,ksize=[1,sequence_length-filter_size,1,1],strides=[1,1,1,1],padding='VALID',name="pool2")
                # #
                # pooled_outputs.append(pooled2)

        #汇总所有的池化后的特征
        num_filters_total = num_filters  * len( filter_sizes)
        self.h_pool = tf.concat(pooled_outputs,3)
        # h_pool_flat.shape = [batch, num_filters_total]
        self.h_pool_flat = tf.reshape(self.h_pool,[-1,num_filters_total])

        #添加 dropout，规则化
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat,self.dropout_keep_prob)

        # 构造损失函数，优化函数，和 预测 输出层
        with tf.name_scope("output"):

            # 建立两层全连接层
            # 第一层
            W1 = tf.get_variable(
                "W1",
                shape=[num_filters_total,num_filters],
                initializer=tf.contrib.layers.xavier_initializer()
            )
            b1 = tf.Variable(tf.constant(0.1,shape=[num_filters]),name="b1")
            # fc1.shape = [batch,num_classes]
            fc1  = tf.nn.xw_plus_b(self.h_drop,W1,b1,name="fc1")

            # 第二层
            W2 = tf.get_variable(
                "W2",
                shape=[num_filters,num_classes],
                initializer=tf.contrib.layers.xavier_initializer()
            )
            b2 = tf.Variable(tf.constant(0.1,shape=[num_classes]),name="b2")
            fc2 = tf.nn.xw_plus_b(fc1,W2,b2,name="scores")

            #使用L2范式，
            l2_loss += tf.nn.l2_loss(W2)
            l2_loss += tf.nn.l2_loss(b2)

            # softmax no test
            self.scores = tf.nn.softmax(fc2,name="scores")
            self.predictions = tf.argmax(self.scores,1,name="prediction")

            #Calculate mean cross-entrpy loss
            with tf.name_scope("loss") :

                losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores,labels=self.input_y)
                self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

            #Accuracy
            with tf.name_scope("accuracy") :

                correct_prediction = tf.equal(self.predictions,tf.argmax(self.input_y,1))
                self.accuracy = tf.reduce_mean( tf.cast(correct_prediction,"float"),name="accuracy")