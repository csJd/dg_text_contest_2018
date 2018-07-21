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
                filter_shape = [filter_size,embedding_size,1,num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape,stddev=0.1),name="W")
                b = tf.Variable(tf.constant(0.1,shape=[num_filters]),name="b")

                #卷积层
                conv = tf.nn.conv2d(self.embedding_chars_expanded,W,strides=[1,1,1,1],padding='VALID',name="pool")

                #非线性激活函数
                h  = tf.nn.relu( tf.nn.bias_add(conv,b),name="relu")

                #采用最大池化层
                pooled = tf.nn.max_pool(h,ksize=[1,sequence_length - filter_size + 1,1,1],strides=[1,1,1,1],padding='VALID',name="pool")

                #汇总所有不同filter_size的池化结果
                pooled_outputs.append(pooled)

        #汇总所有的池化后的特征
        num_filters_total = num_filters * len( filter_sizes)
        self.h_pool = tf.concat(pooled_outputs,3)
        self.h_pool_flat = tf.reshape(self.h_pool,[-1,num_filters_total])

        #添加 dropout，规则化
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat,self.dropout_keep_prob)

        # 构造损失函数，优化函数，和 预测 输出层
        with tf.name_scope("output"):

            W = tf.get_variable(
                "W",
                shape=[num_filters_total,num_classes],
                initializer=tf.contrib.layers.xavier_initializer()
            )
            b = tf.Variable(tf.constant(0.1,shape=[num_classes]),name="b")
            #使用L2范式，
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)

            self.scores  = tf.nn.xw_plus_b(self.h_drop,W,b,name="scores")
            self.predictions = tf.argmax(self.scores,1,name="prediction")

            #Calculate mean cross-entrpy loss
            with tf.name_scope("loss") :

                losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores,labels=self.input_y)
                self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

            #Accuracy
            with tf.name_scope("accuracy") :

                correct_prediction = tf.equal(self.predictions,tf.argmax(self.input_y,1))
                self.accuracy = tf.reduce_mean( tf.cast(correct_prediction,"float"),name="accuracy")


