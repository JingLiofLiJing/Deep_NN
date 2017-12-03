# -*- coding: utf-8 -*- 
import tensorflow as tf
import numpy as np

import random
from numpy.random import RandomState

class MLP(object):
    def __init__(self,sess,layers_number = [10,15,20,5,20,15,10],batch_size = 16, dataset = 'deafault',
                 input_fname_pattern='*tfrecords',crop = True):
        
        '''
        Args:
            sess:TensorFlow session
            layers_number:list为层描述，长度为总的层数，第i个值表示第i层有多少个神经元，第0层为输入层
            batch_size:The size of batch. Should be specified before training.
            dataset:默认随机生成，可以选择导入处理好的数据
            input_fname_pattern:读取文件后缀，默认为读入tfrecords
        '''
        self.sess = sess
        self.batch_size = batch_size
        self.layers_number = layers_number
        self.dataset = dataset
        self.input_fname_pattern = input_fname_pattern
        self.crop = crop
    
        self.layer_names = []
        self.tot_layers = len(self.layers_number)
        for i in range(1,self.tot_layers):
            name = "hidden_layer%d" % i
            self.layer_names.append(name)
            
            
        if self.dataset == 'deafault':
            self.data_X,self.data_Y,self.test_X,self.test_Y = self.ramdon_input()
        else:
            pass
        
        self.build_model()
        
    
    def get_weight_variable(self,shape):
        weights = tf.get_variable("weights",shape,initializer=tf.truncated_normal_initializer(stddev=0.1))
        return weights
    
    def get_bias(self,shape):
        biases =  tf.get_variable("biases",shape,initializer=tf.constant_initializer(0))
        return biases
    
    def build_model(self):
        input_dims = [None,self.layers_number[0]]
        output_dims = [None,self.layers_number[-1]]
        self.input = tf.placeholder(tf.float32, shape = input_dims, name = "x_input")
        self.y_real = tf.placeholder(tf.float32, shape = output_dims, name = "y_real")
        hidden_input = self.input
        for i in range(len(self.layer_names)):
            name = self.layer_names[i]
            with tf.variable_scope(name):
                w_shape = [self.layers_number[i],self.layers_number[i+1]]
                b_shape = [self.layers_number[i+1]]
                w = self.get_weight_variable(w_shape)
                b = self.get_bias(b_shape)
                if i < len(self.layer_names) - 1:
                    a = tf.nn.relu(tf.matmul(hidden_input,w)+b,name = "relu")
                else:
                    a = tf.nn.sigmoid(tf.matmul(hidden_input,w)+b,name = "sigmoid")
                if i < len(self.layer_names) - 1:
                    print(i)
                    bn_a = tf.contrib.layers.batch_norm(a,decay = 0.9,epsilon=1e-5,scale=True,is_training=self.crop,scope=name)         
                    hidden_input = bn_a
                else:
                    hidden_input = a
        self.y_predict = tf.nn.softmax(hidden_input)
        self.correct_prediction = tf.equal(tf.arg_max(self.y_predict, 1), tf.arg_max(self.y_real, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.y_predict, labels = self.y_real))

        tf.summary.scalar('loss', self.cross_entropy)
        tf.summary.scalar('accuary', self.accuracy)
        self.summary_op =  tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter("D:\\111\\",self.sess.graph)
#         saver = tf.train.Saver()
        
    def train(self,config):
        self.train_optim = tf.train.AdamOptimizer(config.learning_rate).minimize(self.cross_entropy)
        self.sess.run((tf.global_variables_initializer(), tf.local_variables_initializer()))
        for i in range(config.epoch):
            start = (i*self.batch_size)%self.dataset_size
            end = min(start + self.batch_size,self.dataset_size)
#             print(self.data_X[start:end],self.data_Y[start:end])
            summary_op,_,accuary = self.sess.run([self.summary_op,self.train_optim,self.accuracy],feed_dict = {self.input:self.data_X[start:end],self.y_real:self.data_Y[start:end]})
            self.summary_writer.add_summary(summary_op, i)
            print("train cross_entropy %d steps and accuary is %s" % (i,str(accuary)))
        
    def test(self,config):
        self.crop = config.crop
        total_cross_entropy = self.sess.run(self.cross_entropy,feed_dict = {self.input:self.test_X,self.y_real:self.test_Y})
        print("test cross_entropy on all data is %g" % total_cross_entropy)
    
    def ramdon_input(self):
        rdm = RandomState(1)
        self.dataset_size = 512
        X = rdm.rand(self.dataset_size,self.layers_number[0])
        Y = np.zeros(shape=(self.dataset_size,self.layers_number[-1]), dtype = np.int32)
        for i in range(self.dataset_size):
            num = int(random.uniform(0,self.layers_number[-1] - 1))
            Y[i,num] = 1
        return X[:928],Y[:928],X[928:],Y[928:]
        
        
        