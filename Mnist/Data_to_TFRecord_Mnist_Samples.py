# -*- coding: utf-8 -*- 
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
import numpy as np

def init(f_directory):
    mnist = input_data.read_data_sets(f_directory,dtype = tf.uint8,one_hot = True)  
    print("训练集总数：%s" % mnist.train.num_examples)
    print("验证集总数：%s" %mnist.validation.num_examples)
    print("测试集总数：%s" %mnist.test.num_examples)
    train_images = mnist.train.images
    train_labels = mnist.train.labels
    train_len = mnist.train.num_examples
    valid_images = mnist.validation.images
    valid_labels = mnist.validation.labels
    valid_len = mnist.validation.num_examples
    test_images = mnist.test.images
    test_labels = mnist.test.labels
    test_len = mnist.test.num_examples
    assert(train_images.shape[1] == valid_images.shape[1]) and (train_images.shape[1] == test_images.shape[1]) , "图片加载部分缺失"
    assert(train_len == train_labels.shape[0]),"训练集缺失"
    assert(valid_len == valid_labels.shape[0]),"验证集缺失"
    assert(test_len == test_labels.shape[0]),"测试集缺失"
    pixels = train_images.shape[1]
    return pixels,[[train_images,train_labels,train_len],[valid_images,valid_labels,valid_len],[test_images,test_labels,test_len]]

#将值转化为64位整数列表
def _int64_feature(value):
    return tf.train.Feature(int64_list = tf.train.Int64List(value=[value]))
#将值转化为字符串列表
def _bytes_feature(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value=[value]))
#将值转化为浮点数列表
def _float_feature(value):
    return tf.train.Feature(float_list = tf.train.FloatList(value=[value]))

def write_files(path,data,pixels):
    to_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),path)
    print(to_path)
    images,labels,leng = data[0],data[1],data[2]
    print(images.shape)
    writer = tf.python_io.TFRecordWriter(to_path)
    for index in range(leng):
        print(images[index].shape)
        print(np.sum(images[index]))
        image_raw = images[index].tostring()
        example = tf.train.Example(features = tf.train.Features(feature={
            'pixels' : _int64_feature(pixels),
            'label' : _int64_feature(np.argmax(labels[index])),
            'image_row' : _bytes_feature(image_raw)
            }))
        writer.write(example.SerializeToString())

def main():
    print("ROOT_PATH:%s" % os.path.dirname(os.path.realpath(__file__)))
    f_directory = os.path.join(os.path.dirname(os.path.realpath(__file__)),'data')
    to_paths = ["TF_Record_Data\\train_record_data","TF_Record_Data\\validation_record_data","TF_Record_Data\\test_record_data"]
    pixels,data = init(f_directory)
    for i in range(len(to_paths)):
        write_files(to_paths[i],data[i],pixels)
        print("%s---------------Done!" % to_paths[i].split('\\')[1])
     
    
if __name__ == "__main__":
    main()