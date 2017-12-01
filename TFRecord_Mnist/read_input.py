# -*- coding: utf-8 -*- 
import tensorflow as tf

files = tf.train.match_filenames_once(r"D:\Eclipse\Projects\NN_train\input_data_sample\csv_files\data-*")
filename_queue = tf.train.string_input_producer(files, shuffle = True)

init = (tf.global_variables_initializer(), tf.local_variables_initializer())

'''
1）假如你要读取的文件是像 CSV 那样的文本文件，用的文件读取器和解码器就是 TextLineReader 和 decode_csv 。
2）假如你要读取的数据是像 cifar10 那样的 .bin 格式的二进制文件，就用 tf.FixedLengthRecordReader 
    和 tf.decode_raw 读取固定长度的文件读取器和解码器。
'''
reader = tf.TextLineReader()
key,value = reader.read(filename_queue)
record_defaults = [["1."], ["2."], ["3."], ["4."],["5."]]
col1,col2,col3,col4,col5 = tf.decode_csv(value,record_defaults=record_defaults)
x,y = tf.stack([col1,col2,col3,col4]),col5

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(files))
    corrd = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=corrd)
    for i in range(10):
        example,label = sess.run([x,y])
        print(example,label)
    corrd.request_stop()
    corrd.join(threads)

