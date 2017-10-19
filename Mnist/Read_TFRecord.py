# -*- coding: utf-8 -*- 
import tensorflow as tf
#出队中队列元素最少个数，当出队而队列元素不够时，会等待更多元素进队
Min_after_dequeue = 50
Batch_Size = 2
#性能需求
Capacity = 10000
Image_Channels = 1
Image_wid_hei = 28

def Pro_files(path):
    files = tf.train.match_filenames_once(path)
    filename_queue = tf.train.string_input_producer(files, shuffle = True)
    
    reader = tf.TFRecordReader()
    key,value = reader.read(filename_queue)
    features = tf.parse_single_example(
        value,
        features={
            'pixels' : tf.FixedLenFeature([],tf.int64),
            'label' : tf.FixedLenFeature([],tf.int64),
            'image_row' :tf.FixedLenFeature([],tf.string)
        })
    
    image_size = features['pixels']
    image_label = features['label']
    image_row = features['image_row'] 
    
    image = tf.decode_raw(image_row,tf.uint8)
    image = tf.reshape(image, [28,28,1])
    x1,x2 = tf.train.shuffle_batch([image,image_label],batch_size=Batch_Size,capacity =Capacity,min_after_dequeue=Min_after_dequeue)
    
    init = (tf.global_variables_initializer(), tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(init)
        print(sess.run(files))
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess = sess, coord = coord)
        for i in range(1):
            print('------------------%s------------------'%i)
            xx1,xx2 = sess.run([x1,x2])
            print(xx1)
            print('------------------%s------------------'%i)
        coord.request_stop()
        coord.join(threads)

def main():
    path = 'D:\\Eclipse\\Projects\\NN_train\\Mnist_test\\Mnist\\TF_Record_Data\\*'
    Pro_files(path)

if __name__ == "__main__":
    main()

