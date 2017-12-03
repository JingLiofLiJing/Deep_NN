# -*- coding: utf-8 -*- 
import os
import numpy as np
import tensorflow as tf
import pprint
from MLP import *

flags = tf.app.flags
flags.DEFINE_integer("epoch", 2000, "Epoch to train [500]")
flags.DEFINE_float("learning_rate", 0.0001, "Learning rate of for adam [0.0002]")
flags.DEFINE_integer("batch_size", 16, "The size of batch [16]")
flags.DEFINE_string("dataset", "deafault", "The name of dataset")
flags.DEFINE_string("input_fname_pattern", "*.tfrecord", "Glob pattern of filename of input [*]")
flags.DEFINE_boolean("crop", True, "True for training, False for testing [True")
FLAGS = flags.FLAGS

pp = pprint.PrettyPrinter()

def main(_):
    pp.pprint(flags.FLAGS.__flags)
    with tf.Session() as sess:
        MLP_one = MLP(
            sess,
            layers_number = [10,15,20,15,10],
            batch_size = FLAGS.batch_size,
            dataset = FLAGS.dataset,
            input_fname_pattern = FLAGS.input_fname_pattern)

    MLP_one.train(FLAGS)
    FLAGS.crop = False
    MLP_one.test(FLAGS)
    

if __name__ == '__main__':
    tf.app.run()