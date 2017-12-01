# -*- coding: utf-8 -*- 
import os
import time
import math
import tensorflow as tf
import numpy as np


class MLP(object):
    def __init__(self,sess,layers_number,crop = True,batch_size = 16, data_size = 'deafault',
                 input_fname_pattern='*.jpg',):