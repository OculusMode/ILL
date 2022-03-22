import numpy as np
from numpy import random, linalg as LA
# import matplotlib.pyplot as plt
from keras.layers import Dense, Layer, Input, Concatenate, Reshape
from keras import Model
import os
# %tensorflow_version 2.x
import tensorflow as tf


device_name = tf.test.gpu_device_name()
if device_name != "/device:GPU:0":
    device_name = "/cpu:0"
else:
    device_name = "/gpu:0"

with tf.device(device_name):
    pass