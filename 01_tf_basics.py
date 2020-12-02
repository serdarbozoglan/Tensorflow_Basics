import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] ='2'

import tensorflow as tf
import numpy as np
print(tf.__version__)

# initilization of Tensors
x = tf.constant(4)
print(x)

y = tf.constant(4, shape=(1,1))
print(y)

z = tf.constant(4, shape=(1,1), dtype=np.float32)
print(z)

x = tf.constant([])