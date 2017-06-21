# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

print("hello_word!")
node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
print(node1, node2)

x = np.array([1., 2., 3., 4.])
y = np.array([0., -1., -2., -3.])
w = np.multiply(x, y)
print(w)