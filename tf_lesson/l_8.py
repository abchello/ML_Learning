# -*- coding: utf-8 -*-
import tensorflow as tf


"""
    placeholder
"""


input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

output = tf.multiply(input1, input2)

with tf.Session() as sess:
    print(sess.run(output, feed_dict={input2: [4.], input1: [3.]}))
    # print('input1' + sess.run(input1))
    print('-------')

    for i in range(20):
        feed_dict_l = {input1: [4. + i], input2: [3. + i]}
        print(sess.run(output, feed_dict_l))

print('---end---')
