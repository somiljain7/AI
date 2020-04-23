import tensorflow as tf
import numpy as np
from scipy import stats
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
X_train,y_train = mnist.train.next_batch(5000)
X_test, y_test = mnist.test.next_batch(100)
k=3
target_x= tf.placeholder("float",[1784])
X = tf.placeholder("float",[None, 784])
y = tf.placeholder("float",[None, 10])
l1_dist = tf.reduce_sum(tf.abs(tf.sub(x, target_x)), 1)
l2_dist = tf.reduce_sum(tf.square(tf.sub(x, target_x)), 1)
nn = tf.nn.top_k(-l1_dist, k)
init = tf.initialize_all_variables()
accuracy_history = []
with tf.Session() as sess:
	sess.run(init)
