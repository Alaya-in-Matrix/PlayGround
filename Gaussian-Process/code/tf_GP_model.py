import tensorflow as tf
import numpy as np
import cPickle as pickle

with open('./test_bench/enb/enb.pickle','rb') as f:
    dataset = pickle.load(f)

train_x = dataset['train_x'].T
train_y = dataset['train_y'][0]
test_x = dataset['test_x'].T
test_y = dataset['test_y'][0]

num_w1 = 100
num_w2 = 100
num_w3 = 100
num_dim = 8

train_y_zero = train_y - np.mean(train_y)
train_y_zero = train_y_zero.reshape((train_y_zero.size, 1))

X = tf.placeholder(tf.float32, shape=(None,num_dim))
y_ = tf.placeholder(tf.float32, shape=(None,1))

# X = tf.constant(train_x, dtype=tf.float32)
# y_ = tf.constant(train_y_zero, dtype=tf.float32)

def weight_variable(shape):
    init = tf.truncated_normal(shape=shape, dtype=tf.float32, stddev=0.1)
    return tf.Variable(init)

def bias_variable(shape):
    init = tf.constant(0.1, shape=shape)
    return tf.Variable(init)

W1 = weight_variable([num_dim, num_w1])
b1 = bias_variable([num_w1])
layer1 = tf.nn.relu(tf.matmul(X, W1) + b1)

W2 = weight_variable([num_w1, num_w2])
b2 = bias_variable([num_w2])
layer2 = tf.nn.relu(tf.matmul(layer1, W2) + b2)

W3 = weight_variable([num_w2, num_w3])
b3 = bias_variable([num_w3])
Phi = tf.nn.relu(tf.matmul(layer2, W3) + b3)

sn2 = tf.Variable(tf.constant(np.std(train_y_zero)/2, dtype=tf.float32, shape=[1]))
sp2 = tf.Variable(tf.constant(np.std(train_y_zero), dtype=tf.float32, shape=[1]))

tmp = tf.reshape(tf.tile(tf.divide(sn2, sp2), [num_w3]), [1,-1])

A = tf.matmul(Phi, Phi, transpose_a=True) + num_w3 * tf.matmul(tmp, tf.eye(num_rows=num_w3))
A_inv = tf.matrix_inverse(A)


with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    p, pp, ppp, A_matrix = sess.run([Phi, sn2, sp2, A], feed_dict={X: train_x, y_: train_y_zero})
    print p.shape
    print pp
    print ppp
    print A_matrix[0]
    print sess.run(tf.matmul(tmp, tf.eye(num_rows=num_w3)))
    print sess.run(
