import tensorflow as tf
import numpy as np


x = tf.placeholder(dtype=tf.float32, shape=[None,224,244,3])
y_ = tf.placeholdet(dtype=tf.float32, shape=[None,1000])

def weight_variable(shape):
    init = tf.truncated_normal(shape=shape,dtype=tf.float32,stddev=0.1)
    return tf.Variable(init)

def bias_variable(shape):
    init = tf.constant(0.1,shape=shape)
    return tf.Variable(init)

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')


with tf.name_scope('conv11'):
    W_conv11 = weight_variable([3,3,3,64])
    b_conv11 = bias_variable([64])
    h_conv11 = tf.nn.relu(conv2d(x, W_conv11) + b_conv11)

with tf.name_scope('conv12'):
    W_conv12 = weight_variable([3,3,64,64])
    b_conv12 = bias_variable([64])
    h_conv12 = tf.nn.relu(conv2d(h_conv11, W_conv12) + b_conv11)

with tf.name_scope('pool1'):
    h_pool1 = max_pool_2x2(h_conv12)

with tf.name_scope('conv21'):
    W_conv21 = weight_variable([3,3,64,128])
    b_conv21 = bias_variable([128])
    h_conv21 = tf.nn.relu(conv2d(h_pool1, W_conv21) + b_conv21)

with tf.name_scope('conv22'):
    W_conv22 = weight_variable([3,3,128,128])
    b_conv22 = bias_variable([128])
    h_conv22 = tf.nn.relu(conv2d(h_conv21, W_conv22) + b_conv22)

with tf.name_scope('pool2'):
    h_pool2 = max_pool_2x2(h_conv22)

with tf.name_scope('conv31'):
    W_conv31 = weight_variable([3,3,128,256])
    b_conv31 = bias_variable([256])
    h_conv31 = tf.nn.relu(conv2d(h_pool2, W_conv31) + b_conv31)

with tf.name_scope('conv32'):
    W_conv32 = weight_variable([3,3,256,256])
    b_conv32 = bias_variable([256])
    h_conv32 = tf.nn.relu(conv2d(h_conv31, W_conv32) + b_conv32)

with tf.name_scope('conv33'):
    W_conv33 = weight_variable([3,3,256,256])
    b_conv33 = bias_variable([256])
    h_conv33 = tf.nn.relu(conv2d(h_conv32, W_conv33) + b_conv33)

with tf.name_scope('pool3'):
    h_pool3 = max_pool_2x2(h_conv33)

with tf.name_scope('conv41'):
    W_conv41 = weight_variable([3,3,256,512])
    b_conv41 = bias_variable([512])
    h_conv41 = tf.nn.relu(conv2d(h_pool3, W_conv41) + b_conv41)

with tf.name_scope('conv42'):
    W_conv42 = weight_variable([3,3,512,512])
    b_conv42 = bias_variable([512])
    h_conv42 = tf.nn.relu(conv2d(h_conv41, W_conv42) + b_conv42)

with tf.name_scope('conv43'):
    W_conv43 = weight_variable([3,3,512,512])
    b_conv43 = bias_variable([512])
    h_conv43 = tf.nn.relu(conv2d(h_conv42, W_conv43) + b_conv43)

with tf.name_scope('pool4'):
    h_pool4 = max_pool_2x2(h_conv43)

with tf.name_scope('conv51'):
    W_conv51 = weight_variable([3,3,512,512])
    b_conv51 = bias_variable([512])
    h_conv51 = tf.nn.relu(conv2d(h_pool4, W_conv51) + b_conv51)

with tf.name_scope('conv52'):
    W_conv52 = weight_variable([3,3,512,512])
    b_conv52 = bias_variable([512])
    h_conv52 = tf.nn.relu(conv2d(h_conv51, W_conv52) + b_conv52)

with tf.name_scope('conv53'):
    W_conv53 = weight_variable([3,3,512,512])
    b_conv53 = bias_variable([512])
    h_conv53 = tf.nn.relu(conv2d(h_conv52, W_conv53) + b_conv53)
    h_conv53_flat = tf.reshape(h_conv53, [-1,7*7*512])

with tf.name_scope('fc1'):
    W_fc1 = weight_variable([7*7*512, 4096])
    b_fc1 = bias_variable([4096])
    h_fc1 = tf.nn.relu(tf.matmul(h_conv53_flat, W_fc1) + b_fc1)

with tf.name_scope('fc2'):
    W_fc2 = weight_variable([4096, 4096])
    b_fc2 = bias_variable([4096])
    h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

with tf.name_scope('fc3'):
    W_fc3 = weight_variable([4096, 1000])
    b_fc3 = bias_variable([1000])
    y = tf.matmul(h_fc2, W_fc3) + b_fc3

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train = tf.train.GradientDescentOptimizer(0.0001).minimize(loss)

correct_prediction = tf.equal(tf.argmax(y_,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


