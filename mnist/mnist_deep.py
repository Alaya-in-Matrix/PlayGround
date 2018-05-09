import tensorflow as tf
import numpy as np
from mnist import MNIST

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)


mndata = MNIST('dataset')
train_images, labels1 = mndata.load_training()
test_images, labels2 = mndata.load_testing()

train_size = 60000
test_size = 10000

train_labels = []
for i in range(train_size):
    train_labels.append([0]*10)
    train_labels[i][labels1[i]] = 1

test_labels = []
for i in range(test_size):
    test_labels.append([0]*10)
    test_labels[i][labels2[i]] = 1

x = tf.placeholder(tf.float32, [None,784])
img = tf.reshape(x, [-1,28,28,1])
y_ = tf.placeholder(tf.float32, [None,10])


with tf.name_scope('conv1'):
    W_conv1 = weight_variable([5,5,1,32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(img, W_conv1) + b_conv1)

with tf.name_scope('pool1'):
    h_pool1 = max_pool_2x2(h_conv1)

with tf.name_scope('conv2'):
    W_conv2 = weight_variable([5,5,32,64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

with tf.name_scope('pool2'):
    h_pool2 = max_pool_2x2(h_conv2)

with tf.name_scope('fc1'):
    W_fc1 = weight_variable([7*7*64,1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1,7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    
with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

with tf.name_scope('fc2'):
    W_fc2 = weight_variable([1024,10])
    b_fc2 = bias_variable([10])
    y = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y))
train = tf.train.GradientDescentOptimizer(0.0001).minimize(loss)

correct_prediction = tf.equal(tf.argmax(y_,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    batch = 10
    epoch_num = 5
    for _ in range(epoch_num):
        for i in range(train_size/batch):
            start = i*batch
            end = (i+1)*batch
            image = np.asarray(train_images[start:end])
            label = np.asarray(train_labels[start:end])
            l, tr, accur = sess.run([loss, train, accuracy], feed_dict={x:image, y_:label, keep_prob:0.7})
            if i%100 == 0:
                 print "step =", i, "loss =", l, "accuracy =", accur

    print(sess.run([loss, accuracy], feed_dict={x:test_images, y_:test_labels, keep_prob:1.0}))
 
