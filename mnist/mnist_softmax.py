import tensorflow as tf
import numpy as np
from mnist import MNIST

mndata = MNIST('dataset/')
train_images, labels1 = mndata.load_training()
test_images, labels2 = mndata.load_testing()

train_labels = []
for i in range(60000):
    train_labels.append([0]*10)
    train_labels[i][labels1[i]] = 1 

test_labels = []
for i in range(10000):
    test_labels.append([0]*10)
    test_labels[i][labels2[i]] = 1

x = tf.placeholder(tf.float32, [None,784])
y_ = tf.placeholder(tf.float32, [None,10])

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

y = tf.matmul(x, W) + b

cross_prediction = tf.equal(tf.argmax(y_,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(cross_prediction, tf.float32))

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y,labels=y_))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    batch = 10
    for i in range(60000/batch):
        start = i*batch
        end = (i+1)*batch
        img = np.asarray(train_images[start:end])
        label = np.asarray(train_labels[start:end])
        l, tr, accur = sess.run([loss, train, accuracy], feed_dict={x:img, y_:label})
        if i%100 == 0:
            print "step =", i,"loss = ", l, "accuracy = ", accur
    
    accur, l = sess.run([accuracy, loss], feed_dict={x:test_images, y_:test_labels})
    print "accuracy = ", accur, "loss = ", l


