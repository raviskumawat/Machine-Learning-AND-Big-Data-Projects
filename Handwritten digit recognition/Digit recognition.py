import tensorflow as tf
import numpy as np
import matplotlib as mp
from tensorflow.examples.tutorials.mnist import input_data


sess=tf.InteractiveSession()

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
print(mnist)
x=tf.placeholder(tf.float32,[None,784])
y=tf.placeholder(tf.float32,[None,10])
w=tf.Variable(np.zeros([784,10]),dtype=tf.float32)
b=tf.Variable(tf.zeros(10),tf.float32)
print(x,w,b)
sess.run(tf.global_variables_initializer())
#y=tf.placeholder(tf.float32,[None,10])
y_= tf.nn.softmax(tf.matmul(x,w) + b)

cost= tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_))

#x_normal=np.array(tf.train.images)
#x_train=x_normal.reshape(-1,784)
#y_train=tf.train.labels
#x_train=tf.train.images

#sess.run(y_,{x:x_train})

optimizer=tf.train.GradientDescentOptimizer(0.5)
training=optimizer.minimize(cost)

for i in range(1000):
    batch = mnist.train.next_batch(100)
    sess.run(y_, {x: batch[0]})
    sess.run(training,{x: batch[0],y: batch[1]})

#correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#print(sess.run(accuracy,{x: x_train, y_: y_train}))


print(sess.run(tf.argmax(y_,1)))

#data=input("Enter a data to predict its price")
#print("the price of item is",sess.run(y_,{x:data}))