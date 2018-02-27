import tensorflow as tf
import numpy as np


x=tf.placeholder(tf.float32)
w=tf.Variable(.03,dtype=tf.float32)
b=tf.Variable(0.2,dtype=tf.float32)
print(x,w,b)

y=tf.placeholder(tf.float32)
y_=w*x+b
cost=tf.reduce_sum(tf.square(y_ - y))
x_train=np.array([1,2,3,4,5,7])
y_train=np.array([1,2,3,4,5,7])

sess=tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(y_,{x:x_train}))

optimizer=tf.train.GradientDescentOptimizer(0.003)
training=optimizer.minimize(cost)

for i in range(600):
    sess.run(training,{x: x_train,y: y_train})
    print(sess.run(y_, {x: x_train}))

print(sess.run(w))

data=input("Enter a data to predict its price")
print("the price of item is",sess.run(y_,{x:data}))