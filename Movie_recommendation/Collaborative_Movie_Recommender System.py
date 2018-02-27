import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
#mat=scipy.io.loadmat('ex8_movies.mat')
#Y=mat['Y']
#R=mat['R']
#plt.interactive(False)
#plt.plot(Y)
#plt.show()
#print(Y[R==1])
#print(Y[R==0])
#params=scipy.io.loadmat('ex8_movieParams.mat')
#num_users = params['num_users']
#num_movies =params['num_movies']
#num_features=params['num_features']

#Y= np.array([[5,5,0,0], [5,None,None,0], [None,4,0,None], [0,None,5,4], [0,0,5,None]])
Y= np.array([[5,5,1,1], [5,0,0,1], [0,4,1,0], [1,1,5,5], [1,1,5,0]],dtype=float)

X=np.array([[1,0.9,0],[1,1 ,0.01],[1,0.99 ,0],[1,0.1 ,1.0],[1,0,0.9]])
#R= np.array([[1,1,1,1], [1,0,0,1], [0,1,1,0], [1,0,1,1], [1,1,1,0]])
print("Predictions:",Y)


#print("shape of Y:",np.shape(Y))
#print(num_users)
#print(num_movies)
#x=np.random.rand(1682,943)

x=tf.Variable(np.random.rand(5,3),tf.float32)
#X=tf.Variable(tf.zeros([1682,943]),dtype=tf.float32)
thetas=tf.Variable(np.random.rand(4,3),dtype=tf.float32)
y=tf.placeholder(tf.float32,[5 ,4])
sess=tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

J=np.square(tf.matmul(tf.cast(x,dtype=tf.float32),tf.to_float(tf.transpose(thetas)))-y)
#J=J[J>0]
def reduce():
    m=0
    i,j=np.shape(Y)
    for a in range(i):
        for b in range(j):
             if Y[a][b]!=0:
                 m+=J[a][b]
    return m
lamd=100
cost=tf.reduce_mean(reduce())#+(lamd/2)*tf.reduce_sum(tf.square(tf.cast(thetas,dtype=tf.float32)))+(lamd/2)*tf.reduce_sum(tf.square(tf.cast(x,dtype=tf.float32)))

optimizer=tf.train.GradientDescentOptimizer(0.05).minimize(cost)

y_=tf.matmul(tf.cast(x,dtype=tf.float32),tf.transpose(thetas))
print("Initial theta:",sess.run(thetas),"\n\n")

#print(sess.run(cost))
for i in range(20):
    (sess.run(J, feed_dict={ y: Y}),"\n\n")
    (sess.run(reduce(), feed_dict={ y: Y}))
    (sess.run(cost,feed_dict={ y: Y}))
    (sess.run(optimizer,feed_dict={y:Y}))
    print("Predictions:",sess.run(y_),"\n\n")
    prdt=sess.run(y_)
    (sess.run(thetas),"\n\n")


def print1():
    m=0
    i,j=np.shape(Y)
    for a in range(i):
        for b in range(j):
             if Y[a][b]==0:
                 Y[a][b]=prdt[a][b]
    return Y
print(print1())
#print(sess.run(y_+1, feed_dict={X: x}))

#sess.run(tf.reshape(J,[-1]))
#Y.flatten()
#print(J)
#print(Y)

#sess.run(j_t,{J:J[:]})
#print(sess.run(cost))

#print("thetas: \n\n")
#print(sess.run(thetas))

