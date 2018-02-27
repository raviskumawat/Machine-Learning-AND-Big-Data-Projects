import numpy as np
movies=["Guardians Of The Galaxy","Interstellar","Despicable Me","The.Lion.King","The Theory Of Everything",
        "Titanic","Doctor Strange","Captian America:Civil War","Iron man","X-men","A walk to remember",
        "Letter to the Juliet","Angry Birds","Ice-Age","A beautiful Mind","Kung fu Panda","Inception",
        "Wonder Woman","R.Ramanujan","Beauty and the Beast"]
users=["Ravi","Surya","Vishnu","Rudranshu","Aditya"]

Y=np.array([[9, 8, 0, 5, 0],
       [0, 9, 8, 0, 5],
       [8, 9, 0, 8, 7],
       [9, 0, 0, 9, 7],
       [8, 8, 8, 0, 0],
       [9, 8, 0, 0, 9],
       [7, 0, 0, 7, 8],
       [0, 9, 2, 6, 5],
       [8, 9, 7, 0, 8],
       [9, 8, 8, 5, 0],
       [5, 9, 6, 0, 5],
       [0, 0, 7, 9, 0],
       [9, 0, 7, 5, 0],
       [0, 9, 6, 8, 8],
       [8, 4, 0, 8, 0],
       [9, 0, 9, 9, 6],
       [8, 0, 0, 8, 7],
       [0, 0, 3, 0, 8],
       [0, 9, 0, 5, 9],
       [0, 0, 0, 7, 9]],dtype=float)

Y=Y[:,0]
print("Y",np.shape(Y))

genre=["Documentary Film","Action","Science Fiction","Cartoon/comedy","Romantic"]

X=np.array([[ 0.1, 0.7, 0.7, 0.1, 0.1],
           [0.9, 0.3, 0.7, 0.0, 0.2],
           [0.1, 0.7, 0.7, 0.9, 0.2],
           [0.1, 0.4, 0.1, 0.9, 0.2],
           [0.9, 0.1, 0.7, 0.1, 0.5],
           [0.1, 0.3, 0.2, 0.2, 0.9],
           [0.2, 0.7, 0.7, 0.2, 0.1],
           [0.2, 0.9, 0.7, 0.1, 0.1],
           [0.1, 0.7, 0.7, 0.3, 0.3],
           [0.2, 0.7, 0.4, 0.1, 0.1],
           [0.1, 0.1, 0.1, 0.2, 1.0],
           [0.3, 0.2, 0.1, 0.3, 0.9],
           [0.2, 0.7, 0.2, 0.9, 0.2],
           [0.2, 0.7, 0.3, 0.9, 0.2],
           [0.9, 0.1, 0.2, 0.1, 0.5],
           [0.1, 0.8, 0.1, 0.8, 0.1],
           [0.1, 0.4, 0.6, 0.3, 0.4],
           [0.2, 0.8, 0.3, 0.1, 0.4],
           [0.8, 0.2, 0.4, 0.1, 0.2],
           [0.2, 0.3, 0.2, 0.6, 0.8]])
print(X)
N=len(X)
#Y=mx+b
b=np.random.rand(20,1)
m=np.random.rand(5,1)
print("b",b)

def cal_gradient(m,bias,X,Y):
    a,b=np.shape(X)
    grad_m=0
    grad_b=0
    N=0
    print("m",m)

    for i in range(0,a):

            x_i=X[i,:]
            print("x_i",x_i)

            y_i=Y[i]
            b_i=bias[i,:]
            print("b_i=",b_i)
            print("y_i=",y_i)

            if(y_i!=0):
                print("IN")
                N+=1

                grad_m+=-np.sum((y_i-(np.dot(x_i,m)+b_i))*x_i)
                print("Grad_m",grad_m)
                grad_b+=-(y_i-(np.dot(x_i,m)+b_i))
                print("Grad_b",grad_b)
    print(np.shape(grad_b))
    new_gm=(2/N)*grad_m
    new_gb=(2/N)*grad_b
    return new_gm,new_gb


def gradient_descent(m,b,X,Y,alpha,num_iter):
    #m=np.random.rand(5,5)
    #b=np.random.rand(20,5)
    for i in range(0,num_iter):
        new_gm,new_gb=cal_gradient(m,b,X,Y)
        m=m-alpha*new_gm
        b=b-alpha*new_gb
        y_=np.dot(X,m)+b
        print("Prediction:",y_)

    return m,b

m,b=gradient_descent(m,b,X,Y,0.05,50)
#print(np.shape(Y))
y_=np.dot(X,m)+b
prd=Y
a=20
d = dict()
for i in range(0,a):
    if(Y[i]==0):
        prd[i]=y_[i]
        x=float(y_[i])
        d[x]=i
print(d)
print("Prediction:",prd)
