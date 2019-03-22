#!/usr/bin/env python
# coding: utf-8

# In[65]:


import os
import random
import numpy as np
from sklearn.utils import shuffle
import face_recognition
import dlib
import re
import cv2
from keras.utils import to_categorical
from sklearn.svm import SVC
import pickle

def give_embedding(img_):
    img=cv2.resize(img_,(256,256),interpolation=cv2.INTER_CUBIC)
    
    #extract gray img
    #gray_img=np.eye(256)
    gray_img=cv2.cvtColor(img.copy(),cv2.COLOR_BGR2GRAY)


    # Create a HOG face detector using the built-in dlib class
    face_detector = dlib.get_frontal_face_detector()
    #print(image)
    face=face_detector(gray_img,1)
    if len(face)<1:
        
        return []
    #detected_faces=face
    face=face[0]
    #face_img=cv2.rectangle(img,(detected_faces[0].left(),detected_faces[0].top()),(detected_faces[0].right(),detected_faces[0].bottom()),(0,255,0),1)
    face_img=img[face.top():face.bottom(),face.left():face.right()]
    face_embedding=np.array(face_recognition.face_encodings(face_img)).ravel()
    return np.array(face_embedding).ravel()


def load_small_img_dataset(path=os.getcwd()):
    data=[]
    labels=[]
    for i,image in enumerate(os.listdir(path)):
        if image.endswith(('.jpg','.jpeg','.png','bmp')):
            #print(image)
            label=re.findall('^(.*)_.*',image)[0]
            pixels=cv2.imread(os.path.join(path,image))
            #first make all images of same size using crop
            face_embedding=give_embedding(pixels)
            if len(face_embedding)<1:
                print('face cannot be detected in {0}  [IGNORING]'.format(image))
                continue
            data.append(face_embedding)
            labels.append(label)
            
    return data,labels

def test_train_dev_split(input_data,output_data,train=0.7,dev=0.2,test=0.1):
    #make seed for exact results everything
    #random.sort(dataset)
    #np.random.seed(2)
    #np.random.shuffle(input_data)
    #np.random.shuffle(output_data)
    input_data, output_data = shuffle(input_data, output_data, random_state=0)
    split1=int(train*len(input_data))
    split2=int((train+dev)*len(input_data))
    train_input=input_data[:split1]
    dev_input=input_data[split1:split2]
    test_input=input_data[split2:]
    
    
    train_output=output_data[:split1]
    dev_output=output_data[split1:split2]
    test_output=output_data[split2:]
    
    return np.array(train_input),np.array(train_output),np.array(dev_input),np.array(dev_output),np.array(test_input),np.array(test_output)



def train_svm(datapath):
    x,y=load_small_img_dataset(datapath)
    X=np.ones((len(x),128),dtype=np.float64)
    for i,sample in enumerate(x):
        for j,val in enumerate(sample):
            X[i][j]=val

    persons=set(y)
    num_persons=len(set(y))
    categorical_mapping={}

    #convert into one hot encoding
    for i,name in enumerate(persons):
        #print(i,name)
        categorical_mapping[name]=i

    output_d=[]

    for i_ in y:
        i_=categorical_mapping[i_]
        output_d.append(i_)
    #output_d=to_categorical(output_d)

    oneHot2Name={}
    for i in categorical_mapping.keys():
        oneHot2Name[categorical_mapping[i]]=i
        
    np.save('oneHot2Name.npy',oneHot2Name)



    train_input,train_output,dev_input,dev_output,test_input,test_output=test_train_dev_split(X,output_d)






    '''
    #No need as dealing with encodings not images
    
    cv2.imshow(oneHot2Name[np.argmax(train_output[13])],train_input[13])
    cv2.waitKey()
    cv2.destroyAllWindows()
    
    # to convert our data type to float32 and normalize our database
    train_input=train_input.astype('float32')
    dev_input=dev_input.astype('float32')
    test_input=test_input.astype('float32')
    print(train_input.shape)
    print(test_input.shape)
    
    
    # Z-scoring or Gaussian Normalization
    train_input=train_input - np.mean(train_input) / train_input.std()
    dev_input=dev_input - np.mean(dev_input) / dev_input.std()
    test_input=test_input - np.mean(test_input) / test_input.std()
    categorical_mapping
    
    
    train_input=train_input/255.0
    dev_input=dev_input/255.0
    test_input=test_input/255.0'''
    

    # Train SVM classifier
    SVMmodel= SVC(C=1.0, kernel="linear", probability=True)
    SVMmodel.fit(train_input,train_output) 
    print("Model score: {0}".format(SVMmodel.score(test_input, test_output)))
    print("predicted: {0}    Actual: {1}".format(SVMmodel.predict(test_input),test_output))
    
    with open('SVMmodel.pickle', 'wb') as file:
        file.write(pickle.dumps(SVMmodel))
    
    return SVMmodel


def recognise_img_SVM(test_img,train=False):
    if train==True:
        model=train_svm('D:\dataset\Image\Face Dataset custom')
    else:
        with open('SVMmodel.pickle', 'rb') as file:
            model = pickle.loads(open('SVMmodel.pickle', "rb").read())
    oneHot2Name=np.load('oneHot2Name.npy').item()
    
    #Test Image
    embd=give_embedding(test_img)
    
    if len(embd)<1:
        return 'unknown'
    predicted=model.predict(np.array(embd).reshape(1,128))
    print("predicted: {0} ".format(oneHot2Name[predicted[0]]))
    #cv2.imshow(oneHot2Name[predicted[0]],cv2.imread(test_img))
    #cv2.waitKey()
    #cv2.destroyAllWindows()
    return oneHot2Name[predicted[0]]


# In[66]:


if __name__ == '__main__':
    test_img=cv2.imread('S_.jpg')
    x=recognise_img_SVM(test_img,train=True)
    print(x)


# In[ ]:





# In[ ]:




