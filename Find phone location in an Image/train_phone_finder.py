#!/usr/bin/env python
# coding: utf-8

# Import modules
import os
from tqdm import tqdm
import sys

#TErminate if necessary Train folder is not provided
if len(sys.argv)<=1:
    print("Folder location not provided.....EXITING...")
    quit()


import os
datapath=os.getcwd()+sys.argv[1][1:]

print("Train Datapath: ",datapath)


from PIL import Image
import matplotlib.pyplot as plt

photo_loc_mapping={}
input_data=[]
labels=[]


def preprocess_photo(photopath,name):
    img=Image.open(photopath+'/'+name)
    input_data.append(img.resize((224,224)))
    labels.append(photo_loc_mapping[name])
    #plt.imshow(img)
    #plt.show()
    #img.show()
    return 
    


def get_data(datapath):
    
    # First load txt file
    for f in os.listdir(datapath):
        if f.endswith('.txt'):
            with open(datapath+'/'+f) as txt_file:
                lines=txt_file.readlines()
                #print(lines)
            for line in lines:
                line_data=line.split()
                photo_loc_mapping[line_data[0]]=[float(line_data[1]),float(line_data[2])]
        
        
    for f in os.listdir(datapath):  
        if f.endswith('.jpg'):
            preprocess_photo(datapath,f)        
    return 

get_data(datapath)


# visualize input_data
from PIL import ImageDraw
def visualize(img_arr,loc):
    width, height = img_arr.size
    #print(loc, img_arr.size)
    #print("Line 1: ",(0,int(float(loc[1])*height), width,int(float(loc[1])*height))
    #print("Line 2: ",(int(float(loc[0])*width),0,int(float(loc[0])*width),height))
    
    draw = ImageDraw.Draw(img_arr) 
    #print(0,int(float(loc[0])*width), height,int(float(loc[0])*width))
    #print(0,int(float(loc[1])*height), width,int(float(loc[1])*height))
    #print(int(float(loc[0])*width),0,int(float(loc[0])*width),height)
    
    x=int(float(loc[1])*height)
    y=int(float(loc[0])*width)
    r=int(loc[0]*0.05)+30
    #draw.line((0,x, width,x), fill=124,width=1)
    #draw.line((y,0,y,height), fill=124,width=1)
    
    print(x-r, y-r, x+r, y+r)
    draw.ellipse((y-r, x-r, y+r, x+r),outline ='blue')
    
    plt.imshow(img_arr)
    plt.show()

    
for photo,label in zip(input_data,labels):
    plt.title("Sample input")
    visualize(photo,label)
    break



from keras.activations import relu
from keras.callbacks import EarlyStopping,TensorBoard, ModelCheckpoint
from keras.layers import *
from keras.optimizers import adam
from keras.losses import mse
from keras.models import Sequential,Model
from keras.utils import to_categorical
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
#get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.utils import shuffle
import numpy as np


from keras.models import Model
from keras.applications import mobilenetv2
from keras.applications.mobilenetv2 import preprocess_input
model_mobile=mobilenetv2.MobileNetV2(input_shape=(224,224,3), include_top=True, weights='imagenet',classes=1000)
model2=Model(input=model_mobile.input,output=model_mobile.layers[-2].output)
print(model2.output_shape)


from keras.utils import to_categorical
from keras.utils import to_categorical
def test_train_dev_split(input_data, output_data, train=0.8, dev=0.1,
                         test=0.1):
    #make seed for exact results everything
    #input_data=preprocess_input(input_data)
    input_data, output_data = shuffle(input_data, output_data, random_state=0)
    split1 = int(train * len(input_data))
    split2 = int((train + dev) * len(input_data))
    train_input = input_data[:split1]
    dev_input = input_data[split1:split2]
    test_input = input_data[split2:]
    

    train_output = output_data[:split1]
    dev_output = output_data[split1:split2]
    test_output = output_data[split2:]
    
    
    train_input=model2.predict(preprocess_input(np.array([np.array(i.resize((224,224))) for i in train_input])))
    dev_input=model2.predict(preprocess_input(np.array([np.array(i.resize((224,224))) for i in dev_input])))
    test_input=model2.predict(preprocess_input(np.array([np.array(i.resize((224,224))) for i in test_input])))
    print(train_input[0])

    return train_input,np.array(train_output),dev_input, np.array(dev_output),test_input, np.array(test_output)

from keras import backend as K 

def euc_dist_keras(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_true - y_pred), axis=-1, keepdims=True))

from keras import backend as K 

def euc_dist_keras(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_true - y_pred), axis=-1, keepdims=True))


def phone_finder_model_dnn(input_data, output_data):

    train_in, train_out, dev_in, dev_out, test_in, test_out = test_train_dev_split(
        input_data, output_data)

    #https://arxiv.org/pdf/1509.05371v2.pdf
    
    print(np.array(train_in).shape)
    model=Sequential()
    model.add(Dense(64,activation='relu',input_shape=(1280,)))
    model.add(Dropout(0.4))
    
    '''model.add(Dense(256,activation='relu'))
    model.add(Dropout(0.4))
    model.add(BatchNormalization())
    
    model.add(Dense(128,activation='relu'))
    model.add(Dropout(0.4))'''
    
    model.add(Dense(8,activation='relu'))
    model.add(Dropout(0.4))
    
    model.add(Dense(2,activation='linear'))
    print(model.input_shape,model.output_shape)
    print(model.summary())

    '''
    model.compile(
        optimizer=adam(0.001),
        loss=[focal_loss(alpha=.25, gamma=2)],
        metrics=['accuracy'])
    
    '''
    model.compile(
        optimizer=adam(0.0001),
        loss='mse',
        metrics=['mae'])
    
    early = EarlyStopping(patience=100)
    
    check = ModelCheckpoint(
        'find_phone_dnn_model.h5',
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode='auto',
        period=1)

    model_history=model.fit(
        train_in,
        train_out,
        batch_size=32,
        callbacks=[early, check],
        validation_data=(dev_in, dev_out),
        epochs=5000)

    loss,mse = model.evaluate(test_in, test_out)
    print("Loss: {0}    MAE: {1}".format(loss, mse))
    
    plt.plot(model_history.history['loss'])
    plt.plot(model_history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    

    return model

dnn_model=phone_finder_model_dnn(input_data,labels)
print("DNN Model trained and Saved in the current Directory")



def test_train_dev_split2(input_data, output_data, train=0.8, dev=0.1,
                         test=0.1):
    #make seed for exact results everything
    #input_data=preprocess_input(input_data)
    input_data, output_data = shuffle(input_data, output_data, random_state=0)
    
    
    for num in range(0,len(input_data)):
        input_data[num]=np.array(input_data[num].resize((224,224))).astype('float')
        input_data[num] -= input_data[num].min() # shifted to 0..max
        input_data[num] *= 1 / input_data[num].max()
        
    
    split1 = int(train * len(input_data))
    split2 = int((train + dev) * len(input_data))
    train_input = input_data[:split1]
    dev_input = input_data[split1:split2]
    test_input = input_data[split2:]
    

    train_output = output_data[:split1]
    dev_output = output_data[split1:split2]
    test_output = output_data[split2:]
    print(train_input[0])

    return np.array(train_input),np.array(train_output),np.array(dev_input), np.array(dev_output),np.array(test_input), np.array(test_output)


def r2_keras(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

# Experiment 2 
"""
'Direct model: raw image classification'
 
"""
def phone_finder_model_cnn(input_data, output_data):

    train_in, train_out, dev_in, dev_out, test_in, test_out = test_train_dev_split2(
        input_data, output_data)

    #https://arxiv.org/pdf/1509.05371v2.pdf
    
    #print(np.array(train_in).shape)
    model=Sequential()
    model.add(Conv2D(5,110,activation='relu',input_shape=(224,224,3)))
    model.add(MaxPool2D())
    model.add(Dropout(0.4))
    
    '''model.add(Conv2D(9,55,activation='relu'))
    model.add(MaxPool2D())
    model.add(Dropout(0.4))'''
    
    model.add(Conv2D(3,27,activation='relu'))
    model.add(MaxPool2D())
    model.add(Dropout(0.4))
    
    
    model.add(Flatten())
    model.add(BatchNormalization())
              
    model.add(Dense(8,activation='relu'))
    model.add(Dropout(0.3))
              
    model.add(Dense(2,activation='linear'))
    
    print(model.input_shape,model.output_shape)
    print(model.summary())

    '''
    model.compile(
        optimizer=adam(0.001),
        loss=[focal_loss(alpha=.25, gamma=2)],
        metrics=['accuracy'])
    
    '''
    model.compile(
        optimizer=adam(0.0001),
        loss='mse',
        metrics=['mae'])
    
    early = EarlyStopping(patience=100)
    
    check = ModelCheckpoint(
        'find_phone_cnn_model.h5',
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode='auto',
        period=1)

    model_history=model.fit(
        train_in,
        train_out,
        batch_size=32,
        callbacks=[early, check],
        validation_data=(dev_in, dev_out),
        epochs=5000)

    loss,mse = model.evaluate(test_in, test_out)
    print("Loss: {0}    MSE: {1}".format(loss, mse))
    
    plt.plot(model_history.history['loss'])
    plt.plot(model_history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    

    return model

model=phone_finder_model_cnn(input_data,labels)
print("CNN Model trained and Saved in the current Directory")

'''

from PIL import Image
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
img=Image.open('test.jpg').resize((224,224))
test_img=np.expand_dims(np.array(img),axis=0)
#print(test_img.shape)
test_input=model2.predict(preprocess_input(test_img))
loc=model.predict(test_input)

visualize(img,loc[0])


# In[ ]:





# In[ ]:





# In[ ]:


from keras.models import load_model
predictor_model=load_model('find_phone_model.h5')
predictor_model.output_shape


# In[ ]:


from PIL import Image
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
img=Image.open('test.jpg').resize((224,224))
test_img=np.expand_dims(np.array(img),axis=0)
#print(test_img.shape)
test_input=model2.predict(preprocess_input(test_img))
loc=model.predict(test_input)
visualize(img,loc[0])


# In[ ]:




'''