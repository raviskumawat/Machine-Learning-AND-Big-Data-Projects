#!/usr/bin/env python
# coding: utf-8

import sys
import numpy as np
import os


# Terminate if no Test image is provided
if len(sys.argv)<=1:
    print("Test Image not provided.....EXITING...")
    quit()

datapath=os.getcwd()+sys.argv[1][1:]
print("Test Image location: ",datapath)


# Load pre-trained model
from keras.models import load_model
dnn_model=load_model('find_phone_dnn_model.h5')
cnn_model=load_model('find_phone_cnn_model.h5')

from keras import backend as K 

def euc_dist_keras(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_true - y_pred), axis=-1, keepdims=True))

# In[5]:

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
    #plt.show()

# In[6]:

# Load Pre-trained MobileNet for converting images into encodings
from keras.models import Model
from keras.applications import mobilenetv2
from keras.applications.mobilenetv2 import preprocess_input
model_mobile=mobilenetv2.MobileNetV2(input_shape=(224,224,3), include_top=True, weights='imagenet',classes=1000)
model2=Model(input=model_mobile.input,output=model_mobile.layers[-2].output)
model2.output_shape


# In[15]:


from PIL import Image
import matplotlib.pyplot as plt

img1=Image.open(datapath)
test_img=np.expand_dims(np.array(img1.resize((224,224))).astype('float'),axis=0)
test_input=model2.predict(preprocess_input(test_img))
loc=dnn_model.predict(test_input)
plt.title('DNN model: '+str(loc[0]))
visualize(img1,loc[0])
img1.save("DNN Output.jpg","JPEG")
plt.show()


img1_cnn=Image.open(datapath)
img_cnn=np.array(img1_cnn.resize((224,224))).astype('float')
img_cnn -= img_cnn.min() # shifted to 0..max
img_cnn *= 1 / img_cnn.max()
test_img_cnn=np.expand_dims(img_cnn,axis=0)
loc_cnn=cnn_model.predict(test_img_cnn)
plt.title('CNN model: '+str(loc_cnn[0]))
visualize(img1_cnn,loc_cnn[0])
img1_cnn.save("CNN Output.jpg","JPEG")
plt.show()

