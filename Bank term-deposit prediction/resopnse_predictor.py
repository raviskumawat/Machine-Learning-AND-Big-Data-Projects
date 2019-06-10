#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


# lOad CSV file
df=pd.read_csv('C:/Users/Ravi/Desktop/Upgrad assignment_ Bank model/bank-additional/bank-additional/bank-additional-full.csv',sep=';')
df.head()


# In[3]:


# Check if there are any NaN values present
df[df.isna().any(axis=1)]


# # Result from above analysis
# 
# job:  330 unknown values present
# 
# martial: 80 unknowns
#     
# education: 1731 unknowns
#     
# default: 8597 unknowns
#     
# housing: 990 unknowns
#     
# loan: 990 unknowns
#     
# 
# 

#  Input variables:
#  
#  
#    # bank client data:
#    1 - age (numeric)
#     
#    2 - job : type of job (categorical: "admin.","blue-collar","entrepreneur","housemaid","management","retired","self-employed","services","student","technician","unemployed","unknown")
#     
#    3 - marital : marital status (categorical: "divorced","married","single","unknown"; note: "divorced" means divorced or widowed)
#     
#    4 - education (categorical: "basic.4y","basic.6y","basic.9y","high.school","illiterate","professional.course","university.degree","unknown")
#      
#      
#    5 - default: has credit in default? (categorical: "no","yes","unknown")
#     
#    6 - housing: has housing loan? (categorical: "no","yes","unknown")
#    
#    7 - loan: has personal loan? (categorical: "no","yes","unknown")
#    
#    
#    # related with the last contact of the current campaign:
#    8 - contact: contact communication type (categorical: "cellular","telephone") 
#     
#    9 - month: last contact month of year (categorical: "jan", "feb", "mar", ..., "nov", "dec")
#     
#   10 - day_of_week: last contact day of the week (categorical: "mon","tue","wed","thu","fri")
#    
#   11 - duration: last contact duration, in seconds (numeric). Important note:  this attribute highly affects the output target (e.g., if duration=0 then y="no"). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.
#   
#    
#    # other attributes:
#   12 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
#    
#   13 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
#   
#   14 - previous: number of contacts performed before this campaign and for this client (numeric)
#    
#   15 - poutcome: outcome of the previous marketing campaign (categorical: "failure","nonexistent","success")
#    
#    # social and economic context attributes
#   16 - emp.var.rate: employment variation rate - quarterly indicator (numeric)
#    
#   17 - cons.price.idx: consumer price index - monthly indicator (numeric)    
#   
#   18 - cons.conf.idx: consumer confidence index - monthly indicator (numeric)  
#   
#   19 - euribor3m: euribor 3 month rate - daily indicator (numeric)
#   
#   20 - nr.employed: number of employees - quarterly indicator (numeric)
#   
# 
#   Output variable (desired target):
#   21 - y - has the client subscribed a term deposit? (binary: "yes","no")
# 

# In[4]:


# check all data column-wise for any discrepencies

for i in df.columns:
    print('[COLUMN]: ',i)
    print(df[i].value_counts())
    print('........\n\n\n')


# In[5]:


# Let's try using 'unknown' as a class name

# Problem:   'default class bias' : 35.5k yes vs 3 no  [TRY excluding and check if accuracy increases?]



# Convert all categorical classes into numerical ones
for i in df.columns:
    print(df[i].dtype.name)
    if df[i].dtype.name in ['category','object']:
        print('Converting [{0}] into categories: '.format(i))
        df[i]=df[i].astype('category')
        df[i]=df[i].cat.codes
        print('........\n\n\n')
    
df.head()


# In[6]:


df.tail()

'''# Clean data:
 Example pdays=999 means the person was never contacted before
 Exclude 'duration' attribute as it is not Known beforehand while bilding a predictive model
'''
# In[7]:


# Normalize data:
for i in df.columns:
    print(df[i].dtype.name)
    
    if df[i].dtype.name in ['int64','float64','int8'] and i!='y':
        # Normalize Data
        print('Normalizing [{0}] : '.format(i))
        df[i]=(df[i]-df[i].mean())/df[i].std()


df.head()


# In[8]:


# Convert data into numpy 'input and labels array'
input_data=[i[:10]+i[11:-1] for i in df.values.tolist()] #Exclude 'duration' attribute
labels=[int(i[-1]) for i in df.values.tolist()]
input_data[0]


# In[ ]:





# In[9]:


from collections import Counter
Counter(labels),input_data[0]


# In[10]:


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
#get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle

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

    return np.array(train_input),np.array(train_output),np.array(dev_input), np.array(dev_output),np.array(test_input), np.array(test_output)


# In[ ]:





# In[12]:


# Model experiment-1:
'''
* pdays=999 [normalized to 0-1]
* Inculde 'default' attribute
'''
def response_predictor_model(input_data, output_data):

    train_in, train_out, dev_in, dev_out, test_in, test_out = test_train_dev_split(
        input_data, output_data)

    #https://arxiv.org/pdf/1509.05371v2.pdf
    
    print(np.array(train_in).shape)
    
    model=Sequential()
    model.add(Dense(32,activation='relu',input_shape=(19,)))
    model.add(Dropout(0.4))
    '''model.add(Dense(200,activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(128,activation='relu'))
    model.add(Dropout(0.4))
    '''
    model.add(Dense(16,activation='relu'))
    model.add(Dropout(0.4))
    
    model.add(Dense(1,activation='sigmoid'))
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
        loss='binary_crossentropy',
        metrics=['accuracy'])
    
    early = EarlyStopping(patience=50)
    
    check = ModelCheckpoint(
        'find_phone_model1.h5',
        monitor='val_acc',
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode='auto',
        period=1)

    model_history=model.fit(
        train_in,
        train_out,
        batch_size=256,
        callbacks=[early, check],
        validation_data=(dev_in, dev_out),
        epochs=500)

    loss, acc = model.evaluate(test_in, test_out)
    print("Loss: {0}    Accuracy: {1}".format(loss, acc))
    
    plt.plot(model_history.history['loss'])
    plt.plot(model_history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
    print("CONFUSION MATRIX:")
    y_pred = model.predict(test_in)
    
    y_pred = np.argmax(y_pred, axis=1)
    print("y_pred: ",y_pred)
    #print(y_pred)
    #print(y_pred.shape)
    #print("Test_out:",test_out)
    y_true =test_out
    print("Y_true:",y_true)
    array = confusion_matrix(y_true, y_pred)
    #df_cm = pd.DataFrame(array)
    df_cm = pd.DataFrame(array, index = ['No','YES'],columns = ['No','YES'])
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True)
    
    return model

model=response_predictor_model(input_data,labels)
# Random Basline model: 88.7% 


# In[13]:


# Model experiment-2:
'''
* Enculde 'default' attribute
* Try Different metrics such as AUC, precision, recall, F1_score
* Use class_weights to counter for class imbalancing 

'''
# Convert data into numpy 'input and labels array'
input_data2=[i[:4]+i[5:10]+i[11:-1] for i in df.values.tolist()] #Exclude 'duration' attribute
labels2=[int(i[-1]) for i in df.values.tolist()]
#input_data2[0]

from sklearn.utils import class_weight
class_weights = class_weight.compute_class_weight('balanced',np.unique(labels2),labels2)
#class_weights

def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision
    
def f1(y_true, y_pred):
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))



import tensorflow as tf
def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc

def response_predictor_model2(input_data, output_data):

    train_in, train_out, dev_in, dev_out, test_in, test_out = test_train_dev_split(
        input_data, output_data)

    #https://arxiv.org/pdf/1509.05371v2.pdf
    
    print(np.array(train_in).shape)
    
    model=Sequential()
    model.add(Dense(64,activation='relu',input_shape=(18,)))
    model.add(Dropout(0.4))
    '''model.add(Dense(200,activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(128,activation='relu'))
    model.add(Dropout(0.4))
    '''
    model.add(Dense(32,activation='relu'))
    model.add(Dropout(0.4))
    
    model.add(Dense(1,activation='sigmoid'))
    print(model.input_shape,model.output_shape)
    print(model.summary())

    '''
    model.compile(
        optimizer=adam(0.001),
        loss=[focal_loss(alpha=.25, gamma=2)],
        metrics=['accuracy'])
    
    '''
    model.compile(
        optimizer=adam(0.001),
        loss='binary_crossentropy',
        metrics=[recall])
    
    early = EarlyStopping(patience=50)
    
    check = ModelCheckpoint(
        'find_phone_model2_f1.h5',
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode='auto',
        period=1)

    model_history=model.fit(
        train_in,
        train_out,
        batch_size=256,
        callbacks=[early, check],
        validation_data=(dev_in, dev_out),
        class_weight=class_weights,
        epochs=500)

    loss, acc = model.evaluate(test_in, test_out)
    print("Loss: {0}    Accuracy: {1}".format(loss, acc))
    
    plt.plot(model_history.history['loss'])
    plt.plot(model_history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
    print("CONFUSION MATRIX:")
    y_pred = model.predict(test_in)
    
    y_pred = np.argmax(y_pred, axis=1)
    print("y_pred: ",y_pred)
    #print(y_pred)
    #print(y_pred.shape)
    #print("Test_out:",test_out)
    y_true =test_out
    print("Y_true:",y_true)
    array = confusion_matrix(y_true, y_pred)
    #df_cm = pd.DataFrame(array)
    df_cm = pd.DataFrame(array, index = ['No','YES'],columns = ['No','YES'])
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True)
    
    return model
model=response_predictor_model2(input_data2,labels2)
# Random Basline model: 88.7% 


# In[14]:


# SVM model 
from sklearn.model_selection import train_test_split

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(input_data, labels, test_size=0.1,random_state=109)
# 70% training and 30% test
Counter(y_train)

from sklearn import svm

#Create a svm Classifier
clf = svm.SVC(C=1,kernel='rbf',class_weight='balanced') # Rbf Kernel

#Train the model using the training sets
clf.fit(X_train, y_train)


# ![](https://cdn-images-1.medium.com/max/960/1*pOtBHai4jFd-ujaNXPilRg.png)
# 

# In[15]:


#Predict the response for test dataset
y_pred = clf.predict(X_test)



from sklearn import metrics

# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# Model Precision: what percentage of responses are classified correctly?
print("Precision:",metrics.precision_score(y_test, y_pred))

# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:",metrics.recall_score(y_test, y_pred))


array = confusion_matrix(y_test, y_pred)
#df_cm = pd.DataFrame(array)
df_cm = pd.DataFrame(array, index = ['No','YES'],columns = ['No','YES'])
plt.figure(figsize=(10, 7))
sn.heatmap(df_cm, annot=True)
    

from sklearn import metrics

y_preds = clf.predict(X_test)

# --classification report --
print(metrics.classification_report(y_test, y_preds, labels=[0,1]))


# Save the SVM model
import pickle
pickle.dump(clf, open('svm_model_balanced', 'wb'))


# In[ ]:


'''

We can predict a prospect beforehand and thus only call only those customers who are likely to opt for a term deposit, 
Thus saving the cost of calling customers which would not have opted for a term deposit. 

Thus as can be seen above, we a total of around 7400(Positives)=4600(False positives)+2800(True positives) 

In random calling:    we get a total of 4640(True positives) out of 41,188 calls
In Selective calling: we get a total of 2800(True Positive) out of 7400 calls 

Financial Benefit: 
Thus the model enables us to give 60.3% of the prospects in just ~17.9% cost and resources as compared to random calling.

'''


# In[ ]:




