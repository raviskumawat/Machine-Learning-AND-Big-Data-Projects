
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import math


# In[2]:


#import dataset optimised i.e. the items having atleast 20 bought items by each user and at least 6 co-ratings
data=pd.read_csv('spih_user_pid_user_user.csv')
data.head()


# In[36]:


item1=1
item2=2

def calculate_avg_difference(item1,item2):
    x1=data[data.columns[item1]]
    x2=data[data.columns[item2]]
    #x1=[5,3,0]
    #x2=[3,4,2]
    rating=0
    counter=0
    diff=0
    for ir in range(1,942):
        r1=x1[ir]
        r2=x2[ir]
        #print(r)
        #z=r.astype(int)
        #print(r)
        #print(rating)
        if r1.any()!=0 and r2.any()!=0:
            diff+=r1-r2
            #print(rating)
            counter+=1
    if(counter==0):
        return (0,0)
    return diff/counter,counter

calculate_avg_difference(1,2)
    
    


# In[65]:


item=3
user=2
#user_i_data=data[data.uid==user]
#rating_useri_itemt=user_i_data[user_i_data.columns[item]]
#print(rating_useri_itemt)
def calculate_predicted_rating(user,item):
    numerator=0
    denominator=0
    predicted_rating=0
    for ii in range(1,974):
        user_data=data[data.uid==user]
        rating_item_i=user_data[user_data.columns[ii]]
        #print(rating_item_i)
        avg_diff,count=calculate_avg_difference(item,ii)
        if(rating_item_i.any()!=0 and item!=ii):
            numerator+=(avg_diff+rating_item_i)*count
            denominator+=count
    predicted_rating=numerator/denominator 
    return predicted_rating

print("Predicted rating for user u and item is: "+str(calculate_predicted_rating(user,item)))
    


