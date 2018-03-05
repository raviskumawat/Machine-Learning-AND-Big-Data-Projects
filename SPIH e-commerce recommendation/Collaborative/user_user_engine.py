
# coding: utf-8

# In[9]:


import pandas as pd
import numpy as np
import math


# In[138]:


#import dataset optimised i.e. the items having atleast 20 bought items by each user and at least 6 co-ratings
data=pd.read_csv('spih_user_pid_user_user.csv')
data.head()


# In[158]:


user1=1
user2=2

def calculate_avg_rating(userid):
    x=data[data.uid==userid]
    #print(x)
    rating=0
    counter=0
    for ir in range(1,974):
        r=x[x.columns[ir]]
        #print(type(r))
        #z=r.astype(int)
        #print(r)
        #print(rating)
        if r.any()!=0:
            rating+=int(r)
            #print(rating)
            counter+=1
    return rating/counter

calculate_avg_rating(user1)
    
    


# In[140]:


user1=1
user2=2

def calculate_avg_corating(user1,user2):
    x1=data[data.uid==user1]
    x2=data[data.uid==user2]
    #print(x)
    rating1=0
    rating2=0
    counter=0
    for i in range(1,974):
        r1=x1[x1.columns[i]]
        r2=x2[x2.columns[i]]
        #print(type(r))
        #z=r.astype(int)
        #print(r)
        #print(rating)
        if r1.any()!=0 and r2.any()!=0:
            rating1+=int(r1)
            rating2+=int(r2)
            #print(rating)
            counter+=1
    print("Total co-ratings:"+str(counter))
    if counter==0:
        return 0
    return rating1/counter,rating2/counter

x,y=calculate_avg_corating(user1,2)
print(x,y)    


# In[142]:


user1=1
user2=2
#calculate the similarity between users p and q
def calculate_similarity(p,q):
    x1=data[data.uid==p]
    x2=data[data.uid==q]
    #print(x)
    sum_rating_p=0
    sum_rating_q=0
    sq_sum_rating_p=0
    sq_sum_rating_q=0
    counter=0
    numerator=0
    denominator=0
    print("calculating similarity between user "+str(p)+" and user "+str(q))
    avg_corated_p,avg_corated_q=calculate_avg_corating(p,q)
    for ic in range(1,974):
        r_pi=x1[x1.columns[ic]]
        r_qi=x2[x2.columns[ic]]
        #print(avg_corated_p,avg_corated_q)
        if r_pi.any()!=0 and r_qi.any()!=0:
            sum_rating_p+=int(r_pi)-avg_corated_p
            sum_rating_q+=int(r_qi)-avg_corated_q
            sq_sum_rating_p+=(int(r_pi)-avg_corated_p)*(int(r_pi)-avg_corated_p)
            sq_sum_rating_q+=(int(r_qi)-avg_corated_q)*(int(r_qi)-avg_corated_q)
            #print(r_pi,r_qi)
            #print(sum_rating_p,sum_rating_q,sq_sum_rating_p,sq_sum_rating_q)
            numerator+=float(sum_rating_p)*float(sum_rating_q)
            #print(numerator)
    denominator=math.sqrt(float(sq_sum_rating_p)*float(sq_sum_rating_q))
    #print(denominator)
    if denominator==0:
        return 0
    return numerator/denominator

x=calculate_similarity(3,2)
print("Similarity:"+str(x))    


# In[157]:


item=1
user=5
def calculate_predicted_rating(user,item):
    avg_rating_target_user=calculate_avg_rating(user)
    numerator=0
    denominator=0
    for ii in range(1,100):
        user_i_data=data[data.uid==ii]
        rating_useri_itemt=user_i_data[user_i_data.columns[item]]
        avg_rating_useri=calculate_avg_rating(ii)
        similarity=calculate_similarity(user,ii)
        print("Similarity:"+str(similarity)+"\n\n\n")
        if rating_useri_itemt.any()!=0 and i!=user:
            numerator+=(int(rating_useri_itemt)-avg_rating_useri)*(similarity)
            denominator+=similarity
    if denominator==0:
        t=0
    else:
        t=(numerator/denominator)
    
    predicted_rating=avg_rating_target_user+t    
    #print(rating_useri_itemt)
    return predicted_rating

print("Predicted rating for user u and item is: "+str(calculate_predicted_rating(user,item)))
    


