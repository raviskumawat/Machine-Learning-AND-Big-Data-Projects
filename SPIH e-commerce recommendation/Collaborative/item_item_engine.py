
# coding: utf-8

# In[8]:


import pandas as pd
import numpy as np
import math


# In[2]:


#import dataset optimised i.e. the items having atleast 20 bought items
data=pd.read_csv('spih_user_pid_item_item_binary.csv')
data.head()


# In[47]:


def cosine_similarity(data1, data2):
    numerator=np.dot(data1,data2)
    sum1=0
    sum2=0
    for i,j in zip(data1,data2):
        sum1+=i*i
        sum2+=j*j
    
    denominator=math.sqrt(sum1)*math.sqrt(sum2)
    cosim=numerator/denominator
    return cosim
    


# In[53]:


data1=data[data.columns[1]]
data2=data[data.columns[2]]
cosine_similarity(data1,data2)


# In[75]:



def find_bought_together_items(product_bought):
    data1=data[product_bought]
    similarities=[]
    #List of product names
    product_list=list(data)

    #Exclude the particular product bought
    product_list.remove(product_bought)

    #Create a dictionary
    d = {k:0 for k in product_list}
  
    for i in product_list[1:]:
        data2=data[i]
        cosim=cosine_similarity(data1,data2)
        d[i]=cosim
    
    d=sorted(d.items(), key=lambda x:x[1],reverse=True)
    return d


# In[135]:


product_bought='ACBECFGT6YXGSNU1'
similar20dict=find_bought_together_items(product_bought)[0:20]
similar20=[i[0] for i in similar20dict]
similar20




# In[136]:


#load item dataset and print the names of the items
metadata=pd.read_csv('spih_dataset_csv.csv')
metadata


# In[137]:


product_bought_data=metadata[metadata.pid==product_bought]
#product_bought_data=product_bought_data['pid','product_name']
print("Showing items to be bought with ")
print(product_bought_data['product_name']+"......")
print("........ \n\n")
for i in similar20:
    x=(metadata[metadata.pid==i])
    print(i+"  "+x['product_name'])
    

