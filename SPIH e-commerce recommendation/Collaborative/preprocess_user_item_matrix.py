
# coding: utf-8

# In[1]:



import numpy as np
import pandas as pd
import math



# In[29]:


metadata=pd.read_csv('user_item_rating_less.csv')


# In[30]:


metadata.head()


# In[31]:


metadata.describe()


# In[32]:


metadata=metadata.dropna(axis=1, how='all')


# In[67]:


metadata.describe()


# In[105]:


df=metadata.count()
df=df.to_frame()
df.columns=['name']
df=df.to_csv('count.csv')


# In[111]:


df=pd.read_csv('count_greater20.csv')
df


# In[117]:


items_rated=df['name']
items_rated


# In[121]:


metadata.filter(items=items_rated)


# In[122]:


metadata.to_csv('spih_user_pid.csv')

