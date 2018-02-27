
# coding: utf-8

# In[7]:


import numpy as np
import pandas as pd
import math


# In[14]:


def show_popular_15():
    metadata=pd.read_csv('spih_dataset_csv.csv')
    c=metadata['average_review_rating']
    c=[float(x) for x in c]
    c_tmp=[value for value in c if not math.isnan(value)]
    c_tmp=sum(c_tmp)/len(c_tmp)
    c=c_tmp
    m = metadata['number_of_reviews']
    m=[int(x) for x in m]
    m=np.percentile(m,90)
    q_items = metadata.copy().loc[metadata['number_of_reviews'] >= m]
    # Function that computes the weighted rating of each movie
    def weighted_rating(x, m=m, C=c):
        v = x['number_of_reviews']
        R = x['average_review_rating']
        # Calculation based on the IMDB formula
        return (v/(v+m) * R) + (m/(m+v) * C)
    q_items['score'] = q_items.apply(weighted_rating, axis=1)
    q_items = q_items.sort_values('score', ascending=False)
    print("printing the top 15 item details....\n\n\n")
    #print(q_items[['pid','product_name','number_of_reviews','average_review_rating','score']].head(15))
    return q_items['pid'].head(15)
    


# In[15]:


#print(show_popular_15())


# In[ ]:




