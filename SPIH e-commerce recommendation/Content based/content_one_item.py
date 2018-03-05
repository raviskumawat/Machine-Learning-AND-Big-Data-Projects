
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


# In[19]:


metadata=pd.read_csv('spih_dataset_csv.csv')

#metadata.describe()
#metadata.head()


# In[20]:


#metadata['description'].head(15)


# In[22]:


#temp_category_tree=metadata['product_category_tree'].split('>>')
metadata['soup']=metadata['description']+metadata['product_name']+metadata['brand']+metadata['product_category_tree']
metadata['soup']=metadata['soup'].fillna('')
#metadata['soup']


# In[23]:


#Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
tfidf = TfidfVectorizer(stop_words='english')

#Replace NaN with an empty string
#metadata['description'] = metadata['description'].fillna('')

#Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(metadata['soup'])

#Output the shape of tfidf_matrix
#tfidf_matrix.shape


# In[24]:


cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)


# In[25]:



indices = pd.Series(metadata.index, index=metadata['pid']).drop_duplicates()


# In[26]:


# Function that takes in movie title as input and outputs most similar movies
def get_recommendations(title, cosine_sim=cosine_sim):
    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:21]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return metadata['pid'].iloc[movie_indices]


# In[27]:


#print(get_recommendations('SRTEH2FF9KEDEFGF'))

