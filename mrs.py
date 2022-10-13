#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import ast


# In[2]:


credits = pd.read_csv('tmdb_5000_credits.csv')


# In[3]:


movies=pd.read_csv('tmdb_5000_movies.csv')   


# In[4]:


movies.shape


# In[5]:


movies = movies.merge(credits,on='title')


# In[6]:


movies.head(1)


# In[7]:


# genres, id,  keywords
# overview, cast ,crew
movies=movies[['movie_id','title','overview','genres','keywords','cast','crew']]


# In[8]:


movies.head(1)


# In[9]:


movies.isnull().sum()


# In[10]:


# drop a null colum raw
movies.dropna(inplace=True)


# In[11]:


movies.duplicated().sum()


# In[12]:


movies.iloc[0].genres


# In[13]:


# '[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]'
# covert into
# ['Action','Adventure','Fantasy','Science Function']


# In[14]:


def convert(obj):
    L = []
    for i in ast.literal_eval(obj): # ast.literal_eval is used for convert in list
        L.append(i['name'])
    return L


# In[15]:


import ast
ast.literal_eval('[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]')


# In[16]:


movies['genres'] = movies['genres'].apply(convert)


# In[17]:


movies.head()


# In[18]:


movies['keywords']


# In[19]:


movies['keywords']=movies['keywords'].apply(convert)


# In[20]:


movies.head(2)


# 

# In[21]:


# for arranging the cast value
def convert3(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj): 
        if counter != 3:
            L.append(i['name'])
            counter+=1
        else:
            break
    return L


# In[22]:


movies['cast'] = movies['cast'].apply(convert3)


# In[23]:


movies.head()


# In[24]:


movies['crew'][0]


# In[25]:


# find director
def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj): 
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L


# In[26]:


movies['crew'] = movies['crew'].apply(fetch_director)


# In[27]:


movies.head(1)


# In[28]:


movies['overview']=movies['overview'].apply(lambda x:x.split())


# In[29]:


movies.head()


# In[ ]:





# In[30]:


# for removing the space
movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])


# In[31]:


movies.head()


# In[32]:


movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords']+ movies['cast']+movies['crew']


# In[33]:


movies.head()


# In[34]:


new_df = movies[['movie_id','title','tags']]


# In[35]:


new_df


# In[36]:


new_df['tags']=new_df['tags'].apply(lambda x: " ".join(x))


# In[37]:


new_df


# In[38]:


import nltk
from nltk.stem.porter import PorterStemmer
# used for [loving,loves] to [love,love]
ps = PorterStemmer()


# In[39]:


def stem(text):
    y = []
    
    for i in text.split():
        y.append(ps.stem(i))
        
    return " ".join(y)


# In[40]:


# testing purpose
ps.stem('dances')


# In[41]:


new_df['tags'] = new_df['tags'].apply(stem)


# In[42]:


new_df['tags'][0]


# In[43]:


#convert tags into lower case
new_df['tags']=new_df['tags'].apply(lambda x:x.lower())


# In[44]:


new_df.head()


# In[45]:


# Comparision the strings 

from sklearn.feature_extraction.text import CountVectorizer


# In[46]:


cv = CountVectorizer(max_features = 5000,stop_words='english')


# In[47]:


vectors = cv.fit_transform(new_df['tags']).toarray()


# In[48]:


vectors[0]


# In[49]:


cv.get_feature_names()


# In[50]:


# Finding the distance between two movies
from sklearn.metrics.pairwise import cosine_similarity


# In[51]:


cosine_similarity(vectors)


# In[52]:


similarity = cosine_similarity(vectors)


# In[53]:


similarity[0]


# In[54]:


#Finding the recomended movie
def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    for i in movie_list:
        print(new_df.iloc[i[0]].title)
        


# In[55]:


recommend('Batman Begins')


# In[56]:


new_df.iloc[1216].title


# In[57]:


# Fetching the movie data in website
import pickle


# In[62]:


#pickle.dump(new_df,open('movies.pkl','wb'))
pickle.dump(new_df.to_dict(),open('movie_dict.pkl','wb'))


# In[59]:


pickle.dump(similarity,open('similarity.pkl','wb'))


# In[ ]:




