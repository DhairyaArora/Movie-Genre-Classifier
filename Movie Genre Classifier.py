#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import json
import nltk
import re
import csv
import matplotlib.pyplot as plt 
import seaborn as sns
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

get_ipython().run_line_magic('matplotlib', 'inline')
pd.set_option('display.max_colwidth', 300)


# In[29]:


meta = pd.read_csv(r"C:\Users\vibhu\Downloads\MovieSummaries\movie.metadata.tsv", sep = '\t', header = None)
meta.head()


# In[30]:


meta.columns = ["movie_id",1,"movie_name",3,4,5,6,7,"genre"]


# In[49]:


plots = []

with open(r"C:\Users\vibhu\Downloads\MovieSummaries\plot_summaries.txt", 'r',encoding="utf8") as f:
       reader = csv.reader(f, dialect='excel-tab') 
       for row in tqdm(reader):
            plots.append(row)


# In[50]:


movie_id = []
plot = []

# extract movie Ids and plot summaries
for i in tqdm(plots):
  movie_id.append(i[0])
  plot.append(i[1])

# create dataframe
movies = pd.DataFrame({'movie_id': movie_id, 'plot': plot})


# In[51]:


movies.head()


# In[52]:


meta['movie_id'] = meta['movie_id'].astype(str)

# merge meta with movies
movies = pd.merge(movies, meta[['movie_id', 'movie_name', 'genre']], on = 'movie_id')

movies.head()


# In[53]:


movies['genre'][0]


# In[54]:


type(json.loads(movies['genre'][0]))


# In[55]:


json.loads(movies['genre'][0]).values()


# In[56]:


genres = [] 

# extract genres
for i in movies['genre']: 
  genres.append(list(json.loads(i).values())) 

# add to 'movies' dataframe  
movies['genre_new'] = genres


# In[57]:


# remove samples with 0 genre tags
movies_new = movies[~(movies['genre_new'].str.len() == 0)]


# In[58]:


movies_new.shape, movies.shape


# In[59]:


movies.head()


# In[60]:


# get all genre tags in a list
all_genres = sum(genres,[])
len(set(all_genres))


# In[61]:



all_genres = nltk.FreqDist(all_genres) 

# create dataframe
all_genres_df = pd.DataFrame({'Genre': list(all_genres.keys()), 
                              'Count': list(all_genres.values())})


# In[62]:


g = all_genres_df.nlargest(columns="Count", n = 50) 
plt.figure(figsize=(12,15)) 
ax = sns.barplot(data=g, x= "Count", y = "Genre") 
ax.set(ylabel = 'Count') 
plt.show()


# In[63]:


# function for text cleaning
def clean_text(text):
    # remove backslash-apostrophe
    text = re.sub("\'", "", text)
    # remove everything alphabets
    text = re.sub("[^a-zA-Z]"," ",text)
    # remove whitespaces
    text = ' '.join(text.split())
    # convert text to lowercase
    text = text.lower()
    
    return text


# In[64]:


movies_new['clean_plot'] = movies_new['plot'].apply(lambda x: clean_text(x))


# In[65]:


movies_new[['plot', 'clean_plot']].sample(3)


# In[66]:


def freq_words(x, terms = 30):
  all_words = ' '.join([text for text in x])
  all_words = all_words.split()
  
  fdist = nltk.FreqDist(all_words)
  words_df = pd.DataFrame({'word':list(fdist.keys()), 'count':list(fdist.values())})
  
  # selecting top 20 most frequent words
  d = words_df.nlargest(columns="count", n = terms) 
  plt.figure(figsize=(12,15))
  ax = sns.barplot(data=d, x= "count", y = "word")
  ax.set(ylabel = 'Word')
  plt.show()


# In[67]:


# print 100 most frequent words
freq_words(movies_new['clean_plot'], 100)


# In[68]:


nltk.download('stopwords')


# In[69]:


from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

# function to remove stopwords
def remove_stopwords(text):
    no_stopword_text = [w for w in text.split() if not w in stop_words]
    return ' '.join(no_stopword_text)
  
movies_new['clean_plot'] = movies_new['clean_plot'].apply(lambda x: remove_stopwords(x))


# In[70]:


freq_words(movies_new['clean_plot'], 100)


# In[71]:


from sklearn.preprocessing import MultiLabelBinarizer

multilabel_binarizer = MultiLabelBinarizer()
multilabel_binarizer.fit(movies_new['genre_new'])

# transform target variable
y = multilabel_binarizer.transform(movies_new['genre_new'])


# In[72]:


tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=10000)


# In[73]:


# split dataset into training and validation set
xtrain, xval, ytrain, yval = train_test_split(movies_new['clean_plot'], y, test_size=0.2, random_state=9)


# In[74]:


# create TF-IDF features
xtrain_tfidf = tfidf_vectorizer.fit_transform(xtrain)
xval_tfidf = tfidf_vectorizer.transform(xval)


# In[75]:


from sklearn.linear_model import LogisticRegression

# Binary Relevance
from sklearn.multiclass import OneVsRestClassifier

# Performance metric
from sklearn.metrics import f1_score


# In[76]:


lr = LogisticRegression()
clf = OneVsRestClassifier(lr)


# In[77]:


# fit model on train data
clf.fit(xtrain_tfidf, ytrain)


# In[78]:



# make predictions for validation set
y_pred = clf.predict(xval_tfidf)


# In[79]:


y_pred[3]


# In[80]:


multilabel_binarizer.inverse_transform(y_pred)[3]


# In[81]:


# evaluate performance
f1_score(yval, y_pred, average="micro")


# In[82]:



# predict probabilities
y_pred_prob = clf.predict_proba(xval_tfidf)


# In[83]:


t = 0.3 # threshold value
y_pred_new = (y_pred_prob >= t).astype(int)


# In[84]:


# evaluate performance
f1_score(yval, y_pred_new, average="micro")


# In[85]:


def infer_tags(q):
    q = clean_text(q)
    q = remove_stopwords(q)
    q_vec = tfidf_vectorizer.transform([q])
    q_pred = clf.predict(q_vec)
    return multilabel_binarizer.inverse_transform(q_pred)


# In[86]:


for i in range(5):
    k = xval.sample(1).index[0]
    print("Movie: ", movies_new['movie_name'][k], "\nPredicted genre: ", infer_tags(xval[k])), print("Actual genre: ",movies_new['genre_new'][k], "\n")

