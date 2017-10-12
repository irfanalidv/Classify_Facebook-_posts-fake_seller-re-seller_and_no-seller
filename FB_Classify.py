
# coding: utf-8

# In[1]:

##Loading all the required libraries
import pandas as pd
import numpy as np

from sklearn.naive_bayes import MultinomialNB,GaussianNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import time

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.lancaster import LancasterStemmer
stem = LancasterStemmer()

#library for regular expretion
import re

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# In[2]:

Fb_Data1=pd.read_csv("FB_User_Classification.csv", delimiter="\t")


# In[3]:

df=Fb_Data1[['description', 'INDEX New']]
df.columns=['description', 'CATEGORY']
df['description'].replace('', np.nan, inplace=True)
df.dropna(subset=['description'], inplace=True)
df['CATEGORY'].replace('NaN', np.nan, inplace=True)
df.dropna(subset=['CATEGORY'], inplace=True)
df.head()


# In[4]:

# function to clean data
#without cleaning the accuracy is more
#try without once
stops = set(stopwords.words("english")) #Removing stop words
def cleanData(text, lowercase = False, remove_stops = False, stemming = False):
    txt = str(text)
    txt = re.sub(r'[^A-Za-z\s]',r'',txt)
    #Removing non-alpha numeric characters
    txt = re.sub(r'\n',r' ',txt)    
    if lowercase:
        txt = " ".join([w.lower() for w in txt.split()])
        
    if remove_stops:
        txt = " ".join([w for w in txt.split() if w not in stops])
    
    if stemming:
        stemmer = LancasterStemmer()
        txt = " ".join([stemmer.stem(w) for w in txt.split()])

    return txt


# In[5]:

df['description'] = df['description'].map(lambda x: cleanData(x, lowercase=True, remove_stops=True, stemming=True))


# In[6]:

df['description']


# In[7]:

#converting category column into numeric target NUM_CATEGORY column
df['NUM_CATEGORY']=df.CATEGORY.map({'No Seller':0,'Reseller':1,'Fake Seller':2})


# In[8]:

#df.drop('CATEGORY', axis=1)


# In[9]:

#used in model 1,2
x=df['description']
y=df['NUM_CATEGORY']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=.85)

vect =CountVectorizer(ngram_range=(2,2))
#converting features into numeric vector
X_train = vect.fit_transform(x_train)
#converting target into numeric vector
X_test = vect.transform(x_test)


# In[10]:

#Training and Predicting the data

mnb = MultinomialNB(alpha=0.2)
mnb.fit(X_train,y_train)
result= mnb.predict(X_test)

accuracy_score(result,y_test)


# In[11]:

svc = SVC(kernel = 'linear')
svc.fit(X_train,y_train)
result_svc= svc.predict(X_test)

accuracy_score(result_svc,y_test)

