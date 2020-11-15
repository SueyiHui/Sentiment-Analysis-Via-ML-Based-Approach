#!/usr/bin/env python
# coding: utf-8

# # MMA865 Individual Assignment


"""
# [Shuyi, Hui]
# [20198085]
# [MMA]
# [2021W]
# [MMA869]
# [09/28/2020]

"""

# ## Submission to Question [2], Part [a]
# ### Load, Explore & Clean the Dataset

import pandas as pd
import numpy as np

import re, string, unicodedata
import nltk
from bs4 import BeautifulSoup
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier
import nltk


# ### Load & Clean

df = pd.read_csv("C:\\Users\\user\\sentiment_train.csv")
df.info()
df.head()


df_test = pd.read_csv("sentiment_test.csv")
df.info()
df.head()


# Check the balance of the combined data in train dataset
np.bincount(df['Polarity'])
## Polarity ratio is balanced [1213, 1187],which indecates data is not heavily imbalanced


# Drop useless content in "Sectence" 
df.drop(df.index[df['Sentence'] == '#NAME?'], inplace = True)
df.drop(df.index[df['Sentence'] == '10-Oct'], inplace = True)


# ### Train & Test Data Setup

#Train Dataset
x_train = df['Sentence']
y_train = df['Polarity']

#Test Dataset
x_test = df_test['Sentence']
y_test = df_test['Polarity']

type(x_train)
x_train.shape
x_train.head()

type(y_train)
y_train.shape
y_train.head()


# ### Preprocessing

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import unidecode
import textstat
import string  

lemmer = WordNetLemmatizer()

def preprocess(doc):
    
    # Convert all to lowercase
    doc = doc.lower()
    
    # Replace URL with URL string
    doc = re.sub(r'http\S+', 'URL', doc)
    
    # Replace AT with AT string
    doc = re.sub(r'@', 'AT', doc)
    
    # Replace all numbers/digits with the string NUM
    doc = re.sub(r'\b\d+\b', 'NUM', doc)
    
    #Remove numbers
    doc = re.sub(r'\d+',' ', doc) 
        
    #Removing a Single Character
    doc = re.sub(r"\s+[a-zA-Z]\s+", " ", doc)
    
    #Replace every stand-alone "i" into "I"
    doc = re.sub(r'\si\s', " I ", doc)
    
    # Remove extra newlines
    doc = re.sub(r'[\r|\n|\r\n]+', ' ',doc)
    
    # Remove .,!,-    
    doc = re.sub('[.!-]', '', doc)
    
    # Lemmatize each word.
    doc = ' '.join([lemmer.lemmatize(w) for w in doc.split()])

    return doc


# ## Submission to Question [2], Part [b]
# ### Construct the Pipeline

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_extraction import stop_words
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import FunctionTransformer
from sklearn.decomposition import NMF
from sklearn.neural_network import MLPClassifier

# Need to preprocess the stopwords, because scikit learn's TfidfVectorizer
# removes stopwords _after_ preprocessing
stop_words = [preprocess(word) for word in stop_words.ENGLISH_STOP_WORDS]

# This vectorizer will be used to create the BOW features
vectorizer = TfidfVectorizer(preprocessor=preprocess, 
                             analyzer = 'char_wb',
                             binary = 1,
                             max_features = 1000, 
                             ngram_range=[1,3],
                             stop_words=None,
                             strip_accents="unicode", 
                             lowercase=False, max_df=0.45,use_idf=True)

dt = DecisionTreeClassifier(criterion = 'entropy', max_depth = 1000,presort=True, random_state=42, max_leaf_nodes=60)
rf = RandomForestClassifier(max_features = 0.05, criterion = 'entropy',n_estimators = 300, max_depth = 120, random_state=42)
knn = KNeighborsClassifier(n_neighbors = 20,weights = 'distance', leaf_size = 120, p = 2, metric = 'euclidean' )
xgboost = XGBClassifier(n_estimators = 55, max_depth = 150, Learning_rate = 0.0009, gamma = 0.05 )
reg = LogisticRegression(verbose = 2, penalty = 'l2', solver = 'liblinear', max_iter = 10, tol = 1e-5, C = 45, random_state=42 )
adboost = AdaBoostClassifier(n_estimators = 200, learning_rate = 0.8, algorithm = 'SAMME.R' )
gbt = GradientBoostingClassifier(n_estimators = 100, learning_rate = 0.8, loss = 'deviance',criterion = 'friedman_mse')
mlp = MLPClassifier(random_state=42, verbose=2, max_iter=200)
nmf = NMF(n_components=25, random_state=1, init='nndsvda', solver='mu', alpha=.1, l1_ratio=.5)


pipe = Pipeline([('cv', vectorizer),('clf',reg)])


#reg 0.71167 0.75000 0.70000    0.75000
#dt 0.62667 0.62833 0.60000    0.66833
#rf 0.71167 0.72167 0.68167    0.71167
#knn 0.69167 0.72000 0.68000   0.72333
#xgb 0.70167 0.71167 0.66333   0.72167 
#adboost 0.69333 0.69667 0.63667    0.68000
#gbt 0.69833 0.68500 0.62667     0.68667
#mlp 0.72333 0.73667 0.68667      0.74167


# ## Submission to Question [2], Part [c]
# ### Fit Model

pipe.fit(x_train, y_train)


# ## Submission to Question [2], Part [d]
# ### Estimate Model Performance on Test Data

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, f1_score, accuracy_score

pred_val = pipe.predict(x_test)

print("Confusion matrix:")
print(confusion_matrix(y_test, pred_val))

print("\nF1 Score = {:.5f}".format(f1_score(y_test, pred_val, average='micro')))
print("\nAccuracy = {:.5f}".format(accuracy_score(y_test, pred_val)))

print("\nClassification Report:")
print(classification_report(y_test, pred_val))


# ### Export Result to File

pred_test = pipe.predict(df_test['Sentence'])

# Export predictions 
output = pd.DataFrame({'Sentence': df_test.Sentence, 'Polarity' : df_test.Polarity, 'predicted': pred_test})
output.head()
#output.to_csv('C:\\Users\\user\\Result865.csv', index=False)
