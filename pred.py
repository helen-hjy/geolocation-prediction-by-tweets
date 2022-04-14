# -*- coding: utf-8 -*-
"""
Apply classifier on test file

"""
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
import time

from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
import ast

#read tfidf files
train_df_tfidf = pd.read_csv('train_tfidf.csv')
#split into label and tweet set
train_tweet_tfidf = train_df_tfidf['tweet']
train_labels = train_df_tfidf['region']
train_labels=train_labels.tolist()
#convert to 2D array
train_tweet_tfidf=[ast.literal_eval(row) for row in train_tweet_tfidf.tolist()]
#get number of words 
words=open("vocab.txt","r").readlines()
no_words=len(words) #2038 words
#process train feature set 
train_features=[]
for x in train_tweet_tfidf:
    each_instance = [0] * no_words  #initialize 1D-array of 0 for all words      
    for t in x:        
        each_instance[t[0]]=t[1]    #assign value to the list with the index
    train_features.append(each_instance)
    
#read test file
test_df_tfidf = pd.read_csv('test_tfidf.csv')
#split into label and tweet df
test_tweet_tfidf = test_df_tfidf['tweet']
test_label = test_df_tfidf['region']
test_label=test_label.tolist()
#convert to 2D array
test_tweet_tfidf=[ast.literal_eval(row) for row in test_tweet_tfidf.tolist()]
#process test feature set
test_features=[]
for x in test_tweet_tfidf:
    each_instance = [0] * no_words  #initialize 1D-array of 0 for all words      
    for t in x:        
        each_instance[t[0]]=t[1]    #assign value to the list with the index
    test_features.append(each_instance)

#get number of words 
words=open("vocab.txt","r").readlines()
no_words=len(words) #2038 words


#apply classifier
clf = MLPClassifier(activation='logistic',max_iter=2,learning_rate='adaptive')
clf.fit(train_features,train_labels)
y_pred=clf.predict(test_features)


#write id name
id_name=[x for x in range(1,len(test_features)+1)] 
#write prediction csv
output=pd.DataFrame({
    "id":id_name,
    "region":y_pred
    })
#export csv file
output.to_csv("pred_MLP.csv",index=False) #replace file name
print("output prediction file successfully!")
