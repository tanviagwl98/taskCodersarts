#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

data = pd.read_csv("data-final (1).csv", sep='\t')
print(data)
data.head

print(data['introelapse'].isnull().values.any())
print(data['testelapse'].isnull().values.any())

print(data['introelapse'].isnull().sum())
print(data['testelapse'].isnull().sum())

replacement= {'introelapse': 0, 'testelapse': 0, 'screenw' : 0, 'screenh' : 0}
data = data.fillna(value=replacement)

print(data['introelapse'].isnull().values.any())
print(data['testelapse'].isnull().values.any())

print(data['country'].isnull().values.any())
replacementc= {'country': 'Unknown'}
data = data.fillna(value=replacementc)
print(data['country'].isnull().values.any())

data.info

for col in data.columns: 
    print(col) 

from sklearn.model_selection import train_test_split

X=data.iloc[:,[101,102,103,104]].values
y=data.iloc[:,107].values

from sklearn.model_selection import train_test_split
x1,x2,y1,y2 =train_test_split(X, y, random_state=0, train_size =0.2)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, classification_report

KNN = KNeighborsClassifier(n_neighbors=1)
BNB = BernoulliNB()
LR = LogisticRegression()

KNN.fit(x1,y1)
y2_KNN_model = KNN.predict(x2)
print("KNN Accuracy :", accuracy_score(y2, y2_KNN_model))
#Accuracy: 0.31.....

LR.fit(x1,y1)
y2_LR_model = LR.predict(x2)
print("LR Accuracy :", accuracy_score(y2, y2_LR_model))
#Accuracy: 0.536...

BNB.fit(x1,y1)
y2_BNB_model = BNB.predict(x2)
print("BNB Accuracy :", accuracy_score(y2, y2_BNB_model))
#Accuracy: 0.537...
