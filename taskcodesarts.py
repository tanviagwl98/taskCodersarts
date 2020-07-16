#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data = pd.read_csv("data-final (1).csv", sep='\t')
print(data)


# In[15]:


data.head


# In[16]:


print(data['introelapse'].isnull().values.any())
print(data['testelapse'].isnull().values.any())


# In[5]:


print(data['introelapse'].isnull().sum())
print(data['testelapse'].isnull().sum())


# In[6]:


replacement= {'introelapse': 0, 'testelapse': 0, 'screenw' : 0, 'screenh' : 0}
data = data.fillna(value=replacement)


# In[17]:


print(data['introelapse'].isnull().values.any())
print(data['testelapse'].isnull().values.any())


# In[18]:


data.info


# In[19]:


count = 0
for col in data.columns: 
    print(col) 
    count = count+1
print(count)    


# In[20]:


data.columns.get_loc("introelapse")


# In[21]:


print(data['country'].isnull().values.any())
replacement= {'country': 'Unknown'}
data = data.fillna(value=replacement)


# In[22]:


print(data['country'].isnull().values.any())


# In[23]:


from sklearn.model_selection import train_test_split

X=data.iloc[:,[101,102,103,104]].values
y=data.iloc[:,107].values




# In[24]:


from sklearn.model_selection import train_test_split
x1,x2,y1,y2 =train_test_split(X, y, random_state=0, train_size =0.2)


# In[25]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression


# In[26]:


from sklearn.metrics import accuracy_score, classification_report

KNN = KNeighborsClassifier(n_neighbors=1)
BNB = BernoulliNB()
LR = LogisticRegression()


# In[27]:


KNN.fit(x1,y1)
y2_KNN_model = KNN.predict(x2)
print("KNN Accuracy :", accuracy_score(y2, y2_KNN_model))


# In[ ]:


LR.fit(x1,y1)
y2_LR_model = LR.predict(x2)
print("LR Accuracy :", accuracy_score(y2, y2_LR_model))


# In[ ]:


BNB.fit(x1,y1)
y2_BNB_model = BNB.predict(x2)
print("BNB Accuracy :", accuracy_score(y2, y2_BNB_model))


# In[ ]:





# In[ ]:




