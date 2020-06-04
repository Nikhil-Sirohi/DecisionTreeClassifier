#!/usr/bin/env python
# coding: utf-8

# In[18]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier


# In[19]:


df=pd.read_csv('diabetes.csv')
df.head()


# In[20]:


x=df.drop(['Outcome'],axis=1)
y=df['Outcome']


# In[21]:


#from sklearn.model_selection import train_test_split
#x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[22]:


#print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)


# In[29]:


dec=DecisionTreeClassifier(max_depth=5,criterion='entropy')
dec.fit(x,y)


# In[30]:


y_pred=dec.predict(x)


# In[31]:


from sklearn import metrics
print("Accuracy",metrics.accuracy_score(y,y_pred))


# In[ ]:





# In[ ]:




