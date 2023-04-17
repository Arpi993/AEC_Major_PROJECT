#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd    # a wonderful dataframe to work with
import numpy as np     # adding a number of mathematical and science functions
import seaborn as sns  # a very easy to use statistical data visualization package
import matplotlib.pyplot as plt # a required plotting tool
import warnings
# sklearn is a big source of pre-written and mostly optimized ML algorithms.
# Here we use their Decision trees, Support Vector Machines, and the classic Perceptron. 
from sklearn import preprocessing, svm   
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
#ignore warnings
warnings.filterwarnings('ignore')




# In[9]:


data = pd.read_csv('..\Downloads\stu_performance.csv')
data.head()


# In[8]:


data.tail()


# In[10]:


ax = sns.countplot(x='Class', data=data, order=['L', 'M', 'H'])
for p in ax.patches:
    ax.annotate('{:.2f}%'.format((p.get_height() * 100) / len(data)), (p.get_x() + 0.24, p.get_height() + 2))
plt.show()


# In[10]:


fig, axarr  = plt.subplots(2,figsize=(10,10))
sns.countplot(x='gender', data=data, order=['M','F'], ax=axarr[0])
sns.countplot(x='gender', hue='Class', data=data, order=['M', 'F'],hue_order = ['L', 'M', 'H'], ax=axarr[1])
plt.show()


# In[11]:


fig, axarr  = plt.subplots(2,figsize=(10,10))
sns.countplot(x='NationalITy', data=data, ax=axarr[0])
sns.countplot(x='NationalITy', hue='Class', data=data,hue_order = ['L', 'M', 'H'], ax=axarr[1])
plt.show()


# In[12]:


fig, axarr  = plt.subplots(2,figsize=(10,10))
sns.countplot(x='PlaceofBirth', data=data, ax=axarr[0])
sns.countplot(x='PlaceofBirth', hue='Class', data=data,hue_order = ['L', 'M', 'H'], ax=axarr[1])
plt.show()


# In[13]:


data.shape


# In[11]:


fig, axarr  = plt.subplots(2,figsize=(10,10))
sns.countplot(x='StageID', data=data, ax=axarr[0])
sns.countplot(x='StageID', hue='Class', data=data, hue_order = ['L', 'M', 'H'], ax=axarr[1])
plt.show()


# In[12]:


fig, axarr  = plt.subplots(2,figsize=(10,10))
sns.countplot(x='GradeID', data=data, order=['G-02', 'G-04', 'G-05', 'G-06', 'G-07', 'G-08', 'G-09', 'G-10', 'G-11', 'G-12'], ax=axarr[0])
sns.countplot(x='GradeID', hue='Class', data=data, order=['G-02', 'G-04', 'G-05', 'G-06', 'G-07', 'G-08', 'G-09', 'G-10', 'G-11', 'G-12'], hue_order = ['L', 'M', 'H'], ax=axarr[1])
plt.show()


# In[13]:


data.loc[data['GradeID'] == 'G-05']


# In[14]:


data.loc[data['GradeID'] == 'G-09']


# In[15]:


fig, axarr  = plt.subplots(2,figsize=(10,10))
sns.countplot(x='SectionID', data=data, order=['A', 'B', 'C'], ax = axarr[0])
sns.countplot(x='SectionID', hue='Class', data=data, order=['A', 'B', 'C'],hue_order = ['L', 'M', 'H'], ax = axarr[1])
plt.show()


# In[16]:


fig, axarr  = plt.subplots(2,figsize=(10,10))
sns.countplot(x='Topic', data=data, ax = axarr[0])
sns.countplot(x='Topic', hue='Class', data=data,hue_order = ['L', 'M', 'H'], ax = axarr[1])
plt.show()


# In[17]:


fig, axarr  = plt.subplots(2,figsize=(10,10))
sns.countplot(x='Semester', data=data, ax = axarr[0])
sns.countplot(x='Semester', hue='Class', data=data,hue_order = ['L', 'M', 'H'], ax = axarr[1])
plt.show()


# In[18]:


fig, axarr  = plt.subplots(2,figsize=(10,10))
sns.countplot(x='Relation', data=data, ax = axarr[0])
sns.countplot(x='Relation', hue='Class', data=data,hue_order = ['L', 'M', 'H'], ax = axarr[1])
plt.show()


# In[20]:


sns.pairplot(data, hue="Class", diag_kind="kde", hue_order = ['L', 'M', 'H'], markers=["o", "s", "D"])
plt.show()


# In[21]:


data.groupby('Topic').median()


# In[22]:


data.groupby('GradeID').median()


# In[23]:


fig, axarr  = plt.subplots(2,figsize=(10,10))
sns.countplot(x='ParentAnsweringSurvey', data=data, order=['Yes', 'No'], ax = axarr[0])
sns.countplot(x='ParentAnsweringSurvey', hue='Class', data=data, order=['Yes', 'No'], hue_order = ['L', 'M', 'H'], ax = axarr[1])
plt.show()


# In[24]:


fig, axarr  = plt.subplots(2,figsize=(10,10))
sns.countplot(x='ParentschoolSatisfaction', data=data, order=['Good', 'Bad'], ax = axarr[0])
sns.countplot(x='ParentschoolSatisfaction', hue='Class', data=data, order=['Good', 'Bad'],hue_order = ['L', 'M', 'H'], ax = axarr[1])
plt.show()


# In[25]:


fig, axarr  = plt.subplots(2,figsize=(10,10))
sns.countplot(x='StudentAbsenceDays', data=data, order=['Under-7', 'Above-7'], ax = axarr[0])
sns.countplot(x='StudentAbsenceDays', hue='Class', data=data, order=['Under-7', 'Above-7'],hue_order = ['L', 'M', 'H'], ax = axarr[1])
plt.show()


# In[ ]:




