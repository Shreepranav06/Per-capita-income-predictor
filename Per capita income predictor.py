#!/usr/bin/env python
# coding: utf-8

# In[24]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model


# In[25]:


df= pd.read_csv('Saved Games/canada_per_capita_income.csv')


# In[26]:


df


# In[27]:


plt.scatter(df['year'],df['per capita income (US$)'])
y_lim = plt.ylim()
x_lim = plt.xlim()
plt.plot(x_lim, y_lim, 'k-', color = 'r')


# In[28]:


reg= linear_model.LinearRegression()
reg.fit(df[['year']],df['per capita income (US$)'])


# In[29]:



a=int(input('Enter which year you want to predict: '))
p=reg.predict([[a]])
print('the price is',p[0])

