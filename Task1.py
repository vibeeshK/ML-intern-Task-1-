#!/usr/bin/env python
# coding: utf-8

# In[20]:


#Import the required libraries
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error


# In[21]:


#Import the data
url = "http://bit.ly/w-data"
data = pd.read_csv(url)
data1=data
data


# In[25]:


#Visualize the data
plt.scatter(x=data['Hours'], y=data['Scores'])
plt.ylabel('Scores')
plt.xlabel('Hours')
plt.show()


# In[22]:


#Splitting training and testing data
x = data.iloc[:, :-1].values  
y = data.iloc[:, 1].values
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.70,test_size=0.30, random_state=0)


# In[ ]:





# In[23]:





# In[24]:


#Linear Regression
linearRegressor = LinearRegression()
linearRegressor.fit(x_train, y_train)
y_predicted = linearRegressor.predict(x_test)
mse = mean_squared_error(y_test, y_predicted)
r = r2_score(y_test, y_predicted)
mae = mean_absolute_error(y_test,y_predicted)
print("Mean Squared Error:",mse)
print("R score:",r)
print("Mean Absolute Error:",mae)


# In[27]:


#Prediction- We need to predict for 9.5 hours in the day-
#Therefore we create a data set with 9.5 hours as:


# In[28]:


data


# In[34]:


df_empty = data[0:0]
df_empty['Hours']=[9.5]
df_empty=df_empty.drop('Scores',axis=1)
df_empty


# In[39]:


#Making the prediction-
linearRegressor = LinearRegression()
linearRegressor.fit(x, y)
prediction = linearRegressor.predict(df_empty)
df_empty['Scores']=prediction


# In[45]:


#Therefore the prediction is--
df_empty


# In[ ]:





# In[ ]:




