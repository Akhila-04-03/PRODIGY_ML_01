#!/usr/bin/env python
# coding: utf-8

# # loading required libraries
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error
# from sklearn.impute import SimpleImputer
# from sklearn.ensemble import HistGradientBoostingRegressor
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import cross_val_score
# import seaborn as sns
# import statsmodels.api as sm
# 

# In[55]:


train_data = pd.read_csv('train.csv')
test_data=pd.read_csv('test.csv')
data=pd.concat([train_data,test_data])
plt.figure(figsize=(18,6))
plt.title('Heatmap of missing values')
sns.heatmap(data.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[61]:


# separate numerical and categorical columns
numerical_cols = train_data.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = train_data.select_dtypes(include=['object']).columns

# replace missing values with mean for numerical columns
imputer = SimpleImputer(strategy='mean')
train_data[numerical_cols] = imputer.fit_transform(train_data[numerical_cols])

# replace missing values with mode for categorical columns
imputer = SimpleImputer(strategy='most_frequent')
train_data[categorical_cols] = imputer.fit_transform(train_data[categorical_cols])

# verify that missing values are replaced
print(train_data.isnull().sum())




# In[62]:


# Select the predictor and target variable (for both training and testing set)
x_train = train_data[['LotArea', 'BedroomAbvGr', 'BsmtFullBath','BsmtHalfBath','FullBath','HalfBath']]
y_train = train_data['SalePrice']

x_test=test_data[['LotArea', 'BedroomAbvGr', 'BsmtFullBath','BsmtHalfBath','FullBath','HalfBath']]


# In[63]:


# separate numerical and categorical columns for test data
numerical_cols2 = x_test.select_dtypes(include=['int64', 'float64']).columns

# replace missing values with mean for numerical columns
imputer = SimpleImputer(strategy='mean')
x_test[numerical_cols2] = imputer.fit_transform(x_test[numerical_cols2])

# verify that missing values are replaced
print(x_test.isnull().sum())


# In[59]:


# Initializing a linear regression model and fit this model to train set
model = LinearRegression()
print(model)
model.fit(x_train, y_train)


# In[64]:


# Making predictions on the test set using the fitted model

y_pred = model.predict(x_test)
print(y_pred)


# In[24]:


#instead of simple linear regression we are using HistGradientBoosting which can effectively handle large datasets and during training, the algorithm learns at each split point whether samples with missing values should go to the left or right child, based on the potential gain. When predicting, samples with 
#missing values are assigned to the left or right child accordingly1. This makes it convenient for handling missing data without additional preprocessing steps. 


# In[65]:


train_data = pd.read_csv('train.csv')
test_data=pd.read_csv('test.csv')

x_train = train_data[['LotArea', 'BedroomAbvGr', 'BsmtFullBath','BsmtHalfBath','FullBath','HalfBath']]
y_train = train_data['SalePrice']

x_test=test_data[['LotArea', 'BedroomAbvGr', 'BsmtFullBath','BsmtHalfBath','FullBath','HalfBath']]
# Initializing a linear regression model and fit this model to train set
histgrad_model =  HistGradientBoostingRegressor().fit(x_train, y_train)

# Making predictions on the test set using the fitted model
y_pred = histgrad_model.predict(x_test)
print("predicted saleprices;",y_pred)


# In[40]:


plt.figure()
plt.title('Comparison of Predicted Sale Price and Original Price')
plt.scatter(y_train, model.predict(x_train), label='Simple Linear Regression')
plt.legend()
plt.show()


# In[38]:


plt.figure()
plt.title('Comparison of Predicted Sale Price and Original Price')
plt.scatter(y_train, histgrad_model.predict(x_train), label='HistGradientBoostingRegressor',color='green')
plt.legend()
plt.show()


# In[46]:


#preparing the submission data
y_pred = model.predict(x_test)
sub_data=pd.DataFrame()
sub_data['Id']=test_data['Id']
sub_data['SalePrice']=y_pred
sub_data
sub_data.to_csv('submission.csv', index=False)


# In[70]:


###LINEAR REGRESSION USING ORDINARY LEAST SQUARE METHOD

data = pd.read_csv('train.csv')

# Select the relevant features: square footage, number of bedrooms, and number of bathrooms
X1 = data[['LotArea', 'BedroomAbvGr', 'BsmtFullBath','BsmtHalfBath','FullBath','HalfBath']]
y1 = data['SalePrice']  # Target variable (house price)

# Add a constant term for the intercept in the regression model
X1 = sm.add_constant(X1)

# Fit the linear regression model
model = sm.OLS(y1, X1).fit()
model
# Print the summary of the model
print(model.summary())


# In[ ]:




