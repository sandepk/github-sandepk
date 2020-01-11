#!/usr/bin/env python
# coding: utf-8

# # Predicting House Sales Prices
# 
# ## Import Libraries and Train Dataset

# In[5]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')
pd.options.display.max_columns = 999
data = pd.read_csv("train.csv")


# In[6]:


data.shape  ## Dimension of the Dataset


# In[7]:


data.head() ## Sample of the Dataset


# In[61]:


data.info() ## Datatypes of the attributes


# In[62]:


SalePrice = data['SalePrice']  ## Store the target column into a Variable


# # Histogram of Saleprices of Train Dataset

# In[63]:


sns.distplot(data['SalePrice'])
data['SalePrice'].describe()


# # Converting Object Datatypes into Categorical Datatypes

# In[64]:


category_col = data.select_dtypes(include=['object']).columns  ## columns which have object datatypes
category_col


# In[ ]:





# In[65]:


for x in category_col:
    data[x] = data[x].astype('category')  ## converted that object Datatypes into Category datatypes


# In[66]:


data['YearBuilt'] = data['YearBuilt'].astype('category')
data['YearRemodAdd'] = data['YearRemodAdd'].astype('category')
data['YrSold'] = data['YrSold'].astype('category')                    ## These are also some categorical datatypes stored in
data['MoSold'] = data['MoSold'].astype('category')                    ##  numeric datatypes.
data['GarageYrBlt'] = data['GarageYrBlt'].astype('category')


# In[67]:


data.info()   ## All the object datatypes are changed into Categorical datatypes


# # Handling Missing values
# 

# In[68]:


missing = data.isnull()  ## tells about the missing values
missing.head()


# In[69]:


missing_attribute = missing.sum()  ## Number of missing values in each attribute
missing_attribute


# In[70]:


missing_cols = missing_attribute[(missing_attribute > len(data)/20)].sort_values() 
## columns in which missing values are more than 5%
missing_cols


# In[71]:


drop_missing_cols = missing_cols.index
drop_missing_cols     


# In[72]:


data= data.drop(drop_missing_cols , axis=1) ## deleting these attributes


# In[73]:


data.shape  ## Dimension of the Dataset


# In[74]:


num_missing = data.isnull().sum()
fixable_cols = num_missing[(num_missing > 0)].sort_values() ## Now the Columns which have missing values
fixable_cols


# In[75]:


replacement_values_dict = data[fixable_cols.index].mode().to_dict(orient='records')[0] ##doubt
replacement_values_dict   ## create a dictionary which contains the mode of the aattributes


# In[76]:


data = data.fillna(replacement_values_dict) ## impute the missing values with the mode of that particular attribute


# In[77]:


data.isnull().sum() ## number of missing values in each attribute


# # Creating Scatter Plots for Numerical Attributes

# In[78]:


numeric_col = data.select_dtypes(include=['int64', 'float64']).columns
for var in numeric_col:
    df = pd.concat([data['SalePrice'],data[var]],axis =1)
    df.plot.scatter(x = var, y = 'SalePrice')


# # Creating Box Plots for Categorical Attributes

# In[22]:


category_col = data.select_dtypes(include = ['category']).columns
for var in category_col:
    df = pd.concat([data['SalePrice'], data[var]], axis=1)
    f, ax = plt.subplots(figsize=(8,6))
    fig = sns.boxplot(x=var, y="SalePrice", data=df)
    fig.axis(ymin=0, ymax=800000)


# # Imputing Outliers with Mean of that Outlier

# In[79]:


numeric_col.drop('SalePrice')          ## selecting numeric columns to determine outliers


# In[24]:


for x in numeric_col:
    iqr = (data[x].quantile(.75) - data[x].quantile(.25))
    upper_outlier = data[x].quantile(.75) + (iqr*1.5)
    lower_outlier = data[x].quantile(.25) - (iqr*1.5)
    mean = data[x].mean()                                ## imputing the outliers with the mean of that particular attribute
    data[x] = data[x].mask((data[x] < lower_outlier), mean)
    data[x] = data[x].mask((data[x] > upper_outlier), mean)


# In[25]:


data


# # Normalizing Numerical Attributes

# In[80]:


normalizing_col = list(data.select_dtypes(include=['int64', 'float64']).columns)
normalizing_col.remove('Id')
normalizing_col.remove('SalePrice')
## list of numerical attributes


# In[81]:


for x in normalizing_col:
    maximum = data[x].max()
    minimum = data[x].min()                                           ## Normalizing the numerical attributes
    data[x] = data[x].apply(lambda x : (x-minimum)/(maximum-minimum))

data


# # Correlation Matrix and Heatmap

# In[28]:


data.corr() ## correlation between each attribute


# In[82]:


corrmat = data.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=1, square=True)


# # Data Reduction

# In[30]:


category_df = data.select_dtypes(include=['category']).columns
data[category_df].nunique()  ## number of unique values in categorical data types


# In[31]:


cat_cols = data[category_df].nunique()  
drop_col = cat_cols[cat_cols>10].index   ## Deleting the categorical attributes which have more than 10 unique values
data= data.drop(drop_col , axis=1)


# In[32]:


data.shape


# # Data Transformation with Dummy Variables

# In[33]:


text_cols = data.select_dtypes(include=['category'])
data = pd.concat([
    data,                                                      ## create dummy variable of categorical attribute
    pd.get_dummies(data.select_dtypes(include=['category']))
], axis=1).drop(text_cols,axis=1)


# In[34]:


data.shape ## total number of attributes increased due to dummy variables


# # Data Reduction by Feature Engineering

# In[35]:


correlations = data.corr()['SalePrice'].abs().sort_values()
corr_cols = list(correlations[correlations<.3].index)      ## columns which have correlation between -.3 and .3 with SalePrice
corr_cols.remove('Id')                   


# In[36]:


data= data.drop(corr_cols , axis=1) ## deleting these attributes


# In[37]:


list(data.columns) ## dimension of the dataset


# # Preprocessed Train Data

# In[38]:


preprocessed_train_data = data


# # Creating & Fitting Linear Regression Model on Validation Dataset
# 
# ## Finding mean square error(MSE), root mean square error(RMSE) and root mean square log error(RMSLE)

# In[39]:


from sklearn.model_selection import KFold
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error

train = data[:730]
test = data[730:]
numeric_train = train.select_dtypes(include=['integer', 'float','category'])
numeric_test = test.select_dtypes(include=['integer', 'float','category'])
    
    ## You can use `pd.Series.drop()` to drop a value.
features = numeric_train.columns.drop(["SalePrice",'Id'])
lr = linear_model.LinearRegression()
lr.fit(train[features], train["SalePrice"])
predictions = lr.predict(test[features])
mse = mean_squared_error(test["SalePrice"], predictions)
print(mse)
rmse = np.sqrt(mse)
print(rmse)
rmsle = np.sqrt(mean_squared_log_error( test["SalePrice"], predictions ))
print(rmsle)


# # Creating & Fitting Decision Tree Model on Validation Dataset
# ## Finding mean square error(MSE), root mean square error(RMSE) and root mean square log error(RMSLE)

# In[40]:


from sklearn.tree import DecisionTreeRegressor
train = data[:730]
test = data[730:]
predictors = list(data.columns)
predictors.remove("SalePrice")
predictors.remove("Id")


reg = DecisionTreeRegressor(min_samples_leaf=5)

reg.fit(train[predictors], train["SalePrice"])

predictions = reg.predict(test[predictors])
mse = mean_squared_error(test["SalePrice"], predictions)
print(mse)
rmse = np.sqrt(mse)
print(rmse)
rmsle = np.sqrt(mean_squared_log_error( test["SalePrice"], predictions ))
print(rmsle)


# # Creating & Fitting Random Forest Model on Validation Dataset
# ## Finding mean square error(MSE), root mean square error(RMSE) and root mean square log error(RMSLE)

# In[41]:


from sklearn.ensemble import RandomForestRegressor
train = data[:730]
test = data[730:]
predictors = list(data.columns)
predictors.remove("SalePrice")
predictors.remove("Id")

reg = RandomForestRegressor(min_samples_leaf=5)
reg.fit(train[predictors], train["SalePrice"])

predictions = reg.predict(test[predictors])
mse = mean_squared_error(test["SalePrice"], predictions)
print(mse)
rmse = np.sqrt(mse)
print(rmse)
rmsle = np.sqrt(mean_squared_log_error( test["SalePrice"], predictions ))
print(rmsle)


# # Importing and Preprocessing of Test Dataset

# In[42]:


import pandas as pd
import matplotlib.pyplot as plt
pd.options.display.max_columns = 999
data = pd.read_csv("test.csv")
category_col = data.select_dtypes(include=['object']).columns  ## columns which have object datatypes
for x in category_col:
    data[x] = data[x].astype('category')  ## converted that object Datatypes into Category datatypes
data['YearBuilt'] = data['YearBuilt'].astype('category')
data['YearRemodAdd'] = data['YearRemodAdd'].astype('category')
data['YrSold'] = data['YrSold'].astype('category')                    ## These are also some categorical datatypes stored in
data['MoSold'] = data['MoSold'].astype('category')                    ##  numeric datatypes.
data['GarageYrBlt'] = data['GarageYrBlt'].astype('category')
missing = data.isnull()  ## tells about the missing values
missing_attribute = missing.sum()  ## Number of missing values in each attribute
missing_cols = missing_attribute[(missing_attribute > len(data)/20)].sort_values() 
## columns in which missing values are more than 5%
drop_missing_cols = missing_cols.index
data= data.drop(drop_missing_cols , axis=1) ## deleting these attributes
num_missing = data.isnull().sum()
fixable_cols = num_missing[(num_missing > 0)].sort_values() ## Now the Columns which have missing values
replacement_values_dict = data[fixable_cols.index].mode().to_dict(orient='records')[0]
replacement_values_dict   ## create a dictionary which contains the mode of the aattributes
data = data.fillna(replacement_values_dict) ## impute the missing values with the mode of that particular attribute
data.isnull().sum() ## number of missing values in each attribute
numeric_col = data.select_dtypes(include=['int64', 'float64']).columns
for x in numeric_col:
    iqr = (data[x].quantile(.75) - data[x].quantile(.25))
    upper_outlier = data[x].quantile(.75) + (iqr*1.5)
    lower_outlier = data[x].quantile(.25) - (iqr*1.5)
    mean = data[x].mean()                                ## imputing the outliers with the mean of that particular attribute
    data[x] = data[x].mask((data[x] < lower_outlier), mean)
    data[x] = data[x].mask((data[x] > upper_outlier), mean)
normalizing_col = list(data.select_dtypes(include=['int64', 'float64']).columns)
normalizing_col.remove('Id')
for x in normalizing_col:
    maximum = data[x].max()
    minimum = data[x].min()                                           ## Normalizing the numerical attributes
    data[x] = data[x].apply(lambda x : (x-minimum)/(maximum-minimum))

category_df = data.select_dtypes(include=['category']).columns
data[category_df].nunique()  ## number of unique values in categorical data types
cat_cols = data[category_df].nunique()  
drop_col = cat_cols[cat_cols>10].index   ## Deleting the categorical attributes which have more than 10 unique values
data= data.drop(drop_col , axis=1)
text_cols = data.select_dtypes(include=['category'])
data = pd.concat([
    data,                                                      ## create dummy variable of categorical attribute
    pd.get_dummies(data.select_dtypes(include=['category']))
], axis=1).drop(text_cols,axis=1)
data = data[['Id',
 'LotArea',
 'OverallQual',
 'MasVnrArea',
 'TotalBsmtSF',
 '1stFlrSF',
 '2ndFlrSF',
 'GrLivArea',
 'FullBath',
 'TotRmsAbvGrd',
 'Fireplaces',
 'GarageCars',
 'GarageArea',
 'OpenPorchSF',
 'MSZoning_RM',
 'MasVnrType_None',
 'ExterQual_Gd',
 'ExterQual_TA',
 'Foundation_CBlock',
 'Foundation_PConc',
 'BsmtQual_Ex',
 'BsmtQual_Gd',
 'BsmtQual_TA',
 'BsmtFinType1_GLQ',
 'HeatingQC_Ex',
 'HeatingQC_TA',
 'KitchenQual_Gd',
 'KitchenQual_TA']]      

preprocessed_test_data = data


# # Creating & Fitting Linear Regression Model on Test Dataset
# 
# ## Saved the predictions and submitted into kaggle
# ##  Histogram of predictions

# In[43]:


from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error

train = preprocessed_train_data
test = preprocessed_test_data
numeric_train = train.select_dtypes(include=['integer', 'float','category'])
numeric_test = test.select_dtypes(include=['integer', 'float','category'])
    
    ## You can use `pd.Series.drop()` to drop a value.
features = numeric_train.columns.drop(["SalePrice",'Id'])
lr = linear_model.LinearRegression()
lr.fit(train[features], train["SalePrice"])
predictions = lr.predict(test[features])
preprocessed_test_data['SalePrice'] = predictions
submit_lr = preprocessed_test_data[['Id','SalePrice']]
submit_lr.to_csv("LR.csv", sep = ',', index=False)

sns.distplot(submit_lr['SalePrice'])
print("skewness: %f" % submit_lr['SalePrice'].skew())
print("kurtosis: %f" %submit_lr['SalePrice'].kurtosis())


# # Creating & Fitting Decision Tree Model on Test Dataset
# 
# ## Saved the predictions and submitted into kaggle
# ##  Histogram of predictions

# In[44]:


from sklearn.tree import DecisionTreeRegressor
train = preprocessed_train_data
test = preprocessed_test_data
predictors = list(train.columns)
predictors.remove("SalePrice")
predictors.remove("Id")


reg = DecisionTreeRegressor(min_samples_leaf=5)

reg.fit(train[predictors], train["SalePrice"])

predictions = reg.predict(test[predictors])

preprocessed_test_data['SalePrice'] = predictions
submit_dt = preprocessed_test_data[['Id','SalePrice']]
submit_dt.to_csv("DT.csv", sep = ',', index=False)

sns.distplot(submit_dt['SalePrice'])
print("skewness: %f" % submit_dt['SalePrice'].skew())
print("kurtosis: %f" %submit_dt['SalePrice'].kurtosis())


# # Creating & Fitting Random Forest Model on Test Dataset
# 
# ## Saved the predictions and submitted into kaggle
# ##  Histogram of predictions

# In[45]:


from sklearn.ensemble import RandomForestRegressor
train = preprocessed_train_data
test = preprocessed_test_data
predictors = list(train.columns)
predictors.remove("SalePrice")
predictors.remove("Id")

reg = RandomForestRegressor(min_samples_leaf=5)
reg.fit(train[predictors], train["SalePrice"])

predictions = reg.predict(test[predictors])
preprocessed_test_data['SalePrice'] = predictions
submit_rf = preprocessed_test_data[['Id','SalePrice']]
submit_rf.to_csv("RF.csv", sep = ',', index=False)

sns.distplot(submit_rf['SalePrice'])

print("skewness: %f" % submit_rf['SalePrice'].skew())
print("kurtosis: %f" %submit_rf['SalePrice'].kurtosis())


# 
# 
# # Thank You!!

# In[ ]:




