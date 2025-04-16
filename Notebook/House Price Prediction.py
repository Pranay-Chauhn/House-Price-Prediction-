#!/usr/bin/env python
# coding: utf-8

# In[131]:


import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


# In[132]:


data  = pd.read_csv("C:\\Users\\rishi\\Downloads\\house-prices-advanced-regression-techniques\\train.csv")
data
test = pd.read_csv("C:\\Users\\rishi\\Downloads\\house-prices-advanced-regression-techniques\\test.csv")


# In[92]:


data.info()


# In[93]:


data.columns


# In[94]:


#correlation matrix
corrmat = data.corr(numeric_only=True)
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True)


# In[95]:


sns.distplot(data['SalePrice'])


# In[96]:


var = 'GrLivArea'
df = pd.concat([data['SalePrice'],data[var]],axis=1)
df.plot.scatter(x=var , y='SalePrice' , ylim=(0,800000))


# In[97]:


var = 'OverallQual'
df = pd.concat([data['SalePrice'],data[var]],axis=1)
sns.boxplot(x=var , y='SalePrice',data=data)


# In[98]:


var = 'TotalBsmtSF'
df = pd.concat([data['SalePrice'],data[var]],axis=1)
df.plot.scatter(x=var , y='SalePrice' , ylim=(0,800000))


# In[99]:


plt.figure(figsize=(20,8))
var = 'YearBuilt'
df = pd.concat([data['SalePrice'],data[var]],axis=1)
sns.boxplot(x=var , y='SalePrice',data=data)
plt.xticks(rotation=90);


# In[100]:


#saleprice correlation matrix
k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(data[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


# In[101]:


sns.set()
cols = ['SalePrice','OverallQual','GrLivArea','GarageArea','TotalBsmtSF','1stFlrSF','FullBath','TotRmsAbvGrd','YearBuilt'
    ]
sns.pairplot(data[cols],size=2.5)


# In[133]:


# dealing with the missing_value
total=data.isnull().sum().sort_values(ascending = False)
percent = (data.isnull().sum()/data.isnull().count()).sort_values(ascending = False)
missing_values = pd.concat([total,percent],axis = 1 , keys=['Total','Percent'] )
missing_values.head(20).index


# In[134]:


data = data.drop((missing_values[missing_values['Total'] > 1]).index,axis= 1)
data = data.drop(data.loc[data.Electrical.isnull()].index)
data.isnull().sum().max()
test = test.drop((missing_values[missing_values['Total'] > 1]).index,axis= 1)


# In[135]:


test


# In[136]:


data = data.drop(["KitchenAbvGr","EnclosedPorch","Id"],axis = 1)
test = test.drop(["KitchenAbvGr","EnclosedPorch","Id"],axis = 1)


# In[138]:


data.info()


# In[139]:


test.info()


# In[107]:


'''data = data.drop(data[data.Street=='Grvl'].index,axis = 0)
data.shape'''


# In[ ]:





# In[108]:


plt.scatter(data.SalePrice,data.LotArea)
plt.xlabel('Sale Price')
plt.ylabel('Lot Size')


# In[140]:


numerical_data = [col for col in data.columns if data[col].dtype != object]
categorical_data = [col for col in data.columns if data[col].dtype == object]


# In[141]:


def outliers(df,ft) :
    Q1 = df[ft].quantile(0.25)
    Q3= df[ft].quantile(0.75)
    IQR = Q3-Q1
    lower_bound = Q1 - 1.5*IQR
    upper_bound = Q3 + 1.5*IQR
    ls=  df.index[(df[ft]<lower_bound) | (df[ft]>upper_bound)]
    return ls


# In[142]:


index_list = []
for features in numerical_data :
    index_list.extend(outliers(data,features))
index_list


# In[143]:


df = data.copy()
df.shape


# In[144]:


def remove(df,ls) :
    ls = sorted(set(ls))
    df = df.drop(ls)
    return df


# In[145]:


data_cleaned = remove(df,index_list)
data_cleaned.shape


# In[146]:


data_cleaned


# In[116]:


df_cat = data_cleaned.select_dtypes(object)
df_num = data_cleaned.select_dtypes(int)
df_num_test = test.select_dtypes(int)
df_num.reset_index(drop=True,inplace=True)
df_num


# In[147]:


cols = list(df_cat.columns)
cols


# In[148]:


from sklearn.preprocessing import OneHotEncoder
'''ohe = OneHotEncoder(categories='auto')
feature_arr = ohe.fit_transform(data_cleaned[['MSZoning',
 'Street',
 'LotShape',
 'LandContour',
 'Utilities',
 'LotConfig',
 'LandSlope',
 'Neighborhood',
 'Condition1',
 'Condition2',
 'BldgType',
 'HouseStyle',
 'RoofStyle',
 'RoofMatl',
 'Exterior1st',
 'Exterior2nd',
 'ExterQual',
 'ExterCond',
 'Foundation',
 'Heating',
 'HeatingQC',
 'CentralAir',
 'Electrical',
 'KitchenQual',
 'Functional',
 'PavedDrive',
 'SaleType',
 'SaleCondition']]).toarray()
feature_labels = ohe.categories_'''


# In[149]:


labels = ['MSZoning',
 'Street',
 'LotShape',
 'LandContour',
 'Utilities',
 'LotConfig',
 'LandSlope',
 'Neighborhood',
 'Condition1',
 'Condition2',
 'BldgType',
 'HouseStyle',
 'RoofStyle',
 'RoofMatl',
 'Exterior1st',
 'Exterior2nd',
 'ExterQual',
 'ExterCond',
 'Foundation',
 'Heating',
 'HeatingQC',
 'CentralAir',
 'Electrical',
 'KitchenQual',
 'Functional',
 'PavedDrive',
 'SaleType',
 'SaleCondition']

categorical_data = data_cleaned[labels]

ohe = OneHotEncoder(categories='auto')

feature_arr = ohe.fit_transform(categorical_data).toarray()

ohe_labels = ohe.get_feature_names_out(labels)

features = pd.DataFrame(
               feature_arr,
               columns=ohe_labels)
features.shape


# In[150]:


categorical_data_test = test[labels]

ohe2 = OneHotEncoder(categories='auto')

feature_arr_test = ohe2.fit_transform(categorical_data_test).toarray()

ohe_labels_test = ohe2.get_feature_names_out(labels)

features_test = pd.DataFrame(
               feature_arr_test,
               columns=ohe_labels_test)
features_test.shape


# In[155]:


test1 = pd.concat([test.select_dtypes(int),features_test],axis = 1)
test1.shape


# In[152]:


final_data = pd.concat([df_num,features],axis = 1)
final_data.shape


# In[153]:


y = final_data['SalePrice']
X = final_data.drop('SalePrice',axis = 1)


# In[124]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)


# In[125]:


X_train.shape


# In[156]:


test2 = test1[X_train.columns]


# In[ ]:


X_train.columns


# In[157]:


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=40)


# In[158]:


model.fit(X_train,y_train)


# In[159]:


model.score(X_test,y_test)


# In[160]:


from sklearn.model_selection import cross_val_score
cross_val_score(LogisticRegression(max_iter=1000),X_train,y_train,cv=3)


# In[161]:


cross_val_score(SVC(),X_train,y_train,cv=3)


# In[162]:


np.mean(cross_val_score(DecisionTreeRegressor(),X_train,y_train,cv=3))


# In[163]:


model = DecisionTreeRegressor()
model.fit(X_train,y_train)


# In[169]:


model.predict(test1)


# In[ ]:




