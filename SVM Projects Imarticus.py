# -*- coding: utf-8 -*-
"""
Created on Sun May 20 16:16:43 2018

@author: Saurabh
"""

#%%
import pandas as pd
import numpy as np
train_data=pd.read_csv('risk_analytics_train.csv',header=0)
test_data=pd.read_csv('risk_analytics_test.csv',header=0)
#train_data_df = pd.DataFrame(train_data)
#test_data_df= pd.DataFrame(test_data)
print(train_data.head())
#%%
# 1 step to remove missing values.
#finding misssing values
print(train_data.isnull().sum())
train_data.describe(include='all')
#%%
#step 2 <- filling missing values by considering categoical variables
colname1=['Gender','Married','Dependents' ,'Self_Employed','Loan_Amount_Term']
for x in colname1[:]:
    train_data[x].fillna(train_data[x].mode()[0],inplace=True)# inplace will  make the changes permanent

# now chwck missing valus
print(train_data.isnull().sum())
#%%
#continuation of step2 fiiling numericalvalues with mean values
train_data["LoanAmount"].fillna(train_data["LoanAmount"].mean(),inplace=True)# inplace will  make the changes permanent
print(train_data.isnull().sum())
#%%
#0<- not took loan
#1<- took a loan
#continuation of step2 fiiling numericalvalues with mean values

train_data["Credit_History"].fillna(value=0,inplace=True)# inplace will  make the changes permanent
print(train_data.isnull().sum())
#%%
#step 3 <-  converting categorical data into numerical
from sklearn import preprocessing
colname=['Gender','Married','Education','Self_Employed','Property_Area','Loan_Status']
#sklearn gives acurate answer if data is in numerical format
le={}

for x in colname:
    le[x]=preprocessing.LabelEncoder()
#%%
for x in colname:
    train_data[x]=le[x].fit_transform(train_data.__getattr__(x))
#%%
print(train_data.head())
#train_data.describe(include='all')
#converted loan status as .y<--1 and N<-- 0

#%%
#%%
# 1 step to remove missing values.
#finding misssing values
print(test_data.isnull().sum())
#test_data.describe(include='all')
#%%
#step 2 <- filling missing values by considering categoical variables
colname1=['Gender','Dependents' ,'Self_Employed','Loan_Amount_Term']
for x in colname1[:]:
    test_data[x].fillna(test_data[x].mode()[0],inplace=True)# inplace will  make the changes permanent

# now chwck missing valus
print(test_data.isnull().sum())
#%%
#continuation of step2 fiiling numericalvalues with mean values
test_data["LoanAmount"].fillna(test_data["LoanAmount"].mean(),inplace=True)# inplace will  make the changes permanent
print(train_data.isnull().sum())
#%%
#0<- not took loan
#1<- took a loan
#continuation of step2 fiiling numericalvalues with mean values

test_data["Credit_History"].fillna(value=0,inplace=True)# inplace will  make the changes permanent
print(test_data.isnull().sum())
#%%
#step 3 <-  converting categorical data into numerical
from sklearn import preprocessing
colname=['Gender','Married','Education','Self_Employed','Property_Area']
#sklearn gives acurate answer if data is in numerical format
le={}

for x in colname:
    le[x]=preprocessing.LabelEncoder()
#%%
for x in colname:
    test_data[x]=le[x].fit_transform(test_data.__getattr__(x))
#%%
print(test_data.head())
#train_data.describe(include='all')
#converted loan status as .y<--1 and n<-- 0
#%%
#step 5<-converting dataset into training and testing data  
x_train=train_data.values[:,1:-1]
y_train=train_data.values[:,-1]# y is a dependent variable of train data set
y_train=y_train.astype(int)#unknown label can occur that ys we are converting it
#%%
x_test=test_data.values[:,1:]
type(x_test)
#%%

#step 6<- maintaing range size
#preventive step not mandate
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)

scaler.fit(x_test)
x_test=scaler.transform(x_test)
#%%
##step 7 <- creating a model and and running the model
from sklearn import svm
svc_model=svm.SVC(kernel='rbf',C=1.0,gamma=0.1)
svc_model.fit(x_train,y_train)
y_pred=svc_model.predict(x_test)
print(list(y_pred))
y_pred_col=list(y_pred)
#%%
test_data=pd.read_csv('risk_analytics_test.csv',header=0)
test_data["y_predictions"]=y_pred_col
test_data.head()
#%%
#converting fileto csv
test_data.to_csv('test_data.csv')
#%%





















