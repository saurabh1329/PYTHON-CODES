# -*- coding: utf-8 -*-
"""
Created on Sun May 13 16:21:13 2018

@author: Saurabh
"""
#%%
import pandas as pd
import numpy as np
pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)
#%%
adult_df=pd.read_csv('adult_data.csv',header=None , delimiter=' *, *',engine='python')
adult_df.head()
adult_df.shape
#%% GIVING A COLUMNS NAMES
adult_df.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
'marital_status', 'occupation', 'relationship',
'race', 'sex', 'capital_gain', 'capital_loss',
'hours_per_week', 'native_country', 'income']
adult_df.head()
#%%  notice
adult_df.isnull().sum()
for value in ['workclass','education','marital_status',
              'occupation','relationship','race','sex',
              'native_country','income']:
    print(value,":",sum(adult_df[value]=="?"))
#%%
#create a copy of original dataframe/dataset
adult_df_rev=pd.DataFrame.copy(adult_df)
adult_df_rev.describe(include='all')
for value in ['workclass','occupation','native_country']:
    adult_df_rev[value].replace(["?"],
                [adult_df_rev.describe(include="all")[value][2]],
                 inplace=True)
adult_df_rev.head(20)
#%%
#adult_df.isnull().sum()
for value in ['workclass','education','marital_status',
              'occupation','relationship','race','sex',
              'native_country','income']:
    print(value,":",sum(adult_df_rev[value]=="?"))
#%%
colname=['workclass','education','marital_status','occupation','relationship','race','sex','native_country','income']
#sklearn gives acurate answer if data is in numerical format
from sklearn import preprocessing
le={}

for x in colname:
    le[x]=preprocessing.LabelEncoder()

for x in colname:
    adult_df_rev[x]=le[x].fit_transform(adult_df_rev.__getattr__(x))
adult_df_rev.head(2)
adult_df_rev.describe(include='all')
#0<=50k
#%%
x=adult_df_rev.values[:,:-1]
y=adult_df_rev.values[:,-1]
x
#%%
#preventive step not mandate
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(x)
x=scaler.transform(x)

print(x)
#%%
#it might be a case that package will coneert dpendet varable as object and we dont want that so manuallywe are converting it into nuber i.e int
#so do it for safer side
y=y.astype(int)
#%%
#RUNNING BASIC MODEL
from sklearn.model_selection import train_test_split

#split the data into train and test

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3
                                                    ,random_state=10)



#%%
from sklearn.linear_model import LogisticRegression
#create a model
#fitting  traning data tp the model
classifier=(LogisticRegression())
classifier.fit(x_train,y_train)
y_pred =classifier.predict(x_test)
#y_pred(list(zip(y_test,y_pred)))
#print(y_test)
#%%
from sklearn.metrics import confusion_matrix , accuracy_score ,classification_report
cfm=confusion_matrix(y_test,y_pred)

print(cfm)

print('classification report:')

print(classification_report(y_test,y_pred))

accuracy_score=accuracy_score(y_test,y_pred)
print('Accuracy model: ' , accuracy_score)



#%%
#ADJUSTING THE THRESOLD
#%%
#store the predicted probabiliies,IT AVE HE PEDICTED VALUES THATLOGISTIC REGRESSION REGRESSION GENERATED
y_pred_prob = classifier.predict_proba(x_test)
print(y_pred_prob)
#%%
y_pred_class=[]
for value in y_pred_prob[:,0]:
    if value < 0.6:
        y_pred_class.append(1)
    else:
        y_pred_class.append(0)
#print(u_pred_class)

#%%
from sklearn.metrics import confusion_matrix , accuracy_score,classification_report
cfm=confusion_matrix(y_test.tolist(),y_pred_class)#both  sholud be list

print(cfm)

#print('classification report:')

#print(classification_report(y_test.tolist(),y_pred_class))

accuracy_score=accuracy_score(y_test,y_pred_class)
print('Accuracy model: ' , accuracy_score)
#%%
for a in np.arange(0,1,0.05):
    predict_mine=np.where(y_pred_prob[:,0]<a,1,0)
    cfm=confusion_matrix(y_test.tolist(),predict_mine)#both  sholud be list
    total_err=cfm[0,1]+cfm[1,0]
    print("Error at threshold", a,":", total_err,"type 2 error:",cfm[1,0])
#[1,0] and [0,1] aere elements position in matrics

#%%
########### FEATURE SELECTION
#%%
x=adult_df_rev.values[:,:-1]
y=adult_df_rev.values[:,-1]
from sklearn.model_selection import train_test_split

#split the data into train and test

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3
                                                    ,random_state=10)
from sklearn.linear_model import LogisticRegression
#create a model
#fitting  traning data tp the model
classifier=(LogisticRegression())
#%%
colname=adult_df_rev.columns[:]
#%%
from sklearn.feature_selection import RFE
rfe =RFE(classifier,6)
model_rfe =rfe.fit(x_train,y_train)
print('Num feature: ',model_rfe.n_features_)
print('Selected Feature:' )
print(list(zip(colname,model_rfe.support_)))
print('Feature Ranking: ',model_rfe.ranking_)
#%%
#model created by "rfe" itself,model has been madee after feature selection.
y_pred=model_rfe.predict(x_test)
#print(list(zip(y_test,y_pred)))
print(y_pred)
from sklearn.metrics import confusion_matrix , accuracy_score ,classification_report
cfm=confusion_matrix(y_test,y_pred)

print(cfm)

print('classification report:')

print(classification_report(y_test,y_pred))

accuracy_score=accuracy_score(y_test,y_pred)
print('Accuracy model: ' , accuracy_score)
#%%
x = adult_df_rev.values[:,:-1]
y = adult_df_rev.values[:,-1]
#%%
 
#skbest selects those columns who have more rank ie 9, 8 ie in descending order
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

test = SelectKBest(score_func=chi2, k=5)

fit1 = test.fit(x, y)

print(fit1.scores_)#it gives list of variable scores
new_x = fit1.transform(x)

print(new_x)
#%%
 

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(new_x)#

x = scaler.transform(new_x)#it transform variable , can give warning bcs it converts data type
#%%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3
                                                    ,random_state=10)
#why random =10 <-
from sklearn.linear_model import LogisticRegression
#create a model
#fitting  traning data tp the model
#classifier=(LogisticRegression())
from sklearn.linear_model import LogisticRegression
#create a model
#fitting  traning data tp the model
classifier=(LogisticRegression())
classifier.fit(x_train,y_train)
y_pred =classifier.predict(x_test)
#%%
from sklearn.metrics import confusion_matrix , accuracy_score ,classification_report
cfm=confusion_matrix(y_test,y_pred)

print(cfm)

print('classification report:')

print(classification_report(y_test,y_pred))

accuracy_score=accuracy_score(y_test,y_pred)
print('Accuracy model: ' , accuracy_score)
#%%











































