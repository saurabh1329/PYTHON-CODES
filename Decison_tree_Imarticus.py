# -*- coding: utf-8 -*-
"""
Created on Sat May 26 14:46:31 2018

@author: Saurabh
"""

#%%
import pandas as pd
import numpy as np
train_data=pd.read_csv('OBS_Network_data.csv',header=None)
#%%
train_data.columns=["Node","Utilised Bandwith Rate","Packet Drop Rate","Full_Bandwidth","Average_Delay_Time_Per_Sec",
"Percentage_Of_Lost_Pcaket_Rate","Percentage_Of_Lost_Byte_Rate","Packet Received Rate","of Used_Bandwidth",
"Lost_Bandwidth","Packet Size_Byte","Packet_Transmitted","Packet_Received","Packet_lost","Transmitted_Byte",
"Received_Byte","10-Run-AVG-Drop-Rate","10-Run-AVG-Bandwith-Use","10-Run-Delay","Node Status","Flood Status","Class"]
#%%
train_data.head(2)
train_data.shape
train_data.isnull().sum()
#%%
#create a copy of file
network_data=pd.DataFrame.copy(train_data)
network_data.head(2)
#%%
network_data=network_data.drop('Packet Size_Byte', axis=1)
network_data.shape
#%%
colname=["Node","Full_Bandwidth","Node Status","Class"]
colname
from sklearn import preprocessing
le={}
for x in colname:
    le[x]=preprocessing.LabelEncoder()
for x in colname:
    network_data[x]=le[x].fit_transform(network_data.__getattr__(x))
#%%
network_data.head(2)
x=network_data.values[:,:-1]
y=network_data.values[:,-1]
#%%
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(x)
x=scaler.transform(x)

scaler.fit(x)
x=scaler.transform(x)
#%%
from sklearn.model_selection import train_test_split
#split thedata into test and train

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=10)
#%%
#predicting using the decison tree
from sklearn.tree import DecisionTreeClassifier

model_DecisionTree=DecisionTreeClassifier()
model_DecisionTree.fit(x_train,y_train)
#%%
#fit the model on the data and predict the values
y_pred=model_DecisionTree.predict(x_test)
print(list(zip(y_test,y_pred)))

#%%
from sklearn.metrics import confusion_matrix , accuracy_score ,classification_report
cfm=confusion_matrix(y_test,y_pred)

print(cfm)

print('classification report:')

print(classification_report(y_test,y_pred))

accuracy_score=accuracy_score(y_test,y_pred)
print('Accuracy model: ' , accuracy_score)
#%%
#confirrming with model 
classifier=(DecisionTreeClassifier())
from sklearn import cross_validation
#performing kfold validation 
kfold_cv=cross_validation.KFold(n=len(x_train),n_folds=10)
print(kfold_cv)

#running the model using scoring metric as acuracy 
kfold_cv_result=cross_validation.cross_val_score(estimator=classifier,X=x_train,y=y_train,scoring="accuracy",cv=kfold_cv)
print(kfold_cv_result)

#finding the mean
print(kfold_cv_result.mean())

"""for train_value,test_value in kfold_cv:
classifier.fit(X_train[train_value],Y_train[train_value]).predict(X_train[test_value])

Y_pred=classifier.predict(X_test)
#print(list(zip)(Y_test,Y_pred)))"""
#%%

#%%

#GRAPHICAL MODEL OF DECISION TREE

from sklearn import tree
with open("model_Decison_Tree.txt","w") as f:
    f=tree.export_graphviz(model_DecisionTree,out_file=f)
#%%
#WEBGRAPHVIZ.COM
#%%
#RUNNING A EXTRA TREES CLASSIFIER MODEL
    
#predicting using the Bagging_Classifier
from sklearn.ensemble import ExtraTreesClassifier

model=(ExtraTreesClassifier(21))
#fit the model on the data and predict the values
model=model.fit(x_train,y_train)

Y_pred=model.predict(x_test)

 

from sklearn.metrics import confusion_matrix, accuracy_score,classification_report
#confusion matrix
print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))
#%%

#random forest classifier
#predicting using the Bagging_Classifier
from sklearn.ensemble import RandomForestClassifier

model=(RandomForestClassifier(501))
#501 is the no. of decison of tree to be run, or no of times the model run
#fit the model on the data and predict the values
model=model.fit(x_train,y_train)

Y_pred=model.predict(x_test)
#%%
#Running  ADABOOST Classifier
from sklearn.ensemble import AdaBoostClassifier

model_AdBoost=(AdaBoostClassifier(base_estimator=DecisionTreeClassifier()))
#501 is the no. of decison of tree to be run, or no of times the model run
#fit the model on the data and predict the values
model=model.fit(x_train,y_train)

Y_pred=model.predict(x_test)
#%%
#Running  GradientBoostingClassifier Classifier

from sklearn.ensemble import GradientBoostingClassifier

model_Gradient=GradientBoostingClassifier()
#501 is the no. of decison of tree to be run, or no of times the model run
#fit the model on the data and predict the values
model=model.fit(x_train,y_train)

Y_pred=model.predict(x_test)
#%%
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier

 

# create the sub models
estimators = []
model1 = LogisticRegression()
estimators.append(('log', model1))
model2 = DecisionTreeClassifier()
estimators.append(('cart', model2))
model3 = SVC()
estimators.append(('svm', model3))

 

# create the ensemble model
ensemble = VotingClassifier(estimators)
ensemble.fit(x_train,y_train)
Y_pred=ensemble.predict(x_test)
#print(Y_pred)

 

from sklearn.metrics import confusion_matrix, accuracy_score,classification_report
#confusion matrix
print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,Y_pred))
print(classification_report(y_test,Y_pred))
#%%




