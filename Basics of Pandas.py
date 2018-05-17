# -*- coding: utf-8 -*-
"""
Created on Sat May 12 12:33:15 2018

@author: Saurabh
"""
#%%
import pandas as pd
import numpy as np
data=np.array(['101','102','103','104'])
s=pd.Series(data)
print(s)
print(s[1])
#%%
data=[['Amit',40],['Nikita',23],['Saurabh',21]]
df=pd.DataFrame(data,columns=['Name','Age'])
print(df)

data=[['Amit',40],['Nikita',23],['Saurabh',21]] # see difference with above one
df=pd.DataFrame(data)
print(df)
#%%

data={'Name':['Amit','Nikita','Clara'],'Age':[40,23,45]}

df=pd.DataFrame(data)
print(df)


df=pd.DataFrame(data,index=['Rank1','Rank2','Rank3'],columns=['Name','Age'])
print(df)
#%%

#ADDING A COLUMNS
df['Addres']=["Mumbai","Pune","Mumbai"]
print(df)
#%% #DID CALCULATION PART WITH COLUMN
df["Newcol"]=5
print(df)

df["Revised"]=df["Newcol"]*2
print(df)
#%% deleting a column

del df['Newcol']
print(df)

#%%
#DELTING COLUMN
#df=df.drop('Rank1')
df=df.drop('Revised',axis=1)
print(df)

#%%   Loc & iloc
#use it wgeen you have to access the row

print(df.loc["Rank2"])
print(df.loc["Rank2":"Rank3"])

print(df.iloc[0:2])
 #%%  ffetcihing a particular value woth its key value

print(df["Name"])

#accessing a subset
print(df[["Name" , "Age" , "Addres"]])

#%%
#Importing your work in a file 
df.to_csv('Sample.csv',index=False,header=True)

df.to_excel('Excel_sample.xls',index=False,header=True)
#print(df)
#%%

#reading your file

df=pd.read_csv('sample.csv')
print(df)

df=pd.read_excel('Excel_sample.xls')
print(df)

#%%
print(df.dtypes)# data types of all columns
print(df.Age.dtype)#dtatype of particular column
print(df.info)# whole details of data frame
print(df.shape)#to get dimenson into rows and columns

#%%
# sorting or rearranging
df.sort_values(["Name"],ascending=False)
#%%

print(df["Name"].unique()) # to get unique names
print(df["Name"].value_counts()) #  to check occurences of particular values in column ,its used in data cleaning



















