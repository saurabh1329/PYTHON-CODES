# -*- coding: utf-8 -*-
"""
Created on Sun May  6 15:37:27 2018

@author: Saurabh
"""

import numpy as np
np.__version__
a=np.array([1,2,3,4])
print(a)
print(type(a))
#ndarray <- n dimensional array
#%%
a=np.array([(1,2,3),(4,5,6)])
print(a)


#created 2 dimensional array and always same no. of elements
#%%
a=np.array([(1,2,3),4,(4,5,6)])
print(a)
print(a.ndim)#gives dimension
#%%
np.ones((2,3,4)) #bydefault data type is float
np.zeros((2,3,4),dtype=np.int)
np.full((2,2),7.5)
a=np.identity(3,dtype=np.int8)#int8 is used to allocate 8 bit
print(a)
np.random.random((5,2))
#%%
a=np.array([(1,2,3,4),(5,6,7,8)])
print(a)
print(a.size)#no. of elements
print(a.shape)
#%%
a=np.array([(1,2,3,4),(5,6,7,8)])
a=np.resize(a,(6,4))
print(a)
a=np.reshape(a,(8,3))# reshape must be a multple of array size
print(a) #
#%%
#fetchin articular element
a=np.array([(1,2,3,4),(5,6,7,8)])
print(a[0,2])
#%%

a=np.array([(8,9),(10,11),(12,13)])
print(a[0:2,:])
#%%
a=np.array([(1,2,3,4),(5,6,7,8)])
print(a[:,2])

#%%
# used to divide in equal parts  where 1,100 is range and divison is 5
a=np.linspace(1,100,5)
print(a)
#%%
a=np.arange(10,26,5)
print(a)
a=np.arange(10,25,5)# last value is excluded
print(a)
#%%
a=np.array([(8,9),(10,11),(12,13)])
print(a)
print(len(a))
print(a.min())
print(a.max())
print(a.sum())
#%%

#math functoion
np.array([(1,0,3),(3,4,5)])
print(np.sqrt(a))
print(np.std(a))
#%%
x=np.array([(1,2),(3,4)])
y=np.array([(5,6),(3,4)])
print(x+y)
print(x-y)
print(x*y)
print(x/y)
print(x@y)
print(x%y)
print(np.dot(x,y))
print(np.add(x,y))
#np.add, np.subtract , np.multiply,np.divide,np.remainder

#%%
#help in jupyter
#?np.mean()
#%%
#CONCATENATION OF ARRAYS
print(x)

print(y)

d=np.concatenate((x,y))
print(d)

print(np.vstack((x,y)))

print(np.hstack((x,y)))

#%%

print(x)
print(x.ravel())
print(x.reshape(2,2))
#%%
z=np.loadtxt(r'C:\Users\Saurabh\Desktop\array_data.txt',skiprows=1,delimiter=",",dtype=int,unpack=True)
print(z)
#unpack is used to make transpose or not depend on tRue or false value of it .
#%%
array_2=np.genfromtxt(r'C:\Users\Saurabh\Desktop\array_data.txt',skip_header=1,delimiter=",",dtype=int,filling_values=0)
print(array_2)
#%%

#conversion of data tye of array after creation of it
print(array_2.astype(int))
print(type(array_2))
m=array_2.astype(int)
array_2.dtype
#%%
x=np.array([(1,2,3),(3,4,5)])
y=np.array([(6,7,8),(9,10,11)])
new_array=np.append(x,y)
new_array
#%%
y=np.array([(6,7,8),(9,10,11)])

new_array=np.insert(y,[1,4],5)
print(new_array)
#%%

y=np.array([(6,7,8),(20,10,11)])

new_array=np.delete(y,[1])
print(new_array)
























