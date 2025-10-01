# PANDAS (17/09/25) 

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import re
import math
import warnings
warnings.filterwarnings("ignore")
Series
Create Series
# Create series from Nump Array
v = np.array([1,2,3,4,5,6,7])
s1 = pd.Series(v)
s1

dtype: int32
#Datatype of Series
s1.dtype
dtype('int32')
```
# Number of bytes consumed by Series
s1.nbytes

# Shape of the Series
s1.shape

# number of dimensions
s1.ndim

# Length of Series
len(s1)

s1.count()

s1.size

# Create series from List 
s0 = pd.Series([1,2,3],index = ['a','b','c'])
s0

dtype: int64
# Modifying index in Series
s1.index = ['a' , 'b' , 'c' , 'd' , 'e' , 'f' , 'g']
s1

dtype: int32
# Create Series using Random and Range function
v2 = np.random.random(10)
ind2 = np.arange(0,10)
s = pd.Series(v2,ind2)
v2 , ind2 , s
(array([0.87790351, 0.21256923, 0.2833476 , 0.84976498, 0.17274437,
        0.36953613, 0.92661933, 0.13005525, 0.25394528, 0.43563311]),
 array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
 
 dtype: float64)
# Creating Series from Dictionary
dict1 = {'a1' :10 , 'a2' :20 , 'a3':30 , 'a4':40}
s3 = pd.Series(dict1)
s3

dtype: int64
pd.Series(99, index=[0, 1, 2, 3, 4, 5]) 

dtype: int64

Slicing Series

# Return all elements of the series
s[:]

dtype: float64
# First three element of the Series
s[0:3]

dtype: float64
# Last element of the Series
s[-1:]

dtype: float64
# Fetch first 4 elements in a series
s[:4]

dtype: float64
# Return all elements of the series except last two elements.
s[:-2]

dtype: float64

# Return all elements of the series except last element.
s[:-1]

dtype: float64
# Return last two elements of the series
s[-2:]

dtype: float64
# # Return last element of the series
s[-1:]

dtype: float64
s[-3:-1]

dtype: float64
Append Series
s2 = s1.copy()
s2

dtype: int32
s3

dtype: int64
# Append S2 & S3 Series
s4 = s2.append(s3)
s4

dtype: int64
# When "inplace=False" it will return a new copy of data with the operation performed
s4.drop('a4' , inplace=False)

dtype: int64
s4

dtype: int64
# When we use "inplace=True" it will affect the dataframe
s4.drop('a4', inplace=True)
s4

dtype: int64
s4 = s4.append(pd.Series({'a4': 7}))
s4

dtype: int64
