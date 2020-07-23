# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 10:33:44 2019

@author: Dell
"""

import pandas as pd
import numpy as np
ser=([[5,6,7,8],[8,9,10,11]])
ser.name='age'
ser
type(ser)
ser.values
type(ser.values)
ser.index
ser1=pd.Series([5,6,7,8],index=['a','b','c','d'])
ser=pd.Series([5,6,7,8],index=range(0,8,2))
ser1.index
ser1.reindex(['a'])
sample={'a':2,'b':3,'c':3,'d':5}
pd.Series(sample)
l1=pd.Series((sample),index=['a','b','c'])
l1.reindex(['e'])
import pandas as pd
import numpy as np
ser2=pd.Series([5,6,7,8])
ser2
ser2=pd.Series([[1,2,3,4],[5,6,7]])
type(ser2)
'''to change the index'''
p1=pd.Series(ser, index=['a','b','c','d'])
'''slicing using strings it includes the ending points also'''
p1['a':'d']
'''DataFrame'''
p2=pd.DataFrame(ser)
p3=pd.Series(ser,index=['e','f'])
p4=pd.DataFrame(p3)
'''index'''
p4=pd.DataFrame(ser,index=['a','d'],columns=['d','e','e','e'])

import pandas as pd
import numpy as np
data = np.array(['a','b','c','d'])
s = pd.Series(data,index=[100,101,102,103])
print (s)
'''for Dictionary'''
data = {'a' : 0., 'b' : 1., 'c' : 2.}
s = pd.Series(data)
print (s)

'''for Scalar value'''
#import the pandas library and aliasing as pd
import pandas as pd
import numpy as np
s = pd.Series(5, index=[0, 1, 2, 3])
print (s)
'''Column selection'''
d = {'one' : pd.Series([1, 2, 3], index=['a', 'b', 'c']),
   'two' : pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])}

df = pd.DataFrame(d)
print (df ['one'])
''''adding new column'''
df['three']=pd.Series([10,20,30],index=['a','b','c'])
print (df)
df['Four']=pd.Series([10,20,30],index=['a','b','c'])
print (df)
print ("Adding a new column using the existing columns in DataFrame:")
df['four']=df['one']+df['three']

print (df)
# Using the previous DataFrame, we will delete a column
# using del function
d = {'one' : pd.Series([1, 2, 3], index=['a', 'b', 'c']), 
   'two' : pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd']), 
   'three' : pd.Series([10,20,30], index=['a','b','c'])}

df = pd.DataFrame(d)
print ("Our dataframe is:")
print (df)

# using del function
print ("Deleting the first column using DEL function:")
del df['one']
print (df)

# using pop function
print ("Deleting another column using POP function:")
df.pop('two')
print (df)
'''finding the location'''
d = {'one' : pd.Series([1, 2, 3], index=['a', 'b', 'c']), 
   'two' : pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])}

df = pd.DataFrame(d)
print (df.loc['b'])
'''slicing '''
df = pd.DataFrame(d)
print (df[0:1])
#addition
import pandas as pd

df = pd.DataFrame([[1, 2], [3, 4]], columns = ['a','b'])
df2 = pd.DataFrame([[5, 6], [7, 8]], columns = ['c','d'])

df = df.append(df2)

#drop
# Drop rows with label 0
df = df.drop(0)

print (df)
df.pop('a')
del df['c']
df
print (df)
###panels 3d array
# creating an empty panel
import pandas as pd
import numpy as np

data = np.random.rand(2,4,5)
p = pd.Panel(data)
print (p)
#using rows
# creating an empty panel
import pandas as pd
import numpy as np
data = {'Item1' : pd.DataFrame(np.random.randn(4, 3)), 
   'Item2' : pd.DataFrame(np.random.randn(4, 2))}
p = pd.Panel(data)
print (p['Item1'])
#major axis
import pandas as pd
import numpy as np
data = {'Item1' : pd.DataFrame(np.random.randn(4, 3)), 
   'Item2' : pd.DataFrame(np.random.randn(4, 2))}
p = pd.Panel(data)
print (p.major_xs(1))
#minor axis
data = {'Item1' : pd.DataFrame(np.random.randn(4, 3)), 
   'Item2' : pd.DataFrame(np.random.randn(4, 2))}
p = pd.Panel(data)
print (p.minor_xs(1))


#####''axes' for series
ser2.axes
ser2.dtype
ser2.empty
ser2.ndim
ser2.size
ser2.values
ser2.head(2)
ser2.tail(0)

#### basic functionalities for data frame
d=([[1,2,3,4],
   [5,6,7,8],
   [5,8,74]])
c=pd.DataFrame(d)
c.T
c.axes
c.dtypes
c.empty
c.ndim
c.shape
c.size
c.head(2)
c.tail(3)
#for rename
print(c.rename (index=str, columns=({0:'a'}), inplace=True))
 
#descriptive statistics
import pandas as pd
import numpy as np
 
#Create a Dictionary of series
d = {'Name':pd.Series(['Tom','James','Ricky','Vin','Steve','Smith','Jack',
   'Lee','David','Gasper','Betina','Andres']),
   'Age':pd.Series([25,26,25,23,30,29,23,34,40,30,51,46]),
   'Rating':pd.Series([4.23,3.24,3.98,2.56,3.20,4.6,3.8,3.78,2.98,4.80,4.10,3.65])
}

#Create a DataFrame
df = pd.DataFrame(d)
print (df.sum())
#axis=1
#Create a Dictionary of series
d = {'Name':pd.Series(['Tom','James','Ricky','Vin','Steve','Smith','Jack',
   'Lee','David','Gasper','Betina','Andres']),
   'Age':pd.Series([25,26,25,23,30,29,23,34,40,30,51,46]),
   'Rating':pd.Series([4.23,3.24,3.98,2.56,3.20,4.6,3.8,3.78,2.98,4.80,4.10,3.65])
}
 
#Create a DataFrame
df = pd.DataFrame(d)
print (df.sum())
df.cumsum()
df.abs()
df.describe()
#tables
''''1	count()	Number of non-null observations
2	sum()	Sum of values
3	mean()	Mean of Values
4	median()	Median of Values
5	mode()	Mode of values
6	std()	Standard Deviation of the Values
7	min()	Minimum Value
8	max()	Maximum Value
9	abs()	Absolute Value
10	prod()	Product of Values
11	cumsum()	Cumulative Sum
12	cumprod()	Cumulative Product'''