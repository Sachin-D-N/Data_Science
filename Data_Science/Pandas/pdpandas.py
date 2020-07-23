"Pandas"
#it is used for data analysis
#popular to process tabular data
#used in Data maniculation,visualization,building ML models
#seies, 
import pandas as pd
pd.Series()
a=pd.Series(['sachin','1','8','s','NaN']) #it gives the indexes
pd.Series([[1,2],[5,6]])
a[0]
type(a[0])

a #numpy gives in terms as row pandas gives as a columns
a[0]='p'
a.values
a.index # gives the RangeIndex(start=0, stop=4, step=1)


a.sort_values(inplace=True,na_position='first')
a.sort_values(inplace=False,na_position='last')

#we can also give the series name
a.name='sachin'

a.index=[1,5,'p',8] #we can also set the index values

a.index=range(0,8,2)

a.value_counts

a.reindex([4]) #it gives the value of the index is have, if there is no value it returns Nan
#by index we will get the values

s_dict={'a':'sachin','b':'paddu','c':'love','d':None}

l1=pd.Series(s_dict) #in dict keys are taken as indexes
l1.name='life'
l1.sort_values(inplace=False,na_position='last')
l1.sort_values(inplace=False,na_position='first')

import numpy as np
l2=pd.Series(np.array([1,2,3,4]),index=np.array(['a','b','f','']))


#if there is an duplicates in series we can't reindex it
l2=pd.Series(np.array([1,2,3,]),index=np.array(['a','a','f','NaN']))
l2.reindex(['a']) #because duplicates keys are preetnt

l2.index(('a')) #error becase ibdex objects not callable

type(l2)
type(l2[0])

import pandas as pd
df=pd.read_csv('C:\\Users\\Dell\Desktop\\assinment ai courses\Pandas\\nyc_weather.csv')
df['Temperature'].max()

df=df.rename(columns={'Temperature':'Temp'}) #to change the column name

"data frame"
#it consits of rows and columns

df.shape #rows and columns

df.head() # it will prints the initial five rows

df.tail() #it will prints the last five rows

df.columns #gives the columns of the dataset

df.Temperature #which one you want you will see
df.Temperature.tail()
df['Temperature']='sachin'#in this way also saw the data

df.index
df.values
df.reindex([8])
df[['Temperature','EST']] #to get two columns

df.describe() #to get the satistical analysis


df[2:5]



df[df.Temperature==df.Temperature.max()] #it gives the which row is equal to max temperature
df.reindex([9])
df.EST[df.Temperature==df.Temperature.max()]


#to read the excel files
#pip3 install xlrd
import pandas as pd
df=pd.read_excel('C:\\Users\\Dell\Desktop\\assinment ai courses\Pandas\\weather_data.xlsx')
df.reindex([0])
df[2:3]

df.to_csv('new_csv')#writing df to csv
df.to_csv('newnoindex_csv',index=False) #writing df to csv

#pip 3 install openpyxl
df.to_excel('new.xlsx',sheet_name='weather_data') #it converts the file and store the file in excel format
                                                    #sheet name means=that sheet name we want to provide
df.info()         #it gives the information about the current data frame

df.values                                         

df=pd.read_csv('C:\\Users\\Dell\\Desktop\\assinment ai courses\\Pandas\\weather_data_cities.csv')
g=df.groupby('city')
for i,j in g:
    print(i)
    print(j) #indexes are copied from original data
    
#to get specific group
g.get_group('mumbai')

#to get the highest in eachgroip
g.max() 

#.loc v/s .iloc
#by default index and row numbers are same
#.loc gives the values by the index
#.iloc gives the values by the row

df=pd.DataFrame([1,2,3,4,5,6,7,8,9])

#to change the index 
df=pd.DataFrame([1,2,3,4,5,6,7,8,9],index=[10,11,12,13,14,15,16,17,18])
 
df.loc[15] 
df.iloc[2]  
df.loc[:15] #up to index no 15
df.iloc[:3] #uo to row no 3
df.iloc[::2] #step size

#concatinate
c=pd.DataFrame([5,6,7,8])
d=pd.DataFrame([9,10,11,12])    
pd.concat([c,d]) #it will concatinate the two data frames and both index are copied and concatinated
pd.concat([c,d],ignore_index=True) #index are not copied
pd.concat([c,d],axis=1) #both are comes side by side

#merge dataframes
e=pd.DataFrame({'city':['mysore','bangalore','hyd'],'humidity':[68,69,40]})
f=pd.DataFrame({'city':['mysore','bangalore','hyd','pun'],'temp':[5,6,8,9]})

e.columns
e=e.rename(columns={'city':'place'})

pd.merge(e,f,on='city') #on the city basis it will merge both ,this not includes pun because pun is not in e

#outer join
pd.merge(e,f,on='city',how='inner')#or left
g=pd.merge(e,f,on='city',how='outer')#or right it also includes pun

e.iloc[[0,1]] #it gives the oth ,and 1st row 

e.iloc[1:3,0] #rows and columns

e.iloc[[False,True,True]] #gives the 1st,2nd row


g.isnull
g.fillna(method='ffill') #nan filled with previous number
g.fillna(method='bfill') #filled with next value

#use of pandas
#Allows the use of labels for rows and columns
#Can calculate rolling statistics on time series data
#Easy handling of NaN values
#Is able to load data of different formats into DataFrames
#Can join and merge different datasets together
#It integrates with NumPy and Matplotlib 

# We import NumPy as np to be able to use the mathematical functions
import numpy as np
# We create a Pandas Series that stores a grocery list of just fruits
fruits= pd.Series(data = [10, 6, 3,], index = ['apples', 'oranges', 'bananas'])

# We display the fruits Pandas Series
fruits

# We print fruits for reference
print('Original grocery list of fruits:\n', fruits)

# We apply different mathematical functions to all elements of fruits
print()
print('EXP(X) = \n', np.exp(fruits))
print() 
print('SQRT(X) =\n', np.sqrt(fruits))
print()
print('POW(X,2) =\n',np.power(fruits,2)) # We raise all elements of fruits to the power of 2


#dataframe
# We import Pandas as pd into Python
import pandas as pd

# We create a dictionary of Pandas Series 
items = {'Bob' : pd.Series(data = [245, 25, 55], index = ['bike', 'pants', 'watch']),
         'Alice' : pd.Series(data = [40, 110, 500, 45], index = ['book', 'glasses', 'bike', 'pants'])}

# We print the type of items to see that it is a dictionary
print(type(items))

# We create a Pandas DataFrame by passing it a dictionary of Pandas Series
shopping_carts = pd.DataFrame(items)

# We display the DataFrame
shopping_carts

# We create a dictionary of Pandas Series without indexes
data = {'Bob' : pd.Series([245, 25, 55]),
        'Alice' : pd.Series([40, 110, 500, 45])}

# We create a DataFrame
df = pd.DataFrame(data)

# We display the DataFrame
df


# We print some information about shopping_carts
print('shopping_carts has shape:', shopping_carts.shape)
print('shopping_carts has dimension:', shopping_carts.ndim)
print('shopping_carts has a total of:', shopping_carts.size, 'elements')
print()
print('The data in shopping_carts is:\n', shopping_carts.values)
print()
print('The row index in shopping_carts is:', shopping_carts.index)
print()
print('The column index in shopping_carts is:', shopping_carts.columns)


#We Create a DataFrame that only has Bob's data
bob_shopping_cart = pd.DataFrame(items, columns=['Bob'])

# We display bob_shopping_cart
bob_shopping_cart

# We Create a DataFrame that only has selected items for both Alice and Bob
sel_shopping_cart = pd.DataFrame(items, index = ['pants', 'book'])

# We display sel_shopping_cart
sel_shopping_cart

# We Create a DataFrame that only has selected items for Alice
alice_sel_shopping_cart = pd.DataFrame(items, index = ['glasses', 'bike'], columns = ['Alice'])

# We display alice_sel_shopping_cart
alice_sel_shopping_cart


# We create a dictionary of lists (arrays)
data = {'Integers' : [1,2,3],
        'Floats' : [4.5, 8.2, 9.6]}

# We create a DataFrame 
df = pd.DataFrame(data)

# We display the DataFrame
df


# We create a dictionary of lists (arrays)
data = {'Integers' : [1,2,3],
        'Floats' : [4.5, 8.2, 9.6]}

# We create a DataFrame and provide the row index
df = pd.DataFrame(data, index = ['label 1', 'label 2', 'label 3'])

# We display the DataFrame
df


# We create a list of Python dictionaries
items2 = [{'bikes': 20, 'pants': 30, 'watches': 35}, 
          {'watches': 10, 'glasses': 50, 'bikes': 15, 'pants':5}]

# We create a DataFrame 
store_items = pd.DataFrame(items2)

# We display the DataFrame
store_items


# We create a list of Python dictionaries
items2 = [{'bikes': 20, 'pants': 30, 'watches': 35}, 
          {'watches': 10, 'glasses': 50, 'bikes': 15, 'pants':5}]

# We create a DataFrame  and provide the row index
store_items = pd.DataFrame(items2, index = ['store 1', 'store 2'])

# We display the DataFrame
store_items


# We print the store_items DataFrame
print(store_items)

# We access rows, columns and elements using labels
print()
print('How many bikes are in each store:\n', store_items[['bikes']])
print()
print('How many bikes and pants are in each store:\n', store_items[['bikes', 'pants']])
print()
print('What items are in Store 1:\n', store_items.loc[['store 1']])
print()
print('How many bikes are in Store 2:', store_items['bikes']['store 2'])

# We add a new column named shirts to our store_items DataFrame indicating the number of
# shirts in stock at each store. We will put 15 shirts in store 1 and 2 shirts in store 2
store_items['shirts'] = [15,2]#column ,row

# We display the modified DataFrame
store_items


# We add a new column named shirts to our store_items DataFrame indicating the number of
# shirts in stock at each store. We will put 15 shirts in store 1 and 2 shirts in store 2
store_items['shirts'] = [15,2]

# We display the modified DataFrame
store_items


# We make a new column called suits by adding the number of shirts and pants
store_items['suits'] = store_items['pants'] + store_items['shirts']

# We display the modified DataFrame
store_items


# We create a dictionary from a list of Python dictionaries that will number of items at the new store
new_items = [{'bikes': 20, 'pants': 30, 'watches': 35, 'glasses': 4}]

# We create new DataFrame with the new_items and provide and index labeled store 3
new_store = pd.DataFrame(new_items, index = ['store 3'])

# We display the items at the new store
new_store


# We append store 3 to our store_items DataFrame
store_items = store_items.append(new_store)

# We display the modified DataFrame
store_items

# We add a new column using data from particular rows in the watches column
store_items['new watches'] = store_items['watches'][1:]

# We display the modified DataFrame
store_items



"""to insert new columns into the DataFrames anywhere we want. 
The dataframe.insert(loc,label,data) method allows us to insert a new column in ,
the dataframe at location loc, with the given column label, and given data. 
Let's add new column named shoes right before the suits column.
 Since suits has numerical index value 4 then we will use this value as loc."""
 
# We insert a new column with label shoes right before the column with numerical index 4
store_items.insert(4, 'shoes', [8,5,0])

# we display the modified DataFrame
store_items


"""To delete rows and columns from our DataFrame we will use the .pop() and .drop() methods.
The .pop() method only allows us to delete columns, while the .drop() method,
can be used to delete both rows and columns by use of the axis keyword"""

# We remove the new watches column
store_items.pop('new watches')

# we display the modified DataFrame
store_items


# We remove the watches and shoes columns
store_items = store_items.drop(['watches', 'shoes'], axis = 1)

# we display the modified DataFrame
store_items

# We remove the store 2 and store 1 rows
store_items = store_items.drop(['store 2', 'store 1'], axis = 0)

# we display the modified DataFrame
store_items

"rename"
# We change the column label bikes to hats
store_items = store_items.rename(columns = {'bikes': 'hats'})

# we display the modified DataFrame
store_items

# We change the row label from store 3 to last store
store_items = store_items.rename(index = {'store 3': 'last store'})

# we display the modified DataFrame
store_items


# We change the row index to be the data in the pants column
store_items = store_items.set_index('pants')

# we display the modified DataFrame
store_items


# We create a list of Python dictionaries
items2 = [{'bikes': 20, 'pants': 30, 'watches': 35, 'shirts': 15, 'shoes':8, 'suits':45},
{'watches': 10, 'glasses': 50, 'bikes': 15, 'pants':5, 'shirts': 2, 'shoes':5, 'suits':7},
{'bikes': 20, 'pants': 30, 'watches': 35, 'glasses': 4, 'shoes':10}]

# We create a DataFrame  and provide the row index
store_items = pd.DataFrame(items2, index = ['store 1', 'store 2', 'store 3'])

# We display the DataFrame
store_items


# We count the number of NaN values in store_items
x =  store_items.isnull().sum().sum()

# We print x
print('Number of NaN values in our DataFrame:', x)


store_items.isnull() #it gives in boolean


store_items.isnull().sum() #nan counts gives by column


# We print the number of non-NaN values in our DataFrame
print()
print('Number of non-NaN values in the columns of our DataFrame:\n', store_items.count())


# We drop any rows with NaN values
store_items.dropna(axis = 0)


# We drop any columns with NaN values
store_items.dropna(axis = 1)


"""Notice that the .dropna() method eliminates (drops) the rows or columns with NaN values out of place. 
This means that the original DataFrame is not modified. You can always remove the desired rows or columns ,
in place by setting the keyword inplace = True inside the dropna() function."""


# We replace all NaN values with 0
store_items.fillna(0)


"""We can also use the .fillna() method to replace NaN values with previous values in the DataFrame, 
this is known as forward filling. When replacing NaN values with forward filling, we can use previous 
values taken from columns or rows. The .fillna(method = 'ffill', axis) will use the forward filling (ffill) method to
replace NaN values using the previous known value along the given axis. """
 
#We replace NaN values with the previous value in the column
store_items.fillna(method = 'ffill', axis = 0)

#We replace NaN values with the previous value in the row
store_items.fillna(method = 'ffill', axis = 1)

#We replace NaN values with the next value in the column
store_items.fillna(method = 'bfill', axis = 0)
store_items.fillna(method = 'backfill', axis = 0)

#We replace NaN values with the next value in the row
store_items.fillna(method = 'backfill', axis = 1)

#We replace NaN values by using linear interpolation using column values
store_items.interpolate(method = 'linear', axis = 0)

#We replace NaN values by using linear interpolation using row values
store_items.interpolate(method = 'linear', axis = 1)

store_items.interpolate(axis=1)




"""pandas practice"""
import numpy as np
import pandas as pd

df=pd.read_csv('H:\\pandas material\\Video_Lecture_NBs\\titanic.csv')
pd.options.display.max_rows=20 #it displays the max rows starting 100 and ending 100 total 200

df.columns

df.head() #default it gives the starting 5 rows
df.tail() #default it gives the ending 5 rows

df.head(n=10) #how many we mentioned that much of starting rows it given
df.tail(n=8) #how many we mentioned that much of ending rows it given

df.index #gives the index of start ,stop,step

df.info() #gives information about dataframe

df.size

df.shape #gives the shape (rows,columns)

df.describe() #gives the statistical calculations

type(df)

len(df)

round(df,0) #round the values to zero

min(df) #min alphabetically starting letter

max(df) #max alphabetically starting letter

df.min() #min elements in the dataframe

df.max() #max elements in the dataframe

df.sort_values(by='pclass',ascending=False)

#selecting the columns

df[['age','sex']] #gives the two columns

df.age#it gives only single column

df.head()

df[0:8] #rows [start:end]

df[0:8:2] #rows {start,stop,step}

#iloc by row

df.iloc[0]
"""
survived       0
pclass         3
sex         male
age           22
sibsp          1
parch          0
fare        7.25
embarked       S
deck         NaN
Name: 0, dtype: object"""

df.iloc[[0,1,2,3]]

df.iloc[-1]

df.iloc[1:8] #row by slicing

df.head(n=25)


df.iloc[:,1] #rows,column

df.iloc[1:3,2:3] #rows ,column slicing


df.iloc[1:4,[4,5,6,7]]

df.columns


#loc index based

df.loc[28] #it gives the 28 the index values

df.head(n=29)

df.loc[1:2] #both one and 2


df.loc[0:4:2] #it incluses the end

df.loc[[1,2,4]]

df.columns

df.loc[1:5,['sex','age']] #row ,column index based or label based

df[['year','age']].loc[1:5]

df.head()

df=pd.read_csv('H:\\pandas material\\Video_Lecture_NBs\\summer.csv')

df.columns


age=df['age']

age.count()

age.describe()

age.max()

age.min()

age.ptp()#it gives the max-min


age.unique() #it is only available for the series

len(age.unique()) #includes the missing values

age.nunique() #excludes the missing values

age.value_counts(dropna=True)

age.value_counts(dropna=False) #including missing values

age.value_counts(dropna=True,sort=True,ascending=True)

age.value_counts(dropna=True,sort=True,ascending=False)

age.value_counts(dropna=True,sort=True,ascending=False,normalize=True) #count/total count


age.value_counts(dropna=True,sort=True,ascending=False,normalize=True,bins=10) #gives the bins and devides


df['age'].unique()


df.sum( )

df=pd.read_csv('H:\\pandas material\\Video_Lecture_NBs\\summer.csv')

df.columns

df.info()

type(df)

athlete=df['Athlete']

type(athlete)

athlete.size

athlete.count() #it will not include incase of missing value is present in it.

athlete.unique()

athlete.nunique()

len(athlete.unique())

df.Athlete.head(n=20)

df.iloc[1:2,:3] #difference b/w loc and iloc is iloc slices column also but loc slices rows only

df['Athlete']
df.columns
df[['Athlete','Year','Sport']]

df=pd.read_csv('H:\\pandas material\\Video_Lecture_NBs\\summer.csv',usecols=['Year','Athlete'],squeeze=False)


year=df['Year']

year[1:8:2]


df.iloc[:,:-1]

import pandas as pd

df=pd.read_csv('H:\\pandas material\\Video_Lecture_NBs\\summer.csv',index_col='Year') #we used the index column as the year column

df.loc[1996:2010]

df.iloc[1:2,:2]


df.sort_index(ascending=False) #sort

df.sort_index(ascending=False,inplace=True) #if inplace =True it doesnot return any value and changes in original data frame


df.sort_index(ascending=True,inplace=False)

df.sort_values(by ='Athlete' ,inplace=True)

df.sort_values(by ='Athlete' ,inplace=True)

df.nlargest(n=3,'Year') #gives the year row largest 3 years

df.nsmallest(n=3,columns='Year')

df.columns

df.Year.idmin()

df.axes #gives both column names and index ranges


