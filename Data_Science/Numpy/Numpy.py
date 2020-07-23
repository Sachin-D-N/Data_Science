"num py"
#it will efficient
#memory efficient
#fast Numerical operations

l=range(1000)
%timeit [i**2 for i in l] #it gives how much time it takes
#948 µs ± 149 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

import numpy as sp
a=sp.arange(1000)
%timeit (a**2)

#2.87 µs ± 213 ns per loop this is the reason numpy is fast

#creating numpy array
a=sp.array([[1,2,3,4], #two dimensional array
            [5,6,7,8]])

#one dimensional array
a=sp.array([1,2,3,5])

a.ndim #gives the dimension of the array

a.shape #gives the shape of array

len(a) #gives the how many  dimensions containing rows

a.ndim

#3d array
import numpy as sp
c=sp.array([[[1,5,6],[7,8,9,3]],[[8,7,6,5],[5,1,5,8]]]) #if the size in list is different it comes like list
c.ndim
"""array([[list([1, 5, 6]), list([7, 8, 9, 3])],
       [list([8, 7, 6, 5]), list([5, 1, 5, 8])]], dtype=object)"""

type(c)

c=sp.array([[1,2,3,4]])

c.shape

c.ndim


d=sp.array([[[1,2,3,4],[1,2,3,4]],[[1,2,3,4],[1,2,3,4]]])

type(d) #numpy.ndarray [dimensional array]

d.ndim #how many square brackets insides it

d.shape

d.dtype #dtype('int32')

array=sp.array([[[[1,2,3,4],
                 [5,6,8,9],
                 [1,4,5,7]],
                 [[1,2,3,4],
                 [5,6,8,9],
                 [1,4,5,7]]],
                [[[1,2,3,4],
                 [5,6,8,9],
                 [1,4,5,7]],
                 [[1,2,3,4],
                 [5,6,8,9],
                 [1,4,5,7]]]])

array.ndim
array.shape
array.size #total elements in the array


#1D array called vector
#2d array called matrix
#nd array called tensor

c=sp.array([[1,2,3,4],
           [5,6,7,8]])
a=sp.array([1,2,3,4])
for i in a:
    print('\n',i)
for i in c:
    print(i,'\n')
    
for i in array:   #it is by shape(2,2,3,4)
    print('\n',i)
    print('next')
    for j in i:
        print('\n',j)
        print('next1')
        for k in j:
            print('\n',k)
            print('next 2')
            for l in k:
                print('\n',l)
                print('next3')

array.itemsize #gives it count of elements in array
                
#it is complex and running out of time that's why in a numpy flat function is there to get single element
for i in array.flat: #it is slow and occufies more memory
    print(i,end=' ')
    
#we can store this in list or tuple
%timeit list(array.flat)
array.flat[0]=5 #we can also change the value 
    
for i in sp.ravel(array): #ravel also doing same as the 
    print(i,end=" ")      #it is occupies less memory fast compare to flat

array.ravel()[0]=3


%timeit sp.ravel(array) 

%timeit (array.flat)  
#arrays are homogeneous we want to give the same type of data

#if give the integers and strings defaultly it converts it in to the string dtype is unicode
    
a=sp.array([[1,2,3,4],['sachin','paddu','good','pair']]) #dtype=u11 means unicode

#using arange function create array
range(0.1) #in range we can't take flot values get type error

# but in arange we can use arange
sp.arange(5.0,10,.1) 


z=sp.arange(10)
#array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) #it is same as the range function

z=sp.arange(1,10,1) #start ,stop,step

#we can also use linspace
import numpy as sp
z=sp.linspace(1,10,12) #start,stop,how many points

"""array([ 1.        ,  1.81818182,  2.63636364,  3.45454545,  4.27272727,
        5.09090909,  5.90909091,  6.72727273,  7.54545455,  8.36363636,
        9.18181818, 10.        ])"""
z=sp.linspace(0,.1,12,endpoint=False) #if we give like that end poind is exclusive

#using zero ,ones and full
import numpy as sp
sp.ones((2,3))  #row ,column
"""array([[1., 1., 1.],
       [1., 1., 1.]])"""
    
sp.ones(10)

sp.zeros(5)# array([0., 0., 0., 0., 0.]) 1D

import numpy as sp
# We create a 3 x 4 ndarray full of zeros. 
X = sp.zeros((3,4))

# We print X
print()
print('X = \n', X)
print()

# We print information about X
print('X has dimensions:', X.shape)
print('X is an object of type:', type(X))
print('The elements in X are of type:', X.dtype)

sp.zeros((5,3,2))

sp.full(10,5) #(how many times, fill with)

sp.full((10,5),5) #(rows,columns),fill with

#random number 
"random numbers are comes b/w 0 and 1"

sp.random.random(5) #how many times
"""
array([0.92279252, 0.63123068, 0.55243296, 0.49906866, 0.05030687])"""

sp.random.random((5,2)) #rows,columns

#NumPy also allows us to create ndarrays with random integers within a particular interval. 
#The function np.random.randint(start, stop, size = shape) creates an 
#ndarray of the given shape with random integers in the half-open interval [start, stop). Let's see an example:

# We create a 3 x 2 ndarray with random integers in the half-open interval [4, 15).
X = sp.random.randint(4,15,size=(3,2))

# We print X
print()
print('X = \n', X)
print()

# We print information about X
print('X has dimensions:', X.shape)
print('X is an object of type:', type(X))
print('The elements in X are of type:', X.dtype)

# The function np.random.normal(mean, standard deviation, size=shape), for example,
# creates an ndarray with the given shape that contains random numbers picked from a normal (Gaussian) distribution with the given mean and standard deviation
# We create a 1000 x 1000 ndarray of random floats drawn from normal (Gaussian) distribution
# with a mean of zero and a standard deviation of 0.1.
X = sp.random.normal(0, 0.1, size=(1000,1000))

# We print X
print()
print('X = \n', X)
print()

# We print information about X
print('X has dimensions:', X.shape)
print('X is an object of type:', type(X))
print('The elements in X are of type:', X.dtype)
print('The elements in X have a mean of:', X.mean())
print('The maximum value in X is:', X.max())
print('The minimum value in X is:', X.min())
print('X has', (X < 0).sum(), 'negative numbers')
print('X has', (X > 0).sum(), 'positive numbers')

#empty

sp.empty((5,3)) #it is fill with zunk values are almost equal to zero

sp.empty(5)#array([0.92279252, 0.63123068, 0.55243296, 0.49906866, 0.05030687])

sp.empty((5,2,2,3))#shape

#tolist converts array into list
sp.arange(0,5,1).tolist()
#[0, 1, 2, 3, 4]
type(sp.arange(0,5,1).tolist()) #type is list

#using eye means identity matrix
"""a fundamental array in Linear Algebra is the Identity Matrix. 
An Identity matrix is a square matrix that has only 1s in its main diagonal and zeros everywhere else. 
The function np.eye(N) creates a square N x N ndarray corresponding to the Identity matrix.
 Since all Identity Matrices are square, the np.eye() function only takes a single integer as an argument. """
 # We create a 5 x 5 Identity matrix. 
X = sp.eye(5)

# We print X
print()
print('X = \n', X)
print()

# We print information about X
print('X has dimensions:', X.shape)
print('X is an object of type:', type(X))
print('The elements in X are of type:', X.dtype) 


sp.eye(3) #diagonal elements are equal to one

sp.eye(3,5) #rows ,columns
"""
[[1., 0., 0., 0., 0.],
 [0., 1., 0., 0., 0.],
 [0., 0., 1., 0., 0.]])

    """
    
sp.eye(3,5,1)   #the diagonal elements are moving to next column
"""
array([[0., 1., 0., 0., 0.],
       [0., 0., 1., 0., 0.],
       [0., 0., 0., 1., 0.]])"""
    
sp.eye(3,3,1) 
"""
array([[0., 1., 0.],
       [0., 0., 1.],
       [0., 0., 0.]])"""
    
    
    
a=sp.array([[1,2,3],[5,6,7]])
a[1][0]

d=sp.array([[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4]])

#using diagonal

sp.diag([5,6]) #create Diagonal elements
"""array([[5, 0],
       [0, 6]])"""

sp.diag([[5,6],[6,5]]) #gives the diagonal elements of this array
#5,5

sp.diag(d,2) #gives the extract the diagonal elements

import numpy as sp
sp.random.rand(4)
sp.random.randn(4)

#change data type
sp.array([1,2,3,4],dtype=float) #giving in float

sp.zeros(3,dtype=complex)

#complex datatype
sp.array([1+2j,2j])

sp.array([1+2j,2j],dtype=str)

sp.array([1,2,3,4],dtype=bool) #giving in float

sp.array([ True,  True,  True,  True]) #boolean dtype


#indexing and slicing

d=sp.array([[1,2,3,4],[4,2,3,4],[5,2,3,4],[6,2,3,4]])
d
d[0]
d[1]
d[0][1]=8 #assigning value

d[3][2]=5

d[3][3]=6

d[3][0]=9

d[0:4:2] #slicing
"""
array([[1, 8, 3, 4],
       [5, 2, 3, 4]])"""
b=sp.array([2,0,1,5])    
    
d[1:3:2]=b[::-1]
d


sp.shares_memory(b,d)

a=sp.array([1,2,3,4,5,6,7])
b=a[::2]
sp.shares_memory(b,a) #both are sharing the memory using  by views internally


a
b
a[0]=5 #both are same in the memory because if 1 will changed other also changed

a  
b  #a and b are both changed


c=sp.copy(a)
sp.shares_memory(c,a) #using copy both memory is changed
c[0]=2
c
a
a[1]=7


#fancy indexing

#numpy arrays are indexing with boolian, integer arrays[by masks] called fancy indexing


f=sp.array([2,5,0,4,2,7,9,6,22])
mask=(f%2!=0)

f[mask] #where is !=0 in the array are will stay other can remove


#mask cann't create veiws

f[mask]=-2 #allo not equal to zero elements are will changes to -2

f=sp.array([2,5,0,4,2,7,9,6,22])

#index by another method

f[[1,2,0,5,2]] 

# array([5, 0, 2, 7, 0])

f[[2,5,7]]='1' #assigning new variables to all indexes

#reshaping
f=sp.array([2,5,0,4,2,7,9,6,22])
f.shape

f.reshape(3,3)

f.reshape(-1,4) #can't get 9 with multiply of 4

f.reshape(-1,3)

f.reshape(3,-5) #if u don't no both sides give one side -value it will atomatically calculated

e=sp.arange(200)
e.reshape(5,4,-1) #it will automatically calculated

e.reshape(-1,5,5)


import numpy as sp
a=[1,2,3,4]
b=[2,3,4,6]
a+b #in list it will concatinated


sp.array(a)+sp.array(b)
#array addittion added to index by index

sp.array(a)-sp.array(b)
sp.array(b)-sp.array(a) #array or matrix substraction

sp.array(a)*sp.array(b) #array multiplication

sp.array(a)/sp.array(b) #array devision


sp.array([1,3,4])+sp.array([2])
sp.array([1])+sp.array([2,3,4])

b=sp.arange(15).reshape(3,-1)
b.min()
b.max()
b.mean()
b.mean(axis=0) #collaps by row but operations by columns
b.mean(axis=1) #collaps by column but operations by rows
b.mean(axis=2) #error because dimensions is 2 only

c=sp.arange(50).reshape(2,5,5)
sp.diag(c) #error input must be ,1 or 2d
c[0][0][4]=40
c.min()
c.min(axis=0) #compares with both of the axis and print the which is minimum collaps with array
c.min(axis=1) #collups it by rows
c.min(axis=2) #collapsing by columns

"""
array([[[ 0,  1,  2,  3,  4], # 0        axis=1 compares with row   axis=2 compares with by column
        [ 5,  6,  7,  8,  9],  #5                                          and formed as row
        [10, 11, 12, 13, 14],  # 10 axis =2
        [15, 16, 17, 18, 19],  #15
        [20, 21, 22, 23, 24]],  #20                                      =[0,5,10,15,20]
       0,1,2,3,4                         =[0,1,2,3,4]                    [25,30,35,40,45]
             axis=1                      [25,26,27,28,29]

    [[25, 26, 27, 28, 29],     #25
        [30, 31, 32, 33, 34],   #30  axis=2
        [35, 36, 37, 38, 39],    #35
        [40, 41, 42, 43, 44],    #40
        [45, 46, 47, 48, 49]]])   #45
        25,26,27,28,29
          axis=1
    """
    

#broadcasting in numpy
"bias term"
 #to perform broadcasting atlisr one of the array shape is equal to 1

a=sp.array([1,5,6,8])  
c=sp.array([1])
a+c #broad casting

a=sp.array([5,4,6,2])

a.shape
a.ndim
b=sp.array([5,2,6])
b.shape
a+b #operands could not be broadcast together with shapes (4,) (3,) shapes are different

a=sp.array([[5,4,6,2],[5,4,6,2]]) #because b shape is (1*4)
b=sp.array([5,2,6,6])
a+b #because b shape is (1*4)

#statndard scalar
a=sp.array([[6,4,6,2],[5,3,2,1]])
(a-a.mean(axis=0))/a.std(axis=0)

a.T #for transpose

a.transpose() #it also used for transpose

#array slicing
a=sp.array([[6,4,6,2],[5,3,2,1],[1,2,5,7]])
a[2,0] #used for slicing
a[0][2] #it can also used for slicing

a[0,1]
a[[0,1]]  #either this
a[0:2]#or this

a[[0,1],[1,0]] #4,5
a[0,1],[1,0] # (4, [1, 0])
a[0,1][1,0] #error

d=sp.array((1,2,3))
#array([1, 2, 3])

sp.median(d)
sp.add(d,d)
sp.sub(d,d)
sp.mean(d)
sp.multiply(d,d)
sp.division
sp.divide(d,d)
sp.divmod(d,d)

divmod(4,2)

sp.dot(d,a) #array([19, 16, 25, 25])

sp.dot(d,d) #multiply a with a,b with b, c with c

d.T
d.transpose()

import numpy as sp
sp.mean(d)
sp.average(d, c=[5,8,7,3]) #22/11=2
a=sp.array((1,0,3))
n=sp.array((5,6,7))
c=sp.array((True,False,False))
sp.where(c,a,n) #if true return from a ,else return from c

sp.where(c,n,a) #if true return from n ,else return from a

sp.where(~c,n,a)


p=sp.array([1,2,3])
p+1 #one is added to every element of the array

#elementwise comparison

a=sp.ones(4)
b=sp.full(4,4)
a==b #array([False, False, False, False])
sp.equal(a,b)
a>b

#array wise comparision
sp.array_equal(a,b) # False

sp.logical_or(a,b) #for or operation in logic

sp.logical_and(a,b)

sp.logical_not(a)

a

sp.logical_xor(a,b)

sp.logical_xor(a)#error because we want give two arguments

#transcendental functions

sp.sin(a)

sp.cos(a)
b=sp.logical_not(a)
sp.sin(sp.logical_not(a))

sp.log(a)

sp.log(0) #-inf


sp.exp(sp.array([1,5,6,8,0])) # exp(x) means=e^x

sp.exp(0)


#basic reductions

x=sp.array([1,2,3,4])

sp.sum(x,axis=0)
sp.sum(x,axis=1) #error because we have only one dimension

y=sp.array([1,2,3,4,5,6,7,8,9,10]).reshape(2,-1)

sp.sum(y)
sp.sum(y,axis=0)
sp.sum(y,axis=1)


sp.min(y)

sp.argmin(y) #it,s gives the min of the index of an array

sp.argmin(y,axis=0)


sp.all([1,2,1])#all are true it gives true otherwise gives false

sp.any([0,0,2,0]) #if one is true in the array it gives true

a=sp.array([5,4,2,8])
b=sp.array([2,4,6,9])
(a>b).all() #it checks the all the elements in the array


#statistics
sp.mean([5,6,8,9])

sp.median([5,6,8,7])


sp.std([5,6,8,9])

#load the data into a object



FH =("C:\\Users\\Dell\\populations.txt")

data=sp.loadtxt(FH)

#y="G:\\ML Notes\\auto_mpg_dataset.csv"
#sp.loadtxt(y)

year,hares,lynxes,carrots=data.T

p=data[:,1:]

sp.std(p,axis=0)

sp.std(p,axis=1)

sp.argmax(p,axis=1)

"Broadcasting"
#tile

#use to create the more and more arrays (replicates the array)

a=sp.arange(5)
b=sp.tile(a,(2,1))#(how many rows ,how many times)

sp.tile(a,(2,2))

sp.tile(a,(2,3))
"""array([[0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4],
       [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4]])"""
    
b.T


c=sp.array([1,2,3,4,5])

b+c

#broadcasting by newaxis

d=sp.array([10,20,30,40])
e=sp.array([1,2,3])

#d is converted from 1D to 2 D

d=d[:,sp.newaxis] #if we give this it will converted to 2D array

d.shape

f=d+e

#flattening
#ravel convert to the 1 d array
sp.ravel(f)


#sometime reshape gives a copy

f=sp.arange(6)
g=f.reshape(3,-1)
g[0]=1

a=np.zeros((2,3))
b=a.T.reshape(3,2)

b[0]=5
a

#dimensions suffeling

a=sp.arange(6*2*3).reshape(3,3,4) #()#3 matrix , 3 rows and 4 columns

a[0,2,1] #oth matrix second row first column value slicing

#resizing
b=sp.arange(5)
b.resize((4)) #resizes from 4 to 3

b.resize((6,)) #if we extends it will added with zero

#sort
b=np.array([[1,4,8],[8,7,5]])
b.sort()
b.sort(axis=1)
b.sort(axis=0)
help(np.sort)

#fancy indexing
#it will gives the positions of the indexing after sorting
c=b.argsort()

d=np.array([5,2,1,3])
f=np.argsort(d)
d[f] #it gives the sorted array
f #gives the index of array


#stacks
"""NumPy also allows us to stack ndarrays on top of each other, or to stack them side by side. 
The stacking is done using either the np.vstack() function for vertical stacking, or the np.hstack() function for horizontal stacking. 
It is important to note that in order to stack ndarrays, the shape of the ndarrays must match"""

import numpy as np

# We create a rank 1 ndarray 
x = np.array([1,2])

# We create a rank 2 ndarray 
Y = np.array([[3,4],[5,6]])

# We print x
print()
print('x = ', x)

# We print Y
print()
print('Y = \n', Y)

# We stack x on top of Y
z = np.vstack((x,Y))

# We stack x on the right of Y. We need to reshape x in order to stack it on the right of Y. 
w = np.hstack((Y,x.reshape(2,1)))

# We print z
print()
print('z = \n', z)

# We print w
print()
print('w = \n', w)


p=sp.array([5,6,7,8,5])
s=sp.array([[1,2,3,4,5],[5,0,8,7,3],[5,2,7,2,8]])
sp.vstack((s[0],p))
sp.vstack((p,s[0]))

sp.hstack((s[0],p))

sp.vstack(s)
sp.hstack(s) #both are comes in horizontally

s.cumsum() #it will add end to end

sp.cumsum(s)

sp.cumsum(s,axis=0)
sp.cumsum(s,axis=1)

#it will gives the unique array
names = sp.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])

sp.unique(names)

set(names)

sp.in1d(names,["Bob"])

sp.isin(names,["Bob","Joe"])

sp.std(a)
sp.var(a)  # populations


s=sp.array([[1,2,3],[5,0,8],[5,2,8]])
sp.linalg.eig(s)


#for delete

"""Now, let's take a look at how we can add and delete elements from ndarrays.
 We can delete elements using the np.delete(ndarray, elements, axis) function. 
 This function deletes the given list of elements from the given ndarray along the specified axis.
 For rank 1 ndarrays the axis keyword is not required. For rank 2 ndarrays, 
axis = 0 is used to select rows, and axis = 1 is used to select columns. Let's see some examples:"""


# We create a rank 1 ndarray 
x = np.array([1, 2, 3, 4, 5])

# We create a rank 2 ndarray
Y = np.array([[1,2,3],[4,5,6],[7,8,9]])

# We print x
print()
print('Original x = ', x)

# We delete the first and last element of x
x = sp.delete(x, [0,4])

# We print x with the first and last element deleted
print()
print('Modified x = ', x)

# We print Y
print()
print('Original Y = \n', Y)

# We delete the first row of y
w = sp.delete(Y, 0, axis=0)

# We delete the first and last column of y
v = sp.delete(Y, [0,2], axis=1)

# We print w
print()
print('w = \n', w)

# We print v
print()
print('v = \n', v)

#append values
We can append values to ndarrays using the np.append(ndarray, elements, axis) function. This function appends the given list of elements to ndarray along the specified axis. Let's see some examples:

# We create a rank 1 ndarray 
x = sp.array([1, 2, 3, 4, 5])

# We create a rank 2 ndarray 
Y = sp.array([[1,2,3],[4,5,6]])

# We print x
print()
print('Original x = ', x)

# We append the integer 6 to x
x = p.append(x, 6)

# We print x
print()
print('x = ', x)

# We append the integer 7 and 8 to x
x = sp.append(x, [7,8])

# We print x
print()
print('x = ', x)

# We print Y
print()
print('Original Y = \n', Y)

# We append a new row containing 7,8,9 to y
v = sp.append(Y, [[7,7,8]], axis=0)

# We append a new column containing 9 and 10 to y
q = sp.append(Y,[[9],[10]], axis=1)

# We print v
print()
print('v = \n', v)

# We print q
print()
print('q = \n', q)

#insert

We can insert values to ndarrays using the np.insert(ndarray, index, elements, axis) function. This function inserts the given list of elements to ndarray right before the given index along the specified axis. Let's see some examples:

# We create a rank 1 ndarray 
x = sp.array([1, 2, 5, 6, 7])

# We create a rank 2 ndarray 
Y = sp.array([[1,2,3],[7,8,9]])

# We print x
print()
print('Original x = ', x)

# We insert the integer 3 and 4 between 2 and 5 in x. 
x = sp.insert(x,2,[3,4])

# We print x with the inserted elements
print()
print('x = ', x)

# We print Y
print()
print('Original Y = \n', Y)

# We insert a row between the first and last row of y
w = sp.insert(Y,1,[4,5,6],axis=0)

# We insert a column full of 5s between the first and second column of y
v = sp.insert(Y,1,5, axis=1)

# We print w
print()
print('w = \n', w)

# We print v
print()
print('v = \n', v)




"slicing of arrays"

#1. ndarray[start:end]
#2. ndarray[start:]
#3. ndarray[:end]


"""
The first method is used to select elements between the start and end indices. 
The second method is used to select all elements from the start index till the last index. 
The third method is used to select all elements from the first index till the end index. 
We should note that in methods one and three, the end index is excluded. """

# We create a 4 x 5 ndarray that contains integers from 0 to 19
X = sp.arange(20).reshape(4, 5)

# We print X
print()
print('X = \n', X)
print()

# We select all the elements that are in the 2nd through 4th rows and in the 3rd to 5th columns
Z = X[1:4,2:5]

# We print Z
print('Z = \n', Z)

# We can select the same elements as above using method 2
W = X[1:,2:5]

# We print W
print()
print('W = \n', W)

# We select all the elements that are in the 1st through 3rd rows and in the 3rd to 4th columns
Y = X[:3,2:5]

# We print Y
print()
print('Y = \n', Y)

# We select all the elements in the 3rd row
v = X[2,:]
v=X[2:]

# We print v
print()
print('v = ', v)

# We select all the elements in the 3rd column
q = X[:,2]

# We print q
print()
print('q = ', q)

# We select all the elements in the 3rd column but return a rank 2 ndarray
R = X[:,2:3]

# We print R
print()
print('R = \n', R)


"""
riginal array X is not copied in the variable Z. Rather, X and Z are now just two different names for the same ndarray. 
We say that slicing only creates a view of the original array. 
This means that if you make changes in Z you will be in effect changing the elements in X as well."""

# We create a 4 x 5 ndarray that contains integers from 0 to 19
X = np.arange(20).reshape(4, 5)

# We print X
print()
print('X = \n', X)
print()

# We select all the elements that are in the 2nd through 4th rows and in the 3rd to 4th columns
Z = X[1:4,2:5]

# We print Z
print()
print('Z = \n', Z)
print()

# We change the last element in Z to 555
Z[2,2] = 555

# We print X
print()
print('X = \n', X)
print()


#We can clearly see in the above example that if we make changes to Z, X changes as well.


"Diogonal elements"

 """The np.diag(ndarray, k=N) function extracts the elements along the diagonal defined by N. 
 As default is k=0, which refers to the main diagonal. 
 Values of k > 0 are used to select elements in diagonals above the main diagonal, and 
 values of k < 0 are used to select elements in diagonals below the main diagonal"""
 
 # We create a 4 x 5 ndarray that contains integers from 0 to 19
X = np.arange(25).reshape(5, 5)

# We print X
print()
print('X = \n', X)
print()

# We print the elements in the main diagonal of X
print('z =', np.diag(X))
print()

# We print the elements above the main diagonal of X
print('y =', np.diag(X, k=1))
print()

# We print the elements below the main diagonal of X
print('w = ', np.diag(X, k=-1))

print('w = ', np.diag(X, k=-2))


#unique elements in the array
"""We can find the unique elements in an ndarray by using the np.unique() function. 
The np.unique(ndarray) function returns the unique elements in the given ndarray,"""

# Create 3 x 3 ndarray with repeated values
X = sp.array([[1,2,3],[5,2,8],[1,2,3]])

# We print X
print()
print('X = \n', X)
print()

# We print the unique elements of X 
print('The unique elements in X are:',sp.unique(X))


sp.count_nonzero(X)


#boolean indexing

# We create a 5 x 5 ndarray that contains integers from 0 to 24
X = np.arange(25).reshape(5, 5)

# We print X
print()
print('Original X = \n', X)
print()

# We use Boolean indexing to select elements in X:
print('The elements in X that are greater than 10:', X[X > 10])
print('The elements in X that less than or equal to 7:', X[X <= 7])
print('The elements in X that are between 10 and 17:', X[(X > 10) & (X < 17)])

# We use Boolean indexing to assign the elements that are between 10 and 17 the value of -1
X[(X > 10) & (X < 17)] = -1

# We print X
print()
print('X = \n', X)
print()


#set operations
# We create a rank 1 ndarray
x = np.array([1,2,3,4,5])

# We create a rank 1 ndarray
y = np.array([6,7,2,8,4])

# We print x
print()
print('x = ', x)

# We print y
print()
print('y = ', y)

# We use set operations to compare x and y:
print()
print('The elements that are both in x and y:', np.intersect1d(x,y))
print('The elements that are in x that are not in y:', np.setdiff1d(x,y))
print('The elements that are in x that are not in y:', np.setdiff1d(y,x))
print('All the elements of x and y:',np.union1d(x,y))


#sort
"""#We can also sort ndarrays in NumPy. We will learn how to use the np.sort() function to sort rank 1 
#and rank 2 ndarrays in different ways. Like with other functions we saw before, 
#the sort function can also be used as a method. However, there is a big difference on how the data is stored in memory in this case.
# When np.sort() is used as a function, it sorts the ndrrays out of place, meaning, that it doesn't change the original ndarray being sorted. 
#However, when you use sort as a method, ndarray.
#sort() sorts the ndarray in place, meaning, that the original array will be changed to the sorted one. """


# We create an unsorted rank 1 ndarray
x = np.random.randint(1,11,size=(10,))

# We print x
print()
print('Original x = ', x)

# We sort x and print the sorted array using sort as a function.
print()
print('Sorted x (out of place):', np.sort(x))

# When we sort out of place the original array remains intact. To see this we print x again
print()
print('x after sorting:', x)

np.sort(np.unique(x))

x.sort()

#by axis
# We create an unsorted rank 2 ndarray
X = np.random.randint(1,11,size=(5,5))

# We print X
print()
print('Original X = \n', X)
print()

# We sort the columns of X and print the sorted array
print()
print('X with sorted columns :\n', np.sort(X, axis = 0))

# We sort the rows of X and print the sorted array
print()
print('X with sorted rows :\n', np.sort(X, axis = 1))


#to load the data of csv & reading we use delimiter
np.loadtxt('C:\\Users\\Dell\\Desktop\\assinment ai courses\\haberman.csv', delimiter=',')

#skip rows and how much want columns
np.loadtxt('C:\\Users\\Dell\\Desktop\\assinment ai courses\\haberman.csv', delimiter=',',skiprows=2,usecols=[1,2,0])


#creating the lower triangular matrix and upper triangular matrix
x=np.tri(5,5) #defalt lower triangular matrix ,k=0

np.tri(5,5,k=1) #k=1 #extended to upside
np.tri(5,k=-1) #decresed to downside


a=np.ones((5,5))
    
np.tril(a) #lower Triangular

np.triu(a) #upper Triangular



