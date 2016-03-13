---
layout: post
title: "Lesson - 04"
tags:
    - python
    - notebook
--- 
## Lesson 04 - NumPy and pandas

So far we have covered the base Python environment - built in data types and
structures, and looked at how we can use Python as a general purpose programming
language.

Now we will hone in on our goal of data science - using the data science modules
developed by the data science community. Download todays [notebook
here](/pythoncourse/assets/applied/Lesson - 04.ipynb).

This lesson and the next lesson are based on the book [Python for data
analysis](http://shop.oreilly.com/product/0636920023784.do), by Wes McKinney,
the primary developer of pandas. Feel free to go without the book - we will
cover much of its content in the class, and it is a little outdated.

The first module we will examine is NumPy and we will the move onto pandas.
NumPy provides arrays, while pandas provides DataFrames.

### NumPy

NumPy stands for numerical Python. So far the Python data structures have
worked, but have not been tailored for large scale data analysis.

Think of how Python works under the hood when multiplying every element of a
list by 2: 

**In [1]:**

{% highlight python %}
l = [1,2,3,4,5]
print(l*2)
#probably not what we want as statisticians!
print([i *2 for i in l])
#this works, but what about - 
l = [1,2,3,4,'a']
print([i *2 for i in l])
{% endhighlight %}

    [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]
    [2, 4, 6, 8, 10]
    [2, 4, 6, 8, 'aa']

 
Python needs to check each data type to find the times method associated with
it. In small examples like this, the overhead is very low, but when we are
dealing with millions of rows, it quickly adds up.

To work better with numeric (or other large scale data), numpy introduces the
array, a data structure which may only contain one type of data: 

**In [9]:**

{% highlight python %}
import numpy as np
l = [1,2,3,4,5]
k = np.array(l)
print(k * 2)
{% endhighlight %}

    [ 2  4  6  8 10]

 
It is also much faster (by a process called vectorisation): 

**In [3]:**

{% highlight python %}
l = range(10000)
k = np.array(l)
%timeit [i * 2 for i in l]
%timeit k * 2
{% endhighlight %}

    100 loops, best of 3: 2.1 ms per loop
    10000 loops, best of 3: 57.6 µs per loop

 
As well as the array data type, numpy contains broadcasting methods, built in
functions utilising the array structure to work extremely fast (by going through
C), linear algebra, random numbers and good integration into C and Fortran code 
 
### NumPy  basics

The array is a new class, with a lot of its own methods. The exact
implementation is outside the scope of the class, take a look at the [NumPy
website](http://www.numpy.org/) for source code and official documentation.

Technically, we use the np.array to create an instance of class ndarray. I'll
refer to them as arrays in this lesson.

We can access the type of data contained in an array: 

**In [4]:**

{% highlight python %}
a = np.array([1,2,3,4,5])
print(a.dtype)
print(np.sctypes)
{% endhighlight %}

    int64
    {'float': [<class 'numpy.float16'>, <class 'numpy.float32'>, <class 'numpy.float64'>, <class 'numpy.float128'>], 'others': [<class 'bool'>, <class 'object'>, <class 'str'>, <class 'str'>, <class 'numpy.void'>], 'complex': [<class 'numpy.complex64'>, <class 'numpy.complex128'>, <class 'numpy.complex256'>], 'int': [<class 'numpy.int8'>, <class 'numpy.int16'>, <class 'numpy.int32'>, <class 'numpy.int64'>], 'uint': [<class 'numpy.uint8'>, <class 'numpy.uint16'>, <class 'numpy.uint32'>, <class 'numpy.uint64'>]}

 
We can initialise arrays in a number of ways 

**In [8]:**

{% highlight python %}
#empty - uninitialised, random numbers!
print(np.empty((2, 3, 2)))
#all zeros
print(np.zeros((3, 6)))
#from a list
print(np.array([1,2,3,5]))
#arange is the numpy version of range - gives an array
print(np.arange(10))
{% endhighlight %}

    [[[  6.90469329e-310   2.13249800e-316]
      [  0.00000000e+000   0.00000000e+000]
      [  0.00000000e+000   8.60952352e-072]]
    
     [[  4.46535817e-090   1.39938874e-076]
      [  1.55075695e+184   1.43927482e+160]
      [  3.99910963e+252   2.32204073e-056]]]
    [[ 0.  0.  0.  0.  0.  0.]
     [ 0.  0.  0.  0.  0.  0.]
     [ 0.  0.  0.  0.  0.  0.]]
    [1 2 3 5]
    [0 1 2 3 4 5 6 7 8 9]


**In [205]:**

{% highlight python %}
#can specify data type
arr1 = np.array([1, 2, 3], dtype=np.float64)
#and cast it:
arr1.astype(np.int32)
#changing type copies, rather than modifies in place
{% endhighlight %}




    array([1, 2, 3])



**In [25]:**

{% highlight python %}
#numpy is vectorized!
print([1,2,3,4] *2)
print(np.array([1,2,3,4]) *2)
#we do this by broadcasting....
{% endhighlight %}

    [1, 2, 3, 4, 1, 2, 3, 4]
    [2 4 6 8]


**In [10]:**

{% highlight python %}
print(np.array([1,2,3,4]) * np.array([1,2,3,4]))
#times, / etc
#this is not matrix multiplication!!
print(np.array([[1,2],[3,5]]) * np.array([[1,2],[3,5]]))
{% endhighlight %}

    [ 1  4  9 16]
    [[ 1  4]
     [ 9 25]]

 
### Subsetting

We can subset much like lists, with the addition of broadcasting for assignment 

**In [217]:**

{% highlight python %}
l = list(range(10))
l[2:5] = 3
print(l)
#error!
l = np.arange(10)
l[2:5] = 3 #broadcasting
print(l)
{% endhighlight %}


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-217-3d54943f2f76> in <module>()
          1 l = list(range(10))
    ----> 2 l[2:5] = 3
          3 print(l)
          4 #error!
          5 l = np.arange(10)


    TypeError: can only assign an iterable

 
We need to be careful about assigning slices: 

**In [218]:**

{% highlight python %}
#linked! reference
l = np.arange(10)
l_slice = l[5:8]
l_slice[:] = 4
print(l)
#use .copy()
{% endhighlight %}

    [0 1 2 3 4 4 4 4 8 9]

 
We can make 2d and 3,4,5... matrices using nested lists, and subset them
appropriately: 

**In [13]:**

{% highlight python %}
#nested lists are made to be 2d
arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(arr2d.shape)
print(arr2d.ndim)
arr2d[1][1]
arr2d[1,1]

arr2d[1:,2:]
arr2d[1,2:]
#multid
arr = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]],[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
print(arr.shape)
print(arr.ndim)
#broadcasting - must match or be one in each dimension!
arr * np.array([[[1,2,3],[4,5,6],[7,8,9]]])
{% endhighlight %}

    (3, 3)
    2
    (2, 3, 3)
    3





    array([[[ 1,  4,  9],
            [16, 25, 36],
            [49, 64, 81]],
    
           [[ 1,  4,  9],
            [16, 25, 36],
            [49, 64, 81]]])



**In [30]:**

{% highlight python %}
#on multi dimensional arrays, we can use axis calls to do things by rows or columns:
arr2d.sum()
arr2d.sum(axis = 0)
arr2d.sum(axis = 1)
{% endhighlight %}




    array([ 6, 15, 24])


 
We don't need filter: We can subset with booleans much like R: 

**In [17]:**

{% highlight python %}
#no filter, just use booleans
l = np.random.randn(10)
#could have done from numpy import random
l[l > 0]
k = np.random.randn(10)
k[l > 0] #arbitrary computation!
#works on 2d
arr2d[arr2d > 3]
#or just get the booleans:
arr2d > 3
#and by assignment
arr2d[arr2d > 3] = 0
arr2d
{% endhighlight %}




    array([[1, 2, 3],
           [0, 0, 0],
           [0, 0, 0]])


 
We can subset to rearrange: 

**In [53]:**

{% highlight python %}
arr = np.empty((8, 4))
for i in range(8):
    arr[i] = i
arr
{% endhighlight %}




    array([[ 0.,  0.,  0.,  0.],
           [ 1.,  1.,  1.,  1.],
           [ 2.,  2.,  2.,  2.],
           [ 3.,  3.,  3.,  3.],
           [ 4.,  4.,  4.,  4.],
           [ 5.,  5.,  5.,  5.],
           [ 6.,  6.,  6.,  6.],
           [ 7.,  7.,  7.,  7.]])



**In [244]:**

{% highlight python %}
arr[[1,2]]
arr[[7,5]]
#negative to start at end
{% endhighlight %}




    array([[ 7.,  7.,  7.,  7.],
           [ 5.,  5.,  5.,  5.]])



**In [246]:**

{% highlight python %}
#np.where
l = np.array([[ 30,  80, 130],
       [ 80, 255, 430],
       [130, 430, 730]])
k = np.array([[ 50,  90, 100],
       [ 70, 245, 420],
       [110, 490, 0]])
j = l > k
j
{% endhighlight %}




    array([[False, False,  True],
           [ True,  True,  True],
           [ True, False,  True]], dtype=bool)



**In [248]:**

{% highlight python %}
np.where(j, l, k)
{% endhighlight %}




    array([[ 50,  90, 130],
           [ 80, 255, 430],
           [130, 490, 730]])



**In [249]:**

{% highlight python %}
#or, using broadcasting
l = np.random.randn(20).reshape(4,5)
np.where(l > 0, l, 0)
{% endhighlight %}




    array([[ 0.43077853,  0.        ,  2.18006544,  0.        ,  0.99805323],
           [ 0.62200891,  0.        ,  1.51734812,  0.        ,  0.4610735 ],
           [ 0.35047453,  0.78169552,  0.        ,  1.40064949,  0.        ],
           [ 0.62944507,  0.        ,  0.        ,  0.        ,  0.        ]])


 
### Reshape and Matrix methods

Using the built in linear algebra methods, we can carry out matrix operations
easily.

We will endeavour to cover more matrix algebra in a lesson including sympy,
sciPy and linear optimization.

Reshape allows us to reshape our matrices: 

**In [61]:**

{% highlight python %}
#reshape
np.arange(15).reshape((3, 5))
{% endhighlight %}




    array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14]])



**In [65]:**

{% highlight python %}
# .T transposes a matrix
np.arange(15).reshape((3, 5)).T
{% endhighlight %}




    array([[ 0,  5, 10],
           [ 1,  6, 11],
           [ 2,  7, 12],
           [ 3,  8, 13],
           [ 4,  9, 14]])



**In [70]:**

{% highlight python %}
#matrix multiplication
#not
np.arange(15).reshape((3, 5)) * np.arange(15).reshape((3, 5))
#this is elementwise, not useful
{% endhighlight %}




    array([[  0,   1,   4,   9,  16],
           [ 25,  36,  49,  64,  81],
           [100, 121, 144, 169, 196]])



**In [14]:**

{% highlight python %}
#use np.dot for matrix operations
np.dot(np.arange(15).reshape((3, 5)), np.arange(15).reshape((3, 5)).T)
{% endhighlight %}




    array([[ 30,  80, 130],
           [ 80, 255, 430],
           [130, 430, 730]])



**In [18]:**

{% highlight python %}
x = np.arange(1000000).reshape((100, 10000))
np.dot(x, x.T)
{% endhighlight %}




    array([[-1724114088,     9679576,  1743473240, ..., -1049853224,
              683940440, -1877233192],
           [    9679576,  1016093272,  2022506968, ..., -1152439720,
             -146026024,   860387672],
           [ 1743473240,  2022506968, -1993426600, ..., -1255026216,
             -975992488,  -696958760],
           ..., 
           [-1049853224, -1152439720, -1255026216, ...,  1884158552,
             1781572056,  1678985560],
           [  683940440,  -146026024,  -975992488, ...,  1781572056,
              951605592,   121639128],
           [-1877233192,   860387672,  -696958760, ...,  1678985560,
              121639128, -1435707304]])


 
NumPy has the expected array of matrix functions, implemented in standard C or
fortran code. See the [website
documentation](https://docs.scipy.org/doc/numpy/reference/routines.linalg.html)
for examples. Some of the functions are in the linalg submodule: 

**In [18]:**

{% highlight python %}
from numpy import linalg as LA
w, v = LA.eig(np.diag((1, 2, 3)))
print(np.diag((1,2,3)))
print(w)
print(v)
{% endhighlight %}

    [[1 0 0]
     [0 2 0]
     [0 0 3]]
    [ 1.  2.  3.]
    [[ 1.  0.  0.]
     [ 0.  1.  0.]
     [ 0.  0.  1.]]


**In [20]:**

{% highlight python %}
#solving:
#5 * x + y = 91 and x + 20 * y = 80
a = np.array([[5,1], [1,20]])
b = np.array([91,80])
LA.solve(a, b)
{% endhighlight %}




    array([ 17.57575758,   3.12121212])


 
More on linear algebra later in the course 
 
### NumPy universal functions

NumPy has a number of 'ufuncs' built in. These are fast, as they are (mostly)
implemented in C, and are a great choice for carrying out element wise
operations. For the full list, see the [official
docs](https://docs.scipy.org/doc/numpy/reference/ufuncs.html) 

**In [253]:**

{% highlight python %}
l = np.array([[ 30,  80, 130],
       [ 80, 255, 430],
       [130, 430, 730]])
np.sqrt(l)
#exp, abs etc
np.mean(l)
np.mean(l, axis = 0)
{% endhighlight %}




    array([  80.,  255.,  430.])


 
We have two main classes of ufuncs, unary, which operate on one array, and
binary which operate on two: 

**In [19]:**

{% highlight python %}
#binary ufuncs:
l = np.array([[ 30,  80, 130],
       [ 80, 255, 430],
       [130, 430, 730]])
k = np.array([[ 50,  90, 100],
       [ 70, 245, 420],
       [110, 490, 0]])
np.maximum(l,k)
{% endhighlight %}




    array([[ 50,  90, 130],
           [ 80, 255, 430],
           [130, 490, 730]])


 
We can write our own ufuncs using `frompyfunc`. The main benefit of this is to
allow broadcasting instead of having to use a loop: 

**In [27]:**

{% highlight python %}
def myfun(x):
    return(oct(x))

print(myfun(l))
{% endhighlight %}


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-27-423e7f25fa0f> in <module>()
          2     return(oct(x))
          3 
    ----> 4 print(myfun(l))
    

    <ipython-input-27-423e7f25fa0f> in myfun(x)
          1 def myfun(x):
    ----> 2     return(oct(x))
          3 
          4 print(myfun(l))


    TypeError: only integer arrays with one element can be converted to an index


**In [28]:**

{% highlight python %}
myfun2 = np.frompyfunc(myfun, 1, 1)
myfun2(l)
{% endhighlight %}




    array([['0o36', '0o120', '0o202'],
           ['0o120', '0o377', '0o656'],
           ['0o202', '0o656', '0o1332']], dtype=object)


 
### Reading in data

pandas `read_csv` function is much easier, but as a stop gap, and to keep the
numbers in NumPy, we can use NumPys built in csv reader. You can see the
[offical docs
here](https://docs.scipy.org/doc/numpy/user/basics.io.genfromtxt.html) 

**In [25]:**

{% highlight python %}
np.genfromtxt('http://www.jeremy.kiwi.nz/pythoncourse/assets/data/data.txt',
              skip_header=1,
              missing_values=('MISSING','MISSING','MISSING'),
              filling_values=(-999,-999,-999))
{% endhighlight %}




    array([[  4.83900000e-01,   4.53600000e-01,   3.56100000e-01],
           [  1.29200000e-01,   6.87500000e-01,  -9.99000000e+02],
           [  1.78100000e-01,   3.04900000e-01,   8.92800000e-01],
           [ -9.99000000e+02,   5.80100000e-01,   2.03800000e-01],
           [  5.99300000e-01,   4.35700000e-01,   7.41000000e-01]])


 
### NumPy Summary

NumPy is a large library - we haven't touched on its sorting, sets, or random
number generations capabilities. However, as pandas is based on NumPy arrays, we
will continue to cover it's functionality here.

Here is a quick overview of the example given in the install instructions: 

**In [104]:**

{% highlight python %}
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline

steps, repeats = 100, 10
stepstaken = np.random.randn(steps, repeats)
plt.plot(stepstaken.cumsum(axis = 0));
plt.xlabel('time');
plt.ylabel('position');
{% endhighlight %}

 
![png]({{ BASE_PATH }}/images/lesson---04_49_0.png) 

 
## pandas

pandas, short for Python and data analysis (or panel datasets) was created by
Wes McKinney, while he was working as a financial analyst (amongst other
projects, he is currently working for Apache to get [pandas to work with the
Apache Arrow format](http://wesmckinney.com/blog/pandas-and-apache-arrow/)). He
initially began it as a port of R into Python for speed, but quickly diverged
into a slightly different model.

It is primarily made for time series and tabular data, and its main point of use
are the new classes, DataFrame and Series, modelled on Rs dataframe. 

**In [2]:**

{% highlight python %}
from pandas import Series, DataFrame
import pandas as pd
{% endhighlight %}
 
### Series

Series are effectively NumPy arrays, but with an added index, which is retained
through operations: 

**In [107]:**

{% highlight python %}
obj = Series([3,6,9,12])
obj
{% endhighlight %}




    0     3
    1     6
    2     9
    3    12
    dtype: int64



**In [256]:**

{% highlight python %}
#series are arrays, with an index:
obj.values
obj.index
#indexes are immutable!
obj.index[1] = 5
{% endhighlight %}


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-256-1f81be84565f> in <module>()
          3 obj.index
          4 #indexes are immutable!
    ----> 5 obj.index[1] = 5
    

    C:\Anaconda3\lib\site-packages\pandas\core\index.py in __setitem__(self, key, value)
       1128 
       1129     def __setitem__(self, key, value):
    -> 1130         raise TypeError("Index does not support mutable operations")
       1131 
       1132     def __getitem__(self, key):


    TypeError: Index does not support mutable operations

 
You can think of a Series, as a fixed length, ordered dict, and we can easily
convert a dict to a Series: 

**In [111]:**

{% highlight python %}
sdata = {'Ohio': 35000, 'Texas': 71000, 'Oregon': 16000, 'Utah': 5000}
Series(sdata)
{% endhighlight %}




    Ohio      35000
    Oregon    16000
    Texas     71000
    Utah       5000
    dtype: int64



**In [259]:**

{% highlight python %}
#or:
l = Series([35000,71000,16000,5000], index=['Ohio','Texas','Oregon','Utah'])
print(l)
#indices must be unique~!
{% endhighlight %}

    Ohio      35000
    Texas     71000
    Oregon    16000
    Utah       5000
    dtype: int64


**In [118]:**

{% highlight python %}
#subsetting by index
l['Ohio']
{% endhighlight %}




    35000



**In [120]:**

{% highlight python %}
#can use all our numpy stuff! And retain our indexes
l[l.values > 30000]
{% endhighlight %}




    Ohio     35000
    Texas    71000
    dtype: int64



**In [260]:**

{% highlight python %}
#reindex with index, fills as NaN
k = Series(l, index=['Ohio','Texas','Oregon','Ontario'])
k
{% endhighlight %}




    Ohio       35000
    Texas      71000
    Oregon     16000
    Ontario      NaN
    dtype: float64



**In [132]:**

{% highlight python %}
#NaN is the NA/missing for pandas!
k.isnull()
None == None
np.nan == np.nan
{% endhighlight %}




    False



**In [261]:**

{% highlight python %}
#add - watch, NaN overwrites!
k + l
{% endhighlight %}




    Ohio        70000
    Ontario       NaN
    Oregon      32000
    Texas      142000
    Utah          NaN
    dtype: float64


 
### Summary

That's it for today.

Next lesson we will cover in more detail how to read in and clean data, as well
as merging, and the split, apply, combine methods on DataFrames.

### Exercises

1\. Create an array of size 10, with all 0s

2\. Reshape the above vector to have dims of (5,2)

3\. Create a 4\*4 identity matrix in NumPy (use google or the docs for a
function)

4\. Create a 10 by 10 matrix or random values (randn, mean = 3), and find the
minimum and maximum values. Find the index of these values

5\. Normalise the above matrix to have a mean of 0

6\. Create a Series, which has an index of NY, SF, TO, and CH, and the values
0.2, 0.9, 3.5 and 2.4

7\. Reindex this series to include VA with NA
 
