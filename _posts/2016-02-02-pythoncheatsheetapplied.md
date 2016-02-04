---
layout: post
title: "01 Python Cheat Sheet - Python @ Precima – Applied Statistics Stream"
author: Jeremy
tags:
comments: true
---

This is a quick walk-through of basic Python syntax - [download the Jupyter notebook here](/pythoncourse/assets/cheatsheet/cheatsheet.ipynb)

### Basic Math

Python can carry out basic mathematical operations:

**In [1]:**

{% highlight python %}
1 * 5
{% endhighlight %}




    5



**In [2]:**

{% highlight python %}
2 / 5
{% endhighlight %}




    0.4



**In [3]:**

{% highlight python %}
2 ** 5
{% endhighlight %}




    32



**In [4]:**

{% highlight python %}
5 % 3
{% endhighlight %}




    2



### Basic Data Types

Python has a number of basic data types and containers, here is a list of those
we will use frequently.


We are assigning them to a variable, using the =.


In code blocks, a # indicates a comment, and is not read by the interpreter.

**In [5]:**

{% highlight python %}
#characters, use " or ' to contain them
string = "string"

#whole numbers
integer = 5

#real numbers
floats = 12.23

#True, False or None
boolean = True
{% endhighlight %}

Now the common data 'containers'

**In [6]:**

{% highlight python %}
#similar to an array, any mix of data types
#lists are the most common base python container
lists = ['anymixoftypes', False, 5, 12.3]

#tuples are similar to a list, but immutable - you cannot change their contents
tuples = ("immutable", "lists")

#Dictionaries are similar to a hash table, key:value pairs. Unordered
dictionary = {'key1' : 'value', 'key2' : 34}

#Sets only keep unique elements. Not able to index
sets = set([1,1,2,3,4,5])
{% endhighlight %}

### Subsetting

We can get objects out of our data structures by subsetting.

Each data type has slightly different ways of subsetting

**In [7]:**

{% highlight python %}
a = [1,2,3,4,5]
#python is 0 indexed = 0 is the first element
a[0]
{% endhighlight %}




    1



**In [8]:**

{% highlight python %}
#negative numbers start at the end
a[-1]
{% endhighlight %}




    5



**In [9]:**

{% highlight python %}
#use : to get everything
a[2:]
{% endhighlight %}




    [3, 4, 5]



**In [10]:**

{% highlight python %}
#nested lists can be subset
a = [1,2,3,[4,5,6],7]
a[3][1]
{% endhighlight %}




    5



**In [11]:**

{% highlight python %}
#dictionaries are accessed by their keys
a = {'key1':12, 'key2':[1,2,3]}
a['key2']
{% endhighlight %}




    [1, 2, 3]



### Importing Modules

External libraries, or modules, allow Python to be extended. We can access these
libraries by importing them.

Anaconda comes with a lot of included modules, more can be installed using pip
or conda at command line

**In [12]:**

{% highlight python %}
#Modules are imported using 'import'
import numpy as np
import pandas as pd
#now if we want to use a numpy function, we use np.function

#we can import certain functions, so we don't have to use the module name
from pandas import Series, DataFrame
#now Series and Dataframe will work without the pd. first
{% endhighlight %}

### Advanced Data Types

Numpy arrays, and Pandas Series and DataFrames are optimised for large data and
data science applications. They can only contain one type of data (or one type
per column in DataFrames) allowing code to be executed much faster, as well as
vectorized operations. More on subsetting of these data types in the full course


**In [13]:**

{% highlight python %}
#an array allows vectorized operations
array = np.array([1,2,3,4,5])
#can only contain one type of data
array - 1
{% endhighlight %}




    array([0, 1, 2, 3, 4])



**In [14]:**

{% highlight python %}
#arrays can be multidimensional
array2d = np.array([[1,2,3,4,5],
                    [6,7,8,9,10]])
#still only one type
array2d - 1
{% endhighlight %}




    array([[0, 1, 2, 3, 4],
           [5, 6, 7, 8, 9]])



**In [15]:**

{% highlight python %}
#Series allows 'indexed' data in arrays
series = Series([1,2,3,4,5], index = ['a', 'b', 'c', 'd', 'e'])
series
#still only one data type (index does not count)
{% endhighlight %}




    a    1
    b    2
    c    3
    d    4
    e    5
    dtype: int64



**In [16]:**

{% highlight python %}
#Dataframes are 2d collections of data - like a spreadsheet or SQL table
dataframe = DataFrame(np.array([1,2,3,4]).reshape(2, 2),
                      columns = list('ab'),
                      index = ['X','Y'])
dataframe
#has column names and indexes
{% endhighlight %}




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>X</th>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Y</th>
      <td>3</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>



### Control Flow

Python has the usual for, while and if else statements.

Python does not use braces like most other languages, it's indentation is
controlled by whitespace and :.

This enforces at least some degree of readability on code.

**In [17]:**

{% highlight python %}
#for loops
for i in [1,2,3,4]:
    print(i ** 2)
{% endhighlight %}

    1
    4
    9
    16


**In [18]:**

{% highlight python %}
#while loops
x = 0
while x < 5:
    print('%s is less than 5' %(x))
    x += 1
{% endhighlight %}

    0 is less than 5
    1 is less than 5
    2 is less than 5
    3 is less than 5
    4 is less than 5


**In [19]:**

{% highlight python %}
#if else
a = [1, 2, 3, 4]
for number in a:
    if number == 1:
        print(number)
    elif number == 2:
        print("two")
    else:
        print("I'm not sure")
{% endhighlight %}

    1
    two
    I'm not sure
    I'm not sure


List (and dict) comprehensions are a shortcut (or syntactic sugar) for 'for'
loops. They allow sucinct, 'pythonic' code:

**In [20]:**

{% highlight python %}
#shortcut for a for loop returning a list
a = [1,2,3,4,5]
[i * i for i in a]
{% endhighlight %}




    [1, 4, 9, 16, 25]



### Functions

Functions are reusbale pieces of code, either built in or custom written. They
take arguments inside brackets and `return` a result

**In [21]:**

{% highlight python %}
#to call a function, use its name and arguments
sum([1,2])
#use help(function) to get help on a function
#help(sum)
{% endhighlight %}




    3



**In [22]:**

{% highlight python %}
#to make your own
def mysum(a, b):
    '''
    This is the help (docstring), to see this type help(mysum)
    '''
    return(a + b)
{% endhighlight %}

**In [23]:**

{% highlight python %}
help(mysum)
{% endhighlight %}

    Help on function mysum in module __main__:

    mysum(a, b)
        This is the help (docstring), to see this type help(mysum)



**In [24]:**

{% highlight python %}
mysum(1,2)
{% endhighlight %}




    3



### Methods

Methods are functions, which are particular to a certain data type. For example,
a list has different methods than a set.
They are called using a .method() after your variable name

**In [25]:**

{% highlight python %}
set([1,2,3]).pop()
{% endhighlight %}




    1



**In [26]:**

{% highlight python %}
#Get help using help(object.method), note no ()
help(set([1,2,3]).pop)
{% endhighlight %}

    Help on built-in function pop:

    pop(...) method of builtins.set instance
        Remove and return an arbitrary set element.
        Raises KeyError if the set is empty.



**In [27]:**

{% highlight python %}
#methods will often modify in place!
a = [5, 4, 3, 2, 1]
a.sort()
a
{% endhighlight %}




    [1, 2, 3, 4, 5]



**In [28]:**

{% highlight python %}
#arguments go in the braces
a.sort(reverse = True)
a
{% endhighlight %}




    [5, 4, 3, 2, 1]



### Objects

An object is a data type, either built in or one which we can custom define. All
items in Python are objects.

**In [29]:**

{% highlight python %}
#define objects using class
#with a way to init a new one
#and any other methods
class myclass:
    '''
    help goes here!
    '''
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def mymethod(self, text):
        print(text)
{% endhighlight %}

**In [30]:**

{% highlight python %}
#make an object
a = myclass(1,2)
a.x, a.y
{% endhighlight %}




    (1, 2)



**In [31]:**

{% highlight python %}
# call a method
a.mymethod("mytext")
{% endhighlight %}

    mytext
