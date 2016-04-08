---
layout: post
title: "Lesson 08 - Optimization"
author: Jeremy
tags:
 - R&D stream
comments: true
---
## Lesson 08 - Python Optimization

Welcome to lesson 9. This lesson we will deepen our understanding of Python, and
learn how to optimise our code through profiling, benchmarking and timing. We
will also learn the most common methods of writing code for parallel processing.

We will also learn the general methods of error handling and raising, and how to
automatically test code to make sure it is carrying out what we want it to do.


Firstly however, we will learn a little more about generators.

Download [todays notebook here](/pythoncourse/assets/notebooks/r&d/Lesson 08 - Optimization.ipynb)

### Iterables and Generators

Let's go back to when we introduced zip:

**In [49]:**

{% highlight python %}
#taken from comments on the site
l=[1,2,3]
k=[4,5,6]
a=zip(l,k)
print(list(a))
print(list(a))
{% endhighlight %}

    [(1, 4), (2, 5), (3, 6)]
    []


Huh, why did't the second one work?

It turns out, many of the maps, zips etc in python 3 are implemented as
iterators. These objects allow us to generate a single part at each step,
without storing it all in memory (these are based on range, but work a little
differently).

**In [54]:**

{% highlight python %}
a = zip(l,k)
print(next(a))
print(next(a))
print(next(a))
print(list(a))
{% endhighlight %}

    (1, 4)
    (2, 5)
    (3, 6)
    []


We can iterate over any iterable in a for loop:

**In [57]:**

{% highlight python %}
for i in l:
    print(i)
{% endhighlight %}

    1
    2
    3


But to eplicit make it an iterator, we use the iter() function:

**In [84]:**

{% highlight python %}
j = iter(l)
print(next(j))
print(next(j))
print(next(j))
print(next(j))
{% endhighlight %}

    1
    2
    3



    ---------------------------------------------------------------------------

    StopIteration                             Traceback (most recent call last)

    <ipython-input-84-31e9e4b4925c> in <module>()
          3 print(next(j))
          4 print(next(j))
    ----> 5 print(next(j))


    StopIteration:


Turning a preexisitng object into an iterator is not very useful however, as we
already have it in memory.

If we want to create a function to make our output, we can use a generator
function

Generator functuions work very similar to standard functions, but use the yield
keyword, rather than return:

**In [67]:**

{% highlight python %}
def mygen(n):
    yield n
    yield n + 1

g = mygen(10)
print(g)
print(next(g))
print(next(g))
{% endhighlight %}

    <generator object mygen at 0x00000241871AE518>
    10
    11


Or, a fibonacci implementation:

**In [80]:**

{% highlight python %}
def fib(n):
    a, b = 1, 1
    for i in range(n):
        yield a
        a, b = b, a + b

for num in fib(10):
    print(num)
{% endhighlight %}

    1
    1
    2
    3
    5
    8
    13
    21
    34
    55


This is also why we can't do tuple comprehensions - the syntax is reserved for
making generator expressions:

**In [81]:**

{% highlight python %}
l=[1,2,3]
g = (i for i in l)
print(next(g))
print(next(g))
print(next(g))
print(next(g))
{% endhighlight %}

    1
    2
    3



    ---------------------------------------------------------------------------

    StopIteration                             Traceback (most recent call last)

    <ipython-input-81-0f1ad886ef79> in <module>()
          4 print(next(g))
          5 print(next(g))
    ----> 6 print(next(g))


    StopIteration:


In general, we can think of generators as a 'lazy list' - a way of stroign how
to get the next object, without taking up all the memory.

### Working with large files

In general Python holds the data we have in memory. We need to come up with ways
to handle larger data out of memory in piecemeal (or buy more RAM). Most methods
are specific to a certain type of data, but we will cover a general method for
now.

We can open a file on the disk in Python, as long as we use the correct
permissions (read_csv from pandas took care of this for us). Let's download the
test example data -
http://jeremy.kiwi.nz/pythoncourse/assets/tests/r&d/test1data.csv

**In [90]:**

{% highlight python %}
g = open('c:/users/jeremy/downloads/test1data.csv', 'r')
print(g)
{% endhighlight %}

    <_io.TextIOWrapper name='c:/users/jeremy/downloads/test1data.csv' mode='r' encoding='cp1252'>


We need to specify a 'mode' to open our file - I have chosen r for read, we can
also use w for writing (this deletes the exsiting file), a for appending, and r+
for writing/andor reading.

The file is not read in straight away - we merely have a pointer to the file. We
can read the next line as though it was created using a generator:

**In [91]:**

{% highlight python %}
#nextline
print(g.readline())
print(g.readline())
{% endhighlight %}

    TripType,VisitNumber,Weekday,Upc,ScanCount,DepartmentDescription,FinelineNumber

    999,5,Friday,68113152929,-1,FINANCIAL SERVICES,1000



If you want to read all the lines of a file in a list you can also use list(f),
f.readlines() or f.read().

Once we are done with a file, we need to close it:

**In [92]:**

{% highlight python %}
g.close()
{% endhighlight %}

But, this doesn't help use too much - we can imagine reading in enough files ot
fill our memory, and then carrying out some analysis, then reading in more.

Luckily, we have the with statement and generators:

**In [7]:**

{% highlight python %}
with open('c:/users/jeremy/downloads/test1data.csv', 'r') as file:
    head = [next(file).strip() for x in range(5)]

print(head)
{% endhighlight %}

    ['TripType,VisitNumber,Weekday,Upc,ScanCount,DepartmentDescription,FinelineNumber', '999,5,Friday,68113152929,-1,FINANCIAL SERVICES,1000', '30,7,Friday,60538815980,1,SHOES,8931', '30,7,Friday,7410811099,1,PERSONAL CARE,4504', '26,8,Friday,2238403510,2,PAINT AND ACCESSORIES,3565']


**In [14]:**

{% highlight python %}
def partread(file):
     with open(file) as myfile:
        for i in myfile:
            yield i

lines = 5
g = partread('c:/users/jeremy/downloads/test1data.csv')
[next(g).strip() for i in range(lines)]
[next(g).strip() for i in range(lines)]
{% endhighlight %}




    ['26,8,Friday,2006613744,2,PAINT AND ACCESSORIES,1017',
     '26,8,Friday,2006618783,2,PAINT AND ACCESSORIES,1017',
     '26,8,Friday,2006613743,1,PAINT AND ACCESSORIES,1017',
     '26,8,Friday,7004802737,1,PAINT AND ACCESSORIES,2802',
     '26,8,Friday,2238495318,1,PAINT AND ACCESSORIES,4501']



Pandas also has a built-in methods to generate an interator:

**In [17]:**

{% highlight python %}
import pandas as pd
x = pd.read_csv('c:/users/jeremy/downloads/test1data.csv', iterator = True)
print(x)
{% endhighlight %}

    <pandas.io.parsers.TextFileReader object at 0x000002A6A7AB2198>


**In [20]:**

{% highlight python %}
x.get_chunk(5)
{% endhighlight %}




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>TripType</th>
      <th>VisitNumber</th>
      <th>Weekday</th>
      <th>Upc</th>
      <th>ScanCount</th>
      <th>DepartmentDescription</th>
      <th>FinelineNumber</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>26</td>
      <td>8</td>
      <td>Friday</td>
      <td>5200010239</td>
      <td>1</td>
      <td>DSD GROCERY</td>
      <td>4606</td>
    </tr>
    <tr>
      <th>1</th>
      <td>26</td>
      <td>8</td>
      <td>Friday</td>
      <td>88679300501</td>
      <td>2</td>
      <td>PAINT AND ACCESSORIES</td>
      <td>3504</td>
    </tr>
    <tr>
      <th>2</th>
      <td>26</td>
      <td>8</td>
      <td>Friday</td>
      <td>22006000000</td>
      <td>1</td>
      <td>MEAT - FRESH &amp; FROZEN</td>
      <td>6009</td>
    </tr>
    <tr>
      <th>3</th>
      <td>26</td>
      <td>8</td>
      <td>Friday</td>
      <td>2236760452</td>
      <td>1</td>
      <td>PAINT AND ACCESSORIES</td>
      <td>7</td>
    </tr>
    <tr>
      <th>4</th>
      <td>26</td>
      <td>8</td>
      <td>Friday</td>
      <td>88679300501</td>
      <td>-1</td>
      <td>PAINT AND ACCESSORIES</td>
      <td>3504</td>
    </tr>
  </tbody>
</table>
</div>



There are more sensible workflows using large data technologies - for now we
will move on.

### Error and Exception handling

It's easier to ask forgiveness than it is to get permission. - Grace Hopper

We can often program more easily, if we simply try to do something, and then
handle the failure. Errors will however break our code if we are not careful, so
we can build in fail safe methods to handle errors:

**In [22]:**

{% highlight python %}
f = open('testfile','r')
{% endhighlight %}


    ---------------------------------------------------------------------------

    FileNotFoundError                         Traceback (most recent call last)

    <ipython-input-22-9b568190695a> in <module>()
    ----> 1 f = open('testfile','r')


    FileNotFoundError: [Errno 2] No such file or directory: 'testfile'


We can try to do this, using the try statement, and an exception:

**In [26]:**

{% highlight python %}
try:
    f = open('testfile','r')
except:
    print('file not found')
{% endhighlight %}

    file not found


Now we have no longer raised an error serious enough to stop our script (whether
this is bad or good is up to you). We can also specify the [type of
error](https://docs.python.org/3/library/exceptions.html#bltin-exceptions) we
will catch (more specific is better):

**In [29]:**

{% highlight python %}
try:
    f = open('testfile','r')
except FileNotFoundError:
    print('file not found')
except TypeError:
    print('type error!')
{% endhighlight %}

    file not found


**In [31]:**

{% highlight python %}
try:
    s = (1,2,3,4)
    s[3] = 4
except FileNotFoundError:
    print('file not found')
except TypeError:
    print('type error!')
{% endhighlight %}

    type error!


We can add on a final else which is only completed if we did not raise an error:

**In [34]:**

{% highlight python %}
try:
    s = [1,2,3,4]
    s[3] = 4
except FileNotFoundError:
    print('file not found')
except TypeError:
    print('type error!')
else:
    print('operation sucessful')
{% endhighlight %}

    operation sucessful


We can use finally to run a piece of code whether or not we were sucessful,
which is useful for cleanup:

**In [37]:**

{% highlight python %}
try:
    s = (1,2,3,4)
    s[3] = 4
except FileNotFoundError:
    print('file not found')
except TypeError:
    print('type error!')
else:
    print('operation sucessful')
finally:
    del(s)
    print('cleanedup')
{% endhighlight %}

    type error!
    cleanedup


Still to come - unit testing,  profiling, benchmarking and parallel prcoessing

**In [39]:**

{% highlight python %}
%prun pass
{% endhighlight %}
