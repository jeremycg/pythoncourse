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
learn how to optimise our code through profiling, benchmarking and timing.

We will also learn the general methods of error handling and raising, and how to
automatically test code to make sure it is carrying out what we want it to do.


Firstly however, we will learn a little more about generators.

Download [todays notebook here](/pythoncourse/notebooks/r&d/Lesson 08 - Optimization.ipynb)

### Iterables and Generators

Let's go back to when we introduced zip:

**In [118]:**

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

**In [119]:**

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

**In [120]:**

{% highlight python %}
for i in l:
    print(i)
{% endhighlight %}

    1
    2
    3


But to eplicit make it an iterator, we use the iter() function:

**In [121]:**

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

    <ipython-input-121-31e9e4b4925c> in <module>()
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

**In [129]:**

{% highlight python %}
def mygen(n):
    yield n
    yield n + 1

g = mygen(10)
print(g)
print(next(g))
print(next(g))
{% endhighlight %}

    <generator object mygen at 0x7f18a3ed0830>
    10
    11


Or, a fibonacci implementation:

**In [130]:**

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

**In [131]:**

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

    <ipython-input-131-0f1ad886ef79> in <module>()
          4 print(next(g))
          5 print(next(g))
    ----> 6 print(next(g))


    StopIteration:


In general, we can think of generators as a 'lazy list' - a way of storing how
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

**In [132]:**

{% highlight python %}
g = open('/home/jeremy/Downloads/test1data.csv', 'r')
print(g)
{% endhighlight %}

    <_io.TextIOWrapper name='/home/jeremy/Downloads/test1data.csv' mode='r' encoding='UTF-8'>


We need to specify a 'mode' to open our file - I have chosen r for read, we can
also use w for writing (this deletes the exsiting file), a for appending, and r+
for writing/andor reading.

The file is not read in straight away - we merely have a pointer to the file. We
can read the next line as though it was created using a generator:

**In [133]:**

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

**In [134]:**

{% highlight python %}
g.close()
{% endhighlight %}

But, this doesn't help use too much - we can imagine reading in enough files ot
fill our memory, and then carrying out some analysis, then reading in more.

Luckily, we have the with statement and generators:

**In [135]:**

{% highlight python %}
with open('/home/jeremy/Downloads/test1data.csv', 'r') as file:
    head = [next(file).strip() for x in range(5)]

print(head)
{% endhighlight %}

    ['TripType,VisitNumber,Weekday,Upc,ScanCount,DepartmentDescription,FinelineNumber', '999,5,Friday,68113152929,-1,FINANCIAL SERVICES,1000', '30,7,Friday,60538815980,1,SHOES,8931', '30,7,Friday,7410811099,1,PERSONAL CARE,4504', '26,8,Friday,2238403510,2,PAINT AND ACCESSORIES,3565']


**In [136]:**

{% highlight python %}
def partread(file):
     with open(file) as myfile:
        for i in myfile:
            yield i

lines = 5
g = partread('/home/jeremy/Downloads/test1data.csv')
[next(g).strip() for i in range(lines)]
[next(g).strip() for i in range(lines)]
{% endhighlight %}




    ['26,8,Friday,2006613744,2,PAINT AND ACCESSORIES,1017',
     '26,8,Friday,2006618783,2,PAINT AND ACCESSORIES,1017',
     '26,8,Friday,2006613743,1,PAINT AND ACCESSORIES,1017',
     '26,8,Friday,7004802737,1,PAINT AND ACCESSORIES,2802',
     '26,8,Friday,2238495318,1,PAINT AND ACCESSORIES,4501']



Pandas also has a built-in methods to generate an interator:

**In [137]:**

{% highlight python %}
import pandas as pd
x = pd.read_csv('/home/jeremy/Downloads/test1data.csv', iterator = True)
print(x)
{% endhighlight %}

    <pandas.io.parsers.TextFileReader object at 0x7f1883699ac8>


**In [138]:**

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
      <td>999</td>
      <td>5</td>
      <td>Friday</td>
      <td>68113152929</td>
      <td>-1</td>
      <td>FINANCIAL SERVICES</td>
      <td>1000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>30</td>
      <td>7</td>
      <td>Friday</td>
      <td>60538815980</td>
      <td>1</td>
      <td>SHOES</td>
      <td>8931</td>
    </tr>
    <tr>
      <th>2</th>
      <td>30</td>
      <td>7</td>
      <td>Friday</td>
      <td>7410811099</td>
      <td>1</td>
      <td>PERSONAL CARE</td>
      <td>4504</td>
    </tr>
    <tr>
      <th>3</th>
      <td>26</td>
      <td>8</td>
      <td>Friday</td>
      <td>2238403510</td>
      <td>2</td>
      <td>PAINT AND ACCESSORIES</td>
      <td>3565</td>
    </tr>
    <tr>
      <th>4</th>
      <td>26</td>
      <td>8</td>
      <td>Friday</td>
      <td>2006613744</td>
      <td>2</td>
      <td>PAINT AND ACCESSORIES</td>
      <td>1017</td>
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

**In [139]:**

{% highlight python %}
f = open('testfile','r')
{% endhighlight %}


    ---------------------------------------------------------------------------

    FileNotFoundError                         Traceback (most recent call last)

    <ipython-input-139-9b568190695a> in <module>()
    ----> 1 f = open('testfile','r')


    FileNotFoundError: [Errno 2] No such file or directory: 'testfile'


We can try to do this, using the try statement, and an exception:

**In [140]:**

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

**In [141]:**

{% highlight python %}
try:
    f = open('testfile','r')
except FileNotFoundError:
    print('file not found')
except TypeError:
    print('type error!')
{% endhighlight %}

    file not found


**In [142]:**

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

**In [143]:**

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

**In [144]:**

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


Still to come - unit testing and parallel prcoessing

### Debugging

We have an interactive debugger in iPython, called after an error using the
%debug command. Using this we can trace back our errors, see current values, and
step forward in code:

**In [145]:**

{% highlight python %}
thisisnotadefinedvariable
{% endhighlight %}


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-145-091615ac5c44> in <module>()
    ----> 1 thisisnotadefinedvariable


    NameError: name 'thisisnotadefinedvariable' is not defined


**In [146]:**

{% highlight python %}
%debug
{% endhighlight %}

    > [1;32m<ipython-input-145-091615ac5c44>[0m(1)[0;36m<module>[1;34m()[0m
    [1;32m----> 1 [1;33m[0mthisisnotadefinedvariable[0m[1;33m[0m[0m
    [0m
    ipdb> exit


In the debugger we have a lot of commands - use h or help to get them. Useful
are c for continue, q for quit, n for next line, s for step into function, u/d
for up/down in the call stack and l for listing the code.

The [help page is available
online](https://docs.python.org/3.5/library/pdb.html).

The easiest way to debug problematic code is to manually enter a breakpoint,
using pdb.set_trace(). From here you will enter the debugger, and have access to
all the variables available in the cirrent environment.

**In [147]:**

{% highlight python %}
import pdb
{% endhighlight %}

**In [148]:**

{% highlight python %}
def fib(num):
    acounter, bcounter = 1, 1
    for i in range(num):
        yield acounter
        acounter, bcounter = bcounter, acounter + bcounter
        pdb.set_trace()

for each in fib(10):
    print(each)
{% endhighlight %}

    1
    > <ipython-input-148-113b0f967fab>(3)fib()
    -> for i in range(num):
    (Pdb) exit



    ---------------------------------------------------------------------------

    BdbQuit                                   Traceback (most recent call last)

    <ipython-input-148-113b0f967fab> in <module>()
          6         pdb.set_trace()
          7
    ----> 8 for each in fib(10):
          9     print(each)


    <ipython-input-148-113b0f967fab> in fib(num)
          1 def fib(num):
          2     acounter, bcounter = 1, 1
    ----> 3     for i in range(num):
          4         yield acounter
          5         acounter, bcounter = bcounter, acounter + bcounter


    <ipython-input-148-113b0f967fab> in fib(num)
          1 def fib(num):
          2     acounter, bcounter = 1, 1
    ----> 3     for i in range(num):
          4         yield acounter
          5         acounter, bcounter = bcounter, acounter + bcounter


    /home/jeremy/anaconda3/lib/python3.5/bdb.py in trace_dispatch(self, frame, event, arg)
         46             return # None
         47         if event == 'line':
    ---> 48             return self.dispatch_line(frame)
         49         if event == 'call':
         50             return self.dispatch_call(frame, arg)


    /home/jeremy/anaconda3/lib/python3.5/bdb.py in dispatch_line(self, frame)
         65         if self.stop_here(frame) or self.break_here(frame):
         66             self.user_line(frame)
    ---> 67             if self.quitting: raise BdbQuit
         68         return self.trace_dispatch
         69


    BdbQuit:


### Profiling and Timing

We have already briefly covered %timeit: This is magic function which runs code
and gives us the execution time. We have the very similar magic function %time:

**In [149]:**

{% highlight python %}
import numpy as np
%time np.arange(100000)
%time np.array(range(100000))
{% endhighlight %}

    CPU times: user 510 µs, sys: 0 ns, total: 510 µs
    Wall time: 581 µs
    CPU times: user 27.6 ms, sys: 20.1 ms, total: 47.7 ms
    Wall time: 51.2 ms





    array([    0,     1,     2, ..., 99997, 99998, 99999])



%time runs our command once, and reports the CPU time and wall time.

%timeit runs our command multiple times (It aims for five seconds), and reports
the average times. One caveat is timeit turns off the garbage collector - so if
we are deleting a lot of things we might be misled. We can use the [timeit
module](https://docs.python.org/3.5/library/timeit.html) for more fine grain
control if needed

**In [150]:**

{% highlight python %}
%timeit np.arange(100000)
%timeit np.array(range(100000))
{% endhighlight %}

    1000 loops, best of 3: 456 µs per loop
    10 loops, best of 3: 44.6 ms per loop


We can also use profiling tools!

First, we have the %prun magic method. This allows us to profile multiple
function calls:

**In [151]:**

{% highlight python %}
%%prun #-l fibo
#two % tell us to do a multi line magic!
def fibo(x):
    if x < 3:
        return 1
    a,b,counter = 1,2,3
    while counter < x:
        a,b,counter = b,a+b,counter+1
    return(b)

fibo(500000)
np.arange(5000000)
{% endhighlight %}



Now this is not super useful - only if we have a large file with multiple
functions. We could probably just use %time or %%timeit.

If we want to go line by line, we need the line profiler module (conda install
line profiler). We then need to load it as an iPython extension, rather than a
module:

**In [152]:**

{% highlight python %}
%load_ext line_profiler
{% endhighlight %}

    The line_profiler extension is already loaded. To reload it, use:
      %reload_ext line_profiler


**In [153]:**

{% highlight python %}
def fibo(x):
    if x < 3:
        return 1
    a,b,counter = 1,2,3
    while counter < x:
        a,b,counter = b,a+b,counter+1
    return(b)

%lprun -f fibo fibo(5000)
#we do lprun -f function statement
{% endhighlight %}

In the same manner, we can do memory profiling, using the [memory_profiler
module](https://pypi.python.org/pypi/memory_profiler)

**In [154]:**

{% highlight python %}
%load_ext memory_profiler
{% endhighlight %}

    The memory_profiler extension is already loaded. To reload it, use:
      %reload_ext memory_profiler


**In [155]:**

{% highlight python %}
def fibo(x):
    if x < 3:
        return 1
    a,b,counter = 1,2,3
    while counter < x:
        a,b,counter = b,a+b,counter+1
    return(b)

%mprun -f fibo fibo(5000)
#doesnt work unless we run a file!
{% endhighlight %}

    ERROR: Could not find file <ipython-input-155-4a4111679972>
    NOTE: %mprun can only be used on functions defined in physical files, and not in the IPython environment.



We can also use the %memit magic (Here I'm showing I was not lieing about the
range function being efficient):

**In [156]:**

{% highlight python %}
%memit range(1000000)
%memit list(range(1000000))
{% endhighlight %}

    peak memory: 182.58 MiB, increment: 0.15 MiB
    peak memory: 217.49 MiB, increment: 34.91 MiB


### Magic Commands

Just as an aside, we can see we can import magic commands from multiple
packages, and [have a lot built in](https://ipython.readthedocs.org/en/stable/in
teractive/magics.html?highlight=magics). Here are two of my favourites:

**In [157]:**

{% highlight python %}
%%bash
for i in `seq 1 10`; do
echo $i
done
{% endhighlight %}

    1
    2
    3
    4
    5
    6
    7
    8
    9
    10


**In [158]:**

{% highlight python %}
%%latex
\begin{align}
\nabla \times \vec{\mathbf{B}} -\, \frac1c\, \frac{\partial\vec{\mathbf{E}}}{\partial t} & = \frac{4\pi}{c}\vec{\mathbf{j}} \\
\end{align}
{% endhighlight %}


\begin{align}
\nabla \times \vec{\mathbf{B}} -\, \frac1c\, \frac{\partial\vec{\mathbf{E}}}{\partial t} & = \frac{4\pi}{c}\vec{\mathbf{j}} \\
\end{align}


### Testing

Test driven development is a development style where we write tests that out
completed code should pass, then attempt to write code to pass them. In this
manner, we can ensure that our code works as desired, and gives outputs that we
desire.

To a lesser extent, all code should include tests - a lot of time spent
debugging and writing code is simply manual testing - why didn't my code work?
Why did this particular data give me an error? What about special edge cases?

This informal testing is often all that code goes through. Code reviews, and
Pair Programming have been shown to help reduce bugs, but a good start is unit
testing.

Unit testing allows us to test each part (unit) of our code automatically, and
can greatly help in refactoring large code bases, or prexisting code bases. We
could theoretically completely rewrite entire scripts and keep the same tests,
so that our inputs and outputs stay identical.

There are a [wide range of testing suites available](http://docs.python-
guide.org/en/latest/writing/tests/), here we will use the [unittest
module](https://docs.python.org/3/library/unittest.html) from the standard
library.

**In [159]:**

{% highlight python %}
import unittest
{% endhighlight %}

unittest works best with scripts - Let's make one with our fibonacci functions
form the second lesson:

**In [None]:**

{% highlight python %}
def fibo(x):
    if x < 3:
        return 1
    a,b,counter = 1,2,3
    while counter < x:
        a,b,counter = b,a+b,counter+1
    return(b)
#saved as fibo.py
{% endhighlight %}

The we create a seperate script, which imports unittest, and define our tests as
a class which inherits from unittest.TestCase. All of our tests are methods of
this class, and must start with test.

We then use the range of assert\* methods built in to the class to say what our
functions should do:

**In [None]:**

{% highlight python %}
import unittest
from fibo import fibo

class testfibo(unittest.TestCase):
    def test_one(self):
        self.assertEqual(fibo(1), 1)

    def test_zero(self):
        self.assertEqual(fibo(0), 0)

    def test_negative(self):
        self.assertRaises(ValueError, fibo, -1)

    def test_ten(self):
        self.assertEqual(fibo(10), 55)

if __name__ == '__main__':
    unittest.main()
#saved as tests.py
{% endhighlight %}

`python tests.py`


```
jeremy@thin:~$ python tests.py
F..F
======================================================================
FAIL: test_negative (__main__.testfibo)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "tests.py", line 10, in test_negative
    self.assertRaises(ValueError, fibo, -1)
AssertionError: ValueError not raised by fibo

======================================================================
FAIL: test_zero (__main__.testfibo)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "tests.py", line 8, in test_zero
    self.assertEqual(fibo(0), 0)
AssertionError: 1 != 0

----------------------------------------------------------------------
Ran 4 tests in 0.002s

FAILED (failures=2)
```




Then we can fix our function:

**In [None]:**

{% highlight python %}
def fibo(x):
    if x < 1:
        if x < 0:
            raise ValueError()
        else:
            return(0)
    if x < 3:
        return 1
    a,b,counter = 1,2,3
    while counter < x:
        a,b,counter = b,a+b,counter+1
    return(b)
#saved as fibo.py
{% endhighlight %}

And rerun our tests.

We know that this is a slow function, so maybe we would like to refactor it. We
can do this, leaving the tests as is:

**In [None]:**

{% highlight python %}
def memoize(myfunction):
    cache = {}
    def function_to_cache(*args):
        if args in cache:
            return cache[args]
        else:
            cache[args] = myfunction(*args)
            return cache[args]
    return function_to_cache

def fiborecur(x):
    if x < 3:
        return 1
    return fiborecur(x - 1) + fiborecur(x - 2)

#saved as fibo.py
{% endhighlight %}

Whoops - our refactor didn't define fibo - We could do this in our script, but
maybe we don't want to for now.

We have the setUp and tearDown methods - using these we can run code to set up
our tests - eg connect to a database or download some data. In general, we
should keep any set up inside our class - we don't want to modify the global
environment for any other tests.

**In [None]:**

{% highlight python %}
import unittest
from fibo import fiborecur
from fibo import memoize

class testfibo(unittest.TestCase):
    def setUp(self):
        self.fibo = memoize(fiborecur)

    def test_one(self):
        self.assertEqual(self.fibo(1), 1)

    def test_zero(self):
        self.assertEqual(self.fibo(0), 0)

    def test_negative(self):
        self.assertRaises(ValueError, self.fibo, -1)

    def test_ten(self):
        self.assertEqual(self.fibo(10), 55)

if __name__ == '__main__':
    unittest.main()
#saved as tests.py
{% endhighlight %}

We forgot our initial bug fixes - lucky we had tests!
