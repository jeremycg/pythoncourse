---
layout: post
title: "Lesson 00 Basic Data Types"
author: Jeremy
tags:
 - Applied Statistics stream
comments: true
---
## Welcome to the Python course @ Precima

Hopefully everyone has read the syllabus, cheat sheets and course outline on the
website.

They are available at www.jeremy.kiwi.nz/pythoncourse.

All lessons are made as iPython notebooks and will be available on this site to
download. Grab todays lesson from [this
link!](/pythoncourse/assets/notebooks/applied/lesson 00.ipynb).

Today we will briefly go over course logistics and the Anaconda environment and
then dive into Python data types.

### Python

Python is a programming language first released in 1991 by Guido van Rossum and
one of the most popular languages used in computing today. Python currently
ranks as the [5th most common language used on
github](https://github.com/blog/2047-language-trends-on-github). Guido is
officially the Benevolent- Dictator For Life (BDFL) of Python, and currently
works at Dropbox, after being on staff at Google from 2005-2012. Python is open
source and free - you can download and read the [source
online](https://www.python.org/downloads/release/python-351/).

Python is used as a scripting language, as well as in web development and to
create applications - some of the more popular websites and applications running
at least partially on Python include: Google, Youtube, Facebook, Instagram,
Reddit, Dropbox, Civilization IV, EVE Online and BitTorrent.

Python as a language is based on readability, flexibility, simplicity and
extensibility.

To see the Philosophy of Python, enter the below command into your interpreter,
without the #:

**In [214]:**

{% highlight python %}
#import this
{% endhighlight %}

The extensibility of Python has caused it to be adopted, along with R, as one of
the premier data science programming languages.

Python has had many added on modules (or libraries) added to it, to allow data
science work which we will cover in this course.

In general compared to R, Python is faster, more programmer focussed and less
restrictive in licensing. The downside is new statistical methods tend to appear
in R before Python, although as the community grows, this has become less of a
problem.

### 2.7 vs 3

In 2008, Python version 3.0 was released. Due to the number and nature of
changes, Python 2.6 and Python 3.0 were not compatible. This has lead to a split
in the Python community, as many users were unwilling to fix existing code to 3
compatibility, and have since continued to develop Python 2.6 into 2.7, while
3.0 has been developed in parallel and currently stands at version 3.5.

In this course, we will use Python 3. This is due to the majority of users
having no legacy code to worry about, the better memory management, and the
availability of the data science stack in Python 3. If you are a die hard Python
2.7 user, feel free to continue using it, although you will need to fix the code
yourself.

In the level we will be coding at the changes are not too big, the largest
differences we will see are `print()` vs `print`, `xrange` vs `range` and other
generators. Python code found online will often be 2.7, but should be readable
by a 3 trained user.

### Anaconda, iPython and Spyder

We will use Anaconda, a distribution of Python by Continuum Analytics, put
together for use in data science.

Anaconda comes with most of the modules we need for data analysis, as well as
Jupyter notebooks, and the IDE we will use for the first couple of lessons,
Spyder.

We have installed the launcher, which allows updating and launching these apps,
the Spyder IDE which allows coding and running scripts in an integrated
environment, the iPython-QT console, which is an advanced console allowing
inline graphing, and the iPython notebook, which allows development in
interactive notebooks in the browser. These programs all run Python code - the
difference is in how you interact with the environment.

iPython notebooks are currently in the process of being rebranded into Jupyter
notebooks - the launcher and documentation will refer to them interchangeably.

Coding along with the lesson is encouraged!


### Course Logistics

This stream of the course will cover applied data science in Python. We will
start by covering the basic data types and structures, move on to loops and
statements, then functions and classes. In lessons 4 and 5 we will introduce the
standard data  modules, Pandas and NumPy, then move on into graphing,
reproducible research and machine learning in the later stages of the course.
Please let me know if you have any particular applications or problems, and we
will try and address the common ones in class.

We are tracking progress, to make sure this course is giving value for your time
- Please make sure you answer the quizzes and assessments we will send out later
in the course.

### Outside Resources

A number of people have asked about text books or online resources for the
course. Most Python tutorials online will assume a knowledge at around the level
we will reach at the end of lesson 3. Until then, follow along on the website.
From there I will recommend a text book and online resources.



## Basic Data Types

### Numbers

Python acts as a calculator:

**In [1]:**

{% highlight python %}
3 + 2
{% endhighlight %}




    5



**In [2]:**

{% highlight python %}
5 - 2
{% endhighlight %}




    3



**In [3]:**

{% highlight python %}
3 * 2
{% endhighlight %}




    6



**In [4]:**

{% highlight python %}
3//2
{% endhighlight %}




    1



**In [5]:**

{% highlight python %}
3 * 3
{% endhighlight %}




    9



**In [6]:**

{% highlight python %}
15 % 4
{% endhighlight %}




    3



**In [7]:**

{% highlight python %}
4 ** 3
{% endhighlight %}




    64



Under the hood, Python has two (actually more) ways of representing numbers.

We can see which one we are using `type()`. Python 2 users will know of floating
division errors:

**In [8]:**

{% highlight python %}
3 / 2
{% endhighlight %}




    1.5



**In [9]:**

{% highlight python %}
type(3 / 2)
{% endhighlight %}




    float



**In [10]:**

{% highlight python %}
type(2)
{% endhighlight %}




    int



**In [11]:**

{% highlight python %}
type(2.0)
{% endhighlight %}




    float



We can convert using `int`, or `float` - Python is not statically typed

**In [12]:**

{% highlight python %}
type(int(2.0))
{% endhighlight %}




    int



**In [13]:**

{% highlight python %}
type(float(2))
{% endhighlight %}




    float



We can assign values to variables:

**In [14]:**

{% highlight python %}
a = 23
b = 2 ** 4
c, d = 2 - 4, 3 / 2
print(a, b, c, d)
{% endhighlight %}

    23 16 -2 1.5


Think of the value as being stored somewhere in the computers memory, and the
variable being a sticky note referring to that point in memory.

There are some rules about variable names - they cannot start with numbers, must
be alphanumeric with no spaces, and should be lowercase with underscores and
somewhat informative about what they represent:

**In [15]:**

{% highlight python %}
1a = 5
{% endhighlight %}


      File "<ipython-input-15-bad23dd73176>", line 1
        1a = 5
         ^
    SyntaxError: invalid syntax



**In [16]:**

{% highlight python %}
a1 = 5
{% endhighlight %}

**In [17]:**

{% highlight python %}
b^ = 5
{% endhighlight %}


      File "<ipython-input-17-8dd4b022c92a>", line 1
        b^ = 5
           ^
    SyntaxError: invalid syntax



**In [18]:**

{% highlight python %}
b g = 5
{% endhighlight %}


      File "<ipython-input-18-034da690133e>", line 1
        b g = 5
          ^
    SyntaxError: invalid syntax



**In [19]:**

{% highlight python %}
this_is_good_practice = 5
#snake_case. See also camelCase, kebab-case
#Python recommends UpperCamelCase for class names,
#CAPITALIZED_WITH_UNDERSCORES for constants, and lowercase_separated_by_underscores for other names.
{% endhighlight %}

Check out the [official Python style guide](http://docs.python-
guide.org/en/latest/writing/style/), or [Googles Python style
guide](https://google.github.io/styleguide/pyguide.html) for more tips regarding
variable names and general coding style.

In iPython based interpreters, we can use `whos` to see all our declared
variables (or use the spyder IDE)

**In [20]:**

{% highlight python %}
whos
{% endhighlight %}

    Variable                Type     Data/Info
    ------------------------------------------
    a                       int      23
    a1                      int      5
    b                       int      16
    c                       int      -2
    d                       float    1.5
    this_is_good_practice   int      5


### Built in functions

A number of built in functions exist for use on numbers. A function is called
using the function name, followed by arguments or args in the brackets.

To get help on how to use a built in function, use the help:

**In [21]:**

{% highlight python %}
#help(abs)
# # means a comment - the python interpreter will ignore the rest of the line
{% endhighlight %}

**In [22]:**

{% highlight python %}
#absolute value
abs(-23)
{% endhighlight %}




    23



**In [23]:**

{% highlight python %}
#hexidecimal representation
hex(3735928559)
{% endhighlight %}




    '0xdeadbeef'



**In [24]:**

{% highlight python %}
# binary
bin(1048576)
{% endhighlight %}




    '0b100000000000000000000'



more on functions later, but remember we can get help using `help(abs)`

### Warning - Floating Point Errors

We would expect the below to be equal to -0.2 - what is going on?

**In [25]:**

{% highlight python %}
-0.1 + 0.2 - 0.3
{% endhighlight %}




    -0.19999999999999998



This is a floating point error. It occurs as computers work in binary and 1/10
is a repeating fraction in binary, much like 1/3 is in base 10. We will discuss
mitigation techniques later in the course - for now be aware when working with
floats.

### Booleans

We have the standard boolean operators:

**In [26]:**

{% highlight python %}
1 > 2
{% endhighlight %}




    False



**In [27]:**

{% highlight python %}
1 < 2
{% endhighlight %}




    True



**In [28]:**

{% highlight python %}
1 == 1
{% endhighlight %}




    True



**In [29]:**

{% highlight python %}
1 != 2
{% endhighlight %}




    True



Which can then be chained using `and` `or`

**In [30]:**

{% highlight python %}
1 == 1 and 2 >= 3
{% endhighlight %}




    False



**In [31]:**

{% highlight python %}
1 <= 1 or 2 > 3
{% endhighlight %}




    True



In addition to True and False, we also have None, which we will discuss later in
the course

### Warning - Bitwise Operations

You probably don't want to use `|`, `&` or `^` which many other languages do for
`or` `and` and `not`- they are for [bitwise
comparison](https://wiki.python.org/moin/BitwiseOperators).

**In [32]:**

{% highlight python %}
12 | 9
{% endhighlight %}




    13



**In [33]:**

{% highlight python %}
print(bin(12))
print(bin(9))
print(bin(13))
{% endhighlight %}

    0b1100
    0b1001
    0b1101


### Strings

Strings are character type data:

**In [34]:**

{% highlight python %}
a = "abcde"
b = 'abcde'
d = "they're"
e = 'He said "Hello"'
f = 'using "both" types isn\'t hard'
print(type(a))
{% endhighlight %}

    <class 'str'>


We can do (some) math on strings

**In [35]:**

{% highlight python %}
print(a * 2)
print(a + d)
print(a * a)
{% endhighlight %}

    abcdeabcde
    abcdethey're



    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-35-badeb938c1f1> in <module>()
          1 print(a * 2)
          2 print(a + d)
    ----> 3 print(a * a)


    TypeError: can't multiply sequence by non-int of type 'str'


Using a \ we can escape or use special characters - we mostly will use \n and \t
:

**In [36]:**

{% highlight python %}
a = 'atgc\nttt'
print(a)
b = 'atgc\tgcggt'
print(b)
{% endhighlight %}

    atgc
    ttt
    atgc	gcggt


or using triple quotes we can enter multiple lines

**In [37]:**

{% highlight python %}
a ='''
this is
a long
string'''
print(a)
{% endhighlight %}


    this is
    a long
    string


strings have functions too but the numeric ones won't work:

**In [38]:**

{% highlight python %}
len('abcde')
{% endhighlight %}




    5



**In [39]:**

{% highlight python %}
abs('abcde')
{% endhighlight %}


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-39-75279fad068d> in <module>()
    ----> 1 abs('abcde')


    TypeError: bad operand type for abs(): 'str'


## Subsetting

When we want to get a subset of a string (or many other Python objects) we use
subsetting.

Python is a 0-indexed language, meaning we start counting at 0. For a
digression, see here for a [possible origin of 0
indexing](http://exple.tive.org/blarg/2013/10/22/citation-needed/).

This leads to the famous quote "The two big problems in computer science are
cache invalidation, naming things and off by one errors".

**In [40]:**

{% highlight python %}
my_string = 'Hello world'
#a single element
print(my_string[0])
#a slice
print(my_string[0:3])
#from the start
print(my_string[:3])
#to the end
print(my_string[3:])
#negative indices count from the end
print(my_string[-3:])
{% endhighlight %}

    H
    Hel
    Hel
    lo world
    rld


Indexing can also use a third place ie `x[from:to:by]`, but this is not used
often in practice.

**In [41]:**

{% highlight python %}
print(my_string[::2])
print(my_string[::-1])
print(my_string[0:6:2])
{% endhighlight %}

    Hlowrd
    dlrow olleH
    Hlo


Strings are immutable - we cannot easily change elements of them in place:

**In [42]:**

{% highlight python %}
my_string[0] = 'h'
{% endhighlight %}


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-42-8e10d58363b9> in <module>()
    ----> 1 my_string[0] = 'h'


    TypeError: 'str' object does not support item assignment


### Methods

Methods are functions which are specific to a certain type of object. They are
called using x.method()

We can see a list in Spyder by using tab

**In [43]:**

{% highlight python %}
my_string = 'abcde'
print(my_string.upper())
print(my_string.upper().lower())
print(my_string.capitalize())
print(my_string.count('a'))
print(my_string.index('c'))
print(my_string.split('d'))
{% endhighlight %}

    ABCDE
    abcde
    Abcde
    1
    2
    ['abc', 'e']


### Print Formatting

We can format our print output using our defined variables:

**In [44]:**

{% highlight python %}
x = 13.13
print('blah blah %s' %(x))
print('Floating point numbers: %1.2f' %(13.144))
print('First: %s, Second: %1.2f, Third: %r' %('hi!', 3.14, 22))
print('First: {y}, Second: {x}, Third: {z}'.format(y = "hi", z = 12.1, x = 12))
{% endhighlight %}

    blah blah 13.13
    Floating point numbers: 13.14
    First: hi!, Second: 3.14, Third: 22
    First: hi, Second: 12, Third: 12.1


## Break

As some motivation, [here is an iPython
notebook](http://nbviewer.jupyter.org/github/donnemartin/data-science-ipython-
notebooks/blob/master/kaggle/titanic.ipynb) which is at the Python level you
will be by the end of the course.

In this notebook, the author takes the publically available Titanic dataset,
does some exploratory analysis and trains a machine learning model to predict
survival of the Titanic sinking based on ticket class, family size, fare paid,
age and sex.

Don't worry too much about specific details here - we will cover the methods and
data structures used as we continue on in the course. This is a great example of
the exploratory analysis, graphing and machine learning powers of Python, and an
example of the notebooks you will be able to produce on your own data.

## Basic Data Containers


### Lists

Lists are the workhorse of Python - they are a way of containing multiple pieces
of data in a single place.

Lists may be recognised as similar to arrays in other languages - we don't need
to preassign type, size or any other attributes


**In [45]:**

{% highlight python %}
my_list = [1, 2, 3, 4]
#subsetting similar to strings
print(my_list[0:2])
#add similarly to strings
print(my_list * 2)
print(my_list + [1, 2, 3])
print(my_list + [[1, 2, 3, 4]])
{% endhighlight %}

    [1, 2]
    [1, 2, 3, 4, 1, 2, 3, 4]
    [1, 2, 3, 4, 1, 2, 3]
    [1, 2, 3, 4, [1, 2, 3, 4]]


Lists can contain multiple data types, including other lists:

**In [46]:**

{% highlight python %}
my_list = [True, "hi", 1, 3.4, [1, 2, 3]]
print(type(my_list))
#nested subset
print([my_list[1][1], my_list[-1][-1]])
{% endhighlight %}

    <class 'list'>
    ['i', 3]


Lists are mutable

**In [47]:**

{% highlight python %}
my_list[1] = "bye"
print(my_list)
{% endhighlight %}

    [True, 'bye', 1, 3.4, [1, 2, 3]]


### List Methods

Lists have their own methods. Many of these will modify a list in place, so
beware

**In [48]:**

{% highlight python %}
my_list = [True, "hi", 1, 3.4, [1, 2, 3]]
my_list.append('a new list item')
my_list
{% endhighlight %}




    [True, 'hi', 1, 3.4, [1, 2, 3], 'a new list item']



**In [49]:**

{% highlight python %}
my_list.extend([1,2,3])
my_list
{% endhighlight %}




    [True, 'hi', 1, 3.4, [1, 2, 3], 'a new list item', 1, 2, 3]



**In [50]:**

{% highlight python %}
my_list = [3,4,1,2]
my_list.sort()
print(my_list)
my_list.sort(reverse=True)
print(my_list)
{% endhighlight %}

    [1, 2, 3, 4]
    [4, 3, 2, 1]


**In [51]:**

{% highlight python %}
print(my_list.pop())
my_list
{% endhighlight %}

    1





    [4, 3, 2]



### Use Case

Python lists are the basic data carrier. You can use them to contain the most
basic of data, to multiply nested lists

**In [52]:**

{% highlight python %}
my_transactions = [["oranges",3],["apples", 2],["grapefruit", 3]]
print(my_transactions[0][1])

my_dna = ["a", "t", "g", "c", ["a", "t", "c", "g", "g"], "t", "a", ["a", "t", "a", "a", "a"]]
print(my_dna[4].count("g"))
{% endhighlight %}

    3
    2


### Warning

Python lists are not copied on reassignment. This will cause problems!

**In [53]:**

{% highlight python %}
my_list = [1, 2, 3, 4]
my_list2 = my_list
my_list2[0] = 5
my_list
{% endhighlight %}




    [5, 2, 3, 4]



We can fix it using slicing, copying or the list() function:

**In [54]:**

{% highlight python %}
my_list = [1, 2, 3, 4]
my_list2 = my_list[:]
my_list2[1] = 5
my_list
{% endhighlight %}




    [1, 2, 3, 4]



**In [55]:**

{% highlight python %}
my_list = [1, 2, 3, 4]
my_list2 = list(my_list)
my_list2[1] = 5
my_list
{% endhighlight %}




    [1, 2, 3, 4]



**In [56]:**

{% highlight python %}
my_list = [1, 2, 3, 4]
my_list2 = my_list.copy()
my_list2[1] = 5
my_list
{% endhighlight %}




    [1, 2, 3, 4]



### Tuples

Tuples are more or less immutable lists, and as such have less methods
associated with them

**In [57]:**

{% highlight python %}
my_tup = (1,2,3,4,5,1)
print(my_tup.index(5))
print(my_tup[2])
print(my_tup.count(1))
print(list(my_tup))
my_tup[0] = 1
{% endhighlight %}

    4
    3
    2
    [1, 2, 3, 4, 5, 1]



    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-57-e9f3c06e3e04> in <module>()
          4 print(my_tup.count(1))
          5 print(list(my_tup))
    ----> 6 my_tup[0] = 1


    TypeError: 'tuple' object does not support item assignment


**In [58]:**

{% highlight python %}
#multiple assignment technically uses tuples
a,b,c,d = 1,2,3,4
{% endhighlight %}

Tuples are a great choice when using parameters in a script - we cannot
overwrite them by mistake without reassigning the entire tuple

### Sets

Sets work like the mathematical notion of sets - only unique elements are
allowed, and they are unordered

**In [59]:**

{% highlight python %}
my_set = {1,2,3,4,5,1,2}
print(my_set)
print(type(my_set))
{% endhighlight %}

    {1, 2, 3, 4, 5}
    <class 'set'>


**In [187]:**

{% highlight python %}
# no subsetting by index!
# hence no mutability either
my_set[0]
{% endhighlight %}


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-187-820e4eb6e783> in <module>()
          1 # no subsetting by index!
          2 # hence no mutability either
    ----> 3 my_set[0]


    TypeError: 'set' object does not support indexing


**In [60]:**

{% highlight python %}
my_set.intersection({1,2,6,7,8})
{% endhighlight %}




    {1, 2}



**In [61]:**

{% highlight python %}
my_set.union({1,2,6,7,8})
{% endhighlight %}




    {1, 2, 3, 4, 5, 6, 7, 8}



**In [62]:**

{% highlight python %}
my_set.add(10)
print(my_set)
{% endhighlight %}

    {1, 2, 3, 4, 5, 10}


### Use Cases

Sets are useful to enumerate every possibility a data set has taken on:

**In [63]:**

{% highlight python %}
things_i_buy = ["razors", "apples", "oranges", "apples", "bananas", "apples", "bananas"]
unique_items = set(things_i_buy)
print(unique_items)
{% endhighlight %}

    {'apples', 'oranges', 'bananas', 'razors'}


**In [64]:**

{% highlight python %}
my_dna = ["a", "t", "g", "c", ["a", "t", "g", "g", "a"], "t", "a", ["a", "t", "a", "a", "a"]]
possible_bases = set(my_dna[4])
print(possible_bases)
{% endhighlight %}

    {'a', 't', 'g'}


### Dictionaries

Python dictionaries are a very useful data structure, which we will modify into
DataFrames inside pandas.

For now they are very similar to hashes or lookup tables from other languages -
we access data inside them by key, rather than index. As such, they are
unordered, similar to sets.


**In [65]:**

{% highlight python %}
#key:values pairs, keys must be unique
my_dict = {'key1' : "val1", 'key2':'val2', 1:[1,2,3,4], 2:"a"}
print(my_dict)
{% endhighlight %}

    {'key2': 'val2', 'key1': 'val1', 2: 'a', 1: [1, 2, 3, 4]}


**In [66]:**

{% highlight python %}
print(my_dict[1])
print(my_dict['key2'])
print(my_dict[0])
{% endhighlight %}

    [1, 2, 3, 4]
    val2



    ---------------------------------------------------------------------------

    KeyError                                  Traceback (most recent call last)

    <ipython-input-66-cb18261125fd> in <module>()
          1 print(my_dict[1])
          2 print(my_dict['key2'])
    ----> 3 print(my_dict[0])


    KeyError: 0


**In [67]:**

{% highlight python %}
my_dict['newkey'] = 'newval'
my_dict.update({"newkey2":'newval2'})
print(my_dict)
{% endhighlight %}

    {'key2': 'val2', 1: [1, 2, 3, 4], 2: 'a', 'newkey': 'newval', 'newkey2': 'newval2', 'key1': 'val1'}


**In [68]:**

{% highlight python %}
print(my_dict.keys())
print(my_dict.items())
print(my_dict.values())
{% endhighlight %}

    dict_keys(['key2', 1, 2, 'newkey', 'newkey2', 'key1'])
    dict_items([('key2', 'val2'), (1, [1, 2, 3, 4]), (2, 'a'), ('newkey', 'newval'), ('newkey2', 'newval2'), ('key1', 'val1')])
    dict_values(['val2', [1, 2, 3, 4], 'a', 'newval', 'newval2', 'val1'])


**In [69]:**

{% highlight python %}
my_dict['key1']
{% endhighlight %}




    'val1'



### Use Cases

We can store a 'key' as a unique id, then take a list to hold its' values:

**In [70]:**

{% highlight python %}
my_transactions = {'cust1':{'transaction1':['apples', 'bananas', 'coke'], 'transaction2':['cookies', 'coke']},
                   'cust2':{'transaction1':['oranges', 'razors', 'coke'], 'transaction2':['bananas', 'coke']}}
print(set(my_transactions['cust1']['transaction1'] + my_transactions['cust1']['transaction2']))
print(my_transactions['cust2']['transaction1'])
{% endhighlight %}

    {'apples', 'coke', 'bananas', 'cookies'}
    ['oranges', 'razors', 'coke']


**In [71]:**

{% highlight python %}
common_mispellings = {"sucess":"success", "succes":"success", "success":"suces", "success":"success"}
common_mispellings["succes"]
{% endhighlight %}




    'success'



## Motivation

How much can we actually learn and predict from Data Science using Python?

Recently, a match fixing ring in professional tennis was alleged by a joint
investigation between [BBC news](http://www.bbc.com/sport/tennis/35319202) and
[Buzzfeed](http://www.buzzfeed.com/heidiblake/the-tennis-racket#.eplO3d4px),
resulting in a large amount of news coverage.

The data analysis carried out was done in Python, and released online as an
[iPython Notebook](https://github.com/BuzzFeedNews/2016-01-tennis-betting-
analysis/blob/master/notebooks/tennis-analysis.ipynb). This story, and its
continued fall out, was [front page news on the
Guardian](http://www.theguardian.com/sport/2016/feb/09/revealed-tennis-umpires-
secretly-banned-gambling-scam) yesterday (9-Feb-2016).

Have a read through the notebook, take note of the functions and methods used on
the data, and see if you believe the analysis.

## Exercises

* We did not discuss in depth the order of operations of math in Python.
Luckily, it follows standard BODMAS ordering. Without evaluating, what will the
following expressions give:
  - 15 * (2 + 5)
  - 15 * 2 + 5
  - 15 + 2 * 5
  - 15 \*\* 2 * 5

* One type of numeric data not mentioned was complex numbers. What methods exist
for complex numbers (use the help)?

**In [72]:**

{% highlight python %}
type(complex(2,3))
{% endhighlight %}




    complex




* A previously used Numeric type, `long`, has been removed from Python 3. Can
you find the largest possible number representable in Python 3.x (feel free to
use google)? (Advanced) Can you explain how Python represents extremely large
numbers?



* I want to make a string, which will contain 3 backslashes, then a tab, a t,
then two newlines, a backslash and an n. Like this:

\\\\\&nbsp;&nbsp;&nbsp;&nbsp;t

\n

&nbsp;&nbsp;&nbsp;&nbsp;Write a string and print it out so it appears like this.
Do it using both triple quotes and multiple lines, and a single line with
special characters

* Using a single subset, print out every 4th letter of the following string,
starting at the second last letter, and ending at the 3rd letter: string =
"ajtougjlpfglnhfejfghu"


* Using only two methods on the following string, make a list ["ABC", "EFG",
"HIJ"]. string = 'abcdefgdhij'


* Subset the 2 from this dictionary:

**In [158]:**

{% highlight python %}
my_dict = {'key1':[{'key2':['hi',{'nested':[1,['hello',2]]}]}, 1, 5]}
{% endhighlight %}

* For the following lists - modify the second item in mylist to be 3. Multiply
that by the second item in mylist2. Is the answer what you would expect?

**In [159]:**

{% highlight python %}
mylist = [1,2,3]
mylist2 = mylist
{% endhighlight %}

* For the following dictionary, make a list containing one of each of the unique
items bought in transaction1 by any customer:

**In [160]:**

{% highlight python %}
my_transactions = {'cust1':{'transaction1':['apples', 'bananas', 'coke'], 'transaction2':['cookies', 'coke']},
                   'cust2':{'transaction1':['oranges', 'razors', 'coke'], 'transaction2':['bananas', 'coke']}}
{% endhighlight %}

* Using the functions `int()`, `str()`, `float()`, `bool()` we can cast objects
into the required type. What do the following commands give (try and predict
without running). After running, see if you can explain why each result is as it
is:
   - int(None)
   - int(True)
   - int(3.0)
   - int("hi")
   - str(2.0)
   - str(2)
   - str(None)
   - str(True)
   - float(True)
   - float("hi")
   - bool(3.0)
   - bool(0)
   - bool(-3)
   - bool("hi")
   - bool([])


* We have a list, my_house, with room names and sizes. Add a new room,
'kitchen', with a size of 47.5 to the list. follow the convention of one list
per room.

**In [162]:**

{% highlight python %}
my_house = [['bedroom', 45], ['living room', 34],
            ['bathroom', 30], ['den', 38]]
{% endhighlight %}

* It turns out we need to remove the den from our plans - use the function `del`
(hint, help(del)) to remove it

* Make a copy of my_house, as my_other_house. Change the name of the 'kitchen'
to 'laundry'

* Change the areas of my_other_house rooms to be half the size they are
currently (do this one by one for now).

* Check my_house - is it as you expect?


* Type in help(my_dict.update). From this help, how would you add one dict to
another? Does this happen in place or a new variable?


* Use the .join() method, to create a string that looks like 'a-b-c-d' from the
list ['a','b','c','d']. Find the method first.


* Do you think 2 \* [1, 2, 3, 4, 5] give a sensible result? Why/Why not? What
should 2 \* [1, 2, 3, 4, "a string"] return?
