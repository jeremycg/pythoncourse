---
layout: post
title: "Lesson 01 Statements in Python"
author: Jeremy
tags:
 - Applied Statistics stream
comments: true
---
## Welcome to Lesson 01

If you did not make it to lesson 00, please check it out on the website, it's
assumed you attended [Lesson 00 - Applied
Section](/pythoncourse/2016/02/09/lesson00applied.html).

Today we go over a quick review and some extensions of last lesson, then dive
into Python statements and functions.

Grab todays lesson from [this
link!](/pythoncourse/assets/notebooks/applied/lesson 01 - applied.ipynb).

### Statements

Python was written with clarity and simplicity in mind, and one of the reasons
this has worked is the basic Python syntax.

Unlike many other programming languages, Python relies on indentation rather
than braces - Thus we cannot easily create a tangled mess like this (my code, in
R):

`sapply(1:length(factors), function(index)
{any(sapply(1:length(factors[[index]]), function(zz)
{all((rep(holding[[index]][1:(length(holding[[index]])/factors[[index]][zz])],
factors[[index]][zz]))==holding[[index]])}))})
`

Instead, code is based on indentation, forcing at least some sensible structure
into the code:

**In [1]:**

{% highlight python %}
l = [1,2,3,4]
for i in l:
    print(i)
{% endhighlight %}

    1
    2
    3
    4


Indentation will likely be taken care of by your IDE/editor, but it is either 4
spaces, or two tabs (not recommended). Choose one and stay with it, as you will
get errors if you mix them or do not indent:


**In [56]:**

{% highlight python %}
for i in l:
print(i)
{% endhighlight %}


      File "<ipython-input-56-4076db310a51>", line 2
        print(i)
            ^
    IndentationError: expected an indented block



Indenting and unindenting is equivalent to closing brackets.

This clarity has lead to people comparing Python to Pseudocode, but we will
shortly see some counterexamples, where Python syntax becomes very confusing.

### Outside Resources

A number of people have asked about text books or online resources for the
course. Most Python tutorials online will assume a knowledge at around the level
we will reach after a few lessons. I'm using [Python for Data
Analysis](http://shop.oreilly.com/product/0636920023784.do) for the Numpy and
Pandas section - it won't be necessary to purchase but is a good book (if a
little outdated). For the other lessons, links will be provided to the relevant
docs.



### Basic Data Types Review

| Type       | Example     | Mutable | Ordered | Duplicates | Subsetting |
|------------|-------------|---------|---------|------------|------------|
| int        | 2           | NA      | NA      | NA         | NA         |
| float      | 2.5         | NA      | NA      | NA         | NA         |
| complex    | 2.5 + 0.6J  | NA      | NA      | NA         | NA         |
| boolean    | True        | NA      | NA      | NA         | NA         |
| string     | 'abcd'      | No      | Yes     | Yes        | x[1]       |
| list       | [1,2,3,'a'] | Yes     | Yes     | Yes        | x[1]       |
| tuple      | (1,2,3,'a') | No      | Yes     | Yes        | x[1]       |
| set        | {1,2,3,4}   | Yes     | No      | No         | No         |
| dictionary | {1:1,2:2}   | Yes     | No      | No         | x['key']   |




### Booleans - Advanced

In addition to the standard boolean operators (<,>,==,!=,) mentioned in the
first
class, there are five others: `in`,`is`,`not`, `any` and `all`. Any and all will
be introduced in the functional programming section.

**In [2]:**

{% highlight python %}
l = ['oranges','apples','bananas','bananas','kiwis']
'oranges' in l
{% endhighlight %}




    True



**In [3]:**

{% highlight python %}
l = 5
5 is l
##very similar to ==
{% endhighlight %}




    True



**In [4]:**

{% highlight python %}
#we are checking identity, not equality
l = ['oranges','apples','bananas']
k = ['oranges','apples','bananas']
print(l is k)
k = l
print(l is k)
{% endhighlight %}

    False
    True


**In [5]:**

{% highlight python %}
not True
{% endhighlight %}




    False



In Python (and most sensible languages), boolean comparisons are short circuit
operators - once one conditon is impossible we stop evaluating. This can get a
good speed up on code if you have one expensive comparison and one that is not -
Therefore, put the cheap calculation first!

**In [6]:**

{% highlight python %}
#del(l)
x = "bananas"
#l is not defined! This would give and error if it hits l
print(x == "bananas" or l == 4)
print(x == "bananas" and l == 4)
{% endhighlight %}

    True
    False


### Data Structures Review

**In [7]:**

{% highlight python %}
customer_1 = {'loyaltyids':(1234, 3456),
              'transactions':{1:['oranges','apples','bananas','pears','kiwis'],
                              2:['bread', 'milk','bananas','bananas']},
              'postalcodes':{'M5T1V1', 'M5S3B2'},
              'transactionspends':[100.46, 34.55],
              'usedcoupons':[True, False]}
{% endhighlight %}

**In [8]:**

{% highlight python %}
#subsetting here
{% endhighlight %}

## Statements

### If, Else, Elif

We can do if and else statements

**In [9]:**

{% highlight python %}
x = 4.99

if not True:
    print(x)
elif x < 5:
    print('on sale')
else:
    print('regular price')


#if True:
#print(x)
{% endhighlight %}

    on sale


and nest them as deep as we would like:

**In [10]:**

{% highlight python %}
pricehistory = [11]

if pricehistory[-1] >= 10 and len(pricehistory) == 1:
    if pricehistory[-1] >= 12:
        pricehistory.append(pricehistory[0] - 1)
        print('new price is ' + str(pricehistory[-1]))
    else:
        pricehistory.append(pricehistory[0] - 0.10 * pricehistory[-1])
        print('new price is ' + str(pricehistory[-1]))
elif len(pricehistory) > 1:
    pricehistory.append(min(pricehistory))
    print('new price is ' + str(pricehistory[-1]))
else:
    pricehistory.append(pricehistory[0]*0.9)
    print('new price is ' + str(pricehistory[-1]))

pricehistory
{% endhighlight %}

    new price is 9.9





    [11, 9.9]



### For loops

For loops work similarly in syntax to if and else. We can use them on any of the
major data structures we just introduced

**In [11]:**

{% highlight python %}
l = ['oranges','bread','bananas','milk']
fruits = ['oranges','apples','bananas','kiwis','strawberries', 'pears']

for item in l:
    if item in fruits:
        print("fruit")
    else:
        print("other")

print("\n")
s = ['oranges','bread','bananas','milk']

for item in s:
    if item in fruits:
        print("fruit")
    else:
        print("other")
{% endhighlight %}

    fruit
    other
    fruit
    other


    fruit
    other
    fruit
    other


On dictionaries, we have to do it a little differently

**In [12]:**

{% highlight python %}
d = {'produce':'banana', 'dairy':'milk', 'canned':'chickpeas'}

for k, v in d.items():
    print(k + ": " + v)
#unordered!

#similar for nested lists:

l = [[1,2],[3,4],[5,6],[7,8]]

for one, two in l:
    print(one * two)
    #no modifying!
    one = one * 2
l
{% endhighlight %}

    canned: chickpeas
    dairy: milk
    produce: banana
    2
    12
    30
    56





    [[1, 2], [3, 4], [5, 6], [7, 8]]



### While loops

Similarly, we can use while loops, and run a loop until a condition is met

**In [13]:**

{% highlight python %}
x = 5

while x > 1:
    print(x)
    x -= 1
#also +=, *=, /=
{% endhighlight %}

    5
    4
    3
    2


### Break, Pass, Continue

We can skip, break or pass elements in our loops

**In [14]:**

{% highlight python %}
l = [1,2,3,4,"oranges",6]

for num in l:
    if type(num) == int:
        print(num)
    else:
        print(num + " is not a number")
        break

for num in l:
    if type(num) == int:
        print(num)
    else:
        print(num + " is not a number")
        continue
        print("how did you get here?")

for num in l:
    if type(num) == int:
        print(num)
    else:
        print(num + " is not a number")
        pass
        print("how did you get here?")


{% endhighlight %}

    1
    2
    3
    4
    oranges is not a number
    1
    2
    3
    4
    oranges is not a number
    6
    1
    2
    3
    4
    oranges is not a number
    how did you get here?
    6


### Range

We saw earlier, we cannot modify in place using a loop. We can instead use
`range` to generate indices and use these (NB range in Python 3 is similar to
xrange in Python 2.7, not `range`). Range objects are not instantly enumerated -
we dont have to hold them in memory.

**In [15]:**

{% highlight python %}
#similar to the subset from:to:by
range(0,10,2)
type(range(0,10))
{% endhighlight %}




    range



**In [16]:**

{% highlight python %}
#not instantly enumerated! this would fill the memory of nearly all computers
range(100000000000000000)
{% endhighlight %}




    range(0, 100000000000000000)



**In [17]:**

{% highlight python %}
l = [1,2,3,4,5,6]
range(len(l))
{% endhighlight %}




    range(0, 6)



**In [18]:**

{% highlight python %}
for i in range(len(l)):
    l[i] = l[i]*2
l
{% endhighlight %}




    [2, 4, 6, 8, 10, 12]



**In [19]:**

{% highlight python %}
#also useful for generating simple lists:
list(range(10))
{% endhighlight %}




    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]



We also have the built in function, `enumerate` which can give us a tuple of the
index and value at each position in a sequence:

**In [20]:**

{% highlight python %}
cheese = ['gouda','cheddar','edam','brie']
print(list(enumerate(cheese)))
for i,j in enumerate(cheese):
    print('item {i} is {j}'.format(i = i, j = j))
{% endhighlight %}

    [(0, 'gouda'), (1, 'cheddar'), (2, 'edam'), (3, 'brie')]
    item 0 is gouda
    item 1 is cheddar
    item 2 is edam
    item 3 is brie


### Zip

zip is a built in python function, that 'zips' sequences into tuples:

**In [21]:**

{% highlight python %}
item = ['oranges','apples','bananas','kiwis','strawberries']
cost = [7,8,9,10,11]
units = [12,13,14,15,16]
list(zip(item,cost,units))
{% endhighlight %}




    [('oranges', 7, 12),
     ('apples', 8, 13),
     ('bananas', 9, 14),
     ('kiwis', 10, 15),
     ('strawberries', 11, 16)]



### Comprehensions

Comprehensions are are syntactic sugar for for loops. We can make lists (or set
or dicts) using this method. Tuples might seem like they should also have a
method, but this is reserved for generators, which will discuss later on in the
course.


**In [22]:**

{% highlight python %}
#for loop
l = []
for i in range(10):
    l.append(i * 5)
l
{% endhighlight %}




    [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]



**In [23]:**

{% highlight python %}
#comprehension
[i * 5 for i in range(10)]
{% endhighlight %}




    [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]



**In [24]:**

{% highlight python %}
#dict comprehension
{i * 10 : j * 2  for i, j in {1: 'a', 2: 'b'}.items()}
{% endhighlight %}




    {10: 'aa', 20: 'bb'}



**In [25]:**

{% highlight python %}
celsius = [0,100,25,37]
[((9/5)*temp + 32) for temp in celsius]
{% endhighlight %}




    [32.0, 212.0, 77.0, 98.60000000000001]



### Ternary expressions

Ternary expressions are a fancy way of doing `if` `else` and are very pythonic:

**In [26]:**

{% highlight python %}
a = 12
a if a > 11 else b
{% endhighlight %}




    12



**In [27]:**

{% highlight python %}
a = 10
b = "no"
a if a > 11 else b
{% endhighlight %}




    'no'



we can chain this with our comprehensions

**In [28]:**

{% highlight python %}
[a if a else 2 for a in [0,1,0,3]]
{% endhighlight %}




    [2, 1, 2, 3]



or our nested comprehensions

**In [29]:**

{% highlight python %}
matrix = [[1,2,3],[4,5,6],[7,8,9]]
[[el * 2 if el % 2 == 0 else el for el in row] for row in matrix]
{% endhighlight %}




    [[1, 4, 3], [8, 5, 12], [7, 16, 9]]



As these comprehensions are not faster than a for loop, it is worth using for
loops once you can't follow clearly what the comprehension is doing (I would
almost never use a nested comprehension)!

## Intro to Functions

Functions are one of the key parts of any programming language. Today we will
touch on basic syntax and definition, and move into a more thorough exploration
in the next class.

**In [30]:**

{% highlight python %}
def celcius_to_fahr(temp):
    return (9/5)*temp + 32

celcius_to_fahr(100)
{% endhighlight %}




    212.0



**In [31]:**

{% highlight python %}
def kelvin_to_celcius(temp):
    return temp - 273

kelvin_to_celcius(100)
{% endhighlight %}




    -173



**In [32]:**

{% highlight python %}
def kelvin_to_fahr(temp):
    return celcius_to_fahr(kelvin_to_celcius(temp))

kelvin_to_fahr(273)
{% endhighlight %}




    32.0



**In [33]:**

{% highlight python %}
def myfirstfun(arg1, arg2):
    '''Here is the docstring. this will be displayed to a user calling help(myfirstfun)
    It can be as long as you'd like.
    This argument takes two arguments, and returns the sum of them
    '''
    return(arg1 + arg2)

help(myfirstfun)
{% endhighlight %}

    Help on function myfirstfun in module __main__:

    myfirstfun(arg1, arg2)
        Here is the docstring. this will be displayed to a user calling help(myfirstfun)
        It can be as long as you'd like.
        This argument takes two arguments, and returns the sum of them



**In [34]:**

{% highlight python %}
myfirstfun(1, 2)
{% endhighlight %}




    3



**In [35]:**

{% highlight python %}
myfirstfun("h", "i")
{% endhighlight %}




    'hi'



We might not want our function to work on strings! We can Raise an error is the
arguments are not numeric:

**In [36]:**

{% highlight python %}
def myfirstfun(arg1, arg2):
    '''Here is the docstring. this will be displayed to a user calling help(myfirstfun)
    It can be as long as you'd like.
    This argument takes two arguments, and returns the sum of them
    '''
    for i in arg1, arg2:
        assert type(i) == int or type(i) == float, "This function requires numerics"
    return(arg1 + arg2)

myfirstfun(1,2)
{% endhighlight %}




    3



**In [37]:**

{% highlight python %}
myfirstfun("h", "i")
{% endhighlight %}


    ---------------------------------------------------------------------------

    AssertionError                            Traceback (most recent call last)

    <ipython-input-37-b9e7ba21c90e> in <module>()
    ----> 1 myfirstfun("h", "i")


    <ipython-input-36-f4e6512fed1f> in myfirstfun(arg1, arg2)
          5     '''
          6     for i in arg1, arg2:
    ----> 7         assert type(i) == int or type(i) == float, "This function requires numerics"
          8     return(arg1 + arg2)
          9


    AssertionError: This function requires numerics


### Passing Multiple Arguments and Default Values

Some functions will need to take an unknown number of arguments, or a default
value:

**In [38]:**

{% highlight python %}
def myfirstfun(arg1, arg2 = 2, *moreargs):
    '''
    This function takes two arguments, and returns the sum of them
    any extra arguments are printed to the screen
    '''
    for i in arg1, arg2:
        assert type(i) == int or type(i) == float
    print("here are the extra arguments {x}".format(x = list(moreargs)))
    return(arg1 + arg2)

myfirstfun(1,2,3,4,5,6)
{% endhighlight %}

    here are the extra arguments [3, 4, 5, 6]





    3



**In [39]:**

{% highlight python %}
myfirstfun(10)
{% endhighlight %}

    here are the extra arguments []





    12



### Warning

For default values, we evaluate them at function definition - this saves time
but can lead to error when using mutable defaults

**In [40]:**

{% highlight python %}
def myfunction(a = 1, b = 2^45):
    return(a + b)
print(myfunction())
print(myfunction(2,4))
print(b)
{% endhighlight %}

    48
    6
    no


**In [41]:**

{% highlight python %}
def myfunction(item = "apples", basket = []):
    basket.append(item)
    return(basket)

print(myfunction("apples"))
print(myfunction("bananas"))
#extremely weird behaviour for anyone coming from R!
{% endhighlight %}

    ['apples']
    ['apples', 'bananas']


**In [42]:**

{% highlight python %}
def myfunction(item = 1, basket = None):
    if basket is None:
        basket = []
    basket.append(item)
    return(basket)
print(myfunction("apples"))
print(myfunction("bananas"))
{% endhighlight %}

    ['apples']
    ['bananas']


### Closures

We can do almost anything we'd like inside a function, including defining other
functions:

**In [43]:**

{% highlight python %}
def internaldef(a,b):
    def helper(c):
        return(c ** 5)
    return(helper(a) + helper(b))
internaldef(2,2)
{% endhighlight %}




    64



Internal functions have access to the enclosing variables:

**In [44]:**

{% highlight python %}
def internaldef(a,b):
    def helper(c):
        return(c ** b)
    return(helper(a) + helper(b))
internaldef(2,2)
{% endhighlight %}




    8



And we can even return them (a function generating function is technically a
'closure'):

**In [45]:**

{% highlight python %}
def makepow(a):
    def pown(num):
        return(num ** a)
    return(pown)
pow4 = makepow(4)
pow4(3)
{% endhighlight %}




    81



### Lambda Functions and map

Lambda functions are functions which are defined and used in the same place -
they are 'anonymous' (see Rs version `lapply(x, function(z) z^2`)). They are
generally used inside another statement, where defining a function is not worth
the time.

We can also use the lamdba to define a function:

**In [46]:**

{% highlight python %}
myfun = lambda x,y: x+y
myfun(1,2)
{% endhighlight %}




    3



**In [47]:**

{% highlight python %}
l = [[1,2],[3,4],[5,6]]
[myfun(x,y) for x,y in l]
{% endhighlight %}




    [3, 7, 11]



`map` takes two arguments - a function (often a lambda function) and a sequence
(or sequences) to iterate over

**In [48]:**

{% highlight python %}
l = [1,2,3,4,5,6]
x = map(lambda x: x if x % 2 == 1 else x **2, l)
list(x)
{% endhighlight %}




    [1, 4, 3, 16, 5, 36]



**In [49]:**

{% highlight python %}
l = range(6)
k = range(6,12)
x = map(lambda x,y : x*y, l,k)
list(x)
{% endhighlight %}




    [0, 7, 16, 27, 40, 55]



**In [50]:**

{% highlight python %}
x = map(lambda x, y: x if x%2 == 1 else y, l,k)
list(x)
{% endhighlight %}




    [6, 1, 8, 3, 10, 5]



## Functional Programming : Filter, Any, All

In addition to map, several other built in functions use lambda functions, or
take functions as arguments. Python purists might say to use the more explicit
loops, but functional programming is highly optimized.

### Filter

Filter is used to remove certain items from a sequence:

**In [51]:**

{% highlight python %}
basket = ['oranges','bread','bananas','milk']
fruits = ['oranges','apples','bananas','kiwis','strawberries', 'pears']
out = []
for item in basket:
    if item in fruits:
        out.append(item)
out
{% endhighlight %}




    ['oranges', 'bananas']



**In [52]:**

{% highlight python %}
x = filter(lambda x: x in fruits, basket)
list(x)
{% endhighlight %}




    ['oranges', 'bananas']



### Any and all

any and all are functions which check if every (or any) member or an iterable
are True:

**In [53]:**

{% highlight python %}
l = [True, True, True]
print(all(l))
print(any(l))
{% endhighlight %}

    True
    True


We can use them similar to generators however:

**In [54]:**

{% highlight python %}
basket = ['oranges','bread','bananas','milk']
fruits = ['oranges','apples','bananas','kiwis','strawberries', 'pears']
print(any(item in fruits for item in basket))
print(all(item in fruits for item in basket))
{% endhighlight %}

    True
    False


**In [55]:**

{% highlight python %}
#empty lists have differing behaviour
l = []
print(any(l))
print(all(l))
{% endhighlight %}

    False
    True
