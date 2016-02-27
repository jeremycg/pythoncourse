---
layout: post
title: "Lesson 02 Functions in Python"
author: Jeremy
tags:
 - Applied Statistics stream
comments: true
---
## Welcome to Lesson 02

Everyone should have covered the previous lessons, and the exercises. If there
are any problems with any of the exercises, please post on the forum - if there
is enough interest, we can cover a question or two in the next class.

I'll go over a couple of questions from last lesson now.

Today we will cover functions, lambda functions, functional programming, scoping
in Python, dynamic programming and decorators. Download the [notebook
here](/pythoncourse/assets/notebooks/applied/lesson%2002%20applied.ipynb).

As an aside, this course is being developed and hosted using github - if you
want access to all the notebooks, you can find [them
here](https://github.com/jeremycg/pythoncourse/tree/raw/assets/notebooks). If
you know how to use git and github feel free to file an issue or pull request if
you spot a typo. If not, we will cover git later in the course.

### Intro to Functions

We have already seen how to use the built-in Python functions. You can see a
list of every fucntion included in Python 3.5 [here on the official Python
documentation site](https://docs.python.org/3.5/library/functions.html). Today
we will learn how to create our own.

Functions are one of the key parts of any programming language. We can use them
when we have a chunk of code we want to reuse and don't want to type out a lot
of times.

Key parts of a function are its name, the arguments, the function body, the
return statement, and the docstring.

### Writing our first function

**In [38]:**

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



**In [39]:**

{% highlight python %}
myfirstfun(1, 2)
{% endhighlight %}




    3



**In [40]:**

{% highlight python %}
myfirstfun("h", "i")
{% endhighlight %}




    'hi'



We might not want our function to work on strings! We can raise an error if the
arguments are not numeric:

**In [41]:**

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



**In [42]:**

{% highlight python %}
myfirstfun("h", "i")
{% endhighlight %}


    ---------------------------------------------------------------------------

    AssertionError                            Traceback (most recent call last)

    <ipython-input-42-b9e7ba21c90e> in <module>()
    ----> 1 myfirstfun("h", "i")


    <ipython-input-41-f4e6512fed1f> in myfirstfun(arg1, arg2)
          5     '''
          6     for i in arg1, arg2:
    ----> 7         assert type(i) == int or type(i) == float, "This function requires numerics"
          8     return(arg1 + arg2)
          9


    AssertionError: This function requires numerics


### Passing Multiple Arguments and Default Values

Some functions will need to take an unknown number of arguments, or a default
value:

**In [43]:**

{% highlight python %}
def myfirstfun(arg1, arg2 = 2, *moreargs):
    '''
    This function takes two arguments, and returns the sum of them
    any extra arguments are printed to the screen
    '''
    for i in arg1, arg2:
        assert type(i) == int or type(i) == float #can also use isinstance()
    print("here are the extra arguments {x}".format(x = list(moreargs)))
    return(arg1 + arg2)

myfirstfun(1,2,3,4,5,6)
{% endhighlight %}

    here are the extra arguments [3, 4, 5, 6]





    3



**In [44]:**

{% highlight python %}
myfirstfun(10)
{% endhighlight %}

    here are the extra arguments []





    12



### Named Arguments

Python supports named arguments, unlike most languages, but similar to R:

**In [3]:**

{% highlight python %}
def myfunction(customer, basket):
    return({customer: basket})
myfunction(basket = [1,2,3,4], customer = "customer1")
#by position, expect error
{% endhighlight %}




    {'customer1': [1, 2, 3, 4]}



We go by name, then order

**In [46]:**

{% highlight python %}
def myfunction(item, number = 1, cost = 3, weight = 1):
    for i in item,number,cost,weight:
        print(i)
myfunction("bananas", cost = 5)
print("\n")
myfunction("oranges", 3)
{% endhighlight %}

    bananas
    1
    5
    1


    oranges
    3
    3
    1


We can combine named and unnamed values. Any positional arguments are passed as
a tuple, any named arguments are passed as a dictionary. We add a \* to
positional, and \*\* to keyword arguments. Very often, we will see this
arguments denoted as \*args and \*\*kwargs, but we can name them whatever we'd
like.

Named arguments must go last:

**In [47]:**

{% highlight python %}
def myfunction(*positional, **keywords):
    print("Positional:", positional)
    print("Keywords:", keywords)

myfunction('one', 'two', 'three')
myfunction('one', a = 'two', b = 'three')
{% endhighlight %}

    Positional: ('one', 'two', 'three')
    Keywords: {}
    Positional: ('one',)
    Keywords: {'b': 'three', 'a': 'two'}


**In [48]:**

{% highlight python %}
myfunction(a = 'one', b = 'two', 'three')
{% endhighlight %}


      File "<ipython-input-48-80cc4c5ccb1f>", line 1
        myfunction(a = 'one', b = 'two', 'three')
                                        ^
    SyntaxError: positional argument follows keyword argument



### Warning

For default values, we evaluate them at function definition - this saves time
but can lead to error when using mutable defaults

**In [10]:**

{% highlight python %}
def myfunction(customer = "1", spend = 2**45):
    return(customer + ": " + str(spend))
print(myfunction())
print(myfunction("2",4))
print(b) #only locally defined
{% endhighlight %}

    1: 35184372088832
    2: 4



    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-10-77c7705ebe6e> in <module>()
          3 print(myfunction())
          4 print(myfunction("2",4))
    ----> 5 print(b) #only locally defined


    NameError: name 'b' is not defined


**In [17]:**

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


**In [18]:**

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

**In [19]:**

{% highlight python %}
def internaldef(a,b):
    def helper(c):
        return(c ** 5)
    return(helper(a) + helper(b))
internaldef(2,2)
{% endhighlight %}




    64



Internal functions have access to the enclosing variables:

**In [20]:**

{% highlight python %}
def internaldef(a,b):
    def helper(c):
        return(c ** b)
    return(helper(a) + helper(b))
internaldef(2,2)
{% endhighlight %}




    8



And we can even return them.

**In [23]:**

{% highlight python %}
def makepow(a):
    def pown(num):
        return(num ** a)
    return(pown)
pow4 = makepow(4)
pow4(3)
{% endhighlight %}




    81



A function defined in this manner comes with its own enclosing environment and
is termed a closure

**In [27]:**

{% highlight python %}
import dis
dis.dis(pow4)
{% endhighlight %}

      3           0 LOAD_FAST                0 (num)
                  3 LOAD_DEREF               0 (a)
                  6 BINARY_POWER
                  7 RETURN_VALUE


**In [40]:**

{% highlight python %}
pow4.__closure__[0].cell_contents
{% endhighlight %}




    4



More about this once we begin object orientated programming! For now, know that
everything in Python is an object, and we have called the \__closure\__ method
on our function object to get out its values

### Lambda Functions and map

Lambda functions are functions which are defined and used in the same place -
they are 'anonymous' (see Rs version `lapply(x, function(z) z^2`)). They are
generally used inside another statement, where defining a function is not worth
the time.

We can also use the lamdba to define a function:

**In [22]:**

{% highlight python %}
myfun = lambda x,y: x+y
myfun(1,2)
{% endhighlight %}




    3



**In [23]:**

{% highlight python %}
l = [[1,2],[3,4],[5,6]]
[myfun(x,y) for x,y in l]
{% endhighlight %}




    [3, 7, 11]



`map` takes two arguments - a function (often a lambda function) and a sequence
(or sequences) to iterate over

**In [49]:**

{% highlight python %}
l = [1,2,3,4,5,6]
x = map(lambda x: x if x % 2 == 1 else x **2, l)
list(x)
{% endhighlight %}




    [1, 4, 3, 16, 5, 36]



**In [50]:**

{% highlight python %}
l = range(6)
k = range(6,12)
x = map(lambda x,y : x*y, l,k)
list(x)
{% endhighlight %}




    [0, 7, 16, 27, 40, 55]



**In [51]:**

{% highlight python %}
x = map(lambda x, y: x if x%2 == 1 else y, l,k)
list(x)
{% endhighlight %}




    [6, 1, 8, 3, 10, 5]



### Functional Programming : Filter, Any, All and Reduce

In addition to map, several other built in functions use lambda functions, or
take functions as arguments. Python purists might say to use the more explicit
loops, but functional programming is highly optimized.

### Filter

Filter is used to remove certain items from a sequence:

**In [52]:**

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



**In [53]:**

{% highlight python %}
x = filter(lambda x: x in fruits, basket)
list(x)
{% endhighlight %}




    ['oranges', 'bananas']



### Any and all

any and all are functions which check if every (or any) member or an iterable
are True:

**In [54]:**

{% highlight python %}
l = [True, True, True]
print(all(l))
print(any(l))
{% endhighlight %}

    True
    True


We can use them similar to generators however:

**In [55]:**

{% highlight python %}
basket = ['oranges','bread','bananas','milk']
fruits = ['oranges','apples','bananas','kiwis','strawberries', 'pears']
print(any(item in fruits for item in basket))
print(all(item in fruits for item in basket))
{% endhighlight %}

    True
    False


**In [56]:**

{% highlight python %}
#empty lists have differing behaviour
l = []
print(any(l))
print(all(l))
{% endhighlight %}

    False
    True


### Reduce

Reduce was removed from base Python in version 3.0, but remains a source of
controversy, as it is a key function in higher level programming languages. The
official advice is to use a loop:

**In [11]:**

{% highlight python %}
l = [2,3,4,5]
result = l[0]

for i in l[1:]:
    result = result**i
result
{% endhighlight %}




    1152921504606846976



We can import reduce from functools:

**In [12]:**

{% highlight python %}
from functools import reduce

l = [2,3,4,5]
reduce(lambda x, y: x ** y, l)
{% endhighlight %}




    1152921504606846976



If you can't figure out what we've done, that's one of the reasons why reduce
was removed. It works by taking a list of arguments, applying the first function
to the first two members, then to the output of that, and the next variable....

(((2 \*\* 3) \*\* 4) \*\* 5)

### Call by reference and side effects

In programming, we have two ways of passing values into functions, calling by
reference, or calling by value. Python uses both (and calls it passing by name)!

Call by value uses the values of an argument, and never modifies the actual
passed value (R does this, as do most versions of SQL):


**In [25]:**

{% highlight python %}
a = 45
def myfunc(x):
    x = x + 5
    return(x)
print(myfunc(a))
print(a)
{% endhighlight %}

    50
    45


For immutable objects, we always pass by value. Under the hood, Python keeps the
variable in the same location in memory, until we modify it (this is a
considerable speed up over copying if we pass a large object). Using the id
function, we can see what is going on:

**In [31]:**

{% highlight python %}
a = 45
print("a is at "+ str(id(a)))
def myfunc(x):
    print("x is at " + str(id(x)))
    x = x + 5
    print("x is now at " + str(id(x)))
    return(x)
print(myfunc(a))
print(a)
{% endhighlight %}

    a is at 1935078544
    x is at 1935078544
    a is at 1935078704
    50
    45


For mutable objects, we pass by reference and we can cause all sorts of trouble
(or desirable things). Again, this is similar to c++ pointers and can be
implemented in some SQL builds using IN OUT parameters:

**In [37]:**

{% highlight python %}
a = [1,2,3]
def myfunc(x):
    x += [4]
    return 3
print(myfunc(a))
print(a)
#a is changed! even though we didn't return it!
{% endhighlight %}

    3
    [1, 2, 3, 4]


Again, we can track ids to see what is happening under the hood:

**In [36]:**

{% highlight python %}
a = [1,2,3]
print("a is at "+ str(id(a)))
def myfunc(x):
    print("x is at "+ str(id(x)))
    x += [4]
    print("x is now at " + str(id(x)))
    return 3
print(myfunc(a))
print(a)
{% endhighlight %}

    a is at 2253675131528
    x is at 2253675131528
    x is now at 2253675131528
    3
    [1, 2, 3, 4]


Technically, we have caused a "side effect" in that we have modifed the global
environment inside a function. In functional programming langauges, this is a
major no-no. In object orientated languages, this is perfectly reasonable as
long as you are careful.

### Scoping in Python

Scoping is a technical term for where and how a language searches for, and
modifies variables. Hopefully the below makes sense to you:

**In [38]:**

{% highlight python %}
x = 1
def myfunc(y):
    x = y
    return(x)
print(myfunc(25))
print(x)
{% endhighlight %}

    25
    1


We have not modified the global x, as x is local to the function.
Python uses LEGB scoping rules to search for variables:

* Local - is there a local definition available inside the current enclosing
environment?
* Enclosing - is there an enclosing defintion?
* Global - is there a global definition?
* Built-in - is there a built in definition?

Here is an example:

**In [40]:**

{% highlight python %}
x = "x"
a = "a"
b = "b"
def myfunc(a,b):
    def myfunc2():
        print(a + " is enclosed")
    myfunc2()
    print(b + " is local")
    print(x + " is global")
    print(abs)

myfunc("c", "d")
{% endhighlight %}

    c is enclosed
    d is local
    x is global
    <built-in function abs>


If we want to change a global variable from inside a function, we can if it is
mutable (see the above side effects section), otherwise we can use the `global`
command:

**In [42]:**

{% highlight python %}
x = 2
def myfunc(y):
    global x
    x /= 10
    return(y)
print(myfunc(10))
print(x)
{% endhighlight %}

    10
    0.2


Again, this is causing side effects, which may or may not be desirable. Use
`global` with care.

If we are bugfixing, we can use the `globals()` and `locals()` functions to
print all the variables we can see in an environment.

### Dynamic programming and caching

Dynamic programming is a method of optimizing functions by cutting the problem
into small pieces. Here we will run through a recursive example, however many
other methods exist. I'd love to teach you guys some examples of where they
might help - however algorithms is outside the scope of the course.

You might recall from the initial pre test, we can create the fibonacci sequence
in a couple of ways:

**In [14]:**

{% highlight python %}
def fibo(x):
    if x < 3:
        return 1
    a,b,counter = 1,2,3
    while counter < x:
        a,b,counter = b,a+b,counter+1
    return(b)
print(list(map(fibo, range(1,10))))
print([fibo(x) for x in range(1,10)])
{% endhighlight %}

    [1, 1, 2, 3, 5, 8, 13, 21, 34]
    [1, 1, 2, 3, 5, 8, 13, 21, 34]


**In [19]:**

{% highlight python %}
def fiborecur(x):
    if x < 3:
        return 1
    return fiborecur(x - 1) + fiborecur(x - 2)
list(map(fiborecur, range(1,10)))
{% endhighlight %}




    [1, 1, 2, 3, 5, 8, 13, 21, 34]



The second of these is dynamic! We have made a recursive function, to do a very
simple task multiple times

However, let's check the performance using %timeit

**In [65]:**

{% highlight python %}
%timeit list(map(fibo, range(1,15)))
%timeit list(map(fiborecur, range(1,15)))
{% endhighlight %}

    100000 loops, best of 3: 19.3 µs per loop
    1000 loops, best of 3: 457 µs per loop


It performs much worse! And only gets worse as we increase x. Why?

We can think a little about how it works - the non-recursive solution has to do
x calculations, one for each increase.

The recursive solution has to do 2^x calculations - each call will have two
offspring calls.

Can we make it better? One way is to use a dictionary to remember our outputs.

**In [17]:**

{% highlight python %}
fib_cache = {}
def fiborecur2(x):
    if x in fib_cache:
        return fib_cache[x]
    fib_cache[x] = x if x < 2 else fiborecur2(x - 1) + fiborecur2(x - 2)
    return fib_cache[x]
print(fiborecur2(10))
print(fib_cache)
{% endhighlight %}

    55
    {0: 0, 1: 1, 2: 1, 3: 2, 4: 3, 5: 5, 6: 8, 7: 13, 8: 21, 9: 34, 10: 55}


**In [10]:**

{% highlight python %}
%timeit list(map(fibo, range(1,15)))
%timeit list(map(fiborecur, range(1,15)))
%timeit list(map(fiborecur2, range(1,15)))
{% endhighlight %}

    100000 loops, best of 3: 18 µs per loop
    1000 loops, best of 3: 478 µs per loop
    100000 loops, best of 3: 6.04 µs per loop


Great! Can we generalise this method? We could create a closure:

**In [22]:**

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

fiborecur3 = memoize(fiborecur)
{% endhighlight %}

**In [16]:**

{% highlight python %}
%timeit fiborecur3(15)
%timeit fiborecur2(15)
%timeit fiborecur(15)
{% endhighlight %}

    The slowest run took 5.62 times longer than the fastest. This could mean that an intermediate result is being cached
    1000000 loops, best of 3: 342 ns per loop
    The slowest run took 8.69 times longer than the fastest. This could mean that an intermediate result is being cached
    1000000 loops, best of 3: 265 ns per loop
    1000 loops, best of 3: 289 µs per loop


### Decorators

We can modify functions in Python using decorators. This is denoted by the @
symbol:

**In [18]:**

{% highlight python %}
@memoize
def fibn(x):
    if x < 3:
        return 1
    return fiborecur(x - 1) + fiborecur(x - 2)

%timeit fibn(15)
{% endhighlight %}

    The slowest run took 885.25 times longer than the fastest. This could mean that an intermediate result is being cached
    1000000 loops, best of 3: 347 ns per loop


Decorators allow us to modify functions:

**In [20]:**

{% highlight python %}
def myfunc(x):
    print("Hello " + x)

myfunc("Precima")
{% endhighlight %}

    Hello Precima


**In [24]:**

{% highlight python %}
def mydecorator(func):
    def wrapper(x):
        func(x)
        print("That's the end of class two")
    return wrapper

@mydecorator
def myfunc(x):
    print("Hello " + x)

myfunc("Precima")
{% endhighlight %}

    Hello Precima
    That's the end of class two


We will cover in more detail how decorators work towards the end of the course

## Exercises

The exercises are designed to be difficult - evidence shows that leaners learn by coding,
in combination with lectures. You should be using google, stackoverflow and your colleagues
 to help with your work, and you are allocated a good amount fo time to do them.

With that said, if you run out of time, feel free to skip the remaining exercises,
and I will split out the advanced questions into an optional section.

* Read through the dynamic programming section that we skipped in class. Don't worry
 too much about specifics, the main take home message is that we can return the output of the same function.

* Rewrite your loop from last lessons exercises to find if all the letter in a
string are in another string as a function. (Advance, optional) Can you think of a dynamic
programming way to remove duplicates?

* Rewrite your zip loop from last lesson exercises as a function (Write a
modification of the current zip function, to work until the longest of the
inputs)

* Make your own function for converting fahrenheit to celsius

* Update the above function to have a docstring, explaining what it does

* Update the function to only take int or floats - use an assertion

* Write a function to take an arbitrary number of unnamed arguments, and return
their sum. Make sure you have a docstring, and test for numerics.

* Modify the above argument to use `reduce`, Remember you need to import reduce.

* Write a function that takes an arbitrary number of named arguments, and
returns the key of those that had even numbers as their input. myfunc(a = 1, b =
2, c = 3, d = 4) should return ['b', 'd']

* Write a function to make a function to check if an item is in a list: def myfruits(fruits):
, myfruits(["apple", 'banana', 'potato', 'cauliflower']) should
return [True, True, False, False]. fruits =
['oranges','apples','bananas','kiwis','strawberries', 'pears'].

* Update the above function, to generate a function which will return a single
True or False if any members of the list are present.

* Update the above function to return as above, but with a return of True when
called on an empty list: myfruits([]) will give True

* (Advanced, optional) Write a function to find the xth prime number: myprime(x)

* (Advanced, optional) Modify the above function to be recursive

* (Advanced, optional) Modify this function to be cached - does it help with performance? Why/Why
not?
