---
layout: post
title: "Test 1 answers - r&d"
author: Jeremy
tags:
 - R&D stream
comments: true
---
## Test 1 - R&D stream

The data science department at Walmart often posts problems on the data science
challenge website, [kaggle](www.kaggle.com). In this test, we will download some
of their data, and use our current understanding of Python to do some
exploratory data analysis.

The challenge this data comes from was to revise the method they use to classify
trip types. The have a number of categories which shopping trips are clustered
into, and challenged entrants to recapitulate their clustering. The challenge is
[archived here](https://www.kaggle.com/c/walmart-recruiting-trip-type-
classification/). The training data set contains the TripType, as well as
VisitNumber, day of the week, Upc of product purchased, the scancount,
department, and fineline number (a categorical description of the item). Each
VisitNumber is a unique basket, with a line for each item scanned. I am
providing you with the first 50,000 lines of data, from the 650,000 total.

The Desired clustering is the categorical variable TripType.

You can complete this course without any knowledge of Pandas and Numpy - I am
loading the data in like this, as it is the easiest way (by far). Please leave
the code blocks in the Data Import section untouched - run them as needed. Feel
free to download the csv from the website and check it out, but use Python for
the analysis!

### Data Import

Here I'm loading the data from the course website, showing the first 5 lines,
and putting it into a dictionary, where each VisitNumber has its own entry.

The dict is called groups.

Please run the below (cursor in cell, then `ctrl-enter`, or click run cell.

**In [1]:**

{% highlight python %}
import pandas as pd
from pandas import DataFrame, Series
{% endhighlight %}

**In [2]:**

{% highlight python %}
dat = pd.read_csv("http://jeremy.kiwi.nz/pythoncourse/assets/tests/r&d/test1data.csv")
{% endhighlight %}

**In [3]:**

{% highlight python %}
dat.head()
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



**In [4]:**

{% highlight python %}
#convert DataFrame to dict
groups = dict(list(dat.groupby("VisitNumber")))
#convert dataframe rows to lists in dict
groups = {key: val.values.tolist() for key,val in groups.items()}
{% endhighlight %}

### Data Checking

Please run the below cell, you should get:

`
[[999, 5, 'Friday', 68113152929.0, -1, 'FINANCIAL SERVICES', 1000.0]]
[[30, 7, 'Friday', 60538815980.0, 1, 'SHOES', 8931.0], [30, 7, 'Friday',
7410811099.0, 1, 'PERSONAL CARE', 4504.0]]
`

If not, please redownload the notebook from the website.

**In [5]:**

{% highlight python %}
print(groups[5])
print(groups[7])
{% endhighlight %}

    [[999, 5, 'Friday', 68113152929.0, -1, 'FINANCIAL SERVICES', 1000.0]]
    [[30, 7, 'Friday', 60538815980.0, 1, 'SHOES', 8931.0], [30, 7, 'Friday', 7410811099.0, 1, 'PERSONAL CARE', 4504.0]]


### Test

Please fill the below questions in the cells and run them for output.

If you'd like another cell, use `alt-enter` or the insert menu.

If you'd like to enter text to explain, either use # for comments, or add a new
cell, then use the dropdown box above to convert it to markdown (from code).

Some data is missing or non-numeric, remember to check and remove or fix these
data points!

Don't worry about printing the outputs - assign them to a variable so I check
them

1\. Create a new dict, which contains the same keys, but a list of unique
DepartmentDescription of items for each visit: ie {7:['SHOES', 'PERSONAL CARE']}

**In [6]:**

{% highlight python %}
#a nested comprehension - set i:the 5th element for each j for i,j in dict
x = {i: set([j[q][5] for q in range(len(j))]) for i,j in groups.items()}
{% endhighlight %}

2\. If you used a function to do this, use a comprehension, if you used a
comprehension, use a function

**In [7]:**

{% highlight python %}
#nested loop
#use the temp variable to hold temporary values
def newdict(olddict):
    dict2 = {}
    for i,j in olddict.items():
        temp = []
        for x in j:
            temp.append(x[5])
        dict2[i] = set(temp)
    return(dict2)
y = newdict(groups)
{% endhighlight %}

**In [8]:**

{% highlight python %}
#check both methods are the same
x == y
{% endhighlight %}




    True



**In [9]:**

{% highlight python %}
#or using our dict of visits (after defining the class):
#dict2 = {}
#for i, j in newdict.items():
#    dict2.update({i:set(j.departments)})
{% endhighlight %}

3\. Create a new dict, with the total number of each category each customer
bought. It should look like `{7:[['SHOES',1], ['PERSONAL CARE',1]], ....}`

**In [10]:**

{% highlight python %}
#same as previous, but don't use set
def newdict2(olddict):
    dict2 = {}
    for i,j in olddict.items():
        temp = []
        for x in j:
            temp.append(x[5])
        dict2[i] = temp
    return(dict2)

#temp dict
y = newdict2(groups)

newdict = {}

#use the .count method to count each item
for i,j in y.items():
    temp = []
    for x in set(j):
        temp.append([x, j.count(x)])
    newdict[i] = temp
{% endhighlight %}

4\. Create a new dict, which contains each customer as a key, with a list of day
shopped, TripType, and summed ScanCount (total items bought).

**In [11]:**

{% highlight python %}
#add all the totalitems,
#take the first for each of day and trip type
dictnew = {}
for i,j in groups.items():
    totalitems = 0
    for k in j:
        totalitems += k[4]
    dictnew[i] = [j[0][0], j[0][2], totalitems]
{% endhighlight %}

5\. Create a Visit Class, which contains the total data we have about each
visit, with the minimum amount of repetition.

**In [12]:**

{% highlight python %}
#im making it easy to init
#there are a wide way of making this class
class Visit:
    def __init__(self, val):
        self.triptype = val[0][0]
        self.number = val[0][1]
        self.day = val[0][2]
        self.upcs = []
        self.scans = []
        self.departments = []
        self.fineline = []
        for i in val:
            self.upcs.append(i[3])
            self.scans.append(i[4])
            self.departments.append(i[5])
            self.fineline.append(i[6])
{% endhighlight %}

6\. Add methods into the class which will describe total items scanned, most
common DepartmentDescription and most common FinelineNumber

**In [13]:**

{% highlight python %}
class Visit:
    def __init__(self, val):
        self.triptype = val[0][0]
        self.number = val[0][1]
        self.day = val[0][2]
        self.upcs = []
        self.scans = []
        self.departments = []
        self.fineline = []
        for i in val:
            self.upcs.append(i[3])
            self.scans.append(i[4])
            self.departments.append(i[5])
            self.fineline.append(i[6])

    def totalscans(self):
        return sum(self.scans)

    def commondepartment(self):
        #we use max, with a key, which is the function we run on each item inthe first argument
        #this would be like running:
        #for i in set(self.departments):
        #       self.departments.count
        #then taking the max value
        return max(set(self.departments), key = self.departments.count)

    def commonfineline(self):
        return max(set(self.fineline), key = self.fineline.count)

{% endhighlight %}

7\. Turn the current dict into a dict of {VisitNumber : Vist} using the new
class.

**In [14]:**

{% highlight python %}
#depending on the init method
newdict = {}
for i,j in groups.items():
    newdict[i] = Visit(j)
{% endhighlight %}

8\. Create a method in your class to calculate the similarity of a visit to
another visit based on DepartmentDescription (you could use total number of
shared categories, or Cosine Similarity)

**In [15]:**

{% highlight python %}
class Visit:
    def __init__(self, val):
        self.triptype = val[0][0]
        self.number = val[0][1]
        self.day = val[0][2]
        self.upcs = []
        self.scans = []
        self.departments = []
        self.fineline = []
        for i in val:
            self.upcs.append(i[3])
            self.scans.append(i[4])
            self.departments.append(i[5])
            self.fineline.append(i[6])

    def totalscans(self):
        return sum(self.scans)

    def commondepartment(self):
        return max(set(self.departments), key = self.departments.count)

    def commonfineline(self):
        return max(set(self.fineline), key = self.fineline.count)

    def similarity(self,b):
        #this is for speed - if we share no elements, we can give a 0
        if set(self.departments).isdisjoint(b.departments):
            return 0
        acount = []
        bcount = []
        for i in set(self.departments).union(b.departments):
            acount.append(self.departments.count(i))
            bcount.append(b.departments.count(i))
        #we could do the cosine part much faster with numpy!
        #taken from https://stackoverflow.com/questions/18424228/cosine-similarity-between-2-number-lists
        sumxx, sumxy, sumyy = 0, 0, 0
        for i in range(len(acount)):
            x = acount[i]; y = bcount[i]
            sumxx += x*x
            sumyy += y*y
            sumxy += x*y
            #I am scaling by the number of categories, may or may not be sensible!
        return sumxy/((sumxx*sumyy)**0.5) * len(acount)
{% endhighlight %}

9\. Compare every basket to every other basket using your similarity score -
Which baskets are the two most similar? How did you score this?

**In [19]:**

{% highlight python %}
#need to remake the dict with the new attribute
newdict = {}
for i,j in groups.items():
    newdict[i] = Visit(j)

#create a list to hold the best result
foundmax = [[0,0],0]

#get a list of values
data = list(newdict.values())
keys = list(newdict.keys())
length = len(data)


for i in range(length):
    for j in range(i+1, length):
        if data[i].similarity(data[j]) > foundmax[1]:
            foundmax = [[i,j],data[i].similarity(data[j])]
#takes around 10 minutes on my computer, not fast, but not toooo slow...
{% endhighlight %}

**In [23]:**

{% highlight python %}
print(foundmax)
#groups[keys[foundmax[0][0]]]
#groups[keys[foundmax[0][1]]]
{% endhighlight %}

    [[5564, 7008], 23.421313765753897]


10\. Optional, open ended question - Can you get an idea how the TripType
categories are determined? Hint, tally TripTypes against categories, fineline
categories, day of the week, number of items.

Include your code, and a quick description. They used a machine learning
algorithm, so don't worry about complete accuracy, a qualatative explanation is
perfect.

**In [None]:**

{% highlight python %}
#some great answers here!!
{% endhighlight %}
