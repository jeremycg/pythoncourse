---
layout: post
title: "Lesson 05 - pandas part 2"
author: Jeremy
tags:
 - Applied Statistics stream
comments: true
---
## Lesson 05 - Pandas part 2

Welcome to lesson 6! In this lesson we will introduce pandas as our main way of
storing data. NumPy will be useful when we have a uniform multidimensional data
set, but for now pandas will be our default.

Again, an exhaustive run through of pandas is too much for this class - see the
book - [Python for Data
Analysis](http://shop.oreilly.com/product/0636920023784.do) or the [official
pandas website](http://pandas.pydata.org/pandas-docs/version/0.17.1/).

If you are familiar with [R](http://pandas.pydata.org/pandas-
docs/version/0.17.1/comparison_with_r.html),
[SAS](http://pandas.pydata.org/pandas-
docs/version/0.17.1/comparison_with_sas.html), and/or
[SQL](http://pandas.pydata.org/pandas-
docs/version/0.17.1/comparison_with_sql.html), click on the links to lead you to
the intro to pandas for users of each language.

Please download todays notebook [here](/pythoncourse/assests/applied/lesson 05
applied.ipynb).

### Data Import

Importing data is the most important first step to get our data in. Today we
will cover read_csv, before we finish the course we will talk about how to
connect to your netezza (and other SQL) databases

**In [1]:**

{% highlight python %}
from pandas import DataFrame, Series
import pandas as pd
import io
import numpy as np
{% endhighlight %}

We have a ton of ways of reading data into and writing data out of pandas. See
the [dataIO page](http://pandas.pydata.org/pandas-docs/stable/io.html) for more
details.

**In [3]:**

{% highlight python %}
#using a string as example
#we could refer to file names if we had the file saved
data = '''
date,A,B,C
20090101,a,1,2
20090102,b,3,4
20090103,c,4,5
'''
print(data)
{% endhighlight %}


    date,A,B,C
    20090101,a,1,2
    20090102,b,3,4
    20090103,c,4,5



**In [3]:**

{% highlight python %}
#by default, the index is arange(nrows)
pd.read_csv(io.StringIO(data))
{% endhighlight %}




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>20090101</td>
      <td>a</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20090102</td>
      <td>b</td>
      <td>3</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20090103</td>
      <td>c</td>
      <td>4</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>



**In [4]:**

{% highlight python %}
#we can specify the index:
pd.read_csv(io.StringIO(data), index_col=0)
#also index_col='date'
{% endhighlight %}




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>20090101</th>
      <td>a</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>20090102</th>
      <td>b</td>
      <td>3</td>
      <td>4</td>
    </tr>
    <tr>
      <th>20090103</th>
      <td>c</td>
      <td>4</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>



**In [40]:**

{% highlight python %}
#we can also use nested indices:
pd.read_csv(io.StringIO(data), index_col=['date','A'])
{% endhighlight %}




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>B</th>
      <th>C</th>
    </tr>
    <tr>
      <th>date</th>
      <th>A</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>20090101</th>
      <th>a</th>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>20090102</th>
      <th>b</th>
      <td>3</td>
      <td>4</td>
    </tr>
    <tr>
      <th>20090103</th>
      <th>c</th>
      <td>4</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>



**In [5]:**

{% highlight python %}
#we can specify data type (it will speed things up, or avoid conversion)
pd.read_csv(io.StringIO(data), index_col=['date'], dtype={'A' : str, 'B':np.int32, 'C':np.float64})
{% endhighlight %}




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>20090101</th>
      <td>a</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>20090102</th>
      <td>b</td>
      <td>3</td>
      <td>4</td>
    </tr>
    <tr>
      <th>20090103</th>
      <td>c</td>
      <td>4</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>



**In [6]:**

{% highlight python %}
#We can throw out names and use our own
pd.read_csv(io.StringIO(data), index_col=[0],
            dtype={'A' : str, 'B':np.int32, 'C':np.float64},
           names=["foo", 'bar', "baz"],
           header = 0)
{% endhighlight %}




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>foo</th>
      <th>bar</th>
      <th>baz</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>20090101</th>
      <td>a</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>20090102</th>
      <td>b</td>
      <td>3</td>
      <td>4</td>
    </tr>
    <tr>
      <th>20090103</th>
      <td>c</td>
      <td>4</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>



**In [57]:**

{% highlight python %}
#filter out some unneeded columns:
pd.read_csv(io.StringIO(data),
           names=['date', 'foo', 'bar', "baz"],
           header = 0,
           usecols = ['foo', 'baz'])
{% endhighlight %}




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>foo</th>
      <th>baz</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>b</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>c</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>



**In [5]:**

{% highlight python %}
#dates! more to come
dat = pd.read_csv(io.StringIO(data),
           parse_dates = True,
            index_col = [0]
           )
dat.index
{% endhighlight %}




    DatetimeIndex(['2009-01-01', '2009-01-02', '2009-01-03'], dtype='datetime64[ns]', name='date', freq=None)



### DataFrames

DataFrames are similar to a dict of a series - technically they are a 2d series
with some linking between levels.

Columns are arrays (must be one data type), and rows are similar to dicts.

However, the row/column mapping is not as strictly enforced as R.

**In [7]:**

{% highlight python %}
dat = pd.read_csv("http://jeremy.kiwi.nz/pythoncourse/assets/tests/r&d/test1data.csv")[1:20]
dat
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
    <tr>
      <th>5</th>
      <td>26</td>
      <td>8</td>
      <td>Friday</td>
      <td>2006618783</td>
      <td>2</td>
      <td>PAINT AND ACCESSORIES</td>
      <td>1017</td>
    </tr>
    <tr>
      <th>6</th>
      <td>26</td>
      <td>8</td>
      <td>Friday</td>
      <td>2006613743</td>
      <td>1</td>
      <td>PAINT AND ACCESSORIES</td>
      <td>1017</td>
    </tr>
    <tr>
      <th>7</th>
      <td>26</td>
      <td>8</td>
      <td>Friday</td>
      <td>7004802737</td>
      <td>1</td>
      <td>PAINT AND ACCESSORIES</td>
      <td>2802</td>
    </tr>
    <tr>
      <th>8</th>
      <td>26</td>
      <td>8</td>
      <td>Friday</td>
      <td>2238495318</td>
      <td>1</td>
      <td>PAINT AND ACCESSORIES</td>
      <td>4501</td>
    </tr>
    <tr>
      <th>9</th>
      <td>26</td>
      <td>8</td>
      <td>Friday</td>
      <td>2238400200</td>
      <td>-1</td>
      <td>PAINT AND ACCESSORIES</td>
      <td>3565</td>
    </tr>
    <tr>
      <th>10</th>
      <td>26</td>
      <td>8</td>
      <td>Friday</td>
      <td>5200010239</td>
      <td>1</td>
      <td>DSD GROCERY</td>
      <td>4606</td>
    </tr>
    <tr>
      <th>11</th>
      <td>26</td>
      <td>8</td>
      <td>Friday</td>
      <td>88679300501</td>
      <td>2</td>
      <td>PAINT AND ACCESSORIES</td>
      <td>3504</td>
    </tr>
    <tr>
      <th>12</th>
      <td>26</td>
      <td>8</td>
      <td>Friday</td>
      <td>22006000000</td>
      <td>1</td>
      <td>MEAT - FRESH &amp; FROZEN</td>
      <td>6009</td>
    </tr>
    <tr>
      <th>13</th>
      <td>26</td>
      <td>8</td>
      <td>Friday</td>
      <td>2236760452</td>
      <td>1</td>
      <td>PAINT AND ACCESSORIES</td>
      <td>7</td>
    </tr>
    <tr>
      <th>14</th>
      <td>26</td>
      <td>8</td>
      <td>Friday</td>
      <td>88679300501</td>
      <td>-1</td>
      <td>PAINT AND ACCESSORIES</td>
      <td>3504</td>
    </tr>
    <tr>
      <th>15</th>
      <td>26</td>
      <td>8</td>
      <td>Friday</td>
      <td>2238400200</td>
      <td>2</td>
      <td>PAINT AND ACCESSORIES</td>
      <td>3565</td>
    </tr>
    <tr>
      <th>16</th>
      <td>26</td>
      <td>8</td>
      <td>Friday</td>
      <td>3019294203</td>
      <td>1</td>
      <td>PAINT AND ACCESSORIES</td>
      <td>2801</td>
    </tr>
    <tr>
      <th>17</th>
      <td>26</td>
      <td>8</td>
      <td>Friday</td>
      <td>72450408840</td>
      <td>1</td>
      <td>PAINT AND ACCESSORIES</td>
      <td>1028</td>
    </tr>
    <tr>
      <th>18</th>
      <td>26</td>
      <td>8</td>
      <td>Friday</td>
      <td>25541500000</td>
      <td>2</td>
      <td>DAIRY</td>
      <td>1305</td>
    </tr>
    <tr>
      <th>19</th>
      <td>26</td>
      <td>8</td>
      <td>Friday</td>
      <td>2310010776</td>
      <td>1</td>
      <td>PETS AND SUPPLIES</td>
      <td>3300</td>
    </tr>
  </tbody>
</table>
</div>



**In [12]:**

{% highlight python %}
# get the column names
dat.columns
#get the first five rows
dat.head()
#pick out specific columns
DataFrame(dat,columns=['TripType','VisitNumber'])
#same as
dat[['TripType','VisitNumber']]
#get one specific column
dat.TripType
#get one specific column
dat['TripType']
#use ix (index) to get the 10th row
dat.ix[10]
#add a new column
dat['foo']="spam"
#using other columns:
dat['foo'] = dat['VisitNumber'] + dat['ScanCount']
#add a new column with specific values
dat['foo']=Series(['spam', 'more spam'],index=[4,10])
{% endhighlight %}

**In [13]:**

{% highlight python %}
#delete a column
del dat['foo']
#'http://pandas.pydata.org/pandas-docs/dev/generated/pandas.DataFrame.html'
{% endhighlight %}

**In [15]:**

{% highlight python %}
#recall, indexes are immutable?
#how to reindex?
dat = dat.reindex(np.arange(5))
dat
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
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
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



**In [16]:**

{% highlight python %}
dat.reindex(np.arange(7),fill_value=0)
dat.reindex(np.arange(10),method='ffill')
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
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
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
    <tr>
      <th>5</th>
      <td>26</td>
      <td>8</td>
      <td>Friday</td>
      <td>2006613744</td>
      <td>2</td>
      <td>PAINT AND ACCESSORIES</td>
      <td>1017</td>
    </tr>
    <tr>
      <th>6</th>
      <td>26</td>
      <td>8</td>
      <td>Friday</td>
      <td>2006613744</td>
      <td>2</td>
      <td>PAINT AND ACCESSORIES</td>
      <td>1017</td>
    </tr>
    <tr>
      <th>7</th>
      <td>26</td>
      <td>8</td>
      <td>Friday</td>
      <td>2006613744</td>
      <td>2</td>
      <td>PAINT AND ACCESSORIES</td>
      <td>1017</td>
    </tr>
    <tr>
      <th>8</th>
      <td>26</td>
      <td>8</td>
      <td>Friday</td>
      <td>2006613744</td>
      <td>2</td>
      <td>PAINT AND ACCESSORIES</td>
      <td>1017</td>
    </tr>
    <tr>
      <th>9</th>
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



**In [18]:**

{% highlight python %}
dat.drop(1)
#dat.drop('foo', axis = 1)
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
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
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



**In [19]:**

{% highlight python %}
#getting data
dat[['TripType','Upc']]
dat['ScanCount']>1
dat[dat['ScanCount']>1]
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



### Quick Review

I mentioned in the previous lecture we can use all our base and NumPy methods on
pandas DataFrames: Here is a quick review taken from the SQL lesson:

**In [8]:**

{% highlight python %}
tips = pd.read_csv('https://raw.github.com/pydata/pandas/master/pandas/tests/data/tips.csv')
tips.head()
{% endhighlight %}




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>total_bill</th>
      <th>tip</th>
      <th>sex</th>
      <th>smoker</th>
      <th>day</th>
      <th>time</th>
      <th>size</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>16.99</td>
      <td>1.01</td>
      <td>Female</td>
      <td>No</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10.34</td>
      <td>1.66</td>
      <td>Male</td>
      <td>No</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>21.01</td>
      <td>3.50</td>
      <td>Male</td>
      <td>No</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>23.68</td>
      <td>3.31</td>
      <td>Male</td>
      <td>No</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>24.59</td>
      <td>3.61</td>
      <td>Female</td>
      <td>No</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>



SQL select:
```
SELECT total_bill, tip, smoker, time
FROM tips
LIMIT 5;
```

**In [9]:**

{% highlight python %}
tips[['total_bill', 'tip', 'smoker', 'time']].head(5)
{% endhighlight %}




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>total_bill</th>
      <th>tip</th>
      <th>smoker</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>16.99</td>
      <td>1.01</td>
      <td>No</td>
      <td>Dinner</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10.34</td>
      <td>1.66</td>
      <td>No</td>
      <td>Dinner</td>
    </tr>
    <tr>
      <th>2</th>
      <td>21.01</td>
      <td>3.50</td>
      <td>No</td>
      <td>Dinner</td>
    </tr>
    <tr>
      <th>3</th>
      <td>23.68</td>
      <td>3.31</td>
      <td>No</td>
      <td>Dinner</td>
    </tr>
    <tr>
      <th>4</th>
      <td>24.59</td>
      <td>3.61</td>
      <td>No</td>
      <td>Dinner</td>
    </tr>
  </tbody>
</table>
</div>



SQL where:
```
SELECT *
FROM tips
WHERE time = 'Dinner' AND tip > 5.00;
LIMIT 5;
```

**In [11]:**

{% highlight python %}
tips[(tips['time'] == 'Dinner') & (tips['tip'] > 5.00)].head(5)
{% endhighlight %}




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>total_bill</th>
      <th>tip</th>
      <th>sex</th>
      <th>smoker</th>
      <th>day</th>
      <th>time</th>
      <th>size</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>23</th>
      <td>39.42</td>
      <td>7.58</td>
      <td>Male</td>
      <td>No</td>
      <td>Sat</td>
      <td>Dinner</td>
      <td>4</td>
    </tr>
    <tr>
      <th>44</th>
      <td>30.40</td>
      <td>5.60</td>
      <td>Male</td>
      <td>No</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>4</td>
    </tr>
    <tr>
      <th>47</th>
      <td>32.40</td>
      <td>6.00</td>
      <td>Male</td>
      <td>No</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>4</td>
    </tr>
    <tr>
      <th>52</th>
      <td>34.81</td>
      <td>5.20</td>
      <td>Female</td>
      <td>No</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>4</td>
    </tr>
    <tr>
      <th>59</th>
      <td>48.27</td>
      <td>6.73</td>
      <td>Male</td>
      <td>No</td>
      <td>Sat</td>
      <td>Dinner</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>



### Groupby

Grouping and summarising data allows us to carry out the key data analysis steps
of [split, apply,
combine](https://www.jstatsoft.org/article/view/v040i01/v40i01.pdf). The journal
article by Hadley Wickham was one of the first formalisations of the split apply
combine paradigm, and we can of course do it in Python.

* Splitting the data into groups based on some criteria
* Applying a function to each group independently
* Combining the results into a data structure

Let's continue on with our analysis of the tips data:

**In [12]:**

{% highlight python %}
tipsgroups = tips.groupby('sex')
tipsgroups
{% endhighlight %}




    <pandas.core.groupby.DataFrameGroupBy object at 0x7f5f60520128>



We now have a new data type, the groupby object.
We can access the attribute, groups. This is a dict, with each level as it's own
entry and the indices of the original data frame:

**In [13]:**

{% highlight python %}
for i,j in tipsgroups.groups.items():
    print(i)
{% endhighlight %}

    Female
    Male


We can do grouping on any axis, or with a custom function (this example is
pathological):

**In [14]:**

{% highlight python %}
def myfun(index):
    if len(index) >= 5:
        return 1
    else:
        return 0

group2 = tips.groupby(myfun, axis = 1)
group2.groups
{% endhighlight %}




    {0: ['tip', 'sex', 'day', 'time', 'size'], 1: ['total_bill', 'smoker']}



We can use tab completion to see all out methods and attributes:

**In [43]:**

{% highlight python %}
from matplotlib import pyplot as plt
%matplotlib inline
tipsgroups.mean()
#tipsgroups.boxplot();
{% endhighlight %}




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>total_bill</th>
      <th>tip</th>
      <th>size</th>
    </tr>
    <tr>
      <th>sex</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Female</th>
      <td>18.056897</td>
      <td>2.833448</td>
      <td>2.459770</td>
    </tr>
    <tr>
      <th>Male</th>
      <td>20.744076</td>
      <td>3.089618</td>
      <td>2.630573</td>
    </tr>
  </tbody>
</table>
</div>



We can iterate through groups:

**In [36]:**

{% highlight python %}
for name, group in tipsgroups:
    print(name)
    print(group.head(5))
{% endhighlight %}

    Female
        total_bill   tip     sex smoker  day    time  size
    0        16.99  1.01  Female     No  Sun  Dinner     2
    4        24.59  3.61  Female     No  Sun  Dinner     4
    11       35.26  5.00  Female     No  Sun  Dinner     4
    14       14.83  3.02  Female     No  Sun  Dinner     2
    16       10.33  1.67  Female     No  Sun  Dinner     3
    Male
       total_bill   tip   sex smoker  day    time  size
    1       10.34  1.66  Male     No  Sun  Dinner     3
    2       21.01  3.50  Male     No  Sun  Dinner     3
    3       23.68  3.31  Male     No  Sun  Dinner     2
    5       25.29  4.71  Male     No  Sun  Dinner     4
    6        8.77  2.00  Male     No  Sun  Dinner     2


To apply, we can use .aggregate:

**In [15]:**

{% highlight python %}
tipsgroups.aggregate(np.mean)
#selecting columns:
tipsgroups['tip'].aggregate(np.mean)
{% endhighlight %}




    sex
    Female    2.833448
    Male      3.089618
    Name: tip, dtype: float64



**In [16]:**

{% highlight python %}
#.agg is short for agg
tipsgroups.agg([np.mean, np.sum, np.std])
{% endhighlight %}




<div>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="3" halign="left">total_bill</th>
      <th colspan="3" halign="left">tip</th>
      <th colspan="3" halign="left">size</th>
    </tr>
    <tr>
      <th></th>
      <th>mean</th>
      <th>sum</th>
      <th>std</th>
      <th>mean</th>
      <th>sum</th>
      <th>std</th>
      <th>mean</th>
      <th>sum</th>
      <th>std</th>
    </tr>
    <tr>
      <th>sex</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Female</th>
      <td>18.056897</td>
      <td>1570.95</td>
      <td>8.009209</td>
      <td>2.833448</td>
      <td>246.51</td>
      <td>1.159495</td>
      <td>2.459770</td>
      <td>214</td>
      <td>0.937644</td>
    </tr>
    <tr>
      <th>Male</th>
      <td>20.744076</td>
      <td>3256.82</td>
      <td>9.246469</td>
      <td>3.089618</td>
      <td>485.07</td>
      <td>1.489102</td>
      <td>2.630573</td>
      <td>413</td>
      <td>0.955997</td>
    </tr>
  </tbody>
</table>
</div>



**In [17]:**

{% highlight python %}
#we can also use a dict, to do different things to different rows:
tipsgroups.agg({'tip': [np.mean, np.sum], 'size':lambda x: max(x)})
{% endhighlight %}




<div>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="2" halign="left">tip</th>
      <th>size</th>
    </tr>
    <tr>
      <th></th>
      <th>mean</th>
      <th>sum</th>
      <th>&lt;lambda&gt;</th>
    </tr>
    <tr>
      <th>sex</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Female</th>
      <td>2.833448</td>
      <td>246.51</td>
      <td>6</td>
    </tr>
    <tr>
      <th>Male</th>
      <td>3.089618</td>
      <td>485.07</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>



We can also filter, transform, plot, count etc etc. Take a look in the help for
more details!

### Joins

We can use a variety of joins in pandas, the most basic using the concat
function:

**In [18]:**

{% highlight python %}
df1 = DataFrame({'A': ['A0', 'A1', 'A2', 'A3'],
                    'B': ['B0', 'B1', 'B2', 'B3'],
                    'C': ['C0', 'C1', 'C2', 'C3'],
                    'D': ['D0', 'D1', 'D2', 'D3']},
                     index=[0, 1, 2, 3])
df2 = DataFrame({'A': ['A4', 'A5', 'A6', 'A7'],
                    'B': ['B4', 'B5', 'B6', 'B7'],
                    'C': ['C4', 'C5', 'C6', 'C7'],
                    'D': ['D4', 'D5', 'D6', 'D7']},
                     index=[4, 5, 6, 7])
{% endhighlight %}

**In [19]:**

{% highlight python %}
#joins on index
pd.concat([df1, df2])
{% endhighlight %}




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A0</td>
      <td>B0</td>
      <td>C0</td>
      <td>D0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>A1</td>
      <td>B1</td>
      <td>C1</td>
      <td>D1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>A2</td>
      <td>B2</td>
      <td>C2</td>
      <td>D2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>A3</td>
      <td>B3</td>
      <td>C3</td>
      <td>D3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>A4</td>
      <td>B4</td>
      <td>C4</td>
      <td>D4</td>
    </tr>
    <tr>
      <th>5</th>
      <td>A5</td>
      <td>B5</td>
      <td>C5</td>
      <td>D5</td>
    </tr>
    <tr>
      <th>6</th>
      <td>A6</td>
      <td>B6</td>
      <td>C6</td>
      <td>D6</td>
    </tr>
    <tr>
      <th>7</th>
      <td>A7</td>
      <td>B7</td>
      <td>C7</td>
      <td>D7</td>
    </tr>
  </tbody>
</table>
</div>



**In [20]:**

{% highlight python %}
#joins on index!
pd.concat([df1, df2], axis = 1)
{% endhighlight %}




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A0</td>
      <td>B0</td>
      <td>C0</td>
      <td>D0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>A1</td>
      <td>B1</td>
      <td>C1</td>
      <td>D1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>A2</td>
      <td>B2</td>
      <td>C2</td>
      <td>D2</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>A3</td>
      <td>B3</td>
      <td>C3</td>
      <td>D3</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>A4</td>
      <td>B4</td>
      <td>C4</td>
      <td>D4</td>
    </tr>
    <tr>
      <th>5</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>A5</td>
      <td>B5</td>
      <td>C5</td>
      <td>D5</td>
    </tr>
    <tr>
      <th>6</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>A6</td>
      <td>B6</td>
      <td>C6</td>
      <td>D6</td>
    </tr>
    <tr>
      <th>7</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>A7</td>
      <td>B7</td>
      <td>C7</td>
      <td>D7</td>
    </tr>
  </tbody>
</table>
</div>



**In [21]:**

{% highlight python %}
#we can ignore the index!
df1.append(df2, ignore_index=True)
{% endhighlight %}




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A0</td>
      <td>B0</td>
      <td>C0</td>
      <td>D0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>A1</td>
      <td>B1</td>
      <td>C1</td>
      <td>D1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>A2</td>
      <td>B2</td>
      <td>C2</td>
      <td>D2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>A3</td>
      <td>B3</td>
      <td>C3</td>
      <td>D3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>A4</td>
      <td>B4</td>
      <td>C4</td>
      <td>D4</td>
    </tr>
    <tr>
      <th>5</th>
      <td>A5</td>
      <td>B5</td>
      <td>C5</td>
      <td>D5</td>
    </tr>
    <tr>
      <th>6</th>
      <td>A6</td>
      <td>B6</td>
      <td>C6</td>
      <td>D6</td>
    </tr>
    <tr>
      <th>7</th>
      <td>A7</td>
      <td>B7</td>
      <td>C7</td>
      <td>D7</td>
    </tr>
  </tbody>
</table>
</div>



For more control, we might want to explicitly use merge!

We have the standard joins - inner, outer, left, right, full and union:

**In [78]:**

{% highlight python %}
df1 = DataFrame({'key': ['A', 'B', 'C', 'D'],
                 'value': np.random.randn(4)})
df2 = pd.DataFrame({'key': ['B', 'D', 'D', 'E'],
                    'value': np.random.randn(4)})
#SQL:
#SELECT *
#FROM df1
#INNER JOIN df2
#  ON df1.key = df2.key;
pd.merge(df1, df2, on='key')
#SQL:
#SELECT *
#FROM df1
#LEFT OUTER JOIN df2
#  ON df1.key = df2.key;
pd.merge(df1, df2, on='key', how='left')
#SQL:
#SELECT *
#FROM df1
#RIGHT OUTER JOIN df2
#  ON df1.key = df2.key;
pd.merge(df1, df2, on='key', how='right')
#SQL:
#SELECT *
#FROM df1
#FULL OUTER JOIN df2
#  ON df1.key = df2.key;
pd.merge(df1, df2, on='key', how='outer')
{% endhighlight %}




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>key</th>
      <th>value_x</th>
      <th>value_y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A</td>
      <td>-2.275516</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>B</td>
      <td>0.050553</td>
      <td>2.301355</td>
    </tr>
    <tr>
      <th>2</th>
      <td>C</td>
      <td>0.943035</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>D</td>
      <td>-0.237517</td>
      <td>0.449717</td>
    </tr>
    <tr>
      <th>4</th>
      <td>D</td>
      <td>-0.237517</td>
      <td>-1.692712</td>
    </tr>
    <tr>
      <th>5</th>
      <td>E</td>
      <td>NaN</td>
      <td>-1.079463</td>
    </tr>
  </tbody>
</table>
</div>



### Example

From here, we will look at a worked example of data analysis using pandas. In
the first lesson, we looked at the example of the tennis fixing scandal, and
briefly ran through it. Now we have the skills and knowledge to walk through it,
and assess the analysis.

Here's the link to the [original
article](http://www.buzzfeed.com/heidiblake/the-tennis-racket) and the [notebook
on github](https://github.com/BuzzFeedNews/2016-01-tennis-betting-analysis)
