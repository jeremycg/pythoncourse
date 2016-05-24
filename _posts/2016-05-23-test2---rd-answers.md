---
layout: post
title: "Test 02 - Answers"
author: Jeremy
tags:
 - R&D stream
comments: true
---

### Test 2: Machine learning

Here are the answers, see [the notebook here](/pythoncourse/assets/tests/test2 - rd answers.ipynb).


In this test we will use the entire dataset from the walmart kaggle challenge,
do some feature engineering and data munging, then fit a random forest model to
our data.

Again, the data is a csv file which contains one line for each scan on their
system, with a Upc, Weekday, ScanCount, DepartmentDescription and
FinelineNumber.

The VisitNumber column groups our data into baskets - Every unique VisitNumber
is a unique basket, with a basket possibly containing multiple scans.

The label is the TripType column, which is Walmarts proprietary way of
clustering their visits into categories. We wish to match their algorithm, and
predict the category of some of our held out data.

This time we will use the full dataset - we have about 650,000 lines, in about
100,000 baskets. Just as a heads up, using 100 classifiers, my answer to the
test takes less than 3 minutes to run - no need for hours and hours of
computation.

If you do need to run this script multiple times, download the dataset from the
website rather than redownloading each time, as it's around 30 mb.

Please answer the questions in the cells below them - feel free to answer out of
order, but leave comments saying where you carried out the answer. I am working
more or less step by step through my answer - Feel free to add on extra
predictors if you can think of them.

1\. Import the modules you will use for the rest of the test:

**In [1]:**

{% highlight python %}
import pandas as pd
import numpy as np
from sklearn import ensemble
from sklearn.cross_validation import train_test_split
import operator
from sklearn import pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
{% endhighlight %}

2\. Read in the data, and check its head. The data is available on the website
at: http://jeremy.kiwi.nz/pythoncourse/assets/tests/test2data.csv

**In [2]:**

{% highlight python %}
dat = pd.read_csv("c:/users/jeremy/desktop/kaglewalmart/data/train.csv")
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
      <td>6.811315e+10</td>
      <td>-1</td>
      <td>FINANCIAL SERVICES</td>
      <td>1000.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>30</td>
      <td>7</td>
      <td>Friday</td>
      <td>6.053882e+10</td>
      <td>1</td>
      <td>SHOES</td>
      <td>8931.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>30</td>
      <td>7</td>
      <td>Friday</td>
      <td>7.410811e+09</td>
      <td>1</td>
      <td>PERSONAL CARE</td>
      <td>4504.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>26</td>
      <td>8</td>
      <td>Friday</td>
      <td>2.238404e+09</td>
      <td>2</td>
      <td>PAINT AND ACCESSORIES</td>
      <td>3565.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>26</td>
      <td>8</td>
      <td>Friday</td>
      <td>2.006614e+09</td>
      <td>2</td>
      <td>PAINT AND ACCESSORIES</td>
      <td>1017.0</td>
    </tr>
  </tbody>
</table>
</div>



3\. Fix the Weekday and DepartmentDescription into dummified data. For now they
can be seperate dataframes

**In [3]:**

{% highlight python %}
#now fix the categorical variables
weekdum = pd.get_dummies(dat['Weekday'])
weekdum.head()
departdum = pd.get_dummies(dat['DepartmentDescription'])
departdum.head()
{% endhighlight %}




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>1-HR PHOTO</th>
      <th>ACCESSORIES</th>
      <th>AUTOMOTIVE</th>
      <th>BAKERY</th>
      <th>BATH AND SHOWER</th>
      <th>BEAUTY</th>
      <th>BEDDING</th>
      <th>BOOKS AND MAGAZINES</th>
      <th>BOYS WEAR</th>
      <th>BRAS &amp; SHAPEWEAR</th>
      <th>...</th>
      <th>SEAFOOD</th>
      <th>SEASONAL</th>
      <th>SERVICE DELI</th>
      <th>SHEER HOSIERY</th>
      <th>SHOES</th>
      <th>SLEEPWEAR/FOUNDATIONS</th>
      <th>SPORTING GOODS</th>
      <th>SWIMWEAR/OUTERWEAR</th>
      <th>TOYS</th>
      <th>WIRELESS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 68 columns</p>
</div>



4\. Drop the unneeded columns from the raw data - I suggest removing -
'Weekday', 'Upc', 'DepartmentDescription' and 'FinelineNumber' (we could dummify
Upc and FineLine, but this will massively increase our data size.)

**In [4]:**

{% highlight python %}
#drop the useless columns:
dat = dat.drop(['Weekday', 'Upc', 'DepartmentDescription', 'FinelineNumber'], axis = 1)
dat.head()
{% endhighlight %}




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>TripType</th>
      <th>VisitNumber</th>
      <th>ScanCount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>999</td>
      <td>5</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>30</td>
      <td>7</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>30</td>
      <td>7</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>26</td>
      <td>8</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>26</td>
      <td>8</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



5\. Correct the Dummified data for number bought in each ScanCount. I would
recommend something like:

`departdummies.multiply(dat['ScanCount'], axis = 0)`

**In [5]:**

{% highlight python %}
#correct for scancount
departdum = departdum.multiply(dat['ScanCount'], axis = 0)
departdum['ScanCount'] = dat['ScanCount']
dat = dat.drop(['ScanCount'], axis = 1)
{% endhighlight %}

6\. Concatenate back together the dummy variables with the main dataframe

**In [6]:**

{% highlight python %}
dat = pd.concat([dat, weekdum, departdum], axis = 1)
dat.head()
{% endhighlight %}




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>TripType</th>
      <th>VisitNumber</th>
      <th>Friday</th>
      <th>Monday</th>
      <th>Saturday</th>
      <th>Sunday</th>
      <th>Thursday</th>
      <th>Tuesday</th>
      <th>Wednesday</th>
      <th>1-HR PHOTO</th>
      <th>...</th>
      <th>SEASONAL</th>
      <th>SERVICE DELI</th>
      <th>SHEER HOSIERY</th>
      <th>SHOES</th>
      <th>SLEEPWEAR/FOUNDATIONS</th>
      <th>SPORTING GOODS</th>
      <th>SWIMWEAR/OUTERWEAR</th>
      <th>TOYS</th>
      <th>WIRELESS</th>
      <th>ScanCount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>999</td>
      <td>5</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-0.0</td>
      <td>...</td>
      <td>-0.0</td>
      <td>-0.0</td>
      <td>-0.0</td>
      <td>-0.0</td>
      <td>-0.0</td>
      <td>-0.0</td>
      <td>-0.0</td>
      <td>-0.0</td>
      <td>-0.0</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>30</td>
      <td>7</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>30</td>
      <td>7</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>26</td>
      <td>8</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>26</td>
      <td>8</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 78 columns</p>
</div>



7\. Summarise the data for each basket (hint, if you groupby columns, an .agg()
method will not apply to them)

**In [7]:**

{% highlight python %}
dat1 = dat.groupby(['TripType', 'VisitNumber']).agg(sum)
dat1.head()
{% endhighlight %}




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>Friday</th>
      <th>Monday</th>
      <th>Saturday</th>
      <th>Sunday</th>
      <th>Thursday</th>
      <th>Tuesday</th>
      <th>Wednesday</th>
      <th>1-HR PHOTO</th>
      <th>ACCESSORIES</th>
      <th>AUTOMOTIVE</th>
      <th>...</th>
      <th>SEASONAL</th>
      <th>SERVICE DELI</th>
      <th>SHEER HOSIERY</th>
      <th>SHOES</th>
      <th>SLEEPWEAR/FOUNDATIONS</th>
      <th>SPORTING GOODS</th>
      <th>SWIMWEAR/OUTERWEAR</th>
      <th>TOYS</th>
      <th>WIRELESS</th>
      <th>ScanCount</th>
    </tr>
    <tr>
      <th>TripType</th>
      <th>VisitNumber</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
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
      <th rowspan="5" valign="top">3</th>
      <th>106</th>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>121</th>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>153</th>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>162</th>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>164</th>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 76 columns</p>
</div>



8\. Use the reset_index() method to remove your groupings. As we did not cover
multiple indices in the lesson, my answer was

`dat1 = dat1.reset_index()`

**In [8]:**

{% highlight python %}
dat1 = dat1.reset_index()
{% endhighlight %}

9\. Split the data into training and testing sets: Use 0.25 of the data in the
test set.

**In [9]:**

{% highlight python %}
classes = dat1.TripType
dat1 = dat1.drop('TripType', axis = 1)
classes.head()

X_train, X_test, y_train, y_test = \
    train_test_split(dat1, classes, test_size = 0.25, random_state = 0)
{% endhighlight %}

10\. Construct at least two more features for the data - For Example, a 1/0
variable for if any product was returned (ScanCount < 0). You might want to do
this step before splitting the data as above

**In [10]:**

{% highlight python %}
#lots of good answers here!
{% endhighlight %}

11\. Plot the training data using matplotlib or seaborn. Choose at least 3
meaningful plots to present aspects of the data.

**In [11]:**

{% highlight python %}
#lots of good answers here
{% endhighlight %}

12\. Take out the TripType from our dataframe - we don't want our label as a
feature.

Make sure to save it somewhere though, as our model needs to be fit to these
labels.

**In [12]:**

{% highlight python %}
#see part 9
{% endhighlight %}

13\. Describe and fit a pipeline that carries out a kfold crossvalidation
randomforest model on the data. Include any relevant preprocessing steps such as
centering and scaling. The kfold might need to be outside the pipeline.

**In [13]:**

{% highlight python %}
pipe = pipeline.Pipeline(steps=[
                ('rf', RandomForestClassifier())
        #we don't really need preprocessing, the bootstrapped nature of RF takes care of it for use
        #most people chose a sensible option anyway!
        ])

#kfold, using the Kfold package
#most people did this
kf = KFold(len(X_train.index), n_folds=3)
for train_index, test_index in kf:
    pipe.fit(X_train.iloc[train_index], y_train.iloc[train_index])
    print(pipe.score(X_test, y_test))
{% endhighlight %}

    0.626322170659
    0.626991094945
    0.622684894853


**In [14]:**

{% highlight python %}
#I really wanted this, but explained it very poorly in the original question.
#apologies!
scores = cross_val_score(pipe, X_test, y_test, cv=5)
scores
{% endhighlight %}




    array([ 0.61009174,  0.61923639,  0.6049718 ,  0.62408377,  0.61556208])



14\. Modify your pipeline to include a grid search for a variable in the
RandomForest model. Try at least 3 values, choose a sensible variable to
optimise. (NB this question has changed from the initial version)

**In [15]:**

{% highlight python %}
estimators = {'rf__n_estimators':list(range(10, 30, 10))}
gs = GridSearchCV(pipe, param_grid=estimators)
gs.fit(X_train, y_train)
{% endhighlight %}




    GridSearchCV(cv=None, error_score='raise',
           estimator=Pipeline(steps=[('rf', RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=None, max_features='auto', max_leaf_nodes=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
                oob_score=False, random_state=None, verbose=0,
                warm_start=False))]),
           fit_params={}, iid=True, n_jobs=1,
           param_grid={'rf__n_estimators': [10, 20]}, pre_dispatch='2*n_jobs',
           refit=True, scoring=None, verbose=0)



**In [16]:**

{% highlight python %}
gs.best_estimator_
{% endhighlight %}




    Pipeline(steps=[('rf', RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=None, max_features='auto', max_leaf_nodes=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, n_estimators=20, n_jobs=1,
                oob_score=False, random_state=None, verbose=0,
                warm_start=False))])



15\. What is the score of the model on the training data?

**In [21]:**

{% highlight python %}
#we could just use our fitted CV here...
model = pipeline.Pipeline(steps=[
                ('rf', RandomForestClassifier(n_estimators = 20))])
model.fit(X_train, y_train)
model.score(X_train, y_train)
{% endhighlight %}




    0.99396557731168556



16\. What is the score of the model on the testing data?

**In [22]:**

{% highlight python %}
model.score(X_test, y_test)
{% endhighlight %}




    0.64045319620385466



17\. What is the most important variable? Can you explain the model?

**In [23]:**

{% highlight python %}
importances = model.named_steps['rf'].feature_importances_
max_index, max_value = max(enumerate(importances), key=operator.itemgetter(1))
print('Feature {x} was the most important, with an importance value of {y}'.format(x = dat1.columns[max_index], y = max_value))
{% endhighlight %}

    Feature ScanCount was the most important, with an importance value of 0.16822170595499625


Thanks for taking the Python Course!

Please save your notebook file as 'your name - test2.ipynb', and email it to
jeremycgray+pythoncourse@gmail.com by the 2nd of May.
