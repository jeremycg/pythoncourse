---
layout: post
title: The matplotlib Plotting System - Part 1
author: Jeremy
tags:
comments: true
---

## What is matplotlib?

matplotlib is a python 2D plotting library which produces publication quality figures in a variety of hardcopy formats and interactive environments across platforms. matplotlib can be used in python scripts, the python and ipython shell, web application servers, and six graphical user interface toolkits.
- Web site: http://matplotlib.org/ (better documentation)

---


## Plotting Systems in Python: Base

- No built in support

---


## Plotting Systems in Python: ggplot


- A semi-abandoned port of ggplot from R is available at http://ggplot.yhathq.com/

---


## The Basics: `plot(y,x)`

- plots a given y for a given x

---

## Example Dataset




    %pylab inline
    import pandas as pd

    mpg=pd.read_csv('https://vincentarelbundock.github.io/Rdatasets/csv/ggplot2/mpg.csv')
    mpg[0:6]


    Populating the interactive namespace from numpy and matplotlib


    WARNING: pylab import has clobbered these variables: ['ylim', 'xlim']
    `%matplotlib` prevents importing * from pylab and numpy





<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>manufacturer</th>
      <th>model</th>
      <th>displ</th>
      <th>year</th>
      <th>cyl</th>
      <th>trans</th>
      <th>drv</th>
      <th>cty</th>
      <th>hwy</th>
      <th>fl</th>
      <th>class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>audi</td>
      <td>a4</td>
      <td>1.8</td>
      <td>1999</td>
      <td>4</td>
      <td>auto(l5)</td>
      <td>f</td>
      <td>18</td>
      <td>29</td>
      <td>p</td>
      <td>compact</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>audi</td>
      <td>a4</td>
      <td>1.8</td>
      <td>1999</td>
      <td>4</td>
      <td>manual(m5)</td>
      <td>f</td>
      <td>21</td>
      <td>29</td>
      <td>p</td>
      <td>compact</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>audi</td>
      <td>a4</td>
      <td>2.0</td>
      <td>2008</td>
      <td>4</td>
      <td>manual(m6)</td>
      <td>f</td>
      <td>20</td>
      <td>31</td>
      <td>p</td>
      <td>compact</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>audi</td>
      <td>a4</td>
      <td>2.0</td>
      <td>2008</td>
      <td>4</td>
      <td>auto(av)</td>
      <td>f</td>
      <td>21</td>
      <td>30</td>
      <td>p</td>
      <td>compact</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>audi</td>
      <td>a4</td>
      <td>2.8</td>
      <td>1999</td>
      <td>6</td>
      <td>auto(l5)</td>
      <td>f</td>
      <td>16</td>
      <td>26</td>
      <td>p</td>
      <td>compact</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>audi</td>
      <td>a4</td>
      <td>2.8</td>
      <td>1999</td>
      <td>6</td>
      <td>manual(m5)</td>
      <td>f</td>
      <td>18</td>
      <td>26</td>
      <td>p</td>
      <td>compact</td>
    </tr>
  </tbody>
</table>
</div>



---

## ggplot2 “Hello, world!”



    plot(mpg["displ"], mpg["hwy"],"o")
    show()


![png](/assets/matplotlib/output_3_0.png)


---

## Modifying aesthetics




    groups = mpg.groupby('drv')
    for name, group in groups:
        plot(group["displ"], group["hwy"], marker='o', linestyle='')
    show()


![png](/assets/matplotlib/output_5_0.png)


---

## Adding a line




    plot(mpg["displ"], mpg["hwy"],"o")
    m,b=np.polyfit(mpg["displ"], mpg["hwy"], 1)
    plot(mpg["displ"], m*mpg["displ"] + b, '-')
    show()


![png](/assets/matplotlib/output_7_0.png)


---

## Histograms




    hist([mpg[mpg["drv"]=="f"]["hwy"],
              mpg[mpg["drv"]=="4"]["hwy"],
              mpg[mpg["drv"]=="r"]["hwy"]],stacked="TRUE")
    show()


![png](/assets/matplotlib/output_9_0.png)


---

## Facets




    ax1=subplot(1, 3, 1)
    ax1.set_xlim([0,7])
    ax1.set_ylim([0,45])
    plot(mpg[mpg["drv"]=="f"]["displ"],mpg[mpg["drv"]=="f"]["hwy"],"o")
    ax2=subplot(1, 3, 2)
    ax2.set_xlim([0,7])
    ax2.set_ylim([0,45])
    plot(mpg[mpg["drv"]=="4"]["displ"],mpg[mpg["drv"]=="4"]["hwy"],"o")
    ax3=subplot(1, 3, 3)
    ax3.set_xlim([0,7])
    ax3.set_ylim([0,45])
    plot(mpg[mpg["drv"]=="r"]["displ"],mpg[mpg["drv"]=="r"]["hwy"],"o")
    show()


![png](/assets/matplotlib/output_11_0.png)



    ax1=subplot(1, 3, 1)
    ax1.set_xlim([0,45])
    ax1.set_ylim([0,35])
    mpg[mpg["drv"]=="f"]["hwy"].hist()
    ax2=subplot(1, 3, 2)
    ax2.set_xlim([0,45])
    ax2.set_ylim([0,35])
    mpg[mpg["drv"]=="4"]["hwy"].hist()
    ax3=subplot(1, 3, 3)
    ax3.set_xlim([0,45])
    ax3.set_ylim([0,35])
    mpg[mpg["drv"]=="r"]["hwy"].hist()
    show()


![png](/assets/matplotlib/output_12_0.png)



## Resources

- The matplotlib website http://matplotlib.org/
- matplotlib gallery http://matplotlib.org/gallery.html
- matplotlib mailing list (http://sourceforge.net/mail/?group_id=80706), primarily for developers
