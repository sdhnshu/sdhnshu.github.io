+++
date = "2020-04-29"
title = "Feature Engineering with Random Forests (Part 1)"
showonlyimage = false
draft = false
image = "https://miro.medium.com/max/573/0*easpKibgbYJgCSG8.png"
weight = 2
+++

Learn how to leverage Random Forests to do Feature Engineering and more.
<!--more-->

- Originally published on [Medium](https://medium.com/@sdhnshu/feature-engineering-with-random-forests-part1-601c66dfb09b)
- Grab the code from [Github](https://github.com/sdhnshu/Random-forests-and-Feature-Engineering)
- Link to [Part 2](/showcase/random-forests-2)


### Introduction

Making an educated guess on what price should you sell a product is not everyone’s cup of tea. It needs a good amount of domain knowledge and at least some experience. And if we are giving this job to an algorithm, it makes it even more complicated.

> *An algorithm is computationally scalable but logically bound.*

Unless you set these bounds the right way, you might not get the best output from it. This is what we are going to look at in this post.

We’ll go through the first 7 lectures of the [Fast.ai machine learning](http://course18.fast.ai/lessonsml1/lesson1.html) course and learn how to predict the price of a bulldozer (`SalePrice`) using the [data available](https://www.kaggle.com/c/bluebook-for-bulldozers/data) about its manufacturing and usage. In this post, we'll cover the random forests algorithm and feature engineering in the [part 2](/showcase/random-forests-2).

![img](https://miro.medium.com/max/573/0*easpKibgbYJgCSG8.png)

### Why Random Forests?

Logically speaking, there are very few constraints in the Random Forests algorithm. Thus when we use it, we are introducing no biases from our end. And because it is simple, it is highly interpretable, unlike the black box algorithms that lack this feature.

> *It takes the simplicity of a decision tree but generalizes it over a forest over the dataset to give us an exploratory tool that we can use as a flashlight in the dark.*

If you want to follow along, the jupyter notebook used in this post is available at the [github repo](https://github.com/sdhnshu/Random-forests-and-Feature-Engineering) along with the Fast.ai library (need a local copy of v0.7, because it has been upgraded with major changes) and an environment file to set up an Anaconda environment. Now that you have everything ready, let us jump right in.

----

### Preprocessing

If you are new to Kaggle competitions, here are some [handy first steps](https://www.kaggle.com/getting-started/44997). If not, make sure you have gone through the [overview](https://www.kaggle.com/c/bluebook-for-bulldozers/overview) and start looking at the data.

![img](https://miro.medium.com/max/573/0*xuijDvckOqpBhTqn.png)
![img](https://miro.medium.com/max/573/0*s-G2Z18Bb5LGTYNd.png)

By eyeballing, we can see that there are:

- Numerical features — measurement or count (`SalePrice`, `BladeWidth`)
- Categorical features — ordinal and nominal (`TrackType`, `EnclosureType`, `ProductSize`, `UsageBand`)
- DateTime features (`SaleDate`)
- Ids (`MachineId`, `ModelId`)

Download the [data](https://www.kaggle.com/c/bluebook-for-bulldozers/data) locally in a folder ‘bluebook-for-bulldozers’ and extract the Train.zip.

Then let us look into some preprocessing techniques like:

- Log transform
- Extracting data from DateTime
- Handling categorical features
- Imputing missing values

#### i. Log Transform

If you see the [Evaluation metrics](https://www.kaggle.com/c/bluebook-for-bulldozers/overview/evaluation), they want the Root Mean Squared Log Error.

> *RMSE: standard deviation of our model’s prediction errors*

```python
def rmse(actual, pred):
    return np.sqrt(np.mean((actual-pred)**2))
```

This is common for a feature like a price. They are more interested if you missed by 10% than if you missed by $10. If it is $10,000 item and you are off by a $1000 or if it is a $100 item and you are off by $10, they would be considered as equivalent issues.

```python
df_raw = pd.read_csv('bluebook-for-bulldozers/Train.csv',
                     parse_dates=['saledate'])
df_raw.SalePrice = np.log(df_raw.SalePrice)
```

Taking a log brings the data closer to a normal distribution. So, if the prediction errors are normally distributed, around 95% of the prediction errors would be in ±2*RMSE. It also decreases the effect of any outliers in the data. RMSE is the metric we’ll mathematically optimize and try to bring it down to 0.

![img](https://miro.medium.com/max/338/0*x_XL8FNSnnBOagEe.png)

#### ii. Extracting Data from DateTime

If some of your features are DateTime varaiables, we can decompose it into some useful features like — was it a weekend, what year was it, etc. Such engineering is often neglected but is very useful to extract the temporal information from the data. At the end of the post, we’ll also see how to remove any kind of temporal dependency from the data. This helps the model to generalize across time and will help us predict the `SalePrice` for the next month.

```python
from fastai.structured import *
from fastai.imports import *add_datepart(df_raw, 'saledate')
```
[add_datepart:](https://github.com/fastai/fastai/blob/775dcd03f49f4f6385f19b503f954468b83afbea/old/fastai/structured.py#L70-L124) Converts a datetime into `Year`, `Month`, `Week`, `Day`, `Dayofweek`, `Dayofyear`, `Is_month_end`, `Is_month_start`, `Is_quarter_end`, `Is_quarter_start`, `Is_year_end`, `Is_year_start` (and `Hour`, `Minute`, `Second`)

![img](https://miro.medium.com/max/182/0*VBpJuWcoh_FNtBo4.png)
![img](https://miro.medium.com/max/573/0*4K_9QfdrE82tANuX.png)

#### iii. Handling Categorical Features

Pandas can handle categorical features by maintaining a number:string hash.

```python
train_cats(df_raw)
```

[train_cats:](https://github.com/fastai/fastai/blob/775dcd03f49f4f6385f19b503f954468b83afbea/old/fastai/structured.py#L128-L153) This function from the Fast.ai library converts any columns of strings to a column of categorical values, in-place. It also fills empty/Nan values with code=-1.

![img](https://miro.medium.com/max/247/0*LN88EvcZzCZucYTw.png)
![img](https://miro.medium.com/max/290/0*v51i5LqgXYGxfiTr.png)
![img](https://miro.medium.com/max/296/0*XFb1QCF0VIk_gdKd.png)

#### iv. Imputing Missing Values

It is hard to find a real-life dataset without having missing values generated due to miscellaneous reasons. You can neglect them by simply removing entire rows containing some missing features. But there is more to it. Exploring them can lead you to the issues spawning them from behind the scenes. Here’s Jeremy Howard, creator of the Fast.ai course and ex-president of Kaggle giving [an example](https://youtu.be/YSFG_W8JxBo?t=1h11m16s).

Instead of deleting entire rows and reducing the dataset size, we can fill them using their mean, median or mode. You can also use algorithms like k-NN and [MICE](https://datascienceplus.com/imputing-missing-data-with-r-mice-package/) to do so. There’s no best way to fill the missing values. A technique that works on a dataset with certain missing values may not work on another.

```python
df, y, nas = proc_df(df_raw, 'SalePrice')
```

[proc_df:](https://github.com/fastai/fastai/blob/775dcd03f49f4f6385f19b503f954468b83afbea/old/fastai/structured.py#L298-L392) This Fast.ai function fills `empty/Nan` values with the median of the feature and splits the dataframe into dependent y and independent df variables. It also adds 1 to the categorical codes so they start from 0 instead of -1 (pandas default). You can choose a random subset from `df_raw` and can also make features 1-hot encoded.

You should now have a clean matrix of integers as your df with no empty values. We can save the df and split it into a training and validation set.

> *We want our validation set the same properties of the test set. Thus, keep the size of the validation set the same as the test set.*
```python
# Pandas dataframes saved in Feather format
df.to_feather('tmp/bulldozers-clean')
df = pd.read_feather('tmp/bulldozers-clean')

# Splitting df and y into train and validation set
def split_vals(a,n):
    return a[:n].copy(), a[n:].copy()

n_valid = 12000  # Same size as test set
n_trn = len(df) - n_valid

X_train, X_valid = split_vals(df, n_trn)
y_train, y_valid = split_vals(y, n_trn)
```
- Training set size: 389,125
- Validation set size: 12,000

![img](https://miro.medium.com/max/325/0*dbY-LZKoAoQNe2Jo.png)

### Random Forest Interpretation

A set of [decision trees](https://en.wikipedia.org/wiki/Decision_tree) trained on a [bootstrapped dataset](https://en.wikipedia.org/wiki/Bootstrapping_(statistics)) (random sampling with replacement of the same size as the original dataset (389,125)) is called a random forest. Scikit-learn provides a `RandomForestRegressor` for predicting continuous floating-point numbers and a `RandomForestClassifier` for discrete numbers.

Let us train a random forest regressor and take a look at its predictions.

```python
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

m = RandomForestRegressor(n_jobs=-1)
m.fit(X_train, y_train)

m.predict(X_valid), m.score(X_valid, y_valid), np.mean(m.predict(X_valid)), np.std(m.predict(X_valid))

-------
Output:

> (array([9.17702, 9.16897, 9.25441, ..., 9.44273, 9.31048, 9.31048]),
> 0.8913676814741088,
> 10.012604788545604,
> 0.6932527067952876)
```
![img](https://miro.medium.com/max/282/0*WaUnLoDw6sK6v_Eq.png)


### Predictions and Feature Contributions

The main output of a model is its predictions. They lie around 10.01 and have a standard deviation of 0.69. If we want to know how much each feature contributes to the prediction, we can do that using a [Tree Interpreter](https://github.com/andosa/treeinterpreter).

```python
from treeinterpreter import treeinterpreter as ti
prediction, bias, contributions = ti.predict(m, row)
```
The bias (10.10561) mentioned above is the mean of y in the training set. That is what we'll add to the contributions[-0.00458, -0.00525, ... 0.00227, 0], to get the prediction (9.17702). Let us see them in a sorted manner.

```python
idxs = np.argsort(contributions[0])
[o for o in zip(X_valid.columns[idxs], X_valid.iloc[0][idxs], contributions[0][idxs])]
# Feature,    Value in the row,    Contribution

----
Output:

> [('ProductSize', 5, -0.8274220325463135),
>  ('saleElapsed', 1284595200, -0.06203234395696224),
>  ('fiProductClassDesc', 17, -0.05853101269602874),
>  ('fiModelDesc', 3232, -0.03869971271031911),
>  ...
>  ('fiModelSeries', 69, 0.00821771617784126),
>  ('fiBaseModel', 1111, 0.018305760128304628),
>  ('YearMade', 1999, 0.024869657037691794),
>  ('Coupler_System', 0, 0.10542811579794246)]
```

### R² Score

An equally important output of the model is the R² score.

> *R² score: the ratio between how good your model is vs. how good a naïve mean model is (both measured using RMSE).*
```python
def r2_score(actual, pred):
    return 1 - rmse(actual, pred)/ rmse(actual, np.mean(actual))
```
It is not the value we are optimizing for, but it lets you compare different models and get a sense of how a score of 0.8 compares to 0.9. A common misconception is that its value can only range from 0 to 1. But,

> *If you predicted infinity for every row, R² = 1 −∞. So when your R² is negative, it means your model is worse than predicting the mean.*

And our model gives an R² score of 89% on the validation set with a 69% variance in the predictions, out of the box. We’ll treat this as our baseline.

### Decision Tree Interpretation

Let us zoom into a single tree:

```python
m = RandomForestRegressor(n_estimators=1, max_depth=3,
                          bootstrap=False, n_jobs=-1)
m.fit(X_train, y_train)draw_tree(m.estimators_[0], X_train, precision=3)
```

![img](https://miro.medium.com/max/573/0*XoBMr4-jU-Iy7Wn6.png)


If `max_depth=3` were removed, the tree would continue to split until every node has the no. of samples=1. But now the tree is only 3 layers deep. At the root node of the tree, is the whole training set (samples=389125) and a value (avg y) of 10.106. Then it is split into 2 samples with size 348,393 and 40,732 at `Coupler_system <= 0.5`.

### How Do We Find the Best Split?

Let us consider that we are trying to split a node containing dogs and cats. If we split so that one node has all dogs and the other has all cats, we would see that the standard deviation of both is 0. Also the size of the child nodes matters.

Thus, we need to minimize this split score using brute force:

> *Std_dev(y) in left child node * no. of points in the left child node*
>
> plus
>
> *Std_dev(y) in right child node * no. of points in the right child node*


Which is mathematically equivalent to:

> *Mean squared error between each point in y and the avg y*

We can control the architecture of the forest using these parameters:

- Min samples leaf
- Max features
- No of estimators

#### i. Min Samples Leaf

You can control the negative depth of the tree using this argument. It stops dividing the nodes once there are 3 samples or less in the node. This gives a performance increase, as the depth of the tree has been reduced by one or two.

```python
m = RandomForestRegressor(n_estimators=40, n_jobs=-1,
                          min_samples_leaf=3)
```
It makes an individual tree’s predictions less accurate. As in each leaf node, it averages those 3 data points to give a prediction. But for the same reason, it generalizes better. The numbers that work well are 1, 3, 5, 10, 25, but it is relative to your overall dataset size.

#### ii. Max Features

Similar to sampling rows while bootstrapping, we can sample features before choosing to split on it.

```python
m = RandomForestRegressor(n_estimators=40, n_jobs=-1,
                          min_samples_leaf=3, max_features=0.5)
```
That’s important because, if there exists a super important feature, every tree would make that their first split. Thus your trees will have a high correlation. Which we do not want. There might exist other important interactions between features that might not be explored.

The above forest samples `0.5*no_of_features`, and then brute forces through them to find the best split. Other good values are `sqrt` and `log2`.

#### iii. No of Estimators

Boosting the no. of trees might increase the accuracy. But the graph flattens out. We can use this parameter as our last resort to stretch it out to its limits.

```python
from sklearn import metrics
preds = np.stack([tree.predict(X_valid) for tree in m.estimators_])
plt.plot([metrics.r2_score(y_valid, np.mean(preds[:i+1], axis=0)) for i in range(40)]);
```
![img](https://miro.medium.com/max/247/0*K52fjVdWcfT1lNz4.png)

### Bagging and Subsampling

As mentioned earlier, random forests by default are built on a [bootstrapped](https://en.wikipedia.org/wiki/Bootstrapping_(statistics)) sample. The sample is picked randomly with replacement but is the same size as the entire training set for each tree (389,125). And the collective predictions from the trees is averaged in the end. This is what [Leo Breiman](https://en.wikipedia.org/wiki/Leo_Breiman), the creator of random forests describes as bootstrap aggregation (bagging for short).

```python
m = RandomForestRegressor(n_estimators=40, n_jobs=-1)
%time m.fit(X_train, y_train)
print_score(m)

----
Output:

> CPU times: user 6min 50s, sys: 4.37 s, total: 6min 54s
> Wall time: 2min 15s
> RMSE Train:  7.856880831112418
> RMSE Valid:  23.892947209645595
>  R2  Train:  98.70986499747674
>  R2  Valid:  89.80499551721383
```
Training on so much duplicate data is unnecessary for experimentation. It raises the training time to more than 2 mins and is overfitting on our training set. And unlike 1999, when Breiman introduced the world to bagging, we have a lot of data to train our models on.

> *So instead of sampling rows with replacement for each tree during bagging, why not sample them without replacement?*

![img](https://miro.medium.com/max/247/0*IiHNFeVCYVTn0WFo.jpg)

This is called subsampling. It is like having a team of scientists, working on different datasets of a common problem. Because their datasets have nothing in common, the system generalizes better. And given enough scientists, the team will surely go through all the data.

```python
set_rf_samples(20000)
m = RandomForestRegressor(n_estimators=40, n_jobs=-1)
%time m.fit(X_train, y_train)
print_score(m)

----
Output:

> CPU times: user 40.6 s, sys: 890 ms, total: 41.5 s
> Wall time: 19.3 s
> RMSE Train:  22.767005816704206
> RMSE Valid:  26.17177986481096
>  R2  Train:  89.16705188450162
>  R2  Valid:  87.7675206461405
```

This function, [set_rf_samples(20000)](https://github.com/fastai/fastai/blob/775dcd03f49f4f6385f19b503f954468b83afbea/old/fastai/structured.py#L398-L409) gives each tree a different random sample of 20,000 rows to train on. It increases the variance in our predictions and trains faster as each tree gets a smaller dataset.

This function is not available in scikit-learn, and is a workaround made by Jeremy Howard as a part of the fast.ai library. You can reset to a normal bootstrapped model by using `reset_rf_samples()`.

### When to Use What?

According to Breiman, there are 2 things you are trying to balance while using random forests:

> *Each tree (aka estimators) is trained as well as possible.*
> *But across estimators, the correlation between them is as low as possible.*

Subsampling gives speed and an ability to generalize. Thus it can be used for any kind of feature engineering or experimentation. Bagging helps train the trees very well. We’ll use this after we freeze the architecture and features.

### Bonus Metric — OOB Score:

Whether we sample with (bootstrapping) or without replacement (subsampling), not all the rows in the training set will be used for each tree. This gives random forests a free validation set. Calculating the R² using these out of the bag (OOB) rows as a validation set gives us an idea of how well is the model generalizing over the training set. It is mostly used with bagging.


```python
reset_rf_samples()
m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3,
                      max_features=0.5, n_jobs=-1, oob_score=True)
%time m.fit(X_train, y_train)
print_score(m)

----
Output:

> CPU times: user 3min 13s, sys: 2.6 s, total: 3min 16s
> Wall time: 1min 6s
> RMSE Train:  11.906626005969184
> RMSE Valid:  22.80435877833463
>  R2  Train:  97.03713235875033
>  R2  Valid:  90.71282339508942
>    OOB    :  91.19405377806355
```
Be careful while using the OOB score with `set_rf_samples(20000)`. If the training set consists of a million rows and you give each tree around 20k rows, the validation set for an OOB score would be huge.

![img](https://miro.medium.com/max/304/0*Cj4l8QQ0nYREk3bC.jpg)

That’s enough detail about the random forests for now. In the [part 2](/showcase/random-forests-2) we’ll see how this algorithm helps us with feature engineering.

---

__*Continue reading [Part 2](/showcase/random-forests-2)*__

### Further Reading

- [Fast.ai ML course notes](https://medium.com/@hiromi_suenaga/machine-learning-1-lesson-1-84a1dc2b5236)
- [Log-transformation and its implications for data analysis](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4120293/)
- [6 Different Ways to Compensate for Missing Values In a Dataset](https://towardsdatascience.com/6-different-ways-to-compensate-for-missing-values-data-imputation-with-examples-6022d9ca0779)
- [How to Calculate Nonparametric Rank Correlation in Python](https://machinelearningmastery.com/how-to-calculate-nonparametric-rank-correlation-in-python/)
- [Data leakage in machine learning](https://machinelearningmastery.com/data-leakage-machine-learning/)