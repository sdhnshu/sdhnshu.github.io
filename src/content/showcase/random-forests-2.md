+++
date = "2020-04-29"
title = "Feature Engineering with Random Forests (Part 2)"
showonlyimage = false
draft = false
image = "https://miro.medium.com/max/573/0*Wzx31D2CkGFSkYcO.png"
weight = 3
+++

Learn how to leverage Random Forests to do Feature Engineering and more.
<!--more-->

- Originally published on [Medium](https://medium.com/@sdhnshu/feature-engineering-with-random-forests-part2-160eb0356172)
- Grab the code from [Github](https://github.com/sdhnshu/Random-forests-and-Feature-Engineering)
- Link to [Part 1](/showcase/random-forests)

### Introduction
In the [part 1](/showcase/random-forests) of the series, we looked at the random forests algorithm and figured how to get accurate predictions from the model. But the part we are forgetting is:

> *The features that get into the model are as important as the model itself.*

Especially if you work at an institution, like the one making [bulldozers](https://www.kaggle.com/c/bluebook-for-bulldozers/data), you are trying to understand the following things about the features:

- What is the __importance__ of each feature?
- How __similar__ are they to each other?
- How do these features __interact__ with each other?
- How to make features truly independent from __time__?

Let us look at each of them using concepts from random forests.
### i. Feature Importance

The only way to find the feature importance is rightly described by one of my oldest inspirations in this verse:

> *You don’t know what you’ve got, until its gone — [Chester Bennington](https://www.youtube.com/watch?v=oM-XJD4J36U)*

Technically speaking, if we randomly shuffle the values in a feature but keep everything else the same, how much worse does our model perform. The worse the model performs, the higher the importance of that feature.

```python
# Feature Importance Intuition
def feature_importance(self, model, x_valid):

    def shuffle_col(colname, x_valid):
        df_temp = x_valid.copy()
        df_temp[colname] = np.random.permutation(
            df_temp[colname].values)
        return df_temp

    pred_actual = model.predict(x_valid)
    return [(1 - r2_score(pred_actual, model.predict(
        shuffle_col(col, x_valid)))) for col in x_valid.columns]
```

The above function describes the functionality of feature importance and the one below is a convenient Fast.ai wrapper around the scikit learn’s optimized functionality. Let us plot 30 most important features out of the 66 we have.

```python
set_rf_samples(50000)
m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3,
                          max_features=0.5, n_jobs=-1,
                          oob_score=True)
m.fit(X_train, y_train)
# rf_feat_importance wrapper from Fast.ai
fi = rf_feat_importance(m, df_trn)
fi[:30].plot('cols', 'imp', 'barh', figsize=(12,7), legend=False)
```

![img](https://miro.medium.com/max/573/0*BcS_MXeq3Kv-ibQf.png)

We can remove the ones lower than the importance of 0.005. This leaves us with 23 features. Let us plot the importance again and score the model.

```python
to_keep = fi[fi.imp>0.005].cols
df_keep = df_trn[to_keep].copy()
X_train, X_valid = split_vals(df_keep, n_trn)
m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3,
                       max_features=0.5, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)
fi = rf_feat_importance(m, df_keep)
plot_fi(fi[:30]);

----
Output:
> RMSE Train:  20.720376603129303
> RMSE Valid:  24.570217524283752
>  R2  Train:  91.02715603822325
>  R2  Valid:  89.21882794959521
>    OOB    :  89.39924413165484
```
![img](https://miro.medium.com/max/573/0*soe_Udg2GtaRkXWC.png)

`YearMade`, `Coupler_System` and `ProductSize` turn out to be the top 3 most important features and we have a validation score of 89.2%
### ii. Feature correlation

Combining similar features can be done by going through all the pairs and clustering the ones that are closest to each other. This kind of hierarchical clustering works based on the ordinality of those features. Which is similar to how random forests work. While splitting a node, it only cares about the sorted order of the data and not their actual values. A common coefficient to measure for this kind of ranked correlation is spearman’s ranked coefficient.

```python
from scipy.cluster import hierarchy as hc
corr = np.round(scipy.stats.spearmanr(df_keep).correlation, 4)
plt.figure(figsize=(16,10))
hc.dendrogram(hc.linkage(hc.distance.squareform(1-corr),
                         method='average'),
              labels=df_keep.columns, orientation='left',
              leaf_font_size=16)
plt.show()
```
![img](https://miro.medium.com/max/573/0*8zHuirGqrQL55zdU.png)


The features that are correlated connect faster. With that, we can see that `saleYear` and `saleElapsed` are similar which is expected as they both are time-dependent. `Hydraulics_flow` and `Grouser_Tracks` are similar and also `fiBaseModel` and `fiModelDesc`.

If two features are similar removing one of them won’t affect the accuracy of our model, it will only make it simpler. So let us try dropping them and measuring the drop in score. Before that let’s establish a baseline.
```python
def get_oob(df):
    m = RandomForestRegressor(n_estimators=30, min_samples_leaf=5,
                              max_features=0.6, n_jobs=-1,
                              oob_score=True)
    x, _ = split_vals(df, n_trn)
    m.fit(x, y_train)
    return m.oob_score_ * 100

get_oob(df_keep)

---
Output:

> 89.03929035253717   # Baseline

```
Let’s drop them one by one.
```python
for c in ['saleYear', 'saleElapsed', 'Grouser_Tracks', 'Hydraulics_Flow', 'fiModelDesc', 'fiBaseModel']:
    print(c, get_oob(df_keep.drop(c, axis=1)))

-----
Output:

> saleYear 88.96577294296006
> saleElapsed 88.72699565865227
> Grouser_Tracks 89.02477486920333
> Hydraulics_Flow 88.98468803187907
> fiModelDesc 88.93420811313398
> fiBaseModel 88.91058697856653
```
Because they didn’t do much damage, `saleYear`, `Hydraulics_Flow`, and `fiBaseModel` shall be removed.


```python
to_drop = ['saleYear', 'Hydraulics_Flow', 'fiBaseModel']
get_oob(df_keep.drop(to_drop, axis=1))

---
Output:

> 88.89865421652972
```
The overall OOB score dropped a little but not significantly. Let us check how a full bootstrapped model is looking with the cleaned data. It happens in 30 seconds rather than 1 min in the last post!

```python
reset_rf_samples()

df_keep.drop(to_drop, axis=1, inplace=True)
X_train, X_valid = split_vals(df_keep, n_trn)

m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)
%time m.fit(X_train, y_train)

print_score(m)

-----
Output:

> CPU times: user 1min 45s, sys: 922 ms, total: 1min 46s
> Wall time: 30.4 s
> RMSE Train:  12.51548564415119
> RMSE Valid:  22.79049206209574
>  R2  Train:  96.72636512376931
>  R2  Valid:  90.72411452493563
>    OOB    :  90.86147528243829
```
### iii. Partial dependence

In this section we’ll see how two features interact. But before we get into it, it would be interesting to see the interactions between specific categories from a categorical variable as well. We can one-hot encode categorical variables with less than 7 categories to do that. Let us check their feature importance.

```python
set_rf_samples(50000) # Subsampled model

df_trn2, y_trn, nas = proc_df(df_raw, 'SalePrice', max_n_cat=7)  # 1hot encoded
X_train, X_valid = split_vals(df_trn2, n_trn)

m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.6, n_jobs=-1)
m.fit(X_train, y_train);

fi = rf_feat_importance(m, df_trn2)
plot_fi(fi[:20]);
```
![img](https://miro.medium.com/max/573/0*WYgcNZUdWFGQeDOW.png)

Enclosure was a categorical feature but is now one-hot encoded. And `Enclosure_EROPS w AC` has gained the highest importance among all. Some search on the internet reveals that it stands for an enclosed space with an AC. Which, if in a bulldozer will surely be a deciding factor for its price.

The second most important feature is `YearMade`. It would be logical to see its interacts with 7th most important, `saleElapsed`.

```python
df_raw.plot('YearMade', 'saleElapsed', 'scatter', alpha=0.01, figsize=(10,8));
```
![img](https://miro.medium.com/max/450/0*gHFbPT5HE9JLgWRl.png)

Oh no! There are some outliers in the data with `YearMade=1000`. This might be due to some technical error. We can remove them from our analyses for now.

```python
df_raw = df_raw[df_raw.YearMade>1930]
```

Let us see how `YearMade` affects our dependent variable, `SalePrice` by simply plotting them together.
```python
from plotnine import *

x_all = get_sample(df_raw[df_raw.YearMade>1930], 500)  # sample 500 pts excluding years with wrong year=1000
ggplot(x_all, aes('YearMade', 'SalePrice'))+stat_smooth(se=True, method='loess')
```
![img](https://miro.medium.com/max/408/0*eRH40X0btXLy10ae.png)

This plot shows ( `YearMade` + the rest of the features)'s affect on `SalePrice`.

To single out how a feature affects another, we need to change one and keep the rest constant. In the example below, we sampled 500 rows and changed all values of `YearMade` in the dataset to a value on x, eg: 1990 and predicted a value of `SalePrice`. Doing so for all values of `YearMade` gives us the plot.
```python
x = get_sample(X_train[X_train.YearMade>1930], 500)

def plot_pdp(feat, clusters=None, feat_name=None):
    feat_name = feat_name or feat
    p = pdp.pdp_isolate(m, x, feat)
    return pdp.pdp_plot(p, feat_name, plot_lines=True,
                        cluster=clusters is not None,
                        n_cluster_centers=clusters)
plot_pdp('YearMade')
plot_pdp('YearMade', 5)
```
![img](https://miro.medium.com/max/573/0*vu4fB_9e_nDarbv7.png)
![img](https://miro.medium.com/max/573/0*QGbSRpn3XQMHy_Wq.png)

Similarly to find how `YearMade` and `SaleElapsed` together affect `SalePrice`, we can use an interaction plot.

```python
feats = ['saleElapsed', 'YearMade']
p = pdp.pdp_interact(m, x, feats)
pdp.pdp_interact_plot(p, feats)
```
![img](https://miro.medium.com/max/573/0*Wzx31D2CkGFSkYcO.png)

We can see from the interaction plot that if `YearMade` is high and `saleElapsed` low, the `SalePrice` would be higher. So the `age` = `saleYear` - `YearMade` of the truck is important. You can add `age` as another feature to our dataset.

```python
df_raw.YearMade[df_raw.YearMade<1950] = 1950
df_keep['age'] = df_raw['age'] = df_raw.saleYear-df_raw.YearMade
X_train, X_valid = split_vals(df_keep, n_trn)
m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3,
                          max_features=0.6, n_jobs=-1)
m.fit(X_train, y_train)
plot_fi(rf_feat_importance(m, df_keep));
```
![img](https://miro.medium.com/max/573/0*DjzkQSca-kwhl0UO.png)

`Age` quickly becomes our most important feature.
### iv. Removing time dependency

Unlike linear models or neural networks, random forests are not very good at extrapolating data. So extrapolating over time is tough. It is better, we remove the dependency on time from our features.

__Q: But how to we find the time dependent features?__

> *If the validation set were a random sample of the training set, it would be difficult to predict if a row is in the validation set.*

So if a model can successfully learn whether a value is in the validation set, then it has some temporal dependency helping it to do so. And the most important features in such a model will be the most time dependent features.

All we have to do is, make `is_valid` our dependent variable, and train the model. Because it is a 0/1 discrete prediction we are using a `RandomForestClassifier` instead of a regressor.
```python
df_ext = df_keep.copy()
df_ext['is_valid'] = 1
df_ext.is_valid[:n_trn] = 0
x, y, nas = proc_df(df_ext, 'is_valid')
m = RandomForestClassifier(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)
m.fit(x, y);
m.oob_score_

----
Output:

> 0.9999950140230601
```
There is a very high score of prediction, thus we have features with time dependency. The validation set is not a random sample from the dataset. Let us see which features are the most important to a purely temporal dependent forest.
```python
fi = rf_feat_importance(m, x); fi[:10]
```
![img](https://miro.medium.com/max/187/0*AAeuknC1bCxFAQRq.png)

Our top 3 time dependent features are: `SalesId`, `saleElapsed` and `MachineId`. Let us remove them and see their effect on the score.
```python
x.drop(['SalesID', 'saleElapsed', 'MachineID'], axis=1, inplace=True)
m = RandomForestClassifier(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)
m.fit(x, y);
m.oob_score_

---
Output:

> 0.9788444998441882
```
Checking the importance after removing those features
```python
fi = rf_feat_importance(m, x); fi[:10]
```
![img](https://miro.medium.com/max/204/0*onbQPrnDELkYqKS0.png)

Now we see new time-dependent features pop up like `age`, `YearMade`, and `saleDayofyear`. Let us drop these 6 one by one, and check their negative affects. Before that let us look at our baseline.
````python
set_rf_samples(50000)
X_train, X_valid = split_vals(df_keep, n_trn)
m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3,
                        max_features=0.5, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)

----
Output:

> RMSE Train:  20.633388115029838
> RMSE Valid:  24.491469116919422
>  R2  Train:  91.1023376550528
>  R2  Valid:  89.2878252704439
>    OOB    :  89.46118267044744
````
Now let us drop them one by one.
```python
feats = ['SalesID', 'saleElapsed', 'MachineID', 'age', 'YearMade', 'saleDayofyear']
for f in feats:
    df_subs = df_keep.drop(f, axis=1)
    X_train, X_valid = split_vals(df_subs, n_trn)
    m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3,
            max_features=0.5, n_jobs=-1, oob_score=True)
    m.fit(X_train, y_train)
    print(f)
    print_score(m)
```
![img](https://miro.medium.com/max/191/0*j5LmuOYm8Y5RPYFe.png)

The R² score on the validation set gets better by dropping `SalesId`, `MachineId`, and `saleDayofyear`.

With that, our features are as time-independent as could be. So, let us drop them and train a complete bootstrap model.
```python
df_subs = df_keep.drop(['SalesID', 'MachineID', 'saleDayofyear'], axis=1)
reset_rf_samples()

X_train, X_valid = split_vals(df_subs, n_trn)
m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3,
               max_features=0.5, n_jobs=-1, oob_score=True)

%time m.fit(X_train, y_train)
print_score(m)

plot_fi(rf_feat_importance(m, X_train));

----
Output:

> CPU times: user 1min 15s, sys: 991 ms, total: 1min 16s
> Wall time: 23.4 s
> RMSE Train:  13.775059373372189
> RMSE Valid:  21.81697224146163
>  R2  Train:  96.03428239799979
>  R2  Valid:  91.49964757192211
>    OOB    :  90.91069189446978
```
![img](https://miro.medium.com/max/573/0*041BsubE6iLfLlkK.png)

Congratulations on achieving a __91.4%__ validation score in __23 seconds__ on the whole dataset!
### Boost it up!

Now that we have our features all engineered and model tuned, we can turn the number of trees __all the way up__ to get an accuracy of __92%__ on the validation set.
```python
m = RandomForestRegressor(n_estimators=160, max_features=0.5,
          n_jobs=-1, oob_score=True)
%time m.fit(X_train, y_train)
print_score(m)

----
Output:

> CPU times: user 5min 52s, sys: 11.4 s, total: 6min 4s
> Wall time: 1min 53s
> RMSE Train:  8.013775394813543
> RMSE Valid:  21.09451703745362
>  R2  Train:  98.65782495931056
>  R2  Valid:  92.05329397797038
>    OOB    :  91.45711650192492
```
This gets us into the __top 1%__ of the competition and it is trained in under __2 mins__.

![img](https://miro.medium.com/max/247/0*lqg4n8qMeEUkVRmM.png)

----

*__Link to [Part 1](/showcase/random-forests)__*

### Further reading

- [Fast.ai ML course notes](https://medium.com/@hiromi_suenaga/machine-learning-1-lesson-1-84a1dc2b5236)
- [Model interpretability](https://medium.com/@ag.ds.bubble/model-interpretability-a4244d82ffb2)
- [Intuitive interpretation of random forests](https://medium.com/usf-msds/intuitive-interpretation-of-random-forest-2238687cae45)
- [Coloring with random forests](https://structuringtheunstructured.blogspot.com/2017/11/coloring-with-random-forests.html)
- [Gradient Boosting vs Random Forest](https://medium.com/@aravanshad/gradient-boosting-versus-random-forest-cfa3fa8f0d80)