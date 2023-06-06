# Random Forest

!!! quote "From Ho TK. Random decision forests"
    "The essence of the method is to build multiple trees in randomly selected subspaces of the feature space."[@Ho1995-gk]

Random forest is an ensemble method based on decision trees which are dubbed as base-learners. Instead of using one single decision tree and model on all the features, we utilize a bunch of decision trees and each tree can model on a subset of features (feature subspace). To make predictions, the results from each tree are combined using some democratization.

Translating to math language, given a proper dataset $\mathscr D(\mathbf X, \mathbf y)$, random forest or the ensemble of trees, denoted as $\{f_i\}$, will predict an ensemble of results $\{f_i(\mathbf X_i)\}$, with $\mathbf X_i \subseteq \mathbf X$.

In this section, we ask ourselves the following questions.

1. How to democratize the ensemble of results from each tree?
2. What determines the quality of the predictions?
3. Why does it even work?

## Margin, Strength, and Correlations

The margin of the model, the strength of the trees, and the correlation between the trees can help us understand how random forest work.

### Margin

The **margin** of the tree is defined as[@Breiman2001-oj]

$$
M(\mathbf X, \mathbf y) = {\color{green}P (\{f_i(\mathbf X)=\mathbf y \})} - \operatorname{max}_{\mathbf j\neq \mathbf y} {\color{red}P (\{ f_i(\mathbf X) = \mathbf j \})}.
$$

!!! note "Terms in the Margin Definition"
    The first term, ${\color{green}P (\{f_i(\mathbf X)=\mathbf y \})}$ is the probability of predicting the exact value in the dataset. In a random forest model, it can be calculated using

    $$
    {\color{green}P (\{f_i(\mathbf X)=\mathbf y \})} = \frac{\sum_{i=1}^N I(f_i(\mathbf X) = \mathbf y)}{N},
    $$

    where $I$ is the indicator function that maps the correct predictions to 1 and the incorrect predictions to 0. The summation is over all the trees.

    The term ${\color{red}P (\{f_i(\mathbf X) = \mathbf j \})}$ is the probability of predicting values $\mathbf j$. The second term $\operatorname{max}_{\mathbf j\neq \mathbf y} {\color{red}P ( \{f_i(\mathbf X) = \mathbf j\})}$ finds the highest misclassification probabilities, i.e., the max probabilities of predicting values $\mathbf j$ other than $\mathbf y$.

To make it easier to interpret this quantity, we only consider two possible predictions:

- $M(\mathbf X, \mathbf y) \to 1$: We will always predict the true value, for all the trees.
- $M(\mathbf X, \mathbf y) \to -1$: We will always predict the wrong value, for all the trees.
- $M(\mathbf X, \mathbf y) \to 0$, we have an equal probability of predicting the correct value and the wrong value.

In general, we prefer a model with higher $M(\mathbf X, \mathbf y)$.

### Strength

However, the margin of the same model is different in different problems. The same model for one problem may give us margin 1 but it might not work that well for a different problem. This can be seen in [our decision tree examples](tree.basics.md).

To bring the idea of margin to a specific problem, Breiman defined the **strength** as the expected value of the margin over the dataset$s$[@Breiman2001-oj],

$$
s = E_{\mathscr D}[M(\mathbf X, \mathbf y)].
$$


### Raw Margin

Instead of using the probability of the predictions in the margin, the indicator function itself is also a measure of how well the predictions are. The **raw margin** is then defined as[@Breiman2001-oj]

$$
M_{R,i}(\mathbf X, \mathbf y) = I (f_i(\mathbf X)=\mathbf y ) - \operatorname{max}_{\mathbf j\neq \mathbf y} I ( f_i(\mathbf X) = \mathbf j ).
$$


### Correlation

The **correlation** between the trees is[@Breiman2001-oj]

$$
\rho_{ij} = \operatorname{corr}(M_{R,i}, M_{R,j}) = \frac{\operatorname{cov}(M_{R,i}, M_{R,j})}{\sigma_{M_{R,i}} \sigma_{M_{R,j}}}  = \frac{E[(M_{R,i} - \bar M_{R,i})(M_{R,j} - \bar M_{R,j})]}{\sigma_{M_{R,i}} \sigma_{M_{R,j}}}.
$$

The correlation tells us how strong the two trees are correlated. If all trees are similar, the correlation is high. Ensembling won't help improving the model in this situation.

!!! note ""
    To get a scalar value of the whole model, the average correlation $\bar \rho$ over all the possible pairs is calculated.


## Predicting Power

The power of the ensemble can be measured by the generalization error,

$$
P_{err} = P_{\mathscr D}(M(\mathbf X, \mathbf y)< 0),
$$

i.e., the probability of getting the correct answer over the whole dataset.

It has been proved that **the ensemble will converge in the random forest as the number of trees gets large.** And the generalization error is proven to be related to the strength and the mean correlation,

$$
P_{err} \leq \frac{\bar \rho (1-s^2) }{s^2}.
$$

We conclude that

1. **The stronger the strength is, the lower the generalization error is.**
2. **The smaller the correlation is, the lower the generalization error is.**


![Upper Limit of generalization error](../assets/tree.random-forest/rf_generalization_error.png)
> Upper Limit of generalization error as functions of $\bar \rho$ and $s$.


## Random Forest Regressor

Similar to decision trees, a random forest can also be used as regressors. A similar conclusion about the regressors can be proved.

To see how the regressor works, we construct an artificial problem. The code can be accessed [on GitHub](https://github.com/datumorphism/mini-code/blob/master/random_forest/random_forest_benchmark.ipynb).

The data we will use is generated by sin function.

```python
X_sin = [[6*random()] for i in range(10000)]
y_sin = np.sin(X_sin)

X_sin_test = [[6*random()] for i in range(10000)]
y_sin_test = np.sin(X_sin_test)
```

A random forest model is constructed and a random search cross validation is applied

```python
model = RandomizedSearchCV(
    pipeline,
    cv=10,
    param_distributions = rf_random_grid,
    verbose=3,
    n_jobs=-1
)
```
