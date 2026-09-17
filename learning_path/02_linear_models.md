# 2 · Understand the linked chapter

**Prerequisite:** Step 1. **Destination:** Explain the chapter's models, loss functions and penalties in your own words, calculate small examples, and choose reasonable experiments.

Read one section, do its checkpoint, then revisit the corresponding section in [the official chapter](https://ideal.umd.edu/ML4ME_Textbook/part1/reviewing_supervised_linear_models.html). You do not need to understand matrix inverses or derivatives yet; Step 3 develops them.

## 1. Linear means linear in the fitted weights

For one input, $\hat y=b+wx$. For two inputs, $\hat y=b+w_1x_1+w_2x_2$. You might use force and temperature to predict displacement. Each weight tells you how the prediction changes when its feature increases by one unit while the others remain fixed; this association alone does not establish causation.

A **feature map** constructs useful inputs from the raw measurement. With $\phi(x)=[1,x,x^2]$ and $w=[1,2,3]$,

$$\hat y=w^T\phi(x)=1+2x+3x^2.$$

At $x=2$, the feature vector is $[1,2,4]$ and the prediction is $1+4+12=17$. The superscript $T$ means transpose; the dot product multiplies corresponding entries and adds them. This is just the familiar sum written compactly.

The curve bends as $x$ changes, but each unknown weight still appears as a coefficient to the first power. It is therefore a **linear model in its weights**. $\sin(x)$ can also be a fixed feature. By contrast, $\sin(wx)$ is generally nonlinear in the unknown $w$.

### Checkpoint 1

Which are linear in their unknown weights: $w_0+w_1x^3$, $w_0+w_1\sin x$, and $w_0+\sin(w_1x)$? For $w=[2,-1,0.5]$, predict the output at $x=2$ using $[1,x,x^2]$.

<details><summary>Worked answer</summary>

The first two are linear in their weights; the third is not. The numerical prediction is $2-1\times2+0.5\times4=2$. A nonlinear transformation of the input is compatible with a linear model if that transformation is fixed before fitting the weights.

</details>

## 2. Fitting means choosing weights to minimize an objective

Ordinary least squares selects weights that minimize the sum of squared residuals. Dividing that sum by a fixed positive sample count changes its scale, but not the minimizer if no penalty is involved. A fitted model can be used on inputs it did not see during training.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from ipywidgets import interact, FloatSlider, IntSlider

X_small = np.array([[0.], [1.], [2.]])
y_small = np.array([1., 3., 5.])
fitted = LinearRegression().fit(X_small, y_small)
print("Intercept:", fitted.intercept_)
print("Slope:", fitted.coef_[0])
print("Prediction at 3 N:", fitted.predict([[3.]])[0], "mm")
assert np.allclose(fitted.predict(X_small), y_small)
```

`fit` estimates parameters from inputs and targets. `predict` applies those parameters. `coef_` contains the fitted feature weights; `intercept_` contains the fitted intercept. Their numerical values depend on the data and the model assumptions.

The book sometimes displays a regression model's `.score(...)`. For these scikit-learn regressors this is **R²**, not classification accuracy: $R^2=1-\sum_i(y_i-\hat y_i)^2/\sum_i(y_i-\bar y)^2$, for nonconstant targets. A value of 1 means a perfect fit; 0 matches always predicting that dataset's mean; a negative value is worse than that mean baseline. A high **training** R² still does not establish generalization.

## 3. Why the best training fit can be a bad model

A flexible model can follow random fluctuations in its training data. Those fluctuations need not repeat in new experiments. **Overfitting** means fitting sample-specific details at the expense of generalization. **Underfitting** means the model is too restrictive to capture the useful relationship.

Two useful terms are **bias**, systematic prediction error averaged over hypothetical repeated training datasets, and **variance**, sensitivity of the fitted prediction to which training examples were observed. Under the usual squared-error setup with an independent new noise realization, expected prediction error decomposes into squared bias, prediction variance, and irreducible noise variance. Regularization often increases bias while reducing variance; validation tells us whether the tradeoff helps. This use of “bias” differs from calling the intercept a bias term.

Polynomial degree is a **hyperparameter**: a choice about the model family. Polynomial coefficients are **parameters**: quantities fitted within that family. We need data that did not determine the fitted coefficients to compare hyperparameters fairly.

- **Training data:** fit weights and preprocessing statistics.
- **Validation data:** compare degrees, penalties, or other design choices.
- **Test data:** make a final assessment after those choices are fixed.

Cross-validation repeats train/validation partitions within the development data. For grouped experiments or time-dependent data, the split must respect groups or time; random splitting can leak information between related examples. Repeatedly choosing models based on a test score turns that set into another validation set.

We use synthetic data below, so we know the hidden relationship and can draw it. With real data, that dashed “truth” curve would not be available.

```python
rng = np.random.default_rng(17)
def hidden_relation(x):
    return 0.7 + 1.1 * x + 0.35 * np.sin(7 * x)

X_train = np.sort(rng.uniform(-1, 1, 16)).reshape(-1, 1)
y_train = hidden_relation(X_train[:, 0]) + rng.normal(0, 0.16, len(X_train))
X_val = np.linspace(-1, 1, 70).reshape(-1, 1)
y_val = hidden_relation(X_val[:, 0]) + rng.normal(0, 0.16, len(X_val))
```

## 4. Regularization: add a cost for the weights

Our convention throughout this experiment is

$$J(b,w)=\underbrace{\frac1N\sum_i(y_i-b-w^T\phi(x_i))^2}_{\text{data loss}}+\underbrace{\lambda\sum_j w_j^2}_{\text{penalty}}.$$

$\lambda\geq0$ is a hyperparameter. Large $\lambda$ makes large feature weights more expensive. The intercept is not penalized in this example. Increasing regularization can reduce overfitting, but too much produces underfitting. It is a tradeoff to validate, not a rule that “larger is better.”

Here $\phi(x)=[x,x^2,\ldots,x^d]$ excludes the constant feature because $b$ already supplies the intercept. An equivalent notation includes 1 among the features and folds $b$ into the weight vector; do not accidentally include the same intercept twice.

Weights have units. A force measured in N gives different numerical coefficients from the same force measured in kN. This changes the effect of a coefficient penalty. We therefore standardize feature columns using training-set means and standard deviations. Standardization makes a feature roughly mean zero and scale one; it is not a guarantee of a Gaussian distribution.

For a nonconstant feature, the transformation is $z=(x-\bar x_{train})/s_{train}$. Apply those same training statistics to later observations. Computing them from the full dataset before splitting would allow validation/test data to influence preprocessing.

**A software convention:** scikit-learn's `Ridge(alpha=a)` minimizes a **sum** of squared residuals plus $a\|w\|_2^2$. To match our **mean**-loss convention, the code sets `alpha = N * lambda`. The pipeline fits scaling only on the training split. [Ridge documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html).

```python
def fit_experiment(degree=3, log10_lambda=-4.0):
    penalty = 10 ** log10_lambda
    model = make_pipeline(
        PolynomialFeatures(degree=degree, include_bias=False),
        StandardScaler(),
        Ridge(alpha=len(X_train) * penalty, solver="svd")
    )
    model.fit(X_train, y_train)
    grid = np.linspace(-1, 1, 250).reshape(-1, 1)
    train_mse = mean_squared_error(y_train, model.predict(X_train))
    val_mse = mean_squared_error(y_val, model.predict(X_val))
    fig, ax = plt.subplots(figsize=(8, 4.3))
    ax.scatter(X_train[:, 0], y_train, label="Training observations", color="#126782")
    ax.scatter(X_val[:, 0], y_val, label="Validation observations", color="#888888", s=12, alpha=0.45)
    ax.plot(grid[:, 0], hidden_relation(grid[:, 0]), "--", color="#3b7f52", label="Synthetic truth")
    ax.plot(grid[:, 0], model.predict(grid), color="#d46b32", label="Fitted model")
    ax.set(xlabel="Normalized load (dimensionless)", ylabel="Response (arbitrary units)",
           ylim=(-1.5, 3.0), title=f"Degree {degree} · λ={penalty:.1e} · train MSE={train_mse:.4f} · validation MSE={val_mse:.4f}")
    ax.legend(fontsize=9); ax.grid(alpha=0.2); plt.show()

interact(fit_experiment,
         degree=IntSlider(value=3, min=1, max=12, continuous_update=False),
         log10_lambda=FloatSlider(value=-4, min=-8, max=2, step=0.5, continuous_update=False));
```

**Try:** degree 1, then 6, then 12, with little regularization. Watch both errors. Keep degree 12 and increase $\lambda$. Find a compromise on validation error. “MSE units” are squared response units. The plotted vertical range is fixed for comparison, so an unstable fitted curve may extend outside it.

### Checkpoint 2

Model A has training MSE 0.001 and validation MSE 1.2. Model B has training MSE 0.06 and validation MSE 0.09. Which is the better candidate for deployment, assuming the validation data represent the intended use? Is this a final test?

<details><summary>Worked answer</summary>

B is the better candidate because its error on unseen validation examples is much lower. A's near-perfect training score is not the objective of model selection. This remains model selection; assess the selected workflow once on an untouched test set. Dataset shift or safety requirements would require additional evaluation.

</details>

## 5. Norms: different ways to measure the size of weights

For a vector $w=[3,-4]$:

| Quantity | Definition | Value |
|---|---|---:|
| L1 | $\sum_j|w_j|$ | 7 |
| L2 | $\sqrt{\sum_jw_j^2}$ | 5 |
| Squared L2 | $\sum_jw_j^2$ | 25 |
| L-infinity | $\max_j|w_j|$ | 4 |
| L0 count | Number of nonzero entries | 2 |

For $p\geq1$, $\|w\|_p=(\sum_j|w_j|^p)^{1/p}$. For $0<p<1$, the same expression is a **quasi-norm**, not a true norm. The nonzero count called “L0” is not a norm either.

**Ridge** uses a squared-L2 penalty and typically shrinks weights continuously. **Lasso** uses an L1 penalty and can set some weights exactly to zero. **Elastic Net** combines L1 with squared L2. “L1 loss” and “L1 penalty” act on different things: the former on prediction errors, the latter on model weights.

Why can L1 produce exact zeros? Consider one coefficient and the objective $\tfrac12(w-a)^2+\lambda|w|$. Away from zero, the penalty adds a fixed pull towards zero. The kink at zero permits the optimum to stay exactly there over a range of $a$ values. Its solution is $\operatorname{sign}(a)\max(|a|-\lambda,0)$. The comparable squared-L2 problem $\tfrac12(w-a)^2+\tfrac\lambda2w^2$ has solution $a/(1+\lambda)$.

These simple formulas apply to this separable toy objective, not arbitrary correlated feature matrices. The experiment shows two coefficients with unpenalized optimum $(3,1)$.

Squared loss in linear-model weights and the L1 or squared-L2 penalties are **convex**: intuitively, their objectives have no separate, worse local minima. There can still be several equally good global minimizers if the problem is not strictly convex. Many penalties with orders below 1 are nonconvex; an optimization routine may then find a local solution that is not globally best. Sparse selected features also need interpretation: correlated features can substitute for each other, so a nonzero coefficient is not automatic evidence of a unique physical cause.

```python
def shrinkage_experiment(penalty=1.0):
    a = np.array([3., 1.])
    ridge = a / (1 + penalty)
    lasso = np.sign(a) * np.maximum(np.abs(a) - penalty, 0)
    values = np.linspace(0, 5, 150)
    ridge_path = a[None, :] / (1 + values[:, None])
    lasso_path = np.maximum(a[None, :] - values[:, None], 0)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ridge_path[:, 0], ridge_path[:, 1], label="Squared L2 (Ridge)")
    ax.plot(lasso_path[:, 0], lasso_path[:, 1], label="L1 (Lasso)")
    ax.scatter(*ridge, s=80); ax.scatter(*lasso, s=80)
    ax.scatter(*a, marker="*", s=120, color="black", label="Unpenalized optimum")
    ax.set(xlabel="Weight 1", ylabel="Weight 2", xlim=(-0.1, 3.2), ylim=(-0.1, 1.2),
           title=f"λ={penalty:.1f}: Ridge={ridge.round(2)}, Lasso={lasso.round(2)}")
    ax.legend(); ax.grid(alpha=0.2); plt.show()

interact(shrinkage_experiment, penalty=FloatSlider(value=1, min=0, max=5, step=0.1, continuous_update=False));
```

### Checkpoint 3

For $w=[0,-2,2]$, calculate L1, L2, squared L2, L-infinity and the nonzero count. For $a=0.6$ and $\lambda=1$, what do the two one-dimensional shrinkage rules give?

<details><summary>Worked answer</summary>

L1 = 4; L2 = $\sqrt8\approx2.828$; squared L2 = 8; L-infinity = 2; count = 2. Ridge gives $0.6/2=0.3$. Lasso gives $\max(0.6-1,0)=0$. The distinction between L2 and squared L2 changes the objective and its optimum.

</details>

## 6. Regression losses: which mistakes matter?

Let $r=y-\hat y$. We choose a loss according to what errors mean in the application.

| Loss | Formula | Interpretation |
|---|---|---|
| Squared | $r^2$ | Large residuals become very expensive |
| Absolute | $|r|$ | Cost grows linearly with residual magnitude |
| Huber | $r^2/2$ if $|r|\leq\delta$; $\delta(|r|-\delta/2)$ otherwise | Smooth quadratic centre with less aggressive tails |
| Epsilon-insensitive | $\max(0,|r|-\epsilon)$ | An explicit tolerance band incurs zero loss |
| Squared epsilon-insensitive | $\max(0,|r|-\epsilon)^2$ | Zero inside the band, quadratic outside it |

Squared errors $[1,1,10]$ cost $1+1+100=102$, so the largest residual dominates. Absolute errors cost $1+1+10=12$. This explains why a squared-loss fit may move substantially towards an outlier. A residual-robust loss does not protect against every kind of outlier: extreme or corrupted input features can still exert high leverage.

```python
def regression_losses(epsilon=0.7):
    r = np.linspace(-3, 3, 301)
    huber = np.where(np.abs(r) <= 1, 0.5*r**2, np.abs(r)-0.5)
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for values, name in [(r**2, "Squared"), (np.abs(r), "Absolute"),
                         (huber, "Huber, δ=1"),
                         (np.maximum(0, np.abs(r)-epsilon), "ε-insensitive"),
                         (np.maximum(0, np.abs(r)-epsilon)**2, "Squared ε-insensitive")]:
        ax.plot(r, values, label=name)
    ax.set(xlabel="Residual (normalized units)", ylabel="Loss (normalized units)",
           title=f"Effect of the error tolerance ε={epsilon:.1f}")
    ax.legend(fontsize=9); ax.grid(alpha=0.2); plt.show()

interact(regression_losses, epsilon=FloatSlider(value=0.7, min=0, max=2, step=0.1, continuous_update=False));
```

### Checkpoint 4

For $r=2$, $\delta=1$, $\epsilon=0.5$, calculate all five losses. Which would explicitly ignore a residual of 0.3?

<details><summary>Worked answer</summary>

Squared = 4; absolute = 2; Huber = $1(2-0.5)=1.5$; epsilon-insensitive = 1.5; squared epsilon-insensitive = 2.25. Both epsilon-insensitive losses ignore 0.3 when $\epsilon=0.5$. These comparisons use normalized residuals; different losses have different natural units and scalings.

</details>

## 7. Classification: separate score, label, probability and margin

Suppose we predict whether a component fails. A linear rule first produces a real-valued **score** $s=b+w^Tx$. That score can be negative or larger than one; it is not automatically a probability.

For labels in $\{-1,+1\}$, use the sign of $s$ to predict the class, with an explicit convention for ties. Define the **functional margin** $m=ys$. If the label is +1 and score is +2, $m=2$: correct sign. If the label is −1 and score is +2, $m=-2$: wrong sign. A larger positive margin reflects stronger separation in score units.

The geometric distance from an example to the hyperplane, for nonzero $w$, is $|s|/\|w\|_2$. This matters because scaling both $b$ and $w$ changes the functional margin without changing the decision boundary. A maximum-margin SVM combines hinge loss with control of the weight norm; hinge loss alone does not fix that scaling.

| Classification loss | Margin formula | What it rewards |
|---|---|---|
| Zero-one | 1 for an incorrect class, otherwise 0 | Counts mistakes, but gives little optimization guidance |
| Perceptron | $\max(0,-m)$ | Stops penalizing once the sign is correct |
| Hinge | $\max(0,1-m)$ | Also penalizes correctly signed examples with margin below 1 |
| Squared hinge | $\max(0,1-m)^2$ | A quadratic hinge violation |
| Modified Huber | 0 if $m\geq1$; $(1-m)^2$ if $-1\leq m<1$; $-4m$ if $m<-1$ | A quadratic margin region with linear growth for strongly wrong scores |
| Logistic | $\log(1+e^{-m})$ | Smoothly rewards larger correct margins |

For zero-one loss, specify the tie rule at $s=0$; plots often count the zero-margin case as a failure. Zero-one loss is useful for evaluation even though its flat regions and discontinuity make direct gradient optimization difficult. Perceptron loss can be zero at a zero score, another reason to distinguish the loss from the decision rule.

```python
margin = np.linspace(-3, 3, 301)
modified_huber = np.where(margin >= 1, 0, np.where(margin >= -1, (1-margin)**2, -4*margin))
fig, ax = plt.subplots(figsize=(7.5, 4))
for values, name in [((margin <= 0).astype(float), "Zero-one (tie counted as error)"),
                     (np.maximum(0, -margin), "Perceptron"),
                     (np.maximum(0, 1-margin), "Hinge"),
                     (modified_huber, "Modified Huber"),
                     (np.logaddexp(0, -margin), "Logistic")]:
    ax.plot(margin, values, label=name)
ax.set(xlabel="Functional margin m = y × score", ylabel="Loss", title="Correct sign is only one possible training goal")
ax.axvline(0, color="gray", linewidth=0.8); ax.legend(); ax.grid(alpha=0.2); plt.show()
```

Classification's modified Huber loss is a function of the **margin**. Regression's Huber loss is a function of the **residual**. Their names are related, but their arguments and formulas are different.

**Logistic regression** converts a score into a model probability with the sigmoid:

$$p(y=1\mid x)=\sigma(s)=\frac{1}{1+e^{-s}}.$$

It is called regression but is used here for classification. A score of 0 gives 0.5; a score of 2 gives approximately 0.881. Model probabilities need validation and may be miscalibrated. A threshold of 0.5 is a convention, not a requirement; asymmetric costs can justify another threshold chosen on validation data.

For targets $y\in\{0,1\}$, binary cross-entropy is $-[y\log p+(1-y)\log(1-p)]$. A true positive assigned probability 0.9 costs $-\log0.9\approx0.105$; assigned probability 0.1 it costs about 2.303. Confidence in the wrong answer is expensive.

```python
def probability_experiment(score=0.0):
    probability = 1 / (1 + np.exp(-score))
    scores = np.linspace(-6, 6, 250)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(scores, 1 / (1 + np.exp(-scores)))
    ax.scatter([score], [probability], s=75)
    ax.set(xlabel="Linear score", ylabel="Model probability of class 1", ylim=(-0.03, 1.03),
           title=f"Score {score:.1f} → probability {probability:.3f}")
    ax.grid(alpha=0.2); plt.show()

interact(probability_experiment, score=FloatSlider(value=0, min=-6, max=6, step=0.2, continuous_update=False));
```

### Checkpoint 5

If $y=-1$ and $s=-0.2$, is the sign correct? Calculate the margin, hinge loss, and perceptron loss. Can a linear score of 3 be interpreted as a probability?

<details><summary>Worked answer</summary>

The sign is correct. Margin = $(-1)(-0.2)=0.2$; hinge = 0.8; perceptron = 0. The score 3 is not a probability; applying the sigmoid gives approximately 0.953. Even then it is a model estimate, not a guaranteed observed frequency.

</details>

## 8. Choose the objective before choosing the optimizer

The **loss** expresses which prediction errors matter. The **penalty** expresses a preference over fitted weights. The **optimizer** finds parameters for that objective. Gradient descent, SGD and a least-squares solver are optimization methods; changing the method is not the same as changing the prediction model or loss.

For a numerical engineering output, begin with an interpretable baseline and squared loss. If residual outliers are plausible, investigate robust alternatives. If only errors beyond a specified tolerance matter, an epsilon-insensitive loss expresses that. For binary probabilities, logistic regression is a useful baseline. For sparse coefficients on comparable scales, investigate Lasso or Elastic Net. Check assumptions and compare on appropriate validation data.

### Checkpoint 6

A sensor occasionally sends an absurd target reading. You also want only a few of 100 input features to have nonzero coefficients. Are these the same requirement? Which parts of the objective address them?

<details><summary>Worked answer</summary>

They are different. A robust residual loss addresses sensitivity to large target errors. An L1 coefficient penalty encourages sparsity. You can combine objectives to address both, then validate the resulting model. Removing features and reducing outlier influence are not interchangeable.

</details>

## 9. Reading the book precisely

These notes clarify the inspected version of the chapter; they do not change the official material.

- In its displayed general $p$-norm formula, the final outer exponent is printed as $p$ after correctly showing $1/p$ earlier. **Use $1/p$.** NumPy also distinguishes norms from the $p<1$ case. [NumPy reference](https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html).
- Distinguish **L2** from **squared L2**. Ridge uses squared L2, whereas the chapter's later generic `norm(..., ord=2)` experiment uses unsquared L2. These are different objectives. Standard Elastic Net combines L1 and **squared** L2.
- The plotted expression `abs(error) - epsilon` needs a lower bound of zero to be epsilon-insensitive loss. Use `maximum(0, abs(error) - epsilon)`.
- The expression $y_i(w^Tx_i)$ is the **margin**, not itself a classification loss to minimize. Hinge or logistic loss acts on that margin.
- “BLUE” refers to an **unregularized** least-squares estimator under the Gauss–Markov assumptions, including full column rank and errors with zero conditional mean and covariance $\sigma^2I$. Features need not be mutually uncorrelated; they must not have exact linear dependence for this full-rank statement. A Ridge estimator is generally biased. “Best” here means lowest covariance among linear unbiased estimators, not universally best predictions.
- L1's nondifferentiability at zero does not prevent optimization: coordinate, subgradient and proximal methods can handle it. L0 subset selection is difficult in general but not impossible or universally useless. Ordinary unconstrained smooth optimizers are not a reliable way to solve every nonconvex or discontinuous penalty problem.

## 10. Your chapter mastery challenge

You have 200 paired load/deflection measurements and suspect mild nonlinear behaviour and occasional erroneous target readings. Write a plan before opening the answer:

1. Identify inputs, targets and units.
2. Propose a baseline and a more flexible linear-in-weights model.
3. Specify a train/validation/test approach, including preprocessing.
4. Explain which hyperparameters you would compare.
5. Explain which loss and penalty choices address different concerns.
6. Describe evidence of overfitting.
7. Explain why low training error does not establish safe extrapolation.

<details><summary>Worked plan</summary>

Use load as input and deflection as target. Start with a straight line; compare modest polynomial degrees. Reserve a test set, and use cross-validation inside the remaining data, respecting specimen groups if repeated measurements come from the same specimen. Put feature construction and scaling inside the pipeline. Compare degree and regularization strength; investigate a robust loss if erroneous readings materially affect fits. Ridge can stabilize flexible polynomial coefficients; L1 has a different goal when feature selection is useful. A widening training/validation error gap suggests overfitting. Finally assess the selected workflow on the test set and report errors in meaningful units. Evidence on the observed load range does not guarantee behaviour outside that range or outside the spring's physical operating regime.

</details>

You are ready to read the linked chapter when you can solve the six checkpoints and defend this plan without opening the answers. Next, go to the book's cross-validation and gradient-descent chapters, supported by Step 3 below.
