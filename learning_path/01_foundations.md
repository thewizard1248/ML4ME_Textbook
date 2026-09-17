# 1 · Foundations: from numbers to a learning model

**Starting point:** You can add and multiply numbers. No Python or ML is assumed. **Destination:** You can read a simple model, calculate its mistakes, and run your first experiment.

We will use a deliberately simple engineering example: measuring how a spring's extension changes with applied force. Our numbers are teaching data, not measurements of a real component.

## 1. Variables are named quantities

Let $x$ mean applied force, measured in newtons (N), and $y$ mean extension, measured in millimetres (mm). Writing $x=3$ says “this measurement used a force of 3 N.” A variable can hold a different value for another measurement.

An **equation** describes a relationship. In $y=2x+1$, the symbol next to $x$ means multiplication: $2x=2\times x$. If $x=3$, then $y=2\times3+1=7$. If $x=0$, then $y=1$.

The function notation $f(x)=2x+1$ is a name for the same rule. Think “put in $x$, apply the rule, get an answer.” The graph places $x$ on the horizontal axis and $f(x)$ on the vertical axis.

The **slope**, 2, means one additional newton raises predicted extension by 2 mm. Its units are mm/N. The **intercept**, 1, is the prediction at zero force, with units mm. An intercept might represent a sensor offset; whether it is physically sensible depends on the experiment.

### Checkpoint 1

For $f(x)=3x-2$, calculate $f(0)$ and $f(4)$. What changes when the slope changes from 3 to 5?

<details><summary>Worked answer</summary>

$f(0)=-2$ and $f(4)=10$. With slope 5, each one-unit increase in the input changes the output by 5 rather than 3. The intercept remains −2; changing the slope pivots the line around its intercept.

</details>

## 2. The small amount of arithmetic we will reuse

An exponent means repeated multiplication: $3^2=9$ and $2^3=8$. A square root reverses squaring for nonnegative numbers: $\sqrt{9}=3$. Absolute value measures magnitude without sign: $|-3|=3$. A fraction divides: $6/3=2$.

Parentheses control the order. $(2+3)^2=25$, while $2+3^2=11$. A negative squared is positive: $(-3)^2=9$. In Python, write `(-3)**2` to make that intention explicit.

Later we need $e^x$ and $\log x$. The exponential $e^x$ grows multiplicatively; $e\approx2.718$. The natural logarithm reverses it: $\log(e^x)=x$. Thus $\log1=0$, $\log e=1$, and $\log(ab)=\log a+\log b$ for positive $a,b$. A small positive probability has a negative log. Logs turn products of many probabilities into easier sums.

The symbol $\sum$ means “add these terms.” For $r=[1,-2,3]$,

$$\sum_{i=1}^{3}r_i^2=1^2+(-2)^2+3^2=14.$$

The subscript $i$ identifies an entry; it is not multiplication. Dividing the sum by 3 gives the average squared value, $14/3$.

### Checkpoint 2

Calculate $|-4|$, $(-4)^2$, and the average of $2,4,9$. Why must $\log p$ be negative when $0<p<1$?

<details><summary>Worked answer</summary>

The results are 4, 16, and $(2+4+9)/3=5$. Since $e^0=1$ and the exponential is increasing, an input below zero is required to produce a number between zero and one. Therefore its logarithm is negative.

</details>

## 3. A dataset is a table of examples

| Measurement | Force $x$ (N) | Observed extension $y$ (mm) |
|---|---:|---:|
| 1 | 0 | 1 |
| 2 | 1 | 3 |
| 3 | 2 | 5 |

Each row is an **example** or **sample**. The input column is a **feature**. The desired output is the **target**. If we also recorded temperature, we would have two input features per row. With observed targets, learning is called **supervised**.

Predicting extension is **regression**, because the output is a numerical quantity. Predicting “failed” or “working” is **classification**, because the target is a category. The word *regression* does not mean that values must decrease.

Real measurements contain noise. A rule predicts $\hat y$, pronounced “y hat.” The actual measurement remains $y$. Keeping these separate is essential: a model's output is a claim about reality, not reality itself.

For our three examples, a model $\hat y=1+2x$ predicts every target exactly. This is possible because we constructed a tiny, perfectly straight dataset. Real datasets generally do not behave this neatly.

## 4. Python: the essentials, one line at a time

A notebook has text cells and code cells. A code cell runs when you press **Shift + Enter**. It can create variables used by later cells. If you restart the kernel, those variables disappear until you run their cells again.

In Python, `=` assigns a value. `==` asks whether two values are equal. Multiplication needs `*`; powers use `**`, not `^`. A `#` starts a comment for the reader. `print(...)` displays a value.

```python
force = 3                 # a number, in newtons
slope = 2                 # millimetres per newton
intercept = 1             # millimetres
prediction = intercept + slope * force
print("Predicted extension:", prediction, "mm")
print("Prediction equals 7:", prediction == 7)
```

A **function** packages a rule for reuse. `def` defines it; the indented `return` line specifies the answer. Changing the input does not require rewriting the rule.

```python
def predict_extension(force, slope=2, intercept=1):
    return intercept + slope * force

for force in [0, 1, 2, 3]:
    print(force, "N ->", predict_extension(force), "mm")
```

`[0,1,2,3]` is a list. The `for` loop repeats its indented instruction for each entry. Libraries provide useful functions; `import numpy as np` makes NumPy available under the short name `np`. NumPy **arrays** allow arithmetic over many measurements at once.

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.array([0., 1., 2.])
y = np.array([1., 3., 5.])
y_hat = 1 + 2 * x
print("Predictions:", y_hat)
print("First observation:", y[0])  # indexing starts at zero
print("Array shape:", x.shape)
```

`(3,)` means three entries in a one-dimensional array. Most scikit-learn models want a table of shape `(number_of_samples, number_of_features)`. For one feature and three samples, that is `(3,1)`. `x.reshape(-1,1)` creates this table; `-1` means “infer the necessary size.”

```python
X = x.reshape(-1, 1)
print(X)
print("Table shape:", X.shape)
```

### Checkpoint 3

A dataset contains 80 experiments, each with force, temperature, and thickness as inputs. What should the shape of `X` be? What does `X[0]` represent?

<details><summary>Worked answer</summary>

The shape is `(80,3)`: 80 rows and 3 feature columns. `X[0]` is the first experiment's three input measurements. The row index begins at zero.

</details>

## 5. How wrong is the model?

Define the residual as $r_i=y_i-\hat y_i$. Positive means the model predicted too little; negative means it predicted too much. Squaring removes the sign and makes large errors count more.

The **mean squared error**, MSE, is

$$\mathrm{MSE}=\frac1N\sum_{i=1}^{N}(y_i-\hat y_i)^2.$$

For our data, try $\hat y=1+x$. Predictions are $[1,2,3]$, residuals are $[0,1,2]$, squared residuals are $[0,1,4]$, and MSE is $5/3\approx1.667\ \mathrm{mm}^2$. The root mean squared error, RMSE, is $\sqrt{5/3}\approx1.291$ mm. RMSE restores the output's units.

```python
y_hat = 1 + x
residuals = y - y_hat
mse = np.mean(residuals ** 2)
print("Residuals:", residuals)
print("MSE:", round(mse, 3), "mm²")
print("RMSE:", round(np.sqrt(mse), 3), "mm")
assert np.isclose(mse, 5 / 3)
```

## 6. Experiment: fit the line yourself

Before touching the controls: what should happen if you increase the slope? Which control moves every prediction up by the same amount? The dashed vertical segments below show residuals. Our goal here is just to make their squared lengths small.

```python
from ipywidgets import interact, FloatSlider

def line_experiment(slope=1.0, intercept=1.0):
    predicted = intercept + slope * x
    mse_value = np.mean((y - predicted) ** 2)
    grid = np.linspace(-0.2, 2.2, 100)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(x, y, s=65, color="#126782", label="Measurements", zorder=3)
    ax.plot(grid, intercept + slope * grid, color="#d46b32", label="Your model")
    ax.vlines(x, y, predicted, linestyles="dashed", color="gray", label="Residuals")
    ax.set(xlabel="Force (N)", ylabel="Extension (mm)", ylim=(-3, 12),
           title=f"MSE = {mse_value:.3f} mm²")
    ax.legend(); ax.grid(alpha=0.2); plt.show()

interact(line_experiment,
         slope=FloatSlider(value=1, min=-1, max=4, step=0.1, continuous_update=False),
         intercept=FloatSlider(value=1, min=-2, max=3, step=0.1, continuous_update=False));
```

Try to reach zero MSE. Then keep the slope correct and make the intercept wrong. Describe the residual pattern. Finally make the slope wrong: why do the errors grow with force?

## 7. What the computer “learns”

We choose a family of rules, here $\hat y=b+wx$. Training selects values for the **parameters**, $b$ and $w$, that reduce an objective calculated from data. The computer does not have to understand springs to find those numbers.

This separation will return throughout the course:

| Component | Question | Our example |
|---|---|---|
| Model | How are predictions made? | $b+wx$ |
| Parameters | What numbers are fitted? | $b,w$ |
| Loss | How do we score mistakes? | Squared residuals |
| Optimization | How are good parameters found? | A least-squares solver or gradient descent |
| Evaluation | Does the fitted model work on new examples? | Error on data not used to fit it |

### Checkpoint 4

For $x=[1,2]$, $y=[3,5]$, and $\hat y=2x$, calculate the predictions, residuals and MSE. Can zero error on these two measurements guarantee performance at $x=100$?

<details><summary>Worked answer</summary>

Predictions are $[2,4]$, residuals $[1,1]$, and MSE is $(1+1)/2=1$. Even an exact fit to two measurements would not establish how the spring behaves far away at 100 N. Extrapolation needs additional evidence and physical assumptions; the spring might leave its elastic regime.

</details>

## Ready to continue?

Without looking back, explain `y`, `y_hat`, a feature, a parameter, a residual and MSE. Write one prediction in Python. Explain why `(80,3)` is different from `(3,80)`. If you can do these, go to **02_linear_models**. If not, change the tiny numbers above and repeat the checkpoints.
