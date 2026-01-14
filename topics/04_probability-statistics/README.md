# Probability and Statistics


During this class we will:

- Finish building models using probability
- Demonstrate fitting models to recover parameters

To share their solution for Exercise 03, the randomly selected student
is \*\*\_\_\_\*\*.

## Learn

Plotting distributions.

``` python
import numpy as np
import polars as pl
import seaborn.objects as so

# Set randomization seed and n
rng = np.random.default_rng(42)
n = 10_000_000

# Discrete uniform distribution
disc_unif = np.random.randint(1, 7, size=n)
draws = pl.DataFrame(disc_unif, schema=['disc_unif'])

disc_unif_p = (draws
    .group_by(pl.col('disc_unif'))
    .agg(n = pl.len())
    .with_columns(
        p = pl.col('n') / pl.col('n').sum()
    )
    .sort('disc_unif')
)

(so.Plot(disc_unif_p, x = 'disc_unif', y = 'p')
    .add(so.Bar())
    .label(x='support', y='p')
)

# Continuous uniform distribution
draws = (draws
    .with_columns(
        pl.Series('cont_unif', rng.uniform(1, 6, size=n)),
    )
)

(so.Plot(draws, x='cont_unif')
    .add(so.Area(alpha=0.5), so.Hist(stat='density'))
    .label(x='support', y='density')
)

# Normal distribution
draws = (draws
    .with_columns(
        pl.Series('norm', rng.normal(10, 5, size=n))
    )
)

(so.Plot(draws, x='norm')
    .add(so.Area(alpha=0.5), so.Hist(stat='density'))
    .label(x='support', y='density')
)

# Binomial distribution
draws = (draws
    .with_columns(
        pl.Series('binom', rng.binomial(1, 0.20, size=n))
    )
)

binom_p = (draws
    .group_by(pl.col('binom'))
    .agg(n = pl.len())
    .with_columns(
        p = pl.col('n') / pl.col('n').sum()
    )
    .with_columns(pl.col('binom').cast(pl.Utf8))
    .sort('binom')
)

(so.Plot(binom_p, x = 'binom', y = 'p')
    .add(so.Bar())
    .label(x='support', y='p')
)
```

Producing a normal likelihood.

``` python
import numpy as np
import polars as pl
import seaborn.objects as so

# Set randomization seed and n
rng = np.random.default_rng(42)
n = 100_000

# Normal likelihood
draws = (pl.DataFrame()
    .with_columns(
        pl.Series('norm', rng.normal(1.8, 3, size=n))
    )
)

(so.Plot(draws, x='norm')
    .add(so.Bars(), so.Hist(stat='density'))
    .label(x='beta', y='likelihood')
)
```

Fitting both a frequentist and a Bayesian model.

``` python
import numpy as np
import polars as pl
import statsmodels.formula.api as smf
import bambi as bmb
import arviz as az

# Set randomization seed
rng = np.random.default_rng(42)

# Specify a function to simulate data
def sim_data(n, beta_0, beta_x1, beta_x2, sigma):
    x1 = rng.normal(10, 3, size=n)
    x2 = rng.binomial(1, 0.5, size=n)
    error = rng.normal(0, sigma, size=n)
    y = beta_0 + beta_x1 * x1 + beta_x2 * x2 + error
    return y, x1, x2

# Call the function and save as an array
data_arr = sim_data(n = 100, beta_0 = 2, beta_x1 = -0.3, beta_x2 = 4, sigma = 5)

# Convert to a dataframe
data_df = pl.DataFrame(data_arr, schema = ['y', 'x1', 'x2']).to_pandas()

# Fit a frequentist linear regression
fr_fit = smf.ols('y ~ x1 + x2', data = data_df).fit()

fr_fit.params     # Point estimates
fr_fit.conf_int() # Interval estimates

# Fit a Bayesian linear regression
ba_fit = bmb.Model('y ~ x1 + x2', data = data_df).fit()

az.summary(ba_fit) # Posterior estimates
```

You can also download the slides as an .html file. Once you’ve previewed
the material and identified any questions, start watching the lecture.

Copy the Monte Carlo code for a distribution and try different parameter
values. How does the distribution change? Go the discussions to share
before continuing with the lecture.

Both frequentist and Bayesian statistics use probability to quantify
uncertainty, so how are they different? Go the discussions to share
before finishing the lecture.

## Apply

### Exercise 04

1.  Return to your simulated sales data from the previous exercise (or
    start with the exercise solution)
2.  Demonstrate that you can recover the parameters you used to simulate
    the data using either a frequentist or Bayesian model
3.  Reflect on how the interval estimates of the two approaches differ
    and how you might interpret each of them
4.  Submit your code, output, and reflection as a single PDF on Canvas

### Milestone 04

Continue working on simulating data that reflects the data generating
process/the ideal data for your project. Can you recover parameters for
your project’s simulated data using either a frequentist or Bayesian
model?
