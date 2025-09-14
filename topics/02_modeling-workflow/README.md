# Modeling Workflow


Running statistical models.

``` python
import os
import polars as pl
import seaborn.objects as so
import statsmodels.formula.api as smf
import bambi as bmb

# Import data
sl_data = pl.read_parquet(os.path.join('data', 'original_df.parquet'))

# Peanut butter purchases
pb_data = sl_data.filter(pl.col('units') > 0)

# Visualize price and units
(so.Plot(pb_data, x = 'price', y = 'units')
    .add(so.Dot(alpha = 0.2), so.Jitter(x = 0.2))
    .add(so.Line(), so.PolyFit(order=1))
)

# Preprocess data
lm_data = pb_data.to_dummies(
    # Dummy code brand
    columns = ['brand'], drop_first = True
).with_columns(
    # Log units and price
    (pl.col('units').fill_nan(0) + 1).log().alias('log_units'),
    (pl.col('price') + 1).log().alias('log_price')
).select(
    ['log_units', 'log_price', 'brand_Skippy', 'brand_PeterPan', 'brand_Harmons']
).to_pandas()

# Fit a frequentist linear regression
fr_fit = smf.ols('log_units ~ log_price + brand_Skippy + brand_PeterPan + brand_Harmons', data = lm_data).fit()

# Fit a Bayesian linear regression
ba_fit = bmb.Model('log_units ~ log_price + brand_Skippy + brand_PeterPan + brand_Harmons', data = lm_data).fit()
```

Producing a visualization comparing frequentist and Bayesian estimates.

``` python
import arviz as az
import numpy as np
import matplotlib.pyplot as plt

# Comparing OLS and Bayesian estimates
# --- 1) Pull the OLS point estimate and 95% CI for log_price
ols_coef = fr_fit.params['log_price']
ols_ci_low, ols_ci_high = fr_fit.conf_int().loc['log_price']

# --- 2) Get posterior draws for log_price from the Bambi fit
# Convert the xarray to a flat 1-D numpy array
post = ba_fit.posterior['log_price'].stack(sample=('chain','draw')).values.flatten()

# --- 3) Plot: posterior density + OLS CI band and point
fig, ax = plt.subplots(figsize=(7, 4))

# Posterior density (ArviZ helper)
az.plot_kde(post, ax=ax)

# OLS 95% CI as a vertical band, and the OLS point estimate
ax.axvspan(ols_ci_low, ols_ci_high, alpha=0.2, label='OLS 95% CI')
ax.axvline(ols_coef, linestyle='--', linewidth=2, label='OLS estimate')

# Cosmetics
ax.set_title('Frequentist Confidence Interval vs. Bayesian Posterior Distribution')
ax.set_xlabel('Parameter Estimate')
ax.set_ylabel('Density')
ax.legend()
plt.tight_layout()
plt.show()
```
