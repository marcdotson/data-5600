import numpy as np
import polars as pl
import seaborn.objects as so
from scipy.stats import norm

rng = np.random.default_rng(42)

# Simulate data
def sim_data(n_obs, beta_0, beta_1, sigma):
  X = rng.normal(0, 5, size = n_obs)
  y = beta_0 + beta_1 * x + rng.normal(0, sigma, size = n_obs)
  return y, X

y, X = sim_data(n_obs=100, beta_0=2, beta_1=3.5, sigma=1)

theta_support = np.arange(-100, 100, step = 1)
theta_density = np.zeros(len(theta_support))
for theta in range(len(theta_support)):
  # mu = -10 + theta_support[theta] * X
  # temp = norm.pdf(y, loc=mu, scale=1)
  # theta_density[theta] = np.mean(temp)
  mu = 2 + theta_support[theta] * X[0]
  theta_density[theta] = norm.pdf(y[0], loc=mu, scale=1)

# theta_df = pl.DataFrame({'theta': theta_density.ravel()})
# temp_df = pl.DataFrame(temp, schema = ['temp'])
theta_df = pl.DataFrame({'support': theta_support, 'density': theta_density})

(so.Plot(theta_df, x='support', y='density')
    .add(so.Area(), so.Hist())
    # .label(x='support', y='density')
)

# Infer parameters
def inf_theta(n_eval, y, X):
    n = len(y)
    
    return beta_0_hat, beta_1_hat, sigma_hat


# -----------------------------
# 3. Evaluate likelihood surface
# -----------------------------
beta_0_grid = np.linspace(0, 4, 60)
beta_1_grid = np.linspace(0.2, 1.4, 60)
sigma = 1.5  # fix sigma for visualization

loglik_values = []
for b0 in beta_0_grid:
    for b1 in beta_1_grid:
        ll = loglik_linear((b0, b1, sigma), y, x)
        loglik_values.append((b0, b1, ll))

import polars as pl
df_ll = pl.DataFrame(loglik_values, schema=["beta_0", "beta_1", "loglik"])

# Normalize for nicer color scale
df_ll = df_ll.with_columns((pl.col("loglik") - pl.col("loglik").max()).alias("rel_loglik"))

# -----------------------------
# 4. Visualize with seaborn.objects
# -----------------------------
p = (
    so.Plot(df_ll.to_pandas(), x="beta_0", y="beta_1", color="rel_loglik")
    .add(so.Contour(), levels=20)
    .label(x="β₀", y="β₁", color="Relative log-likelihood")
)
p.show()
plt.title("Log-likelihood surface for simple linear regression")
plt.show()

