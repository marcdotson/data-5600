# 1. Predictors must be **standardized** for fitting penalized regression models
# 2. The **regularization hyperparameter(s)** must be tuned via cross-validation
# 3. Confidence intervals for parameter estimates must be obtained via **bootstrapping**

import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.utils import resample

# ---------------------------------------------------------------
# 3. Confidence intervals for parameter estimates must be obtained via **bootstrapping**

# Set randomization seed
rng = np.random.default_rng(42)

# Specify a function to simulate data
def sim_data(n, beta_0, beta_x1, beta_x2):
    x1 = rng.normal(10, 3, size=n)
    x2 = rng.binomial(1, 0.5, size=n)
    x3 = 2 * x1 + 3 * x2
    prob_y = (
      np.exp(beta_0 + beta_x1 * x1 + beta_x2 * x2) / 
      (1 + np.exp(beta_0 + beta_x1 * x1 + beta_x2 * x2))
    )
    y = rng.binomial(1, prob_y, size=n)
    return y, x1, x2, x3

# Simulate data
y, x1, x2, x3 = sim_data(n = 500, beta_0 = 0.25, beta_x1 = -0.15, beta_x2 = 0.75)

# Create a design matrix X
# X = np.column_stack([x1, x2, x3])
X = np.column_stack([x1, x2])

# # Split data into training (temp) and testing data
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size = 0.2, random_state = 42, stratify = y
# )

fit_01 = LogisticRegression().fit(X, y)

# Extract point estimates
intercept = fit_01.intercept_.ravel()
slopes = fit_01.coef_.ravel()
point_est = np.concatenate([intercept, slopes])

# Bootstrap confidence intervals
n_samples = 100
boot_est = np.empty((n_samples, len(point_est)))
for b in range(n_samples):
    # Resample data with replacement
    X_b, y_b = resample(X, y, replace=True, random_state=42 + b)
    
    # Fit logistic regression on resampled data
    fit_b = LogisticRegression().fit(X_b, y_b)
    
    # Extract point estimates using resampled data
    intercept_b = fit_b.intercept_.ravel()
    slopes_b = fit_b.coef_.ravel()
    point_est_b = np.concatenate([intercept_b, slopes_b])

    # Save point estimates using resampled data
    boot_est[b, :] = point_est_b

ci_lower = np.percentile(boot_est, 2.5, axis=0)
ci_upper = np.percentile(boot_est, 97.5, axis=0)

# Specify predictors
# predictors = ['x1', 'x2', 'x3']
predictors = ['x1', 'x2']

# Output of point and interval estimates
pl.DataFrame({
    'point_est': point_est,
    'ci_lower': ci_lower,
    'ci_upper': ci_upper
})

# import bambi as bmb
# import arviz as az

# # Combine y and X
# data = pl.DataFrame({
#     'y': y,
#     'x1': x1,
#     'x2': x2,
#     # 'x3': x3
# }).to_pandas()

# # Specify priors
# priors_dict = {
#     'Intercept': bmb.Prior('Normal', mu = 0, sigma = 10),
#     **{term: bmb.Prior('Normal', mu = 0, sigma = 1) for term in predictors}
# }

# # Fit a Bayesian logistic regression
# ba_model = bmb.Model(
#     'y ~ ' + ' + '.join(predictors), 
#     data = data,
#     family = 'bernoulli'
# )

# ba_model.set_priors(priors = priors_dict)
# ba_fit = ba_model.fit(progressbar = False)

# print(az.summary(ba_fit))
# ---------------------------------------------------------------

# 1. Predictors must be **standardized** for fitting penalized regression models
# 2. The **regularization hyperparameter(s)** must be tuned via cross-validation

# Create a design matrix X
X = np.column_stack([x1, x2, x3])
predictors = ['x1', 'x2', 'x3']

# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Create a pipeline to handle standardization and model fitting
ridge_pipe = Pipeline([
    ('feature_engineering', StandardScaler()),
    ('classification', LogisticRegression(penalty='l2'))
])

# Tune C using a log scale (inverse of regularization strength)
hyper_grid = {'classification__C': np.logspace(-10, 10, 30)}

kfold_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
tune = GridSearchCV(ridge_pipe, hyper_grid, scoring='accuracy', cv=kfold_cv, n_jobs=-1, refit=True)
tune.fit(X_train, y_train)

best_C = tune.best_params_['classification__C']
best_cv_score = tune.best_score_
best_estimator = tune.best_estimator_  # pipeline refit on full training set

# evaluate on test set
probs_test = best_estimator.predict_proba(X_test)[:, 1]
preds_test = (probs_test >= 0.5).astype(int)
test_accuracy = accuracy_score(y_test, preds_test)
test_auc = roc_auc_score(y_test, probs_test)

# collect results (useful object as final expression)
results = {
    "feature_names": predictors,
    "best_C": best_C,
    "best_cv_accuracy": best_cv_score,
    "best_estimator": best_estimator,
    "test_accuracy_0.5": test_accuracy,
    "test_auc": test_auc,
    "probs_test": probs_test,
    "preds_test": preds_test,
    "X_train": X_train,
    "y_train": y_train,
    "X_test": X_test,
    "y_test": y_test
}

results

# 3. Confidence intervals for parameter estimates must be obtained via **bootstrapping**
# Extract point estimates from refit best_estimator
ridge_final = best_estimator.named_steps['classification']
intercept = ridge_final.intercept_.ravel()
slopes = ridge_final.coef_.ravel()
point_est = np.concatenate([intercept, slopes])

# Bootstrap confidence intervals
n_samples = 100
boot_est = np.empty((n_samples, len(point_est)))
for b in range(n_samples):
    # Resample data with replacement
    X_b, y_b = resample(X, y, replace=True, random_state=42 + b)
    
    # Fit logistic regression on resampled data
    fit_b = LogisticRegression(penalty='l2', C=best_C).fit(X_b, y_b)
    
    # Extract point estimates using resampled data
    intercept_b = fit_b.intercept_.ravel()
    slopes_b = fit_b.coef_.ravel()
    point_est_b = np.concatenate([intercept_b, slopes_b])

    # Save point estimates using resampled data
    boot_est[b, :] = point_est_b

ci_lower = np.percentile(boot_est, 2.5, axis=0)
ci_upper = np.percentile(boot_est, 97.5, axis=0)

# Output of point and interval estimates
pl.DataFrame({
    'point_est': point_est,
    'ci_lower': ci_lower,
    'ci_upper': ci_upper
})

# 4. Visualize the bootstrap estimates
# 5. Compare with Bayesian estimation

