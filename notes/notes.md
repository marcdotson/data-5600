# DATA 5600 Course Notes


Provided more opportunities for student participation beyond live coding
by including question slides as prompts for activities, including
discussing with a neighbor, raising hands, speed dialogue, and
whiteboarding concepts.

- Slow down, visualize/illustrate – especially with code, ask questions,
  activities
- Use minimal scales for grades (e.g., “Excellent,” “Good,” “Needs
  Improvement”)
- Create rubrics as a set of guiding questions rather than a checklist
- Learn names so you can call on students to share
- Create QR code for adding me on LinkedIn

# Week 01 (Aug 25, 27; Jan 5, 7; May 4, 6)

- Regression and Machine Learning (Aug 25; Jan 5; May 4)
- Modeling Workflow (Aug 27; Jan 7; May 6)

# Week 02 (Sept 3; Jan 12, 14; May 11, 13)

- Decisions and Data (Sept 3; Jan 12; May 11)

# Week 03 (Sept 8, 10; Jan 21; May 18, 20)

## Probability and Statistics (Sept 8; Jan 14; May 13)

A probabilistic approach to machine learning enhances the principled,
unifying framework provided by decision theory. This includes Bayesian
statistics, where we treat all unknowns as random variables and can
directly use probability distributions to quantify uncertainty in our
estimates, and frequentist statistics, where we treat data as random and
indirectly use probability distributions to quantify uncertainty in our
estimates. The direct, Bayesian approach is arguably more intuitive,
especially for applied students, while the indirect, frequentist
approach is less computationally intensive. The hope is that introducing
them in contrast and as complements will enhance student understanding
of modeling generally and combat the problem of student overfitting to a
set list of procedures.

- Probability as uncertainty: directly and indirectly, Bayesian and
  frequentist (PML p. 33)

- Try using the introduction from my teaching demo?

- Why teach both? If all you have is a hammer, everything looks like a
  nail. And it’s often easier to understand a thing in contrast.

- Deterministic vs. probabilistic models

- Interpretable models are typically parametric models – so not as
  flexible

- Probability as frequency assumes symmetry and repeatability

- Frequentist (objective) vs. Bayesian (subjective) interpretations of
  probability

- Intercept-only model – no story – intercept or bias

- Using linear models to formalize this story and probability
  distributions to quantify uncertainty

- No free lunch theorem: which model is best depends on the application

- Example likelihoods (as models of the data generating process)

### Uncertainty

Recover parameters when there is only one kind of uncertainty recovering
parameters? MLE and Bayes and Bootstrap? (Build, Fit, Evaluate?)

- Probability as a unifying framework for machine learning (PML p. 1)

- Bayes vs. frequentist probability?

- Frequentist uncertainty is based on imaginary/theoretical resampling
  of data (sampling distributions)

- So only measurements – estimators can have probability distributions
  (sampling)

- Bayesian uncertatinytreats randomness as a property of information,
  not the world

- Use randomness to describe our uncertainty in the face of incomplete
  knowledge

- Uncertainty from not knowing the true data generating process
  vs. naturally occuring uncertainty in the data (PML p. 7, 33-34)

- Axioms and basic definitions, set theory

### Random Variables and Distributions

- Use an experiment to illustrate the difference between random and
  fixed variables: Need something fixed to learn about the random
  variable

- RVs, discrete and continous, PMF and PDF

- Distributions, supports

- Normal vs. uniform distributions, etc.

- Joint and marginal distributions

- Expectations, conditional expectation; Means, variance, standard
  deviation, mode, limitations of summaries (PML p. 43)

- Bayes rule

- Why Normal? (PML p. 60)

- Activity: Speed dating to explain critical concept(s)

### Recovering Parameters

- CLT? Monte Carlo? (PML p. 71-72)
- Covariance and correlation?
- Correlation does not imply causation, and the lack of correlation does
  not imply lack of causation
- Data and parameters $f(X; \theta)$

Since the data generating process is the unobserved process that
generates the data, before we work with real data where we never know
the data generating process, we can assume that our model *is* the data
generating process, *choose* values for the parameters, and generate or
**simulate** data using the model.

For example, we can choose values for $\beta_0$ and $\beta_1$, simulate
possible `promotion_spend` data and then simulate `sales` data.

Why? Two big reasons:

1.  Prepare our data analysis before getting real data.
2.  Prove our code is working by recovering parameter values.

- Wrap up

### Statistics

- Learning parameters from data, estimating parameters, model fitting,
  training, calibration

- Likelihood as our data generating process, the evidence for the data
  we see?

- MLE: Pick the parameters that assign the highest probability to the
  training data (PML p. 105)

- MLE as a point approximation of the posterior distribution with a
  uniform prior (PML p. 106)

- MLE to OLS

- Points, intervals, and distributions

- Bayes: Grid Approximation to ABC?

- Sampling distributions vs. posterior distributions, bootstrap as poor
  man’s posterior

- Interpreting parameter estimates

- Confidence intervals and bootstrapping

See https://allendowney.github.io/ElementsOfDataScience/ See
https://allendowney.github.io/ThinkStats/ See
https://sta210-s22.github.io/website/

- Exercise: Produce a meme summarizing a key statistics concept

### Exploratory Data Analysis?

- Continuous and Discrete Variables?

## Linear Regression (Sept 10; Jan 21; May 18)

- Regression is also referred to as linear regression or, more
  generally, a linear model

- Key property is the expected value (mu) of the output is assumed to be
  a linear function of the inputs (X)

- Line isn’t great, but we can do non-linear transofmrations on the
  predictors while still keeping a linear model

- SLR, MLR

- Simulating data and recovering parameters

### Module 1 (Slides 1-17, Slides 18-38)

Correlation coefficient Outcome vs. explanatory variables Definition of
SLR Deterministic vs. probabilistic Parameter definitions Error is a
function of unknown parameters. A residual is an “estimate” of this
error. Interpreting the slope parameter Residuals OLS intuition Danger
with extrapolating MSE as an estimate of variance Code: Scatterplots,
correlation, OLS, and MSE by hand

### Module 2 (Slides 1-8)

Diagnostics for assumptions before inference How is the iid assumption
formally described? Does iid persist in the case of grouped
observations? Using graphical diagnostics of assumptions

# Week 04 (Sept 15, 17; Jan 26, 28; May 27) POSIT CONF

## Continuous Predictors

- Exploration and preparation

- Feature engineering, transformations

- Training/testing split

- Flexible ML is about automating feature engineering

- Be sure to include +1 for log transforms

- Rescaling/normalizing/standardizing predictors

## Discrete Predictors

- Exploration and preparation
- Dummy/one-hot and index coding

### Module 2 (Slides 9-34, 35-54)

Are all these assumptions really required to use regression? Why does
the Q-Q plot say “standardized residuals”? Where did DFBETA and DFFITS
come from? Using numerical diagnostics of assumptions How to remediate
invalidated assumptions Interpreting estimates with transformations
Code: All graphical and numerical diagnostics

# Week 05 (Sept 22, 24; Feb 2, 4; Jun 1, 3)

## Assumptions and Diagnostics

- Validity: Data is relevant to the objective, no missing variables.

- Representativeness: Data is representative of the population or
  process.

- Additivity: The relationship between the outcome and predictors is
  additive.

- Linearity: The mean is a linear function of the predictors.

- Independence: Observations are independent of each other.

- Constant Variance: Homoscedasticity or constant variance of errors
  across all levels of predictors.

- Normality of Errors: Errors are normally distributed?

- Omitted and included variable bias

- Multicollinearity

## Fitting and Interpreting Models

- Parameter estimates
- Significance, confidence intervals, and p-values

### Module 3 (Slides 1-6, 7-28)

Using the CLT, assuming null is true, to test hypotheses Computing and
using the standard error for hypothesis testing Manually computing a
test statistic and looking up a p-value Probability of observing
something as or more extreme, assuming the null is true Manually
computing a confidence interval using the t-distribution and margin of
error Equivalence of p-values and confidence intervals Confidence
intervals are about uncertainty in parameter estimates (i.e., parameters
are fixed)

# Week 06 (Sept 29, Oct 1; Feb 9, 11; Jun 17)

## Model Evaluation and Prediction

- In-sample vs. out-of-sample vs. decision theoretic evaluation
- Overfitting and underfitting, variance vs. bias tradeoff
- Use theory-model-evidence.png

## Communicating Results

- Demonstrate presenting on a project as part of the communication
  session

### Module 3 (Slides 29-36)

Why do we compute a confidence band for the average of y? How do we get
the confidence band out of a fixed confidence interval? Prediction
intervals are about uncertainty in new data (i.e., new data is random)
Why even produce a confidence interval around the mean if it isn’t for a
confidence band? What’s the extra term in the standard error for the
prediction interval? SD used twice. Are prediction intervals different
in Bayesian statistics since data is fixed? Properties of MSE, RMSE,
MAE, R-squared, adjusted R-squared, and F statistic Code: Confidence
intervals, test statistics, p-value, prediction intervals, confidence
and prediction bands, and model fit statistics

### Module 4 (Slides 1-16)

Interpreting multiple slopes Adding multicollinearity to the list of
assumptions

# Week 07 (Oct 6, 8; Feb 18; Jun 22, 24)

Presentations

—– CASE TWO —–

- Student evaluations
- Student panel invitation
- Post-quiz for research project

# Week 08 (Oct 13, 15; Feb 23, 25; Jun 29, Jul 1)

### Module 4 (Slides 17-39, 40-55)

Interpreting the F-test and model vs. coefficient p-values What is
multiplicity or the multiple comparisons problem? Using partial
regression to have multiple regression diagnostics Underfitting and
overfitting Code: Create a scatterplot matrix, correlation matrix (plus
a heat map), fitting multiple linear regression, diagnostics (including
partial regression plots)

# Week 09 (Oct 20, 22; Mar 2, 4; Jul 6, 8)

## Regularization

- Penalize the regression to learn “regular” (i.e., generalizable)
  features
- Shrinkage toward the MLE

## Hyperparameter Tuning

## Module 5 (Slides 1-12, 13-38)

Stepwise regression for model selection when focusing on prediction
Discussed the possibility of p \> n Working from a full model and using
backward or forward-selection No discussion on overfitting needing
predictive fit? Is stepwise regression used in ML practice?
Cross-validation without one static test dataset? What about leakage?
Are variable selection and shrinkage methods primarily used for
multicollinearity? AIC and BIC

# Week 10 (Oct 27, 29; Mar 16, 18; Jul 13, 15)

## Feature Engineering

## Dummy Coding

## Module 5 (Slides 39-59)

Ridge regression, LASSO, elastic net, best subsets, sequential
replacement Bias-variance tradeoff and shrinkage methods Accuracy and
precision is just bias and variance Code: Clunky, implementing stepwise
regression and shrinkage methods with manual hyperparameter tuning

## Module 6 (Slides 1-14)

Dummy variables and the “dummy variable trap” Interpretation, synonyms,
etc.

# Week 11 (Nov 3, 5; Mar 23, 25; Jul 20, 22)

## PCR

## Module 6 (Slides 15-27, 28-58)

Interactions (including higher-order interactions), including
interactions Why do we just interpret interaction effects and not main
effects? Does this all have to be done with the statsmodels API? What
about https://www.scikit-yb.org/en/latest/index.html? What about models
built to bring assumption verification, etc. to scikit-learn? Code: EDA,
interactions plots, dummy coding, creating interactions

# Week 12 (Nov 10, 12; Mar 30, Apr 1; Jul 27, 29)

## Multilevel Models

- Motivate with Simpson’s paradox (PML p. 80)

## Module 7 (Slides 1-21, 22-35; Aug 3, 5)

Logistic regression basics, including maximum likelihood estimation Why
not introduce training/testing split earlier without cross-validation?
Probability vs. (Log) Odds

# Week 13 (Nov 17, 19; Apr 6, 8)

## GLMs

## Module 7 (Slides 36-54, 55-61)

Logistic regression assumptions and diagnostics If the assumptions we
used previously break down, why use them at all? Interpreting a
confusion matrix Code: Splitting the data into train and test, maximum
likelihood, ROC/AUC

# Week 14 (Nov 24; Apr 13, 15)

## Logistic Regression

## Module 8 (Slides 1-15, 16-36)

Dimension reduction as opposed to variable selection and regularization
Should this all be in a “feature engineering” module? Shouldn’t we have
decision trees and random forests as well/instead? Are there students
who *only* take the introduction to ML? PCA on its own and then PCA as
part of PCR Code: Using PCA and PCR for continuous and discrete outcomes

# Week 15 (Dec 1, 3; Apr 20)

Presentations
