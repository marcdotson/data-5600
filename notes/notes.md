# DATA 5600 Course Notes


Provided more opportunities for student participation beyond live coding
by including question slides as prompts for activities, including
discussing with a neighbor, raising hands, speed dialogue, and
whiteboarding concepts.

- Slow down and illustrate – especially with case, code, questions, and
  activities
- Use minimal scales for grades (e.g., “Excellent,” “Good,” “Needs
  Improvement”)
- Create rubrics as a set of guiding questions rather than a checklist
- Learn names so you can call on students to share
- Create QR code for adding me on LinkedIn
- Produce a meme to summarize a concept
- Speed dating to explain concepts

## Week 01 (Aug 25, 27; Jan 5, 7; May 4, 6)

- Regression and Machine Learning (Aug 25; Jan 5; May 4)
- Modeling Workflow (Aug 27; Jan 7; May 6)

## Week 02 (Sept 3; Jan 12, 14; May 11, 13)

- Decisions and Data (Sept 3; Jan 12; May 11)

## Week 03 (Sept 8, 10; Jan 21; May 18, 20)

- Probability and Statistics (Sept 8; Jan 14; May 13)
- Linear Models (Sept 10; Jan 21; May 18)

# Week 04 (Sept 15, 17; Jan 26, 28; May 27)

- Diagnostics and Remedies (Part 01) (Sept 15; Jan 26; May 27)
- Diagnostics and Remedies (Part 02) (Sept 17; Jan 28; Jun 1)

# Week 05 (Sept 22, 24; Feb 2, 4; Jun 1, 3)

- Feature Engineering and Ordinary Least Squares (Sept 22; Feb 2; Jun 3)
- Frequentist and Bayesian Inference (Sept 24; Feb 4; Jun 17)

## Estimation

- MLE as a decision theoretic extension of OLS (p. 8-9 of PML, p. 105 of
  PML, p. 143 of PML?)
- Using OLS to estimate the parameters in a linear regression is
  equivalent to using Bayesian inference with a \_\_\_\_\_\_ prior.
  (Show with the two equation forms.)
- Illustrate how a posterior is an updated version of the prior taking
  into the account the likelihood
- Sampling distributions vs. posterior distributions

## Parameter Estimates

- Parameter estimates

- Statistical significance

- Points, intervals, bootstrap, and distributions (estimate, estimator,
  estimand meme) – danger of summarizing/summary statistics?

- That this is an illustration of shrinkage that happens automatically
  (p. 89 of PML)

- Bootstrap as poor man’s posterior

- Significance, confidence intervals, and p-values

- Probability of observing something as or more extreme, assuming the
  null is true

- Manually computing a confidence interval using the t-distribution and
  margin of error

- Equivalence of p-values and confidence intervals (meme)

- Confidence intervals are about uncertainty in parameter estimates
  (i.e., parameters are fixed)

## Interpretations

- Create a chart showing the differences between Bayesian and
  frequentist statistics?
- Statistical models capture association, not causation
- Correlation does not imply causation, and the lack of correlation does
  not imply lack of causation
- Do I need matrix multiplication – does that notation work in both
  statsmodels and Bambi?

# Week 06 (Sept 29, Oct 1; Feb 9, 11; Jun 17)

## Model Evaluation and Prediction

- Overall model fit
- Comparing predictions and real data
- No free lunch theorem: which model is best depends on the application?

Properties of MSE, RMSE, MAE, R-squared, adjusted R-squared, and F
statistic Code: Confidence intervals, test statistics, p-value,
prediction intervals, confidence and prediction bands, and model fit
statistics

Why do we compute a confidence band for the average of y? How do we get
the confidence band out of a fixed confidence interval? Prediction
intervals are about uncertainty in new data (i.e., new data is random)
Why even produce a confidence interval around the mean if it isn’t for a
confidence band? What’s the extra term in the standard error for the
prediction interval? SD used twice. Are prediction intervals different
in Bayesian statistics since data is fixed?

- In-sample vs. out-of-sample vs. decision theoretic evaluation
- Overfitting and underfitting, variance vs. bias tradeoff
- Use theory-model-evidence.png

## Communicating Results

- Demonstrate presenting on a project as part of the communication
  session
- Time to work on the project in class?

# Week 07 (Oct 6, 8; Feb 18; Jun 22, 24)

Presentations

—– CASE TWO —–

- Update the syllabus to reflect the changed schedule topics
- Student evaluations
- Student panel invitation
- Post-quiz for research project

# Week 08 (Oct 13, 15; Feb 23, 25; Jun 29, Jul 1)

- Week 09 notes from pre-PhD seminar

### Module 4 (Slides 17-39, 40-55)

Interpreting the F-test and model vs. coefficient p-values Underfitting
and overfitting, the complexity vs. error plot (bias variance tradeoff?)
to navigate between the two

# Week 09 (Oct 20, 22; Mar 2, 4; Jul 6, 8)

## Regularization

- Penalize the regression to learn “regular” (i.e., generalizable)
  features
- Shrinkage toward the MLE

Why Bayes? Carefully and directly model uncertainty This prior
predictive distribution is the expected distribution of our data, given
how we’ve specified our likelihood and priors. Does this look
reasonable? No one has a negative height, for a start. At this point we
can iterate on how we’ve specified our likelihood and priors, produce
another prior predictive distribution and evaluate again, etc.

> “Prior predictive simulation is very useful for assigning sensible
> priors, because it can be quite hard to anticipate how priors
> influence the observable variables.”

Does Bambi have an easy way to do prior predictive checks? Or just
expand our Monte Carlo simulation? Facet in seaborn.objects to compare
the distribution of the outcome vs. the prior predictive check

## Hyperparameter Tuning

## Module 5 (Slides 1-12, 13-38)

Stepwise regression for model selection when focusing on prediction
Discussed the possibility of p \> n Working from a full model and using
backward or forward-selection No discussion on overfitting needing
predictive fit? Is stepwise regression used in ML practice?
Cross-validation without one static test dataset? What about leakage?
Cross-validation is needed for hyperparameter tuning, and model
selection is a form of “hyperparameter tuning” Cross-validation for
model selection as well in order to keep a larger training dataset
(p. 123 of PML) Are variable selection and shrinkage methods primarily
used for multicollinearity? AIC and BIC

# Week 10 (Oct 27, 29; Mar 16, 18; Jul 13, 15)

## Feature Engineering

## Dummy Coding

## Module 5 (Slides 39-59)

Ridge regression, LASSO, elastic net, best subsets, sequential
replacement Bias-variance tradeoff and shrinkage methods Accuracy and
precision is just bias and variance Code: Clunky, implementing stepwise
regression and shrinkage methods with manual hyperparameter tuning

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
PCA on its own and then PCA as part of PCR Code: Using PCA and PCR for
continuous and discrete outcomes

# Week 15 (Dec 1, 3; Apr 20)

Presentations
