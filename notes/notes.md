# DATA 5600 Course Notes


# What I Did

I revised the course consistent with the pedagogical research begun last
year to study the effect of a probabilistic and decision theoretic
approach to machine learning on student confidence and understanding.
This included framing the course with decision theoretic objectives and
a probabilistic modeling workflow and referring to this unifying
framework throughout the semester. I also responded to student and peer
feedback.

- Organized the course better, including using the decision theoretic
  objectives and probabilistic modeling workflow as a unifying
  framework.
- Replaced homework assignments with exercises leading up to two instead
  of three projects. Exercises provide more consistent practice and
  include opportunities for reflection and preparation for upcoming
  class topics.
- Detailed two business cases, one for each unit, to reduce the
  cognitive burden of switching between many datasets and allowing for
  greater depth for each application to better inform student project
  work.
- Clarified the expectations for each project by ensuring the rubrics
  make the purpose, task, skills, and criteria clear and by having a
  single initial submission deadline so students can get feedback before
  presenting and so students who present later in the week don’t have an
  advantage over others.
- Revised the course content to make it my own, including slides to
  preview and summarize each lecture.
- Included more live coding in class instruction (as well as discussing
  error messages) and by starting class with a student called on at
  random to share their solution to the exercise due for that class.
- Provided more opportunities for student participation beyond live
  coding by including question slides as prompts for activities,
  including discussing with a neighbor, raising hands, speed dialogue,
  and whiteboarding concepts.

# Assessment

- Exercises (20%): At the beginning of each class, a student will be
  called on at random to walk through and share their solution.
  Additionally, for each exercise every student will be randomly
  assigned to review one other student’s exercise. Students won’t get
  credit if they don’t submit their solution on time, aren’t present and
  prepared to share when called on at random, or don’t complete their
  randomly assigned review.
- Interviews (30%): During the first two weeks of class, during the two
  weeks in the middle of the semester, and during the final two weeks of
  class each student will have a required 10-minute interview with me
  where we discuss the course and their efforts and evaluate their
  understanding in the form of an oral exam.
- Projects (50%): Students will complete two group projects, one focused
  on multiple regression and the second focused on multiple regression
  or classification. The groups will both present and submit a technical
  report developed on a GitHub repository. A week before the
  presentations, they will submit a draft of their slides to get
  feedback and have time for revision. The other students in the class,
  as well as the group members themselves, will help evaluate each of
  the presentations.

# Week 01 (Aug 25, 27; Jan 5, 7)

## Regression and Machine Learning (for Analytics) (Aug 25; Jan 5)

### Introductions

- Get started (schedule)
- What is your major? What business problems interest you?
- Personal introduction
- Activity: Form groups (of size X), move to sit together, discuss
  interests

### Course Overview

- Analytics is using data to inform decisions in the presence of
  uncertainty
- Summarize the course objectives, skills
- CV entry (also in the syllabus)
- What makes this course different from DATA 3100 and 3300?
- What makes this course different from DATA 3500?
- Discuss DATA 5610, 5620, and 5630
- How to learn and the place of assessments
- Breakdown of grades: At this level, your assessmenmts should be
  aligned with what you will be expected to do in practice
- Reference data-stack as recommended tools for the course
- Discuss the Goldilocks zone for AI use, reference Copilot
- Getting help
- Activity: Look at syllabus, discuss what you’re excited/nervous about,
  questions

### Stories

- Mathematics, Statistics, Economics; Computer Science, AI, Machine
  Learning; and Data Analytics
- This fractured history is a reason why there are is so much
  terminology to keep track of
- What is regression and ML? What is the relationship and overlap?
- Data analytics vs. data mining vs. data science (PML p. 27-28)
- Supervised vs. unsupervised vs. reinforcement learning
- Supervised learning is learning a mapping function from inputs to
  outputs $f: X \rightarrow Y$
- Regression and classification, $y \in \\R$
  vs. $y \in \{1, \ldots, C\}$
- Descriptive statistics, inferential statistics, and decision theory?
- Models *extract information* from the data to inform decisions in the
  presence of uncertainty
- All models predict, some are interpretable, fewer still are causal; is
  the model consistent with your objective?
- Business applications for analytics
- Exercise: Read case and summarize the objective and the ideal data;
  narrate a business problem, it’s objective, and the ideal data
- Wrap up

## Modeling Workflow (Aug 27; Jan 7)

### Before Data

- Get started (solution, schedule, and workflow)
- High-level overview of the (interpretable) modeling workflow
- Focus on being transparent and direct with objectives and assumptions
- Plan/Narrate: objective/loss function, data generation process/ideal
  dataset
- Stories – modeling as storytelling, starts as a narrative and then
  gets mathematized
- Build/Assemble: Translate story into mathematical models – a
  functional mapping of inputs to output and how you will evaluate it
  (loss function)
- Activity: Discuss with your groups what kind of loss function, data
  generating process makes sense for your problem

### Using Data

- Explore: Summarize and visualize the data, data dictionary; X as
  design matrix, N and P
- Reconcile: Data and Model. Preprocess the data (what
  transformations?), compare the data to what you’ve simulated to see if
  you’re missing something from your model. Check assumptions, specific
  to linear models or other kinds of models (or is that just part of
  prepare?). Careful with overfitting. Sensitivity analysis.
- Fit: Fit/train/calibrate the model on training data (lots of different
  models, libraries we can use – needs to be consistent with our
  objective)
- Evaluate: Parameter estimates (including uncertainty), considering
  significance, and overall (predictive) model fit on testing data
- Predict: Use the model to make predictions on new data, including
  uncertainty in those predictions
- Activity: Speed dialogue to discuss favorite libraries you use for
  working with data

### Communicating Results

- Communicate: Report and present the results in a way that is
  understood by a mixed audience
- Everything you’ve worked on is to inform the managerial decision, so
  if you don’t it’s like giving up at the end of the race
- Discuss good and bad examples of communicating modeling results
- Exercise: Finish setting up a data stack, write about which part of
  the modeling workflow you’re most/least comfortoable with, use Quarto
  to render the document
- Wrap up

# Week 02 (Sept 3; Jan 12, 14)

## Decisions and Data / Loss and Likelihood Functions / Payoffs, Losses, and Likelihoods (Sept 3; Jan 12)

### Decision Theory

- Get started (solution, schedule, and workflow)
- Introduce the case and start illustrating in more detail the modeling
  workflow
- Illustrate decision theory with a pros and cons list? then a
  payoff/loss matrix? maximize beneift/minimize loss
- Decision theory is a principled, unifying framework for informing
  decision making in the presence of uncertainty
- Actions, states of the world, objective function (payoff/loss)
  (starting on BID p. 239)
- Activity:

### Data Generating Process

- Stories – modeling as storytelling, starts as a narrative and then
  gets mathematized
- Reference DAGs in DATA 5620
- Using linear models to formalize this story and probability
  distributions to quantify uncertainty
- Models are simplifications of reality, but the goal is to capture the
  essence of the data generation process
- No free lunch theorem: which model is best depends on the application
- Activity:

### Loss and Likelihoods Functions

- Example loss functions – or just focus on a linear profit function?
- Risk neutral, risk averse, risk seeking loss functions (BID p. 227)
- Maximize profit, maximize market share, minimize churn, etc.
- What are the inputs into the function? Parameters – the states of the
  world – along with what actions to take?
- Example likelihoods (as models of the data generating process)
- Generative models, use to generate or simulate data to evaluate
  assumptions, prepare for an analysis, test code (recover parameters)
- Signal simulating data
- Exercise:
- Wrap up

# Week 03 (Sept 8, 10; Jan 21)

## Probability and Statistics / Probabilistic Machine Learning (Sept 8; Jan 14)

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

### Uncertainty

- Get started (solution, schedule, and workflow)
- Probability as a unifying framework for machine learning (PML p. 1)
- Bayes vs. frequentist probability?
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

### 

- CLT? Monte Carlo? (PML p. 71-72)
- Covariance and correlation?
- Correlation does not imply causation, and the lack of correlation does
  not imply lack of causation
- Data and parameters $f(X; \theta)$
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

## Linear Regression (Sept 10; Jan 21)

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

# Week 04 (Sept 15, 17; Jan 26, 28) POSIT CONF

## Continuous Predictors

- Exploration and preparation
- Feature engineering, transformations
- Be sure to include +1 for log transforms

## Discrete Predictors

- Exploration and preparation
- Dummy/one-hot and index coding

### Module 2 (Slides 9-34, 35-54)

Are all these assumptions really required to use regression? Why does
the Q-Q plot say “standardized residuals”? Where did DFBETA and DFFITS
come from? Using numerical diagnostics of assumptions How to remediate
invalidated assumptions Interpreting estimates with transformations
Code: All graphical and numerical diagnostics

# Week 05 (Sept 22, 24; Feb 2, 4)

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

# Week 06 (Sept 29, Oct 1; Feb 9, 11)

## Model Evaluation and Prediction

- In-sample vs. out-of-sample vs. decision theoretic evaluation
- Overfitting and underfitting

## Communicating Results

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

# Week 07 (Oct 6, 8; Feb 18)

Presentations

—– CASE TWO —–

- Student evaluations
- Student panel invitation
- Post-quiz for research project

# Week 08 (Oct 13, 15; Feb 23, 25)

### Module 4 (Slides 17-39, 40-55)

Interpreting the F-test and model vs. coefficient p-values What is
multiplicity or the multiple comparisons problem? Using partial
regression to have multiple regression diagnostics Underfitting and
overfitting Code: Create a scatterplot matrix, correlation matrix (plus
a heat map), fitting multiple linear regression, diagnostics (including
partial regression plots)

# Week 09 (Oct 20, 22; Mar 2, 4)

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

# Week 10 (Oct 27, 29; Mar 16, 18)

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

# Week 11 (Nov 3, 5; Mar 23, 25)

## PCR

## Module 6 (Slides 15-27, 28-58)

Interactions (including higher-order interactions), including
interactions Why do we just interpret interaction effects and not main
effects? Does this all have to be done with the statsmodels API? What
about https://www.scikit-yb.org/en/latest/index.html? What about models
built to bring assumption verification, etc. to scikit-learn? Code: EDA,
interactions plots, dummy coding, creating interactions

# Week 12 (Nov 10, 12; Mar 30, Apr 1)

## Multilevel Models

- Motivate with Simpson’s paradox (PML p. 80)

## Module 7 (Slides 1-21, 22-35)

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
