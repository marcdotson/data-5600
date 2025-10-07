# DATA 5600 Course Notes


Provided more opportunities for student participation with question
slides as prompts for activities, including coding, reflection,
discussing with a neighbor, raising hands, speed dialogue, and
whiteboarding concepts.

- Slow down and illustrate – especially with case, code, questions, and
  activities
- Use minimal rubric for grades (e.g., “Excellent,” “Good,” “Needs
  Improvement”)
- Have the exercises map to the project and project rubric map to the
  modeling workflow
- Learn names so you can call on students to share

TODO:

- Add iceberg illustration of projects: presentations up top and report
  underneath
- Include use the workflow explicitly as part of the project report
  expectations
- Add weekly milestones for each project?
- Exercise 02: Have this exercise tied to the case and not the project.
- Lecture 03: Didn’t get to do the second activity, and this is when we
  need to rely heavily on activities to provide project and
  case-specific context.
- Do Feature Engineering before model assumptions, including
  log-transformations, dummy-coding, and interaction terms. Not as much
  EDA.
- Divide assumptions into deterministic and probabilistic assumptions?
- Figure out a way to simplify and streamline assumption evaluation.
  Provide functions – something!
- Compare more before and after plots when the assumptions are violated
  and when they are satisfied using the same tools provided.
- Demonstrate likelihood being used for inference as the inverse of it
  being used for simulation as part of OLS-as-special-case-of-MLE
  discussion.
- Exercise 11: Simplify or spread out to previous exercises

## Week 01 (Aug 25, 27; Jan 5, 7; May 4, 6)

- Regression and Machine Learning (Aug 25; Jan 5; May 4)
- Modeling Workflow (Aug 27; Jan 7; May 6)

## Week 02 (Sept 3; Jan 12, 14; May 11, 13)

- Decisions and Data (Sept 3; Jan 12; May 11)

## Week 03 (Sept 8, 10; Jan 21; May 18, 20)

- Probability and Statistics (Sept 8; Jan 14; May 13)
- Linear Models (Sept 10; Jan 21; May 18)

## Week 04 (Sept 15, 17; Jan 26, 28; May 27)

- Diagnostics and Remedies (Part 01) (Sept 15; Jan 26; May 20)
- Diagnostics and Remedies (Part 02) (Sept 17; Jan 28; May 27)

## Week 05 (Sept 22, 24; Feb 2, 4; Jun 1, 3)

- Feature Engineering and Ordinary Least Squares (Sept 22; Feb 2; Jun 1)
- Frequentist and Bayesian Inference (Sept 24; Feb 4; Jun 3)

## Week 06 (Sept 29, Oct 1; Feb 9, 11; Jun 17)

- Model Evaluation and Prediction (Sept 29; Feb 9; Jun 17)
- Communicating Results (Oct 1; Feb 11; Jun 22)

## Week 07 (Oct 6, 8; Feb 18; Jun 22, 24)

- Presentations (Oct 6, 8; Feb 18; Jun 24)

## Week 08 (Oct 13, 15; Feb 23, 25; Jun 29, Jul 1)

- Loss and Utility Functions (Oct 13; Feb 23; Jun 29)
- Generalized Linear Models (Oct 15; Feb 25; Jul 1)

### Decision Making Under Uncertainty

- Need to revisit the modeling workflow and the place of decision theory
  as a frame for it.
- Use this discussion to illustrate the second case and to motivate the
  next project.

Wald’s basic intuition was that statistics is nothing but the science of
decision making under uncertainty. Statistical problems should be
considered as special instances of general decision problems, where
decisions have to be taken in the face of uncertainty. “Acts have
consequences for the actor, and these consequences depend on facts, not
all of which are generally known to him. The unknown facts will often be
referred to as states of the world” -Jimmy Savage

- Generalize loss and utility functions to construct non-symmetric or
  non-linear utility functions.
- Statistics to inform what to *do* not just what to *say*.
- Reference to Gossett, Student T.

### Generalized Linear Models

- Generalize Linear Models to GLMs
- GLMs are parametric models – the number of parameters don’t increase
  with n
- No free lunch theorem: which model is best depends on the application?
- Week 09 notes from pre-PhD seminar

## Week 09 (Oct 20, 22; Mar 2, 4; Jul 6, 8)

- Logistic Regression (Oct 20; Mar 2; Jul 6)
- Maximum Likelihood Estimation (Oct 22; Mar 4; Jul 8)

### Logistic Regression

- Most common GLM
- Revisit assumptions and reconciliation
- Logistic regression assumptions and diagnostics
- If the assumptions we used previously break down, why use them at all?
- Interpreting a confusion matrix
- Prior predictive checks for helping to set priors
- This prior predictive distribution is the expected distribution of our
  data, given how we’ve specified our likelihood and priors. Does this
  look reasonable? No one has a negative height, for a start. At this
  point we can iterate on how we’ve specified our likelihood and priors,
  produce another prior predictive distribution and evaluate again, etc.
- Does Bambi have an easy way to do prior predictive checks? Or just
  expand our Monte Carlo simulation?
- Facet in seaborn.objects to compare the distribution of the outcome
  vs. the prior predictive check

### Maximum Likelihood Estimation

- Generalize OLS to MLE, Generalize Confidence Intervals to Bootstrap
- Illustrate MLE vs. posterior, MLE as an approximation to a posterior
- Bootstrap as poor man’s posterior
- estimate, estimator, estimand meme
- They don’t know how sampling theory works (what did they get before?),
  you need to illustrate it as well as Bayesian updating
- Can a confidence interval be interpreted the way it is for both OLS
  and MLE? Are we intereting confidence interval incorrecly for OLS and
  MLE? At what point do we need bootstrapping?
- Create a chart showing the differences between Bayesian and
  frequentist statistics?
- Week 03 notes from pre-PhD seminar

## Week 10 (Oct 27, 29; Mar 16, 18; Jul 13, 15)

- Hyperparameter Tuning (Oct 27; Mar 16; Jul 13)
- Cross-Validation (Oct 29; Mar 18; Jul 15)

### Hyperparameter Tuning

- Maybe introduce interactions as well? To decide on whether or not to
  include them?
- Start with model selection as a form of “hyperparameter tuning”

### Cross-Validation

- Cross-validation is needed for hyperparameter tuning
- Cross-validation for model selection as well in order to keep a larger
  training dataset (p. 123 of PML)

## Week 11 (Nov 3, 5; Mar 23, 25; Jul 20, 22)

- Penalized Regression (Nov 3; Mar 23; Jul 20)
- Variable Selection (Nov 5; Mar 25; Jul 22)

### Penalized Regression

- Generalize Regression to Penalized Regression/Bayesian Models (with
  PyMC?)
- Penalize the regression to learn “regular” (i.e., generalizable)
  features
- Ridge regression, LASSO, elastic net, best subsets, sequential
  replacement
- Bias-variance tradeoff and shrinkage methods
- Accuracy and precision is just bias and variance
- Shrinkage toward the MLE
- Why Bayes? Carefully and directly model uncertainty

### Variable Selection

- Stepwise regression for model selection when focusing on prediction
- Discussed the possibility of p \> n
- Working from a full model and using backward or forward-selection

## Week 12 (Nov 10, 12; Mar 30, Apr 1; Jul 27, 29)

- Dimension Reduction (Nov 10; Mar 30; Jul 27)
- Principal Components Regression (Nov 12; Apr 1; Jul 29)

### Dimension Reduction

### Principal Components Regression

- Dimension reduction as opposed to variable selection and
  regularization
- PCA on its own and then PCA as part of PCR

## Week 13 (Nov 17, 19; Apr 6, 8; Aug 3, 5)

- Interactions (Nov 17; Apr 6; Aug 3)
- Multilevel Models (Nov 19; Apr 8; Aug 5)

### Interactions

- Interactions (including higher-order interactions), including
  interactions
- Why do we just interpret interaction effects and not main effects?
- Does this all have to be done with the statsmodels API? What about
  https://www.scikit-yb.org/en/latest/index.html?

### Multilevel Models

- Motivate with Simpson’s paradox (PML p. 80)

## Week 14 (Nov 24; Apr 13, 15)

- Thanksgiving Break (Nov 24)

## Week 15 (Dec 1, 3; Apr 20)

- Presentations (Dec 1, 3; Apr 13, 15, 20)

- Student evaluations (IDEA)

- Student panel invitation

- Post-quiz for research project

- Share LinkedIn QR code
