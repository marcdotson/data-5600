# Frequentist and Bayesian Inference


During this class we will compare frequentist and Bayesian inference.

Start by downloading the loyalty data.

## Learn

View the slides:

You can also download the slides as an .html file. Once you’ve previewed
the material and identified any questions, start watching the lecture.

Why do we care about model parameters? What is missing from our
estimates so far? Go the discussions to share before continuing with the
lecture.

When might prior information be especially important in an analysis? Go
the discussions to share before finishing the lecture.

## Data Dictionary

Loyalty data includes consumers who have signed up for the loyalty
program.

- **customer_id**: Unique customer identifier
- **units**: Number of peanut butter jars purchased
- **loyal**: Enrolled in Harmon’s loyalty program
- **age**: Age
- **avg_spend**: Average spend per week
- **gender**: Gender
- **points**: Points accrued
- **email**: Signed up for email promotions

## Apply

### Exercise 09

1.  Harmon’s has made a portion of their loyalty CRM data available, but
    after walking through the assumption diagnostics, I’ve found that
    only `gender`, `email`, and `points` (logged) should be included as
    additional predictors
2.  Join the two data sets, split into training and testing data, update
    the feature engineering, and fit a new model with the additional
    predictors using OLS and, if you can, a Bayesian model with uniform
    priors
3.  Interpret the interval estimates, being careful to track transformed
    scales, reference levels, the presence of multiple predictors in the
    model, statistical significance, and the differences between
    credible and confidence intervals
4.  Submit both your Quarto document and your code, output, and
    interpretations as a single PDF on Canvas

### Milestone 09

Update your project’s parameter estimate interpretations using interval
estimates and statistical significance. If you can, run a Bayesian model
and compare credible and confidence intervals.
