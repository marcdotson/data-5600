# Confusion and Cross-Validation


During this class we will:

- Discuss additional measures of overall classification model fit
- Introduce cross-validation for model selection and hyperparameter
  tuning

## Learn

View the slides:

You can also download the slides as an .html file. Once you’ve previewed
the material and identified any questions, start watching the lecture.

What could go wrong with using the test data to choose the cutoff
probability? Go the discussions to share before continuing with the
lecture.

How would you implement cross-validation to tune the cutoff probability?
Go the discussions to share before finishing the lecture.

## Apply

### Exercise 17

1.  Start with the latest coding exercise solution
2.  Implement 5-fold cross-validation to tune the cutoff probability
    hyperparameter so that you maximize overall accuracy
3.  When splitting the data into training, validation, and testing data,
    remember to use the `stratify` parameter
4.  Given the computational expense of cross-validation, using MLE is
    sufficient
5.  Submit both your Quarto document and your code, output, and
    explanation as a single PDF on Canvas

### Milestone 17

Start working on hyperparameter tuning for your project by using
cross-validation to tune the cutoff probability.
