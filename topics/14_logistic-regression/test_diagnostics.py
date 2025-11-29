import os
import polars as pl
import numpy as np
import matplotlib.pyplot as plt

model_diagnostics = os.path.join('topics', '14_logistic-regression', 'model_diagnostics.py')
exec(open(model_diagnostics).read(), globals())

# Use non-interactive backend for testing
plt.switch_backend('Agg')

def test_linear_regression():
    print("\n" + "="*50)
    print("TESTING LINEAR REGRESSION DIAGNOSTICS (POLARS)")
    print("="*50)
    
    # Generate synthetic data (Ideal case)
    np.random.seed(42)
    n = 100
    X1 = np.random.normal(0, 1, n)
    X2 = np.random.normal(0, 1, n)
    # Linear relationship, constant variance, normal errors
    Y = 2*X1 + 3*X2 + np.random.normal(0, 1, n)
    
    data = pl.DataFrame({'X1': X1, 'X2': X2, 'Y': Y})
    features = ['X1', 'X2']
    target = 'Y'
    
    print("\n--- Test Case 1: Ideal Linear Data ---")
    check_validity(data)
    check_representativeness(data, target, 'linear')
    check_linearity(data, features, target, 'linear')
    check_independence(data, target)
    check_constant_variance(data, features, target, 'linear')
    check_normality(data, target, 'linear')
    check_identifiability(data, features)

    # Generate synthetic data (Problematic case: Multicollinearity + Non-linear)
    X3 = X1 * 0.95 + np.random.normal(0, 0.1, n) # High correlation with X1
    Y_bad = X1**2 + np.random.normal(0, 1, n) # Non-linear
    
    data_bad = pl.DataFrame({'X1': X1, 'X3': X3, 'Y': Y_bad})
    features_bad = ['X1', 'X3']
    
    print("\n--- Test Case 2: Problematic Linear Data ---")
    check_identifiability(data_bad, features_bad)
    check_linearity(data_bad, features_bad, 'Y', 'linear')

def test_logistic_regression():
    print("\n" + "="*50)
    print("TESTING LOGISTIC REGRESSION DIAGNOSTICS (POLARS)")
    print("="*50)
    
    # Generate synthetic data (Ideal case)
    np.random.seed(42)
    n = 200
    X1 = np.random.normal(0, 1, n)
    # Logistic relationship
    logit = 2*X1
    prob = 1 / (1 + np.exp(-logit))
    Y = np.random.binomial(1, prob, n)
    
    data = pl.DataFrame({'X1': X1, 'Y': Y})
    features = ['X1']
    target = 'Y'
    
    print("\n--- Test Case 3: Ideal Logistic Data ---")
    check_validity(data)
    check_representativeness(data, target, 'logistic')
    check_linearity(data, features, target, 'logistic')
    check_independence(data, target)
    check_constant_variance(data, features, target, 'logistic') # Should skip
    check_normality(data, target, 'logistic') # Should skip
    check_identifiability(data, features)

if __name__ == "__main__":
    test_linear_regression()
    test_logistic_regression()

