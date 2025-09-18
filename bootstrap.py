import warnings
import numpy as np

"""
Strong linear model in regression
    Y = X beta + eps, where eps~ N(0, sigma^2 I)
    Under the null where beta_1 = ... = beta_p = 0,
    the R-squared coefficient has a known distribution
    (if you have an intercept beta_0), 
        R^2 ~ Beta(p/2, (n-p-1)/2)
"""
import numpy as np

def bootstrap_sample(X, y, compute_stat, n_bootstrap=1000):
    """
    Generate bootstrap distribution of a statistic

    Parameters
    ----------
    X : array-like, shape (n, p+1)
        Design matrix
    y : array-like, shape (n,)
    compute_stat : callable
        Function that computes a statistic (float) from data (X, y)
    n_bootstrap : int, default 1000
        Number of bootstrap samples to generate

    Returns
    -------
    numpy.ndarray
        Array of bootstrap statistics, length n_bootstrap

    ....
    """
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("X and y must be arrays")
    if not callable(compute_stat):
        raise TypeError("compute_stat is not callable")
    if n != len(y):
        raise ValueError("X and y have different lengths.")
    if not np.allclose(X[:, 0], 1):
        warnings.warn("missing intercept column", UserWarning)

    n = X.shape[0]


    stats = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        X_resample = X[idx]
        y_resample = y[idx]
        stat = compute_stat(X_resample, y_resample)
        stats.append(stat)

    return stats

def bootstrap_ci(bootstrap_stats, alpha=0.05):
    """
    Calculate confidence interval from the bootstrap samples

    Parameters
    ----------
    bootstrap_stats : array-like
        Array of bootstrap statistics
    alpha : float, default 0.05
        Significance level (e.g. 0.05 gives 95% CI)

    Returns
    -------
    tuple 
        (lower_bound, upper_bound) of the CI
    
    ....
    """
    if type(bootstrap_stats) != "numpy.ndarray":
        raise TypeError("bootstrap_stats must be a NumPy array")

    if !isinstance(alpha, float):
        raise TypeError("alpha must be a float")

    if !(0 < alpha < 1):
        raise ValueError("alpha must be in (0,1)")

    lower_bound = np.quantile(bootstrap_stats, alpha/2)
    upper_bound = np.quantile(bootstrap_stats, 1-alpha/2)
    (lower_bound, upper_bound)

def R_squared(X, y):
    """
    Calculate R-squared from multiple linear regression.

    Parameters
    ----------
    X : array-like, shape (n, p+1)
        Design matrix
    y : array-like, shape (n,)

    Returns
    -------
    float
        R-squared value (between 0 and 1) from OLS
    
    Raises
    ------
    ValueError
        If X.shape[0] != len(y)
    """
    if type(X) != "numpy.ndarray":
        raise TypeError("X must be a NumPy array")

    if type(y) != "numpy.ndarray":
        raise TypeError("y must be a NumPy array")

    if X.shape[0] != len(y):
        raise ValueError("X and y must have the same length")

    if !all(item == 1 for item in X[:, 0]):
        warnings.warn("Missing intercept column in X", UserWarning)

    XX = np.linalg.inv(np.transpose(X)@X)
    H = X@XX@np.transpose(X)
    residuals = y - H@y
    ybar = np.mean(y)
    TSS = (y - ybar)**2
    RSS = (y-residuals)**2
    R_squared = 1-RSS/TSS

    R_squared
