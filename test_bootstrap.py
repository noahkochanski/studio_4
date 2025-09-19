import pytest
import numpy as np
from bootstrap import bootstrap_sample, bootstrap_ci, R_squared

def test_bootstrap_integration():
    """Test that bootstrap_sample and bootstrap_ci work together"""
    X = np.random.normal(0, 1, (100, 2))
    X = np.concatenate((np.ones((100,1)), X), axis=1)
    y = 2 + X[:,1]*3 - X[:,2]*1 + np.random.normal(0, 1, 100)
    stats = bootstrap_sample(X, y, R_squared)
    confint = bootstrap_ci(stats)
    assert isinstance(confint, tuple)

def test_boostrap_CI():
    stats = np.random.rand(100)
    # Testing alpha between 0 and 1
    with pytest.raises(ValueError, match="alpha needs to be between 0 and 1"):
        bootstrap_ci(stats, alpha=-0.1)
    with pytest.raises(ValueError, match="alpha needs to be between 0 and 1"):
        bootstrap_ci(stats, alpha=1.1)
   
    # Testing statistics are an array
    with pytest.raises(TypeError, match = "stats should be array-like"):
        bootstrap_ci("not an array", alpha = 0.05)

    # Testing statistics are not empty
    with pytest.raises(ValueError, match = "stats should not be empty"):
        bootstrap_ci(np.array([]), alpha=0.05)

def test_bootstrap_sample_output():
    """ Test output of bootstrap sample has length n_bootstrap """
    y = np.random.normal(0, 1, 100)
    X = np.random.normal(0, 1, (100,3))
    X = np.concatenate((np.ones((100,1)), X), axis=1)

    results = bootstrap_sample(X, y, R_squared)
    assert len(results) == 1000
    
def test_bootstrap_sample_arg_lengths():
    """ Test y and X have same length """
    y = np.random.normal(0, 1, 50)
    X = np.random.normal(0, 1, (100, 2))
    X = np.concatenate((np.ones((100,1)), X), axis=1)
    with pytest.raises(ValueError, match="different lengths"):
        bootstrap_sample(X, y, R_squared)

def test_bootstrap_sample_stat_type():
    """ Test stat function is callable """
    y = np.random.normal(0, 1, 100)
    X = np.random.normal(0, 1, (100, 2))
    X = np.concatenate((np.ones((100,1)), X), axis=1)
    with pytest.raises(TypeError, match="is not callable"):
        bootstrap_sample(X, y, 3, n_bootstrap = 1000)

def test_bootstrap_sample_happy_path():
    """ Test that bootstrap sample correctly implements paired bootstrap"""
    y = np.random.normal(0, 1, 100)
    X = np.concatenate((np.ones((100,1)), np.reshape(y, (100,1))), axis=1)
    results = bootstrap_sample(X, y, R_squared)
    assert np.all(results == 1)

def test_bootstrap_sample_test_intercept():
    """ Test intercept column is present in X """
    y = np.random.normal(0, 1, 100)
    X = np.random.normal(0, 1, (100, 2))
    with pytest.warns(UserWarning, match = "missing intercept column"):
        bootstrap_sample(X, y, R_squared)

def test_bootstrap_sample_data_types():
    """ Test that X and y are arrays """

    # y is not an array
    y = 2
    X = np.random.normal(0, 1, (1, 2))
    with pytest.raises(TypeError, match="must be arrays"):
        bootstrap_sample(X, y, R_squared)

    # X is not an array
    y = np.random.normal(0, 1, 2)
    X = (3, 4)
    with pytest.raises(TypeError, match="must be arrays"):
        bootstrap_sample(X, y, R_squared)

    # both y and X not arrays
    y = "Hello"
    X = "World"
    with pytest.raises(TypeError, match="must be arrays"):
        bootstrap_sample(X, y, R_squared)

def test_R_squared():
    X = np.ones([1,2])
    y = np.ones(4)
    
    # Length mismatch
    with pytest.raises(ValueError, match = "X and y do not match in size"):
        R_squared(X, y)
    
    # Wrong input type
    with pytest.raises(TypeError, match = "X is not an array"):
        R_squared("not an array", y)
    with pytest.raises(TypeError, match = "y is not an array"):
        R_squared(X, "not an array")

def test_R_squared_happy_path():
    y = np.random.normal(0, 1, 100)
    X = np.concatenate((np.ones((100,1)), np.reshape(y, (100,1))), axis=1)
    result = R_squared(X, y)
    assert result == 1
