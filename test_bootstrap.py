import pytest
import numpy as np
from bootstrap import bootstrap_sample, bootstrap_ci, r_squared

def test_bootstrap_integration():
    """Test that bootstrap_sample and bootstrap_ci work together"""
    # This test should initially fail
    pass

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
        boostrap_ci([], alpha=0.05)


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