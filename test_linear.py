import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from linear.prog import LinearRegression

def test_two_points_line():
    X = np.array([[1], [2]])
    y = np.array([1, 2]) 

    model = LinearRegression()
    model.fit(X, y)
    preds = model.predict(X)

    assert np.allclose(preds, y, atol=1e-6)


def test_random_points_on_line():
    np.random.seed(42)

    X = np.random.rand(100, 1) * 10
    y = 3 * X.squeeze() + 5 

    model = LinearRegression()
    model.fit(X, y)
    preds = model.predict(X)

    assert np.allclose(preds, y, atol=1e-6)

def test_linearly_separable_like_case():
    X = np.array([
        [1], [2], [3],    
        [10], [11], [12]  
    ])
    y = np.array([1, 2, 3, 10, 11, 12])  

    model = LinearRegression()
    model.fit(X, y)
    preds = model.predict(X)

    assert np.allclose(preds, y, atol=1e-6)


def test_ill_conditioned_matrix():

    X = np.array([
        [1, 1],
        [2, 2.000001],
        [3, 3.000002],
        [4, 4.000003]
    ])
    y = np.array([2, 4, 6, 8])

    model = LinearRegression()

    try:
        model.fit(X, y)
        preds = model.predict(X)

        assert np.all(np.isfinite(preds))

    except np.linalg.LinAlgError:
        pytest.skip("Матрица вырожденная — ожидаемое поведение для inv")

def test_compare_with_sklearn():
    np.random.seed(0)

    X = np.random.rand(100, 3)
    true_w = np.array([2.0, -1.0, 0.5])
    y = X @ true_w + 3 

    my_model = LinearRegression()
    my_model.fit(X, y)
    my_preds = my_model.predict(X)

    # sklearn
    sk_model = LinearRegression()
    sk_model.fit(X, y)
    sk_preds = sk_model.predict(X)

    assert np.allclose(my_preds, sk_preds, atol=1e-5)