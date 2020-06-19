import numpy as np
import oracles


def grad_finite_diff(function, w, eps=1e-8):
    """
    Возвращает численное значение градиента, подсчитанное по следующией формуле:
        result_i := (f(w + eps * e_i) - f(w)) / eps,
        где e_i - следующий вектор:
        e_i = (0, 0, ..., 0, 1, 0, ..., 0)
                          >> i <<
    """

    d = w.shape[0]
    grad = np.zeros(d)

    for i in range(d):
        w_test = w.copy()
        w_test[i] += eps
        grad[i] = (function(w_test) - function(w))
    grad /= eps

    return grad

# oracle = oracles.BinaryLogistic(l2_coef=1)
# X = np.random.normal(0, 1, size=(100, 3))
# y = np.random.randint(0, 1 + 1, size=100)
# print(grad_finite_diff(lambda w: oracle.func(X, y, w), np.ones(3), 1e-6))
