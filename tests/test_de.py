import unittest
# -*- coding: utf-8 -*-
import numpy as np
from sko.DE import DE


class TestDE(unittest.TestCase):

    def test_best_y_is_best_evaluated(self):
        # best_y must not be worse than any point DE evaluated (issue #248)
        for seed in (25, 41, 47):
            calls = []

            def func(x):
                value = float(np.sum((x - 1.3) ** 2))
                calls.append(value)
                return value

            np.random.seed(seed)
            de = DE(func=func, n_dim=10, size_pop=50, max_iter=3, lb=[-5] * 10, ub=[5] * 10)
            best_x, best_y = de.run()
            self.assertLessEqual(float(np.ravel(best_y)[0]), min(calls) + 1e-9,
                                 msg="DE best_y is worse than an evaluated point, seed {}".format(seed))

    def test_y_matches_population(self):
        np.random.seed(0)
        de = DE(func=lambda x: float(np.sum(x ** 2)), n_dim=3, size_pop=10, max_iter=5, lb=-1, ub=1)
        de.run()
        np.testing.assert_allclose(de.Y, np.sum(de.X ** 2, axis=1))


if __name__ == '__main__':
    unittest.main()
