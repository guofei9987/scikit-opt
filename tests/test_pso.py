import unittest

import numpy as np

from sko.PSO import PSO


class TestPSO(unittest.TestCase):
    def setUp(self):
        self.random_state = np.random.get_state()
        np.random.seed(0)

    def tearDown(self):
        np.random.set_state(self.random_state)

    def test_initial_best(self):
        pso = PSO(lambda x: np.sum(x ** 2), n_dim=2, pop=5, max_iter=0)
        np.testing.assert_array_equal(pso.pbest_y, pso.Y)
        best = pso.Y.argmin()
        np.testing.assert_array_equal(pso.best_x, pso.X[best])
        np.testing.assert_array_equal(pso.best_y, pso.Y[best])
        np.testing.assert_array_equal(pso.run()[0], pso.X[best])

    def test_initial_best_respects_constraints(self):
        for constraint in (lambda x: -x[0], lambda x: 2):
            with self.subTest(constraint=constraint):
                pso = PSO(lambda x: x[0], n_dim=1, pop=10, lb=-1, ub=1,
                          constraint_ueq=(constraint,))
                feasible = np.array([constraint(x) <= 0 for x in pso.X])
                expected = np.where(feasible[:, None], pso.Y, np.inf)
                np.testing.assert_array_equal(pso.pbest_y, expected)
                np.testing.assert_array_equal(pso.gbest_y, expected.min(axis=0))
                if feasible.any():
                    np.testing.assert_array_equal(pso.gbest_x, pso.X[expected.argmin()])

    def test_first_move_retains_better_initial_point(self):
        pso = PSO(lambda x: x[0], n_dim=1, pop=1, max_iter=1,
                  lb=0, ub=1, w=1, c1=0, c2=0)
        initial_x = pso.X[0].copy()
        best_x, best_y = pso.run()
        self.assertGreater(pso.X[0, 0], initial_x[0])
        np.testing.assert_array_equal(best_x, initial_x)
        np.testing.assert_array_equal(best_y, initial_x)


if __name__ == '__main__':
    unittest.main()
