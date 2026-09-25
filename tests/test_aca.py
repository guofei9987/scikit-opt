# -*- coding: utf-8 -*-
import numpy as np
from sko.ACA import ACA_TSP

import unittest


class TestACA(unittest.TestCase):

    def test_zero_probability_row_does_not_crash(self):
        n_dim = 5
        distance_matrix = np.ones((n_dim, n_dim)) - np.eye(n_dim)

        def cal_total_distance(routine):
            num_points, = routine.shape
            return sum([distance_matrix[routine[i % num_points], routine[(i + 1) % num_points]]
                        for i in range(num_points)])

        aca = ACA_TSP(func=cal_total_distance, n_dim=n_dim, size_pop=4, max_iter=1,
                      distance_matrix=distance_matrix)
        aca.Tau = np.zeros((n_dim, n_dim))  # force every probability row to sum to 0

        best_x, best_y = aca.run()

        self.assertFalse(np.isnan(best_y))
        self.assertEqual(sorted(best_x.tolist()), list(range(n_dim)))


if __name__ == '__main__':
    unittest.main()
