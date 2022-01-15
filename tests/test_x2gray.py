# -*- coding: utf-8 -*-
# @Time    : 2019/10/15
# @Author  : github.com/Agrover112 , github.com/guofei9987

import numpy as np
from sko.GA import GA
from sko.tool_kit import x2gray

import unittest


class TestX2Gray(unittest.TestCase):

    def test_x2gray(self):
        cases = [
            [2, [-1, -1], [1, 1], 1e-7],
            [5, [-10, -1, -3, -4.5, 1.5], [1, 3, 5, 7.8, 9.8], 1e-7],
            [3, [0, -5, -10], [15, 10, 5], 1],

        ]
        for n_dim, lb, ub, precision in cases:
            ga = GA(func=lambda x: x, n_dim=n_dim, size_pop=200, max_iter=800, lb=lb, ub=ub, precision=precision)

            value = ga.chrom2x(ga.Chrom)
            chrom2 = x2gray(x=value, n_dim=n_dim, lb=lb, ub=ub, precision=precision)
            self.assertTrue(np.allclose(ga.Chrom, chrom2), msg='x2gray error')


if __name__ == '__main__':
    unittest.main()
