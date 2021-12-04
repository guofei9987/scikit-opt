import unittest
# -*- coding: utf-8 -*-
# @Time    : 2019/10/15
# @Author  : github.com/Agrover112
from sko.demo_func import ackley, cigar, function_for_TSP, rastrigrin, rosenbrock, sixhumpcamel, sphere, schaffer, zakharov, shubert, griewank, rastrigrin, rosenbrock, sixhumpcamel, zakharov, ackley, cigar


class TestDemoFunc(unittest.TestCase):

    def test_function_for_TSP(self):
        pass

    def test_sphere(self):
        self.assertEqual(sphere((0, 0)), 0.0,msg="sphere failed for 2 arguments ")

    def test_schaffer(self):
        self.assertEqual(schaffer((0, 0)), 0.0,msg="schaffer failed for 2 arguments ")

    def test_shubert(self):
        self.assertEqual(
            shubert((-7.08350643, -7.70831395)), -186.7309088309155 ,msg="shubert failed for 2 arguments ")

    def test_griewank(self):
        self.assertEqual(griewank((0, 0)), 0.0,msg="griewank failed for 2 arguments ")
        self.assertEqual(griewank((0, 0, 0)), 0.0,msg="griewank failed for 3 arguments ")

    def test_rastrigrin(self):
        self.assertEqual(rastrigrin((0, 0)), 0.0,msg="rastrigrin failed for 2 arguments ")
        self.assertEqual(rastrigrin((0, 0, 0)), 0.0,msg="rastrigrin failed for 3 arguments ")

    def test_rosenbrock(self):
        self.assertEqual(rosenbrock((1, 1, 1)), 0.0,msg="rosenbrock failed for 3 arguments ")
        self.assertEqual(rosenbrock((1, 1)), 0.0,msg="rosenbrock failed for 2 arguments ")

    def test_zakharov(self):
        self.assertEqual(zakharov((0, 0)), 0.0,msg="zakharov failed for 2 arguments ")
        self.assertEqual(zakharov((0, 0, 0)), 0.0,msg="zakharov failed for 3 arguments ")

    def test_ackley(self):
        self.assertEqual(ackley((0, 0)),-200.0,msg="ackley failed for 2 arguments ")

    def test_cigar(self):
        self.assertEqual(cigar((0, 0)), 0.0,msg="cigar failed for 2 arguments ")
        self.assertEqual(cigar((0, 0, 0, 0)), 0.0,msg="cigar failed for 4 arguments ")

    def test_sixhumpcamel(self):
        self.assertEqual(sixhumpcamel(
            (-0.08984201368301331, 0.7126564032704135)), -1.0316284534898774,msg="sixhumpcamel failed for 2 arguments ")
    
if __name__ == '__main__':
        unittest.main()
