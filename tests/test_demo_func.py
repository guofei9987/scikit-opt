import unittest

from sko.demo_func import ackley, cigar, function_for_TSP, rastrigrin, rosenbrock, sixhumpcamel, sphere, schaffer, zakharov, shubert, griewank, rastrigrin, rosenbrock, sixhumpcamel, zakharov, ackley, cigar


class TestDemoFunc(unittest.TestCase):

    def test_function_for_TSP(self):
        pass

    def sphere(self):
        val=sphere((1, 0))
        print(val)
        self.assertEqual(val, 0.0)

    def schaffer(self):
        self.assertEqual(schaffer((0, 0)), 0.0,msg="schaffer error: schaffer failed for 2 arguments ")

    def test_shubert(self):
        self.assertEqual(
            shubert((-7.08350643, -7.70831395)), -186.7309088309155 ,msg="shubert error: shubert failed for 2 arguments ")

    def test_griewank(self):
        self.assertEqual(griewank((0, 0)), 0.0,msg="griewank error: griewank failed for 2 arguments ")
        self.assertEqual(griewank((0, 0, 0)), 0.0,msg="griewank error: griewank failed for 3 arguments ")

    def test_rastrigrin(self):
        self.assertEqual(rastrigrin((0, 0)), 0.0,msg="rastrigrin error: rastrigrin failed for 2 arguments ")
        self.assertEqual(rastrigrin((0, 0, 0)), 0.0,msg="rastrigrin error: rastrigrin failed for 3 arguments ")

    def test_rosenbrock(self):
        self.assertEqual(rosenbrock((1, 1, 1)), 0.0,msg="rosenbrock error: rosenbrock failed for 3 arguments ")
        self.assertEqual(rosenbrock((1, 1)), 0.0,msg="rosenbrock error: rosenbrock failed for 2 arguments ")

    def test_zakharov(self):
        self.assertEqual(zakharov((0, 0)), 0.0,msg="zakharov error: zakharov failed for 2 arguments ")
        self.assertEqual(zakharov((0, 0, 0)), 0.0,msg="zakharov error: zakharov failed for 3 arguments ")

    def test_ackley(self):
        self.assertEqual(ackley((0, 0)),-200.0,msg="ackley error: ackley failed for 2 arguments ")

    def test_cigar(self):
        self.assertEqual(cigar((0, 0)), 0.0,msg="cigar error: cigar failed for 2 arguments ")
        self.assertEqual(cigar((0, 0, 0, 0)), 0.0,msg="cigar error: cigar failed for 4 arguments ")

    def test_sixhumpcamel(self):
        self.assertEqual(sixhumpcamel(
            (-0.08984201368301331, 0.7126564032704135)), -1.0316284534898774,msg="sixhumpcamel error: sixhumpcamel failed for 2 arguments ")
    
if __name__ == '__main__':
        unittest.main(verbosity=2,exit=False)
