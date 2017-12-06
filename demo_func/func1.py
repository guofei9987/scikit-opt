def demo_func1(p):
    x,y,z=p
    return -x**2+y**2+z**2

import scipy.optimize as opt
import numpy as np

init_point=(-2,-2,-2)


result=opt.fmin(demo_func1,init_point)
print(result)