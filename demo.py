import ga
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def demo_func2(p):
    x, y, z = p
    return -(x ** 2 + y ** 2 + z ** 2)


# general_best,func_general_best=ga.ga(func=demo_func2, pop=50, iter_max=200, lb=[-1, -10, -5], ub=[2, 10, 2],precision=[1e-7, 1e-7, 1e-7],Pm=0.001)
# print(general_best,func_general_best)
general_best,func_general_best,FitV_history=ga.ga(func=demo_func2,  lb=[-1, -10, -5], ub=[2, 10, 2])
print('best_x:',general_best)
print('best_y:',func_general_best)

ga.plot_FitV(FitV_history)

