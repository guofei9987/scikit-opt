from pso import PSO
import ga
from test_func import sphere as obj_func


print('------------')
print('starting PSO...')
# my_pso = PSO(func=obj_func, pop=30, dim=5, max_iter=100)
my_pso = PSO(func=obj_func, dim=3)
fitness = my_pso.fit()
print('best_x is ',my_pso.gbest_x)
print('best_y is ',my_pso.gbest_y)
my_pso.plot_history()


print('-------------')
print('starting GA...')

# general_best,func_general_best=ga.ga(func=demo_func2, pop=50, iter_max=200, lb=[-1, -10, -5], ub=[2, 10, 2],precision=[1e-7, 1e-7, 1e-7],Pm=0.001)
# print(general_best,func_general_best)
general_best,func_general_best,FitV_history=ga.ga(func=obj_func,  lb=[-1, -10, -5], ub=[2, 10, 2])
print('best_x:',general_best)
print('best_y:',func_general_best)

ga.plot_FitV(FitV_history)