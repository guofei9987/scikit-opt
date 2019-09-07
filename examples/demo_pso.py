demo_func = lambda x: x[0] ** 2 + (x[1] - 0.05) ** 2 + x[2] ** 2

from sko.PSO import PSO

from test_func import sphere as obj_func

print('------------')
print('starting PSO...')


pso = PSO(func=obj_func, dim=3)
fitness = pso.fit()
print('best_x is ', pso.gbest_x)
print('best_y is ', pso.gbest_y)
pso.plot_history()