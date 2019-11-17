def demo_func(x):
    x1, x2, x3 = x
    return x1 ** 2 + (x2 - 0.05) ** 2 + x3 ** 2


from sko.PSO import PSO

pso = PSO(func=demo_func, dim=3)
fitness = pso.run()
print('best_x is ', pso.gbest_x, 'best_y is', pso.gbest_y)

import matplotlib.pyplot as plt

plt.plot(pso.gbest_y_hist)
plt.show()
# %% PSO with constrain:
print('*' * 50, '\n PSO with constrain:')

pso = PSO(func=demo_func, dim=3, lb=[0, -1, 0.5], ub=[1, 1, 1])
pso.run()
print('best_x is ', pso.gbest_x, 'best_y is', pso.gbest_y)
