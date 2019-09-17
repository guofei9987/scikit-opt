def demo_func(x):
    x1, x2, x3 = x
    return x1 ** 2 + (x2 - 0.05) ** 2 + x3 ** 2


from sko.PSO import PSO
pso = PSO(func=demo_func, dim=3)
fitness = pso.fit()
print('best_x is ', pso.gbest_x, 'best_y is', pso.gbest_y)
pso.plot_history()

# from test_func import sphere as obj_func
