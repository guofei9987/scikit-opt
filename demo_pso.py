from pso import PSO
def demo_func2(p):
    # Sphere函数
    out_put = 0
    for i in p:
        out_put += i ** 2
    return out_put
my_pso = PSO(func=demo_func2, pop=30, dim=5, max_iter=100)
fitness = my_pso.fit()
print(my_pso.gbest_x)
print(my_pso.gbest_y)
my_pso.plot_history()