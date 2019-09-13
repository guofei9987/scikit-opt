import numpy as np
from sko.GA import GA, GA_TSP, ga_with_udf


def selection_elite(self):
    print('udf selection actived')
    FitV = self.FitV
    FitV = (FitV - FitV.min()) / (FitV.max() - FitV.min() + 1e-10) + 0.2
    # the worst one should still has a chance to be selected
    # the elite(defined as the best one for a generation) must survive the selection
    elite_index = np.array([FitV.argmax()])

    # do Roulette to select the next generation
    sel_prob = FitV / FitV.sum()
    roulette_index = np.random.choice(range(self.size_pop), size=self.size_pop - 1, p=sel_prob)
    sel_index = np.concatenate([elite_index, roulette_index])
    self.Chrom = self.Chrom[sel_index, :]  # next generation
    return self.Chrom


options = {'selection': {'udf': selection_elite}}
GA_1 = ga_with_udf(GA, options)

demo_func = lambda x: x[0] ** 2 + (x[1] - 0.05) ** 2 + x[2] ** 2
ga = GA_1(func=demo_func, n_dim=3, max_iter=500, lb=[-1, -10, -5], ub=[2, 10, 2])
best_x, best_y = ga.fit()

print('best_x:', best_x, '\n', 'best_y:', best_y)
