import numpy as np


def mutation(self):
    '''
    mutation of 0/1 type chromosome
    faster than `self.Chrom = (mask + self.Chrom) % 2`
    :param self:
    :return:
    '''
    #
    mask = (np.random.rand(self.size_pop, self.len_chrom) < self.prob_mut)
    self.Chrom ^= mask
    return self.Chrom


def mutation_TSP_1(self):
    '''
    every gene in every chromosome mutate
    :param self:
    :return:
    '''
    for i in range(self.size_pop):
        for j in range(self.n_dim):
            if np.random.rand() < self.prob_mut:
                n = np.random.randint(0, self.len_chrom, 1)
                self.Chrom[i, j], self.Chrom[i, n] = self.Chrom[i, n], self.Chrom[i, j]
    return self.Chrom


def reverse(self):
    '''
    Reverse
    :param self:
    :return:
    '''
    for i in range(self.size_pop):
        if np.random.rand() < self.prob_mut:
            n1, n2 = np.random.randint(0, self.len_chrom - 1, 2)
            if n1 >= n2:
                n1, n2 = n2, n1 + 1
            self.Chrom[i, n1:n2] = self.Chrom[i, n1:n2][::-1]
    return self.Chrom


def mutation_TSP_3(self):
    for i in range(self.size_pop):
        if np.random.rand() < self.prob_mut:
            n1, n2 = np.random.randint(0, self.len_chrom, 2)
            self.Chrom[i, n1], self.Chrom[i, n2] = self.Chrom[i, n2], self.Chrom[i, n1]
    return self.Chrom
