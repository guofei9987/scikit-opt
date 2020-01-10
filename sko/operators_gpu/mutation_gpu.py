import torch


def mutation(self):
    '''
    mutation of 0/1 type chromosome
    faster than `self.Chrom = (mask + self.Chrom) % 2`
    :param self:
    :return:
    '''
    mask = (torch.rand(size=(self.size_pop, self.len_chrom), device=self.device) < self.prob_mut).type(torch.int8)
    self.Chrom ^= mask
    return self.Chrom
