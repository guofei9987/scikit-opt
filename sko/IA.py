#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019/9/16
# @Author  : github.com/guofei9987


import numpy as np
from sko.GA import GA_TSP, ga_with_udf
from scipy import spatial


# %%
def immune_ranking(self, T=0.7, alpha=0.95):
    '''

    :param self: GA_TSP class, IA_TSP can be established upon Genetic Algorithm
    :param T: float, 抗体与抗体之间的亲和度阈值，大于这个阈值认为亲和，否则认为不亲和
    :param alpha: float, 多样性评价指数，也就是抗体和抗原的重要性/抗体浓度重要性
    :return: numpy.array 期望繁殖概率
    '''
    # part1：抗体与抗原的亲和度
    A = 1 / self.Y

    # part2：抗体浓度（抗体与抗体之间的亲和度）
    dist_matrix1 = spatial.distance.cdist(self.Chrom, self.Chrom, metric='hamming')  # 抗体之间的hamming距离矩阵
    similiar_matrix1 = dist_matrix1 < 1 - T  # 是否与其他抗体的相似度多于T
    similiar_matrix2 = similiar_matrix1.sum(axis=1)  # 每个抗体与其他抗体相似，计数
    S = (similiar_matrix2 - 1) / (self.size_pop - 1)  # 抗体浓度。减一是因为自己与自己一定相似，应当排除
    self.FitV = alpha * A / A.sum() + (1 - alpha) * S / (S.sum() + 1e-5)
    return self.FitV


class IA_TSP(GA_TSP):
    ranking = immune_ranking
