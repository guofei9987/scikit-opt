import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# def gray2rv(gray_code):
#     # demo of Gray Code to real value
#     len_gray_code = len(gray_code)
#     b = gray_code.cumsum() % 2
#     return sum(b * 0.5 ** np.array(1, len_gray_code + 1))


def bs2rv(Chrom, FieldD):
    # Gray Code to real value
    # FieldD = [len_segment,lb_segment,ub_segment]
    # len_segment每个片段的长度
    # lb_segment每个片段的最大值
    # ub_segment每个片段的最小值
    # Phen输出值
    # Chrom = ga.crtbp(50, 20)
    # FieldD = np.array([[10, 10], [0, 1], [2, 6]])
    # bs2rv(Chrom, FieldD)

    len_segment, lb_segment, ub_segment = FieldD
    cumsum_len_segment = len_segment.cumsum()
    Phen = np.zeros(shape=(Chrom.shape[0], len(len_segment)))
    for i, j in enumerate(cumsum_len_segment):
        if i == 0:
            Chrom_temp = Chrom[:, :cumsum_len_segment[0]]
        else:
            Chrom_temp = Chrom[:, cumsum_len_segment[i - 1]:cumsum_len_segment[i]]
        temp1 = Chrom_temp.cumsum(axis=1) % 2
        temp2 = temp1 * np.logspace(start=1, stop=len_segment[i], base=0.5, num=len_segment[i])
        temp3 = temp2.sum(axis=1)
        adjust = (np.ones(shape=(1, len_segment[i])) * np.logspace(start=1, stop=len_segment[i], base=0.5,
                                                                   num=len_segment[i])).sum()
        temp4 = lb_segment[i] + (ub_segment[i] - lb_segment[i]) * temp3 / adjust
        Phen[:, i] = temp4
    return Phen


def ranking(func, Phen):
    FitV = []
    for i in Phen:
        FitV.append(- func(i)) # 为了统一求最小值，所以设置函数值越小，适应性越高
    FitV = np.array(FitV)
    return FitV


def selection(FitV):
    # 轮盘赌
    # FitV: 适应度
    # index
    # sel_index=[]选出的序号
    FitV = FitV - FitV.min() + 1e-10
    cum_FitV = FitV.cumsum() / FitV.sum()

    sel_index = []
    for k in FitV:
        p = np.random.rand()
        for i, j in enumerate(cum_FitV):
            if p <= j:
                sel_index.append(i)
                break
    return sel_index


def crtbp(pop=10, Lind=30):
    # pop: 个体数量
    # Lind: 基因数量
    # Chrom: 种群，shape=pop, Lind
    Chrom = np.random.randint(low=0, high=2, size=(pop, Lind))
    return Chrom


def crossover(Chrom):
    # 奇数个的处理
    pop, Lind = Chrom.shape
    i = int(pop / 2)
    Chrom1 = np.concatenate([Chrom[::2, :i], Chrom[1::2, i:]], axis=1)
    Chrom2 = np.concatenate([Chrom[1::2, :i], Chrom[0::2, i:]], axis=1)
    Chrom = np.concatenate([Chrom1, Chrom2], axis=0)
    return Chrom


def mut(Chrom, Pm=0.01):
    # 变异
    pop, Lind = Chrom.shape
    mask = (np.random.rand(pop, Lind) < Pm) * 1
    return (mask + Chrom) % 2


def demo_func1(p):
    x, y, z = p
    return x ** 2 + y ** 2 + z ** 2


def ga(func=demo_func1, pop=50, iter_max=200, lb=[-1, -10, -5], ub=[2, 10, 2], precision=None, Pm=0.001):
    Lind = []
    if precision is None: precision = [None for i in lb]
    for i in range(len(lb)):
        if precision[i] is None: precision[i] = 1e-7
        Lind_segment = 1
        while 2 ** Lind_segment < (ub[i] - lb[i]) / precision[i]:
            Lind_segment += 1
        Lind.append(Lind_segment)

    Chrom = crtbp(pop, sum(Lind))
    FieldD = np.array([Lind, lb, ub])

    FitV_history = []
    for i in range(iter_max):
        Chrom = crossover(Chrom)
        Chrom = mut(Chrom, Pm=Pm)
        Phen = bs2rv(Chrom, FieldD)  # func的输入
        FitV = ranking(func, Phen)  # func的输出，ndarray
        sel_index = selection(FitV)  # index，选中的基因
        Chrom = Chrom[sel_index, :]  # 选出基因

        general_best = Phen[FitV.argmax(), :]
        FitV_history.append(FitV)
    return general_best, func(general_best), FitV_history

def plot_FitV(FitV_history):
    FitV_history = pd.DataFrame(FitV_history)
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    for i in FitV_history.columns:
        ax1.plot(FitV_history.index, FitV_history.loc[:, i], '.', color='red')

    plt_mean = FitV_history.mean(axis=1)
    plt_max = FitV_history.max(axis=1)
    ax1.plot(plt_mean.index, plt_mean, label='mean')
    ax1.plot(plt_max.index, plt_max, label='max')

    ax2 = fig.add_subplot(212)
    ax2.plot(plt_max.index, plt_max.cummax())
    plt.show()