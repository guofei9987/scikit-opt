import numpy as np

def ranking(self):
    # GA select the biggest one, but we want to minimize func, so we put a negative here
    if not self.has_constraint:
        self.FitV = -self.Y
    else:
        penalty_eq = np.array([np.sum(np.abs([c_i(x) for c_i in self.constraint_eq])) for x in self.X])
        penalty_ueq = np.array([np.sum(np.abs([max(0, c_i(x)) for c_i in self.constraint_ueq])) for x in self.X])
        self.FitV = -self.Y - 1e5 * penalty_eq - 1e5 * penalty_ueq
    return self.FitV


def ranking_linear(self):
    '''
    For more details see [Baker1985]_.

    :param self:
    :return:

    .. [Baker1985] Baker J E, "Adaptive selection methods for genetic
    algorithms, 1985.
    '''
    self.FitV = np.argsort(np.argsort(-self.Y))
    return self.FitV

