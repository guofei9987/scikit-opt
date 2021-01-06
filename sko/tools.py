import numpy as np
from functools import lru_cache 


def func_transformer(func):
    '''
    transform this kind of function:
    ```
    def demo_func(x):
        x1, x2, x3 = x
        return x1 ** 2 + x2 ** 2 + x3 ** 2
    ```
    into this kind of function:
    ```
    def demo_func(x):
        x1, x2, x3 = x[:,0], x[:,1], x[:,2]
        return x1 ** 2 + (x2 - 0.05) ** 2 + x3 ** 2
    ```
    getting vectorial performance if possible
    :param func:
    :return:
    '''

    prefered_function_format = '''
    def demo_func(x):
        x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]
        return x1 ** 2 + (x2 - 0.05) ** 2 + x3 ** 2
    '''

    is_vector = getattr(func, 'is_vector', False)
    is_parallel = getattr(func, 'is_parallel', False)
    is_cached = getattr(func, 'is_cached', False)

    if is_cached:
        @lru_cache(maxsize = None) 
        def func_cached(X):
            return func
        func = func_cached

    if is_vector:
        return func
        
    if is_parallel:
        from multiprocessing.dummy import Pool

        pool = Pool()

        def func_transformed(X):
            return np.array(pool.map(func, X))

        return func_transformed
    else:
        if func.__code__.co_argcount == 1:
            def func_transformed(X):
                return np.array([func(x) for x in X])

            return func_transformed
        elif func.__code__.co_argcount > 1:

            def func_transformed(X):
                return np.array([func(*tuple(x)) for x in X])

            return func_transformed

    raise ValueError('''
    object function error,
    function should be like this:
    ''' + prefered_function_format)
