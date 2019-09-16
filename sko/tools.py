


def func_transformer(func):
    '''
    transform this kind of function:
    ```
    def demo_func(x):
        x1, x2, x3 = x
        return x1 ** 2 + (x2 - 0.05) ** 2 + x3 ** 2
    ```
    into this kind of function:
    ```
    def demo_func(x):
        x1, x2, x3 = x
        return x1 ** 2 + x2 ** 2 + x3 ** 2
    ```
    :param func:
    :return:
    '''

    prefered_function_format = '''
    def demo_func(x):
        x1, x2, x3 = x
        return x1 ** 2 + (x2 - 0.05) ** 2 + x3 ** 2
    '''
    if func.__code__.co_argcount == 1:
        return func
    elif func.__code__.co_argcount > 1:

        def func_transformed(x):
            args = tuple(x)
            return func(*args)

        return func_transformed
    else:
        raise ValueError('''
        object function error,
        function should be like this:
        ''' + prefered_function_format)

