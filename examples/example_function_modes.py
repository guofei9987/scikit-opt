import time
import datetime
import os

print(__name__,os.getpid())

def io_task(name):
    ppid = os.getppid()
    pid = os.getpid()
    start_time = datetime.datetime.now().strftime('%S.%f')
    time.sleep(1)
    end_time = datetime.datetime.now().strftime('%S.%f')
    print('io task, name={name},__name__={__name__}, pid={pid}, ppid ={ppid}, start_time={start_time}, end_time={end_time}\n'.
          format(name=name,__name__=__name__, pid=pid, ppid=ppid, start_time=start_time, end_time=end_time))
    return name

def cpu_task(name):
    ppid = os.getppid()
    pid = os.getpid()
    start_time = datetime.datetime.now().strftime('%S.%f')
    g_search_list = list(range(10000))
    count = 0
    for i in range(10000):
        count += pow(3 * 2, 3 * 2) if i in g_search_list else 0

    end_time = datetime.datetime.now().strftime('%S.%f')
    print('cpu task, name={name}, pid={pid}, ppid ={ppid}, start_time={start_time}, end_time={end_time}\n'.
          format(name=name, pid=pid, ppid=ppid, start_time=start_time, end_time=end_time))
    return name




import datetime
# from tmp5 import io_task
# from tmp5 import cpu_task
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import Pool

# %%
# if __name__ == '__main__':
start = datetime.datetime.now()
pool = ThreadPool()  # ThreadPool(4), 不指定进程数，则使用全部线程
pool.map(io_task, range(10))  # 返回list，就是结果
print(datetime.datetime.now() - start)

start = datetime.datetime.now()
pool = Pool()
pool.map(io_task, range(10))  # 返回list，就是结果
print(datetime.datetime.now() - start)

start = datetime.datetime.now()
list(map(cpu_task, range(10)))
print('普通任务', datetime.datetime.now() - start)

start = datetime.datetime.now()
pool = ThreadPool()  # ThreadPool(4), 不指定进程数，则使用全部线程
pool.map(cpu_task, range(10))  # 返回list，就是结果
print(datetime.datetime.now() - start)

start = datetime.datetime.now()
pool = Pool()
pool.map(cpu_task, range(10))  # 返回list，就是结果
print(datetime.datetime.now() - start)

