## 3 types of Simulated Annealing
模拟退火有三种具体形式  
‘fast’:
```
u ~ Uniform(0, 1, size = d)
y = sgn(u - 0.5) * T * ((1 + 1/T)**abs(2*u - 1) - 1.0)

xc = y * (upper - lower)
x_new = x_old + xc

c = n * exp(-n * quench)
T_new = T0 * exp(-c * k**quench)
```

‘cauchy’:
```
u ~ Uniform(-pi/2, pi/2, size=d)
xc = learn_rate * T * tan(u)
x_new = x_old + xc

T_new = T0 / (1 + k)
```

‘boltzmann’:
```
std = minimum(sqrt(T) * ones(d), (upper - lower) / (3*learn_rate))
y ~ Normal(0, std, size = d)
x_new = x_old + learn_rate * y

T_new = T0 / log(1 + k)
```
### 代码示例
#### 1. Fast Simulated Annealing
-> Demo code: [examples/demo_sa.py#s4](https://github.com/guofei9987/scikit-opt/blob/master/examples/demo_sa.py#L17)
```python
from sko.SA import SAFast

sa_fast = SAFast(func=demo_func, x0=[1, 1, 1], T_max=1, T_min=1e-9, q=0.99, L=300, max_stay_counter=150)
sa_fast.run()
print('Fast Simulated Annealing: best_x is ', sa_fast.best_x, 'best_y is ', sa_fast.best_y)

```
#### 2. Boltzmann Simulated Annealing
-> Demo code: [examples/demo_sa.py#s5](https://github.com/guofei9987/scikit-opt/blob/master/examples/demo_sa.py#L24)
```python
from sko.SA import SABoltzmann

sa_boltzmann = SABoltzmann(func=demo_func, x0=[1, 1, 1], T_max=1, T_min=1e-9, q=0.99, L=300, max_stay_counter=150)
sa_boltzmann.run()
print('Boltzmann Simulated Annealing: best_x is ', sa_boltzmann.best_x, 'best_y is ', sa_fast.best_y)

```
#### 3. Cauchy Simulated Annealing
-> Demo code: [examples/demo_sa.py#s6](https://github.com/guofei9987/scikit-opt/blob/master/examples/demo_sa.py#L31)
```python
from sko.SA import SACauchy

sa_cauchy = SACauchy(func=demo_func, x0=[1, 1, 1], T_max=1, T_min=1e-9, q=0.99, L=300, max_stay_counter=150)
sa_cauchy.run()
print('Cauchy Simulated Annealing: best_x is ', sa_cauchy.best_x, 'best_y is ', sa_cauchy.best_y)
```
