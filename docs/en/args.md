
## input parameters

Use help (like `help(GA)`, `GA?`) to see the details.
```python
import sko

help(sko.GA.GA)
help(sko.GA.GA_TSP)
help(sko.PSO.PSO)
help(sko.DE.DE)
help(sko.SA.SA)
help(sko.SA.SA_TSP)
help(sko.ACA.ACA_TSP)
help(sko.IA.IA_TSP)
help(sko.AFSA.AFSA)
```

### GA

| input parameters | default value    | meaning     |
|-----------------|--------|------------------------|
| func            | \-     | objective function    |
| n\_dim          | \-     | dim of objective function (number of input parameters) |
| size\_pop       | 50     | size of population  |
| max\_iter       | 200    | max iteration     |
| prob\_mut       | 0\.001 | probability of mutation     |
| lb              | \-1    | lower bound of variables. Can be int/float/list  |
| ub              | 1      | upper bound of variables. Can be int/float/list  |
| constraint\_eq  | tuple()    | equal constraint        |
| constraint\_ueq | tuple()    | unequal constraint                  |
| precision       | 1e\-7  | precision，int/float or list |


### GA_TSP

| input parameters | default value    | meaning     |
|-----------|--------|--------|
| func      | \-     | objective function  |
| n\_dim    | \-     | num of cities  |
| size\_pop | 50     | size of population   |
| max\_iter | 200    | max iteration |
| prob\_mut | 0\.001 | probability of mutation |


### PSO

| input parameters | default value    | meaning     |
|-----------|------|----------|
| func      | \-   | objective function  |
| n\_dim    | \-   | dim of objective function  |
| size\_pop | 50   | size of population   |
| max\_iter | 200  | max iteration  |
| lb        | None | lower bound of variables |
| ub        | None | upper bound of variables |
| w         | 0\.8 | inertia weight  |
| c1        | 0\.5 | cognitive parameter |
| c2        | 0\.5 | social parameter |
| constraint\_ueq | tuple()    | unequal constraint                  |

### DE

| input parameters | default value    | meaning     |
|-----------|--------|----------|
| func      | \-     | objective function   |
| n\_dim    | \-     | dim of objective function  |
| size\_pop | 50     | size of population     |
| max\_iter | 200    | max iteration   |
| prob\_mut | 0\.001 | probability of mutation     |
| F         | 0\.5   | coefficient of mutation     |
| lb        | \-1    | lower bound of variables |
| ub        | 1      | upper bound of variables |
| constraint\_eq  | tuple()   | equal constraint                   |
| constraint\_ueq | tuple()    | unequal constraint                 |

### SA

| input parameters | default value    | meaning     |
|--------------------|-------|-------|
| func               | \-    | objective function  |
| x0                 | \-    | initial point |
| T\_max             | 100   | max temperature |
| T\_min             | 1e\-7 | min temperature  |
| L                  | 300   | long of chain  |
| max\_stay\_counter | 150   | cooldown time  |
| lb              | \-1    | lower bound of variables. Can be int/float/list  |
| ub              | 1      | upper bound of variables. Can be int/float/list  |



### SA_TSP

| input parameters | default value    | meaning     |
|--------------------|-------|-------|
| func               | \-    | objective function  |
| x0                 | \-    | initial point |
| T\_max             | 100   | max temperature |
| T\_min             | 1e\-7 | min temperature  |
| L                  | 300   | long of chain  |
| max\_stay\_counter | 150   | cooldown time  |


### ACA_TSP

| input parameters | default value    | meaning     |
|------------------|------|----------------------|
| func             | \-   | objective function                 |
| n\_dim           | \-   | number of cities ( also called dim of objective function)                 |
| size\_pop        | 10   | number of ants                 |
| max\_iter        | 20   | max iteration               |
| distance\_matrix | \-   | distance matrix between cities |
| alpha            | 1    | importance of pheromone         |
| beta             | 2    | importance of fitness       |
| rho              | 0\.1 | evaporation speed of pheromone        |


### IA_TSP

| input parameters | default value    | meaning     |
|-----------|--------|----------------------------------|
| func      | \-     | objective function                             |
| n\_dim    | \-     | number of cities ( also called dim of objective function)                             |
| size\_pop | 50     | size of population                             |
| max\_iter | 200    | max iteration                           |
| prob\_mut | 0\.001 | probability of mutation                             |
| T         | 0\.7   | concentration of antibody  |
| alpha     | 0\.95  | importance of diversity compared to concentration of antibody   |


### AFSA

| input parameters | default value    | meaning     |
|---------------|-------|------------------|
| func          | \-    | objective function             |
| n\_dim        | \-    | dim of objective function          |
| size\_pop     | 50    | size of population             |
| max\_iter     | 300   | max iteration           |
| max\_try\_num | 100   | max try of prey in one movement      |
| step          | 0\.5  | max scale of movement       |
| visual        | 0\.3  | max range of perception         |
| q             | 0\.98 | perception of fish will go down every movement  |
| delta         | 0\.5  | fishes toleration of crowd |

## outputs
*(in this part, `x` is also called value of objective funtion. `x` is also called inputs of objective funtion)*

### GA&GA_TSP

- `ga.generation_best_Y` best Y of every generation
- `ga.generation_best_X` X for best Y of every generation
- `ga.all_history_FitV` fitness value of every generation
- `ga.all_history_Y` function value of every generation and every individual
- `ga.best_y` best y
- `ga.best_x` best x

### DE


- `de.generation_best_Y` best Y of every generation
- `de.generation_best_X` X for best Y of every generation
- `de.all_history_Y` Y of every generation and every individual
- `de.best_y` best y
- `de.best_x` best x


### PSO
- `pso.record_value` location, velocity, function value of every generation and every particles. only when `pso.record_mode = True`.
- `pso.gbest_y_hist` best y every generation
- `pso.best_y` best y （In PSO, use `pso.gbest_x`, `pso.gbest_y`）
- `pso.best_x` best x



### SA

- `sa.generation_best_Y` best Y of every generation
- `sa.generation_best_X` X for best Y of every generation
- `sa.best_x` best x
- `sa.best_y` best y

### ACA
- `sa.generation_best_Y` best Y of every generation
- `sa.generation_best_X` X for best Y of every generation
- `aca.best_y` best y
- `aca.best_x` best x

### AFSA
- `afsa.best_x` best x
- `afsa.best_y` best y

### IA

- `ia.generation_best_Y` best Y of every generation
- `ia.generation_best_X` X for best Y of every generation
- `ia.all_history_FitV` 每一代的每个个体的适应度
- `ia.all_history_Y` 每一代每个个体的函数值
- `ia.best_y` best y
- `ia.best_x` best x
