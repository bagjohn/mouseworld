
# coding: utf-8

# In[2]:

#%%writefile mouseworld/evolve_Value_Layer.py

import numpy as np
import matplotlib.pyplot as plt
from mouseworld.myspace import *
from joblib import Parallel, delayed
import multiprocessing
import time

a = time.time()
grid0 = Value_layer('example0',100,100)
grid1 = Value_layer('example1',100,100)
grid2 = Value_layer('example2',100,100)
grid3 = Value_layer('example3',100,100)

num_cores = multiprocessing.cpu_count()
def show(grid) :
    cell_values = np.zeros((grid.width, grid.height))
    for cell in grid.coord_iter():
        cell_value, x, y = cell
        cell_values[x][y] = cell_value
    plt.imshow(cell_values, interpolation='nearest')
    plt.colorbar()
    plt.show()

b = time.time()   
for i in range(10) :
    grid0.add_value((50,50),1)
    grid1.add_value((50,50),1)
    grid2.add_value((50,50),1)
    grid3.add_value((50,50),1)
#     Parallel(n_jobs=num_cores)(delayed(layer.diffuse)(0.95,0.8) for layer in [grid0,grid1,grid2,grid3])
    grid0.diffuse(0.95,0.8)
    grid1.diffuse(0.95,0.8)
    grid2.diffuse(0.95,0.8)
    grid3.diffuse(0.95,0.8)

c = time.time()

print(b-a)
print(c-b)

show(grid0)
show(grid1)
show(grid2)
show(grid3)

# print(grid.get_value((50,50)))
# print(grid.get_value((45,45)))
# print(grid.get_value((40,40)))
# print(grid.get_value((35,35)))
# print(grid.get_value((30,30)))
# print(grid.get_value((25,25)))
# print(grid.get_value((20,20)))


# In[1]:

import itertools

pos =[(4,3) for i in range(100)]
header = [7]
params = (pos, header)
param_combs = list(itertools.product(*params))  
param_combs


# In[26]:

import numpy as np

b = dict(zip(['aa', 'ba', 'ca', 'da', 'ea'],[0] * 5))
#a = dict(zip(['a', 'b', 'c', 'd', 'e'],[np.zeros(5)] * 5))
a = dict(zip(['a', 'b', 'c', 'd', 'e'],[b]*5))
# a['c'][2] = 5
#a['a'] = 5
# sum(a['b'])
# a['b'][3]=4
a['a']['ba'] = 4
a


# In[55]:

import pandas as pd

a=['g1', 'g2', 'g3', 'g4', 'g5']
b=['o1', 'o2', 'o3', 'o4', 'o5']
c=pd.DataFrame([np.zeros(5)] * 5, index = a, columns = b)

c.ix['g3']['o3'] = 4
# sum(c.ix['g2'])
d = c.ix['g3']
d['o1'] = 2
d
c.ix['g1']['o3'] = -2
c['o3'].index.min()
c


# In[57]:

import numpy as np
sensor_vector = [0,0]
sensor_vector = [np.mean(sensor_vector), sensor_vector[0]-sensor_vector[1]]
sensor_vector


# In[4]:

ind = 3
print(ind)
ind = "{0:03}".format(ind)
print(ind)


# In[97]:

primary_values = dict(zip(['a','b'], [1]*2))
primary_values['a']


# In[95]:

a = [[0,0]] * 3
a[1][1] = 5
a


# In[13]:



def update_odor_values (gain) :
        vector = sensor_vector
        values = secondary_values
        for i in range(len(vector)) :
            if vector[i][0] > 0 :
                
                # BIO : CLASSICAL CONDITIONING
                error = gain * vector[i][0] - values[odor_layer_names[i]]
                values[odor_layer_names[i]] += error * 0.1
        return values

sensor_vector = [[0,0], [0.5, 0.5], [0,0], [0.3, 0], [0,0], [0.8,0.4], [0.9, 0], [0.1,0.9], [0,0], [0.5,0.8]]
odor_layer_names =['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']    
secondary_values = dict(zip(odor_layer_names, [0]*10))

for i in range(10):
    secondary_values = update_odor_values(20)

secondary_values

