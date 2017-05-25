
# coding: utf-8

# In[15]:

import numpy as np
import matplotlib.pyplot as plt
from mouseworld.myspace import *
# from decimal import *

# getcontext().prec = 7

grid = Value_layer('example',100,100,True)
#grid2 = Value_layer(100,100,True)
#grid.add_value((99,99),100)

def show(grid) :
    cell_values = np.zeros((grid.width, grid.height))
    for cell in grid.coord_iter():
        cell_value, x, y = cell
        cell_values[x][y] = cell_value
    plt.imshow(cell_values, interpolation='nearest')
    plt.colorbar()
    plt.show()

#show(grid)
for i in range(100) :
    grid.add_value((50,50),0.3)
    grid.diffuse(0.95,0.8)
show(grid)
print(grid.get_value((50,50)))
print(grid.get_value((45,45)))
print(grid.get_value((40,40)))
print(grid.get_value((35,35)))
print(grid.get_value((30,30)))
print(grid.get_value((25,25)))
print(grid.get_value((20,20)))


# In[2]:

from mouseworld.myspace import *

grid = Value_layer('dada',100,100,True)
grid.get_value((50,50))


# In[14]:

from decimal import *
getcontext().prec = 7
getcontext()

