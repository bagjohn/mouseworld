
# coding: utf-8

# In[5]:

import numpy as np
import matplotlib.pyplot as plt
from mouseworld.myspace import *
import time
import ctypes

# getcontext().prec = 7

grid = Value_layer('example',40,40,True)
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
for i in range(40) :
    grid.add_value((20,20),2)
#     grid.add_value((i,i),1)
#     grid.add_value((80,20),0.3)
#     grid.add_value((50,i),0.3)
#     grid.diffuse(0.95,0.8)
    print ctypes.CDLL('library.so').square(4) # linux or when mingw used on windows# from decimal import *

#     show(grid)
#     time.sleep(1)

# show(grid)

# for i in range(10) :
#     print(grid.get_value((i*10+9,i*10+9)))

# for i in range(10) :
#     print(grid.get_value((50,i*10+9)))

# for i in range(10) :
#     print(grid.get_value((20+i,80+i)))
    
for i in range(10) :
    
    print(grid.get_value((20+i,20+i)))
#     print(grid.get_value((90,90)))
#     print(grid.get_value((96,96)))
#     print(grid.get_value((97,97)))
#     print(grid.get_value((98,98)))
#     print(grid.get_value((99,99)))


# In[ ]:



