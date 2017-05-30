
import numpy as np
import matplotlib.pyplot as plt
#from mouseworld.myspace import *
from joblib import Parallel, delayed
import multiprocessing

class Value_layer :

    def __init__(self, unique_id, width, height, torus):
        
        self.height = height
        self.width = width
        self.torus = torus
        self.unique_id =unique_id
        
        self.grid = np.zeros(shape=(width,height))

#         for x in range(self.width):
#             col = []
#             for y in range(self.height):
#                 col.append(0)
#             self.grid.append(col)
    
    def torus_adj(self, coord, dim_len):
        if self.torus:
            coord %= dim_len
        return coord
    
    def out_of_bounds(self, pos):
        x, y = pos
        return x < 0 or x >= self.width or y < 0 or y >= self.height
    
    def coord_iter(self):
        """ An iterator that returns coordinates as well as cell contents. """
        for row in range(self.width):
            for col in range(self.height):
                yield self.grid[row][col], row, col 
    
    def iter_neighbors(self, pos, moore,
                       include_center=False, radius=1):
        neighborhood = self.iter_neighborhood(
            pos, moore, include_center, radius)
        return self.iter_cell_list_contents(neighborhood)
    
    def iter_cell_list_contents(self, cell_list):
        return (self.grid[x][y] for x, y in cell_list)
    
    def avg_neighborhood(self, pos):
        val=0
        x, y = pos
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                if dx == 0 and dy == 0:
                    continue
        
                px = self.torus_adj(x + dx, self.width)
                py = self.torus_adj(y + dy, self.height)

                # Skip if new coords out of bounds.
                if(self.out_of_bounds((px, py))):
                    continue

                val += self.grid[px][py]
                
        return val/8
    
    def add_value(self, pos, value) :
        x, y = pos
        self.grid[x][y] += value

    def neighbor_avg(self,pos) :
        val = self.iter_neighbors(pos, moore = True, include_center=False, radius=1)
        return sum(val)/8
    
    def diffuse(self, evap_const, diff_const) :
        old = self
        for row in range(self.width):
            for col in range(self.height):
                self.grid[row][col] = evap_const * (old.grid[row][col] + diff_const * (old.avg_neighborhood((row,col)) - old.grid[row][col]))
            

grid0 = Value_layer('example0',100,100,True)
grid1 = Value_layer('example1',100,100,True)
grid2 = Value_layer('example2',100,100,True)
grid3 = Value_layer('example3',100,100,True)

num_cores = multiprocessing.cpu_count()
def show(grid) :
    cell_values = np.zeros((grid.width, grid.height))
    for cell in grid.coord_iter():
        cell_value, x, y = cell
        cell_values[x][y] = cell_value
    plt.imshow(cell_values, interpolation='nearest')
    plt.colorbar()
    plt.show()

    
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