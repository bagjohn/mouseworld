
# coding: utf-8

# In[16]:

import pandas as pd
import numpy as np
from inspect import signature

def wait(x) :
    print('lala%i'%x)
    

def search() :
    print('lbbla')
    
#df = pd.DataFrame(np.nan, index=[1], columns=('Verb','Noun_type','Noun_group','Function'))
df = pd.DataFrame([['Wait', 3, -0, wait, None], ['Search', 12, 0, search, 12]], columns=('Verb', 'Noun_group', 'Value', 'Function', 'Arg_1'))

cf = pd.DataFrame([['Approach', 'Food_2', -1, wait, np.nan]], columns=('Verb', 'Noun_group', 'Value', 'Function', 'Arg_1'))    
df.loc[df.index.max() + 1] = ['Approach','Food_1',0,1,1]
df.loc[df.index.max() + 1] = ['Feed','Food_1',0,1,1]
#df.append(['Feed','Food', 5, 'self.feed(food)'])
#df['Function'][1](df['Arg_1'][1])
sig = signature(df['Function'][0])
num_args = len(sig.parameters)

a = df.loc[(df['Verb'] == 'Feed') & (df['Noun_group'] == cf.loc[0]['Noun_group'])]
#df['Value'].min()
a =df.loc[[0]]

a=df.loc[df['Verb'] =='Approach']
#b =cf
#a['Verb'][0]
#df['Value'].idxmin()
df['Noun_group'][1] == df['Arg_1'][1]
df.head(n=1).loc[0]#['Verb']
df.ix[1]
#a['Value'].idxmax()
#a.loc[a['Value'].idxmax()]
g = None
g is not None#a.loc[]

df['Value'][df['Verb'] == 'Search'] = 0
# temp = df['Verb'] == 'Wait'
# temp2 = df['Noun_group'].isnull
# temp2
#cf.loc[0]['Noun_group']
df.loc['fdfd'] = ['Feed','Food_1',0,1,1]
df.set_value('fdfd','Value',0)
df.as_matrix()
df
print(df['Value'].idxmax())


# In[4]:

import pandas as pd

predator_groups = [('Predator_group_%i'%i) for i in range(10)]
food_groups = [('Food_group_%i'%i) for i in range(8)]
groups = food_groups + predator_groups

primary_values = dict(zip(groups, [0.0001]*18))
primary_values[food_groups[3]] += 8
primary_values
#food_groups[3]

trivial_possible_actions = pd.DataFrame([['Wait', None, 0, 'self.wait', None], 
                                                      ['Search', None, 0.00001, 'self.search_for_odor', None]], 
                                                     columns=('Verb', 'Noun_group', 'Value', 'Function', 'Arg_1'))

print(trivial_possible_actions)
actions = trivial_possible_actions
print(actions)
actions.loc[actions.index.max() + 1] = ['Approach', 'groups[i]', 'temp[i][0] * self.hunger_status * value', 'self.approach', 'temp[i]']
print(trivial_possible_actions)
print(actions)

actions = trivial_possible_actions
print(trivial_possible_actions)
print(actions)


# In[26]:

import pandas as pd
import numpy as np

a=pd.DataFrame([[18, 'F'],[50, 'M']],columns = ('Age','Sex'))
print(a)
print(type(a))
b=a
print(type(b))
print(b)
b.loc[b.index.max() + 1] = [30,'M']
print(b)
print(a)

e=b.loc[b.index.max()]
print(e)
print(type(e))
b.loc[b.index.max() + 1] = [40,'F']
print(b)
print(e)


c=np.array([[18,'F'],[50,'M']])
print(c)
d=c
print(d)
d = np.append(d,[[30,'M']], axis=0)
print(d)
print(c)


# In[19]:

import pandas as pd
import numpy as np

a=[[1,2],[3,4]]
b=pd.DataFrame(a,columns = ('Age','Sex'))

df = pd.DataFrame({"A":[], "B":[]})
df.loc[0]=[5,6]
df.index.max() +1
a = pd.Series([5,6], index=("A", "B"))
print(a)
df2=df.add(a)
df.empty
df.index[1]


# In[24]:

import numpy as np

np.ones((5,1))*4


# In[3]:


class Sex :
    def __init__(self) :
        self.a = 5
    def add1(self, x) :
        x=x+1
        self.a += 3
        return False

sex = Sex()
#sex.add1(3)
if sex.add1(3) :
    pass
print(sex.a)


# In[10]:

import pandas as pd
import numpy as np

action_history = pd.DataFrame({"Verb":[], "Noun_group":[], "Duration":[], "Benefit":[], "Termination":[]})
action_history = pd.DataFrame([], columns=('Verb', 'Noun_group', 'Duration', 'Benefit', 'Closure'))
action_history


# In[81]:

import numpy as np

def mutate_genome(genome) :
    #genome = self.genome
    for i in range(len(genome)) :
        rand = np.random.uniform(low=-1.0, high=1.0, size=None)
        if abs(rand) <= 0.1 :
            genome[i] += np.sign(rand) * 0.1
            if genome[i] <= 0 :
                genome[i] = 0
    return np.around(genome, decimals = 2)

a = [ 0.18,  0.57,  0.01,  0.91,  0.69]
a
len(a)
mutate_genome(a)


# In[8]:

get_ipython().run_cell_magic('writefile', 'mouseworld/check_multiprocessing.py', '\nimport numpy as np\nimport matplotlib.pyplot as plt\n#from mouseworld.myspace import *\nfrom joblib import Parallel, delayed\nimport multiprocessing\n\nclass Value_layer :\n\n    def __init__(self, unique_id, width, height, torus):\n        \n        self.height = height\n        self.width = width\n        self.torus = torus\n        self.unique_id =unique_id\n        \n        self.grid = []\n\n        for x in range(self.width):\n            col = []\n            for y in range(self.height):\n                col.append(0)\n            self.grid.append(col)\n    \n    def torus_adj(self, coord, dim_len):\n        if self.torus:\n            coord %= dim_len\n        return coord\n    \n    def out_of_bounds(self, pos):\n        x, y = pos\n        return x < 0 or x >= self.width or y < 0 or y >= self.height\n    \n    def coord_iter(self):\n        """ An iterator that returns coordinates as well as cell contents. """\n        for row in range(self.width):\n            for col in range(self.height):\n                yield self.grid[row][col], row, col \n    \n    def iter_neighbors(self, pos, moore,\n                       include_center=False, radius=1):\n        neighborhood = self.iter_neighborhood(\n            pos, moore, include_center, radius)\n        return self.iter_cell_list_contents(neighborhood)\n    \n    def iter_cell_list_contents(self, cell_list):\n        return (self.grid[x][y] for x, y in cell_list)\n    \n    def iter_neighborhood(self, pos, moore,\n                          include_center=False, radius=1):\n        x, y = pos\n        coordinates = set()\n        for dy in range(-radius, radius + 1):\n            for dx in range(-radius, radius + 1):\n                if dx == 0 and dy == 0 and not include_center:\n                    continue\n                # Skip diagonals in Von Neumann neighborhood.\n                if not moore and dy != 0 and dx != 0:\n                    continue\n                # Skip diagonals in Moore neighborhood when distance > radius\n                if moore and radius > 1 and (dy ** 2 + dx ** 2) ** .5 > radius:\n                    continue\n                # Skip if not a torus and new coords out of bounds.\n                if not self.torus and (not (0 <= dx + x < self.width) or\n                                       not (0 <= dy + y < self.height)):\n                    continue\n\n                px = self.torus_adj(x + dx, self.width)\n                py = self.torus_adj(y + dy, self.height)\n\n                # Skip if new coords out of bounds.\n                if(self.out_of_bounds((px, py))):\n                    continue\n\n                coords = (px, py)\n                if coords not in coordinates:\n                    coordinates.add(coords)\n                    yield coords\n    \n    def add_value(self, pos, value) :\n        x, y = pos\n        self.grid[x][y] += value\n\n    def neighbor_avg(self,pos) :\n        val = self.iter_neighbors(pos, moore = True, include_center=False, radius=1)\n        return sum(val)/8\n    \n    def diffuse(self, evap_const, diff_const) :\n        old = self\n        for row in range(self.width):\n            for col in range(self.height):\n                self.grid[row][col] = evap_const * (old.grid[row][col] + diff_const * (old.neighbor_avg((row,col)) - old.grid[row][col]))\n            \n\ngrid0 = Value_layer(\'example0\',100,100,True)\ngrid1 = Value_layer(\'example1\',100,100,True)\ngrid2 = Value_layer(\'example2\',100,100,True)\ngrid3 = Value_layer(\'example3\',100,100,True)\n\nnum_cores = multiprocessing.cpu_count()\ndef show(grid) :\n    cell_values = np.zeros((grid.width, grid.height))\n    for cell in grid.coord_iter():\n        cell_value, x, y = cell\n        cell_values[x][y] = cell_value\n    plt.imshow(cell_values, interpolation=\'nearest\')\n    plt.colorbar()\n    plt.show()\n\n    \nfor i in range(10) :\n    grid0.add_value((50,50),1)\n    grid1.add_value((50,50),1)\n    grid2.add_value((50,50),1)\n    grid3.add_value((50,50),1)\n#     Parallel(n_jobs=num_cores)(delayed(layer.diffuse)(0.95,0.8) for layer in [grid0,grid1,grid2,grid3])\n    grid0.diffuse(0.95,0.8)\n    grid1.diffuse(0.95,0.8)\n    grid2.diffuse(0.95,0.8)\n    grid3.diffuse(0.95,0.8)\n    \nshow(grid0)\nshow(grid1)\nshow(grid2)\nshow(grid3)\n\n# print(grid.get_value((50,50)))\n# print(grid.get_value((45,45)))\n# print(grid.get_value((40,40)))\n# print(grid.get_value((35,35)))\n# print(grid.get_value((30,30)))\n# print(grid.get_value((25,25)))\n# print(grid.get_value((20,20)))')


# In[11]:

import numpy as np
import math

class Example :
    
    def __init__(self) :
        self.g = 0
    
def add1(x) :
    x += 1
    return x

def xxx(y) :
    add1(y)
    return y

w = 2
print(w)
w =xxx(w)
print(w)

ex = Example()
ex.g

hbjbhhbj = add1(ex.g)


# In[13]:

import pandas as pd

current_action = pd.DataFrame([['Wait', None, 0, 0, False]], 
                                           columns=('Verb', 'Noun_group', 'Duration', 'Benefit', 'Closure'))
a = current_action.loc[0]
if current_action['Verb'][0] == 'Wait' :
    print('dsds')
type(a)
current_action.loc[3] = a
b = pd.Series(['Wait', None, 0, 0, False], 
                                           index=('Verb', 'Noun_group', 'Duration', 'Benefit', 'Closure'))
current_action.loc[2] = b
current_action

# b= None
b is not None


# In[4]:

a=None

a!=None


# In[2]:

from joblib import Parallel, delayed

import multiprocessing

     

# what are your inputs, and what operation do you want to
# perform on each input. For example...

inputs = range(10)

def processInput(i):

    return i * i

 

num_cores = multiprocessing.cpu_count()

     

results = Parallel(n_jobs=num_cores)(delayed(processInput)(i) for i in inputs)
results


# In[150]:

get_ipython().run_cell_magic('writefile', 'mouseworld/space_surface.py', "\nfrom scipy.stats import multivariate_normal\n#import matplotlib.pyplot as plt\nimport numpy as np\nimport pandas as pd\n\n#from mouseworld.myspace import ContinuousSpace\n\nclass Space_surface :\n    def __init__(self, unique_id): \n        self.unique_id = unique_id\n        #self.space  = space\n        #self.agent_list = []\n        columns = ['weight', 'loc[x]', 'loc[y]', 'scale']\n        #dtype={'weight':float,'loc':(float, float), 'scale':int}\n        self.agent_list = pd.DataFrame(data=np.zeros((0,len(columns))), columns=columns) \n        \n    def add_agent(self, agent) :\n        unique_id = agent.unique_id\n        scale = agent.odor_std\n        weight = agent.odor_strength\n        loc = agent.pos\n        self.agent_list.loc[unique_id] = [weight, loc[0], loc[1], scale]\n        #self.agent_list.append([unique_id, weight, loc, scale])\n        \n    def update_surface(self) :\n        data = self.agent_list.as_matrix()\n        #data = self.agent_list.drop(['unique_id'], axis=1)\n        #data = data.values\n        self.surface = [[0, None]]*len(data)\n        for i in range(len(data)):\n            self.surface[i][0] = data[i,0]\n            pos = [data[i,1], data[i,2]]\n            self.surface[i][1] = multivariate_normal(pos, data[i,3])\n        #print(data)\n        \n    def update_agent_location(self, agent_unique_id, agent_pos) :\n#         temp = self.agent_list\n#         temp.ix[agent_unique_id]['loc'] = agent_pos\n#         ind = temp[:,0] == agent.unique_id\n#         temp[ind][2] = agent_pos\n        self.agent_list.set_value(agent_unique_id,'loc[x]',agent_pos[0])\n        self.agent_list.set_value(agent_unique_id,'loc[y]',agent_pos[1])\n\n        \n    def remove_agent(self, agent) :\n        temp = self.agent_list\n        #self.agent_list = temp[temp[:,0] != agent.unique_id]\n        temp.drop(temp.index[agent.unique_id])\n        self.agent_list = temp\n         \n    def get_value(self, pos) :\n        value = 0\n        for i in self.surface :\n            value += i[0] * i[1].pdf(pos)\n        return value\n    ")


# In[153]:

d = dict(zip(['a', 'b', 'c'], [[0.002, 0]]*3))
d['a']


# In[58]:

from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

from mouseworld.myspace import ContinuousSpace

width = 100
height = 100

space = ContinuousSpace(width, height, True, x_min=0, y_min=0,
            grid_width=width, grid_height=height)

x, y = np.mgrid[0:width:1, 0:height:1]
pos = np.empty(x.shape + (2,))
pos[:, :, 0] = x; pos[:, :, 1] = y
rv1 = multivariate_normal([5, 5], 32) 
rv2 = multivariate_normal([80, 80], 8)
rv3 = multivariate_normal([90, 3], 16)

def rv(x) :
    rv = rv1.pdf(x) + rv2.pdf(x) + rv3.pdf(x)
    return rv
plt.contourf(x, y, rv(pos))
#plt.contourf(x, y, rv2.pdf(pos))
#plt.contourf(x, y, 0.5*rv.pdf(pos) + 0.5*rv2.pdf(pos))
plt.show()
c = [1,1]
a = [0.5, -0.2]
print(rv1.pdf(a))
print(rv2.pdf(a))
print(rv3.pdf(a))
print(rv(a))

print(rv1.pdf(c))
print(rv2.pdf(c))
print(rv3.pdf(c))
print(rv(c))

#pos[:, :, 0]
#pos


# In[37]:

"""
=========================================
Density Estimation for a Gaussian mixture
=========================================
Plot the density estimation of a mixture of two Gaussians. Data is
generated from two Gaussians with different centers and covariance
matrices.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn import mixture

n_samples = 300

# generate random sample, two components
np.random.seed(0)

# generate spherical data centered on (20, 20)
shifted_gaussian = np.random.randn(n_samples, 2) + np.array([20, 20])

# generate zero centered stretched Gaussian data
C = np.array([[0., -0.7], [3.5, .7]])
stretched_gaussian = np.dot(np.random.randn(n_samples, 2), C)

# concatenate the two datasets into the final training set
X_train = np.vstack([shifted_gaussian, stretched_gaussian])

# fit a Gaussian Mixture Model with two components
clf = mixture.GaussianMixture(n_components=2, covariance_type='full')
clf.fit(X_train)

# display predicted scores by the model as a contour plot
x = np.linspace(-20., 30.)
y = np.linspace(-20., 40.)
X, Y = np.meshgrid(x, y)
XX = np.array([X.ravel(), Y.ravel()]).T
Z = -clf.score_samples(XX)
Z = Z.reshape(X.shape)

CS = plt.contour(X, Y, Z, norm=LogNorm(vmin=1.0, vmax=1000.0),
                 levels=np.logspace(0, 3, 10))
CB = plt.colorbar(CS, shrink=0.8, extend='both')
plt.scatter(X_train[:, 0], X_train[:, 1], .8)

plt.title('Negative log-likelihood predicted by a GMM')
plt.axis('tight')
plt.show()


# In[16]:

import numpy as np

a = ['rere','trtr','fdsfs']
b = ['ewew', 'cdcd', 'sasa']
#np.concatenate((a, b), axis=0)[2]
(a+b)[2]


# In[94]:

from numpy.random import randn
from pandas import DataFrame
df = DataFrame(randn(10, 2), columns=('kaka','lala'))
a = df.query('kaka == 0.2' and 'lala == 0.2')
if a.empty :
    print('dssd')
else :
    print('dsds')
#df[df.a > df.b]  # same result as the previous expression


# In[67]:

import numpy as np

def search() :
    print('lbbla')
#a = [4,5,,3]
search(())


# In[26]:

def a(x):
    y = x*x*x
    return y

def b(x):
    y = x*2
    return y

def c(x):
    y = x/2
    return y

def d(x):
    y = c(x)*b(x)*a(x)
    return y

def e(z):
    print('lala_%s'%z)
    
ls=[]
print('new22')
ls.append(c(3))
ls.append(b(3))
ls.append(3)
ls.append(d)
print('new')
ls[0]
print('new')
ls[1]
print('new')
ls[2]
print('new')
ls[3](3)


# In[ ]:



