
# coding: utf-8

# In[145]:

import pandas as pd
import numpy as np
from inspect import signature

def wait(x) :
    print('lala%i'%x)
    

def search() :
    print('lbbla')
    
#df = pd.DataFrame(np.nan, index=[1], columns=('Verb','Noun_type','Noun_group','Function'))
df = pd.DataFrame([['Wait', 3, -1, wait, None], ['Search', 12, 0, search, 12]], columns=('Verb', 'Noun_group', 'Value', 'Function', 'Arg_1'))

cf = pd.DataFrame([['Approach', 'Food_2', -1, wait, np.nan]], columns=('Verb', 'Noun_group', 'Value', 'Function', 'Arg_1'))    
df.loc[df.index.max() + 1] = ['Approach','Food_1',(3,3),1,1]
df.loc[df.index.max() + 1] = ['Feed','Food_1',1,1,1]
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

df['Value'][df['Verb'] == 'Search'] = 5
# temp = df['Verb'] == 'Wait'
# temp2 = df['Noun_group'].isnull
# temp2
#cf.loc[0]['Noun_group']
df.loc['fdfd'] = ['Feed','Food_1',1,1,1]
df.set_value('fdfd','Value',5)
df.as_matrix()


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


# In[17]:

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



