
from scipy.stats import multivariate_normal
#import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#from mouseworld.myspace import ContinuousSpace

class Space_surface :
    def __init__(self, unique_id): 
        self.unique_id = unique_id
        #self.space  = space
        #self.agent_list = []
        columns = ['weight', 'loc[x]', 'loc[y]', 'scale']
        #dtype={'weight':float,'loc':(float, float), 'scale':int}
        self.agent_list = pd.DataFrame(data=np.zeros((0,len(columns))), columns=columns) 
        
    def add_agent(self, agent) :
        unique_id = agent.unique_id
        scale = agent.odor_std
        weight = agent.odor_strength
        loc = agent.pos
        self.agent_list.loc[unique_id] = [weight, loc[0], loc[1], scale]
        #self.agent_list.append([unique_id, weight, loc, scale])
        
    def update_surface(self) :
        data = self.agent_list.as_matrix()
        #data = self.agent_list.drop(['unique_id'], axis=1)
        #data = data.values
        self.surface = [[0, None]]*len(data)
        for i in range(len(data)):
            self.surface[i][0] = data[i,0]
            pos = [data[i,1], data[i,2]]
            self.surface[i][1] = multivariate_normal(pos, data[i,3])
        #print(data)
        
    def update_agent_location(self, agent_unique_id, agent_pos) :
#         temp = self.agent_list
#         temp.ix[agent_unique_id]['loc'] = agent_pos
#         ind = temp[:,0] == agent.unique_id
#         temp[ind][2] = agent_pos
        self.agent_list.set_value(agent_unique_id,'loc[x]',agent_pos[0])
        self.agent_list.set_value(agent_unique_id,'loc[y]',agent_pos[1])

        
    def remove_agent(self, agent) :
        temp = self.agent_list
        #self.agent_list = temp[temp[:,0] != agent.unique_id]
        temp.drop(temp.index[agent.unique_id])
        self.agent_list = temp
         
    def get_value(self, pos) :
        value = 0
        for i in self.surface :
            value += i[0] * i[1].pdf(pos)
        return value
    