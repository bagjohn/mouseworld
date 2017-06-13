
# coding: utf-8

# In[5]:

import pandas as pd
import numpy as np

a=np.ndarray([2,5],dtype = float)
df_old=pd.DataFrame([[a[0],'m'], [a[1],'f']], index = [0,1], columns = ('Genome','Sex'))
df_old.to_csv('df.csv', sep='\t')
df_new = pd.read_csv('df.csv', sep='\t')

print(df_old['Genome'][0])
print(df_new['Genome'][0])
print(type(df_old['Genome'][0]))
print(type(df_new['Genome'][0]))


# In[117]:

import pandas as pd
import numpy as np
import ast
df = pd.read_csv('results/simulation_139/genome_data.csv', sep='\t')

#print(df)
# df = df['Genome', 'motor_NN_on', 'learning_on','mousebrain_sim'].values
# df = df['Genome'].values
c=df['Genome']
# d=c.to_array()
# print(a.dtypes)
# b= a.values
# print(type(b))
# print(b)
# d=a.as_matrix()
# print(type(d))
# print(d)
# a[0].as_matrix
# b=pd.to_numeric(a)
# df['Genome'] = df['Genome'].astype([float]) 
# a=df['Genome']
b=c.values.tolist()
d=b[1].strip()
print(d)
e=np.fromstring(d, dtype=float, sep=' ')
print(e)
print(type(e))


# In[31]:

a = []
genome = 3
temp_list = [([genome, True, True, None]),
            ([genome, True, False, None]),([genome, False, False, None])]
a.append(tuple([genome, True, True, None]))
a.append(tuple([genome, True, False, None]))
a.append(tuple([genome, False, False, None]))
a


# In[13]:

# import os
# import mouseworld
import ipyparallel
import itertools
clients = ipyparallel.Client()
dview = clients.direct_view()

with dview.sync_imports():
    from mouseworld.multi_mousetest import Multi_Mousetest
    import time


# In[6]:

mouse_list = [5,4,3,5,6]

file = open('results/stats2.txt','w') 
for i in range(len(mouse_list)) :
    if i%3 == 0:
        file.write(str(mouse_list[i]) + '\n')
#     file.write(str(all_first_actions[i]))
file.close() 


# In[2]:

#%%writefile multi_mousetest_run_parallel.py

# at terminal : ipcluster start -n 4

import ipyparallel
import itertools
import mouseworld


clients = ipyparallel.Client()
dview = clients.direct_view()

# mouse_list_file = sys.argv[1]
mouse_list= []
# max_speed = [0, 0.2, 0.4, 0.6, 0.8, 1]
# antenna_length = [0, 0.2, 0.4, 0.6, 0.8, 1]
# antenna_angle = [0, 0.2, 0.4, 0.6, 0.8, 1]
max_speed = [0.2]
antenna_length = [0.4]
antenna_angle = [0.2]
params = (max_speed, antenna_length, antenna_angle)
param_combs = list(itertools.product(*params))
for params in param_combs :
    genome = [params[0], 0, 0, params[1], params[2]]
    mouse_list.append(tuple([genome, True, True, None]))
    mouse_list.append(tuple([genome, True, False, None]))
    mouse_list.append(tuple([genome, False, False, None]))
# mouse_list=[([[0.2,0.4,0.6,0.4,0.2], True, True, None]),
#             ([[0.2,0.4,0.6,0.4,0.2], True, False, None]),([[0.2,0.4,0.6,0.4,0.2], False, False, None])]
# pos =[(1,1), (1,2), (1,4), (1,5), (2,2), (2,3), (2,4), (2,5), (3,3), (3,4), (3,5), (4,4), (4,5), (5,5)]
# # pos =[(1,1), (2,2), (3,3)]
# header = [0,1,2,3,4,5,6,7]
# # header = [0]

# params = (pos, header, mouse_list)
# param_combs = list(itertools.product(*params))

with dview.sync_imports():
    from mouseworld.multi_mousetest import Multi_Mousetest
    import time
dview.push({"Multi_Mousetest": Multi_Mousetest})

def make_model(mouse_data):
    #header = params[0]
#     pos = params[0]
#     header = params[1]
#     mouse_list = params[2]
    model = Multi_Mousetest(mouse_data, -10, 1, 0, 100, 100)
    for i in range(10) :
        model.food_schedule.step()
        #model.predator_schedule.step()
        model.diffuse_odor_layers(model.odor_layers)
    #counter = 0
    myrange = 40
    for i in range(myrange) :
        #c=time.time()
        #counter += 1
        model.step()
        #d=time.time()
    model.final_datacollector.collect(model,model.all_mice_schedule)
    final_agent_data = model.final_datacollector.get_agent_vars_dataframe()
    mouse_statistics = final_agent_data[['first_action_duration', 'first_action_termination']]
    mouse_statistics = mouse_statistics.reset_index('Step', drop = True)
    #mouse_statistics = mouse_statistics.reset_index('AgentID', drop = True)
    succesful_trials = mouse_statistics.loc[(mouse_statistics['first_action_termination'] == 'Closure')]
    num_trials = len(mouse_statistics.index)
    num_succesful_trials = len(succesful_trials.index)
    performance = num_succesful_trials / num_trials
    mean_time = succesful_trials['first_action_duration'].mean()
#     sensor_vector = final_agent_data['sensor_vector'][0].values[0]
#     sensor_position = final_agent_data['sensor_position'][0].values[0]
#     motor_vector = final_agent_data['motor_vector'][0].values[0]
#     first_action = final_agent_data['action_history'][0].values[0].loc[0]
#     first_action = mousetest_data
#     first_action = (mousetest_data['Duration'], mousetest_data['Termination'])
    #first_action = final_agent_data['action_history'][0].values[0].loc[0]
    return (performance,  mean_time)
#     return (params, mousetest_data.loc['Mouse_1'])

#     return (first_action, sensor_vector, sensor_position, motor_vector)
 
all_first_actions = dview.map_sync(make_model, mouse_list)
file = open('results/stats.txt','w') 
for i in range(len(mouse_list)) :
    if i%3 == 0 :
        file.write('')
    file.write(str(mouse_list[i]) + '\n')
    file.write(str(all_first_actions[i]) + '\n')
file.close() 


# In[1]:

get_ipython().run_cell_magic('writefile', 'mouseworld/multi_mousetest.py', '\n# Testing ground for multiple mice\n# Use :\n# Test approach or avoid efficiency of mouse relative to same genome mice. \n# If mouse has mousebrain check also whether it learns\n# Data from mose needed :\n# 1. genome\n# 2. motor_NN_on, learning_on\n# 3. mousebrain_sim\n# Arguments :\n# 1. Mouse_list to be tested. For every mouse create 1 mouse of each type\n# ((motor_NN_on, learning_on) = [(False,False), (True,False), (True,True)] \n# with same genome plus a mouse with the given mousebrain if it has one.\n# It includes the fields :\n# 1.a. Genome \n# 1.b. motor_NN_on\n# 1.c. learning_on\n# 1.d. mousebrain_weights  ([None, None, None] where there is no mousebrain)\n# World parameters\n# 2. position relative to food/predator\n# 3. header\n# 4. value of food/predator. If positive check approach, else check avoid\n\nfrom mesa import Agent, Model\n#from mesa.time import RandomActivation\nfrom mesa.datacollection import DataCollector\n\nimport itertools\nimport numpy as np\nimport math\nimport pandas as pd\nimport random\nfrom scipy.stats import norm\n\nfrom mouseworld.mytime import *\nfrom mouseworld.myspace import *\nfrom mouseworld.mouseworld import Mouseworld\nfrom mouseworld.mouse import Mouse\nfrom mouseworld.food import Food\nfrom mouseworld.predator import Predator\nfrom mouseworld.mydatacollector import MyDataCollector\n\nfrom joblib import Parallel, delayed\nimport multiprocessing\n\nclass Multi_Mousetest(Mouseworld):\n#     def __init__(self, mouse_list, pos, header, food_odor_value, num_food, num_predators, width, height, mousebrain_inheritance = False):\n    def __init__(self, mouse_data, food_odor_value, num_food, num_predators, width, height, mousebrain_inheritance = False):\n    \n        # for parallel processing\n        self.num_cores = multiprocessing.cpu_count()\n        \n        self.genome = mouse_data[0]\n        self.motor_NN_on = mouse_data[1]\n        self.learning_on = mouse_data[2]\n        self.mousebrain_weights = mouse_data[3]\n        \n        # define model variables from args\n#         self.num_genomes = len(mouse_list)\n#         self.num_mice = 4 * self.num_genomes\n        self.num_unborn_mice = 0\n        self.num_genes = 5\n        self.num_food = num_food\n        self.num_predators = num_predators\n        self.mousebrain_inheritance = mousebrain_inheritance\n        # build model continuous space\n        self.space = ContinuousSpace(width, height, True, x_min=0, y_min=0,\n            grid_width=width, grid_height=height)\n        \n        # initialize food parameters\n        self.food_amount_range = (10000,10000)\n        self.food_odor_strength = [1] #[0.7,1]\n        self.food_odor_std = [8]\n        self.nutritional_value = [1]\n        self.food_params = (self.food_odor_strength, self.nutritional_value, self.food_odor_std)\n        self.food_param_combs = list(itertools.product(*self.food_params))\n        self.food_groups_num = len(self.food_param_combs)\n        self.food_groups = [(\'Food_group_%i\'%i) for i in range(self.food_groups_num)]\n        self.food_layers = [Value_layer(\'Food_odor_%i\'%i, width, height, True) for i in range(self.food_groups_num)]\n        self.food_layer_names = [(\'Food_odor_%i\'%i) for i in range(self.food_groups_num)]\n#         for i in range(self.food_groups_num) :\n#             self.food_groups[i] = (\'Food_group_%i\'%i) \n#             self.food_layers[i] = (\'Food_odor_%i\'%i)\n        \n        # initialize predator parameters\n        self.predator_odor_strength = [1] # [0.7,1]\n        self.predator_odor_std = [8]\n        self.damage_level = [1] #[0.3,1]\n        self.hunt_rule = [1]\n        self.hunt_radius = [1] #[0.5,1]\n        self.predator_params = (self.predator_odor_strength, self.predator_odor_std, self.damage_level,\n                                self.hunt_rule, self.hunt_radius)\n        self.predator_param_combs = list(itertools.product(*self.predator_params))\n        self.predator_groups_num = len(self.predator_param_combs)\n        self.predator_groups = [(\'Predator_group_%i\'%i) for i in range(self.predator_groups_num)]\n        self.predator_layers = [Value_layer(\'Predator_odor_%i\'%i, width, height, True) for i in range(self.predator_groups_num)]\n        self.predator_layer_names = [(\'Predator_odor_%i\'%i) for i in range(self.predator_groups_num)]\n#         for i in range(self.predator_groups_num) :\n#             self.predator_groups[i] = (\'Predator_group_%i\'%i)\n#             self.predator_layers[i] = (\'Predator_odor_%i\'%i)\n            \n        # all agents (food & predator)\n        self.groups_num = self.food_groups_num + self.predator_groups_num\n        self.groups = self.food_groups + self.predator_groups\n        self.odor_layers = self.food_layers + self.predator_layers\n        self.odor_layer_names = self.food_layer_names + self.predator_layer_names\n\n        # build schedules\n        self.schedule = RandomActivation(self)\n        self.all_mice_schedule = RandomActivation(self)\n        self.food_schedule = RandomActivation(self)\n        self.predator_schedule = RandomActivation(self)\n        self.mouseworld_date = 0\n        \n        #initialize ids\n        self.initialize_ids([\'Mouse\', \'Food\', \'Predator\'])\n        \n        #initialize sensor_vector\n#         self.sensor_num = 2\n#         temp = [np.zeros(self.sensor_num)] * self.groups_num\n#         self.zero_sensor_vector = pd.Series(temp, index=self.odor_layers)\n        \n        pos =[(1,1), (1,2), (1,4), (1,5), (2,2), (2,3), (2,4), (2,5), (3,3), (3,4), (3,5), (4,4), (4,5), (5,5)]\n#         pos =[(1,1)]\n        header = [0,1,2,3,4,5,6,7]\n        params = (pos, header)\n        param_combs = list(itertools.product(*params))    \n        \n        self.num_mice = len(param_combs)\n        \n        # Create agents (Version : One agent for each point and header)\n        for params in param_combs:\n            header = params[1]\n            x, y = params[0]\n\n            mouse = Mouse(self, None, self.genome, 0, motor_NN_on = self.motor_NN_on, learning_on = self.learning_on,\n                           appraisal_NN_on = False, initial_mousebrain_weights = self.mousebrain_weights)\n\n            mouse.header = header * math.pi / 4\n            mouse.primary_values[self.food_groups[0]] = 100\n            mouse.secondary_values.ix[self.food_groups[0]][self.food_layer_names[0]]= food_odor_value\n            self.schedule.add(mouse)\n            self.all_mice_schedule.add(mouse)\n            self.space.place_agent(mouse, (50 + x, 50 + y))\n            #self.place_agent_randomly(mouse)\n            #print(mouse.unique_id)\n            #print(mouse.genome)\n        \n        # Create agents (Version : 4 agents for each genome)\n#         for i in range(self.num_genomes):\n#             #temp_genome = self.initialization_genome[i]\n#             temp_genome = mouse_list[i][0]\n#             temp_mousebrain = mouse_list[i][3]\n#             mouse0 = Mouse(self, None, temp_genome, 0, motor_NN_on = mouse_list[i][1], learning_on = mouse_list[i][2],\n#                            appraisal_NN_on = False, initial_mousebrain_weights = temp_mousebrain)\n#             mouse1 = Mouse(self, None, temp_genome, 0, motor_NN_on = False, learning_on = False, appraisal_NN_on = False)\n#             mouse2 = Mouse(self, None, temp_genome, 0, motor_NN_on = True, learning_on = False, appraisal_NN_on = False)\n#             mouse3 = Mouse(self, None, temp_genome, 0, motor_NN_on = True, learning_on = True, appraisal_NN_on = False)\n#             for mouse in [mouse0, mouse1, mouse2, mouse3] :\n#                 mouse.header = header * math.pi / 4\n#                 mouse.primary_values[self.food_groups[0]] = food_odor_value * 10\n#                 mouse.secondary_values.ix[self.food_groups[0]][self.food_layer_names[0]]= food_odor_value\n#                 self.schedule.add(mouse)\n#                 self.all_mice_schedule.add(mouse)\n#                 self.space.place_agent(mouse, (50 + x, 50 + y))\n#             #self.place_agent_randomly(mouse)\n#             #print(mouse.unique_id)\n#             #print(mouse.genome)\n            \n        \n        for i in range(self.num_food):\n            j = i%(self.food_groups_num)\n            food = Food(self.food_groups[j], j, self.food_layers[j], self.food_amount_range, self)\n            self.food_schedule.add(food)\n            self.space.place_agent(food, (50, 50))\n            #self.place_agent_randomly(food)\n            #self.food_layers[j].add_agent(food)\n            \n        for i in range(self.num_predators):\n            j = i%(self.predator_groups_num)\n            predator = Predator(self.predator_groups[j], j, self.predator_layers[j], self)\n            self.predator_schedule.add(predator)\n            self.place_agent_randomly(predator)\n            #self.predator_layers[j].add_agent(predator)\n                \n        self.initial_datacollector = MyDataCollector(\n            model_reporters={"Initial genome distribution": lambda a: a.initialization_genome})\n        \n        self.model_datacollector = MyDataCollector(\n            model_reporters={"Alive_mice": lambda a: a.num_mice, \n                             "Unborn_mice": lambda a: a.num_unborn_mice})\n        \n        self.datacollector = MyDataCollector(\n            model_reporters={"Alive_mice": lambda a: a.num_mice, \n                             "Unborn_mice": lambda a: a.num_unborn_mice,\n                             "Food_groups_num": lambda a: a.food_groups_num},\n            agent_reporters={"Header": lambda a: a.header,\n                             "Age": lambda a: a.age, \n                             "Energy": lambda a: a.energy,\n                             "max_speed": lambda a: a.max_speed,\n                             "incubation_period": lambda a: a.incubation_period,\n                             "pos": lambda a: a.pos,\n                             "Genome": lambda a: a.genome})\n        \n        self.mousebrain_datacollector = MyDataCollector(\n            agent_reporters={"odor": lambda a: a.mousebrain_sim.data[a.mousebrain.p_odor],\n                             "state": lambda a: a.mousebrain_sim.data[a.mousebrain.p_state], \n                             "approach": lambda a: a.mousebrain_sim.data[a.mousebrain.p_approach],\n                             "avoid": lambda a: a.mousebrain_sim.data[a.mousebrain.p_avoid],\n                             "search": lambda a: a.mousebrain_sim.data[a.mousebrain.p_search],\n                             "change": lambda a: a.mousebrain_sim.data[a.mousebrain.p_change],\n                             "errors0": lambda a: a.mousebrain_sim.data[a.mousebrain.p_errors0],\n                             "errors1": lambda a: a.mousebrain_sim.data[a.mousebrain.p_errors1],\n                             "errors2": lambda a: a.mousebrain_sim.data[a.mousebrain.p_errors2]})\n\n        self.test_datacollector = MyDataCollector(\n            agent_reporters={"sensor_vector": lambda a: a.sensor_vector})       \n#         self.test_datacollector = MyDataCollector(\n#             agent_reporters={"sensor_vector": lambda a: a.sensor_vector,\n#                              "Action": lambda a: a.current_action[\'Verb\'],\n#                              "Noun_group": lambda a: a.current_action[\'Noun_group\'],\n#                              "food_gained_energy": lambda a: a.food_gained_energy,\n#                              "food_lost_energy": lambda a: a.food_lost_energy,\n#                              "metabolism_buffer": lambda a: a.metabolism_buffer,\n#                             "energy_to_predators": lambda a: a.energy_to_predators,\n#                             "total_distance": lambda a: a.total_distance})\n        \n#         self.final_datacollector = MyDataCollector(\n#             agent_reporters={"total_distance": lambda a: a.total_distance,\n#                              "Energy": lambda a: a.energy,\n#                              "food_lost_energy": lambda a: a.food_lost_energy,\n#                             "food_gained_energy": lambda a: a.food_gained_energy})\n        \n        self.final_datacollector = MyDataCollector(\n            model_reporters={"Alive_mice": lambda a: a.schedule.get_agent_count(), \n                             "All_mice": lambda a: a.all_mice_schedule.get_agent_count(), \n                             "Unborn_mice": lambda a: a.num_unborn_mice,\n                            "odor_layer_names": lambda a: a.odor_layer_names},\n            agent_reporters={"age": lambda a: a.age,\n                             "energy": lambda a: a.energy,\n                             "generation": lambda a: a.generation,\n                             "num_offspring": lambda a: a.num_offspring,\n                             "hunger_status": lambda a: a.hunger_status,\n                             "action_history": lambda a: a.action_history,\n                             "first_action_duration": lambda a: a.action_history[\'Duration\'][0],\n                             "first_action_termination": lambda a: a.action_history[\'Termination\'][0],\n                            "possible_actions": lambda a: a.possible_actions,\n                             "primary_values": lambda a: a.primary_values,\n                             "secondary_values": lambda a: a.secondary_values,\n                            "sensor_vector": lambda a: a.sensor_vector,\n                             "motor_vector": lambda a: a.motor_vector,\n                            "sensor_position": lambda a: a.sensor_position})')


# In[ ]:




# In[ ]:



