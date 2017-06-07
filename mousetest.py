
# coding: utf-8

# In[4]:

import itertools

# header = [0,1,2,3,4,5,6,7]
# x = [-5,-4,-3,-2,-1,0,1,2,3,4,5]
# y = [-5,-4,-3,-2,-1,0,1,2,3,4,5]
# max_speed = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
# antenna_length = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
# antenna_angle = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
header = [0,1,2,3,4,5,6,7]
x = [0,1,2,3,4,5]
y = [0,1,2,3,4,5]
max_speed = [0,0.2,0.4,0.6,0.8,1]
antenna_length = [0,0.2,0.4,0.6,0.8,1]
antenna_angle = [0,0.2,0.4,0.6,0.8,1]

params = (header, x, y, max_speed, antenna_length, antenna_angle)
param_combs = list(itertools.product(*params))
len(param_combs)
#param_combs[111111]


# In[14]:

#%%writefile mousetest_run_parallel.py

# at terminal : ipcluster start -n 4

import ipyparallel
import itertools

clients = ipyparallel.Client()
dview = clients.direct_view()


# pos =[(0,0), (0,1), (1,0), (1,1)]
pos =[(1,1), (2,2), (3,3)]
header = [0,1,2,3,4,5,6,7]
# x = [0,1]
# y = [0,1]
# x = [0,1,2,3,4,5]
# y = [0,1,2,3,4,5]
max_speed = [0,0.4,1]
# antenna_length = [0,0.2,0.4,0.6,0.8,1]
# antenna_angle = [0,0.2,0.4,0.6,0.8,1]
antenna_angle = [0.4]

# max_speed = [0]
antenna_length = [0.4]
# antenna_angle = [0]

params = (max_speed, antenna_length, antenna_angle, pos, header)
param_combs = list(itertools.product(*params))

with dview.sync_imports():
    from mouseworld.mousetest import Mousetest
    import time
dview.push({"Mousetest": Mousetest})

def make_model(params):
    #header = params[0]
    pos = params[3]
    header = params[4]
    max_speed = params[0]
    antenna_length = params[1]
    antenna_angle = params[2]
    model = Mousetest([1, 0, 0], [max_speed, 0.5, 0.5, antenna_length, antenna_angle], pos, header, 10, 1, 0, 100, 100)
    for i in range(10) :
        model.food_schedule.step()
        #model.predator_schedule.step()
        model.diffuse_odor_layers(model.odor_layers)
    #counter = 0
    myrange = 20
    for i in range(myrange) :
        #c=time.time()
        #counter += 1
        model.step()
        #d=time.time()
    model.final_datacollector.collect(model,model.all_mice_schedule)
    final_agent_data = model.final_datacollector.get_agent_vars_dataframe()
    sensor_vector = final_agent_data['sensor_vector'][0].values[0]
    sensor_position = final_agent_data['sensor_position'][0].values[0]
    motor_vector = final_agent_data['motor_vector'][0].values[0]
    first_action = final_agent_data['action_history'][0].values[0].loc[0]
    first_action = (first_action['Duration'], first_action['Termination'])
    #first_action = final_agent_data['action_history'][0].values[0].loc[0]
    return (params, first_action)
#     return (first_action, sensor_vector, sensor_position, motor_vector)
 
all_first_actions = dview.map_sync(make_model, param_combs)
for i in range(len(param_combs)) :
    print(all_first_actions[i])


# In[6]:

get_ipython().run_cell_magic('writefile', 'mousetest_run.py', "\nfrom mouseworld import mousetest\nimport time\n\nx = 1\ny = 1\nheader = 1 # 0-7\nantenna_length = 0.5\nantenna_angle = 0.5\n# Build the model\nmodel = mousetest.Mousetest([0, 0, 1], [0.5, 0.5, 0.5, antenna_length, antenna_angle], (x,y), header, 10, 1, 0, 100, 100)\n\n# Gather initial randomly distributed data\n# model.initial_datacollector.collect(model,model.schedule)\n# initial_model_data = model.initial_datacollector.get_model_vars_dataframe()\n# initial_model_data.to_csv('results/initial_model_data.csv', sep='\\t')\n\n# Prepare environment by stepping food and predators and diffusing odors\na=time.time()\nfor i in range(10) :\n    model.food_schedule.step()\n    model.predator_schedule.step()\n    model.diffuse_odor_layers(model.odor_layers)\nb=time.time()\nprint(b-a)\n\n#Run for discrete number of timesteps\ncounter = 0\nmyrange = 1\nfor i in range(myrange) :\n    c=time.time()\n    counter += 1\n    model.step()\n    d=time.time()\n    #print('sim step : %i in %f'%(counter, d-c))\n\n# Run until all mice perish\n# while model.num_mice > 0 :\n#     print('sim step : %i'%counter)\n#     model.step()\n    \nmodel.final_datacollector.collect(model,model.all_mice_schedule)\nfinal_model_data = model.final_datacollector.get_model_vars_dataframe()\n#final_model_data.to_csv('results/final_model_data.csv', sep='\\t')\nfinal_agent_data = model.final_datacollector.get_agent_vars_dataframe()\nfirst_action = final_agent_data['action_history'][0].values[0].loc[0]\n# model.food_datacollector.collect(model,model.food_schedule)\n# food_data = model.food_datacollector.get_agent_vars_dataframe()\n# print(food_data)\n\n#print(final_model_data)\n\nfor i in range(len(final_agent_data)) :\n#     print('Name : %s'%final_agent_data.index[i][1])\n#     print('Age : %i'%final_agent_data['age'][0].values[i])\n#     print('Generation : %i'%final_agent_data['generation'][0].values[i])\n#     print('Offspring : %i'%final_agent_data['num_offspring'][0].values[i])\n#     print('Energy : %f'%final_agent_data['energy'][0].values[i])\n#     print('Hunger_status : %f'%final_agent_data['hunger_status'][0].values[i])\n#     print(final_agent_data['action_history'][0].values[i])\n    print(final_agent_data['action_history'][0].values[i].loc[0])\n#     print(final_agent_data['possible_actions'][0].values[i])\n#     print(final_agent_data['primary_values'][0].values[i])\n#     print(final_agent_data['primary_values'])\n#     print(final_agent_data['secondary_values'][0].values[i])\n    print(final_agent_data['sensor_vector'][0].values[i])\n#     print(final_agent_data['sensor_position'][0].values[i])")


# In[9]:

get_ipython().run_cell_magic('writefile', 'mouseworld/mousetest.py', '\n# This is a testing ground for a single mouse. Initiate with num_mice e.g. [0,0,1].\n# A food is placed at (50,50) and we define the initial placement of the mouse by the pos argument (it will be (50+x,50+y)).\n# We define the genome the mouse will have and its initial header\n# We define the secondary value for the food as positive so that the mouse will immediately "approach"\n# or as negative so that the mouse will immediately "avoid"\n\nfrom mesa import Agent, Model\n#from mesa.time import RandomActivation\nfrom mesa.datacollection import DataCollector\n\nimport itertools\nimport numpy as np\nimport math\nimport pandas as pd\nimport random\nfrom scipy.stats import norm\n\nfrom mouseworld.mytime import *\nfrom mouseworld.myspace import *\nfrom mouseworld.mouseworld import Mouseworld\nfrom mouseworld.mouse import Mouse\nfrom mouseworld.food import Food\nfrom mouseworld.predator import Predator\nfrom mouseworld.mydatacollector import MyDataCollector\n\nfrom joblib import Parallel, delayed\nimport multiprocessing\n\nclass Mousetest(Mouseworld):\n    def __init__(self, num_mice, genome, pos, header, food_odor_value, num_food, num_predators, width, height):\n        \n        # for parallel processing\n        self.num_cores = multiprocessing.cpu_count()\n        \n        # define model variables from args\n        self.num_mice = sum(num_mice)\n        self.num_unborn_mice = 0\n        self.num_genes = 5\n        self.num_food = num_food\n        self.num_predators = num_predators\n        # build model continuous space\n        self.space = ContinuousSpace(width, height, True, x_min=0, y_min=0,\n            grid_width=width, grid_height=height)\n        \n        # initialize genome\n        self.initialization_genome = self.initialize_genome()\n        \n        # initialize food parameters\n        self.food_amount_range = (40,200)\n        self.food_odor_strength = [1] #[0.7,1]\n        self.food_odor_std = [8]\n        self.nutritional_value = [1]\n        self.food_params = (self.food_odor_strength, self.nutritional_value, self.food_odor_std)\n        self.food_param_combs = list(itertools.product(*self.food_params))\n        self.food_groups_num = len(self.food_param_combs)\n        self.food_groups = [(\'Food_group_%i\'%i) for i in range(self.food_groups_num)]\n        self.food_layers = [Value_layer(\'Food_odor_%i\'%i, width, height, True) for i in range(self.food_groups_num)]\n        self.food_layer_names = [(\'Food_odor_%i\'%i) for i in range(self.food_groups_num)]\n#         for i in range(self.food_groups_num) :\n#             self.food_groups[i] = (\'Food_group_%i\'%i) \n#             self.food_layers[i] = (\'Food_odor_%i\'%i)\n        \n        # initialize predator parameters\n        self.predator_odor_strength = [1] # [0.7,1]\n        self.predator_odor_std = [8]\n        self.damage_level = [1] #[0.3,1]\n        self.hunt_rule = [1]\n        self.hunt_radius = [1] #[0.5,1]\n        self.predator_params = (self.predator_odor_strength, self.predator_odor_std, self.damage_level,\n                                self.hunt_rule, self.hunt_radius)\n        self.predator_param_combs = list(itertools.product(*self.predator_params))\n        self.predator_groups_num = len(self.predator_param_combs)\n        self.predator_groups = [(\'Predator_group_%i\'%i) for i in range(self.predator_groups_num)]\n        self.predator_layers = [Value_layer(\'Predator_odor_%i\'%i, width, height, True) for i in range(self.predator_groups_num)]\n        self.predator_layer_names = [(\'Predator_odor_%i\'%i) for i in range(self.predator_groups_num)]\n#         for i in range(self.predator_groups_num) :\n#             self.predator_groups[i] = (\'Predator_group_%i\'%i)\n#             self.predator_layers[i] = (\'Predator_odor_%i\'%i)\n            \n        # all agents (food & predator)\n        self.groups_num = self.food_groups_num + self.predator_groups_num\n        self.groups = self.food_groups + self.predator_groups\n        self.odor_layers = self.food_layers + self.predator_layers\n        self.odor_layer_names = self.food_layer_names + self.predator_layer_names\n\n        # build schedules\n        self.schedule = RandomActivation(self)\n        self.all_mice_schedule = RandomActivation(self)\n        self.food_schedule = RandomActivation(self)\n        self.predator_schedule = RandomActivation(self)\n        \n        #initialize ids\n        self.initialize_ids([\'Mouse\', \'Food\', \'Predator\'])\n        \n        #initialize sensor_vector\n#         self.sensor_num = 2\n#         temp = [np.zeros(self.sensor_num)] * self.groups_num\n#         self.zero_sensor_vector = pd.Series(temp, index=self.odor_layers)\n        \n        x, y = pos\n            \n        # Create agents\n        for i in range(self.num_mice):\n            #temp_genome = self.initialization_genome[i]\n            temp_genome = genome\n            if i < num_mice[0] :\n                mouse = Mouse(self, temp_genome, 0, motor_NN_on = False, learning_on = False, appraisal_NN_on = False)\n            elif i < num_mice[1]:\n                mouse = Mouse(self, temp_genome, 0, motor_NN_on = True, learning_on = False, appraisal_NN_on = False)\n            else :\n                mouse = Mouse(self, temp_genome, 0, motor_NN_on = True, learning_on = True, appraisal_NN_on = False)\n            mouse.header = header * math.pi / 4\n            mouse.primary_values[self.food_groups[0]] = food_odor_value * 10\n            mouse.secondary_values.ix[self.food_groups[0]][self.food_layer_names[0]]= food_odor_value\n            self.schedule.add(mouse)\n            self.all_mice_schedule.add(mouse)\n            self.space.place_agent(mouse, (50 + x, 50 + y))\n            #self.place_agent_randomly(mouse)\n            #print(mouse.unique_id)\n            #print(mouse.genome)\n            \n        \n        for i in range(self.num_food):\n            j = i%(self.food_groups_num)\n            food = Food(self.food_groups[j], j, self.food_layers[j], self.food_amount_range, self)\n            self.food_schedule.add(food)\n            self.space.place_agent(food, (50, 50))\n            #self.place_agent_randomly(food)\n            #self.food_layers[j].add_agent(food)\n            \n        for i in range(self.num_predators):\n            j = i%(self.predator_groups_num)\n            predator = Predator(self.predator_groups[j], j, self.predator_layers[j], self)\n            self.predator_schedule.add(predator)\n            self.place_agent_randomly(predator)\n            #self.predator_layers[j].add_agent(predator)\n                \n        self.initial_datacollector = MyDataCollector(\n            model_reporters={"Initial genome distribution": lambda a: a.initialization_genome})\n        \n        self.datacollector = MyDataCollector(\n            model_reporters={"Alive_mice": lambda a: a.num_mice, \n                             "Unborn_mice": lambda a: a.num_unborn_mice,\n                             "Food_groups_num": lambda a: a.food_groups_num},\n            agent_reporters={"Header": lambda a: a.header,\n                             "Age": lambda a: a.age, \n                             "Energy": lambda a: a.energy,\n                             "max_speed": lambda a: a.max_speed,\n                             "incubation_period": lambda a: a.incubation_period,\n                             "pos": lambda a: a.pos,\n                             "Genome": lambda a: a.genome})\n        \n        self.mousebrain_datacollector = MyDataCollector(\n            agent_reporters={"odor": lambda a: a.mousebrain_sim.data[a.mousebrain.p_odor],\n                             "state": lambda a: a.mousebrain_sim.data[a.mousebrain.p_state], \n                             "approach": lambda a: a.mousebrain_sim.data[a.mousebrain.p_approach],\n                             "avoid": lambda a: a.mousebrain_sim.data[a.mousebrain.p_avoid],\n                             "search": lambda a: a.mousebrain_sim.data[a.mousebrain.p_search],\n                             "change": lambda a: a.mousebrain_sim.data[a.mousebrain.p_change],\n                             "errors0": lambda a: a.mousebrain_sim.data[a.mousebrain.p_errors0],\n                             "errors1": lambda a: a.mousebrain_sim.data[a.mousebrain.p_errors1],\n                             "errors2": lambda a: a.mousebrain_sim.data[a.mousebrain.p_errors2]})\n\n        self.test_datacollector = MyDataCollector(\n            agent_reporters={"sensor_vector": lambda a: a.sensor_vector})       \n#         self.test_datacollector = MyDataCollector(\n#             agent_reporters={"sensor_vector": lambda a: a.sensor_vector,\n#                              "Action": lambda a: a.current_action[\'Verb\'],\n#                              "Noun_group": lambda a: a.current_action[\'Noun_group\'],\n#                              "food_gained_energy": lambda a: a.food_gained_energy,\n#                              "food_lost_energy": lambda a: a.food_lost_energy,\n#                              "metabolism_buffer": lambda a: a.metabolism_buffer,\n#                             "energy_to_predators": lambda a: a.energy_to_predators,\n#                             "total_distance": lambda a: a.total_distance})\n        \n#         self.final_datacollector = MyDataCollector(\n#             agent_reporters={"total_distance": lambda a: a.total_distance,\n#                              "Energy": lambda a: a.energy,\n#                              "food_lost_energy": lambda a: a.food_lost_energy,\n#                             "food_gained_energy": lambda a: a.food_gained_energy})\n        \n        self.final_datacollector = MyDataCollector(\n            model_reporters={"Alive_mice": lambda a: a.schedule.get_agent_count(), \n                             "All_mice": lambda a: a.all_mice_schedule.get_agent_count(), \n                             "Unborn_mice": lambda a: a.num_unborn_mice,\n                            "odor_layer_names": lambda a: a.odor_layer_names},\n            agent_reporters={"age": lambda a: a.age,\n                             "energy": lambda a: a.energy,\n                             "generation": lambda a: a.generation,\n                             "num_offspring": lambda a: a.num_offspring,\n                             "hunger_status": lambda a: a.hunger_status,\n                             "action_history": lambda a: a.action_history,\n                            "possible_actions": lambda a: a.possible_actions,\n                             "primary_values": lambda a: a.primary_values,\n                             "secondary_values": lambda a: a.secondary_values,\n                            "sensor_vector": lambda a: a.sensor_vector,\n                             "motor_vector": lambda a: a.motor_vector,\n                            "sensor_position": lambda a: a.sensor_position})\n        \n        self.predator_datacollector = MyDataCollector(\n            agent_reporters={"Victims_num": lambda a: a.victims_num,\n                             "odor_strength": lambda a: a.odor_strength,\n                             "hunt_rule": lambda a: a.hunt_rule,\n                             "odor_std": lambda a: a.odor_std,\n                             "Damage_level": lambda a: a.damage_level})\n        \n        self.food_datacollector = MyDataCollector(\n            agent_reporters={"Pos": lambda a: a.pos})')


# In[ ]:



