
# This is a testing ground for a single mouse. Initiate with num_mice e.g. [0,0,1].
# A food is placed at (50,50) and we define the initial placement of the mouse by the pos argument (it will be (50+x,50+y)).
# We define the genome the mouse will have and its initial header
# We define the secondary value for the food as positive so that the mouse will immediately "approach"
# or as negative so that the mouse will immediately "avoid"

from mesa import Agent, Model
#from mesa.time import RandomActivation
from mesa.datacollection import DataCollector

import itertools
import numpy as np
import math
import pandas as pd
import random
from scipy.stats import norm

from mouseworld.mytime import *
from mouseworld.myspace import *
from mouseworld.mouseworld import Mouseworld
from mouseworld.mouse import Mouse
from mouseworld.food import Food
from mouseworld.predator import Predator
from mouseworld.mydatacollector import MyDataCollector

from joblib import Parallel, delayed
import multiprocessing

class Mousetest(Mouseworld):
    def __init__(self, num_mice, genome, pos, header, food_odor_value, num_food, num_predators, width, height):
        
        # for parallel processing
        self.num_cores = multiprocessing.cpu_count()
        
        # define model variables from args
        self.num_mice = sum(num_mice)
        self.num_unborn_mice = 0
        self.num_genes = 5
        self.num_food = num_food
        self.num_predators = num_predators
        # build model continuous space
        self.space = ContinuousSpace(width, height, True, x_min=0, y_min=0,
            grid_width=width, grid_height=height)
        
        # initialize genome
        self.initialization_genome = self.initialize_genome()
        
        # initialize food parameters
        self.food_amount_range = (40,200)
        self.food_odor_strength = [1] #[0.7,1]
        self.food_odor_std = [8]
        self.nutritional_value = [1]
        self.food_params = (self.food_odor_strength, self.nutritional_value, self.food_odor_std)
        self.food_param_combs = list(itertools.product(*self.food_params))
        self.food_groups_num = len(self.food_param_combs)
        self.food_groups = [('Food_group_%i'%i) for i in range(self.food_groups_num)]
        self.food_layers = [Value_layer('Food_odor_%i'%i, width, height, True) for i in range(self.food_groups_num)]
        self.food_layer_names = [('Food_odor_%i'%i) for i in range(self.food_groups_num)]
#         for i in range(self.food_groups_num) :
#             self.food_groups[i] = ('Food_group_%i'%i) 
#             self.food_layers[i] = ('Food_odor_%i'%i)
        
        # initialize predator parameters
        self.predator_odor_strength = [1] # [0.7,1]
        self.predator_odor_std = [8]
        self.damage_level = [1] #[0.3,1]
        self.hunt_rule = [1]
        self.hunt_radius = [1] #[0.5,1]
        self.predator_params = (self.predator_odor_strength, self.predator_odor_std, self.damage_level,
                                self.hunt_rule, self.hunt_radius)
        self.predator_param_combs = list(itertools.product(*self.predator_params))
        self.predator_groups_num = len(self.predator_param_combs)
        self.predator_groups = [('Predator_group_%i'%i) for i in range(self.predator_groups_num)]
        self.predator_layers = [Value_layer('Predator_odor_%i'%i, width, height, True) for i in range(self.predator_groups_num)]
        self.predator_layer_names = [('Predator_odor_%i'%i) for i in range(self.predator_groups_num)]
#         for i in range(self.predator_groups_num) :
#             self.predator_groups[i] = ('Predator_group_%i'%i)
#             self.predator_layers[i] = ('Predator_odor_%i'%i)
            
        # all agents (food & predator)
        self.groups_num = self.food_groups_num + self.predator_groups_num
        self.groups = self.food_groups + self.predator_groups
        self.odor_layers = self.food_layers + self.predator_layers
        self.odor_layer_names = self.food_layer_names + self.predator_layer_names

        # build schedules
        self.schedule = RandomActivation(self)
        self.all_mice_schedule = RandomActivation(self)
        self.food_schedule = RandomActivation(self)
        self.predator_schedule = RandomActivation(self)
        
        #initialize ids
        self.initialize_ids(['Mouse', 'Food', 'Predator'])
        
        #initialize sensor_vector
#         self.sensor_num = 2
#         temp = [np.zeros(self.sensor_num)] * self.groups_num
#         self.zero_sensor_vector = pd.Series(temp, index=self.odor_layers)
        
        x, y = pos
            
        # Create agents
        for i in range(self.num_mice):
            #temp_genome = self.initialization_genome[i]
            temp_genome = genome
            if i < num_mice[0] :
                mouse = Mouse(self, temp_genome, 0, motor_NN_on = False, learning_on = False, appraisal_NN_on = False)
            elif i < num_mice[1]:
                mouse = Mouse(self, temp_genome, 0, motor_NN_on = True, learning_on = False, appraisal_NN_on = False)
            else :
                mouse = Mouse(self, temp_genome, 0, motor_NN_on = True, learning_on = True, appraisal_NN_on = False)
            mouse.header = header * math.pi / 4
            mouse.primary_values[self.food_groups[0]] = food_odor_value * 10
            mouse.secondary_values.ix[self.food_groups[0]][self.food_layer_names[0]]= food_odor_value
            self.schedule.add(mouse)
            self.all_mice_schedule.add(mouse)
            self.space.place_agent(mouse, (50 + x, 50 + y))
            #self.place_agent_randomly(mouse)
            #print(mouse.unique_id)
            #print(mouse.genome)
            
        
        for i in range(self.num_food):
            j = i%(self.food_groups_num)
            food = Food(self.food_groups[j], j, self.food_layers[j], self.food_amount_range, self)
            self.food_schedule.add(food)
            self.space.place_agent(food, (50, 50))
            #self.place_agent_randomly(food)
            #self.food_layers[j].add_agent(food)
            
        for i in range(self.num_predators):
            j = i%(self.predator_groups_num)
            predator = Predator(self.predator_groups[j], j, self.predator_layers[j], self)
            self.predator_schedule.add(predator)
            self.place_agent_randomly(predator)
            #self.predator_layers[j].add_agent(predator)
                
        self.initial_datacollector = MyDataCollector(
            model_reporters={"Initial genome distribution": lambda a: a.initialization_genome})
        
        self.datacollector = MyDataCollector(
            model_reporters={"Alive_mice": lambda a: a.num_mice, 
                             "Unborn_mice": lambda a: a.num_unborn_mice,
                             "Food_groups_num": lambda a: a.food_groups_num},
            agent_reporters={"Header": lambda a: a.header,
                             "Age": lambda a: a.age, 
                             "Energy": lambda a: a.energy,
                             "max_speed": lambda a: a.max_speed,
                             "incubation_period": lambda a: a.incubation_period,
                             "pos": lambda a: a.pos,
                             "Genome": lambda a: a.genome})
        
        self.mousebrain_datacollector = MyDataCollector(
            agent_reporters={"odor": lambda a: a.mousebrain_sim.data[a.mousebrain.p_odor],
                             "state": lambda a: a.mousebrain_sim.data[a.mousebrain.p_state], 
                             "approach": lambda a: a.mousebrain_sim.data[a.mousebrain.p_approach],
                             "avoid": lambda a: a.mousebrain_sim.data[a.mousebrain.p_avoid],
                             "search": lambda a: a.mousebrain_sim.data[a.mousebrain.p_search],
                             "change": lambda a: a.mousebrain_sim.data[a.mousebrain.p_change],
                             "errors0": lambda a: a.mousebrain_sim.data[a.mousebrain.p_errors0],
                             "errors1": lambda a: a.mousebrain_sim.data[a.mousebrain.p_errors1],
                             "errors2": lambda a: a.mousebrain_sim.data[a.mousebrain.p_errors2]})

        self.test_datacollector = MyDataCollector(
            agent_reporters={"sensor_vector": lambda a: a.sensor_vector})       
#         self.test_datacollector = MyDataCollector(
#             agent_reporters={"sensor_vector": lambda a: a.sensor_vector,
#                              "Action": lambda a: a.current_action['Verb'],
#                              "Noun_group": lambda a: a.current_action['Noun_group'],
#                              "food_gained_energy": lambda a: a.food_gained_energy,
#                              "food_lost_energy": lambda a: a.food_lost_energy,
#                              "metabolism_buffer": lambda a: a.metabolism_buffer,
#                             "energy_to_predators": lambda a: a.energy_to_predators,
#                             "total_distance": lambda a: a.total_distance})
        
#         self.final_datacollector = MyDataCollector(
#             agent_reporters={"total_distance": lambda a: a.total_distance,
#                              "Energy": lambda a: a.energy,
#                              "food_lost_energy": lambda a: a.food_lost_energy,
#                             "food_gained_energy": lambda a: a.food_gained_energy})
        
        self.final_datacollector = MyDataCollector(
            model_reporters={"Alive_mice": lambda a: a.schedule.get_agent_count(), 
                             "All_mice": lambda a: a.all_mice_schedule.get_agent_count(), 
                             "Unborn_mice": lambda a: a.num_unborn_mice,
                            "odor_layer_names": lambda a: a.odor_layer_names},
            agent_reporters={"age": lambda a: a.age,
                             "energy": lambda a: a.energy,
                             "generation": lambda a: a.generation,
                             "num_offspring": lambda a: a.num_offspring,
                             "hunger_status": lambda a: a.hunger_status,
                             "action_history": lambda a: a.action_history,
                            "possible_actions": lambda a: a.possible_actions,
                             "primary_values": lambda a: a.primary_values,
                             "secondary_values": lambda a: a.secondary_values,
                            "sensor_vector": lambda a: a.sensor_vector,
                             "motor_vector": lambda a: a.motor_vector,
                            "sensor_position": lambda a: a.sensor_position})
        
        self.predator_datacollector = MyDataCollector(
            agent_reporters={"Victims_num": lambda a: a.victims_num,
                             "odor_strength": lambda a: a.odor_strength,
                             "hunt_rule": lambda a: a.hunt_rule,
                             "odor_std": lambda a: a.odor_std,
                             "Damage_level": lambda a: a.damage_level})
        
        self.food_datacollector = MyDataCollector(
            agent_reporters={"Pos": lambda a: a.pos})