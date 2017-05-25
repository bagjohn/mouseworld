
from mesa import Agent, Model
#from mesa.time import RandomActivation
#from mesa.space import ContinuousSpace

import nengo
import random
import math
import numpy as np
import pandas as pd
from inspect import signature

from mouseworld.myspace import ContinuousSpace
from mouseworld.input_manager import Input_manager
#from mouseworld.mousebrain import build_mousebrain
from mouseworld.mousebrain import Mousebrain
#import mouseworld.mousebrain
from mouseworld.food import Food
#from mouseworld.predator import Predator
import mouseworld.predator


class Mouse(Agent):
    
    def __init__(self, model, genome, motor_NN_on, appraisal_NN_on):
        
        # Initial parameter setting
        self.model = model
        self.unique_id = model.give_next_id('Mouse')
        
        # Constants
        self.max_energy = 1200
        self.max_gastric_content = 200
        self.energy = 1000
        self.maturity_age = 800        
        
        # Constants per turn, for energy loss to predators and amount of food to consume. 
        self.suffering_amount = 20
        self.feeding_amount = 20
        
        # Initialization of variables to be changed throughout life. Not to be retrieved
        self.age = 0
        self.energy_change = 0
        self.food_to_energy = 0
        self.gastric_content = 0
        self.hunger_status = 0
        self.incubation_time = 0   
        self.pregnant = False
        self.unborn = False
        self.header = random.uniform(0, 2*math.pi)
        self.unborn_child = None
        
        # Initialization of cumulative counters. To be retrieved once per mouse        
        self.energy_to_predators = 0
        self.total_distance = 0
        self.food_gained_energy = 0
        self.food_lost_energy = 0
        
        # Genome to phenotype
        self.genome = genome
        self.max_speed = genome[0] * (5 - 1) + 1
        self.incubation_period = genome[1] * 300
        self.metabolism_range = genome[2]
        self.antenna_length = genome[3] * 5
        self.antenna_angle = genome[4] * math.pi/2
        
        # Sensor and actor initialization
        self.brain_iterations_per_step = 10
        self.sensor_num = 2
        self.sensor_vector = [[0] * self.sensor_num] * (self.model.groups_num)
        self.sensor_threshold = 0.0001
        self.sensor_position = [(0,0)] * self.sensor_num
        self.motor_num = 2
        self.motor_vector = np.zeros(self.motor_num)
        #self.possible_actions = [['wait'],['search_for_odor']]
        #self.trivial_possible_actions = [self.wait(),self.search_for_odor()]
        #self.current_action = [self.wait()]
        self.trivial_possible_actions = pd.DataFrame([['Wait', None, 0, self.wait, None], 
                                                      ['Search', None, 0.00001, self.search_for_odor, None]], 
                                                     columns=('Verb', 'Noun_group', 'Value', 'Function', 'Arg_1'))
        self.possible_actions = self.trivial_possible_actions
        
        #self.current_action = self.possible_actions.loc[0]
        self.action_history = pd.DataFrame([['Wait', None, 0, 0, False]], 
                                           columns=('Verb', 'Noun_group', 'Duration', 'Benefit', 'Closure'))
        
        self.current_action = self.action_history.loc[0]
#         self.action_value = pd.DataFrame([['Wait', np.nan, np.nan], ['Search', np.nan, np.nan]], 
#                                                      columns=('Verb', 'Noun_group', 'Value'))
        
        #self.primary_values = dict(zip(self.model.groups, [np.zeros(2)]*self.model.groups_num))
        self.primary_values = dict(zip(self.model.groups, [0.0001]*self.model.groups_num))
        self.secondary_values = dict(zip(self.model.odor_layers, [0]*self.model.groups_num))
        
        # Mousebrain initialization
        self.motor_NN_on = motor_NN_on
        self.appraisal_NN_on = appraisal_NN_on
        
        if appraisal_NN_on:
            pass
        
        if motor_NN_on :
            self.input_manager = Input_manager()
            self.mousebrain = Mousebrain()
            self.mousebrain.build(self.input_manager)
            self.mousebrain_sim = nengo.Simulator(self.mousebrain, dt=0.001)
        
    def die(self):
        if (self.pregnant) :
            #self.unborn_child.die()
            self.model.num_unborn_mice -= 1
        self.model.space.remove_agent(self)
        self.model.schedule.remove(self)
        self.model.num_mice -= 1
        self.death_age = self.age
    
    def mutate_genome(self) :
        genome = self.genome
        for i in range(len(genome)) :
            rand = np.random.uniform(low=-1.0, high=1.0, size=None)
            if abs(rand) <= 0.1 :
                genome[i] += np.sign(rand) * 0.1
                if genome[i] <= 0 :
                    genome[i] = 0
                elif genome[i] >= 1 :
                    genome[i] = 1
        return np.around(genome, decimals = 2)
    
    def conceive(self):
        child_genome = self.mutate_genome()
        mouse = Mouse(self.model, child_genome, self.motor_NN_on, self.appraisal_NN_on)
        mouse.unborn = True
        self.model.num_unborn_mice += 1
        self.pregnant = True
        self.unborn_child = mouse
        
    def give_birth(self):
        self.model.place_agent_randomly(self.unborn_child)
        self.model.schedule.add(self.unborn_child)
        self.model.num_mice += 1
        self.model.num_unborn_mice -= 1
        self.pregnant = False
        self.incubation_time = 0
        self.unborn_child = None
    
    def sense(self) :
        odor_layers = self.model.odor_layers
        groups_num = self.model.groups_num
        pos = self.sensor_position
        for i in range(self.sensor_num) :
            pos[i]=self.model.space._point_to_cell(pos[i])
        sensor_vector = [[0] * self.sensor_num] * groups_num
        for i in range(groups_num) :
            for j in range(self.sensor_num) :
                temp = odor_layers[i].get_value(pos[j])
                if temp > self.sensor_threshold :
                    sensor_vector[i][j] = temp
            # trivial transformation for test purposes
            sensor_vector[i] = (np.mean(sensor_vector[i]), sensor_vector[i][0]-sensor_vector[i][1])
        return sensor_vector      
    
    def update_possible_actions (self) :
        a = self.trivial_possible_actions 
        temp = self.sensor_vector
        groups = self.model.groups
        #layer  = self.model.odor_layers
        for i in range(len(temp)) :
            if temp[i][0] > 0 :
                # The secondary_values (as well as the primary_values) array, is indexed after model.groups_num.
                # The sensor_vector is also indexed after model.odor_layers, therefore they agree
                value = self.secondary_values[self.model.odor_layers[i]]
                #In the primary and secondary values arrayys, the [0] is reward and the [1] is punishment, both positive
                if value > 0 :
                    a.loc[a.index.max() + 1] = ['Approach', groups[i], temp[i][0] * self.hunger_status * value, self.approach, temp[i]]
                elif value < 0 : # & (isinstance (agent, Predator)) :
                    a.loc[a.index.max() + 1] = ['Avoid', groups[i], (-1) * temp[i][0] * value, self.avoid, temp[i]]
        return a
    
    def detect_action_closure(self, current_action, possible_actions) :
        verb = current_action['Verb']
        if verb == 'Suffer' :
            a = possible_actions.loc[(possible_actions['Verb'] == 'Suffer') & 
                                     (possible_actions['Noun_group'] == current_action['Noun_group'])]
            if not a.empty :
                return (True , a.loc[a['Value'].idxmin()])
            else :
                return (False, None)
        elif verb == 'Approach' :
            a = possible_actions.loc[(possible_actions['Verb'] == 'Feed') & 
                                     (possible_actions['Noun_group'] == current_action['Noun_group'])]
            if not a.empty :
                return (True , a.loc[a['Value'].idxmax()])
            else :
                return (False, None)
        elif verb == 'Avoid' :
            a = possible_actions.loc[(possible_actions['Verb'] == 'Avoid') & 
                                     (possible_actions['Noun_group'] == current_action['Noun_group'])]
            if a.empty :
                return (True, None)
            else :
                return (False, None)
        elif verb == 'Search' :
            a = possible_actions.loc[(possible_actions['Verb'] == 'Approach') & 
                                     (possible_actions['Value'] > 0)]
            if not a.empty :
                return (True, a.loc[a['Value'].idxmax()])
            else :
                return (False, None)
        else :
            return (False, None)
    
#     def evaluate(self, current_action, new_action) :
#         if (current_action['Verb'][0] == new_action['Verb'][0]) & 
#         (current_action['Noun_group'][0] == new_action['Noun_group'][0]) :
            
        
    def decide(self, closure, current_action, possible_actions) :
        #print(possible_actions)
        if closure[0] == True :
            self.current_action['Closure'] = True
        
        if  closure[1] is not None :
            new_action = closure[1]
        else :
            #max_reward = possible_actions['Value'].max()
            max_reward_action_ind = possible_actions['Value'].idxmax()
            new_action = possible_actions.loc[max_reward_action_ind]
#             max_reward = possible_actions['Reward_Value'].max()
#             min_punishment = possible_actions['Punishment_Value'].min()
#             if max_reward >= abs(min_punishment) :
#                 max_reward_action_ind = possible_actions['Reward_Value'].idxmax()
#                 new_action = possible_actions.loc[max_reward_action_ind]
#             else :
#                 min_punishment_action_ind = possible_actions['Punishment_Value'].idxmin()
#                 new_action = possible_actions.loc[min_punishment_action_ind]
            
        #print(current_action)    
        #print(new_action) 
        if (current_action['Verb'] == new_action['Verb']) & ((current_action['Noun_group'] is None) or (current_action['Noun_group'] == new_action['Noun_group'])):
            self.current_action['Duration'] += 1
        else :
            self.action_history.loc[self.action_history.index.max() + 1] = self.current_action
            self.current_action = pd.Series([new_action['Verb'], new_action['Noun_group'], 0, 0, False], 
                                           index =('Verb', 'Noun_group', 'Duration', 'Benefit', 'Closure'))
        
        return new_action
    
    def act(self, action) :
        function = action['Function']
        sig = signature(function)
        num_args = len(sig.parameters)
        if num_args == 0 :
            function()
        elif num_args == 1 :
            arg = action['Arg_1']
            function(arg)
            
    def wait(self) :
        self.motor_vector = np.zeros(self.motor_num)
        
    def search_for_odor(self) :
        if self.motor_NN_on :
            self.input_manager.value = (0,0)
            self.input_manager.state = [0,-1,-1]
#             with self.mousebrain_sim :
#                 self.mousebrain_sim.step()
            for i in range(self.brain_iterations_per_step) :
                self.mousebrain_sim.step()
            #print(self.mousebrain_sim.data[self.mousebrain.p_search])
            temp = self.mousebrain_sim.data[self.mousebrain.p_search]
            self.motor_vector = np.mean(temp[-self.brain_iterations_per_step : ], axis = 0)
            #print(self.motor_vector)
        else :
            #motor_vector = [random.uniform(0, 1)]*self.motor_num
            self.motor_vector = [1,0]
    
    def approach(self, goal_sense) :
        #goal_sense = self.sensor_vector[odor_layer]
        
        if self.motor_NN_on :
            self.input_manager.value = goal_sense
            self.input_manager.state = [-1,0,-1]
#             with self.mousebrain_sim :
#                 self.mousebrain_sim.step()
            for i in range(self.brain_iterations_per_step) :
                self.mousebrain_sim.step()
            temp = self.mousebrain_sim.data[self.mousebrain.p_approach]
            self.motor_vector = np.mean(temp[-self.brain_iterations_per_step : ], axis = 0)
        else :
            self.motor_vector = [np.exp(-goal_sense[0]), -goal_sense[1]]
        
    def avoid(self, goal_sense) :
        #goal_sense = self.sensor_vector[odor_layer]
        
        if self.motor_NN_on :
            self.input_manager.value = goal_sense
            self.input_manager.state = [-1,-1,0]
#             with self.mousebrain_sim :
#                 self.mousebrain_sim.step()
            for i in range(self.brain_iterations_per_step) :
                self.mousebrain_sim.step()
            temp = self.mousebrain_sim.data[self.mousebrain.p_avoid]
            self.motor_vector = np.mean(temp[-self.brain_iterations_per_step : ], axis = 0)
        else :
            self.motor_vector = [np.exp(goal_sense[0])-1, goal_sense[1]]
                
    def move(self, motor_vector) :
        distance = motor_vector[0] * self.max_speed
        self.total_distance += distance
        self.header = (self.header + motor_vector[1] * math.pi)%(2*math.pi)
        new_pos = (self.pos[0] + math.cos(self.header) * distance, self.pos[1] + math.sin(self.header) * distance)
        self.model.space.move_agent(self, self.model.space.torus_adj(new_pos))
        return distance
    
    def add_feed_suffer_possibility(self, actions) :
        cellmates = self.model.space.get_neighbors(self.pos, 1, include_center=True)
        food_cellmates = [x for x in cellmates if isinstance (x, Food)]
        predator_cellmates = [x for x in cellmates if isinstance (x, mouseworld.predator.Predator)]
        if len(predator_cellmates) != 0:
            for predator in predator_cellmates :
                if np.random.uniform() < predator.escape_chance :
                    value = self.primary_values[predator.group]
                    actions.loc[actions.index.max() + 1] = ['Suffer', predator.group, value, self.suffer, predator]                  
        if len(food_cellmates) != 0:
            for food in food_cellmates :
                value = self.primary_values[food.group]
                actions.loc[actions.index.max() + 1] = ['Feed', food.group, self.hunger_status * value, self.feed, food]                  
        return actions
    
    def suffer(self,predator) :
        self.motor_vector = np.zeros(self.motor_num)
        loss =  self.suffering_amount * predator.damage_level
        self.energy -= loss
        self.energy_to_predators += loss
        
        self.current_action['Benefit'] -= loss
        self.secondary_values = self.update_odor_values (-loss)
        self.primary_values[predator.group] -= loss

    
    def feed(self, food) :
        self.motor_vector = np.zeros(self.motor_num)
        feeding_amount = min(self.feeding_amount, food.food_amount)
        food.food_amount -= feeding_amount
        self.gastric_content += feeding_amount
        gain = feeding_amount * food.nutritional_value
        self.food_to_energy += gain
        if food.nutritional_value > 0 :
            self.food_gained_energy += gain
            #self.primary_values[food.group] = [gain, 0]
        elif food.nutritional_value < 0 :
            self.food_lost_energy -= gain
            #self.primary_values[food.group] = [0,-gain]
        # set the new value for the specific food group. At a later stage it must be elaborated
        self.current_action['Benefit'] += gain
        self.secondary_values = self.update_odor_values (gain)
        self.primary_values[food.group] += gain
        #self.secondary_values[food.odor_layer] = self.primary_values[food.group]    
        
    def update_odor_values (self, gain) :
        vector = self.sensor_vector
        values = self.secondary_values
        for i in range(len(vector)) :
            if vector[i][0] != 0 :
                values[self.model.odor_layers[i]] += vector[i][0] * gain
        return values
    
    def pay_metabolic_cost(self, pregnant, distance) :
        old_energy = self.energy
        self.energy += 0.05 * self.food_to_energy
        if self.energy >= self.max_energy :
            self.energy = self.max_energy
        self.food_to_energy = 0.95 * self.food_to_energy
        if pregnant:
            self.energy -= (1 + distance)*2
        else :
            self.energy -= 1 + distance
        self.energy_change = self.energy - old_energy
        #self.current_action['Energy_loss'] +=
        self.gastric_content = 0.95 * self.gastric_content
        self.hunger_status = abs((0.9 * self.max_gastric_content - self.gastric_content)) / (0.9 * self.max_gastric_content)
    
    def set_sensor_position(self) :
        left_antenna_header = (self.header - self.antenna_angle) % (math.pi*2)
        right_antenna_header = (self.header + self.antenna_angle) % (math.pi*2)
        left_antenna_pos = (self.pos[0] + math.cos(left_antenna_header) * self.antenna_length, self.pos[1] + math.sin(left_antenna_header) * self.antenna_length)
        right_antenna_pos = (self.pos[0] + math.cos(right_antenna_header) * self.antenna_length, self.pos[1] + math.sin(right_antenna_header) * self.antenna_length)
        self.sensor_position = [self.model.space.torus_adj(left_antenna_pos), self.model.space.torus_adj(right_antenna_pos)]
    
    def step(self):
        if self.energy <= 0 :
            self.die()
        else :
            if (self.age >= self.maturity_age and self.pregnant == False) :
                self.conceive()
            if (self.pregnant) :
                self.incubation_time += 1
                if self.incubation_time >= self.incubation_period :
                    self.give_birth()
            self.sensor_vector = self.sense()
            self.possible_actions = self.update_possible_actions()
            self.possible_actions = self.add_feed_suffer_possibility(self.possible_actions)
            self.closure = self.detect_action_closure(self.current_action, self.possible_actions)
            #print(self.closure)
            #print(self.closure[0])
            #print(self.closure[1])
            action = self.decide(self.closure, self.current_action, self.possible_actions)
            self.act(action)
            distance = self.move(self.motor_vector)
            self.pay_metabolic_cost (self.pregnant, distance)
            self.age += 1
            #self.sensor_vector = self.model.zero_sensor_vector
            self.set_sensor_position()
            #self.possible_actions = self.trivial_possible_actions 