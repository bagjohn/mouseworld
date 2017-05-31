
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
    
    def __init__(self, model, genome, generation, motor_NN_on, learning_on, appraisal_NN_on):
        
        # Initial parameter setting
        self.model = model
        self.unique_id = model.give_next_id('Mouse')
        self.generation = generation
        
        # Constants
        self.max_energy = 1200
        self.max_gastric_content = 200
        self.energy = 500
        self.maturity_age = 200        
        self.metabolism_rate = 0.95
        self.primary_learning_rate = 0.1
        self.secondary_learning_rate = 0.01
        
        # Constants per turn, for energy loss to predators and amount of food to consume. 
        self.suffering_amount = 20
        self.feeding_amount = 20
        
        # Initialization of variables to be changed throughout life. Not to be retrieved
        self.age = 0
        self.energy_change = 0
        self.metabolism_buffer = 0
        self.gastric_content = 0
        self.hunger_status = 1
        self.incubation_time = 0   
        self.pregnant = False
        self.unborn = False
        self.header = random.uniform(0, 2*math.pi)
        #self.header = random.uniform(0, 2*math.pi)
        self.unborn_child = None
        self.num_offspring = 0
        
        # Initialization of cumulative counters. To be retrieved once per mouse        
        self.energy_to_predators = 0
        self.total_distance = 0
        self.food_gained_energy = 0
        self.food_lost_energy = 0
        
        # Genome to phenotype
        self.genome = genome
        self.max_speed = genome[0] * (5 - 1) + 1
        self.incubation_period = genome[1] * 200 + 100
        self.metabolism_range = genome[2]
        self.antenna_length = genome[3]
        self.antenna_angle = genome[4] * math.pi/2
        
        # Sensor and actor initialization
        self.brain_iterations_per_step = 50
        self.sensor_num = 2
        self.sensor_vector = np.zeros(shape = (self.model.groups_num,self.sensor_num))
        self.sensor_threshold = 0.0001
        self.sensor_position = [(0,0)] * self.sensor_num
        self.motor_num = 2
        self.motor_vector = np.zeros(self.motor_num)
        #self.possible_actions = [['wait'],['search_for_odor']]
        #self.trivial_possible_actions = [self.wait(),self.search_for_odor()]
        #self.current_action = [self.wait()]
        self.trivial_actions = [['Wait', None, 0, self.wait, None], 
                                ['Search', None, 0.00001, self.search_for_odor, None]]
        
        self.possible_actions = pd.DataFrame(self.trivial_actions, 
                                                     columns=('Verb', 'Noun_group', 'Value', 'Function', 'Arg_1'))
        
        #self.current_action = self.possible_actions.loc[0]
        self.action_history = pd.DataFrame([], columns=('Verb', 'Noun_group', 'Duration', 'Benefit', 'Termination', 'Distance'))
#         self.action_history = pd.DataFrame([['Wait', None, 0, 0, False]], 
#                                            columns=('Verb', 'Noun_group', 'Duration', 'Benefit', 'Closure'))
        
        self.current_action = self.possible_actions.loc[0]
#         self.action_value = pd.DataFrame([['Wait', np.nan, np.nan], ['Search', np.nan, np.nan]], 
#                                                      columns=('Verb', 'Noun_group', 'Value'))
        
        #self.primary_values = dict(zip(self.model.groups, [np.zeros(2)]*self.model.groups_num))
        self.primary_values = dict(zip(self.model.groups, [1 for i in self.model.groups]))
        self.secondary_values = pd.DataFrame([np.zeros(self.model.groups_num)] * self.model.groups_num, 
                                             index = self.model.groups, columns = self.model.odor_layer_names)
        #self.secondary_values = dict(zip(self.model.groups, [np.zeros(self.model.groups_num)] * self.model.groups_num))
        
        # Mousebrain initialization
        self.motor_NN_on = motor_NN_on
        self.appraisal_NN_on = appraisal_NN_on
        self.learning_on = learning_on
        
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
        self.action_history['Termination'][self.action_history.index.max()] = 'Death'
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
        
        # BIO : Genome passed to offspring 
        child_genome = self.mutate_genome()
        
        # BIO : New unborn mouse creation
        mouse = Mouse(self.model, child_genome, self.generation + 1, self.motor_NN_on, self.appraisal_NN_on)
        mouse.unborn = True
        
        # BIO : Pass primary values to offspring (TEST)
        mouse.primary_values = self.primary_values 
        
        # BIO : Parent mouse pregnant
        self.pregnant = True
        self.unborn_child = mouse
        
        # COUNTER
        self.model.num_unborn_mice += 1
        
        
    def give_birth(self):
        
        # TECH : Place newborn in the world
        self.model.place_agent_randomly(self.unborn_child)
        self.model.schedule.add(self.unborn_child)
        self.model.all_mice_schedule.add(self.unborn_child)
        
        # COUNTER
        self.model.num_mice += 1
        self.model.num_unborn_mice -= 1
        self.num_offspring +=1
        
        # BIO : Parent no longer pregnant
        self.pregnant = False
        self.incubation_time = 0
        self.unborn_child = None
      
    def set_sensor_position(self, pos, header) :
        left_antenna_header = (header + self.antenna_angle) % (math.pi*2)
        right_antenna_header = (header - self.antenna_angle) % (math.pi*2)
        left_antenna_pos = (pos[0] + math.cos(left_antenna_header) * self.antenna_length, pos[1] + math.sin(left_antenna_header) * self.antenna_length)
        right_antenna_pos = (pos[0] + math.cos(right_antenna_header) * self.antenna_length, pos[1] + math.sin(right_antenna_header) * self.antenna_length)
        sensor_position = [self.model.space.torus_adj(left_antenna_pos), self.model.space.torus_adj(right_antenna_pos)]
        for i in range(self.sensor_num) :
            sensor_position[i]=self.model.space._point_to_cell(sensor_position[i])
        return sensor_position
    
    def sense(self, pos) :
        odor_layers = self.model.odor_layers
        groups_num = self.model.groups_num
#         for i in range(self.sensor_num) :
#             pos[i]=self.model.space._point_to_cell(pos[i])
        #sensor_vector = [[0] * self.sensor_num] * groups_num
        sensor_vector = np.zeros(shape = (groups_num,self.sensor_num))
        for i in range(groups_num) :
            for j in range(self.sensor_num) :
                temp = odor_layers[i].get_value(pos[j])
                if temp > self.sensor_threshold :
                    sensor_vector[i][j] = temp
                else :
                    sensor_vector[i][j] = 0
            # trivial transformation for test purposes
            sensor_vector[i] = [np.mean(sensor_vector[i]), sensor_vector[i][0]-sensor_vector[i][1]]
        return sensor_vector      
    
    def update_possible_actions (self, actions) :
        #a = self.trivial_possible_actions 
        temp = self.sensor_vector
        groups = self.model.groups
        #layer  = self.model.odor_layers
        for i in range(len(temp)) :
            if temp[i][0] > 0 :
                # The secondary_values (as well as the primary_values) array, is indexed after model.groups_num.
                # The sensor_vector is also indexed after model.odor_layers, therefore they agree
                values_for_odor = self.secondary_values[self.model.odor_layer_names[i]]
                max_stim = values_for_odor.argmax()
                max_value = values_for_odor.max()
                min_stim = values_for_odor.argmin()
                min_value = values_for_odor.min()
                #In the primary and secondary values arrayys, the [0] is reward and the [1] is punishment, both positive
                if max_value > 0 :
                    # BIO : The expected reward of an approaching action is a function of 
                    # BIO : the currently perceived stimulus strength and its reward value for the organism
                    # BIO : 
                    #reward = self.hunger_status * value
                    actions.loc[actions.index.max() + 1] = ['Approach', max_stim, temp[i][0] * self.hunger_status * max_value, self.approach, temp[i]]
                if min_value < 0 : # & (isinstance (agent, Predator)) :
                    actions.loc[actions.index.max() + 1] = ['Avoid', min_stim, (-1) * temp[i][0] * min_value, self.avoid, temp[i]]
        return actions
    
    def add_feed_suffer_possibility(self, actions) :
        cellmates = self.model.space.get_neighbors(self.pos, 1, include_center=True)
        food_cellmates = [x for x in cellmates if isinstance (x, Food)]
        predator_cellmates = [x for x in cellmates if isinstance (x, mouseworld.predator.Predator)]
        if len(predator_cellmates) != 0:
            for predator in predator_cellmates :
                if np.random.uniform() > predator.escape_chance :
                    value = self.primary_values[predator.group]
                    actions.loc[actions.index.max() + 1] = ['Suffer', predator.group, value, self.suffer, predator]                  
        if len(food_cellmates) != 0:
            for food in food_cellmates :
                value = self.primary_values[food.group]
                actions.loc[actions.index.max() + 1] = ['Feed', food.group, self.hunger_status * value, self.feed, food]                  
        return actions
    
    def check_inheritance(self, current_action, possible_actions) :
        verb = current_action['Verb']
        noun = current_action['Noun_group']
        
        if verb == 'Approach' :
            a = possible_actions.loc[(possible_actions['Verb'] == 'Feed') & 
                                     (possible_actions['Noun_group'] == noun)]
            b = possible_actions.loc[(possible_actions['Verb'] == 'Approach') & 
                                     (possible_actions['Noun_group'] == noun)]
            if not a.empty :
                self.action_history['Termination'][self.action_history.index.max()] = 'Closure'
                return (a.loc[a['Value'].idxmax()])
            elif b.empty :
                self.action_history['Termination'][self.action_history.index.max()] = 'Failure'
                return (None)
            else :
                return (None)
            
        elif verb == 'Avoid' :
            a = possible_actions.loc[(possible_actions['Verb'] == 'Avoid') & 
                                     (possible_actions['Noun_group'] == noun)]
            if a.empty :
                self.action_history['Termination'][self.action_history.index.max()] = 'Closure'
            return (None)   
        elif verb == 'Search' :
            a = possible_actions.loc[(possible_actions['Verb'] == 'Approach') & 
                                     (possible_actions['Value'] > 0)]
            if not a.empty :
                self.action_history['Termination'][self.action_history.index.max()] = 'Closure'
                return (a.loc[a['Value'].idxmax()])
            else :
                return (None)
        else :
            return (None)
        
#     def evaluate(self, current_action, new_action) :
#         if (current_action['Verb'][0] == new_action['Verb'][0]) & 
#         (current_action['Noun_group'][0] == new_action['Noun_group'][0]) :self.action_history.loc[self.action_history.index.max()]
                
    def decide(self, current_action, possible_actions) :
        a = possible_actions.loc[(possible_actions['Verb'] == 'Suffer')]
        if not a.empty :
            return a.loc[a['Value'].idxmin()]
        else :
            temp = self.check_inheritance(current_action, possible_actions)       
            if temp is not None :
                return temp
            else :
                max_reward_action_ind = possible_actions['Value'].idxmax()
                new_action = possible_actions.loc[max_reward_action_ind]
                return new_action
            
            #max_reward = possible_actions['Value'].max()
            
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
        
        
        
#         if (current_action['Verb'] == new_action['Verb']) & ((current_action['Noun_group'] is None) or (current_action['Noun_group'] == new_action['Noun_group'])):
#             self.current_action['Duration'] += 1
#         else :
#             self.action_history.loc[self.action_history.index.max() + 1] = self.current_action
#             self.current_action = pd.Series([new_action['Verb'], new_action['Noun_group'], 0, 0, False], 
#                                            index =('Verb', 'Noun_group', 'Duration', 'Benefit', 'Closure'))
        
#         return new_action
    
    def update_action_history(self, action, action_history) :
        if action_history.empty :
            action_history.loc[0] = [action['Verb'], action['Noun_group'], 1, 0, None, 0]
        else :
            last_action = action_history.loc[action_history.index.max()]
            if (action['Verb'] == last_action['Verb']) & ((action['Noun_group'] is None) or (action['Noun_group'] == last_action['Noun_group'])):
                action_history['Duration'][action_history.index.max()] += 1
            else :
                action_history.loc[action_history.index.max() + 1] = [action['Verb'], action['Noun_group'], 1, 0, None, 0]
        return action_history
        
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
            if self.learning_on :
                self.input_manager.state = [0,-1,-1]
            else :
                self.input_manager.state = [-1,-1,-1]
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
        #self.motor_vector = [1,0]
    
    def approach(self, goal_sense) :
        #goal_sense = self.sensor_vector[odor_layer]
        
        if self.motor_NN_on :
            self.input_manager.value = goal_sense
            if self.learning_on :
                self.input_manager.state = [-1,0,-1]
            else :
                self.input_manager.state = [-1,-1,-1]
            
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
            if self.learning_on :
                self.input_manager.state = [-1,-1,0]
            else :
                self.input_manager.state = [-1,-1,-1]
            
#             with self.mousebrain_sim :
#                 self.mousebrain_sim.step()
            for i in range(self.brain_iterations_per_step) :
                self.mousebrain_sim.step()
            temp = self.mousebrain_sim.data[self.mousebrain.p_avoid]
            self.motor_vector = np.mean(temp[-self.brain_iterations_per_step : ], axis = 0)
        else :
            self.motor_vector = [np.exp(goal_sense[0])-1, goal_sense[1]]
            
    def suffer(self,predator) :
        
        # BIO : ENERGY LOSS due to suffering
        loss =  self.suffering_amount * predator.damage_level
        self.energy -= loss
        
        # COUNTER
        self.energy_to_predators += loss
        self.action_history['Benefit'][self.action_history.index.max()] -= loss
        
        # BIO : LEARNING when suffering
        self.primary_values[predator.group] += (-loss - self.primary_values[predator.group]) * self.primary_learning_rate
        self.update_odor_values (predator.group)
        
        # TECH : Motion when suffering is 0.
        self.motor_vector = np.zeros(self.motor_num)
    
    def feed(self, food) :
        
        # TECH : Feeding amount must be the minimum of [consumption per turn, space left in stomach and available food amount]
        feeding_amount = self.feeding_amount
        if feeding_amount > self.max_gastric_content-self.gastric_content :
            self.action_history['Termination'][self.action_history.index.max()] = 'Saturation'
            feeding_amount = self.max_gastric_content-self.gastric_content
        if feeding_amount > food.food_amount :
            self.action_history['Termination'][self.action_history.index.max()] = 'No food!'
            feeding_amount = food.food_amount
        food.food_amount -= feeding_amount
                
        # ENV : Nutritional value per unit is defined by the food parameter
        gain = feeding_amount * food.nutritional_value
        
        # BIO : ENERGY BUFFER GAIN due to feeding
        self.metabolism_buffer += gain
        # BIO : STOMACH FILLING due to feeding
        self.gastric_content += feeding_amount
        
        # COUNTER
        if food.nutritional_value > 0 :
            self.food_gained_energy += gain
            #self.primary_values[food.group] = [gain, 0]
        elif food.nutritional_value < 0 :
            self.food_lost_energy -= gain
            #self.primary_values[food.group] = [0,-gain]
        self.action_history['Benefit'][self.action_history.index.max()] += gain
        
        # BIO : LEARNING when feeding
        #self.secondary_values = self.update_odor_values(gain)
        # set the new value for the specific food group. At a later stage it must be elaborated
        self.primary_values[food.group] += (gain - self.primary_values[food.group]) * self.primary_learning_rate
        self.update_odor_values(food.group)
        
        # TECH : Motion when feeding is 0. 
        self.motor_vector = np.zeros(self.motor_num)
        
    def update_odor_values (self, group) :
        layer_names = self.model.odor_layer_names
        vector = self.sensor_vector
        primary_value = self.primary_values[group]
        values_for_group = self.secondary_values.ix[group]
        sum_values = sum(values_for_group)
        error = primary_value - sum_values
        
        for i in range(len(vector)) :
            if vector[i][0] > 0 :
                
                # BIO : Wikipedia on CLASSICAL CONDITIONING : The Rescorla-Wagner equation dV=a*b*(l-sum(V))
                # BIO :  V represents the current associative strength of the CS
                # BIO : dV is the change in this strength that happens on a given trial
                # BIO : sum(V) is the sum of the strengths of all stimuli present in the situation. 
                # BIO : l is the maximum associative strength that a given US will support; 
                # BIO : its value is usually set to 1 on trials when the US is present, and 0 when the US is absent
                # BIO : a and b are constants related to the salience of the CS and the speed of learning for a given US
                
                # BIO : a is the value of a specific odor at the point where conditioning occurs
                # BIO : b is the learning rate
                # BIO : 
                
                values_for_group[layer_names[i]] += error * vector[i][0] * self.secondary_learning_rate 
            else :
                continue
            
        #return secondary_values
    
    def move(self, motor_vector) :
        
        # BIO : Translate motor signal to behavior (how much to turn, how much to move)
        distance = motor_vector[0] * self.max_speed
        self.header = (self.header + motor_vector[1] * math.pi)%(2*math.pi)
        
        # COUNTER
        self.total_distance += distance
        self.action_history['Distance'][self.action_history.index.max()] += distance
        
        # TECH : Move the agent
        new_pos = (self.pos[0] + math.cos(self.header) * distance, self.pos[1] + math.sin(self.header) * distance)
        self.model.space.move_agent(self, self.model.space.torus_adj(new_pos))
        
        return distance
    
    def pay_metabolic_cost(self, pregnant, distance) :
        old_energy = self.energy
        
        # BIO : KATABOLISM RATE
        self.energy += 0.05 * self.metabolism_buffer
        if self.energy >= self.max_energy :
            self.energy = self.max_energy
        self.metabolism_buffer = self.metabolism_rate * self.metabolism_buffer
        if pregnant:
            self.energy -= (1 + distance)*2
        else :
            self.energy -= 1 + distance
        self.energy_change = self.energy - old_energy
        #self.current_action['Energy_loss'] +=
        self.gastric_content = 0.95 * self.gastric_content
        self.hunger_status = abs((1 * self.max_gastric_content - self.gastric_content)) / (1 * self.max_gastric_content)
    
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
            self.possible_actions = pd.DataFrame(self.trivial_actions, 
                                                     columns=('Verb', 'Noun_group', 'Value', 'Function', 'Arg_1'))
            self.sensor_position = self.set_sensor_position(self.pos, self.header)
            self.sensor_vector = self.sense(self.sensor_position)
            self.possible_actions = self.update_possible_actions(self.possible_actions)
            self.possible_actions = self.add_feed_suffer_possibility(self.possible_actions)
            self.current_action = self.decide(self.current_action, self.possible_actions)
            self.action_history = self.update_action_history(self.current_action, self.action_history)
            self.act(self.current_action)
            distance = self.move(self.motor_vector)
            self.pay_metabolic_cost (self.pregnant, distance)
            self.age += 1
            