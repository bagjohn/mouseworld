
from mesa import Agent
#from mesa.time import RandomActivation
#from mesa.space import ContinuousSpace

import nengo
import random
import math
import os
import numpy as np
import pandas as pd
from inspect import signature

#from mouseworld.myspace import ContinuousSpace
from mouseworld.input_manager import Input_manager
#from mouseworld.mousebrain import build_mousebrain
from mouseworld.mousebrain import Mousebrain
#import mouseworld.mousebrain
from mouseworld.food import Food
#from mouseworld.predator import Predator
import mouseworld.predator


class Mouse(Agent):
    
    def __init__(self, model, parent_ID, genome, generation, motor_NN_on, learning_on, appraisal_NN_on, 
                 sex = random.choice(['male', 'female']), header = random.uniform(0, 2*math.pi), 
                 initial_mousebrain_weights = None, mousebrain_seed = None, brain_iterations_per_step = 10, 
                 control_population = False):
        
        # Initial parameter setting
        self.model = model
        self.unique_id = model.give_next_id('Mouse')
        self.generation = generation
        self.parent_ID = parent_ID
        self.header = header
        self.control_population = control_population
        
        # Constants
        self.max_energy = self.model.mouse_max_energy
        self.max_gastric_content = 200
        self.energy = self.model.mouse_initial_energy
        self.maturity_age = 200       
        self.metabolism_rate = 0.95
        self.primary_learning_rate = 0.1
        self.secondary_learning_rate = 0.01
        self.conception_date = self.model.mouseworld_date
        self.sex = sex
        if self.sex == 'male' :
            self.mating_odor_layer = self.model.male_mating_odor_layer
        elif self.sex == 'female' :
            self.mating_odor_layer = self.model.female_mating_odor_layer
        
        # Constants per turn, for energy loss to predators and amount of food to consume. 
        self.suffering_amount = 20
        self.feeding_amount = 20
        
        # Initialization of variables to be changed throughout life. Not to be retrieved
        self.age = 0
        self.death_date = None
        self.birth_date = 0
        self.energy_change = 0
        self.metabolism_buffer = 0
        self.gastric_content = 0
        self.hunger_status = 1
        self.sexual_drive = 0
        self.incubation_time = 0   
        #self.header = random.uniform(0, 2*math.pi)
        self.unborn_child = None
        
        
        # Initialization of life cycle flags
        self.pregnant = False
        self.unborn = False
        self.mature = False
        
        # Initialization of cumulative counters. To be retrieved once per mouse        
        self.energy_to_predators = 0
        self.total_distance = 0
        self.food_gained_energy = 0
        self.food_lost_energy = 0
        self.num_offspring = 0
        self.offspring = []
        
        # Genome to phenotype
        self.genome = genome
        self.phenotype = self.gen_to_pheno(self.genome, self.model.genetic_map)
        
        self.max_speed = self.phenotype[0] * (5 - 1) + 1
        self.incubation_period = self.phenotype[1] * 200 + 100
#         self.incubation_period = genome[1] * 2 + 1
#         self.incubation_period = 0
        self.metabolism_range = self.phenotype[2]
        self.antenna_length = self.phenotype[3] * 3 + 0.5
        self.antenna_angle = self.phenotype[4] * math.pi/2
        self.feeding_behavioral_preference = self.phenotype[5]
        self.fear_behavioral_preference = self.phenotype[6]
        self.reproduction_behavioral_preference = self.phenotype[7]
        
        # Sensor and actor initialization
        self.sensor_num = 2
        self.sensor_vector = np.zeros(shape = (len(self.model.odor_layers),self.sensor_num))
#         self.sensor_vector = np.zeros(self.sensor_num, dtype={'names':self.model.odor_layer_names, 
#                                                               'formats':[('f4',)*self.sensor_num]*len(self.model.odor_layer_names)})
        self.sensor_threshold = 0.001
        self.sensor_position = [(0,0)] * self.sensor_num
        self.motor_num = 2
        self.motor_vector = np.zeros(self.motor_num)
        #self.possible_actions = [['wait'],['search_for_odor']]
        #self.trivial_possible_actions = [self.wait(),self.search_for_odor()]
        #self.current_action = [self.wait()]
        self.trivial_actions = [['Wait', None, 0, self.wait, None]]
        
        self.possible_actions = pd.DataFrame(self.trivial_actions, columns=('Verb', 'Noun_group', 'Value', 'Function', 'Arg_1'))
        
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
                                             index = self.model.groups, columns = self.model.group_odor_layer_names)
        #self.secondary_values = dict(zip(self.model.groups, [np.zeros(self.model.groups_num)] * self.model.groups_num))
        
        # Mousebrain initialization
        self.initial_mousebrain_weights = initial_mousebrain_weights
        self.final_mousebrain_weights = None
        self.mousebrain_steps = [0, 0, 0]
    
        self.mousebrain_seed = mousebrain_seed
        self.brain_iterations_per_step = brain_iterations_per_step
        self.motor_NN_on = motor_NN_on
        self.appraisal_NN_on = appraisal_NN_on
        self.learning_on = learning_on
        
        # TECH : Provide initialization steps to avoid switch artifact of NN.
        # TECH : Switch from one action to another, therefore from one goal sense to another.
        # TECH : From inspecting nengo_gui, it takes about 0.05 sec (50 * 0.001) to stabilize during manual switch. 
        # TECH : So 100 steps should be enough
        self.num_mousebrain_initialization_steps = 100
        
        if self.appraisal_NN_on:
            pass
        
        if self.motor_NN_on :
            self.input_manager = Input_manager()
            self.mousebrain = Mousebrain(seed = self.mousebrain_seed)
            self.mousebrain.build(self.input_manager, self.initial_mousebrain_weights)
            self.mousebrain_sim = nengo.Simulator(self.mousebrain, dt=0.001)
        else : 
            self.mousebrain = None
            self.mousebrain_sim = None
    
    def gen_to_pheno(self, genome, genetic_map) :
        temp = list(map(int, genome))
        phenotype = []
        for x in range(len(genetic_map)) :
            t = [temp[i] for i in genetic_map[x]]
            pheno = sum([t[i]*(2**i) for i in range(len(t))])/(2**self.model.num_positions_per_gene)
            phenotype.append(pheno)
        return phenotype
    
    def get_mousebrain_weights(self) :
#         if self.mousebrain_steps not in ([0,0,0],[None, None, None]) :
        if self.mousebrain_steps != [0,0,0] :
            temp0 = self.mousebrain_sim.data[self.mousebrain.p_approach_weights][-1]
            temp1 = self.mousebrain_sim.data[self.mousebrain.p_avoid_weights][-1]
            temp2 = self.mousebrain_sim.data[self.mousebrain.p_search_weights][-1]
            return [temp0, temp1, temp2]
        else :
            return self.initial_mousebrain_weights
    
    def store_mousebrain_weights(self) :
        directory = ('%s/veteran_mousebrains'%self.model.directory)
        if not os.path.exists(directory):
            os.makedirs(directory)
        filename = ('%s/veteran_%i_%i_%i.npz'%(directory, self.mousebrain_steps[0], self.mousebrain_steps[1], self.mousebrain_steps[2]))
        weights = self.get_mousebrain_weights()
        if weights :
            np.savez(filename, genome = self.genome, motor_NN_on = self.motor_NN_on, learning_on = self.learning_on, 
                     seed = self.mousebrain_seed, brain_iterations_per_step = self.brain_iterations_per_step, 
                     mousebrain_steps = self.mousebrain_steps, w_search=weights[0], w_approach=weights[1], w_avoid=weights[2],
                    simulation_num = self.model.simulation_num)
        else :
            np.savez(filename, genome = self.genome, motor_NN_on = self.motor_NN_on, learning_on = self.learning_on, 
                     seed = self.mousebrain_seed, brain_iterations_per_step = self.brain_iterations_per_step, 
                     mousebrain_steps = self.mousebrain_steps, simulation_num = self.model.simulation_num)
#         directory = ('results/veteran_mousebrains/%i'%self.mousebrain_seed)
#         if not os.path.exists(directory):
#             os.makedirs(directory)
#         filename = [('%s/%s_weights'%(directory, i)) for i in ['search', 'approach', 'avoid']]
#         weights = self.get_mousebrain_weights()
#         for i in range(len(filename)) :
#             f = filename[i]
#             w = weights[i]
#             np.save(f, w)
#         with open('results/veteran_mousebrains/mousebrain_exp.txt', 'a') as myfile:
#             myfile.write(str(self.mousebrain_steps) + '\t')
#             myfile.write(str(self.mousebrain_seed) + '\n')
    
    def die(self):
        if (self.pregnant) :
            #self.unborn_child.die()
            self.model.num_unborn_mice -= 1
        if self.motor_NN_on :
#             self.mousebrain.approach_ws.save(self.mousebrain_sim)
#             self.mousebrain.avoid_ws.save(self.mousebrain_sim)
#             self.mousebrain.search_ws.save(self.mousebrain_sim)
            self.final_mousebrain_weights = self.get_mousebrain_weights()
            self.mousebrain_sim.close()
        self.model.space.remove_agent(self)
        self.model.schedule.remove(self)
        self.action_history['Termination'][self.action_history.index.max()] = 'Death'
        self.model.num_mice -= 1
        self.death_age = self.age
        self.death_date = self.model.mouseworld_date
    
    def call_mate (self, partner_list) :
        if self.sex == 'male' :
            for mouse in partner_list :
                self.model.candidate_pairs.append(((self, mouse),self.sexual_drive * mouse.sexual_drive))
        elif self.sex == 'female' :
            for mouse in partner_list :
                self.model.candidate_pairs.append(((mouse, self),mouse.sexual_drive * self.sexual_drive))
    
#     def mate (self, partner) :
#         self.sexual_drive = 0
#         if self.sex == 'Female' :
#             self.conceive()
            
#     def mutate_genome(self) :
#         genome = self.genome
#         for i in range(len(genome)) :
#             rand = np.random.uniform(low=-1.0, high=1.0, size=None)
#             if abs(rand) <= 0.1 :
#                 genome[i] += np.sign(rand) * 0.1
#                 if genome[i] <= 0 :
#                     genome[i] = 0
#                 elif genome[i] >= 1 :
#                     genome[i] = 1
#         return np.around(genome, decimals = 2)
    
    def conceive(self, mouse):
        
#         # BIO : Genome passed to offspring 
# #         child_genome = self.mutate_genome()
        
#         # BIO : New unborn mouse creation
#         if self.model.mousebrain_inheritance :
#              # BIO : Child inherits parent knowledge (TEST)
#             mouse = Mouse(self.model, self.unique_id, child_genome, self.generation + 1, self.motor_NN_on, 
#                           self.learning_on, self.appraisal_NN_on, initial_mousebrain_weights = self.get_mousebrain_weights(),
#                          brain_iterations_per_step = self.brain_iterations_per_step)
#         else :
#             mouse = Mouse(self.model, self.unique_id, child_genome, self.generation + 1, self.motor_NN_on, 
#                           self.learning_on, self.appraisal_NN_on, initial_mousebrain_weights = None, 
#                           brain_iterations_per_step = self.brain_iterations_per_step)
#         mouse.unborn = True
#         self.offspring.append(mouse)
        
        # BIO : Pass primary and secondary values to offspring (TEST)
        if self.model.appraisal_knowledge_inheritance :
            mouse.primary_values = self.primary_values 
            mouse.secondary_values = self.secondary_values
        
        # BIO : Child mouse gets unborn status
        mouse.unborn = True
        
        # BIO : Parent mouse gets pregnant status
        self.pregnant = True
        self.unborn_child = mouse
        
        # COUNTER
        self.model.num_unborn_mice += 1
        
        
    def give_birth(self):
        
        # TECH : Place newborn in the world
        child_pos = (self.pos[0] + random.uniform(-1, 1), self.pos[1] + random.uniform(-1, 1))
        self.model.space.place_agent(self.unborn_child, child_pos)
        self.model.schedule.add(self.unborn_child)
        self.model.all_mice_schedule.add(self.unborn_child)
        
        # COUNTER
        self.model.num_mice += 1
        self.model.num_unborn_mice -= 1
        
        self.unborn_child.birth_date = self.model.mouseworld_date + 1
        
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
        odor_layers_num = len(odor_layers)
        sv = self.sensor_vector
#         for i in range(self.sensor_num) :
#             pos[i]=self.model.space._point_to_cell(pos[i])
        #sensor_vector = [[0] * self.sensor_num] * groups_num
#         sensor_vector = np.zeros(shape = (odor_layers_num,self.sensor_num))
        for i in range(odor_layers_num) :
            for j in range(self.sensor_num) :
                temp = odor_layers[i].get_value(pos[j])
                if temp > self.sensor_threshold :
                    sv[i][j] = temp
                else :
                    sv[i][j] = 0
            # trivial transformation for test purposes
            sv[i] = [np.mean(sv[i]), (sv[i][0]-sv[i][1])/np.mean(sv[i])]
        return sv      
    
    def update_possible_actions (self, actions) :
        actions.loc[actions.index.max() + 1] = ['Search', 'Food', self.feeding_behavioral_preference * 0.001 * self.hunger_status, 
                                                self.search_for_odor, None]
        actions.loc[actions.index.max() + 1] = ['Search', 'Mate', self.reproduction_behavioral_preference * 0.001 * self.sexual_drive, 
                                                self.search_for_odor, None]
        sv = self.sensor_vector
        groups = self.model.groups
        #layer  = self.model.odor_layers
        for i in range(self.model.groups_num) :
            if sv[i][0] > 0 :
                # The secondary_values (as well as the primary_values) array, is indexed after model.groups_num.
                # The sensor_vector is also indexed after model.odor_layers, therefore they agree
                values_for_odor = self.secondary_values[self.model.group_odor_layer_names[i]]
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
                    actions.loc[actions.index.max() + 1] = ['Approach', max_stim, 
                                                            self.feeding_behavioral_preference * sv[i][0] * self.hunger_status * max_value, 
                                                            self.approach, sv[i]]
                if min_value < 0 : # & (isinstance (agent, Predator)) :
                    actions.loc[actions.index.max() + 1] = ['Avoid', min_stim, 
                                                            self.fear_behavioral_preference * (-1) * sv[i][0] * min_value, 
                                                            self.avoid, sv[i]]
        
        # Add approach mating odor
        male_mating_sv = sv[self.model.groups_num]
        female_mating_sv = sv[self.model.groups_num + 1]
        if (male_mating_sv[0] > 0 and self.sex == 'female'):
            actions.loc[actions.index.max() + 1] = ['Approach', 'Male_mating_odor', 
                                                    self.reproduction_behavioral_preference * male_mating_sv[0] * self.sexual_drive,
                                                    self.approach, male_mating_sv]
        elif (female_mating_sv[0] > 0 and self.sex == 'male'):
            actions.loc[actions.index.max() + 1] = ['Approach', 'Female_mating_odor', 
                                                    self.reproduction_behavioral_preference * female_mating_sv[0] * self.sexual_drive,
                                                    self.approach, female_mating_sv]
        return actions
    
    def add_feed_suffer_mate_possibility(self, actions) :
        cellmates = self.model.space.get_neighbors(self.pos, 1, include_center=True)
        mice_cellmates = [x for x in cellmates if (isinstance (x, Mouse) and x.unique_id != self.unique_id)]
        male_mice_cellmates = [x for x in mice_cellmates if x.sex == 'male']
        female_mice_cellmates = [x for x in mice_cellmates if x.sex == 'female']
        food_cellmates = [x for x in cellmates if isinstance (x, Food)]
        predator_cellmates = [x for x in cellmates if isinstance (x, mouseworld.predator.Predator)]
        if self.sex == 'male' :
            f_ids = [female.unique_id for female in female_mice_cellmates]
            actions.loc[actions.index.max() + 1] = ['Call_mate', f_ids, self.reproduction_behavioral_preference * self.sexual_drive, self.call_mate, female_mice_cellmates]
        elif self.sex == 'female' :
            m_ids = [male.unique_id for male in male_mice_cellmates]
            actions.loc[actions.index.max() + 1] = ['Call_mate', m_ids, self.reproduction_behavioral_preference * self.sexual_drive, self.call_mate, male_mice_cellmates]
        if len(predator_cellmates) != 0:
            for predator in predator_cellmates :
                if np.random.uniform() > predator.escape_chance :
                    value = self.primary_values[predator.group]
                    actions.loc[actions.index.max() + 1] = ['Suffer', predator.group, 
                                                            self.fear_behavioral_preference * value, self.suffer, predator]                  
        if len(food_cellmates) != 0:
            for food in food_cellmates :
                value = self.primary_values[food.group]
                actions.loc[actions.index.max() + 1] = ['Feed', food.group, 
                                                        self.feeding_behavioral_preference * self.hunger_status * value, 
                                                        self.feed, food]                  
        return actions
    
    def check_inheritance(self, current_action, possible_actions) :
        verb = current_action['Verb']
        noun = current_action['Noun_group']
        
        if verb == 'Approach' :
            if (noun == 'Male_mating_odor' or noun == 'Female_mating_odor') :
                c = possible_actions.loc[(possible_actions['Verb'] == 'Call_mate') & (possible_actions['Value'] > 0)]
                if not c.empty :
                    self.action_history['Termination'][self.action_history.index.max()] = 'Closure'
                    return (c.loc[c['Value'].idxmax()])
                else :
                    return (None)
            else :
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
    def initialize_mousebrain(self, goal_sense) :
        if self.motor_NN :
            self.input_manager.value = goal_sense
            self.input_manager.state = [-1,-1,-1]
            self.mousebrain_sim.run_steps(self.num_mousebrain_initialization_steps, progress_bar=False)
            
    def update_action_history(self, action, action_history) :
        if action_history.empty :
            action_history.loc[0] = [action['Verb'], action['Noun_group'], 1, 0, None, 0]
            if any(t == action['Verb'] for t in ['approach', 'avoid']):
                self.initialize_mousebrain(action['Arg_1'])
        else :
            last_action = action_history.loc[action_history.index.max()]
            if (action['Verb'] == last_action['Verb']) & ((action['Noun_group'] is None) or (action['Noun_group'] == last_action['Noun_group'])):
                action_history['Duration'][action_history.index.max()] += 1
            else :
                action_history.loc[action_history.index.max() + 1] = [action['Verb'], action['Noun_group'], 1, 0, None, 0]
                if any(t == action['Verb'] for t in ['approach', 'avoid']):
                    self.initialize_mousebrain(action['Arg_1'])        
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
#         if self.motor_NN_on :
#                 self.current_mousebrain_weights = self.get_mousebrain_weights()
            
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
#             for i in range(self.brain_iterations_per_step) :
#                 self.mousebrain_sim.step()
            self.mousebrain_sim.run_steps(self.brain_iterations_per_step, progress_bar=False)
            self.mousebrain_steps[0] += 1
            #print(self.mousebrain_sim.data[self.mousebrain.p_search])
            temp = self.mousebrain_sim.data[self.mousebrain.p_search]
#             self.current_mousebrain_weights = self.get_mousebrain_weights()
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
#             for i in range(self.brain_iterations_per_step) :
#                 self.mousebrain_sim.step()
            self.mousebrain_sim.run_steps(self.brain_iterations_per_step, progress_bar=False)
            self.mousebrain_steps[1] += 1
            temp = self.mousebrain_sim.data[self.mousebrain.p_approach]
#             self.current_mousebrain_weights = self.get_mousebrain_weights()
            self.motor_vector = np.mean(temp[-self.brain_iterations_per_step : ], axis = 0)
        else :
            self.motor_vector = [np.exp(-goal_sense[0])-0.4, goal_sense[1]]
        
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
#             for i in range(self.brain_iterations_per_step) :
#                 self.mousebrain_sim.step()
            self.mousebrain_sim.run_steps(self.brain_iterations_per_step, progress_bar=False)
            self.mousebrain_steps[2] += 1
            temp = self.mousebrain_sim.data[self.mousebrain.p_avoid]
#             self.current_mousebrain_weights = self.get_mousebrain_weights()
            self.motor_vector = np.mean(temp[-self.brain_iterations_per_step : ], axis = 0)
        else :
            self.motor_vector = [np.exp(goal_sense[0])-0.9, -goal_sense[1]]
            
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
        
        for i in range(self.model.groups_num) :
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
        self.header = (self.header + motor_vector[1] * math.pi / 2)%(2*math.pi)
        
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
        self.energy += (1 - self.metabolism_rate) * self.metabolism_buffer
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
    
    def update_status(self):
        if self.energy <= 0 :
            self.die()
        else :
            if self.model.mouse_reproduction :
                if self.age == self.maturity_age :
                    self.mature = True
            if (self.mature and not self.pregnant) :
                self.sexual_drive += (1-self.sexual_drive)*0.1
                grid_pos = self.model.space._point_to_cell(self.pos)
                self.mating_odor_layer.add_value(grid_pos, self.sexual_drive) 
            if (self.pregnant) :
                self.incubation_time += 1
                if self.incubation_time >= self.incubation_period :
                    self.give_birth()
            
        
    def step(self):
        self.possible_actions = pd.DataFrame(self.trivial_actions, 
                                                 columns=('Verb', 'Noun_group', 'Value', 'Function', 'Arg_1'))
        self.sensor_position = self.set_sensor_position(self.pos, self.header)
        self.sensor_vector = self.sense(self.sensor_position)
        self.possible_actions = self.update_possible_actions(self.possible_actions)
        self.possible_actions = self.add_feed_suffer_mate_possibility(self.possible_actions)
        self.current_action = self.decide(self.current_action, self.possible_actions)
        self.action_history = self.update_action_history(self.current_action, self.action_history)
        self.act(self.current_action)
        distance = self.move(self.motor_vector)
        self.pay_metabolic_cost (self.pregnant, distance)
        self.age += 1
            
            