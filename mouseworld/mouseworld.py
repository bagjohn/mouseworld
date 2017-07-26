
from mesa import Model
#from mesa.time import RandomActivation
#from mesa.space import ContinuousSpace
#from mesa.datacollection import DataCollector

import itertools
import numpy as np
import math
import pandas as pd
import random
import os
import collections
from scipy.stats import norm

# from mouseworld.myspace import ContinuousSpace
# from mouseworld.myspace import Value_layer
from mouseworld.mytime import *
from mouseworld.myspace import *
from mouseworld.mouse import Mouse
from mouseworld.food import Food
from mouseworld.predator import Predator
from mouseworld.mydatacollector import MyDataCollector
#from mouseworld.space_surface import Space_surface

from joblib import Parallel, delayed
import multiprocessing

class Mouseworld(Model):
    def __init__(self, num_mice_wo_MNN = 20, num_mice_w_MNN = 0,num_mice_w_lMNN = 0, 
                 num_food = 10, num_predators = 5, simulation_num = None,
                 phenotype_range = [(0,1), (0,1), (0,1), (0,1), (0,1), (0,1), (0,1), (0,1)],
                 mouse_initial_energy = 1000, mouse_max_energy = 1200,
                 mouse_position = 'random', food_position = 'random', predator_position = 'random',
                 primary_values = None, secondary_values = None, 
                 food_amount_range = (20,400), nutritional_value = [-1, 0.7, 1], food_growth_rate = [1],
                 width = 100, height = 100, mousebrain_inheritance = False, appraisal_knowledge_inheritance = False, 
                 mouse_reproduction = True, brain_iterations_per_step = 10, initial_mousebrain_weights = None, 
                 mousebrain_seed = None, test_veteran = False):
        
        # This attribute has been added for visualization purposes in order to be compatible with mesa ModularServer
        self.running = True
        
        # for parallel processing
        self.num_cores = multiprocessing.cpu_count()
        
        # create results folder
        if simulation_num :
            self.simulation_num = simulation_num
            self.directory = ('results/simulation_%s'%simulation_num)
            if not os.path.exists(self.directory):
                os.makedirs(self.directory)
            
        # define model variables from args
        self.num_mice = num_mice_wo_MNN + num_mice_w_MNN + num_mice_w_lMNN
        self.num_unborn_mice = 0
        self.phenotype_range = phenotype_range
        self.mouse_initial_energy = mouse_initial_energy
        self.mouse_max_energy = mouse_max_energy
        #self.num_genes = len(phenotype_range)
        self.num_food = num_food
        self.num_predators = num_predators
        self.mousebrain_inheritance = mousebrain_inheritance
        self.brain_iterations_per_step = brain_iterations_per_step
        self.mouse_reproduction = mouse_reproduction
        self.initial_mousebrain_weights = initial_mousebrain_weights
        self.mousebrain_seed = mousebrain_seed
        
        # build model continuous space
        self.space = ContinuousSpace(width, height, True, x_min=0, y_min=0,
            grid_width=width, grid_height=height)
        
        # define genomic parameters
        self.num_genes = 8
        self.num_positions_per_gene = 10
        self.genetic_map_seed = 0.3
        self.mutation_rate = 0.01
        self.recombination_rate = 0.01
        
        # initialize genome
        self.initialization_genome = self.initialize_genome(self.num_genes * self.num_positions_per_gene, self.num_mice)
        self.genetic_map = self.generate_genetic_map(self.num_genes, self.num_positions_per_gene, self.genetic_map_seed)
        
        # initialize positions
        if mouse_position == 'random' :
            self.initial_mouse_positions = self.initialize_pos_randomly(self.num_mice, header = True)
        elif mouse_position == 'in_quadrant' :
            self.initial_mouse_positions = self.initialize_pos_in_quadrant(self.num_mice)
        else :
            self.initial_mouse_positions = [mouse_position for i in range(self.num_mice)]
            
        if food_position == 'random' :
            self.initial_food_positions = self.initialize_pos_randomly(self.num_food)
        else : 
            self.initial_food_positions = [food_position for i in range(self.num_food)]
            
        if predator_position == 'random' :
            self.initial_predator_positions = self.initialize_pos_randomly(self.num_predators)
        else : 
            self.initial_predator_positions = [predator_position for i in range(self.num_predators)]
            
        # initialize food parameters
        self.food_amount_range = food_amount_range
        self.food_growth_rate = food_growth_rate
        
        self.food_odor_strength = [2] #[0.7,1]
        self.food_odor_std = [8]
        self.nutritional_value = nutritional_value #[-1, 0.7, 1]
        self.food_params = (self.food_odor_strength, self.nutritional_value, self.food_odor_std, self.food_growth_rate)
        self.food_param_combs = list(itertools.product(*self.food_params))
        self.food_groups_num = len(self.food_param_combs)
        self.food_groups = [('Food_group_%i'%i) for i in range(self.food_groups_num)]
        self.food_layers = [Value_layer('Food_odor_%i'%i, width, height, True) for i in range(self.food_groups_num)]
        self.food_layer_names = [('Food_odor_%i'%i) for i in range(self.food_groups_num)]

        
        # initialize predator parameters
        self.predator_odor_strength = [1] # [0.7,1]
        self.predator_odor_std = [8]
        self.damage_level = [1] #[0.3,1]
        self.hunt_rule = [1] #[0, 1]
        self.hunt_radius = [1] #[0.5, 1] #[0.5,1]
        self.predator_params = (self.predator_odor_strength, self.predator_odor_std, self.damage_level,
                                self.hunt_rule, self.hunt_radius)
        self.predator_param_combs = list(itertools.product(*self.predator_params))
        self.predator_groups_num = len(self.predator_param_combs)
        self.predator_groups = [('Predator_group_%i'%i) for i in range(self.predator_groups_num)]
        self.predator_layers = [Value_layer('Predator_odor_%i'%i, width, height, True) for i in range(self.predator_groups_num)]
        self.predator_layer_names = [('Predator_odor_%i'%i) for i in range(self.predator_groups_num)]

            
        # all agents (food & predator)
        self.groups_num = self.food_groups_num + self.predator_groups_num
        self.groups = self.food_groups + self.predator_groups
        self.group_odor_layers = self.food_layers + self.predator_layers
        self.group_odor_layer_names = self.food_layer_names + self.predator_layer_names
        
        # mating odor
        self.male_mating_odor_layer = Value_layer('Male_mating_odor', width, height, True)
        self.female_mating_odor_layer = Value_layer('Female_mating_odor', width, height, True)
        self.pregnancy_odor_layer = Value_layer('Pregnancy_odor', width, height, True)
        self.mouse_layers = [self.male_mating_odor_layer, self.female_mating_odor_layer, self.pregnancy_odor_layer]
        self.mouse_layer_names = [self.male_mating_odor_layer.unique_id, self.female_mating_odor_layer.unique_id,
                                  self.pregnancy_odor_layer.unique_id]
        
        # all odor layers
        self.odor_layers = self.group_odor_layers + self.mouse_layers
        self.odor_layer_names = self.group_odor_layer_names + self.mouse_layer_names

        # build schedules
        self.schedule = RandomActivation(self)
        self.all_mice_schedule = RandomActivation(self)
        self.food_schedule = RandomActivation(self)
        self.predator_schedule = RandomActivation(self)
        self.mouseworld_date = 0
        
        #initialize ids
        self.initialize_ids(['Mouse', 'Food', 'Predator'])
        
        #initialize ranks
        self.age_rank = []
        self.exp_search_rank = []
        self.exp_approach_rank = []
        self.exp_avoid_rank = []
        
        #initialize pair list
        self.candidate_pairs = []
        self.selected_pairs = []
        
        #initialize sensor_vector
#         self.sensor_num = 2
#         temp = [np.zeros(self.sensor_num)] * self.groups_num
#         self.zero_sensor_vector = pd.Series(temp, index=self.odor_layers)
        
        # Create environment agents (food & predators)
        for i in range(self.num_food):
            temp_position = self.initial_food_positions[i]
            j = i%(self.food_groups_num)
            food = Food(self.food_groups[j], j, self.food_layers[j], self.food_amount_range, self)
            self.food_schedule.add(food)
            self.space.place_agent(food, temp_position)
            #self.food_layers[j].add_agent(food)
            
        for i in range(self.num_predators):
            temp_position = self.initial_predator_positions[i]
            j = i%(self.predator_groups_num)
            predator = Predator(self.predator_groups[j], j, self.predator_layers[j], self)
            self.predator_schedule.add(predator)
            self.space.place_agent(predator, temp_position)
            #self.predator_layers[j].add_agent(predator)
    
        # Create acting agents (mice)
        for i in range(self.num_mice):
            temp_genome = self.initialization_genome[i]
            temp_position = self.initial_mouse_positions[i]
            if i < num_mice_wo_MNN :
                temp = [False, False, False]
            elif i < (num_mice_wo_MNN + num_mice_w_MNN):
                temp = [True, False, False]
            else :
                temp = [True, True, False]
            mouse = Mouse(self, [None,None], temp_genome, [0,0], 
                          motor_NN_on = temp[0], learning_on = temp[1], appraisal_NN_on = temp[2],
                          header = temp_position[1], brain_iterations_per_step = self.brain_iterations_per_step,
                          initial_mousebrain_weights = self.initial_mousebrain_weights, 
                          mousebrain_seed = self.mousebrain_seed)
            self.schedule.add(mouse)
            self.all_mice_schedule.add(mouse)
            self.space.place_agent(mouse, temp_position[0])
            
            if test_veteran :
                mouse = Mouse(self, [None,None], temp_genome, [0,0], 
                          motor_NN_on = temp[0], learning_on = temp[1], appraisal_NN_on = temp[2],
                          header = temp_position[1], brain_iterations_per_step = self.brain_iterations_per_step,
                          mousebrain_seed = self.mousebrain_seed, control_population = True)
                self.schedule.add(mouse)
                self.all_mice_schedule.add(mouse)
                self.space.place_agent(mouse, temp_position[0])
                
        for mouse in self.schedule.agents :
            if primary_values is not None :
                mouse.primary_values[self.food_groups[0]] = primary_values[0]
                mouse.primary_values[self.predator_groups[0]] = primary_values[1]

            if secondary_values is not None :
                mouse.secondary_values.ix[self.food_groups[0]][self.food_layer_names[0]]= secondary_values[0]
                mouse.secondary_values.ix[self.predator_groups[0]][self.predator_layer_names[0]]= secondary_values[1]

#             self.place_agent_randomly(mouse)   
        
        # Create data collectors        
#         self.initial_datacollector = MyDataCollector(
#             model_reporters={"Initial genome distribution": lambda a: a.initialization_genome})
        
        self.initial_datacollector = MyDataCollector(
            agent_reporters={"Genome": lambda a: a.genome})
#         self.datacollector = MyDataCollector(
#             model_reporters={"Alive_mice": lambda a: a.num_mice, 
#                              "Unborn_mice": lambda a: a.num_unborn_mice}
#             agent_reporters={"Header": lambda a: a.header,
#                              "Age": lambda a: a.age, 
#                              "Energy": lambda a: a.energy,
#                              "max_speed": lambda a: a.max_speed,
#                              "incubation_period": lambda a: a.incubation_period,
#                              "pos": lambda a: a.pos,
#                              "Genome": lambda a: a.genome})
        
        self.model_datacollector = MyDataCollector(
            model_reporters={"Alive_mice": lambda a: a.num_mice, 
                             "Unborn_mice": lambda a: a.num_unborn_mice})
        
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
                            "groups_num": lambda a: a.groups_num},
            agent_reporters={"age": lambda a: a.age,
                             "energy": lambda a: a.energy,
                             "sex": lambda a: a.sex,
                             "generation": lambda a: a.generation,
                             "num_offspring": lambda a: a.num_offspring,
                             "action_history": lambda a: a.action_history,
                            "possible_actions": lambda a: a.possible_actions,
                             "primary_values": lambda a: a.primary_values,
                             "secondary_values": lambda a: a.secondary_values,
                            "sensor_vector": lambda a: a.sensor_vector,
                             "motor_vector": lambda a: a.motor_vector,
                             "sensor_position": lambda a: a.sensor_position,
                            "motor_NN_on": lambda a: a.motor_NN_on,
                             "learning_on": lambda a: a.learning_on,
                             "appraisal_NN_on": lambda a: a.appraisal_NN_on,
                            "parent_ID": lambda a: a.parent_ID,
                            "offspring": lambda a: a.offspring,
                            "birth_date": lambda a: a.birth_date,
                            "death_date": lambda a: a.death_date,
                            "Genome": lambda a: a.genome,
                            "mousebrain_sim": lambda a: a.mousebrain_sim,
                            "initial_mousebrain_weights": lambda a: a.initial_mousebrain_weights,
#                              "current_mousebrain_weights": lambda a: a.current_mousebrain_weights,
                            "final_mousebrain_weights": lambda a: a.final_mousebrain_weights,
                            "first_action_duration": lambda a: a.action_history['Duration'][0],
                             "first_action_termination": lambda a: a.action_history['Termination'][0],
                            "control_population": lambda a: a.control_population})
        
        self.predator_datacollector = MyDataCollector(
            agent_reporters={"Victims_num": lambda a: a.victims_num,
                             "odor_strength": lambda a: a.odor_strength,
                             "hunt_rule": lambda a: a.hunt_rule,
                             "odor_std": lambda a: a.odor_std,
                             "Damage_level": lambda a: a.damage_level})
    
#     def show_odor_to_mice(self, agent) :
#         std = agent.odor_std
#         agents_in_radius = self.space.get_neighbors(agent.pos, std*3, include_center=True)
#         mice_in_radius = [x for x in agents_in_radius if isinstance (x, Mouse)]
#         num_mice_in_radius = len(mice_in_radius)
#         if len(mice_in_radius) != 0 :
#             for mouse in mice_in_radius :
#                 #get the appropriate odor value per sensor
#                 odor_value = []
#                 sensor_num = mouse.sensor_num
#                 sensor_position = mouse.sensor_position
#                 for i in range(sensor_num) :
#                     pos = sensor_position[i]
#                     distance = self.space.get_distance(agent.pos, pos)
#                     odor_value.append(norm.pdf(distance, scale = std)*10)
                    
#                 # trivial transformation for test purposes
#                 odor_value = (np.mean(odor_value), odor_value[0]-odor_value[1])
                
#                 #update the sensor vector
#                 mouse.sensor_vector[agent.odor_layer] = odor_value
                
#                 #update the mouse's possible actions
#                 self.update_mouse_possible_actions(mouse, mouse.possible_actions, agent.odor_layer, odor_value[0], agent)
    
#     def update_mouse_possible_actions (self, mouse, possible_actions, odor_layer, odor_value, agent) :
#         #IMPORTANT : Primary (Food, Predator) and Secondary (Odor) values are a [x,y] where x is the reward and y the punishment
#         # both positive
        
#         value = mouse.secondary_values[odor_layer]
#         #mouse.possible_actions
#         if value[0] > 0 :
#             possible_actions.loc[possible_actions.index.max() + 1] = ['Approach', agent.group, odor_value * mouse.hunger_status * value[0], 0, 0, mouse.approach, odor_layer]
#         if (value[1] > 0) & (isinstance (agent, Predator)) :
#             possible_actions.loc[possible_actions.index.max() + 1] = ['Avoid', agent.group, 0, (-1) * odor_value * value[1], 0, mouse.avoid, odor_layer]

#     def update_mouse_sensor_vector(self, mouse, odor_layer, odor_value) :
#         sensor_vector = mouse.sensor_vector
#         sensor_vector[odor_layer] = odor_value
        
    def initialize_ids(self, classes) :
        self.next_ids = np.ones(1, dtype={'names':classes, 'formats':[int]*len(classes)})
            
    def give_next_id(self, class_name) :
        ind = int(self.next_ids[class_name])
        ind = "{0:03}".format(ind)
        next_id = '%s_%s'%(class_name, ind)
        self.next_ids[class_name] += 1
        return next_id
    
    def generate_genetic_map (self, num_genes, num_positions_per_gene, genetic_map_seed) :
        positions = list(range(num_genes * num_positions_per_gene))
        random.shuffle(positions, lambda: genetic_map_seed)
        genetic_map = [positions[x:x+num_positions_per_gene] for x in range(0, len(positions),num_positions_per_gene)]
        return genetic_map
   
    def initialize_genome (self, genome_length, num_mice) :
        rand_str = lambda n: ''.join([random.choice(['0','1']) for i in range(n)])
        # Now to generate a random string of length 10
        genome = [rand_str(genome_length) for i in range(num_mice)]
        return genome
    
    def initialize_phenotype(self) :
        phenotype = [[np.random.uniform(low=low, high=high) for (low, high) in self.phenotype_range] for i in range(self.num_mice)]
        phenotype = np.around(phenotype, decimals = 2)
        #print(genome)
        return phenotype

    # Creates 19 * 8 = 152 combinations of position and header in a quadrant of space around (0,0)
    # Then adjusts to num
    def initialize_pos_in_quadrant(self, num) :
        pos =[(0,2), (0,3), (0,4), (0,5), (1,1), (1,2), (1,3), (1,4), (1,5), (2,2), (2,3), (2,4), (2,5), (3,3), (3,4), (3,5), (4,4), (4,5), (5,5)]
        header = [0,1,2,3,4,5,6,7]
        params = (pos, header)
        param_combs = list(itertools.product(*params))
        num_combs = len(param_combs)
        positions = [param_combs[i%num_combs] for i in range(num)]
        return positions
    
    # Add the agent to a random space point
    def place_agent_randomly(self, agent) :
        x = random.randrange(self.space.width)
        y = random.randrange(self.space.height)
        self.space.place_agent(agent, (x,y))
        
    # Creates num random positions in space with/without header 
    def initialize_pos_randomly(self, num, header = False):
        if header == True :
            positions = [((random.randrange(self.space.width), random.randrange(self.space.height)), 
                          random.uniform(0, 2*math.pi)) for i in range(num)]
        else :
            positions = [(random.randrange(self.space.width), random.randrange(self.space.height)) for i in range(num)]
        return positions
        
#         if hasattr(agent, 'sensor_position'):
#             agent.set_sensor_position()
    
#     def update_surfaces(self) :
#         for i in self.odor_layers :
#             i.update_surface()
    
    def diffuse_odor_layers(self, layers) :
        for layer in layers :
            layer.diffuse(0.95,0.8) 
            
    def diffuse_odor_layers_parallel(self, layers) :
        
        Parallel(n_jobs=self.num_cores)(delayed(layer.diffuse)(0.85,0.8) for layer in layers)
    
    def update_ranks(self) :
        self.age_rank = sorted(self.all_mice_schedule.agents, key=lambda x: x.age, reverse=False)
        self.exp_search_rank = sorted(self.all_mice_schedule.agents, key=lambda x: x.mousebrain_steps[0], reverse=True)
        self.exp_approach_rank = sorted(self.all_mice_schedule.agents, key=lambda x: x.mousebrain_steps[1], reverse=True)
        self.exp_avoid_rank = sorted(self.all_mice_schedule.agents, key=lambda x: x.mousebrain_steps[2], reverse=True)

    def define_pairs(self, candidate_pairs) :
        selected_pairs = []
        counter = collections.Counter(candidate_pairs)
        mutual_pairs = [pair for pair in counter if counter[pair]> 1]
        sorted_pairs = sorted(mutual_pairs, key=lambda x: -x[1])
        sorted_pairs = [x for (x,y) in sorted_pairs]
        while sorted_pairs :
            selected_pairs.append(sorted_pairs[0])
            sorted_pairs = [(z,w) for (z,w) in sorted_pairs if (z is not sorted_pairs[0][0] and w is not sorted_pairs[0][1])]
        return selected_pairs
    
    def generate_genome (self, male_genome, female_genome) :
        # BIO : Recombination might happen at a single site
        rand_rec = np.random.uniform(low=-1, high=1, size=None)
        if abs(rand_rec) < self.recombination_rate :
            rec_site = np.random.randint(low=1, high=len(male_genome)-1, size=None)
        else :
            rec_site = 0
        if rand_rec >= 0 :
            child_genome = male_genome[:rec_site] + female_genome[rec_site:]
        else :
            child_genome = female_genome[:rec_site] + male_genome[rec_site:]
        # BIO : Mutation might happen at multiple sites (Poisson process)
        mut_sites = np.random.poisson(self.mutation_rate, len(child_genome))
        temp=list(child_genome)
        for i in range(len(mut_sites)) :
            if mut_sites[i] == 1 :
                if temp[i] == '0' :
                    temp[i] = '1'
                elif temp[i] == '1' :
                    temp[i] = '0'
        #     print('%i,%s'%(s[i],c[i]))

        child_genome =''.join(temp)
        return child_genome
    
    def create_offspring (self, selected_pairs) :
        for pair in selected_pairs :
            male_genome = pair[0].genome
            female_genome = pair[1].genome
            child_genome = self.generate_genome(male_genome, female_genome)
            
            # BIO : New unborn mouse creation
            if self.mousebrain_inheritance :
            # BIO : Child inherits parent knowledge (TEST)
                mouse = Mouse(self, [pair[0].unique_id, pair[1].unique_id], child_genome, 
                              [pair[0].generation + 1, pair[1].generation + 1],pair[1].motor_NN_on, pair[1].learning_on, 
                              pair[1].appraisal_NN_on, initial_mousebrain_weights = pair[1].get_mousebrain_weights(),
                             brain_iterations_per_step = pair[1].brain_iterations_per_step)
            else :
                mouse = Mouse(self, [pair[0].unique_id, pair[1].unique_id], child_genome, [pair[0].generation + 1, pair[1].generation + 1],
                              pair[1].motor_NN_on, pair[1].learning_on, pair[1].appraisal_NN_on, initial_mousebrain_weights = None, 
                              brain_iterations_per_step = pair[1].brain_iterations_per_step)
            
            # BIO : Sexual drive drops to 0
            pair[0].sexual_drive = 0
            pair[1].sexual_drive = 0
            
            # BIO : Mother mouse will conceive
            pair[1].conceive(mouse)
            
            # COUNTER
            pair[0].offspring.append(mouse)
            pair[1].offspring.append(mouse)
            pair[0].num_offspring +=1
            pair[1].num_offspring +=1
            
    
    

    def step(self):
        '''Advance the model by one step.'''
        #self.predator_datacollector.collect(self,self.predator_schedule)
        self.candidate_pairs = []
        self.food_schedule.step()
        self.predator_schedule.step()
        self.schedule.update_status()
        #self.diffuse_odor_layers_parallel(self.odor_layers)
        self.diffuse_odor_layers(self.odor_layers)
        self.schedule.step() 
        self.selected_pairs = self.define_pairs(self.candidate_pairs)
        self.create_offspring(self.selected_pairs)
        self.update_ranks()
        self.test_datacollector.collect(self, self.schedule)
        self.model_datacollector.collect(self, self.schedule)
        self.mouseworld_date += 1