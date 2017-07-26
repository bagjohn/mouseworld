
from mesa import Agent, Model
#from mesa.time import RandomActivation
#from mesa.space import ContinuousSpace
from mouseworld.myspace import ContinuousSpace
import mouseworld.mouse 
#import Mouse

import math
import numpy as np

class Predator(Agent):
    
    def __init__(self, group, group_num, odor_layer, model):
        #super().__init__(unique_id, model)
        self.model = model
        self.unique_id = model.give_next_id('Predator')
        self.group = group
        self.group_num = group_num
        self.odor_layer = odor_layer
        self.victims_num = 0
        temp_predator_param_comb = self.model.predator_param_combs[group_num]
        self.get_params(temp_predator_param_comb)
        self.escape_chance = 0.1
        
    def get_params(self, params) :
        self.odor_strength = params[0]
        self.odor_std = params[1]
        self.damage_level = params[2]
        self.hunt_rule = params[3]
        self.hunt_radius = params[4]
        
    def move_naive(self):
        # move by 1 towards a random direction
        header = np.random.uniform (low = 0.0, high = math.pi*2)
        new_position = (self.pos[0] + math.cos(header), self.pos[1] + math.sin(header))
        self.model.space.move_agent(self, self.model.space.torus_adj(new_position))
    
    def move_smart(self):
        # move by 1 towards a random mouse within 3 radius
        neighbors = self.model.space.get_neighbors(self.pos, self.hunt_radius*3, include_center=True)
        neighbor_mice = [x for x in neighbors if isinstance (x, mouseworld.mouse.Mouse)]
        if neighbor_mice :
            unlucky_mouse = np.random.choice(neighbor_mice)
            distance = self.model.space.get_distance(unlucky_mouse.pos, self.pos)
            trans_vector = [(unlucky_mouse.pos[0] - self.pos[0])/distance,(unlucky_mouse.pos[1] - self.pos[1])/distance]
            new_position = (self.pos[0] + trans_vector[0], self.pos[1] + trans_vector[1])
            self.model.space.move_agent(self, self.model.space.torus_adj(new_position))
        else :
            self.move_naive()
        
    def find_mice(self):
        cellmates = self.model.space.get_neighbors(self.pos, self.hunt_radius, include_center=True)
        mice_cellmates = [x for x in cellmates if isinstance (x, mouseworld.mouse.Mouse)]
        #self.model.space._grid.iter_cell_list_contents(self.model.space._point_to_cell(self.pos))
        return mice_cellmates
        #if len(mice_cellmates) != 0:
            #self.hungry = False
            #self.victims_num += 1
            #return True
#             for o in mice_cellmates :
#                 self.victims_num += 1
#                 loss = o.energy * self.damage_level
#                 o.energy -= loss
#                 o.primary_values[self.group] = [0, loss]
#                 o.secondary_values[self.odor_layer] = o.primary_values[self.group]
    
    def step(self):
        #self.hungry = True
        mice_cellmates = self.find_mice()
        if not mice_cellmates :
            if self.hunt_rule == 0 :
                self.move_naive()
            elif self.hunt_rule == 1 :
                self.move_smart()
        #self.find_mice
        grid_pos = self.model.space._point_to_cell(self.pos)
        self.odor_layer.add_value(grid_pos, self.odor_strength)
        #self.odor_layer.update_agent_location(self.unique_id, self.pos)
        #self.model.show_odor_to_mice(self)
        #self.model.odor_matrix['predator_odor_%i'%(self.group_num)][self.pos[0]][self.pos[1]] = self.odor_strength
        