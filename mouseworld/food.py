
from mesa import Agent, Model
#from mesa.time import RandomActivation
#from mesa.space import ContinuousSpace
from mouseworld.myspace import ContinuousSpace

import random

class Food(Agent):
    
    def __init__(self, group, group_num, odor_layer, food_amount_range, model):
        self.model = model
        self.unique_id = model.give_next_id('Food')
        self.group = group
        self.group_num = group_num
        self.odor_layer = odor_layer
        self.food_amount = random.randint(food_amount_range[0],food_amount_range[1])
        temp_food_param_comb = self.model.food_param_combs[group_num]
        self.get_params(temp_food_param_comb)
        
    def get_params(self, params) :
        self.odor_strength = params[0]
        self.nutritional_value = params[1]
        self.odor_std = params[2]   
    
    def replace(self) :
        self.model.space.remove_agent(self)
        self.model.food_schedule.remove(self)
        #self.odor_layer.remove_agent(self)
        # for now we replace food
        food = Food(self.group, self.group_num, self.odor_layer, self.model.food_amount_range, self.model)
        
        self.model.place_agent_randomly(food)
        self.model.food_schedule.add(food)
        #self.odor_layer.add_agent(self)
        
    def step(self):
        if self.food_amount <= 0 :
            self.replace()
        else :
            grid_pos = self.model.space._point_to_cell(self.pos)
            self.odor_layer.add_value(grid_pos, self.odor_strength)            
            #self.model.show_odor_to_mice(self)
            #self.model.odor_matrix['food_odor_%i'%(self.group_num)][self.pos[0]][self.pos[1]] = self.odor_strength