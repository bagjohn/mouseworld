
import nengo
import numpy as np 
import math

# requires CCMSuite https://github.com/tcstewar/ccmsuite/
import ccm.lib.grid
import ccm.lib.continuous
import ccm.ui.nengo

mymap="""
#################################################################
#                                                               #
#                                                               #                       
#                                                               #
#                                                               #
#                                                               #        
#                                                               #                                    
#                                                               #            
#                                                               #
#                                                               #        
#                                                               #
#                                                               # 
#                                                               #                       
#                                                               #
#                                                               #
#                                                               #        
#                                                               #                                    
#                                                               #            
#                                                               #
#                                                               #        
#                                                               #
#                                                               #                
#                                                               #
#                                                               #                       
#                                                               #
#                                                               #
#                                                               #        
#                                                               #                                    
#                                                               #            
#                                                               #
#                                                               #        
#                                                               #
#                                                               #                
#                                                               #   
#                                                               #
#                                                               #
#                                                               #
#                                              f                #
#                                                               #
#                                                               #   
#                                                               #
#                                                               #
#                                                               #
#                                                               #
#                                                               #
#                                                               #
#################################################################

"""

class Cell(ccm.lib.grid.Cell):
    def color(self):
        return 'black' if self.wall else None
    def load(self, char):
        if char == '#':
            self.wall = True
        if char == 'f':
            self.reward = 10

world = ccm.lib.cellular.World(Cell, map=mymap, directions=8)

body = ccm.lib.continuous.Body()
food = ccm.lib.continuous.Body()
#antenna = ccm.lib.continuous.Body()

world.add(body, x=20, y=10, dir=2)
world.add(food, x=48, y=9, dir=2)   



model = nengo.Network(seed=8)
with model:
    def move(t, x):
        speed, rotation = x
        dt = 0.001
        max_speed = 20.0
        max_rotate = 10.0
        body.turn(rotation * dt * max_rotate)
        body.go_forward(speed * dt * max_speed)
        

    
    
  
    #nengo.Connection(movement, movement_node)

    env = ccm.ui.nengo.GridNode(world, dt=0.005)

    def distance2f(t):
        (dx, dy) = world.get_offset_in_direction(body.x, body.y, (int(body.dir - 1)%world.directions))
        (x_left, y_left) = (body.x + dx, body.y +dy)
        (dx, dy) = world.get_offset_in_direction(body.x, body.y, (int(body.dir + 1)%world.directions))
        (x_right, y_right) = (body.x + dx, body.y +dy)
        for a in [x_left, x_right] :
            if a < 0 or a >= world.width :
                a = body.x
        for a in [y_left, y_right] :
            if a < 0 or a >= world.height :
                a = body.y
                
        max_distance = math.sqrt(world.width**2 + world.height**2)
        dis_left = (math.sqrt((food.x-x_left)**2 + (food.y-y_left)**2))/max_distance
        dis_right = (math.sqrt((food.x-x_right)**2 + (food.y-y_right)**2))/max_distance   
        return (dis_left, dis_right)
    
    odor = nengo.Node(distance2f)

    odor_neurons = nengo.Ensemble(n_neurons=50, dimensions=2)
    nengo.Connection(odor, odor_neurons)
    
    odor_mean  = nengo.Ensemble(n_neurons=50, dimensions=1)
    nengo.Connection(odor[0], odor_mean, transform = 0.5)
    nengo.Connection(odor[1], odor_mean, transform = 0.5)
    
    odor_memory = nengo.Ensemble(n_neurons=50, dimensions=1)
    nengo.Connection(odor_mean, odor_memory, transform = 1, synapse=0.05)
    
    odor_change = nengo.Node(size_in=1, label='reward')
    nengo.Connection(odor_memory, odor_change, transform = -10, synapse=0.01)
    nengo.Connection(odor_mean, odor_change, transform = 10, synapse=0.01)
    
    
    odor2motor = nengo.Ensemble(n_neurons=100, dimensions=4, radius=4, seed=2,
                noise=nengo.processes.WhiteSignal(10, 0.1, rms=1))
    
    nengo.Connection(odor_change, odor2motor[0],synapse=0.01)
    nengo.Connection(odor_neurons[0], odor2motor[1],synapse=0.01)
    nengo.Connection(odor_neurons[1], odor2motor[2],synapse=0.01)
    nengo.Connection(odor_mean, odor2motor[3],synapse=0.01)
    
    movement = nengo.Node(size_in=2)
    
    def braiten(x):
        turn = x[1] - x[2]
        spd = math.exp(x[0])*x[3]
        return spd, turn
    nengo.Connection(odor2motor, movement, function=braiten)  
    
   
    
