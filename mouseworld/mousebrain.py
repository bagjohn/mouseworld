
import numpy as np
import matplotlib.pyplot as plt

#import mouseworld.mouse
from mouseworld.input_manager import Input_manager

#from mouseworld.mouse import Input_manager
import nengo

class Mousebrain(nengo.Network) :
    
    def approach(self, x):
        spd = np.exp(-x[0])
        turn = -x[1]
        return spd, turn

    def avoid(self, x) :
        spd = np.exp(x[0])-0.9    
        turn = x[1]        
        return spd, turn

    def search(self, x) :
        spd = 1
        turn = 0
        return spd, turn

    
    def build(self, input_manager):

        #mousebrain  = nengo.Network()
        with self:
            odor = nengo.Node(input_manager.return_value,size_out = 2)

            state = nengo.Node(input_manager.return_state,size_out = 3)

            odor_neurons = nengo.Ensemble(n_neurons=50, dimensions=2, radius=1)
            nengo.Connection(odor, odor_neurons, synapse = None)

            odor_memory = nengo.Ensemble(n_neurons=200, dimensions=2)
            nengo.Connection(odor[0], odor_memory[0], transform = 1, synapse=None)
            nengo.Connection(odor_memory[0], odor_memory[1], transform = 1, synapse=0.3)
            #nengo.Connection(odor_memory[1], odor_memory[0], transform = -0.5, synapse=0.3)

            odor_change = nengo.Ensemble(n_neurons = 200, dimensions = 1, radius=0.1,  
                            max_rates=nengo.dists.Uniform(300, 400), intercepts=nengo.dists.Uniform(0, 0.1))
            nengo.Connection(odor_memory[0], odor_change, transform = 1, synapse=0.01)
            nengo.Connection(odor_memory[1], odor_change, transform = -1, synapse=0.01)
            #nengo.Connection(odor_neurons[0], odor_change, transform = -0.1, synapse=0.1)

            #hub = nengo.Node(size_in = 1,size_out=3)
            errors = nengo.networks.EnsembleArray(n_neurons=100, n_ensembles=3, ens_dimensions=2)
            #nengo.Connection(odor_change, hub)
            nengo.Connection(odor_change, errors.input, transform = [[1]]*6, synapse=0.1)
            nengo.Connection(state[0], errors.ensembles[0].neurons, transform=np.ones((100,1))*1)
            nengo.Connection(state[1], errors.ensembles[1].neurons, transform=np.ones((100,1))*1)
            nengo.Connection(state[2], errors.ensembles[2].neurons, transform=np.ones((100,1))*1)

            #nengo.Connection(odor_memory, odor_change[0], transform = -10, synapse=None)
            #nengo.Connection(odor_neurons[0], odor_change[0], transform = 10, synapse=None)
            #nengo.Connection(odor_memory, odor_change[1], transform = -10, synapse=None)
            #nengo.Connection(odor_neurons[0], odor_change[1], transform = 10, synapse=None)

            #reward = nengo.Node(size_in = 1)
            #nengo.Connection(odor_memory, reward, transform = -10, synapse=0.05)
            #nengo.Connection(odor_neurons[0], reward, transform = 10, synapse=0.05)

            #odor2motor = nengo.Ensemble(n_neurons=100, dimensions=2, radius=2, seed=2, 
            # yy          noise=nengo.processes.WhiteSignal(10, 0.1, rms=1))
            #odor2motor = nengo.Ensemble(n_neurons=200, dimensions=3, radius=2)

            #nengo.Connection(odor_change, odor2motor[0],synapse=0.01)

            approach_neurons = nengo.Ensemble(n_neurons=200, dimensions=2)
            avoid_neurons = nengo.Ensemble(n_neurons=200, dimensions=2)
            search_neurons = nengo.Ensemble(n_neurons=200, dimensions=2)
            #nengo.Connection(odor_neurons[0], odor2motor[0],synapse=0.01)
            #nengo.Connection(odor_neurons[1], odor2motor[1],synapse=0.01)

            approach_node = nengo.Node(size_in=2, size_out = 2)
            avoid_node = nengo.Node(size_in=2, size_out = 2)
            search_node = nengo.Node(size_in=2, size_out = 2)

            # Arbitrary, intuitive functions are implemented as initialization for the various ensembles.
            # Input: x[0] for mean stimulus strength (higher when closer to the source)
            # Input: x[1] for left-right stimulus (positive when source to the left)
            # Turning is implemented by default to the right
            # When approaching we need high speed when low input and turning towards source
            
            #nengo.Connection(odor2motor, motor_neurons, function=braiten) 
            conn_approach = nengo.Connection(odor_neurons, approach_neurons, function=self.approach, 
                        learning_rule_type=nengo.PES(learning_rate=1e-4, pre_tau=0.1))
            conn_avoid = nengo.Connection(odor_neurons, avoid_neurons, function=self.avoid, 
                        learning_rule_type=nengo.PES(learning_rate=1e-4, pre_tau=0.1))
            conn_search = nengo.Connection(odor_neurons, search_neurons, function=self.search, 
                        learning_rule_type=nengo.PES(learning_rate=1e-4, pre_tau=0.1))

            nengo.Connection(errors.ensembles[0], conn_approach.learning_rule, synapse = 0.01)
            nengo.Connection(errors.ensembles[1], conn_avoid.learning_rule, synapse = 0.01)
            nengo.Connection(errors.ensembles[2], conn_search.learning_rule, synapse = 0.01)



            nengo.Connection(approach_neurons, approach_node) 
            nengo.Connection(avoid_neurons, avoid_node) 
            nengo.Connection(search_neurons, search_node) 
            #learning = nengo.Node(size_out = 2, output = [-1,-1])
            #nengo.Connection(learning, conn.learning_rule, synapse=None)  

#             mousebrain.p_approach = nengo.Probe(approach_node)
#             mousebrain.p_avoid = nengo.Probe(avoid_node)
#             mousebrain.p_search = nengo.Probe(search_node)

            self.p_approach = nengo.Probe(approach_node)
            self.p_avoid = nengo.Probe(avoid_node)
            self.p_search = nengo.Probe(search_node)
            self.p_odor = nengo.Probe(odor)
            self.p_state = nengo.Probe(state)
            self.p_change = nengo.Probe(odor_change)
            self.p_errors0 = nengo.Probe(errors.ensembles[0])
            self.p_errors1 = nengo.Probe(errors.ensembles[1])
            self.p_errors2 = nengo.Probe(errors.ensembles[2])
            
        #return mousebrain