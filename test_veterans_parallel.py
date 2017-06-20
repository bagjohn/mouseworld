
import math
import ipyparallel
import itertools
import os
import numpy as np
# at terminal : ipcluster start -n 4

clients = ipyparallel.Client()
dview = clients.direct_view()

print('Loading modules to engines') 
with dview.sync_imports():
    from mouseworld import mouseworld
    from mouseworld.mouseworld import Mouseworld
    import time
    import sys
    import numpy
dview.push({"Mouseworld": Mouseworld})
print('Defining filenames')

directory = ('results/simulation_%s/veteran_mousebrains'%sys.argv[1])

def run_experiment(file) :

    mouse_data = numpy.load(file)

    if mouse_data['motor_NN_on'] :
        mousebrain_weights = [mouse_data['w_search'], mouse_data['w_approach'], mouse_data['w_avoid']]
        if mouse_data['learning_on'] :
            num_mice = [0, 0, 152]
        else :
            num_mice = [0, 152, 0]
    else :
        mousebrain_weights = None
        num_mice = [152, 0, 0]
    genome = numpy.around(mouse_data['genome'], decimals = 2)
    genome_range = [(x,x) for x in genome]
    mousebrain_seed = mouse_data['seed']
    brain_iterations_per_step = mouse_data['brain_iterations_per_step']
    mousebrain_steps = mouse_data['mousebrain_steps']
    
    simulation_num = mouse_data['simulation_num']

    # Build the model
    print('Building mouseworld')
    model = mouseworld.Mouseworld(num_mice, 1, 0, simulation_num, genome_range = genome_range, 
                                  mouse_position = 'in_quadrant', food_position = (0,0), 
                                  primary_values = [100, 0], secondary_values = [100,0], food_amount_range = (10000,10000), 
                                  nutritional_value = [1], mouse_reproduction = False, 
                                  brain_iterations_per_step = brain_iterations_per_step, mousebrain_seed = mousebrain_seed,
                                 initial_mousebrain_weights = mousebrain_weights, test_veteran = True)

    # Gather initial randomly distributed data
    # model.initial_datacollector.collect(model,model.schedule)
    # initial_model_data = model.initial_datacollector.get_model_vars_dataframe()
    # initial_model_data.to_csv('results/initial_model_data.csv', sep='\t')

    # Prepare environment by stepping food and predators and diffusing odors
    print('Preparing environment')
    for i in range(40) :
        model.food_schedule.step()
        model.predator_schedule.step()
        model.diffuse_odor_layers(model.odor_layers)

    counter = 0
    myrange = 40
    # Run for discrete number of timesteps
    print('Simulatimg for %i timesteps'%myrange)
    for i in range(myrange) :
        c=time.time()
        counter += 1
        model.step()
        d=time.time()
        print('sim step : %i in %f'%(counter, d-c))
    print('Simulation terminated - Number of time steps reached')


    # Run until all mice perish
    # print('Simulatimg until all mice perish')
    # while model.num_mice > 0 :
    #     c=time.time()
    #     counter += 1
    #     model.step()
    #     d=time.time()
    #     print('sim step : %i in %f'%(counter, d-c))
    # print('Simulation terminated - No alive mice')

    # Gather final model and agent data
    print('Gathering simulation data')
    model.final_datacollector.collect(model,model.all_mice_schedule)

    final_agent_data = model.final_datacollector.get_agent_vars_dataframe()
    mouse_statistics = final_agent_data[['control_population', 'first_action_duration', 'first_action_termination']]

    mouse_statistics = mouse_statistics.reset_index('Step', drop = True)
    print('Counting successful trials')
    veteran_stats = mouse_statistics.loc[mouse_statistics['control_population'] == False]
    control_stats = mouse_statistics.loc[mouse_statistics['control_population'] == True]
    num_trials = len(veteran_stats.index)
    succesful_veteran_trials = veteran_stats.loc[(veteran_stats['first_action_termination'] == 'Closure')]
    succesful_control_trials = control_stats.loc[(control_stats['first_action_termination'] == 'Closure')]
    unsuccesful_veteran_trials = veteran_stats.loc[(veteran_stats['first_action_termination'] == 'Failure')]
    unsuccesful_control_trials = control_stats.loc[(control_stats['first_action_termination'] == 'Failure')]

    num_succesful_veteran_trials = len(succesful_veteran_trials.index)
    num_succesful_control_trials = len(succesful_control_trials.index)                                     
    num_unsuccesful_veteran_trials = len(unsuccesful_veteran_trials.index)
    num_unsuccesful_control_trials = len(unsuccesful_control_trials.index)

    mean_time_veteran_success = succesful_veteran_trials['first_action_duration'].mean()
    mean_time_control_success = succesful_control_trials['first_action_duration'].mean()                 
    mean_time_veteran_failure = unsuccesful_veteran_trials['first_action_duration'].mean()
    mean_time_control_failure = unsuccesful_control_trials['first_action_duration'].mean()

    veteran_results = [num_succesful_veteran_trials, mean_time_veteran_success, num_unsuccesful_veteran_trials, mean_time_veteran_failure]
    control_results = [num_succesful_control_trials, mean_time_control_success, num_unsuccesful_control_trials, mean_time_control_failure]

    exp_data = [mousebrain_steps, mousebrain_seed, brain_iterations_per_step, num_mice, veteran_results, control_results]
    return exp_data

filenames = []
for file in os.listdir(directory):
    if file.endswith(".npz"):
        filenames.append('%s/%s'%(directory,file))
        
print('Initializing parallel execution')      
all_exp_data = dview.map_sync(run_experiment, filenames)

print('Writing results to file')
file = open('%s/test_veterans.txt'%directory,'w') 
counter = 0
for exp_data in all_exp_data :
    counter +=1
    mousebrain_steps = exp_data[0]
    mousebrain_seed = exp_data[1]
    brain_iterations_per_step = exp_data[2] 
    num_mice = exp_data[3]
    veteran_results = exp_data[4]
    control_results = exp_data[5]
    file.write('-----Veteran Mouse : %i-----\n'%counter)
    file.write('num_mice : %s' %str(num_mice) + '\t')
    file.write('experience : %s'%str(mousebrain_steps) + '\t')
    file.write('mousebrain seed : %s' %str(mousebrain_seed) + '\t')
    file.write('brain iterations per step : %s' %str(brain_iterations_per_step) + '\n')
    file.write('veteran results' + '\t')
    file.write(str(veteran_results) + '\n')
    file.write('control results' + '\t')
    file.write(str(control_results) + '\n')

file.close() 