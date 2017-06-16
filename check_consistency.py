
# coding: utf-8

# In[33]:

get_ipython().run_cell_magic('writefile', 'check_consistency_parallel.py', '\nimport math\nimport ipyparallel\nimport itertools\n\n# at terminal : ipcluster start -n 4\n\nclients = ipyparallel.Client()\ndview = clients.direct_view()\n\ndef initialize_pos_in_quadrant(num) :\n    pos =[(0,1), (0,2), (0,3), (0,4), (0,5), (1,1), (1,2), (1,3), (1,4), (1,5), (2,2), (2,3), (2,4), (2,5), (3,3), (3,4), (3,5), (4,4), (4,5), (5,5)]\n    temp = [0,1,2,3,4,5,6,7]\n    header = [i*math.pi/4 for i in temp]\n    params = (pos, header)\n    param_combs = list(itertools.product(*params))\n    num_combs = len(param_combs)\n    positions = [param_combs[i%num_combs] for i in range(num)]\n    return positions\n\nmouse_positions = initialize_pos_in_quadrant(160)\n# mouse_positions = [[(45,45), i/4*math.pi] for i in(range(8))]\n\nwith dview.sync_imports():\n    from mouseworld import mouseworld\n    from mouseworld.mouseworld import Mouseworld\n    import time\ndview.push({"Mouseworld": Mouseworld})\n\ndef run_experiment(mouse_position) :\n#    \n    num_mice = [0, 0, 100]\n\n    # Build the model\n    print(\'Building mouseworld\')\n    model = mouseworld.Mouseworld(num_mice, 1, 0, genome_range = [(0.6,0.6), (0,1), (0,1), (0.8,0.8), (0.4,0.4)], \n                     mouse_position = mouse_position, food_position = (0,0),\n                     primary_values = [100, 0], secondary_values = [100, 0], \n                     food_amount_range = (10000,10000), nutritional_value = [1], brain_iterations_per_step = 10)\n\n    # Gather initial randomly distributed data\n    # model.initial_datacollector.collect(model,model.schedule)\n    # initial_model_data = model.initial_datacollector.get_model_vars_dataframe()\n    # initial_model_data.to_csv(\'results/initial_model_data.csv\', sep=\'\\t\')\n\n    # Prepare environment by stepping food and predators and diffusing odors\n    print(\'Preparing environment\')\n    for i in range(40) :\n        model.food_schedule.step()\n        model.predator_schedule.step()\n        model.diffuse_odor_layers(model.odor_layers)\n\n    counter = 0\n    myrange = 60\n    # Run for discrete number of timesteps\n    print(\'Simulatimg for %i timesteps\'%myrange)\n    for i in range(myrange) :\n        c=time.time()\n        counter += 1\n        model.step()\n        d=time.time()\n        print(\'sim step : %i in %f\'%(counter, d-c))\n    print(\'Simulation terminated - Number of time steps reached\')\n\n    # Run until all mice perish\n    # while model.num_mice > 0 :\n    #     counter += 1\n    #     print(\'sim step : %i\'%counter)\n    #     model.step()\n\n    # Gather final model and agent data\n    model.final_datacollector.collect(model,model.all_mice_schedule)\n    # final_model_data = model.final_datacollector.get_model_vars_dataframe()\n    # print(final_model_data)\n\n    final_agent_data = model.final_datacollector.get_agent_vars_dataframe()\n    mouse_statistics = final_agent_data[[\'first_action_duration\', \'first_action_termination\']]\n    # mouse_statistics = final_agent_data[\'action_history\']\n    # mouse_statistics = final_agent_data[\'secondary_values\']\n    # mouse_statistics = final_agent_data[\'sensor_vector\']\n\n    mouse_statistics = mouse_statistics.reset_index(\'Step\', drop = True)\n        #mouse_statistics = mouse_statistics.reset_index(\'AgentID\', drop = True)\n    num_trials = len(mouse_statistics.index)\n    # for i in range(num_trials)\n    succesful_trials = mouse_statistics.loc[(mouse_statistics[\'first_action_termination\'] == \'Closure\')]\n    unsuccesful_trials = mouse_statistics.loc[(mouse_statistics[\'first_action_termination\'] == \'Failure\')]\n    # incomplete_trials = mouse_statistics.loc[(mouse_statistics[\'first_action_termination\'] == \'None\')]\n\n    num_succesful_trials = len(succesful_trials.index)\n    num_unsuccesful_trials = len(unsuccesful_trials.index)\n    # num_incomplete_trials = len(incomplete_trials.index)\n\n    #performance = num_succesful_trials / num_trials\n    mean_time_success = succesful_trials[\'first_action_duration\'].mean()\n    mean_time_failure = unsuccesful_trials[\'first_action_duration\'].mean()\n\n    sim_params = [mouse_position, num_mice, myrange]\n    results = [num_succesful_trials, mean_time_success, num_unsuccesful_trials, mean_time_failure]\n    exp_data = [results, sim_params]\n    return exp_data\n    # mouse_statistics = [performance,  mean_time]\n    #     sensor_vector = final_agent_data[\'sensor_vector\'][0].values[0]\n#     sensor_position = final_agent_data[\'sensor_position\'][0].values[0]\n#     motor_vector = final_agent_data[\'motor_vector\'][0].values[0]\n#     first_action = final_agent_data[\'action_history\'][0].values[0].loc[0]\n#     first_action = mousetest_data\n#     first_action = (mousetest_data[\'Duration\'], mousetest_data[\'Termination\'])\n    #first_action = final_agent_data[\'action_history\'][0].values[0].loc[0]\n#     return (performance,  mean_time)\n#     return (mouse_statistics)\n# print(performance,  mean_time)\nall_exp_data = dview.map_sync(run_experiment, mouse_positions)\n\nfile = open(\'results/check_approach_consistency_001_2.txt\',\'w\') \nfor exp_data in all_exp_data :\n    file.write(str(exp_data) + \'\\n\')\n# file.write(str(results) + \'\\n\')\nfile.close() \n# mouse_statistics.to_csv(\'results/mouse_statistics.csv\', sep=\'\\t\')')


# In[1]:

#%%writefile check_consistency.py

from mouseworld import mouseworld
import time
import math

mouse_position = [(45,45), 1/4*math.pi]
num_mice = [0, 0, 10]

# Build the model
print('Building mouseworld')
model = mouseworld.Mouseworld(num_mice, 1, 0, genome_range = [(0.6,0.6), (0,1), (0,1), (0.8,0.8), (0.4,0.4)], 
                 mouse_position = mouse_position, food_position = (50,50),
                 primary_values = [100, 0], secondary_values = [100, 0], 
                 food_amount_range = (10000,10000), nutritional_value = [1], brain_iterations_per_step = 10)

# Gather initial randomly distributed data
# model.initial_datacollector.collect(model,model.schedule)
# initial_model_data = model.initial_datacollector.get_model_vars_dataframe()
# initial_model_data.to_csv('results/initial_model_data.csv', sep='\t')

# Prepare environment by stepping food and predators and diffusing odors
print('Preparing environment')
for i in range(4) :
    model.food_schedule.step()
    model.predator_schedule.step()
    model.diffuse_odor_layers(model.odor_layers)

counter = 0
myrange = 2
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
# while model.num_mice > 0 :
#     counter += 1
#     print('sim step : %i'%counter)
#     model.step()
    
# Gather final model and agent data
model.final_datacollector.collect(model,model.all_mice_schedule)
# final_model_data = model.final_datacollector.get_model_vars_dataframe()
# print(final_model_data)

final_agent_data = model.final_datacollector.get_agent_vars_dataframe()
mouse_statistics = final_agent_data[['first_action_duration', 'first_action_termination']]
# mouse_statistics = final_agent_data['action_history']
# mouse_statistics = final_agent_data['secondary_values']
# mouse_statistics = final_agent_data['sensor_vector']

mouse_statistics = mouse_statistics.reset_index('Step', drop = True)
    #mouse_statistics = mouse_statistics.reset_index('AgentID', drop = True)
num_trials = len(mouse_statistics.index)
# for i in range(num_trials)
succesful_trials = mouse_statistics.loc[(mouse_statistics['first_action_termination'] == 'Closure')]
unsuccesful_trials = mouse_statistics.loc[(mouse_statistics['first_action_termination'] == 'Failure')]
# incomplete_trials = mouse_statistics.loc[(mouse_statistics['first_action_termination'] == 'None')]

num_succesful_trials = len(succesful_trials.index)
num_unsuccesful_trials = len(unsuccesful_trials.index)
# num_incomplete_trials = len(incomplete_trials.index)

#performance = num_succesful_trials / num_trials
mean_time_success = succesful_trials['first_action_duration'].mean()
mean_time_failure = unsuccesful_trials['first_action_duration'].mean()

sim_params = [mouse_position, num_mice, myrange]
results = [num_succesful_trials, mean_time_success, num_unsuccesful_trials, mean_time_failure]
# mouse_statistics = [performance,  mean_time]
#     sensor_vector = final_agent_data['sensor_vector'][0].values[0]
#     sensor_position = final_agent_data['sensor_position'][0].values[0]
#     motor_vector = final_agent_data['motor_vector'][0].values[0]
#     first_action = final_agent_data['action_history'][0].values[0].loc[0]
#     first_action = mousetest_data
#     first_action = (mousetest_data['Duration'], mousetest_data['Termination'])
    #first_action = final_agent_data['action_history'][0].values[0].loc[0]
#     return (performance,  mean_time)
#     return (mouse_statistics)
# print(performance,  mean_time)
file = open('results/check_consistency.txt','w') 
file.write(str(sim_params) + '\t')
file.write(str(results) + '\n')
file.close() 
# mouse_statistics.to_csv('results/mouse_statistics.csv', sep='\t')


# In[ ]:



