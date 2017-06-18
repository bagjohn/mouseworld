
# coding: utf-8

# In[ ]:

get_ipython().run_cell_magic('writefile', 'create_veterans_parallel.py', '\nimport math\nimport ipyparallel\nimport itertools\n\n# at terminal : ipcluster start -n 4\n\nclients = ipyparallel.Client()\ndview = clients.direct_view()\n\nwith dview.sync_imports():\n    from mouseworld import mouseworld\n    from mouseworld.mouseworld import Mouseworld\n    import time\ndview.push({"Mouseworld": Mouseworld})\n\ndef run_experiment() :\n#    \n    num_mice = [0, 0, 100]\n\n    # Build the model\n    print(\'Building mouseworld\')\n    model = mouseworld.Mouseworld(num_mice, 100, 0, mouse_initial_energy = 1000, mouse_max_energy = 1200,\n                     food_amount_range = (20,40), nutritional_value = [1], mouse_reproduction = False, brain_iterations_per_step = 10)\n\n    # Gather initial randomly distributed data\n    # model.initial_datacollector.collect(model,model.schedule)\n    # initial_model_data = model.initial_datacollector.get_model_vars_dataframe()\n    # initial_model_data.to_csv(\'results/initial_model_data.csv\', sep=\'\\t\')\n\n    # Prepare environment by stepping food and predators and diffusing odors\n    print(\'Preparing environment\')\n    for i in range(40) :\n        model.food_schedule.step()\n        model.predator_schedule.step()\n        model.diffuse_odor_layers(model.odor_layers)\n\n    counter = 0\n    myrange = 60\n    # Run for discrete number of timesteps\n#     print(\'Simulatimg for %i timesteps\'%myrange)\n#     for i in range(myrange) :\n#         c=time.time()\n#         counter += 1\n#         model.step()\n#         d=time.time()\n#         print(\'sim step : %i in %f\'%(counter, d-c))\n#     print(\'Simulation terminated - Number of time steps reached\')\n\n    # Run until all mice perish\n    print(\'Simulatimg until all mice perish\')\n    while model.num_mice > 0 :\n        c=time.time()\n        counter += 1\n        model.step()\n        d=time.time()\n        print(\'sim step : %i in %f\'%(counter, d-c))\n    print(\'Simulation terminated - No alive mice\')\n        \n    # Gather final model and agent data\n    model.final_datacollector.collect(model,model.all_mice_schedule)\n    # final_model_data = model.final_datacollector.get_model_vars_dataframe()\n    # print(final_model_data)\n\n    final_agent_data = model.final_datacollector.get_agent_vars_dataframe()\n    mouse_statistics = final_agent_data[[\'first_action_duration\', \'first_action_termination\']]\n    # mouse_statistics = final_agent_data[\'action_history\']\n    # mouse_statistics = final_agent_data[\'secondary_values\']\n    # mouse_statistics = final_agent_data[\'sensor_vector\']\n\n    mouse_statistics = mouse_statistics.reset_index(\'Step\', drop = True)\n        #mouse_statistics = mouse_statistics.reset_index(\'AgentID\', drop = True)\n    num_trials = len(mouse_statistics.index)\n    # for i in range(num_trials)\n    succesful_trials = mouse_statistics.loc[(mouse_statistics[\'first_action_termination\'] == \'Closure\')]\n    unsuccesful_trials = mouse_statistics.loc[(mouse_statistics[\'first_action_termination\'] == \'Failure\')]\n    # incomplete_trials = mouse_statistics.loc[(mouse_statistics[\'first_action_termination\'] == \'None\')]\n\n    num_succesful_trials = len(succesful_trials.index)\n    num_unsuccesful_trials = len(unsuccesful_trials.index)\n    # num_incomplete_trials = len(incomplete_trials.index)\n\n    #performance = num_succesful_trials / num_trials\n    mean_time_success = succesful_trials[\'first_action_duration\'].mean()\n    mean_time_failure = unsuccesful_trials[\'first_action_duration\'].mean()\n\n    sim_params = [mouse_position, num_mice, myrange]\n    results = [num_succesful_trials, mean_time_success, num_unsuccesful_trials, mean_time_failure]\n    exp_data = [results, sim_params]\n    return exp_data\n    # mouse_statistics = [performance,  mean_time]\n    #     sensor_vector = final_agent_data[\'sensor_vector\'][0].values[0]\n#     sensor_position = final_agent_data[\'sensor_position\'][0].values[0]\n#     motor_vector = final_agent_data[\'motor_vector\'][0].values[0]\n#     first_action = final_agent_data[\'action_history\'][0].values[0].loc[0]\n#     first_action = mousetest_data\n#     first_action = (mousetest_data[\'Duration\'], mousetest_data[\'Termination\'])\n    #first_action = final_agent_data[\'action_history\'][0].values[0].loc[0]\n#     return (performance,  mean_time)\n#     return (mouse_statistics)\n# print(performance,  mean_time)\nall_exp_data = dview.map_sync(run_experiment, mouse_positions)\n\nfile = open(\'results/check_approach_consistency_001_2.txt\',\'w\') \nfor exp_data in all_exp_data :\n    file.write(str(exp_data) + \'\\n\')\n# file.write(str(results) + \'\\n\')\nfile.close() \n# mouse_statistics.to_csv(\'results/mouse_statistics.csv\', sep=\'\\t\')')


# In[1]:

#%%writefile create_veterans.py

from mouseworld import mouseworld
import time
import sys

simulation_num = sys.argv[1]
# simulation_num = 999


num_mice = [0, 0, 10]

 # Build the model
print('Building mouseworld')
model = mouseworld.Mouseworld(num_mice, 100, 0, simulation_num, mouse_initial_energy = 10000, mouse_max_energy = 12000,
                 food_amount_range = (20,40), nutritional_value = [1], mouse_reproduction = False, 
                              brain_iterations_per_step = 10, mousebrain_seed = 8)

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
myrange = 2
# Run for discrete number of timesteps
# print('Simulatimg for %i timesteps'%myrange)
# for i in range(myrange) :
#     c=time.time()
#     counter += 1
#     model.step()
#     d=time.time()
#     print('sim step : %i in %f'%(counter, d-c))
# print('Simulation terminated - Number of time steps reached')


# Run until all mice perish
print('Simulatimg until all mice perish')
while model.num_mice > 0 :
    c=time.time()
    counter += 1
    model.step()
    d=time.time()
    print('sim step : %i in %f'%(counter, d-c))
print('Simulation terminated - No alive mice')

print('Storing mousebrains')
# print(model.exp_approach_rank)
# print(model.exp_approach_rank[:1])
for i in model.exp_approach_rank[:5] :
    i.store_mousebrain_weights()
    
# # Gather final model and agent data
# model.final_datacollector.collect(model,model.all_mice_schedule)
# # final_model_data = model.final_datacollector.get_model_vars_dataframe()
# # print(final_model_data)

# final_agent_data = model.final_datacollector.get_agent_vars_dataframe()
# mouse_statistics = final_agent_data[['first_action_duration', 'first_action_termination']]


# In[2]:

get_ipython().run_cell_magic('writefile', 'test_veterans_parallel.py', '\nimport math\nimport ipyparallel\nimport itertools\nimport os\nimport numpy as np\n# at terminal : ipcluster start -n 4\n\nclients = ipyparallel.Client()\ndview = clients.direct_view()\n\nprint(\'Loading modules to engines\') \nwith dview.sync_imports():\n    from mouseworld import mouseworld\n    from mouseworld.mouseworld import Mouseworld\n    import time\n    import sys\n    import numpy\ndview.push({"Mouseworld": Mouseworld})\nprint(\'Defining filenames\')\n\ndirectory = (\'results/simulation_%s/veteran_mousebrains\'%sys.argv[1])\n\ndef run_experiment(file) :\n\n    mouse_data = numpy.load(file)\n\n    if mouse_data[\'motor_NN_on\'] :\n        if mouse_data[\'learning_on\'] :\n            num_mice = [0, 0, 152]\n        else :\n            num_mice = [0, 152, 0]\n    else :\n        num_mice = [152, 0, 0]\n    genome = numpy.around(mouse_data[\'genome\'], decimals = 2)\n    genome_range = [(x,x) for x in genome]\n    mousebrain_seed = mouse_data[\'seed\']\n    brain_iterations_per_step = mouse_data[\'brain_iterations_per_step\']\n    mousebrain_steps = mouse_data[\'mousebrain_steps\']\n    mousebrain_weights = [mouse_data[\'w_search\'], mouse_data[\'w_approach\'], mouse_data[\'w_avoid\']]\n    simulation_num = mouse_data[\'simulation_num\']\n\n    # Build the model\n    print(\'Building mouseworld\')\n    model = mouseworld.Mouseworld(num_mice, 1, 0, simulation_num, genome_range = genome_range, \n                                  mouse_position = \'in_quadrant\', food_position = (0,0), \n                                  primary_values = [100, 0], secondary_values = [100,0], food_amount_range = (10000,10000), \n                                  nutritional_value = [1], mouse_reproduction = False, \n                                  brain_iterations_per_step = brain_iterations_per_step, mousebrain_seed = mousebrain_seed,\n                                 initial_mousebrain_weights = mousebrain_weights, test_veteran = True)\n\n    # Gather initial randomly distributed data\n    # model.initial_datacollector.collect(model,model.schedule)\n    # initial_model_data = model.initial_datacollector.get_model_vars_dataframe()\n    # initial_model_data.to_csv(\'results/initial_model_data.csv\', sep=\'\\t\')\n\n    # Prepare environment by stepping food and predators and diffusing odors\n    print(\'Preparing environment\')\n    for i in range(40) :\n        model.food_schedule.step()\n        model.predator_schedule.step()\n        model.diffuse_odor_layers(model.odor_layers)\n\n    counter = 0\n    myrange = 40\n    # Run for discrete number of timesteps\n    print(\'Simulatimg for %i timesteps\'%myrange)\n    for i in range(myrange) :\n        c=time.time()\n        counter += 1\n        model.step()\n        d=time.time()\n        print(\'sim step : %i in %f\'%(counter, d-c))\n    print(\'Simulation terminated - Number of time steps reached\')\n\n\n    # Run until all mice perish\n    # print(\'Simulatimg until all mice perish\')\n    # while model.num_mice > 0 :\n    #     c=time.time()\n    #     counter += 1\n    #     model.step()\n    #     d=time.time()\n    #     print(\'sim step : %i in %f\'%(counter, d-c))\n    # print(\'Simulation terminated - No alive mice\')\n\n    # Gather final model and agent data\n    print(\'Gathering simulation data\')\n    model.final_datacollector.collect(model,model.all_mice_schedule)\n\n    final_agent_data = model.final_datacollector.get_agent_vars_dataframe()\n    mouse_statistics = final_agent_data[[\'control_population\', \'first_action_duration\', \'first_action_termination\']]\n\n    mouse_statistics = mouse_statistics.reset_index(\'Step\', drop = True)\n    print(\'Counting successful trials\')\n    veteran_stats = mouse_statistics.loc[mouse_statistics[\'control_population\'] == False]\n    control_stats = mouse_statistics.loc[mouse_statistics[\'control_population\'] == True]\n    num_trials = len(veteran_stats.index)\n    succesful_veteran_trials = veteran_stats.loc[(veteran_stats[\'first_action_termination\'] == \'Closure\')]\n    succesful_control_trials = control_stats.loc[(control_stats[\'first_action_termination\'] == \'Closure\')]\n    unsuccesful_veteran_trials = veteran_stats.loc[(veteran_stats[\'first_action_termination\'] == \'Failure\')]\n    unsuccesful_control_trials = control_stats.loc[(control_stats[\'first_action_termination\'] == \'Failure\')]\n\n    num_succesful_veteran_trials = len(succesful_veteran_trials.index)\n    num_succesful_control_trials = len(succesful_control_trials.index)                                     \n    num_unsuccesful_veteran_trials = len(unsuccesful_veteran_trials.index)\n    num_unsuccesful_control_trials = len(unsuccesful_control_trials.index)\n\n    mean_time_veteran_success = succesful_veteran_trials[\'first_action_duration\'].mean()\n    mean_time_control_success = succesful_control_trials[\'first_action_duration\'].mean()                 \n    mean_time_veteran_failure = unsuccesful_veteran_trials[\'first_action_duration\'].mean()\n    mean_time_control_failure = unsuccesful_control_trials[\'first_action_duration\'].mean()\n\n    veteran_results = [num_succesful_veteran_trials, mean_time_veteran_success, num_unsuccesful_veteran_trials, mean_time_veteran_failure]\n    control_results = [num_succesful_control_trials, mean_time_control_success, num_unsuccesful_control_trials, mean_time_control_failure]\n\n    exp_data = [mousebrain_steps, mousebrain_seed, veteran_results, control_results]\n    return exp_data\n\nfilenames = []\nfor file in os.listdir(directory):\n    if file.endswith(".npz"):\n        filenames.append(\'%s/%s\'%(directory,file))\n        \nprint(\'Initializing parallel execution\')      \nall_exp_data = dview.map_sync(run_experiment, filenames)\n\nprint(\'Writing results to file\')\nfile = open(\'%s/test_veterans.txt\'%directory,\'w\') \ncounter = 0\nfor exp_data in all_exp_data :\n    counter +=1\n    mousebrain_steps = exp_data[0]\n    mousebrain_seed = exp_data[1]\n    veteran_results = exp_data[2]\n    control_results = exp_data[3]\n    file.write(\'-----Veteran Mouse : %i-----\\n\'%counter)\n    file.write(\'experience : %s\'%str(mousebrain_steps) + \'\\t\')\n    file.write(\'mousebrain seed : %s\' %str(mousebrain_seed) + \'\\n\')\n    file.write(\'veteran results\' + \'\\t\')\n    file.write(str(veteran_results) + \'\\n\')\n    file.write(\'control results\' + \'\\t\')\n    file.write(str(control_results) + \'\\n\')\n\nfile.close() ')


# In[11]:

get_ipython().run_cell_magic('writefile', 'test_veterans.py', "\nfrom mouseworld import mouseworld\nimport time\nimport sys\nimport numpy as np\n\nmouse_data = np.load(sys.argv[1])\n\nif mouse_data['motor_NN_on'] :\n    if mouse_data['learning_on'] :\n        num_mice = [0, 0, 152]\n    else :\n        num_mice = [0, 152, 0]\nelse :\n    num_mice = [152, 0, 0]\ngenome = np.around(mouse_data['genome'], decimals = 2)\ngenome_range = [(x,x) for x in genome]\nmousebrain_seed = mouse_data['seed']\nbrain_iterations_per_step = mouse_data['brain_iterations_per_step']\nmousebrain_steps = mouse_data['mousebrain_steps']\nmousebrain_weights = [mouse_data['w_search'], mouse_data['w_approach'], mouse_data['w_avoid']]\n                 \n# Build the model\nprint('Building mouseworld')\nmodel = mouseworld.Mouseworld(num_mice, 1, 0, genome_range = genome_range, \n                              mouse_position = 'in_quadrant', food_position = (0,0), \n                              primary_values = [100, 0], secondary_values = [100,0], food_amount_range = (10000,10000), \n                              nutritional_value = [1], mouse_reproduction = False, \n                              brain_iterations_per_step = brain_iterations_per_step, mousebrain_seed = mousebrain_seed,\n                             initial_mousebrain_weights = mousebrain_weights, test_veteran = True)\n\n# Gather initial randomly distributed data\n# model.initial_datacollector.collect(model,model.schedule)\n# initial_model_data = model.initial_datacollector.get_model_vars_dataframe()\n# initial_model_data.to_csv('results/initial_model_data.csv', sep='\\t')\n\n# Prepare environment by stepping food and predators and diffusing odors\nprint('Preparing environment')\nfor i in range(40) :\n    model.food_schedule.step()\n    model.predator_schedule.step()\n    model.diffuse_odor_layers(model.odor_layers)\n\ncounter = 0\nmyrange = 40\n# Run for discrete number of timesteps\nprint('Simulatimg for %i timesteps'%myrange)\nfor i in range(myrange) :\n    c=time.time()\n    counter += 1\n    model.step()\n    d=time.time()\n    print('sim step : %i in %f'%(counter, d-c))\nprint('Simulation terminated - Number of time steps reached')\n\n\n# Run until all mice perish\n# print('Simulatimg until all mice perish')\n# while model.num_mice > 0 :\n#     c=time.time()\n#     counter += 1\n#     model.step()\n#     d=time.time()\n#     print('sim step : %i in %f'%(counter, d-c))\n# print('Simulation terminated - No alive mice')\n    \n# Gather final model and agent data\nprint('Gathering simulation data')\nmodel.final_datacollector.collect(model,model.all_mice_schedule)\n\nfinal_agent_data = model.final_datacollector.get_agent_vars_dataframe()\nmouse_statistics = final_agent_data[['control_population', 'first_action_duration', 'first_action_termination']]\n\nmouse_statistics = mouse_statistics.reset_index('Step', drop = True)\nprint('Counting successful trials')\nveteran_stats = mouse_statistics.loc[mouse_statistics['control_population'] == False]\ncontrol_stats = mouse_statistics.loc[mouse_statistics['control_population'] == True]\nnum_trials = len(veteran_stats.index)\nsuccesful_veteran_trials = veteran_stats.loc[(veteran_stats['first_action_termination'] == 'Closure')]\nsuccesful_control_trials = control_stats.loc[(control_stats['first_action_termination'] == 'Closure')]\nunsuccesful_veteran_trials = veteran_stats.loc[(veteran_stats['first_action_termination'] == 'Failure')]\nunsuccesful_control_trials = control_stats.loc[(control_stats['first_action_termination'] == 'Failure')]\n\nnum_succesful_veteran_trials = len(succesful_veteran_trials.index)\nnum_succesful_control_trials = len(succesful_control_trials.index)                                     \nnum_unsuccesful_veteran_trials = len(unsuccesful_veteran_trials.index)\nnum_unsuccesful_control_trials = len(unsuccesful_control_trials.index)\n\nmean_time_veteran_success = succesful_veteran_trials['first_action_duration'].mean()\nmean_time_control_success = succesful_control_trials['first_action_duration'].mean()                 \nmean_time_veteran_failure = unsuccesful_veteran_trials['first_action_duration'].mean()\nmean_time_control_failure = unsuccesful_control_trials['first_action_duration'].mean()\n\nveteran_results = [num_succesful_veteran_trials, mean_time_veteran_success, num_unsuccesful_veteran_trials, mean_time_veteran_failure]\ncontrol_results = [num_succesful_control_trials, mean_time_control_success, num_unsuccesful_control_trials, mean_time_control_failure]\n\nfile = open('results/veteran_mousebrains/test_veterans.txt','w') \nfile.write('experience : %s'%str(mousebrain_steps) + '\\t')\nfile.write('mousebrain seed : %s' %str(mousebrain_seed) + '\\n')\nfile.write('veteran results' + '\\t')\nfile.write(str(veteran_results) + '\\n')\nfile.write('control results' + '\\t')\nfile.write(str(control_results) + '\\n')\n\nfile.close() ")


# In[ ]:



