
import math
import ipyparallel
import itertools
import sys
# at terminal : ipcluster start -n 4

clients = ipyparallel.Client()
dview = clients.direct_view()

with dview.sync_imports():
    from mouseworld import mouseworld
    from mouseworld.mouseworld import Mouseworld
    import time
    import sys
dview.push({"Mouseworld": Mouseworld})

def run_experiment(exp_data) :
    simulation_num = sys.argv[1]
    num_mice = [10, 10, 10]

    # Build the model
    print('Building mouseworld')
    model = mouseworld.Mouseworld(num_mice, 100, 40, simulation_num, mouse_initial_energy = 1000, mouse_max_energy = 1200,
                     food_amount_range = (20,200), nutritional_value = [1], mouse_reproduction = True, 
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
    myrange = 20
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
#     print('Simulatimg until all mice perish')
#     while model.num_mice > 0 :
#         c=time.time()
#         counter += 1
#         model.step()
#         d=time.time()
#         print('sim step : %i in %f'%(counter, d-c))
#     print('Simulation terminated - No alive mice')
        
    print('Storing mousebrains')
# print(model.exp_approach_rank)
# print(model.exp_approach_rank[:1])
    for i in model.exp_approach_rank[:30] :
        i.store_mousebrain_weights()
    
#     exp_data = 5
    return exp_data
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
print('Starting parallel execution')
e=time.time()
all_exp_data = dview.map_sync(run_experiment, [5])
f=time.time()
print('parallel simulation complete in : %f'%(f-e))