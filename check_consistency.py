
from mouseworld import mouseworld
import time
import math

mouse_position = [(45,45), 1/4*math.pi]
num_mice = [0, 0, 1]

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
for i in range(40) :
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