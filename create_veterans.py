#ooo


from mouseworld import mouseworld
import time
import sys

simulation_num = sys.argv[1]
# simulation_num = 999


num_mice = [10, 10, 10]

 # Build the model
print('Building mouseworld')
model = mouseworld.Mouseworld(num_mice, 100, 0, simulation_num, mouse_initial_energy = 10000, mouse_max_energy = 12000,
                 food_amount_range = (20,40), nutritional_value = [1], mouse_reproduction = False, 
                              brain_iterations_per_step = 1, mousebrain_seed = 8)

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
for i in model.exp_approach_rank[:30] :
    i.store_mousebrain_weights()
    
# # Gather final model and agent data
# model.final_datacollector.collect(model,model.all_mice_schedule)
# # final_model_data = model.final_datacollector.get_model_vars_dataframe()
# # print(final_model_data)

# final_agent_data = model.final_datacollector.get_agent_vars_dataframe()
# mouse_statistics = final_agent_data[['first_action_duration', 'first_action_termination']]