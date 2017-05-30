from mouseworld import mouseworld
import time
import matplotlib.pyplot as plt
import numpy as np

num_mice = 200

# Build the model
model = mouseworld.Mouseworld(num_mice, 5, 100, 50, 100, 100)


# Prepare environment by stepping food and predators and diffusing odors
# for i in range(100) :
#     model.food_schedule.step()
#     model.predator_schedule.step()
#     model.diffuse_odor_layers_parallel(model.odor_layers)
a=time.time()
for i in range(10) :
    model.food_schedule.step()
    model.predator_schedule.step()
    model.diffuse_odor_layers(model.odor_layers)
#Run for discrete number of timesteps
b=time.time()
print(b-a)
counter = 0
myrange = 1000
# for i in range(myrange) :
#     c=time.time()
#     counter += 1
#     model.step()
#     d=time.time()
#     print('sim step : %i in %f'%(counter, d-c))
#t = np.arange(1, myrange*10 +1, 1)
#t = np.arange(1, (myrange-2)*10 +1, 1)
#Run until all mice perish
while model.num_mice > 0 :
    c=time.time()
    counter += 1
    model.step()
    d=time.time()
    print('sim step : %i in %f'%(counter, d-c))
# Gather final model and agent data
#model.mousebrain_datacollector.collect(model,model.schedule)
#mousebrain_data = model.mousebrain_datacollector.get_agent_vars_dataframe()
#mousebrain_data.to_csv('results/mousebrain_data.csv', sep='\t')
model.final_datacollector.collect(model,model.all_mice_schedule)
final_model_data = model.final_datacollector.get_model_vars_dataframe()
#final_model_data.to_csv('results/final_model_data.csv', sep='\t')
final_agent_data = model.final_datacollector.get_agent_vars_dataframe()

print(final_model_data)

for i in range(len(final_agent_data)) :
    print('Name : %s'%final_agent_data.index[i][1])
    print('Age : %i'%final_agent_data['age'][0].values[i])
    print('Generation : %i'%final_agent_data['generation'][0].values[i])
    print('Offspring : %i'%final_agent_data['num_offspring'][0].values[i])
    print('Energy : %f'%final_agent_data['energy'][0].values[i])
    print(final_agent_data['action_history'][0].values[i])
    #print(final_agent_data['possible_actions'][0].values[i])
    print(final_agent_data['primary_values'][0].values[i])
    print(final_agent_data['secondary_values'][0].values[i])
    print(final_agent_data['sensor_vector'][0].values[i])
    print(final_agent_data['sensor_position'][0].values[i])