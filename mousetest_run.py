
from mouseworld import mousetest
import time


   
x = 1
y = 1
header = 1 # 0-7
antenna_length = 0.5
antenna_angle = 0.5
# Build the model
model = mousetest.Mousetest([0, 0, 1], [0.5, 0.5, 0.5, antenna_length, antenna_angle], (x,y), header, 10, 1, 0, 100, 100)

# Gather initial randomly distributed data
# model.initial_datacollector.collect(model,model.schedule)
# initial_model_data = model.initial_datacollector.get_model_vars_dataframe()
# initial_model_data.to_csv('results/initial_model_data.csv', sep='\t')

# Prepare environment by stepping food and predators and diffusing odors
a=time.time()
for i in range(10) :
    model.food_schedule.step()
    model.predator_schedule.step()
    model.diffuse_odor_layers(model.odor_layers)
b=time.time()
print(b-a)

#Run for discrete number of timesteps
counter = 0
myrange = 1
for i in range(myrange) :
    c=time.time()
    counter += 1
    model.step()
    d=time.time()
    #print('sim step : %i in %f'%(counter, d-c))

# Run until all mice perish
# while model.num_mice > 0 :
#     print('sim step : %i'%counter)
#     model.step()
    
model.final_datacollector.collect(model,model.all_mice_schedule)
final_model_data = model.final_datacollector.get_model_vars_dataframe()
#final_model_data.to_csv('results/final_model_data.csv', sep='\t')
final_agent_data = model.final_datacollector.get_agent_vars_dataframe()
first_action = final_agent_data['action_history'][0].values[0].loc[0]
# model.food_datacollector.collect(model,model.food_schedule)
# food_data = model.food_datacollector.get_agent_vars_dataframe()
# print(food_data)

#print(final_model_data)

for i in range(len(final_agent_data)) :
#     print('Name : %s'%final_agent_data.index[i][1])
#     print('Age : %i'%final_agent_data['age'][0].values[i])
#     print('Generation : %i'%final_agent_data['generation'][0].values[i])
#     print('Offspring : %i'%final_agent_data['num_offspring'][0].values[i])
#     print('Energy : %f'%final_agent_data['energy'][0].values[i])
#     print('Hunger_status : %f'%final_agent_data['hunger_status'][0].values[i])
#     print(final_agent_data['action_history'][0].values[i])
    print(final_agent_data['action_history'][0].values[i].loc[0])
#     print(final_agent_data['possible_actions'][0].values[i])
#     print(final_agent_data['primary_values'][0].values[i])
#     print(final_agent_data['primary_values'])
#     print(final_agent_data['secondary_values'][0].values[i])
    print(final_agent_data['sensor_vector'][0].values[i])
#     print(final_agent_data['sensor_position'][0].values[i])