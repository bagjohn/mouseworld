
from mouseworld import mouseworld
import time

# Build the model
model = mouseworld.Mouseworld([0, 0, 100], 100, 40, mousebrain_inheritance = True)

# Gather initial randomly distributed data
model.initial_datacollector.collect(model,model.schedule)
initial_model_data = model.initial_datacollector.get_model_vars_dataframe()
initial_model_data.to_csv('results/initial_model_data.csv', sep='\t')

# Prepare environment by stepping food and predators and diffusing odors
for i in range(2) :
    model.food_schedule.step()
    model.predator_schedule.step()
    model.diffuse_odor_layers_parallel(model.odor_layers)

counter = 0
# Run for discrete number of timesteps
for i in range(2) :
    counter += 1
    print('sim step : %i'%counter)
    model.step()

# Run until all mice perish
# while model.num_mice > 0 :
#     counter += 1
#     print('sim step : %i'%counter)
#     model.step()
    
# Gather final model and agent data
model.final_datacollector.collect(model,model.all_mice_schedule)
final_model_data = model.final_datacollector.get_model_vars_dataframe()
final_model_data.to_csv('results/final_model_data.csv', sep='\t')
final_agent_data = model.final_datacollector.get_agent_vars_dataframe()
final_agent_data.to_csv('results/final_agent_data.csv', sep='\t')

# Gather test model and agent data
test_agent_data = model.test_datacollector.get_agent_vars_dataframe()
test_agent_data.to_csv('results/test_agent_data.csv', sep='\t')