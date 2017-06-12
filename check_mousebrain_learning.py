
from mouseworld import mouseworld
import time

# Build the modelprint('Building mouseworld')
print('Building mouseworld')
model = mouseworld.Mouseworld([0, 0, 1], 100, 40, 100, 100, mousebrain_inheritance = True)

print('Preparing odor layers')
for i in range(10) :
    model.food_schedule.step()
    model.predator_schedule.step()
    model.diffuse_odor_layers(model.odor_layers)

counter = 0  

while model.num_mice > 0 :
        c=time.time()
        counter += 1
        model.step()
        d=time.time()
        print('sim step : %i in %f'%(counter, d-c))
    print('Simulation terminated - No alive mice')

print('Gathering agent mousebrain data')
model.final_datacollector.collect(model,model.all_mice_schedule)
final_agent_data = model.final_datacollector.get_agent_vars_dataframe()
mousebrain_data = final_agent_data[['initial_mousebrain_weights', 'final_mousebrain_weights']]

print(mousebrain_data['initial_mousebrain_weights'])
print(mousebrain_data['final_mousebrain_weights'])