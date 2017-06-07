from mouseworld import mouseworld
import time
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

num_mice = [10, 0, 0]

# Build the model
model = mouseworld.Mouseworld(num_mice, 100, 50, 100, 100)


# Prepare environment by stepping food and predators and diffusing odors
# for i in range(100) :
#     model.food_schedule.step()
#     model.predator_schedule.step()
#     model.diffuse_odor_layers_parallel(model.odor_layers)
a=time.time()
for i in range(0) :
    model.food_schedule.step()
    model.predator_schedule.step()
    model.diffuse_odor_layers(model.odor_layers)
    
#Run for discrete number of timesteps
b=time.time()
print(b-a)
counter = 0
myrange = 20
for i in range(myrange) :
    c=time.time()
    counter += 1
    model.step()
    d=time.time()
    print('sim step : %i in %f'%(counter, d-c))

#Run until all mice perish
# while model.num_mice > 0 :
#     c=time.time()
#     counter += 1
#     model.step()
#     d=time.time()
#     print('sim step : %i in %f'%(counter, d-c))

# Gather final model and agent data
model.final_datacollector.collect(model,model.all_mice_schedule)
final_model_data = model.final_datacollector.get_model_vars_dataframe()
final_agent_data = model.final_datacollector.get_agent_vars_dataframe()
tree_data = final_agent_data[['parent_ID', 'birth_date', 'age']]
tree_data = tree_data.reset_index('AgentID').values


def rearrange_data (tree_data) :
    temp = [mouse for mouse in tree_data if mouse[1] is None]
    new_tree_data = []
    while len(temp) != 0 :
        offspring = [mouse for mouse in tree_data if mouse[1] == temp[0][0]]      
        new_tree_data.append(temp[0])
        temp = np.delete(temp, 0, 0)
        if len(offspring) != 0 :
            offspring.sort(key=lambda x: -x[2])
            for x in range(len(offspring)) :
                temp = np.insert(temp, 0, offspring[x], axis=0)
    return new_tree_data


new_tree_data = rearrange_data(tree_data)
for i in range(len(new_tree_data)) :
    mouse = new_tree_data[i]
    plt.plot((mouse[2], mouse[2] + mouse[3]), (i, i), 'k-')
plt.show()