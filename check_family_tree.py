from mouseworld import mouseworld
import time
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
#import networkx as nx

simulation_number = 112
result_folder = 'results/simulation_%i'%simulation_number
os.makedirs(result_folder)

num_mice = [100, 0, 0]

# Build the model
model = mouseworld.Mouseworld(num_mice, 100, 50, 100, 100)


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
# b=time.time()
# print(b-a)
counter = 0
# myrange = 2
# for i in range(myrange) :
#     c=time.time()
#     counter += 1
#     model.step()
#     d=time.time()
#     print('sim step : %i in %f'%(counter, d-c))
# print('Simulation terminated - Number of time steps reached')

#Run until all mice perish
while model.num_mice > 0 :
    c=time.time()
    counter += 1
    model.step()
    d=time.time()
    print('sim step : %i in %f'%(counter, d-c))
print('Simulation terminated - No alive mice')
# Gather model and data
print('Gathering model data and ploting number of mice')
model_data = model.model_datacollector.get_model_vars_dataframe()
model_data = model_data[['Alive_mice', 'Unborn_mice']]
model_data.to_csv('%s/num_mice.csv'%result_folder, sep='\t')
plt.plot(model_data['Alive_mice'])
plt.plot(model_data['Unborn_mice'])
plt.savefig('%s/num_mice.png'%result_folder, bbox_inches='tight')
plt.show()
#     plt.legend(bbox_to_anchor=(0, 1), loc='best')
# plt.savefig('results/check_family_tree.png', bbox_inches='tight')
# plt.show()

# Gather final model and agent data
print('Gathering agent data and ploting family tree')
model.final_datacollector.collect(model,model.all_mice_schedule)
final_model_data = model.final_datacollector.get_model_vars_dataframe()
final_agent_data = model.final_datacollector.get_agent_vars_dataframe()
genome_data = final_agent_data[['Genome']]
genome_data = genome_data.reset_index('Step', drop = True)
genome_data.to_csv('%s/genome_data.csv'%result_folder, sep='\t')
# print(type(genome_data))
tree_data = final_agent_data[['parent_ID', 'birth_date', 'age', 'generation']]
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

cmap = mpl.cm.Set1
# print(cmap.N)
new_tree_data = rearrange_data(tree_data)
for i in range(len(new_tree_data)) :
    mouse = new_tree_data[i]
#     plt.plot((mouse[2], mouse[2] + mouse[3]), (i, i), 'k-')
    plt.plot((mouse[2], mouse[2] + mouse[3]), (i, i), color=cmap(mouse[4]%cmap.N), label=mouse[0])
    plt.legend(bbox_to_anchor=(0, 1), loc='best')
plt.savefig('%s/family_tree.png'%result_folder, bbox_inches='tight')
plt.show()
print('Over!!!')