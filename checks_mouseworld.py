
# coding: utf-8

# In[4]:

#%%writefile check_family_tree.py
from mouseworld import mouseworld
import time
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
#import networkx as nx

simulation_number = 111
result_folder = 'results/simulation_%i'%simulation_number
os.makedirs(result_folder)

num_mice = [10, 0, 0]

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
myrange = 2
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

# Gather model and data
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
model.final_datacollector.collect(model,model.all_mice_schedule)
final_model_data = model.final_datacollector.get_model_vars_dataframe()
final_agent_data = model.final_datacollector.get_agent_vars_dataframe()
genome_data = final_agent_data[['Genome']]
genome_data = genome_data.reset_index('Step', drop = True)
genome_data.to_csv('results/simulation_%i/genome_data.csv'%result_folder, sep='\t')
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
plt.savefig('results/simulation_%i/family_tree.png'%result_folder, bbox_inches='tight')
plt.show()


# In[6]:

get_ipython().run_cell_magic('writefile', 'check_action_history.py', "from mouseworld import mouseworld\nimport time\nimport matplotlib.pyplot as plt\nimport numpy as np\n\nnum_mice = [0, 0, 200]\n\n# Build the model\nmodel = mouseworld.Mouseworld(num_mice, 100, 50, 100, 100)\n\n\n# Prepare environment by stepping food and predators and diffusing odors\n# for i in range(100) :\n#     model.food_schedule.step()\n#     model.predator_schedule.step()\n#     model.diffuse_odor_layers_parallel(model.odor_layers)\na=time.time()\nfor i in range(10) :\n    model.food_schedule.step()\n    model.predator_schedule.step()\n    model.diffuse_odor_layers(model.odor_layers)\n#Run for discrete number of timesteps\nb=time.time()\nprint(b-a)\ncounter = 0\nmyrange = 1000\n# for i in range(myrange) :\n#     c=time.time()\n#     counter += 1\n#     model.step()\n#     d=time.time()\n#     print('sim step : %i in %f'%(counter, d-c))\n#t = np.arange(1, myrange*10 +1, 1)\n#t = np.arange(1, (myrange-2)*10 +1, 1)\n#Run until all mice perish\nwhile model.num_mice > 0 :\n    c=time.time()\n    counter += 1\n    model.step()\n    d=time.time()\n    print('sim step : %i in %f'%(counter, d-c))\n# Gather final model and agent data\n#model.mousebrain_datacollector.collect(model,model.schedule)\n#mousebrain_data = model.mousebrain_datacollector.get_agent_vars_dataframe()\n#mousebrain_data.to_csv('results/mousebrain_data.csv', sep='\\t')\nmodel.final_datacollector.collect(model,model.all_mice_schedule)\nfinal_model_data = model.final_datacollector.get_model_vars_dataframe()\n#final_model_data.to_csv('results/final_model_data.csv', sep='\\t')\nfinal_agent_data = model.final_datacollector.get_agent_vars_dataframe()\n\nprint(final_model_data)\n\nfor i in range(len(final_agent_data)) :\n    print('Name : %s'%final_agent_data.index[i][1])\n    print('Age : %i'%final_agent_data['age'][0].values[i])\n    print('Generation : %i'%final_agent_data['generation'][0].values[i])\n    print('Offspring : %i'%final_agent_data['num_offspring'][0].values[i])\n    print('Energy : %f'%final_agent_data['energy'][0].values[i])\n    print(final_agent_data['action_history'][0].values[i])\n    #print(final_agent_data['possible_actions'][0].values[i])\n    print(final_agent_data['primary_values'][0].values[i])\n    print(final_agent_data['secondary_values'][0].values[i])\n    print(final_agent_data['sensor_vector'][0].values[i])\n    print(final_agent_data['sensor_position'][0].values[i])")


# In[7]:

get_ipython().run_cell_magic('writefile', 'check_mousebrain.py', "\nfrom mouseworld import mouseworld\nimport time\n\nimport matplotlib.pyplot as plt\nimport numpy as np\n\n# Build the model\nmodel = mouseworld.Mouseworld([0, 0, 1], 0, 0, 100, 100)\n\n\n# Prepare environment by stepping food and predators and diffusing odors\n# for i in range(100) :\n#     model.food_schedule.step()\n#     model.predator_schedule.step()\n#     model.diffuse_odor_layers_parallel(model.odor_layers)\n# for i in range(10) :\n#     model.food_schedule.step()\n#     model.predator_schedule.step()\n#     model.diffuse_odor_layers_parallel(model.odor_layers)\n#Run for discrete number of timesteps\ncounter = 0\nmyrange = 20\nfor i in range(myrange) :\n    counter += 1\n    print('sim step : %i'%counter)\n    model.step()\n#t = np.arange(1, myrange*10 +1, 1)\n#t = np.arange(1, (myrange-2)*10 +1, 1)\n# Run until all mice perish\n# while model.num_mice > 0 :\n#     print('sim step : %i'%counter)\n#     model.step()\n    \n# Gather final model and agent data\nmodel.mousebrain_datacollector.collect(model,model.schedule)\nmousebrain_data = model.mousebrain_datacollector.get_agent_vars_dataframe()\n#mousebrain_data.to_csv('results/mousebrain_data.csv', sep='\\t')\n\nodor = mousebrain_data['odor'][0].values[0]\nstate = mousebrain_data['state'][0].values[0]\napproach = mousebrain_data['approach'][0].values[0]\navoid = mousebrain_data['avoid'][0].values[0]\nsearch = mousebrain_data['search'][0].values[0]\nchange = mousebrain_data['change'][0].values[0]\nerrors0 = mousebrain_data['errors0'][0].values[0]\nerrors1 = mousebrain_data['errors1'][0].values[0]\nerrors2 = mousebrain_data['errors2'][0].values[0]\n\n# print (type(a))\n# print(a)\n\nodor1 = [row[0] for row in odor]\nodor2 = [row[1] for row in odor]\n\napproach1 = [row[0] for row in approach]\napproach2 = [row[1] for row in approach]\n\navoid1 = [row[0] for row in avoid]\navoid2 = [row[1] for row in avoid]\n\nsearch1 = [row[0] for row in search]\nsearch2 = [row[1] for row in search]\n\n\n\nstate0 = [row[0] for row in state]\nstate1 = [row[1] for row in state]\nstate2 = [row[2] for row in state]\n\ndata = []\n\ndata.append(odor1)\ndata.append(odor2)\ndata.append(change)\ndata.append(search)\ndata.append(errors0)\ndata.append(errors1)\ndata.append(errors2)\n\ndata.append(avoid)\ndata.append(approach)\n# plt.plot(t, odor, 'r--', t, state, 'r--', t, avoid, 'r--',\n#          t, avoid, 'r--', t, search, 'r--', t, errors0, 'r--',\n#          t, errors0, 'r--', t, errors1, 'r--', t, errors2, 'r--')\n\n# plt.plot(odor1, color='blue', odor2, color = 'red', approach1, color = 'green', approach2, color = 'black')\n# plt.show()\n#x = np.linspace(0, 1, 10)\nfor i in [3]:\n    plt.plot(data[i], label='$data{i}$'.format(i=i))\nplt.legend(loc='best')\nplt.show()\n\nfor i in [2,4]:\n    plt.plot(data[i], label='$data{i}$'.format(i=i))\nplt.legend(loc='best')\nplt.show()\n\nfor i in [2,5]:\n    plt.plot(data[i], label='$data{i}$'.format(i=i))\nplt.legend(loc='best')\nplt.show()\n\nfor i in [2,6]:\n    plt.plot(data[i], label='$data{i}$'.format(i=i))\nplt.legend(loc='best')\nplt.show()\n\nplt.plot(odor1, 'bs', change, 'r--')\nplt.show()\n\nplt.plot(errors0, 'bs', errors1, 'r--', errors2, 'g^')\nplt.show()\n\n# plt.plot(search1, 'bs', search2, 'r--')\n# plt.show()\n\nplt.plot(approach1, 'bs', approach2, 'r--')\nplt.show()\n\nplt.plot(state0, 'bs', state1, 'r--', state2, 'g^')\nplt.show()\n\nplt.plot(odor1, 'bs', odor2, 'r--')\nplt.show()")


# In[ ]:



