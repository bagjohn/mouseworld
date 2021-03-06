
# coding: utf-8

# In[1]:

#%%writefile check_mousebrain_learning.py

from mouseworld import mouseworld
import time

# Build the modelprint('Building mouseworld')
print('Building mouseworld')
model = mouseworld.Mouseworld([0, 0, 1], 100, 40, 100, 100, mousebrain_inheritance = True)

print('Preparing odor layers')
for i in range(0) :
    model.food_schedule.step()
    model.predator_schedule.step()
    model.diffuse_odor_layers(model.odor_layers)

counter = 0  
# Run for discrete number of timesteps
for i in range(10) :
    counter += 1
    print(model.num_mice)
    print('sim step : %i'%counter)
    model.step()
    
# while model.num_mice > 0 :
#         c=time.time()
#         counter += 1
#         model.step()
#         d=time.time()
#         print('sim step : %i in %f'%(counter, d-c))
# print('Simulation terminated - No alive mice')

print('Gathering agent mousebrain data')
model.final_datacollector.collect(model,model.all_mice_schedule)
final_agent_data = model.final_datacollector.get_agent_vars_dataframe()
# mousebrain_data = final_agent_data[['generation', 'initial_mousebrain_weights', 'current_mousebrain_weights','final_mousebrain_weights']]
mousebrain_data = final_agent_data[['generation', 'initial_mousebrain_weights', 'final_mousebrain_weights']]

print(mousebrain_data['generation'])
print(mousebrain_data['initial_mousebrain_weights'])
# print(mousebrain_data['current_mousebrain_weights'])
print(mousebrain_data['final_mousebrain_weights'])
# print(mousebrain_data['final_mousebrain_weights'][0].values)


# In[4]:

get_ipython().run_cell_magic('writefile', 'check_family_tree.py', "\nfrom mouseworld import mouseworld\nimport time\nimport matplotlib.pyplot as plt\nimport matplotlib as mpl\nimport numpy as np\nimport os\nimport sys\n\n# Arguments\n# argv[1] TO_DEATH means simulation runs until all mice die. Otherwise it defines the number of time steps\n# argv[2] defines the simulation number therefore the results folder name\n\nsimulation_number = int(sys.argv[2])\nresult_folder = 'results/simulation_%i'%simulation_number\nos.makedirs(result_folder)\n\nnum_mice = [2, 2, 2]\n\n# Build the model\nprint('Building mouseworld')\nmodel = mouseworld.Mouseworld(num_mice, 200, 100, 100, 100)\n\n\n# Prepare environment by stepping food and predators and diffusing odors\n# for i in range(100) :\n#     model.food_schedule.step()\n#     model.predator_schedule.step()\n#     model.diffuse_odor_layers_parallel(model.odor_layers)\nprint('Preparing odor layers')\nfor i in range(10) :\n    model.food_schedule.step()\n    model.predator_schedule.step()\n    model.diffuse_odor_layers(model.odor_layers)\n\ncounter = 0    \n\n#Run for discrete number of timesteps\nif sys.argv[1] == 'TO_DEATH' :\n    while model.num_mice > 0 :\n        c=time.time()\n        counter += 1\n        model.step()\n        d=time.time()\n        print('sim step : %i in %f'%(counter, d-c))\n    print('Simulation terminated - No alive mice')\n\n#Run until all mice perish\nelse :\n    myrange = int(sys.argv[1])\n    for i in range(myrange) :\n        c=time.time()\n        counter += 1\n        model.step()\n        d=time.time()\n        print('sim step : %i in %f'%(counter, d-c))\n    print('Simulation terminated - Number of time steps reached')\n    \n\n# Gather model and data\n\nprint('Gathering model data and ploting number of mice')\n\nmodel_data = model.model_datacollector.get_model_vars_dataframe()\nmodel_data = model_data[['Alive_mice', 'Unborn_mice']]\nmodel_data.to_csv('%s/num_mice.csv'%result_folder, sep='\\t')\nplt.figure(1)\nplt.plot(model_data['Alive_mice'])\nplt.plot(model_data['Unborn_mice'])\nplt.savefig('%s/num_mice.png'%result_folder, bbox_inches='tight')\n#plt.show()\n\n# Gather final model and agent data\n\nprint('Gathering agent data and ploting family tree')\n\nmodel.final_datacollector.collect(model,model.all_mice_schedule)\nfinal_model_data = model.final_datacollector.get_model_vars_dataframe()\nfinal_agent_data = model.final_datacollector.get_agent_vars_dataframe()\ngenome_data = final_agent_data[['Genome', 'motor_NN_on', 'learning_on', 'mousebrain_sim']]\ngenome_data = genome_data.reset_index('Step', drop = True)\ngenome_data.to_csv('%s/genome_data.csv'%result_folder, sep='\\t')\ntree_data = final_agent_data[['parent_ID', 'birth_date', 'age', 'generation', 'motor_NN_on', 'learning_on']]\ntree_data = tree_data.reset_index('AgentID').values\n\n\ndef rearrange_data (tree_data) :\n    temp = [mouse for mouse in tree_data if mouse[1] is None]\n    new_tree_data = []\n    while len(temp) != 0 :\n        offspring = [mouse for mouse in tree_data if mouse[1] == temp[0][0]]      \n        new_tree_data.append(temp[0])\n        temp = np.delete(temp, 0, 0)\n        if len(offspring) != 0 :\n            offspring.sort(key=lambda x: -x[2])\n            for x in range(len(offspring)) :\n                temp = np.insert(temp, 0, offspring[x], axis=0)\n    return new_tree_data\n\ncmap1 = mpl.cm.Set1\ncmap2 = mpl.cm.Dark2\ncmap3 = mpl.cm.Set3\nnew_tree_data = rearrange_data(tree_data)\nall_mice = len(new_tree_data)\nplt.figure(figsize=(40, 25))\nfor i in range(all_mice) :\n    mouse = new_tree_data[i]\n    if not mouse[5] :\n        cmap = cmap1\n    elif not mouse[6] :\n        cmap = cmap2\n    else :\n        cmap = cmap3\n    plt.plot((mouse[2], mouse[2] + mouse[3]), (all_mice - i, all_mice - i), color=cmap(mouse[4]%cmap.N), label=mouse[0])\n    plt.legend(bbox_to_anchor=(0, 1), loc='best')\nplt.savefig('%s/family_tree.png'%result_folder, bbox_inches='tight')\n#plt.show()\nprint('Over!!!')")


# In[4]:

get_ipython().run_cell_magic('writefile', 'check_action_history.py', "from mouseworld import mouseworld\nimport time\nimport matplotlib.pyplot as plt\nimport numpy as np\n\nnum_mice = [0, 0, 200]\n\n# Build the model\nmodel = mouseworld.Mouseworld(num_mice, 100, 50, 100, 100)\n\n\n# Prepare environment by stepping food and predators and diffusing odors\n# for i in range(100) :\n#     model.food_schedule.step()\n#     model.predator_schedule.step()\n#     model.diffuse_odor_layers_parallel(model.odor_layers)\na=time.time()\nfor i in range(10) :\n    model.food_schedule.step()\n    model.predator_schedule.step()\n    model.diffuse_odor_layers(model.odor_layers)\n#Run for discrete number of timesteps\nb=time.time()\nprint(b-a)\ncounter = 0\nmyrange = 1000\n# for i in range(myrange) :\n#     c=time.time()\n#     counter += 1\n#     model.step()\n#     d=time.time()\n#     print('sim step : %i in %f'%(counter, d-c))\n#t = np.arange(1, myrange*10 +1, 1)\n#t = np.arange(1, (myrange-2)*10 +1, 1)\n#Run until all mice perish\nwhile model.num_mice > 0 :\n    c=time.time()\n    counter += 1\n    model.step()\n    d=time.time()\n    print('sim step : %i in %f'%(counter, d-c))\n# Gather final model and agent data\n#model.mousebrain_datacollector.collect(model,model.schedule)\n#mousebrain_data = model.mousebrain_datacollector.get_agent_vars_dataframe()\n#mousebrain_data.to_csv('results/mousebrain_data.csv', sep='\\t')\nmodel.final_datacollector.collect(model,model.all_mice_schedule)\nfinal_model_data = model.final_datacollector.get_model_vars_dataframe()\n#final_model_data.to_csv('results/final_model_data.csv', sep='\\t')\nfinal_agent_data = model.final_datacollector.get_agent_vars_dataframe()\n\nprint(final_model_data)\n\nfor i in range(len(final_agent_data)) :\n    print('Name : %s'%final_agent_data.index[i][1])\n    print('Age : %i'%final_agent_data['age'][0].values[i])\n    print('Generation : %i'%final_agent_data['generation'][0].values[i])\n    print('Offspring : %i'%final_agent_data['num_offspring'][0].values[i])\n    print('Energy : %f'%final_agent_data['energy'][0].values[i])\n    print(final_agent_data['action_history'][0].values[i])\n    #print(final_agent_data['possible_actions'][0].values[i])\n    print(final_agent_data['primary_values'][0].values[i])\n    print(final_agent_data['secondary_values'][0].values[i])\n    print(final_agent_data['sensor_vector'][0].values[i])\n    print(final_agent_data['sensor_position'][0].values[i])")


# In[11]:

get_ipython().run_cell_magic('writefile', 'check_mousebrain.py', "\nfrom mouseworld import mouseworld\nimport time\n\nimport matplotlib.pyplot as plt\nimport numpy as np\n\n# Build the model\nmodel = mouseworld.Mouseworld([0, 0, 1], 0, 0, 100, 100)\n\n\n# Prepare environment by stepping food and predators and diffusing odors\n# for i in range(100) :\n#     model.food_schedule.step()\n#     model.predator_schedule.step()\n#     model.diffuse_odor_layers_parallel(model.odor_layers)\n# for i in range(10) :\n#     model.food_schedule.step()\n#     model.predator_schedule.step()\n#     model.diffuse_odor_layers_parallel(model.odor_layers)\n#Run for discrete number of timesteps\ncounter = 0\nmyrange = 20\nfor i in range(myrange) :\n    counter += 1\n    print('sim step : %i'%counter)\n    model.step()\n#t = np.arange(1, myrange*10 +1, 1)\n#t = np.arange(1, (myrange-2)*10 +1, 1)\n# Run until all mice perish\n# while model.num_mice > 0 :\n#     print('sim step : %i'%counter)\n#     model.step()\n    \n# Gather final model and agent data\nmodel.mousebrain_datacollector.collect(model,model.schedule)\nmousebrain_data = model.mousebrain_datacollector.get_agent_vars_dataframe()\n#mousebrain_data.to_csv('results/mousebrain_data.csv', sep='\\t')\n\nodor = mousebrain_data['odor'][0].values[0]\nstate = mousebrain_data['state'][0].values[0]\napproach = mousebrain_data['approach'][0].values[0]\navoid = mousebrain_data['avoid'][0].values[0]\nsearch = mousebrain_data['search'][0].values[0]\nchange = mousebrain_data['change'][0].values[0]\nerrors0 = mousebrain_data['errors0'][0].values[0]\nerrors1 = mousebrain_data['errors1'][0].values[0]\nerrors2 = mousebrain_data['errors2'][0].values[0]\n\n# print (type(a))\n# print(a)\n\nodor1 = [row[0] for row in odor]\nodor2 = [row[1] for row in odor]\n\napproach1 = [row[0] for row in approach]\napproach2 = [row[1] for row in approach]\n\navoid1 = [row[0] for row in avoid]\navoid2 = [row[1] for row in avoid]\n\nsearch1 = [row[0] for row in search]\nsearch2 = [row[1] for row in search]\n\n\n\nstate0 = [row[0] for row in state]\nstate1 = [row[1] for row in state]\nstate2 = [row[2] for row in state]\n\ndata = []\n\ndata.append(odor1)\ndata.append(odor2)\ndata.append(change)\ndata.append(search)\ndata.append(errors0)\ndata.append(errors1)\ndata.append(errors2)\n\ndata.append(avoid)\ndata.append(approach)\n# plt.plot(t, odor, 'r--', t, state, 'r--', t, avoid, 'r--',\n#          t, avoid, 'r--', t, search, 'r--', t, errors0, 'r--',\n#          t, errors0, 'r--', t, errors1, 'r--', t, errors2, 'r--')\n\n# plt.plot(odor1, color='blue', odor2, color = 'red', approach1, color = 'green', approach2, color = 'black')\n# plt.show()\n#x = np.linspace(0, 1, 10)\nfor i in [3]:\n    plt.plot(data[i], label='$data{i}$'.format(i=i))\nplt.legend(loc='best')\nplt.show()\n\nfor i in [2,4]:\n    plt.plot(data[i], label='$data{i}$'.format(i=i))\nplt.legend(loc='best')\nplt.show()\n\nfor i in [2,5]:\n    plt.plot(data[i], label='$data{i}$'.format(i=i))\nplt.legend(loc='best')\nplt.show()\n\nfor i in [2,6]:\n    plt.plot(data[i], label='$data{i}$'.format(i=i))\nplt.legend(loc='best')\nplt.show()\n\nplt.plot(odor1, 'bs', change, 'r--')\nplt.show()\n\nplt.plot(errors0, 'bs', errors1, 'r--', errors2, 'g^')\nplt.show()\n\n# plt.plot(search1, 'bs', search2, 'r--')\n# plt.show()\n\nplt.plot(approach1, 'bs', approach2, 'r--')\nplt.show()\n\nplt.plot(state0, 'bs', state1, 'r--', state2, 'g^')\nplt.show()\n\nplt.plot(odor1, 'bs', odor2, 'r--')\nplt.show()")


# In[ ]:



