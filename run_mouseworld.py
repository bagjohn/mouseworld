
# coding: utf-8

# In[2]:

get_ipython().run_cell_magic('writefile', 'mouseworld_run.py', "\n#from ipywidgets import widgets\nfrom mouseworld import mouseworld\nimport time\n\nempty_model = mouseworld.Mouseworld(50, 5, 100, 40, 100, 100)\n#for i in range(5000) :\ncounter = 0\n\n#a = time.time()\nfor i in range(100) :\n    empty_model.diffuse_odor_layers_parallel(empty_model.odor_layers)\n#b = time.time()\n#print(b-a)\nwhile empty_model.num_mice > 0 :\n#for i in range(20) :\n    counter += 1\n    print('sim step : %i'%counter)\n    #print('num_mice : %i'%empty_model.num_mice)\n    #print('num_unborn_mice : %i'%empty_model.num_unborn_mice)\n    #print('empty_model.schedule.get_agent_count() : %i'%empty_model.schedule.get_agent_count())\n    empty_model.step()\n\nempty_model.final_datacollector.collect(empty_model,empty_model.schedule)\n    \ngini4 = empty_model.final_datacollector.get_agent_vars_dataframe()\n#gini1 = empty_model.datacollector.get_model_vars_dataframe()\ngini2 = empty_model.test_datacollector.get_agent_vars_dataframe()\n#gini3 = empty_model.predator_datacollector.get_agent_vars_dataframe()\ngini4.to_csv('results2.csv', sep='\\t')\ngini2.to_csv('results.csv', sep='\\t')\n#gini2\ngini4")


# In[ ]:



