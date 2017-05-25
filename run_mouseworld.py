
# coding: utf-8

# In[1]:

get_ipython().run_cell_magic('writefile', 'mouseworld_run.py', "\nfrom mouseworld import mouseworld\nimport time\n\n# Build the model\nmodel = mouseworld.Mouseworld(100, 5, 100, 40, 100, 100)\n\n# Gather initial randomly distributed data\nmodel.initial_datacollector.collect(model,model.schedule)\ninitial_model_data = model.datacollector.get_model_vars_dataframe()\ninitial_model_data.to_csv('results/initial_model_data.csv', sep='\\t')\n\n# Prepare environment by stepping food and predators and diffusing odors\nfor i in range(100) :\n    model.food_schedule.step()\n    model.predator_schedule.step()\n    model.diffuse_odor_layers_parallel(model.odor_layers)\n\n# Run for discrete number of timesteps\n# counter = 0\n# for i in range(20) :\n#     counter += 1\n#     print('sim step : %i'%counter)\n#     model.step()\n\n# Run until all mice perish\nwhile model.num_mice > 0 :\n    print('sim step : %i'%counter)\n    model.step()\n    \n# Gather final model and agent data\nmodel.final_datacollector.collect(model,model.schedule)\nfinal_model_data = model.final_datacollector.get_model_vars_dataframe()\nfinal_model_data.to_csv('results/final_model_data.csv', sep='\\t')\nfinal_agent_data = model.final_datacollector.get_agent_vars_dataframe()\nfinal_agent_data.to_csv('results/final_agent_data.csv', sep='\\t')\n\n# Gather test model and agent data\ntest_agent_data = model.test_datacollector.get_agent_vars_dataframe()\ntest_agent_data.to_csv('results/test_agent_data.csv', sep='\\t')")


# In[ ]:



