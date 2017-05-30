
# coding: utf-8

# In[ ]:

get_ipython().run_cell_magic('writefile', 'check_mousebrain.py', "\nfrom mouseworld import mouseworld\nimport time\n\n# Build the model\nmodel = mouseworld.Mouseworld(1, 5, 100, 40, 100, 100)\n\n\n# Prepare environment by stepping food and predators and diffusing odors\n# for i in range(100) :\n#     model.food_schedule.step()\n#     model.predator_schedule.step()\n#     model.diffuse_odor_layers_parallel(model.odor_layers)\n\n#Run for discrete number of timesteps\ncounter = 0\nfor i in range(200) :\n    counter += 1\n    print('sim step : %i'%counter)\n    model.step()\n\n# Run until all mice perish\n# while model.num_mice > 0 :\n#     print('sim step : %i'%counter)\n#     model.step()\n    \n# Gather final model and agent data\nmodel.mousebrain_datacollector.collect(model,model.schedule)\nmousebrain_data = model.mousebrain_datacollector.get_agent_vars_dataframe()\nmousebrain_data.to_csv('results/mousebrain_data.csv', sep='\\t')\n")

