
# coding: utf-8

# In[5]:

get_ipython().run_cell_magic('writefile', 'mouseworld_run.py', "\nfrom mouseworld import mouseworld\nimport time\n\n# Build the model\nmodel = mouseworld.Mouseworld([100, 0, 0], 100, 40, 100, 100)\n\n# Gather initial randomly distributed data\nmodel.initial_datacollector.collect(model,model.schedule)\ninitial_model_data = model.initial_datacollector.get_model_vars_dataframe()\ninitial_model_data.to_csv('results/initial_model_data.csv', sep='\\t')\n\n# Prepare environment by stepping food and predators and diffusing odors\nfor i in range(10) :\n    model.food_schedule.step()\n    model.predator_schedule.step()\n    model.diffuse_odor_layers_parallel(model.odor_layers)\n\n# Run for discrete number of timesteps\ncounter = 0\nfor i in range(20) :\n    counter += 1\n    print('sim step : %i'%counter)\n    model.step()\n\n# Run until all mice perish\n# while model.num_mice > 0 :\n#     print('sim step : %i'%counter)\n#     model.step()\n    \n# Gather final model and agent data\nmodel.final_datacollector.collect(model,model.schedule)\nfinal_model_data = model.final_datacollector.get_model_vars_dataframe()\nfinal_model_data.to_csv('results/final_model_data.csv', sep='\\t')\nfinal_agent_data = model.final_datacollector.get_agent_vars_dataframe()\nfinal_agent_data.to_csv('results/final_agent_data.csv', sep='\\t')\n\n# Gather test model and agent data\ntest_agent_data = model.test_datacollector.get_agent_vars_dataframe()\ntest_agent_data.to_csv('results/test_agent_data.csv', sep='\\t')")


# In[2]:

get_ipython().run_cell_magic('writefile', 'mouseworld_run2.py', '\nfrom mouseworld import mouseworld\n\n\n# Build the model\nmodel = mouseworld.Mouseworld([0, 0, 10], 100, 40, 100, 100)\n\n# Run for discrete number of timesteps\n\nfor i in range(1) :\n\n    model.step()\n\n')


# In[3]:

get_ipython().run_cell_magic('writefile', 'plot_genome.py', "\nfrom mouseworld import mouseworld\nimport time\nimport matplotlib.pyplot as plt\nimport numpy as np\n\n# Build the model\nmodel = mouseworld.Mouseworld([0, 0, 100], 100, 40, 100, 100)\ngenome = model.initialization_genome\nfig = plt.figure()\nnum_genes = genome[0].size\nfor i in range(num_genes) :\n    x = genome[:,i]\n    ax = fig.add_subplot(3,3,(i+1))\n    n, bins, patches = ax.hist(x, 20, normed=1, facecolor='g', alpha=0.75)\n\n\n#plt.xlabel('Gene value')\n#plt.ylabel('Probability')\n#plt.title('Histogram of gene distribution')\n#plt.text(60, .025, r'$\\mu=100,\\ \\sigma=15$')\n#plt.axis([0, 1, 0, 20])\n#plt.grid(True)\nplt.show()\n")


# In[ ]:



