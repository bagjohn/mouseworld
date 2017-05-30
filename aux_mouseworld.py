
# coding: utf-8

# In[1]:

get_ipython().run_cell_magic('writefile', 'mouseworld/input_manager.py', '\nclass Input_manager(object):\n    """because we need to contain state, the easier way to do that in\n    Python is to make a class"""\n\n    def __init__(self):\n        self.state = \'wait\'\n        self.value = 0\n        \n#     def set_state(x):\n#             if x == \'wait\' :\n#                 current_state = [-1,-1,-1]\n#             if x == \'search\' :\n#                 current_state = [0,-1,-1]\n#             if x == \'approach\' :\n#                 current_state = [-1,0,-1]\n#             if x == \'avoid\' :\n#                 current_state = [-1,-1,0]\n#             return current_state\n        \n    def modify_value(self, modifyer):\n        """you can modify the state value or over-write it here\n        or you can just modify the state parameter directly"""\n        #print("Manage the input here.")\n        self.value += modifyer\n\n    def return_state(self, t):\n        return self.state\n    \n    def return_value(self, t):\n        return self.value\n    \n        ')


# In[2]:

get_ipython().run_cell_magic('writefile', 'mouseworld/mydatacollector.py', '\nfrom mesa.datacollection import DataCollector\n\nclass MyDataCollector(DataCollector):\n    ## subclass DataCollector to only collect data on certain agents\n    ## in this case, I only report them if they are NOT alive\n    ## self.alive is an attribute that I track for my agents\n    def __init__(self, model_reporters={}, agent_reporters={}, tables={}):\n        """ Instantiate a DataCollector with lists of model and agent reporters.\n        Both model_reporters and agent_reporters accept a dictionary mapping a\n        variable name to a method used to collect it.\n        For example, if there was only one model-level reporter for number of\n        agents, it might look like:\n            {"agent_count": lambda m: m.schedule.get_agent_count() }\n        If there was only one agent-level reporter (e.g. the agent\'s energy),\n        it might look like this:\n            {"energy": lambda a: a.energy}\n        The tables arg accepts a dictionary mapping names of tables to lists of\n        columns. For example, if we want to allow agents to write their age\n        when they are destroyed (to keep track of lifespans), it might look\n        like:\n            {"Lifespan": ["unique_id", "age"]}\n        Args:\n            model_reporters: Dictionary of reporter names and functions.\n            agent_reporters: Dictionary of reporter names and functions.\n        """\n        self.model_reporters = {}\n        self.agent_reporters = {}\n\n        self.model_vars = {}\n        self.agent_vars = {}\n        self.tables = {}\n\n        for name, func in model_reporters.items():\n            self._new_model_reporter(name, func)\n\n        for name, func in agent_reporters.items():\n            self._new_agent_reporter(name, func)\n\n        for name, columns in tables.items():\n            self._new_table(name, columns)\n\n    \n    def collect(self, model, schedule):\n        """ Collect all the data for the given model object. """\n        if self.model_reporters:\n            for var, reporter in self.model_reporters.items():\n                self.model_vars[var].append(reporter(model))\n\n        if self.agent_reporters:\n            for var, reporter in self.agent_reporters.items():\n                agent_records = []\n                #add an if clause to only append to agent records if our agent meets a certain condition\n                for agent in schedule.agents:\n                    agent_records.append((agent.unique_id, reporter(agent)))\n                self.agent_vars[var].append(agent_records)\n                \n## When I define the datacollector for my model, I use MyDataCollector rather than the default DataCollector')


# In[ ]:


