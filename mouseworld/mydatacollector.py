
from mesa.datacollection import DataCollector

class MyDataCollector(DataCollector):
    ## subclass DataCollector to only collect data on certain agents
    ## in this case, I only report them if they are NOT alive
    ## self.alive is an attribute that I track for my agents
    def __init__(self, model_reporters={}, agent_reporters={}, tables={}):
        """ Instantiate a DataCollector with lists of model and agent reporters.
        Both model_reporters and agent_reporters accept a dictionary mapping a
        variable name to a method used to collect it.
        For example, if there was only one model-level reporter for number of
        agents, it might look like:
            {"agent_count": lambda m: m.schedule.get_agent_count() }
        If there was only one agent-level reporter (e.g. the agent's energy),
        it might look like this:
            {"energy": lambda a: a.energy}
        The tables arg accepts a dictionary mapping names of tables to lists of
        columns. For example, if we want to allow agents to write their age
        when they are destroyed (to keep track of lifespans), it might look
        like:
            {"Lifespan": ["unique_id", "age"]}
        Args:
            model_reporters: Dictionary of reporter names and functions.
            agent_reporters: Dictionary of reporter names and functions.
        """
        self.model_reporters = {}
        self.agent_reporters = {}

        self.model_vars = {}
        self.agent_vars = {}
        self.tables = {}

        for name, func in model_reporters.items():
            self._new_model_reporter(name, func)

        for name, func in agent_reporters.items():
            self._new_agent_reporter(name, func)

        for name, columns in tables.items():
            self._new_table(name, columns)

    
    def collect(self, model, schedule):
        """ Collect all the data for the given model object. """
        if self.model_reporters:
            for var, reporter in self.model_reporters.items():
                self.model_vars[var].append(reporter(model))

        if self.agent_reporters:
            for var, reporter in self.agent_reporters.items():
                agent_records = []
                #add an if clause to only append to agent records if our agent meets a certain condition
                for agent in schedule.agents:
                    agent_records.append((agent.unique_id, reporter(agent)))
                self.agent_vars[var].append(agent_records)
                
## When I define the datacollector for my model, I use MyDataCollector rather than the default DataCollector