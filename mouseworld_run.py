
#from ipywidgets import widgets
from mouseworld import mouseworld

empty_model = mouseworld.Mouseworld(100, 5, 100, 20, 100, 100)
#for i in range(5000) :
counter = 0
for i in range(10) :
    empty_model.diffuse_odor_layers(empty_model.odor_layers)
while empty_model.num_mice > 0 :
#for i in range(20) :
    counter += 1
    print('sim step : %i'%counter)
    #print('num_mice : %i'%empty_model.num_mice)
    #print('num_unborn_mice : %i'%empty_model.num_unborn_mice)
    #print('empty_model.schedule.get_agent_count() : %i'%empty_model.schedule.get_agent_count())
    empty_model.step()

empty_model.final_datacollector.collect(empty_model,empty_model.schedule)
    
gini4 = empty_model.final_datacollector.get_agent_vars_dataframe()
#gini1 = empty_model.datacollector.get_model_vars_dataframe()
gini2 = empty_model.test_datacollector.get_agent_vars_dataframe()
#gini3 = empty_model.predator_datacollector.get_agent_vars_dataframe()
gini4.to_csv('results2.csv', sep='\t')
gini2.to_csv('results.csv', sep='\t')
#gini2
gini4