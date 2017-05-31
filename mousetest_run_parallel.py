
# at terminal : ipcluster start -n 4

import ipyparallel

clients = ipyparallel.Client()
dview = clients.direct_view()

with dview.sync_imports():
    from mouseworld.mousetest import Mousetest
    import time
dview.push({"Mousetest": Mousetest})

def make_model(header):
    x = 1
    y = 1
    antenna_length = 1
    antenna_angle = 1
    model = Mousetest([0, 0, 1], [0.5, 0.5, 0.5, antenna_length, antenna_angle], (x,y), header, 10, 1, 0, 100, 100)
    for i in range(10) :
        model.food_schedule.step()
        model.predator_schedule.step()
        model.diffuse_odor_layers(model.odor_layers)
    counter = 0
    myrange = 1
    for i in range(myrange) :
        c=time.time()
        counter += 1
        model.step()
        d=time.time()
    model.final_datacollector.collect(model,model.all_mice_schedule)
    final_agent_data = model.final_datacollector.get_agent_vars_dataframe()
    sensor_vector = final_agent_data['sensor_vector'][0].values[0]
    sensor_position = final_agent_data['sensor_position'][0].values[0]
    motor_vector = final_agent_data['motor_vector'][0].values[0]
    first_action = final_agent_data['action_history'][0].values[0].loc[0]
    first_action = (first_action['Duration'], first_action['Termination'])
    #first_action = final_agent_data['action_history'][0].values[0].loc[0]
    return (first_action, sensor_vector, sensor_position, motor_vector)
 
all_first_actions = dview.map_sync(make_model, [0,1,2,3,4,5,6,7])
print(all_first_actions)