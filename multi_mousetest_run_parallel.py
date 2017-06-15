
# at terminal : ipcluster start -n 4

import ipyparallel
import itertools
import mouseworld


clients = ipyparallel.Client()
dview = clients.direct_view()

# mouse_list_file = sys.argv[1]
mouse_list= []
# max_speed = [0, 0.2, 0.4, 0.6, 0.8, 1]
# antenna_length = [0, 0.2, 0.4, 0.6, 0.8, 1]
# antenna_angle = [0, 0.2, 0.4, 0.6, 0.8, 1]
max_speed = [0.2]
antenna_length = [0.4]
antenna_angle = [0.2]
params = (max_speed, antenna_length, antenna_angle)
param_combs = list(itertools.product(*params))
for params in param_combs :
    genome = [params[0], 0, 0, params[1], params[2]]
    mouse_list.append(tuple([genome, True, True, None]))
    mouse_list.append(tuple([genome, True, False, None]))
    mouse_list.append(tuple([genome, False, False, None]))
# mouse_list=[([[0.2,0.4,0.6,0.4,0.2], True, True, None]),
#             ([[0.2,0.4,0.6,0.4,0.2], True, False, None]),([[0.2,0.4,0.6,0.4,0.2], False, False, None])]
# pos =[(1,1), (1,2), (1,4), (1,5), (2,2), (2,3), (2,4), (2,5), (3,3), (3,4), (3,5), (4,4), (4,5), (5,5)]
# # pos =[(1,1), (2,2), (3,3)]
# header = [0,1,2,3,4,5,6,7]
# # header = [0]

# params = (pos, header, mouse_list)
# param_combs = list(itertools.product(*params))

with dview.sync_imports():
    from mouseworld.multi_mousetest import Multi_Mousetest
    import time
dview.push({"Multi_Mousetest": Multi_Mousetest})

def make_model(mouse_data):
    #header = params[0]
#     pos = params[0]
#     header = params[1]
#     mouse_list = params[2]
    model = Multi_Mousetest(mouse_data, 10, 1, 0, 100, 100)
    for i in range(10) :
        model.food_schedule.step()
        #model.predator_schedule.step()
        model.diffuse_odor_layers(model.odor_layers)
    #counter = 0
    myrange = 60
    for i in range(myrange) :
        #c=time.time()
        #counter += 1
        model.step()
        #d=time.time()
    model.final_datacollector.collect(model,model.all_mice_schedule)
    final_agent_data = model.final_datacollector.get_agent_vars_dataframe()
    mouse_statistics = final_agent_data[['first_action_duration', 'first_action_termination']]
    mouse_statistics = mouse_statistics.reset_index('Step', drop = True)
    #mouse_statistics = mouse_statistics.reset_index('AgentID', drop = True)
    succesful_trials = mouse_statistics.loc[(mouse_statistics['first_action_termination'] == 'Closure')]
    num_trials = len(mouse_statistics.index)
    num_succesful_trials = len(succesful_trials.index)
    performance = num_succesful_trials / num_trials
    mean_time = succesful_trials['first_action_duration'].mean()
#     sensor_vector = final_agent_data['sensor_vector'][0].values[0]
#     sensor_position = final_agent_data['sensor_position'][0].values[0]
#     motor_vector = final_agent_data['motor_vector'][0].values[0]
#     first_action = final_agent_data['action_history'][0].values[0].loc[0]
#     first_action = mousetest_data
#     first_action = (mousetest_data['Duration'], mousetest_data['Termination'])
    #first_action = final_agent_data['action_history'][0].values[0].loc[0]
    return (performance,  mean_time)
#     return (params, mousetest_data.loc['Mouse_1'])

#     return (first_action, sensor_vector, sensor_position, motor_vector)
 
all_first_actions = dview.map_sync(make_model, mouse_list)
file = open('results/stats.txt','w') 
for i in range(len(mouse_list)) :
    if i%3 == 0 :
        file.write('------------------NEW GENOME------------------\n')
    file.write(str(mouse_list[i]) + '\n')
    file.write(str(all_first_actions[i]) + '\n')
file.close() 