
# coding: utf-8

# In[1]:

from mouseworld.mouseworld import Mouseworld
from mouseworld.predator import Predator
from mouseworld.mouse import Mouse
from mouseworld.food import Food

from mesa_mouseworld.visualization.modules.CanvasContinuousSpaceVisualization import CanvasContinuousSpace
from mesa_mouseworld.visualization.modules.CanvasGridVisualization import CanvasGrid
from mesa_mouseworld.visualization.modules.ChartVisualization import ChartModule
#from mesa.visualization.ModularVisualization import ModularServer
from mesa_mouseworld.visualization.ModularVisualization import ModularServer
from mesa_mouseworld.visualization.UserParam import UserSettableParameter

import math

def agent_portrayal(agent):
    if isinstance(agent,Predator):
        portrayal = {"Shape": "circle",
                     "Filled": "true",
                     "Layer": 0,
                     "Color": "Red",
                     "r": 1,
                     "x":agent.pos[0],
                     "y":agent.pos[1]}
        
    elif isinstance(agent,Food):
        # Define the relative amount of the food in terms of the max amount. Constrain to [0,0.5]
        relative_amount = agent.food_amount/(2 * agent.food_amount_range[1])
        portrayal = {"Shape": "rect",
                     "Filled": "true",
                     "Layer": 0,
                     "Color": "Green",
                     "w": 0.5 + relative_amount,
                     "h": 0.5 + relative_amount,
                     "x":agent.pos[0],
                     "y":agent.pos[1]}
        
    
        
    elif isinstance(agent,Mouse):
        portrayal = {"Shape": "mouse",
                     "Filled": "true",
                     "Layer": 0,
                     "Color": "Blue",
                     "header": agent.header,
                     "antenna_length":agent.antenna_length,
                     "antenna_angle":agent.antenna_angle, #math.sin(agent.header),
                     "r": 1,
                     "x":agent.pos[0],
                     "y":agent.pos[1]}
    return portrayal

width = 100
height = 100

space_element = CanvasContinuousSpace(agent_portrayal, width, height, 500, 500)

#odor_layer_element = CanvasGrid(odor_layer_agent_portrayal, width, height, 500, 500)

chart_element = ChartModule([{"Label": "Alive_mice", "Color": "#AA0000"},
                             {"Label": "Unborn_mice", "Color": "#666666"}],
                            data_collector_name="model_datacollector")

model_params=dict( 
        num_mice_wo_MNN=UserSettableParameter('slider', 'Number of mice w/o NN',20, 0, 200, 1),
        num_mice_w_MNN=UserSettableParameter('slider', 'Number of mice w NN w/o learning',0, 0, 200, 1),
        num_mice_w_lMNN=UserSettableParameter('slider', 'Number of mice w NN w learning',0, 0, 200, 1),
        num_food=UserSettableParameter('slider', 'Number of food',10, 0, 200, 1),
        num_predators=UserSettableParameter('slider', 'Number of predators', 5, 0, 200, 1),
        width=width,
        height=height,
        mouse_initial_energy = UserSettableParameter('slider', 'Mouse initial energy',1000, 100, 10000, 100),
        mouse_max_energy = UserSettableParameter('slider', 'Mouse maximum energy',1200, 100, 10000, 100),
        #mouse_position = UserSettableParameter('choice', 'Mouse initial position',value = 'random',choices = ['random', 'in_quadrant']),
        #food_position = UserSettableParameter('choice', 'Mouse initial position',value = 'random',choices = ['random', '(0,0)']),
        predator_position = 'random',
        primary_values = None, 
        secondary_values = None, 
        food_amount_range = (20,400), 
        nutritional_value = [-1, 0.7, 1], 
        food_growth_rate = [1],
        mousebrain_inheritance = False,
        mouse_reproduction = UserSettableParameter('checkbox', 'Mouse reproduction',True),
        brain_iterations_per_step = 10,
        test_veteran = False
)




server = ModularServer(Mouseworld,
                       [space_element, chart_element],
                       "Mouseworld_continuous_space",
                       model_params)
#                       [20,0,0], 20, 10, width = width, height = height)
server.port = 8521 # The default
server.launch()


# In[2]:

#get_ipython().run_cell_magic('writefile', 'mouseworld/my_viz.py', '\n# -*- coding: utf-8 -*-\n"""\nModular Canvas Rendering\n========================\nModule for visualizing model objects in grid cells.\n"""\nfrom collections import defaultdict\nfrom mesa.visualization.ModularVisualization import VisualizationElement\n\n\nclass CanvasGrid(VisualizationElement):\n    """ A CanvasGrid object uses a user-provided portrayal method to generate a\n    portrayal for each object. A portrayal is a JSON-ready dictionary which\n    tells the relevant JavaScript code (GridDraw.js) where to draw what shape.\n    The render method returns a dictionary, keyed on layers, with values as\n    lists of portrayals to draw. Portrayals themselves are generated by the\n    user-provided portrayal_method, which accepts an object as an input and\n    produces a portrayal of it.\n    A portrayal as a dictionary with the following structure:\n        "x", "y": Coordinates for the cell in which the object is placed.\n        "Shape": Can be either "circle", "rect" or "arrowHead"\n            For Circles:\n                "r": The radius, defined as a fraction of cell size. r=1 will\n                     fill the entire cell.\n            For Rectangles:\n                "w", "h": The width and height of the rectangle, which are in\n                          fractions of cell width and height.\n            For arrowHead:\n            "scale": Proportion scaling as a fraction of cell size.\n            "heading_x": represents x direction unit vector.\n            "heading_y": represents y direction unit vector.\n        "Color": The color to draw the shape in; needs to be a valid HTML\n                 color, e.g."Red" or "#AA08F8"\n        "Filled": either "true" or "false", and determines whether the shape is\n                  filled or not.\n        "Layer": Layer number of 0 or above; higher-numbered layers are drawn\n                 above lower-numbered layers.\n        "text": The text to be inscribed inside the Shape. Normally useful for\n                showing the unique_id of the agent.\n        "text_color": The color to draw the inscribed text. Should be given in\n                      conjunction of "text" property.\n    Attributes:\n        portrayal_method: Function which generates portrayals from objects, as\n                          described above.\n        grid_height, grid_width: Size of the grid to visualize, in cells.\n        canvas_height, canvas_width: Size, in pixels, of the grid visualization\n                                     to draw on the client.\n        template: "canvas_module.html" stores the module\'s HTML template.\n    """\n    package_includes = ["GridDraw.js", "CanvasModule.js"]\n    portrayal_method = None  # Portrayal function\n    canvas_width = 500\n    canvas_height = 500\n\n    def __init__(self, portrayal_method, grid_width, grid_height,\n                 canvas_width=500, canvas_height=500):\n        """ Instantiate a new CanvasGrid.\n        Args:\n            portrayal_method: function to convert each object on the grid to\n                              a portrayal, as described above.\n            grid_width, grid_height: Size of the grid, in cells.\n            canvas_height, canvas_width: Size of the canvas to draw in the\n                                         client, in pixels. (default: 500x500)\n        """\n        self.portrayal_method = portrayal_method\n        self.grid_width = grid_width\n        self.grid_height = grid_height\n        self.canvas_width = canvas_width\n        self.canvas_height = canvas_height\n\n        new_element = ("new CanvasModule({}, {}, {}, {})"\n            .format(self.canvas_width, self.canvas_height,\n                self.grid_width, self.grid_height))\n\n        self.js_code = "elements.push(" + new_element + ");"\n\n    def render(self, model):\n        grid_state = defaultdict(list)\n        grid = model.space._grid\n        for x in range(grid.width):\n            for y in range(grid.height):\n                cell_objects = grid.get_cell_list_contents([(x, y)])\n                for obj in cell_objects:\n                    portrayal = self.portrayal_method(obj)\n                    if portrayal:\n                        portrayal["x"] = x\n                        portrayal["y"] = y\n                        grid_state[portrayal["Layer"]].append(portrayal)\n\n        return grid_state')


# In[ ]:



