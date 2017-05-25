
# coding: utf-8

# In[3]:

get_ipython().run_cell_magic('writefile', 'mouseworld/mytime.py', '\n# -*- coding: utf-8 -*-\n"""\nMesa Time Module\n================\n\nObjects for handling the time component of a model. In particular, this module\ncontains Schedulers, which handle agent activation. A Scheduler is an object\nwhich controls when agents are called upon to act, and when.\n\nThe activation order can have a serious impact on model behavior, so it\'s\nimportant to specify it explicitly. Example simple activation regimes include\nactivating all agents in the same order every step, shuffling the activation\norder every time, activating each agent *on average* once per step, and more.\n\nKey concepts:\n    Step: Many models advance in \'steps\'. A step may involve the activation of\n    all agents, or a random (or selected) subset of them. Each agent in turn\n    may have their own step() method.\n\n    Time: Some models may simulate a continuous \'clock\' instead of discrete\n    steps. However, by default, the Time is equal to the number of steps the\n    model has taken.\n\n\nTODO: Have the schedulers use the model\'s randomizer, to keep random number\nseeds consistent and allow for replication.\n\n"""\nimport random\n\nfrom joblib import Parallel, delayed\nimport multiprocessing\n\n     \n\n\nclass BaseScheduler:\n    """ Simplest scheduler; activates agents one at a time, in the order\n    they were added.\n\n    Assumes that each agent added has a *step* method which takes no arguments.\n\n    (This is explicitly meant to replicate the scheduler in MASON).\n\n    """\n    model = None\n    steps = 0\n    time = 0\n    agents = []\n\n    def __init__(self, model):\n        """ Create a new, empty BaseScheduler. """\n        self.model = model\n        self.steps = 0\n        self.time = 0\n        self.agents = []\n\n    def add(self, agent):\n        """ Add an Agent object to the schedule.\n\n        Args:\n            agent: An Agent to be added to the schedule. NOTE: The agent must\n            have a step() method.\n\n        """\n        self.agents.append(agent)\n\n    def remove(self, agent):\n        """ Remove all instances of a given agent from the schedule.\n\n        Args:\n            agent: An agent object.\n\n        """\n        while agent in self.agents:\n            self.agents.remove(agent)\n\n    def step(self):\n        """ Execute the step of all the agents, one at a time. """\n        for agent in self.agents[:]:\n            agent.step()\n        self.steps += 1\n        self.time += 1\n\n    def get_agent_count(self):\n        """ Returns the current number of agents in the queue. """\n        return len(self.agents)\n    \nclass RandomActivation(BaseScheduler):\n    """ A scheduler which activates each agent once per step, in random order,\n    with the order reshuffled every step.\n\n    This is equivalent to the NetLogo \'ask agents...\' and is generally the\n    default behavior for an ABM.\n\n    Assumes that all agents have a step(model) method.\n\n    """\n    def step(self):\n        """ Executes the step of all agents, one at a time, in\n        random order.\n\n        """\n        random.shuffle(self.agents)\n        \n        for agent in self.agents[:]:\n            agent.step()\n        self.steps += 1\n        self.time += 1\n\nclass ParallelRandomActivation(BaseScheduler):\n    """ A scheduler which activates each agent once per step, in random order,\n    with the order reshuffled every step.\n\n    This is equivalent to the NetLogo \'ask agents...\' and is generally the\n    default behavior for an ABM.\n\n    Assumes that all agents have a step(model) method.\n\n    """\n    def step(self):\n        """ Executes the step of all agents, one at a time, in\n        random order.\n\n        """\n        random.shuffle(self.agents)\n        num_cores = multiprocessing.cpu_count()\n\n        Parallel(n_jobs=num_cores)(delayed(agent.step)() for agent in self.agents[:])\n\n        \n#         for agent in self.agents[:]:\n#             agent.step()\n        self.steps += 1\n        self.time += 1\n\n\nclass SimultaneousActivation(BaseScheduler):\n    """ A scheduler to simulate the simultaneous activation of all the agents.\n\n    This scheduler requires that each agent have two methods: step and advance.\n    step() activates the agent and stages any necessary changes, but does not\n    apply them yet. advance() then applies the changes.\n\n    """\n    def step(self):\n        """ Step all agents, then advance them. """\n        for agent in self.agents[:]:\n            agent.step()\n        for agent in self.agents[:]:\n            agent.advance()\n        self.steps += 1\n        self.time += 1\n\n\nclass StagedActivation(BaseScheduler):\n    """ A scheduler which allows agent activation to be divided into several\n    stages instead of a single `step` method. All agents execute one stage\n    before moving on to the next.\n\n    Agents must have all the stage methods implemented. Stage methods take a\n    model object as their only argument.\n\n    This schedule tracks steps and time separately. Time advances in fractional\n    increments of 1 / (# of stages), meaning that 1 step = 1 unit of time.\n\n    """\n    stage_list = []\n    shuffle = False\n    shuffle_between_stages = False\n    stage_time = 1\n\n    def __init__(self, model, stage_list=["step"], shuffle=False,\n                 shuffle_between_stages=False):\n        """ Create an empty Staged Activation schedule.\n\n        Args:\n            model: Model object associated with the schedule.\n            stage_list: List of strings of names of stages to run, in the\n                         order to run them in.\n            shuffle: If True, shuffle the order of agents each step.\n            shuffle_between_stages: If True, shuffle the agents after each\n                                    stage; otherwise, only shuffle at the start\n                                    of each step.\n\n        """\n        super().__init__(model)\n        self.stage_list = stage_list\n        self.shuffle = shuffle\n        self.shuffle_between_stages = shuffle_between_stages\n        self.stage_time = 1 / len(self.stage_list)\n\n    def step(self):\n        """ Executes all the stages for all agents. """\n        if self.shuffle:\n            random.shuffle(self.agents)\n        for stage in self.stage_list:\n            for agent in self.agents[:]:\n                getattr(agent, stage)()  # Run stage\n            if self.shuffle_between_stages:\n                random.shuffle(self.agents)\n            self.time += self.stage_time\n\n        self.steps += 1')


# In[2]:

get_ipython().run_cell_magic('writefile', 'mouseworld/myspace.py', '\n# -*- coding: utf-8 -*-\n"""\nMesa Space Module\n=================\nObjects used to add a spatial component to a model.\nGrid: base grid, a simple list-of-lists.\nSingleGrid: grid which strictly enforces one object per cell.\nMultiGrid: extension to Grid where each cell is a set of objects.\n"""\n# Instruction for PyLint to suppress variable name errors, since we have a\n# good reason to use one-character variable names for x and y.\n# pylint: disable=invalid-name\n\nimport itertools\nimport random\nimport math\n\n\ndef accept_tuple_argument(wrapped_function):\n    """ Decorator to allow grid methods that take a list of (x, y) position tuples\n    to also handle a single position, by automatically wrapping tuple in\n    single-item list rather than forcing user to do it.\n    """\n    def wrapper(*args):\n        if isinstance(args[1], tuple) and len(args[1]) == 2:\n            return wrapped_function(args[0], [args[1]])\n        else:\n            return wrapped_function(*args)\n    return wrapper\n\n\nclass Grid:\n    """ Base class for a square grid.\n    Grid cells are indexed by [x][y], where [0][0] is assumed to be the\n    bottom-left and [width-1][height-1] is the top-right. If a grid is\n    toroidal, the top and bottom, and left and right, edges wrap to each other\n    Properties:\n        width, height: The grid\'s width and height.\n        torus: Boolean which determines whether to treat the grid as a torus.\n        grid: Internal list-of-lists which holds the grid cells themselves.\n    Methods:\n        get_neighbors: Returns the objects surrounding a given cell.\n        get_neighborhood: Returns the cells surrounding a given cell.\n        get_cell_list_contents: Returns the contents of a list of cells\n            ((x,y) tuples)\n        neighbor_iter: Iterates over position neightbors.\n        coord_iter: Returns coordinates as well as cell contents.\n        place_agent: Positions an agent on the grid, and set its pos variable.\n        move_agent: Moves an agent from its current position to a new position.\n        iter_neighborhood: Returns an iterator over cell coordinates that are\n        in the neighborhood of a certain point.\n        torus_adj: Converts coordinate, handles torus looping.\n        out_of_bounds: Determines whether position is off the grid, returns\n        the out of bounds coordinate.\n        iter_cell_list_contents: Returns an iterator of the contents of the\n        cells identified in cell_list.\n        get_cell_list_contents: Returns a list of the contents of the cells\n        identified in cell_list.\n        remove_agent: Removes an agent from the grid.\n        is_cell_empty: Returns a bool of the contents of a cell.\n    """\n    def __init__(self, width, height, torus):\n        """ Create a new grid.\n        Args:\n            width, height: The width and height of the grid\n            torus: Boolean whether the grid wraps or not.\n        """\n        self.height = height\n        self.width = width\n        self.torus = torus\n\n        self.grid = []\n\n        for x in range(self.width):\n            col = []\n            for y in range(self.height):\n                col.append(self.default_val())\n            self.grid.append(col)\n\n        # Add all cells to the empties list.\n        self.empties = list(itertools.product(\n                            *(range(self.width), range(self.height))))\n\n    @staticmethod\n    def default_val():\n        """ Default value for new cell elements. """\n        return None\n\n    def __getitem__(self, index):\n        return self.grid[index]\n\n    def __iter__(self):\n        # create an iterator that chains the\n        #  rows of grid together as if one list:\n        return itertools.chain(*self.grid)\n\n    def coord_iter(self):\n        """ An iterator that returns coordinates as well as cell contents. """\n        for row in range(self.width):\n            for col in range(self.height):\n                yield self.grid[row][col], row, col    # agent, x, y\n\n    def neighbor_iter(self, pos, moore=True):\n        """ Iterate over position neighbors.\n        Args:\n            pos: (x,y) coords tuple for the position to get the neighbors of.\n            moore: Boolean for whether to use Moore neighborhood (including\n                   diagonals) or Von Neumann (only up/down/left/right).\n        """\n        neighborhood = self.iter_neighborhood(pos, moore=moore)\n        return self.iter_cell_list_contents(neighborhood)\n\n    def iter_neighborhood(self, pos, moore,\n                          include_center=False, radius=1):\n        """ Return an iterator over cell coordinates that are in the\n        neighborhood of a certain point.\n        Args:\n            pos: Coordinate tuple for the neighborhood to get.\n            moore: If True, return Moore neighborhood\n                        (including diagonals)\n                   If False, return Von Neumann neighborhood\n                        (exclude diagonals)\n            include_center: If True, return the (x, y) cell as well.\n                            Otherwise, return surrounding cells only.\n            radius: radius, in cells, of neighborhood to get.\n        Returns:\n            A list of coordinate tuples representing the neighborhood. For\n            example with radius 1, it will return list with number of elements\n            equals at most 9 (8) if Moore, 5 (4) if Von Neumann (if not\n            including the center).\n        """\n        x, y = pos\n        coordinates = set()\n        for dy in range(-radius, radius + 1):\n            for dx in range(-radius, radius + 1):\n                if dx == 0 and dy == 0 and not include_center:\n                    continue\n                # Skip diagonals in Von Neumann neighborhood.\n                if not moore and dy != 0 and dx != 0:\n                    continue\n                # Skip diagonals in Moore neighborhood when distance > radius\n                if moore and radius > 1 and (dy ** 2 + dx ** 2) ** .5 > radius:\n                    continue\n                # Skip if not a torus and new coords out of bounds.\n                if not self.torus and (not (0 <= dx + x < self.width) or\n                                       not (0 <= dy + y < self.height)):\n                    continue\n\n                px = self.torus_adj(x + dx, self.width)\n                py = self.torus_adj(y + dy, self.height)\n\n                # Skip if new coords out of bounds.\n                if(self.out_of_bounds((px, py))):\n                    continue\n\n                coords = (px, py)\n                if coords not in coordinates:\n                    coordinates.add(coords)\n                    yield coords\n\n    def get_neighborhood(self, pos, moore,\n                         include_center=False, radius=1):\n        """ Return a list of cells that are in the neighborhood of a\n        certain point.\n        Args:\n            pos: Coordinate tuple for the neighborhood to get.\n            moore: If True, return Moore neighborhood\n                   (including diagonals)\n                   If False, return Von Neumann neighborhood\n                   (exclude diagonals)\n            include_center: If True, return the (x, y) cell as well.\n                            Otherwise, return surrounding cells only.\n            radius: radius, in cells, of neighborhood to get.\n        Returns:\n            A list of coordinate tuples representing the neighborhood;\n            With radius 1, at most 9 if Moore, 5 if Von Neumann (8 and 4\n            if not including the center).\n        """\n        return list(self.iter_neighborhood(pos, moore, include_center, radius))\n\n    def iter_neighbors(self, pos, moore,\n                       include_center=False, radius=1):\n        """ Return an iterator over neighbors to a certain point.\n        Args:\n            pos: Coordinates for the neighborhood to get.\n            moore: If True, return Moore neighborhood\n                    (including diagonals)\n                   If False, return Von Neumann neighborhood\n                     (exclude diagonals)\n            include_center: If True, return the (x, y) cell as well.\n                            Otherwise,\n                            return surrounding cells only.\n            radius: radius, in cells, of neighborhood to get.\n        Returns:\n            An iterator of non-None objects in the given neighborhood;\n            at most 9 if Moore, 5 if Von-Neumann\n            (8 and 4 if not including the center).\n        """\n        neighborhood = self.iter_neighborhood(\n            pos, moore, include_center, radius)\n        return self.iter_cell_list_contents(neighborhood)\n\n    def get_neighbors(self, pos, moore,\n                      include_center=False, radius=1):\n        """ Return a list of neighbors to a certain point.\n        Args:\n            pos: Coordinate tuple for the neighborhood to get.\n            moore: If True, return Moore neighborhood\n                    (including diagonals)\n                   If False, return Von Neumann neighborhood\n                     (exclude diagonals)\n            include_center: If True, return the (x, y) cell as well.\n                            Otherwise,\n                            return surrounding cells only.\n            radius: radius, in cells, of neighborhood to get.\n        Returns:\n            A list of non-None objects in the given neighborhood;\n            at most 9 if Moore, 5 if Von-Neumann\n            (8 and 4 if not including the center).\n        """\n        return list(self.iter_neighbors(\n            pos, moore, include_center, radius))\n\n    def torus_adj(self, coord, dim_len):\n        """ Convert coordinate, handling torus looping. """\n        if self.torus:\n            coord %= dim_len\n        return coord\n\n    def out_of_bounds(self, pos):\n        """\n        Determines whether position is off the grid, returns the out of\n        bounds coordinate.\n        """\n        x, y = pos\n        return x < 0 or x >= self.width or y < 0 or y >= self.height\n\n    @accept_tuple_argument\n    def iter_cell_list_contents(self, cell_list):\n        """\n        Args:\n            cell_list: Array-like of (x, y) tuples, or single tuple.\n        Returns:\n            An iterator of the contents of the cells identified in cell_list\n        """\n        return (\n            self[x][y] for x, y in cell_list if not self.is_cell_empty((x, y)))\n\n    @accept_tuple_argument\n    def get_cell_list_contents(self, cell_list):\n        """\n        Args:\n            cell_list: Array-like of (x, y) tuples, or single tuple.\n        Returns:\n            A list of the contents of the cells identified in cell_list\n        """\n        return list(self.iter_cell_list_contents(cell_list))\n\n    def move_agent(self, agent, pos):\n        """\n        Move an agent from its current position to a new position.\n        Args:\n            agent: Agent object to move. Assumed to have its current location\n                   stored in a \'pos\' tuple.\n            pos: Tuple of new position to move the agent to.\n        """\n        self._remove_agent(agent.pos, agent)\n        self._place_agent(pos, agent)\n        agent.pos = pos\n\n    def place_agent(self, agent, pos):\n        """ Position an agent on the grid, and set its pos variable. """\n        self._place_agent(pos, agent)\n        agent.pos = pos\n\n    def _place_agent(self, pos, agent):\n        """ Place the agent at the correct location. """\n        x, y = pos\n        self.grid[x][y] = agent\n        if pos in self.empties:\n            self.empties.remove(pos)\n\n    def remove_agent(self, agent):\n        """ Remove the agent from the grid and set its pos variable to None. """\n        pos = agent.pos\n        self._remove_agent(pos, agent)\n        agent.pos = None\n\n    def _remove_agent(self, pos, agent):\n        """ Remove the agent from the given location. """\n        x, y = pos\n        self.grid[x][y] = None\n        self.empties.append(pos)\n\n    def is_cell_empty(self, pos):\n        """ Returns a bool of the contents of a cell. """\n        x, y = pos\n        return True if self.grid[x][y] == self.default_val() else False\n\n    def move_to_empty(self, agent):\n        """ Moves agent to a random empty cell, vacating agent\'s old cell. """\n        pos = agent.pos\n        new_pos = self.find_empty()\n        if new_pos is None:\n            raise Exception("ERROR: No empty cells")\n        else:\n            self._place_agent(new_pos, agent)\n            agent.pos = new_pos\n            self._remove_agent(pos, agent)\n\n    def find_empty(self):\n        """ Pick a random empty cell. """\n        if self.exists_empty_cells():\n            pos = random.choice(self.empties)\n            return pos\n        else:\n            return None\n\n    def exists_empty_cells(self):\n        """ Return True if any cells empty else False. """\n        return len(self.empties) > 0\n\n\nclass SingleGrid(Grid):\n    """ Grid where each cell contains exactly at most one object. """\n    empties = []\n\n    def __init__(self, width, height, torus):\n        """ Create a new single-item grid.\n        Args:\n            width, height: The width and width of the grid\n            torus: Boolean whether the grid wraps or not.\n        """\n        super().__init__(width, height, torus)\n\n    def position_agent(self, agent, x="random", y="random"):\n        """ Position an agent on the grid.\n        This is used when first placing agents! Use \'move_to_empty()\'\n        when you want agents to jump to an empty cell.\n        Use \'swap_pos()\' to swap agents positions.\n        If x or y are positive, they are used, but if "random",\n        we get a random position.\n        Ensure this random position is not occupied (in Grid).\n        """\n        if x == "random" or y == "random":\n            coords = self.find_empty()\n            if coords is None:\n                raise Exception("ERROR: Grid full")\n        else:\n            coords = (x, y)\n        agent.pos = coords\n        self._place_agent(coords, agent)\n\n    def _place_agent(self, pos, agent):\n        if self.is_cell_empty(pos):\n            super()._place_agent(pos, agent)\n        else:\n            raise Exception("Cell not empty")\n\n\nclass MultiGrid(Grid):\n    """ Grid where each cell can contain more than one object.\n    Grid cells are indexed by [x][y], where [0][0] is assumed to be at\n    bottom-left and [width-1][height-1] is the top-right. If a grid is\n    toroidal, the top and bottom, and left and right, edges wrap to each other.\n    Each grid cell holds a set object.\n    Properties:\n        width, height: The grid\'s width and height.\n        torus: Boolean which determines whether to treat the grid as a torus.\n        grid: Internal list-of-lists which holds the grid cells themselves.\n    Methods:\n        get_neighbors: Returns the objects surrounding a given cell.\n    """\n    @staticmethod\n    def default_val():\n        """ Default value for new cell elements. """\n        return set()\n\n    def _place_agent(self, pos, agent):\n        """ Place the agent at the correct location. """\n        x, y = pos\n        self.grid[x][y].add(agent)\n        if pos in self.empties:\n            self.empties.remove(pos)\n\n    def _remove_agent(self, pos, agent):\n        """ Remove the agent from the given location. """\n        x, y = pos\n        self.grid[x][y].remove(agent)\n        if self.is_cell_empty(pos):\n            self.empties.append(pos)\n\n    @accept_tuple_argument\n    def iter_cell_list_contents(self, cell_list):\n        """\n        Args:\n            cell_list: Array-like of (x, y) tuples, or single tuple.\n        Returns:\n            A iterator of the contents of the cells identified in cell_list\n        """\n        return itertools.chain.from_iterable(\n            self[x][y] for x, y in cell_list if not self.is_cell_empty((x, y)))\n\n\nclass ContinuousSpace:\n    """ Continuous space where each agent can have an arbitrary position.\n    Assumes that all agents are point objects, and have a pos property storing\n    their position as an (x, y) tuple. This class uses a MultiGrid internally\n    to store agent objects, to speed up neighborhood lookups.\n    """\n    _grid = None\n\n    def __init__(self, x_max, y_max, torus, x_min=0, y_min=0,\n                 grid_width=100, grid_height=100):\n        """ Create a new continuous space.\n        Args:\n            x_max, y_max: Maximum x and y coordinates for the space.\n            torus: Boolean for whether the edges loop around.\n            x_min, y_min: (default 0) If provided, set the minimum x and y\n                          coordinates for the space. Below them, values loop to\n                          the other edge (if torus=True) or raise an exception.\n            grid_width, _height: (default 100) Determine the size of the\n                                 internal storage grid. More cells will slow\n                                 down movement, but speed up neighbor lookup.\n                                 Probably only fiddle with this if one or the\n                                 other is impacting your model\'s performance.\n        """\n        self.x_min = x_min\n        self.x_max = x_max\n        self.width = x_max - x_min\n        self.y_min = y_min\n        self.y_max = y_max\n        self.height = y_max - y_min\n        self.torus = torus\n\n        self.cell_width = (self.x_max - self.x_min) / grid_width\n        self.cell_height = (self.y_max - self.y_min) / grid_height\n\n        self._grid = MultiGrid(grid_width, grid_height, torus)\n\n    def place_agent(self, agent, pos):\n        """ Place a new agent in the space.\n        Args:\n            agent: Agent object to place.\n            pos: Coordinate tuple for where to place the agent.\n        """\n        pos = self.torus_adj(pos)\n        self._place_agent(pos, agent)\n        agent.pos = pos\n\n    def move_agent(self, agent, pos):\n        """ Move an agent from its current position to a new position.\n        Args:\n            agent: The agent object to move.\n            pos: Coordinate tuple to move the agent to.\n        """\n        pos = self.torus_adj(pos)\n        self._remove_agent(agent.pos, agent)\n        self._place_agent(pos, agent)\n        agent.pos = pos\n\n    def _place_agent(self, pos, agent):\n        """ Place an agent at a given point, and update the internal grid. """\n        cell = self._point_to_cell(pos)\n        self._grid._place_agent(cell, agent)\n        \n    def remove_agent(self, agent):\n        """ Remove the agent from the grid and set its pos variable to None. """\n        pos = agent.pos\n        self._remove_agent(pos, agent)\n        agent.pos = None\n\n    def _remove_agent(self, pos, agent):\n        """ Remove an agent at a given point, and update the internal grid. """\n        cell = self._point_to_cell(pos)\n        self._grid._remove_agent(cell, agent)\n\n    def get_neighbors(self, pos, radius, include_center=True):\n        """ Get all objects within a certain radius.\n        Args:\n            pos: (x,y) coordinate tuple to center the search at.\n            radius: Get all the objects within this distance of the center.\n            include_center: If True, include an object at the *exact* provided\n                            coordinates. i.e. if you are searching for the\n                            neighbors of a given agent, True will include that\n                            agent in the results.\n        """\n        # Get candidate objects\n        scale = max(self.cell_width, self.cell_height)\n        cell_radius = math.ceil(radius / scale)\n        cell_pos = self._point_to_cell(pos)\n        possible_objs = self._grid.get_neighbors(cell_pos,\n                                                 True, True, cell_radius)\n        neighbors = []\n        # Iterate over candidates and check actual distance.\n        for obj in possible_objs:\n            dist = self.get_distance(pos, obj.pos)\n            if dist <= radius and (include_center or dist > 0):\n                neighbors.append(obj)\n        return neighbors\n\n    def get_distance(self, pos_1, pos_2):\n        """ Get the distance between two point, accounting for toroidal space.\n        Args:\n            pos_1, pos_2: Coordinate tuples for both points.\n        """\n        x1, y1 = pos_1\n        x2, y2 = pos_2\n        if not self.torus:\n            dx = x1 - x2\n            dy = y1 - y2\n        else:\n            d_x = abs(x1 - x2)\n            d_y = abs(y1 - y2)\n            dx = min(d_x, self.width - d_x)\n            dy = min(d_y, self.height - d_y)\n        return math.sqrt(dx ** 2 + dy ** 2)\n\n    def torus_adj(self, pos):\n        """ Adjust coordinates to handle torus looping.\n        If the coordinate is out-of-bounds and the space is toroidal, return\n        the corresponding point within the space. If the space is not toroidal,\n        raise an exception.\n        Args:\n            pos: Coordinate tuple to convert.\n        """\n        if not self.out_of_bounds(pos):\n            return pos\n        elif not self.torus:\n            raise Exception("Point out of bounds, and space non-toroidal.")\n        else:\n            x = self.x_min + ((pos[0] - self.x_min) % self.width)\n            y = self.y_min + ((pos[1] - self.y_min) % self.height)\n            return (x, y)\n\n    def _point_to_cell(self, pos):\n        """ Get the cell coordinates that a given x,y point falls in. """\n        if self.out_of_bounds(pos):\n            raise Exception("Point out of bounds.")\n\n        x, y = pos\n        cell_x = math.floor((x - self.x_min) / self.cell_width)\n        cell_y = math.floor((y - self.y_min) / self.cell_height)\n        return (cell_x, cell_y)\n\n    def out_of_bounds(self, pos):\n        """ Check if a point is out of bounds. """\n        x, y = pos\n        return (x < self.x_min or x >= self.x_max or \n        y < self.y_min or y >= self.y_max)\n\n    \n# Custom class following the Repast Simphony ValueLayer and ValueLayerDiffuser classes \n    \nclass Value_layer(Grid) :\n\n    def __init__(self, unique_id, width, height, torus):\n        \n        self.height = height\n        self.width = width\n        self.torus = torus\n        self.unique_id =unique_id\n        \n        self.grid = []\n\n        for x in range(self.width):\n            col = []\n            for y in range(self.height):\n                col.append(self.default_val())\n            self.grid.append(col)\n            \n    @staticmethod\n    def default_val():\n        """ Default value for new cell elements. """\n        return 0\n    \n    def neighbor_avg(self,pos) :\n        val = self.iter_neighbors(pos, moore = True, include_center=False, radius=1)\n        return sum(val)/8\n    \n    def add_value(self, pos, value) :\n        x, y = pos\n        self.grid[x][y] += value\n    \n    # It might be called from Continuous Space, so pos must first be set to grid dims through _point_to_cell\n    def get_value(self, pos) :\n        x, y = pos\n        return self.grid[x][y]\n    \n    def set_value(self, pos, value) :\n        x, y = pos\n        self.grid[x][y] = value\n    \n    def update_grid(self, new_grid):\n        for row in range(self.width):\n            for col in range(self.height):\n                self.grid[row][col] = new_grid[row][col]\n    \n    def _point_to_cell(self, pos):\n        """ Get the cell coordinates that a given x,y point falls in. """\n        if self.out_of_bounds(pos):\n            raise Exception("Point out of bounds.")\n\n        x, y = pos\n        cell_x = math.floor((x - self.x_min) / self.cell_width)\n        cell_y = math.floor((y - self.y_min) / self.cell_height)\n        return (cell_x, cell_y)\n    \n    def diffuse(self, evap_const, diff_const) :\n        old = self\n        for row in range(self.width):\n            for col in range(self.height):\n                self.grid[row][col] = evap_const * (old.grid[row][col] + diff_const * (old.neighbor_avg((row,col)) - old.grid[row][col]))\n                \n#     def diffuse(self, evap_const, diff_const) :\n#         new_grid = np.zeros((self.width, self.height))\n#         for cell in self.coord_iter():\n#             cell_value, x, y = cell\n#             new_grid[x][y] = evap_const * (cell_value + diff_const * (self.neighbor_avg((x,y)) - cell_value))\n#         self.update_grid(new_grid)\n#         return self')


# In[ ]:



