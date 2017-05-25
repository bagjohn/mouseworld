
import numpy as np
import matplotlib.pyplot as plt

agent_counts = np.zeros((empty_model.grid.width, empty_model.grid.height))
for cell in empty_model.grid.coord_iter():
    cell_content, x, y = cell
    agent_count = len(cell_content)
    agent_counts[x][y] = agent_count
plt.imshow(agent_counts, interpolation='nearest')
plt.colorbar()
plt.show()