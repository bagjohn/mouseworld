
from mouseworld import mouseworld


# Build the model
model = mouseworld.Mouseworld(10, 5, 100, 40, 100, 100)

# Run for discrete number of timesteps

for i in range(1) :

    model.step()

