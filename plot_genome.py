
import numpy as np
import matplotlib.pyplot as plt

# Fixing random state for reproducibility


genome = empty_model.initialization_genome
#x = genome[:,0]
# the histogram of the data

fig = plt.figure()
num_genes = genome[0].size
for i in range(num_genes) :
    x = genome[:,i]
    ax = fig.add_subplot(3,3,(i+1))
    n, bins, patches = ax.hist(x, 20, normed=1, facecolor='g', alpha=0.75)


#plt.xlabel('Gene value')
#plt.ylabel('Probability')
#plt.title('Histogram of gene distribution')
#plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
#plt.axis([0, 1, 0, 20])
#plt.grid(True)
plt.show()