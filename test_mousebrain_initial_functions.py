
# coding: utf-8

# In[7]:

import numpy as np
import matplotlib.pyplot as plt


def main():
    x = np.linspace(0, 1, 1000)
    y = np.exp(-x)

    plt.figure()
    plt.plot(x, y)
    plt.xlabel('$x$')
    plt.ylabel('$\exp(x)$')

    plt.figure()
    plt.plot(x, -np.exp(-x))
    plt.xlabel('$x$')
    plt.ylabel('$-\exp(-x)$')

    plt.show()

if __name__ == '__main__':
    main()


# In[ ]:



