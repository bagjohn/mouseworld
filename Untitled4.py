
# coding: utf-8

# In[3]:

#%%writefile test_ccode.py

import ctypes
print ctypes.CDLL('library.so').square(4) # linux or when mingw used on windows


# In[ ]:



