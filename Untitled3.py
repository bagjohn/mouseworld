
# coding: utf-8

# In[5]:

from mouseworld.mouse import Input_manager
from mouseworld.mousebrain import Mousebrain
import pickle


b=Input_manager()
a=Mousebrain()
a.build(b)

pickle.dump(a, open( "save.p", "wb" ) )


# In[ ]:



