
class Input_manager(object):
    """because we need to contain state, the easier way to do that in
    Python is to make a class"""

    def __init__(self):
        self.state = 'wait'
        self.value = 0
        
#     def set_state(x):
#             if x == 'wait' :
#                 current_state = [-1,-1,-1]
#             if x == 'search' :
#                 current_state = [0,-1,-1]
#             if x == 'approach' :
#                 current_state = [-1,0,-1]
#             if x == 'avoid' :
#                 current_state = [-1,-1,0]
#             return current_state
        
    def modify_value(self, modifyer):
        """you can modify the state value or over-write it here
        or you can just modify the state parameter directly"""
        #print("Manage the input here.")
        self.value += modifyer

    def return_state(self, t):
        return self.state
    
    def return_value(self, t):
        return self.value
    
        