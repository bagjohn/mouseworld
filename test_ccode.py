
import ctypes
# print ctypes.windll.library.square(4) # windows
print ctypes.CDLL('library.so').square(4) # linux or when mingw used on windows