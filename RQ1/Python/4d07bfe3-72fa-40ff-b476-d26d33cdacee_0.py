import ctypes
import numpy as np

class STRUCT_2(ctypes.Structure):
    _fields_ = [('field_1', ctypes.c_short),
                ('field_2', ctypes.c_short),
                ('field_3', ctypes.c_short),
                ('field_4', ctypes.c_short),
                ('field_5', ctypes.c_short),
                ('field_6', ctypes.c_short),
                ('field_7', ctypes.c_short),
                ('field_8', ctypes.c_short)]

class STRUCT_1(ctypes.Structure):
    _fields_ = [('elements', ctypes.c_short),
                ('STRUCT_ARRAY', ctypes.POINTER(STRUCT_2))]

    def __init__(self, num_of_structs):
        elems = (STRUCT_2 * num_of_structs)()
        self.STRUCT_ARRAY = ctypes.cast(elems, ctypes.POINTER(STRUCT_2))
        self.elements = num_of_structs

        for num in range(0, num_of_structs):
            self.STRUCT_ARRAY[num].field_1 = 1
            self.STRUCT_ARRAY[num].field_2 = 2
            self.STRUCT_ARRAY[num].field_3 = 3
            self.STRUCT_ARRAY[num].field_4 = 4

# Creating the ctypes structure
num_of_structs = 100
test = STRUCT_1(num_of_structs)

# Creating a NumPy array that directly maps to the memory of the ctypes array
np_array = np.ctypeslib.as_array(test.STRUCT_ARRAY, shape=(num_of_structs,)).view(np.recarray)

# Accessing and modifying the NumPy array elements directly affects the underlying ctypes data
for i in range(num_of_structs):
    np_array[i].field_1 = 10
    np_array[i].field_2 = 20

# Printing the changes made in the ctypes array via the NumPy array
for i in range(num_of_structs):
    print(f"Element {i}: field_1 = {test.STRUCT_ARRAY[i].field_1}, field_2 = {test.STRUCT_ARRAY[i].field_2}")
