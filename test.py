import numpy as np
from math import sin, cos

a=np.array([3,4,2,7,9])
max=max(a)
min=min(a)
print((a-np.mean(a))/(max-min))
