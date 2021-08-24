import numpy as np
names=np.array([[1,1,2],[5,6,7],[5,6,7],[1,1,2],[5,8,4],[5,8,4],[3,4,5]])
print(names.shape)
print('去重方案2:\n',np.array(list(set([tuple(t) for t in names]))))