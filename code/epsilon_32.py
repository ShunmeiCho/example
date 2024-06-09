import numpy as np

x = np.float32(1.0)
while np.float32(1.0) + x > np.float32(1.0):
    x = np.float32(x / np.float32(2))
print(x * np.float32(2))
