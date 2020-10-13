import matplotlib.pyplot as plt
import numpy as np 
depth_list = [5,9,10,56]
plt.scatter(list(range(len(depth_list))), depth_list)
x = np.linspace(0, 10, 30)
y = np.sin(x)

# plt.plot(x, y, 'o', color='black');
plt.show()

