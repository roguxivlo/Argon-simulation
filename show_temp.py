# import numpy and matplotlib
import numpy as np
import matplotlib.pyplot as plt

# read data from energy.csv:
data = np.loadtxt('energy.csv', delimiter=',')
# plot data:
plt.plot(data[:,0], data[:,1], 'r-')

# show plot:
plt.show()
