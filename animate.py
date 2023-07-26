import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# Process "animation.csv" line by line and store in a list of lists
data = []
with open('animation.csv', 'r') as f:
  for line in f:
    # remove '\n':
    line = line[:-1]
    if (len(line) > 0):
      # slice by ';':
      line = line.split(';')
      
      for i in range(len(line)):
          if (len(line[i]) > 0):
            line[i] = line[i][1:-1]
            coordinates = line[i].split(',')
            line[i] = [float(coordinates[0]), float(coordinates[1]), float(coordinates[2])]
            # print(line[i])
      data.append(line[:-1])

# print(data)

# convert to numpy ndarray
data = np.array(data)

# print data dimensions
print(data.shape)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
line, = ax.plot([], [], [], 'o')

def init():
  line.set_data([], [])
  line.set_3d_properties([])
  return line,

def update(frame):
    x = data[frame, :, 0]  # X coordinates of atoms
    y = data[frame, :, 1]  # Y coordinates of atoms
    z = data[frame, :, 2]  # Z coordinates of atoms
    line.set_data(x, y)
    line.set_3d_properties(z)
    return line,

ax.set_xlim(np.min(data[:,:,  0]), np.max(data[:,:, 0]))
ax.set_ylim(np.min(data[:,:, 1]), np.max(data[:,:, 1]))
ax.set_zlim(np.min(data[:,:, 2]), np.max(data[:,:, 2]))

anim = animation.FuncAnimation(fig, update, frames=len(data), init_func=init, blit=True)

plt.show()