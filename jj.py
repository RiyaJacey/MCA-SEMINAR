import matplotlib.pyplot as plt
import numpy as np

# create some data
x = np.arange(0, 10, 0.1)
y1 = np.sin(x)
y2 = np.cos(x)

# create the first plot in a new figure
fig1 = plt.figure()
plt.plot(x, y1)
plt.title('Plot 1')

# create the second plot in a new figure
fig2 = plt.figure()
plt.plot(x, y2)
plt.title('Plot 2')

# show the plots
plt.show()