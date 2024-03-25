import numpy as np
import matplotlib.pyplot as plt


fig, ax = plt.subplots()

# aspect ratio
ax.set_aspect('equal')

# manual scaling
ax.set(xlim=(-5,5), ylim=(-5,5))

# auto scaling
ax.autoscale()

# draw x,y axes
ax.axvline(x=0, c="grey", label="x=0")
ax.axhline(y=0, c="grey", label="y=0")

# plot title
ax.set_title('Title')

# axis labels
ax.set_xlabel('x_label')
ax.set_ylabel('y_label')

# axis major ticks
ax.set_xticks(np.arange(start=0,stop=1000, step=100)) 
ax.set_yticks(np.arange(start=0,stop=1000, step=100)) 

# axis minor ticks
ax.set_xticks(np.arange(start=0,stop=1000, step=50), minor=True) 
ax.set_yticks(np.arange(start=0,stop=1000, step=50), minor=True)

# show gridlines
ax.grid()

plt.show()