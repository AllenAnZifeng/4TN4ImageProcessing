import matplotlib.pyplot as plt
import numpy as np

# Generate some example data
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# Create a 1x2 subplot grid
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# Plot the data on the first subplot and add a legend
ax1.plot(x, y1, label='Sine')
ax1.set_title('Sine')
ax1.legend(loc='upper right')

# Plot the data on the second subplot and add a legend
ax2.plot(x, y2, label='Cosine')
ax2.set_title('Cosine')
ax2.legend(loc='upper right')

# Show the plot
plt.show()