import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D

def sigmoid(x):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-x))

def neuron(x, y, w1, w2, b):
    """Computes the output of a single neuron with sigmoid activation."""
    linear_output = w1 * x + w2 * y + b
    return sigmoid(linear_output)

def update(val):
    """Update the 3D plot based on the current slider values."""
    # Get current slider values
    weight_1 = weight_1_slider.val
    weight_2 = weight_2_slider.val
    bias = bias_slider.val

    # Recalculate outputs for the 3D surface plot
    Z = neuron(X, Y, weight_1, weight_2, bias)

    # Clear the old plot and replot the new surface
    ax.clear()
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')

    # Set labels and title
    ax.set_xlabel('Input 1', labelpad=10)
    ax.set_ylabel('Input 2', labelpad=10)
    ax.set_zlabel('Output', labelpad=10)
    ax.set_title('Neuron Output with Sigmoid Activation')

    # Adjust the number of ticks to reduce clutter
    ax.set_xticks(np.linspace(-1, 1, 5))
    ax.set_yticks(np.linspace(-1, 1, 5))
    ax.set_zticks(np.linspace(0, 1, 5))  # Output is now in [0, 1] due to sigmoid

    # Update color bar legend
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Output Value')

    fig.canvas.draw_idle()

# Create input space for the 3D plot
x = np.linspace(-1, 1, 100)
y = np.linspace(-1, 1, 100)
X, Y = np.meshgrid(x, y)

# Initialize figure and 3D axis
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Initialize sliders for weights and bias
ax_weight_1 = plt.axes([0.2, 0.15, 0.65, 0.03], facecolor='lightgoldenrodyellow')
ax_weight_2 = plt.axes([0.2, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
ax_bias = plt.axes([0.2, 0.05, 0.65, 0.03], facecolor='lightgoldenrodyellow')

weight_1_slider = Slider(ax_weight_1, 'Weight 1', -2.0, 2.0, valinit=1.0)
weight_2_slider = Slider(ax_weight_2, 'Weight 2', -2.0, 2.0, valinit=1.0)
bias_slider = Slider(ax_bias, 'Bias', -5.0, 5.0, valinit=0.0)

# Initial plot
Z = neuron(X, Y, 1.0, 1.0, 0.0)
surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')

# Set labels and title
ax.set_xlabel('Input 1', labelpad=10)
ax.set_ylabel('Input 2', labelpad=10)
ax.set_zlabel('Output', labelpad=10)
ax.set_title('Neuron Output with Sigmoid Activation')

# Adjust the number of ticks to reduce clutter
ax.set_xticks(np.linspace(-1, 1, 5))
ax.set_yticks(np.linspace(-1, 1, 5))
ax.set_zticks(np.linspace(0, 1, 5))  # Sigmoid output in range [0, 1]

# Add color bar legend
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Output Value')

# Connect the sliders to the update function
weight_1_slider.on_changed(update)
weight_2_slider.on_changed(update)
bias_slider.on_changed(update)

# Show the plot
plt.show()
