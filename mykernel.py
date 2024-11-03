import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

# Sample small image (a simple 2D numpy array)
sample_image = np.array([
    [150, 150, 150, 150, 100],
    [100, 100, 100, 150, 100],
    [100, 100, 100, 150, 100],
    [100, 100, 100, 150, 100],
    [100, 100, 100, 150, 100]
], dtype=np.uint8)

# Vertical edge detection kernel (Sobel operator)
vertical_kernel = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])

# Apply convolution using the vertical edge detection kernel
vertical_edges = convolve2d(sample_image, vertical_kernel, mode='valid')

print(sample_image.shape)
print(vertical_kernel.shape)
print(vertical_edges.shape)
print("---")
print(sample_image)
print("---")
print(vertical_edges)

# Plotting the original and the edge-detected images
plt.figure(figsize=(10, 4))

# Original image
plt.subplot(1, 2, 1)
plt.imshow(sample_image, cmap='gray', vmin=0, vmax=255)
plt.title('Original Image')
plt.xticks(list(range(5)))
plt.yticks(list(range(5)))

# Annotate each cell with its value
for i in range(sample_image.shape[0]):
    for j in range(sample_image.shape[1]):
        plt.text(j, i, str(sample_image[i, j]), ha='center', va='center', color='white')

# Edge-detected image
plt.subplot(1, 2, 2)
plt.imshow(vertical_edges, cmap='gray')
plt.title('Vertical Edges Detected')
plt.xticks(list(range(3)))
plt.yticks(list(range(3)))

# Annotate each cell with its value
for i in range(vertical_edges.shape[0]):
    for j in range(vertical_edges.shape[1]):
        plt.text(j, i, str(-vertical_edges[i, j]), ha='center', va='center', color='white')

plt.tight_layout()
plt.show()
