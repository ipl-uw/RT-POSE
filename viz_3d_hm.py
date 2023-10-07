import torch
import matplotlib.pyplot as plt
import numpy as np

# Create a dummy 3D tensor

# hms = np.load('7_000226_hm.npy')

# for i in range(hms.shape[0]):
#     hm = hms[i]
#     fig = plt.figure()
#     # ax = fig.gca(projection='3d')
#     # ax.voxels(hm, edgecolor='k')
#     plt.imshow(hm.mean(axis=2))

#     plt.show()
#     plt.close()


import numpy as np
import matplotlib.pyplot as plt

# Define a 3D Gaussian function
def gaussian_3d(x, y, z, mu_x, mu_y, mu_z, sigma_x, sigma_y, sigma_z):
    return (1 / (2 * np.pi * sigma_x * sigma_y * sigma_z)) * \
           np.exp(-0.5 * ((x - mu_x) ** 2 / sigma_x ** 2 +
                         (y - mu_y) ** 2 / sigma_y ** 2 +
                         (z - mu_z) ** 2 / sigma_z ** 2))

# Create a grid of points
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
z = np.linspace(-5, 5, 100)

X, Y, Z = np.meshgrid(x, y, z)

# Set Gaussian parameters (mu and sigma for x, y, and z)
mu_x, mu_y, mu_z = 0, 0, 0
sigma_x, sigma_y, sigma_z = 1, 1, 1

# Calculate Gaussian values for each point in the grid
values = gaussian_3d(X, Y, Z, mu_x, mu_y, mu_z, sigma_x, sigma_y, sigma_z)
values /= values.max()

# Save to file as comma-separated values
with open('data.txt', 'w') as f:
    f.write(','.join(map(str, values.flatten())))

# # Now, visualize it
# fig = plt.figure(figsize=(8, 8))
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(X, Y, Z, c=values.flatten(), cmap='viridis', marker='.')
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# plt.colorbar(ax.scatter(X, Y, Z, c=values.flatten(), cmap='viridis', marker='.'))
# plt.title("3D Gaussian Distribution")
# plt.show()