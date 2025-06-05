import numpy as np
import matplotlib.pyplot as plt

def generate_spiral(n_points, n_turns, noise=0.0):
    t = np.linspace(0, n_turns * 2 * np.pi, n_points)
    x = t * np.cos(t) + np.random.normal(0, noise, n_points)
    y = t * np.sin(t) + np.random.normal(0, noise, n_points)
    return np.column_stack((x, y))

X = generate_spiral(1000, 2, noise=0.1)

# plt.scatter(X[:, 0], X[:, 1], s=5)
# plt.show()


def generate_3d_spiral(n_points, n_turns, noise=0.0):
    t = np.linspace(0, n_turns * 2 * np.pi, n_points)
    x = t * np.cos(t) + np.random.normal(0, noise, n_points)
    y = t * np.sin(t) + np.random.normal(0, noise, n_points)
    z = t + np.random.normal(0, noise, n_points)
    
    return np.column_stack((x, y, z)), y

X_3d, y = generate_3d_spiral(1000, 2, noise=0.1)

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2], s=5, c=y)
plt.show()
