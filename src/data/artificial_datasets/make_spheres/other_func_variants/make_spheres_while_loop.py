import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def generate_points_in_nd_sphere(n, dim = 3, radius=1, thresh = 0.9):
    """Generates n random points within a n-d sphere of given radius."""
    cnt = 0
    points = []
    while cnt < n:
        pnts = np.random.rand(dim) * 2 * radius - radius
        pnts_nrm = np.linalg.norm(pnts)
        if (pnts_nrm <= radius) & (pnts_nrm >= radius*thresh):
            points.append(pnts)
            cnt += 1
    points = np.asarray(points)
    return points

dim = 10
radius1 = 5
radius2 = radius1 * 0.5
n = 500
Xa = generate_points_in_nd_sphere(n, dim = dim, radius=radius1, thresh = 0.9)
Xb = generate_points_in_nd_sphere(n, dim = dim, radius=radius2, thresh = 0.9)
X = np.concatenate((Xa, Xb))
y = [0]*len(Xa) + [1]*len(Xb)

print(X.shape)

# # One way to go about this: Iterate over the number of dimensions, and make a dict on the fly
# sphere_dict = dict()
# for i in range(X.shape[1]):
# 	xi = [x[i] for x in X.tolist()]
# 	sphere_dict["X{}".format(i)] = xi
# sphere_dict["y"] = y
# df = pd.DataFrame(sphere_dict)
# df.to_csv('test.csv', index=0)

# # Lazy man's way of going about this, which could seem like overkill
X_df = pd.DataFrame(X)
y_dict = {'class':y}
y_df = pd.DataFrame(y_dict)
df = pd.concat([X_df, y_df], axis=1)
df.to_csv('spheres_data.csv')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1],X[:,2], c= y, cmap='viridis')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, dim-3], X[:, dim-2],X[:,dim-1], c= y, cmap='viridis')
plt.show()
