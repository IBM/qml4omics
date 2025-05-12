import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

def generate_points_in_nd_sphere(n, dim = 3, radius=1):
    """Generates n random points within a n-d sphere of given radius."""
    points = np.random.rand(n, dim) * 2 * radius - radius
    # This will generate points for the small sphere (when <= radius), and the large sphere (>=radius*0.9)
    # I assume the 0.9 ensures that there is atleast a litte overlap in the data between both spheres.
    return points[(np.linalg.norm(points, axis=1) <= radius) & (np.linalg.norm(points, axis=1) >= radius*0.9)]
dim_list=[]
radius_list=[]
n_samples = []
for i in range(5, 15):
    dim = i + 1
    radius = i
    Xa = generate_points_in_nd_sphere(2000*i, dim = dim, radius=radius)
    Xb = generate_points_in_nd_sphere(2000*i, dim = dim, radius=0.4*radius)
    X = np.concatenate((Xa, Xb))
    print(X)
    print(X.shape[0])
    n_samples.append(X.shape[0])
    dim_list.append(dim)
    radius_list.append(radius)
    y = [0]*len(Xa) + [1]*len(Xb)
rates = plt.figure()
ax = rates.add_subplot(111, projection='3d')
ax.scatter(n_samples, dim_list, radius_list)
ax.set_xlabel('# of points')
ax.set_ylabel('# of dimensions')
ax.set_zlabel('radius')
plt.show()
# plt.savefig('rates_{}.png'.format(1))

# # One way to go about this: Iterate over the number of dimensions, and make a dict on the fly
# sphere_dict = dict()
# for i in range(X.shape[1]):
# 	xi = [x[i] for x in X.tolist()]
# 	sphere_dict["X{}".format(i)] = xi
# sphere_dict["y"] = y
# df = pd.DataFrame(sphere_dict)
# df.to_csv('test.csv', index=0)

# # Lazy man's way of going about this, which could seem like overkill
    # X_df = pd.DataFrame(X)
    # y_dict = {'class':y}
    # y_df = pd.DataFrame(y_dict)
    # df = pd.concat([X_df, y_df], axis=1)
    # df.to_csv('test_{}.csv'.format(i))


    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(X[:, 0], X[:, 1],X[:,2], c= y, cmap='viridis')
    # plt.savefig('spheres_data_{}.png'.format(i))

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(X[:, 2], X[:, 3],X[:,4], c= y, cmap='viridis')
    # #plt.show()
print(n_samples)
