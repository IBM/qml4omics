import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import json


np.random.seed(42)


def generate_points_in_nd_sphere(n_s, dim = 3, radius=1, thresh = 0.9):
    """Generates n random points within a n-d sphere of given radius."""
    cnt = 0
    points = []
    while cnt < n_s:
        pnts = np.random.rand(dim) * 2 * radius - radius
        pnts_nrm = np.linalg.norm(pnts)
        if (pnts_nrm <= radius) & (pnts_nrm >= radius*thresh):
            points.append(pnts)
            cnt += 1
    points = np.asarray(points)
    return points

# parameters to vary across the configurations
N_SAMPLES = list(range(100, 300, 25))
DIM = list(range(5, 15, 5))
RAD = list(range(5, 20, 5))

def my_make_spheres(
    n_s, 
    dim,
    radius,
):
        
    # enumerate all possible combinations of parameters based on ranges above
    configurations = list(itertools.product(*[n_s, dim, radius]))
    print(configurations)
    print(len(configurations))
    count_configs = 1

    dataset_config = {}

    # populate all the configs with the corresponding argument values
    for n_s, n_d, n_r in configurations:
            config = "samples={}, dimensions={}, radius={}".format(
                n_s, n_d, n_r
            )
            print(count_configs)
            radius1 = n_r
            radius2 = radius1 * 0.5
            Xa = generate_points_in_nd_sphere(n_s, dim = n_d, radius=radius1, thresh = 0.9)
            Xb = generate_points_in_nd_sphere(n_s, dim = n_d, radius=radius2, thresh = 0.9)
            X = np.concatenate((Xa, Xb))
            y = [0]*len(Xa) + [1]*len(Xb)

            # # One way to go about this: Iterate over the number of dimensions, and make a dict on the fly
            # sphere_dict = dict()
            # for i in range(X.shape[1]):
            # 	xi = [x[i] for x in X.tolist()]
            # 	sphere_dict["X{}".format(i)] = xi
            # sphere_dict['class'] = y
            # df = pd.DataFrame(sphere_dict)
            # df.to_csv('test.csv', index=0)
            
            print("Configuration {}/{}: {}".format(count_configs, len(configurations), config))
            X_df = pd.DataFrame(X)
            y_dict = {'class':y}
            y_df = pd.DataFrame(y_dict)
            df = pd.concat([X_df, y_df], axis=1)
            with open('spheres_data/dataset_config.json', 'w') as outfile:
                dataset_config.update({'spheres_data-{}.csv'.format(count_configs):
                {
                'n_samples':n_s,
                'dimensions': n_d,
                'radius': n_r}})  
                json.dump(dataset_config, outfile, indent=4) 
            new_dataset = df.to_csv('spheres_data/spheres_data-{}.csv'.format(count_configs), index=False)

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            # ax.scatter(X[:, 0], X[:, 1],X[:,2], c= y, cmap='viridis')
            ax.scatter(X[:, n_d-3], X[:, n_d-2],X[:, n_d-1], c=y, cmap='viridis')
            plt.savefig('spheres_data/spheres_data-{}.png'.format(count_configs))
            count_configs += 1
            # print(X.shape)
            # print(y.shape)
    return

my_make_spheres(
    n_s=N_SAMPLES,
    dim=DIM,
    radius=RAD
)

