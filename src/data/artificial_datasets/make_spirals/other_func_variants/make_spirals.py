import numpy as np
from numpy import pi
import matplotlib.pyplot as plt


def make_spirals(n_samples):
    
    N=n_samples
    theta = np.sqrt(np.random.rand(N))*2*pi # np.linspace(0,2*pi,100)

    r_a = 2*theta + pi
    data_a = np.array([np.cos(theta)*r_a, np.sin(theta)*r_a]).T
    x_a = data_a + np.random.randn(N,2)

    r_b = -2*theta - pi
    data_b = np.array([np.cos(theta)*r_b, np.sin(theta)*r_b]).T
    x_b = data_b + np.random.randn(N,2)

    res_a = np.append(x_a, np.zeros((N,1)), axis=1)
    res_b = np.append(x_b, np.ones((N,1)), axis=1)

    res = np.append(res_a, res_b, axis=0)
    np.random.shuffle(res)

    np.savetxt("result.csv", res, delimiter=",", header="x,y,label", comments="", fmt='%.5f')

    plt.scatter(x_a[:,0],x_a[:,1])
    plt.scatter(x_b[:,0],x_b[:,1])
    plt.show()
    
make_spirals(400)
