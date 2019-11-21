import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')

from scipy.spatial import Voronoi, voronoi_plot_2d

def buildChart(neurons, data_X, data_Y):
    #neurons.append(tuple([np.min(data_X), np.min(data_Y)]))
    #neurons.append(tuple([np.max(data_X), np.max(data_Y)]))
    # neurons.append((0,0))
    # neurons.append((1,1))
    vor = Voronoi(neurons)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # voronoi
    voronoi_plot_2d(vor, ax=ax, show_vertices=False)
    nX = []
    nY = []
    for x in neurons:
        nX.append(x[0])
        nY.append(x[1])
    ax.scatter(nX, nY)
    ax.scatter(data_X, data_Y)

    #ax.axis([np.min(data_X)-0.05, np.max(data_X)+0.05, np.min(data_Y)-0.05, np.max(data_Y)+0.05])
    ax.axis([0,1,0,1])
    plt.show()
