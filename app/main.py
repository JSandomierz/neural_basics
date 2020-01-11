
from app.wta import WTM
from app.gen import Gen
import app.plot as plt

import numpy as np
import math
import random
#sprawozdanie - implementacja sieci WTM
def __main__():
    gen = Gen()
    gen.addPointGauss(5, 0.08, 0.8, 0.8)
    gen.addPointGauss(10, 0.05, 0.4, 0.4)
    gen.addPointGauss(15, 0.03, 0.2, 0.8)
    gen.addPointGauss(20, 0.06, 0.8, 0.2)
    inputs = [[0.3, 0.3], [0.7, 0.7]]
    inputs = [[x,y] for x,y in zip(gen.points[0], gen.points[1])]
    print("inputs:",inputs)
    steps = 50
    net = WTM(num_neurons=20, num_inputs=2, numSteps=steps)
    for i in range(steps):
        errors = []
        random.shuffle(inputs)
        # print(inputs)
        errors.append(net.train(inputs))
        print("Epoch {}, error {}".format(i, math.fabs(np.sum(errors))))
    netVec = []
    for p in net.weights:
        r = []
        for d in p:
            r.append(d)
        netVec.append(tuple(r))
    print("Weights:",net.weights)
    plt.buildChart(net.weights, gen.points[0], gen.points[1])
__main__()
