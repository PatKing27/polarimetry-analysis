from Observer import *
from Stats import *
import numpy as np
from math import *
import matplotlib.pyplot as plt

St = Stats()

N = 50

x1 = np.random.normal(0.0, 1.0, (N,N))
y1 = np.random.normal(0.0, 1.0, (N,N))

xx1, f1 = St.Gaussian1DKDE(x1, [-2.0, 2.0])
xx2, yy2, f2 = St.Gaussian2DKDE(x1, y1, [-2.0,2.0], [-2.0,2.0])

X = Observable([x1, N, 'linear', 'x', 'x', None, 'viridis',['x','y'], None, None])
Y = Observable([y1, N, 'linear', 'y', 'y', None, 'viridis',['x','y'], None, None])

xx3, f3 = St.Gaussian1DKDE(X, [-2.0, 2.0])
xx4, yy4, f4 = St.Gaussian2DKDE(X, Y, [-2.0, 2.0], [-2.0, 2.0])

print('plotting')
fig = plt.figure()
plt.plot(xx1, f1, 'r-')
plt.plot(xx3, f3, 'b-')
fig.savefig('1dtest.png')
plt.gca()
fig = plt.figure()
plt.contour(xx2, yy2, f2)
plt.contour(xx4, yy4, f4)
fig.savefig('2dtest.png')