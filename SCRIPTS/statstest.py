# Here are some examples of how to use the new Stats capabilitiesself.

from math import *
import numpy as np
from Stats import *
import matplotlib.pyplot as plt

# Create instance of Stats. This has changed now so that you don't have to
# give it anything; those numbers were just deprecated anyway.

St = Stats()

# Lets create a couple random fields to study.
N = 100
x1 = 3.0*np.random.rand(N) + 2.0
y1 = 1.5*np.random.rand(N) + 2.0
x2 = 1.5*np.random.rand(N) - 2.0
y2 = 3.0*np.random.rand(N) - 2.0

bounds = [-4.0,6.0]

print('Created random data.')

# Lets do 2D KDE. Some names have changed, so make sure you check that the
# names are correct if you run any old scripts.
xx1, yy1, f1 = St.Gaussian2DKDE(x1, y1, bounds, bounds)
xx2, yy2, f2 = St.Gaussian2DKDE(x2, y2, bounds, bounds)

print('Initial 2DKDE of random data complete.')

# Lets plot this.
fig1 = plt.figure(1)
plt.contour(xx1, yy1, f1, 20, colors='red')
plt.contour(xx2, yy2, f2, 20, colors='blue')
fig1.savefig('statstest.png')

print('First plot complete.')

# Now lets try the saving capability. You have to specify whether you used
# Gaussian or Fast, since the header contains info about what you're saving.
St.Save2DKDE(xx1, yy1, f1, '2DKDE1.dat', 'Gaussian')
St.Save2DKDE(xx2, yy2, f2, '2DKDE2.dat', 'Gaussian')

print('KDEs saved.')

# Lets check that we can load these.
xx1loaded, yy1loaded, f1loaded = St.Read2DKDE('2DKDE1.dat')
xx2loaded, yy2loaded, f2loaded = St.Read2DKDE('2DKDE2.dat')

print('KDEs loaded.')
# Now plot them so we can make sure we loaded them correctly.
fig2 = plt.figure(2)
plt.contour(xx1loaded, yy1loaded, f1loaded, 20, colors='red')
plt.contour(xx2loaded, yy2loaded, f2loaded, 20, colors='blue')
fig2.savefig('loadtest.png')

print('Second plot complete.')
# Now lets try changing the resolution of the KDE. The first number is your
# desired resolution, and the second is the order of the KDE you are setting
# (either 2 for 2DKDE or 1 for 1DKDE.) The default is 100 for 2D and 500 for 1D.
St.SetKDERes(50, 2)

xx1low, yy1low, f1low = St.Gaussian2DKDE(x1, y1, bounds, bounds)
xx2low, yy2low, f2low = St.Gaussian2DKDE(x2, y2, bounds, bounds)

print('Lower res KDE complete.')

# Finally, plot to see what you get.
fig3 = plt.figure(3)
plt.contour(xx1low, yy1low, f1low, 20, colors='red')
plt.contour(xx2low, yy2low, f2low, 20, colors='blue')
fig3.savefig('lowrestest.png')

print('Third plot complete.')