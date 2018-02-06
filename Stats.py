#*********************************Stats.py*************************************#
#
# Author: Patrick King, Date: 02/06/18
#
#******************************************************************************#

from   math        import *
import numpy       as     np
import scipy.stats as     stats
from   Observer    import Observable

class Stats(object):
    # Constructor.
    def __init__(self, args):
        self.fitres = args[0]
        self.boots  = args[1]

    def GaussianKDE(self, X, Y, xb, yb):
        xx, yy = np.mgrid[xb[0]:xb[-1]:100j, yb[0]:yb[-1]:100j]
        pos    = np.vstack([xx.ravel(),yy.ravel()])
        vals   = np.vstack([X,Y])
        kernel = stats.gaussian_kde(vals)
        f      = np.reshape(kernel(pos).T, xx.shape)
        return xx, yy, f

    def Gaussian1DKDE(self, X, xb):
        xx     = np.linspace(xb[0],xb[-1],500)
        kernel = stats.gaussian_kde(X)
        f      = np.reshape(kernel(xx), xx.shape)
        return xx, f

    def Gaussian3DKDE(self, X, Y, Z, xb, yb, zb):
        xx, yy, zz = np.mgrid[xb[0]:xb[-1]:50j, yb[0]:yb[-1]:50j,zb[0]:zb[-1]:50j]
        pos        = np.vstack([xx.ravel(),yy.ravel(),zz.ravel()])
        vals       = np.vstack([X,Y,Z])
        kernel     = stats.gaussian_kde(vals)
        f          = np.reshape(kernel(pos).T,xx.shape)
        return xx, yy, zz, f

    def Fast1DKDE(self, X):
        from fastkde import fastKDE
        pdf, axes = fastKDE.pdf(X)
        return axes, pdf

    def Fast2DKDE(self, X, Y):
        from fastkde import fastKDE
        pdf, axes = fastKDE.pdf(X,Y)
        ax1, ax2 = axes
        return ax1, ax2, pdf

    def PCA(self, X, Y):
        covmatrix    = np.cov(X, Y)
        evals, evecs = np.linalg.eig(covmatrix)
        return evals, evecs
