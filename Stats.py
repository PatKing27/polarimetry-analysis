#*************************************Stats.py*********************************#
#
# Author: Patrick King, Date: 02/06/18
#
# Update (PKK) 04/17/18: Ensured compatibility with Observer and Observable
# update. Folded PCA sorting into PCA method, so that PCA always returns the
# eval/evec pair corresponding to the smaller eigenvalue first. Eliminated
# deprecated attributes, and added variability to the GaussianKDE resolution.
# Eliminated Gaussian3DKDE. Renamed GaussianKDE to Gaussian2DKDE.
#
# Update (PKK) 04/24/18: Added read/write capability to KDE operations. Passed
# testing. TO DO: PCA still needs testing.
#
# Update (PKK) 06/27/18: Adding some consistency checks in the GaussianKDE
# methods to ensure that you can pass it arbitarily shaped arrays, provided
# they are square. Additionally, adding some functionality wherein you can send
# them observables naturally.
#
#******************************************************************************#

from   math        import *
import numpy       as     np
import scipy.stats as     stats
from   Observer    import *

class Stats(object):
    # Constructor.
    def __init__(self):
        self.G1Dres = 500
        self.G2Dres = 100

    # Mutator to change the resolution of the GaussianKDE procedures.
    def SetKDERes(self, res, order):
        assert order in [1, 2]
        if   order == 1:
            self.G1Dres = res
        elif order == 2:
            self.G2Dres = res

    def Gaussian1DKDE(self, X, xb):
        if type(X) is Observable:
            X = X.data
        if len(X.shape) != 1:
            n = 1
            for s in X.shape:
                n *= s
            X = X.reshape(n)
        xx     = np.linspace(xb[0],xb[-1],self.G1Dres)
        kernel = stats.gaussian_kde(X)
        f      = np.reshape(kernel(xx), xx.shape)
        return xx, f

    def Gaussian2DKDE(self, X, Y, xb, yb):
        if type(X) is Observable:
            X = X.data
        if type(Y) is Observable:
            Y = Y.data
        assert X.shape == Y.shape
        if len(X.shape) != 1:
            n = 1
            for s in X.shape:
                n *= s
            X = X.reshape(n)
            Y = Y.reshape(n)
        xx, yy = np.mgrid[xb[0]:xb[-1]:self.G2Dres*1j,
                          yb[0]:yb[-1]:self.G2Dres*1j]
        pos    = np.vstack([xx.ravel(),yy.ravel()])
        vals   = np.vstack([X,Y])
        kernel = stats.gaussian_kde(vals)
        f      = np.reshape(kernel(pos).T, xx.shape)
        return xx, yy, f

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
        sort         = evals.argsort()
        evals.sort()
        evecs        = evecs[:, sort]
        return evals, evecs

    def CorrCoeffs(self, X, Y, return_p):
        r_pearson,  p_pearson  = stats.pearsonr(X, Y)
        r_spearman, p_spearman = stats.spearmanr(X, Y)
        return r_pearson, r_spearman

    def Save2DKDE(self, xx, yy, f, filename, method):
        assert method in ('Gaussian','Fast')
        if method == 'Gaussian':
            datastack = np.vstack((xx.ravel(),
                                   yy.ravel(),
                                    f.ravel()))
            header = str(xx.shape[0])
        elif method == 'Fast':
            header = str(len(xx))
            xx_new, yy_new = np.meshgrid(xx, yy)

            datastack = np.vstack((xx_new.ravel(),
                                   yy_new.ravel(),
                                        f.ravel()))
        np.savetxt(filename,    np.transpose(datastack), header = header,
                   comments='', newline='\n')
        return

    def Read2DKDE(self, source):
        res = np.genfromtxt(source, skip_header=0,  max_rows=1,
                            delimiter =',', autostrip=True,
                            comments=None,  dtype=int)
        datastack = np.genfromtxt(source, skip_header=1)
        xx = datastack[:,0].reshape((res,res))
        yy = datastack[:,1].reshape((res,res))
        f  = datastack[:,2].reshape((res,res))
        return xx, yy, f

    def Save1DKDE(self, x, f, filename):
        datastack = np.vstack((x.ravel(), f.ravel()))
        np.savetxt(filename, np.transpose(datastack), comments='', newline='\n')
        return

    def Read1DKDE(self, source):
        datastack = np.genfromtxt(source)
        x = datastack[:,0]
        f = datastack[:,1]
        return x, f
