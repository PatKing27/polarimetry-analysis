#***************************VMAPP Class File: Stats****************************#
#
# Author: Patrick King, Date: 11/01/16
#
# This class file defines the Stats object. This contains statistical routines,
# primarily fitting.
#
#******************************************************************************#

from   math        import *
import numpy       as     np
import scipy.stats as     stats
from   sklearn.neighbors import KernelDensity
from   Observer    import Observable

class Stats(object):
    # Constructor.
    def __init__(self, args):
        self.fitres = args[0]
        self.boots  = args[1]

    # Performs a linear regression of the two observables. The scales of the
    # observables determines whether a simple power law (log-log), hybrid
    # linear-log, or simple linear fit is performed. Returns the fit line, the
    # slope of the fit, the intercept of the fit, the coefficient of
    # determination, the p-value of the null (slope = 0) hypothesis, and the
    # standard error of the regression.
    def LinearRegression(self, O1, O2):
        xdata = O1.nyquist
        ydata = O2.nyquist
        xnorm = O1.norm
        ynorm = O2.norm
        xmin  = np.min(O1.bins)
        xmax  = np.max(O1.bins)
        if xnorm == 'log':
            xdata = np.ma.masked_array(xdata, xdata <= 0.0)
            xdata = np.ma.log10(xdata)
            xfit  = np.linspace(log10(xmin),log10(xmax),self.fitres)
            if ynorm == 'log':
                ydata = np.ma.masked_array(ydata, ydata <= 0.0)
                ydata = np.ma.log10(ydata)
                sl, cpt, r, p, er = stats.linregress(xdata,ydata)
                yfit = cpt + sl*xfit
                xfit = 10.0**xfit
                yfit = 10.0**yfit
            else:
                sl, cpt, r, p, er = stats.linregress(xdata,ydata)
                yfit = cpt + sl*xfit
                xfit = 10.0**xfit
        else:
            xfit  = np.linspace(xmin,xmax,self.fitres)
            if ynorm == 'log':
                ydata = np.ma.masked_array(ydata, ydata <= 0.0)
                ydata = np.ma.log10(ydata)
                sl, cpt, r, p, er = stats.linregress(xdata,ydata)
                yfit = cpt + sl*xfit
                yfit = 10.0**yfit
            else:
                sl, cpt, r, p, er = stats.linregress(xdata,ydata)
                yfit = cpt + sl*xfit
        return xfit, yfit, sl, cpt, r, p, er

    def BootstrapError(self, O1, O2):
        N = len(O1.nyquist)
        fits = np.zeros(self.boots)
        for t in range(self.boots):
            indices  = np.floor(np.random.rand(N)*N).astype(int)
            O1resamp = O1.nyquist[indices]
            O2resamp = O2.nyquist[indices]
            O1R = Observable([O1resamp,O1.N,O1.norm,O1.lname,O1.sname,O1.units,O1.colmap,O1.ax,O1.beam,O1.binnum])
            O2R = Observable([O2resamp,O2.N,O2.norm,O2.lname,O2.sname,O2.units,O2.colmap,O2.ax,O2.beam,O2.binnum])
            xfit, yfit, fits[t], cpt, r, p, er = self.LinearRegression(O1R,O2R)
        return np.std(fits)

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

    def SquareResidual(self, X, Y):
        assert np.shape(X) == np.shape(Y)
        residual = np.square(X-Y)
        restotal = np.sum(residual)
        return restotal, residual

    def JensenShannonDistance(self, P, Q):
        from scipy.stats import entropy
        M = 0.5*(P+Q)
        return 0.5*(entropy(P, M) + entropy(Q, M))

    def PCA(self, X, Y):
        covmatrix    = np.cov(X, Y)
        evals, evecs = np.linalg.eig(covmatrix)
        return evals, evecs
