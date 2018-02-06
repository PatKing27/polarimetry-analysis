#********************************Smoother.py***********************************#
#
# Author: Patrick King, Date: 02/06/18
#
#******************************************************************************#

import numpy                 as     np
from   math                  import *
import scipy.ndimage.filters as     filters
from   Observer              import Observable
from   Nabla                 import *
import functools

class Smoother(object):

    def __init__(self, args):
        self.fwhm   = args[0]
        self.N      = args[1]
        self.boxlen = args[2]
        self.order  = args[3]
        self.sigma   = self.fwhm/(sqrt(8.0*log(2)))
        self.hwhm    = self.fwhm/2.0
        self.posres  = max(int(self.N*self.sigma/(self.boxlen)), 1)
        self.poshwhm = max( int(self.N*self.hwhm/(self.boxlen)), 1)
        self.nbn     = int((self.N/self.posres))

    def __Nyquist(self, O):
        img       = O.data
        inc       = int(self.posres)
        h         = int(self.posres/2)
        i, j      = np.meshgrid(*map(np.arange, img.shape), indexing = 'ij')
        masks     = [i >= self.N-h,j >= self.N-h,(i+h)%inc != 0,(j+h)%inc != 0]
        totmask   = functools.reduce(np.logical_or, masks)
        img       = np.ma.masked_array(img, totmask)
        imgds     = img.compressed()
        O.nyquist = imgds
        return
        
    def Smooth(self, O):
        dn  = filters.gaussian_filter(O.data,self.posres,self.order,mode='wrap')
        olst = [dn,O.N,O.norm,O.lname,O.sname,O.units,O.colmap,O.ax]
        olst.append(self.poshwhm)
        olst.append(self.nbn)
        nobs = Observable(olst)
        self.__Nyquist(nobs)
        # Keep bounds the same as simulation resolution, for consistency.
        nobs.bounds = O.bounds
        # Keep bins the same as simulation resolution, but update the binnum.
        if nobs.norm == 'log':
            bmin      = np.floor(np.log10(np.min(O.data)))
            bmax      =  np.ceil(np.log10(np.max(O.data)))
            nobs.bins = np.logspace(bmin,bmax,nobs.binnum)
        elif nobs.norm == 'symlog':
            bmin = np.round(np.log10(np.min(np.absolute(O.data))))
            bmax = np.round(np.log10(np.max(np.absolute(O.data))))
            binh        = int((nobs.binnum - 1)/2)
            neg         = -np.logspace(bmax,bmin,binh)
            zero        = np.array([0.0])
            pos         = np.logspace(bmin,bmax,binh)
            nobs.bins   = np.concatenate((neg,zero,pos)).astype(np.double)
        else:
            b = max(np.abs(np.min(O.data)),np.max(O.data))
            if not np.any(O.data < 0.0):
                nobs.bins = np.linspace(0.0,b,nobs.binnum)
            else:
                nobs.bins = np.linspace(-b,b,nobs.binnum)
        return nobs

    def BeamSample(self, O):
        self.__Nyquist(O)

    def SmoothGradient(self, O):
        Nab = Nabla([self.fwhm,self.N,self.boxlen])
        G   = Nab.ComputeGradient(O)
        self.__Nyquist(G)
        return G

    def SmoothAngleGradient(self, Q, U):
        Nab = Nabla([self.fwhm,self.N,self.boxlen])
        S   = Nab.ComputeAngleGradient(Q,U)
        self.__Nyquist(S)
        return S

