#*********************************Smoother.py**********************************#
#
# Author: Patrick King, Date: 02/06/18
#
# Update (PKK) 04/17/18: Ensured compatibility with Observer and Observable
# update. Made Nyquist public. Changed SmoothPolarizationFraction to accomodate
# the more general dust polarization efficiency physics. Added writing
# capability.
#
# Update (PKK) 04/24/18: Bugfixes related to mask operations. Passed testing.
#
#******************************************************************************#

import numpy                 as     np
from   math                  import *
import scipy.ndimage.filters as     filters
from   Observer              import Observable
from   Nabla                 import *
import functools

class Smoother(object):

    # Constructor.
    def __init__(self, args):
        self.fwhm    = args[0]
        self.N       = args[1]
        self.boxlen  = args[2]
        self.order   = args[3]
        self.sigma   = self.fwhm/(sqrt(8.0*log(2)))
        self.hwhm    = self.fwhm/2.0
        self.posres  = max(int(self.N*self.sigma/(self.boxlen)), 1)
        self.poshwhm = max( int(self.N*self.hwhm/(self.boxlen)), 1)
        self.Writer  = Observer([None, self.N, self.boxlen, './'])

    def ChangeOptLabel(self, new_optlabel):
        self.Writer.ChangeOptLabel(new_optlabel)
        return

    def     ChangePath(self, new_path):
        self.Writer.ChangePath(new_path)
        return

    def Nyquist(self, O):
        img       = O.data
        inc       = int(self.posres)
        h         = int(self.posres/2)
        i, j      = np.meshgrid(*map(np.arange, img.shape), indexing = 'ij')
        masks     = [i >= self.N-h,j >= self.N-h,(i+h)%inc != 0,(j+h)%inc != 0]
        masks.append(O.data.mask)
        totmask   = functools.reduce(np.logical_or, masks)
        img       = np.ma.masked_array(img, totmask)
        imgds     = img.compressed()
        O.nyquist = imgds
        return

    def Smooth(self, O):
        dn  = filters.gaussian_filter(O.data,self.posres,self.order,mode='wrap')
        dn = np.ma.masked_array(dn, O.data.mask)
        olst = [dn,
                O.N,
                O.norm,
                O.lname,
                O.sname,
                O.units,
                O.colmap,
                O.axes,
                O.rotation,
                self.poshwhm]
        nobs = Observable(olst)
        self.Nyquist(nobs)
        # Keep bounds the same as simulation resolution, for consistency.
        nobs.bounds = O.bounds
        self.Writer.WriteObservable(nobs)
        return nobs

    def SmoothGradient(self, O):
        Nab = Nabla([self.fwhm,self.N,self.boxlen])
        G   = Nab.ComputeGradient(O, O.data.mask)
        self.Nyquist(G)
        return G

    def SmoothAngleGradient(self, Q, U):
        Nab = Nabla([self.fwhm,self.N,self.boxlen])
        S   = Nab.ComputeAngleGradient(Q,U,Q.data.mask)
        self.Nyquist(S)
        return S

    def SmoothPolarizationFraction(self, Q, U, I):
        pdata = np.sqrt(Q.data**2 + U.data**2)/I.data
        pdata = np.ma.masked_array(pdata, I.data.mask)
        p = Observable([pdata,
                        Q.N,
                        'log',
                        'Polarization Fraction',
                        '$p$',
                        'None',
                        'plasma',
                        Q.axes,
                        Q.rotation,
                        Q.beam])
        self.Nyquist(p)
        self.Writer.WriteObservable(p)
        return p
