#**************************VMAPP Class File: Smoother**************************#
#
# Author: Patrick King, Date: 10/17/16
#
# This class defines the Smoother object, which is used to produce observables
# that are convolved with a Gaussian beam, mimicking the effects of a real beam.
# Requires the fwhm beamwidth, in parsecs, for a symmetric gaussian beam, as
# well as the desired order of the gaussian; the pixel scale of the standard
# deviation of the beam (which is what is actually convolved) is computed.
#
#******************************************************************************#

# Packages: Dependencies include Numpy, Math, Scipy, VMAPP
import numpy                 as     np
from   math                  import *
import scipy.ndimage.filters as     filters
from   Observer              import Observable
from   Nabla                 import *
import functools

class Smoother(object):

    # Constructor for the Smoother class. Sets important parameters for a given
    # 'telescopic' beam. Takes the fwhm of the beam and the length of the box in
    # parsecs, the number of pixels along an axis, the order of the gaussian
    # beam, and the binnum of the unsampled phase histograms.
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

    # Private method: sampling of smoothed data for phase histograms. Once the
    # data has been smoothed, it is necessary to adjust the number of samples
    # that one uses in producing a phase histogram, because the information
    # contained in the image has been reduced. This sampling method attempts to
    # reduce the resolution to approximately 4 samples per beam (2 for each
    # beamwidth.)
    #def __Nyquist(self, O):
    #    img       = O.data
    #    inc       = int(self.posres)
    #    h         = int(self.posres/2)
    #    i, j      = np.meshgrid(*map(np.arange, img.shape), indexing = 'ij')
    #    masks     = [i >= self.N-h,j >= self.N-h,(i+h)%inc != 0,(j+h)%inc != 0]
    #    totmask   = functools.reduce(np.logical_or, masks)
    #    img       = np.ma.masked_array(img, totmask)
    #    imgds     = img.compressed()
    #    O.nyquist = imgds

    def __Nyquist(self, O):
        img       = O.data
        #print(str(np.shape(img)))
        #print(str(img.count()))
        inc       = int(self.posres)
        #print(str(self.fwhm))
        #print(str(inc))
        h         = int(self.posres/2)
        i, j      = np.meshgrid(*map(np.arange, img.shape), indexing = 'ij')
        masks     = [i >= self.N-h,j >= self.N-h,(i+h)%inc != 0,(j+h)%inc != 0]
        totmask   = functools.reduce(np.logical_or, masks)
        img       = np.ma.masked_array(img, totmask)
        imgds     = img.compressed()
        O.nyquist = imgds
        #print(str(len(imgds)))

    # This method performs the primary function of the Smoother class. It
    # produces an entirely new observable which is produced from an old one.
    # This new observable inherits several of the old one's characteristics -
    # names, units, colormap, axes. A new binnum, nyquist-sampling, and data are
    # associated with the new object. At the moment, only symmetric beams are
    # allowed.
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
        #self.__SpecialNyquist(O)

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

    def SmoothFraction(self, Q, U, I, p0):
        pdata = p0*np.sqrt(Q.data**2 + U.data**2)/I.data
        axes = Q.ax
        beam = Q.beam
        N = Q.N
        binnum = Q.binnum
        olst = [pdata,N,'log','Polarization Fraction','$p$','None','plasma',axes,beam,binnum]
        p = Observable(olst)
        self.__Nyquist(p)
        return p
