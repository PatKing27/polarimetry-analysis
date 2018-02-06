#*********************************Nabla.py*************************************#
#
# Author: Patrick King, Date: 02/06/18
#
#
#******************************************************************************#

import numpy                 as     np
from   math                  import *
import scipy.ndimage.filters as     filters
from   Observer              import Observable

class Nabla(object):

    # Constructor.
    def __init__(self,args):
        self.beam       = args[0]
        self.N          = args[1]
        self.boxlen     = args[2]
        self.sigma      = self.beam/(sqrt(8.0*log(2)))
        self.pixelwidth = max(int(self.N*self.sigma/(self.boxlen)), 1)
        self.scale = self.__CalculateScales()

    # This function produces the x,y combinations of pixels that correspond to a
    # circle the width of the pixelwidth chosen to compute our gradients. These
    # combinations tell the gradient functions the locations of the different
    # points away from the central point to calculate the finite differencing
    # with. This enables gradients to be computed at different scales on the
    # POS.
    def __CalculateScales(self):
        scales = []
        l = self.pixelwidth
        for a in range(-l,l+1):
            for b in range(-l,l+1):
                r = np.sqrt(a**2 + b**2)
                if (r <= l + 0.5) and (r >= l - 0.5):
                    scales.append([a,b])
        return scales

    # The inferred magnetic field angle on the plane of the sky (chi) is a
    # quantity specified entirely by the Stokes parameters Q and U. Computing
    # the angle difference must be done understanding that the angle is
    # degenerate with the same POS angle that is pi/2 out of phase with it,
    # making the maximum difference 90 degrees. The difference is efficiently
    # computed directly from the stokes parameters. The result is given in
    # degrees.
    def __DQU(self, Qc, Qi, Uc, Ui):
        return np.rad2deg(0.5*np.arctan2((Qi*Uc - Qc*Ui),(Qi*Qc + Ui*Uc)))

    # This function computes the angular scalar gradient (using the DQU function
    # to properly account for angular differences) for angular quantities.
    def __AngleGradient(self, Q, U, mask):
        q = np.array(Q.data)
        u = np.array(U.data)
        xlim = q.shape[0]
        ylim = q.shape[1]
        S2 = np.zeros((xlim,ylim))
        for i in range(xlim):
            for j in range(ylim):
                num = 0
                if not mask[i,j]:
                    for vec in self.scale:
                        a = vec[0]
                        b = vec[1]
                        if (i + a) > (xlim - 1):
                            ip = i + a - xlim
                            if (j + b) > (ylim - 1):
                                jp  = j + b - ylim
                            elif (j + b) < 0:
                                jp = j + b + ylim
                            else:
                                jp = j + b
                        elif (i + a) < 0:
                            ip = i + a + xlim
                            if (j + b) > (ylim - 1):
                                jp = j + b - ylim
                            elif (j + b) < 0:
                                jp = j + b + ylim
                            else:
                                jp = j + b
                        else:
                            ip = i + a
                            if (j + b) > (ylim - 1):
                                jp = j + b - ylim
                            elif (j + b) < 0:
                                jp = j + b + ylim
                            else:
                                jp = j + b
                        if not mask[ip,jp]:
                            inc = self.__DQU(q[i,j],q[ip,jp],u[i,j],u[ip,jp])**2
                            S2[i,j] += inc
                            num += 1
                if num != 0:
                    S2[i,j] = S2[i,j]/num
        return np.sqrt(S2)

    # This function computes the ordinary scalar gradient, simply computing the
    # squared finite differences at the specified scales for typical POS
    # quantities.
    def __Gradient(self, O):
        v  = O.data
        xlim = v.shape[0]
        ylim = v.shape[1]
        G2 = np.zeros((xlim,ylim))
        for i in range(xlim):
            for j in range(ylim):
                num = 0
                for vec in self.scale:
                    a = vec[0]
                    b = vec[1]
                    if (i + a) > (xlim - 1):
                        ip = i + a - xlim
                        if (j + b) > (ylim - 1):
                            jp = j + b - ylim
                            inc = (v[ip,jp] - v[i,j])**2
                            G2[i,j] += inc
                            num += 1
                        elif (j + b) < 0:
                            jp = j + b + ylim
                            inc = (v[ip,jp] - v[i,j])**2
                            G2[i,j] += inc
                            num += 1
                        else:
                            jp = j + b
                            inc = (v[ip,jp] - v[i,j])**2
                            G2[i,j] += inc
                            num += 1
                    elif (i + a) < 0:
                        ip = i + a + xlim
                        if (j + b) > (ylim - 1):
                            jp = j + b - ylim
                            inc = (v[ip,jp] - v[i,j])**2
                            G2[i,j] += inc
                            num += 1
                        elif (j + b) < 0:
                            jp = j + b + ylim
                            inc = (v[ip,jp] - v[i,j])**2
                            G2[i,j] += inc
                            num += 1
                        else:
                            jp = j + b
                            inc = (v[ip,jp] - v[i,j])**2
                            G2[i,j] += inc
                            num += 1
                    else:
                        ip = i + a
                        if (j + b) > (ylim - 1):
                            jp = j + b - ylim
                            inc = (v[ip,jp] - v[i,j])**2
                            G2[i,j] += inc
                            num += 1
                        elif (j + b) < 0:
                            jp = j + b + ylim
                            inc = (v[ip,jp] - v[i,j])**2
                            G2[i,j] += inc
                            num += 1
                        else:
                            jp = j + b
                            inc = (v[ip,jp] - v[i,j])**2
                            G2[i,j] += inc
                            num += 1
                G2[i,j] = G2[i,j]/num
        return np.sqrt(G2)

    # Compute the Angle Gradient and construct and Observable object, then
    # return it.
    def ComputeAngleGradient(self, Q, U, mask):
        sdata  = self.__AngleGradient(Q, U, mask)
        snorm  = 'log'
        slname = 'Dispersion in Polarization Angles'
        ssname = 'S'
        sunits = 'Degrees'
        sargs  = [sdata,self.N,snorm,slname,ssname,sunits,'gist_heat',Q.ax]
        sargs.append(self.pixelwidth)
        sargs.append(self.N/self.pixelwidth)
        S      = Observable(sargs)
        return S
    
    # Compute S with numpy arrays instead of observables. 
    def OLessComputeAngleGradient(self, Q, U, mask):
        return self.__AngleGradient(Q,U, mask)

    # Compute an ordinary scalar gradient, construct an Observable object, and
    # return it.
    def ComputeGradient(self, O):
        gdata  = self.__Gradient(O)
        gdata = gdata/max(self.pixelwidth,(1.0/self.N))
        gnorm  = 'log'
        glname = O.lname + ' POS Gradient'
        gsname = 'D' + O.sname
        if O.units is not None:
            gunits = O.units + ' pc$^{-1}$'
        else:
            gunits = 'pc$^{-1}$'
        gargs  = [gdata,self.N,gnorm,glname,gsname,gunits,O.colmap,O.ax]
        if self.pixelwidth == 1:
            gargs.append(None)
        else:
            gargs.append(self.pixelwidth)
        gargs.append(self.N/self.pixelwidth)
        G      = Observable(gargs)
        return G
