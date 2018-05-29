#********************************Nabla.py**************************************#
#
# Author: Patrick King, Date: 02/06/18
#
# Update (PKK), 04/17/18: Changes made in accord with modifications to Observer
# and Observable. Made AngleGradient and Gradient public for numpy arrays.
# Added mask support for Gradient. Added write support for new Gradient and
# AngleGradient Observables.
#
# Update (PKK) 04/24/18: Bugfixes related to mask operations. Passed testing.
#
#******************************************************************************#

import numpy                 as     np
from   math                  import *
import scipy.ndimage.filters as     filters
from   Observer              import *

class Nabla(object):

    # Constructor for the Nabla class.
    def __init__(self,args):
        self.beam       = args[0]
        if self.beam is None:
            self.beam = 0.0
        self.N          = args[1]
        self.boxlen     = args[2]
        self.sigma      = self.beam/(sqrt(8.0*log(2)))
        self.pixelwidth = max(int(self.N*self.sigma/(self.boxlen)), 1)
        self.scale      = self.__CalculateScales()
        self.path       = './'
        self.Writer = Observer([None, self.N, self.boxlen, './'])

    # Mutator to change the optional label for saving.
    def ChangeOptLabel(self, new_optlabel):
        Writer.ChangeOptLabel(new_optlabel)
        return

    # Mutator to change the path to save.
    def     ChangePath(self, new_path):
        Writer.ChangePath(new_path)
        return

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
    def AngleGradient(self, Q, U, mask):
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
        return np.ma.masked_array(np.sqrt(S2),mask)

    # This function computes the ordinary scalar gradient, simply computing the
    # squared finite differences at the specified scales for typical POS
    # quantities.
    def      Gradient(self, O, mask):
        v  = O.data
        xlim = v.shape[0]
        ylim = v.shape[1]
        G2 = np.zeros((xlim,ylim))
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
        return np.ma.masked_array(np.sqrt(G2),mask)

    # Compute the Angle Gradient and construct and Observable object, then
    # return it.
    def ComputeAngleGradient(self, Q, U, mask):
        sdata  = self.AngleGradient(Q, U, mask)
        sargs  = [sdata,
                  self.N,
                  'log',
                  'Dispersion in Polarization Angles',
                  '$S$',
                  'Degrees',
                  'gist_heat',
                  Q.axes,
                  Q.rotation]
        if self.pixelwidth == 1:
            sargs.append(None)
        else:
            sargs.append(self.pixelwidth)
        S = Observable(sargs)
        self.Writer.WriteObservable(S)
        return S

    # Compute an ordinary scalar gradient, construct an Observable object, and
    # return it.
    def      ComputeGradient(self, O, mask):
        gdata  = self.Gradient(O, mask)
        gdata = gdata/max(self.pixelwidth,(1.0/self.N))
        if O.units is not None:
            gunits = O.units + ' pc$^{-1}$'
        else:
            gunits = 'pc$^{-1}$'
        gargs  = [gdata,
                  self.N,
                  'log',
                  O.lname + ' POS Gradient',
                  'D' + O.sname,
                  gunits,
                  O.colmap,
                  O.axes,
                  O.rotation]
        if self.pixelwidth == 1:
            gargs.append(None)
        else:
            gargs.append(self.pixelwidth)
        G = Observable(gargs)
        self.Writer.WriteObservable(G)
        return G
