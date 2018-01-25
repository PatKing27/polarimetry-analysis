#*************************VMAPP Class File: Observer***************************#
#
# Author: Patrick King, Date: 10/11/16
#
# This class defines the Observer object and the Observable object. Each field
# in the pipeline is contained in an Observable object, which associates with it
# a number of important attributes that can be referenced quickly in the data
# structure and interpreted by the other modules. The Observer object is a base
# class for the yt based operations on the simulation data, including reading in
# the data and computing the column integrations.
#
#******************************************************************************#

# Packages: Dependencies include Numpy, Math, yt, VMAPP
import yt
import numpy                 as     np
from   math                  import *
from   yt.units.yt_array     import YTQuantity
from   Rotator               import *

class Observable(object):

    # Constructor for the Observable class. Associates important characteristics
    # of the quantity with the data. Computes bins and bounds from data.
    def __init__(self, args):
        self.data   = args[0]             # symmetric 2d Numpy array of data
        self.N      = args[1]             # length of each axis
        self.norm   = args[2]
        self.lname  = args[3]
        self.sname  = args[4]
        self.units  = args[5]
        self.colmap = args[6]
        self.ax     = args[7]
        self.beam   = args[8]
        self.binnum = args[9]

        assert self.norm in ['log','linear','symlog']

        # Determine bounds and bins from the data.
        if  self.norm == 'log':
            self.data = np.ma.masked_array(self.data, self.data <= 0.0)
            bmin = np.ma.floor(np.ma.log10(np.ma.min(self.data)))
            bmax =  np.ma.ceil(np.ma.log10(np.ma.max(self.data)))
            self.bounds =[10**bmin,10**bmax]
            try:
                self.bins = np.logspace(bmin,bmax,self.binnum)
            except TypeError:
                print('Something wrong.')
        elif self.norm == 'symlog':
            bmin = np.round(np.log10(np.min(np.absolute(self.data))))
            bmax = np.round(np.log10(np.max(np.absolute(self.data))))
            self.bounds = [-10**bmax,10**bmax,10**bmin]
            binh        = int((self.binnum - 1)/2)
            neg         = -np.logspace(bmax,bmin,binh)
            zero        = np.array([0.0])
            pos         = np.logspace(bmin,bmax,binh)
            self.bins   = np.concatenate((neg,zero,pos)).astype(np.double)
        else:
            b = max(np.abs(np.min(self.data)),np.max(self.data))
            if not np.any(self.data < 0.0):
                self.bounds = [0.0,b]
                self.bins = np.linspace(0.0,b,self.binnum)
            else:
                self.bounds = [-b,b]
                self.bins = np.linspace(-b,b,self.binnum)

        # nyquist-sampled, 1d array of data for statistics. Observables are
        # always initialized with simulation resolution, so downsampling is not
        # done yet.
        self.nyquist = self.data.flatten()

    def SetScale(self, bnds):
        bmin = bnds[0]
        bmax = bnds[1]
        if self.norm == 'log':
            self.bounds = bnds
            bmin = np.round(log10(bmin))
            bmax = np.round(log10(bmax))
            self.bins = np.logspace(bmin,bmax,self.binnum)
        elif self.norm == 'linear':
            self.bounds = bnds
            self.bins = np.linspace(bmin,bmax,self.binnum)
        return

class ShortObservable(object):

    def __init__(self, args):
        self.nyquist = args[0]
        self.norm    = args[1]
        self.sname   = args[2]
        self.lname   = args[3]
        self.units   = args[4]
        self.binnum  = args[5]

        #determine bins and bounds from data
        if self.norm == 'log':
            bmin = np.floor(np.ma.log10(np.min(self.nyquist)))
            bmax =  np.ceil(np.ma.log10(np.max(self.nyquist)))
            self.bounds = [bmin,bmax]
            self.bins = np.logspace(bmin,bmax,self.binnum)
        else:
            b = max(np.abs(np.min(self.nyquist)),np.max(self.nyquist))
            if not np.any(self.data < 0.0):
                self.bounds = [0.0,b]
                self.bins = np.linspace(0.0,b,self.binnum)
            else:
                self.bounds = [-b,b]
                self.bins = np.linspace(-b,b,self.binnum)

    def SetScale(self, bnds):
        bmin = bnds[0]
        bmax = bnds[1]
        if self.norm == 'log':
            self.bounds = bnds
            bmin = np.round(log10(bmin))
            bmax = np.round(log10(bmax))
            self.bins = np.logspace(bmin,bmax,self.binnum)
        elif self.norm == 'linear':
            self.bounds = bnds
            self.bins = np.linspace(bmin,bmax,self.binnum)
        return

class Observer(object):
    # Constuctor for the Observer class. Sets some yt parameters and initializes
    # some important class variables.
    def __init__(self, args):
        yt.funcs.mylog.setLevel(50)
        self.src       = args[0]                             # data src path+name
        self.emmindex  = args[1]                             # emm. power law ind
        self.densref   = args[2]                             # max emm. (cm^-3)
        self.denscut   = args[3]                             # sens. lim. (cm^-3)
        self.velcut    = args[4]
        self.poscut    = args[5]
        self.p0        = args[6]                             # max polarization
        self.N         = args[7]                             # sim resolution
        self.boxlen    = args[8]                             # boxlength in pc
        self.ax        = args[9]                             # axes (x,y,z)
        self.rot       = args[10]
        self.reselmt   = self.boxlen*(3.086E18)/self.N       # ds in centimeters

    def __DC2Compute(self, rho, Bx, By, Bz, B2, axes):
        dc2 = 0.0
        if axes == ['x','y']:
            LOSVec = np.array([0,0,1])
        elif axes == ['z','x']:
            LOSVec = np.array([0,1,0])
        else:
            LOSVec = np.array([1,0,0])
        for j in range(len(rho)):
            for i in range(j):
                Bvec1 = np.array([Bx[i],By[i],Bz[i]])
                Bvec2 = np.array([Bx[j],By[j],Bz[j]])
                dc2 += 4.0*rho[i]*rho[j]*(np.dot(np.cross(Bvec1,Bvec2),LOSVec)**2)/(B2[i]*B2[j])
        return dc2

    def Decompose(self):
        ds       = yt.load(self.src)
        ad       = ds.all_data()
        d        = [self.N,self.N,self.N]
        adcg     = ds.covering_grid(level=0,left_edge=[0.0,0.0,0.0],dims=d)
        denscube = adcg['density'].to_ndarray()
        mxcube   = adcg['momentum_x'].to_ndarray()
        mycube   = adcg['momentum_y'].to_ndarray()
        mzcube   = adcg['momentum_z'].to_ndarray()
        Bxcube   = adcg['magnetic_field_x'].to_ndarray()
        Bycube   = adcg['magnetic_field_y'].to_ndarray()
        Bzcube   = adcg['magnetic_field_z'].to_ndarray()
        B2cube   = np.square(Bxcube) + np.square(Bycube) + np.square(Bzcube)
        vzcube   = np.absolute(mzcube/denscube)
        im1      = (denscube >  self.denscut)
        im2      = (vzcube   <= self.velcut )
        intmask = np.logical_and(im1,im2)
        intcube  = intmask.astype(float)*denscube
        if self.poscut is not None:
            imin = int(self.N*self.poscut/self.boxlen)
            imax = int(512 - imin)
            for i in range(imin):
                intcube[:,:,i]            = 0.0
                intcube[:,:,self.N-(i+1)] = 0.0
        if self.emmindex == 0.0:
            emmcube = np.ones((self.N,self.N,self.N))
        else:
            emmcube  = (denscube > self.densref).astype(float)
            emmcube *= np.power(denscube/self.densref,self.emmindex)
            emmcube += (denscube <= self.densref).astype(float)
        if self.rot is not None:
            R = Rotator([self.rot[0],self.rot[1],self.rot[2],self.N, 0])
            intcube  = R.ScalarRotate(intcube)
            emmcube  = R.ScalarRotate(emmcube)
            denscube = R.ScalarRotate(denscube)
            B2cube   = R.ScalarRotate(B2cube)
            mxcube, mycube, mzcube = R.VectorRotate(mxcube,mycube,mzcube)
            Bxcube, Bycube, Bzcube = R.VectorRotate(Bxcube,Bycube,Bzcube)
        if self.ax == ['x','y']:
            N2cube =intcube*emmcube*((np.square(Bxcube)+np.square(Bycube))/B2cube - (2.0/3.0))
            Qcube = emmcube*intcube*((Bycube**2 - Bxcube**2)/B2cube)
            Ucube = emmcube*intcube*(2.0*Bxcube*Bycube/B2cube)
            Gcube = emmcube*intcube*((Bxcube**2 + Bycube**2)/B2cube)
        elif self.ax == ['z','x']:
            N2cube =intcube*emmcube*((np.square(Bzcube)+np.square(Bxcube))/B2cube - (2.0/3.0))
            Qcube = emmcube*intcube*((Bxcube**2 - Bzcube**2)/B2cube)
            Ucube = emmcube*intcube*(2.0*Bzcube*Bxcube/B2cube)
            Gcube = emmcube*intcube*((Bxcube**2 + Bzcube**2)/B2cube)
        else:
            N2cube =intcube*emmcube*((np.square(Bycube)+np.square(Bzcube))/B2cube - (2.0/3.0))
            Qcube = emmcube*intcube*((Bzcube**2 - Bycube**2)/B2cube)
            Ucube = emmcube*intcube*(2.0*Bycube*Bzcube/B2cube)
            Gcube = emmcube*intcube*((Bycube**2 + Bzcube**2)/B2cube)
        CD = np.zeros((self.N,self.N))
        Q  = np.zeros((self.N,self.N))
        U  = np.zeros((self.N,self.N))
        N2 = np.zeros((self.N,self.N))
        G  = np.zeros((self.N,self.N))
        C  = np.zeros((self.N,self.N))
        for j in range(self.N):
            for i in range(self.N):
                if self.ax == ['x','y']:
                    CD[j,i] = np.sum(intcube[i,j,:])
                    Q[j,i]  = np.sum(  Qcube[i,j,:])
                    U[j,i]  = np.sum(  Ucube[i,j,:])
                    N2[j,i] = np.sum( N2cube[i,j,:])
                    G[j,i]  = np.sum(  Gcube[i,j,:])
                    C[j,i]  = self.__DC2Compute(denscube[i,j,:],Bxcube[i,j,:],Bycube[i,j,:],Bzcube[i,j,:],B2cube[i,j,:],self.ax)
                elif self.ax == ['z','x']:
                    CD[j,i] = np.sum(intcube[j,:,i])
                    Q[j,i]  = np.sum(  Qcube[j,:,i])
                    U[j,i]  = np.sum(  Ucube[j,:,i])
                    N2[j,i] = np.sum( N2cube[j,:,i])
                    G[j,i]  = np.sum(  Gcube[j,:,i])
                    C[j,i]  = self.__DC2Compute(denscube[j,:,i],Bxcube[j,:,i],Bycube[j,:,i],Bzcube[j,:,i],B2cube[j,:,i],self.ax)
                else:
                    CD[j,i] = np.sum(intcube[:,i,j])
                    Q[j,i]  = np.sum(  Qcube[:,i,j])
                    U[j,i]  = np.sum(  Ucube[:,i,j])
                    N2[j,i] = np.sum( N2cube[:,i,j])
                    G[j,i]  = np.sum(  Gcube[:,i,j])
                    C[j,i]  = self.__DC2Compute(denscube[:,i,j],Bxcube[:,i,j],Bycube[:,i,j],Bzcube[:,i,j],B2cube[:,i,j],self.ax)

        mask = np.ma.make_mask(CD <= 0.0)
        CD   = np.ma.masked_array(CD, mask)
        Q    = np.ma.masked_array(Q,  mask)
        U    = np.ma.masked_array(U,  mask)
        N2   = np.ma.masked_array(N2, mask)
        CD  *=self.reselmt
        Q   *=self.reselmt
        U   *=self.reselmt
        N2  *=self.reselmt
        G   *=self.reselmt
        I    = CD - self.p0*N2
        p    = self.p0*np.sqrt(np.square(Q)+np.square(U))/I
        I    = np.ma.masked_array(I,   mask)
        p    = np.ma.masked_array(p,   mask)
        G    = np.ma.masked_array(G/I, mask)
        C    = np.ma.masked_array(C/I, mask)
        c1   = 'viridis'
        c2   = 'magma'
        n    = 'log'
        u1   = 'cm$^{-2}$'
        u2   = 'None'
        Is   = 'Stokes I'
        Qs   = 'Stokes Q'
        Us   = 'Stokes U'
        ps   = 'Polarization Fraction'
        Gs   = 'LOS Mean Inclination'
        Cs   = 'LOS Mean Cancellation'
        Gh   = '$\langle \cos^2 \gamma \rangle$'
        Ch   = 'S\langle \sin^2 \theta \cos^2 \beta \rangle$'
        OI   = Observable([I, self.N, n, Is, 'I',   u1, c1, self.ax, None, self.N])
        OQ   = Observable([Q, self.N, n, Qs, 'Q',   u1, c1, self.ax, None, self.N])
        OU   = Observable([U, self.N, n, Us, 'U',   u1, c1, self.ax, None, self.N])
        Op   = Observable([p, self.N, n, ps, '$p$', u2, c2, self.ax, None, self.N])
        OG   = Observable([G, self.N, n, Gs, Gh,    u2, c2, self.ax, None, self.N])
        OC   = Observable([C, self.N, n, Cs, Ch,    u2, c2, self.ax, None, self.N])
        return [OI, OQ, OU, Op, OG, OC]

    def Polarimetry(self):
        ds       = yt.load(self.src)
        ad       = ds.all_data()
        d        = [self.N,self.N,self.N]
        adcg     = ds.covering_grid(level=0,left_edge=[0.0,0.0,0.0],dims=d)
        denscube = adcg['density'].to_ndarray()
        mxcube   = adcg['momentum_x'].to_ndarray()
        mycube   = adcg['momentum_y'].to_ndarray()
        mzcube   = adcg['momentum_z'].to_ndarray()
        Bxcube   = adcg['magnetic_field_x'].to_ndarray()
        Bycube   = adcg['magnetic_field_y'].to_ndarray()
        Bzcube   = adcg['magnetic_field_z'].to_ndarray()
        B2cube   = np.square(Bxcube) + np.square(Bycube) + np.square(Bzcube)
        vzcube   = np.absolute(mzcube/denscube)
        im1      = (denscube >  self.denscut)
        im2      = (vzcube   <= self.velcut )
        intmask = np.logical_and(im1,im2)
        intcube  = intmask.astype(float)*denscube
        if self.poscut is not None:
            imin = int(self.N*self.poscut/self.boxlen)
            imax = int(512 - imin)
            for i in range(imin):
                intcube[:,:,i]            = 0.0
                intcube[:,:,self.N-(i+1)] = 0.0
        if self.emmindex == 0.0:
            emmcube = np.ones((self.N,self.N,self.N))
        else:
            emmcube  = (denscube > self.densref).astype(float)
            emmcube *= np.power(denscube/self.densref,self.emmindex)
            emmcube += (denscube <= self.densref).astype(float)
        if self.rot is not None:
            R = Rotator([self.rot[0],self.rot[1],self.rot[2],self.N, 0])
            intcube  = R.ScalarRotate(intcube)
            emmcube  = R.ScalarRotate(emmcube)
            denscube = R.ScalarRotate(denscube)
            B2cube   = R.ScalarRotate(B2cube)
            mxcube, mycube, mzcube = R.VectorRotate(mxcube,mycube,mzcube)
            Bxcube, Bycube, Bzcube = R.VectorRotate(Bxcube,Bycube,Bzcube)
        if self.ax == ['x','y']:
            N2cube =intcube*emmcube*((np.square(Bxcube)+np.square(Bycube))/B2cube - (2.0/3.0))
            Qcube = emmcube*intcube*((Bycube**2 - Bxcube**2)/B2cube)
            Ucube = emmcube*intcube*(2.0*Bxcube*Bycube/B2cube)
        elif self.ax == ['z','x']:
            N2cube =intcube*emmcube*((np.square(Bzcube)+np.square(Bxcube))/B2cube - (2.0/3.0))
            Qcube = emmcube*intcube*((Bxcube**2 - Bzcube**2)/B2cube)
            Ucube = emmcube*intcube*(2.0*Bzcube*Bxcube/B2cube)
        else:
            N2cube =intcube*emmcube*((np.square(Bycube)+np.square(Bzcube))/B2cube - (2.0/3.0))
            Qcube = emmcube*intcube*((Bzcube**2 - Bycube**2)/B2cube)
            Ucube = emmcube*intcube*(2.0*Bycube*Bzcube/B2cube)
        CD = np.zeros((self.N,self.N))
        Q  = np.zeros((self.N,self.N))
        U  = np.zeros((self.N,self.N))
        N2 = np.zeros((self.N,self.N))
        #for j in range(self.N):
        #    for i in range(self.N):
        #        if self.ax == ['x','y']:
        #            CD[j,i] = np.sum(intcube[i,j,:])
        #            Q[j,i]  = np.sum(Qcube[i,j,:])
        #            U[j,i]  = np.sum(Ucube[i,j,:])
        #            N2[j,i] = np.sum(N2cube[i,j,:])
        #        elif self.ax == ['z','x']:
        #            CD[j,i] = np.sum(intcube[j,:,i])
        #            Q[j,i]  = np.sum(Qcube[j,:,i])
        #            U[j,i]  = np.sum(Ucube[j,:,i])
        #            N2[j,i] = np.sum(N2cube[j,:,i])
        #        else:
        #            CD[j,i] = np.sum(intcube[:,i,j])
        #            Q[j,i]  = np.sum(Qcube[:,i,j])
        #            U[j,i]  = np.sum(Ucube[:,i,j])
        #            N2[j,i] = np.sum(N2cube[:,i,j])
        if self.ax == ['x','y']:
            CD = np.sum(intcube, axis=2).T
            Q  = np.sum(Qcube,   axis=2).T
            U  = np.sum(Ucube,   axis=2).T
            N2 = np.sum(N2cube,  axis=2).T
        elif self.ax == ['z','x']:
            CD = np.sum(intcube, axis=1).T
            Q  = np.sum(Qcube,   axis=1).T
            U  = np.sum(Ucube,   axis=1).T
            N2 = np.sum(N2cube,  axis=1).T
        else:
            CD = np.sum(intcube, axis=0).T
            Q  = np.sum(Qcube,   axis=0).T
            U  = np.sum(Ucube,   axis=0).T
            N2 = np.sum(N2cube,  axis=0).T
        mask = np.ma.make_mask(CD <= 0.0)
        CD = np.ma.masked_array(CD, mask)
        Q  = np.ma.masked_array(Q,  mask)
        U  = np.ma.masked_array(U,  mask)
        N2 = np.ma.masked_array(N2, mask)
        CD *=self.reselmt
        Q  *=self.reselmt
        U  *=self.reselmt
        N2 *=self.reselmt
        I   = CD - self.p0*N2
        I   = np.ma.masked_array(I,  mask)
        P   = np.sqrt(np.square(Q)+np.square(U))
        P   = np.ma.masked_array(P,  mask)
        p   = self.p0*np.sqrt(np.square(Q)+np.square(U))/I
        p   = np.ma.masked_array(p,  mask)
        ch  = np.rad2deg(0.5*(np.pi + np.arctan2(U,Q)))
        ch  = np.ma.masked_array(ch, mask)
        c  = 'viridis'
        n  = 'log'
        u1 = 'cm$^{-2}$'
        u2 = 'None'
        u3 = 'Degrees'
        Is  = 'Stokes I'
        Qs  = 'Stokes Q'
        Us  = 'Stokes U'
        Ps  = 'Polarized Intensity'
        Ns  = 'Column Density'
        N2s = 'Column Inclination Correction'
        ps  = 'Polarization Fraction'
        chs = 'POS Magnetic Field Angle'
        chh = '$\chi$'
        OI  = Observable([I,  self.N, n, Is,  'I',     u1 ,c, self.ax, None, self.N])
        OQ  = Observable([Q,  self.N, n, Qs,  'Q',     u1 ,c, self.ax, None, self.N])
        OU  = Observable([U,  self.N, n, Us,  'U',     u1 ,c, self.ax, None, self.N])
        OCD = Observable([CD,  self.N, n, Ns,  'N',     u1 ,c, self.ax, None, self.N])
        ON2 = Observable([N2, self.N, n, N2s, 'N$_2$', u1 ,c, self.ax, None, self.N])
        OP  = Observable([P,  self.N, n, Ps,  'P',     u1 ,c, self.ax, None, self.N])
        Op  = Observable([p,  self.N, n, ps,  '$p$',   u2 ,'magma', self.ax, None, self.N])
        Och = Observable([ch, self.N, 'linear', chs, chh,     u3 ,c, self.ax, None, self.N])
        return [OI, OQ, OU, OCD, ON2, OP, Op, Och]

    # This method actually performs the computation of the quantities and stores
    # them in an array of Observable objects. As of now, the integration will
    # yield the continuum quantities, I, Q, and U; and the line quantities,
    # M0, M1, and M2.
    def Integrate(self):
        # Load Simulation Data.
        ds = yt.load(self.src)
        ad = ds.all_data()

        # Extract covering grids of data for Numpy-based manipulation.
        d = [self.N, self.N, self.N]
        adcg = ds.covering_grid(level=0,left_edge=[0.0,0.0,0.0],dims=d)

        denscube = adcg['density'].to_ndarray()
        mxcube   = adcg['momentum_x'].to_ndarray()
        mycube   = adcg['momentum_y'].to_ndarray()
        mzcube   = adcg['momentum_z'].to_ndarray()
        Bxcube   = adcg['magnetic_field_x'].to_ndarray()
        Bycube   = adcg['magnetic_field_y'].to_ndarray()
        Bzcube   = adcg['magnetic_field_z'].to_ndarray()
        B2cube   = np.square(Bxcube) + np.square(Bycube) + np.square(Bzcube)

        # intcube - specify what regions we are sensitive to. Currently using a
        # logical or combination of a density cut and a inflow velocity cut
        vzcube  = np.absolute(mzcube/denscube)
        im1     = (denscube >  self.denscut)
        im2     = (vzcube   <= self.velcut )
        intmask = np.logical_and(im1,im2)
        intcube = intmask.astype(float)*denscube

        # emmcube
        if self.emmindex == 0.0:
            emmcube = np.ones((self.N,self.N,self.N))
        else:
            emmcube  = (denscube >  self.densref).astype(float)
            emmcube *= np.power(denscube/self.densref, self.emmindex)
            emmcube += (denscube <= self.densref).astype(float)

        # If we are rotating, use the Rotator to produce the rotated cubes.
        # The Line of Sight is Always assumed to be z.
        if self.rot is not None:
            R = Rotator([self.rot[0],self.rot[1],self.rot[2],self.N,4])
            intcube                = R.ScalarRotate(intcube)
            emmcube                = R.ScalarRotate(emmcube)
            denscube               = R.ScalarRotate(denscube)
            B2cube                 = R.ScalarRotate(B2cube)
            mxcube, mycube, mzcube = R.VectorRotate(mxcube,mycube,mzcube)
            Bxcube, Bycube, Bzcube = R.VectorRotate(Bxcube,Bycube,Bzcube)

        # Produce Stokes Parameters from Rotated Values
        tmp   = (2.0/3.0) - (np.square(Bxcube)+np.square(Bycube))/B2cube
        Icube = intcube*(1.0 + self.p0*emmcube*tmp)
        Qcube = emmcube*intcube*((Bycube**2 - Bxcube**2)/B2cube)
        Ucube = emmcube*intcube*(2.0*Bxcube*Bycube/B2cube)

        # Produce VLOS from Rotated Values
        Vcube = mzcube/denscube

        # Column Integrations: Compute I, Q, U; compute M0, Bz
        I  = np.zeros((self.N,self.N))
        Q  = np.zeros((self.N,self.N))
        U  = np.zeros((self.N,self.N))
        M0 = np.zeros((self.N,self.N))
        M1 = np.zeros((self.N,self.N))
        M2 = np.zeros((self.N,self.N))
        BZ = np.zeros((self.N,self.N))

        for j in range(self.N):
            for i in range(self.N):
                I[j,i]  = np.sum(Icube[i,j,:])
                Q[j,i]  = np.sum(Qcube[i,j,:])
                U[j,i]  = np.sum(Ucube[i,j,:])
                M0[j,i] = np.sum(intcube[i,j,:])
                M1[j,i] = np.sum(intcube[i,j,:]*Vcube[i,j,:])
                BZ[j,i] = np.average(np.absolute(Bzcube[i,j,:]))

        # Mask regions where I, M0 are along sightlines with no sensitivity.
        m1   = np.ma.make_mask(I  <= 0.0)
        m2   = np.ma.make_mask(M0 <= 0.0)
        mask = np.ma.mask_or(m1,m2)

        I  = np.ma.masked_array(I,  mask)
        Q  = np.ma.masked_array(Q,  mask)
        U  = np.ma.masked_array(U,  mask)
        M0 = np.ma.masked_array(M0, mask)
        M1 = np.ma.masked_array(M1, mask)
        BZ = np.ma.masked_array(BZ, mask)

        # NRAO CASA DEFINITION: Divide M1 by M0
        M1 = M1/M0

        # NRAO CASA DEFINITION: Moment 2 calculation
        for j in range(self.N):
            for i in range(self.N):
                v = Vcube[i,j,:] - M1[j,i]
                M2[j,i] = np.sum(intcube[i,j,:]*np.square(v[:]))

        M2 = np.ma.masked_array(M2, mask)

        # NRAO CASA DEFINITION: Correct M2
        M2 = np.sqrt(M2/M0)

        # Correct I, Q, U, M0 units - must multiply by physical length of the
        # resolution element (ds) along the line of sight.
        I  =  I*self.reselmt
        Q  =  Q*self.reselmt
        U  =  U*self.reselmt
        M0 = M0*self.reselmt

        # Compute p and chi - Polarimetric Variables
        p   = self.p0*np.sqrt(np.square(Q)+np.square(U))/I
        p   = np.ma.masked_array(p,  mask)
        ch  = np.rad2deg(0.5*(np.pi + np.arctan2(U,Q)))
        ch  = np.ma.masked_array(ch, mask)

        # Colormap reference variables
        c1 = 'viridis'
        c2 = 'plasma'
        c3 = 'Spectral_r'

        # Norm reference variables
        n1 = 'log'
        n2 = 'symlog'
        n3 = 'linear'

        # Unit reference variables
        u1 = 'cm$^{-2}$'
        u2 = 'km s$^{-1}$'
        u3 = '$\mu$G'
        u4 = 'None'
        u5 = 'Degrees'

        # Names for Observables
        Is  = 'Stokes Parameter I'
        Qs  = 'Stokes Parameter Q'
        Us  = 'Stokes Parameter U'
        M0s = 'Moment 0'
        M1s = 'Moment 1'
        M2s = 'Moment 2'
        Bzs = 'Average LOS Magnetic Field'
        ps  = 'Polarization Fraction'
        chs = 'POS Magnetic Field Angle'
        chh = '$\chi$'

        # Short name for N
        N = self.N

        # Create Observable Objects
        OI  = Observable([I,  N, n1, Is,  'I',     u1 ,c1, self.ax, None, N])
        OQ  = Observable([Q,  N, n2, Qs,  'Q',     u1 ,c3, self.ax, None, N])
        OU  = Observable([U,  N, n2, Us,  'U',     u1 ,c3, self.ax, None, N])
        OM0 = Observable([M0, N, n1, M0s, 'M$_0$', u1 ,c1, self.ax, None, N])
        OM1 = Observable([M1, N, n3, M1s, 'M$_1$', u2 ,c3, self.ax, None, N])
        OM2 = Observable([M2, N, n1, M2s, 'M$_2$', u2 ,c2, self.ax, None, N])
        OBz = Observable([BZ, N, n1, Bzs, 'B$_z$', u3 ,c2, self.ax, None, N])
        Op  = Observable([p,  N, n1, ps,  '$p$',   u4 ,c2, self.ax, None, N])
        Och = Observable([ch, N, n3, chs, chh,     u5 ,c2, self.ax, None, N])

        # Return list containing observables objects.
        return [OI, OQ, OU, OM0, OM1, OM2, OBz, Op, Och]

    def ExpPolarimetry(self):
        ds       = yt.load(self.src)
        ad       = ds.all_data()
        d        = [self.N,self.N,self.N]
        adcg     = ds.covering_grid(level=0,left_edge=[0.0,0.0,0.0],dims=d)
        denscube = adcg['density'].to_ndarray()
        mxcube   = adcg['momentum_x'].to_ndarray()
        mycube   = adcg['momentum_y'].to_ndarray()
        mzcube   = adcg['momentum_z'].to_ndarray()
        Bxcube   = adcg['magnetic_field_x'].to_ndarray()
        Bycube   = adcg['magnetic_field_y'].to_ndarray()
        Bzcube   = adcg['magnetic_field_z'].to_ndarray()
        vzcube   = np.absolute(mzcube/denscube)
        im1      = (denscube >  self.denscut)
        im2      = (vzcube   <= self.velcut )
        intmask = np.logical_and(im1,im2)
        #**EXPERIMENTAL**
        ## Cancel the ordered field for the colliding flow sims.
        #B2cubetmp = np.square(Bxcube) + np.square(Bycube) + np.square(Bzcube)
        #print(np.sqrt(10.0**np.ma.mean(np.ma.masked_array(np.log10(np.abs(B2cubetmp)),intmask))))
        #print(10.0**np.ma.mean(np.ma.masked_array(np.log10(np.abs(Bxcube)),intmask)))
        #print(10.0**np.ma.mean(np.ma.masked_array(np.log10(np.abs(Bycube)),intmask)))
        #print(10.0**np.ma.mean(np.ma.masked_array(np.log10(np.abs(Bzcube)),intmask)))
        #Bxcube += -(10.0**np.ma.mean(np.ma.masked_array(np.log10(np.abs(Bxcube)),intmask)))
        Bxcube *= 0.0
        Bxcube += 0.001
        B2cube   = np.square(Bxcube) + np.square(Bycube) + np.square(Bzcube)
        #print(np.sqrt(10.0**np.ma.mean(np.ma.masked_array(np.log10(np.abs(B2cube)),intmask))))
        intcube  = intmask.astype(float)*denscube
        if self.poscut is not None:
            imin = int(self.N*self.poscut/self.boxlen)
            imax = int(512 - imin)
            for i in range(imin):
                intcube[:,:,i]            = 0.0
                intcube[:,:,self.N-(i+1)] = 0.0
        if self.emmindex == 0.0:
            emmcube = np.ones((self.N,self.N,self.N))
        else:
            emmcube  = (denscube > self.densref).astype(float)
            emmcube *= np.power(denscube/self.densref,self.emmindex)
            emmcube += (denscube <= self.densref).astype(float)
        if self.rot is not None:
            R = Rotator([self.rot[0],self.rot[1],self.rot[2],self.N, 0])
            intcube  = R.ScalarRotate(intcube)
            emmcube  = R.ScalarRotate(emmcube)
            denscube = R.ScalarRotate(denscube)
            B2cube   = R.ScalarRotate(B2cube)
            mxcube, mycube, mzcube = R.VectorRotate(mxcube,mycube,mzcube)
            Bxcube, Bycube, Bzcube = R.VectorRotate(Bxcube,Bycube,Bzcube)
        if self.ax == ['x','y']:
            N2cube =intcube*emmcube*((np.square(Bxcube)+np.square(Bycube))/B2cube - (2.0/3.0))
            Qcube = emmcube*intcube*((Bycube**2 - Bxcube**2)/B2cube)
            Ucube = emmcube*intcube*(2.0*Bxcube*Bycube/B2cube)
        elif self.ax == ['z','x']:
            N2cube =intcube*emmcube*((np.square(Bzcube)+np.square(Bxcube))/B2cube - (2.0/3.0))
            Qcube = emmcube*intcube*((Bxcube**2 - Bzcube**2)/B2cube)
            Ucube = emmcube*intcube*(2.0*Bzcube*Bxcube/B2cube)
        else:
            N2cube =intcube*emmcube*((np.square(Bycube)+np.square(Bzcube))/B2cube - (2.0/3.0))
            Qcube = emmcube*intcube*((Bzcube**2 - Bycube**2)/B2cube)
            Ucube = emmcube*intcube*(2.0*Bycube*Bzcube/B2cube)
        CD = np.zeros((self.N,self.N))
        Q  = np.zeros((self.N,self.N))
        U  = np.zeros((self.N,self.N))
        N2 = np.zeros((self.N,self.N))
        for j in range(self.N):
            for i in range(self.N):
                if self.ax == ['x','y']:
                    CD[j,i] = np.sum(intcube[i,j,:])
                    Q[j,i]  = np.sum(Qcube[i,j,:])
                    U[j,i]  = np.sum(Ucube[i,j,:])
                    N2[j,i] = np.sum(N2cube[i,j,:])
                elif self.ax == ['z','x']:
                    CD[j,i] = np.sum(intcube[j,:,i])
                    Q[j,i]  = np.sum(Qcube[j,:,i])
                    U[j,i]  = np.sum(Ucube[j,:,i])
                    N2[j,i] = np.sum(N2cube[j,:,i])
                else:
                    CD[j,i] = np.sum(intcube[:,i,j])
                    Q[j,i]  = np.sum(Qcube[:,i,j])
                    U[j,i]  = np.sum(Ucube[:,i,j])
                    N2[j,i] = np.sum(N2cube[:,i,j])
        mask = np.ma.make_mask(CD <= 0.0)
        CD = np.ma.masked_array(CD, mask)
        Q  = np.ma.masked_array(Q,  mask)
        U  = np.ma.masked_array(U,  mask)
        N2 = np.ma.masked_array(N2, mask)
        CD *=self.reselmt
        Q  *=self.reselmt
        U  *=self.reselmt
        N2 *=self.reselmt
        I   = CD - self.p0*N2
        I   = np.ma.masked_array(I,  mask)
        P   = np.sqrt(np.square(Q)+np.square(U))
        P   = np.ma.masked_array(P,  mask)
        p   = self.p0*np.sqrt(np.square(Q)+np.square(U))/I
        p   = np.ma.masked_array(p,  mask)
        ch  = np.rad2deg(0.5*(np.pi + np.arctan2(U,Q)))
        ch  = np.ma.masked_array(ch, mask)
        c  = 'viridis'
        n  = 'log'
        u1 = 'cm$^{-2}$'
        u2 = 'None'
        u3 = 'Degrees'
        Is  = 'Stokes I'
        Qs  = 'Stokes Q'
        Us  = 'Stokes U'
        Ps  = 'Polarized Intensity'
        Ns  = 'Column Density'
        N2s = 'Column Inclination Correction'
        ps  = 'Polarization Fraction'
        chs = 'POS Magnetic Field Angle'
        chh = '$\chi$'
        OI  = Observable([I,  self.N, n, Is,  'I',     u1 ,c, self.ax, None, self.N])
        OQ  = Observable([Q,  self.N, n, Qs,  'Q',     u1 ,c, self.ax, None, self.N])
        OU  = Observable([U,  self.N, n, Us,  'U',     u1 ,c, self.ax, None, self.N])
        OCD = Observable([CD,  self.N, n, Ns,  'N',     u1 ,c, self.ax, None, self.N])
        ON2 = Observable([N2, self.N, n, N2s, 'N$_2$', u1 ,c, self.ax, None, self.N])
        OP  = Observable([P,  self.N, n, Ps,  'P',     u1 ,c, self.ax, None, self.N])
        Op  = Observable([p,  self.N, n, ps,  '$p$',   u2 ,'magma', self.ax, None, self.N])
        Och = Observable([ch, self.N, n, chs, chh,     u3 ,c, self.ax, None, self.N])
        return [OI, OQ, OU, OCD, ON2, OP, Op, Och]

    def Gamma(self):
        ds       = yt.load(self.src)
        ad       = ds.all_data()
        d        = [self.N,self.N,self.N]
        adcg     = ds.covering_grid(level=0,left_edge=[0.0,0.0,0.0],dims=d)
        denscube = adcg['density'].to_ndarray()
        mxcube   = adcg['momentum_x'].to_ndarray()
        mycube   = adcg['momentum_y'].to_ndarray()
        mzcube   = adcg['momentum_z'].to_ndarray()
        Bxcube   = adcg['magnetic_field_x'].to_ndarray()
        Bycube   = adcg['magnetic_field_y'].to_ndarray()
        Bzcube   = adcg['magnetic_field_z'].to_ndarray()
        B2cube   = np.square(Bxcube) + np.square(Bycube) + np.square(Bzcube)
        vzcube   = np.absolute(mzcube/denscube)
        im1      = (denscube >  self.denscut)
        im2      = (vzcube   <= self.velcut )
        intmask = np.logical_and(im1,im2)
        intcube  = intmask.astype(float)*denscube
        if self.poscut is not None:
            imin = int(self.N*self.poscut/self.boxlen)
            imax = int(512 - imin)
            for i in range(imin):
                intcube[:,:,i]            = 0.0
                intcube[:,:,self.N-(i+1)] = 0.0
        if self.emmindex == 0.0:
            emmcube = np.ones((self.N,self.N,self.N))
        else:
            emmcube  = (denscube > self.densref).astype(float)
            emmcube *= np.power(denscube/self.densref,self.emmindex)
            emmcube += (denscube <= self.densref).astype(float)
        if self.rot is not None:
            R = Rotator([self.rot[0],self.rot[1],self.rot[2],self.N, 0])
            intcube  = R.ScalarRotate(intcube)
            emmcube  = R.ScalarRotate(emmcube)
            denscube = R.ScalarRotate(denscube)
            B2cube   = R.ScalarRotate(B2cube)
            mxcube, mycube, mzcube = R.VectorRotate(mxcube,mycube,mzcube)
            Bxcube, Bycube, Bzcube = R.VectorRotate(Bxcube,Bycube,Bzcube)
        if self.ax == ['x','y']:
            N2cube =intcube*emmcube*((np.square(Bxcube)+np.square(Bycube))/B2cube - (2.0/3.0))
            Qcube = emmcube*intcube*((Bycube**2 - Bxcube**2)/B2cube)
            Ucube = emmcube*intcube*(2.0*Bxcube*Bycube/B2cube)
            Gcube = emmcube*intcube*((Bxcube**2 + Bycube**2)/B2cube)
        elif self.ax == ['z','x']:
            N2cube =intcube*emmcube*((np.square(Bzcube)+np.square(Bxcube))/B2cube - (2.0/3.0))
            Qcube = emmcube*intcube*((Bxcube**2 - Bzcube**2)/B2cube)
            Ucube = emmcube*intcube*(2.0*Bzcube*Bxcube/B2cube)
            Gcube = emmcube*intcube*((Bxcube**2 + Bzcube**2)/B2cube)
        else:
            N2cube =intcube*emmcube*((np.square(Bycube)+np.square(Bzcube))/B2cube - (2.0/3.0))
            Qcube = emmcube*intcube*((Bzcube**2 - Bycube**2)/B2cube)
            Ucube = emmcube*intcube*(2.0*Bycube*Bzcube/B2cube)
            Gcube = emmcube*intcube*((Bycube**2 + Bzcube**2)/B2cube)
        CD = np.zeros((self.N,self.N))
        Q  = np.zeros((self.N,self.N))
        U  = np.zeros((self.N,self.N))
        N2 = np.zeros((self.N,self.N))
        G  = np.zeros((self.N,self.N))
        for j in range(self.N):
            for i in range(self.N):
                if self.ax == ['x','y']:
                    CD[j,i] = np.sum(intcube[i,j,:])
                    Q[j,i]  = np.sum(  Qcube[i,j,:])
                    U[j,i]  = np.sum(  Ucube[i,j,:])
                    N2[j,i] = np.sum( N2cube[i,j,:])
                    G[j,i]  = np.sum(  Gcube[i,j,:])
                elif self.ax == ['z','x']:
                    CD[j,i] = np.sum(intcube[j,:,i])
                    Q[j,i]  = np.sum(  Qcube[j,:,i])
                    U[j,i]  = np.sum(  Ucube[j,:,i])
                    N2[j,i] = np.sum( N2cube[j,:,i])
                    G[j,i]  = np.sum(  Gcube[j,:,i])
                else:
                    CD[j,i] = np.sum(intcube[:,i,j])
                    Q[j,i]  = np.sum(  Qcube[:,i,j])
                    U[j,i]  = np.sum(  Ucube[:,i,j])
                    N2[j,i] = np.sum( N2cube[:,i,j])
                    G[j,i]  = np.sum(  Gcube[:,i,j])
        mask = np.ma.make_mask(CD <= 0.0)
        CD = np.ma.masked_array(CD, mask)
        Q  = np.ma.masked_array(Q,  mask)
        U  = np.ma.masked_array(U,  mask)
        N2 = np.ma.masked_array(N2, mask)
        CD *=self.reselmt
        Q  *=self.reselmt
        U  *=self.reselmt
        N2 *=self.reselmt
        G  *=self.reselmt
        I   = CD - self.p0*N2
        I   = np.ma.masked_array(I,  mask)
        P   = np.sqrt(np.square(Q)+np.square(U))
        P   = np.ma.masked_array(P,  mask)
        p   = self.p0*np.sqrt(np.square(Q)+np.square(U))/I
        p   = np.ma.masked_array(p,  mask)
        ch  = np.rad2deg(0.5*(np.pi + np.arctan2(U,Q)))
        ch  = np.ma.masked_array(ch, mask)
        G   = np.ma.masked_array(G/I,  mask)
        c  = 'viridis'
        n  = 'log'
        u1 = 'cm$^{-2}$'
        u2 = 'None'
        u3 = 'Degrees'
        Is  = 'Stokes I'
        Qs  = 'Stokes Q'
        Us  = 'Stokes U'
        Ps  = 'Polarized Intensity'
        Ns  = 'Column Density'
        N2s = 'Column Inclination Correction'
        ps  = 'Polarization Fraction'
        chs = 'POS Magnetic Field Angle'
        Gs  = 'LOS Mean Inclination'
        chh = '$\chi$'
        Gh  = '$\langle \cos^2 \gamma \rangle$'
        OI  = Observable([I,  self.N, n, Is,  'I',     u1 ,c, self.ax, None, self.N])
        OQ  = Observable([Q,  self.N, n, Qs,  'Q',     u1 ,c, self.ax, None, self.N])
        OU  = Observable([U,  self.N, n, Us,  'U',     u1 ,c, self.ax, None, self.N])
        OCD = Observable([CD, self.N, n, Ns,  'N',     u1 ,c, self.ax, None, self.N])
        ON2 = Observable([N2, self.N, n, N2s, 'N$_2$', u1 ,c, self.ax, None, self.N])
        OP  = Observable([P,  self.N, n, Ps,  'P',     u1 ,c, self.ax, None, self.N])
        Op  = Observable([p,  self.N, n, ps,  '$p$',   u2 ,'magma', self.ax, None, self.N])
        Och = Observable([ch, self.N, n, chs, chh,     u3 ,c, self.ax, None, self.N])
        OG  = Observable([G,  self.N, n, Gs,  Gh,      u2, 'magma', self.ax, None, self.N])
        return [OI, OQ, OU, OCD, ON2, OP, Op, Och, OG]
