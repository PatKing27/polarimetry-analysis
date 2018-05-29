#********************************Observer.py***********************************#
#
# Author: Patrick King, Date: 02/06/18
#
# Update (PKK) 04/17/18: Greatly changed behavior of the Observer class and the
# behavior of Observables. Eliminated deprecated attributes of the Observable
# class. Added mutator methods for appropriate attributes. Updated Polarimetry
# method. Added Zeeman and VelocityMoments methods. Added read/write Observable
# capabilities. Eliminated the ShortObservable class. Added grain alignment
# efficiency models based on radiative torque theory and grain populations.
# Added significant documentation.
#
# Update (PKK) 04/24/18: Updates to fix observable read/write behavior. Testing
# completed for Constant and Power-Law Dust Grain Emissivity laws. TO DO: need
# to complete testing for MRN, WD01 RATs; verify rotation still works; test
# Zeeman and VelocityMoments calculators.
#
# Update (PKK) 05/11/18: Updates to fix WD01 RATs. Computation facilitated using
# numpy interpolation of the polarization efficiency. Testing completed for MRN
# and WD01 RATs. Added optional plotting of the polarization efficiency as a
# function of both gas density and minimum aligned grain size. More TO DO: add
# output for stacks of 2D observables as FITS files.
#
#******************************************************************************#

import yt
import numpy                 as     np
from   math                  import *
from   yt.units.yt_array     import YTQuantity
from   Rotator               import *

class Observable(object):

    # Constructor for the Observable class. Associates important characteristics
    # of the quantity with the data. Computes bins and bounds from data.
    def __init__(self, args):
        self.data     = args[0]             # symmetric 2d Numpy array of data
        self.N        = args[1]             # integer length per axis
        self.norm     = args[2]             # colorscale norm
        self.lname    = args[3]             # long name for label
        self.sname    = args[4]             # short name (latex style) for label
        self.units    = args[5]             # Associated units
        self.colmap   = args[6]             # desired colormap
        self.axes     = args[7]
        self.rotation = args[8]
        self.beam     = args[9]

        assert self.norm in ['log','linear','symlog']
        assert self.axes in [['x','y'], ['z','x'], ['y','z']]

        # Determine bounds from data and norm.
        if  self.norm == 'log':
            mask = np.logical_or(np.ma.getmask(self.data),self.data <= 0.0)
            self.data.mask = np.ma.nomask
            self.data = np.ma.masked_array(self.data, mask)
            bmin = np.ma.floor(np.ma.log10(np.ma.min(self.data)))
            bmax =  np.ma.ceil(np.ma.log10(np.ma.max(self.data)))
            self.bounds =[10**bmin,10**bmax]
        elif self.norm == 'symlog':
            bmin = np.round(np.log10(np.min(np.absolute(self.data))))
            bmax = np.round(np.log10(np.max(np.absolute(self.data))))
            self.bounds = [-10**bmax,10**bmax,10**bmin]
        else:
            b = max(np.abs(np.min(self.data)),np.max(self.data))
            if not np.any(self.data < 0.0):
                self.bounds = [0.0,b]
            else:
                self.bounds = [-b,b]

        # nyquist-sampled, 1d array of data for statistics. Observables are
        # always initialized with simulation resolution, so downsampling is not
        # done yet.
        self.nyquist = np.ma.compressed(self.data)

    # Mutator to change the bounds of the object manually.
    def   SetBounds(self, new_bnds):
        self.bounds = new_bnds
        return

    # Mutator to change the colormap.
    def SetColormap(self, new_cmap):
        self.colmap = new_cmap
        return

    # Mutator to change the colorscale norm.
    def     SetNorm(self, new_norm):
        assert new_norm in ['log','linear','symlog']
        self.norm = new_norm
        return

class Observer(object):
    # Constructor.
    def __init__(self, args):
        # Suppress yt output.
        yt.funcs.mylog.setLevel(50)
        # Basic arguments.
        self.src      = args[0]                         # data source path+name
        self.N        = args[1]                         # simulation resolution
        self.boxlen   = args[2]                         # length of box in pc
        self.path     = args[3]                         # base destination path
        self.reselmt  = self.boxlen*(3.086E18)/self.N   # ds in centimeters
        self.optlabel = None

        # default yt name handles for fields.
        self.densityhandle   = 'density'
        self.magneticxhandle = 'magnetic_x'
        self.magneticyhandle = 'magnetic_y'
        self.magneticzhandle = 'magnetic_z'
        self.momentumxhandle = 'momentum_x'
        self.momentumyhandle = 'momentum_y'
        self.momentumzhandle = 'momentum_z'

    # Mutator for the units of the box to alter the size of reselmt. This can
    # also effectively change the boxlen. Send the length unit in cm.
    def ChangeLengthUnits(self, new_length):
        self.reselmt = self.boxlen*new_length/self.N
        return

    # Mutator to change the path to save observables in. Must have
    def        ChangePath(self, new_path):
        self.path = new_path
        return

    # Mutator to give an optional label to append to written observables.
    def    ChangeOptLabel(self, new_optlabel):
        self.optlabel = new_optlabel
        return

    # Mutators to change the naming of the density, magnetic, and momentum
    # fields, as yt demands.
    def  ChangeDensityHandle(self, new_handle):
        self.densityhandle = new_handle
        return

    def ChangeMagneticHandle(self, new_handles):
        assert len(new_handles) == 3
        self.magneticxhandle = new_handles[0]
        self.magneticyhandle = new_handles[1]
        self.magneticzhandle = new_handles[2]
        return

    def ChangeMomentumHandle(self, new_handles):
        assert len(new_handles) == 3
        self.momentumxhandle = new_handles[0]
        self.momentumyhandle = new_handles[1]
        self.momentumzhandle = new_handles[2]
        return

    # Helper method to compute the MRN/LD05 polarization efficiency model.
    def   __MRN(self, amin, amax):
        return 2.0*(np.sqrt(amax)-np.sqrt(amin))

    # Helper method to calculate the silicate contribution to the WD01/LD05
    # polarization efficiency model.
    def __WDS01(self, amin, amax, args):
        argscp        = args.copy()
        intres        = argscp.pop()
        a             = 10.0**np.linspace(np.log10(amin),
                                          np.log10(amax),
                                          intres)
        integral      = 0.0
        if amin < amax:
            for i in range(int(intres-1)):
                integral += 0.5*(self.__WDIntegrand(a[i+1], argscp) +
                                 self.__WDIntegrand(a[i],   argscp))* \
                                 (a[i+1]-a[i])
        return integral

    # Helper method to calculate the carbonaceous contribution to the WD01/LD05
    # polarization efficiency model.
    def __WDC01(self, amin, amax, args):
        argscp        = args.copy()
        intres        = argscp.pop()
        Cabund        = argscp.pop()
        a             = 10.0**np.linspace(np.log10(amin),
                                          np.log10(amax),
                                          intres)
        integral      = 0.0
        if amin < amax:
            for i in range(int(intres-1)):
                integral += 0.5*((self.__WDIntegrand(a[i+1], argscp) +
                                  self.__Micrograins(a[i+1], Cabund)) +
                                 (self.__WDIntegrand(a[i],   argscp) +
                                  self.__Micrograins(a[i],   Cabund)))* \
                                  (a[i+1]-a[i])
        return integral

    # Helper method to calculate the integrand for the WD01/LD05 polarization
    # efficiency model, for both silicate and carbonaceous grains.
    def __WDIntegrand(self, a, args):
        if a > args[2]:
            integrand      = (args[2]**3.0)*((a/args[2])**(args[0]+3.0))* \
                             (args[4]/a)*exp(-((a-args[2])/args[3])**3.0)
            if args[1] < 0.0:
                integrand *= np.power((1.0 - args[1]*(a/args[2])), -1.0)
            else:
                integrand *= (1.0 + args[1]*(a/args[2]))
        else:
            integrand      = (args[2]**3.0)*((a/args[2])**(args[0]+3.0))* \
                             (args[4]/a)
            if args[1] < 0.0:
                integrand *= np.power((1.0 - args[1]*(a/args[2])), -1.0)
            else:
                integrand *= (1.0 + args[1]*(a/args[2]))
        return integrand

    # Helper method to calculate the carbonaceous grain population from WD01 at
    # the tiniest scales.
    def __Micrograins(self, a, Cabund):
        B1    = (Cabund/2.0E-5)*4.0787E-7
        B2    = (Cabund/2.0E-5)*1.9105E-10
        term1 = B1*exp(-0.5*(np.log(a/(3.5E-4))/0.4)**2.0)/a
        term2 = B2*exp(-0.5*(np.log(a/(3.0E-3))/0.4)**2.0)/a
        return term1 + term2

    # Base method for computing synthetic polarimetry along specified axis.
    # Writes observables computed at the end. Uses dust emissivity prescription
    # specified in pol_args, and excludes material in the simulation using
    # exc_args.
    def     Polarimetry(self, exc_args, rot_args, pol_args):
        # Load data. Basic simulation requires a density (scalar) field; a
        # momentum (vector) field; and a magnetic (vector) field. If your
        # simulation uses velocity field instead of a momentum field, modify
        # the formulation here. Additionally, this code may need to be modified
        # if your simulation uses a different naming scheme than here. See the
        # yt documentation for more details, but examining your dataset should
        # give you the correct handle for the fields.
        ds           = yt.load(self.src)
        ad           = ds.all_data()
        d            = [self.N, self.N, self.N]
        adcg         = ds.covering_grid(level=0,left_edge=[0.0,0.0,0.0],dims=d)
        denscube     = adcg[self.densityhandle  ].to_ndarray()
        Bxcube       = adcg[self.magneticxhandle].to_ndarray()
        Bycube       = adcg[self.magneticyhandle].to_ndarray()
        Bzcube       = adcg[self.magneticzhandle].to_ndarray()
        B2cube       = np.square(Bxcube) + np.square(Bycube) + np.square(Bzcube)
        # exc_args. Modify these if you wish to adopt a different type of
        # exclusion criteria, or more of them.
        # First check if we will exclude by density threshold, and if so, create
        # such a mask.
        masks = []
        if exc_args[0] is not None:
            assert exc_args[0][0] in ['gt','lt']
            if   exc_args[0][0] == 'gt':
                masks.append(denscube >= exc_args[0][1])
            elif exc_args[0][0] == 'lt':
                masks.append(denscube <= exc_args[0][1])
        # Next check if we will adopt a z-velocity magnitude threshold, and if
        # so, create such a mask.
        if exc_args[1] is not None:
            mxcube   = adcg[self.momentumxhandle].to_ndarray()
            mycube   = adcg[self.momentumyhandle].to_ndarray()
            mzcube   = adcg[self.momentumzhandle].to_ndarray()
            vzcube   = np.absolute(mzcube/denscube)
            assert exc_args[1][0] in ('gt','lt')
            if   exc_args[1][0] == 'gt':
                masks.append(vzcube >= exc_args[1][1])
            elif exc_args[1][0] == 'lt':
                masks.append(vzcube <= exc_args[1][1])
        # Finally, combine the masks using logical_and, so that any voxel that
        # is masked by any mask is in the final mask. (Different flavors of
        # logic can be implemented at will.)
        tot_mask = np.zeros((self.N,self.N,self.N)).astype(bool)
        for m in masks:
            tot_mask = np.logical_and(tot_mask, m)
        maskcube = np.logical_not(tot_mask).astype(float)
        # Next, create the dust emmisivity physics. New models can be
        # implemented here.
        assert pol_args[0] in ['Constant', 'Power-Law', 'RAT']
        if   pol_args[0] == 'Constant':
            p0 = pol_args[1]
            emmcube = p0*np.ones((self.N,self.N,self.N))
        elif pol_args[0] == 'Power-Law':
            p0 = pol_args[1]
            dens0 = pol_args[2]
            dens0inv = dens0**(-1.0)
            index = pol_args[3]
            emmcube  = (denscube >  dens0).astype(float)
            emmcube *= np.power(denscube*dens0inv,index)
            emmcube += (denscube <= dens0).astype(float)
            emmcube *= p0
        elif pol_args[0] == 'RAT':
            assert pol_args[1] in ['MRN','WD01']
            assert pol_args[2] in ['ISOSPHERE','DIFFUSE']
            if pol_args[2] == 'ISOSPHERE':
                A_vcube = 0.00856*np.power(denscube,0.5)
            elif pol_args[2] == 'DIFFUSE':
                A_vcube = np.zeros((self.N,self.N,self.N))
            a_algcube = (A_vcube+5.0)*(np.power(np.log10(denscube),3.0))/4800.0
            a_min = pol_args[3]
            a_max = pol_args[4]
            # check that the minimum aligned grain sizes are within the grain
            # population supplied.
            for a_alg in np.nditer(a_algcube):
                if a_alg < a_min:
                    a_alg = a_min
                if a_alg > a_max:
                    a_alg = a_max
            emmcube = np.zeros((self.N,self.N,self.N))
            if pol_args[1] == 'MRN':
                denom = 3.9*self.__MRN(a_min, a_max)
                for k in range(self.N):
                    for j in range(self.N):
                        for i in range(self.N):
                            emmcube[i,j,k] = self.__MRN(a_algcube[i,j,k], a_max)
            elif pol_args[1] == 'WD01':
                assert pol_args[5] in [4.0, 5.5]
                assert pol_args[6] in ['1A','2A','3A','4A','5A',
                                       '1B','2B','3B','4B']
                if   pol_args[5] == 4.0:
                    if   pol_args[6] == '1A':
                        paramsS = [-2.03,  0.66800, 0.18900, 0.100, 5.20E-14]
                        paramsC = [-2.26, -0.19900, 0.02410, 0.861, 5.47E-12, 0.0E-5]
                    elif pol_args[6] == '2A':
                        paramsS = [-2.05,  0.83200, 0.18800, 0.100, 4.81E-14]
                        paramsC = [-2.16, -0.08620, 0.00867, 0.803, 4.58E-11, 1.0E-5]
                    elif pol_args[6] == '3A':
                        paramsS = [-2.06,  0.99500, 0.18500, 0.100, 4.70E-14]
                        paramsC = [-2.01, -0.09730, 0.00811, 0.696, 3.96E-11, 2.0E-5]
                    elif pol_args[6] == '4A':
                        paramsS = [-2.08,  1.29000, 0.18400, 0.100, 4.26E-14]
                        paramsC = [-1.83, -0.17500, 0.01170, 0.604, 1.42E-11, 3.0E-5]
                    elif pol_args[6] == '5A':
                        paramsS = [-2.09,  1.58000, 0.18300, 0.100, 3.94E-14]
                        paramsC = [-1.64, -0.24700, 0.01520, 0.536, 5.83E-12, 4.0E-5]
                    elif pol_args[6] == '1B':
                        paramsS = [-2.01,  0.89400, 0.19800, 0.100, 4.95E-14]
                        paramsC = [-2.62, -0.01440, 0.01870, 5.740, 6.46E-12, 0.0E-5]
                    elif pol_args[6] == '2B':
                        paramsS = [-2.11,  1.58000, 0.19700, 0.100, 3.69E-14]
                        paramsC = [-2.52, -0.05410, 0.03660, 6.650, 1.08E-12, 1.0E-5]
                    elif pol_args[6] == '3B':
                        paramsS = [-2.05,  1.19000, 0.19700, 0.100, 4.37E-14]
                        paramsC = [-2.36, -0.09570, 0.03050, 6.440, 1.62E-12, 2.0E-5]
                    elif pol_args[6] == '4B':
                        paramsS = [-2.10,  1.64000, 0.19800, 0.100, 3.63E-14]
                        paramsC = [-2.09, -0.19300, 0.01990, 4.600, 4.21E-12, 3.0E-5]
                    elif pol_args[6] == '5B':
                        paramsS = [-2.11,  2.10000, 0.19800, 0.100, 3.13E-14]
                        paramsC = [-1.96, -0.81300, 0.06930, 3.480, 2.95E-13, 4.0E-5]
                elif pol_args[5] == 5.5:
                    if   pol_args[6] == '1A':
                        paramsS = [-1.57,  1.10000, 0.19800, 0.100, 4.24E-14]
                        paramsC = [-2.35, -0.66800, 0.14800, 1.960, 4.82E-14, 0.0E-5]
                    elif pol_args[6] == '2A':
                        paramsS = [-1.57,  1.25000, 0.19700, 0.100, 4.00E-14]
                        paramsC = [-2.12, -0.67000, 0.06860, 1.350, 3.65E-13, 1.0E-5]
                    elif pol_args[6] == '3A':
                        paramsS = [-1.55,  1.33000, 0.19500, 0.100, 4.05E-14]
                        paramsC = [-1.94, -0.85300, 0.07860, 0.921, 2.57E-13, 2.0E-5]
                    elif pol_args[6] == '4A':
                        paramsS = [-1.59,  2.12000, 0.19300, 0.100, 3.20E-14]
                        paramsC = [-1.61, -0.72200, 0.04180, 0.720, 7.58E-13, 3.0E-5]
                    elif pol_args[6] == '1B':
                        paramsS = [-1.09, -0.37000, 0.21800, 0.100, 1.17E-13]
                        paramsC = [-2.80,  0.03560, 0.02030, 3.430, 2.74E-12, 0.0E-5]
                    elif pol_args[6] == '2B':
                        paramsS = [-1.14, -0.19500, 0.21600, 0.100, 1.05E-13]
                        paramsC = [-2.67,  0.01290, 0.01340, 3.440, 7.25E-12, 1.0E-5]
                    elif pol_args[6] == '3B':
                        paramsS = [-1.08, -0.33600, 0.21600, 0.100, 1.17E-13]
                        paramsC = [-2.45, -0.00132, 0.02750, 5.140, 8.79E-13, 2.0E-5]
                    elif pol_args[6] == '4B':
                        paramsS = [-1.13, -0.10900, 0.21100, 0.100, 1.04E-13]
                        paramsC = [-1.90, -0.05170, 0.01200, 7.280, 2.86E-12, 3.0E-5]
                paramsS.append(pol_args[7])
                paramsC.append(pol_args[7])
                denom   = 1.5*(    self.__WDS01(a_min, a_max, paramsS)+
                               1.6*self.__WDC01(a_min, a_max, paramsC))
                #for k in range(self.N):
                #    for j in range(self.N):
                #        for i in range(self.N):
                #            emmcube[i,j,k] = self.__WDS01(a_algcube[i,j,k],
                #                                          a_max, paramsS)
                # Above is too slow. Instead compute integral once, and then
                # interpolate. This way, you use numpy efficiently. These
                # interpolation techniques have been verified correct to within
                # a tolerance of about 5E-5.
                densinterp = 10.0**np.linspace(np.log10(np.min(denscube)),
                                               np.log10(np.max(denscube)),
                                               pol_args[7])
                # First use the density range to compute minimum aligned grain
                # sizes.
                a_alginterp = np.zeros(pol_args[7])
                for i in range(pol_args[7]):
                    if pol_args[2] == 'ISOSPHERE':
                        A_vi = 0.00856*np.power(densinterp[i],0.5)
                    elif pol_args[2] == 'DIFFUSE':
                        A_vi = 0.0
                    a_alginterp[i] = (A_vi + 5.0)* \
                                     (np.power(np.log10(densinterp[i]),3.0)) \
                                     /4800.0
                # Next, compute the effective polarization efficiency with these
                # grain sizes.
                p_effinterp = np.zeros(pol_args[7])
                for i in range(pol_args[7]):
                    p_effinterp[i] = self.__WDS01(a_alginterp[i],
                                                  a_max, paramsS)
                    if not np.isfinite(p_effinterp[i]):
                        p_effinterp[i] = 1.0E-200
                    elif p_effinterp[i] > 1.0:
                        p_effinterp[i] = 1.0
                    elif p_effinterp[i] == 0.0:
                        p_effinterp[i] = 1.0E-200
                p_effinterp = np.abs(p_effinterp)
                # Optional Plot of polarization efficiency with density and
                # grain sizes.
                if pol_args[8]:
                    import matplotlib.pyplot as plt
                    fig = plt.figure()
                    plt.loglog(a_alginterp,p_effinterp*np.power(denom,-1.0))
                    plt.ylim(1E-4,1E1)
                    plt.ylabel('Polarization Efficiency')
                    plt.xlabel('Minimum Aligned Grain Size (um)')
                    if self.optlabel:
                        fig.savefig(self.path+'poleffa'+self.optlabel+'.png')
                    else:
                        fig.savefig(self.path+'poleffa.png')
                    fig = plt.figure()
                    plt.loglog(densinterp,p_effinterp*np.power(denom,-1.0))
                    plt.ylim(1E-4,1E1)
                    plt.ylabel('Polarization Efficiency')
                    plt.xlabel(r'Gas Number Density (cm$^{-3}$)')
                    if self.optlabel:
                        fig.savefig(self.path+'poleffd'+self.optlabel+'.png')
                    else:
                        fig.savefig(self.path+'poleffd.png')
                # Finally, interpolate the effective polarization efficiency for
                # each voxel's minimum aligned grain size computed earlier.
                emmcube = np.interp(a_algcube, a_alginterp, p_effinterp)
            # Apply grain population normalization.
            emmcube *= (np.power(denom,-1.0))
            # Ensure no negative polarization efficiencies got through.
            emmcube = np.abs(emmcube)

        # Next check if we are rotating by our angles, and if so, rotate our
        # simulation values using Rotator.
        axes     = rot_args[0]
        assert axes in [['x','y'],['z','x'],['y','z']]
        rotation = rot_args[1]
        if rotation is not None:
            assert len(rotation) == 3
            R = Rotator([rotation[0], rotation[1], rotation[2],
                         self.N, 0])
            maskcube               = R.ScalarRotate(maskcube)
            emmcube                = R.ScalarRotate(emmcube)
            denscube               = R.ScalarRotate(denscube)
            B2cube                 = R.ScalarRotate(B2cube)
            Bxcube, Bycube, Bzcube = R.VectorRotate(Bxcube,Bycube,Bzcube)
        # Next, create the total cubes that will be integrated along the correct
        # axis. Then, integrate.
        Ncube  = maskcube*denscube
        if   axes == ['x','y']:
            N2cube = maskcube*emmcube*denscube* \
                     (((np.square(Bxcube)+np.square(Bycube))* \
                                               np.power(B2cube,-1.0))-(2.0/3.0))
            Qcube  = maskcube*emmcube*denscube* \
                     (np.square(Bycube)-np.square(Bxcube))*np.power(B2cube,-1.0)
            Ucube  = maskcube*emmcube*denscube* \
                                       (2.0*Bxcube*Bycube)*np.power(B2cube,-1.0)
            CD     = self.reselmt*np.sum(Ncube,  axis=2).T
            Q      = self.reselmt*np.sum(Qcube,  axis=2).T
            U      = self.reselmt*np.sum(Ucube,  axis=2).T
            N2     = self.reselmt*np.sum(N2cube, axis=2).T
        elif axes == ['z','x']:
            N2cube = maskcube*emmcube*denscube* \
                     (((np.square(Bzcube)+np.square(Bxcube))* \
                                               np.power(B2cube,-1.0))-(2.0/3.0))
            Qcube  = maskcube*emmcube*denscube* \
                     (np.square(Bxcube)-np.square(Bzcube))*np.power(B2cube,-1.0)
            Ucube  = maskcube*emmcube*denscube* \
                                       (2.0*Bzcube*Bxcube)*np.power(B2cube,-1.0)
            CD     = self.reselmt*np.sum(Ncube,  axis=1).T
            Q      = self.reselmt*np.sum(Qcube,  axis=1).T
            U      = self.reselmt*np.sum(Ucube,  axis=1).T
            N2     = self.reselmt*np.sum(N2cube, axis=1).T
        elif axes == ['y','z']:
            N2cube = maskcube*emmcube*denscube* \
                     (((np.square(Bycube)+np.square(Bzcube))* \
                                               np.power(B2cube,-1.0))-(2.0/3.0))
            Qcube  = maskcube*emmcube*denscube* \
                     (np.square(Bzcube)-np.square(Bycube))*np.power(B2cube,-1.0)
            Ucube  = maskcube*emmcube*denscube* \
                                       (2.0*Bycube*Bzcube)*np.power(B2cube,-1.0)
            CD     = self.reselmt*np.sum(Ncube,  axis=0).T
            Q      = self.reselmt*np.sum(Qcube,  axis=0).T
            U      = self.reselmt*np.sum(Ucube,  axis=0).T
            N2     = self.reselmt*np.sum(N2cube, axis=0).T

        # Now mask any elements where the column density is not positive, in
        # case something bad happened in the simulation (unlikely) or there are
        # sightlines totally excluded by your cutoff criteria (possible).
        safetymask = np.zeros((self.N,self.N)).astype(bool)
        for j in range(self.N):
            for i in range(self.N):
                if CD[i,j] <= 0.0:
                    safetymask[i,j] = True
        CD = np.ma.masked_array(CD, safetymask)
        Q  = np.ma.masked_array(Q,  safetymask)
        U  = np.ma.masked_array(U,  safetymask)
        N2 = np.ma.masked_array(N2, safetymask)

        # Compute derived quantities. Apply safety mask.
        I  = CD - N2
        Pi = np.sqrt(np.square(Q)+np.square(U))
        p  = Pi*np.ma.power(I,-1.0)
        ch = np.rad2deg(0.5*(np.pi + np.arctan2(U,Q)))
        Pi = np.ma.masked_array(Pi, safetymask)
        I  = np.ma.masked_array(I,  safetymask)
        p  = np.ma.masked_array(p,  safetymask)
        ch = np.ma.masked_array(ch, safetymask)

        # Create Observables and add them to the list to return them.
        col = ['viridis','Spectral_r','magma']
        unt = ['cm$^{-2}$', 'None', 'Degrees']
        snm = ['I', 'Q', 'U', '$\Sigma$', '$\Sigma_2$', 'P', '$p$', '$\chi$']
        lnm = ['Stokes I', 'Stokes Q', 'Stokes U', 'Column Density',
               'Column Density Correction', 'Polarized Intensity',
               'Polarization Fraction', 'Magnetic Field Angle']

        Observables = []
        Observables.append(Observable([I,
                                       self.N,
                                       'log',
                                       lnm[0],
                                       snm[0],
                                       unt[0],
                                       col[0],
                                       axes,
                                       rotation,
                                       None]))
        Observables.append(Observable([Q,
                                       self.N,
                                       'symlog',
                                       lnm[1],
                                       snm[1],
                                       unt[0],
                                       col[1],
                                       axes,
                                       rotation,
                                       None]))
        Observables.append(Observable([U,
                                       self.N,
                                       'symlog',
                                       lnm[2],
                                       snm[2],
                                       unt[0],
                                       col[1],
                                       axes,
                                       rotation,
                                       None]))
        Observables.append(Observable([CD,
                                       self.N,
                                       'log',
                                       lnm[3],
                                       snm[3],
                                       unt[0],
                                       col[0],
                                       axes,
                                       rotation,
                                       None]))
        Observables.append(Observable([N2,
                                       self.N,
                                       'linear',
                                       lnm[4],
                                       snm[4],
                                       unt[0],
                                       col[2],
                                       axes,
                                       rotation,
                                       None]))
        Observables.append(Observable([Pi,
                                       self.N,
                                       'log',
                                       lnm[5],
                                       snm[5],
                                       unt[0],
                                       col[0],
                                       axes,
                                       rotation,
                                       None]))
        Observables.append(Observable([p,
                                       self.N,
                                       'log',
                                       lnm[6],
                                       snm[6],
                                       unt[1],
                                       col[2],
                                       axes,
                                       rotation,
                                       None]))
        Observables.append(Observable([ch,
                                       self.N,
                                       'linear',
                                       lnm[7],
                                       snm[7],
                                       unt[2],
                                       col[1],
                                       axes,
                                       rotation,
                                       None]))

        # Write the observables.
        for o in Observables:
            self.WriteObservable(o)
        # Finally, return the observables.
        return Observables

    # Base method for computing synthetic Zeeman observations of the magnetic
    # field strength.
    def          Zeeman(self, exc_args, rot_args):
        ds           = yt.load(self.src)
        ad           = ds.all_data()
        d            = [self.N, self.N, self.N]
        adcg         = ds.covering_grid(level=0,left_edge=[0.0,0.0,0.0],dims=d)
        denscube     = adcg[self.densityhandle].to_ndarray()
        Bxcube       = adcg[self.magneticxhandle].to_ndarray()
        Bycube       = adcg[self.magneticyhandle].to_ndarray()
        Bzcube       = adcg[self.magneticzhandle].to_ndarray()
        # exc_args. Modify these if you wish to adopt a different type of
        # exclusion criteria, or more of them.
        # First check if we will exclude by density threshold, and if so, create
        # such a mask.
        masks = []
        if exc_args[0] is not None:
            assert exc_args[0][0] in ['gt','lt']
            if   exc_args[0][0] == 'gt':
                masks.append(denscube >= exc_args[0][1])
            elif exc_args[0][0] == 'lt':
                masks.append(denscube <= exc_args[0][1])
        # Next check if we will adopt a z-velocity magnitude threshold, and if
        # so, create such a mask.
        if exc_args[1] is not None:
            mxcube   = adcg[self.momentumxhandle].to_ndarray()
            mycube   = adcg[self.momentumyhandle].to_ndarray()
            mzcube   = adcg[self.momentumzhandle].to_ndarray()
            vzcube   = np.absolute(mzcube/denscube)
            assert exc_args[1][0] in ('gt','lt')
            if   exc_args[1][0] == 'gt':
                masks.append(vzcube >= exc_args[1][1])
            elif exc_args[1][0] == 'lt':
                masks.append(vzcube <= exc_args[1][1])
        # Finally, combine the masks using logical or, so that any voxel that
        # is masked by any mask is in the final mask. (Different flavors of
        # logic can be implemented at will.)
        tot_mask = np.zeros((self.N,self.N,self.N)).astype(bool)
        for m in masks:
            tot_mask = np.logical_and(tot_mask, m)
        maskcube = np.logical_not(tot_mask).astype(float)

        # Next check if we are rotating by our angles, and if so, rotate our
        # simulation values using Rotator.
        axes     = rot_args[0]
        assert axes in [['x','y'],['z','x'],['y','z']]
        rotation = rot_args[1]
        if rotation is not None:
            assert len(rotation) == 3
            R = Rotator([rotation[0], rotation[1], rotation[2],
                         self.N, 0])
            maskcube               = R.ScalarRotate(maskcube)
            denscube               = R.ScalarRotate(denscube)
            Bxcube, Bycube, Bzcube = R.VectorRotate(Bxcube,Bycube,Bzcube)

        # Next, create the total cubes that will be integrated along the correct
        # axis. Then, integrate.
        Ncube  = maskcube*denscube
        if   axes == ['x','y']:
            CD     = self.reselmt*np.sum(    Ncube,               axis=2).T
            BZee   =              np.average(np.absolute(Bzcube), axis=2).T
        elif axes == ['z','x']:
            CD     = self.reselmt*np.sum(    Ncube,               axis=1).T
            BZee   =              np.average(np.absolute(Bycube), axis=1).T
        elif axes == ['y','z']:
            CD     = self.reselmt*np.sum(    Ncube,               axis=0).T
            BZee   =              np.average(np.absolute(Bxcube), axis=0).T

        # Now mask any elements where the column density is not positive, in
        # case something bad happened in the simulation (unlikely) or there are
        # sightlines totally excluded by your cutoff criteria (possible).
        safetymask = np.zeros((self.N,self.N)).astype(bool)
        for j in range(self.N):
            for i in range(self.N):
                if CD[i,j] <= 0.0:
                    safetymask[i,j] = True
        CD   = np.ma.masked_array(CD,   safetymask)
        BZee = np.ma.masked_array(BZee, safetymask)

        # Create Observables and add them to the list to return them.
        lnm = ['Column Density', 'Zeeman Magnetic Field Magnitude']
        snm = ['$Sigma$', 'B$_{Zee}$']
        unt = ['cm$^{-2}$', 'uG']
        col = ['viridis', 'plasma']

        Observables = []
        Observables.append(Observable([CD,
                                       self.N,
                                       'log',
                                       lnm[0],
                                       snm[0],
                                       unt[0],
                                       col[0],
                                       axes,
                                       rotation,
                                       None]))
        Observables.append(Observable([BZee,
                                       self.N,
                                       'linear',
                                       lnm[1],
                                       snm[1],
                                       unt[1],
                                       col[1],
                                       axes,
                                       rotation,
                                       None]))

        # Write observables.
        for o in Observables:
            self.WriteObservable(o)
        # Finally, return the observables.
        return Observables

    # Base method for computing synthetic velocity moments. Uses NRAO CASA
    # definitions for the velocity moments.
    def VelocityMoments(self, exc_args, rot_args):
        ds           = yt.load(self.src)
        ad           = ds.all_data()
        d            = [self.N, self.N, self.N]
        adcg         = ds.covering_grid(level=0,left_edge=[0.0,0.0,0.0],dims=d)
        denscube     = adcg[self.densityhandle  ].to_ndarray()
        mxcube       = adcg[self.momentumxhandle].to_ndarray()
        mycube       = adcg[self.momentumyhandle].to_ndarray()
        mzcube       = adcg[self.momentumzhandle].to_ndarray()
        vxcube       = mxcube/denscube
        vycube       = mxcube/denscube
        vzcube       = mxcube/denscube
        # exc_args. Modify these if you wish to adopt a different type of
        # exclusion criteria, or more of them.
        # First check if we will exclude by density threshold, and if so, create
        # such a mask.
        masks = []
        if exc_args[0] is not None:
            assert exc_args[0][0] in ['gt','lt']
            if   exc_args[0][0] == 'gt':
                masks.append(denscube >= exc_args[0][1])
            elif exc_args[0][0] == 'lt':
                masks.append(denscube <= exc_args[0][1])
        # Next check if we will adopt a z-velocity magnitude threshold, and if
        # so, create such a mask.
        if exc_args[1] is not None:
            vzabscube = np.absolute(mzcube/denscube)
            assert exc_args[1][0] in ('gt','lt')
            if   exc_args[1][0] == 'gt':
                masks.append(vzcube >= exc_args[1][1])
            elif exc_args[1][0] == 'lt':
                masks.append(vzcube <= exc_args[1][1])
        # Finally, combine the masks using logical or, so that any voxel that
        # is masked by any mask is in the final mask. (Different flavors of
        # logic can be implemented at will.)
        tot_mask = np.zeros((self.N,self.N,self.N)).astype(bool)
        for m in masks:
            tot_mask = np.logical_and(tot_mask, m)
        maskcube = np.logical_not(tot_mask).astype(float)

        # Next check if we are rotating by our angles, and if so, rotate our
        # simulation values using Rotator.
        axes     = rot_args[0]
        assert axes in [['x','y'],['z','x'],['y','z']]
        rotation = rot_args[1]
        if rotation is not None:
            assert len(rotation) == 3
            R = Rotator([rotation[0], rotation[1], rotation[2],
                         self.N, 0])
            maskcube               = R.ScalarRotate(maskcube)
            denscube               = R.ScalarRotate(denscube)
            vxcube, vycube, vzcube = R.VectorRotate(vxcube,vycube,vzcube)

        # Next, create the total cubes that will be integrated along the correct
        # axis. Then, integrate.
        intcube  = maskcube*denscube
        if   axes == ['x','y']:
            M0     = self.reselmt*np.sum(intcube, axis=2).T
            M0inv  = np.power(M0, -1.0)
            M1     = self.reselmt*np.sum(intcube*vzcube, axis=2).T
            M1    *= M0inv
            v2cube = np.power((vzcube - np.stack([M1 for _ in range(self.N)],
                              axis=2)), 2.0)
            M2     = self.reselmt*np.sum(intcube*v2cube, axis=2).T
            M2     = np.power(M2*M0inv, 0.5)
        elif axes == ['z','x']:
            M0     = self.reselmt*np.sum(intcube, axis=1).T
            M0inv  = np.power(M0, -1.0)
            M1     = self.reselmt*np.sum(intcube*vycube, axis=1).T
            M1    *= M0inv
            v2cube = np.power((vycube - np.stack([M1 for _ in range(self.N)],
                              axis=1)), 2.0)
            M2     = self.reselmt*np.sum(intcube*v2cube, axis=1).T
            M2     = np.power(M2*M0inv, 0.5)
        elif axes == ['y','z']:
            M0     = self.reselmt*np.sum(intcube, axis=0).T
            M0inv  = np.power(M0, -1.0)
            M1     = self.reselmt*np.sum(intcube*vxcube, axis=0).T
            M1    *= M0inv
            v2cube = np.power((vxcube - np.stack([M1 for _ in range(self.N)],
                              axis=0)), 2.0)
            M2     = self.reselmt*np.sum(intcube*v2cube, axis=0).T
            M2     = np.power(M2*M0inv, 0.5)

        # Now mask any elements where the column density is not positive, in
        # case something bad happened in the simulation (unlikely) or there are
        # sightlines totally excluded by your cutoff criteria (possible).
        safetymask = np.zeros((self.N,self.N)).astype(bool)
        for j in range(self.N):
            for i in range(self.N):
                if CD[i,j] <= 0.0:
                    safetymask[i,j] = True
        M0 = np.ma.masked_array(M0, safetymask)
        M1 = np.ma.masked_array(M1, safetymask)
        M2 = np.ma.masked_array(M2, safetymask)

        # Correct the units of the moment 1 and 2 maps to km/s from cm/s.
        M1 *= 1.0E-5
        M2 *= 1.0E-5

        # Create Observables and add them to the list to return them.
        lnm = ['Velocity Moment 0', 'Velocity Moment 1', 'Velocity Moment 2']
        snm = ['M$_0$', 'M$_1$', 'M$_2$']
        unt = ['cm$^{-2}$', 'km/s']
        col = ['viridis', 'Spectral_r', 'magma']

        Observables = []
        Observables.append(Observable([M0,
                                       self.N,
                                       'log',
                                       lnm[0],
                                       snm[0],
                                       unt[0],
                                       col[0],
                                       axes,
                                       rotation,
                                       None]))
        Observables.append(Observable([M1,
                                       self.N,
                                       'log',
                                       lnm[1],
                                       snm[1],
                                       unt[1],
                                       col[1],
                                       axes,
                                       rotation,
                                       None]))
        Observables.append(Observable([M2,
                                       self.N,
                                       'log',
                                       lnm[2],
                                       snm[2],
                                       unt[1],
                                       col[2],
                                       axes,
                                       rotation,
                                       None]))

        # Write observables.
        for o in Observables:
            self.WriteObservable(o)
        # Finally, return the observables.
        return Observables

    # Method to write an observable to a simple text file. Creates a header with
    # 9 rows, each corresponding to the metadata attached to the observable.
    # Then it unmasks the data, stores the mask in an integer array of the same
    # size of the data, and ravels both and stacks them, creating two columns -
    # data and mask.
    def WriteObservable(self, O):
        if   O.axes == ['x','y']:
            ax = 0
        elif O.axes == ['z','x']:
            ax = 1
        elif O.axes == ['y','z']:
            ax = 2
        header    = str(O.lname)    + '\n'
        header   += str(O.sname)    + '\n'
        header   += str(O.N)        + '\n'
        header   += str(O.norm)     + '\n'
        header   += str(O.units)    + '\n'
        header   += str(O.colmap)   + '\n'
        header   += str(ax)         + '\n'
        header   += str(O.rotation) + '\n'
        header   += str(O.beam)     + '\n'
        filename = O.lname.replace(' ', '_')
        if self.optlabel is not None:
            filename += self.optlabel
        filename +='.txt'
        if len(np.asarray(O.data.mask))  == 0:
            O.data.mask = np.zeros(O.data.shape)
        datastack = np.vstack((np.ravel(np.array(O.data)),
                               np.ravel(np.ma.getmask(O.data).astype(int))))
        np.savetxt(self.path+filename, np.transpose(datastack),
                   header=header, comments='', newline='\n')
        return

    # Reads an observable file of the same form as written using
    # WriteObservable. Returns the observable.
    def ReadObservable(self, source):
        header    = np.genfromtxt(source, skip_header=0,  max_rows=9,
                                          delimiter =',', autostrip=True,
                                          comments=None,  dtype=str)
        lname     =     header[0]
        sname     =     header[1]
        N         = int(header[2])
        norm      =     header[3]
        units     =     header[4]
        colmap    =     header[5]
        axes      = int(header[6])
        rotation  =     header[7]
        beam      =     header[8]
        if   axes == 0:
            ax = ['x','y']
        elif axes == 1:
            ax = ['z','x']
        elif axes == 2:
            ax = ['y','z']
        if rotation == 'None':
            rotation = None
        if beam     == 'None':
            beam = None
        datastack = np.genfromtxt(source, skip_header=10)
        mask =                    datastack[:,1].reshape((N,N)).astype(bool)
        data = np.ma.masked_array(datastack[:,0].reshape((N,N)), mask)
        O = Observable([data,   N,  norm,     lname,  sname, units,
                        colmap, ax, rotation, beam])
        return O
