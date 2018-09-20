from Observer import *
from Rotator  import *
from Nabla    import *
from Smoother import *
from Stats    import *
from Plotter  import *
from math     import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

indices = [-1.0/3.0, -2.0/3.0, -1.0]
cutoffs = [1.0E3, 1.0E2]

boxlenA = 20.0 
boxlenB = 10.0 
boxlenC =  5.0
boxlenD =  1.0 

scaleA = boxlenA/boxlenB 
scaleC = boxlenC/boxlenB 
scaleD = boxlenD/boxlenB 

N = 512 

p0 = 0.15

Nbnds = [20.0, 24.0]
pbnds = [-3.5, -0.5]
Sbnds = [-1.5,  2.0]

colors = ['#0000ff', '#80b381', '#ffcc00', '#ec8013', '#b30000']

exc_args = [None, None]
rot_args = [['x','y'], None]

St = Stats() 

figNp, axNp = plt.subplots(2, 4, sharex=True, sharey=True, figsize=(13,7))
figSp, axSp = plt.subplots(2, 4, sharex=True, sharey=True, figsize=(13,7))
figNS, axNS = plt.subplots(2, 4, sharex=True, sharey=True, figsize=(13,7))

figp, axp = plt.subplots(4, 1, sharex=True, sharey=True, figsize=(4,13))
figS, axS = plt.subplots(4, 1, sharex=True, sharey=True, figsize=(4,13))

# Plot the BLASTPol data for comparison. 
BPData     = np.loadtxt('VelaC_pNS_500micron_noRCW36.dat',skiprows=2)
NBPdata    = BPData[:,0]
SBPdata    = np.rad2deg(BPData[:,1])
pBPdata    = BPData[:,2] 
xpBP, fpBP = St.Gaussian1DKDE(np.log10(pBPdata), pbnds)
xSBP, fSBP = St.Gaussian1DKDE(np.log10(SBPdata), Sbnds)
NpBPeval, NpBPevec = St.PCA(np.log10(NBPdata), np.log10(pBPdata))
SpBPeval, SpBPevec = St.PCA(np.log10(SBPdata), np.log10(pBPdata))
NSBPeval, NSBPevec = St.PCA(np.log10(NBPdata), np.log10(SBPdata))
xxNpBP, yyNpBP, fNpBP = St.Gaussian2DKDE(np.log10(NBPdata), np.log10(pBPdata), Nbnds, pbnds)
xxSpBP, yySpBP, fSpBP = St.Gaussian2DKDE(np.log10(SBPdata), np.log10(pBPdata), Sbnds, pbnds)
xxNSBP, yyNSBP, fNSBP = St.Gaussian2DKDE(np.log10(NBPdata), np.log10(SBPdata), Nbnds, Sbnds)
meanCDBP = np.average(np.log10(NBPdata))
meanpBP  = np.average(np.log10(pBPdata))
meanSBP  = np.average(np.log10(SBPdata))
axp[0].semilogy(xpBP, fpBP, '-', linewidth=2.0, color=colors[0], zorder=1)
axS[0].semilogy(xSBP, fSBP, '-', linewidth=2.0, color=colors[0], zorder=1)
axp[1].semilogy(xpBP, fpBP, '-', linewidth=2.0, color=colors[0], zorder=1)
axS[1].semilogy(xSBP, fSBP, '-', linewidth=2.0, color=colors[0], zorder=1)
axp[2].semilogy(xpBP, fpBP, '-', linewidth=2.0, color=colors[0], zorder=1)
axS[2].semilogy(xSBP, fSBP, '-', linewidth=2.0, color=colors[0], zorder=1)
axp[3].semilogy(xpBP, fpBP, '-', linewidth=2.0, color=colors[0], zorder=1)
axS[3].semilogy(xSBP, fSBP, '-', linewidth=2.0, color=colors[0], zorder=1)
axNp[1,0].contour(xxNpBP, yyNpBP, fNpBP, 10, colors=colors[0],zorder=1)
axNp[1,0].quiver(meanCDBP, meanpBP, np.sqrt(NpBPeval[0])*NpBPevec[0,0], np.sqrt(NpBPeval[0])*NpBPevec[1,0], scale=1.25, zorder=2)
axNp[1,0].quiver(meanCDBP, meanpBP, np.sqrt(NpBPeval[1])*NpBPevec[0,1], np.sqrt(NpBPeval[1])*NpBPevec[1,1], scale=1.25, zorder=2)
axNp[1,0].plot(meanCDBP, meanpBP, 'k.', markersize=5)
axSp[1,0].contour(xxSpBP, yySpBP, fSpBP, 10, colors=colors[0],zorder=1)
axSp[1,0].quiver(meanSBP, meanpBP, np.sqrt(SpBPeval[0])*abs(SpBPevec[0,0]),  np.sqrt(SpBPeval[0])*abs(SpBPevec[1,0]), scale=1.25, zorder=2)
axSp[1,0].quiver(meanSBP, meanpBP, np.sqrt(SpBPeval[1])*abs(SpBPevec[0,1]), -np.sqrt(SpBPeval[1])*abs(SpBPevec[1,1]), scale=1.25, zorder=2)
axSp[1,0].plot(meanSBP, meanpBP, 'k.', markersize=5)
axNS[1,0].contour(xxNSBP, yyNSBP, fNSBP, 10, colors=colors[0],zorder=1)
axNS[1,0].quiver(meanCDBP, meanSBP, np.sqrt(NSBPeval[0])*NSBPevec[0,0], np.sqrt(NSBPeval[0])*NSBPevec[1,0], scale=1.25, zorder=2)
axNS[1,0].quiver(meanCDBP, meanSBP, np.sqrt(NSBPeval[1])*NSBPevec[0,1], np.sqrt(NSBPeval[1])*NSBPevec[1,1], scale=1.25, zorder=2)
axNS[1,0].plot(meanCDBP, meanSBP, 'k.', markersize=5)
print('BlastPol Complete.')

# Open Files to write stats to. 
fileNA  = open('N_XY_A_stats.txt', 'w')
filepA  = open('p_XY_A_stats.txt', 'w')
fileSA  = open('S_XY_A_stats.txt', 'w')
fileNpA = open('Np_XY_A_stats.txt', 'w')
fileNSA = open('NS_XY_A_stats.txt', 'w')
fileSpA = open('Sp_XY_A_stats.txt', 'w')

fileNA.write( '# Model N_geomean N_geomed N_geostd N_geoskew N_geokurt \n')
filepA.write( '# Model p_geomean p_geomed p_geostd p_geoskew p_geokurt \n')
fileSA.write( '# Model S_geomean S_geomed S_geostd S_geoskew S_geokurt \n')
fileNpA.write('# Model Npevalmax Npevecmax1 Npevecmax2 Npevalmin Npevecmin1 Npevecmin2 NpPearson NpSpearman \n')
fileNSA.write('# Model NSevalmax NSevecmax1 NSevecmax2 NSevalmin NSevecmin1 NSevecmin2 NSPearson NSSpearman \n')
fileSpA.write('# Model Spevalmax Spevecmax1 Spevecmax2 Spevalmin Spevecmin1 Spevecmin2 SpPearson SpSpearman \n')

# Load Sim A data. 
OA = Observer(['L20B5M5_0029.vtk',N,boxlenA,'./'])
OA.ChangeMagneticHandle(['magnetic_field_x', 'magnetic_field_y', 'magnetic_field_z'])
OA.DataLoad()
NA = Nabla([None,N,boxlenA])
PltA = Plotter([N, boxlenA, 5, 100, './'])
print('Sim A Data Loaded.')

# First, plot the unmodified distributions. 
pol_args = ['Constant', p0]
ObsA = OA.Polarimetry(exc_args, rot_args, pol_args)
CDA = ObsA[3]
CDA.data *= scaleA
QA = ObsA[1]
UA = ObsA[2]
pA = ObsA[6]
SA = NA.ComputeAngleGradient(QA, UA, QA.data.mask)
# 1D KDE.
xNA, fNA = St.Gaussian1DKDE(np.log10(np.ma.compressed(CDA.data)), Nbnds)
xpA, fpA = St.Gaussian1DKDE(np.log10(np.ma.compressed(pA.data)), pbnds)
xSA, fSA = St.Gaussian1DKDE(np.log10(np.ma.compressed(SA.data)), Sbnds)
# PCA.
NpAeval, NpAevec = St.PCA(np.log10(np.ma.compressed(CDA.data)), np.log10(np.ma.compressed(pA.data)))
SpAeval, SpAevec = St.PCA(np.log10(np.ma.compressed(SA.data)),  np.log10(np.ma.compressed(pA.data)))
NSAeval, NSAevec = St.PCA(np.log10(np.ma.compressed(CDA.data)), np.log10(np.ma.compressed(SA.data)))
# 2D KDE.
xxNpA, yyNpA, fNpA = St.Gaussian2DKDE(np.log10(np.ma.compressed(CDA.data)), np.log10(np.ma.compressed(pA.data)), Nbnds, pbnds)
xxSpA, yySpA, fSpA = St.Gaussian2DKDE(np.log10(np.ma.compressed(SA.data)),  np.log10(np.ma.compressed(pA.data)), Sbnds, pbnds)
xxNSA, yyNSA, fNSA = St.Gaussian2DKDE(np.log10(np.ma.compressed(CDA.data)), np.log10(np.ma.compressed(SA.data)), Nbnds, Sbnds)
# Compute Correlation Coefficients.
NpApr, NpAsr = St.CorrCoeffs(np.log10(np.ma.compressed(CDA.data)), np.log10(np.ma.compressed(pA.data)), False)
SpApr, SpAsr = St.CorrCoeffs(np.log10(np.ma.compressed(SA.data)),  np.log10(np.ma.compressed(pA.data)), False)
NSApr, NSAsr = St.CorrCoeffs(np.log10(np.ma.compressed(CDA.data)), np.log10(np.ma.compressed(SA.data)), False)
# Compute means.
meanCDA = np.average(np.log10(np.ma.compressed(CDA.data)))
meanpA  = np.average(np.log10(np.ma.compressed( pA.data)))
meanSA  = np.average(np.log10(np.ma.compressed( SA.data))) 
# Compute geometric medians. 
medCDA = np.median(np.log10(np.ma.compressed(CDA.data)))
medpA  = np.median(np.log10(np.ma.compressed( pA.data)))
medSA  = np.median(np.log10(np.ma.compressed( SA.data)))
# Compute geometric std deviations. 
stdCDA = np.std(np.log10(np.ma.compressed(CDA.data)))
stdpA  = np.std(np.log10(np.ma.compressed( pA.data)))
stdSA  = np.std(np.log10(np.ma.compressed( SA.data)))
# Compute geometric skews. 
skewCDA = stats.skew(np.log10(np.ma.compressed(CDA.data)))
skewpA  = stats.skew(np.log10(np.ma.compressed( pA.data)))
skewSA  = stats.skew(np.log10(np.ma.compressed( SA.data)))
# Compute geometric kurtoses. 
kurtCDA = stats.kurtosis(np.log10(np.ma.compressed(CDA.data)), fisher=True)
kurtpA  = stats.kurtosis(np.log10(np.ma.compressed( pA.data)), fisher=True)
kurtSA  = stats.kurtosis(np.log10(np.ma.compressed( SA.data)), fisher=True)
# Write Stats. 
fileNA.write( 'Homogeneous, {: f}, {: f}, {: f}, {: f}, {: f} \n'.format(meanCDA, medCDA, stdCDA, skewCDA, kurtCDA))
filepA.write( 'Homogeneous, {: f}, {: f}, {: f}, {: f}, {: f} \n'.format(meanpA,  medpA,  stdpA,  skewpA,  kurtpA))
fileSA.write( 'Homogeneous, {: f}, {: f}, {: f}, {: f}, {: f} \n'.format(meanSA,  medSA,  stdSA,  skewSA,  kurtSA))
fileNpA.write('Homogeneous, {: f}, {: f}, {: f}, {: f}, {: f}, {: f}, {: f}, {: f} \n'.format(NpAeval[1], NpAevec[1,0], NpAevec[1,1], NpAeval[0], NpAevec[0,0], NpAevec[0,1], NpApr, NpAsr))
fileNSA.write('Homogeneous, {: f}, {: f}, {: f}, {: f}, {: f}, {: f}, {: f}, {: f} \n'.format(NSAeval[1], NSAevec[1,0], NSAevec[1,1], NSAeval[0], NSAevec[0,0], NSAevec[0,1], NSApr, NSAsr))
fileSpA.write('Homogeneous, {: f}, {: f}, {: f}, {: f}, {: f}, {: f}, {: f}, {: f} \n'.format(SpAeval[1], SpAevec[1,0], SpAevec[1,1], SpAeval[0], SpAevec[0,0], SpAevec[0,1], SpApr, SpAsr))
# Save 1D KDE.
St.Save1DKDE(xNA, fNA, 'N_XY_A_homo')
St.Save1DKDE(xpA, fpA, 'p_XY_A_homo')
St.Save1DKDE(xSA, fSA, 'S_XY_A_homo')
# Save 2D KDE. 
St.Save2DKDE(xxNpA, yyNpA, fNpA, 'Np_XY_A_homo', 'Gaussian')
St.Save2DKDE(xxNSA, yyNSA, fNSA, 'NS_XY_A_homo', 'Gaussian')
St.Save2DKDE(xxSpA, yySpA, fSpA, 'Sp_XY_A_homo', 'Gaussian')       
# plotting. 
axp[0].semilogy(xpA, fpA, '-', linewidth=1.25, color=colors[4], zorder=2)
axS[0].semilogy(xSA, fSA, '-', linewidth=1.25, color=colors[4], zorder=2)

axNp[0,0].contour(xxNpA, yyNpA, fNpA, 10, colors=colors[3],zorder=1)
axNp[0,0].quiver(meanCDA, meanpA, -np.sqrt(NpAeval[0])*NpAevec[0,0], -np.sqrt(NpAeval[0])*NpAevec[1,0], scale=1.25, zorder=2)
axNp[0,0].quiver(meanCDA, meanpA, -np.sqrt(NpAeval[1])*NpAevec[0,1], -np.sqrt(NpAeval[1])*NpAevec[1,1], scale=1.25, zorder=2)
axNp[0,0].plot(meanCDA, meanpA, 'k.', markersize=5)

axSp[0,0].contour(xxSpA, yySpA, fSpA, 10, colors=colors[3],zorder=1)
axSp[0,0].quiver(meanSA, meanpA, np.sqrt(SpAeval[0])*abs(SpAevec[0,0]),  np.sqrt(SpAeval[0])*abs(SpAevec[1,0]), scale=1.25, zorder=2)
axSp[0,0].quiver(meanSA, meanpA, np.sqrt(SpAeval[1])*abs(SpAevec[0,1]), -np.sqrt(SpAeval[1])*abs(SpAevec[1,1]), scale=1.25, zorder=2)
axSp[0,0].plot(meanSA, meanpA, 'k.', markersize=5)

axNS[0,0].contour(xxNSA, yyNSA, fNSA, 10, colors=colors[3],zorder=1)
axNS[0,0].quiver(meanCDA, meanSA, np.sqrt(NSAeval[0])*NSAevec[0,0], np.sqrt(NSAeval[0])*NSAevec[1,0], scale=1.25, zorder=2)
axNS[0,0].quiver(meanCDA, meanSA, np.sqrt(NSAeval[1])*NSAevec[0,1], np.sqrt(NSAeval[1])*NSAevec[1,1], scale=1.25, zorder=2)
axNS[0,0].plot(meanCDA, meanSA, 'k.', markersize=5)

del xNA, xpA, xSA, fNA, fpA, fSA
del xxNpA, yyNpA, fNpA 
del xxNSA, yyNSA, fNSA 
del xxSpA, yySpA, fSpA
del NpApr, NpAsr, NSApr, NSAsr, SpApr, SpAsr
del meanCDA, meanpA, meanSA
del medCDA, medpA, medSA 
del stdCDA, stdpA, stdSA 
del skewCDA, skewpA, skewSA 
del kurtCDA, kurtpA, kurtSA 
del NpAeval, NpAevec 
del NSAeval, NSAevec 
del SpAeval, SpAevec
print("Sim A Index Constant p0 Complete.")

# Now, loop for model A.
for i in range(len(indices)):
    for j in range(len(cutoffs)):
        pol_args = ['Power-Law', p0, cutoffs[j]/(scaleA**2), indices[i]]
        ObsA = OA.Polarimetry(exc_args, rot_args, pol_args)
        CDA = ObsA[3]
        CDA.data *= scaleA 
        QA  = ObsA[1]
        UA  = ObsA[2]
        pA  = ObsA[6]
        SA  = NA.ComputeAngleGradient(QA, UA, QA.data.mask)
        # 1D KDE.
        xNA, fNA = St.Gaussian1DKDE(np.log10(np.ma.compressed(CDA.data)), Nbnds)
        xpA, fpA = St.Gaussian1DKDE(np.log10(np.ma.compressed(pA.data)), pbnds)
        xSA, fSA = St.Gaussian1DKDE(np.log10(np.ma.compressed(SA.data)), Sbnds)
        # PCA.
        NpAeval, NpAevec = St.PCA(np.log10(np.ma.compressed(CDA.data)), np.log10(np.ma.compressed(pA.data)))
        SpAeval, SpAevec = St.PCA(np.log10(np.ma.compressed(SA.data)),  np.log10(np.ma.compressed(pA.data)))
        NSAeval, NSAevec = St.PCA(np.log10(np.ma.compressed(CDA.data)), np.log10(np.ma.compressed(SA.data)))
        # 2D KDE.
        xxNpA, yyNpA, fNpA = St.Gaussian2DKDE(np.log10(np.ma.compressed(CDA.data)), np.log10(np.ma.compressed(pA.data)), Nbnds, pbnds)
        xxSpA, yySpA, fSpA = St.Gaussian2DKDE(np.log10(np.ma.compressed(SA.data)),  np.log10(np.ma.compressed(pA.data)), Sbnds, pbnds)
        xxNSA, yyNSA, fNSA = St.Gaussian2DKDE(np.log10(np.ma.compressed(CDA.data)), np.log10(np.ma.compressed(SA.data)), Nbnds, Sbnds)
        # Compute Correlation Coefficients.
        NpApr, NpAsr = St.CorrCoeffs(np.log10(np.ma.compressed(CDA.data)), np.log10(np.ma.compressed(pA.data)), False)
        SpApr, SpAsr = St.CorrCoeffs(np.log10(np.ma.compressed(SA.data)),  np.log10(np.ma.compressed(pA.data)), False)
        NSApr, NSAsr = St.CorrCoeffs(np.log10(np.ma.compressed(CDA.data)), np.log10(np.ma.compressed(SA.data)), False)
        # Compute means. 
        meanCDA = np.average(np.log10(np.ma.compressed(CDA.data)))
        meanpA  = np.average(np.log10(np.ma.compressed( pA.data)))
        meanSA  = np.average(np.log10(np.ma.compressed( SA.data))) 
        # Compute geometric medians. 
        medCDA = np.median(np.log10(np.ma.compressed(CDA.data)))
        medpA  = np.median(np.log10(np.ma.compressed( pA.data)))
        medSA  = np.median(np.log10(np.ma.compressed( SA.data)))
        # Compute geometric std deviations. 
        stdCDA = np.std(np.log10(np.ma.compressed(CDA.data)))
        stdpA  = np.std(np.log10(np.ma.compressed( pA.data)))   
        stdSA  = np.std(np.log10(np.ma.compressed( SA.data)))
        # Compute geometric skews. 
        skewCDA = stats.skew(np.log10(np.ma.compressed(CDA.data)))
        skewpA  = stats.skew(np.log10(np.ma.compressed( pA.data)))
        skewSA  = stats.skew(np.log10(np.ma.compressed( SA.data)))
        # Compute geometric kurtoses. 
        kurtCDA = stats.kurtosis(np.log10(np.ma.compressed(CDA.data)), fisher=True)
        kurtpA  = stats.kurtosis(np.log10(np.ma.compressed( pA.data)), fisher=True)
        kurtSA  = stats.kurtosis(np.log10(np.ma.compressed( SA.data)), fisher=True)
        # Write Stats. 
        fileNA.write( 'Homogeneous, {: f}, {: f}, {: f}, {: f}, {: f} \n'.format(meanCDA, medCDA, stdCDA, skewCDA, kurtCDA))
        filepA.write( 'Homogeneous, {: f}, {: f}, {: f}, {: f}, {: f} \n'.format(meanpA,  medpA,  stdpA,  skewpA,  kurtpA))
        fileSA.write( 'Homogeneous, {: f}, {: f}, {: f}, {: f}, {: f} \n'.format(meanSA,  medSA,  stdSA,  skewSA,  kurtSA))
        fileNpA.write('Homogeneous, {: f}, {: f}, {: f}, {: f}, {: f}, {: f}, {: f}, {: f} \n'.format(NpAeval[1], NpAevec[1,0], NpAevec[1,1], NpAeval[0], NpAevec[0,0], NpAevec[0,1], NpApr, NpAsr))
        fileNSA.write('Homogeneous, {: f}, {: f}, {: f}, {: f}, {: f}, {: f}, {: f}, {: f} \n'.format(NSAeval[1], NSAevec[1,0], NSAevec[1,1], NSAeval[0], NSAevec[0,0], NSAevec[0,1], NSApr, NSAsr))
        fileSpA.write('Homogeneous, {: f}, {: f}, {: f}, {: f}, {: f}, {: f}, {: f}, {: f} \n'.format(SpAeval[1], SpAevec[1,0], SpAevec[1,1], SpAeval[0], SpAevec[0,0], SpAevec[0,1], SpApr, SpAsr))
        # Save 1D KDE.
        St.Save1DKDE(xNA, fNA, 'N_XY_A_ind'+str(i)+'_cut'+str(j))
        St.Save1DKDE(xpA, fpA, 'p_XY_A_ind'+str(i)+'_cut'+str(j))
        St.Save1DKDE(xSA, fSA, 'S_XY_A_ind'+str(i)+'_cut'+str(j))
        # Save 2D KDE. 
        St.Save2DKDE(xxNpA, yyNpA, fNpA, 'Np_XY_A_ind'+str(i)+'_cut'+str(j), 'Gaussian')
        St.Save2DKDE(xxNSA, yyNSA, fNSA, 'NS_XY_A_ind'+str(i)+'_cut'+str(j), 'Gaussian')
        St.Save2DKDE(xxSpA, yySpA, fSpA, 'Sp_XY_A_ind'+str(i)+'_cut'+str(j), 'Gaussian')       
        # plotting. 
        if j == 0:
            axp[0].semilogy(xpA, fpA, '-', linewidth=1.25, color=colors[3-i], zorder=2)
            axS[0].semilogy(xSA, fSA, '-', linewidth=1.25, color=colors[3-i], zorder=2)
        else:
            axp[0].semilogy(xpA, fpA, '--', linewidth=1.25, color=colors[3-i], zorder=2)
            axS[0].semilogy(xSA, fSA, '--', linewidth=1.25, color=colors[3-i], zorder=2)

        axNp[j,i+1].contour(xxNpA, yyNpA, fNpA, 10, colors=colors[3],zorder=1)
        axNp[j,i+1].quiver(meanCDA, meanpA, np.sqrt(NpAeval[0])*abs(NpAevec[0,0]),  np.sqrt(NpAeval[0])*abs(NpAevec[1,0]), scale=1.25, zorder=2)
        axNp[j,i+1].quiver(meanCDA, meanpA, np.sqrt(NpAeval[1])*abs(NpAevec[0,1]), -np.sqrt(NpAeval[1])*abs(NpAevec[1,1]), scale=1.25, zorder=2)
        axNp[j,i+1].plot(meanCDA, meanpA, 'k.', markersize=5)

        axSp[j,i+1].contour(xxSpA, yySpA, fSpA, 10, colors=colors[3],zorder=1)
        axSp[j,i+1].quiver(meanSA, meanpA, np.sqrt(SpAeval[0])*abs(SpAevec[0,0]),  np.sqrt(SpAeval[0])*abs(SpAevec[1,0]), scale=1.25, zorder=2)
        axSp[j,i+1].quiver(meanSA, meanpA, np.sqrt(SpAeval[1])*abs(SpAevec[0,1]), -np.sqrt(SpAeval[1])*abs(SpAevec[1,1]), scale=1.25, zorder=2)
        axSp[j,i+1].plot(meanSA, meanpA, 'k.', markersize=5)

        axNS[j,i+1].contour(xxNSA, yyNSA, fNSA, 10, colors=colors[3],zorder=1)
        axNS[j,i+1].quiver(meanCDA, meanSA, np.sqrt(NSAeval[0])*NSAevec[0,0], np.sqrt(NSAeval[0])*NSAevec[1,0], scale=1.25, zorder=2)
        axNS[j,i+1].quiver(meanCDA, meanSA, np.sqrt(NSAeval[1])*NSAevec[0,1], np.sqrt(NSAeval[1])*NSAevec[1,1], scale=1.25, zorder=2)
        axNS[j,i+1].plot(meanCDA, meanSA, 'k.', markersize=5)

        del xNA, xpA, xSA, fNA, fpA, fSA
        del xxNpA, yyNpA, fNpA 
        del xxNSA, yyNSA, fNSA 
        del xxSpA, yySpA, fSpA
        del NpApr, NpAsr, NSApr, NSAsr, SpApr, SpAsr
        del meanCDA, meanpA, meanSA
        del medCDA, medpA, medSA 
        del stdCDA, stdpA, stdSA 
        del skewCDA, skewpA, skewSA 
        del kurtCDA, kurtpA, kurtSA 
        del NpAeval, NpAevec 
        del NSAeval, NSAevec 
        del SpAeval, SpAevec 
        print("Sim A Index "+str(i)+" Cutoff "+str(j)+" Complete.")

# Close files.
fileNA.close()
filepA.close() 
fileSA.close()
fileNpA.close()
fileNSA.close()
fileSpA.close()

# Remove simdata to prevent memoryerror. 
del SA, pA, UA, QA, CDA 
del ObsA 
del OA

# Open files to write stats to.
fileNB  = open('N_XY_B_stats.txt', 'w')
filepB  = open('p_XY_B_stats.txt', 'w')
fileSB  = open('S_XY_B_stats.txt', 'w')
fileNpB = open('Np_XY_B_stats.txt', 'w')
fileNSB = open('NS_XY_B_stats.txt', 'w')
fileSpB = open('Sp_XY_B_stats.txt', 'w')

fileNB.write( '# Model N_geomean N_geomed N_geostd N_geoskew N_geokurt \n')
filepB.write( '# Model p_geomean p_geomed p_geostd p_geoskew p_geokurt \n')
fileSB.write( '# Model S_geomean S_geomed S_geostd S_geoskew S_geokurt \n')
fileNpB.write('# Model Npevalmax Npevecmax1 Npevecmax2 Npevalmin Npevecmin1 Npevecmin2 NpPearson NpSpearman \n')
fileNSB.write('# Model NSevalmax NSevecmax1 NSevecmax2 NSevalmin NSevecmin1 NSevecmin2 NSPearson NSSpearman \n')
fileSpB.write('# Model Spevalmax Spevecmax1 Spevecmax2 Spevalmin Spevecmin1 Spevecmin2 SpPearson SpSpearman \n')

# Load Sim B.
OB = Observer(['ccldL10B5M10n_0025.vtk',N,boxlenB,'./'])
OB.ChangeMagneticHandle(['cell_centered_B_x','cell_centered_B_y','cell_centered_B_z'])
OB.DataLoad()
NB = Nabla([None,N,boxlenB])
PltB = Plotter([N, boxlenB, 5, 100, './'])
print('Sim B Data Loaded.')

# Next plot the unmodified data for Sim B.
pol_args = ['Constant', p0]
ObsB = OB.Polarimetry(exc_args, rot_args, pol_args)
CDB = ObsB[3]
QB = ObsB[1]
UB = ObsB[2]
pB = ObsB[6]
SB = NB.ComputeAngleGradient(QB, UB, QB.data.mask)
# 1D KDE.
xNB, fNB = St.Gaussian1DKDE(np.log10(np.ma.compressed(CDB.data)), Nbnds)
xpB, fpB = St.Gaussian1DKDE(np.log10(np.ma.compressed(pB.data)), pbnds)
xSB, fSB = St.Gaussian1DKDE(np.log10(np.ma.compressed(SB.data)), Sbnds)
# PCA.
NpBeval, NpBevec = St.PCA(np.log10(np.ma.compressed(CDB.data)), np.log10(np.ma.compressed(pB.data)))
SpBeval, SpBevec = St.PCA(np.log10(np.ma.compressed(SB.data)),  np.log10(np.ma.compressed(pB.data)))
NSBeval, NSBevec = St.PCA(np.log10(np.ma.compressed(CDB.data)), np.log10(np.ma.compressed(SB.data)))
# KDE.
xxNpB, yyNpB, fNpB = St.Gaussian2DKDE(np.log10(np.ma.compressed(CDB.data)), np.log10(np.ma.compressed(pB.data)), Nbnds, pbnds)
xxSpB, yySpB, fSpB = St.Gaussian2DKDE(np.log10(np.ma.compressed(SB.data)),  np.log10(np.ma.compressed(pB.data)), Sbnds, pbnds)
xxNSB, yyNSB, fNSB = St.Gaussian2DKDE(np.log10(np.ma.compressed(CDB.data)), np.log10(np.ma.compressed(SB.data)), Nbnds, Sbnds)
# Compute Correlation Coefficients.
NpBpr, NpBsr = St.CorrCoeffs(np.log10(np.ma.compressed(CDB.data)), np.log10(np.ma.compressed(pB.data)), False)
SpBpr, SpBsr = St.CorrCoeffs(np.log10(np.ma.compressed(SB.data)),  np.log10(np.ma.compressed(pB.data)), False)
NSBpr, NSBsr = St.CorrCoeffs(np.log10(np.ma.compressed(CDB.data)), np.log10(np.ma.compressed(SB.data)), False)
# Compute means. 
meanCDB = np.average(np.log10(np.ma.compressed(CDB.data)))
meanpB  = np.average(np.log10(np.ma.compressed( pB.data)))
meanSB  = np.average(np.log10(np.ma.compressed( SB.data)))  
# Compute geometric medians. 
medCDB = np.median(np.log10(np.ma.compressed(CDB.data)))
medpB  = np.median(np.log10(np.ma.compressed( pB.data)))
medSB  = np.median(np.log10(np.ma.compressed( SB.data)))
# Compute geometric std deviations. 
stdCDB = np.std(np.log10(np.ma.compressed(CDB.data)))
stdpB  = np.std(np.log10(np.ma.compressed( pB.data)))
stdSB  = np.std(np.log10(np.ma.compressed( SB.data)))
# Compute geometric skews. 
skewCDB = stats.skew(np.log10(np.ma.compressed(CDB.data)))
skewpB  = stats.skew(np.log10(np.ma.compressed( pB.data)))
skewSB  = stats.skew(np.log10(np.ma.compressed( SB.data)))
# Compute geometric kurtoses. 
kurtCDB = stats.kurtosis(np.log10(np.ma.compressed(CDB.data)), fisher=True)
kurtpB  = stats.kurtosis(np.log10(np.ma.compressed( pB.data)), fisher=True)
kurtSB  = stats.kurtosis(np.log10(np.ma.compressed( SB.data)), fisher=True)
# Write Stats. 
fileNB.write( 'Homogeneous, {: f}, {: f}, {: f}, {: f}, {: f} \n'.format(meanCDB, medCDB, stdCDB, skewCDB, kurtCDB))
filepB.write( 'Homogeneous, {: f}, {: f}, {: f}, {: f}, {: f} \n'.format(meanpB,  medpB,  stdpB,  skewpB,  kurtpB))
fileSB.write( 'Homogeneous, {: f}, {: f}, {: f}, {: f}, {: f} \n'.format(meanSB,  medSB,  stdSB,  skewSB,  kurtSB))
fileNpB.write('Homogeneous, {: f}, {: f}, {: f}, {: f}, {: f}, {: f}, {: f}, {: f} \n'.format(NpBeval[1], NpBevec[1,0], NpBevec[1,1], NpBeval[0], NpBevec[0,0], NpBevec[0,1], NpBpr, NpBsr))
fileNSB.write('Homogeneous, {: f}, {: f}, {: f}, {: f}, {: f}, {: f}, {: f}, {: f} \n'.format(NSBeval[1], NSBevec[1,0], NSBevec[1,1], NSBeval[0], NSBevec[0,0], NSBevec[0,1], NSBpr, NSBsr))
fileSpB.write('Homogeneous, {: f}, {: f}, {: f}, {: f}, {: f}, {: f}, {: f}, {: f} \n'.format(SpBeval[1], SpBevec[1,0], SpBevec[1,1], SpBeval[0], SpBevec[0,0], SpBevec[0,1], SpBpr, SpBsr))  
# Save 1D KDE.
St.Save1DKDE(xNB, fNB, 'N_XY_B_homo')
St.Save1DKDE(xpB, fpB, 'p_XY_B_homo')
St.Save1DKDE(xSB, fSB, 'S_XY_B_homo')
# Save 2D KDE. 
St.Save2DKDE(xxNpB, yyNpB, fNpB, 'Np_XY_B_homo', 'Gaussian')
St.Save2DKDE(xxNSB, yyNSB, fNSB, 'NS_XY_B_homo', 'Gaussian')
St.Save2DKDE(xxSpB, yySpB, fSpB, 'Sp_XY_B_homo', 'Gaussian')     
# plotting. 
axp[1].semilogy(xpB, fpB, '-', linewidth=1.25, color=colors[4], zorder=2)
axS[1].semilogy(xSB, fSB, '-', linewidth=1.25, color=colors[4], zorder=2)

axNp[0,0].contour(xxNpB, yyNpB, fNpB, 10, colors=colors[1],zorder=1)
axNp[0,0].quiver(meanCDB, meanpB, -np.sqrt(NpBeval[0])*NpBevec[0,0], -np.sqrt(NpBeval[0])*NpBevec[1,0], scale=1.25, zorder=2)
axNp[0,0].quiver(meanCDB, meanpB, -np.sqrt(NpBeval[1])*NpBevec[0,1], -np.sqrt(NpBeval[1])*NpBevec[1,1], scale=1.25, zorder=2)
axNp[0,0].plot(meanCDB, meanpB, 'k.', markersize=5)

axSp[0,0].contour(xxSpB, yySpB, fSpB, 10, colors=colors[1],zorder=1)
axSp[0,0].quiver(meanSB, meanpB, np.sqrt(SpBeval[0])*abs(SpBevec[0,0]),  np.sqrt(SpBeval[0])*abs(SpBevec[1,0]), scale=1.25, zorder=2)
axSp[0,0].quiver(meanSB, meanpB, np.sqrt(SpBeval[1])*abs(SpBevec[0,1]), -np.sqrt(SpBeval[1])*abs(SpBevec[1,1]), scale=1.25, zorder=2)
axSp[0,0].plot(meanSB, meanpB, 'k.', markersize=5)

axNS[0,0].contour(xxNSB, yyNSB, fNSB, 10, colors=colors[1],zorder=1)
axNS[0,0].quiver(meanCDB, meanSB, np.sqrt(NSBeval[0])*NSBevec[0,0], np.sqrt(NSBeval[0])*NSBevec[1,0], scale=1.25, zorder=2)
axNS[0,0].quiver(meanCDB, meanSB, np.sqrt(NSBeval[1])*NSBevec[0,1], np.sqrt(NSBeval[1])*NSBevec[1,1], scale=1.25, zorder=2)
axNS[0,0].plot(meanCDB, meanSB, 'k.', markersize=5)

del xNB, xpB, xSB, fNB, fpB, fSB
del xxNpB, yyNpB, fNpB 
del xxNSB, yyNSB, fNSB 
del xxSpB, yySpB, fSpB
del NpBpr, NpBsr, NSBpr, NSBsr, SpBpr, SpBsr
del meanCDB, meanpB, meanSB
del medCDB, medpB, medSB 
del stdCDB, stdpB, stdSB 
del skewCDB, skewpB, skewSB 
del kurtCDB, kurtpB, kurtSB 
del NpBeval, NpBevec 
del NSBeval, NSBevec 
del SpBeval, SpBevec 
print("Sim B Index Constant p0 Complete.")

# Now loop for B. 
for i in range(len(indices)):
    for j in range(len(cutoffs)):
        pol_args = ['Power-Law', p0, cutoffs[j], indices[i]]
        ObsB = OB.Polarimetry(exc_args, rot_args, pol_args)
        CDB = ObsB[3]
        QB  = ObsB[1]
        UB  = ObsB[2]
        pB  = ObsB[6]
        SB  = NB.ComputeAngleGradient(QB, UB, QB.data.mask)
        # 1D KDE.
        xNB, fNB = St.Gaussian1DKDE(np.log10(np.ma.compressed(CDB.data)), Nbnds)
        xpB, fpB = St.Gaussian1DKDE(np.log10(np.ma.compressed(pB.data)), pbnds)
        xSB, fSB = St.Gaussian1DKDE(np.log10(np.ma.compressed(SB.data)), Sbnds)
        # PCA.
        NpBeval, NpBevec = St.PCA(np.log10(np.ma.compressed(CDB.data)), np.log10(np.ma.compressed(pB.data)))
        SpBeval, SpBevec = St.PCA(np.log10(np.ma.compressed(SB.data)),  np.log10(np.ma.compressed(pB.data)))
        NSBeval, NSBevec = St.PCA(np.log10(np.ma.compressed(CDB.data)), np.log10(np.ma.compressed(SB.data)))
        # 1D KDE.
        xxNpB, yyNpB, fNpB = St.Gaussian2DKDE(np.log10(np.ma.compressed(CDB.data)), np.log10(np.ma.compressed(pB.data)), Nbnds, pbnds)
        xxSpB, yySpB, fSpB = St.Gaussian2DKDE(np.log10(np.ma.compressed(SB.data)),  np.log10(np.ma.compressed(pB.data)), Sbnds, pbnds)
        xxNSB, yyNSB, fNSB = St.Gaussian2DKDE(np.log10(np.ma.compressed(CDB.data)), np.log10(np.ma.compressed(SB.data)), Nbnds, Sbnds)
        # Compute Correlation Coefficients.
        NpBpr, NpBsr = St.CorrCoeffs(np.log10(np.ma.compressed(CDB.data)), np.log10(np.ma.compressed(pB.data)), False)
        SpBpr, SpBsr = St.CorrCoeffs(np.log10(np.ma.compressed(SB.data)),  np.log10(np.ma.compressed(pB.data)), False)
        NSBpr, NSBsr = St.CorrCoeffs(np.log10(np.ma.compressed(CDB.data)), np.log10(np.ma.compressed(SB.data)), False)
        # Compute means. 
        meanCDB = np.average(np.log10(np.ma.compressed(CDB.data)))
        meanpB  = np.average(np.log10(np.ma.compressed( pB.data)))
        meanSB  = np.average(np.log10(np.ma.compressed( SB.data)))
        # Compute geometric medians. 
        medCDB = np.median(np.log10(np.ma.compressed(CDB.data)))
        medpB  = np.median(np.log10(np.ma.compressed( pB.data)))
        medSB  = np.median(np.log10(np.ma.compressed( SB.data)))
        # Compute geometric std deviations. 
        stdCDB = np.std(np.log10(np.ma.compressed(CDB.data)))
        stdpB  = np.std(np.log10(np.ma.compressed( pB.data)))
        stdSB  = np.std(np.log10(np.ma.compressed( SB.data)))
        # Compute geometric skews. 
        skewCDB = stats.skew(np.log10(np.ma.compressed(CDB.data)))
        skewpB  = stats.skew(np.log10(np.ma.compressed( pB.data)))
        skewSB  = stats.skew(np.log10(np.ma.compressed( SB.data)))
        # Compute geometric kurtoses. 
        kurtCDB = stats.kurtosis(np.log10(np.ma.compressed(CDB.data)), fisher=True)
        kurtpB  = stats.kurtosis(np.log10(np.ma.compressed( pB.data)), fisher=True)
        kurtSB  = stats.kurtosis(np.log10(np.ma.compressed( SB.data)), fisher=True)
        # Write Stats. 
        fileNB.write( 'Homogeneous, {: f}, {: f}, {: f}, {: f}, {: f} \n'.format(meanCDB, medCDB, stdCDB, skewCDB, kurtCDB))
        filepB.write( 'Homogeneous, {: f}, {: f}, {: f}, {: f}, {: f} \n'.format(meanpB,  medpB,  stdpB,  skewpB,  kurtpB))
        fileSB.write( 'Homogeneous, {: f}, {: f}, {: f}, {: f}, {: f} \n'.format(meanSB,  medSB,  stdSB,  skewSB,  kurtSB))
        fileNpB.write('Homogeneous, {: f}, {: f}, {: f}, {: f}, {: f}, {: f}, {: f}, {: f} \n'.format(NpBeval[1], NpBevec[1,0], NpBevec[1,1], NpBeval[0], NpBevec[0,0], NpBevec[0,1], NpBpr, NpBsr))
        fileNSB.write('Homogeneous, {: f}, {: f}, {: f}, {: f}, {: f}, {: f}, {: f}, {: f} \n'.format(NSBeval[1], NSBevec[1,0], NSBevec[1,1], NSBeval[0], NSBevec[0,0], NSBevec[0,1], NSBpr, NSBsr))
        fileSpB.write('Homogeneous, {: f}, {: f}, {: f}, {: f}, {: f}, {: f}, {: f}, {: f} \n'.format(SpBeval[1], SpBevec[1,0], SpBevec[1,1], SpBeval[0], SpBevec[0,0], SpBevec[0,1], SpBpr, SpBsr))  
        # Save 1D KDE.
        St.Save1DKDE(xNB, fNB, 'N_XY_B_ind'+str(i)+'_cut'+str(j))
        St.Save1DKDE(xpB, fpB, 'p_XY_B_ind'+str(i)+'_cut'+str(j))
        St.Save1DKDE(xSB, fSB, 'S_XY_B_ind'+str(i)+'_cut'+str(j))
        # Save 2D KDE. 
        St.Save2DKDE(xxNpB, yyNpB, fNpB, 'Np_XY_B_ind'+str(i)+'_cut'+str(j), 'Gaussian')
        St.Save2DKDE(xxNSB, yyNSB, fNSB, 'NS_XY_B_ind'+str(i)+'_cut'+str(j), 'Gaussian')
        St.Save2DKDE(xxSpB, yySpB, fSpB, 'Sp_XY_B_ind'+str(i)+'_cut'+str(j), 'Gaussian')        
        # plotting. 
        if j == 0:
            axp[1].semilogy(xpB, fpB, '-', linewidth=1.25, color=colors[3-i], zorder=2)
            axS[1].semilogy(xSB, fSB, '-', linewidth=1.25, color=colors[3-i], zorder=2)
        else:
            axp[1].semilogy(xpB, fpB, '--', linewidth=1.25, color=colors[3-i], zorder=2)
            axS[1].semilogy(xSB, fSB, '--', linewidth=1.25, color=colors[3-i], zorder=2)

        axNp[j,i+1].contour(xxNpB, yyNpB, fNpB, 10, colors=colors[1],zorder=1)
        axNp[j,i+1].quiver(meanCDB, meanpB, np.sqrt(NpBeval[0])*abs(NpBevec[0,0]),  np.sqrt(NpBeval[0])*abs(NpBevec[1,0]), scale=1.25, zorder=2)
        axNp[j,i+1].quiver(meanCDB, meanpB, np.sqrt(NpBeval[1])*abs(NpBevec[0,1]), -np.sqrt(NpBeval[1])*abs(NpBevec[1,1]), scale=1.25, zorder=2)
        axNp[j,i+1].plot(meanCDB, meanpB, 'k.', markersize=5)

        axSp[j,i+1].contour(xxSpB, yySpB, fSpB, 10, colors=colors[1],zorder=1)
        axSp[j,i+1].quiver(meanSB, meanpB, np.sqrt(SpBeval[0])*abs(SpBevec[0,0]),  np.sqrt(SpBeval[0])*abs(SpBevec[1,0]), scale=1.25, zorder=2)
        axSp[j,i+1].quiver(meanSB, meanpB, np.sqrt(SpBeval[1])*abs(SpBevec[0,1]), -np.sqrt(SpBeval[1])*abs(SpBevec[1,1]), scale=1.25, zorder=2)
        axSp[j,i+1].plot(meanSB, meanpB, 'k.', markersize=5)

        axNS[j,i+1].contour(xxNSB, yyNSB, fNSB, 10, colors=colors[1],zorder=1)
        axNS[j,i+1].quiver(meanCDB, meanSB, np.sqrt(NSBeval[0])*NSBevec[0,0], np.sqrt(NSBeval[0])*NSBevec[1,0], scale=1.25, zorder=2)
        axNS[j,i+1].quiver(meanCDB, meanSB, np.sqrt(NSBeval[1])*NSBevec[0,1], np.sqrt(NSBeval[1])*NSBevec[1,1], scale=1.25, zorder=2)
        axNS[j,i+1].plot(meanCDB, meanSB, 'k.', markersize=5)

        del xNB, xpB, xSB, fNB, fpB, fSB
        del xxNpB, yyNpB, fNpB 
        del xxNSB, yyNSB, fNSB 
        del xxSpB, yySpB, fSpB
        del NpBpr, NpBsr, NSBpr, NSBsr, SpBpr, SpBsr
        del meanCDB, meanpB, meanSB
        del medCDB, medpB, medSB 
        del stdCDB, stdpB, stdSB 
        del skewCDB, skewpB, skewSB 
        del kurtCDB, kurtpB, kurtSB 
        del NpBeval, NpBevec 
        del NSBeval, NSBevec 
        del SpBeval, SpBevec 
        print("Sim B Index "+str(i)+" Cutoff "+str(j)+" Complete.")

# Remove simdata to prevent memoryerror. 
del SB, pB, UB, QB, CDB 
del ObsB 
del OB

# Close files.
fileNB.close()
filepB.close() 
fileSB.close()
fileNpB.close()
fileNSB.close()
fileSpB.close()

# open files to write stats to.
fileNC  = open('N_XY_C_stats.txt', 'w')
filepC  = open('p_XY_C_stats.txt', 'w')
fileSC  = open('S_XY_C_stats.txt', 'w')
fileNpC = open('Np_XY_C_stats.txt', 'w')
fileNSC = open('NS_XY_C_stats.txt', 'w')
fileSpC = open('Sp_XY_C_stats.txt', 'w')

fileNC.write( '# Model N_geomean N_geomed N_geostd N_geoskew N_geokurt \n')
filepC.write( '# Model p_geomean p_geomed p_geostd p_geoskew p_geokurt \n')
fileSC.write( '# Model S_geomean S_geomed S_geostd S_geoskew S_geokurt \n')
fileNpC.write('# Model Npevalmax Npevecmax1 Npevecmax2 Npevalmin Npevecmin1 Npevecmin2 NpPearson NpSpearman \n')
fileNSC.write('# Model NSevalmax NSevecmax1 NSevecmax2 NSevalmin NSevecmin1 NSevecmin2 NSPearson NSSpearman \n')
fileSpC.write('# Model Spevalmax Spevecmax1 Spevecmax2 Spevalmin Spevecmin1 Spevecmin2 SpPearson SpSpearman \n')

# load Sim C data.
OC = Observer(['L5B2M10_0026.vtk',N,boxlenC,'./'])
OC.ChangeMagneticHandle(['cell_centered_B_x', 'cell_centered_B_y', 'cell_centered_B_z'])
OC.DataLoad()
NC = Nabla([None,N,boxlenC])
PltC = Plotter([N, boxlenC, 5, 100, './'])
print('Sim C Data Loaded.')

# Next plot the unmodified data for sim C. 
pol_args = ['Constant', p0]
ObsC = OC.Polarimetry(exc_args, rot_args, pol_args)
CDC = ObsC[3]
CDC.data *= scaleC
QC = ObsC[1]
UC = ObsC[2]
pC = ObsC[6]
SC = NC.ComputeAngleGradient(QC, UC, QC.data.mask)
# 1D KDE.
xNC, fNC = St.Gaussian1DKDE(np.log10(np.ma.compressed(CDC.data)), Nbnds)
xpC, fpC = St.Gaussian1DKDE(np.log10(np.ma.compressed(pC.data)), pbnds)
xSC, fSC = St.Gaussian1DKDE(np.log10(np.ma.compressed(SC.data)), Sbnds)
# PCA.
NpCeval, NpCevec = St.PCA(np.log10(np.ma.compressed(CDC.data)), np.log10(np.ma.compressed(pC.data)))
SpCeval, SpCevec = St.PCA(np.log10(np.ma.compressed(SC.data)),  np.log10(np.ma.compressed(pC.data)))
NSCeval, NSCevec = St.PCA(np.log10(np.ma.compressed(CDC.data)), np.log10(np.ma.compressed(SC.data)))
# 2D KDE.
xxNpC, yyNpC, fNpC = St.Gaussian2DKDE(np.log10(np.ma.compressed(CDC.data)), np.log10(np.ma.compressed(pC.data)), Nbnds, pbnds)
xxSpC, yySpC, fSpC = St.Gaussian2DKDE(np.log10(np.ma.compressed(SC.data)),  np.log10(np.ma.compressed(pC.data)), Sbnds, pbnds)
xxNSC, yyNSC, fNSC = St.Gaussian2DKDE(np.log10(np.ma.compressed(CDC.data)), np.log10(np.ma.compressed(SC.data)), Nbnds, Sbnds)
# Compute Correlation Coefficients.
NpCpr, NpCsr = St.CorrCoeffs(np.log10(np.ma.compressed(CDC.data)), np.log10(np.ma.compressed(pC.data)), False)
SpCpr, SpCsr = St.CorrCoeffs(np.log10(np.ma.compressed(SC.data)),  np.log10(np.ma.compressed(pC.data)), False)
NSCpr, NSCsr = St.CorrCoeffs(np.log10(np.ma.compressed(CDC.data)), np.log10(np.ma.compressed(SC.data)), False)
# Compute means. 
meanCDC = np.average(np.log10(np.ma.compressed(CDC.data)))
meanpC  = np.average(np.log10(np.ma.compressed( pC.data)))
meanSC  = np.average(np.log10(np.ma.compressed( SC.data)))  
# Compute geometric medians. 
medCDC = np.median(np.log10(np.ma.compressed(CDC.data)))
medpC  = np.median(np.log10(np.ma.compressed( pC.data)))
medSC  = np.median(np.log10(np.ma.compressed( SC.data)))
# Compute geometric std deviations. 
stdCDC = np.std(np.log10(np.ma.compressed(CDC.data)))
stdpC  = np.std(np.log10(np.ma.compressed( pC.data)))
stdSC  = np.std(np.log10(np.ma.compressed( SC.data)))
# Compute geometric skews. 
skewCDC = stats.skew(np.log10(np.ma.compressed(CDC.data)))
skewpC  = stats.skew(np.log10(np.ma.compressed( pC.data)))
skewSC  = stats.skew(np.log10(np.ma.compressed( SC.data)))
# Compute geometric kurtoses. 
kurtCDC = stats.kurtosis(np.log10(np.ma.compressed(CDC.data)), fisher=True)
kurtpC  = stats.kurtosis(np.log10(np.ma.compressed( pC.data)), fisher=True)
kurtSC  = stats.kurtosis(np.log10(np.ma.compressed( SC.data)), fisher=True)
# Write Stats. 
fileNC.write( 'Homogeneous, {: f}, {: f}, {: f}, {: f}, {: f} \n'.format(meanCDC, medCDC, stdCDC, skewCDC, kurtCDC))
filepC.write( 'Homogeneous, {: f}, {: f}, {: f}, {: f}, {: f} \n'.format(meanpC,  medpC,  stdpC,  skewpC,  kurtpC))
fileSC.write( 'Homogeneous, {: f}, {: f}, {: f}, {: f}, {: f} \n'.format(meanSC,  medSC,  stdSC,  skewSC,  kurtSC))
fileNpC.write('Homogeneous, {: f}, {: f}, {: f}, {: f}, {: f}, {: f}, {: f}, {: f} \n'.format(NpCeval[1], NpCevec[1,0], NpCevec[1,1], NpCeval[0], NpCevec[0,0], NpCevec[0,1], NpCpr, NpCsr))
fileNSC.write('Homogeneous, {: f}, {: f}, {: f}, {: f}, {: f}, {: f}, {: f}, {: f} \n'.format(NSCeval[1], NSCevec[1,0], NSCevec[1,1], NSCeval[0], NSCevec[0,0], NSCevec[0,1], NSCpr, NSCsr))
fileSpC.write('Homogeneous, {: f}, {: f}, {: f}, {: f}, {: f}, {: f}, {: f}, {: f} \n'.format(SpCeval[1], SpCevec[1,0], SpCevec[1,1], SpCeval[0], SpCevec[0,0], SpCevec[0,1], SpCpr, SpCsr)) 
# Save 1D KDE.
St.Save1DKDE(xNC, fNC, 'N_XY_C_homo')
St.Save1DKDE(xpC, fpC, 'p_XY_C_homo')
St.Save1DKDE(xSC, fSC, 'S_XY_C_homo')
# Save 2D KDE. 
St.Save2DKDE(xxNpC, yyNpC, fNpC, 'Np_XY_C_homo', 'Gaussian')
St.Save2DKDE(xxNSC, yyNSC, fNSC, 'NS_XY_C_homo', 'Gaussian')
St.Save2DKDE(xxSpC, yySpC, fSpC, 'Sp_XY_C_homo', 'Gaussian')       
# plotting. 
axp[2].semilogy(xpC, fpC, '-', linewidth=1.25, color=colors[4], zorder=2)
axS[2].semilogy(xSC, fSC, '-', linewidth=1.25, color=colors[4], zorder=2)

axNp[0,0].contour(xxNpC, yyNpC, fNpC, 10, colors=colors[2],zorder=1)
axNp[0,0].quiver(meanCDC, meanpC, np.sqrt(NpCeval[0])*NpCevec[0,0], np.sqrt(NpCeval[0])*NpCevec[1,0], scale=1.25, zorder=2)
axNp[0,0].quiver(meanCDC, meanpC, np.sqrt(NpCeval[1])*NpCevec[0,1], np.sqrt(NpCeval[1])*NpCevec[1,1], scale=1.25, zorder=2)
axNp[0,0].plot(meanCDC, meanpC, 'k.', markersize=5)

axSp[0,0].contour(xxSpC, yySpC, fSpC, 10, colors=colors[2],zorder=1)
axSp[0,0].quiver(meanSC, meanpC, np.sqrt(SpCeval[0])*abs(SpCevec[0,0]),  np.sqrt(SpCeval[0])*abs(SpCevec[1,0]), scale=1.25, zorder=2)
axSp[0,0].quiver(meanSC, meanpC, np.sqrt(SpCeval[1])*abs(SpCevec[0,1]), -np.sqrt(SpCeval[1])*abs(SpCevec[1,1]), scale=1.25, zorder=2)
axSp[0,0].plot(meanSC, meanpC, 'k.', markersize=5)

axNS[0,0].contour(xxNSC, yyNSC, fNSC, 10, colors=colors[2],zorder=1)
axNS[0,0].quiver(meanCDC, meanSC, np.sqrt(NSCeval[0])*NSCevec[0,0], np.sqrt(NSCeval[0])*NSCevec[1,0], scale=1.25, zorder=2)
axNS[0,0].quiver(meanCDC, meanSC, np.sqrt(NSCeval[1])*NSCevec[0,1], np.sqrt(NSCeval[1])*NSCevec[1,1], scale=1.25, zorder=2)
axNS[0,0].plot(meanCDC, meanSC, 'k.', markersize=5)

del xNC, xpC, xSC, fNC, fpC, fSC
del xxNpC, yyNpC, fNpC 
del xxNSC, yyNSC, fNSC 
del xxSpC, yySpC, fSpC
del NpCpr, NpCsr, NSCpr, NSCsr, SpCpr, SpCsr
del meanCDC, meanpC, meanSC
del medCDC, medpC, medSC 
del stdCDC, stdpC, stdSC 
del skewCDC, skewpC, skewSC 
del kurtCDC, kurtpC, kurtSC 
del NpCeval, NpCevec 
del NSCeval, NSCevec 
del SpCeval, SpCevec
print("Sim C Index Constant p0 Complete.")

# Now loop for C. 
for i in range(len(indices)):
    for j in range(len(cutoffs)):
        pol_args = ['Power-Law', p0, cutoffs[j]/(scaleC**2), indices[i]]
        ObsC = OC.Polarimetry(exc_args, rot_args, pol_args)
        CDC = ObsC[3]
        CDC.data *= scaleC
        QC  = ObsC[1]
        UC  = ObsC[2]
        pC  = ObsC[6]
        SC  = NC.ComputeAngleGradient(QC, UC, QC.data.mask)
        # 1D KDE.
        xNC, fNC = St.Gaussian1DKDE(np.log10(np.ma.compressed(CDC.data)), Nbnds)
        xpC, fpC = St.Gaussian1DKDE(np.log10(np.ma.compressed(pC.data)), pbnds)
        xSC, fSC = St.Gaussian1DKDE(np.log10(np.ma.compressed(SC.data)), Sbnds)
        # PCA.
        NpCeval, NpCevec = St.PCA(np.log10(np.ma.compressed(CDC.data)), np.log10(np.ma.compressed(pC.data)))
        SpCeval, SpCevec = St.PCA(np.log10(np.ma.compressed(SC.data)),  np.log10(np.ma.compressed(pC.data)))
        NSCeval, NSCevec = St.PCA(np.log10(np.ma.compressed(CDC.data)), np.log10(np.ma.compressed(SC.data)))
        # 2D KDE.
        xxNpC, yyNpC, fNpC = St.Gaussian2DKDE(np.log10(np.ma.compressed(CDC.data)), np.log10(np.ma.compressed(pC.data)), Nbnds, pbnds)
        xxSpC, yySpC, fSpC = St.Gaussian2DKDE(np.log10(np.ma.compressed(SC.data)),  np.log10(np.ma.compressed(pC.data)), Sbnds, pbnds)
        xxNSC, yyNSC, fNSC = St.Gaussian2DKDE(np.log10(np.ma.compressed(CDC.data)), np.log10(np.ma.compressed(SC.data)), Nbnds, Sbnds)
        # Compute Correlation Coefficients.
        NpCpr, NpCsr = St.CorrCoeffs(np.log10(np.ma.compressed(CDC.data)), np.log10(np.ma.compressed(pC.data)), False)
        SpCpr, SpCsr = St.CorrCoeffs(np.log10(np.ma.compressed(SC.data)),  np.log10(np.ma.compressed(pC.data)), False)
        NSCpr, NSCsr = St.CorrCoeffs(np.log10(np.ma.compressed(CDC.data)), np.log10(np.ma.compressed(SC.data)), False)
        # Compute means. 
        meanCDC = np.average(np.log10(np.ma.compressed(CDC.data)))
        meanpC  = np.average(np.log10(np.ma.compressed( pC.data)))
        meanSC  = np.average(np.log10(np.ma.compressed( SC.data)))  
        # Compute geometric medians. 
        medCDC = np.median(np.log10(np.ma.compressed(CDC.data)))
        medpC  = np.median(np.log10(np.ma.compressed( pC.data)))
        medSC  = np.median(np.log10(np.ma.compressed( SC.data)))
        # Compute geometric std deviations. 
        stdCDC = np.std(np.log10(np.ma.compressed(CDC.data)))
        stdpC  = np.std(np.log10(np.ma.compressed( pC.data)))
        stdSC  = np.std(np.log10(np.ma.compressed( SC.data)))
        # Compute geometric skews. 
        skewCDC = stats.skew(np.log10(np.ma.compressed(CDC.data)))
        skewpC  = stats.skew(np.log10(np.ma.compressed( pC.data)))
        skewSC  = stats.skew(np.log10(np.ma.compressed( SC.data)))
        # Compute geometric kurtoses. 
        kurtCDC = stats.kurtosis(np.log10(np.ma.compressed(CDC.data)), fisher=True)
        kurtpC  = stats.kurtosis(np.log10(np.ma.compressed( pC.data)), fisher=True)
        kurtSC  = stats.kurtosis(np.log10(np.ma.compressed( SC.data)), fisher=True)
        # Write Stats. 
        fileNC.write( 'Homogeneous, {: f}, {: f}, {: f}, {: f}, {: f} \n'.format(meanCDC, medCDC, stdCDC, skewCDC, kurtCDC))
        filepC.write( 'Homogeneous, {: f}, {: f}, {: f}, {: f}, {: f} \n'.format(meanpC,  medpC,  stdpC,  skewpC,  kurtpC))
        fileSC.write( 'Homogeneous, {: f}, {: f}, {: f}, {: f}, {: f} \n'.format(meanSC,  medSC,  stdSC,  skewSC,  kurtSC))
        fileNpC.write('Homogeneous, {: f}, {: f}, {: f}, {: f}, {: f}, {: f}, {: f}, {: f} \n'.format(NpCeval[1], NpCevec[1,0], NpCevec[1,1], NpCeval[0], NpCevec[0,0], NpCevec[0,1], NpCpr, NpCsr))
        fileNSC.write('Homogeneous, {: f}, {: f}, {: f}, {: f}, {: f}, {: f}, {: f}, {: f} \n'.format(NSCeval[1], NSCevec[1,0], NSCevec[1,1], NSCeval[0], NSCevec[0,0], NSCevec[0,1], NSCpr, NSCsr))
        fileSpC.write('Homogeneous, {: f}, {: f}, {: f}, {: f}, {: f}, {: f}, {: f}, {: f} \n'.format(SpCeval[1], SpCevec[1,0], SpCevec[1,1], SpCeval[0], SpCevec[0,0], SpCevec[0,1], SpCpr, SpCsr))  
        # Save 1D KDE.
        St.Save1DKDE(xNC, fNC, 'N_XY_C_ind'+str(i)+'_cut'+str(j))
        St.Save1DKDE(xpC, fpC, 'p_XY_C_ind'+str(i)+'_cut'+str(j))
        St.Save1DKDE(xSC, fSC, 'S_XY_C_ind'+str(i)+'_cut'+str(j))
        # Save 2D KDE. 
        St.Save2DKDE(xxNpC, yyNpC, fNpC, 'Np_XY_C_ind'+str(i)+'_cut'+str(j), 'Gaussian')
        St.Save2DKDE(xxNSC, yyNSC, fNSC, 'NS_XY_C_ind'+str(i)+'_cut'+str(j), 'Gaussian')
        St.Save2DKDE(xxSpC, yySpC, fSpC, 'Sp_XY_C_ind'+str(i)+'_cut'+str(j), 'Gaussian')         
        # plotting. 
        if j == 0:
            axp[2].semilogy(xpC, fpC, '-', linewidth=1.25, color=colors[3-i], zorder=2)
            axS[2].semilogy(xSC, fSC, '-', linewidth=1.25, color=colors[3-i], zorder=2)
        else:
            axp[2].semilogy(xpC, fpC, '--', linewidth=1.25, color=colors[3-i], zorder=2)
            axS[2].semilogy(xSC, fSC, '--', linewidth=1.25, color=colors[3-i], zorder=2)

        axNp[j,i+1].contour(xxNpC, yyNpC, fNpC, 10, colors=colors[2],zorder=1)
        axNp[j,i+1].quiver(meanCDC, meanpC, np.sqrt(NpCeval[0])*abs(NpCevec[0,0]),  np.sqrt(NpCeval[0])*abs(NpCevec[1,0]), scale=1.25, zorder=2)
        axNp[j,i+1].quiver(meanCDC, meanpC, np.sqrt(NpCeval[1])*abs(NpCevec[0,1]), -np.sqrt(NpCeval[1])*abs(NpCevec[1,1]), scale=1.25, zorder=2)
        axNp[j,i+1].plot(meanCDC, meanpC, 'k.', markersize=5)

        axSp[j,i+1].contour(xxSpC, yySpC, fSpC, 10, colors=colors[2],zorder=1)
        axSp[j,i+1].quiver(meanSC, meanpC, np.sqrt(SpCeval[0])*abs(SpCevec[0,0]),  np.sqrt(SpCeval[0])*abs(SpCevec[1,0]), scale=1.25, zorder=2)
        axSp[j,i+1].quiver(meanSC, meanpC, np.sqrt(SpCeval[1])*abs(SpCevec[0,1]), -np.sqrt(SpCeval[1])*abs(SpCevec[1,1]), scale=1.25, zorder=2)
        axSp[j,i+1].plot(meanSC, meanpC, 'k.', markersize=5)

        axNS[j,i+1].contour(xxNSC, yyNSC, fNSC, 10, colors=colors[2],zorder=1)
        axNS[j,i+1].quiver(meanCDC, meanSC, np.sqrt(NSCeval[0])*NSCevec[0,0], np.sqrt(NSCeval[0])*NSCevec[1,0], scale=1.25, zorder=2)
        axNS[j,i+1].quiver(meanCDC, meanSC, np.sqrt(NSCeval[1])*NSCevec[0,1], np.sqrt(NSCeval[1])*NSCevec[1,1], scale=1.25, zorder=2)
        axNS[j,i+1].plot(meanCDC, meanSC, 'k.', markersize=5)

        del xNC, xpC, xSC, fNC, fpC, fSC
        del xxNpC, yyNpC, fNpC 
        del xxNSC, yyNSC, fNSC 
        del xxSpC, yySpC, fSpC
        del NpCpr, NpCsr, NSCpr, NSCsr, SpCpr, SpCsr
        del meanCDC, meanpC, meanSC
        del medCDC, medpC, medSC 
        del stdCDC, stdpC, stdSC 
        del skewCDC, skewpC, skewSC 
        del kurtCDC, kurtpC, kurtSC
        del NpCeval, NpCevec 
        del NSCeval, NSCevec 
        del SpCeval, SpCevec 
        print("Sim C Index "+str(i)+" Cutoff "+str(j)+" Complete.")

# Remove simdata to prevent memoryerror. 
del SC, pC, UC, QC, CDC 
del ObsC 
del OC

# Close files.
fileNC.close()
filepC.close() 
fileSC.close()
fileNpC.close()
fileNSC.close()
fileSpC.close()

# Open files to write stats to.
fileND  = open('N_XY_D_stats.txt', 'w')
filepD  = open('p_XY_D_stats.txt', 'w')
fileSD  = open('S_XY_D_stats.txt', 'w')
fileNpD = open('Np_XY_D_stats.txt', 'w')
fileNSD = open('NS_XY_D_stats.txt', 'w')
fileSpD = open('Sp_XY_D_stats.txt', 'w')

fileND.write( '# Model N_geomean N_geomed N_geostd N_geoskew N_geokurt \n')
filepD.write( '# Model p_geomean p_geomed p_geostd p_geoskew p_geokurt \n')
fileSD.write( '# Model S_geomean S_geomed S_geostd S_geoskew S_geokurt \n')
fileNpD.write('# Model Npevalmax Npevecmax1 Npevecmax2 Npevalmin Npevecmin1 Npevecmin2 NpPearson NpSpearman \n')
fileNSD.write('# Model NSevalmax NSevecmax1 NSevecmax2 NSevalmin NSevecmin1 NSevecmin2 NSPearson NSSpearman \n')
fileSpD.write('# Model Spevalmax Spevecmax1 Spevecmax2 Spevalmin Spevecmin1 Spevecmin2 SpPearson SpSpearman \n')

# Load Sim D data. 
OD = Observer(['M10B10R1_0051.vtk',N,boxlenD,'./'])
OD.ChangeMagneticHandle(['magnetic_field_x', 'magnetic_field_y', 'magnetic_field_z'])
OD.DataLoad()
ND = Nabla([None,N,boxlenD])
PltD = Plotter([N, boxlenD, 5, 100, './'])
print('Sim D Data Loaded.')

# next plot the unmodified data for sim D. 
pol_args = ['Constant', p0]
ObsD = OD.Polarimetry(exc_args, rot_args, pol_args)
CDD = ObsD[3]
CDD.data *= scaleD
QD = ObsD[1]
UD = ObsD[2]
pD = ObsD[6]
SD = ND.ComputeAngleGradient(QD, UD, QD.data.mask)
# 1D KDE.
xND, fND = St.Gaussian1DKDE(np.log10(np.ma.compressed(CDD.data)), Nbnds)
xpD, fpD = St.Gaussian1DKDE(np.log10(np.ma.compressed(pD.data)), pbnds)
xSD, fSD = St.Gaussian1DKDE(np.log10(np.ma.compressed(SD.data)), Sbnds)
# PCA.
NpDeval, NpDevec = St.PCA(np.log10(np.ma.compressed(CDD.data)), np.log10(np.ma.compressed(pD.data)))
SpDeval, SpDevec = St.PCA(np.log10(np.ma.compressed(SD.data)),  np.log10(np.ma.compressed(pD.data)))
NSDeval, NSDevec = St.PCA(np.log10(np.ma.compressed(CDD.data)), np.log10(np.ma.compressed(SD.data)))
# 2D KDE.
xxNpD, yyNpD, fNpD = St.Gaussian2DKDE(np.log10(np.ma.compressed(CDD.data)), np.log10(np.ma.compressed(pD.data)), Nbnds, pbnds)
xxSpD, yySpD, fSpD = St.Gaussian2DKDE(np.log10(np.ma.compressed(SD.data)),  np.log10(np.ma.compressed(pD.data)), Sbnds, pbnds)
xxNSD, yyNSD, fNSD = St.Gaussian2DKDE(np.log10(np.ma.compressed(CDD.data)), np.log10(np.ma.compressed(SD.data)), Nbnds, Sbnds)
# Compute Correlation Coefficients.
NpDpr, NpDsr = St.CorrCoeffs(np.log10(np.ma.compressed(CDD.data)), np.log10(np.ma.compressed(pD.data)), False)
SpDpr, SpDsr = St.CorrCoeffs(np.log10(np.ma.compressed(SD.data)),  np.log10(np.ma.compressed(pD.data)), False)
NSDpr, NSDsr = St.CorrCoeffs(np.log10(np.ma.compressed(CDD.data)), np.log10(np.ma.compressed(SD.data)), False)
# Compute means. 
meanCDD = np.average(np.log10(np.ma.compressed(CDD.data)))
meanpD  = np.average(np.log10(np.ma.compressed( pD.data)))
meanSD  = np.average(np.log10(np.ma.compressed( SD.data)))
# Compute geometric medians. 
medCDD = np.median(np.log10(np.ma.compressed(CDD.data)))
medpD  = np.median(np.log10(np.ma.compressed( pD.data)))
medSD  = np.median(np.log10(np.ma.compressed( SD.data)))
# Compute geometric std deviations. 
stdCDD = np.std(np.log10(np.ma.compressed(CDD.data)))
stdpD  = np.std(np.log10(np.ma.compressed( pD.data)))
stdSD  = np.std(np.log10(np.ma.compressed( SD.data)))
# Compute geometric skews. 
skewCDD = stats.skew(np.log10(np.ma.compressed(CDD.data)))
skewpD  = stats.skew(np.log10(np.ma.compressed( pD.data)))
skewSD  = stats.skew(np.log10(np.ma.compressed( SD.data)))
# Compute geometric kurtoses. 
kurtCDD = stats.kurtosis(np.log10(np.ma.compressed(CDD.data)), fisher=True)
kurtpD  = stats.kurtosis(np.log10(np.ma.compressed( pD.data)), fisher=True)
kurtSD  = stats.kurtosis(np.log10(np.ma.compressed( SD.data)), fisher=True)
# Write Stats. 
fileND.write( 'Homogeneous, {: f}, {: f}, {: f}, {: f}, {: f} \n'.format(meanCDD, medCDD, stdCDD, skewCDD, kurtCDD))
filepD.write( 'Homogeneous, {: f}, {: f}, {: f}, {: f}, {: f} \n'.format(meanpD,  medpD,  stdpD,  skewpD,  kurtpD))
fileSD.write( 'Homogeneous, {: f}, {: f}, {: f}, {: f}, {: f} \n'.format(meanSD,  medSD,  stdSD,  skewSD,  kurtSD))
fileNpD.write('Homogeneous, {: f}, {: f}, {: f}, {: f}, {: f}, {: f}, {: f}, {: f} \n'.format(NpDeval[1], NpDevec[1,0], NpDevec[1,1], NpDeval[0], NpDevec[0,0], NpDevec[0,1], NpDpr, NpDsr))
fileNSD.write('Homogeneous, {: f}, {: f}, {: f}, {: f}, {: f}, {: f}, {: f}, {: f} \n'.format(NSDeval[1], NSDevec[1,0], NSDevec[1,1], NSDeval[0], NSDevec[0,0], NSDevec[0,1], NSDpr, NSDsr))
fileSpD.write('Homogeneous, {: f}, {: f}, {: f}, {: f}, {: f}, {: f}, {: f}, {: f} \n'.format(SpDeval[1], SpDevec[1,0], SpDevec[1,1], SpDeval[0], SpDevec[0,0], SpDevec[0,1], SpDpr, SpDsr))     
# Save 1D KDE.
St.Save1DKDE(xND, fND, 'N_XY_D_homo')
St.Save1DKDE(xpD, fpD, 'p_XY_D_homo')
St.Save1DKDE(xSD, fSD, 'S_XY_D_homo')
# Save 2D KDE. 
St.Save2DKDE(xxNpD, yyNpD, fNpD, 'Np_XY_D_homo', 'Gaussian')
St.Save2DKDE(xxNSD, yyNSD, fNSD, 'NS_XY_D_homo', 'Gaussian')
St.Save2DKDE(xxSpD, yySpD, fSpD, 'Sp_XY_D_homo', 'Gaussian')     
# plotting. 
axp[3].semilogy(xpD, fpD, '-', linewidth=1.25, color=colors[4], zorder=2)
axS[3].semilogy(xSD, fSD, '-', linewidth=1.25, color=colors[4], zorder=2)

axNp[0,0].contour(xxNpD, yyNpD, fNpD, 10, colors=colors[4],zorder=1)
axNp[0,0].quiver(meanCDD, meanpD, np.sqrt(NpDeval[0])*NpDevec[0,0], np.sqrt(NpDeval[0])*NpDevec[1,0], scale=1.25, zorder=2)
axNp[0,0].quiver(meanCDD, meanpD, np.sqrt(NpDeval[1])*NpDevec[0,1], np.sqrt(NpDeval[1])*NpDevec[1,1], scale=1.25, zorder=2)
axNp[0,0].plot(meanCDD, meanpD, 'k.', markersize=5)

axSp[0,0].contour(xxSpD, yySpD, fSpD, 10, colors=colors[4],zorder=1)
axSp[0,0].quiver(meanSD, meanpD, np.sqrt(SpDeval[0])*abs(SpDevec[0,0]), np.sqrt(SpDeval[0])*abs(SpDevec[1,0]), scale=1.25, zorder=2)
axSp[0,0].quiver(meanSD, meanpD, np.sqrt(SpDeval[1])*abs(SpDevec[0,1]), -np.sqrt(SpDeval[1])*abs(SpDevec[1,1]), scale=1.25, zorder=2)
axSp[0,0].plot(meanSD, meanpD, 'k.', markersize=5)

axNS[0,0].contour(xxNSD, yyNSD, fNSD, 10, colors=colors[4],zorder=1)
axNS[0,0].quiver(meanCDD, meanSD, np.sqrt(NSDeval[0])*NSDevec[0,0], np.sqrt(NSDeval[0])*NSDevec[1,0], scale=1.25, zorder=2)
axNS[0,0].quiver(meanCDD, meanSD, np.sqrt(NSDeval[1])*NSDevec[0,1], np.sqrt(NSDeval[1])*NSDevec[1,1], scale=1.25, zorder=2)
axNS[0,0].plot(meanCDD, meanSD, 'k.', markersize=5)

del xND, xpD, xSD, fND, fpD, fSD
del xxNpD, yyNpD, fNpD 
del xxNSD, yyNSD, fNSD 
del xxSpD, yySpD, fSpD
del NpDpr, NpDsr, NSDpr, NSDsr, SpDpr, SpDsr
del meanCDD, meanpD, meanSD
del medCDD, medpD, medSD 
del stdCDD, stdpD, stdSD 
del skewCDD, skewpD, skewSD 
del kurtCDD, kurtpD, kurtSD
del NpDeval, NpDevec 
del NSDeval, NSDevec 
del SpDeval, SpDevec
print("Sim D Index Constant p0 Complete.")

# Finally loop for D. 
for i in range(len(indices)):
    for j in range(len(cutoffs)):
        pol_args = ['Power-Law', p0, cutoffs[j]/(scaleD**2), indices[i]]
        ObsD = OD.Polarimetry(exc_args, rot_args, pol_args)
        CDD = ObsD[3]
        CDD.data *= scaleD 
        QD  = ObsD[1]
        UD  = ObsD[2]
        pD  = ObsD[6]
        SD  = ND.ComputeAngleGradient(QD, UD, QD.data.mask)
        # 1D KDE.
        xND, fND = St.Gaussian1DKDE(np.log10(np.ma.compressed(CDD.data)), Nbnds)
        xpD, fpD = St.Gaussian1DKDE(np.log10(np.ma.compressed(pD.data)), pbnds)
        xSD, fSD = St.Gaussian1DKDE(np.log10(np.ma.compressed(SD.data)), Sbnds)
        # PCA.
        NpDeval, NpDevec = St.PCA(np.log10(np.ma.compressed(CDD.data)), np.log10(np.ma.compressed(pD.data)))
        SpDeval, SpDevec = St.PCA(np.log10(np.ma.compressed(SD.data)),  np.log10(np.ma.compressed(pD.data)))
        NSDeval, NSDevec = St.PCA(np.log10(np.ma.compressed(CDD.data)), np.log10(np.ma.compressed(SD.data)))
        # KDE.
        xxNpD, yyNpD, fNpD = St.Gaussian2DKDE(np.log10(np.ma.compressed(CDD.data)), np.log10(np.ma.compressed(pD.data)), Nbnds, pbnds)
        xxSpD, yySpD, fSpD = St.Gaussian2DKDE(np.log10(np.ma.compressed(SD.data)),  np.log10(np.ma.compressed(pD.data)), Sbnds, pbnds)
        xxNSD, yyNSD, fNSD = St.Gaussian2DKDE(np.log10(np.ma.compressed(CDD.data)), np.log10(np.ma.compressed(SD.data)), Nbnds, Sbnds)
        # Compute Correlation Coefficients.
        NpDpr, NpDsr = St.CorrCoeffs(np.log10(np.ma.compressed(CDD.data)), np.log10(np.ma.compressed(pD.data)), False)
        SpDpr, SpDsr = St.CorrCoeffs(np.log10(np.ma.compressed(SD.data)),  np.log10(np.ma.compressed(pD.data)), False)
        NSDpr, NSDsr = St.CorrCoeffs(np.log10(np.ma.compressed(CDD.data)), np.log10(np.ma.compressed(SD.data)), False)
        # Compute means. 
        meanCDD = np.average(np.log10(np.ma.compressed(CDD.data)))
        meanpD  = np.average(np.log10(np.ma.compressed( pD.data)))
        meanSD  = np.average(np.log10(np.ma.compressed( SD.data))) 
        # Compute geometric medians. 
        medCDD = np.median(np.log10(np.ma.compressed(CDD.data)))
        medpD  = np.median(np.log10(np.ma.compressed( pD.data)))
        medSD  = np.median(np.log10(np.ma.compressed( SD.data)))
        # Compute geometric std deviations. 
        stdCDD = np.std(np.log10(np.ma.compressed(CDD.data)))
        stdpD  = np.std(np.log10(np.ma.compressed( pD.data)))
        stdSD  = np.std(np.log10(np.ma.compressed( SD.data)))
        # Compute geometric skews. 
        skewCDD = stats.skew(np.log10(np.ma.compressed(CDD.data)))
        skewpD  = stats.skew(np.log10(np.ma.compressed( pD.data)))
        skewSD  = stats.skew(np.log10(np.ma.compressed( SD.data)))
        # Compute geometric kurtoses. 
        kurtCDD = stats.kurtosis(np.log10(np.ma.compressed(CDD.data)), fisher=True)
        kurtpD  = stats.kurtosis(np.log10(np.ma.compressed( pD.data)), fisher=True)
        kurtSD  = stats.kurtosis(np.log10(np.ma.compressed( SD.data)), fisher=True)
        # Write Stats. 
        fileND.write( 'Homogeneous, {: f}, {: f}, {: f}, {: f}, {: f} \n'.format(meanCDD, medCDD, stdCDD, skewCDD, kurtCDD))
        filepD.write( 'Homogeneous, {: f}, {: f}, {: f}, {: f}, {: f} \n'.format(meanpD,  medpD,  stdpD,  skewpD,  kurtpD))
        fileSD.write( 'Homogeneous, {: f}, {: f}, {: f}, {: f}, {: f} \n'.format(meanSD,  medSD,  stdSD,  skewSD,  kurtSD))
        fileNpD.write('Homogeneous, {: f}, {: f}, {: f}, {: f}, {: f}, {: f}, {: f}, {: f} \n'.format(NpDeval[1], NpDevec[1,0], NpDevec[1,1], NpDeval[0], NpDevec[0,0], NpDevec[0,1], NpDpr, NpDsr))
        fileNSD.write('Homogeneous, {: f}, {: f}, {: f}, {: f}, {: f}, {: f}, {: f}, {: f} \n'.format(NSDeval[1], NSDevec[1,0], NSDevec[1,1], NSDeval[0], NSDevec[0,0], NSDevec[0,1], NSDpr, NSDsr))
        fileSpD.write('Homogeneous, {: f}, {: f}, {: f}, {: f}, {: f}, {: f}, {: f}, {: f} \n'.format(SpDeval[1], SpDevec[1,0], SpDevec[1,1], SpDeval[0], SpDevec[0,0], SpDevec[0,1], SpDpr, SpDsr))     
        # Save 1D KDE.
        St.Save1DKDE(xND, fND, 'N_XY_D_ind'+str(i)+'_cut'+str(j))
        St.Save1DKDE(xpD, fpD, 'p_XY_D_ind'+str(i)+'_cut'+str(j))
        St.Save1DKDE(xSD, fSD, 'S_XY_D_ind'+str(i)+'_cut'+str(j))
        # Save 2D KDE. 
        St.Save2DKDE(xxNpD, yyNpD, fNpD, 'Np_XY_D_ind'+str(i)+'_cut'+str(j), 'Gaussian')
        St.Save2DKDE(xxNSD, yyNSD, fNSD, 'NS_XY_D_ind'+str(i)+'_cut'+str(j), 'Gaussian')
        St.Save2DKDE(xxSpD, yySpD, fSpD, 'Sp_XY_D_ind'+str(i)+'_cut'+str(j), 'Gaussian')    
        # plotting. 
        if j == 0:
            axp[3].semilogy(xpD, fpD, '-', linewidth=1.25, color=colors[3-i], zorder=2)
            axS[3].semilogy(xSD, fSD, '-', linewidth=1.25, color=colors[3-i], zorder=2)
        else:
            axp[3].semilogy(xpD, fpD, '--', linewidth=1.25, color=colors[3-i], zorder=2)
            axS[3].semilogy(xSD, fSD, '--', linewidth=1.25, color=colors[3-i], zorder=2)

        axNp[j,i+1].contour(xxNpD, yyNpD, fNpD, 10, colors=colors[4],zorder=1)
        axNp[j,i+1].quiver(meanCDD, meanpD, np.sqrt(NpDeval[0])*abs(NpDevec[0,0]),  np.sqrt(NpDeval[0])*abs(NpDevec[1,0]), scale=1.25, zorder=2)
        axNp[j,i+1].quiver(meanCDD, meanpD, np.sqrt(NpDeval[1])*abs(NpDevec[0,1]), -np.sqrt(NpDeval[1])*abs(NpDevec[1,1]), scale=1.25, zorder=2)
        axNp[j,i+1].plot(meanCDD, meanpD, 'k.', markersize=5)

        axSp[j,i+1].contour(xxSpD, yySpD, fSpD, 10, colors=colors[4],zorder=1)
        axSp[j,i+1].quiver(meanSD, meanpD, np.sqrt(SpDeval[0])*abs(SpDevec[0,0]),  np.sqrt(SpDeval[0])*abs(SpDevec[1,0]), scale=1.25, zorder=2)
        axSp[j,i+1].quiver(meanSD, meanpD, np.sqrt(SpDeval[1])*abs(SpDevec[0,1]), -np.sqrt(SpDeval[1])*abs(SpDevec[1,1]), scale=1.25, zorder=2)
        axSp[j,i+1].plot(meanSD, meanpD, 'k.', markersize=5)

        axNS[j,i+1].contour(xxNSD, yyNSD, fNSD, 10, colors=colors[4],zorder=1)
        axNS[j,i+1].quiver(meanCDD, meanSD, np.sqrt(NSDeval[0])*NSDevec[0,0], np.sqrt(NSDeval[0])*NSDevec[1,0], scale=1.25, zorder=2)
        axNS[j,i+1].quiver(meanCDD, meanSD, np.sqrt(NSDeval[1])*NSDevec[0,1], np.sqrt(NSDeval[1])*NSDevec[1,1], scale=1.25, zorder=2)
        axNS[j,i+1].plot(meanCDD, meanSD, 'k.', markersize=5)

        del xND, xpD, xSD, fND, fpD, fSD
        del xxNpD, yyNpD, fNpD 
        del xxNSD, yyNSD, fNSD 
        del xxSpD, yySpD, fSpD
        del NpDpr, NpDsr, NSDpr, NSDsr, SpDpr, SpDsr
        del meanCDD, meanpD, meanSD
        del medCDD, medpD, medSD 
        del stdCDD, stdpD, stdSD 
        del skewCDD, skewpD, skewSD 
        del kurtCDD, kurtpD, kurtSD
        del NpDeval, NpDevec 
        del NSDeval, NSDevec 
        del SpDeval, SpDevec 
        print("Sim D Index "+str(i)+" Cutoff "+str(j)+" Complete.")

# Close files.
fileND.close()
filepD.close() 
fileSD.close()
fileNpD.close()
fileNSD.close()
fileSpD.close()

# Remove simdata to prevent memoryerror. 
del SD, pD, UD, QD, CDD 
del ObsD 
del OD

# Figure settings. 
axp[3].set_xlim(pbnds)
axp[3].set_ylim([1E-3,1E1])
axp[3].yaxis.set_ticks([1E-2,1E0])
axp[3].xaxis.set_ticks([-3.0, -2.0, -1.0])
axp[0].set_title('Z-LOS')
axp[0].plot([-1.0,-2.0], [-1.0,-2.0],color=colors[4],label=r'Homogeneous')
axp[0].plot([-1.0,-2.0], [-1.0,-2.0],color=colors[3],label=r'$\gamma = -\frac{1}{3}$')
axp[0].plot([-1.0,-2.0], [-1.0,-2.0],color=colors[2],label=r'$\gamma = -\frac{2}{3}$')
axp[0].plot([-1.0,-2.0], [-1.0,-2.0],color=colors[1],label=r'$\gamma = -1$')
axp[0].plot([-1.0,-2.0], [-1.0,-2.0],color=colors[0],label=r'BLASTPol')
axp[0].text(-3.4, 1.65E-3, r'Model A',fontsize='large').set_bbox(dict(facecolor='white',alpha=0.5))
axp[1].text(-3.4, 1.85E-3, r'Model B',fontsize='large').set_bbox(dict(facecolor='white',alpha=0.5))
axp[2].text(-3.4, 1.85E-3, r'Model C',fontsize='large').set_bbox(dict(facecolor='white',alpha=0.5))
axp[3].text(-3.4, 1.75E-3, r'Model D',fontsize='large').set_bbox(dict(facecolor='white',alpha=0.5))
axp[0].legend(loc='upper left',fontsize='small')
axp[0].grid()
axp[1].grid()
axp[2].grid()
axp[3].grid()
axp[3].set_xlabel(r'Polarization Fraction',fontsize='large')
figp.subplots_adjust(wspace=0,hspace=0)

axS[3].set_xlim(Sbnds)
axS[3].set_ylim([1E-3,1E1])
axS[3].yaxis.set_ticks([1E-2,1E0])
axS[3].xaxis.set_ticks([-1.0 ,0.0, 1.0])
axS[0].set_title('Z-LOS')
axS[0].yaxis.tick_right()
axS[0].yaxis.set_label_position('right')
axS[0].plot([-1.0,-2.0], [-1.0,-2.0],color=colors[4],label=r'Homogeneous')
axS[0].plot([-1.0,-2.0], [-1.0,-2.0],color=colors[3],label=r'$\gamma = -\frac{1}{3}$')
axS[0].plot([-1.0,-2.0], [-1.0,-2.0],color=colors[2],label=r'$\gamma = -\frac{2}{3}$')
axS[0].plot([-1.0,-2.0], [-1.0,-2.0],color=colors[1],label=r'$\gamma = -1$')
axS[0].plot([-1.0,-2.0], [-1.0,-2.0],color=colors[0],label=r'BLASTPol')
axS[0].text(-3.4, 1.65E-3, r'Model A',fontsize='large').set_bbox(dict(facecolor='white',alpha=0.5))
axS[1].text(-3.4, 1.85E-3, r'Model B',fontsize='large').set_bbox(dict(facecolor='white',alpha=0.5))
axS[2].text(-3.4, 1.85E-3, r'Model C',fontsize='large').set_bbox(dict(facecolor='white',alpha=0.5))
axS[3].text(-3.4, 1.75E-3, r'Model D',fontsize='large').set_bbox(dict(facecolor='white',alpha=0.5))
axS[0].legend(loc='upper left',fontsize='small')
axS[0].grid()
axS[1].grid()
axS[2].grid()
axS[3].grid()
axS[3].set_xlabel(r'Dispersion in Polarization Angles ($^{\circ}$)',fontsize='large')
figS.subplots_adjust(wspace=0,hspace=0)

axNp[0,0].plot([1.0,2.0],[1.0,2.0],color=colors[3],label='Model A')
axNp[0,0].plot([1.0,2.0],[1.0,2.0],color=colors[1],label='Model B')
axNp[0,0].plot([1.0,2.0],[1.0,2.0],color=colors[2],label='Model C')
axNp[0,0].plot([1.0,2.0],[1.0,2.0],color=colors[4],label='Model D')
axNp[0,0].plot([1.0,2.0],[1.0,2.0],color=colors[0],label='BLASTPol')
axNp[0,0].legend(loc='lower right',fontsize='x-small')
axNp[0,0].text(20.15,-3.325, 'Homogeneous', fontsize='small').set_bbox(dict(facecolor='white',alpha=0.5))
axNp[1,0].text(20.15,-3.325, 'BLASTPol', fontsize='small').set_bbox(dict(facecolor='white',alpha=0.5))
axNp[0,1].text(20.15,-3.325, r'$\beta = -\frac{1}{3}$, $n_{crit}=10^3$ cm$^{-3}$', fontsize='x-small').set_bbox(dict(facecolor='white',alpha=0.5))
axNp[0,2].text(20.15,-3.325, r'$\beta = -\frac{2}{3}$, $n_{crit}=10^3$ cm$^{-3}$', fontsize='x-small').set_bbox(dict(facecolor='white',alpha=0.5))
axNp[0,3].text(20.15,-3.325, r'$\beta = -1$, $n_{crit}=10^3$ cm$^{-3}$',           fontsize='x-small').set_bbox(dict(facecolor='white',alpha=0.5))
axNp[1,1].text(20.15,-3.325, r'$\beta = -\frac{1}{3}$, $n_{crit}=10^2$ cm$^{-3}$', fontsize='x-small').set_bbox(dict(facecolor='white',alpha=0.5))
axNp[1,2].text(20.15,-3.325, r'$\beta = -\frac{2}{3}$, $n_{crit}=10^2$ cm$^{-3}$', fontsize='x-small').set_bbox(dict(facecolor='white',alpha=0.5))
axNp[1,3].text(20.15,-3.325, r'$\beta = -1$, $n_{crit}=10^2$ cm$^{-3}$',            fontsize='x-small').set_bbox(dict(facecolor='white',alpha=0.5))

axNS[0,0].plot([1.0,2.0],[-3.0,-2.0],color=colors[3],label='Model A')
axNS[0,0].plot([1.0,2.0],[-3.0,-2.0],color=colors[1],label='Model B')
axNS[0,0].plot([1.0,2.0],[-3.0,-2.0],color=colors[2],label='Model C')
axNS[0,0].plot([1.0,2.0],[-3.0,-2.0],color=colors[4],label='Model D')
axNS[0,0].plot([1.0,2.0],[-3.0,-2.0],color=colors[0],label='BLASTPol')
axNS[0,0].legend(loc='lower right',fontsize='x-small')
axNS[0,0].text(20.15,-1.35, 'Homogeneous', fontsize='small').set_bbox(dict(facecolor='white',alpha=0.5))
axNS[1,0].text(20.15,-1.35, 'BLASTPol', fontsize='small').set_bbox(dict(facecolor='white',alpha=0.5))
axNS[0,1].text(20.15,-1.35, r'$\beta = -\frac{1}{3}$, $n_{crit}=10^3$ cm$^{-3}$', fontsize='x-small').set_bbox(dict(facecolor='white',alpha=0.5))
axNS[0,2].text(20.15,-1.35, r'$\beta = -\frac{2}{3}$, $n_{crit}=10^3$ cm$^{-3}$', fontsize='x-small').set_bbox(dict(facecolor='white',alpha=0.5))
axNS[0,3].text(20.15,-1.35, r'$\beta = -1$, $n_{crit}=10^3$ cm$^{-3}$', fontsize='x-small').set_bbox(dict(facecolor='white',alpha=0.5))
axNS[1,1].text(20.15,-1.35, r'$\beta = -\frac{1}{3}$, $n_{crit}=10^2$ cm$^{-3}$', fontsize='x-small').set_bbox(dict(facecolor='white',alpha=0.5))
axNS[1,2].text(20.15,-1.35, r'$\beta = -\frac{2}{3}$, $n_{crit}=10^2$ cm$^{-3}$', fontsize='x-small').set_bbox(dict(facecolor='white',alpha=0.5))
axNS[1,3].text(20.15,-1.35, r'$\beta = -1$, $n_{crit}=10^2$ cm$^{-3}$', fontsize='x-small').set_bbox(dict(facecolor='white',alpha=0.5))

axSp[0,0].plot([10.0,20.0],[-30.0,-20.0],color=colors[3],label='Model A')
axSp[0,0].plot([10.0,20.0],[-30.0,-20.0],color=colors[1],label='Model B')
axSp[0,0].plot([10.0,20.0],[-30.0,-20.0],color=colors[2],label='Model C')
axSp[0,0].plot([10.0,20.0],[-30.0,-20.0],color=colors[4],label='Model D')
axSp[0,0].plot([10.0,20.0],[-30.0,-20.0],color=colors[0],label='BLASTPol')
axSp[0,0].legend(loc='lower right',fontsize='x-small')
axSp[0,0].text(-1.35,-3.325, 'Homogeneous', fontsize='small').set_bbox(dict(facecolor='white',alpha=0.5))
axSp[1,0].text(-1.35,-3.325, 'BLASTPol', fontsize='small').set_bbox(dict(facecolor='white',alpha=0.5))
axSp[0,1].text(-1.35,-3.325, r'$\beta = -\frac{1}{3}$, $n_{crit}=10^3$ cm$^{-3}$', fontsize='x-small').set_bbox(dict(facecolor='white',alpha=0.5))
axSp[0,2].text(-1.35,-3.325, r'$\beta = -\frac{2}{3}$, $n_{crit}=10^3$ cm$^{-3}$', fontsize='x-small').set_bbox(dict(facecolor='white',alpha=0.5))
axSp[0,3].text(-1.35,-3.325, r'$\beta = -1$, $n_{crit}=10^3$ cm$^{-3}$', fontsize='x-small').set_bbox(dict(facecolor='white',alpha=0.5))
axSp[1,1].text(-1.35,-3.325, r'$\beta = -\frac{1}{3}$, $n_{crit}=10^2$ cm$^{-3}$', fontsize='x-small').set_bbox(dict(facecolor='white',alpha=0.5))
axSp[1,2].text(-1.35,-3.325, r'$\beta = -\frac{2}{3}$, $n_{crit}=10^2$ cm$^{-3}$', fontsize='x-small').set_bbox(dict(facecolor='white',alpha=0.5))
axSp[1,3].text(-1.35,-3.325, r'$\beta = -1$, $n_{crit}=10^2$ cm$^{-3}$', fontsize='x-small').set_bbox(dict(facecolor='white',alpha=0.5))

axNp[0,0].yaxis.set_ticks([-3.0, -2.0, -1.0])
axNp[1,0].xaxis.set_ticks([21, 22, 23])
axNp[1,1].xaxis.set_ticks([21, 22, 23])
axNp[1,2].xaxis.set_ticks([21, 22, 23])
axNp[0,0].set_xlim(Nbnds)
axNp[0,0].set_ylim(pbnds)
axNp[0,0].grid()
axNp[0,1].grid()
axNp[0,2].grid()
axNp[0,3].grid()
axNp[1,0].grid()
axNp[1,1].grid()
axNp[1,2].grid()
axNp[1,3].grid()
axNp[1,0].set_xlabel(r'Column Density (cm$^{-3}$)',fontsize='x-large')
axNp[1,0].xaxis.set_label_coords(2.0,-0.125)
axNp[0,0].set_ylabel(r'Polarization Fraction',fontsize='x-large')
axNp[0,0].yaxis.set_label_coords(-0.125, 0.0)
figNp.subplots_adjust(wspace=0, hspace=0)

axSp[0,0].yaxis.set_ticks([-3.0, -2.0, -1.0])
axSp[1,0].xaxis.set_ticks([-1.0, 0.0, 1.0])
axSp[1,1].xaxis.set_ticks([-1.0, 0.0, 1.0])
axSp[1,2].xaxis.set_ticks([-1.0, 0.0, 1.0])
axSp[0,0].set_xlim(Sbnds)
axSp[0,0].set_ylim(pbnds)
axSp[0,0].grid()
axSp[0,1].grid()
axSp[0,2].grid()
axSp[0,3].grid()
axSp[1,0].grid()
axSp[1,1].grid()
axSp[1,2].grid()
axSp[1,3].grid()
axSp[1,0].set_xlabel(r'Dispersion in Polarization Angles ($^{\circ}$)',fontsize='x-large')
axSp[1,0].xaxis.set_label_coords(2.0,-0.125)
axSp[0,0].set_ylabel(r'Polarization Fraction',fontsize='x-large')
axSp[0,0].yaxis.set_label_coords(-0.125, 0.0)
figSp.subplots_adjust(wspace=0, hspace=0)

axNS[0,0].yaxis.set_ticks([-1.0, 0.0, 1.0])
axNS[1,0].xaxis.set_ticks([21, 22, 23])
axNS[1,1].xaxis.set_ticks([21, 22, 23])
axNS[1,2].xaxis.set_ticks([21, 22, 23])
axNS[0,0].set_xlim(Nbnds)
axNS[0,0].set_ylim(Sbnds)
axNS[0,0].grid()
axNS[0,1].grid()
axNS[0,2].grid()
axNS[0,3].grid()
axNS[1,0].grid()
axNS[1,1].grid()
axNS[1,2].grid()
axNS[1,3].grid()
axNS[1,0].set_xlabel(r'Column Density (cm$^{-3}$)',fontsize='x-large')
axNS[1,0].xaxis.set_label_coords(2.0,-0.125)
axNS[0,0].set_ylabel(r'Dispersion in Polarization Angles ($^{\circ}$)',fontsize='x-large')
axNS[0,0].yaxis.set_label_coords(-0.125, 0.0)
figNS.subplots_adjust(wspace=0, hspace=0)

plt.tight_layout()

figp.savefig('pPLXY.png')
figS.savefig('SPLXY.png')
figNp.savefig('NpPLXY.png')
figSp.savefig('SpPLXY.png')
figNS.savefig('NSPLXY.png')
