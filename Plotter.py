#************************VMAPP Class File: Plotter*****************************#
#
# Author: Patrick King, Date: 10/11/16
#
# This class file defines the Plotter object, which contains several different
# plotting routines formatted for publication quality results from the VMAPP
# pipeline. Currently, there are two subroutines defined: Imager and PhaseHist.
# Documentation for each can be found below.
#
#******************************************************************************#

# Packages: Dependencies include os, Numpy, Matplotlib, Scipy, Math, VMAPP
import os
import numpy                 as     np
import matplotlib.pyplot     as     plt
from   matplotlib.colors     import LogNorm
from   matplotlib.colors     import Normalize
from   matplotlib.colors     import SymLogNorm
from   matplotlib            import ticker
from   matplotlib            import cm
import scipy.stats           as     stats
from   math                  import *
from   Observer              import Observable
from   Stats                 import *

class Plotter(object):
    # Constructor. Initializes several common variables, and sets global plot
    # characteristics.
    def __init__(self, args):
        self.N      = args[0]
        self.boxlen = args[1]
        self.ntklbl = args[2]
        self.dpi    = args[3]
        self.path   = args[4]
        self.rot    = args[5]
        self.fitres = args[6]
        plt.rcParams['font.family'] = 'serif'
        self.ticks  = np.linspace(0,self.N,self.ntklbl,dtype=int)
        self.labels = np.linspace(-self.boxlen/2.0,self.boxlen/2.0,self.ntklbl)
        self.md     = self.path+'/Maps'
        self.pd     = self.path+'/PhaseHist'
        self.sd     = self.path+'/ParSeries'
        if not os.path.exists(self.path+'/Maps'):
            os.makedirs(self.path+'/Maps')
        if not os.path.exists(self.path+'/PhaseHist'):
            os.makedirs(self.path+'/PhaseHist')
        if not os.path.exists(self.path+'/ParSeries'):
            os.makedirs(self.path+'/ParSeries')

    # This method produces a simple map given a 2d array. It requires the
    # Observable, the name of the figure, and the (optional) beam.
    def    Imager(self, O, fgname):
        # Obtain information from the Observable object.
        img  = O.data
        norm = O.norm
        bds  = O.bounds
        colmap = O.colmap
        hndl = [O.lname, O.units, fgname]
        axes = O.ax
        bm   = O.beam
        # Instantiate matplotlib objects.
        fig = plt.figure()
        ax  = fig.gca()
        # Determine Image Scale: Logarithmic, Linear, or Symmetric Logarithmic.
        # Assign bounds for scale, if applicable.
        if   norm == 'log':
            if bds is None:
                normmap =    LogNorm()
            else:
                normmap =    LogNorm(vmin=bds[0],vmax=bds[1])
        elif norm == 'linear':
            if bds is None:
                normmap =  Normalize()
            else:
                normmap =  Normalize(vmin=bds[0],vmax=bds[1])
        elif norm == 'symlog':
            if bds is None:
                normmap = SymLogNorm()
            else:
                normmap = SymLogNorm(linthresh=bds[2],vmin=bds[0],vmax=bds[1])
        # Make the initial plot, and set scale, tickmarks, colorbar,  and axes
        # labels.
        im = ax.pcolormesh(img, cmap=colmap, norm=normmap)
        ax.axis('image')
        plt.xticks(self.ticks,self.labels)
        plt.yticks(self.ticks,self.labels)
        plt.minorticks_on()
        plt.xlabel(axes[0]+' (pc)')
        plt.ylabel(axes[1]+' (pc)')
        if norm == 'log':
            cb = fig.colorbar(im,ax=ax,pad=0.0)
            cb.ax.minorticks_on()
        elif norm == 'symlog':
            cb = fig.colorbar(im,ax=ax,pad=0.0,format=ticker.LogFormatterMathtext())
        else:
            cb = fig.colorbar(im,ax=ax,pad=0.0)
            cb.ax.minorticks_on()
        # If there is a beam, annotate it.
        if bm is not None:
            bc = plt.Circle((30,30),bm,color='r',linewidth=1.5,fill=False)
            ax.add_artist(bc)
        #if self.rot is not None:
        #    strrot = 'Roll: {:0.3f}, Pitch: {:0.3f}, Yaw: {:0.3f}'.format(self.rot[0],self.rot[1],self.rot[2])
    #        plt.title(hndl[0] + ' (' + hndl[1] + ')' + '\n (' + strrot + ')')
        else:
            plt.title(hndl[0] + ' (' + hndl[1] + ')')
        # Clear the figure and close it.
        # Attach relevant title handle and save the figure.
        plt.savefig(self.md+'/'+hndl[2]+'.png',dpi=self.dpi,bbox_inches='tight')
        plt.clf()
        plt.close('all')
        return

    def Hist(self, O, fgname):
        Odata = O.nyquist
        fig = plt.figure()
        ax  = fig.gca()
        if O.norm == 'log':
            nrm = True
        else:
            nrm = False
        plt.hist(Odata,bins=50,range=O.bounds,log=nrm)
        plt.title(O.lname)
        plt.xlabel(O.sname)
        plt.savefig(self.hd+'/'+fgname+'.eps',dpi=self.dpi,bbox_inches='tight')
        plt.clf()
        plt.close('all')

    # This method produces a 2d histogram phase plot of two observables computed
    # at each point on the sky. Requires the two observables, the desired
    # colormap, and the name of the file.
    def PhaseHist(self, O1, O2, colmap, fgname, titlestr):
        # Obtain information from Observable Objects
        v1    = O1.nyquist
        v2    = O2.nyquist
        bns   = [O1.bins, O2.bins]
        norms = [O1.norm, O2.norm]
        hndl  = [O1.sname+' ('+O1.units+')', O2.sname+' ('+O2.units+')', fgname]
        # Begin plotting.
        fig = plt.figure()
        ax  = fig.gca()
        # Generate 2d histogram and attempt to find the nearest log10 of the
        # maximum count. This must be done prior to masking. Then, create the
        # normalization of the number of counts.
        ct, xe, ye = np.histogram2d(v1,v2,bins=[bns[0],bns[1]])
        try:
            maxct = np.ceil(np.log10(np.max(ct)))
        except OverflowError:
            maxct = 2
        normmap = LogNorm(vmin=1,vmax=10**maxct)
        # Mask entries with no counts so the plot only displays useful data.
        ct = np.ma.masked_array(ct, ct == 0)
        # Create the phase plot of the 2d histogram, and set the scales, axes
        # labels, and colorbar.
        im = ax.pcolormesh(bns[0], bns[1], ct.T, cmap=colmap, norm = normmap)
        ax.set_xscale(norms[0])
        ax.set_yscale(norms[1])
        plt.xlabel(hndl[0])
        plt.ylabel(hndl[1])
        cb = fig.colorbar(im, ax=ax, pad=0.0)
        cb.set_label('Counts')
        ax.axis('tight')
        # Power law fitting: compute linear regression in log-log space, and
        # annotate power law index and coefficient of determination. Only do
        # this if both scales are logarithmic.
        St = Stats([self.fitres,10])
        xfit, yfit ,sl, cpt, r, p, er = St.LinearRegression(O1,O2)
        if norms[0] == 'log' and norms[1] == 'log':
            lbl = 'Index = {:0.3f} \n r$^2$ = {:0.3e}'.format(sl,r**2)
        else:
            lbl = 'Slope = {:0.3f} \n r$^2$ = {:0.3e}'.format(sl,r**2)
        lbl += '\n stderr = {:0.3e}'.format(er)
        plt.plot(xfit, yfit, 'r-', label=lbl)
        plt.legend(loc=4,fontsize='medium')
        plt.title(titlestr)
        # Save the figure.
        fig.savefig(self.pd+'/'+hndl[2]+'.eps',dpi=self.dpi,bbox_inches='tight')
        # Clear the figure and close it.
        plt.clf()
        plt.close('all')
        return

    def PhaseComp(self, O1, O2, X, Y, colmap, fgname, titlestr):
        norms = [O1.norm,O2.norm]
        hndl  = [O1.sname+' ('+O1.units+')', O2.sname+' ('+O2.units+')', fgname]
        fig = plt.figure()
        ax  = fig.gca()
        # First produce phase histogram of X, Y data.
        x1   = X.nyquist
        y1   = Y.nyquist
        bns1 = [X.bins,Y.bins]

        if norms[0] == 'log':
            x1 = np.log10(x1)
            bns1[0] = np.log10(bns1[0])
        if norms[1] == 'log':
            y1 = np.log10(y1)
            bns1[1] = np.log10(bns1[1])

        ct, xe, ye = np.histogram2d(x1,y1,bins=[bns1[0],bns1[1]])

        try:
            maxct = np.ceil(np.log10(np.max(ct)))
        except OverflowError:
            maxct = 2
        normmap = LogNorm(vmin=1,vmax=10**maxct)
        ct = np.ma.masked_array(ct, ct == 0)
        im = ax.pcolormesh(bns1[0], bns1[1], ct.T, cmap=colmap, norm=normmap)
        cb = fig.colorbar(im, ax=ax, pad=0.0)
        cb.set_label('Counts')

        # Next produce contour plot of comparison observables.
        bnds1 = O1.bounds
        bnds2 = O2.bounds

        if norms[0] == 'log':
            x2  = np.log10(O1.nyquist)
            xb1 = np.log10(bnds1[0])
            xb2 = np.log10(bnds1[-1])
        else:
            x2  = O1.nyquist
            xb1 = bnds1[0]
            xb2 = bnds1[-1]
        if norms[1] == 'log':
            y2 = np.log10(O2.nyquist)
            yb1 = np.log10(bnds2[0])
            yb2 = np.log10(bnds2[-1])
        else:
            y2 = O2.nyquist
            yb1 = bnds2[0]
            yb2 = bnds2[-1]

        #xx2, yy2 = np.mgrid[xb1:xb2:100j, yb1:yb2:100j]
        #pos2     = np.vstack([xx2.ravel(),yy2.ravel()])
        #vals2    = np.vstack([x2,y2])
        #kernel2  = stats.gaussian_kde(vals2)
        #f2       = np.reshape(kernel2(pos2).T, xx2.shape)

        St = Stats([self.fitres,10])
        xx2, yy2, f2 = St.GaussianKDE(x2,y2,[xb1,xb2],[yb1,yb2])

        #cf       = ax.contourf(xx2,yy2,f2,10,cmap='viridis')
        #cf.cmap.set_under('w')
        cs       =  ax.contour(xx2,yy2,f2,20,colors='k')

        plt.xlabel(hndl[0])
        plt.ylabel(hndl[1])
        ax.axis('tight')
        # Power law fitting: compute linear regression in log-log space, and
        # annotate power law index and coefficient of determination. Only do
        # this if both scales are logarithmic.
        #St = Stats([self.fitres,10])
        xfit1, yfit1 ,sl1, cpt1, r1, p1, er1 = St.LinearRegression(O1,O2)
        xfit2, yfit2 ,sl2, cpt2, r2, p2, er2 = St.LinearRegression(X,Y)
        if norms[0] == 'log' and norms[1] == 'log':
            lbl1 = 'Index = {:0.3f} \n r$^2$ = {:0.3e}'.format(sl1,r1**2)
            lbl2 = 'Index = {:0.3f} \n r$^2$ = {:0.3e}'.format(sl2,r2**2)
            plt.plot(np.log10(xfit1), np.log10(yfit1), 'r-', label=lbl1)
            plt.plot(np.log10(xfit2), np.log10(yfit2),'b-', label=lbl2)
        else:
            lbl1 = 'Slope = {:0.3f} \n r$^2$ = {:0.3e}'.format(sl1,r1**2)
            lbl2 = 'Slope = {:0.3f} \n r$^2$ = {:0.3e}'.format(sl2,r2**2)
            plt.plot(xfit1, yfit1, 'r-', label=lbl1)
            plt.plot(xfit2, yfit2, 'b-', label=lbl2)
        plt.legend(loc=4,fontsize='small')
        plt.title(titlestr)
        # Save the figure.
        fig.savefig(self.pd+'/'+hndl[2]+'CMP.eps',dpi=self.dpi,bbox_inches='tight')
        # Clear the figure and close it.
        plt.clf()
        plt.close('all')
        return

    def ScaledPhaseComp(self, O1, O2, X, Y, colmap, avgs, fgname, titlestr):
        norms = [O1.norm,O2.norm]
        hndl  = [O1.sname+' ('+O1.units+')', O2.sname+' ('+O2.units+')', fgname]
        fig = plt.figure()
        ax  = fig.gca()
        # First produce phase histogram of X, Y data.
        x1   = X.nyquist
        y1   = Y.nyquist
        bns1 = [X.bins,Y.bins]

        if norms[0] == 'log':
            x1 = np.log10(x1)
            bns1[0] = np.log10(bns1[0])
        if norms[1] == 'log':
            y1 = np.log10(y1)
            bns1[1] = np.log10(bns1[1])

        ct, xe, ye = np.histogram2d(x1,y1,bins=[bns1[0],bns1[1]])

        try:
            maxct = np.ceil(np.log10(np.max(ct)))
        except OverflowError:
            maxct = 2
        normmap = LogNorm(vmin=1,vmax=10**maxct)
        ct = np.ma.masked_array(ct, ct == 0)
        im = ax.pcolormesh(bns1[0], bns1[1], ct.T, cmap=colmap, norm=normmap)
        cb = fig.colorbar(im, ax=ax, pad=0.0)
        cb.set_label('Counts')

        # Next produce contour plot of comparison observables.
        bnds1 = O1.bounds
        bnds2 = O2.bounds

        if norms[0] == 'log':
            x2  = np.log10(O1.nyquist)
            xb1 = np.log10(bnds1[0])
            xb2 = np.log10(bnds1[-1])
        else:
            x2  = O1.nyquist
            xb1 = bnds1[0]
            xb2 = bnds1[-1]
        if norms[1] == 'log':
            y2 = np.log10(O2.nyquist)
            yb1 = np.log10(bnds2[0])
            yb2 = np.log10(bnds2[-1])
        else:
            y2 = O2.nyquist
            yb1 = bnds2[0]
            yb2 = bnds2[-1]

        #xx2, yy2 = np.mgrid[xb1:xb2:100j, yb1:yb2:100j]
        #pos2     = np.vstack([xx2.ravel(),yy2.ravel()])
        #vals2    = np.vstack([x2,y2])
        #kernel2  = stats.gaussian_kde(vals2)
        #f2       = np.reshape(kernel2(pos2).T, xx2.shape)

        St = Stats([self.fitres,10])
        xx2, yy2, f2 = St.GaussianKDE(x2,y2,[xb1,xb2],[yb1,yb2])

        #cf       = ax.contourf(xx2,yy2,f2,10,cmap='viridis')
        #cf.cmap.set_under('w')
        cs       =  ax.contour(xx2,yy2,f2,20,colors='k')

        plt.xlabel(hndl[0])
        plt.ylabel(hndl[1])
        ax.axis('tight')
        # Power law fitting: compute linear regression in log-log space, and
        # annotate power law index and coefficient of determination. Only do
        # this if both scales are logarithmic.
        #St = Stats([self.fitres,10])
        xfit1, yfit1 ,sl1, cpt1, r1, p1, er1 = St.LinearRegression(O1,O2)
        xfit2, yfit2 ,sl2, cpt2, r2, p2, er2 = St.LinearRegression(X,Y)
        if norms[0] == 'log' and norms[1] == 'log':
            lbl1 = 'Index = {:0.3f} \n r$^2$ = {:0.3e}'.format(sl1,r1**2)
            lbl2 = 'Index = {:0.3f} \n r$^2$ = {:0.3e}'.format(sl2,r2**2)
            plt.plot(np.log10(xfit1), np.log10(yfit1), 'r-', label=lbl1)
            plt.plot(np.log10(xfit2), np.log10(yfit2),'b-', label=lbl2)
        else:
            lbl1 = 'Slope = {:0.3f} \n r$^2$ = {:0.3e}'.format(sl1,r1**2)
            lbl2 = 'Slope = {:0.3f} \n r$^2$ = {:0.3e}'.format(sl2,r2**2)
            plt.plot(xfit1, yfit1, 'r-', label=lbl1)
            plt.plot(xfit2, yfit2, 'b-', label=lbl2)
        plt.legend(loc=4,fontsize='small')
        plt.title(titlestr)
        # Annotate the averages.
        if avgs[0] is not None:
            plt.plot(np.log10(avgs[0])*np.ones(len(yfit1)),np.log10(yfit1),'c--')
        if avgs[1] is not None:
            plt.plot(np.log10(xfit1),np.log10(avgs[1])*np.ones(len(xfit1)),'c--')
        # Save the figure.
        fig.savefig(self.pd+'/'+hndl[2]+'CMP.eps',dpi=self.dpi,bbox_inches='tight')
        # Clear the figure and close it.
        plt.clf()
        plt.close('all')
        return

    def DoubleKDE(self, X1, Y1, X2, Y2, xbnds, ybnds, fgname, titlestr):
        assert X1.norm == X2.norm
        assert Y1.norm == Y2.norm
        norms   = [X1.norm,Y1.norm]
        hndl  = ['log$_{10}$('+X1.sname+' ('+X1.units+'))', 'log$_{10}$('+Y1.sname+' ('+Y1.units+'))', fgname]
        fig     = plt.figure()
        ax      = fig.gca()
        # Compute KDEs.
        if norms[0] == 'log':
            x1  = np.log10(X1.nyquist)
            x2  = np.log10(X2.nyquist)
            xb1 = np.log10(xbnds[0])
            xb2 = np.log10(xbnds[-1])
        else:
            x1  = X1.nyquist
            x2  = X2.nyquist
            xb1 = xbnds[0]
            xb2 = xbnds[-1]
        if norms[1] == 'log':
            y1  = np.log10(Y1.nyquist)
            y2  = np.log10(Y2.nyquist)
            yb1 = np.log10(ybnds[0])
            yb2 = np.log10(ybnds[-1])
        else:
            y1  = Y1.nyquist
            y2  = Y2.nyquist
            yb1 = ybnds[0]
            yb2 = ybnds[-1]

        #xx, yy  = np.mgrid[xb1:xb2:100j, yb1:yb2:100j]
        #pos     = np.vstack([xx.ravel(),yy.ravel()])
        #vals1   = np.vstack([x1,y1])
        #vals2   = np.vstack([x2,y2])
        #kernel1 = stats.gaussian_kde(vals1)
        #kernel2 = stats.gaussian_kde(vals2)
        #f1      = np.reshape(kernel1(pos).T, xx.shape)
        #f2      = np.reshape(kernel2(pos).T, xx.shape)
        St = Stats([self.fitres,10])
        xx, yy, f1 = St.GaussianKDE(x1,y1,[xb1,xb2],[yb1,yb2])
        xx, yy, f2 = St.GaussianKDE(x2,y2,[xb1,xb2],[yb1,yb2])

        # Plot KDEs.
        #lvls = np.logspace(-0.5,0.5,10)
        #cf1     = ax.contourf(xx, yy, f1, levels=lvls, cmap   = 'plasma' )
        #cf1.cmap.set_under('w',alpha=0.0)
        cs1     =  ax.contour(xx, yy, f1, 20, cmap = 'plasma' )
        #cf2     = ax.contourf(xx, yy, f2, levels=lvls, cmap   = 'viridis')
        #cf2.cmap.set_under('w',alpha=0.0)
        cs2     =  ax.contour(xx, yy, f2, 20, cmap = 'viridis')

        plt.xlabel(hndl[0])
        plt.ylabel(hndl[1])
        ax.axis('scaled')
        plt.title(titlestr)
        fig.savefig(self.pd+'/'+hndl[2]+'2KDE.eps',dpi=self.dpi,bbox_inches='tight')
        plt.clf()
        plt.close('all')
        return
