#*******************************Plotter.py*************************************#
#
# Author: Patrick King, Date: 02/06/18
#
#******************************************************************************#
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
    # Constructor.
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
    #
    # !!! DEPRECATED !!!
    # Phase histograms will be phased out in favor of kernel density estimation 
    # methods. Replacement with a simple KDE + PCA plotter on to-do list. 
    #
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
