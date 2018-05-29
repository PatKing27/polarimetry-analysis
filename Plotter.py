#******************************Plotter.py**************************************#
#
# Author: Patrick King, Date: 02/06/18
#
# Update (PKK) 04/17/18: Elimiated deprecated methods. Passed testing.
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
        axes = O.axes
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
        #    plt.title(hndl[0] + ' (' + hndl[1] + ')' + '\n (' + strrot + ')')
        else:
            plt.title(hndl[0] + ' (' + hndl[1] + ')')
        # Clear the figure and close it.
        # Attach relevant title handle and save the figure.
        plt.savefig(self.md+'/'+hndl[2]+'.png',dpi=self.dpi,bbox_inches='tight')
        plt.clf()
        plt.close('all')
        return
