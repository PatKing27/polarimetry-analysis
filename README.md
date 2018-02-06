# polarimetry-analysis
Tools to analyze submillimeter polarization observations

## about 
This is a collection of tools used for analyzing, exploring, and characterizing submillimeter (linear) polarization observations. Generally, these datasets are image-like - two spatial dimensions - but contain more than just one number/value per pixel. Associated with each pixel is a vector consisting of the two Stokes parameters, Q and U, normalized by the total intensity I. The magnitude of this vector is called the polarization fraction, and the orientation defined by the vector is called the polarization angle. 

Linearly polarized signals in the typical targets of these observations - giant molecular clouds and other dense structures in the interstellar medium - arise due to the alignment of tiny particles, called dust grains, with the local magnetic field. Even though they are very cold, dust grains will emit blackbody radiation (hence the *submillimeter* part) and since the grains are irregular and tiny, the light they emit is polarized. The magnetic field is quite difficult to study, but quite important in understanding what is going on in these structures - especially in understanding the process of star formation. 

Since the polarization observations contain an imprint of the magnetic field - albeit projected, density weighted, and combined nonlinearly - analyzing them can provide some insight into the magnetic field orientation, dynamics, and maybe even field strength (all of which I collectively call *magnetic field structure.*) The tools in this repository are some of the tools I've implemented to try to gain this insight. Hopefully this will provide a solid foundation that can be built upon and expanded to ever more sophisticated tools. 

While these tools are intended for use by the observer, I have neglected to mention the other important half to these tools. I originally wrote these tools with an eye to analyzing *synthetic* observations of simulated molecular clouds. Accordingly I wrote these tools with that application in mind, and only later applied these techniques to real observations. Included here is the code I wrote to perform these synthetic observations on 3D magnetohydrodynamics simulations.

## code
The code is written entirely in python, utilizing several scipy packages, especially numpy, yt, matplotlib, scipy.stats, and (most recently) fastkde. 

## wish list 
* output 
  * store observational datasets in natural, rapidly readable form
  * store costly calculated products like kernel density estimation PDFs for later use 
* optimization
  * synthetic observations: costly numpy summation replaced with Cython implementation 
* continuum/line observation integration 
  * M0, M1, M2 - line-of-sight velocity 
* new analysis/visualization 
  * Projected Rayleigh Statistic and Histogram of Relative Orientations 
  * Line Integral Convolution (LIC)
  * Power Spectrum - fft 
  * higher order geometric statistics? 
* sampling/resampling 
  * Nyquist sampling 
  * arbitrary beamform 
