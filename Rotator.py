#********************************Rotator.py************************************#
#
# Author: Patrick King, Date: 02/06/18
#
#
#******************************************************************************#

from   math                        import *
import numpy                       as     np
from   scipy.ndimage.interpolation import rotate

class Rotator(object):

    # Constructor. 
    def      __init__(self, args):
        self.roll  = args[0]
        self.pitch = args[1]
        self.yaw   = args[2]
        self.N     = args[3]
        self.order = args[4]

    # Internal method: Rotate an array, with interpolation.
    def      __Rotate(self, A, angle, axis):
        B = np.zeros(np.shape(A))
        if axis == 0:
            for i in range(self.N):
                B[:,:,i] = rotate(A[:,:,i], angle, reshape=False,order=self.order,mode='wrap')
        elif axis == 1:
            for i in range(self.N):
                B[:,i,:] = rotate(A[:,i,:], angle, reshape=False,order=self.order,mode='wrap')
        else:
            for i in range(self.N):
                B[i,:,:] = rotate(A[i,:,:], angle, reshape=False,order=self.order,mode='wrap')
        return B

    # Rotate Scalar Field, defined by S
    def  ScalarRotate(self, S):
        if  self.roll != 0.0:
            S = self.__Rotate(S, self.roll,  axis = 2)
        if self.pitch != 0.0:
            S = self.__Rotate(S, self.pitch, axis = 0)
        if   self.yaw != 0.0:
            S = self.__Rotate(S, self.yaw,   axis = 1)
        return S

    # Rotate Vector Field with coordinate components C1, C2, C3
    def  VectorRotate(self, C1, C2, C3):
        if self.roll  != 0.0:
            C1, C2, C3 =  self.__VectorRoll(C1,C2,C3)
        if self.pitch != 0.0:
            C1, C2, C3 = self.__VectorPitch(C1,C2,C3)
        if self.yaw   != 0.0:
            C1, C2, C3 =   self.__VectorYaw(C1,C2,C3)
        C1R = self.ScalarRotate(C1)
        C2R = self.ScalarRotate(C2)
        C3R = self.ScalarRotate(C3)
        return C1R, C2R, C3R

    def  __VectorRoll(self, C1, C2, C3):
        cosb = np.cos(np.deg2rad(self.roll))
        sinb = np.sin(np.deg2rad(self.roll))
        C1r  =  C1
        C2r  =  cosb*C2 - sinb*C3
        C3r  =  sinb*C2 + cosb*C3
        return C1r, C2r, C3r

    def __VectorPitch(self, C1, C2, C3):
        cosa = np.cos(np.deg2rad(self.pitch))
        sina = np.sin(np.deg2rad(self.pitch))
        C1r  =  cosa*C1 - sina*C2
        C2r  =  sina*C1 + cosa*C2
        C3r  =  C3
        return C1r, C2r, C3r

    def   __VectorYaw(self, C1, C2, C3):
        cosc = np.cos(np.deg2rad(self.yaw))
        sinc = np.sin(np.deg2rad(self.yaw))
        C1r  =  cosc*C1 + sinc*C3
        C2r  =  C2
        C3r  = -sinc*C1 + cosc*C3
        return C1r, C2r, C3r
