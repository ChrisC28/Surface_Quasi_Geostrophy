import numpy as np
from matplotlib import pyplot as plt
from scipy.fftpack import fft2,ifft2,fftshift,ifftshift,fftfreq

nX = 128
nY = 128

permutivity = 8.85e-12
Lx = 1.0
Ly = 1.0

x = np.linspace(0,Lx,nX)
y = np.linspace(0,Ly,nY)

delta_x = x[1]-x[0]
delta_y = y[1]-y[0]

XX,YY = np.meshgrid(x,y)



RHS = np.zeros([nY,nX],dtype='float64')

idx = np.nonzero(x>=0.5*Lx)[0][0]
idy = np.nonzero(y>=0.55*Ly)[0][0]

print idy
RHS[idy,idx] = 1.0 * delta_x * delta_y / permutivity
idy = np.nonzero(y>=0.45*Ly)[0][0]
RHS[idy,idx] = -1.0 * delta_x * delta_y / permutivity


RHS_mirror = np.zeros([2.0*nY,2.0*nX],dtype=RHS.dtype)
RHS_mirror[0:nY,0:nX] = RHS
RHS_mirror[0:nY,nX:2*nX] = RHS[:,::-1]
RHS_mirror[nY:2*nY,0:nX] = RHS[::-1,:]
RHS_mirror[nY:2*nY,nX:2*nX] = RHS[::-1,::-1]


kx = 2.0*np.pi*fftfreq(2*nX,d=delta_x)
ly = 2.0*np.pi*fftfreq(2*nY,d=delta_y)
KX,LY = np.meshgrid(kx, ly)
K2 = KX*KX + LY*LY
K2[0,0] = 1.0

RHS_fft = fft2(RHS_mirror)

soln = ifft2(RHS_fft/(-K2))
soln = soln[0:nY,0:nY]






