# -*- coding: utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt
from netCDF4 import Dataset
import cmocean
import imp
import matplotlib.gridspec as gridspec
import scipy.signal as scipysignal
from scipy import sparse
from netCDF4 import Dataset
from scipy.fftpack import fft2,ifft2,fftshift,ifftshift,fftfreq
import pyamg

THERMAL_COEFF = 2.0e-4
G0            = 9.81
RHO_0         = 1000.0

file_path = '/media/2EC0D69917D1DC10/RIDGE05KM/Output/'
T_file_name = 'RIDGE05KM_5d_02240101_02241231_grid_T.nc'

data_ridge_dir_path  = '/home/cchlod/NEMO_ANALYSIS/RIDGE05KM/'
N2_file_name                = 'N2_MEAN_RIDGE05KM_165_212.nc'
mean_T_file_name            = 'T_mean_5d_RIDGE05KM_165_225.nc'

#===================================#
#Variable Temperature components
#===================================#
mean_T_dataset         = Dataset(file_path+T_file_name,'r')

xT       = mean_T_dataset.variables['nav_lon'][0,:]
yT       = mean_T_dataset.variables['nav_lat'][:,0]
depth_T  = mean_T_dataset.variables['deptht'][:]

SSH_ridge         = mean_T_dataset.variables['sossheig'][0,:,:]
T_ridge           = mean_T_dataset.variables['votemper'][0,0,:,:]
mean_T_dataset.close()

#=======================#
# N2 components
#=======================#
mean_iso_depth_dataset = Dataset(data_ridge_dir_path+N2_file_name,'r')
N2_ridge       = mean_iso_depth_dataset.variables['vobn2'][0,:,:,:]
mean_iso_depth_dataset.close()


#===================================#
#Time Mean Temperature components
#===================================#
mean_T_dataset         = Dataset(data_ridge_dir_path+mean_T_file_name,'r')

xT     = mean_T_dataset.variables['nav_lon'][0,:]
yT     = mean_T_dataset.variables['nav_lat'][:,0]
depth  = mean_T_dataset.variables['deptht'][:]

mean_SSH_ridge         = mean_T_dataset.variables['sossheig'][0,:,:]
mean_T_ridge           = mean_T_dataset.variables['votemper'][0,0,:,:]
mean_MLD_ridge         = mean_T_dataset.variables['mld001'][0,:,:]
mean_T_dataset.close()


x_lim = [0,100]
y_lim = [150,250]

N2_ridge_domain_ave = np.nanmean(np.nanmean(N2_ridge[:,y_lim[0]:y_lim[1]+1,x_lim[0]:x_lim[1]+1],axis=2),axis=1)

SSH_anom_ridge = SSH_ridge-mean_SSH_ridge
T_anom_ridge   = T_ridge-mean_T_ridge


SSH_anom_ridge = SSH_anom_ridge[y_lim[0]:y_lim[1]+1,x_lim[0]:x_lim[1]+1]
T_anom_ridge   = (THERMAL_COEFF*G0/RHO_0) * T_anom_ridge[y_lim[0]:y_lim[1]+1,x_lim[0]:x_lim[1]+1]
x_loc          = xT[x_lim[0]:x_lim[1]+1]
y_loc          = yT[y_lim[0]:y_lim[1]+1]

nY,nX=SSH_anom_ridge.shape 

#Mirror flip to ensure periodicity 
SSH_anom_ridge_mirror = np.zeros([2.0*nY,2.0*nX],dtype=SSH_anom_ridge.dtype)
SSH_anom_ridge_mirror[0:nY,0:nX] = SSH_anom_ridge
SSH_anom_ridge_mirror[0:nY,nX:2*nX] = SSH_anom_ridge[:,::-1]
SSH_anom_ridge_mirror[nY:2*nY,0:nX] = SSH_anom_ridge[::-1,:]
SSH_anom_ridge_mirror[nY:2*nY,nX:2*nX] = SSH_anom_ridge[::-1,::-1]

T_anom_ridge_mirror                   = np.zeros([2.0*nY,2.0*nX],dtype=SSH_anom_ridge.dtype)
T_anom_ridge_mirror[0:nY,0:nX]        = T_anom_ridge
T_anom_ridge_mirror[0:nY,nX:2*nX]     = T_anom_ridge[:,::-1]
T_anom_ridge_mirror[nY:2*nY,0:nX]     = T_anom_ridge[::-1,:]
T_anom_ridge_mirror[nY:2*nY,nX:2*nX]  = T_anom_ridge[::-1,::-1]

#Fourier transform in x and y 

SSH_anom_ridge_FFT = fftshift(fft2(SSH_anom_ridge_mirror))
T_anom_ridge_FFT   = fftshift(fft2(T_anom_ridge_mirror))

kx = 2.0*np.pi*fftfreq(2*nX,d=xT[1]-xT[0])
ly = 2.0*np.pi*fftfreq(2*nY,d=yT[1]-yT[0])




f0 = 1.0e-4
max_depth = depth_T[-1]
delta_z   = 50.0
depth     = np.arange(0,max_depth+delta_z/2,delta_z)
nZ        = depth.size

N2_surf    = 3.0e-5
N2_efold   = 750.0
N2_profile = N2_surf * np.exp(-depth/N2_efold)
S          = f0*f0/N2_profile


RHS  = np.zeros(nZ,dtype='float64')



#nZ = 5
#D2z  = sparse.csr_matrix((nZ,nZ),dtype=np.complex128)
D2z  = np.zeros((nZ,nZ),dtype=np.float64)

D2z[range(1,nZ-1),range(1,nZ-1)] = -(S[2:nZ] + S[1:nZ-1]) / (delta_z)
D2z[range(1,nZ-1),range(0,nZ-2)] =   S[1:nZ-1] /   (delta_z*delta_z)
D2z[range(1,nZ-1),range(2,nZ)]   =   S[2:nZ]   /   (delta_z*delta_z)

#Neumann boundary conditions at the surface
D2z[0,0] = -S[1] / delta_z#1.0/delta_z
D2z[0,1] =  S[1] / delta_z#-1.0 /delta_z
#Neumann conditions at the bottom
D2z[nZ-1,nZ-1] =-S[nZ-1] / delta_z #-1.0/delta_z
D2z[nZ-1,nZ-2] = S[nZ-1] / delta_z#1.0/delta_z

eigenvalues,eigenvectors = np.linalg.eig(D2z)

rossby_radius = eigenvalues[0]

dsa


SQG_streamfunction_FFT = np.zeros([nZ,2*nY,2*nX],dtype='float64')

for iY in range(0,2*nY):
    print 'iY= ', iY, ' of ', 2*nY 
    for iX in range(0,2*nX):
        DK = sparse.csr_matrix((nZ,nZ),dtype=np.float64)
        DK[range(1,nZ-1),range(1,nZ-1)] = K2[iY,iX]

        
        #Modify the RHS to impose the boundary condititions
        top_bc = T_anom_ridge_FFT[iY,iX]/f0
        RHS_current = RHS.copy() + 0.0j
        
        RHS_current[0]    = RHS_current[0]+ top_bc
        #print - (top_bc/ delta_z)
        #print RHS_current[0]
        #print '===================='
        RHS_current[nZ-1] = 0.0
        
        D = D2z - DK
        
        
        #multigrid_obj = pyamg.ruge_stuben_solver(D)     # construct the multigrid hierarchy
        #multigrid_obj = pyamg.smoothed_aggregation_solver(D)
        #SQG_streamfunction_FFT[:,iY,iX] = multigrid_obj.solve(RHS_current, tol=1e-5)         # solve Ax=b to a tolerance of 1e-5
        

SQG_streamfunction = np.zeros_like(SQG_streamfunction_FFT)
for iZ in range(0,nZ):
    SQG_streamfunction[iZ,:,:] = fftshift(ifft2(SQG_streamfunction_FFT[iZ,:,:]))









#D2z[range(0,nZ-2),range(1,nZ-1)] = S[1:nZ-1]

#D2z[range(0,nZ),range(0,nZ)] = -(S[0:nZ-1] + S[1:nZ])
#D2z[range(1,nZ-1),range(2,nZ)] = S[0:nZ-2]
#D2z[range(2,nZ),range(1,nZ-1)] = S[1:nZ]



#D2z[range(0,nZ-1),range(1,nZ-2)] = 1# S[0:nZ-1]
#D2z[range(2,nZ),range(1,nZ-3)]    = 1#S[1:nZ]

