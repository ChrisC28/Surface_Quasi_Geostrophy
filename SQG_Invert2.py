# -*- coding: utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt
from netCDF4 import Dataset
import cmocean
import imp
import matplotlib.gridspec as gridspec
import scipy.signal as scipysignal
from scipy import sparse
from scipy import linalg
from scipy.fftpack import fft2,ifft2,fftshift,ifftshift,fftfreq
import pyamg
SQG_Toolbox = imp.load_source('SQG_Toolbox', '/home/cchlod/PV_Inversion/SQG_Inversion/SQG_ToolBox.py')


THERMAL_COEFF = 2.0e-4
G0            = 9.81
RHO_0         = 1000.0

file_path = '/media/2EC0D69917D1DC10/RIDGE05KM/Output/'
T_file_name = 'RIDGE05KM_5d_02000101_02001231_grid_T.nc'
U_file_name = 'RIDGE05KM_5d_02000101_02001231_grid_U.nc'
V_file_name = 'RIDGE05KM_5d_02000101_02001231_grid_V.nc'

data_ridge_dir_path  = '/home/cchlod/NEMO_ANALYSIS/RIDGE05KM/'
N2_file_name                = 'N2_MEAN_RIDGE05KM_165_212.nc'
mean_T_file_name            = 'T_mean_5d_RIDGE05KM_165_225.nc'
U_depth_mean_file_name      = 'U_mean_5d_RIDGE05KM_165_225.nc'
V_depth_mean_file_name      = 'V_mean_5d_RIDGE05KM_165_225.nc'
sigma_mean_file_name        = 'sig0_RIDGE05KM_165_224.nc'


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


#===================================#
#Variable U and V components
#===================================#
mean_U_dataset = Dataset(file_path+U_file_name,'r')
xU        = mean_U_dataset.variables['nav_lon'][0,:]
yU        = mean_U_dataset.variables['nav_lat'][:,0]
U_ridge   = mean_U_dataset.variables['vozocrtx'][0,:,:,:]
mean_U_dataset.close()

mean_V_dataset = Dataset(file_path+V_file_name,'r')
xV       = mean_V_dataset.variables['nav_lon'][0,:]
yV       = mean_V_dataset.variables['nav_lat'][:,0]
V_ridge  = mean_V_dataset.variables['vomecrty'][0,:,:,:]
mean_V_dataset.close()

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

#===================================#
#Time Mean Density components
#===================================#
mean_sigma_dataset   = Dataset(data_ridge_dir_path+sigma_mean_file_name,'r')
mean_sigma_ridge     = mean_sigma_dataset.variables['vosigma0'][0,:,:]
mean_sigma_dataset.close()

#==========================#
# Mean U depth components
#==========================#
mean_u_dataset = Dataset(data_ridge_dir_path+U_depth_mean_file_name,'r')
mean_U_ridge       = mean_u_dataset.variables['vozocrtx'][0,:,:,:]
x_u     = mean_u_dataset.variables['nav_lon'][0,:]
y_u     = mean_u_dataset.variables['nav_lat'][:,0]
mean_u_dataset.close()
#==========================#
# Mean V depth components
#==========================#
mean_v_dataset = Dataset(data_ridge_dir_path+V_depth_mean_file_name,'r')
x_v            = mean_v_dataset.variables['nav_lon'][0,:]
y_v            = mean_v_dataset.variables['nav_lat'][:,0]
mean_V_ridge   = mean_v_dataset.variables['vomecrty'][0,:,:,:]
mean_v_dataset.close()




delta_x = xT[1]-xT[0]
delta_y = yT[1]-yT[0]

x_lim = [0,100]
y_lim = [150,250]

sigma_ridge_domain_ave = np.nanmean(np.nanmean(mean_sigma_ridge[:,y_lim[0]:y_lim[1]+1,x_lim[0]:x_lim[1]+1],axis=2),axis=1)

SSH_anom_ridge = SSH_ridge-mean_SSH_ridge
T_anom_ridge   = T_ridge-mean_T_ridge
U_anom_ridge   = U_ridge-mean_U_ridge
V_anom_ridge   = V_ridge-mean_V_ridge


SSH_anom_ridge = SSH_anom_ridge[y_lim[0]:y_lim[1]+1,x_lim[0]:x_lim[1]+1]
U_anom_ridge   = U_anom_ridge[:,y_lim[0]:y_lim[1]+1,x_lim[0]:x_lim[1]+1]
V_anom_ridge   = V_anom_ridge[:,y_lim[0]:y_lim[1]+1,x_lim[0]:x_lim[1]+1]


T_anom_ridge   = (THERMAL_COEFF*G0/RHO_0) * T_anom_ridge[y_lim[0]:y_lim[1]+1,x_lim[0]:x_lim[1]+1]
x_loc          = xT[x_lim[0]:x_lim[1]+1]
y_loc          = yT[y_lim[0]:y_lim[1]+1]


rho_profile_1 = np.interp(depth,depth_T,sigma_ridge_domain_ave)

RHO_0 = 1000.0
g0    = 9.81
f0    = -1.0e-4

SQG_object = SQG_Toolbox.SQG(-depth,rho_profile,f0)
rossby_wavenum,barotropic_mode,baroclinic_mode = SQG_object.Vertical_Eigenmodes()
dsa
surf_strfun     = SQG_object.Solve_Surface_Streamfunction(T_anom_ridge,delta_x*1.0e3,delta_y*1.0e3)
interior_strfun = SQG_object.Solve_Interior_Streamfunction((g0/f0)*SSH_anom_ridge,barotropic_mode,baroclinic_mode)
total_strfun = surf_strfun+interior_strfun


_,u_recon,v_recon = np.gradient(total_strfun,1.0,delta_y*1.0e3,delta_x*1.0e3)
u_recon = -u_recon

u_recon_rms  = np.nanmean(np.nanmean(np.sqrt(u_recon*u_recon),axis=2),axis=1)
v_recon_rms  = np.nanmean(np.nanmean(np.sqrt(v_recon*v_recon),axis=2),axis=1)

u_model_rms = np.nanmean(np.nanmean(np.sqrt(U_anom_ridge*U_anom_ridge),axis=2),axis=1)
v_model_rms = np.nanmean(np.nanmean(np.sqrt(V_anom_ridge*V_anom_ridge),axis=2),axis=1)

dasd




for iY in range(nY,2*nY):
    print 'iY= ', iY, ' of ', 2*nY 
    for iX in range(nX,2*nX):
        
        DK= -np.diag(K2[iY,iX]*np.ones(nZ,dtype='float64'),k=0)
        #DK[0,0] = 0.0 
        #DK[nZ-1,nZ-1] = 0.0
        #Modify the RHS to impose the boundary condititions
        top_bc = S*T_anom_ridge_FFT[iY,iX]/(f0*d_rho[0])
        #top_bc = T_anom_ridge_FFT[iY,iX]/f0
        RHS_current = RHS.copy()
        
        #RHS_current[0]    = RHS_current[0] + top_bc
        RHS_current[0]    =  RHS_current[0] - top_bc
        #print - (top_bc/ delta_z)
        #print RHS_current[0]
        #print '===================='
        RHS_current[nZ-1] = 0.0
        
        D = D2z + DK
        SQG_streamfunction_FFT[:,iY,iX]  = linalg.solve(D, RHS_current)
        
        #print SQG_streamfunction_FFT[:,iY,iX]
        #multigrid_obj = pyamg.ruge_stuben_solver(D)     # construct the multigrid hierarchy
        #multigrid_obj = pyamg.smoothed_aggregation_solver(D)
        #SQG_streamfunction_FFT[:,iY,iX] = multigrid_obj.solve(RHS_current, tol=1e-5)         # solve Ax=b to a tolerance of 1e-5
        



#Project the interior solution onto the baroclinc mode

modal_coeffs = np.zeros([2,2*nY,2*nX],dtype='complex128')
modal_LHS = np.zeros([2,2],dtype='complex128')
modal_LHS[0,0] = pm[0,0]
modal_LHS[0,1] = pm[0,1]

modal_LHS[1,0] = pm[-1,0]
modal_LHS[1,1] = pm[-1,1]

for iY in range(0,2*nY):
    for iX in range(0,2*nX):
        
        
        modal_RHS = np.zeros([2,1],dtype='complex128')
        modal_RHS[0] = (g0/f0*SSH_anom_ridge_FFT[iY,iX]) - SQG_streamfunction_FFT[0,iY,iX]
        modal_RHS[1] = -SQG_streamfunction_FFT[-1,iY,iX]

        modal_coeffs[:,iY,iX] = np.squeeze(linalg.solve(modal_LHS, modal_RHS))



IQG_streamfunction_FFT = modal_coeffs[0,:,:] * pm[:,0][:,np.newaxis,np.newaxis] + modal_coeffs[1,:,:] * pm[:,1][:,np.newaxis,np.newaxis] 



SQG_streamfunction = np.zeros_like(SQG_streamfunction_FFT)
IQG_streamfunction = np.zeros_like(IQG_streamfunction_FFT)


for iZ in range(0,nZ):
    #print SQG_streamfunction_FFT[iZ,:,:]
    SQG_streamfunction[iZ,:,:] = ifft2(SQG_streamfunction_FFT[iZ,:,:])
    IQG_streamfunction[iZ,:,:] = ifft2(IQG_streamfunction_FFT[iZ,:,:])


SQG_streamfunction = SQG_streamfunction[:,0:nY,0:nX]
IQG_streamfunction = IQG_streamfunction[:,0:nY,0:nX]

QG_streamfunction = SQG_streamfunction + IQG_streamfunction


_,u_recon,v_recon = np.gradient(QG_streamfunction,1.0,delta_y*1.0e3,delta_x*1.0e3)
u_recon = -u_recon


#D2z[range(0,nZ-2),range(1,nZ-1)] = S[1:nZ-1]

#D2z[range(0,nZ),range(0,nZ)] = -(S[0:nZ-1] + S[1:nZ])
#D2z[range(1,nZ-1),range(2,nZ)] = S[0:nZ-2]
#D2z[range(2,nZ),range(1,nZ-1)] = S[1:nZ]



#D2z[range(0,nZ-1),range(1,nZ-2)] = 1# S[0:nZ-1]
#D2z[range(2,nZ),range(1,nZ-3)]    = 1#S[1:nZ]

