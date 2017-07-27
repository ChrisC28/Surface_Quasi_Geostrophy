# -*- coding: utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt
from netCDF4 import Dataset
import cmocean
import imp
import matplotlib.gridspec as gridspec
from scipy.interpolate import interp1d as scipy_interp1d
from scipy.stats import linregress
from scipy.interpolate import interpn as scipy_interpnd
from os.path import join, isfile, isdir

try:
    SQG_Toolbox = imp.load_source('SQG_Toolbox', '/home/cchlod/PV_Inversion/SQG_Inversion/SQG_ToolBox.py')
except FileNotFoundError:
    import SQG_ToolBox as SQG_Toolbox

def SQG_reconstruction(x_grid,y_grid,depth_grid,SSH,SST,density_profile,x_obs,y_obs,z_obs):
    
    
    F0      = -1.0e-4
    delta_z = 50.0
    delta_x = (x_grid[1]-x_grid[0])*1.0e3
    delta_y = (y_grid[1]-y_grid[0])*1.0e3

    
    depth_SQG = np.arange(0,depth_grid[-1]+delta_z,delta_z) 
    #Compute the background stratification profile
    interp_density_profile = np.interp(depth_SQG,depth_grid,density_profile)

    SQG_object = SQG_Toolbox.SQG(-depth_SQG,interp_density_profile,F0)
    surf_strfun, interior_strfun, total_strfun = SQG_object.Solve_Total_Streamfunction(SSH,SST,delta_x,delta_y)
    
    #Determine the anomalous velocity from the 
    _,u_recon,v_recon = np.gradient(total_strfun,1.0,delta_y,delta_x)
    u_recon = -u_recon
    
    
    sqg_u_at_obs =  scipy_interpnd( [depth_SQG,y_grid,x_grid],u_recon,
                                     np.asarray([z_obs,y_obs,x_obs]).T,
                                      bounds_error=False, fill_value=np.nan)
                                                  
    sqg_v_at_obs =  scipy_interpnd( [depth_SQG,y_grid,x_grid],v_recon,
                                      np.asarray([z_obs,y_obs,x_obs]).T,
                                                  bounds_error=False, fill_value=np.nan)
                                                  
    return sqg_u_at_obs, sqg_v_at_obs
    
    
THERMAL_COEFF = 2.0e-4
G0            = 9.81
RHO_0         = 1000.0
F0            = -1.0e-4


#============================#
#Set the domain limits
#zone 1
#============================#
x_lim = [200,300]
y_lim = [150,250]

#============================================================================#
# Load Time Mean Model Data from Netcdf Files
#============================================================================#

#======================================#
# Time Variable Data File Definitions
#======================================#
file_path          = '/media/2EC0D69917D1DC10/RIDGE05KM/Output/'
if not isdir(file_path):
    file_path = '/net/argos/data/peps/cchlod/CHANNEL_OUTPUT/'
    
if not isdir(file_path):
    raise FileNotFoundError('Directory '+file_path+' does not exist')
    
T_file_name_stem   = 'RIDGE05KM_5d_02000101_02001231_grid_T.nc'
U_file_name_stem   = 'RIDGE05KM_5d_02000101_02001231_grid_U.nc'
V_file_name_stem   = 'RIDGE05KM_5d_02000101_02001231_grid_V.nc'
sig_file_name_stem = 'sig0_RIDGE05KM_02000101_02001231.nc'

#=====================================#
# Time Mean Data File Definitions
#=====================================#
data_ridge_dir_path  = '/home/cchlod/NEMO_ANALYSIS/RIDGE05KM/'
if not isdir(data_ridge_dir_path):
    data_ridge_dir_path = '/net/argos/data/peps/cchlod/CHANNEL_OUTPUT/'
    
if not isdir(data_ridge_dir_path):
    raise FileNotFoundError('Directory '+data_ridge_dir_path+' does not exist')
    


mean_T_file_name            = 'T_mean_5d_RIDGE05KM_165_225.nc'
U_depth_mean_file_name      = 'U_mean_5d_RIDGE05KM_165_225.nc'
V_depth_mean_file_name      = 'V_mean_5d_RIDGE05KM_165_225.nc'
sigma_mean_file_name        = 'sig0_RIDGE05KM_165_224.nc'


#===================================#
#Time Mean Temperature components
#===================================#
mean_T_dataset         = Dataset(data_ridge_dir_path+mean_T_file_name,'r')
xT       = mean_T_dataset.variables['nav_lon'][0,x_lim[0]:x_lim[1]]
yT       = mean_T_dataset.variables['nav_lat'][y_lim[0]:y_lim[1],0]
depth_T  = mean_T_dataset.variables['deptht'][:]

mean_SSH_ridge         = mean_T_dataset.variables['sossheig'][0,y_lim[0]:y_lim[1],x_lim[0]:x_lim[1]]
mean_T_ridge           = mean_T_dataset.variables['votemper'][0,:,y_lim[0]:y_lim[1],x_lim[0]:x_lim[1]]
mean_MLD_ridge         = mean_T_dataset.variables['mld001'][0,y_lim[0]:y_lim[1],x_lim[0]:x_lim[1]]
mean_T_dataset.close()


#===================================#
#Time Mean Density components
#===================================#
mean_sigma_dataset   = Dataset(data_ridge_dir_path+sigma_mean_file_name,'r')
mean_sigma_ridge     = mean_sigma_dataset.variables['vosigma0'][0,:,y_lim[0]:y_lim[1],x_lim[0]:x_lim[1]]
mean_sigma_dataset.close()

#==========================#
# Mean U depth components
#==========================#
mean_u_dataset = Dataset(data_ridge_dir_path+U_depth_mean_file_name,'r')
mean_U_ridge       = mean_u_dataset.variables['vozocrtx'][0,:,y_lim[0]:y_lim[1],x_lim[0]:x_lim[1]]
x_u     = mean_u_dataset.variables['nav_lon'][0,x_lim[0]:x_lim[1]]
y_u     = mean_u_dataset.variables['nav_lat'][y_lim[0]:y_lim[1],0]
mean_u_dataset.close()
#==========================#
# Mean V depth components
#==========================#
mean_v_dataset = Dataset(data_ridge_dir_path+V_depth_mean_file_name,'r')
x_v            = mean_v_dataset.variables['nav_lon'][0,x_lim[0]:x_lim[1]]
y_v            = mean_v_dataset.variables['nav_lat'][y_lim[0]:y_lim[1],0]
mean_V_ridge   = mean_v_dataset.variables['vomecrty'][0,:,y_lim[0]:y_lim[1],x_lim[0]:x_lim[1]]
mean_v_dataset.close()

#============================================================================#
# Finished Loading Files
#============================================================================#

#==============================#
#Determine the domain averaged
#density profile
#=============================#
mean_sigma_ridge_domain_ave = np.nanmean(np.nanmean(mean_sigma_ridge,axis=2),axis=1)

#Parameters for the psuedo obs - w
time_step_to_get = 0
depth_to_get     = 31
n_obs            = 50

#============================================================================#
# Load Time Varying Model Data from Netcdf Files for Reconstruction
#============================================================================#
#===================================#
#Variable Temperature components
#===================================#
T_dataset    = Dataset(file_path+T_file_name_stem,'r')
SSH_ridge    = T_dataset.variables['sossheig'][time_step_to_get,y_lim[0]:y_lim[1],x_lim[0]:x_lim[1]]
T_ridge      = T_dataset.variables['votemper'][time_step_to_get,:,y_lim[0]:y_lim[1],x_lim[0]:x_lim[1]]
T_dataset.close()

#===================================#
#Variable Density components
#===================================#
sig_dataset    = Dataset(file_path+sig_file_name_stem,'r')
sigma0_ridge   = sig_dataset.variables['vosigma0'][time_step_to_get,:,y_lim[0]:y_lim[1],x_lim[0]:x_lim[1]]
sig_dataset.close()

#===================================#
#Variable U and V components
#===================================#
U_dataset = Dataset(file_path+U_file_name_stem,'r')
U_ridge   = U_dataset.variables['vozocrtx'][time_step_to_get,:,y_lim[0]:y_lim[1],x_lim[0]:x_lim[1]]
U_dataset.close()

V_dataset = Dataset(file_path+V_file_name_stem,'r')
V_ridge  = V_dataset.variables['vomecrty'][time_step_to_get,:,y_lim[0]:y_lim[1],x_lim[0]:x_lim[1]]
V_dataset.close()



#====================================#
# SQG theory is only capable of 
# reconstructing _anomalies_
# from _anomalies_. Thus, we form 
# anomaly fields
#====================================#

U_anom_ridge   = U_ridge-mean_U_ridge
V_anom_ridge   = V_ridge-mean_V_ridge


SSH_anom_ridge = SSH_ridge-mean_SSH_ridge
T_anom_ridge   = T_ridge-mean_T_ridge
sigma0_anom_ridge = (sigma0_ridge - mean_sigma_ridge)/RHO_0 #used to form buoyancy



#Generate the psuedo obs
x_obs = np.random.uniform(low=xT[0],high=xT[-1],size=n_obs)
y_obs = np.random.uniform(low=yT[0],high=yT[-1],size=n_obs)
z_obs = np.ones(n_obs,dtype=x_obs.dtype)*depth_T[depth_to_get]


 
#SQG reconstruction at the psuedo obs locations
#u_recon,v_recon = SQG_reconstruction(xT,yT,depth_T,SSH_anom_ridge,sigma0_anom_ridge[0,:,:],mean_sigma_ridge_domain_ave,x_obs,y_obs,z_obs)
u_recon,v_recon = SQG_reconstruction(xT,yT,depth_T,SSH_anom_ridge,T_anom_ridge[0,:,:],mean_sigma_ridge_domain_ave,x_obs,y_obs,z_obs)

#Estimate the 'true' value of u and v at the psuedo obs locations
u_obs =  scipy_interpnd( (x_u,y_u),U_anom_ridge[depth_to_get,:,:].T,
                                       np.asarray([x_obs,y_obs]).T,
                                       bounds_error=False, fill_value=np.nan) 

v_obs =  scipy_interpnd( (x_v,y_v),V_anom_ridge[depth_to_get,:,:].T,
                                       np.asarray([x_obs,y_obs]).T,
                                       bounds_error=False, fill_value=np.nan)




