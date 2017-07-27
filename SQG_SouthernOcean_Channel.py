# -*- coding: utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt
from netCDF4 import Dataset
import cmocean
import imp
import matplotlib.gridspec as gridspec
from scipy.interpolate import interp1d as scipy_interp1d
from scipy.stats import linregress 
SQG_Toolbox = imp.load_source('SQG_Toolbox', '/home/cchlod/PV_Inversion/SQG_Inversion/SQG_ToolBox.py')

THERMAL_COEFF = 2.0e-4
G0            = 9.81
RHO_0         = 1000.0
F0            = -1.0e-4

#zone 1
#x_lim = [200,300]
#zone 2
x_lim = [700,801]
y_lim = [150,250]

#=======================================================#
# Time Variable Data File Definitions
#=======================================================#
file_path          = '/media/2EC0D69917D1DC10/RIDGE05KM/Output/'
T_file_name_stem   = 'RIDGE05KM_5d_02000101_02001231_grid_T.nc'
U_file_name_stem   = 'RIDGE05KM_5d_02000101_02001231_grid_U.nc'
V_file_name_stem   = 'RIDGE05KM_5d_02000101_02001231_grid_V.nc'
sig_file_name_stem = 'sig0_RIDGE05KM_02000101_02001231.nc'

#=======================================================#
# Time Mean Data File Definitions
#=======================================================#
data_ridge_dir_path  = '/home/cchlod/NEMO_ANALYSIS/RIDGE05KM/'
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

delta_x = xT[1]-xT[0]
delta_y = yT[1]-yT[0]
delta_z = 50
depth_SQG = np.arange(0,4000+delta_z,delta_z) 
#Compute the background stratification profile
mean_sigma_ridge_domain_ave = np.nanmean(np.nanmean(mean_sigma_ridge,axis=2),axis=1)
rho_profile                 = np.interp(depth_SQG,depth_T,mean_sigma_ridge_domain_ave)

#Initialise the SQG toolbox and build the vertical differentiation matrix
SQG_object = SQG_Toolbox.SQG(-depth_SQG,rho_profile,F0)

#Solve for the eigenvectors and eigenvalues, as they do not change with time
rossby_wavenum,normal_modes= SQG_object.Vertical_Eigenmodes()



mean_u_recon = np.zeros([depth_T.size-1,yT.size-1,xT.size-1],dtype='float64')
mean_v_recon = np.zeros([depth_T.size-1,yT.size-1,xT.size-1],dtype='float64')
mean_T_recon = np.zeros([depth_T.size-1,yT.size,xT.size],dtype='float64')

mean_u_model = np.zeros([depth_T.size-1,yT.size-1,xT.size-1],dtype='float64')
mean_v_model = np.zeros([depth_T.size-1,yT.size-1,xT.size-1],dtype='float64')
mean_T_model = np.zeros([depth_T.size-1,yT.size,xT.size],dtype='float64')

EKE_recon    = np.zeros([depth_T.size-1,yT.size-1,xT.size-1],dtype='float64')
EKE_model    = np.zeros([depth_T.size-1,yT.size-1,xT.size-1],dtype='float64')


RMS_T_recon    = np.zeros([depth_T.size-1,yT.size,xT.size],dtype='float64')
RMS_T_model    = np.zeros([depth_T.size-1,yT.size,xT.size],dtype='float64')


RMSE_u = np.zeros(depth_T.size-1,dtype='float64')
RMSE_v = np.zeros(depth_T.size-1,dtype='float64')
RMSE_T = np.zeros(depth_T.size-1,dtype='float64')

correlation_u = np.zeros(depth_T.size-1,dtype='float64')
correlation_v = np.zeros(depth_T.size-1,dtype='float64')
correlation_T = np.zeros(depth_T.size-1,dtype='float64')


histogram_bins   = np.arange(-0.1,0.10001,0.005)

err_histogram_u = []
err_histogram_v = []

for i_hist_level in range(0,depth_T.size-1):
    err_histogram_u.append(np.zeros(histogram_bins.size-1,dtype='float64'))
    err_histogram_v.append(np.zeros(histogram_bins.size-1,dtype='float64'))




#Get the number of time steps
mean_T_dataset = Dataset(file_path+T_file_name_stem,'r')
nT   = mean_T_dataset.dimensions['time_counter'].size
mean_T_dataset.close()



for iT in range(0,1):  #nT):
    print iT,  ' of ', nT, 'time steps'
    #===================================#
    #Variable Temperature components
    #===================================#
    T_dataset    = Dataset(file_path+T_file_name_stem,'r')
    SSH_ridge    = T_dataset.variables['sossheig'][iT,y_lim[0]:y_lim[1],x_lim[0]:x_lim[1]]
    T_ridge      = T_dataset.variables['votemper'][iT,:,y_lim[0]:y_lim[1],x_lim[0]:x_lim[1]]
    T_dataset.close()
    
    
    sig_dataset    = Dataset(file_path+sig_file_name_stem,'r')
    sigma0_ridge   = sig_dataset.variables['vosigma0'][iT,:,y_lim[0]:y_lim[1],x_lim[0]:x_lim[1]]
    sig_dataset.close()

    
            
    SSH_anom_ridge = SSH_ridge-mean_SSH_ridge
    T_anom_ridge   = T_ridge-mean_T_ridge
    sigma0_anom_ridge = (sigma0_ridge - mean_sigma_ridge)/RHO_0
    

    #surf_strfun     = SQG_object.Solve_Surface_Streamfunction(0.5*G0 *THERMAL_COEFF * T_anom_ridge[0,:,:],delta_x*1.0e3,delta_y*1.0e3)
    surf_strfun     = SQG_object.Solve_Surface_Streamfunction(-G0*sigma0_anom_ridge[0,:,:],delta_x*1.0e3,delta_y*1.0e3)

    interior_strfun = SQG_object.Solve_Interior_Streamfunction((G0/F0)*SSH_anom_ridge,normal_modes,[0.1])
    total_strfun = surf_strfun + interior_strfun
    
    plt.figure(1)
    plt.contourf(xT,yT,total_strfun[20,:,:],25,cmap=plt.cm.jet)
    plt.show()
    
    
    plt.figure(2)
    plt.contourf(xT,yT,(G0/F0)*SSH_anom_ridge,25,cmap=plt.cm.jet)
    plt.show()
    
    dsa
    #Reconstruct the temperature annomaly
    T_recon =  -(0.5*F0/(G0)) * (total_strfun[0:-1,:,:]-total_strfun[1::,:,:])/(depth_SQG[1::]-depth_SQG[0:-1])[:,np.newaxis,np.newaxis]
    
    
    #===================================#
    #Variable U and V components
    #===================================#
    U_dataset = Dataset(file_path+U_file_name_stem,'r')
    U_ridge   = U_dataset.variables['vozocrtx'][iT,:,y_lim[0]:y_lim[1],x_lim[0]:x_lim[1]]
    U_dataset.close()

    V_dataset = Dataset(file_path+V_file_name_stem,'r')
    V_ridge  = V_dataset.variables['vomecrty'][iT,:,y_lim[0]:y_lim[1],x_lim[0]:x_lim[1]]
    V_dataset.close()

    U_anom_ridge   = U_ridge-mean_U_ridge
    V_anom_ridge   = V_ridge-mean_V_ridge

    #Interpolate the U and V data to the T points 
    U_anom_ridge = 0.5*(U_anom_ridge[:,1::,1::]+U_anom_ridge[:,1::,0:-1])
    V_anom_ridge = 0.5*(V_anom_ridge[:,0:-1,1::]+V_anom_ridge[:,1::,1::])
    #Interpolate the SQG solution to the model vertical grid
    
    total_strfun_interp = np.zeros([depth_T.size-1,yT.size,xT.size],dtype='float64')
    T_recon_interp      = np.zeros([depth_T.size-1,yT.size,xT.size],dtype='float64')
    for iY in range(0,yT.size):
        for iX in range(0,xT.size):
            
            total_strfun_interp[:,iY,iX] = np.interp(depth_T[0:-1],depth_SQG,total_strfun[:,iY,iX])
            T_recon_interp[:,iY,iX]      = np.interp(depth_T[0:-1],0.5*(depth_SQG[0:-1]+depth_SQG[1::]),T_recon[:,iY,iX])
    
    _,u_recon,v_recon = np.gradient(total_strfun_interp,1.0,delta_y*1.0e3,delta_x*1.0e3)
    u_recon = -u_recon
    
    #b_recon = (1.0/F0) * (total_strfun_interp[0:-1,:,:]-total_strfun_interp[1::,:,:])/((depth_T[1:-1]-depth_T[0:-2])[:,np.newaxis,np.newaxis])
    
    mean_u_recon = mean_u_recon + u_recon[:,1::,1::]
    mean_v_recon = mean_v_recon + v_recon[:,1::,1::]
    mean_T_recon = mean_T_recon + T_recon_interp
    
    mean_u_model = mean_u_model + U_anom_ridge[0:-1,:,:]
    mean_v_model = mean_v_model + V_anom_ridge[0:-1,:,:]
    mean_T_model = mean_T_model + T_anom_ridge[0:-1,:,:]
    
    
    EKE_recon    = EKE_recon + 0.5 * (u_recon[:,1::,1::]*u_recon[:,1::,1::] + v_recon[:,1::,1::]*v_recon[:,1::,1::])
    EKE_model    = EKE_model + 0.5 * (U_anom_ridge*U_anom_ridge + V_anom_ridge*V_anom_ridge)[0:-1,:,:]

    RMS_T_recon    = RMS_T_recon +  (T_recon_interp*T_recon_interp)
    RMS_T_model    = RMS_T_model +  (T_anom_ridge[0:-1,:,:]*T_anom_ridge[0:-1,:,:])

                        
    RMSE_u  = RMSE_u + np.nanmean(np.nanmean((u_recon[:,1::,1::] - U_anom_ridge[0:-1,:,:]) * (u_recon[:,1::,1::] - U_anom_ridge[0:-1,:,:]),axis=2),axis=1)
    RMSE_v  = RMSE_v + np.nanmean(np.nanmean((v_recon[:,1::,1::] - V_anom_ridge[0:-1,:,:]) * (v_recon[:,1::,1::] - V_anom_ridge[0:-1,:,:]),axis=2),axis=1)
    RMSE_T  = RMSE_T + np.nanmean(np.nanmean((T_recon_interp - T_anom_ridge[0:-1,:,:]) * (T_recon_interp - T_anom_ridge[0:-1,:,:]),axis=2),axis=1)

    for iZ in range(0,depth_T.size-1):
        slope, intercept, r_value, p_value, std_err = linregress( U_anom_ridge[iZ,:,:].flatten(), u_recon[iZ,1::,1::].flatten())
        correlation_u[iZ] = correlation_u[iZ] + r_value*r_value
        slope, intercept, r_value, p_value, std_err = linregress( V_anom_ridge[iZ,:,:].flatten(), v_recon[iZ,1::,1::].flatten())
        correlation_v[iZ] = correlation_v[iZ]+ r_value*r_value
        slope, intercept, r_value, p_value, std_err = linregress( T_anom_ridge[iZ,:,:].flatten(), T_recon_interp[iZ,:,:].flatten())
        correlation_T[iZ] = correlation_T[iZ] + r_value*r_value
        
        error_histogram_u, errror_bins_edges = np.histogram(u_recon[iZ,1::,1::].flatten()-U_anom_ridge[iZ,:,:].flatten(),histogram_bins)
        error_histogram_v, errror_bins_edges = np.histogram(v_recon[iZ,1::,1::].flatten()-V_anom_ridge[iZ,:,:].flatten(),histogram_bins)

        err_histogram_u[iZ][:] = err_histogram_u[iZ][:] + error_histogram_u
        err_histogram_v[iZ][:] = err_histogram_v[iZ][:] + error_histogram_v

            
    
        
        
RMSE_u = np.sqrt(RMSE_u/float(nT))
RMSE_v = np.sqrt(RMSE_v/float(nT))
RMSE_T = np.sqrt(RMSE_T/float(nT))

mean_u_recon = mean_u_recon/float(nT)
mean_v_recon = mean_v_recon/float(nT)
mean_T_recon = mean_T_recon/float(nT)

mean_u_model = mean_u_model/float(nT)
mean_v_model = mean_v_model/float(nT)
mean_T_model = mean_T_model/float(nT)

EKE_recon    = EKE_recon/float(nT)
EKE_model    = EKE_model/float(nT)
RMS_T_recon  = np.sqrt(RMS_T_recon/float(nT))
RMS_T_model  = np.sqrt(RMS_T_model/float(nT))

for i_hist_level in range(0,depth_T.size-1):
    err_histogram_u[i_hist_level][:] = err_histogram_u[i_hist_level][:]/float(nT)
    err_histogram_v[i_hist_level][:] = err_histogram_v[i_hist_level][:]/float(nT)

correlation_T = correlation_T/float(nT)
correlation_u = correlation_u/float(nT)
correlation_v = correlation_v/float(nT)

output_path          = '/home/cchlod/PV_Inversion/SQG_Inversion/'
output_stat_file     = 'SQG_performance_stats_model_region2.npz'
output_snapshot_file = 'SQG_performance_snapshot_region2.npz'


np.savez(output_path+output_stat_file,RMSE_u=RMSE_u,RMSE_v=RMSE_v,RMSE_T=RMSE_T,EKE_recon=EKE_recon,EKE_model=EKE_model,RMS_T_recon=RMS_T_recon,RMS_T_model=RMS_T_model,
                                      correlation_u=correlation_u,correlation_v=correlation_v,correlation_T=correlation_T,
                                      err_histogram_u=err_histogram_u,err_histogram_v=err_histogram_v,histogram_bins=histogram_bins)
                                      
np.savez(output_path+output_snapshot_file,U_anom_ridge=U_anom_ridge,V_anom_ridge=V_anom_ridge,T_anom_ridge=T_anom_ridge,
                                          u_recon=u_recon,v_recon=v_recon,T_recon_interp=T_recon_interp)
                                          

