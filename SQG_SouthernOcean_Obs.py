# -*- coding: utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt
from netCDF4 import Dataset
import cmocean
import imp
import matplotlib.gridspec as gridspec
from scipy.interpolate import interp1d as scipy_interp1d
from mpl_toolkits.basemap import Basemap, addcyclic, shiftgrid
import pyproj as pyproj

SQG_Toolbox = imp.load_source('SQG_Toolbox', '/home/cchlod/PV_Inversion/SQG_Inversion/SQG_ToolBox.py')

#==========================#
# Constants
#==========================#
THERMAL_COEFF = 2.0e-4
G0            = 9.81
RHO_0         = 1000.0
EARTH_ROTATION = 2.0*np.pi/(24.0*60.0*60.0)
DEG_2_RAD      = np.pi/180.0
EARTH_RADIUS   = 6380.0e3
#==========================#
# Netcdf File scale factor
#==========================#
sst_scale_factor = 0.01
ssh_scale_factor = 1.0e-4

#==========================#
# Domain Limits
#==========================#

LON_WEST = 185.0
LON_EAST = 190.0

LAT_SOUTH = -55.0
LAT_NORTH = -50.0

#==========================#
# Input file names
#==========================#
SST_file_path = '/home/cchlod/OISST/'
SST_file_name = 'sst_SouthernOcean_2009.nc'

SSH_file_path         = '/home/cchlod/AVISO/AVISO_Gridded_2/'
SSH_file_name         = 'sla_dt_SouthernOcean_2009.nc'
output_file_path      = 

uv_file_path         = '/home/cchlod/AVISO/AVISO_Gridded_UV/'
uv_file_name          = 'uv_dt_SouthernOcean_2009.nc'

density_file_path     = '/home/cchlod/ARGO_Analysis/Mapped_Output/'
density_file_name     = 'mapped_gamma_all_sources_2009.nc'
dyn_height_file_name  = 'mapped_streamfunction_abs_all_sources_2009.nc'
#===================================#
# Load the (constant) density 
#===================================#
gamma_dataset    = Dataset(density_file_path+density_file_name,'r')
lat_gamma        =  gamma_dataset.variables['lat'][:]
lon_gamma        =  gamma_dataset.variables['lon'][:]
depth            =  gamma_dataset.variables['depth'][:]
gamma            =  gamma_dataset.variables['gamma'][:,:,:]
gamma_dataset.close()


#===================================#
# Load the (constant) dynamic height 
#===================================#
dyn_hgt_dataset    = Dataset(density_file_path+dyn_height_file_name,'r')
dyn_hgt            =  dyn_hgt_dataset.variables['streamfunction_absolute'][:,:,:]
dyn_hgt_dataset.close()



#===================================#
# Load latitude, longitude and time 
#===================================#
SST_dataset   = Dataset(SST_file_path+SST_file_name,'r')
time          = SST_dataset.variables['time'][:]
lat           = SST_dataset.variables['lat'][:]
lon           = SST_dataset.variables['lon'][:]
SST_dataset.close()

#=============================#
#Determine the domain limits
#=============================#
lon_west_idx = np.nonzero(lon>=LON_WEST)[0][0]
lon_east_idx = np.nonzero(lon>=LON_EAST)[0][0]

lat_south_idx = np.nonzero(lat>=LAT_SOUTH)[0][0]
lat_north_idx = np.nonzero(lat>=LAT_NORTH)[0][0]


lon_gamma_west_idx = np.nonzero(lon_gamma>=LON_WEST)[0][0]
lon_gamma_east_idx = np.nonzero(lon_gamma>=LON_EAST)[0][0]

lat_gamma_south_idx = np.nonzero(lat_gamma>=LAT_SOUTH)[0][0]
lat_gamma_north_idx = np.nonzero(lat_gamma>=LAT_NORTH)[0][0]

gamma      = gamma[:,lat_gamma_south_idx:lat_gamma_north_idx,lon_gamma_west_idx:lon_gamma_east_idx]
dyn_hgt    = dyn_hgt[:,lat_gamma_south_idx:lat_gamma_north_idx,lon_gamma_west_idx:lon_gamma_east_idx]

delta_lon = lon[1]-lon[0]
delta_lat = lat[1]-lat[0]

lat = lat[lat_south_idx:lat_north_idx]
lon = lon[lon_west_idx:lon_east_idx]
#==============================================================================#
#Compute the geostrophic current on isopycnal layer
#==============================================================================#
grad_Z,grad_Y,grad_X = np.gradient(dyn_hgt,1.0,delta_lat*DEG_2_RAD,delta_lon*DEG_2_RAD)
v = np.zeros_like(dyn_hgt)
u = np.zeros_like(dyn_hgt)

for i_lat in range(0,lat.size):
    f = 2.0*EARTH_ROTATION*np.sin(DEG_2_RAD*lat[i_lat])
    u[:,i_lat,:] = -1.0/(EARTH_RADIUS*f) * grad_Y [:,i_lat,:]
    v[:,i_lat,:] =  1.0/(EARTH_RADIUS*np.cos(DEG_2_RAD*lat[i_lat])*f) * grad_X [:,i_lat,:]

f  = 2.0*EARTH_ROTATION * np.sin(DEG_2_RAD*lat)


gamma_domain_ave = np.nanmean(np.nanmean(gamma,axis=2),axis=1)

delta_x = 10.0e3
delta_y = 10.0e3


f0 = 2.0*EARTH_ROTATION * np.sin(DEG_2_RAD*lat[lat.size/2])
m = Basemap(projection='merc',llcrnrlat=LAT_SOUTH,urcrnrlat=LAT_NORTH,\
                llcrnrlon=LON_WEST,urcrnrlon=LON_EAST,lat_ts=0.5*(LAT_SOUTH+LAT_NORTH),resolution='c')
nX_map = int((m.xmax-m.xmin)/delta_x)+1
nY_map = int((m.ymax-m.ymin)/delta_y)+1
u_mean_proj = np.zeros([depth.size,nY_map,nX_map],dtype='float64')
v_mean_proj = np.zeros([depth.size,nY_map,nX_map],dtype='float64')

for iZ in range(0,depth.size):
    u_mean_proj[iZ,:,:] = m.transform_scalar(u[iZ,:,:],lon,lat,nX_map,nY_map,order=1)
    v_mean_proj[iZ,:,:] = m.transform_scalar(v[iZ,:,:],lon,lat,nX_map,nY_map,order=1)     


x_map = np.arange(m.xmin,m.xmax,delta_x)
y_map = np.arange(m.ymin,m.ymax,delta_y)

#SLA_proj = m(SLA)
XX,YY = np.meshgrid(x_map,y_map)
lon_map, lat_map = m(XX,YY,inverse=True)
      
#Instansiate the SQG toolbox
SQG_object = SQG_Toolbox.SQG(-depth,gamma_domain_ave,f0)
for iT in range(0,time.size):
    
    print 'iT = ', iT, ' of ', time.size
    #==================================#
    #Load the SST anomaly
    #==================================#
    SST_dataset   = Dataset(SST_file_path+SST_file_name,'r')
    SST_anom         = sst_scale_factor * SST_dataset.variables['anom'][iT,0,lat_south_idx:lat_north_idx,lon_west_idx:lon_east_idx]
    SST_dataset.close()

    #===================================#
    # SLA
    #===================================#
    SSH_dataset         = Dataset(SSH_file_path+SSH_file_name,'r')
    SLA         =  ssh_scale_factor*SSH_dataset.variables['sla'][iT,lat_south_idx:lat_north_idx,lon_west_idx:lon_east_idx]
    SSH_dataset.close()
    #===================================#
    # U and V
    #===================================#
    uv_dataset  = Dataset(uv_file_path+uv_file_name,'r')
    u_surf      = uv_dataset.variables['u'][iT,lat_south_idx:lat_north_idx,lon_west_idx:lon_east_idx]
    v_surf      = uv_dataset.variables['v'][iT,lat_south_idx:lat_north_idx,lon_west_idx:lon_east_idx]
    uv_dataset.close()


    SLA_proj = m.transform_scalar(SLA,lon,lat,nX_map,nY_map,order=1)
    SST_proj = m.transform_scalar(SST_anom,lon,lat,nX_map,nY_map,order=1)   
    
                      


    #Compute the SQG and interior streamfunctions based on the methodology
    #of Wang et al. (2013) and Lacase and Wang (2016)
    surf_strfun, interior_strfun, total_strfun = SQG_object.Solve_Total_Streamfunction(SLA_proj,SST_proj,delta_x,delta_y)
    
    #Compute reconstructed temperatures
    T_recon = (f0/(G0*THERMAL_COEFF)) * (total_strfun[0:-1,:,:]-total_strfun[1::,:,:])/(depth[1::]-depth[0:-1])[:,np.newaxis,np.newaxis]
    _,u_recon,v_recon = np.gradient(total_strfun,1.0,delta_y,delta_x)
    u_recon = -u_recon
    
    u_total = u_mean_proj + u_recon 
    v_total = v_mean_proj + v_recon 

    if iT==0:    
        dataset_out   = Dataset(base_sla_path+output_file_name+'thres_' + str(grad_thres) + '_' + str(i_year)+'.nc',
                                'w',clobber=True, format='NETCDF4')
    dataset_out.createDimension('time', None)
    var_time = dataset_out.createVariable('time', 'f8', ['time'])


    dataset_out.createDimension('lat', n_lat)
    dataset_out.createDimension('lon', n_lon)
    var_lat = dataset_out.createVariable('lat', 'f8', ['lat'])
    var_lon = dataset_out.createVariable('lon', 'f8', ['lon'])
    var_lat[:] = lat
    var_lon[:] = lon
    var_hist = dataset_out.createVariable('jet_loc_hist', 'f8', ['lat','lon'])
    var_locations = dataset_out.createVariable('jet_locations', 'f8', ['time','lat','lon'])

    sla         = dataset_sla.variables['sla'][:,:,:]
    time        = dataset_sla.variables['time'][:]
    nT  = time.shape[0]
    
    

    
        
