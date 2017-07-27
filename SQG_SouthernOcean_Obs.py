# -*- coding: utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt
from netCDF4 import Dataset
import cmocean
import imp
import matplotlib.gridspec as gridspec
from scipy.interpolate import interp1d as scipy_interp1d
from scipy.interpolate import interp2d as scipy_interp2d

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

LAT_SOUTH = -60.0
LAT_NORTH = -40.0

#==========================#
# Time Limits
#==========================#
START_YEAR = 2005
END_YEAR   = 2011


#==========================#
# Input file names
#==========================#
SST_file_path      = '/media/2EC0D69917D1DC10/OISST/'
SST_file_name_stem = 'sst_SouthernOcean_'

SSH_file_path         = '/home/cchlod/AVISO/AVISO_Gridded_2/'
SSH_file_name_stem         = 'sla_dt_SouthernOcean_'

output_file_path      = '/media/2EC0D69917D1DC10/SQG_reconstructions/'
output_file_name_stem = 'SQG_Reconstruction_lon' +  str(int(LON_WEST)) + '_' + str(int(LON_EAST)) + '_lat_' + str(int(-LAT_SOUTH)) + 'S_' + str(int(-LAT_NORTH)) + 'S_'

density_file_path           = '/home/cchlod/ARGO_Analysis/Mapped_Output/'
density_file_name_stem      = 'mapped_gamma_all_sources_'
dyn_height_file_name_stem   = 'mapped_streamfunction_abs_all_sources_'
#===================================#
# Load the mapped internal  density 
#===================================#
gamma_dataset    = Dataset(density_file_path+density_file_name_stem + str(START_YEAR) + '.nc','r')
lat_gamma        =  gamma_dataset.variables['lat'][:]
lon_gamma        =  gamma_dataset.variables['lon'][:]
depth            =  gamma_dataset.variables['depth'][:]
gamma_dataset.close()

#===================================#
# Load latitude, longitude and time 
#===================================#
SST_dataset   = Dataset(SST_file_path+SST_file_name_stem + str(START_YEAR) + '.nc','r')
lat           = SST_dataset.variables['lat'][:]
lon           = SST_dataset.variables['lon'][:]
SST_dataset.close()


file_name      = '/home/cchlod/ARGO_Analysis/ANDRO_Data/ANDRO_20140218.dat'
topo_file_name = '/home/cchlod/ETOPO01/ETOPO1_SO_SubSampled.nc'
#READ the topop
topo_file = Dataset(topo_file_name, 'r')
topo_lon = np.asarray(topo_file.variables['lon'][:])
topo_lat = np.asarray(topo_file.variables['lat'][:])
topo = np.asarray(topo_file.variables['topo'][:,:])
topo_file.close()


topo_mask = np.zeros_like(topo)
topo_mask[topo>-500] = 1



#Set up the grid for the SQG validation
delta_lat = 5.0
delta_lon = 5.0
lon_validation_grid = np.arange(0+delta_lon/2,360.0-delta_lon/2+1,delta_lon)
lat_validation_grid = np.arange(LAT_SOUTH+delta_lat/2,LAT_NORTH-delta_lon/2+1,delta_lat)




delta_x = 10.0e3
delta_y = 10.0e3


for i_year in range(START_YEAR,END_YEAR):

    for i_lat in range(0,1): #lat_validation_grid.size):
        print i_lat
        lat_south_idx = np.nonzero(lat>=lat_validation_grid[i_lat]-(delta_lat/2))[0][0]
        lat_north_idx = np.nonzero(lat>=lat_validation_grid[i_lat]+(delta_lat/2))[0][0]
    
        lat_gamma_south_idx = np.nonzero(lat_gamma>=lat_validation_grid[i_lat]-(delta_lat/2))[0][0]
        lat_gamma_north_idx = np.nonzero(lat_gamma>=lat_validation_grid[i_lat]+(delta_lat/2))[0][0]
        lat_box = lat[lat_south_idx:lat_north_idx]
    
        delta_lat = lat[1]-lat[0]

        for i_lon in range(0,1): #lat_validation_grid.size):
            #===================================#
            # Load the dynamic height and density
            # climatologies to determine the 
            # mean flow and the background stratification
            #===================================#
            #We work in Mercator coordinates to make Fourier transforms easy (otherwise we
            #would need to use spherical harmonics

            #=============================#
            #Determine the domain limits
            #=============================#
            lon_west_idx = np.nonzero(lon>=lon_validation_grid[i_lon]-(delta_lon/2))[0][0]
            lon_east_idx = np.nonzero(lon>=lon_validation_grid[i_lon]+(delta_lon/2))[0][0]
            
            lon_gamma_west_idx = np.nonzero(lon_gamma>=lon_validation_grid[i_lon]-(delta_lon/2))[0][0]
            lon_gamma_east_idx = np.nonzero(lon_gamma>=lon_validation_grid[i_lon]+(delta_lon/2))[0][0]
    
            delta_lon = lon[1]-lon[0]
            lon_box = lon[lon_west_idx:lon_east_idx]

            

            m = Basemap(projection='merc',llcrnrlat=lat_box[0],urcrnrlat=lat_box[-1],
                llcrnrlon=lon_box[0],urcrnrlon=lon_box[-1],lat_ts=lat_validation_grid[i_lat],resolution='c')
    
            nX_map = int((m.xmax-m.xmin)/delta_x)+1
            nY_map = int((m.ymax-m.ymin)/delta_y)+1

            x_map = np.arange(m.xmin,m.xmax,delta_x)
            y_map = np.arange(m.ymin,m.ymax,delta_y)

            XX,YY = np.meshgrid(x_map,y_map)
            lon_map, lat_map = m(XX,YY,inverse=True)
            
    
            dyn_hgt_dataset    = Dataset(density_file_path+dyn_height_file_name_stem + str(i_year) + '.nc','r')
            dyn_hgt            =  dyn_hgt_dataset.variables['streamfunction_absolute'][:,lat_south_idx:lat_north_idx,lon_west_idx:lon_east_idx]
            dyn_hgt_dataset.close()
    
            gamma_dataset    = Dataset(density_file_path+density_file_name_stem + str(i_year) + '.nc','r')
            gamma        =  gamma_dataset.variables['gamma'][:,lat_south_idx:lat_north_idx,lon_west_idx:lon_east_idx]
            gamma_dataset.close()


            #Use the domain averaged density profile
            gamma_domain_ave = np.nanmean(np.nanmean(gamma,axis=2),axis=1)

            #==============================================================================#
            #Compute the time mean geostrophic current on each depth layer
            #==============================================================================#
            grad_Z,grad_Y,grad_X = np.gradient(dyn_hgt,1.0,delta_lat*DEG_2_RAD,delta_lon*DEG_2_RAD)
            u_mean = np.zeros_like(dyn_hgt)
            v_mean = np.zeros_like(dyn_hgt)

            for i_lat_box in range(0,lat_box.size):
                f = 2.0*EARTH_ROTATION*np.sin(DEG_2_RAD*lat_box[i_lat_box])
                u_mean[:,i_lat_box,:] = -1.0/(EARTH_RADIUS*f) * grad_Y [:,i_lat_box,:]
                v_mean[:,i_lat_box,:] =  1.0/(EARTH_RADIUS*np.cos(DEG_2_RAD*lat[i_lat_box])*f) * grad_X [:,i_lat_box,:]

    
            f0 = 2.0*EARTH_ROTATION * np.sin(DEG_2_RAD*lat[lat.size/2])
            u_mean_proj = np.zeros([depth.size,nY_map,nX_map],dtype='float64')
            v_mean_proj = np.zeros([depth.size,nY_map,nX_map],dtype='float64')
            
            #Convert the time mean velocity to Mercator coordinates
            for iZ in range(0,depth.size):
                u_mean_proj[iZ,:,:] = m.transform_scalar(u_mean[iZ,:,:],lon_box,lat_box,nX_map,nY_map,order=1)
                v_mean_proj[iZ,:,:] = m.transform_scalar(v_mean[iZ,:,:],lon_box,lat_box,nX_map,nY_map,order=1)     

            #Instansiate the SQG toolbox
            if i_year == START_YEAR and i_lat==0:
                print 'Starting solution proceedure'
                SQG_object    = SQG_Toolbox.SQG(-depth,gamma_domain_ave,f0)
                SST_dataset   = Dataset(SST_file_path+SST_file_name_stem + str(i_year) +'.nc','r')
                time          = SST_dataset.variables['time'][:]
                SST_dataset.close()

            else:
                SQG_object.Modify_Density(gamma_domain_ave)

            

            for iT in range(0,time.size):
    
                print 'iT = ', iT, ' of ', time.size
                #==================================#
                #Load the SST anomaly
                #==================================#
                SST_dataset   = Dataset(SST_file_path+SST_file_name_stem + str(i_year) + '.nc','r')
                SST_anom         = sst_scale_factor * SST_dataset.variables['anom'][iT,0,lat_south_idx:lat_north_idx,lon_west_idx:lon_east_idx]
                SST_dataset.close()

    
                #===================================#
                # SLA
                #===================================#
                SSH_dataset = Dataset(SSH_file_path+SSH_file_name_stem  + str(i_year) + '.nc','r')
                SLA         =  ssh_scale_factor*SSH_dataset.variables['sla'][iT,lat_south_idx:lat_north_idx,lon_west_idx:lon_east_idx]
                SSH_dataset.close()
        
                #Project the surface data into Mercator coordinates
                SLA_proj = m.transform_scalar(SLA,lon_box,lat_box,nX_map,nY_map,order=1)
                SST_proj = m.transform_scalar(SST_anom,lon_box,lat_box,nX_map,nY_map,order=1)   

                #Compute the SQG and interior streamfunctions based on the methodology
                #of Wang et al. (2013) and Lacase and Wang (2016)
                surf_strfun, interior_strfun, total_strfun = SQG_object.Solve_Total_Streamfunction(SLA_proj,SST_proj,delta_x,delta_y)
    
                #Compute reconstructed temperatures
                T_recon = (f0/(G0*THERMAL_COEFF)) * (total_strfun[0:-1,:,:]-total_strfun[1::,:,:])/(depth[1::]-depth[0:-1])[:,np.newaxis,np.newaxis]
                _,u_recon,v_recon = np.gradient(total_strfun,1.0,delta_y,delta_x)
                u_recon = -u_recon
    
                u_total = u_mean_proj + u_recon 
                v_total = v_mean_proj + v_recon
                
                #Interpolate back to the original lat/lon grid
                total_strfun_lat_lon = np.zeros([depth.size,lat_box.size,lon_box.size],dtype='float64')
                
                for iZ in range(0,depth.size):
                
                    interp_func = scipy_interp2d(lon_map.flatten(),lat_map.flatten(),total_strfun[iZ,:,:].flatten(),kind='linear',bounds_error=False,fill_value=np.nan)
                    total_strfun_lat_lon[iZ,:,:] = interp_func(lon_box,lat_box)
                sada
                '''
                if iT==0:    
            dataset_out   = Dataset(output_file_path+output_file_name_stem + str(i_year) + '.nc',
                                'w',clobber=True, format='NETCDF4')
            dataset_out.createDimension('time', None)
            var_time = dataset_out.createVariable('time', 'f8', ['time'])
    
            dataset_out.createDimension('x', nX_map)
            dataset_out.createDimension('y', nY_map)
            dataset_out.createDimension('depth', depth.size)
            var_lat = dataset_out.createVariable('lat', 'f8', ['y','x'])
            var_lon = dataset_out.createVariable('lon', 'f8', ['y','x'])
            var_depth = dataset_out.createVariable('depth', 'f8', ['depth'])

            var_lat[:,:]   = lat_map
            var_lon[:,:]   = lon_map
            var_depth[:]   = depth

            var_surf_strfun  = dataset_out.createVariable('surf_strfun',     'f8', ['time','depth','y','x'])
            var_int_strfun   = dataset_out.createVariable('interior_strfun', 'f8', ['time','depth','y','x'])
            var_tot_strfun   = dataset_out.createVariable('total_strfun',     'f8', ['time','depth','y','x'])
            var_u_tot_strfun = dataset_out.createVariable('u_total',     'f8', ['time','depth','y','x'])
            var_v_tot_strfun = dataset_out.createVariable('v_total',     'f8', ['time','depth','y','x'])
        
            var_u_anom_strfun = dataset_out.createVariable('u_anom',     'f8', ['time','depth','y','x'])
            var_v_anom_strfun = dataset_out.createVariable('v_anom',     'f8', ['time','depth','y','x'])
               
        var_surf_strfun[iT,:,:,:]  = surf_strfun
        var_int_strfun[iT,:,:,:]   = interior_strfun
        var_tot_strfun[iT,:,:,:]   = total_strfun
        var_u_tot_strfun[iT,:,:,:] = u_total
        var_v_tot_strfun[iT,:,:,:] = v_total
        var_u_anom_strfun[iT,:,:,:]= u_recon
        var_v_anom_strfun[iT,:,:,:]= v_recon
        var_time[iT] = time[iT]
    dataset_out.close()
        '''
