#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 11:53:15 2017
PCA analysis
@author: jbrlod
"""

# %% Import
from os.path import isdir
import xarray as xr
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from SQG_minimise_cost_function_model import SQG_reconstruction
#Load data
data_ridge_dir_path  = '/home/cchlod/NEMO_ANALYSIS/RIDGE05KM/'
if not isdir(data_ridge_dir_path):
    data_ridge_dir_path = '/net/argos/data/peps/cchlod/CHANNEL_OUTPUT/'
    
if not isdir(data_ridge_dir_path):
    raise FileNotFoundError('Directory '+data_ridge_dir_path+' does not exist')
    
sigma_mean_file_name        = 'sig0_RIDGE05KM_165_224.nc'

#%%%%%%%%%%%%%%%%%%%%%%%%%%%
n_components = 4
#TODO : should be parametrized

#%% Load data
#============================#
#Set the domain limits
#zone 1
#============================#
x_lim = [200,300]
y_lim = [150,250]

#Load all sigma
data = xr.open_dataset(os.path.join(data_ridge_dir_path,sigma_mean_file_name))


#%% Preprocessing
#Extract region
sigma = data['vosigma0'].isel(time_counter=0,
             x = slice(*tuple(x_lim)),
             y = slice(*tuple(y_lim)))

count = sigma.count(dim='deptht')
mask = count==data['deptht'].size
Xumasked = sigma.stack(geo=('x','y'))
mask1D = mask.stack(geo=('x','y'))
Xmasked = Xumasked.where(mask1D,drop=True)
Xmasked=Xmasked.transpose('geo','deptht')


# %% pca
dimgeo = 0
if not Xmasked.dims[dimgeo] == 'geo':
    raise ValueError('dim not correct')
scaler = StandardScaler().fit(Xmasked)
X = scaler.transform(Xmasked)


pca = PCA(n_components=n_components)
pca.fit(X)

#%%plot PCA
if __name__ == '__main__':
    plt.plot(scaler.mean_,sigma.deptht)
    plt.gca().invert_yaxis()
    plt.ylabel('depth[m]')
    plt.xlabel('density')
    
    plt.title('mean density profile')
    plt.show()
    
    
    x = range(1,len(pca.explained_variance_ratio_)+1)
    plt.bar(x,pca.explained_variance_ratio_)
    plt.plot(x,pca.explained_variance_ratio_.cumsum(),'r.-')
    plt.xticks(x)
    plt.xlabel('eigen value')
    plt.ylabel('explained variance')
    plt.show()
    
    plt.plot(pca.components_[0,:],sigma.deptht)
    plt.gca().invert_yaxis()
    plt.title('First PCA component')
    plt.xlabel('component value')
    plt.ylabel('depth[m]')


#%% Class
class pcasim:
    def __init__(self,pca,scaler,depth):
        self._pca = pca
        self._scaler = scaler
        self._depth = depth
        self._nfeatures = self._pca.n_components
        
    def transform (self,X):
        if X.ndim ==1:
            X = X[np.newaxis,:]
        Xn = scaler.transform(X)
        return pca.transform(Xn).squeeze()
    
    def inverse_transform(self,proj):
        if proj.ndim ==1:
            proj = proj[np.newaxis,:]
        Xn = pca.inverse_transform(proj)
        return scaler.inverse_transform(Xn).squeeze()
        
    
pca4 = pcasim(pca,scaler,sigma.deptht)

#%% Model Simulator class
class simulator:
    def __init__(self,x_grid,y_grid,depth_grid,SSH,SST,density_profile,x_obs,y_obs,z_obs,pca=None):
        self._x_grid = x_grid
        self._y_grid = y_grid
        self._depth_grid = depth_grid
        self._SSH = SSH
        self._SST = SST
        self._pca = pca
        self._nsimul = 0
        if self._pca is None:
            self._density_profile = density_profile
        else:
            self._density_profile = pca.inverse_transform(density_profile)
        self._x_obs = x_obs
        self._y_obs = y_obs
        self._z_obs = z_obs
        
    #set parameters
    def set_params(self,**parameters):
        #print(self._density_profile)
        for par,val in parameters.items():
         
            if par == '_density_profile' and not self._pca is None:
                if val.ndim == 2:
                    val = np.transpose(val) #beause ensemble dim order != pca dim order
                val = self._pca.inverse_transform(val)
            setattr(self,par,val)
            
        #print(self._density_profile)

    def simulate(self):
        self._nsimul = self._nsimul+1
        #print('Sim nn',self._nsimul)
        if self._density_profile.ndim == 2:
            u = []
            v = []
            for dp in self._density_profile:
                utmp,vtmp = SQG_reconstruction(self._x_grid,self._y_grid,self._depth_grid,
                                  self._SSH,self._SST,dp,
                                  self._x_obs,self._y_obs,self._z_obs)
                u.append(utmp)
                v.append(vtmp)
            u = np.transpose(np.array(u)) #shape = (n,nens)
            v = np.transpose(np.array(v))
            return u,v
        else:    
            return SQG_reconstruction(self._x_grid,self._y_grid,self._depth_grid,
                                  self._SSH,self._SST,self._density_profile,
                                  self._x_obs,self._y_obs,self._z_obs)
    

    
    def sim_gamma(self,density_profile):
        self.set_params(_density_profile=density_profile)
        return self.simulate()
    
    def residual_gamma(self,density_profile,uobs,vobs):
        usim,vsim = self.sim_gamma(density_profile)
        return (np.array((uobs,vobs))-np.array((usim,vsim))).ravel()
    
    def loss_gamma(self,density_profile,uobs,vobs):
        
        J = np.linalg.norm(self.residual_gamma(density_profile,uobs,vobs))
        #print('loss = ',J)
        return J
        
        