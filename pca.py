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
from sklearn.preprocessing import StandardScaler,FunctionTransformer
import matplotlib.pyplot as plt
#Load data
data_ridge_dir_path  = '/home/cchlod/NEMO_ANALYSIS/RIDGE05KM/'
if not isdir(data_ridge_dir_path):
    data_ridge_dir_path = '/net/argos/data/peps/cchlod/CHANNEL_OUTPUT/'
    
if not isdir(data_ridge_dir_path):
    raise FileNotFoundError('Directory '+data_ridge_dir_path+' does not exist')
    
sigma_mean_file_name        = 'sig0_RIDGE05KM_165_224.nc'


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


pca = PCA(n_components=1)
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