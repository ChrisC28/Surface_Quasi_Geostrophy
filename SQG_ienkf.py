#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 09:26:24 2017
SQG_ienkf
Adjust density profile to fit observations
@author: jbrlod
"""

# %% Import
import numpy as np
import matplotlib.pyplot as plt
from etkf import gn_ienkf
from scipy.linalg import block_diag
from scipy.interpolate import interpn as scipy_interpnd
from pca import pca4, simulator, Xmasked
from SQG_minimise_cost_function_model import \
    mean_sigma_ridge_domain_ave,\
    xT,yT,depth_T,\
    SSH_anom_ridge,\
    T_anom_ridge,\
    x_u,y_u,x_v,y_v,\
    V_anom_ridge,\
    U_anom_ridge
    
PLOT = True #To plot the restults
REAL_OBS = True #To take obs in dataset
  
# %% Prepare data    

time_step_to_get = 0
depth_to_get     = 31
n_obs            = 50
n_ens            = 20
n_val            = 50

x_all = np.random.uniform(low=xT[1],high=xT[-2],size=n_obs+n_val)
y_all = np.random.uniform(low=yT[1],high=yT[-2],size=n_obs+n_val)
z_all = np.ones(n_obs+n_val,dtype=x_all.dtype)*depth_T[depth_to_get]

x_obs = x_all[:n_obs]
y_obs = y_all[:n_obs]
z_obs = z_all[:n_obs]
# %% Twin Experiment
sim = simulator(xT,yT,depth_T,SSH_anom_ridge,T_anom_ridge[0,:,:],mean_sigma_ridge_domain_ave,x_obs,y_obs,z_obs)


if REAL_OBS:
#Estimate the 'true' value of u and v at the psuedo obs locations
    urealobs =  scipy_interpnd( (x_u,y_u),U_anom_ridge[depth_to_get,:,:].T,
                                       np.asarray([x_obs,y_obs]).T,
                                       bounds_error=False, fill_value=np.nan) 

    vrealobs =  scipy_interpnd( (x_v,y_v),V_anom_ridge[depth_to_get,:,:].T,
                                       np.asarray([x_obs,y_obs]).T,
                                       bounds_error=False, fill_value=np.nan)
    yrealobs = np.matrix(np.concatenate((urealobs,vrealobs))[:,np.newaxis])
else:
#"True" value (simulated)
    utrue,vtrue = sim.sim_gamma(mean_sigma_ridge_domain_ave)
    ytrue = np.matrix(np.concatenate((utrue,vtrue))[:,np.newaxis])
    

# Perturbation of sigma
#sigma = mean_sigma_ridge_domain_ave.copy()
#sigma = sigma+np.random.normal(loc=0.0,scale=0.01,size=sigma.shape)

sim._pca = pca4

#sigma = np.random.normal(loc=0.0,scale=10,size=pca4._pca.n_components)
#First Guess
#ufg,vfg = sim.sim_gamma(sigma)

# %% Ensemble consitution
#state size 
sigper = 0.1

n = pca4._pca.n_components

p = pca4.transform(Xmasked)

#draw random p for each coordinate of p to consitutute the ensemble
#pens = np.zeros((n_ens,n))
#for i in range(n):
#    pens[:,i] = np.random.choice(p[:,i],n_ens)
iens = np.random.choice(p.shape[0],n_ens)
pens = np.transpose(p[iens,:]) #shape = (n,n_ens)

#add some noise:
B = sigper*np.matrix(np.identity(n))
for i in range(n_ens):
    pens[:,i] += np.random.multivariate_normal(np.zeros(n),B)
    
if PLOT:
    plt.plot(p[:,0],p[:,1],'.',color='gray')
    plt.plot(pens[0,:],pens[1,:],'o',color='red')
    plt.show()
    
#%% First guess
ufg,vfg = sim.sim_gamma(pens)
yfg = np.matrix(np.concatenate((ufg,vfg))[:,np.newaxis])
#Ensemble first guess
E0 = np.matrix(pens)
x0 = np.mean(E0,axis=1)
sigma_fg = pca4.inverse_transform(x0.transpose())

if PLOT:
    plt.plot(mean_sigma_ridge_domain_ave,depth_T,color='black',label='truth')
    plt.plot(sigma_fg,depth_T,color='blue',label='first-guess')
    plt.gca().invert_yaxis()
    plt.legend()
    plt.ylabel('depth[m]')
    plt.xlabel('density')
    plt.show()
    
    plt.plot(sigma_fg-mean_sigma_ridge_domain_ave,depth_T,color='blue',label='first-guess')
    plt.gca().invert_yaxis()
    plt.legend()
    plt.ylabel('depth[m]')
    plt.xlabel('error on density')
    plt.show()
    
    #Compare obs
    if REAL_OBS:
        plt.scatter(urealobs,np.mean(ufg,axis=1),color='blue')    
    else:
        plt.scatter(utrue,np.mean(ufg,axis=1),color='blue')
    plt.plot([-0.15,0.15],[-0.15,0.15],color='black')
    plt.show()
    
    
    if REAL_OBS:
        rfg = np.corrcoef(np.asarray(yrealobs).ravel(),np.asarray(np.mean(yfg,axis=1)).ravel())
        rmsfg = np.sqrt(np.mean(np.square(yrealobs.ravel()-np.mean(yfg,axis=1).ravel())))
    else:
        rfg = np.corrcoef(ytrue.ravel(),np.mean(yfg,axis=1).ravel())
        rmsfg = np.sqrt(np.mean(np.square(ytrue.ravel()-np.mean(yfg,axis=1).ravel())))
    print ('r FG=',rfg[0,1],' rms FG=',rmsfg)

#%% Prepare ienkf

#error on obs
sigobs = 1e-5
R = sigobs*np.matrix(np.identity(n_obs))


if REAL_OBS:
    uobs = urealobs
    vobs = vrealobs
else:
    #obs (from simulated)
    uobs = utrue + np.random.multivariate_normal(np.zeros(n_obs),R)
    vobs = vtrue + np.random.multivariate_normal(np.zeros(n_obs),R)
    
    #obs (from real obs)
    
    if PLOT:
        plt.scatter(utrue,uobs,color='green')
        plt.plot([-0.15,0.15],[-0.15,0.15],color='black')
    
        plt.show()
    
yobs = np.matrix(np.concatenate((uobs,vobs))[:,np.newaxis])
R = block_diag(R,R)
Rinv = np.linalg.inv(R)


epoch=0
A0 = E0 - x0
x = x0

epsilon = 1e-5
w = np.matrix(np.zeros((n_ens,1)))

def H(E0):
    u,v = sim.sim_gamma(E0)
    return np.matrix(np.concatenate((u,v),axis=0))

#%% Run ienkf

w = gn_ienkf(A0,w,x0,yobs,H,Rinv,epsilon)
  
epoch += 1
    
 #   dx,T = ienkf(A0,x,x0,yobs,T,H_ens,R)
    
x = x0 + A0*w
uan,van = sim.sim_gamma(x)
yan = np.matrix(np.concatenate((uan,van))[:,np.newaxis])

#%% Diags

#Compare density profile
sigma_an = pca4.inverse_transform(x.transpose())

plt.plot(mean_sigma_ridge_domain_ave,depth_T,color='black',label='truth')
plt.plot(sigma_an,depth_T,color='red',label='analysis')
plt.plot(sigma_fg,depth_T,color='blue',label='first-guess')
plt.gca().invert_yaxis()
plt.legend()
plt.ylabel('depth[m]')
plt.xlabel('density')
plt.show()

plt.plot(sigma_an-mean_sigma_ridge_domain_ave,depth_T,color='red',label='analysis')
plt.plot(sigma_fg-mean_sigma_ridge_domain_ave,depth_T,color='blue',label='first-guess')
plt.gca().invert_yaxis()
plt.legend()
plt.ylabel('depth[m]')
plt.xlabel('error on density')
plt.show()

#Compare obs
if PLOT:
    if REAL_OBS:
         plt.scatter(urealobs,np.mean(ufg,axis=1),color='blue')
         plt.scatter(urealobs,uan,color='red')
         plt.plot([-0.15,0.15],[-0.15,0.15],color='black')
         plt.show()
    else:
        plt.scatter(utrue,np.mean(ufg,axis=1),color='blue')
        plt.scatter(utrue,uan,color='red')
        plt.plot([-0.15,0.15],[-0.15,0.15],color='black')
        plt.show()


#
if REAL_OBS:
    rfg = np.corrcoef(yrealobs.ravel(),np.mean(yfg,axis=1).ravel())
    rmsfg = np.sqrt(np.mean(np.square(yrealobs.ravel()-np.mean(yfg,axis=1).ravel())))
    ran = np.corrcoef(yrealobs.ravel(),yan.ravel())
    rmsan = np.sqrt(np.mean(np.square(yrealobs.ravel()-yan.ravel())))

else:
    rfg = np.corrcoef(ytrue.ravel(),np.mean(yfg,axis=1).ravel())
    rmsfg = np.sqrt(np.mean(np.square(ytrue.ravel()-np.mean(yfg,axis=1).ravel())))
    rmsan = np.sqrt(np.mean(np.square(ytrue.ravel()-yan.ravel())))
    rmsan = np.sqrt(np.mean(np.square(ytrue.ravel()-yan.ravel())))

print ('r FG=',rfg[0,1],' r an=',ran[0,1],' rms FG=',rmsfg,' rms an=',rmsan)


# %% Validation
x_val = x_all[n_obs:]
y_val = y_all[n_obs:]
z_val = z_all[n_obs:]

sim_val = simulator(xT,yT,depth_T,SSH_anom_ridge,T_anom_ridge[0,:,:],mean_sigma_ridge_domain_ave,x_val,y_val,z_val)

if REAL_OBS:
#Estimate the 'true' value of u and v at the psuedo obs locations
    urealobs_val =  scipy_interpnd( (x_u,y_u),U_anom_ridge[depth_to_get,:,:].T,
                                       np.asarray([x_val,y_val]).T,
                                       bounds_error=False, fill_value=np.nan) 

    vrealobs_val =  scipy_interpnd( (x_v,y_v),V_anom_ridge[depth_to_get,:,:].T,
                                       np.asarray([x_val,y_val]).T,
                                       bounds_error=False, fill_value=np.nan)
    yrealobs_val = np.matrix(np.concatenate((urealobs_val,vrealobs_val))[:,np.newaxis])
else:
#"True" value (simulated)
    utrue_val,vtrue_val = sim_val.sim_gamma(mean_sigma_ridge_domain_ave)
    ytrue_val = np.matrix(np.concatenate((utrue,vtrue))[:,np.newaxis])
    
sim_val._pca = pca4
uan_val,van_val = sim_val.sim_gamma(x)

#TODO :
# Read paper Chris/JB for number of obs
# Same plot as in paper
