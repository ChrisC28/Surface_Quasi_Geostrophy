#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 17:16:35 2017
Optimisation experiments
@author: jbrlod
TODO : add correlation of coefficients
"""

# %% Import
import numpy as np
import sys
from scipy.optimize import minimize, least_squares, basinhopping
import matplotlib.pyplot as plt
from pca import pca4
from SQG_minimise_cost_function_model import \
    SQG_reconstruction,\
    mean_sigma_ridge_domain_ave,\
    xT,yT,depth_T,\
    SSH_anom_ridge,\
    T_anom_ridge,\
    U_anom_ridge,\
    V_anom_ridge

  
# %% Prepare data    

time_step_to_get = 0
depth_to_get     = 31
n_obs            = 50

x_obs = np.random.uniform(low=xT[0],high=xT[-1],size=n_obs)
y_obs = np.random.uniform(low=yT[0],high=yT[-1],size=n_obs)
z_obs = np.ones(n_obs,dtype=x_obs.dtype)*depth_T[depth_to_get]

# %% 

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
                val = self._pca.inverse_transform(val)
            setattr(self,par,val)
            
        #print(self._density_profile)

    def simulate(self):
        self._nsimul = self._nsimul+1
        #print('Sim nn',self._nsimul)
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
        
        

# %% Twin Experiment
sim = simulator(xT,yT,depth_T,SSH_anom_ridge,T_anom_ridge[0,:,:],mean_sigma_ridge_domain_ave,x_obs,y_obs,z_obs)

#"True" value
utrue,vtrue = sim.sim_gamma(mean_sigma_ridge_domain_ave)

# Perturbation of sigma
#sigma = mean_sigma_ridge_domain_ave.copy()
#sigma = sigma+np.random.normal(loc=0.0,scale=0.01,size=sigma.shape)

sim._pca = pca4
sigma = np.random.normal(loc=0.0,scale=10,size=pca4._pca.n_components)
#First Guess
ufg,vfg = sim.sim_gamma(sigma)

#Computation : J = sim.loss_gamma(sigma,*(utrue,vtrue))
# %% Minimize
minimizer = 'Nelder-Mead'

options= dict()
options['BFGS'] = {'disp':True,'maxiter':20,'eps':1.0e-7}
options['Nelder-Mead'] = {'disp':True,'maxiter':20,'maxfev':100}

Nfeval = 1
def callbackF(sigma):
    global Nfeval
    print(sigma,Nfeval,sim.loss_gamma(sigma,*(utrue,vtrue)))
    Nfeval += 1


if minimizer in {'BFGS','Nelder-Mead'}:
    result = minimize(sim.loss_gamma,sigma,args=(utrue,vtrue),
                      method=minimizer,tol = 1e-4,
                      options=options[minimizer],
                      callback = callbackF)
    
elif minimizer == 'least_squares':
    result = least_squares(sim.residual_gamma,sigma,
                           method = 'lm',
                           args=(utrue,vtrue),
                           verbose=2)
    
elif minimizer == 'basinhopping':
    result = basinhopping(sim.loss_gamma,sigma,
                           minimizer_kwargs = {'args':(utrue,vtrue)},
                           disp=True)    

urec,vrec = sim.sim_gamma(result.x)

print ('Final = ',result.x)
#%% Plots
plt.quiver(x_obs,y_obs,utrue,vtrue,color='red',label='true')
plt.quiver(x_obs,y_obs,ufg,vfg,color='green',label='fg')
plt.quiver(x_obs,y_obs,urec,vrec,color='blue',label='rec')
plt.legend()
plt.show()
plt.savefig('quiver.png')
plt.scatter(utrue,ufg,color='green')
plt.scatter(utrue,urec,color='blue')
plt.show()
plt.savefig('scatter.png')
plt.plot(mean_sigma_ridge_domain_ave[1:]-mean_sigma_ridge_domain_ave[:-1],color='red')
plt.plot(pca4.inverse_transform(sigma)[1:]-pca4.inverse_transform(sigma)[:-1],color='green')
plt.plot(sim._density_profile[1:]-sim._density_profile[:-1],color='blue')
plt.show()
plt.savefig('density.png')

#%% Plot cost function
sim = simulator(xT,yT,depth_T,SSH_anom_ridge,T_anom_ridge[0,:,:],mean_sigma_ridge_domain_ave,x_obs,y_obs,z_obs)
sim._pca = pca4
x = np.linspace(-5,5,20)
sigx,sigy = np.meshgrid(x,x)
J = np.zeros(sigx.shape)
count=0

for ix in range(J.shape[0]):
    print(ix+1,'/',J.shape[0])
    for iy in range(J.shape[1]):
        sigma = np.array([sigx[ix,iy],sigy[ix,iy]])
        J[ix,iy] = sim.loss_gamma(sigma,*(utrue,vtrue))
        count +=1 