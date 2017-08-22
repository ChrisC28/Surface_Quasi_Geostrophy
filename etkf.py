#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 09:33:04 2017
ETKF related algorithm
@author: jbrlod
"""
import numpy as np
import scipy
def gn_ienkf(A0,w,x0,yobs,H,Rinv,epsilon=1e-5):
    """iteration of iteratve EnKF using Gauss-Newton as described in
    Bocquet and Sakov 2014, QJRMS - Algo 1
    :param A0: Initial anomaly matrix
    :param w: current value in trasformed coordinate (x = x0+ A0*w)
    :param x0: initial state value
    :param yobs: observation
    :param H: Model+observation operator
    :param Rinv: Inverse of error var-covar observartion matrix
    :param epsilon: parmeeter to perform finite difference
    return w
    """
    nens = A0.shape[1]
    Iens = np.matrix(np.identity(nens))
    
    #state parameter
    x = x0 + A0*w
    
    #Ensemble in state space
    E = x + epsilon*A0
    
    #Ensemble in observation space
    Ey = np.matrix(H(E))
    
    #Ensemle mean in observation space
    y = np.mean(Ey,axis=1)
    
    #Linear tangent (approx)
    Y = (Ey - y)/epsilon
    
    #Cost function gradient
    GradJ = (nens - 1)*w - Y.transpose()*Rinv*(yobs - y)
    
    #Hessian
    He = (nens - 1)*Iens + Y.transpose()*Rinv*Y
    
    #Gauss-Newton increment
    dw = np.linalg.solve(He,GradJ)
    
    w = w - dw
    return w
    
def ienkf(A0,x,x0,yobs,T,H,R):
    """ iteration of iterative EnKF as described in 
    or IenKF : sakov, olivier, bertino 2011
    An Iterative EnKF for Strongly Nonlinear Systems
    :param A0: Initial anomaly matrix
    :param x: current state value (mean of the ensemble)
    :param x0: initial state value
    :param yobs: observation
    :param T: Transform matrix such as A = A0*T
    :param H: Model+observation operator
    :param R: observation error variance-covariance matrix
    :return: (dx,T) the increment for x and the updated transform matrix
    
    """
    nens = A0.shape[1]
    Iens = np.matrix(np.identity(nens))
    
    #Anomalies in state space
    A = A0 * T
    
    #Ensemble in state space
    E = x + A
    
    #Ensemble in observation space
    Ey = np.matrix(H(E))
    
    #Ensemle mean in observation space
    y = np.mean(Ey,axis=1)
    
    #Anomaies in observation space
    Ay = Ey - y
    Ay = Ay*np.linalg.inv(T)
    
    #Innovation vector
    dy = yobs - y
    
    
    Rmsq = np.linalg.inv(scipy.linalg.sqrtm(R))
    s = Rmsq*dy/np.sqrt(nens-1)
    S = Rmsq*Ay/np.sqrt(nens-1)
    V = np.linalg.inv(Iens + S.T*S)
    b = V*S.T*s
    dx = A0*b + A0 * V * np.linalg.pinv(A0.T*A0) * A0.T * (x-x0)
    T = scipy.linalg.sqrtm(V)
    return (dx,T)