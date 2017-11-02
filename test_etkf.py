#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 12:05:34 2017
Test ETKF

@author: jbrlod
"""

import numpy as np
from etkf import gn_ienkf
from scipy.interpolate import interp1d
import  matplotlib.pyplot as plt
VERBOSE = 2
def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise (ValueError, "smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise (ValueError, "Input vector needs to be bigger than window size.")


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise (ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y[(window_len//2):-(window_len//2)]


nobs = 50
nens = 30
n = 100
nepoch = 10
#error standard deviation
sig = 10.0
sigobs = 0.01
window_len = 31
#coordinates
t = np.linspace(-1,1,n)
tobs = np.random.uniform(-1,1,size =nobs)

#model
def M(x):
    return np.power(x,3)-np.power(x,2)+x
#    return 2*x +1 
def H(x):
    global tobs,t
    return interp1d(t,M(x))(tobs)

def H_ens(E):
    ret = np.zeros((nobs,E.shape[1]))
    for i in range(E.shape[1]):
        ret[:,i] = H(np.array(E[:,i]).ravel())
    return ret
#truth
xt = np.sin(np.pi*t)


#FG
E0 = np.matrix(np.zeros((n,nens)))
for i in range(E0.shape[1]):
    E0[:,i] = np.random.standard_normal((t.size,1))
    E0[:,i] = (xt+sig*smooth(E0[:,i].A1,window_len=window_len))[:,np.newaxis]
x0 = np.mean(E0,axis=1)

R = sigobs*np.matrix(np.identity(nobs))
yobs =H(xt) + np.random.multivariate_normal(np.zeros(nobs),R)
yobs = np.matrix(yobs[:,np.newaxis]) # shape = (nobs,1)

plt.plot(t,xt,'-b')
plt.plot(t,M(xt),'-r')
plt.plot(tobs,yobs,'+k')
plt.plot(t,x0,':b')
plt.plot(t,M(x0),':r')
plt.show()

#IenKF
R = np.matrix(np.identity(nobs))
Rinv = np.linalg.inv(R)

epoch=0
A0 = E0 - x0
x = x0
T = np.matrix(np.identity(nens))
Iens = np.matrix(np.identity(nens))

#for GN-IENKF
epsilon = 1e-5
w = np.matrix(np.zeros((nens,1)))

while epoch<nepoch:
    
    w = gn_ienkf(A0,w,x0,yobs,H_ens,Rinv,epsilon)
 
    
    epoch += 1
    
 #   dx,T = ienkf(A0,x,x0,yobs,T,H_ens,R)
    
    x = x0 + A0*w
    
    if VERBOSE>0:
        print('error=',np.linalg.norm(x-np.matrix(xt[:,np.newaxis])))
    if VERBOSE>1:
        print('xt(50)=',xt[50],' x(50)=',x[50])
    

plt.plot(t,xt,'-b',label='xtrue')
plt.plot(t,x0,':b',label='x first guess')
plt.plot(t,x,'-m',label='analysis')
plt.legend()
plt.show()


plt.plot(t,M(xt),'-r',label='M(xtrue)')
plt.plot(tobs,yobs,'+k',label='obs')
plt.plot(t,M(x0),':r',label='M(x first guess)')
plt.plot(t,M(x),'-m',label='M(analysis)')
plt.legend()
plt.show()