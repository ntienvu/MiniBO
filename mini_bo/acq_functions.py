# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 16:05:06 2020

@author: Vu Nguyen
"""

import numpy as np
from scipy.stats import norm

class AcquisitionFunction(object):
    """
    An object to compute the acquisition functions.
    """

    def __init__(self, acq_name):
                
        ListAcq=['ucb', 'ei', 'gp_ucb']
        # check valid acquisition function
        IsTrue=[val for idx,val in enumerate(ListAcq) if val in acq_name]
        #if  not in acq_name:
        if  IsTrue == []:
            err = "The utility function " \
                  "{} has not been implemented, " \
                  "please choose one of ucb, ei, or poi.".format(acq_name)
            raise NotImplementedError(err)
        else:
            self.acq_name = acq_name
                    
        
    def acq_kind(self,gp,x):
            
        y_max=np.max(gp.Y)
        
        if np.any(np.isnan(x)):
            return 0
        
        if self.acq_name == 'ucb' or self.acq_name == 'gp_ucb' :
            return self._gp_ucb( gp,x)
        if self.acq_name == 'ei':
            return self._ei(x, gp, y_max)
  
  
    @staticmethod
    def _gp_ucb(gp,xTest,fstar_scale=0):
        #dim=gp.dim
        mean, var= gp.predict(xTest)
        var.flags['WRITEABLE']=True
        var[var<1e-10]=0
     
        # Linear in D, log in t https://github.com/kirthevasank/add-gp-bandits/blob/master/BOLibkky/getUCBUtility.m
        #beta_t = gp.X.shape[1] * np.log(len(gp.Y))
        beta_t = np.log(len(gp.Y))
      
        #beta=300*0.1*np.log(5*len(gp.Y))# delta=0.2, gamma_t=0.1
        temp=mean + np.sqrt(beta_t) * np.sqrt(var)
        return  temp
    
 
    @staticmethod
    def _ei(x, gp, y_max):
        #y_max=np.asscalar(y_max)
        mean, var = gp.predict(x)
        var2 = np.maximum(var, 1e-10 + 0 * var)
        z = (mean - y_max)/np.sqrt(var2)        
        out=(mean - y_max) * norm.cdf(z) + np.sqrt(var2) * norm.pdf(z)
        
        out[var2<1e-10]=0
        
        return out
       