# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 10:51:04 2020

@author: Lenovo
"""



import numpy as np
#from gp import GaussianProcess
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from mini_bo.gp import GaussianProcess
from mini_bo.utilities import acq_max_with_name
import copy

#======================================================================================================
#======================================================================================================
#======================================================================================================
#======================================================================================================
counter = 0


class BayesOpt:

    def __init__(self, func, SearchSpace,acq_name="ei",verbose=1):
        """      
        Input parameters
        ----------
        
        func:                       a function to be optimized
        SearchSpace:                bounds on parameters        
        acq_name:                   acquisition function name, such as [ei, gp_ucb]
                           
        Returns
        -------
        dim:            dimension
        SearchSpace:         SearchSpace on original scale
        scaleSearchSpace:    SearchSpace on normalized scale of 0-1
        time_opt:       will record the time spent on optimization
        gp:             Gaussian Process object
        """

        self.verbose=verbose
        if isinstance(SearchSpace,dict):
            # Get the name of the parameters
            self.keys = list(SearchSpace.keys())
            
            self.SearchSpace = []
            for key in list(SearchSpace.keys()):
                self.SearchSpace.append(SearchSpace[key])
            self.SearchSpace = np.asarray(self.SearchSpace)
        else:
            self.SearchSpace=np.asarray(SearchSpace)
            
            
        self.dim = len(SearchSpace)

        scaler = MinMaxScaler()
        scaler.fit(self.SearchSpace.T)
        self.Xscaler=scaler
        
        # create a scaleSearchSpace 0-1
        self.scaleSearchSpace=np.array([np.zeros(self.dim), np.ones(self.dim)]).T
                
        # function to be optimised
        self.f = func
    
        # store X in original scale
        self.X_ori= None

        # store X in 0-1 scale
        self.X = None
        
        # store y=f(x)
        # (y - mean)/(max-min)
        self.Y = None
               
        # y original scale
        self.Y_ori = None
                 

        # acquisition function
        self.acq_name = acq_name

        self.gp=GaussianProcess(self.scaleSearchSpace,verbose=verbose)

    
    def init(self, n_init_points=3,seed=1):
        """      
        Input parameters
        ----------
        gp_params:            Gaussian Process structure      
        n_init_points:        # init points
        """

        np.random.seed(seed)
        
        init_X = np.random.uniform(self.SearchSpace[:, 0], self.SearchSpace[:, 1],size=(n_init_points, self.dim))
        
        self.X_ori = np.asarray(init_X)
        
        # Evaluate target function at all initialization           
        y_init=self.f(init_X)
        y_init=np.reshape(y_init,(n_init_points,1))

        self.Y_ori = np.asarray(y_init)      
        self.Y=(self.Y_ori-np.mean(self.Y_ori))/np.std(self.Y_ori)
        self.X = self.Xscaler.transform(init_X)

        self.gp=GaussianProcess(self.scaleSearchSpace,verbose=self.verbose)
        self.gp.fit(self.X, self.Y)

       
        
    def init_with_data(self, init_X,init_Y,isPermutation=False):
        """      
        Input parameters
        ----------
        gp_params:            Gaussian Process structure      
        x,y:        # init data observations (in original scale)
        """
            
        self.Y_ori = np.asarray(init_Y)
        self.Y=(self.Y_ori-np.mean(self.Y_ori))/np.std(self.Y_ori)
        
        self.X_ori=np.asarray(init_X)
        self.X = self.Xscaler.transform(init_X)

        self.gp=GaussianProcess(self.scaleSearchSpace,verbose=self.verbose)
        self.gp.fit(self.X, self.Y)
    
    def set_ls(self,lengthscale):
        self.gp.set_ls(lengthscale)
        
    def posterior(self, Xnew):
        #self.gp.fit(self.X, self.Y,IsOptimize=1)
        self.gp.fit(self.X, self.Y)
        mu, sigma2 = self.gp.predict(Xnew)
        return mu, np.sqrt(sigma2)
        
    def select_batch_of_points(self,B):
        self.B=B
        temp_X=self.X.copy()
        temp_Y=self.Y.copy()

        temp_gp=copy.deepcopy(self.gp)
      
        x_max_batch=[]
                
        for ii in range(B):
            # Finding argmax of the acquisition function.
            x_max=acq_max_with_name(gp=temp_gp,SearchSpace=self.scaleSearchSpace,acq_name="ucb") # by default, we use GP-BUCB for batch setting
            x_max_batch.append(x_max.reshape((1, -1)))
            mu_x,sigma_x=temp_gp.predict(x_max)
        
            temp_X = np.vstack((temp_X, x_max.reshape((1, self.dim))))    
            temp_Y=np.vstack((temp_Y.reshape((-1,1)),mu_x.reshape((1,1))))

            temp_gp.fit(temp_X,temp_Y)

        return x_max_batch

    def select_next_point(self,B=1):
        """
        Main optimization method.

        Input parameters
        ----------
        gp_params: parameter for Gaussian Process

        Returns
        -------
        x: recommented point for evaluation
        """

        #self.Y=np.reshape(self.Y,(-1,1))
        self.gp=GaussianProcess(self.scaleSearchSpace,verbose=self.verbose)
        self.gp.fit(self.X, self.Y)
        self.B=B
        
        # optimize GP parameters after 3*dim iterations
        if  len(self.Y)%(3*self.dim)==0:
            self.gp.optimise()
            
        # Set acquisition function
        if B==1: # Sequential BO
            x_max=acq_max_with_name(gp=self.gp,SearchSpace=self.scaleSearchSpace,acq_name=self.acq_name)
            x_max_ori=self.Xscaler.inverse_transform(np.reshape(x_max,(-1,self.dim)))
        else: # Batch BO
            x_max=self.select_batch_of_points(B) # this is a batch of points
            # convert back to original scale
            x_max_ori=[self.Xscaler.inverse_transform(np.reshape(xx,(-1,self.dim))) for xx in x_max]
            x_max_ori=np.asarray(x_max_ori).reshape((B,self.dim))
            x_max=np.asarray(x_max).reshape((B,self.dim))

        # store X          
        self.X = np.vstack((self.X, x_max.reshape((B, -1))))
        self.X_ori=np.vstack((self.X_ori, x_max_ori))

        if self.f is None:
            return x_max,x_max_ori
              
        # evaluate Y using original X
        if B==1:
            self.Y_ori = np.append(self.Y_ori, self.f(x_max_ori))
        else:
            for idx,val in enumerate(x_max):
                self.Y_ori = np.append(self.Y_ori, self.f(val))

        # update Y after change Y_original
        self.Y=(self.Y_ori-np.mean(self.Y_ori))/np.std(self.Y_ori)

        return x_max#,x_max_ori
    
    