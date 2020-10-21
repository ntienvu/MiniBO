from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

from mini_bo.acq_functions import AcquisitionFunction
import os

cdict = {'red': ((0.0, 0.0, 0.0),
                  (0.5, 1.0, 0.7),
                  (1.0, 1.0, 1.0)),
          'green': ((0.0, 0.0, 0.0),
                    (0.5, 1.0, 0.0),
                    (1.0, 1.0, 1.0)),
          'blue': ((0.0, 0.0, 0.0),
                   (0.5, 1.0, 0.0),
                   (1.0, 0.5, 1.0))}
          

my_cmap = plt.get_cmap('Blues')

        
counter = 0
       
        
def plot_bo(bo):
    if bo.dim==1:
        plot_bo_1d(bo)
    if bo.dim==2:
        plot_bo_2d(bo)
    

def plot_bo_1d(bo):
    global counter
    counter=counter+1
    
    func=bo.f
    #x_original = np.linspace(bo.SearchSpace[0,0], bo.SearchSpace[0,1], 100)
    x = np.linspace(bo.scaleSearchSpace[0,0], bo.scaleSearchSpace[0,1], 1000)
    x_original=bo.Xscaler.inverse_transform(np.reshape(x,(-1,bo.dim)))

    y_original = func(x_original)
    #y = func(x)
    #y_original=mu*np.std(bo.Y_ori)+np.mean(bo.Y_ori)

    fig=plt.figure(figsize=(12, 5.5))
    fig.suptitle('Bayes Opt After {} Points'.format(len(bo.X)),fontsize=22)# fontdict={'size':22}
    
    gs = gridspec.GridSpec(4, 1, height_ratios=[0.05,3, 0.6,1]) 
    axis = plt.subplot(gs[1])
    acq = plt.subplot(gs[3])
    
    mu, sigma = bo.posterior(x)
    #mu_original=mu*(np.max(y_original)-np.min(y_original))+np.mean(y_original)
    mu_original=mu*np.std(bo.Y_ori)+np.mean(bo.Y_ori)
    sigma_original=sigma*np.std(bo.Y_ori)+np.mean(bo.Y_ori)#**2
    
    axis.plot(x_original, y_original, linewidth=3, label='Real Function')
    axis.plot(bo.X_ori.flatten(), bo.Y_ori, 'D', markersize=8, label=u'Observations', color='r')
    axis.plot(x_original, mu_original, '--', color='k', label='GP mean')
    
    temp_xaxis=np.concatenate([x_original, x_original[::-1]])
    #temp_xaxis=temp*bo.max_min_gap+bo.SearchSpace[:,0]
    
    #temp_yaxis_original=np.concatenate([mu_original - 1.9600 * sigma_original, (mu_original + 1.9600 * sigma_original)[::-1]])
    temp_yaxis=np.concatenate([mu - 1.9600 * sigma, (mu + 1.9600 * sigma)[::-1]])
    temp_yaxis_original2=temp_yaxis*np.std(bo.Y_ori)+np.mean(bo.Y_ori)
    axis.fill(temp_xaxis, temp_yaxis_original2,alpha=.6, fc='c', ec='None', label='95% CI')
    axis.set_title("Surrogate Function",fontsize=18)

    
    axis.set_xlim((np.min(x_original), np.max(x_original)))
    #axis.set_ylim((None, None))
    axis.set_ylabel('f(x)', fontdict={'size':16})
    axis.set_xlabel('x', fontdict={'size':16})

    myacq=AcquisitionFunction(acq_name=bo.acq_name)
    utility = myacq.acq_kind(bo.gp,x.reshape((-1, 1)))

    #utility = bo.acq_func.acq_kind(x.reshape((-1, 1)), bo.gp)
    acq.plot(x_original, utility, label='Utility Function', color='purple')
    acq.plot(x_original[np.argmax(utility)], np.max(utility), '*', markersize=15, 
             label=u'Suggested Point', markerfacecolor='gold', markeredgecolor='k', markeredgewidth=1)
  
    max_point=np.max(utility)
    acq.set_title("Acquisition Function",fontsize=18)
    #acq.plot(bo.X_ori[-1:], max_point.repeat(1), 'v', markersize=15, 
         #label=u'Previous Selection', markerfacecolor='green', markeredgecolor='k', markeredgewidth=1)
             
    acq.set_xlim((np.min(x_original), np.max(x_original)))
    acq.set_ylabel(r'$\alpha(x)$', fontdict={'size':16})
    acq.set_xlabel('x', fontdict={'size':16})
    
    axis.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.,fontsize=16)
    acq.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.,fontsize=16)
    
    #plt.legend(fontsize=14)

    strFileName="{:d}_GP_BO_1d.pdf".format(counter)
    #fig.savefig(strFileName, bbox_inches='tight')
    
    
def plot_bo_2d(bo):
    
    x1 = np.linspace(bo.scaleSearchSpace[0,0], bo.scaleSearchSpace[0,1], 60)
    x2 = np.linspace(bo.scaleSearchSpace[1,0], bo.scaleSearchSpace[1,1], 60)
    x1g,x2g=np.meshgrid(x1,x2)
    
    X=np.c_[x1g.flatten(), x2g.flatten()]
    
    x1_ori = np.linspace(bo.SearchSpace[0,0], bo.SearchSpace[0,1], 60)
    x2_ori = np.linspace(bo.SearchSpace[1,0], bo.SearchSpace[1,1], 60)    
    x1g_ori,x2g_ori=np.meshgrid(x1_ori,x2_ori)
    
    X_ori=np.c_[x1g_ori.flatten(), x2g_ori.flatten()]
  

    fig = plt.figure(figsize=(12,8))
    
    axis_mean2d = fig.add_subplot(2, 2, 1)
    axis_var2d = fig.add_subplot(2, 2, 2)
    acq2d = fig.add_subplot(2, 2, 3)
    
    mu, sigma = bo.posterior(X)


    # plot GP mean
    CS_acq_mean=axis_mean2d.contourf(x1g_ori,x2g_ori,mu.reshape(x1g.shape),cmap=my_cmap,origin='lower')
    #CS2_acq_mean = plt.contour(CS_acq_mean, levels=CS_acq_mean.levels[::2],colors='r',origin='lower',hold='on')
    axis_mean2d.scatter(bo.X_ori[:,0],bo.X_ori[:,1],color='r',label='Data')  
    axis_mean2d.set_title('GP Mean',fontsize=16)
    axis_mean2d.set_xlim(bo.SearchSpace[0,0], bo.SearchSpace[0,1])
    axis_mean2d.set_ylim(bo.SearchSpace[1,0], bo.SearchSpace[1,1])
    
    #acq2d.legend(loc=1, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    #axis_mean2d.legend(loc='center left',ncol=3,bbox_to_anchor=(0, -0.2))
      
    fig.colorbar(CS_acq_mean, ax=axis_mean2d, shrink=0.9)


    # plot GP variance
    CS_acq_var=axis_var2d.contourf(x1g_ori,x2g_ori,sigma.reshape(x1g.shape),cmap=my_cmap,origin='lower')
    #CS2_acq_var = plt.contour(CS_acq_var, levels=CS_acq_var.levels[::2],colors='r',origin='lower',hold='on')
    axis_var2d.scatter(bo.X_ori[:,0],bo.X_ori[:,1],color='red',label='Observations')  
    axis_var2d.set_title('GP Var',fontsize=16)
    axis_var2d.set_xlim(bo.SearchSpace[0,0], bo.SearchSpace[0,1])
    axis_var2d.set_ylim(bo.SearchSpace[1,0], bo.SearchSpace[1,1])
    axis_var2d.legend(loc='center left',ncol=3,bbox_to_anchor=(0, -0.2),fontsize=14)
      
    fig.colorbar(CS_acq_var, ax=axis_var2d, shrink=0.9)


    # plot the acquisition function

    #utility = bo.acq_func.acq_kind(X, bo.gp)
    myacq=AcquisitionFunction(acq_name=bo.acq_name)
	
    utility = myacq.acq_kind(bo.gp,X.reshape((-1, 1)))
    #acq3d.plot_surface(x1g,x1g,utility.reshape(x1g.shape))
    
    CS_acq=acq2d.contourf(x1g_ori,x2g_ori,utility.reshape(x1g.shape),cmap=my_cmap,origin='lower')
    #CS2_acq = plt.contour(CS_acq, levels=CS_acq.levels[::2],colors='r',origin='lower',hold='on')
    
    idxBest=np.argmax(utility)
  
    try:  
        B=-bo.B
    except:
        B=-1

    acq2d.scatter(bo.X_ori[:,0],bo.X_ori[:,1],color='r')  
    acq2d.scatter(bo.X_ori[B:,0],bo.X_ori[B:,1],marker='*', color='green',s=140,label='Suggested Points')
    #acq2d.scatter(X_ori[idxBest,0],X_ori[idxBest,1],marker='s',color='yellow',s=30,label='Next Best Guess')

    acq2d.set_title('Acquisition Function',fontsize=16)
    acq2d.set_xlim(bo.SearchSpace[0,0], bo.SearchSpace[0,1])
    acq2d.set_ylim(bo.SearchSpace[1,0], bo.SearchSpace[1,1])
    acq2d.legend(loc='center left',ncol=3,bbox_to_anchor=(0, -0.2),fontsize=14)
      
    fig.colorbar(CS_acq, ax=acq2d, shrink=0.9)

 
  
def plot_original_function(myfunction,X_ori=None,Y_ori=None):
    
    origin = 'lower'

    func=myfunction.func


    if myfunction.input_dim==1:    
        x = np.linspace(myfunction.bounds['x'][0], myfunction.bounds['x'][1], 1000)
        y = func(x)
    
        fig=plt.figure(figsize=(8, 5))
        plt.plot(x, y)
        strTitle="{:s}".format(myfunction.name)

        plt.title(strTitle)
    
    if myfunction.input_dim==2:    
        
        # Create an array with parameters bounds
        if isinstance(myfunction.bounds,dict):
            # Get the name of the parameters        
            bounds = []
            for key in myfunction.bounds.keys():
                bounds.append(myfunction.bounds[key])
            bounds = np.asarray(bounds)
        else:
            bounds=np.asarray(myfunction.bounds)
            
        x1 = np.linspace(bounds[0][0], bounds[0][1], 50)
        x2 = np.linspace(bounds[1][0], bounds[1][1], 50)
        x1g,x2g=np.meshgrid(x1,x2)
        X_plot=np.c_[x1g.flatten(), x2g.flatten()]
        Y = func(X_plot)
    
        #fig=plt.figure(figsize=(8, 5))
        
        #fig = plt.figure(figsize=(12, 3.5))
        fig = plt.figure(figsize=(14, 6))
        
        gs = gridspec.GridSpec(1, 2, width_ratios=[1.6,1]) 
        ax3d = plt.subplot(gs[0], projection='3d')
        ax2d = plt.subplot(gs[1])
  
        alpha = 0.7
        ax3d.plot_surface(x1g,x2g,Y.reshape(x1g.shape),cmap=my_cmap,alpha=alpha) 
        
        
        idxBest=np.argmax(Y)
        #idxBest=np.argmin(Y)
    
        ax3d.scatter(X_plot[idxBest,0],X_plot[idxBest,1],Y[idxBest],marker='*',color='r',s=200,label='Peak')
    
        if X_ori is not None:
            ax3d.scatter(X_ori[:,0],X_ori[:,1],Y_ori,marker='o',color='r',s=100,label='Observations')

        
        #mlab.view(azimuth=0, elevation=90, roll=-90+alpha)
        ax3d.set_xlabel("x1",fontsize=14)
        ax3d.set_ylabel("x2",fontsize=14)
        ax3d.set_zlabel("f(x)",fontsize=14)

        strTitle="{:s}".format(myfunction.name)
        #print strTitle
        ax3d.set_title(strTitle)
        #ax3d.view_init(40, 130)

        
        idxBest=np.argmax(Y)
        CS=ax2d.contourf(x1g,x2g,Y.reshape(x1g.shape),cmap=my_cmap,origin=origin)   
       
        #CS2 = plt.contour(CS, levels=CS.levels[::2],colors='r',origin=origin,hold='on')
        ax2d.scatter(X_plot[idxBest,0],X_plot[idxBest,1],marker='*',color='r',s=300,label='Peak')
        if X_ori is not None:
            ax2d.scatter(X_ori[:,0],X_ori[:,1],marker='o',color='r',s=100,label='Observations')

        plt.colorbar(CS, ax=ax2d, shrink=0.9)

        ax2d.set_xlabel("x1",fontsize=14)
        ax2d.set_ylabel("x2",fontsize=14)
        ax2d.set_title(strTitle)

        
    strFolder=""
    strFileName="{:s}.eps".format(myfunction.name)
    strPath=os.path.join(strFolder,strFileName)
    #fig.savefig(strPath, bbox_inches='tight')
  
