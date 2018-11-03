#
#
# Edison Weik
# Sampling from a Gaussian Process
# gp_sample.py
#
# 
# Sample and plot functions from Gaussian Processes defined by 
# different kernels.
#

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy.interpolate import spline
from variance import *
from kernels import *



# KERNEL FUNCTIONS
# k7: symmetric
# k6: periodic
# k5: ornstein-uhlenbeck (range must be positive)
# k4: brownian motion (range must be positive)
# k3: squared exponential 
# k2: linear
# k1: simple number
    
    
X_min = -5
X_max = 5
    
def get_prior_sample( sample_size, kernel ):
    # get covariance matrix
    X = np.linspace(X_min, X_max, num=sample_size)
    K = variance( X, kernel )
    
    # generate random Gaussian vector with this covariance matrix
    u = np.random.normal( size = sample_size )
    w, v = np.linalg.eig(K)
    return X, np.matmul( v, np.matmul( np.sqrt(w)*np.identity(len(w)), u))
    
    
def describe_sample_plot():
    plt.ylabel('$f(x)$')
    plt.xlabel('$x$')
    #plt.title('$f(x) \sim \mathcal{GP}(0, \mathrm{exp}(-\mathrm{sin}^2( \pi(x_1 - x_2)) ))$')
    plt.title('$f(x) \sim \mathcal{GP}(0, \mathrm{exp}(-\mathrm{min}^2( |x_1 - x_2|, |x_1 + x_2|) ))$')
    #plt.title('$f(x) \sim \mathcal{GP} (0,  \mathrm{exp}( -0.5 |x_1 - x_2|^2 ) )$')
    #plt.title('$f(x) \sim \mathcal{GP} (0, x_1 x_2)$')
    #plt.title('$f(x) \sim \mathcal{GP} (0, 1)$')
    plt.ylim(-5, 5)
    #plt.show()
    plt.savefig("images/gp/gp_symmetric_sample.png")


def plot_sample(samples, kernel):  
    for _ in range(samples):
        X, Y = get_prior_sample(100, kernel)
        Xnew = np.linspace(X.min(), X.max(), 300)
        smooth = spline(X, Y, Xnew)
        plt.plot(Xnew, smooth, c=np.random.rand(3,))
        #plt.plot(X, Y, c=np.random.rand(3,))
    describe_sample_plot()


def plot_normal(mean, variance):
    sigma = np.sqrt(variance)
    x = np.linspace( mean-3*sigma, mean+3*sigma, 500)
    plt.plot(x, mlab.normpdf(x, mean, sigma))
    plt.xlim(-10, 10)
    plt.ylim(0, 0.5)
    plt.title(' $ x \sim \mathcal{N}(0,1) $ ')
    plt.xlabel(' $ x $ ')
    plt.ylabel(' p$(x)$ ')
    plt.savefig("images/normal1.png")
    
    
    
if __name__ == "__main__":
    plot_sample(20, k7)
    #plot_normal(0, 1)
    
    
    
    