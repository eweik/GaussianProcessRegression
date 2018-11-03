#
#
# Edison Weik
# Gaussian Process Regression
# gp_regression.py
#
#


import numpy as np
import matplotlib.pyplot as plt
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


# Generate samples
def get_data( f, x_max, sample_size ):
    rng = np.random.RandomState(1)
    X = x_max * rng.rand(sample_size, 1).ravel()
    X[: int(sample_size/2) ] *= -1
    Y = np.array( [ f(x) for x in X ] )
    dY = 0.1 + 0.2 * np.random.random(Y.shape)
    Y += np.random.normal(0, dY)
    return X, Y


def GP_regression(X, Y, k, noise, x_hat):
    '''
    Algorithm taken from Rasmussen & Williams book
    _Gaussian Processes for Machine Learning_
    (algorithm 2.1 on page 19)
    '''
    n = len(X)
    K = variance( X, k )
    #w, v = np.linalg.eig(K+ noise*np.identity(n))
    #L = np.matmul( v, np.sqrt(w) )
    L = np.linalg.cholesky( K + noise*np.identity(n) )
    
    # predictive mean
    alpha = np.linalg.solve( np.transpose(L), np.linalg.solve( L, Y ) )
    k_hat = covariance( X, x_hat, k )
    mu_hat = np.matmul( np.transpose(k_hat), alpha )
    
    # predictive variance
    v = np.linalg.solve( L, k_hat )
    k_hathat = variance( x_hat, k )
    a = np.matmul( np.transpose(v), v )
    covar_hat = k_hathat - np.matmul( np.transpose(v), v )
    var_hat = covar_hat.diagonal()
    
    # log marginal likelihood
    logML = -0.5 * np.matmul( np.transpose(v), alpha ) - np.matrix.trace(L) - (n/2)*np.log10(2*np.pi)

    return mu_hat, np.sqrt(var_hat), logML


    

def plot_predictions(X, Y, X_test, Y_hat, sigma, f, i):
    plt.plot(X, Y, 'k.', label='Observations')
    #plt.plot(X_test, f(X_test), 'r:', label='Actual $f(x)= \mathrm{sin} (x)$')
    plt.plot(X_test, Y_hat, 'b-', label='Prediction')
    plt.fill(np.concatenate([X_test, X_test[::-1]]), \
             np.concatenate([Y_hat - 1.96 * sigma, (Y_hat + 1.96 * sigma)[::-1]]), \
             alpha = 0.5, fc = 'b', ec = 'None', label='95% confidence interval')
    
    describe_plot(i)
 
 
def describe_plot(i):
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
    plt.ylim(-3, 3)
    #plt.title('$\mathcal{GP}(0, x_1 x_2)$')
    #plt.title('$\mathcal{GP}(0, \mathrm{min}(x_1, x_2)$)')
    plt.title('$\mathcal{GP}(0, \mathrm{exp}(-|x_1 - x_2|^2))$, $n = $'+str(i))
    #plt.title('$\mathcal{GP}(0, \mathrm{exp}(-\mathrm{sin}^2( \pi(x_1 - x_2)) ))$')
    #plt.title('$\mathcal{GP}(0, \mathrm{exp}(-\mathrm{min}^2( |x_1 - x_2|, |x_1 + x_2| ))$')
    plt.legend(loc="upper left")
    plt.savefig('images/gpr/gpr_'+str(i)+'.pdf') 
    plt.close()
    

def evolution():
    f = lambda x: np.sin(x)
    x_hat = np.linspace(-10, 10, 2000)
    indices = np.arange(50)
    np.random.shuffle( indices )
    X_data, Y_data = get_data( f, 5, 50 )
    X_data = X_data[indices]
    Y_data = Y_data[indices]
    for i in [1,2,3,4,5,6,7,8,9,10,15,20,25,30,35,40,45,50]:
        X = X_data[:i]
        Y = Y_data[:i]
        y_hat, sigma, logML = GP_regression(X, Y, k3, 0.1, x_hat)
        plot_predictions(X, Y, x_hat, y_hat, sigma, f, i)
        print(i)
        
 
 
if __name__ == "__main__":
    # range
    X_min = -5
    X_max = 5
    
    f = lambda x: np.sin(x)
    X, Y = get_data( f, X_max, 50 )
    x_hat = np.linspace(-10, 10, 2000)
    y_hat, sigma, logML = GP_regression(X, Y, k3, 0.05, x_hat)
    #y_hat = np.zeros(2000)
    #sigma = np.ones(2000)
    plot_predictions(X, Y, x_hat, y_hat, sigma, f)
    

    
    
    