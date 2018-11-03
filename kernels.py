# Edison Weik
# Kernel functions
# kernel.py

import numpy as np

# KERNEL FUNCTIONS
# k7: symmetric
# k6: periodic
# k5: ornstein-uhlenbeck (range must be positive)
# k4: brownian motion (range must be positive)
# k3: squared exponential 
# k2: linear
# k1: simple number

    
# symmetric process
def k7(xi, xj):
    x = np.minimum( np.absolute(xi-xj), np.absolute(xi+xj) )
    return np.exp( -1 * np.square(x) / 0.25 )

# periodic process
def k6(xi, xj):
    sine = np.sin( 0.5 * np.pi * (xi - xj) )
    return np.exp( -1 * np.square(sine) )

# orstein-uhlenbeck process (brownian velocity)
def k5(xi, xj):
    abs_diff = np.absolute( xi - xj )
    return np.exp( -1 * abs_diff )    

# brownian motion
def k4(xi, xj):
    return np.minimum( xi, xj )

# squared-exponential process (smooth)
def k3(xi, xj):
    sq_diff = np.square( np.absolute( xi - xj ) )
    return np.exp( -0.5 * sq_diff / 4  )
    
# linear process
def k2(xi, xj):
    return 1 * np.dot( xi, xj )
    
# number
def k1(xi, xj):
    return 1;
    