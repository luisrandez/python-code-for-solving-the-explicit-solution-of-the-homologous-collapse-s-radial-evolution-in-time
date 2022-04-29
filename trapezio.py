    
import numpy as np

def trapezio(time_array, N_fft=16, eps=1e-4):
    """
    This function solves the transcendental equation 
    
    u + sin(u) = tau in [0, pi]
    
    by means of the composite trapezoidal rule. 
    
    Based on https://gist.github.com/oliverphilcox/559f086f1bf63b23d55c508b2f47bad3
    
    Args:
        time_array (np.ndarray): Array of times in units of the free-fall time.
        
    Keyword Args:
        N_fft (int): Number of FFT points, default: 32.
        eps (float): Small parameter used to define contours, default 1e-4.

    Returns:
        np.ndarray: solution of the transcendental equation z + np.sin(z) - time_array =0  
        
    """
    # Input checks
    assert N_fft>=1, "At least one FFT grid-point!"
    assert (eps>0)&(eps<0.1), "Epsilon parameter must be small "

# define the function
    
    def f(z, t):
        return z + np.sin(z) - t
# define an auxiliary function    
    def gg(x, t):
        return 1./f(np.pi/2.+(np.pi/2.-eps)*x, t)

# optimizaed code for solving the root of f(z,t) based in the composite trapezoidal rule con N_fft  points
    
    x_array = np.linspace(0, np.pi, N_fft+1)[:,np.newaxis]
    tablae = np.exp(1j*x_array)
    tablaD = gg(tablae, time_array[np.newaxis,:])*tablae
    tablaN = tablae[1:-1]*tablaD[1:-1]

    root = np.pi/2 + (np.pi/2 - eps)*np.real(tablaD[0] - tablaD[N_fft] + 2*sum( tablaN[0:N_fft]))\
         /np.real(tablaD[0] + tablaD[N_fft] + 2*sum( tablaD[1:N_fft]) )      
    return root

#
# print(trapezio(np.array([np.pi/4,np.pi/6,np.pi/8,np.pi/16]), 16, 1e-4))
#
# [0.39790776 0.26331554 0.19698528 0.09825378]
#
