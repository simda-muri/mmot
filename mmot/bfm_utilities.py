import numpy as np
from scipy.fftpack import dctn, idctn
from w2 import BFM

import matplotlib.pyplot as plt 

# Initialize Fourier kernel
def initialize_kernel(n1, n2):
    xx, yy = np.meshgrid(np.linspace(0,np.pi,n1,False), np.linspace(0,np.pi,n2,False))
    kernel = 2*n1*n1*(1-np.cos(xx)) + 2*n2*n2*(1-np.cos(yy))
    kernel[0,0] = 1     # to avoid dividing by zero
    return kernel

# 2d DCT
def dct2(a):
    return dctn(a, norm='ortho')
    
# 2d IDCT
def idct2(a):
    return idctn(a, norm='ortho')

def leg_transform(bf, phi):
    """ Returns the legendre transform of the convex dual variable. """

    output = np.zeros(phi.shape)
    bf.ctransform(output, phi) # <- This should compute the c-transform of the potential in convex form

    return output

def convex_conversion(g,x,y):
  """ 
  Given the dual variable g(x) evaluated at the points x, this function returns 
  the ϕ(x) = ½|x|² - g(x)
  """
  return 0.5*(x*x+y*y) - g
  
def c_transform(bf, dualVar,x,y):
    """ Returns the c transform of the Kantorovich dual variable.  

    First, the dual variable is converted to convex form and then the bf module's
    ctransform function is employed.
    """
    output = np.zeros(dualVar.shape)

    phi = convex_conversion(dualVar,x,y)
    output = leg_transform(bf,phi)
    output = convex_conversion(output,x,y)

    return output

def push_forward(bf, dualVar, marginal, x, y):
  
  # Convert to convex form, which is what the bf module expects
  phi = convex_conversion(dualVar, x,y)
 
  # Push forward
  output = np.zeros(marginal.shape)  
  bf.pushforward(output, phi, marginal)

  return output

def update_potential(f, rho, nu, kernel, sigma):
    """ 
      Update f as 
          f ← f + σ Δ⁻¹(ρ − ν) = f - σ Δ⁻¹(ν - ρ)
      and return the error 
          ∫(−Δ)⁻¹(ρ−ν) (ρ−ν)
      Modifies phi and rho
    """
    
    n1, n2 = nu.shape

    rho -= nu
    workspace = dct2(rho) / kernel
    workspace[0,0] = 0
    workspace = idct2(workspace)
    
    f += sigma * workspace
    h1 = np.sum(workspace * rho) / (n1*n2)
    
    return h1
    
def ascent_direction(rho,nu,kernel):
    """
    Computes the ascent direction 
    
    g = Δ⁻¹(ρ − ν) 

    and the squared norm 
    
    |g|^2

    This function does not change the values of rho and nu
    """

    workspace = dct2(rho-nu) / kernel
    workspace[0,0] = 0
    grad = idct2(workspace)

    gradSqNorm = np.sum(grad * rho) / np.prod(nu.shape)

    return grad, gradSqNorm