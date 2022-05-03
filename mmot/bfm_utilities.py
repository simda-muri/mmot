import numpy as np
from scipy.fftpack import dctn, idctn
from w2 import BFM

from scipy.interpolate import interp2d
from scipy.interpolate import LinearNDInterpolator

import matplotlib.pyplot as plt 
from shapely.geometry import Polygon

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
  
def c_transform(bf, dualVar,x,y, weight=1.0):
    """ Returns the c transform of the Kantorovich dual variable.  

    First, the dual variable is converted to convex form and then the bf module's
    ctransform function is employed.
    """

    phi = convex_conversion(dualVar/weight,x,y)
    temp = leg_transform(bf,phi)
    
    return convex_conversion(temp,x,y) * weight

def gradient(f):
    """
    Computes the gradient of f using finite differences.  Assumes the gradient at the boundaries is
    zero and that values of f are defined at cell centers.

    ARGUMENTS:
        f (np.array) : A function defined at the pixel centers

    RETURNS:
        gradx (np.array) : A finite difference approximation of the derivative of f in the x direction.
        grady (np.array) : A finite difference approximation of the  derivative of f in the y direction.
    """

    assert(len(f.shape)==2)

    ny,nx = f.shape 

    # Compute the gradient of the dual variable.  Assume gradient at boundaries is zero
    dy = 1.0/ny
    grady = np.zeros(f.shape)
    grady[1:-1,:] = (f[2:,:] - f[0:-2,:])/(2.0*dy) # Interior nodes gradient
    grady[0,:] = 0.5*(f[1,:]-f[0,:])/dy # Bottom row 
    grady[-1,:] = 0.5*(f[-1,:]-f[-2,:])/dy # Top row 
    
    dx = 1.0/nx
    gradx = np.zeros(f.shape)
    gradx[:,1:-1] = (f[:,2:] - f[:,0:-2])/(2.0*dx) # Interior nodes gradient
    gradx[:,0] = 0.5*(f[:,1]-f[:,0])/dx # Left 
    gradx[:,-1] = 0.5*(f[:,-1]-f[:,-2])/dx # Top row 

    return gradx, grady


def push_forward2(dualVar, src_dens, X1, X2, weight=1.0):

    ny,nx = dualVar.shape 

    gradx, grady = gradient(dualVar/weight)

    newx = X1 - gradx 
    newy = X2 - grady

    hessxx, hessxy = gradient(gradx)
    hessyx, hessyy = gradient(grady)

    nugget = 1e-8
    jacdet = (1.0-hessxx + nugget)*(1.0-hessyy + nugget) - hessxy*hessyx
    tgt_dens = src_dens / np.abs(jacdet)

    interp = LinearNDInterpolator(list(zip(newx.ravel(), newy.ravel())), tgt_dens.ravel())
    
    tgt_dens_interp = interp(X1,X2)
    tgt_dens_interp[np.isnan(tgt_dens_interp)] = 0.0
    tgt_dens_interp *= (nx*ny) / np.sum(tgt_dens_interp)

    return tgt_dens_interp

    




def push_forward(bf, dualVar, marginal, x, y, weight=1.0):
  """ Computes the push forward of the marginal :math:`\mu` given the dual variable :math:`f`.

    .. math::

        c(x,y) = h(y-x) = \frac{w}{2}\|x-y\|^2

    .. math::

        S(x) = x - (\nabla h)^{-1} (\nabla g)

    The BFM code evaluates 

    .. math::

        \left(\nabla \phi)_\sharp \mu 

    So to evaluate S, we need 

    .. math::

        x - (\nabla h)^{-1} \circ (\nabla g) (x) = (I - w^{-1} (\nabla g))(x)  = \nabla( \frac{1}{2}\|x\|^2 - w^{-1}g)
        

  """
  # Convert to convex form, which is what the bf module expects
  phi = convex_conversion(dualVar/weight, x,y)
 
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

    # fig,axs = plt.subplots(ncols=2)
    # axs[0].imshow(rho)
    # axs[1].imshow(workspace)

    
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