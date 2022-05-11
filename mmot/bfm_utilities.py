import numpy as np
from scipy.fftpack import dctn, idctn
from w2 import BFM

from scipy.interpolate import interp2d
from scipy.interpolate import LinearNDInterpolator

import matplotlib.pyplot as plt 
from shapely.geometry import box, Polygon

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

def hessian(f):
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
    dx = 1.0/nx 

    Hyy = np.zeros((nx,ny))
    Hyy[1:-1,:] = (f[2:,:] - 2.0*f[1:-1,:] + f[0:-2,:])/(dy*dy)
    Hyy[0,:] = (f[2,:] - f[1,:])/(dx*dx)
    Hyy[-1,:] = -(f[-1,:] - f[-2,:])/(dx*dx)
     
    Hyx = np.zeros((nx,ny))
    Hyx[1:-1,1:-1] = (0.25/(dx*dy))*(f[2:,2:] - f[0:-2,2:] - f[2:,0:-2] + f[0:-2,0:-2])
    
    Hxx = np.zeros((nx,ny))
    Hxx[:,1:-1] = (f[:,2:] - 2.0*f[:,1:-1] + f[:,0:-2])/(dx*dx)
    Hxx[:,1] = (f[:,2] - f[:,1])/(dx*dx)
    Hxx[:,-1] = -(f[:,-1] - f[:,-2])/(dx*dx)
    
    return Hxx, Hyy, Hyx


def push_forward2(dualVar, src_dens, X1, X2, weight=1.0):

    ny,nx = dualVar.shape 

    gradx, grady = gradient(dualVar/weight)

    newx = X1 - gradx 
    newy = X2 - grady

    hessxx, hessyy, hessyx = hessian(dualVar/weight)

    nugget = 1e-6
    jacdet = (1.0-hessxx + nugget)*(1.0-hessyy + nugget) - hessyx*hessyx
    tgt_dens = src_dens / np.abs(jacdet)

    interp = LinearNDInterpolator(list(zip(newx.ravel(), newy.ravel())), tgt_dens.ravel())
    
    tgt_dens_interp = interp(X1,X2)
    tgt_dens_interp[np.isnan(tgt_dens_interp)] = 0.0
    tgt_dens_interp *= (nx*ny) / np.sum(tgt_dens_interp)

    return tgt_dens_interp

# @jit
# def clip_to_box(poly, bbox):
#     """ Clips a polygon to a bounding box.  Returns the corners of the clipped polygon.

#     Parameters
#     ----------
#         poly :  list of list of float
#             list of 2d points defining the polygon.   Points must be in CCW order.
#         bbox: list of float
#             xmin, xmax, ymin, ymax defining axis-aligned bounding box

#     Returns
#     ----------
#         list of list of float  
#             Corner points defining the clipped polygon.
#     """

#     # Clip based on left edge of box 
#     in_poly = np.copy(poly)
#     out_poly = []

#     for i in range(len(in_poly)):
#         edge_start = in_poly[i-1]
#         edge_end = in_poly[i]
        
#         if(edge_start[0]>=bbox[0]):
#             if(edge_end[0]>=bbox[0]):
#                 # Entire edge is entirely on the correct side of the halfplane
#                 out_poly.append(edge_end)
#             else:
#                 # Edge starts in halfplane, but finishes outside 
#                 # bbox = edge_start + w*(edge_end-edge_start)
#                 w = (bbox[0]-edge_start[0])/(edge_end[0]-edge_start[0])
#                 intersection = [bbox[0], edge_start[1] + w*(edge_end[1]-edge_start[1])]
#                 out_poly.append(intersection)
#         else:
#             if(edge_end[0]>=bbox[0]):
#                 # Edge starts outside halfplane, but finished inside
#                 w = (bbox[0]-edge_start[0])/(edge_end[0]-edge_start[0])
#                 intersection = [bbox[0], edge_start[1] + w*(edge_end[1]-edge_start[1])]
#                 out_poly.append(intersection)
#                 out_poly.append(edge_end)
#             else:
#                 # Edge is entirely outside halfplane
#                 pass
    
#     # Clip based on right edge of box 
#     in_poly = np.copy(out_poly)
#     out_poly = []

#     for i in range(len(in_poly)):
#         edge_start = in_poly[i-1]
#         edge_end = in_poly[i]
        
#         if(edge_start[0]<=bbox[1]):
#             if(edge_end[0]<=bbox[1]):
#                 # Entire edge is entirely on the correct side of the halfplane
#                 out_poly.append(edge_end)
#             else:
#                 # Edge starts in halfplane, but finishes outside 
#                 # bbox = edge_start + w*(edge_end-edge_start)
#                 w = (bbox[1]-edge_start[0])/(edge_end[0]-edge_start[0])
#                 intersection = [bbox[1], edge_start[1] + w*(edge_end[1]-edge_start[1])]
#                 out_poly.append(intersection)
#         else:
#             if(edge_end[0]<=bbox[1]):
#                 # Edge starts outside halfplane, but finished inside
#                 w = (bbox[1]-edge_start[0])/(edge_end[0]-edge_start[0])
#                 intersection = [bbox[1], edge_start[1] + w*(edge_end[1]-edge_start[1])]
#                 out_poly.append(intersection)
#                 out_poly.append(edge_end)
#             else:
#                 # Edge is entirely outside halfplane
#                 pass
    
#     # Clip based on bottom edge of box 
#     in_poly = np.copy(out_poly)
#     out_poly = []

#     for i in range(len(in_poly)):
#         edge_start = in_poly[i-1]
#         edge_end = in_poly[i]
        
#         if(edge_start[1]>=bbox[2]):
#             if(edge_end[1]>=bbox[2]):
#                 # Entire edge is entirely on the correct side of the halfplane
#                 out_poly.append(edge_end)
#             else:
#                 # Edge starts in halfplane, but finishes outside 
#                 # bbox = edge_start + w*(edge_end-edge_start)
#                 w = (bbox[2]-edge_start[1])/(edge_end[1]-edge_start[1])
#                 intersection = [edge_start[0] + w*(edge_end[0]-edge_start[0]), bbox[2]]
#                 out_poly.append(intersection)
#         else:
#             if(edge_end[1]>=bbox[2]):
#                 # Edge starts outside halfplane, but finished inside
#                 w = (bbox[2]-edge_start[1])/(edge_end[1]-edge_start[1])
#                 intersection = [edge_start[0] + w*(edge_end[0]-edge_start[0]), bbox[2]]
#                 out_poly.append(intersection)
#                 out_poly.append(edge_end)
#             else:
#                 # Edge is entirely outside halfplane
#                 pass

#     # Clip based on top edge of box 
#     in_poly = np.copy(out_poly)
#     out_poly = []

#     for i in range(len(in_poly)):
#         edge_start = in_poly[i-1]
#         edge_end = in_poly[i]
        
#         if(edge_start[1]<=bbox[3]):
#             if(edge_end[1]<=bbox[3]):
#                 # Entire edge is entirely on the correct side of the halfplane
#                 out_poly.append(edge_end)
#             else:
#                 # Edge starts in halfplane, but finishes outside 
#                 # bbox = edge_start + w*(edge_end-edge_start)
#                 w = (bbox[3]-edge_start[1])/(edge_end[1]-edge_start[1])
#                 intersection = [edge_start[0] + w*(edge_end[0]-edge_start[0]), bbox[3]]
#                 out_poly.append(intersection)
#         else:
#             if(edge_end[1]<=bbox[3]):
#                 # Edge starts outside halfplane, but finished inside
#                 w = (bbox[3]-edge_start[1])/(edge_end[1]-edge_start[1])
#                 intersection = [edge_start[0] + w*(edge_end[0]-edge_start[0]), bbox[3]]
#                 out_poly.append(intersection)
#                 out_poly.append(edge_end)
#             else:
#                 # Edge is entirely outside halfplane
#                 pass

#     return out_poly 

# @jit
# def polygon_area(poly):
#     """ Uses the "shoelace" algorithm to compute the area of a polygon.

#     Parameters
#     ----------
#         poly :  list of list of float
#             list of 2d points defining the polygon.   Points must be in CCW order.

#     Retruns
#     ---------
#         float
#             The are aof the polygon.
#     """

# def push_forward3(dualVar, src_dens, X1, X2, weight=1.0):

#     ny,nx = dualVar.shape 
#     dy = 1.0/ny 
#     dx = 1.0/nx 

#     gradx, grady = gradient(dualVar/weight)

#     newx = X1 - gradx 
#     newy = X2 - grady

#     output = np.zeros(src_dens.shape)

#     for i in range(ny):
#         for j in range(nx):
            
#             if(src_dens[i,j]>1e-15):
#                 # Bilinear interpolation to get position of corners 
#                 # bottom left corner
#                 if((j==0)|(i==0)):
#                     x_bl = j*dx
#                     y_bl = i*dy
#                 else:
#                     x_bl = 0.25*(newx[i,j] + newx[i-1,j] + newx[i,j-1] + newx[i-1,j-1])
#                     y_bl = 0.25*(newy[i,j] + newy[i-1,j] + newy[i,j-1] + newy[i-1,j-1])

#                 # bottom right corner 
#                 if((j==nx-1)|(i==0)):   
#                     x_br = (j+1)*dx 
#                     y_br = i*dy 
#                 else:
#                     x_br = 0.25*(newx[i,j] + newx[i-1,j] + newx[i,j+1] + newx[i-1,j+1])
#                     y_br = 0.25*(newy[i,j] + newy[i-1,j] + newy[i,j+1] + newy[i-1,j+1])

#                 # upper right corner
#                 if((j==nx-1) | (i==ny-1)):
#                     x_ur = (j+1)*dx 
#                     y_ur = (i+1)*dy
#                 else:
#                     x_ur = 0.25*(newx[i,j] + newx[i+1,j] + newx[i,j+1] + newx[i+1,j+1])
#                     y_ur = 0.25*(newy[i,j] + newy[i+1,j] + newy[i,j+1] + newy[i+1,j+1])
                
#                 # upper left corner 
#                 if((j==0)|(i==ny-1)):
#                     x_ul = j*dx 
#                     y_ul = (i+1)*dy
#                 else:
#                     x_ul = 0.25*(newx[i,j] + newx[i+1,j] + newx[i,j-1] + newx[i+1,j-1])
#                     y_ul = 0.25*(newy[i,j] + newy[i+1,j] + newy[i,j-1] + newy[i+1,j-1])
                    
#                 # Grid cells in the original uniform discretization that could part of the mapped grid cell 
#                 corner_is = [ int(np.floor(y_bl/dy)), int(np.floor(y_br/dy)), int(np.floor(y_ur/dy)), int(np.floor(y_ul/dy))]
#                 imin = np.min(corner_is)
#                 imax = np.max(corner_is)
                
#                 corner_js = [ int(np.floor(x_bl/dx)), int(np.floor(x_br/dx)), int(np.floor(x_ur/dx)), int(np.floor(x_ul/dx))]
#                 jmin = np.min(corner_js)
#                 jmax = np.max(corner_js)
                
#                 tri1 = Polygon([(x_bl, y_bl), (x_ur, y_ur), (x_ul, y_ul)])
#                 tri1_area = tri1.area 

#                 tri2 = Polygon([(x_bl, y_bl), (x_br, y_br), (x_ur, y_ur)])
#                 tri2_area = tri2.area 

#                 for inew in range(imin,imax+1):
#                     for jnew in range(jmin,jmax+1):
#                         iadd = np.max([np.min([inew,ny-1]), 0])
#                         jadd = np.max([np.min([jnew,nx-1]), 0])
                        
#                         cell = box(jnew*dx, inew*dy, (jnew+1)*dx, (inew+1)*dy)
#                         if((tri1_area>1e-10)&(tri2_area>1e-10)):
#                             output[iadd,jadd] += 0.5*src_dens[i,j]*tri1.intersection(cell).area / tri1_area
#                             output[iadd,jadd] += 0.5*src_dens[i,j]*tri2.intersection(cell).area / tri2_area  

#     output *= np.prod(output.shape) / np.sum(output)
#     return output
    




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