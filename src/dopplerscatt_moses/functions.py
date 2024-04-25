from scipy import ndimage
import numpy as np
import xarray as xr

def gaussian_filter(da, sigma = 5, truncate = 4):
    '''
    https://stackoverflow.com/questions/18697532/gaussian-filtering-a-image-with-nan-in-python
    '''
    ones = xr.ones_like(da)
    U = da.values
    V = U.copy()
    V[np.isnan(U)]=0
    VV = ndimage.gaussian_filter(V,sigma=sigma,truncate=truncate)
    
    W = 0*U+1
    W[np.isnan(U)]=0
    WW = ndimage.gaussian_filter(W,sigma=sigma,truncate=truncate)
    WW[WW==0] = np.nan
    
    Z = VV/WW
    Z[np.isnan(U)] = np.nan
    
    return Z*ones