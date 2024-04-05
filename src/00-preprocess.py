import os
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from glob import glob
from datetime import datetime
from tqdm import tqdm
from xhistogram.xarray import histogram
from pyproj import Transformer
from scipy.ndimage import gaussian_filter
import gsw
import matplotlib.colors as colors
from oa import scaloa
from scipy.interpolate import griddata

import warnings

import dask
from dask.distributed import Client, LocalCluster
cluster = LocalCluster(n_workers = 20, dashboard_address=':4242') #number of cores
client = Client(cluster) # go check dask dashboard

wgs2utm = Transformer.from_crs("EPSG:4326", "EPSG:32610").transform # x, y = f(lat, lon)

dt0 = datetime(2022,10,24)

campaign = "IOP1"

fname = glob(f"../data/external/{campaign}/dopplerscatt/*{dt0.strftime('%Y%m%d')}*")[0]
dscatt = xr.open_dataset(fname)
dscatt = dscatt.where(dscatt.azimuth_diversity_flag_all_lines==0)


def filter(U, sigma = 5, truncate = 4):

    '''
    https://stackoverflow.com/questions/18697532/gaussian-filtering-a-image-with-nan-in-python
    '''
    V=U.copy()
    V[np.isnan(U)]=0
    VV=gaussian_filter(V,sigma=sigma,truncate=truncate)
    
    W=0*U+1
    W[np.isnan(U)]=0
    WW=gaussian_filter(W,sigma=sigma,truncate=truncate)
    
    Z=VV/WW
    Z[np.isnan(U)]=np.nan
    return Z


dscatt["u_current_all_lines"] = xr.ones_like(dscatt["u_current_all_lines"])*filter(dscatt["u_current_all_lines"].values)
dscatt["v_current_all_lines"] = xr.ones_like(dscatt["v_current_all_lines"])*filter(dscatt["v_current_all_lines"].values)

f = gsw.f(dscatt.latitude.mean())
Ro = (
    dscatt["v_current_all_lines"].differentiate("x") - 
    dscatt["u_current_all_lines"].differentiate("y")
).rename("Ro") / f

dscatt["Ro"] = Ro

dscatt["wind_u"] = dscatt.wind_speed_all_lines * np.sin(np.deg2rad(dscatt.wind_dir_all_lines))

dscatt["wind_v"] = dscatt.wind_speed_all_lines * np.cos(np.deg2rad(dscatt.wind_dir_all_lines))
dscatt["current_speed"] = np.sqrt(dscatt["u_current_all_lines"]**2 + dscatt["v_current_all_lines"]**2)


def second_order_polynomial(coords, a, b, c, d, e, f):
    x, y = coords
    return f*x**2 + e*y**2 + d*x*y + c*x + b*y + a


for var in ["u_current_all_lines", "v_current_all_lines"]:
    params = dscatt[var].curvefit(func=second_order_polynomial,
                             param_names=['a', 'b', 'c', 'd', 'e', 'f'],
                             coords=['x', 'y'])
    
    fit = second_order_polynomial([dscatt.x, dscatt.y], *params.curvefit_coefficients.values)
    fit = fit.where(~np.isnan(dscatt[var])).rename(f"mean_{var}")
    dscatt[f"mean_{var}"] = fit

output_fname = f"../data/processed/{campaign}__dopplerscatt_{dt0.strftime('%Y%m%d')}.nc"

dscatt["x"] = dscatt["x"] * 1e-3
dscatt["y"] = dscatt["y"] * 1e-3

dscatt.to_netcdf(output_fname)




fnames = glob(f"../data/external/{campaign}/moses/*{dt0.strftime('%Y%m%d')}*")
fnames.sort()


H = []
for fname in tqdm(fnames):

    ds = xr.open_dataset(fname)
    ds["SST"] = ds["SST"] + 9.8 # bias from Sentinel
    
    x, y = wgs2utm(ds.latitude, ds.longitude)
    ds = ds.assign(
        x = (xr.ones_like(ds.longitude)*x),
        y = (xr.ones_like(ds.latitude)*y),
    )

    xm = (ds.x.where(~np.isnan(ds.SST))).mean("across")
    ym = (ds.y.where(~np.isnan(ds.SST))).mean("across")

    median = ds.SST.median("across")
    
    ds = ds.chunk(along = 1000, across = 100)
    
    dist = np.sqrt((ds.x-xm)**2 + (ds.y-ym)**2).rename("dist")*1e-3
    
    bins = [np.arange(-0.5, 0.3, 0.01), np.arange(0, 2.5, 0.05)]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        Hi = histogram(
            (ds.SST-median),
            dist, bins = bins).load()
        
    H.append(Hi)

H = xr.concat(H, "passing").sum("passing")
Hc = (H.cumsum("SST_bin")/H.sum("SST_bin"))



fig, ax = plt.subplots()
(H*1e-5).plot(ax = ax, cbar_kwargs = dict(label = "histogram [10$^5$ obs]"))
C = Hc.plot.contour(ax = ax, levels = [0.5], colors = ["r"])
ax.plot([], [], "r", label = "median")
ax.legend()
ax.set(xlabel = "cross-swath distance [km]", ylabel = "SST bias [$^\circ$C]")
fig.savefig("../img/moses_sst_bias.png", dpi = 300, bbox_inches = "tight")

dist, ta = C.allsegs[0][0].T
bias = xr.DataArray(ta, dims = ["dist"], coords = dict(dist = ("dist", dist)))



dx = 30
bins = [
    np.arange(dscatt.x.min(), dscatt.x.max(), dx),
    np.arange(dscatt.y.min(), dscatt.y.max(), dx)
]

threshold = 0.2 #seconds per measurements

H = []
T = []

for fname in tqdm(fnames):

    ds = xr.open_dataset(fname)
    ds["SST"] = ds.SST + 9.8 # Sentinel bias

    time = (ds.time.astype("float64").where(~np.isnan(ds.time)).mean("across")*1e-9)
    time = time-time.min()
    time = time.rolling(along = 11, min_periods = 1).mean()
    
    x, y = wgs2utm(ds.latitude, ds.longitude)
    ds = ds.assign(
        x = (xr.ones_like(ds.longitude)*x),
        y = (xr.ones_like(ds.latitude)*y),
    )

    ds["SST"] = ds.SST.where(np.abs(time.differentiate("along"))<threshold)
    
    xm = (ds.x.where(~np.isnan(ds.SST))).mean("across")
    ym = (ds.y.where(~np.isnan(ds.SST))).mean("across")
    
    dist = np.sqrt((ds.x-xm)**2 + (ds.y-ym)**2).rename("dist")*1e-3

    bias_i = bias.interp(dist = dist, method="linear", kwargs={"fill_value": "extrapolate"})

    ds["SST"] = ds["SST"] - bias_i

    # ds["SST"] = ds.SST.where(dist<2.4)
    
    ds = ds.chunk(along = 1000, across = 200)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        Hi = histogram(ds.x, ds.y, bins = bins, weights = ~np.isnan(ds.SST)).load()
        Ti = histogram(ds.x, ds.y, bins = bins, weights = ds.SST.fillna(0)).load() / Hi
    
    H.append(Hi)
    T.append(Ti)
    
H = xr.concat(H, "lines")
T = xr.concat(T, "lines")

moses = ((T*H).sum("lines")/H.sum("lines")).rename("SST")

moses["x_bin"] = moses["x_bin"] * 1e-3
moses["y_bin"] = moses["y_bin"] * 1e-3


output_fname = f"../data/processed/{campaign}__moses_{dt0.strftime('%Y%m%d')}_binned.nc"

moses.to_netcdf(output_fname)






# dT = np.sqrt(moses.differentiate("y_bin")**2 + moses.differentiate("x_bin")**2)

# dT.rolling(x_bin = 31, y_bin = 31, min_periods = 1).mean().plot(x = "x_bin", xlim = [320, 400], ylim = [4060, 4090], robust = True, cmap = "bone_r")

# fig, ax = plt.subplots(figsize = (8,4))
# moses.plot(x = "x_bin", robust = True, cmap = "Spectral_r")

# np.abs(dTdy).plot(x = "x_bin", robust = True, cmap = "bone_r", alpha = 0.3, add_colorbar = False)

# ax.axis("scaled")
# ax.set(xlim = [320, 400], ylim = [4060, 4090])

# fig.savefig("../img/moses_sst_test.png", dpi = 300)

# output_fname = f"../data/processed/{campaign}__moses_{dt0.strftime('%Y%m%d')}_binned.nc"

# moses.to_netcdf(output_fname)


# dx = 20
# std = moses.rolling(x_bin = dx, y_bin = dx, min_periods = 5).std()
# med = moses.rolling(x_bin = dx, y_bin = dx, min_periods = 5).median()

# (np.abs(moses-med)>2*std).plot(x = "x_bin")

# moses_old = xr.open_dataset(f"../data/processed/{campaign}__moses_{dt0.strftime('%Y%m%d')}_binned_biased.nc").SST


# xref = moses.x_bin.mean()
# yref = moses.y_bin.mean()

# moses = moses.assign_coords(
#     x_bin = (moses.x_bin-xref)*1e-3,
#     y_bin = (moses.y_bin-yref)*1e-3
# )

# moses_old = moses_old.assign_coords(
#     x_bin = (moses_old.x_bin-xref)*1e-3,
#     y_bin = (moses_old.y_bin-yref)*1e-3
# )

# kw = dict(
#     vmin = 16.5, vmax = 20,
#     levels = np.arange(16.5, 20.1, 0.1),
#     cmap = "Spectral_r", add_colorbar = False
# )

# fig,ax = plt.subplots(2,1, figsize = (7, 6))

# moses_old.plot(ax = ax[0], x = "x_bin", **kw)
# C = moses.plot(ax = ax[1], x = "x_bin", **kw)

# fig.colorbar(C, ax = ax, label = "SST [$^\circ$C]", fraction = 0.02)

# for a in ax:
#     a.set(
#         xlim = [-50, 40],
#         ylim = [-20, 20]
#     )

# ax[0].set(xlabel = "", ylabel = "y [km]", title = "Before")
# ax[1].set(xlabel = "x [km]", ylabel = "y [km]", title = "After")

# fig.savefig("../img/comparing_moses_sst_bias.png", dpi = 300, bbox_inches = "tight")










import xarray as xr
import numpy as np
from scipy.interpolate import griddata

# Initial setup
n = 30
x = xr.DataArray(np.linspace(0, 2 * np.pi, n), dims="x")
y = xr.DataArray(np.linspace(0, 2 * np.pi, n), dims="y")
z = np.sin(x) * xr.ones_like(y)  # sin(x) across a grid

# Introducing NaNs
mask = xr.DataArray(np.random.randint(0, 2, (n, n)).astype('bool'), dims=['x', 'y'])
z = z.where(mask)

z.stack(k = ["x", "y"])

z.interp(x = 0, y = 3, method="linear", kwargs={"fill_value": "extrapolate"})

pad = 5
i = 10
j = 20

xc = z.x.isel(x = i).values
yc = z.y.isel(y = j).values

# if np.isnan(z.isel(x = i, y = j)):

zi = z.isel(x = slice(i-pad, i+pad), y = slice(j-pad, j+pad))

x, y = np.meshgrid(zi.x, zi.y)
values = zi.values
valid = ~np.isnan(values)

griddata((x[valid], y[valid]), values[valid], (xc, yc))

# else:
#     return z.isel(x = i, y = i)


def custom_interp_func(window):
    # window is now a flattened array of the surrounding values, including the center value
    shape = window.shape[1:]  # Get the shape of the original window
    center_index = np.array(shape) // 2
    center_flat_index = center_index[0] * shape[1] + center_index[1]
    center_value = window[:, center_flat_index]
    
    # If the center is not NaN, return it directly
    if not np.isnan(center_value):
        return center_value
    
    # Otherwise, interpolate using valid values in the window
    valid_mask = ~np.isnan(window)
    if not valid_mask.any():
        return np.nan  # No valid values to interpolate from
    
    # Perform interpolation; this example simply takes the mean of valid values
    return np.nanmean(window, axis=1)

window_size = 3  # Define your window size
pad_size = window_size // 2

# Using construct to create windowed views of the data
windowed = z.pad(x=(pad_size, pad_size), y=(pad_size, pad_size), constant_values=np.nan).construct(
    {"window": (("x", "y"), np.full((window_size, window_size), fill_value=True))},
    fill_value=np.nan
).stack(rolled=('x', 'y', 'window'))

# Apply the custom function over the windows
interpolated = xr.apply_ufunc(
    custom_interp_func, 
    windowed,
    input_core_dims=[["rolled"]],
    vectorize=True,
    dask="parallelized",
    output_dtypes=[float]
).unstack("rolled")

# interpolated is now an xarray DataArray with interpolated NaNs within each window




import xarray as xr
import numpy as np

n = 30
x = xr.DataArray(np.linspace(0,2*np.pi,n),dims=['x'])
y = xr.DataArray(np.linspace(0,2*np.pi,n),dims=['y'])
z = (np.sin(x)*xr.ones_like(y))

mask = xr.DataArray(np.random.randint(0,1+1,(n,n)).astype('bool'),dims=['x','y'])

z = z.where(mask)

zs = z.stack(k=['x','y'])

zvalid = zs.where(~np.isnan(zs),drop=True)
znan = zs.where(np.isnan(zs),drop=True)

xi,yi = znan.k.x.drop('k'),znan.k.y.drop('k')

zi = znan.interp(x=xi,y=yi)

zi = zi.set_index(k = ["x", "y"]).unstack("k")

kw = dict(vmin = -1, vmax = 1, cmap = "RdBu_r")

fig,ax = plt.subplots()
z.where(mask).plot(ax=ax,x = "x", **kw)
ax.scatter(xi,yi,c=zi,**kw,linewidth=1,edgecolor='k')

