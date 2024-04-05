import earthaccess
from earthaccess import Auth, Store
from glob import glob
import xarray as xr
import pandas as pd
import os
import subprocess


# Initialize and attempt login
auth = Auth()
auth.login(strategy = "netrc")


datasets = {
    "drifters": {
        "id": {"doi": "10.5067/SMODE-DRIFT"},
        "variables": {"longitude": "lon", "latitude": "lat"}
    },
    "seagliders": {
        "id": {"doi": "10.5067/SMODE-SEAGL3"},
        "variables": {"time_dive": "time", "lon_dive": "lon", "lat_dive": "lat"}
    },
    "floats": {
        "id": {"doi": "10.5067/SMODE-FLOAT"},
        "variables": {"time_GPS": "time", "longitude_GPS": "lon", "latitude_GPS": "lat"}
    },
    "wavegliders": {
        "id": {"doi": "10.5067/SMODE-GLID2"},
        "variables": {"longitude": "lon", "latitude": "lat"}
    },
    "saildrones": {
        "id": {"short_name": "SMODE_L1_SAILDRONES_V1"},#todo
    },
    "dopplerscatt": {
        "id": {"doi": "10.5067/SMODE-DSCT2-V2"},#todo
    },
    "moses": {
        "id": {"doi": "10.5067/SMODE-MOSE2"}
    },
}


# IOP1
campaign = "IOP1"
time_range = ("2022-10-22", "2022-10-25") 

if not os.path.exists(f"../data/external/{campaign}"):
    os.mkdir(f"../data/external/{campaign}")


for key in datasets:
    print(f"\n\n{key}\n")
    path = f"../data/external/{campaign}/{key}/"
    
    if not os.path.exists(path):
        os.mkdir(path)
        
    results = earthaccess.search_data(
        **datasets[key]["id"],
        cloud_hosted = True,
        temporal = time_range
    )
    
    # Download data files
    store = Store(auth)
    files = store.get(results, path)

    if key == "saildrones":
        fnames = glob(f"../data/external/{campaign}/{key}/*")
        fnames.sort()
        for fname in fnames:
            subprocess.run(["tar", "-xzf", fname, "-C", f"../data/external/{campaign}/{key}/"])

for key in datasets:
    variables = datasets[key]["variables"]
    fnames = glob(f"../data/external/{campaign}/{key}/*")
    fnames.sort()
    fnames = [fname for fname in fnames if "_optics" not in fname]

    ds = []
    for fname in fnames:

        dsi = xr.open_dataset(fname)

        if key == "drifters":
            dsi = dsi.isel(time = dsi.position_QCflag.values.nonzero()[0])
        
        dsi = dsi[list(variables)].rename(variables).reset_coords()
        
        if len(list(dsi.dims)) > 1:
            raise ValueError("The dataset must have only one dimension.")
        dim = list(dsi.dims)[0]
        if dim != "time":
            dsi = dsi.swap_dims({dim: "time"})

        if dsi.time.dtype == "float32":
            dsi = dsi.assign(time = ("time", pd.to_datetime(dsi.time.values, unit='s')))
            
        dsi = dsi.isel(time = (dsi.time.dt.year > 1970).values.nonzero()[0])
        dsi = dsi.resample({"time":"1H"}).mean().interpolate_na("time")
        
        ds.append(dsi)
        
    ds = xr.concat(ds, "instrument")
    ds.to_netcdf(f"../data/processed/{campaign}_{key}.nc")

