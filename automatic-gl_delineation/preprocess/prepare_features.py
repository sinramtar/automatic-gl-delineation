#-----------------------------------------------------------------------------------
#	
#	script     : contains functions to prepare features stack
#   author     : Sindhu Ramanath Tarekere
#   date	   : 24 Nov 2021
#
#-----------------------------------------------------------------------------------

import datetime
import re
from collections import defaultdict
from ast import literal_eval
import pathlib

# Data management modules
import geopandas as gpd
from netCDF4 import Dataset
import pandas as pd

import rasterio as rio
import rasterio.mask
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.merge import merge
from shapely.geometry import box

import pyTMD
from pyTMD.io.OTIS import read_otis_grid

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy import ndimage, signal
import math

from joblib import Parallel, delayed
from tqdm import tqdm

def get_closest_tide_point(long, lat, cats):
    """Finds the closest coordinates containing a valid tidal amplitude value

    Parameters
    ----------
    long : float32
        longitude (decimal degrees)
    lat : float32
        latitude (decimal degrees)
    cats : Path obj
        path containing CATS2008 model
    Parameters
    ----------
    long : float32
        longitude (decimal degrees)
    lat : float32
        latitude (decimal degrees)
    cats : Path obj
        path containing CATS2008 model

    Returns
    -------
    (float32, float32)
        closest longitude and latitude (decimal degrees)
    """    
    modelGridFile = cats / 'CATS2008/grid_CATS2008'
    erosion_in_km = 20.0
    #Read grid file
    xi, yi, _, mz, _, _ = read_otis_grid(modelGridFile)
    #-- grid step size of tide model
    dx = xi[1] - xi[0]
    dy = yi[1] - yi[0]
    minxi = xi.min()
    minyi = yi.min()
    modelGridFile = cats / 'CATS2008/grid_CATS2008'
    erosion_in_km = 20.0
    #Read grid file
    xi, yi, _, mz, _, _ = read_otis_grid(modelGridFile)
    #-- grid step size of tide model
    dx = xi[1] - xi[0]
    dy = yi[1] - yi[0]
    minxi = xi.min()
    minyi = yi.min()

    #Erode grid mask
    erosion_iteration = (int) (erosion_in_km / dx)
    eroded_mask = ndimage.morphology.binary_erosion(mz, iterations = erosion_iteration)
    #Erode grid mask
    erosion_iteration = (int) (erosion_in_km / dx)
    eroded_mask = ndimage.morphology.binary_erosion(mz, iterations = erosion_iteration)

    _, inds = ndimage.distance_transform_edt(np.logical_not(eroded_mask), return_indices = erosion_iteration)
    _, inds = ndimage.distance_transform_edt(np.logical_not(eroded_mask), return_indices = erosion_iteration)

    long = np.atleast_1d(long)
    lat = np.atleast_1d(lat)
    x, y = pyTMD.convert_ll_xy(long, lat, 'CATS2008', 'F')
    long = np.atleast_1d(long)
    lat = np.atleast_1d(lat)
    x, y = pyTMD.convert_ll_xy(long, lat, 'CATS2008', 'F')

    x_ind = int(math.floor(x - minxi)/dx)
    y_ind = int(math.floor(y - minyi)/dy)
    x_ind = int(math.floor(x - minxi)/dx)
    y_ind = int(math.floor(y - minyi)/dy)

    #Get closest index
    closest_y_ind = inds[0][y_ind][x_ind]
    closest_x_ind = inds[1][y_ind][x_ind]
    #Get closest index
    closest_y_ind = inds[0][y_ind][x_ind]
    closest_x_ind = inds[1][y_ind][x_ind]

    closest_x = ((closest_x_ind+0.5)*dx) + minxi
    closest_y = ((closest_y_ind+0.5)*dy) + minyi
    closest_x = ((closest_x_ind+0.5)*dx) + minxi
    closest_y = ((closest_y_ind+0.5)*dy) + minyi

    #convert to lat lon
    closest_lon, closest_lat = pyTMD.convert_ll_xy(np.atleast_1d(closest_x), np.atleast_1d(closest_y), 'CATS2008','B')
    return closest_lon, closest_lat

def get_tidal_amplitude(date_object, lons, lats, cats):
    """Returns tide levels (pixelwise) given time, coordinates and tidal model.
    Function was adapted from:
    https://gitlab.eoc.dlr.de/glaciology/groundingline/-/blob/master/processor/tide_prediction/predictTideForPoints.py

    Parameters
    ----------
    date_object : datetime
        time for which tide leves are calculated
    lons : ndarray (1, )
        longitudes (EPSG:3031)
    lats : ndarray (1, )
        latitudes (EPSG:3031)
    cats : Path obj
        directory containing tidal model

    Returns
    -------
    ndarray
        masked array of tidal corrections
    """   
    tide_amplitude = pyTMD.compute_tide_corrections(lons, \
            lats, np.zeros((len(lons)), dtype = 'float32'), \
            DIRECTORY = cats, \
            MODEL = 'CATS2008', EPSG = 3031, \
            EPOCH = (date_object.year, date_object.month, date_object.day, date_object.hour, date_object.minute, date_object.second), TYPE = 'drift', TIME = 'GPS', \
            METHOD = 'spline', FILL_VALUE = np.nan)

    if np.isnan(tide_amplitude).all():
        print(f'Is nan')
        closest_lats = []
        closest_lons = []
        lons, lats = rio.warp.transform(src_crs = rio.crs.CRS.from_string('EPSG:3031'), dst_crs = rio.crs.CRS.from_string('EPSG:4326'), xs = lons, ys = lats)
        for lon, lat in zip(lons, lats):
            cl_lon, cl_lat = get_closest_tide_point(lon, lat, cats)
            closest_lats.append(cl_lat)
            closest_lons.append(cl_lon)
        
        tide_amplitude = pyTMD.compute_tide_corrections(np.asarray(closest_lons), \
            np.asarray(closest_lats), np.zeros((len(lons)), dtype = 'float32'), \
            DIRECTORY = cats, \
            MODEL = 'CATS2008', EPSG = 4326, \
            EPOCH = (date_object.year, date_object.month, date_object.day, date_object.hour, date_object.minute, date_object.second), TYPE = 'drift', TIME = 'GPS', \
            METHOD = 'spline', FILL_VALUE = np.nan)
    return tide_amplitude

def get_air_pressure(date, longs, lats, dataset_dir, transform = False, src_crs = 'EPSG:3031', dst_crs = 'EPSG:4326'):
    """Calculates air pressure for given date, longitude and latitude coordinates. Dataset is located in dataset_dir.
    Adapted from: https://gitlab.eoc.dlr.de/imf-glaciology/groundingline/-/blob/master/processor/tide_prediction/predictAirPressure.py

    Parameters
    ----------
    date : datetime
        time for which air pressure is calculated
    lons : ndarray
        longitude coordinates (decimal degrees) in a 2d grid
    lat : ndarray
        latitude coordinates (decimal degrees) in a 2d grid
    dataset_dir : Path obj
        location of NCEP/NCAR air pressure model

    Returns
    -------
    ndarray
        air pressure 
    """  
    ap_file = dataset_dir / ("slp." + str(date.year) + ".nc") 
    fileBaseDateTimeString = '1800-01-01T00:00:00'
    fileBaseDateTime = datetime.datetime.strptime(fileBaseDateTimeString, '%Y-%m-%dT%H:%M:%S')

    timeOffset = date - fileBaseDateTime
    timeOffsetHours = timeOffset.total_seconds() / 60.0 / 60.0
    
    try:
        rootgrp = Dataset(ap_file, "r", format = "NETCDF4")
    except:
        print(ap_file)
        return

    if transform:
        longs, lats = rio.warp.transform(src_crs = rio.crs.CRS.from_string(src_crs), dst_crs = rio.crs.CRS.from_string(dst_crs), xs = longs, ys = lats)

    longs = [long + 360.0  if long < 0.0 else long for long in longs]

    #To be able to interpolate bewtween 357.5 and 0.0
    if np.asarray(longs).any() > 357.5:
        lons = np.hstack([rootgrp['lon'][:].data, 360.0])
        values = np.concatenate([rootgrp['slp'][:], np.atleast_3d(rootgrp['slp'][:,:,0])], axis = 2)
    else:
        lons = rootgrp['lon'][:].data
        values = rootgrp['slp'][:]

    #Need to invert latitude to have ascending grid
    points = (rootgrp['time'][:].data, rootgrp['lat'][:].data*-1, lons)
    
    #Need to invert latitude to because of previous inversion
    pressures = []
    interpolated_grid = RegularGridInterpolator(points = points, values = values, method = 'linear', bounds_error = False, fill_value = np.nan)
    fit_points = np.asarray([[timeOffsetHours, lat*-1, long] for long, lat in zip(longs, lats)])
    pressures = interpolated_grid(fit_points)

    if not np.all(np.isnan(pressures)):
        pressures = np.nan_to_num(pressures, nan = np.nanmean(pressures))

    return np.asarray(pressures)

def read_shapefile(shp_file):
    """Reads shapefile and returns geopandas DataFrame

    Parameters
    ----------
    shp_dir : Path obj
        path to shapefile

    Returns
    -------
    GeoDataFrame
        dataframe of shapefile
    """    
    return gpd.read_file(shp_file)

def crop(dd, gl):
    """Crops raster to extent of shapefile, writes cropped raster to savedir

    Parameters
    ----------
    dd_dir : Path obj
        directory containing the original rasters
    gl_df : GeoDataFrame
        dataframe of shapefile
    savedir : Path obj
        directory where cropped rasters would be written to
    """ 
    
    with rio.open(dd, 'r') as src:
        gl_values = gl.geometry.values
        cropped_image, cropped_transform = rasterio.mask.mask(src, gl_values, crop = True, filled = False, nodata = src.nodata, all_touched = True)
    return cropped_image, cropped_transform

def rasterize_grounding_lines(dd, gl_df, res, savedir, projection = 'epsg:3031'):
    """Creates labels (grounding line = 1, non grounding line = 0) for double difference interferograms

    Parameters
    ----------
    dd : Path obj
        double difference raster
    gl_df : GeoDataFrame
        dataframe of shapefile
    res : int
        resolution of resulting labels tiff
    projection : str
        projection of resulting labels tiff
    savedir : Path obj
        directory where labels would be written to
    """    
    if 'tile_' in dd.name:
        uuid = dd.name[:dd.name.rfind('_tile')]
    else:
        uuid = dd.name.removesuffix('.tif')

    shape = gl_df.loc[gl_df['UUID'].str.contains(uuid)]
    dst_crs = rio.crs.CRS.from_string(projection)
    if not shape.empty:            
        with rio.open(dd, 'r') as src:
            if dst_crs == src.crs:
                transform = src.transform
                height = src.height
                width = src.width
            else:            
                transform, width, height = calculate_default_transform(src_crs = src.crs, dst_crs = rio.crs.CRS.from_string(projection), width = src.width,
                                                height = src.height, left = src.bounds.left, bottom = src.bounds.bottom, right = src.bounds.right,
                                                top = src.bounds.top, resolution = res)

            gl_values = shape[['geometry']].values.flatten()
            shapes = ((geom, 1) for geom in gl_values)
            labels = rio.features.rasterize(shapes = shapes, out_shape = (height, width), 
                                            transform = transform, all_touched = True, dtype = 'uint8')

            labels_meta = src.meta
            labels_meta.update({"driver": "GTiff",
                "height": height,
                "width": width,
                'transform': transform,
                'count': 1,
                'dtype': 'uint8'})

            label_name = savedir / dd.name
            with rio.open(label_name, 'w', **labels_meta) as dest:
                dest.write(labels, indexes = 1)

def label_multiclass(dd, gl_df, res, plot_resources, savedir, projection = "EPSG:3031"):
    """Creates labels (grounding line = 1, grounded ice = 2, ocean = 3, ice shelf = 4) for double difference interferograms

    Parameters
    ----------
    dd : Path obj
        double difference raster
    gl_df : GeoDataFrame
        dataframe of shapefile
    res : int
        resolution of resulting labels tiff
    savedir : Path obj
        directory where labels would be written to
    projection : str, optional
        projection of resulting labels tiff. Defaults to "EPSG:3031"
    """  
    basemap = gpd.read_file(plot_resources / 'basemap')
    ice_shelf = basemap.loc[basemap.Category == "Ice shelf"]
    ocean = basemap.loc[basemap.Category == "Ocean"]
    grounded_ice = basemap.loc[basemap.Category == "Land"]

    uuid = dd.name.removesuffix('.tif')
    shape = gl_df.loc[gl_df['UUID'] == uuid]
    if not shape.empty:            
        with rio.open(dd, 'r') as src:
            transform, width, height = calculate_default_transform(src_crs = src.crs, dst_crs = rio.crs.CRS.from_string(projection), width = src.width,
                                                height = src.height, left = src.bounds.left, bottom = src.bounds.bottom, right = src.bounds.right,
                                                top = src.bounds.top, resolution = res)


            bbox = rio.transform.array_bounds(height, width, transform)
            polygon = box(*bbox)
            clipped_ice_shelf = ice_shelf.clip(polygon, keep_geom_type = True)
            clipped_ocean = ocean.clip(polygon, keep_geom_type = True)
            clipped_grounded_ice = grounded_ice.clip(polygon, keep_geom_type = True)

            labels = np.zeros((height, width), dtype = 'uint8')

            shapes_ice_shelf = ((geom, 4) for geom in clipped_ice_shelf.geometry.values)
            if not (clipped_ice_shelf.is_empty).all():
                labels = rio.features.rasterize(shapes = shapes_ice_shelf, out_shape = (height, width),
                                            transform = transform, all_touched = True, dtype = 'uint8')

            shapes_ocean = ((geom, 3) for geom in clipped_ocean.geometry.values)
            if not (clipped_ocean.is_empty).all():
                labels = rio.features.rasterize(shapes = shapes_ocean, out_shape = (height, width), out = labels,
                                            transform = transform, all_touched = True, dtype = 'uint8')

            shapes_grounded_ice = ((geom, 2) for geom in clipped_grounded_ice.geometry.values)
            if not (clipped_grounded_ice.is_empty).all():
                labels = rio.features.rasterize(shapes = shapes_grounded_ice, out_shape = (height, width), out = labels,
                                            transform = transform, all_touched = True, dtype = 'uint8')

            gl_values = shape[['geometry']].values.flatten()
            shapes = ((geom, 1) for geom in gl_values)
            labels = rio.features.rasterize(shapes = shapes, out_shape = (height, width), out = labels,
                                            transform = transform, all_touched = True, dtype = 'uint8')

            if not (labels == 0).all():
                print(uuid)

                labels_meta = src.meta
                labels_meta.update({"driver": "GTiff",
                    "height": height,
                    "width": width,
                    'transform': transform,
                    'count': 1,
                    'dtype': 'uint8',
                    'crs': 'EPSG:3031'})
                

                label_name = savedir / dd.name
                with rio.open(label_name, 'w', **labels_meta) as dest:
                    dest.write(labels, indexes = 1)

def reproject_and_resample_complex(raster, writedir, dst_crs = 'EPSG:3031', pixel_size = (100.0, 100.0)):
    """Converts phase to complex array and reprojects raster to given CRS and resamples to given pixel size

    Parameters
    ----------
    raster : Path obj
        path to raster
    writedir : Path obj
        directory where reprojected raster will be saved
    dst_crs : str, optional
        desired CRS, by default 'EPSG:3031'
    pixel_size : (float, float)
        pixel size (x, y) in units of destination CRS, by default (100.0, 100.0) meters
    """    
    dst_crs = rio.crs.CRS.from_string(dst_crs)
    with rasterio.open(raster) as src:
        name_str_index = src.name.rfind('/')
        writepath = writedir / src.name[name_str_index + 1:]
        transform, width, height = calculate_default_transform(src_crs = src.crs, dst_crs = dst_crs, width = src.width,
        height = src.height, left = src.bounds.left, bottom = src.bounds.bottom, right = src.bounds.right,
        top = src.bounds.top, resolution = pixel_size)

        phase = src.read(1)
        kwargs = src.meta.copy()
        kwargs.update({
                    'crs': dst_crs,
                    'transform': transform,
                    'width': width,
                    'height': height,
                    'count': 4
                    })
    
    complex_phase = np.ones_like(phase) * np.exp(1j * (phase))

    dest_complex = np.full((2, height, width), fill_value = 0, dtype = src.meta['dtype'])

    reproject(source = np.real(complex_phase),
            destination = dest_complex[0, :, :],
            src_transform = src.transform,
            src_crs = src.crs,
            src_nodata = src.nodata,
            dst_nodata = src.nodata,
            dst_transform = transform,
            dst_crs = dst_crs,
            resampling = Resampling.bilinear,
            dst_resolution = pixel_size)
    
    reproject(source = np.imag(complex_phase),
            destination = dest_complex[1, :, :],
            src_transform = src.transform,
            src_crs = src.crs,
            src_nodata = src.nodata,
            dst_nodata = src.nodata,
            dst_transform = transform,
            dst_crs = dst_crs,
            resampling = Resampling.bilinear,
            dst_resolution = pixel_size)

    descriptions = ['Real', 'Imaginary', 'Pseudocoherence', 'Phase']
    with rasterio.open(writepath, 'w', **kwargs) as dst:
        for band, arr in zip(range(2), [dest_complex[0, :, :], dest_complex[1, :, :]]):
            dst.write_band(band + 1, arr)
            dst.set_band_description(band + 1, descriptions[band])
        
        pseudocoherence, phase = rectangular_to_polar(dest_complex)
        dst.write_band(3, pseudocoherence)
        dst.set_band_description(3, descriptions[2])
        dst.write_band(4, phase)
        dst.set_band_description(4, descriptions[3])
                
def rectangular_to_polar(rect):
    complex_resampled =  rect[0, :, :] + 1j * rect[1, :, :]
    return np.abs(complex_resampled), np.angle(complex_resampled)                  

def reproject_and_resample(src, dst, res, src_nodata, dst_nodata, src_transform, dst_transform,
                            src_crs, dst_crs):
    dst_crs = rio.crs.CRS.from_string(dst_crs)
    reproject(source = src,
            destination = dst,
            src_transform = src_transform,
            src_crs = src_crs,
            src_nodata = src_nodata,
            dst_nodata = dst_nodata,
            dst_transform = dst_transform,
            dst_crs = dst_crs,
            resampling = Resampling.bilinear,
            dst_resolution = res)
    return dst

def windowed_read(src_raster, ref_raster, bands, res = None):
    """Reads a portion of source raster, based on bounds from reference raster.

    Parameters
    ----------
    src_raster : Path obj
        source raster, from which window is extracted
    ref_raster : Path obj
        reference raster, defines the bounds of the window
    bands : int
        number of bands to be extracted from source raster
    res : int, optional
        resolution to which the windowed raster would to resampled to. Defaults to None

    Returns
    -------
    ndarray (bands, ), float
        extracted window, nodata value
    """    
    with rio.open(src_raster) as src:
        with rio.open(ref_raster) as ref:
            col_start = round(abs(src.bounds.left - ref.bounds.left)/src.res[0])
            row_start = round(abs(src.bounds.top - ref.bounds.top)/src.res[1])
            col_end = col_start + ref.width
            row_end = row_start + ref.height

            window = rio.windows.Window.from_slices((row_start, row_end), (col_start, col_end))
            
            windowed_data = src.read(indexes = range(1, bands + 1), window = window)
            meta = src.meta.copy()
            meta.update({
                'count': bands,
                'height': windowed_data.shape[1],
                'width': windowed_data.shape[2]})
        
        dst_arr = np.zeros_like(windowed_data)
        if res:
            dst_arr = reproject_and_resample(windowed_data, dst_arr, res, src.nodata, src.nodata,  src.transform, 
                                             src.transform, src.crs, 'EPSG:3031')
        else:
            dst_arr = windowed_data

    return dst_arr, src.nodata

def append_to_raster(src, dest_raster, description):
    """Appends array/raster to another raster

    Parameters
    ----------
    src : ndarray or open DatasetWriter obj
        data to append to raster
    dest_raster : Path obj
        raster to which data is appended
    description : list of str
        band description
    """    
    with rio.open(dest_raster) as dest:
        descriptions = list(dest.descriptions) + description
        if isinstance(src, np.ndarray):
            augmented = np.zeros((src.shape[0] + dest.count, dest.height, dest.width), dtype = src.dtype)
            augmented[:dest.count, :, :] = dest.read(indexes = range(1, dest.count + 1))
            augmented[dest.count:, :, :] = src
            augmented_meta = dest.meta.copy()
            augmented_meta.update({
                        'count': src.shape[0] + dest.count,
                        'dtype': src.dtype})
        else:
            augmented = np.zeros((src.count + dest.count, dest.height, dest.width), dtype = src.meta['dtype'])
            augmented[:dest.count, :, :] = dest.read(indexes = range(1, dest.count + 1))
            augmented[dest.count:, :, :] = src.read(indexes = range (1, src.count + 1))
            augmented_meta = dest.meta.copy()
            augmented_meta.update({
                        'count': src.shape[0] + dest.count,
                        'dtype': src.meta['dtype'],
                        'nodata' : src.nodata })

    with rio.open(dest_raster, 'w', **augmented_meta) as dest:
        for band in range(augmented_meta['count']):
            dest.write_band(band + 1, augmented[band, :, :].reshape(augmented_meta['height'], augmented_meta['width']))
            dest.set_band_description(band + 1, descriptions[band])

def get_colordict(mapdir, ers = False):
    """Creates a dictionary that maps RGB values to phase (-pi to pi)

    Parameters
    ----------
    mapdir : Path
        path to file containing the mapping
    ers : bool, optional
        indicates if the type of mapping. If True, uses ers colormap, defaults to False

    Returns
    -------
    default dict
        dictionary mapping color to phase values
    """    
    colormap_arr = np.zeros((256, 3))

    if ers:
        with open(mapdir) as md:
            colormap_text = md.readlines()
            for index, line in enumerate(colormap_text):
                colormap_arr[index, :] = np.array(literal_eval(line))
    else:
        with open(mapdir) as md:
            colormap_text = md.readlines()[6:]
            for index, line in enumerate(colormap_text):
                colormap_arr[index, :] = re.findall(r'\=([^]]*)\;', line)

    colormap_dictionary = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    phase_mapping = np.linspace(-np.pi, np.pi, colormap_arr.shape[0])
    for index, color in enumerate(colormap_arr):
        colormap_dictionary[color[0]][color[1]][color[2]] = phase_mapping[index]

    return colormap_dictionary

def map_rgb_to_phase(rgb_raster, colordict):
    phase_raster = np.full((rgb_raster.shape[0], rgb_raster.shape[1]), fill_value = 0.0, dtype = np.float32)
    for row in range(rgb_raster.shape[0]):
        for col in range(rgb_raster.shape[1]):
            color = rgb_raster[row, col]
            phase_raster[row, col] = colordict[color[0]][color[1]][color[2]]
    return phase_raster

def rgb_to_phase(raster: rio.DatasetReader, gl: gpd.GeoSeries, colordict_dir: pathlib.Path, savedir: pathlib.Path):
    """Converts RGB to phase values and writes raster to savedir

    Parameters
    ----------
    raster : DatasetWriter obj
        double difference interferogram
    colordict_dir : pathlib.Path
        directory where rgb to phase conversion tables are stored
    Returns
    -------
    ndarray
        phase array
    """ 
    with rio.open(raster) as rgb:   
        meta = rgb.meta.copy()
    
    cropped_image, cropped_transform = crop(dd = raster, gl = gl)
    red = cropped_image[0, :, :]
    green = cropped_image[1, :, :]
    blue = cropped_image[2, :, :]

    rgb = np.dstack((red, green, blue))
    colordict = get_colordict(mapdir = colordict_dir / 'colormap.md')

    phase = map_rgb_to_phase(rgb_raster = rgb, colordict = colordict)

    if np.all(phase == None) or np.all(phase == 0):
        colordict = get_colordict(mapdir = colordict_dir / 'ers_colormap_array.txt', ers = True) 
        phase = map_rgb_to_phase(rgb_raster = rgb, colordict = colordict)

    meta.update({'count': 1, 'dtype': np.float32, 'height': phase.shape[0], 'width': phase.shape[1], 'transform': cropped_transform})
    if savedir and (np.all(phase != 0) or np.all(phase != None)):
        with rio.open(savedir / raster.name, 'w', **meta) as dest:
            dest.write(phase, indexes = 1)

def compute_tides_and_air_pressure(raster, gl, lats, lons, cats_model, ap_model):
    """Generates corrected tidal displacement for given double difference interferogram. Tide amplitudes are calculated at coordinates of
    grounding line.

    Parameters
    ----------
    raster : DatasetWriter obj
        double difference interferogram label
    gl : GeoDataSeries
        grounding line corresponding to interferogram
    lats : ndarray
        latitude (EPSG:3031) of grounding line pixels
    lons : ndarray
        longitude (EPSG:3031) of grounding line pixels
    cats_model : Path obj
        directory containing the tidal model
    ap_model : Path obj
        directory containing the air pressure model 

    Returns
    -------
    (tidal_displacement, air_pressure)
    temporary raster files of corrected tidal displacements and air pressures
    """
    times = np.concatenate(gl[['T1', 'T2', 'T3', 'T4']].values)
    times = [pd.to_datetime(time) for time in times if not pd.isnull(time)]
    num_acqs = len(times)
    
    tides = np.zeros((num_acqs, raster.height, raster.width), dtype = np.float32)
    pressure = np.zeros((num_acqs, raster.height, raster.width), dtype = np.float32)
    
    for ind, time in enumerate(times):
        tides[ind, :, :] = get_tidal_amplitude(time, lons, lats, cats_model).reshape(raster.height, raster.width)
        if not np.all(np.isnan(tides[ind, :, :])):
            tides[ind, :, :] = np.nan_to_num(tides[ind, :, :], nan = np.nanmean(tides[ind, :, :]))
        else:
            print(f'No tides')
            tides[ind, :, :] = 0
        pressure[ind, :, :] = get_air_pressure(time, lons, lats, ap_model, transform = True).reshape(raster.height, raster.width)

    acc_gravity = 9.8
    density_seawater = 1026
    tides = tides - ((101325 - pressure) / (acc_gravity * density_seawater))
    
    tidal_displacement = np.zeros((raster.height, raster.width), dtype = np.float32)
    atmos_pressure = np.zeros((raster.height, raster.width), dtype = np.float32)

    if num_acqs == 2:
        tidal_displacement = tides[0, :, :] - tides[1, :, :]
        atmos_pressure = np.abs(pressure[0, :, :] - pressure[1, :, :])
    elif num_acqs == 3:
        tidal_displacement = (tides[1, :, :] - tides[0, :, :]) + (tides[1, :, :] - tides[2, :, :])
        atmos_pressure = np.abs((pressure[1, :, :] - pressure[0, :, :]) + (pressure[1, :, :] - pressure[2, :, :]))
    elif num_acqs == 4:
        tidal_displacement = (tides[1, :, :] - tides[0, :, :]) + (tides[3, :, :] - tides[2, :, :])
        atmos_pressure = np.abs((pressure[1, :, :] - pressure[0, :, :]) + (pressure[2, :, :] - pressure[3, :, :]))
    
    return tidal_displacement, atmos_pressure

def get_gl_coords(raster):
    """Returns coordinates convered by the double difference interferogram

    Parameters
    ----------
    raster : DatasetWriter obj
        double difference interferogram

    Returns
    -------
    (longs, lats)
        longitudes and latitudes covered by the interferogram
    """    

    lats = np.arange(raster.bounds.bottom, raster.bounds.top + raster.res[1], raster.res[1])
    longs = np.arange(raster.bounds.left, raster.bounds.right + raster.res[0], raster.res[0])

    if lats.shape[0] != raster.height:
        lats = lats[:raster.height]
    if longs.shape[0] != raster.width:
        longs = longs[:raster.width]

    longs, lats = np.meshgrid(longs, lats)
    longs = longs.reshape(-1)
    lats = lats.reshape(-1)
    return longs, lats

def create_patches(raster, overlap, savepatches, patch_size = 256):
    """Creates square overlapping patches from raster. Ignores patches that either have no information (all zeros).

    Parameters
    ----------
    raster : DatasetWriter obj
        raster from which to create patches
    overlap : float (0-1)
        fraction overlap between successive tiles
    savepatches : Path obj
        directory in which the patches are saved
    patch_size : int, optional
        dimension of patch, by default 256
    """ 
    patches = []
    patch_meta = []
    step = int((1 - overlap) * patch_size)
    tolerance = 1e-02
    true_phase_range = 2*np.pi

    with rio.open(raster) as src:
        descriptions = src.descriptions
        for row_offset in range(0, src.height, step):
            for col_offset in range(0, src.width, step):
                window = rio.windows.Window(row_offset, col_offset, patch_size, patch_size)
                transform = rio.windows.transform(window, src.transform)
                patch = np.zeros((src.count, patch_size, patch_size), dtype = src.profile['dtype'])
                patch = src.read(window = window)
                if patch.shape[1] == 0 or patch.shape[2] == 0 or np.any(patch == np.nan) or np.all(patch[3, :, :] == 0) or \
                    (((patch[3, :, :].max() - patch[3, :, :].min()) - true_phase_range) > tolerance): 
                    continue
                else:
                    patches.append(patch)
                    meta = src.meta.copy()
                    meta.update({'transform': transform,
                                    'height': patch.shape[1],
                                    'width': patch.shape[2],
                                    'count': patch.shape[0]})
                    patch_meta.append(meta)
    
    for ind, (patch, meta) in enumerate(zip(patches, patch_meta)):
        patch_name = raster.name[:raster.name.rfind('.tif')] + '_tile_' + str(ind) + '.tif'
        with rio.open(savepatches / patch_name, 'w', **meta) as dest:
            for band in range(meta['count']): 
                dest.write_band(band + 1, patch[band, :, :])
                dest.set_band_description(band + 1, descriptions[band])

def estimate_noise(phase, filter_size = 5): 
    """Computes noise per pixel by considering a moving average. The result is scaled to [0, 1]

    Parameters
    ----------
    phase : ndarray
        phase values
    filter_size : int, optional
        size of averaging window, by default 5

    Returns
    -------
    ndarray
        noise estimate
    """    
    complex_phase = np.ones_like(phase) * np.exp(1j * (phase))
    avg = signal.convolve2d(complex_phase, np.ones((filter_size, filter_size)), mode = 'same', boundary = 'symm') / (filter_size * filter_size)
    return np.abs(avg)

def stack_features(raster: pathlib.Path, dem_raster: pathlib.Path = None,  gl: gpd.GeoDataFrame = None,
                   vel_raster: pathlib.Path = None, cats: pathlib.Path = None, ap_model: pathlib.Path = None):
    """Creates stack of features, with an option to append the features and save to existing tif

    Parameters
    ----------
    raster : Path obj
        path to raster 
    gl : GeoDataFrame obj
        data frame of grounding lines
    dem_raster : Path obj
        path to mosaicked DEM raster
    vel_raster : Path obj
        path to ice velocity raster
    cats : Path obj
        path to CATS2008 tide model
    ap_model : Path obj
        path to NCEP/NCAR air pressure dataset
    """
    if gl:    
        gl = gl.loc[gl['UUID'] == raster.name[:raster.name.rfind('_tile')]]
    
    num_features = 4
    descriptions = ['Real', 'Imaginary', 'Pseudocoherence', 'Phase (radians)']

    non_interferometric_features = [dem_raster, vel_raster, cats, ap_model]
    num_features = num_features + sum(x is not None for x in non_interferometric_features)
    
    band = 3
    with rio.open(raster) as tile:
        features_stack = np.zeros((num_features, tile.height, tile.width), dtype = tile.meta['dtype'])
        features_stack[0:4, :, :] = tile.read([1, 2, 3, 4])
        features_stack[0:4, :, :] = np.nan_to_num(features_stack[0:4, :, :], nan = 0.0)
        
        if dem_raster is not None:
            dem, dem_nodata = windowed_read(dem_raster, raster, 1, tile.res)
            features_stack[band + 1, :, :] = impute(feature = dem, nodata = dem_nodata)
            descriptions.append(f'DEM (m), nodata = {dem_nodata}')
        
        if vel_raster is not None:
            ice_velocity, vel_nodata = windowed_read(vel_raster, raster, 2, tile.res)
            features_stack[(band+1):(band+2), :, :] = impute(feature = ice_velocity, nodata = vel_nodata)
            descriptions = descriptions + ['Ice velocity easting', f'Ice velocity northing, nodata = {vel_nodata}']

        if cats is not None:
            assert(ap_model is not None)
            longs, lats = get_gl_coords(tile)
            tides, ap = compute_tides_and_air_pressure(tile, gl, lats, longs, cats, ap_model)
            features_stack[band + 1, :, :] = impute(feature = tides, nodata = np.nan)
            features_stack[band + 1, :, :] = impute(feature = ap, nodata = 0.0)
            descriptions = descriptions + ['Differential tide level (m), nodata = NaN', 'Atmospheric pressure (Pa), nodata 0.0']

    meta = tile.meta.copy()
    meta.update({'count': num_features})
        
    with rio.open(raster, 'w', **meta) as dest:
        for band in range(features_stack.shape[0]):
            dest.write_band(band + 1, features_stack[band, :, :])
            dest.set_band_description(band + 1, descriptions[band])

def to_npy(arr, name, savedir, dest_dim, channels_last = True):
    """Saves numpy array as a compressed npz file. 

    Parameters
    ----------
    arr : ndarray
        3D numpy array in order (channels, rows, columns) 
    name : str
        name of npz file
    savedir : Path obj
        directory where npz file would be saved
    dest_dim : int
        dimension of array to be saved (needed because not all tiles are the same size)
    channels_last : bool
        decides the array order. If True, arr is saved as (rows, columns, channels), defaults to True
    """     
    
    if (arr.shape[1] != dest_dim) or (arr.shape[2] != dest_dim):
        features = np.zeros((arr.shape[0], dest_dim, dest_dim), dtype = arr.dtype)
        features[:, :arr.shape[1], :arr.shape[2]] = arr
    else:
        features = arr

    if channels_last:
        features = np.swapaxes(features, 0, 2)
        features = np.swapaxes(features, 0, 1) 

    name = name + '.npz'
    np.savez_compressed(savedir / name, features)

def impute(feature, nodata):
    """Impute's raster band partially filled with nodata values with mean value.

    Parameters
    ----------
    feature : ndarray (rows, columns)
        feature
    nodata : float
        nodata value for the channel to be imputed
    """    
    if np.isnan(nodata):
        feature = np.nan_to_num(feature, nan = np.nanmean(feature))
    else:
        feature[feature == nodata] = np.mean(feature[feature != nodata])
    return feature

def get_min_max_npz(npz, bands, ignore_values):
    with np.load(npz) as data:
        features = data['arr_0']
    
    mins = []
    maxes = []
    for band, ignore in zip(bands, ignore_values):
        feature = features[:, :, band]
        if ignore != None:
            feature = np.ma.masked_equal(feature, ignore)
        mins.append(np.nanmin(feature))
        maxes.append(np.nanmax(feature))
    
    return mins, maxes

def get_global_range(tiles: list, bands: list = [4, 5, 6, 7, 8], ignore_values: list = [None, None, None, None, 0.0]) -> list:
    num_tiles = len(tiles)
    num_bands = len(bands)
    
    local_mins, local_maxes = zip(*Parallel(n_jobs = -1)(delayed(get_min_max_npz)(tile, bands, ignore_values) for tile in tiles))
    local_mins = np.asarray(local_mins).reshape(num_tiles, num_bands)
    local_maxes = np.asarray(local_maxes).reshape(num_tiles, num_bands)

    global_max_mins = []        
    global_max = np.nanmax(local_maxes, axis = 0)
    global_min = np.nanmin(local_mins, axis = 0)

    for max_value, min_value in zip(global_max, global_min):
        global_max_mins.append([min_value, max_value])
    
    return global_max_mins
    
def scale_features(tile: pathlib.Path, savedir: pathlib.Path, global_max_mins: list, bands: list) -> None:
    """Scales features between either 0 or 1 or provided global range

    Parameters
    ----------
    tile : Path obj
        .npz file of features
    savedir : Path
        directory where scaled array would be saved
    global_max_mins : list of tuples, optional
        [min, max] desired range of scaled data.
    bands : list
        bands to be scaled
    """    
    path_copy = tile.resolve()
    path_copy = savedir / tile.name

    with np.load(tile) as data:
        features = data['arr_0']
    features_scaled = np.zeros_like(features)
    
    features_scaled[:, :, :4] = features[:, :, :4]
    for ind, band in enumerate(bands):
        desired_range = global_max_mins[ind]
        max_min = desired_range[1] - desired_range[0]
        band_min = desired_range[0]
            
        if max_min == 0:
            features_scaled[:, :, band] = features[:, :, band]
        else:
            features_scaled[:, :, band] = (features[:, :, band] - band_min) / max_min
    
    if path_copy == tile.resolve():
        tile.unlink()
    np.savez_compressed(path_copy, features_scaled)
    
def polar_to_rectangular(tile, savedir):
    """Converts complex features amplitude and phase to their rectanglar components,
    preserving the rest of the features

    Parameters
    ----------
    tile : Path
        path to features .npz file
    savedir : Path
        directory where rectangular form of features is saved
    """    
    with np.load(tile) as data:
        features = data['arr_0']
        amplitude = features[:, :, 0]
        phase = features[:, :, 1]
    
    polar = amplitude * np.exp(1j * (phase))
    features[:, :, 0] = np.real(polar)
    features[:, :, 1] = np.imag(polar)

    np.savez_compressed(savedir / tile.name, features)


def npz_to_tif(npz, ref_tif, savepath):
    """Converts ndarray to geoTiFF

    Parameters
    ----------
    npz : ndarray (height, width, channels)
        numpy array of activation map
    ref_tif : geoTiFF
        reference geotiFF
    savepath : Path
    """  
    if isinstance(npz, pathlib.Path):
        with np.load(npz) as data:
            npz = data['arr_0']
              
    with rio.open(ref_tif) as data:
        meta = data.meta.copy()
        meta.update({'dtype': np.float32})
    
    dims = len(npz.shape)
    
    if dims == 2:
        npz = npz[:, :, np.newaxis]
        count = 1
    elif dims == 3:
        count = npz.shape[-1]
    
    meta.update({'height': npz.shape[0], 'width': npz.shape[1], 'count': count})
    with rio.open(savepath, 'w', **meta) as dest:
        for band in range(count):
            dest.write_band(band + 1, npz[:, :, band])

def merge_tiles(to_be_merged, savedir, name, descriptions = None, delete_orig = True):
    """Mosaics geoTiFFs

    Parameters
    ----------
    to_be_merged : list
        list of rasters
    savedir : Path
    name : str
        name of mosaicked tif
    descriptions : list of str, optional
        band wise descriptions, defaults to None
    delete_orig : bool, optional
        if True, deletes to_be_merged list, defaults to True
    """    
    opened = []
    
    for tile in to_be_merged:
        opened.append(rio.open(tile))
    
    merged, transform = merge(opened)
    print(merged.shape)
    meta = opened[0].meta.copy()
    meta.update({'width': merged.shape[2], 'height': merged.shape[1], 'transform': transform, 'count': merged.shape[0]})

    if not descriptions:
        descriptions = np.repeat('', merged.shape[0])
    
    with rio.open(savedir / (name + '.tif'), 'w', **meta) as dest:
        for band in range(merged.shape[0]):
            dest.write_band(band + 1, merged[band, :, :])
            dest.set_band_description(band + 1, descriptions[band])
    
    if delete_orig:
        for tile in to_be_merged:
            tile.unlink()

def merge_npz_tiles(to_be_merged, ref_tif, savedir, name, overlap = 0.2, tile_dim = 1024): 
    with rio.open(ref_tif) as tif:
        merged_size = (tif.height, tif.width, 1)

    merged = np.zeros(merged_size, dtype = np.uint8)
    overlap_areas = np.ones_like(merged)
    step = int((1 - overlap) * tile_dim)

    to_be_merged.sort()
    tile_count = 0
    row_intervals = np.arange(0, merged_size[0], step)
    column_intervals = np.arange(0, merged_size[1], step)

    for row in row_intervals[:-1]:
        for column in column_intervals[:-1]:
            with np.load(to_be_merged[tile_count]) as data:
                prediction = data['arr_0']
            
            merged[row:row+tile_dim, column:column+tile_dim, :] = merged[row:row+tile_dim, column:column+tile_dim, :] + prediction
            overlap_areas[row:row+tile_dim, column:column+tile_dim, 0] = overlap_areas[row:row+tile_dim, column:column+tile_dim, 0] + 1
            tile_count = tile_count + 1
        tile_count = tile_count + 1
    
    merged = np.divide(merged, overlap_areas)
    np.savez_compressed(savedir / (name + '.npz'), merged)  
   

def find_valids(datasplit: list, labels_npz: pathlib.Path):
    """Finds tiles that contain grounding line

    Parameters
    ----------
    datasplit : list
        tile names
    labels_npz : pathlib.Path
        labels directory

    Returns
    -------
    list
        list of valids (tiles with grounding line) and empties
    """    
    valids = []
    empties = []
    for tile in datasplit:
        with np.load(labels_npz / tile) as data:
            labels = data['arr_0']
            if np.all(labels == 0):
                empties.append(tile + '\n')
            else:
                valids.append(tile + '\n')
    return valids, empties

def is_tile_clean(tile):
    clean = True
    with np.load(tile) as data:
        features = data['arr_0']
    if np.any(np.isnan(features)) or np.all(features[:, :, 3] == 0):
        clean = False
    return clean

def create_dataset_split(tiles_npz: pathlib.Path, split_file: pathlib.Path, labels_npz: pathlib.Path = None, grounding_lines: gpd.GeoDataFrame = None):
    """Splits feature stacks into train/validation/test splits

    Parameters
    ----------
    tiles_npz : pathlib.Path
        feature stacks directory
    labels_npz : pathlib.Path (optional)
        labels directory. Default is None
    grounding_lines : gpd.GeoDataFrame (optional)
        database of manual delineations. Default is None
    split_file : pathlib.Path
        empty text file, where dataset split would be written
    """
    tiles = np.asarray(list(tiles_npz.glob('*.npz')))
    tiles = np.asarray([tile for tile in tiles if 'augmented' not in tile.name])
    clean_indices = Parallel(n_jobs = -1)(delayed(is_tile_clean)(tile) for tile in tiles)

    tiles_clean = tiles[clean_indices]
    print(f'After: {len(tiles_clean)}')
    if grounding_lines is not None:
        train_list = grounding_lines.loc[grounding_lines.dataset == 'train', 'UUID'].to_numpy()
        test_list = grounding_lines.loc[grounding_lines.dataset == 'test', 'UUID'].to_numpy()
        validation_list = grounding_lines.loc[grounding_lines.dataset == 'validation', 'UUID'].to_numpy()

        train_set = []
        validation_set = []
        test_set = []
        for tile in validation_list:
            validation_set.append([clean.name for clean in tiles_clean if tile in clean.name])
        for tile in train_list:
            train_set.append([clean.name for clean in tiles_clean if tile in clean.name])
        for tile in test_list:
            test_set.append([clean.name for clean in tiles_clean if tile in clean.name])
        
        train_set = np.concatenate(train_set)
        validation_set = np.concatenate(validation_set)
        train_valids, train_empties = find_valids(train_set, labels_npz)
        validation_valids, validation_empties = find_valids(validation_set, labels_npz)

        test_set = np.concatenate(test_set)
        test_valids, test_empties = find_valids(test_set, labels_npz)
    
        print(f'Test valids: {len(test_valids)} test empties: {len(test_empties)}')
        print(f'Train valids: {len(train_valids)} train empties: {len(train_empties)}')
        print(f'Validation valids: {len(validation_valids)} validation empties: {len(validation_empties)}')
    else:
        test_valids= [clean.name + '\n' for clean in tiles_clean]
        test_empties = '\n'
        train_valids = '\n'
        train_empties = '\n'
        validation_empties = '\n'
        validation_valids = '\n'

    with open(split_file, 'w') as lines:
        lines.write('train_valids')
        lines.write('\n')
        lines.writelines(train_valids)
        lines.write('\n')
        lines.write('train_empties')
        lines.writelines('\n')
        lines.writelines(train_empties)
        lines.write('\n')
        lines.writelines('validation_valids')
        lines.writelines('\n')
        lines.writelines(validation_valids)
        lines.write('\n')
        lines.writelines('validation_empties')
        lines.writelines('\n')
        lines.writelines(validation_empties)
        lines.write('\n')
        lines.writelines('test_valids')
        lines.writelines('\n')
        lines.writelines(test_valids)
        lines.write('\n')
        lines.writelines('test_empties')
        lines.writelines('\n')
        lines.writelines(test_empties)

def random_flip(tile: pathlib.Path, label: pathlib.Path, savetile: pathlib.Path, savelabel: pathlib.Path):
    """Randomly flips feature stacks

    Parameters
    ----------
    tile : pathlib.Path
        original features
    label : pathlib.Path
        corresponding GLL delineation
    savetile : pathlib.Path
        directory where flipped tile would be saved
    savelabel : pathlib.Path
        directory where flipped labels would be saved
    """    
    with np.load(tile) as data:
        features = data['arr_0']
    
    with np.load(label) as data:
        labels = data['arr_0']
    
    features_labels_combo = np.zeros((features.shape[0], features.shape[0], features.shape[2] + 1), dtype = features.dtype)
    features_labels_combo[:, :, :features.shape[2]] = features
    features_labels_combo[:, :, -1] = labels[:, :, 0]

    rng = np.random.default_rng()
    choice = rng.choice([0, 1], 1)[0]

    if choice:
        flipped = np.fliplr(features_labels_combo)
    else:
        flipped = np.flipud(features_labels_combo)

    np.savez_compressed(savetile / ('augmented_' + tile.name) , flipped[:, :, :features.shape[2]])
    np.savez_compressed(savelabel / ('augmented_' + tile.name), flipped[:, :, -1][:, :, np.newaxis])

def clip_raster(raster, gl, saveraster):
    shapes = gl.geometry.values

    with rio.open(raster) as src:
        out_image, out_transform = rio.mask.mask(src, shapes, crop = True)
        out_meta = src.meta

        out_meta.update({"driver": "GTiff",
                         "height": out_image.shape[1],
                         "width": out_image.shape[2],
                         "transform": out_transform})

    with rasterio.open(saveraster, "w", **out_meta) as dest:
        dest.write(out_image)