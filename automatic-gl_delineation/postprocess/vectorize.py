import numpy as np
import rasterio as rio
from rasterio.transform import Affine
from rasterio.merge import merge
import geopandas as gpd

from shapely.geometry import LineString, box, Point
from shapely import make_valid
from shapely.ops import unary_union
from skimage.morphology import skeletonize
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import CubicSpline

from scipy import signal
from plantcv import plantcv as pcv

from tqdm import tqdm
import pathlib
from copy import deepcopy
from itertools import product
import re

def merge_npz(to_be_merged: list, tif_dir: pathlib.Path, ref_tif: pathlib.Path, savedir: pathlib.Path) -> None:
    """Merges npz prediction tiles to original size

    Parameters
    ----------
    npz_dir : pathlib.Path
        predictions directory, containing predictions to be merged
    tif_dir : pathlib.Path
        corresponding tif label/features stack, for original size reference
    ref_tif : pathlib.Path
        original size tif
    savedir : pathlib.Path
        directory where megerd npz will be saved
    """
    to_be_merged.sort()
    with rio.open(ref_tif) as tif:
        merged_size = (tif.height, tif.width, 1)
        merged_bounds = tif.bounds
    
    merged = np.zeros(merged_size)
    overlap_areas = np.zeros_like(merged, dtype = np.uint8)

    for tile in to_be_merged:
        with np.load(tile) as data:
            prediction = data['arr_0']
        
        tif_path = tif_dir / (tile.name.removesuffix('.npz') + '.tif')
        with rio.open(tif_path) as tif:
            tile_bounds = tif.bounds
            tile_res = tif.res
            tile_dim = (tif.height, tif.width)
        
        col_start = int(round(tile_bounds.left - merged_bounds.left) / tile_res[0])
        row_start = int(round(merged_bounds.top - tile_bounds.top) / tile_res[1])
        
        merged[row_start:row_start+tile_dim[0], col_start:col_start+tile_dim[1], 0] = merged[row_start:row_start+tile_dim[0], col_start:col_start+tile_dim[1], 0] + prediction[:tile_dim[0], :tile_dim[1], 0]
        overlap_areas[row_start:row_start+tile_dim[0], col_start:col_start+tile_dim[1], 0] = overlap_areas[row_start:row_start+tile_dim[0], col_start:col_start+tile_dim[1], 0] + 1
    
    merged = np.divide(merged, overlap_areas, out = np.zeros_like(merged), where = overlap_areas != 0)
    merged = signal.medfilt2d(merged, kernel_size = 11)
    savedir.mkdir(parents = True, exist_ok = True)
    np.savez_compressed(savedir / (ref_tif.name.removesuffix('.tif') + '.npz'), merged)

def threshold_predictions(tile: np.ndarray, threshold: np.float16) -> np.ndarray:
    """Converts prediction to binary array, based on threshold value

    Parameters
    ----------
    tile : np.ndarray
        prediction tile
    threshold : np.float16
        pixel threshold value above which pixel is set to 1

    Returns
    -------
    np.ndarray
        binarized prediction
    """    
    prediction_encoded = np.zeros_like(tile, dtype = 'uint8')
    prediction_encoded[tile >= threshold] = 1
    prediction_encoded = signal.medfilt2d(prediction_encoded, kernel_size = 11)
    return prediction_encoded

def get_segments(filled_mask: np.ndarray) -> dict:
    """Saves the pixels of segments of prediction skeleton to dict, with dict keys corresponding to segment id

    Parameters
    ----------
    filled_mask : np.ndarray
        Filled prediction segment, output from pcv.morphology.fill_segments

    Returns
    -------
    dict
        segments and corresponding pixels
    """    
    segments = {}
    for id in np.unique(filled_mask):
        segments[id] = np.argwhere(filled_mask == id)
    del segments[0]
    return segments

def prune_branches(skeleton: np.ndarray, junctions: np.ndarray, segments: dict, path_lengths: np.ndarray) -> np.ndarray:
    for junction in junctions:
        junction_neighbourhood_rows = np.arange(junction[0] - 3, junction[0] + 4)
        junction_neighbourhood_columns = np.arange(junction[1] - 3, junction[1] + 4)
        junction_neighbourhood_pixels = list(product(junction_neighbourhood_rows, junction_neighbourhood_columns))
        junction_segments = np.unique([segment_id for segment_id, segment in segments.items() for pixel in segment if tuple(pixel) in junction_neighbourhood_pixels])
        if junction_segments.any():
            junction_segments = junction_segments.astype(np.uint8)
            junction_segment_lengths = [path_lengths[id] for id in junction_segments - 1]
            branch_pixels = segments[junction_segments[np.argmin(junction_segment_lengths)]]
            skeleton[branch_pixels[:, 0], branch_pixels[:, 1]] = 0
    return skeleton

def prune_skeleton(skeleton: np.ndarray) -> np.ndarray:
    """Prunes branches of skeletonized prediction

    Parameters
    ----------
    skeleton : np.ndarray
        pixel-wide thinned prediction

    Returns
    -------
    np.ndarray
        Skeleton of prediction with side branches removed
    """    
    pruned, segmented_skeleton, segmented_objs = pcv.morphology.prune(skeleton, size = 30)
    fill_mask = pcv.morphology.fill_segments(mask = pruned, objects = list(segmented_objs))
    junctions = pcv.morphology.find_branch_pts(skel_img = pruned)
    junction_pts = np.argwhere(junctions == 255)
    _ = pcv.morphology.segment_path_length(segmented_img = segmented_skeleton, objects = segmented_objs)
    segment_lengths = pcv.outputs.observations['default']['segment_path_length']['value']
    segments = get_segments(filled_mask = fill_mask)
    pruned = prune_branches(skeleton = pruned, junctions = junction_pts, segments = segments, path_lengths = segment_lengths)
    return pruned

def find_segment_ends(segments: dict, skeleton: np.ndarray) -> dict:
    """Finds one tip of each line segment in skeletonized skeleton. Does not handle closed rings or loops

    Parameters
    ----------
    segments : dict
        pixel positions of the individual segments, output from the function get_segments
    skeleton : np.ndarray
        skeletonized predictions

    Returns
    -------
    dict
        The pixel corresponding to one end of a line segment
    """    
    segment_ends = {}
    max_pixel = skeleton.shape[0] - 1
    for id, segment in segments.items():
        segment_ends[id] = segment[0]
        for pixel in segment:
            if (pixel[0] == 0) or (pixel[1] == 0) or (pixel[0] == max_pixel) or (pixel[1] == max_pixel):
                segment_ends[id] = pixel
            else:
                neighbours = skeleton[pixel[0] - 1:pixel[0] + 2, pixel[1] - 1:pixel[1] + 2]
                if np.sum(neighbours) == 2:
                    segment_ends[id] = pixel
    return segment_ends

def npz_to_lines(tif_tile, npz, threshold = None):
    """Converts ndarray to lines

    Parameters
    ----------
    tif_dir : Path
        path to reference tif tile
    npz : Path
        path to prediction (probabilities)
    threshold : float, optional
        threshold considered for binarizing model output, by default None

    Returns
    -------
    MultiLineString
        vector
    """    
    with rio.open(tif_tile) as tif:
        transform = tif.transform
    
    with np.load(npz) as data:
        tile = data['arr_0'][:, :, 0]

    if threshold:
        encoded = threshold_predictions(tile, threshold)
    else:
        encoded = tile

    skeleton = prune_skeleton(skeletonize(encoded).astype('uint8'))
    pruned, _, segment_objects = pcv.morphology.prune(skel_img = skeleton, size = 0)
    filled = pcv.morphology.fill_segments(mask = pruned, objects = list(segment_objects))
    segments = get_segments(filled_mask = filled)
    segment_ends = find_segment_ends(segments = segments, skeleton = pruned)
    vectors = raster_to_lines(segments = segments, segment_ends = segment_ends, transform = transform)
    vectors = unary_union(smoothen_line(vector = vectors, sigma = 10))
    return vectors

def create_df(geoms, names, dataset):
    """Saves geometries in a GeoDataFrame

    Parameters
    ----------
    geoms : list of MultiLineString
         grounding lines
    names : list of str
        names of tiles
    dataset : list of str
        string describing if tile is part of the validation or test set

    Returns
    -------
    GeoDataFrame
        data frame of geometries
    """    
    df = gpd.GeoDataFrame({'tile': names, 'geometry': geoms, 'dataset': dataset}, crs = 'EPSG:3031')
    return df

def create_predictions_df_predict_only(interferograms: list[pathlib.Path], tiles_dir: pathlib.Path, tif_dir: pathlib.Path, opt_threshold: np.float16 = 0.9) -> gpd.GeoDataFrame:
    predictions_df = gpd.GeoDataFrame(columns=['UUID', 'geometry'], geometry = 'geometry', crs = 'EPSG:3031')
    geometry = []
    uuid = []
    for interferogram in tqdm(interferograms):
        filtered = []
        interferogram_name = interferogram.name.removesuffix('.tif')
        if 'GEO' in interferogram_name:
            interferogram_name = interferogram_name[:interferogram_name.rfind('_GEO')]
        if 'stats' in interferogram_name:
            interferogram_name = interferogram_name[:interferogram_name.rfind('_stats')]
        
        uuid.append(interferogram_name)
        predicted_tiles = list(tiles_dir.glob(interferogram_name + '*.npz'))
        for prediction in predicted_tiles:
            tile_number = re.findall(r'tile_\d+', prediction.name)[-1]
            tif_name = prediction.name[:prediction.name.index(tile_number)] + tile_number + '.tif'
            tif_tile = list(tif_dir.glob(tif_name))[0]
            filtered.append(npz_to_lines(tif_tile, prediction, opt_threshold))
        if filtered:
            geometry.append(unary_union(filtered))
        else:
            geometry.append(np.nan)
    
    predictions_df.geometry = geometry
    predictions_df.UUID = uuid
    return predictions_df

def create_predictions_df(tiles_dir: pathlib.Path, labels_dir: pathlib.Path, gl_df: gpd.GeoDataFrame,  opt_threshold: np.float16 = 0.9) -> gpd.GeoDataFrame:
    """Converts npz predictions to vectors

    Parameters
    ----------
    gl_df : gpd.GeoDataFrame
        ground truth GeoDataFrame
    tiles_dir : pathlib.Path
        Predictions directory
    labels_dir : pathlib.Path
        Ground truth labels directory (rasters)
    opt_threshold : np.float16
        threshold value (between 0 - 1) to binarize predictions

    Returns
    -------
    gpd.GeoDataFrame
        data frame of prediction vectors
    """    
    predictions_df = gpd.GeoDataFrame(columns=['UUID', 'dataset', 'geometry'], geometry = 'geometry', crs = 'EPSG:3031')
    predictions_df[['UUID', 'dataset']] = gl_df[['UUID', 'dataset']]

    geometry = []

    for interferogram in gl_df.UUID.values:
        filtered = []
        predicted_tiles = list(tiles_dir.glob(interferogram + '_tile_*.npz'))
        for prediction in predicted_tiles:
            tile_number = re.findall(r'tile_\d+', prediction.name)[-1]
            tif_name = prediction.name[:prediction.name.index(tile_number)] + tile_number + '.tif'
            tif_tile = list(labels_dir.glob(tif_name))[0]
            filtered.append(npz_to_lines(tif_tile, prediction, opt_threshold))
        if filtered:
            geometry.append(unary_union(filtered))
        else:
            geometry.append(np.nan)
    
    predictions_df.geometry = geometry

    return predictions_df

def remove_outliers(predictions_df, reference, distance_threshold):
    """Removes false detections

    Parameters
    ----------
    predictions_df : GeoDataFrame
        predictions from neural network
    reference : pathlib.Path
        location of ADD simple basemap or measures GL
    distance_threshold : int
        buffer distance within which all predictions are preserved

    Returns
    -------
    GeoDataFrame
        filtered predictions
    """
    if 'dataset' in predictions_df.columns:
        copy_cols = ['UUID', 'dataset']
    else:
        copy_cols = ['UUID']
    filtered_df = deepcopy(predictions_df[copy_cols])
    reference_df = gpd.read_file(reference).to_crs('EPSG:3031')
    filtered_geoms = []
    for _, row in tqdm(filtered_df.iterrows()):
        sample = predictions_df.loc[predictions_df.UUID == row.UUID]
        try:
            clip_box = box(*sample.total_bounds)
            clipped_basemap = reference_df.clip(clip_box)
            basemap_outline = clipped_basemap.boundary.difference(clip_box.boundary)
            basemap_outline = basemap_outline.buffer(distance = distance_threshold)
            basemap_outline = gpd.GeoDataFrame(geometry = basemap_outline, crs = 'epsg:3031')
            filtered_geoms.append(sample.intersection(basemap_outline, align = False).geometry.values[0])
        except:
            filtered_geoms.append(sample.geometry.values[0])

    filtered_df = gpd.GeoDataFrame(filtered_df, geometry = filtered_geoms, crs = 'EPSG:3031')
    return filtered_df

def reorder_points(start_point: Point, remaining_points: list) -> list:
    """Reorders points in line (nearest neighbours)

    Parameters
    ----------
    start_point : Point
        on of the ends of the line segment
    remaining_points : list
        rest of the points in the line segment

    Returns
    -------
    list
        reordered points
    """    
    points_reordered = []
    points_reordered.append(start_point)
    index = 1
    while len(remaining_points) != 0:
        current_point = points_reordered[index - 1]
        distances = [np.sqrt(np.square(current_point[0] - point[0]) + np.square(current_point[1] - point[1])) for point in remaining_points]
        points_reordered.append(remaining_points[np.argmin(distances)])
        remaining_points.remove(points_reordered[index])
        index = index + 1
    return points_reordered

def raster_to_lines(segments: dict, segment_ends: dict, transform: Affine) -> list:
    """Converts skeletonized raster to line strings, handles loops and rings too

    Parameters
    ----------
    segments : dict
        skeletonized prediction segments, output of get_segments
    segment_ends : dict
        pixels that correspond to end of the segments, output of get_segment_ends
    transform : Affine
        affine transform matrix of prediction tile

    Returns
    -------
    list
       line strings
    """    
    transformer = rio.transform.AffineTransformer(transform)
    line_segments = []
    for id, segment in segments.items():
        if id != 0 and len(segment) > 1:
            points = []
            for pixel in segment:
                if (pixel != segment_ends[id]).all():
                    points.append(tuple(pixel))
            points = reorder_points(segment_ends[id], points)
            if len(points) > 1:
                points = [Point(transformer.xy(pixel[0], pixel[1])) for pixel in points]
                line_segments.append(make_valid(LineString(points)))
    return line_segments

def smoothen_line(vector: list, points: np.int16 = 1000, sigma: np.float16 = 10) -> list:
    """Interpolates line coordinates and applies 1D gaussian smoothing

    Parameters
    ----------
    vector : list
        linestrings
    points : np.int16, optional
        total no. of coordinates after interpolation, by default 1000
    sigma : np.float16, optional
        kernel for 1D gaussian, by default 10

    Returns
    -------
    list
        smoothened linestrings
    """    
    interpolated_lines = []
    for line in vector:
        x, y = line.xy
        x = list(x)
        y = list(y)
        values = np.arange(len(x))

        interp_x = CubicSpline(x = values, y = x, bc_type = 'natural')
        interp_y = CubicSpline(x = values, y = y, bc_type = 'natural')
        interpolate_values = np.linspace(values[0], values[-1], points)
        interpolated_x = interp_x(interpolate_values)
        interpolated_y = interp_y(interpolate_values)

        smooth_x = gaussian_filter1d(interpolated_x, sigma)
        smooth_y = gaussian_filter1d(interpolated_y, sigma)
        interp_x = CubicSpline(x = interpolate_values, y = smooth_x, bc_type = 'natural')
        interp_y = CubicSpline(x = interpolate_values, y = smooth_y, bc_type = 'natural')
        interpolated_x = interp_x(values)
        interpolated_y = interp_y(values)
        interpolated_coordinates = [Point(x, y) for x, y in zip(interpolated_x, interpolated_y)]
        interpolated_line = LineString(interpolated_coordinates)
        interpolated_lines.append(interpolated_line)
    return interpolated_lines

def merge_tiles(to_be_merged, savepath, descriptions = None, delete_orig = True):
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
    meta = opened[0].meta.copy()
    meta.update({'width': merged.shape[2], 'height': merged.shape[1], 'transform': transform, 'count': merged.shape[0]})

    if not descriptions:
        descriptions = np.repeat('', merged.shape[0])
    
    with rio.open(savepath, 'w', **meta) as dest:
        for band in range(merged.shape[0]):
            dest.write_band(band + 1, merged[band, :, :])
            dest.set_band_description(band + 1, descriptions[band])
    
    if delete_orig:
        for tile in to_be_merged:
            tile.unlink()