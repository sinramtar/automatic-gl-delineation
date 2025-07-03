import numpy as np
from tqdm import tqdm
import rasterio as rio
from rasterio.features import shapes
from shapely import Polygon, Point, union_all, line_interpolate_point


def count_pixels_in_mask(ensemble_df, ensemble_outputs_dir, sigma = 2):
    pixels_in_mask = 0
    num_pixels = 0
    for sample in ensemble_df.UUID.values:
        if (ensemble_outputs_dir / (sample + '_stats.tif')).is_file():
            with rio.open(ensemble_outputs_dir / (sample + '_stats.tif')) as tif:
                stddev = tif.read(2)
                mean = tif.read(1)
                transform = tif.transform
            
            mean_plus_std = mean + (sigma * stddev)
            mean_plus_std_mask = (mean_plus_std > 0)
            mean_mask = mean > 0
            
            mean_gl = ensemble_df.loc[ensemble_df.UUID == sample]
            shapes = ((geom, 1) for geom in mean_gl.geometry.values)
            
            # mean_gl_raster = rasterize(shapes = shapes, out_shape = mean.shape, transform = transform, all_touched = True, dtype = 'uint8')
            pixels_in_mask = pixels_in_mask + np.sum(mean_mask[mean_plus_std_mask])
            num_pixels = num_pixels + np.sum(mean_mask)
    return pixels_in_mask /num_pixels
    
def get_average_std_predictions(tiles: list) -> tuple[np.ndarray, np.ndarray]:
    """Computes per sample mean and standard deviation of ensemble members predictions

    Parameters
    ----------
    tiles : list
        ensemble members outputs for one sample

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ensemble mean, ensemble standard deviation
    """ 
    predictions = []
    for tile in tiles:
        with np.load(tile) as data:
            predictions.append(data['arr_0'])
        
    predictions = np.concatenate(predictions, axis = -1)
    ensemble_mean = predictions.mean(axis = -1)
    ensemble_stddev = predictions.std(axis = -1)
    return ensemble_mean, ensemble_stddev

def polygonize_uncertainties(thresholded_stats, ref_tif, mean_gl):
    with rio.open(ref_tif) as tif:
        transform = tif.transform
    
    geoms = [{'properties': {'cluster_id': int(v)}, 'geometry': s} 
            for (s, v) in (shapes(thresholded_stats, mask = thresholded_stats.astype(bool), transform = transform, 
                                  connectivity = 8))]
    
    polygons = [Polygon(shape['geometry']['coordinates'][0]) for shape in geoms]
    filtered_polygons = union_all([polygon for line_geom in mean_gl.explode(index_parts = True).geometry.values 
                         for polygon in polygons if line_geom.intersects(polygon)])
    
    return filtered_polygons

def measure_approx_stddev(ensemble_mean_df, stddev_df):
    polygon_boundaries = stddev_df.boundary
    per_sample_stddevs = []
    
    for uuid in tqdm(ensemble_mean_df.UUID.values):
        mean_line = ensemble_mean_df.loc[ensemble_mean_df.UUID == uuid]
        distances = []
        for line in mean_line.explode(index_parts = True).geometry.values:
            line = line.segmentize(10000)
            ten_km_segments = np.arange(0, line.length, 10000)
            line_coords = [line_interpolate_point(line, distance = distance) for distance in ten_km_segments]
            distances.append([coords.distance(polygon_boundaries).min() for coords in line_coords])
        per_sample_stddevs.append([d for dist in distances for d in dist])
        
    return per_sample_stddevs