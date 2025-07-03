#-----------------------------------------------------------------------------------
#	
#	script       : visualization of everything
#   author       : Sindhu Ramanath Tarekere
#   date	     : 14 Dec 2021
#
#-----------------------------------------------------------------------------------

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.cm import ScalarMappable

import pathlib
from adjustText import adjust_text

import numpy as np
import geopandas as gpd
from scipy import signal
from shapely import box, STRtree, intersection

import pandas as pd
import rasterio as rio
from rasterio.windows import from_bounds

import matplotlib.gridspec as gs
import matplotlib.patches as patches

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib_map_utils.core.scale_bar import ScaleBar

from cartopy.mpl.geoaxes import GeoAxes
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl import ticker as c_ticker

import seaborn as sns

import postprocess.vectorize as vec


def plot_features_4326(raster_orig, raster_interferometric_features, inset_params, figparams = {'figsize': (25, 5), 'dpi': 300}):
    
    with rio.open(raster_orig) as tif:
        phase = tif.read(1)
        phase[phase == 0] = np.nan
        roi_bbox = gpd.GeoDataFrame(geometry = [box(xmin = tif.bounds.left, ymin = tif.bounds.bottom, xmax = tif.bounds.right, ymax = tif.bounds.top)], crs = 'EPSG:4326')
    
    roi_bbox = roi_bbox.to_crs('EPSG:3031')
    
    with rio.open(raster_interferometric_features) as tif:
        real = tif.read(1)
        imaginary = tif.read(2)

    pseudo_backscatter = np.ones_like(phase, dtype = np.uint8)
    
    fig, axs = plt.subplots(nrows = 2, ncols = 2, figsize = figparams['figsize'], dpi = figparams['dpi'], tight_layout = True)
    ph = axs[0, 0].imshow(phase, cmap = 'hsv', interpolation = 'None')
    fig.colorbar(ph, ax = axs[0, 0], pad = 0.046, fraction = 0.046, label = 'radians')
    axs[0, 0].set_xticks([])
    axs[0, 0].set_yticks([])

    plot_antarctica_inset(ax = axs[0, 1], inset_params = inset_params, roi_bbox = roi_bbox)
    ps = axs[0, 1].imshow(pseudo_backscatter, cmap = 'binary_r')
    fig.colorbar(ps, ax = axs[0, 1], pad = 0.046, fraction = 0.046)
    axs[0, 1].set_xticks([])
    axs[0, 1].set_yticks([])
    
    re = axs[1, 0].imshow(real, cmap = 'hsv', interpolation = 'None')
    fig.colorbar(re, ax = axs[1, 0], pad = 0.046, fraction = 0.046)
    axs[1, 0].set_xticks([])
    axs[1, 0].set_yticks([])

    im = axs[1, 1].imshow(imaginary, cmap = 'hsv', interpolation = 'None')
    fig.colorbar(im, ax = axs[1, 1], pad = 0.046, fraction = 0.046)
    axs[1, 1].set_xticks([])
    axs[1, 1].set_yticks([])
    
    return fig

def plot_resampled_features(raster_3031, insets, figparams = {'figsize': (25, 5), 'dpi': 300}):
    
    inset_bounds = insets.bounds
    inset_1_bounds = inset_bounds.loc[0].values
    inset_2_bounds = inset_bounds.loc[1].values
    with rio.open(raster_3031) as tif:
        features = tif.read()
        extent = [tif.bounds.left, tif.bounds.right, tif.bounds.bottom, tif.bounds.top]
        
        window1 = from_bounds(left = inset_1_bounds[0], right = inset_1_bounds[2], bottom = inset_1_bounds[1], top = inset_1_bounds[3], transform = tif.transform)
        inset1 = tif.read(window = window1)
        
        window2 = from_bounds(left = inset_2_bounds[0], right = inset_2_bounds[2], bottom = inset_2_bounds[1], top = inset_2_bounds[3], transform = tif.transform)
        inset2 = tif.read(window = window2)
    
    crs_3031 = ccrs.SouthPolarStereo()
    crs_4326 = ccrs.PlateCarree()
    
    fig = plt.figure(figsize = figparams['figsize'], dpi = 300, tight_layout = True)
    grid = gs.GridSpec(nrows = 3, ncols = 4)
    
    ax1 = fig.add_subplot(grid[0,:2], projection = crs_3031)
    g1 = ax1.gridlines(draw_labels = {'bottom': 'x', 'left': 'y'}, linestyle = '--', linewidth = 0.5, color = 'dimgray', crs = crs_4326, dms = True, x_inline = False, y_inline = False,
                        xlabel_style = {'size': 5, 'color': 'gray'}, ylabel_style = {'size': 5, 'color': 'gray'}, xpadding = 3, ypadding = 3, rotate_labels = False)
    
    g1.ylocator = c_ticker.LatitudeLocator(nbins = 'auto')
    g1.xlocator = c_ticker.LongitudeLocator(nbins = 'auto')
    re = ax1.imshow(features[0, :, :], extent = extent, cmap = 'hsv', interpolation = 'None')
    fig.colorbar(re, ax = ax1, pad = 0.046, fraction = 0.046)
    
    ax2 = fig.add_subplot(grid[0, 2:], projection = crs_3031)
    g2 = ax2.gridlines(draw_labels = {'bottom': 'x', 'left': 'y'}, linestyle = '--', linewidth = 0.5, color = 'dimgray', crs = crs_4326, dms = True, x_inline = False, y_inline = False,
                        xlabel_style = {'size': 5, 'color': 'gray'}, ylabel_style = {'size': 5, 'color': 'gray'}, xpadding = 3, ypadding = 3, rotate_labels = False)
    
    g2.ylocator = c_ticker.LatitudeLocator(nbins = 'auto')
    g2.xlocator = c_ticker.LongitudeLocator(nbins = 'auto')
    im = ax2.imshow(features[1, :, :], extent = extent, cmap = 'hsv', interpolation = 'None')
    fig.colorbar(im, ax = ax2, pad = 0.046, fraction = 0.046)
    
    ax3_1 = fig.add_subplot(grid[2, 0], projection = crs_3031)
    ax3_1.imshow(inset1[3, :, :], cmap = 'hsv', extent = [inset_1_bounds[0], inset_1_bounds[2], inset_1_bounds[1], inset_1_bounds[3]]) 
    
    ax3_2 = fig.add_subplot(grid[2, 1], projection = crs_3031)
    ax3_2.imshow(inset1[2, :, :], cmap = 'gray', extent = [inset_1_bounds[0], inset_1_bounds[2], inset_1_bounds[1], inset_1_bounds[3]])
    
    ax4_1 = fig.add_subplot(grid[2, 2], projection = crs_3031)
    ax4_1.imshow(inset2[3, :, :], cmap = 'hsv', extent = [inset_2_bounds[0], inset_2_bounds[2], inset_2_bounds[1], inset_2_bounds[3]]) 
    
    ax4_2 = fig.add_subplot(grid[2, 3], projection = crs_3031)
    ax4_2.imshow(inset2[2, :, :], cmap = 'gray', extent = [inset_2_bounds[0], inset_2_bounds[2], inset_2_bounds[1], inset_2_bounds[3]])  
    
    
    ax5 = fig.add_subplot(grid[1, :2], projection = crs_3031)
    g5 = ax5.gridlines(draw_labels = {'bottom': 'x', 'left': 'y'}, linestyle = '--', linewidth = 0.5, color = 'dimgray', crs = crs_4326, dms = True, x_inline = False, y_inline = False,
                        xlabel_style = {'size': 5, 'color': 'gray'}, ylabel_style = {'size': 5, 'color': 'gray'}, xpadding = 3, ypadding = 3, rotate_labels = False)
    
    g5.ylocator = c_ticker.LatitudeLocator(nbins = 'auto')
    g5.xlocator = c_ticker.LongitudeLocator(nbins = 'auto')
    ps = ax5.imshow(features[2, :, :], extent = extent, cmap = 'gray', interpolation = 'None')
    insets.boundary.plot(ax = ax5, color = 'blue')
    fig.colorbar(ps, ax = ax5, pad = 0.046, fraction = 0.046)
    
    ax6 = fig.add_subplot(grid[1, 2:], projection = crs_3031)
    g6 = ax6.gridlines(draw_labels = {'bottom': 'x', 'left': 'y'}, linestyle = '--', linewidth = 0.5, color = 'dimgray', crs = crs_4326, dms = True, x_inline = False, y_inline = False,
                        xlabel_style = {'size': 5, 'color': 'gray'}, ylabel_style = {'size': 5, 'color': 'gray'}, xpadding = 3, ypadding = 3, rotate_labels = False)
    
    phase = features[3, :, :]
    phase[phase == 0] = np.nan
    g6.ylocator = c_ticker.LatitudeLocator(nbins = 'auto')
    g6.xlocator = c_ticker.LongitudeLocator(nbins = 'auto')
    ph = ax6.imshow(phase, extent = extent, cmap = 'hsv', interpolation = 'None')
    fig.colorbar(ph, ax = ax6, pad = 0.046, fraction = 0.046, label = 'radians')
    
    scalebar = ScaleBar(location = 'lower left', style = "boxes", bar = {"projection":3031, "minor_type":"none", 'height': 0.05, 'major_div': 2, 'max': 20}, 
                        labels = {'style': 'minor_first', 'loc': 'above', 'pad': 0},
                        units = {'loc': 'text', 'pad': 0}, text = {'stroke_color': 'black', 'textcolor': 'black', 'fontsize': 5, 'stroke_width': 0.3, 'fontweight': 'light', 'fontfamily': 'sans-serif'})
    ax6.add_artist(scalebar)
    
    return fig
    

def plot_postprocessing_example(prediction, transform, figparams = {'figsize': (25, 5), 'dpi': 300}):
    fig = plt.figure(figsize = figparams['figsize'], dpi = figparams['dpi'], tight_layout = True)
    ax1 = fig.add_subplot(1, 5, 1)
    
    pred = ax1.imshow(prediction, cmap = 'Blues')
    fig.colorbar(pred, ax = ax1, label = 'probability', pad = 0.046, fraction = 0.046)
    ax1.axis('off')
    
    binarized = np.zeros_like(prediction)
    binarized[prediction >= 0.8] = 1

    ax2 = fig.add_subplot(1, 5, 2)
    ax2.imshow(binarized, cmap = 'binary')
    ax2.axis('off')

    filtered = signal.medfilt2d(binarized, kernel_size = 11)
    ax3 = fig.add_subplot(1, 5, 3)
    ax3.imshow(filtered, cmap = 'binary')
    ax3.axis('off')
    
    skeleton = vec.skeletonize(filtered)
    ax4 = fig.add_subplot(1, 5, 4)
    ax4.imshow(skeleton, cmap = 'binary')
    ax4.axis('off')
    
    skeleton = vec.prune_skeleton(skeleton.astype('uint8'))
    pruned, _, segment_objects = vec.pcv.morphology.prune(skel_img = skeleton, size = 0)
    filled = vec.pcv.morphology.fill_segments(mask = pruned, objects = list(segment_objects))
    segments = vec.get_segments(filled_mask = filled)
    segment_ends = vec.find_segment_ends(segments = segments, skeleton = pruned)
    vectors = vec.raster_to_lines(segments = segments, segment_ends = segment_ends, transform = transform)
    vectors = gpd.GeoDataFrame(geometry = vec.smoothen_line(vector = vectors, sigma = 10), crs = 'EPSG:3031')
    
    ax5 = fig.add_subplot(1, 5, 5)
    vectors.plot(ax = ax5, color = 'black', linewidth = 1.5)
    ax5.axis('off')
    return fig

def plot_preprocessing_example(features_tif, gl, tile_tif, basemap, figparams = {'figsize': (25, 10), 'dpi': 300}):
    crs_3031 = ccrs.SouthPolarStereo()
    
    fig, axs = plt.subplots(nrows = 2, ncols = 6, figsize = figparams['figsize'], dpi = figparams['dpi'], subplot_kw = {'projection': crs_3031}, tight_layout = True)
    
    with rio.open(tile_tif) as tif:
        tile_features = tif.read()
        tile_bbox = box(xmin = tif.bounds.left, ymin  = tif.bounds.bottom, xmax = tif.bounds.right, ymax = tif.bounds.top)
        tile_bbox_df = gpd.GeoDataFrame(geometry = [tile_bbox], crs = 'EPSG:3031')
    
    with rio.open(features_tif) as tif:
        phase = tif.read(4)
        phase[phase == 0] = np.nan
        basemap_roi = basemap.clip(tif.bounds)
    
    extent = [tif.bounds.left, tif.bounds.right, tif.bounds.bottom, tif.bounds.top]
    axs[0, 0].imshow(phase, cmap = 'hsv', extent = extent)
    axs[0, 0].imshow(np.ones_like(phase, dtype = np.uint8), cmap = 'binary_r', alpha = 0.2, extent = extent)
    tile_bbox_df.boundary.plot(ax = axs[0, 0], color = 'red', linewidth = 1.5)
    axs[0, 0].axis('off')
    
    basemap_cmap = mpl.colors.ListedColormap(['lightblue', 'blue', 'white'])
    basemap_roi.plot(ax = axs[1, 0], cmap = basemap_cmap, column = 'Category', categorical = True)
    gl.plot(ax = axs[1, 0], color = 'black', linewidth = 1.2)
    tile_bbox_df.boundary.plot(ax = axs[1, 0], color = 'red', linewidth = 1.5)
    
    gl_part = gl.clip(tile_bbox)
    axs[0, 1].imshow(tile_features[3], cmap = 'hsv', extent = [tile_bbox.bounds[0], tile_bbox.bounds[2], tile_bbox.bounds[1], tile_bbox.bounds[3]])
    gl_part.plot(ax = axs[1, 1], color = 'black', linewidth = 1.5)
    axs[1, 1].axis('off')
    axs[1, 1].spines['top'].set_visible(False)
    axs[1, 1].spines['bottom'].set_visible(False)
    axs[1, 1].spines['left'].set_visible(False)
    axs[1, 1].spines['right'].set_visible(False)
    
    fig = plot_features(features = tile_features, fig = fig, axs = axs[:, 2:].flatten())
    return fig
    

def plot_antarctic_wide_deviations_with_ref_gl(ref_gl, ais_cci, polis_distances, cmap = 'viridis', radarsat_mosaic = None):
    ref_gl_segments = ref_gl.boundary.segmentize(5000).explode(index_parts = True)

    cci_parts = ais_cci.explode(index_parts = True)

    polis_parts = polis_distances.explode(column = 'deviations')

    tree = STRtree(cci_parts.geometry.values)
    deviations = polis_parts.deviations.values

    cmap = mpl.cm.get_cmap(cmap)
    cumulative_polis_deviations = []
    plot_geoms = []

    for segment in ref_gl_segments.geometry.values:
        index = tree.query_nearest(segment, max_distance = 300)
        num_parts = len(index)
        if (num_parts > 0) and not (pd.isnull(deviations[index])).all():
            cumulative_polis_deviations.append(np.nanmedian(deviations[index]))
            plot_geoms.append(segment)
        else:
            continue
    
    norm = plt.Normalize(vmin = np.nanmin(cumulative_polis_deviations), vmax = np.nanmax(cumulative_polis_deviations))
    colors = [cmap(norm(distance)) for distance in cumulative_polis_deviations]

    crs_3031 = ccrs.SouthPolarStereo()
    crs_4326 = ccrs.PlateCarree()

    fig, ax = plt.subplots(figsize = (8, 8), dpi = 300, nrows = 1, ncols = 1, subplot_kw = {'projection': crs_3031})
    grid = ax.gridlines(draw_labels = True, linestyle = '--', linewidth = 0.5, color = 'gray', crs = crs_4326, dms = True, x_inline = False, y_inline = False)
    grid.top_labels = False
    grid.right_labels = False

    if radarsat_mosaic:
        with rio.open(radarsat_mosaic) as tif:
            window = from_bounds(left = ais_cci.total_bounds[0], right = ais_cci.total_bounds[2], bottom = ais_cci.total_bounds[1], top = ais_cci.total_bounds[3], transform = tif.transform)
            mosaic = tif.read(1, window = window)
            
        ax.imshow(mosaic, cmap = 'gray', interpolation = 'None', alpha = 0.8, extent = [ais_cci.total_bounds[0], ais_cci.total_bounds[2], ais_cci.total_bounds[1], ais_cci.total_bounds[3]])
    
    df = gpd.GeoDataFrame(geometry = plot_geoms, crs = 'EPSG:3031')
    df['deviations'] = cumulative_polis_deviations

    df.plot(ax = ax, color = colors, linewidth = 0.8)
    sm = ScalarMappable(norm = norm, cmap = cmap)
    fig.colorbar(mappable = sm, ax = ax, pad = 0.046, fraction = 0.046)
    return fig

def plot_antarctic_wide_deviations_graph(rois, polis_distances, gls, feature_subsets = []):
    plot_df = []
    for lines, polis in zip(gls, polis_distances):
        polis = polis.loc[polis.dataset == 'test', ['UUID', 'deviations']]
        lines = lines.loc[lines.UUID.isin(polis.UUID.values)]
        ensemble_roi_grouped = lines.sjoin_nearest(rois)

        ensemble_roi_polis = pd.DataFrame(data = {'Region of Interest': ensemble_roi_grouped.ROI.values})
        for roi in ensemble_roi_grouped.ROI.values:
            uuids_in_roi = ensemble_roi_grouped.loc[ensemble_roi_grouped.ROI == roi, 'UUID'].values
            ensemble_roi_polis.loc[ensemble_roi_polis['Region of Interest'] == roi, 'Polis [m]'] = polis.loc[polis.UUID.isin(uuids_in_roi), 'deviations'].values

        ensemble_roi_polis = ensemble_roi_polis.explode(column = ['Polis [m]'])
        ensemble_roi_polis = ensemble_roi_polis.dropna()
        plot_df.append(ensemble_roi_polis)

def color_line_by_deviation(lines: gpd.GeoDataFrame, deviations: pd.DataFrame, ax: plt.Axes, cmap: str = 'seismic', **lc_kwargs: dict):
    lines_exploded = lines.explode(index_parts = True)
    for line in lines_exploded.geometry.values:
        x = line.coords.x
        y = line.coords.y

def plot_features(features, fig = None, axs = None, scalebar = False, grid = False):
    """Plots sample features

    Parameters
    ----------
    features : ndarray (1, height, width, channels)
        sample
    extent: list
        plot bounds
    """
    
    crs_3031 = ccrs.SouthPolarStereo()
    crs_4326 = ccrs.PlateCarree()

    if not fig and axs:
        fig, axs = plt.subplots(figsize = (15, 10), nrows = 2, ncols = 4, dpi = 300, subplot_kw = {'projection': crs_3031}, tight_layout = True)
        axs = axs.flatten()

    num_features = features.shape[-1]
    titles = np.array(['Real', 'Imaginary', 'Pseudo coherence', 'Phase', 'DEM', 'Ice velocity easting', 'Ice velocity northing', 'Differential tide level'])
    cmaps = np.array(['hsv', 'hsv', 'gray', 'hsv', 'terrain', 'RdBu_r', 'RdBu_r', 'viridis'])
    labels = np.array(['', '', '', 'radians', 'm', 'm/d', 'm/d', 'm'])
    
    for channel, title, cmap, label in zip(range(num_features), titles, cmaps, labels):
        
        axs[channel].set_facecolor('white')
        axs[channel].set_title(title, fontsize = 16)
        
        if grid:
            gl = axs[channel].gridlines(draw_labels = True, linestyle = '--', linewidth = 0.5, color = 'gray', crs = crs_4326, dms = True, x_inline = False, y_inline = False)
            gl.top_labels = False
            gl.right_labels = False
        
        plot = axs[channel].imshow(features[channel, :, :], cmap = cmap, interpolation = 'None', transform = crs_3031)
        
        fig.colorbar(plot, ax = axs[channel], label = label, fraction = 0.046, pad = 0.04)
        axs[channel].axis('off')

    if scalebar:
        scalebar = ScaleBar(dx = 1, units = 'm', dimension = 'si-length', location = 'lower right')
        axs[channel].add_artist(scalebar)
    
    return fig
    
def plot_predictions(ground_truth, prediction, gt_to_pred, pred_to_gt):
    """Plots ground truth and prediction vectors, colors them according to their distances

    Parameters
    ----------
    ground_truth : GeoDataFrame
        ground truth geometries
    prediction : GeoDataFrame
        prediction geometries
    gt_to_pred : ndarray
        distance from ground truth segments to corresponding prediction segments
    pred_to_gt : ndarray
        distance from prediciton segments to corresponding ground truth segments
    """    
    
    fig = plt.figure(figsize = (10, 10))
    plt.style.use('dark_background')
    ax = fig.add_subplot(1, 1, 1)

    gt_min = np.nanmin(gt_to_pred)
    gt_max = np.nanmax(gt_to_pred)
    pred_min = np.nanmin(pred_to_gt)
    pred_max = np.nanmax(pred_to_gt)
    
    if pred_min != np.nan and pred_max != np.nan and gt_min != np.nan and gt_max != np.nan:
        sm_gt = plt.cm.ScalarMappable(cmap = 'winter', norm = plt.Normalize(vmin =  gt_min, vmax = gt_max))
        sm_pred = plt.cm.ScalarMappable(cmap = 'autumn', norm = plt.Normalize(vmin =  pred_min, vmax = pred_max))  
    else:
        sm_gt = plt.cm.ScalarMappable(cmap = 'winter')
        sm_pred = plt.cm.ScalarMappable(cmap = 'autumn') 

    sm_gt._A = []
    fig.colorbar(sm_gt, ax = ax, label = 'ground truth to prediction (m)', location = 'left')
    ground_truth.plot(cmap = 'winter', ax = ax)

    sm_pred._A = []
    fig.colorbar(sm_pred, ax = ax, label = 'prediction to ground truth (m)')
    prediction.plot(cmap = 'autumn', ax = ax)
    fig.tight_layout()

def plot_antarctica_inset(ax, inset_params, roi_bbox):
    crs_3031 = ccrs.SouthPolarStereo()
    crs_4326 = ccrs.PlateCarree()
    
    antarctica_inset = inset_axes(ax, width = inset_params['width'], height =inset_params['height'], loc = inset_params['loc'], 
                                  axes_class = GeoAxes, axes_kwargs = dict(projection = crs_3031), 
                                  bbox_to_anchor = inset_params['bbox'], bbox_transform = ax.transAxes)
    
    antarctica_inset.set_extent([-170, 170, -85, -65], crs = crs_4326)
    antarctica_inset.add_feature(cfeature.LAND, color = '#f3f1f0')
    antarctica_inset.add_feature(cfeature.OCEAN, color = '#a3bdd1')
    antarctica_inset.spines['left'].set_linewidth(0.5)
    antarctica_inset.spines['right'].set_linewidth(0.5)
    antarctica_inset.spines['top'].set_linewidth(0.5)
    antarctica_inset.spines['bottom'].set_linewidth(0.5)
    
    roi_bbox.plot(ax = antarctica_inset, color = inset_params['bbox_color'])
        
    adjust_text([antarctica_inset.text(x = roi_bbox.total_bounds[1], y = roi_bbox.total_bounds[3], s = inset_params['roi_label'], fontsize = 7)])
    

def plot_colorbar(fig, ax, params, sm):
    colorbar_ax = inset_axes(ax, width = params['colorbar_width'], height = params['colorbar_height'], loc = params['colorbar_loc'], 
                             bbox_transform = ax.transAxes, bbox_to_anchor = params['bbox'])
    cbar = fig.colorbar(sm, cax = colorbar_ax, pad = 0.046, fraction = 0.046, orientation = params['colorbar_orientation'])
    cbar.set_label(label = params['colorbar_label'], size = 5, labelpad = -0.1)
    cbar.set_ticks(params['colorbar_ticks'], labels = params['colorbar_ticklabels'], size = 5)
    cbar.ax.tick_params(length = 0, pad = 0.1)
    cbar.outline.set_linewidth(0.5)


def overlay_gls_on_dinsar_subplots(manual_line: gpd.GeoDataFrame, pred_lines: list[gpd.GeoDataFrame], features_stack: pathlib.Path,
                          deviations: list[np.float32], coverages: list[np.float32],
                          radarsat_mosaic: pathlib.Path,
                          figure_params: dict,
                          feature_plot_params: dict,
                          antarctica_inset_params: dict,
                          metrics_display_params: dict,
                          scalebar_params: dict,
                          uncertainty_buffer: gpd.GeoSeries = None,
                          zoom_ins: list = [],
                          zoom_ins_params: dict = {},
                          shade_subplots: bool = False,
                          uncertainty_buffer_params: dict = {'color': 'maroon', 'alpha': 0.6}):
    """Plots grounding lines against interferogram and radarsat mosaic backdrop (very fancy).

    Parameters
    ----------
    gt : GeoDataFrame
        ground truth geometry (MultiLineString)
    shps : list of GeoDataFrame
        list of predicted grounding line geometries
    labels : str
        labels for plot legend
    feature : xarray Dataset
        feature tif opened as xarray Dataset (one band)
    plot_resources : Path
        directory containing background plot stuff
    title : str
        plot title
    scale : str, optional
        map scale, defaults to ''
    feature_cmap : str
        plot colormap, defaults to 'cyclic'
    colorbar_label : str
        colorbar label
    
    Returns
    -------
    Figure
        a very fancy plot
    """  

    crs_3031 = ccrs.SouthPolarStereo()
    crs_4326 = ccrs.PlateCarree()

    gt_bounds = manual_line.total_bounds
    if uncertainty_buffer is not None:
        gt_bounds = [gt_bounds[0] - 2000, gt_bounds[1] - 2000, gt_bounds[2] + 2000, gt_bounds[3] + 2000]
        
    plot_extent = [gt_bounds[0], gt_bounds[2], gt_bounds[1], gt_bounds[3]]
    
    with rio.open(radarsat_mosaic) as tif:
        window = from_bounds(left = gt_bounds[0], right = gt_bounds[2], bottom = gt_bounds[1], top = gt_bounds[3], transform = tif.transform)
        mosaic = tif.read(1, window = window)
    
    tif = rio.open(features_stack)
    window = from_bounds(left = gt_bounds[0], right = gt_bounds[2], bottom = gt_bounds[1], top = gt_bounds[3], transform = tif.transform)
    feature = tif.read(4, window = window)
        
    feature[feature == 0] = np.nan
    roi_bbox = gpd.GeoDataFrame(geometry=[box(xmin = gt_bounds[0], xmax = gt_bounds[2], ymin = gt_bounds[1], ymax = gt_bounds[3])], crs = 'EPSG:3031')
    
    total_subplots = len(pred_lines) + len(zoom_ins)
    assert (figure_params['rows'] + figure_params['columns']) > total_subplots, 'Enter right num of rows and cols!'

    fig = plt.figure(figsize = figure_params['size'], dpi = 300)
    grid = gs.GridSpec(nrows = figure_params['rows'], ncols = figure_params['columns'])

    for index, (prediction, deviation, coverage, pos, title) in enumerate(zip(pred_lines, deviations, coverages, figure_params['subplot_pos'], figure_params['title'])):
        ax = fig.add_subplot(grid[pos[0], pos[1]], projection = crs_3031)
        ax.set_title(title)
        gl = ax.gridlines(draw_labels = {'bottom': 'x', 'left': 'y'}, linestyle = '--', linewidth = 0.5, color = 'dimgray', crs = crs_4326, dms = True, x_inline = False, y_inline = False,
                          xlabel_style = {'size': 5, 'color': 'gray'}, ylabel_style = {'size': 5, 'color': 'gray'}, xpadding = 3, ypadding = 3, rotate_labels = False)
        
        gl.ylocator = c_ticker.LatitudeLocator(nbins = 'auto')
        gl.xlocator = c_ticker.LongitudeLocator(nbins = 'auto')
                
        ax.imshow(mosaic, alpha = 0.7, cmap = 'gray', interpolation = 'None', extent = plot_extent)
        phase = ax.imshow(feature, alpha = feature_plot_params['alpha'], cmap = feature_plot_params['cmap'], interpolation = 'None', extent = plot_extent)
        if shade_subplots:
            ax.imshow(np.ones_like(feature, dtype = np.uint8), cmap = 'binary_r', alpha = 0.2, extent = plot_extent)
        
        if uncertainty_buffer is not None:
            uncertainty_buffer[index].plot(ax = ax, color = uncertainty_buffer_params['color'], alpha = uncertainty_buffer_params['alpha'])
    
        manual_line.plot(ax = ax, color = 'white', linewidth = 1)
        prediction = gpd.GeoSeries(prediction.clip_by_rect(*manual_line.total_bounds), crs = 'EPSG:3031')
        prediction.plot(ax = ax, linewidth = 1, color = 'black')
        
        # Deviation and coverage box
        ax.text(x = metrics_display_params['loc'][0], y = metrics_display_params['loc'][1], s = f'{deviation:.2f} m \n {(coverage * 100):.2f}%', 
                        bbox={'facecolor': 'white', 'alpha': 1, 'pad': 3},
                        transform = ax.transAxes, fontdict = {'size': metrics_display_params['fontsize']})
        
    main_axes = fig.get_axes()
    scalebar = ScaleBar(location = scalebar_params['loc'], style="boxes", bar={"projection":3031, "minor_type":"none", 'height': 0.05, 'major_div': 2, 'max': scalebar_params['max']}, 
                        labels = {'style': 'minor_first', 'loc': 'above', 'pad': 0},
                        units = {'loc': 'text', 'pad': 0}, text = {'stroke_color': 'black', 'textcolor': 'black', 'fontsize': 5, 'stroke_width': 0.3, 'fontweight': 'light', 'fontfamily': 'sans-serif'})
    main_axes[0].add_artist(scalebar)
    feature_plot_params['colorbar_ticks'] = [np.pi, -np.pi]
    feature_plot_params['colorbar_ticklabels'] = [r'-$\pi$', r'$\pi$']
    feature_plot_params['tick_color'] = 'black'
    plot_colorbar(fig = fig, ax = main_axes[0], params = feature_plot_params, sm = phase)
    plot_antarctica_inset(ax = main_axes[0], inset_params = antarctica_inset_params, roi_bbox = roi_bbox)
    
    if len(zoom_ins) > 0:
        gt_bbox = box(xmin = gt_bounds[0], ymin = gt_bounds[1], xmax = gt_bounds[2], ymax = gt_bounds[3])
        num_subplot = 0
        for bbox in zoom_ins.geometry.values:
            gt_box = manual_line.clip(bbox)
            window = from_bounds(left = bbox.bounds[0], bottom = bbox.bounds[1], right = bbox.bounds[2], 
                                 top = bbox.bounds[3], transform = tif.transform)
            extent = [bbox.bounds[0], bbox.bounds[2], bbox.bounds[1], bbox.bounds[3]]
            zoomed_phase = tif.read(4, window = window)
            zoomed_phase[zoomed_phase == 0] = np.nan
            for main_ax_index, pred in enumerate(pred_lines):
                pos = zoom_ins_params['subplot_pos'][num_subplot]
                pred_box = pred.clip(bbox)
                ax = fig.add_subplot(grid[pos[0], pos[1]], projection = crs_3031)
                ax.text(x = 0.01, y = 0.9, s = f'({chr(97 + num_subplot)})', transform = ax.transAxes)
                # ax.set_title(f'({chr(97 + num_subplot)})', loc = 'left', y = 0.8)
                ax.imshow(zoomed_phase, cmap = 'hsv', interpolation = 'None', extent = extent, alpha = 0.4)
                ax.imshow(np.ones_like(zoomed_phase, dtype = np.uint8), cmap = 'binary_r', extent = extent, alpha = 0.3)
                
                if not gt_box.is_empty.all():
                    gt_box.plot(ax = ax, color = 'white', linewidth = 1.0)
                if not pred_box.is_empty.all():
                    pred_box.plot(ax = ax, color = 'black', linewidth = 1.0)
                if uncertainty_buffer is not None:
                    uncertainty_buffer_box = uncertainty_buffer[main_ax_index].clip(bbox)
                    uncertainty_buffer_box.plot(ax = ax, color = uncertainty_buffer_params['color'], alpha = 0.4)
                
                ax.set_yticks([])
                ax.set_xticks([])
                
                bbox_boundary = gpd.GeoDataFrame(geometry = [intersection(bbox, gt_bbox).boundary], crs = 'EPSG:3031')
                bbox_boundary.plot(ax = main_axes[main_ax_index], color = 'blue', linewidth = 1.0)
                adjust_text([main_axes[main_ax_index].text(x = bbox.bounds[zoom_ins_params['label_pos'][0]], y = bbox.bounds[zoom_ins_params['label_pos'][1]], 
                                              s = f'{chr(97 + num_subplot)}', fontdict = {'size': 7},
                                              bbox = {'facecolor': 'white', 'alpha': 1})], ax = main_axes[main_ax_index])
                num_subplot = num_subplot + 1
    
    tif.close()

    return fig

def overlay_gls_on_dinsar_single(manual_line: gpd.GeoDataFrame = None,
                                pred_lines: gpd.GeoDataFrame = None, 
                                features_stack: list[pathlib.Path] = None, 
                                radarsat_mosaic: pathlib.Path = None, 
                                deviations: np.float32 = None, 
                                coverages: np.float32 = None,
                                figure_params: dict = None,
                                feature_plot_params: dict = None,
                                antarctica_inset_params: dict = None,
                                metrics_display_params: dict = None,
                                scalebar_params: dict = None):
    
    crs_3031 = ccrs.SouthPolarStereo()
    crs_4326 = ccrs.PlateCarree()
    
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = figure_params['size'], dpi = 300, subplot_kw = {'projection': crs_3031})
    scalebar = ScaleBar(location = scalebar_params['loc'], style="boxes", bar={"projection":3031, "minor_type":"none", 'height': 0.05, 'major_div': 2, 'max': scalebar_params['max']}, 
                        labels = {'style': 'minor_first', 'loc': 'above', 'pad': 0},
                        units = {'loc': 'text', 'pad': 0}, text = {'stroke_color': 'black', 'textcolor': 'black', 'fontsize': 5, 'stroke_width': 0.3, 'fontweight': 'light', 'fontfamily': 'sans-serif'})
    
    if np.all(manual_line) is not None:
        bounds = manual_line.total_bounds
        manual_line.plot(ax = ax, color = 'white', zorder = 2.5)
    else:
        bounds = pred_lines.total_bounds

    plot_extent = [bounds[0], bounds[2], bounds[1], bounds[3]]
    roi_bbox = gpd.GeoDataFrame(geometry=[box(xmin = bounds[0], xmax = bounds[2], ymin = bounds[1], ymax = bounds[3])], crs = 'EPSG:3031')
    
    grid = ax.gridlines(draw_labels = {'bottom': 'x', 'left': 'y'}, linestyle = '--', linewidth = 0.5, color = 'dimgray', crs = crs_4326, dms = True, x_inline = False, y_inline = False,
                          xlabel_style = {'size': 5, 'color': 'gray'}, ylabel_style = {'size': 5, 'color': 'gray'}, xpadding = 3, ypadding = 3, rotate_labels = False)
        
    grid.ylocator = c_ticker.LatitudeLocator(nbins = 'auto')
    grid.xlocator = c_ticker.LongitudeLocator(nbins = 'auto')
    
    if radarsat_mosaic:
        with rio.open(radarsat_mosaic) as tif:
            window = from_bounds(left = bounds[0], right = bounds[2], bottom = bounds[1], top = bounds[3], transform = tif.transform)
            mosaic = tif.read(1, window = window)
            ax.imshow(mosaic, cmap = 'gray', extent = plot_extent, interpolation = 'None', alpha = 0.7)
    
    if len(features_stack) >= 1:     
        for dd in features_stack:
            with rio.open(dd) as tif:
                window = from_bounds(left = bounds[0], right = bounds[2], bottom = bounds[1], top = bounds[3], transform = tif.transform)
                dd_phase = tif.read(4, window = window)
                dd_phase[dd_phase == 0] = np.nan
                ph = ax.imshow(dd_phase, cmap = 'hsv', interpolation = 'None', extent = plot_extent, alpha = 0.4)
    
    if pred_lines:
        pred_lines.plot(ax = ax, color = 'black', linewidth = 0.8)
        
    ax.add_artist(scalebar)
    feature_plot_params['colorbar_ticks'] = [np.pi, -np.pi]
    feature_plot_params['colorbar_ticklabels'] = [r'-$\pi$', r'$\pi$']
    feature_plot_params['tick_color'] = 'black'
    plot_colorbar(fig = fig, ax = ax, params = feature_plot_params, sm = ph)
    
    if antarctica_inset_params:
        plot_antarctica_inset(ax = ax, inset_params = antarctica_inset_params, roi_bbox = roi_bbox)
    
    return fig
    

def plot_ensemble_example(predictions: list, stats_tif: pathlib.Path, cmap = 'magma'):
    fig = plt.figure(figsize = (15, 7), dpi = 300, tight_layout = True)
    
    with rio.open(stats_tif) as tif:
        ensemble_stats = tif.read()
        extent_stats = [tif.bounds.left, tif.bounds.right, tif.bounds.bottom, tif.bounds.top]
    
    for index, prediction in enumerate(predictions):
        with rio.open(prediction) as tif:
            pred = tif.read(1)
        ax = fig.add_subplot(2, 4, index + 1)
        ax.set_title(f'Model {index + 1} output', fontdict = {'size': 12})
        ax.imshow(pred, cmap = cmap, interpolation = 'None', extent = extent_stats, vmin = 0, vmax = 1)
        ax.set_xticks([])
        ax.set_yticks([])

    ax6 = fig.add_subplot(2, 4, 6)
    ax6.set_title('Mean prediction', fontdict = {'size': 12})
    ax6.imshow(ensemble_stats[0, :, :], cmap = cmap, interpolation = 'None', extent = extent_stats, vmin = 0, vmax = 1)
    ax6.set_xticks([])
    ax6.set_yticks([])
    divider1 = make_axes_locatable(ax6)
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    sm = ScalarMappable(norm = plt.Normalize(vmin = 0, vmax = 1), cmap = mpl.colormaps[cmap])
    fig.colorbar(sm, cax = cax1)
    
    ax7 = fig.add_subplot(2, 4, 7)
    ax7.set_xticks([])
    ax7.set_yticks([])
    
    ax7.set_title('Standard deviation of predictions', fontdict = {'size': 12})
    stddev = ax7.imshow(ensemble_stats[1, :, :], cmap = 'plasma', interpolation = 'None', extent = extent_stats)
    divider2 = make_axes_locatable(ax7)
    cax2 = divider2.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(stddev, cax = cax2)
    
    return fig
    

    