#-----------------------------------------------------------------------------------
#	
#	script       : metrics for evaluation of ML models
#   author       : Sindhu Ramanath Tarekere
#   date	     : 10 Dec 2021
#
#-----------------------------------------------------------------------------------

import numpy as np
import pandas as pd

from scipy.stats import median_abs_deviation
from shapely.geometry import Point
import rasterio as rio
from tqdm import tqdm
import pathlib

import geopandas as gpd
from shapely import box, is_geometry, intersection

import torch
from torchmetrics import AveragePrecision, F1Score

def plot_pr_curve(precisions, recalls, ax, label):
    """Plots Precision-Recall curve 

    Parameters
    ----------
    precisions : ndarray (1, )
        precision for different thresholds
    recalls : ndarray (1, )
        recall for corresponding thresholds
    ax : matplotlib.pyplot.axes
        plot axis 
    label : str
        label for plot

    Returns
    -------
    matplotlib.pyplot.axes
        axis object 
    """    
    ax.set_title('PR curve')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.plot(recalls, precisions, label = label)
    ax.legend()
    return ax

def flatten_ground_truth_and_predictions(tiles_dir: pathlib.Path, labels_dir: pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
    """Collects ground truth and predicted tiles and flattens them in order to compute ODS F1 score

    Parameters
    ----------
    tiles_dir : pathlib.Path
        predictions directory
    labels_dir : pathlib.Path
        ground truth directory

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        flattened ground truth and prediction tiles arrays
    """    
    y_true = []
    y_pred = []
    for tile in tiles_dir.glob('*.npz'):
        if '_tile' in tile.name:
            with np.load(tile) as pred:
                pred_arr = pred['arr_0'][:, :, 0]
            with np.load(labels_dir / (tile.name)) as gd:
                gd_arr = gd['arr_0'][:, :, 0]

            y_pred.append(pred_arr)
            y_true.append(gd_arr)
    y_pred = np.concatenate([row.flatten() for row in y_pred])
    y_true = np.concatenate([row.flatten() for row in y_true])
    return y_true, y_pred

def compute_f1score(precision, recall):
    """Computes F1 scores from precision and recall

    Parameters
    ----------
    precision : array like
        precision values
    recall : array like
        recall values for corresponding thresholds

    Returns
    -------
    array like
        f1 scores
    """    
    return (2 * precision * recall) / (precision + recall)

def compute_precision_recall_from_confusion_matrix(y_true, y_pred, threshold):
    """Calculates false positive rate given ground truth, prediction probabilities and threshold

    Parameters
    ----------
    y_true : list of ndarray (1, )
        ground truth, each list element corresponds to one sample
    y_pred : list of ndarray (1, )
        prediction probability, each list element corresponds to one sample
    threshold : float
        threshold to binarize y_pred

    Returns
    -------
    float
        false positive rate
    """    
    y_true = y_true.reshape(1, -1).astype('uint8')
    y_pred = y_pred.reshape(1, -1)

    y_pred = encode_prediction_proba(y_pred, threshold)
    tp = np.sum(np.logical_and(y_pred == 1, y_true == 1))
    fp = np.sum(np.logical_and(y_pred == 1, y_true == 0))
    fn = np.sum(np.logical_and(y_pred == 0, y_true == 1))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    return precision, recall

def compute_precision_recall(y_true, y_pred, thresholds):
    """Calculates precision and recall values given ground truth, predicitions and thresholds.
    The scores are weighted by the frequency of classes

    Parameters
    ----------
    y_true : ndarray
        ground truth, 1 - gl pixel, 0 - non gl pixel
    y_pred : ndarray
        predicted probabilities
    thresholds : ndarray
        thresholds for which recall and precision are calculated

    Returns
    -------
    (list, list)
        precision and recall scores
    """    
    y_true = y_true.reshape(1, -1).astype('uint8')
    y_pred = y_pred.reshape(1, -1)
    precision_scores = []
    recall_scores = []

    for threshold in thresholds:
        precision, recall = compute_precision_recall_from_confusion_matrix(y_true, y_pred, threshold)
        precision_scores.append(precision)
        recall_scores.append(recall)
    return np.asarray(precision_scores), np.asarray(recall_scores)

def encode_prediction_proba(y_pred, threshold):
    """Converts labels  equal to or greater than threshold to 1, else 0

    Parameters
    ----------
    y_pred : ndarray (1, )
        predicted probabilities
    threshold : float
        probabilty threshold (0 - 1)

    Returns
    -------
    ndarray (1, )
        predictions
    """ 
    encoded = np.zeros_like(y_pred, dtype = 'uint8')   
    encoded[y_pred >= threshold] = 1
    return encoded

def compute_ods_f1(y_true, y_pred, thresholds):
    """Calculates threshold for which F1 score is maximum

    Parameters
    ----------
    y_true : ndarray (1, )
        ground truth
    y_pred : ndarray (1, )
        predicted probabilities
    thresholds : ndarray (1, )
        thresholds for which to compute f1 score

    Returns
    -------
    (float, float)
        ods score and corresponding threshold
    """ 

    precision_scores, recall_scores = compute_precision_recall(y_true, y_pred, thresholds)
    f1_scores = compute_f1score(precision_scores, recall_scores)
    if not np.all(np.isnan(f1_scores)):
        ods_ind = np.nanargmax(f1_scores)
    else:
        ods_ind = -1
    ods_f1 = f1_scores[ods_ind]
    best_threshold = thresholds[ods_ind]
    return ods_f1, best_threshold

def compute_ois_f1(y_true, y_pred, thresholds):
    """Calculate Optimal Image Set F1 score

    Parameters
    ----------
    y_true : ndarray
        ground truth
    y_pred : ndarray
        prediction probabilities
    thresholds : ndarray (1, )
        thresholds for which to compute f1 score

    Returns
    -------
    float
        f1 score
    """    
    scores = np.zeros((y_true.shape[0], thresholds.shape[0]))
    for sample in range(y_true.shape[0]):
        precision_scores, recall_scores = compute_precision_recall(y_true[sample, :], y_pred[sample, :], thresholds)
        scores[sample, :] = compute_f1score(precision_scores, recall_scores)
    
    sample_maximums = np.nanmax(scores, axis = 0)
    return np.nanmean(sample_maximums)

def compute_mad(deviations):
    """Calculates Median Absolute Deviation

    Parameters
    ----------
    deviations : ndarray (1, )
        distance between predictions and ground truth

    Returns
    -------
    float
        median absolute deviation
    """    
    return median_abs_deviation(deviations, nan_policy = 'omit')

def compute_polis_points(ground_truth, prediction):
    """Calculates deviation between ground truth and prediction using the polis algorithm

    Parameters
    ----------
    ground_truth : GeoDataFrame
        data frame containing ground truth geometries (MultiPoint)
    predictions : GeoDataFrame
        data frame containing prediction geometries (MultiPoint)

    Returns
    -------
    float32, float32, float32
        prediction to ground truth deviation, ground truth to prediction deviation, total deviation
    """
    num_geoms = len(ground_truth)
    pred_to_gt = np.zeros((num_geoms))
    gt_to_pred = np.zeros_like(pred_to_gt)

    for index in range(num_geoms):
        pred = prediction.iloc[index]
        gt = ground_truth.iloc[index]

        pred_distances = []
        gt_distances = []
        for point in pred.geometry.geoms:
            pred_distances.append(point.distance(gt.geometry))
        
        pred_to_gt[index] = np.nanmean(pred_distances)

        for point in gt.geometry.geoms:
            gt_distances.append(point.distance(pred.geometry))
        
        gt_to_pred[index] = np.nanmean(gt_distances)      

    deviation = np.mean(pred_to_gt + gt_to_pred)
    return pred_to_gt, gt_to_pred, deviation

def compute_polis_lines(ground_truth, prediction):
    """Calculates deviation between ground truth and prediction using the polis algorithm

    Parameters
    ----------
    ground_truth : GeoDataFrame
        data frame containing ground truth geometries (MultiLineString)
    predictions : GeoDataFrame
        data frame containing prediction geometries (MultiLineString)

    Returns
    -------
    list, list. list
        prediction to ground truth deviation, ground truth to prediction deviation, overall deviation
    """
    pred_to_gt = []
    gt_to_pred = []
    deviations = []
    for sample in tqdm(prediction.UUID.values):
        pred_row = prediction.loc[prediction.UUID == sample]
        gt_row = ground_truth.loc[ground_truth.UUID == sample]
        if pred_row.is_empty.all() or gt_row.is_empty.all() or pred_row.geometry.isna().all():
            pred_to_gt.append(np.nan)
            gt_to_pred.append(np.nan)
            deviations.append(np.nan)
        else:
            gt_exploded = gt_row.explode(index_parts = True)
            num_gt_lines = len(gt_exploded)
            pred_dists = np.full(num_gt_lines, fill_value = np.nan, dtype = np.float16)
            gt_dists = np.full(num_gt_lines, fill_value = np.nan, dtype = np.float16)
            for index, line in enumerate(gt_exploded.geometry):
                try:
                    if not (line.is_empty) or not (is_geometry(line)):
                        nearest_pred = pred_row.clip_by_rect(*line.bounds)
                        pred_dist = 0
                        gt_dist = 0
                        pred_coords = nearest_pred.get_coordinates()
                        if len(pred_coords) > 0:
                            for coord_index in range(len(pred_coords)):
                                x = pred_coords.x.iloc[coord_index]
                                y = pred_coords.y.iloc[coord_index]
                                geom = Point(x, y)
                                pred_dist += geom.distance(line)
                            pred_dists[index] = pred_dist / len(pred_coords)

                            gt_coords = list(line.coords)
                            for gt_coord_index in range(len(gt_coords)):
                                geom = Point(gt_coords[gt_coord_index]) 
                                gt_dist += geom.distance(nearest_pred)
                            gt_dists[index] = gt_dist.iloc[0] / len(gt_coords)
                except:
                    continue
                
            pred_to_gt.append(pred_dists)
            gt_to_pred.append(gt_dists)
            overall_deviations = (gt_dists + pred_dists) / 2
            deviations.append(overall_deviations)
    return pred_to_gt, gt_to_pred, deviations

def compute_coverage(ground_truth, prediction, line_metrics):
    """Computes the percentage of ground truth length covered by prediction line

    Parameters
    ----------
    ground_truth : GeoDataFrame
        ground truth geometries
    prediction : GeoDataFrame
        prediction geometries
        
    Returns
    -------
    ndarray (len(ground_truth), len(buffers))
        percentage coverage
    """
    
    coverage = []
    for uuid in line_metrics.UUID.values:
        prediction_sample = prediction.loc[prediction.UUID == uuid]
        gt_sample = ground_truth.loc[ground_truth.UUID == uuid]
        distances = line_metrics.loc[line_metrics.UUID == uuid].deviations.values[0]
        if not np.all(np.isnan(distances)):
            buffer_distance = np.nanmedian(distances)
            buffered_sample = gt_sample.buffer(buffer_distance)
            intersecting_geom = intersection(buffered_sample.geometry.values, prediction_sample.geometry.values)[0]
            prediction_length = intersecting_geom.length
            gt_length = gt_sample.geometry.length.values[0]
            coverage.append((prediction_length / gt_length).T)
        else:
            coverage.append(np.nan)
    
    line_metrics['coverage'] = coverage
    return line_metrics

def pixels_in_buffer(gt_vector, buffer_distance, gt_raster, pred_npz):
    buffered = gt_vector.buffer(buffer_distance, resolution = 2)
    shapes = buffered.geometry.values

    with rio.open(gt_raster) as src:
        _, _, window =  rio.mask.raster_geometry_mask(src, shapes, crop = True, all_touched = True)
        y_true = src.read(1, window = window)
    
    y_pred = pred_npz[window.row_off:window.row_off + window.height, window.col_off: window.col_off + window.width]
    
    return y_true, y_pred

def compute_testset_f1_ap(predictions: list, labels: list, thresholds: list, opt_threshold: np.float16 = 0.8) -> tuple[float, float]:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    thresholds = torch.from_numpy(thresholds).to(device)

    ap = AveragePrecision(task = 'binary', thresholds = thresholds, average = 'weighted').to(device)
    f1 = F1Score(task = 'binary', threshold = opt_threshold, average = 'weighted').to(device)
    
    preds_tensor = torch.zeros((len(predictions), 1, 256, 256), dtype = torch.float32, device = device)
    gd_tensor = torch.zeros((len(predictions), 1, 256, 256), dtype = torch.long, device = device)
    
    for index, (prediction, label) in tqdm(enumerate(zip(predictions, labels))):
        preds_tensor[index, 0, :, :] = torch.from_numpy(prediction).float()
        gd_tensor[index, 0, :, :] = torch.from_numpy(label).long()

    average_precision = ap(preds_tensor, gd_tensor)
    ods_f1 = f1(preds_tensor, gd_tensor)
    return average_precision.cpu().detach().numpy(), ods_f1.cpu().detach().numpy()

def compute_dataset_metrics(df: gpd.GeoDataFrame, savemetrics: pathlib.Path) -> None:
    subset_polis = df[['dataset', 'pred_to_gt', 'gt_to_pred', 'deviations']]
    exploded_polis = subset_polis.explode(column = ['pred_to_gt', 'gt_to_pred', 'deviations'])
    exploded_polis = exploded_polis.dropna()
    exploded_polis[['pred_to_gt', 'gt_to_pred', 'deviations']] = exploded_polis[['pred_to_gt', 'gt_to_pred', 'deviations']].apply(lambda x: x.astype(np.float32))

    grouped_polis = exploded_polis.groupby(by = 'dataset')
    means = grouped_polis.mean()
    medians = grouped_polis.median()
    mads = grouped_polis.apply(lambda x: pd.Series(compute_mad(x)))
    mads.columns = ['pred_to_gt', 'gt_to_pred', 'deviations']
    metrics = pd.concat([means, medians, mads], keys = ['mean', 'median', 'mad'], axis = 1)
    
    if 'stddev' in df.columns:
        stddev_subset = df[['dataset', 'stddev']]
        exploded_stddev = stddev_subset.explode(column = ['stddev'])
        exploded_stddev = exploded_stddev.dropna()
        exploded_stddev[['stddev']] = exploded_stddev[['stddev']].apply(lambda x: x.astype(np.float32))
        grouped_stddev = exploded_stddev.groupby(by = 'dataset')
        avg_stddev = grouped_stddev.mean()
        metrics = pd.concat([metrics, avg_stddev], axis = 1)

    np.save(savemetrics, metrics.to_dict())

def clip_geometries_for_evaluation(ground_truth_df: gpd.GeoDataFrame, split: list, labels_tifs_dir: pathlib.Path) -> gpd.GeoDataFrame:
    """Clips ground truth GL geometries to those exactly used in the DL dataset. This is to ensure fair evaluation of model
    performance

    Parameters
    ----------
    ground_truth_df : gpd.GeoDataFrame
        Ground truth geometries
    split : list
        Feature tiles included in dataset
    labels_tifs_dir : pathlib.Path
        Ground truth labels as tifs

    Returns
    -------
    gpd.GeoDataFrame
        Dataframe with clipped geometries
    """      
    valid_ground_truths = ground_truth_df.loc[(ground_truth_df.dataset == 'train') | (ground_truth_df.dataset == 'validation') | 
                                              (ground_truth_df.dataset == 'test')]
    uuids = valid_ground_truths.UUID.values

    for uuid in uuids:
        double_difference = valid_ground_truths.loc[valid_ground_truths.UUID == uuid]
        
        boxes = []
        tiles_of_sample = [tile for tile in split if uuid in tile]
        for tile in tiles_of_sample:
            with rio.open(labels_tifs_dir / (tile.removesuffix('.npz') + '.tif')) as tif:
                boxes.append(box(xmin = tif.bounds.left, xmax = tif.bounds.right, ymin = tif.bounds.bottom, ymax = tif.bounds.top))

        total_covered_area = gpd.GeoSeries(boxes, crs = 'EPSG:3031')
        clipped_geom = double_difference.clip_by_rect(xmin = total_covered_area.total_bounds[0], ymin = total_covered_area.total_bounds[1], 
                                                      xmax = total_covered_area.total_bounds[2], ymax = total_covered_area.total_bounds[3])
        
        valid_ground_truths.loc[valid_ground_truths.UUID == uuid, 'geometry'] = clipped_geom.make_valid()
    return valid_ground_truths