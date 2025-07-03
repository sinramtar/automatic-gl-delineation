import numpy as np
import rasterio as rio
from rasterio.features import rasterize
from sklearn.metrics import balanced_accuracy_score
import geopandas as gpd

import matplotlib.pyplot as plt

def get_predictions_and_label_pixels(samples, ground_truth_tif_dir):
    predictions = []
    ground_truths = []
    for sample in samples:
        with np.load(sample) as data:
            predictions.append(data['arr_0'][:, :, 0])
        ground_truth = np.zeros(shape = (256, 256), dtype = np.uint8)
        with rio.open(ground_truth_tif_dir / (sample.name.removesuffix('.npz') + '.tif')) as tif:
            ground_truth[:tif.height, :tif.width] = tif.read(1)
        ground_truths.append(ground_truth)
    predictions = np.concatenate(predictions).flatten()
    ground_truths = np.concatenate(ground_truths).flatten()
    return predictions, ground_truths

def compute_confidence(predicted_probs):
    return (2 * np.maximum(predicted_probs, 1 - predicted_probs)) - 1

def shannon_entropy(predicted_probs, eps = 1e-6, C = 2):
    predicted_probs = np.clip(predicted_probs, eps, 1 - eps)
    return -np.sum(predicted_probs * np.log(predicted_probs) / np.log(C), axis = -1)

def shannon_confidence(predicted_probs):
    return 1- shannon_entropy(predicted_probs)

def get_pixels_in_buffer(samples, ground_truth_tif_dir, gls, buffer = 1000):
    predictions = []
    ground_truths = []
    for sample in samples:
        ground_truth = np.zeros(shape = (256, 256), dtype = np.uint8)
        with rio.open(ground_truth_tif_dir / (sample.name.removesuffix('.npz') + '.tif')) as tif:
            bounds = tif.bounds
            ground_truth[:tif.height, :tif.width] = tif.read(1)
            
        gt_vector = gls.clip_by_rect(*bounds)
        buffered_gt_vector = gt_vector.buffer(distance = buffer)
        gl_values = buffered_gt_vector.geometry.values
        gl_values = gl_values[~(gl_values.is_empty | gl_values.isna())]
        shapes = ((geom, 1) for geom in gl_values)
        
        buffered_gt_mask = rasterize(shapes = shapes, out_shape = (256, 256), transform = tif.transform, 
                                                  all_touched = True, dtype = 'uint8')
        with np.load(sample) as data:
            predictions.append(data['arr_0'][buffered_gt_mask == 1, 0])
        ground_truths.append(ground_truth[buffered_gt_mask == 1])
    
    predictions = np.concatenate(predictions).flatten()
    ground_truths = np.concatenate(ground_truths).flatten()
    return predictions, ground_truths

def compute_confidence_and_balanced_accuracy(predictions, ground_truths, bins):
    accuracies = []
    confidences = []
    for index, bin in enumerate(bins[:-1]):
        predictions_thresholded = np.zeros_like(predictions)
        predictions_thresholded[predictions >= bin] = 1
        pixels = np.logical_and(predictions >= bin, predictions < bins[index + 1])
        predictions_in_bin = predictions[pixels]
        ground_truths_in_bin = ground_truths[pixels]
        accuracies.append(balanced_accuracy_score(y_true = ground_truths_in_bin, y_pred = predictions_thresholded[pixels]))
        confidences.append(np.mean(predictions_in_bin))
    return accuracies, confidences

def compute_confidence_and_accuracy(predictions, ground_truths, bins, confidence_measure = 'default'):
    accuracies = []
    confidences = []
    for index, bin in enumerate(bins[:-1]):
        predictions_thresholded = np.zeros_like(predictions)
        predictions_thresholded[predictions >= bin] = 1
        accuracy = predictions_thresholded == ground_truths
        pixels = np.logical_and(predictions >= bin, predictions < bins[index + 1])
        predictions_in_bin = predictions[pixels]
        if confidence_measure == 'default':
            confidence = compute_confidence(predictions_in_bin)
        elif confidence_measure == 'shannon':
            confidence = shannon_confidence(predictions_in_bin) 
        accuracies.append(np.sum(accuracy[pixels]) / np.sum(pixels))
        confidences.append(np.mean(confidence))
    return accuracies, confidences

def plot_reliability_diagram(predictions, ground_truths, confidence_measure = 'default', class_balanced = False):
    bins = np.arange(0, 1.1, 0.1)
    fig = plt.figure(figsize = (10, 5), tight_layout = True)
    ax1 = fig.add_subplot(1, 2, 1)
    hist, bins = np.histogram(predictions, bins = bins)
    hist = hist / len(predictions)
    width = bins[1] - bins[0]
    center = (bins[:-1] + bins[1:]) / 2
    ax1.bar(x = center, height = hist, width = width, align = 'center')
    ax1.set_title('Distribution of per-pixel model confidence')
    ax1.set_xlabel('Confidence')
    ax1.set_ylabel('Percentage of samples')
    
    if class_balanced:
        accuracies, confidences = compute_confidence_and_balanced_accuracy(predictions = predictions, ground_truths = ground_truths, bins = bins)
    else:        
        accuracies, confidences = compute_confidence_and_accuracy(predictions = predictions, ground_truths = ground_truths, bins = bins, confidence_measure = confidence_measure)
    print(f'Confidences: {confidences}')
    print(f'Accuracies: {accuracies}')
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(confidences, accuracies, 'bo', markersize = 5)
    # ax2.set_xticks(ticks = bins)
    ax2.set_ylim((0.0, 1.0))
    ax2.set_xlabel('Confidence')
    ax2.set_ylabel('Accuracy')

def get_average_std_predictions(predictions):
    ensemble_predictions = []
    for prediction in predictions:
        with np.load(prediction) as data:
            ensemble_predictions.append(data['arr_0'][:, :, 0])
    
    ensemble_predictions = np.stack(ensemble_predictions, axis = -1)
    ensemble_mean = np.mean(ensemble_predictions, axis = -1)
    ensemble_std = np.std(ensemble_predictions, axis = -1)
    return ensemble_mean, ensemble_std