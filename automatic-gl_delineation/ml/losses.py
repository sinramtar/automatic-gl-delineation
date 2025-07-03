#-----------------------------------------------------------------------------------
#	
#	script       : losses for training ML models
#   author       : Sindhu Ramanath Tarekere
#   date	     : 10 Dec 2021
#
#-----------------------------------------------------------------------------------

import torch
from abc import ABC, abstractmethod

class Loss(ABC, torch.nn.Module):
    def __init__(self) -> None:
        super(Loss, self).__init__()
    
    @abstractmethod
    def forward(self, preds: torch.Tensor, targets: torch.Tensor, kwargs: dict):
        pass


class Polis(Loss):
    def __init__(self, input_shape: tuple) -> None:
        super().__init__()
        self.euclidean_distance_tensor = self.get_euclidean_distance_tensor(input_shape = input_shape)

    def get_euclidean_distance_tensor(self, input_shape: tuple) -> torch.Tensor:
        indices = torch.arange(input_shape[0])
        indices_matrix = torch.tensor([(a, b) for a in indices for b in indices])
        indices_matrix = indices_matrix.float()

        return torch.cdist(indices_matrix, indices_matrix)

    def forward(self, preds: torch.Tensor, targets: torch.Tensor, kwargs: dict) -> torch.Tensor:
        gt_column = targets.reshape(targets.shape[0], targets.shape[1], -1, 1)
        preds_column = preds.reshape(targets.shape[0], targets.shape[1], -1, 1)

        gl_mask_pred = torch.zeros((targets.shape[0], targets.shape[1], self.euclidean_distance_tensor.shape[0], self.euclidean_distance_tensor.shape[1]),
                                    dtype = torch.bool)
        gl_mask_gt = torch.zeros((targets.shape[0], targets.shape[1], self.euclidean_distance_tensor.shape[0], self.euclidean_distance_tensor.shape[1]), 
                                 dtype = torch.bool)

        gl_mask_pred[:, :, :, (preds_column != 0).flatten()] = True
        gl_mask_gt[:, :, (gt_column != 0).flatten(), :] = True

        gl_mask = torch.logical_and(gl_mask_gt, gl_mask_pred)
        distance_matrix = torch.masked.masked_tensor(self.euclidean_distance_tensor, mask = gl_mask, requires_grad = True)
        
        gt_to_pred_min_distances = torch.amin(distance_matrix, dim = 1, keepdim = True)
        pred_to_gt_min_distances = torch.amin(distance_matrix, dim = 0, keepdim = True)
        
        polis = 0.5 * ((gt_to_pred_min_distances.mean(dim = (2, 3), keepdim = True)) + (pred_to_gt_min_distances.mean(dim = (2, 3), keepdim = True)))
        
        if kwargs['reduce_batch'] == 'mean':
            polis_batch = torch.mean(polis**2)
        else:
            polis_batch = torch.sum(polis**2)

        wce = WeightedCrossEntropy()
        wce_batch = wce(preds, targets, kwargs)
        
        combined = (kwargs['polis_weighting'] * polis_batch) + wce_batch
        return combined

class WeightedCrossEntropy(Loss):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, preds: torch.Tensor, targets: torch.Tensor, kwargs: dict):
        epsilon = kwargs['epsilon'] 
        preds = torch.clamp(preds, epsilon, 1 - epsilon)
        mask = targets.clone()

        if kwargs['class_balanced_weighting']:
            num_negatives = torch.sum(1. - targets, dim = (2, 3), keepdim = True)
            num_positives = torch.sum(targets, dim = (2, 3), keepdim = True)
            beta = num_negatives / (num_negatives + num_positives)
            weights_positives = beta * kwargs['weight_positives']
            weights_negatives = (1 - beta) * kwargs['weight_negatives']
        else: 
            weights_positives = kwargs['weight_positives']
            weights_negatives = kwargs['weight_negatives']
        
        mask = torch.where(mask == 1.0, weights_positives, weights_negatives)
        
        loss = torch.nn.BCELoss(weight = mask, reduction = kwargs['reduce_batch'])(preds, targets)
        
        return loss

class Focal(Loss):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, preds: torch.Tensor, targets: torch.Tensor, kwargs) -> torch.Tensor:
        wce = WeightedCrossEntropy()
        wce_loss = wce(preds, targets, kwargs)
        return ((1 - torch.exp(-1 * wce_loss)) ** kwargs['gamma']) * wce_loss

class DiceLoss(Loss):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, preds: torch.Tensor, targets: torch.Tensor, kwargs: dict):
        epsilon = kwargs['epsilon'] 
        preds = torch.clamp(preds, epsilon, 1 - epsilon)
        
        axes = [0] + list(range(2, len(preds.shape)))
        tp, fp, fn, _ = self.get_tp_fp_fn_tn(preds, targets, axes)
        
        numerator = (2 * tp)
        denominator = (2 * tp + fp + fn)
        
        dice_loss = numerator / denominator
        
        wce = WeightedCrossEntropy()
        wce_loss = wce(preds, targets, kwargs = kwargs)

        return (kwargs['dice_alpha'] * dice_loss) + (kwargs['dice_beta'] * wce_loss)
        
    def get_tp_fp_fn_tn(self, net_output, gt, axes = None):
        if axes is None:
            axes = tuple(range(2, len(net_output.size())))

        tp = net_output * gt
        fp = net_output * (1 - gt)
        fn = (1 - net_output) * gt
        tn = (1 - net_output) * (1 - gt)

        if len(axes) > 0:
            tp = torch.sum(tp, dim = axes, keepdim = False)
            fp = torch.sum(fp, dim = axes, keepdim = False)
            fn = torch.sum(fn, dim = axes, keepdim = False)
            tn = torch.sum(tn, dim = axes, keepdim = False)

        return tp, fp, fn, tn