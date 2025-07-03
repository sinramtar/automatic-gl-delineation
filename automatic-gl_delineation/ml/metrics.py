from typing import Any
import torch
from torchmetrics.classification import BinaryAveragePrecision, BinaryPrecisionRecallCurve, BinaryF1Score, BinaryJaccardIndex
from torchmetrics import Metric

import numpy as np

class ODSF1(Metric):
    def __init__(self) -> None:
        super().__init__()
        self.add_state('total', default = torch.tensor(0, dtype = torch.float32), dist_reduce_fx = "sum")
        self.add_state('optimal_threshold', default = torch.tensor(0, dtype = torch.float16), dist_reduce_fx = "sum")
    
    def update(self, preds: torch.Tensor, targets: torch.Tensor, **kwargs) -> torch.Tensor:
        assert preds.shape == targets.shape
        
        if kwargs['thresholds']:
            thresholds = kwargs['thresholds']
        else:
            thresholds = torch.linspace(0.0, 1.0, 10, dtype = torch.float, device = preds.device)
        prcurve = BinaryPrecisionRecallCurve(thresholds = thresholds).to(preds.device)
        precisions, recalls, thresholds = prcurve(preds, targets)
        f1_scores = torch.nan_to_num((2 * precisions * recalls) / (precisions + recalls))
        self.total = torch.max(f1_scores)
        max_index = torch.argmax(f1_scores)
        self.optimal_threshold = thresholds[max_index]
                
    def compute(self):
        return self.total

class OISF1(Metric):
    def __init__(self) -> None:
        super().__init__()
        self.add_state('total', default = torch.tensor(0, dtype = torch.float32), dist_reduce_fx = "sum")
    
    def update(self, preds: torch.Tensor, targets: torch.Tensor, **kwargs) -> torch.Tensor:
        assert preds.shape == targets.shape
        
        if kwargs['thresholds']:
            thresholds = kwargs['thresholds']
        else:
            thresholds = np.linspace(0.0, 1.0, 10)
        
        f1_scores = torch.zeros((10), dtype = torch.float32, device = preds.device)
        for index, threshold in enumerate(thresholds):
            f1 = BinaryF1Score(threshold = threshold, multidim_average = 'samplewise').to(preds.device)
            f1_scores[index] = torch.nanmean(f1(preds, targets))

        self.total = torch.max(f1_scores)

    def compute(self):
        return self.total   

class AveragePrecision(Metric):
    def __init__(self) -> None:
        super().__init__()
        self.add_state('total', default = torch.tensor(0, dtype = torch.float32), dist_reduce_fx = "sum")

    def update(self, preds: torch.Tensor, targets: torch.Tensor, **kwargs) -> torch.Tensor:
        assert preds.shape == targets.shape
        
        if kwargs['thresholds']:
            thresholds = kwargs['thresholds']
        else:
            thresholds = torch.linspace(0, 1, 10, dtype = torch.float, device = preds.device)
            
        self.total = BinaryAveragePrecision(thresholds = thresholds).to(preds.device)(preds, targets)
        
    def compute(self):
        return self.total

class IOU(Metric):
    def __init__(self) -> None:
        super().__init__()
        self.add_state('total', default = torch.tensor(0, dtype = torch.float32), dist_reduce_fx = "sum")
    
    def update(self, preds: torch.Tensor, targets: torch.Tensor, **kwargs) -> torch.Tensor:
        assert preds.shape == targets.shape
        
        if kwargs['threshold']:
            threshold = kwargs['threshold']
        else:
            threshold = 0.5
        
        iou = BinaryJaccardIndex(threshold = threshold).to(preds.device)
        self.total = iou(preds, targets)
    
    def compute(self):
        return self.total
