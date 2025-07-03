#-----------------------------------------------------------------------------------
#	
#	class      : contains classes that build, train and evaluate different deep learning models (pytorch)
#   author     : Sindhu Ramanath Tarekere
#   date	   : 19 July 2022
#
#-----------------------------------------------------------------------------------
from abc import ABC, abstractmethod
import math

import torch
import lightning as pl
import torchmetrics
from lightning.pytorch.utilities import grad_norm

import losses as loss_funcs
import metrics
import layers

class TorchModel(ABC, pl.LightningModule):
    def __init__(self, config):
        """Constructor for TorchModel class. Defines generic neural network functions 
        (training_step, validation_step etc.)

        Parameters
        ----------
        config : dict
            config dictionary
        """         
        super(TorchModel, self).__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Device: {device}')
        self.config = config
        self.lr = self.config['Optimizer']['learning_rate']
        
        metric_collection = torchmetrics.MetricCollection(metrics = [metrics.ODSF1(), metrics.OISF1(), metrics.AveragePrecision(), metrics.IOU()], compute_groups = False)
        
        self.train_metrics = metric_collection.clone(prefix = 'train_')
        self.val_metrics = metric_collection.clone(prefix = 'val_')

        # Loss function
        try:
            if self.config['Loss']['loss']  == 'weighted_cross_entropy':
                self.loss_func = loss_funcs.WeightedCrossEntropy()
            elif self.config['Loss']['loss']  == 'focal':
                self.loss_func = loss_funcs.Focal()
            elif self.config['Loss']['loss'] == 'polis':
                self.loss_func = loss_funcs.Polis()
            elif self.config['Loss']['loss'] == 'dice':
                self.loss_func = loss_funcs.DiceLoss()
            elif self.config['Loss']['loss'] == 'auto_weight_bce':
                self.loss_func = loss_funcs.AutoWeightBCE()
        except:
            raise NotImplementedError(f'Loss function has not yet been implemented!')

    @abstractmethod
    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        pass
        
    @abstractmethod   
    def forward(self, x):
        pass
    
    @abstractmethod
    def training_step(self, batch, batch_idx):
        """Per batch training procedure

        Parameters
        ----------
        batch : torch.utils.data.Dataset
            input data on which the model is to be trained
        batch_idx : int
            batch number

        Returns
        -------
        float
            average training loss for batch
        """        
        inputs, labels = batch
        outputs = self(inputs)
        
        loss = torch.zeros((1), device = inputs.device)

        if isinstance(outputs, list):
            for output in outputs:
                loss = loss + self.loss_func(preds = output, targets = labels, kwargs = self.config['Loss'])
            cal_metrics = outputs[-1]
        else:
            loss = self.loss_func(preds = outputs, targets = labels, kwargs = self.config['Loss'])
            cal_metrics = outputs

        labels = labels.int()
        self.train_metrics(cal_metrics, labels, threshold = self.config['Metrics']['threshold'], thresholds = self.config['Metrics']['thresholds'])

        self.log('train_loss', loss, on_epoch = True, logger = True, prog_bar = True)
        self.log_dict(self.train_metrics, on_epoch = True, logger = True)
        
        return loss

    @abstractmethod
    def validation_step(self, batch, batch_idx):
        """Per batch validation procedure

        Parameters
        ----------
        batch : torch.utils.data.Dataset
            data on which validation of model is carried out
        batch_idx : int
            batch number
        """        
        inputs, labels = batch
        outputs = self(inputs)
        loss = torch.zeros((1), device = inputs.device)

        if isinstance(outputs, list):
            for output in outputs:
                loss = loss + self.loss_func(preds = output, targets = labels, kwargs = self.config['Loss'])
            cal_metrics = outputs[-1]
        else:
            loss = self.loss_func(preds = outputs, targets = labels, kwargs = self.config['Loss'])
            cal_metrics = outputs
        labels = labels.int()
        
        self.log('val_loss', loss, on_epoch = True, logger = True, prog_bar = True)
        self.val_metrics.update(cal_metrics, labels, threshold = self.config['Metrics']['threshold'], thresholds = self.config['Metrics']['thresholds'])

    @abstractmethod
    def on_validation_epoch_end(self) -> None:
        self.log_dict(self.val_metrics.compute(), logger = True, on_epoch = True)
        self.val_metrics.reset()
        
    @abstractmethod
    def test_step(self, batch, batch_idx):
        """Testing procedure per batch

        Parameters
        ----------
        batch : torch.utils.data.Dataset
            test data
        batch_idx : int
            batch number
        """        
        pass

    @abstractmethod
    def configure_optimizers(self):
        """Sets up optimizer as per config (specified in config)

        Returns
        -------
        torch.optim.X
            optimizer object

        """
        # Default optimizer, vanilla Adam without L2 weight decay
        optimizer = torch.optim.Adam(self.parameters(), lr = self.lr)
        
        if self.config['Optimizer']['optimizer'] == 'Adam':
            optimizer = torch.optim.Adam(self.parameters(), lr = self.lr, weight_decay = self.config['Optimizer']['weight_decay'])
        elif self.config['Optimizer']['optimizer'] == 'AdamW':
            optimizer = torch.optim.AdamW(self.parameters(), lr = self.lr, weight_decay = self.config['Optimizer']['weight_decay']) 
            
        return [optimizer]
    
    def save(self, savepath):
        torch.save(self.state_dict(), savepath)
    
    def predict_step(self, batch, batch_idx, dataloader_idx = 0):
        inputs, _ = batch
        return self(inputs)
    
    @abstractmethod
    def initialize_weights(self):
        for name, param in self.named_parameters():
            if name.endswith(".bias"):
                param.data.fill_(0)
            elif name.endswith('.weight'): 
                if len(param.shape) > 1:
                    bound = math.sqrt(6) / math.sqrt(param.shape[0] + param.shape[1])
                else:
                    bound = math.sqrt(6) / math.sqrt(param.shape[0])
                param.data.uniform_(-bound, bound)    
    
class RicherConvFeatures(TorchModel):
    def __init__(self, config):
        """PyTorch implementation of Richer Convolutional Features [Liu et al., 2017]
        https://openaccess.thecvf.com/content_cvpr_2017/papers/Liu_Richer_Convolutional_Features_CVPR_2017_paper.pdf


        Parameters
        ----------
        config['HED'] : dict
           dictionary containing input parameters for the neural network (see config.gl)
        """        
        super().__init__(config = config)
        conv_kernel = config['RCF']['conv_kernel_size']
        stride = config['RCF']['conv_kernel_strides']  
        padding =  config['RCF']['padding']  

        # Model definition

        self.conv_1_1 = torch.nn.Conv2d(in_channels = config['Data']['input_shape'][-1], out_channels = config['RCF']['filters'][0],
                                            kernel_size = conv_kernel, stride = stride, padding = padding)
            
        self.conv_1_2 = torch.nn.Conv2d(in_channels = config['RCF']['filters'][0], out_channels = config['RCF']['filters'][0],
                                        kernel_size = conv_kernel, stride = stride, padding = padding)
        
        self.conv_2_1 = torch.nn.Conv2d(in_channels = config['RCF']['filters'][0], out_channels = config['RCF']['filters'][1],
                                        kernel_size = conv_kernel, stride = stride, padding = padding)

        self.conv_2_2 = torch.nn.Conv2d(in_channels = config['RCF']['filters'][1], out_channels = config['RCF']['filters'][1],
                                        kernel_size = conv_kernel, stride = stride, padding = padding)

        self.conv_3_1 = torch.nn.Conv2d(in_channels = config['RCF']['filters'][1], out_channels = config['RCF']['filters'][2],
                                        kernel_size = conv_kernel, stride = stride, padding = padding)
        
        self.conv_3_2 = torch.nn.Conv2d(in_channels = config['RCF']['filters'][2], out_channels = config['RCF']['filters'][2],
                                        kernel_size = conv_kernel, stride = stride, padding = padding)

        self.conv_3_3 = torch.nn.Conv2d(in_channels = config['RCF']['filters'][2], out_channels = config['RCF']['filters'][2],
                                        kernel_size = conv_kernel, stride = stride, padding = padding)

        self.conv_4_1 = torch.nn.Conv2d(in_channels = config['RCF']['filters'][2], out_channels = config['RCF']['filters'][3],
                                        kernel_size = conv_kernel, stride = stride, padding = padding)
        
        self.conv_4_2 = torch.nn.Conv2d(in_channels = config['RCF']['filters'][3], out_channels = config['RCF']['filters'][3],
                                        kernel_size = conv_kernel, stride = stride, padding = padding)

        self.conv_4_3 = torch.nn.Conv2d(in_channels = config['RCF']['filters'][3], out_channels = config['RCF']['filters'][3],
                                        kernel_size = conv_kernel, stride = stride, padding = padding)

        self.conv_5_1 = torch.nn.Conv2d(in_channels = config['RCF']['filters'][3], out_channels = config['RCF']['filters'][4],
                                        kernel_size = conv_kernel, stride = stride, padding = padding)
        
        self.conv_5_2 = torch.nn.Conv2d(in_channels = config['RCF']['filters'][4], out_channels = config['RCF']['filters'][4],
                                        kernel_size = conv_kernel, stride = stride, padding = padding)

        self.conv_5_3 = torch.nn.Conv2d(in_channels = config['RCF']['filters'][4], out_channels = config['RCF']['filters'][4],
                                        kernel_size = conv_kernel, stride = stride, padding = padding)

        self.activation = torch.nn.ReLU(inplace = True)

        pool_kernel = config['RCF']['pool_kernel_size']
        pool_stride = config['RCF']['pool_kernel_strides']

        self.pool1 = torch.nn.MaxPool2d(pool_kernel, stride = pool_stride, ceil_mode = True)
        self.pool2 = torch.nn.MaxPool2d(pool_kernel, stride = pool_stride, ceil_mode = True)
        self.pool3 = torch.nn.MaxPool2d(pool_kernel, stride = pool_stride, ceil_mode = True)
        self.pool4 = torch.nn.MaxPool2d(pool_kernel, stride = pool_stride, ceil_mode = True)

        self.one_by_one_1_1 = torch.nn.Conv2d(in_channels = config['RCF']['filters'][0], out_channels = 21, 
                                            kernel_size = 1, stride = stride)
        self.one_by_one_1_2 = torch.nn.Conv2d(in_channels = config['RCF']['filters'][0], out_channels = 21, 
                                            kernel_size = 1, stride = stride)

        self.one_by_one_2_1 = torch.nn.Conv2d(in_channels = config['RCF']['filters'][1], out_channels = 21, 
                                            kernel_size = 1, stride = stride)
        self.one_by_one_2_2 = torch.nn.Conv2d(in_channels = config['RCF']['filters'][1], out_channels = 21, 
                                            kernel_size = 1, stride = stride)

        self.one_by_one_3_1 = torch.nn.Conv2d(in_channels = config['RCF']['filters'][2], out_channels = 21, 
                                            kernel_size = 1, stride = stride)
        self.one_by_one_3_2 = torch.nn.Conv2d(in_channels = config['RCF']['filters'][2], out_channels = 21, 
                                            kernel_size = 1, stride = stride)
        self.one_by_one_3_3 = torch.nn.Conv2d(in_channels = config['RCF']['filters'][2], out_channels = 21, 
                                            kernel_size = 1, stride = stride)
        
        self.one_by_one_4_1 = torch.nn.Conv2d(in_channels = config['RCF']['filters'][3], out_channels = 21, 
                                            kernel_size = 1, stride = stride)
        self.one_by_one_4_2 = torch.nn.Conv2d(in_channels = config['RCF']['filters'][3], out_channels = 21, 
                                            kernel_size = 1, stride = stride)
        self.one_by_one_4_3 = torch.nn.Conv2d(in_channels = config['RCF']['filters'][3], out_channels = 21, 
                                            kernel_size = 1, stride = stride)

        self.one_by_one_5_1 = torch.nn.Conv2d(in_channels = config['RCF']['filters'][4], out_channels = 21, 
                                            kernel_size = 1, stride = stride)
        self.one_by_one_5_2 = torch.nn.Conv2d(in_channels = config['RCF']['filters'][4], out_channels = 21, 
                                            kernel_size = 1, stride = stride)
        self.one_by_one_5_3 = torch.nn.Conv2d(in_channels = config['RCF']['filters'][4], out_channels = 21, 
                                            kernel_size = 1, stride = stride)

        self.one_by_one_1 = torch.nn.Conv2d(in_channels = 21, out_channels = 1, kernel_size = 1, stride = 1)
        self.one_by_one_2 = torch.nn.Conv2d(in_channels = 21, out_channels = 1, kernel_size = 1, stride = 1)
        self.one_by_one_3 = torch.nn.Conv2d(in_channels = 21, out_channels = 1, kernel_size = 1, stride = 1)
        self.one_by_one_4 = torch.nn.Conv2d(in_channels = 21, out_channels = 1, kernel_size = 1, stride = 1)
        self.one_by_one_5 = torch.nn.Conv2d(in_channels = 21, out_channels = 1, kernel_size = 1, stride = 1)
        
        self.deconv_1 = torch.nn.ConvTranspose2d(in_channels = 1, out_channels = 1, kernel_size = 2, stride = 2, bias = False)
        self.deconv_2 = torch.nn.ConvTranspose2d(in_channels = 1, out_channels = 1, kernel_size = 4, stride = 4, bias = False)
        self.deconv_3 = torch.nn.ConvTranspose2d(in_channels = 1, out_channels = 1, kernel_size = 8, stride = 8, bias = False)
        self.deconv_4 = torch.nn.ConvTranspose2d(in_channels = 1, out_channels = 1, kernel_size = 16, stride = 16, bias = False)
        
        self.fuse = torch.nn.Conv2d(in_channels = 5, out_channels = 1, kernel_size = 1, stride = 1, bias = False)

        # Initialize weights
        self.initialize_weights()

        # Optimizer
        self.optimizer = self.configure_optimizers()
    
    def initialize_weights(self):
        return super().initialize_weights()

    def forward(self, x):
        """Defines forward path of RCF

        Parameters
        ----------
        x : tensor
            input tensor

        Returns
        -------
        list
            outputs and side outputs of RCF
        """        
        conv_1_1 = self.activation(self.conv_1_1(x))
        conv_1_2 = self.activation(self.conv_1_2(conv_1_1))
        pool1   = self.pool1(conv_1_2)
        conv_2_1 = self.activation(self.conv_2_1(pool1))
        conv_2_2 = self.activation(self.conv_2_2(conv_2_1))
        pool2   = self.pool2(conv_2_2)
        conv_3_1 = self.activation(self.conv_3_1(pool2))
        conv_3_2 = self.activation(self.conv_3_2(conv_3_1))
        conv_3_3 = self.activation(self.conv_3_3(conv_3_2))
        pool3   = self.pool3(conv_3_3)
        conv_4_1 = self.activation(self.conv_4_1(pool3))
        conv_4_2 = self.activation(self.conv_4_2(conv_4_1))
        conv_4_3 = self.activation(self.conv_4_3(conv_4_2))
        pool4   = self.pool4(conv_4_3)
        conv_5_1 = self.activation(self.conv_5_1(pool4))
        conv_5_2 = self.activation(self.conv_5_2(conv_5_1))
        conv_5_3 = self.activation(self.conv_5_3(conv_5_2))

        conv_1_1_down = self.one_by_one_1_1(conv_1_1)
        conv_1_2_down = self.one_by_one_1_2(conv_1_2)
        conv_2_1_down = self.one_by_one_2_1(conv_2_1)
        conv_2_2_down = self.one_by_one_2_2(conv_2_2)
        conv_3_1_down = self.one_by_one_3_1(conv_3_1)
        conv_3_2_down = self.one_by_one_3_2(conv_3_2)
        conv_3_3_down = self.one_by_one_3_3(conv_3_3)
        conv_4_1_down = self.one_by_one_4_1(conv_4_1)
        conv_4_2_down = self.one_by_one_4_2(conv_4_2)
        conv_4_3_down = self.one_by_one_4_3(conv_4_3)
        conv_5_1_down = self.one_by_one_5_1(conv_5_1)
        conv_5_2_down = self.one_by_one_5_2(conv_5_2)
        conv_5_3_down = self.one_by_one_5_3(conv_5_3)

        out1 = self.one_by_one_1(conv_1_1_down + conv_1_2_down)
        out2 = self.one_by_one_2(conv_2_1_down + conv_2_2_down)
        out3 = self.one_by_one_3(conv_3_1_down + conv_3_2_down + conv_3_3_down)
        out4 = self.one_by_one_4(conv_4_1_down + conv_4_2_down + conv_4_3_down)
        out5 = self.one_by_one_5(conv_5_1_down + conv_5_2_down + conv_5_3_down)

        out2 = self.deconv_1(out2)
        out3 = self.deconv_2(out3)
        out4 = self.deconv_3(out4)
        out5 = self.deconv_4(out5)

        fuse = torch.cat((out1, out2, out3, out4, out5), dim = 1)
        out_fuse = self.fuse(fuse)

        results = [out1, out2, out3, out4, out5, out_fuse]
        results = [torch.sigmoid(r) for r in results]
        return results

    def training_step(self, batch, batch_num):
        return super().training_step(batch = batch, batch_idx = batch_num)
    
    def validation_step(self, batch, batch_idx):
        return super().validation_step(batch, batch_idx)
    
    def on_validation_epoch_end(self) -> None:
        return super().on_validation_epoch_end()
    
    def test_step(self, batch, batch_idx):
        return super().test_step(batch, batch_idx)
    
    def configure_optimizers(self):
        return super().configure_optimizers()
    
    def save(self, savepath):
        return super().save(savepath)
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return super().predict_step(batch, batch_idx, dataloader_idx)
    
class HolisticallyNestedEdgeDetection(TorchModel):
    def __init__(self, config):
        super(HolisticallyNestedEdgeDetection, self).__init__(config)

        conv_kernel = config['HED']['conv_kernel_size'] 
        conv_stride = config['HED']['conv_kernel_strides'] 
        padding = config['HED']['padding']  

        # Model definition
        self.conv_1_1 = torch.nn.Conv2d(in_channels = config['Data']['input_shape'][-1], out_channels = config['HED']['filters'][0],
                                        kernel_size = conv_kernel, stride = conv_stride, padding = padding)
        
        self.conv_1_2 = torch.nn.Conv2d(in_channels = config['HED']['filters'][0], out_channels = config['HED']['filters'][0],
                                        kernel_size = conv_kernel, stride = conv_stride, padding = padding)
        
        self.conv_2_1 = torch.nn.Conv2d(in_channels = config['HED']['filters'][0], out_channels = config['HED']['filters'][1],
                                        kernel_size = conv_kernel, stride = conv_stride, padding = padding)

        self.conv_2_2 = torch.nn.Conv2d(in_channels = config['HED']['filters'][1], out_channels = config['HED']['filters'][1],
                                        kernel_size = conv_kernel, stride = conv_stride, padding = padding)

        self.conv_3_1 = torch.nn.Conv2d(in_channels = config['HED']['filters'][1], out_channels = config['HED']['filters'][2],
                                        kernel_size = conv_kernel, stride = conv_stride, padding = padding)

        self.conv_3_2 = torch.nn.Conv2d(in_channels = config['HED']['filters'][2], out_channels = config['HED']['filters'][2],
                                        kernel_size = conv_kernel, stride = conv_stride, padding = padding)

        self.conv_3_3 = torch.nn.Conv2d(in_channels = config['HED']['filters'][2], out_channels = config['HED']['filters'][2],
                                        kernel_size = conv_kernel, stride = conv_stride, padding = padding)

        self.conv_4_1 = torch.nn.Conv2d(in_channels = config['HED']['filters'][2], out_channels = config['HED']['filters'][3],
                                        kernel_size = conv_kernel, stride = conv_stride, padding = padding)
        
        self.conv_4_2 = torch.nn.Conv2d(in_channels = config['HED']['filters'][3], out_channels = config['HED']['filters'][3],
                                        kernel_size = conv_kernel, stride = conv_stride, padding = padding)

        self.conv_4_3 = torch.nn.Conv2d(in_channels = config['HED']['filters'][3], out_channels = config['HED']['filters'][3],
                                        kernel_size = conv_kernel, stride = conv_stride, padding = padding)

        self.conv_5_1 = torch.nn.Conv2d(in_channels = config['HED']['filters'][3], out_channels = config['HED']['filters'][4],
                                        kernel_size = conv_kernel, stride = conv_stride, padding = padding)
        
        self.conv_5_2 = torch.nn.Conv2d(in_channels = config['HED']['filters'][4], out_channels = config['HED']['filters'][4],
                                        kernel_size = conv_kernel, stride = conv_stride, padding = padding)

        self.conv_5_3 = torch.nn.Conv2d(in_channels = config['HED']['filters'][4], out_channels = config['HED']['filters'][4],
                                        kernel_size = conv_kernel, stride = conv_stride, padding = padding)

        self.activation = torch.nn.ReLU(inplace = True)

        pool_kernel = config['HED']['pool_kernel_size']
        pool_stride = config['HED']['pool_kernel_strides']

        self.pool1 = torch.nn.MaxPool2d(pool_kernel, stride = pool_stride)
        self.pool2 = torch.nn.MaxPool2d(pool_kernel, stride = pool_stride)
        self.pool3 = torch.nn.MaxPool2d(pool_kernel, stride = pool_stride)
        self.pool4 = torch.nn.MaxPool2d(pool_kernel, stride = pool_stride)

        self.one_by_one_1 = torch.nn.Conv2d(in_channels = config['HED']['filters'][0], out_channels = 1, kernel_size = 1, stride = conv_stride, padding = padding)
        self.one_by_one_2 = torch.nn.Conv2d(in_channels = config['HED']['filters'][1], out_channels = 1, kernel_size = 1, stride = conv_stride, padding = padding)
        self.one_by_one_3 = torch.nn.Conv2d(in_channels = config['HED']['filters'][2], out_channels = 1, kernel_size = 1, stride = conv_stride, padding = padding)
        self.one_by_one_4 = torch.nn.Conv2d(in_channels = config['HED']['filters'][3], out_channels = 1, kernel_size = 1, stride = conv_stride, padding = padding)
        self.one_by_one_5 = torch.nn.Conv2d(in_channels = config['HED']['filters'][4], out_channels = 1, kernel_size = 1, stride = conv_stride, padding = padding)

        self.deconv_2 = torch.nn.ConvTranspose2d(in_channels = 1, out_channels = 1, kernel_size = 2, stride = 2, bias = False, padding = 0)
        self.deconv_3 = torch.nn.ConvTranspose2d(in_channels = 1, out_channels = 1, kernel_size = 8, stride = 4, bias = False, padding = 2)
        self.deconv_4 = torch.nn.ConvTranspose2d(in_channels = 1, out_channels = 1, kernel_size = 16, stride = 8, bias = False, padding = 4)
        self.deconv_5 = torch.nn.ConvTranspose2d(in_channels = 1, out_channels = 1, kernel_size = 32, stride = 16, bias = False, padding = 8)

        self.fuse = torch.nn.Conv2d(in_channels = 5, out_channels = 1, kernel_size = 1, stride = 1, bias = False)

        # Initialize weights
        self.initialize_weights()

        # Optimizer
        self.optimizer = self.configure_optimizers()

    def initialize_weights(self):
        return super().initialize_weights()
    
    def forward(self, x):
        """Defines forward path of HED

        Parameters
        ----------
        x : tensor
            input tensor

        Returns
        -------
        list
            outputs and side outputs of HED
        """     
        conv_1_1 = self.activation(self.conv_1_1(x))
        conv_1_2 = self.activation(self.conv_1_2(conv_1_1))
        pool1   = self.pool1(conv_1_2)
        conv_2_1 = self.activation(self.conv_2_1(pool1))
        conv_2_2 = self.activation(self.conv_2_2(conv_2_1))
        pool2   = self.pool2(conv_2_2)
        conv_3_1 = self.activation(self.conv_3_1(pool2))
        conv_3_2 = self.activation(self.conv_3_2(conv_3_1))
        conv_3_3 = self.activation(self.conv_3_3(conv_3_2))
        pool3   = self.pool3(conv_3_3)
        conv_4_1 = self.activation(self.conv_4_1(pool3))
        conv_4_2 = self.activation(self.conv_4_2(conv_4_1))
        conv_4_3 = self.activation(self.conv_4_3(conv_4_2))
        pool4   = self.pool4(conv_4_3)
        conv_5_1 = self.activation(self.conv_5_1(pool4))
        conv_5_2 = self.activation(self.conv_5_2(conv_5_1))
        conv_5_3 = self.activation(self.conv_5_3(conv_5_2))

        conv_1_down = self.one_by_one_1(conv_1_2)
        conv_2_down = self.one_by_one_2(conv_2_2)
        conv_3_down = self.one_by_one_3(conv_3_3)
        conv_4_down = self.one_by_one_4(conv_4_3)
        conv_5_down = self.one_by_one_5(conv_5_3)

        out1 = conv_1_down
        out2 = self.deconv_2(conv_2_down)
        out3 = self.deconv_3(conv_3_down)
        out4 = self.deconv_4(conv_4_down)
        out5 = self.deconv_5(conv_5_down)

        fuse = torch.cat((out1, out2, out3, out4, out5), dim = 1)
        out_fuse = self.fuse(fuse)

        results = [out1, out2, out3, out4, out5, out_fuse]
        results = [torch.sigmoid(r) for r in results]
        return results
    
    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        for name, layer in self.named_children():
            if 'conv' in name:
                gradient = grad_norm(layer, norm_type = 2)
                self.log(name, gradient['grad_2.0_norm_total'], logger = True)
    
    def training_step(self, batch, batch_idx):
        """Per batch training procedure

        Parameters
        ----------
        batch : torch.utils.data.Dataset
            input data on which the model is to be trained
        batch_idx : int
            batch number

        Returns
        -------
        float
            average training loss for batch
        """        
        return super().training_step(batch, batch_idx)
    
    def validation_step(self, batch, batch_idx):
        """Per batch validation procedure

        Parameters
        ----------
        batch : torch.utils.data.Dataset
            data on which validation of model is carried out
        batch_idx : int
            batch index
        """        
        return super().validation_step(batch, batch_idx)
    
    def on_validation_epoch_end(self) -> None:
        return super().on_validation_epoch_end()
    
    def test_step(self, batch, batch_idx):
        """Testing procedure per batch

        Parameters
        ----------
        batch : torch.utils.data.Dataset
            test data
        batch_idx : int
            batch index
        """        
        inputs, labels = batch

        outputs = self(inputs)
        loss = torch.zeros((1), device = inputs.device)
        
        for output in outputs:
            loss = loss + self.loss_func(preds = output, targets = labels, kwargs = self.config['Loss'])    

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("test_loss", loss, prog_bar = True)
    
    def configure_optimizers(self):
        return super().configure_optimizers()
    
    def save(self, savepath):
        return super().save(savepath)
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return super().predict_step(batch, batch_idx, dataloader_idx)

class DeepLabV3(TorchModel):
    def __init__(self, config):
        super(DeepLabV3, self).__init__(config)
        self.config = config

        # Model definition

        self.dilated_1 = torch.nn.Conv2d(in_channels = 128, out_channels = 128, groups = 128, kernel_size = self.config['Deeplab']['conv_kernel_size'], 
                                         stride = self.config['Deeplab']['conv_kernel_strides'] , padding = self.config['Deeplab']['padding'], dilation = 1)
        self.dilated_2 = torch.nn.Conv2d(in_channels = 128, out_channels = 128, groups = 128, kernel_size = self.config['Deeplab']['conv_kernel_size'], 
                                         stride = self.config['Deeplab']['conv_kernel_strides'] , padding = self.config['Deeplab']['padding'], dilation = 2)
        self.dilated_3 = torch.nn.Conv2d(in_channels = 128, out_channels = 128, groups = 128, kernel_size = self.config['Deeplab']['conv_kernel_size'], 
                                         stride = self.config['Deeplab']['conv_kernel_strides'] , padding = self.config['Deeplab']['padding'], dilation = 3)
        self.dilated_4 = torch.nn.Conv2d(in_channels = 128, out_channels = 128, groups = 128, kernel_size = self.config['Deeplab']['conv_kernel_size'], 
                                        stride = self.config['Deeplab']['conv_kernel_strides'] , padding = self.config['Deeplab']['padding'], dilation = 4)
        self.dilated_5 = torch.nn.Conv2d(in_channels = 128, out_channels = 128, groups = 128, kernel_size = self.config['Deeplab']['conv_kernel_size'], 
                                         stride = self.config['Deeplab']['conv_kernel_strides'] , padding = self.config['Deeplab']['padding'], dilation = 5)

        self.elu = torch.nn.ELU(inplace = True)
        self.pool = torch.nn.MaxPool2d(kernel_size = self.config['Deeplab']['pool_kernel_size'], stride = self.config['Deeplab']['pool_kernel_strides'])
        self.dropout = torch.nn.Dropout(p  = self.config['Deeplab']['dropout'])
        
        self.upsample = torch.nn.Upsample(scale_factor = 2, mode = 'nearest')
        self.fuse = torch.nn.Conv2d(in_channels = 3, out_channels = 1, kernel_size = 1, stride = 1, bias = False)
        
        
        self.conv_1_1 = torch.nn.Conv2d(in_channels = self.config['Data']['input_shape'][-1], out_channels = 32, kernel_size = self.config['Deeplab']['conv_kernel_size'], 
                                          stride = self.config['Deeplab']['conv_kernel_strides'] , padding = self.config['Deeplab']['padding'])
        self.conv_1_2 = torch.nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = self.config['Deeplab']['conv_kernel_size'], 
                                          stride = self.config['Deeplab']['conv_kernel_strides'] , padding = self.config['Deeplab']['padding'])
        
        self.conv_2_1 = torch.nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = self.config['Deeplab']['conv_kernel_size'], 
                                          stride = self.config['Deeplab']['conv_kernel_strides'] , padding = self.config['Deeplab']['padding'])
        self.conv_2_2 = torch.nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = self.config['Deeplab']['conv_kernel_size'], 
                                          stride = self.config['Deeplab']['conv_kernel_strides'] , padding = self.config['Deeplab']['padding'])
        
        self.conv_3_1 = torch.nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = self.config['Deeplab']['conv_kernel_size'], 
                                          stride = self.config['Deeplab']['conv_kernel_strides'] , padding = self.config['Deeplab']['padding'])
        self.conv_3_2 = torch.nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = self.config['Deeplab']['conv_kernel_size'], 
                                          stride = self.config['Deeplab']['conv_kernel_strides'] , padding = self.config['Deeplab']['padding'])
        
        self.conv_4_1 = torch.nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = self.config['Deeplab']['conv_kernel_size'], 
                                          stride = self.config['Deeplab']['conv_kernel_strides'] , padding = self.config['Deeplab']['padding'])
        self.conv_4_2 = torch.nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = self.config['Deeplab']['conv_kernel_size'], 
                                          stride = self.config['Deeplab']['conv_kernel_strides'] , padding = self.config['Deeplab']['padding'])
        
        self.conv_5_1 = torch.nn.Conv2d(in_channels = 704, out_channels = 64, kernel_size = self.config['Deeplab']['conv_kernel_size'], 
                                          stride = self.config['Deeplab']['conv_kernel_strides'] , padding = self.config['Deeplab']['padding'])
        self.conv_5_2 = torch.nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = self.config['Deeplab']['conv_kernel_size'], 
                                          stride = self.config['Deeplab']['conv_kernel_strides'] , padding = self.config['Deeplab']['padding'])
        
        self.conv_6_1 = torch.nn.Conv2d(in_channels = 128, out_channels = 32, kernel_size = self.config['Deeplab']['conv_kernel_size'], 
                                          stride = self.config['Deeplab']['conv_kernel_strides'] , padding = self.config['Deeplab']['padding'])
        self.conv_6_2 = torch.nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = self.config['Deeplab']['conv_kernel_size'], 
                                          stride = self.config['Deeplab']['conv_kernel_strides'] , padding = self.config['Deeplab']['padding'])
        
        self.conv_7_1 = torch.nn.Conv2d(in_channels = 64, out_channels = 32, kernel_size = self.config['Deeplab']['conv_kernel_size'], 
                                          stride = self.config['Deeplab']['conv_kernel_strides'] , padding = self.config['Deeplab']['padding'])
        self.conv_7_2 = torch.nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = self.config['Deeplab']['conv_kernel_size'], 
                                          stride = self.config['Deeplab']['conv_kernel_strides'] , padding = self.config['Deeplab']['padding'])
        self.conv_7_3 = torch.nn.Conv2d(in_channels = 32, out_channels = 3, kernel_size = self.config['Deeplab']['conv_kernel_size'], 
                                          stride = self.config['Deeplab']['conv_kernel_strides'] , padding = self.config['Deeplab']['padding'])
        
        # Initialize weights
        self.initialize_weights()
        
        # Optimizer
        self.optimizer = self.configure_optimizers()
    
    def conv_block(self, input, in_channels, out_channels):
        conv_1 = self.elu(torch.nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = self.config['Deeplab']['conv_kernel_size'], 
                                          stride = self.config['Deeplab']['conv_kernel_strides'] , padding = self.config['Deeplab']['padding'])(input))
        dropout = self.dropout(conv_1)
        conv_2 = self.elu(torch.nn.Conv2d(in_channels = out_channels, out_channels = out_channels, kernel_size = self.config['Deeplab']['conv_kernel_size'], 
                                          stride = self.config['Deeplab']['conv_kernel_strides'] , padding = self.config['Deeplab']['padding'])(dropout))
        return conv_2
    
    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        return super().on_before_optimizer_step(optimizer)
    
    def initialize_weights(self):
        return super().initialize_weights()

    def forward(self, x):
        """Defines forward path of RCF

        Parameters
        ----------
        x : torch.Tensor
            input tensor

        Returns
        -------
        torch.Tensor
            network output
        """     
        conv_1_1 = self.elu(self.conv_1_1(x))
        dropout_1 = self.dropout(conv_1_1)
        conv_1_2 = self.elu(dropout_1)
        pool_1 = self.pool(conv_1_2)
        
        conv_2_1 = self.elu(self.conv_2_1(pool_1))
        dropout_2 = self.dropout(conv_2_1)
        conv_2_2 = self.elu(self.conv_2_2(dropout_2))
        pool_2 = self.pool(conv_2_2)
        
        conv_3_1 = self.elu(self.conv_3_1(pool_2))
        dropout_3 = self.dropout(conv_3_1)
        conv_3_2 = self.elu(self.conv_3_2(dropout_3))
        pool_3 = self.pool(conv_3_2)
        
        conv_4_1 = self.elu(self.conv_4_1(pool_3))
        dropout_4 = self.dropout(conv_4_1)
        conv_4_2 = self.elu(self.conv_4_2(dropout_4))
        
        dilated_1 = self.elu(self.dilated_1(conv_4_2))
        dilated_2 = self.elu(self.dilated_2(conv_4_2))
        dilated_3 = self.elu(self.dilated_3(conv_4_2))
        dilated_4 = self.elu(self.dilated_4(conv_4_2))
        dilated_5 = self.elu(self.dilated_5(conv_4_2))
        
        cat_1 = torch.cat((dilated_1, dilated_2, dilated_3, dilated_4, dilated_5), dim = 1)
        upsampled_1 = self.upsample(cat_1)
        cat_2 = torch.cat([upsampled_1, conv_3_2], dim = 1)
        
        conv_5_1 = self.elu(self.conv_5_1(cat_2))
        dropout_5 = self.dropout(conv_5_1)
        conv_5_2 = self.elu(self.conv_5_2(dropout_5))
        
        upsampled_2 = self.upsample(conv_5_2)
        cat_3 = torch.cat([upsampled_2, conv_2_2], dim = 1)
        
        conv_6_1 = self.elu(self.conv_6_1(cat_3))
        dropout_6 = self.dropout(conv_6_1)
        conv_6_2 = self.elu(self.conv_6_2(dropout_6))
        
        upsampled_3 = self.upsample(conv_6_2)
        cat_4 = torch.cat([upsampled_3, conv_1_2], dim = 1)
        
        conv_7_1 = self.elu(self.conv_7_1(cat_4))
        dropout_7 = self.dropout(conv_7_1)
        conv_7_2 = self.elu(self.conv_7_2(dropout_7))
        conv_7_3 = self.elu(self.conv_7_3(conv_7_2))
        output = torch.sigmoid(self.fuse(conv_7_3))
        return output
    
    def training_step(self, batch, batch_idx):
        return super().training_step(batch, batch_idx)
    
    def validation_step(self, batch, batch_idx):
        return super().validation_step(batch = batch, batch_idx = batch_idx)
    
    def on_validation_epoch_end(self) -> None:
        return super().on_validation_epoch_end()
    
    def test_step(self, batch, batch_idx):
        inputs, labels = batch

        output = self(inputs)
        
        loss = self.loss_func(preds = output, targets = labels, kwargs = self.config['Loss'])    

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("test_loss", loss, prog_bar = True)
    
    def configure_optimizers(self):
        return super().configure_optimizers()
    
    def save(self, savepath):
        return super().save(savepath)
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return super().predict_step(batch, batch_idx, dataloader_idx) 

class HEDUNet(TorchModel):
    """Implements HEDUNet following this publication: 
       K. Heidler, L. Mou, C. Baumhoer, A. Dietz and X. X. Zhu, "HED-UNet: Combined Segmentation and Edge Detection for Monitoring the Antarctic Coastline," 
       in IEEE Transactions on Geoscience and Remote Sensing, vol. 60, pp. 1-14, 2022, Art no. 4300514, doi: 10.1109/TGRS.2021.3064606.
       
       Code is adapted from their github: https://github.com/khdlr/HED-UNet
       

    Parameters
    ----------
    TorchModel : _type_
        _description_
    """    
    def __init__(self, config):
        super(HEDUNet, self).__init__(config)
        self.config = config
        
        conv_block = layers.Convx2
        if self.config['HEDUNet']['squeeze_excitation']:
            conv_block = layers.WithSE(conv_block)
        
        self.output_channels = 2
        
        self.init = torch.nn.Conv2d(in_channels = self.config['Data']['input_shape'][-1], out_channels = self.config['HEDUNet']['base_channels'], kernel_size = 1)

        conv_args = dict(conv_block = conv_block, bn = self.config['HEDUNet']['batch_norm'], padding_mode = 'replicate')
        
        self.down_blocks = torch.nn.ModuleList([layers.DownBlock(c_in = (1<<i) * self.config['HEDUNet']['base_channels'], c_out = (2<<i) * self.config['HEDUNet']['base_channels'], **conv_args)
                                                for i in range(self.config['HEDUNet']['stack_height'])])
        

        self.up_blocks = torch.nn.ModuleList([layers.UpBlock(c_in = (2<<i) * self.config['HEDUNet']['base_channels'], c_out = (1<<i) * self.config['HEDUNet']['base_channels'], **conv_args) 
                                               for i in reversed(range(self.config['HEDUNet']['stack_height']))])
        
        self.predictors = torch.nn.ModuleList([torch.nn.Conv2d((1<<i) * self.config['HEDUNet']['base_channels'], self.output_channels, 1)
                                                for i in reversed(range(self.config['HEDUNet']['stack_height'] + 1))])
        
        if self.config['HEDUNet']['merging'] == 'attention':
            self.queries = torch.nn.ModuleList([torch.nn.Conv2d((1<<i) * self.config['HEDUNet']['base_channels'], self.output_channels, 1)
                for i in reversed(range(self.config['HEDUNet']['stack_height'] + 1))])
        elif self.config['HEDUNet']['merging'] == 'learned':
            self.merge_predictions = torch.nn.Conv2d(self.output_channels * (self.config['HEDUNet']['stack_height'] + 1), self.output_channels, 1)
        else:
            # no merging
            pass
        
        self.sobel = torch.nn.Conv2d(1, 2, 1, padding = 1, padding_mode = 'replicate', bias = False)
        self.sobel.weight.requires_grad = False
        self.sobel.weight.set_(torch.Tensor([[
            [-1,  0,  1],
            [-2,  0,  2],
            [-1,  0,  1]],
            [[-1, -2, -1],
            [ 0,  0,  0],
            [ 1,  2,  1]
        ]]).reshape(2, 1, 3, 3))
        
        # Initialize weights
        self.initialize_weights()
        
        # Optimizer
        self.optimizer = self.configure_optimizers()
    
    def initialize_weights(self):
        return super().initialize_weights()
        
    def forward(self, x):
        B, _, H, W = x.shape
        x = self.init(x)

        skip_connections = []
        for block in self.down_blocks:
            skip_connections.append(x)
            x = block(x)

        multilevel_features = [x]
        for block, skip in zip(self.up_blocks, reversed(skip_connections)):
            x = block(x, skip)
            multilevel_features.append(x)

        predictions_list = []
        full_scale_preds = []
        for feature_map, predictor in zip(multilevel_features, self.predictors):
            prediction = predictor(feature_map)
            predictions_list.append(prediction)
            full_scale_preds.append(torch.nn.functional.interpolate(prediction, size = (H, W), mode = 'bilinear', align_corners = True))

        predictions = torch.cat(full_scale_preds, dim=1)

        if self.config['HEDUNet']['merging'] == 'attention':
            queries = [torch.nn.functional.interpolate(q(feat), size = (H, W), mode = 'bilinear', align_corners = True)
                    for q, feat in zip(self.queries, multilevel_features)]
            queries = torch.cat(queries, dim = 1)
            queries = queries.reshape(B, -1, self.output_channels, H, W)
            attn = torch.nn.functional.softmax(queries, dim = 1)
            predictions = predictions.reshape(B, -1, self.output_channels, H, W)
            combined_prediction = torch.sum(attn * predictions, dim = 1)
        elif self.config['HEDUNet']['merging'] == 'learned':
            combined_prediction = self.merge_predictions(predictions)
        else:
            combined_prediction = predictions_list[-1]

        if self.config['HEDUNet']['deep_supervision']:
            return combined_prediction, list(reversed(predictions_list))
        else:
            return combined_prediction
    
    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        return super().on_before_optimizer_step(optimizer)
    
    def get_pyramid(self, mask):
        with torch.no_grad():
            self.sobel = self.sobel.to(mask.device)
            masks = [mask]
            ## Build mip-maps
            for _ in range(self.config['HEDUNet']['stack_height']):
                # Pretend we have a batch
                big_mask = masks[-1]
                small_mask = torch.nn.functional.avg_pool2d(big_mask, 2)
                masks.append(small_mask)

            targets = []
            for mask in masks:
                sobel = torch.any(self.sobel(mask) != 0, dim = 1, keepdims = True).float()
                targets.append(torch.cat([mask, sobel], dim = 1))

        return targets
    
    def training_step(self, batch, batch_idx):
        """Per batch training procedure

        Parameters
        ----------
        batch : torch.utils.data.Dataset
            input data on which the model is to be trained
        batch_idx : int
            batch number

        Returns
        -------
        float
            average training loss for batch
        """        
        inputs, labels = batch
        output, outputs_levels = self(inputs)
        targets = self.get_pyramid(labels)
        
        loss_final = self.loss_func(output, targets[0], kwargs = self.config['Loss'])
        loss_levels = []

        for output_level, target in zip(outputs_levels, targets):
            loss_levels.append(self.loss_func(output_level, target, kwargs = self.config['Loss']))
        
        deep_supervision_loss = torch.sum(torch.stack(loss_levels))
        loss = loss_final + deep_supervision_loss

        target = targets[0].int()
        self.train_metrics(output, target, threshold = self.config['Metrics']['threshold'], thresholds = self.config['Metrics']['thresholds'])

        self.log('train_loss', loss, on_epoch = True, logger = True, prog_bar = True)
        self.log_dict(self.train_metrics, on_epoch = True, logger = True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        """Per batch validation procedure

        Parameters
        ----------
        batch : torch.utils.data.Dataset
            data on which validation of model is carried out
        batch_idx : int
            batch index
        """        
        inputs, labels = batch
        output, outputs_levels = self(inputs)
        targets = self.get_pyramid(labels)
        
        loss_final = self.loss_func(output, targets[0], kwargs = self.config['Loss'])
        loss_levels = []

        for output_level, target in zip(outputs_levels, targets):
            loss_levels.append(self.loss_func(output_level, target, kwargs = self.config['Loss']))
        
        deep_supervision_loss = torch.sum(torch.stack(loss_levels))
        loss = loss_final + deep_supervision_loss

        target = targets[0].int()
        
        self.log('val_loss', loss, on_epoch = True, logger = True, prog_bar = True)
        self.val_metrics.update(output, target, threshold = self.config['Metrics']['threshold'], thresholds = self.config['Metrics']['thresholds'])
    
    def on_validation_epoch_end(self) -> None:
        return super().on_validation_epoch_end()
    
    def test_step(self, batch, batch_idx):
        return super().test_step(batch, batch_idx)
    
    def configure_optimizers(self):
        return super().configure_optimizers()
    
    def save(self, savepath):
        return super().save(savepath)
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return super().predict_step(batch, batch_idx, dataloader_idx)

class DexiNed(TorchModel):
    def __init__(self, config):
        super().__init__(config)
        
        self.config = config
    
        self.block_1 = layers.DoubleConvBlock(self.config['Data']['input_shape'][-1], 32, 64, stride = 2)
        self.block_2 = layers.DoubleConvBlock(64, 128, use_act = False)
        self.dblock_3 = layers._DenseBlock(2, 128, 256) # [128,256,100,100]
        self.dblock_4 = layers._DenseBlock(3, 256, 512)
        self.dblock_5 = layers._DenseBlock(3, 512, 512)
        self.dblock_6 = layers._DenseBlock(3, 512, 256)
        self.maxpool = torch.nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)

        # left skip connections, figure in Journal
        self.side_1 = layers.SingleConvBlock(64, 128, 2)
        self.side_2 = layers.SingleConvBlock(128, 256, 2)
        self.side_3 = layers.SingleConvBlock(256, 512, 2)
        self.side_4 = layers.SingleConvBlock(512, 512, 1)

        # right skip connections, figure in Journal paper
        self.pre_dense_2 = layers.SingleConvBlock(128, 256, 2)
        self.pre_dense_3 = layers.SingleConvBlock(128, 256, 1)
        self.pre_dense_4 = layers.SingleConvBlock(256, 512, 1)
        self.pre_dense_5 = layers.SingleConvBlock(512, 512, 1)
        self.pre_dense_6 = layers.SingleConvBlock(512, 256, 1)


        self.up_block_1 = layers.UpConvBlock(64, 1)
        self.up_block_2 = layers.UpConvBlock(128, 1)
        self.up_block_3 = layers.UpConvBlock(256, 2)
        self.up_block_4 = layers.UpConvBlock(512, 3)
        self.up_block_5 = layers.UpConvBlock(512, 4)
        self.up_block_6 = layers.UpConvBlock(256, 4)
        self.block_cat = layers.SingleConvBlock(6, 1, stride=1, use_bs=False) # hed fusion method
        
        self.initialize_weights()
        
        self.optimizer = self.configure_optimizers()
    
    def initialize_weights(self):
        for name, param in self.named_parameters():
            if name.endswith('.weight') and 'conv' in name:
                xavier_std = (math.sqrt(2) / math.sqrt(param.shape[0] + param.shape[1]))
                param.data.normal_(0.0, xavier_std)
                if param.shape[1] == torch.Size([1]):
                    if 'transpose' in name:
                        param.data.normal_(mean = 0.0, std = 0.1)
                    else:
                        param.data.normal_(mean = 0.0)
            elif name.endswith('.bias'):
                param.data.fill_(0)

    
    def slice(self, tensor, slice_shape):
        t_shape = tensor.shape
        height, width = slice_shape
        if t_shape[-1]!=slice_shape[-1]:
            new_tensor = torch.nn.functional.interpolate(
                tensor, size=(height, width), mode = 'bicubic', align_corners = False)
        else:
            new_tensor = tensor
        # tensor[..., :height, :width]
        return new_tensor
    
    def forward(self, x):
        # Block 1
        block_1 = self.block_1(x)
        block_1_side = self.side_1(block_1)

        # Block 2
        block_2 = self.block_2(block_1)
        block_2_down = self.maxpool(block_2)
        block_2_add = block_2_down + block_1_side
        block_2_side = self.side_2(block_2_add)

        # Block 3
        block_3_pre_dense = self.pre_dense_3(block_2_down)
        block_3, _ = self.dblock_3([block_2_add, block_3_pre_dense])
        block_3_down = self.maxpool(block_3) # [128,256,50,50]
        block_3_add = block_3_down + block_2_side
        block_3_side = self.side_3(block_3_add)

        # Block 4
        block_2_resize_half = self.pre_dense_2(block_2_down)
        block_4_pre_dense = self.pre_dense_4(block_3_down+block_2_resize_half)
        block_4, _ = self.dblock_4([block_3_add, block_4_pre_dense])
        block_4_down = self.maxpool(block_4)
        block_4_add = block_4_down + block_3_side
        block_4_side = self.side_4(block_4_add)

        # Block 5
        block_5_pre_dense = self.pre_dense_5(block_4_down) #block_5_pre_dense_512 +block_4_down
        block_5, _ = self.dblock_5([block_4_add, block_5_pre_dense])
        block_5_add = block_5 + block_4_side

        # Block 6
        block_6_pre_dense = self.pre_dense_6(block_5)
        block_6, _ = self.dblock_6([block_5_add, block_6_pre_dense])

        # upsampling blocks
        out_1 = self.up_block_1(block_1)
        out_2 = self.up_block_2(block_2)
        out_3 = self.up_block_3(block_3)
        out_4 = self.up_block_4(block_4)
        out_5 = self.up_block_5(block_5)
        out_6 = self.up_block_6(block_6)
        results = [out_1, out_2, out_3, out_4, out_5, out_6]

        # concatenate multiscale outputs
        block_cat = torch.cat(results, dim=1)  # Bx6xHxW
        block_cat = self.block_cat(block_cat)  # Bx1xHxW

        # return results
        results.append(block_cat)
        results = [torch.sigmoid(result) for result in results]
        return results

    def training_step(self, batch, batch_idx):
        return super().training_step(batch, batch_idx)
    
    def validation_step(self, batch, batch_idx):
        return super().validation_step(batch, batch_idx)
    
    def on_validation_epoch_end(self) -> None:
        return super().on_validation_epoch_end()
    
    def test_step(self, batch, batch_idx):
        return super().test_step(batch, batch_idx)
    
    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        return super().on_before_optimizer_step(optimizer)
    
    def configure_optimizers(self):
        return super().configure_optimizers()
    
    def save(self, savepath):
        return super().save(savepath)
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return super().predict_step(batch, batch_idx, dataloader_idx)