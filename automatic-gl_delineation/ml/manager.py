#-----------------------------------------------------------------------------------
#	
#	class      : contains class that manages all aspects of managing a pytorch neural network
#   author     : Sindhu Ramanath Tarekere
#   date	   : 03 August 2022
#
#-----------------------------------------------------------------------------------
import numpy as np
from tqdm import tqdm
import pathlib

import models
import torch

import prepare_torch_datasets as prep_data
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from lightning.pytorch.tuner import Tuner
from lightning.pytorch.callbacks import Callback


class LossDivergenceCallback(Callback):
    def __init__(self, config, savepath):
        super(LossDivergenceCallback, self).__init__()  
        self.config = config
        self.savepath = savepath
    
    def on_train_epoch_end(self, trainer, model):
        if np.abs(trainer.progress_bar_metrics['train_loss_epoch'] - trainer.progress_bar_metrics['val_loss']) > self.config['Model_Details']['divergence_threshold']:
            trainer.save_checkpoint(self.savepath / f'{trainer.current_epoch}-loss_div.ckpt')

class TorchModelManager():
    def __init__(self, configfile):
        """Constructor for TorchModelManager class. Reads configfile, handles model setup, training/validation and prediction phases
        Parameters
        ----------
        configfile : pathlib.Path
            path to configuration file
        """         
        super(TorchModelManager, self).__init__()     

        # Read config file
        self.config = self._read_config(configfile)
        pl.seed_everything(self.config['Model_Details']['seed'], workers = True)

        # Choose model
        try:
            if self.config['Model_Details']['model'] == 'HED':
                self.model = models.HolisticallyNestedEdgeDetection(self.config)
            elif self.config['Model_Details']['model'] == 'RCF':
                self.model = models.RicherConvFeatures(self.config)
            elif self.config['Model_Details']['model'] == 'Deeplab':
                self.model = models.DeepLabV3(self.config)
            elif self.config['Model_Details']['model'] == 'HEDUNet':
                self.model = models.HEDUNet(self.config)
            elif self.config['Model_Details']['model'] == 'DexiNed':
                self.model = models.DexiNed(self.config)
        except:
            raise NotImplementedError(f'{self.config["Model_Details"]["model"]} has not yet been implemented!')

        # Read input tiles and ground truth
        self.setup_data()


    def setup_trainer(self):
        """Sets up callbacks and instantiates PyTorch Lightning Trainer
        """       
        val_loss_chk_callback = ModelCheckpoint(dirpath = self.config['Directories']['checkpoint'] / self.config['Model_Details']['name'], monitor = "val_loss", filename = '{epoch}-val_loss',
                                                             save_top_k = self.config['Model_Details']['save_top_k'])
        val_ods_f1_chk_callback = ModelCheckpoint(dirpath = self.config['Directories']['checkpoint'] / self.config['Model_Details']['name'] , monitor = "val_ODSF1", filename = '{epoch}-val_ods_f1',
                                                           save_top_k = self.config['Model_Details']['save_top_k'], mode = 'max')
        val_ois_f1_chk_callback = ModelCheckpoint(dirpath = self.config['Directories']['checkpoint'] / self.config['Model_Details']['name'] , monitor = "val_OISF1", filename = '{epoch}-val_ois_f1',
                                                           save_top_k = self.config['Model_Details']['save_top_k'], mode = 'max')
        val_ap_chk_callback = ModelCheckpoint(dirpath = self.config['Directories']['checkpoint'] / self.config['Model_Details']['name'] , monitor = "val_AveragePrecision", filename = '{epoch}-val_ap',
                                                           save_top_k = self.config['Model_Details']['save_top_k'], mode = 'max')
        
        loss_divergence_callback = LossDivergenceCallback(config = self.config, savepath = self.config['Directories']['checkpoint'] / self.config['Model_Details']['name'])
        tb_logger = TensorBoardLogger(save_dir = self.config['Directories']['logs_dir'] / self.config['Model_Details']['name'])

        save_params = {'Model_Details', 'Directories', 'Loss', self.config['Model_Details']['model'], 'Optimizer', 'Data'}
        config_subset = {k: self.config[k] for k in self.config.keys() & save_params}

        tb_logger.log_hyperparams(params = config_subset)
        
        callbacks = [val_loss_chk_callback, val_ods_f1_chk_callback, val_ois_f1_chk_callback, val_ap_chk_callback]
        
        self.trainer = pl.Trainer(logger = tb_logger, callbacks = callbacks, devices = 1, accelerator = 'auto', 
                                    max_epochs = self.config['Model_Details']['epochs'], deterministic = self.config['Model_Details']['deterministic'],
                                    accumulate_grad_batches = self.config['Model_Details']['accumulate_batches'], log_every_n_steps = 5,
                                    check_val_every_n_epoch = 1)
    
    def tune_learning_rate(self):
        tuner = Tuner(self.trainer)
        lr_finder = tuner.lr_find(self.model, train_dataloaders = self.training_loader, val_dataloaders = self.validation_loader)
        self.model.lr = lr_finder.suggestion()
        print(f'Tuned LR: {self.model.lr}')
    
    def train(self, dummy = 0):
        """Handles model training and validation
        """
        # Setup PyTorch trainer
        self.setup_trainer()     
        if self.config['Model_Details']['find_lr']:
            self.tune_learning_rate()   
        if self.config['Model_Details']['resume_training']:
            self.trainer.fit(model = self.model, train_dataloaders = self.training_loader, val_dataloaders = self.validation_loader, 
                            ckpt_path = self.config['Directories']['checkpoint'] / (self.config['Model_Details']['name'] + '.ckpt'))
        else:
            self.trainer.fit(model = self.model, train_dataloaders = self.training_loader, val_dataloaders = self.validation_loader)
    
    def predict(self, savedir, pretrained = None):
        """Generates predictions for samples provided via dataloader and saves them in savedir
        """ 
        self.trainer = pl.Trainer(logger = False, devices = 1, accelerator = 'auto', max_epochs = self.config['Model_Details']['epochs'], deterministic = self.config['Model_Details']['deterministic'],
                                  accumulate_grad_batches = self.config['Model_Details']['accumulate_batches']) 
        if self.config['Model_Details']['model'] == 'HED':
            self.model = models.HolisticallyNestedEdgeDetection.load_from_checkpoint(checkpoint_path = pretrained, config = self.config)
        if self.config['Model_Details']['model'] == 'Deeplab':
            self.model = models.DeepLabV3.load_from_checkpoint(checkpoint_path = pretrained, config = self.config)
        self.model.eval()  
        predictions = self.trainer.predict(model = self.model, dataloaders = self.prediction_loader)
        pathlib.Path.mkdir(savedir, exist_ok = True, parents = True)
        for prediction, name in tqdm(zip(predictions, self.tile_names), desc = 'Writing predictions to savedir...'):
            if 'HEDUNet' in self.config['Model_Details']['model']:
                output = prediction[0].numpy()
            elif 'Deeplab' in self.config['Model_Details']['model']:
                output = prediction.numpy()
            else:
                output = prediction[-1].numpy()
            output = output.swapaxes(1, 3)
            np.savez_compressed(savedir / name, np.squeeze(output, axis = 0))

    def setup_data(self):
        """Instantiates dataloaders for training/validation and test sets
        """        
        datamodule = prep_data.TorchDataModule(self.config)
        if not self.config['Model_Details']['predict']:
            self.training_loader = datamodule.train_dataloader()
            self.validation_loader = datamodule.val_dataloader()
        self.prediction_loader = datamodule.prediction_dataloader()
        self.tile_names = datamodule.tile_names

    def _read_config(self, config):
        """Returns configparser object

        Parameters
        ----------
        config: dict
            YAML config converted to a dictionary

        Returns
        -------
        dict
        """        
        if config['Model_Details']['name'] == 'None':
           config['Model_Details']['name'] = config['Model_Details']['model'] + '_' + config['Directories']['features_stack_npz'].name + '_' + str(config['Model_Details']['loss'])
        
        tiles_list = list(config['Directories']['features_stack_npz'].glob('*.npz'))
        with np.load(tiles_list[0]) as data:
            dim = data['arr_0'].shape[0]
        config['Data']['input_shape'] = (dim, dim, len(config['Data']['select_features']))

        return config