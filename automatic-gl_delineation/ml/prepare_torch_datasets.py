#-----------------------------------------------------------------------------------
#	
#	class      : contains class that handles data loading in a format suitable for pytorch models
#   author     : Sindhu Ramanath Tarekere
#   date	   : 03 August 2022
#
#-----------------------------------------------------------------------------------
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl

class TorchDataset(Dataset):
    """Class reads input tiles and converts them to the format accepted by torch
    """    
    def __init__(self, tiles, features_in_stack, labels_dir = None, dim = (256, 256),
                 n_classes = 2, select_features = ['Real', 'Imaginary', 'Pseudocoherence', 'Phase', 
                                                   'DEM', 'Ice velocity easting', 'ice velocity northing',
                                                   'Differential tide level', 'Atmospheric pressure'], 
                seed = 0):
        """Constructor for DataGenerator

        Parameters
        ----------
        tiles : list of pathlib.Path
            absolute path of tiles
        labels_path : Path obj
            directory containing labels
        dim : tuple, optional
            dimension of features (rows and columns), by default (256, 256)
        n_classes : int, optional
            number of classes, by default 2
        select_features : list, optional
        seed : int, optional
            random seed, defaults to 0 
        """        
        self.dim = dim
        self.labels_dir = labels_dir
        self.tiles = tiles
        self.n_classes = n_classes
        self.select_features = select_features
        self.features_in_stack = features_in_stack
        self.seed = seed
        if self.select_features is not None:
            self.n_channels = len(self.select_features)
    
    def __len__(self):
        """Computes number of batches per epoch

        Returns
        -------
        int
            number of batches
        """        
        return len(self.tiles)
    
    def __getitem__(self, index):
        """Generates one batch of data

        Parameters
        ----------
        index : int
            index of dataset from which to extract images

        Returns
        -------
        (ndarray, ndarray)
            tuple of features and corresponding ground truth
        """        
        # Generate data
        features = np.load(self.tiles[index])['arr_0']
        assert len(self.features_in_stack) == features.shape[-1], f"Mismatch between features in stack{len(self.features_in_stack)} and stack shape {features.shape[-1]}"

        select = np.zeros((features.shape[-1]), dtype = bool)
        for feature in self.select_features:
            select[self.features_in_stack.index(feature)] = True

        tile = features[:, :, select]
        tile = tile.swapaxes(0, 2)
        label = []
        if self.labels_dir:
            label = np.load(self.labels_dir / self.tiles[index].name)['arr_0']
            label = label.swapaxes(0, 2)
            label = torch.from_numpy(label).float()
        
        tile = torch.from_numpy(tile).float()
        
        return tile, label

class TorchDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.setup()

    def setup(self):
        """Collects train/validation/test split from .txt file, creates dataloaders
        """ 
        with open(self.config['Directories']['features_stack_npz'] / 'features_in_stack.txt') as f:
            features_in_stack = f.readlines()
            self.features_in_stack = [elem.strip() for elem in features_in_stack]


        with open(self.config['Directories']['split']) as split_info:
            split = split_info.readlines()
            split = [elem.strip() for elem in split]
    
        train_tiles = split[1:split.index('train_empties')] 
        validation_tiles = split[split.index('validation_valids') + 1:split.index('validation_empties')]
        test_tiles = split[split.index('test_valids') + 1:split.index('test_empties')]
        
        if self.config['Data']['test_interferometric_only']:
            test_interferometric_only = split[split.index('interferometric_valids') + 1:split.index('interferometric_empties')]
            test_tiles = test_tiles + test_interferometric_only
        
        train_list = np.asarray([self.config['Directories']['features_stack_npz'] / tile for tile in train_tiles if (self.config['Directories']['features_stack_npz'] / tile).is_file()])
        validation_list = np.asarray([self.config['Directories']['features_stack_npz'] / tile for tile in validation_tiles if (self.config['Directories']['features_stack_npz'] / tile).is_file()])
        test_list = np.asarray([self.config['Directories']['features_stack_npz'] / tile for tile in test_tiles if (self.config['Directories']['features_stack_npz'] / tile).is_file()])
        
        if self.config['Data']['augment_flipped']:
            augmented = []
            for tile in train_list:
                augmented_tile = self.config['Directories']['features_stack_npz'] / ('augmented_' + tile.name)
                if augmented_tile.is_file():
                    augmented.append(augmented_tile)
            augmented = np.asarray(augmented)
            train_list = np.append(train_list, augmented)
        
        if self.config['Data']['augment_empties']:
            train_empties = split[split.index('train_empties') + 1:split.index('validation_valids')]
            train_empties = np.asarray([self.config['Directories']['features_stack_npz'] / tile for tile in train_empties if (self.config['Directories']['features_stack_npz'] / tile).is_file()])
            train_list = np.append(train_list, train_empties)

        
        self.train_list = train_list
        self.test_list = test_list
        self.validation_list = validation_list
        print(f'Training tiles: {train_list.shape}, validation: {validation_list.shape}, test: {test_list.shape}')
    
    def train_dataloader(self):
        train_split = TorchDataset(tiles = self.train_list, labels_dir = self.config['Directories']['labels_npz'], dim = (self.config['Data']['input_shape'][0], self.config['Data']['input_shape'][1]),
                                    select_features = self.config['Data']['select_features'], features_in_stack = self.features_in_stack)
        
        return DataLoader(dataset = train_split,  batch_size = self.config['Model_Details']['batch_size'], shuffle = True, num_workers = 8)

    def val_dataloader(self):
        val_split = TorchDataset(tiles = self.validation_list, labels_dir = self.config['Directories']['labels_npz'], dim = (self.config['Data']['input_shape'][0], self.config['Data']['input_shape'][1]),
                                    select_features = self.config['Data']['select_features'], features_in_stack = self.features_in_stack)
        return DataLoader(dataset = val_split,  batch_size = self.config['Model_Details']['batch_size'], shuffle = False, num_workers = 8)

    def test_dataloader(self):
        test_split = Dataset(tiles = self.test_list, labels_dir = self.config['Directories']['labels_npz'], dim = (self.config['Data']['input_shape'][0], self.config['Data']['input_shape'][1]),
                                    select_features = self.config['Data']['select_features'], features_in_stack = self.features_in_stack)
        return DataLoader(dataset = test_split,  batch_size = self.config['Model_Details']['batch_size'], shuffle = False, num_workers = 8)
    
    def prediction_dataloader(self):
        # For prediction
        train_list_for_prediction = [tile for tile in self.train_list if 'augmented' not in tile.name]
        predict_tiles = np.hstack((train_list_for_prediction, self.validation_list, self.test_list))
        self.tile_names = [tile.name for tile in predict_tiles]
        predict_gen = TorchDataset(predict_tiles, dim = (self.config['Data']['input_shape'][0], self.config['Data']['input_shape'][1]),
                                    select_features = self.config['Data']['select_features'], features_in_stack = self.features_in_stack)
        return DataLoader(predict_gen, batch_size = 1, shuffle = False, num_workers = 8)
