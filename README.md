# Deep learning based automatic grounding line delineation in DInSAR interferograms
This repository contains the delineation pipeline described in our publication:

Ramanath Tarekere, S., Krieger, L., Floricioiu, D., & Heidler, K. (2024). Deep learning based automatic grounding line delineation in DInSAR interferograms. EGUsphere, 2024, 1–35. https://doi.org/10.5194/egusphere-2024-223

## Environment
Dependencies for the software are listed in environment.yml. A virtual environment with the same dependencies can be setup as follows (with [Anaconda](https://www.anaconda.com/download/))

```
conda env create -f environment.yml
```

We use [snakemake](https://snakemake.readthedocs.io/en/stable/) to manage the preprocessing, neural network training and postprocessing stages of our pipeline. The ```config.yaml``` is a sample config file and is used for every stage of the delineation pipeline.

## Dataset preprocessing
We have provided a [sample dataset](https://zenodo.org/records/10785613?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjlmZGE1YWEwLWE1MjgtNGMyMy1hZTNlLTRmNjc3NDhlYTI1MyIsImRhdGEiOnt9LCJyYW5kb20iOiI5YzE0MmMwMmZhY2ViMmQ0MjMyYTBhMzhiYWQxNGE4NiJ9.qv6qOnYXksClG8KtamXUJum0Nq_cmEYSRjGK0JjTBp_SHDHHdCBmsq-5GIcpVEd8RsT3V32DobxyVd6OxaxQGg). The dataset contains eight interferograms and the corresponding manual grounding line delineations from the AIS_cci GLL product.

### TanDEM-X 90 m PolarDEM
Available for download https://download.geoservice.dlr.de/TDM_POLARDEM90/ANTARCTICA/

### Ice velocity
The cumulative ice velocity maps of Antarctica used in the publication are not open-access. Alternatively, the monthly averaged ice velocity maps
from ENVEO IT are availble for download here: https://cryoportal.enveo.at/data/

### Differential tide levels
The CATS2008 tide model can be accessed from here: https://www.usap-dc.org/view/dataset/601235

### NCEP/NCAR surface level air pressure
The NCEP/NCAR daily 4x sea level pressure data is available here: https://psl.noaa.gov/data/gridded/data.ncep.reanalysis.html

Once the datasets have been downloaded into the corresponding directories as mentioned in the ```config.yaml``` file, the following commands start the preprocessing workflow. If you are using your own dataset, please ensure the split of training, validation and test samples is indicated in the Shapefile/GeoJSON of the manual delineations as we do not generate this split on the fly. 

```
cd preprocess
snakemake --cores <num cores> 
```

```
cd preprocess
snakemake --cores <num cores> --snakefile Snakefile_predict_only
``` 
should be used with our provided sample dataset as we do not indicate the split of training, validation and test samples. 

Please refer to the [snakemake](https://snakemake.readthedocs.io/en/stable/) documentation for all the options. 

## HED training
The rules for training and prediction of HED are present in the ```ml/Snakefile```. We use [Slurm] (https://slurm.schedmd.com/overview.html) to submit jobs to a compute node. For an example please refer to the script ```ml/slurm_train.cmd```. The model checkpoint provided in the repository can be used to directly generate the predictions of the generated features stack.

Without Slurm, the following commands execute the training and prediction stages of the HED
```
cd ml
snakemake --cores <num cores> -R train
snakemake --cores <num cores> -R predict
```

## Postprocessing
The default Snakefile can be executed via the commands. This also includes calculation of PoLiS distances, F1 score and AP metrics.
```
cd postprocess
snakemake --cores <num cores>
```
Snakefile_predict_only should be used for samples without corresponding manual delineations.