Regressrion Pipeline
====================

A regression pipeline that can run a model selection experiment with various
feature sets selected using different feature selection methods. Moreover,
the hyperparameters of the models can be tuned with optuna. 

Various Modules
===============

The pipeline is divided into various modules

* Preprocessing module
* Feature engineering module
* Feature selection module
* Feature transformation module
* Optuna optimization module
* Model evaluation module
* Vizualization module 

Preprocessing module
--------------------

The preprocessing module consists of methods that handles missing
data, encoding of categorical features, scaling of the features, 
and also it creates train and test data sets. Currently, only 
target encoding, catboost encoding and ordinal encoding are 
implemented. The encoders used in this pipeline are taken from
Category Encoders package. If the data dosn't contain any null values,
the the `data_contains_nullval` flag in the `input_data.yaml` file can
be set to `False`. Similarly, if there is no categorical feature in the 
data, then `requires_feature_encoding` falg can be set to `False`.
Moreover, the feature scaling can be controlled by setting the 
`requires_feature_scaling` flag to either `True` or `False`.

Feature engineering module
--------------------------

The feature engineering module is to be included by the user as this is
problem specific. In this distribution of the code, I have provided a 
specific case for car price prediction. The module is an external function
that carries out the necessary feature engineering. One just needs to keep
this method inside the `regression_pipeline` source code directory with the
same module name as it is shown in this specific case. The rest will be
taken care of. Please note if the data provided is already have the engineered
features and no further feature engineering is required, then switch of the
feature engineering module by setting the `requires_feature_engineering` flag to
`False` in the `input_data.yaml` file. 

Feature selection module
------------------------

This module contains methods that implements various feature selection methods.
Most of these methods are taken from scikit-learn. On top of these methods,
BorutaPy method is also implemented. Finally, one of the methods in the module
will select those features which are selected by majority of the methods using
mejority voting. These various feature sets are then used to get the best
performing model for the problem. If the dataset already contains the appropriate
features and no feature selection process is requires then the 
`requires_feature_selection` flag can be set to `False`. 

Feature transformation module
-----------------------------

This module contains various mathematical methods that can be used to transform
the target or any of the features in the data. Many a times, if the data
contains very skewed features, this module can be used to transform the features to
a nearly normal distribution. Various trasnformations such as logarithmic, 
square, sqaure root, reciprocal, and power transformations are implemented in this 
module. If no feature transformation and target transformations are required, then
`requires_feature_transformation` and `requires_target_transformation` flags can 
both be set to `False`.

Optuna optimization module
--------------------------

The hyperparameters of various models can be tuned using optuna. The optuna
optimization process is implemented in this module. This module can run with
all the selected feature sets and can tune Random Forest and Gradient boosting
models to figure out the best performing model. If hyperparameter tuning is not
required then `requires_hyperparam_opt` flag must be set to `False`.

Model evaluation module
-----------------------

This model evaluates the performance of a given model. The model performance is
measured in terms of a suitable metric as defined in scikit-learn.  

Vizualization module
---------------------

This model plots various importnat graphs to understand the model performances
in terms of train and test scores. 


The Directory Structure of the Pipeline
========================================

The pipeline, as explained above, is a combination of different modules. All the main
modules are located in `regression_pipeline directory`. The `config_create` directory
contains a script that generates the necessary config files to run the pipeline. The 
`create_yaml_data.py` script to generate the config files must be run from within the
`config_create` directory only. Once this script is run successfully, the config
files will be generated in the `config` directory. The dataset for the problem must always
be located in the `data` directory. The `output` directory will be generated after successful
running of the `main.py` script. This directory contains all the results of the model selection
process. 


Usage
======

The pipeline can be installed using the requirements.txt file provided. In order
to install the package, run the following command:

```shell

$pip install -r requirements.txt

```
After installing the package, one needs to carry out the following steps to finally run the 
code.

* Create the input configuration file using `create_yaml_data.py` scirpt located in 
`config_creator` directory. The necessary parameters that to be provided are explained
in the comments of the script itself. The input files necessary to run the code will be
stored in config directory. Two files will be created - one, general configuration to run
the model selection experiment and the other, to run the uptuna optimization. If the optuna
optimization is not required, one can ignore this file. The `input_data.yaml` is
the general configuration file and `hyperopt_data.yaml` is the optuna configuration file. 

* To create the config files, go inside the `config_creator` directory, provide all the parameters
in the `create_yaml_data.py` scirpt as per the problem at hand, and then run the script with

```shell

$python create_yaml_data.py

```

* Once the configuration files are ready, go to the parent directory where the `main.py` script is
located. The `main.py` script is to be run to carry out the model selection experiments. This script
can be run with the following command:

```shell

$python main.py --gen-config config/input_data.yaml [--optuna-config config/hyperopt_data.yaml]

``` 

* After the code is run succefully, an `output` direcotory will be created where all the necessary plots
and the results in terms of the perfromance of all the models tried will be generated. 
