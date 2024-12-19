# comboKR2.0

A scaled-up version of the comboKR (https://github.com/aalto-ics-kepaco/comboKR/) for drug combination surface prediction.

## System requirements

The code is developed with python 3.9. The main dependencies are the RLScore (https://github.com/aatapa/RLScore) and synergy (https://github.com/djwooten/synergy) packages. From these, the versions 0.8.2a (RLScore) and 0.5.1 (synergy) have been used. 

**Note:** 
- Synergy package is not backwards compatible! Newer versions than 0.5.1 currently exist, but using those will result in errors.
- The RLScore package (0.8.2a0) is not available in PyPI! When installing the environment with pip, it should be installed separately.

The main algorithm in scalable_comboKR.py has been run with numpy 1.23.5, and scikit-learn 1.0.2. The demo depends additionally on some other usual python packages, such as scipy and matplotlib. 

## Installation guide

### With conda

A suitable conda environment can be created with the provided yml file, with which the algorithm can then be used. 

> conda env create -f environment.yml -n combokr2-env

Alternatively, the package can be installed with pip.

### With pip

Before installing the comboKR2.0 package make sure that latest versions of pip and build are installed:

>`pip3 install --upgrade pip`

>`pip3 install --upgrade build`

There are two options for installing the comboKR package. 

#### Directly from the github

>`pip3 install git+https://github.com/aalto-ics-kepaco/comboKR2.0.git#egg=comboKR2.0`

#### Downloading from github

>`mkdir comboKR2.0`

>`cd comboKR2.0`

>`git clone https://github.com/aalto-ics-kepaco/comboKR2.0`

After downloading the comboKR2.0 package, it can be installed by the following command from the comboKR2.0 directory:

>`pip3 install .`

#### RLScore

The RLScore package (https://github.com/aatapa/RLScore) has to be installed separately.

## Demo

A small-scale demo based on O'Neil dataset [1] is provided in demo.py. Before running it, download and unpack the data.zip. The expected runtime of the demo is about 20 minutes; much less if candidate set optimisation is used instead of the projected gradient descent. 


[1] O'Neil, J., Benita, Y., Feldman, I., Chenard, M., Roberts, B., Liu, Y., ... & Shumway, S. D. (2016). An unbiased oncology compound screen to identify novel combination strategies. Molecular cancer therapeutics, 15(6), 1155-1162.
