# comboKR2.0

A scaled-up version of the comboKR (https://github.com/aalto-ics-kepaco/comboKR/) for drug combination surface prediction.

## System requirements

The code is developed with python 3.8. 

The main algorithm in scalable_comboKR.py is run with numpy 1.23.5, and scikit-learn 1.0.2. It relies on the RLScore (https://github.com/aatapa/RLScore) and synergy (https://github.com/djwooten/synergy) packages.
The demo depends additionally on some other usual python packages, such as scipy and matplotlib. 


## Installation guide

Before installing the comboKR2.0 package make sure that latest versions of pip and build are installed:

>`pip3 install --upgrade pip`

>`pip3 install --upgrade build`

There are two options for installing the comboKR package. 

### Directly from the github

>`pip3 install git+https://github.com/aalto-ics-kepaco/comboKR2.0.git#egg=comboKR2.0`

### Downloading from github

>`mkdir comboKR2.0`

>`cd comboKR2.0`

>`git clone https://github.com/aalto-ics-kepaco/comboKR2.0`

After downloading the comboKR2.0 package, it can be installed by the following command from the comboKR2.0 directory:

>`pip3 install .`

## Demo

A small-scale demo based on O'Neil dataset [1] is provided in demo.py. Before running it, download and unpack the data.zip. The expected runtime of the demo is about 20 minutes; much less if candidate set optimisation is used instead of the projected gradient descent. 


[1] O'Neil, J., Benita, Y., Feldman, I., Chenard, M., Roberts, B., Liu, Y., ... & Shumway, S. D. (2016). An unbiased oncology compound screen to identify novel combination strategies. Molecular cancer therapeutics, 15(6), 1155-1162.
