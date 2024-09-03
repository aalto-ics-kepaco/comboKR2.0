import numpy as np
import pickle

import time
from datetime import timedelta

from scalable_comboKR import SurfaceKernelNystromDifference
from scalable_comboKR import ScalablecomboKRdiff as ComboKR2


"""
An example of how to run comboKR 2.0 code. Skipping the cross-validation here!

Based on O'Neil dataset
"""


# decide with which regularisation parameter (early stopping interations) to run the experiment
n_iters = 5

# ======================================================================================================================
# load the needed data
print("Loading the data and building the required arrays")

# the kernel matrices
Kcell = np.load("data/Kc.npy")
Kdrug = np.load("data/Kd.npy")
# feature names, to identify which row&column in kernel matrices correspond to which cell/drug
cell_ids = np.load("data/cellfeat_ids.npy")
drug_ids = np.load("data/drugfeat_ids.npy")

# training & test split - identified with triplets indetifying cell line, drug1, and drug2.
# note for training: it is good practice to include both drug orders! todo check how I save this
# the ids in first column should all be present in cell_ids
# the ids in the last two columns should be from drug_ids
cdd_tr = np.load("data/cdd_tr.npy")  # of size n_tr*3
cdd_tst = np.load("data/cdd_tst.npy")  # of size n_tst*3
print("cdd_tr shape:", cdd_tr.shape)
print("cdd_tst shape:", cdd_tst.shape)

# load the fitted monotherapy and drug combo surface functions

# monotherapies as Hill equations
# dataframe always indicates the cell line and the drug id
with open("data/fitted_hills_full.df.pkl", "rb") as f:
    hills = pickle.load(f)
# the parameters of the fitted BRAID functions to the surfaces
braids_all = np.load("data/fitted_braid_params_full.npy")
# the functions are in the same order as the rows in this one:
cdd_full = np.load("data/cdd_full.npy")

# for the test data, load the dose concentrations and the measured responses at those concentrations
# these have been vectorised; every row corresponds to one surface
# full data arrays are again ordered as the cdd_full
c1_all = np.load("data/conc1Median.npy")
c2_all = np.load("data/conc2Median.npy")
y_all = np.load("data/measurementsMedian.npy")

# select the subsets of data for training and testing
# find where in cdd_full the cdd_tr and cdd_tst indices are

tr_inds = []
for ii in range(cdd_tr.shape[0]):
    tr_inds.append(np.where((cdd_full[:, 0] == cdd_tr[ii, 0]) &
                            (cdd_full[:, 1] == cdd_tr[ii, 1]) &
                            (cdd_full[:, 2] == cdd_tr[ii, 2]))[0][0])
tst_inds = []
for ii in range(cdd_tst.shape[0]):
    tst_inds.append(np.where((cdd_full[:, 0] == cdd_tst[ii, 0]) &
                             (cdd_full[:, 1] == cdd_tst[ii, 1]) &
                             (cdd_full[:, 2] == cdd_tst[ii, 2]))[0][0])
tr_inds = np.array(tr_inds)
tst_inds = np.array(tst_inds)

braids_tr = braids_all[tr_inds, :]
# it is good practice to "double" the training data, i.e. consider both drug pair orders in training
cdd_tr = np.vstack((cdd_tr, np.hstack((cdd_tr[:, 0][:, None], cdd_tr[:, 2][:, None], cdd_tr[:, 1][:, None]))))
braids_tr2 = braids_tr[:, [0, 2, 1, 3, 5, 4, 7, 6, 8]]
braids_tr = np.vstack((braids_tr, braids_tr2))

c1_tst = c1_all[tst_inds, :]
c2_tst = c2_all[tst_inds, :]
y_tst = y_all[tst_inds, :]

# ======================================================================================================================
# train the model
#
# The output data approximation

output_nystrom = SurfaceKernelNystromDifference(tol=1e-1, kernel="rbf", max_is_100=False)
output_nystrom.create_approximation()

combokr2 = ComboKR2(Kcell, Kdrug, cell_ids, drug_ids, output_nystrom, hills=hills, max_is_100=False)
# train (basically just initialise stuff, due to memory concers the actual training happens in predict)
combokr2.train(cdd_tr, braids_tr, n_iters)
# predict => Expected running time: about 20 minutes
print("Starting to train and test")
print("This might take a while due to the step of fitting BRAID surfaces to the GD solution..")
t0 = time.process_time()
preds, pred_braid_params = combokr2.predict_with_gd_nadam_and_braid(cdd_tst, c1_tst, c2_tst)
t1 = time.process_time()
print("It took this long to train and test comboKR 2.0:", timedelta(seconds=t1-t0))  # 0:18:54.285831

# check how well it went
print("Pearson correlation:", np.round(np.corrcoef(y_tst.ravel(), preds.ravel())[0, 1], 3))
from scipy.stats import spearmanr
print("Spearman correlation:", np.round(spearmanr(y_tst.ravel(), preds.ravel())[0], 3))

# Pearson correlation: 0.939
# Spearman correlation: 0.938

import matplotlib.pyplot as plt
plt.figure()
plt.scatter(y_tst.ravel(), preds.ravel(), s=0.02)
plt.xlabel("groundtruth")
plt.ylabel("prediction")
plt.show()
