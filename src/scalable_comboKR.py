import sys
import time
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
from rlscore.learner import CGKronRLS

from sklearn.metrics.pairwise import rbf_kernel, pairwise_distances

from helpers import mat_to_vec, vec_to_mat, c_from_normalized_c
from braid_surface_model_for_fit import braid_model_with_raw_c_input, MyBraidWithOptimisers, braid_model_with_log_c_input

# the code relies on some functionalities of the synergy package: https://github.com/djwooten/synergy
# from the "braid_surface_model_for_fit" import

"""
## ***********************************************************************/
##    This file contains the code for comboKR2.0, a scalable approach for  
##    predicting drug combination surfaces. 
##
##     MIT License
##     Copyright (c) 2024 KEPACO
##
##     Permission is hereby granted, free of charge, to any person obtaining
##     a copy of this software and associated documentation files (the
##     "Software"), to deal in the Software without restriction, including
##     without limitation the rights to use, copy, modify, merge, publish,
##     distribute, sublicense, and/or sell copies of the Software, and to
##     permit persons to whom the Software is furnished to do so, subject
##     to the following conditions:
##
##     The above copyright notice and this permission notice shall be
##     included in all copies or substantial portions of the Software.
##
##     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
##     EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
##     MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
##     IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
##     CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
##     TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
##     SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
## 
## ***********************************************************************/

@ Riikka Huusari, 2024
"""

"""
This file implements the scalable comboKR 2.0 code. The classes are:

- SurfaceKernelNystrom
- SurfaceKernelNystromDifference
- ScalableComboKR
- ScalableComboKRdiff


The first two implement Nyström approximation for the output surfaces; first for when direct surface prediction is
considered, second for difference surface prediction. 

The last two implement the comboKR 2.0. 
  - "train" is the same for all variants
  - there are multiple different "predict..." functions, depending on the pre-image solver type
    * "predict_with_candidates" uses candidate set optimisation
    * "predict_with_gd_nadam_and_braid" uses projected gradient descent (using nesterov-accelerated Adam)
    * "predict_with_neutral_braid" includes NO LEARNING! It's a baseline prediction
  - note: due to memory concerns, the actual prediction happens in "_predict_raw", and it's called in the beginning of
    the prediction functions. For each output at a time, it trains the learner (using RLScore package), predicts, and
    for the next output overwrites the previous learner. Training all learners separately and then predicting with them
    afterwards takes too much memory. 

"""


class SurfaceKernelNystrom:

    def __init__(self, tol=1e-6, verbosity=0, kernel="rbf", max_is_100=True):

        """

        :param tol: Affects how many surfaces are chosen for the Nyström approximation of the outputs;
                    the smaller this is, the more are chosen
        :param verbosity: How much to print of the progress
        :param kernel: Kernel on outputs. "rbf", "linear", "poly3", "linearn", "poly3n" ("n" = normalised)
        :param max_is_100: Is the response data in [0, 100] (True) or in [0, 1] (False)
        """

        C2, C1 = np.meshgrid(np.arange(0, 1.1, 0.1), np.arange(0, 1.1, 0.1))
        self.c1 = mat_to_vec(C1)
        self.c2 = mat_to_vec(C2)
        self.tol = tol
        self.verbosity=verbosity
        self.kernel = kernel

        if max_is_100:
            # the values are in range [0, 100] (could be higher but most of them); ALMANAC and Jaaks datasets
            self.max_responseval = 100
        else:
            # the values are in range [0, 1]; e.g. O'Neil dataset
            self.max_responseval = 1

    def create_approximation(self, n_rmax=1):

        slopes = [0.2, 1, 3]
        max_responses = [self.max_responseval*0.75, self.max_responseval*0.50, self.max_responseval*0.25, 0]
        kappas = [-1.5, -0.5, 0, 0.5, 5, 20]

        if n_rmax == 3:
            r3maxes = [0, self.max_responseval*0.20, self.max_responseval*0.40]
        if n_rmax == 2:
            r3maxes = [self.max_responseval*0.10, self.max_responseval*0.30]
        if n_rmax == 1:
            r3maxes = [self.max_responseval*0.20]
        else:
            raise NotImplementedError

        braids = []
        for slope1 in slopes:
            for slope2 in slopes:
                for maxr1 in max_responses:
                    for maxr2 in max_responses:
                        for kappa in kappas:
                            br = [self.max_responseval, maxr1, maxr2, np.minimum(maxr1, maxr2), slope1, slope2, 1e-6, 1e-6, kappa]
                            n_braids = len(braids)
                            for rmax3 in r3maxes:
                                # if it doesn't make surface invalid
                                if np.minimum(maxr1, maxr2) - rmax3 >= 0:
                                    br[3] = np.minimum(maxr1, maxr2) - rmax3
                                    braids.append(br)
                            n_braids2 = len(braids)
                            if n_braids2 == n_braids:
                                braids.append(br)
        if self.verbosity>0:
            print("created in total " + str(len(braids)) + " surface functions!")

        braids = np.array(braids)

        # then build the kernel - need to sample all of these at the relevant concentrations
        # only normalised for now - original is not realistic as in all EC_50 is the same

        sampled_surfaces = self._get_samples_on_grid(braids)

        if self.kernel == "rbf":
            sigma = np.mean(pairwise_distances(sampled_surfaces))
            self.gamma = 1 / (2 * (sigma ** 2))
            K = rbf_kernel(sampled_surfaces, gamma=self.gamma)
        elif self.kernel == "linear":
            K = np.dot(sampled_surfaces, sampled_surfaces.T)
        elif self.kernel == "poly3":
            d = 3
            K = (np.dot(sampled_surfaces, sampled_surfaces.T)) ** d
        elif self.kernel == "linearn":
            K = (np.dot(sampled_surfaces, sampled_surfaces.T))
            kww = ((sampled_surfaces * sampled_surfaces).sum(-1))  # diagonal elements of dot products

            # what if there is a zero there? there really shouldn't be...

            K = K / np.sqrt(kww[:, None] * kww[None, :])
        elif self.kernel == "poly3n":
            d = 3
            K = (np.dot(sampled_surfaces, sampled_surfaces.T) ) ** d
            kww = ((sampled_surfaces * sampled_surfaces).sum(-1)) ** d  # diagonal elements of dot products

            K = K / np.sqrt(kww[:, None] * kww[None, :])

        else:
            raise NotImplementedError

        try:
            eigvals = np.linalg.eigvalsh(K)
        except np.linalg.LinAlgError:
            print("zeroes in kww", np.where(kww == 0))
            print("nans in K", np.where(np.isnan(K)))
            raise
        eigvals[eigvals < self.tol] = 0
        rank = np.count_nonzero(eigvals)
        # print("#eigenvalues:", np.count_nonzero(eigvals))

        from scipy.linalg import qr as scqr
        # Q, R = scqr(Phix[rows, :], pivoting=False)
        _, _, P = scqr(K, pivoting=True)
        # print("P:\n", P)

        indices = P[:rank]

        self.braids = braids[indices, :]
        self.nystrom_surfaces = sampled_surfaces[indices, :]

        Ksmall = K[np.ix_(indices, indices)]

        eigvals, eigvecs = np.linalg.eigh(Ksmall)
        zeroinds = np.where(eigvals<self.tol)[0]
        eigvals = 1/np.sqrt(eigvals)
        eigvals[zeroinds] = 0
        self.Wsqinv = np.dot(eigvecs, np.dot(np.diag(eigvals), eigvecs.T))
        if self.verbosity > 0:
            print("nyström approximation created successfully")
            print("using only", Ksmall.shape[0], "surfaces")

    def _get_samples_on_grid(self, braids):

        """
        self.c1 and self.c2 are normalised concentrations on the grid that was specified in __init__

        :param braids: Braid functions, assumes [Emax, E1, E2, E3, h1, h2, C1, C2, kappa] (hmm did I have h and C order correct here in comment?)
        :return:
        """

        sampled_surfaces = []
        for bb in range(braids.shape[0]):
            # sampling the hill equations from BRAID
            c1_original = c_from_normalized_c(self.c1, *braids[bb, [0, 1, 4, 6]])
            c2_original = c_from_normalized_c(self.c2, *braids[bb, [0, 2, 5, 7]])
            surf = braid_model_with_raw_c_input([c1_original, c2_original], *braids[bb, :])
            sampled_surfaces.append(surf)
        sampled_surfaces = np.array(sampled_surfaces)
        return sampled_surfaces

    def get_approximated_features_grid(self, surfs):
        if self.kernel == "rbf":
            K = rbf_kernel(surfs, self.nystrom_surfaces, gamma=self.gamma)
        elif self.kernel == "linear":
            K = np.dot(surfs, self.nystrom_surfaces.T)
        elif self.kernel == "poly3":
            d=3
            K = (np.dot(surfs, self.nystrom_surfaces.T)) ** d
        elif self.kernel == "linearn":
            K = np.dot(surfs, self.nystrom_surfaces.T)
            kd1 = ((surfs * surfs).sum(-1))  # diagonal elements of dot products
            kd2 = ((self.nystrom_surfaces * self.nystrom_surfaces).sum(-1))  # diagonal elements of dot products
            K = K/(np.sqrt(kd1[:, None] * kd2[None, :]))
        elif self.kernel == "poly3n":
            d=3
            K = (np.dot(surfs, self.nystrom_surfaces.T)) ** d
            kd1 = ((surfs * surfs).sum(-1))**d  # diagonal elements of dot products
            kd2 = ((self.nystrom_surfaces * self.nystrom_surfaces).sum(-1))**d  # diagonal elements of dot products
            K = K/(np.sqrt(kd1[:, None] * kd2[None, :]))
        else:
            raise NotImplementedError
        return np.dot(K, self.Wsqinv)

    def get_approximated_features_braid(self, braids):
        surfs = self._get_samples_on_grid(braids)
        surfs = self.fix_Ytr(surfs, int(np.sqrt(len(self.c1))))
        assert not np.any(np.isnan(surfs))
        return self.get_approximated_features_grid(surfs)

    def gradient_over_kernel(self, y, z):

        # !!! z depends on self.Wsqinv!! but it's way more efficient to calculate outside
        # todo merge this in scalable iokr class?

        if self.kernel == "rbf":
            ky = rbf_kernel(y[None, :], self.nystrom_surfaces, gamma=self.gamma)
            res = np.sum(2*self.gamma*z[:, None]*ky.T*self.nystrom_surfaces, axis=0)
            return res
        elif self.kernel == "linear":
            d = 1
            t0 = np.dot(y[None, :], self.nystrom_surfaces.T)
            t1 = np.power(t0, d - 1)
            return d * self.nystrom_surfaces.T * t1[:, np.newaxis]
        elif self.kernel == "poly3":
            d = 3
            t0 = np.dot(y[None, :], self.nystrom_surfaces.T)
            t1 = np.power(t0, d - 1)
            return d * self.nystrom_surfaces.T * t1[:, np.newaxis]
        elif self.kernel == "linearn":
            d = 1
            r=0

            gx = (np.dot(y[None, :], self.nystrom_surfaces.T)) ** d
            t1 = np.power(gx, d - 1)
            dgx = d * self.nystrom_surfaces.T * t1[:, np.newaxis]

            kuu = (np.dot(y[None, :], y[None, :].T) + r) ** d
            kww = ((self.nystrom_surfaces * self.nystrom_surfaces).sum(-1) ) ** d
            hx2 = kuu * kww
            hx = np.sqrt(kuu * kww)
            dkuu = 2 * d * (np.dot(y[None, :], y[None, :].T) + r) ** (d - 1) * y[None, :]
            dhx = dkuu * (0.5 * (kuu * kww) ** (-0.5) * kww)[:, np.newaxis]

            # derivative with quotient rule
            return (dgx * hx[:, np.newaxis] - dhx * gx[:, np.newaxis]) / hx2[:, np.newaxis]
        elif self.kernel == "poly3n":
            d = 3
            r = 0

            gx = (np.dot(y[None, :], self.nystrom_surfaces.T)) ** d
            t1 = np.power(gx, d - 1)
            dgx = d * self.nystrom_surfaces.T * t1[:, np.newaxis]

            kuu = (np.dot(y[None, :], y[None, :].T) + r) ** d
            kww = ((self.nystrom_surfaces * self.nystrom_surfaces).sum(-1) ) ** d
            hx2 = kuu * kww
            hx = np.sqrt(kuu * kww)
            dkuu = 2 * d * (np.dot(y[None, :], y[None, :].T) + r) ** (d - 1) * y[None, :]
            dhx = dkuu * (0.5 * (kuu * kww) ** (-0.5) * kww)[:, np.newaxis]

            # derivative with quotient rule
            return (dgx * hx[:, np.newaxis] - dhx * gx[:, np.newaxis]) / hx2[:, np.newaxis]
        else:
            raise NotImplementedError

    def fix_Ytr(self, Ytr, matsize=4):

        """
        Ytr is assumed to be a matrix, where each row is vectorised sampled surface

        This script "fixes" the surfaces in case there are nan values
        The fix is replicating the previous values in row/column to replace the nan values

        :param Ytr: each row in here is vectorised surface matrix (each matrix should be d*d)
        :param matsize: d
        :return:
        """

        # ¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤
        # rows that are in unique rows here and in inds_in_tr
        [rows, cols] = np.where(np.isnan(Ytr))
        # print("rows and cols with nans before:\n", rows, cols)
        rows_to_fix = np.unique(rows)
        # print("need to fix", len(rows_to_fix), "rows! ")

        bad_braids3 = []

        if (Ytr.shape[1] == 14) or (Ytr.shape[1] != matsize**2):  # if this is not a regular grid I cannot do this
            print("This error is due to non-square matrix to be fixed in fix_Ytr function")
            print("The specific size 14 corresponds to Jaaks data, 7*2")
            print("For me this has not been a problem - with Jaaks data this function seems not to have been ever called")
            raise NotImplementedError

        for rr, rowindx in enumerate(np.unique(rows_to_fix)):

            vec = Ytr[rowindx, :]
            cols_here = np.where(np.isnan(vec))[0]
            # print("row:", rowindx)

            # print(vec)
            # print(cols_here)

            assert np.any(np.isnan(vec))

            # three possibilities: (1) all are nan, (2) only some are, consecutive, or (3) some are but they jump
            if len(cols_here) == matsize ** 2:
                # print("*")
                bad_braids3.append(rowindx)
            elif len(cols_here) == 1:
                if cols_here[0] % (matsize) == matsize - 1:  # at the end of a column
                    if cols_here[0] == matsize - 1:  # first column
                        vec[cols_here[0]] = vec[cols_here[0] - 1]
                    else:
                        vec[cols_here[0]] = np.nanmean([vec[cols_here[0] - 1], vec[cols_here[0] - matsize]])
                elif cols_here[0] >= matsize * (matsize - 1):
                    # last in a row
                    if cols_here[0] == matsize * (matsize - 1):
                        # first row
                        vec[cols_here[0]] = vec[cols_here[0] - matsize]
                    else:
                        vec[cols_here[0]] = np.nanmean([vec[cols_here[0] - matsize], vec[cols_here[0] - 1]])
                else:
                    # it's in the middle... take mean of values around
                    indices_around = []
                    # not in first col
                    if cols_here[0] >= matsize:
                        indices_around.append(cols_here[0] - matsize)
                    if cols_here[0] < matsize * (matsize - 1):  # not in last col
                        indices_around.append(cols_here[0] + matsize)
                    if cols_here[0] % matsize != 0:  # not on first row
                        indices_around.append(cols_here[0] - 1)
                    if cols_here[0] % matsize != (matsize - 1):  # not on last row
                        indices_around.append(cols_here[0] + 1)
                    vec[cols_here[0]] = np.nanmean(vec[indices_around])
            elif (cols_here[-1] - cols_here[0]) == (len(cols_here) - 1):
                # print("-")
                # consecutive indexing; column
                """  example
                0 4 8  12
                1 5 9  13
                2 6 10 14
                3 7 11 15
                """
                # what if this is the first column?

                # take tha last available column and repeat
                if len(cols_here) == 1:
                    vec[cols_here[0]] = vec[cols_here[0] - 1]

                else:
                    # todo was there a bug here?
                    # ok this is available +1 actually
                    colindx_available = (cols_here[0] // matsize) * matsize

                    last_col = vec[colindx_available - 1:colindx_available]
                    indx_for_col_to_fill = cols_here[0] % matsize
                    # while indx_for_col_to_fill < matsize ** 2:
                    try:
                        vec[cols_here[0]:colindx_available] = last_col[indx_for_col_to_fill:]
                    except:
                        print("debugging...")
                        print(vec)
                        print(cols_here)
                        print(indx_for_col_to_fill)
                        print(last_col)
                        raise
                    #     indx_for_col_to_fill += matsize
                Ytr[rowindx, :] = vec
            else:
                # print("¤")
                # jumping; rows

                update_rowindx = np.arange(matsize) * matsize
                prev_rowindx = np.copy(update_rowindx)
                done = False
                while np.all(prev_rowindx < matsize ** 2):
                    new_rowindx = prev_rowindx + 1
                    if new_rowindx[0] in cols_here:
                        # nan, assign as previous row
                        vec[new_rowindx] = vec[update_rowindx]
                    else:
                        # update row that should be used to fill in other rows
                        update_rowindx = new_rowindx
                    prev_rowindx = new_rowindx
                    if matsize ** 2 - 1 in new_rowindx:
                        break
                Ytr[rowindx, :] = vec

                # it can be both! so test still for consecutive once more here
                if np.any(np.isnan(vec)):
                    cols_here = np.where(np.isnan(vec))[0]
                    if (cols_here[-1] - cols_here[0]) == (len(cols_here) - 1):
                        # print("-")
                        # consecutive indexing; column
                        """  example
                        0 3 6
                        1 4 7
                        2 5 8
                        """
                        # take tha last available column and repeat

                        if len(cols_here) == 1:
                            vec[cols_here[0]] = vec[cols_here[0]-1]

                        else:
                            # todo was there a bug here?
                            # ok this is available +1 actually
                            colindx_available = (cols_here[0]//matsize)*matsize

                            last_col = vec[colindx_available-1:colindx_available]
                            indx_for_col_to_fill = cols_here[0]%matsize
                            try:
                                vec[cols_here[0]:colindx_available] = last_col[indx_for_col_to_fill:]
                            except:
                                print("debugging here:")
                                print(vec)
                                print(matsize, len(vec), len(cols_here))
                                print(cols_here)
                                print(matsize)
                                raise
                        Ytr[rowindx, :] = vec

            # finally, just try to fix them one-by-one:
            for iter in range(2):
                for ii in range(len(cols_here)):
                    indices_around = []
                    # not in first col
                    if cols_here[ii] >= matsize:
                        indices_around.append(cols_here[ii] - matsize)
                    if cols_here[ii] < matsize * (matsize - 1):  # not in last col
                        indices_around.append(cols_here[ii] + matsize)
                    if cols_here[ii] % matsize != 0:  # not on first row
                        indices_around.append(cols_here[ii] - 1)
                    if cols_here[ii] % matsize != (matsize - 1):  # not on last row
                        indices_around.append(cols_here[ii] + 1)
                    vec[cols_here[ii]] = np.nanmean(vec[indices_around])
            if np.any(np.isnan(vec)):
                bad_braids3.append(rowindx)

        if len(bad_braids3) > 0:
            if np.any(Ytr > 50):
                Ytr[bad_braids3, :] = 100  # whatever at this point.....
            else:
                Ytr[bad_braids3, :] = 1  # whatever at this point.....

        return Ytr


class SurfaceKernelNystromDifference(SurfaceKernelNystrom):

    def _get_samples_on_grid(self, braids):
        sampled_surfaces = []
        for bb in range(braids.shape[0]):
            c1_original = c_from_normalized_c(self.c1, *braids[bb, [0, 1, 4, 6]])
            c2_original = c_from_normalized_c(self.c2, *braids[bb, [0, 2, 5, 7]])
            zero_br = np.copy(braids[bb, :])
            zero_br[-1] = 0
            zero_surf = braid_model_with_raw_c_input([c1_original, c2_original], *zero_br)
            surf = braid_model_with_raw_c_input([c1_original, c2_original], *braids[bb, :])
            sampled_surfaces.append(surf-zero_surf)
        sampled_surfaces = np.array(sampled_surfaces)
        return sampled_surfaces


class ScalableComboKR:

    """
    More memory efficient version

    It seems that the memory issues are directly from number of output tasks.
    So, cannot save the learners, but have to train and test and discard.

    Training happens at test time.
    """

    def __init__(self, Kcell, Kdrug, cell_ids, drug_ids, output_nystrom, hills=None, verbosity=0, max_is_100=True):

        """

        :param Kcell:
        :param Kdrug:
        :param cell_ids:
        :param drug_ids:
        :param output_nystrom: SurfaceKernelNystrom or SurfaceKernelNystromDifference object
        :param hills:
        :param verbosity: How much is printed during the run: 0=very little, 1=something. Verbosity=2 was used
        in debugging, most of those have been removed when cleaning up the code.
        :param max_is_100: True: response data in [0, 100], False: response data in [0, 1]
        """

        self.Kc = Kcell
        self.Kd = Kdrug
        self.cell_ids = cell_ids
        self.drug_ids = drug_ids

        self.nystrom_y = output_nystrom  # should be object of class above

        self.hills = hills

        if verbosity > 0:
            if self.hills is None:
                print("WARNING: using random initialisation in gd, not the reasonable guess from Hills!")
            else:
                print("Using initial guess in gd based on Hills :) ")

        self.verbosity = verbosity
        print("CREATING verbosity", self.verbosity)

        if max_is_100:
            # the values are in range [0, 100] (could be higher but most of them); ALMANAC and Jaaks datasets
            self.max_responseval = 100
        else:
            # the values are in range [0, 1]; e.g. O'Neil dataset
            self.max_responseval = 1

        nc = Kcell.shape[0]
        nd = Kdrug.shape[0]
        print("n_c, n_d:", nc, nd)
        print("in kron:", nc * nd, nd)

    def _get_empirical_feats(self, K):
        eigvals, eigvecs = np.linalg.eigh(K)
        zeroinds = np.where(eigvals < 1e-12)[0]
        eigvals = 1 / np.sqrt(eigvals)
        eigvals[zeroinds] = 0
        efeats = np.dot(eigvecs, np.dot(np.diag(eigvals), eigvecs.T))
        return efeats

    def _get_inds_from_cdd(self, cdd):

        # there will be (Kc\otimes Kd) \otimes Kd
        # from cdd I get the inds for where in Kc and Kd -> ind_d2 directly usable
        # ind in Kc \otimes Kd will be just ind_c*n_c+ind_d1

        n_c = self.Kc.shape[0]
        n_d = self.Kd.shape[0]
        print("Kc:", self.Kc.shape)

        inds1 = []
        inds2 = []

        for ii in range(cdd.shape[0]):
            cell, drug1, drug2 = cdd[ii, :]
            ind_c = np.where(self.cell_ids == cell)[0]
            ind_d1 = np.where(self.drug_ids == drug1)[0]
            ind_d2 = np.where(self.drug_ids == drug2)[0]
            inds1.append(ind_c * n_d + ind_d1)
            inds2.append(ind_d2)
        return np.array(inds1, dtype=np.int32), np.array(inds2, dtype=np.int32)

    def train(self, cell_drug_drug_tr, braids_tr, maxiter):

        print("BRAIDS shape:", braids_tr.shape)

        self.maxiter = maxiter

        self.cdd_tr = cell_drug_drug_tr
        self.mat1_inds, self.mat2_inds = self._get_inds_from_cdd(self.cdd_tr)
        print("#training samples:", cell_drug_drug_tr.shape[0])

        print("make output features...")
        self.Phiy_tr = self.nystrom_y.get_approximated_features_braid(braids_tr)
        print("..done!", self.Phiy_tr.shape[1])

    def predict_with_candidates(self, cell_drug_drug_tst, c1_tst, c2_tst):

        print("Predicting raw values in the approximated output feature space")
        raw_preds = self._predict_raw(cell_drug_drug_tst)
        print("Predicting raw values done")

        preds = []
        pred_params = []
        # predicting single way
        for ii in range(cell_drug_drug_tst.shape[0]):
            # print("start with indx", ii)

            cell, drug1, drug2 = cell_drug_drug_tst[ii, :]

            # --------------------------------------------------------------------------------------------
            # build the candidates
            h1 = self.hills[(self.hills["drug"] == drug1) & (self.hills["cell"] == cell)]
            h2 = self.hills[(self.hills["drug"] == drug2) & (self.hills["cell"] == cell)]
            h1_np = [self.max_responseval, h1["Emax"].to_numpy()[0], h1["h"].to_numpy()[0], h1["C"].to_numpy()[0]]
            h2_np = [self.max_responseval, h2["Emax"].to_numpy()[0], h2["h"].to_numpy()[0], h2["C"].to_numpy()[0]]
            braid_params = [self.max_responseval, h1["Emax"].to_numpy()[0], h2["Emax"].to_numpy()[0], 0,
                            h1["h"].to_numpy()[0], h2["h"].to_numpy()[0],
                            h1["C"].to_numpy()[0], h2["C"].to_numpy()[0], 0]

            # kappas = [-1.99, -1.5, -1, -0.5, -0.1, 0.01, 0.1, 0.5, 1, 2, 10, 25, 50]
            kappas = [0.01, 0.1, -0.1, 0.5, -0.5, 1, -1, 2, -1.5, 10]
            start = np.maximum(np.minimum(h1["Emax"].to_numpy()[0], h2["Emax"].to_numpy()[0]) - self.max_responseval*0.50, 0)
            end = np.minimum(self.max_responseval*1.20, np.maximum(h1["Emax"].to_numpy()[0], h2["Emax"].to_numpy()[0]) + self.max_responseval*0.10)
            Emaxs = np.arange(start, end, (end - start) / 4)

            candidate_evals = []
            all_param_combinations = []
            for kappa in kappas:
                braid_params[8] = kappa
                for Emax in Emaxs:
                    braid_params[3] = Emax
                    candidate_evals.append(braid_model_with_raw_c_input([c_from_normalized_c(self.nystrom_y.c1, *h1_np),
                                                                         c_from_normalized_c(self.nystrom_y.c2,
                                                                                             *h2_np)],
                                                                        *braid_params))
                    all_param_combinations.append(np.copy(braid_params))
            candidate_evals = np.array(candidate_evals)
            all_param_combinations = np.array(all_param_combinations)
            # if there happens to be a nan in the candidate evaluation, do not include it!
            if np.any(np.isnan(candidate_evals)):
                nonnanrows = np.where(~np.isnan(np.sum(candidate_evals, axis=1)))[0]
                if len(nonnanrows) > 0:
                    candidate_evals = candidate_evals[nonnanrows, :]
                    all_param_combinations = all_param_combinations[nonnanrows, :]
                else:
                    print("all candidate evals contained nan!", cell, drug1, drug2)
                    candidate_evals[np.isnan(candidate_evals)] = self.max_responseval

            phic = self.nystrom_y.get_approximated_features_grid(candidate_evals)

            if self.nystrom_y.kernel == "linear" or self.nystrom_y.kernel == "poly3":
                scores = np.diag(np.dot(phic, phic))-2*np.dot(phic, raw_preds[ii, :])

                best_indx = np.argmin(scores)
            else:
                scores = np.dot(phic, raw_preds[ii, :])

                best_indx = np.argmax(scores)

            preds.append(
                braid_model_with_raw_c_input([c1_tst[ii, :], c2_tst[ii, :]], *all_param_combinations[best_indx, :]))
            pred_params.append(all_param_combinations[best_indx, :])

        return np.array(preds), pred_params

    def _fit_with_torch_surface_nonnormalised(self, z, initial, nystrom_surfaces, nystrom_object):

        import torch
        from torch import nn

        tensorz = torch.tensor(z)
        tensorinitial = torch.tensor(initial)

        tensor_n_surfs = torch.tensor(nystrom_surfaces)
        tensor_wsqinv = torch.tensor(nystrom_object.Wsqinv)

        def torch_kernel(x, Z):

            if nystrom_object.kernel == "linear":
                polyk = (torch.vm(x, Z))
                return polyk
            elif nystrom_object.kernel == "poly3":
                d = 3
                polyk = (torch.vm(x, Z)) ** d
                return polyk

        class Model(nn.Module):
            """Custom Pytorch model for gradient optimization.
            """

            def __init__(self):
                super().__init__()
                weights = torch.tensor(initial)
                # make weights torch parameters
                self.weights = nn.Parameter(weights)

            def forward(self):
                """Implement function to be optimised.
                """

                weights = self.weights

                K = torch_kernel(weights, tensor_n_surfs)
                val = torch.matmul(K, tensor_wsqinv)

                if nystrom_object.kernel == "linear":
                    res = torch.dot(weights, weights)-2*torch.dot(val, tensorz)
                else:
                    res = torch.dot(weights, weights)**3-2*torch.dot(val, tensorz)
                # print(res)
                return res

        def training_loop(model, optimizer, n=500):
            "Training loop for torch model."
            losses = []

            # from https://discuss.pytorch.org/t/want-to-maximise-a-function-do-i-use-a-torch-nn-loss-or-is-there-a-better-way/76842/2
            # how to use just the value of the function here
            # converged = False
            # n_converged = 0
            for i in range(n):
                # commented these out now that I have the closure:
                value = model()  # value is the value of the objective
                optimizer.zero_grad()
                value.backward() # why in places it's backwards, in other places backward?
                optimizer.step()
                losses.append(value.detach().numpy())  # hmm?

                # if not converged:
                #     if i > 50:
                #         # recall that my loss takes negative values
                #         if losses[-20] - losses[-1] < 1e-6:
                #             print("CONVERGED!", i)
                #             converged = True
                #             n_converged = i
                if (i > 50) and (losses[-20] - losses[-1] < 1e-6):
                    break

            return losses#, n_converged

        # instantiate model
        m = Model()
        # Instantiate optimizer
        opt = torch.optim.NAdam(m.parameters(), lr=0.2, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, momentum_decay=0.004)

        # opt = torch.optim.Adam(m.parameters(), lr=0.1)
        # opt = torch.optim.LBFGS(m.parameters(), lr=0.001, line_search_fn="strong_wolfe")
        losses = training_loop(m, opt)
        if self.verbosity>= 3:
            plt.figure()
            plt.plot(np.arange(len(losses)), losses)
            plt.show()
        # plt.figure()
        # plt.plot(np.arange(len(losses)), losses)
        # plt.scatter(n_converged, losses[n_converged])
        # plt.show()

        popt = (m.weights).detach().numpy()

        return popt

    def _fit_with_torch_surface(self, z, initial, nystrom_surfaces, nystrom_object):

        if nystrom_object.kernel == "linear" or nystrom_object.kernel == "poly3":
            return self._fit_with_torch_surface_nonnormalised(z, initial, nystrom_surfaces, nystrom_object)

        import torch
        from torch import nn

        tensorz = torch.tensor(z)
        tensorinitial = torch.tensor(initial)

        tensor_n_surfs = torch.tensor(nystrom_surfaces)
        tensor_wsqinv = torch.tensor(nystrom_object.Wsqinv)

        def torch_kernel(x, Z):

            if nystrom_object.kernel == "rbf":
                gamma = nystrom_object.gamma
                # hmm ok so this is some broadcasting stuff? how to translate to pytorch?
                # omg it seems to work exactly the same way! Nice!
                xx = (Z * Z).sum(-1)  # diagonal of WW^T
                yy = torch.dot(x, x)
                xy = 2 * torch.mv(Z, x)  # note 2 is here!

                return torch.exp(gamma*(-xx + xy - yy))
            elif nystrom_object.kernel == "linearn":
                polyk = (torch.mv(Z, x))
                kd1 = (torch.dot(x, x))  # diagonal elements of dot products
                kd2 = (Z * Z).sum(-1)   # diagonal elements of dot products

                return polyk / torch.sqrt(kd1 * kd2)
            elif nystrom_object.kernel == "poly3n":
                d = 3
                polyk = (torch.mv(Z, x)) ** d
                kd1 = (torch.dot(x, x)) ** d # diagonal elements of dot products
                kd2 = (Z * Z).sum(-1) ** d  # diagonal elements of dot products

                return polyk / torch.sqrt(kd1 * kd2)

        class Model(nn.Module):
            """Custom Pytorch model for gradient optimization.
            """

            def __init__(self):
                super().__init__()
                weights = tensorinitial
                # make weights torch parameters
                self.weights = nn.Parameter(weights)

            def forward(self):
                """Implement function to be optimised.
                """

                weights = self.weights

                K = torch_kernel(weights, tensor_n_surfs)
                val = torch.matmul(K, tensor_wsqinv)

                res = -torch.dot(val, tensorz)
                # print(res)
                return res

        def training_loop(model, optimizer, n=500):
            "Training loop for torch model."
            losses = []

            # from https://discuss.pytorch.org/t/want-to-maximise-a-function-do-i-use-a-torch-nn-loss-or-is-there-a-better-way/76842/2
            # how to use just the value of the function here
            # converged = False
            # n_converged = 0
            for i in range(n):
                value = model()  # value is the value of the objective
                optimizer.zero_grad()
                value.backward() # why in places it's backwards, in other places backward?
                optimizer.step()
                losses.append(value.detach().numpy())  # hmm?

                if (i > 50) and (losses[-20] - losses[-1] < 1e-6):
                    break

            return losses#, n_converged

        # instantiate model
        m = Model()
        # Instantiate optimizer
        if self.max_responseval == 100:
            lr = 0.2
        else:
            lr = 0.02
        opt = torch.optim.NAdam(m.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, momentum_decay=0.004)

        # opt = torch.optim.Adam(m.parameters(), lr=0.1)
        # opt = torch.optim.LBFGS(m.parameters(), lr=0.001, line_search_fn="strong_wolfe")
        losses = training_loop(m, opt)
        if self.verbosity>= 3:
            plt.figure()
            plt.plot(np.arange(len(losses)), losses)
            plt.show()
        # plt.figure()
        # plt.plot(np.arange(len(losses)), losses)
        # plt.scatter(n_converged, losses[n_converged])
        # plt.show()

        popt = (m.weights).detach().numpy()

        return popt

    def _get_braid_e_bounds(self, h1, h2, preds, error=10):

        """
        E1 bound: bounds for hill on drug 1. Should be within the responses of largest concentrations
        E2 bound: bounds for hill on drug 2. As above

        E3 bound: bounds for the largest combo response.
         -> if largest combo response stronger than hills go with that
         -> if hills stronger? lower bound with lowest hill, higher bound from strongest combo

        :param h1:
        :param h2:
        :return:
        """

        if self.max_responseval == 1:
            error /= 100

        if h1[2] < 0:
            h1bounds = (h1[2]*2, h1[2]/2)
        else:
            h1bounds = (h1[2]/2, h1[2]*2)
        if h2[2] < 0:
            h1bounds = (h2[2]*2, h2[2]/2)
        else:
            h2bounds = (h2[2]/2, h2[2]*2)

        return {"E0_bounds": (self.max_responseval-error/2, self.max_responseval+error/2),
                "E1_bounds": (np.maximum(0, h1[1] - error), h1[1] + error),
                "E2_bounds": (np.maximum(0, h2[1] - error), h2[1] + error),
                "E3_bounds": (np.maximum(0, preds[-1]-self.max_responseval*0.2),
                              np.maximum(0, preds[-1]) +self.max_responseval*0.20),
                "h1_bounds": h1bounds,  # todo hmm what values do these usually take? is this ok?
                "h2_bounds": h2bounds,
                "C1_bounds": (h1[3]/2, h1[3]*2),
                "C2_bounds": (h2[3]/2, h2[3]*2)}

    def _fit_braid_to_normalised_grid_prediction(self, cdd, pred):
        cell, drug1, drug2 = cdd

        # get the hills
        h1 = self.hills[(self.hills["drug"] == drug1) & (self.hills["cell"] == cell)]
        h2 = self.hills[(self.hills["drug"] == drug2) & (self.hills["cell"] == cell)]
        h1_np = [self.max_responseval, h1["Emax"].to_numpy()[0], h1["h"].to_numpy()[0], h1["C"].to_numpy()[0]]
        h2_np = [self.max_responseval, h2["Emax"].to_numpy()[0], h2["h"].to_numpy()[0], h2["C"].to_numpy()[0]]

        # what are the concentrations of this prediction?
        c1 = c_from_normalized_c(self.nystrom_y.c1, *h1_np)
        c2 = c_from_normalized_c(self.nystrom_y.c2, *h2_np)

        for error in [10, 20, 30]:  # error is scaled in the get_braid_e_bounds, if needed
            try:
                braid_e_bounds = self._get_braid_e_bounds(h1_np, h2_np, pred, error=error)

                braid = MyBraidWithOptimisers(cell_id=cell, drug1_id=drug1, drug2_id=drug2,
                                              E0_bounds=braid_e_bounds["E0_bounds"],
                                              E1_bounds=braid_e_bounds["E1_bounds"],
                                              E2_bounds=braid_e_bounds["E2_bounds"],
                                              E3_bounds=braid_e_bounds["E3_bounds"],
                                              h1_bounds=braid_e_bounds["h1_bounds"],
                                              h2_bounds=braid_e_bounds["h2_bounds"],
                                              C1_bounds=braid_e_bounds["C1_bounds"],
                                              C2_bounds=braid_e_bounds["C2_bounds"],
                                              kappa_bounds=[-1.5, 10])

                x0_tofit = braid._get_initial_guess(c1, c2, mat_to_vec(pred))
                # note! these parameters are already transformed for fit!
                x0 = braid._transform_params_from_fit(x0_tofit)  # the initial guess in more understandable format

                params_fitted_log = braid.fit_with_scipy_least_squares(np.vstack((c1[None, :], c2[None, :])), mat_to_vec(pred),
                                                                           braid_model_with_log_c_input, x0_tofit, loss="linear")
                params_fitted = braid._transform_params_from_fit(params_fitted_log)
                return params_fitted
            except ValueError:
                pass

        return None

    def predict_with_gd_nadam_and_braid(self, cell_drug_drug_tst, c1_tst, c2_tst):

        # torch and nesterov accelerated adam?

        print("Predicting raw values in the approximated output feature space")
        raw_preds = self._predict_raw(cell_drug_drug_tst)
        print("Predicting raw values done")

        print("raw preds:", raw_preds.shape)
        print("Nyström approx dimension:", self.nystrom_y.Wsqinv.shape[0])

        preds = []
        pred_params = []
        tz = time.process_time()

        failed_fits = 0
        for tt in range(raw_preds.shape[0]):

            cell, drug1, drug2 = cell_drug_drug_tst[tt, :]

            # ------------------------------------------------------------------
            # form the initial guess
            if self.hills is not None:
                # find from dataframe with drug and cell ids
                h1_df = self.hills[((self.hills['drug'] == drug1) & (self.hills['cell'] == cell))]
                h2_df = self.hills[((self.hills['drug'] == drug2) & (self.hills['cell'] == cell))]
                # print(h1_df)
                # print(h2_df)
                h1 = [self.max_responseval, h1_df["Emax"].item(), h1_df["h"].item(), h1_df["C"].item()]
                h2 = [self.max_responseval, h2_df["Emax"].item(), h2_df["h"].item(), h2_df["C"].item()]

                # form the initial guess
                initial_guess_braid = [self.max_responseval, h1[1], h2[1], np.minimum(h1[1], h2[1]), h1[2], h2[2], h1[3], h2[3], 0]

                # then sample it at the concentrations of the normalised grid
                initial_guess_sampled = braid_model_with_raw_c_input([c_from_normalized_c(self.nystrom_y.c1, *h1),
                                                                      c_from_normalized_c(self.nystrom_y.c2, *h2)],
                                                                     *initial_guess_braid)
            else:
                # actually this won't work..... well whatever this else never has happened,
                # I'll leave this here for now, fixing later
                initial_guess_braid = [self.max_responseval, h1[1], h2[1], np.minimum(h1[1], h2[1]), h1[2], h2[2], h1[3], h2[3], 0]
                initial_guess_sampled = np.zeros(len(self.nystrom_y.c1))+self.max_responseval
            # ------------------------------------------------------------------

            z = np.dot(self.nystrom_y.Wsqinv, raw_preds[tt, :])

            result = self._fit_with_torch_surface(z, initial_guess_sampled, self.nystrom_y.nystrom_surfaces,
                                                  self.nystrom_y)

            braid_result = self._fit_braid_to_normalised_grid_prediction([cell, drug1, drug2], result)
            if braid_result is None:
                preds.append(
                    braid_model_with_raw_c_input([c1_tst[tt, :], c2_tst[tt, :]], *initial_guess_braid))
                pred_params.append(initial_guess_braid)
                failed_fits+=1
            else:
                preds.append(
                    braid_model_with_raw_c_input([c1_tst[tt, :], c2_tst[tt, :]], *braid_result))
                pred_params.append(braid_result)

            # if tt%10 == 0:
            #     print("predicted with nadam %d/%d"%(tt, raw_preds.shape[0]))
            #     td = time.process_time()-tz
            #     print("this took ", timedelta(seconds=td))
            #     print("On average per surface: ", timedelta(seconds=td/(tt+1)))

        # print(preds_on_grid.shape)
        print("All predictions with gd done successfully :) ")
        # if self.verbosity > 0:
        #     random_inds = np.random.permutation(preds_on_grid.shape[0])[:6]
        #     plt.figure()
        #     for ii in range(len(random_inds)):
        #         plt.subplot(231+ii)
        #         plt.matshow(vec_to_mat(preds_on_grid[ii, :], (11, 11)), fignum=False)
        #     plt.show()
        print("--> FAILED TO FIT BRAID %d/%d times!!"%(failed_fits, cell_drug_drug_tst.shape[0]))

        return np.array(preds), np.array(pred_params)

    def predict_with_neutral_braid(self, cell_drug_drug_tst, c1_tst, c2_tst):

        """
        No learning here! This is the baseline "prediction", i.e. just assigning neutral surface based on the Hill
        equations
        :param cell_drug_drug_tst:
        :param c1_tst:
        :param c2_tst:
        :return:
        """

        preds = []
        pred_params = []
        tz = time.process_time()
        for tt in range(cell_drug_drug_tst.shape[0]):

            cell, drug1, drug2 = cell_drug_drug_tst[tt, :]

            # ------------------------------------------------------------------
            # form the initial guess
            # find from dataframe with drug and cell ids
            h1_df = self.hills[((self.hills['drug'] == drug1) & (self.hills['cell'] == cell))]
            h2_df = self.hills[((self.hills['drug'] == drug2) & (self.hills['cell'] == cell))]
            # print(h1_df)
            # print(h2_df)
            h1 = [self.max_responseval, h1_df["Emax"].item(), h1_df["h"].item(), h1_df["C"].item()]
            h2 = [self.max_responseval, h2_df["Emax"].item(), h2_df["h"].item(), h2_df["C"].item()]

            # form the initial guess
            initial_guess_braid = [self.max_responseval, h1[1], h2[1], np.minimum(h1[1], h2[1]), h1[2], h2[2], h1[3], h2[3], 0]

            # then sample it at the concentrations of the normalised grid
            initial_guess_sampled = braid_model_with_raw_c_input([c1_tst[tt, :], c2_tst[tt, :]],
                                                                 *initial_guess_braid)

            # ------------------------------------------------------------------

            preds.append(initial_guess_sampled)
            pred_params.append(initial_guess_braid)

        return np.array(preds), np.array(pred_params)

    def _predict_raw(self, cell_drug_drug_tst, difference=False):

        print("Raw, prediction, also containing training")

        mat1_inds, mat2_inds = self._get_inds_from_cdd(cell_drug_drug_tst)

        tr_preds = []
        preds = []
        t00 = time.process_time()
        for ii in range(self.Phiy_tr.shape[1]):
            t0 = time.process_time()
            x1 = np.kron(self.Kc, self.Kd)
            learner = CGKronRLS(K1=x1, K2=self.Kd,
                                Y=self.Phiy_tr[:, ii], label_row_inds=self.mat1_inds, label_col_inds=self.mat2_inds,
                                regparam=0, maxiter=self.maxiter)
            # print(learner.predict(x1, self.drugfeats, mat1_inds, mat2_inds))
            if self.verbosity > 2:
                tr_preds.append(learner.predict(x1, self.Kd, self.mat1_inds, self.mat2_inds))
            preds.append(learner.predict(x1, self.Kd, mat1_inds, mat2_inds))
            t1 = time.process_time()
            if self.verbosity > 0:
                print("training and predicting one iteration took: ", timedelta(seconds=t1 - t0))
        print("training and predicting raw took: ", timedelta(seconds=time.process_time() - t00))
        preds = np.array(preds).T
        # self.learner.predict(X1, X2, inds_X1pred, inds_X2pred, pko)
        # pko; pairwise kernel operator, if None (default) then one is created

        # how do tr_preds and phiy_tr compare?
        if self.verbosity > 2:

            tr_preds = np.array(tr_preds).T

            variantname = "IOKR"
            if difference:
                variantname+="diff"

            np.save("debugging/tr_preds_newmono" + variantname + "newBraids.npy", tr_preds)
            np.save("debugging/tr_cdd_newmono" + variantname + "newBraids.npy", self.cdd_tr)
            np.save("debugging/tr_phiy_newmono" + variantname + "newBraids.npy", self.Phiy_tr)

            print("Pearson correlation on training:", np.corrcoef(self.Phiy_tr.ravel(), tr_preds.ravel())[0, 1])

            exit("only wanted to know this :)")

            plt.matshow(np.dot(self.Phiy_tr, self.Phiy_tr.T))
            plt.title("phiy tr - phiy tr")
            plt.matshow(np.dot(tr_preds, self.Phiy_tr.T))
            plt.title("tr preds - phiy tr")

            print()
            print("phiy tr:")
            print(self.Phiy_tr[:5, :])
            print("phiy tr preds:")
            print(tr_preds[:5, :])

            braids_tst = np.load("debugging/braids_tst.npy")
            phiy_tst = self.nystrom_y.get_approximated_features_braid(braids_tst)
            plt.matshow(np.dot(phiy_tst, phiy_tst.T))
            plt.title("phiy tst - phiy tts")
            plt.matshow(np.dot(preds, phiy_tst.T))
            plt.title("tst preds - phiy tst")

            print()
            print("phiy tst:")
            print(phiy_tst[:5, :])
            print("phiy tst preds:")
            print(preds[:5, :])

            plt.figure()
            for ii in range(9):
                plt.subplot(331+ii)
                plt.scatter(preds[:, ii], phiy_tst[:, ii])
                plt.xlabel("preds")
                plt.ylabel("true")
                minval = np.maximum(np.min(preds[:, ii]), np.min(phiy_tst[:, ii]))
                maxval = np.minimum(np.max(preds[:, ii]), np.max(phiy_tst[:, ii]))
                plt.plot([minval, maxval], [minval, maxval], linestyle="--", c="k")
                plt.title("TST %5.3f"%np.corrcoef(phiy_tst[:, ii], preds[:, ii])[0, 1])

            plt.figure()
            for ii in range(9):
                plt.subplot(331+ii)
                plt.scatter(preds[:, ii], phiy_tst[:, ii])
                plt.xlabel("preds")
                plt.ylabel("true")
                minval = np.maximum(np.min(tr_preds[:, ii]), np.min(self.Phiy_tr[:, ii]))
                maxval = np.minimum(np.max(tr_preds[:, ii]), np.max(self.Phiy_tr[:, ii]))
                plt.plot([minval, maxval], [minval, maxval], linestyle="--", c="k")
                plt.title("TR %5.3f"%np.corrcoef(self.Phiy_tr[:, ii], tr_preds[:, ii])[0, 1])


            plt.show()

        return preds

    def fix_Ytr(self, Ytr, matsize=4):
        # ¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤
        # rows that are in unique rows here and in inds_in_tr
        [rows, cols] = np.where(np.isnan(Ytr))
        # print("rows and cols with nans before:\n", rows, cols)
        rows_to_fix = rows
        # print("need to fix", len(rows_to_fix), "rows! :(")

        bad_braids3 = []

        if (Ytr.shape[1] == 14) or (Ytr.shape[1] != matsize**2):  # if this is not a regular grid I cannot do this
            print("This error is due to non-square matrix to be fixed in fix_Ytr function")
            print("The specific size 14 corresponds to Jaaks data, 7*2")
            print("For me this has not been a problem -"
                  " with Jaaks data this function has not been ever called")
            raise NotImplementedError

        for rr, rowindx in enumerate(np.unique(rows_to_fix)):

            vec = Ytr[rowindx, :]
            cols_here = np.where(np.isnan(vec))[0]

            assert np.any(np.isnan(vec))

            # three possibilities: (1) all are nan, (2) only some are, consequtive, or (3) some are but they jump
            if len(cols_here) == matsize ** 2:
                # print("*")
                bad_braids3.append(rowindx)
            elif (cols_here[-1] - cols_here[0]) == (len(cols_here) - 1):
                # print("-")
                # consecutive indexing; column
                """  example
                0 3 6
                1 4 7
                2 5 8
                """
                # take tha last available column and repeat
                last_col = vec[cols_here[0] - matsize:cols_here[0]]
                indx_for_col_to_fill = cols_here[0]
                while indx_for_col_to_fill < matsize ** 2:
                    vec[indx_for_col_to_fill:indx_for_col_to_fill + matsize] = last_col
                    indx_for_col_to_fill += matsize
                Ytr[rowindx, :] = vec
            else:
                # print("¤")
                # jumping; rows
                update_rowindx = np.arange(matsize) * matsize
                prev_rowindx = np.copy(update_rowindx)
                done = False
                while np.all(prev_rowindx < matsize ** 2):
                    new_rowindx = prev_rowindx + 1
                    if new_rowindx[0] in cols_here:
                        # nan, assign as previous row
                        vec[new_rowindx] = vec[update_rowindx]
                    else:
                        # update row that should be used to fill in other rows
                        update_rowindx = new_rowindx
                    prev_rowindx = new_rowindx
                    if matsize ** 2 - 1 in new_rowindx:
                        break
                Ytr[rowindx, :] = vec

                # it can be both! so test still for consecutive once more here
                if np.any(np.isnan(vec)):
                    cols_here = np.where(np.isnan(vec))[0]
                    if (cols_here[-1] - cols_here[0]) == (len(cols_here) - 1):
                        # print("-")
                        # consecutive indexing; column
                        """  example
                        0 3 6
                        1 4 7
                        2 5 8
                        """
                        # take tha last available column and repeat
                        last_col = vec[cols_here[0] - matsize:cols_here[0]]
                        indx_for_col_to_fill = cols_here[0]
                        while indx_for_col_to_fill < matsize ** 2:
                            vec[
                            indx_for_col_to_fill:indx_for_col_to_fill + matsize] = last_col
                            indx_for_col_to_fill += matsize
                        Ytr[rowindx, :] = vec
            if np.any(np.isnan(vec)):
                bad_braids3.append(rowindx)

        return Ytr


class ScalablecomboKRdiff(ScalableComboKR):

    """
    More memory efficient version

    It seems that the memory issues are directly from number of output tasks.
    So, cannot save the learners, but have to train and test and discard.

    Training happens at test time.
    """

    def predict_with_candidates(self, cell_drug_drug_tst, c1_tst, c2_tst):

        print("Predicting raw values in the approximated output feature space")
        raw_preds = self._predict_raw(cell_drug_drug_tst, difference=True)
        print("Predicting raw values done")

        preds = []
        pred_params = []
        # predicting single way
        for ii in range(cell_drug_drug_tst.shape[0]):
            # print("start with indx", ii)

            cell, drug1, drug2 = cell_drug_drug_tst[ii, :]

            # --------------------------------------------------------------------------------------------
            # build the candidates
            h1 = self.hills[(self.hills["drug"] == drug1) & (self.hills["cell"] == cell)]
            h2 = self.hills[(self.hills["drug"] == drug2) & (self.hills["cell"] == cell)]
            h1_np = [self.max_responseval, h1["Emax"].to_numpy()[0], h1["h"].to_numpy()[0], h1["C"].to_numpy()[0]]
            h2_np = [self.max_responseval, h2["Emax"].to_numpy()[0], h2["h"].to_numpy()[0], h2["C"].to_numpy()[0]]
            braid_params = [self.max_responseval, h1["Emax"].to_numpy()[0], h2["Emax"].to_numpy()[0], 0,
                            h1["h"].to_numpy()[0], h2["h"].to_numpy()[0],
                            h1["C"].to_numpy()[0], h2["C"].to_numpy()[0], 0]

            kappas = [0.01, 0.1, -0.1, 0.5, -0.5, 1, -1, 2, -1.5, 10]
            start = np.maximum(np.minimum(h1["Emax"].to_numpy()[0], h2["Emax"].to_numpy()[0]) - self.max_responseval*0.50, 0)
            end = np.minimum(self.max_responseval*1.20, np.maximum(h1["Emax"].to_numpy()[0], h2["Emax"].to_numpy()[0]) + self.max_responseval*0.10)
            Emaxs = np.arange(start, end, (end - start) / 4)

            candidate_evals = []
            # candidate_full_evals = []
            all_param_combinations = []
            for kappa in kappas:
                braid_params[8] = kappa
                for Emax in Emaxs:
                    braid_params[3] = Emax
                    zero_br = np.copy(braid_params)
                    zero_br[-1]  = 0
                    surf = braid_model_with_raw_c_input([c_from_normalized_c(self.nystrom_y.c1, *h1_np),
                                                                         c_from_normalized_c(self.nystrom_y.c2,
                                                                                             *h2_np)],
                                                                        *braid_params)
                    zerosurf = braid_model_with_raw_c_input([c_from_normalized_c(self.nystrom_y.c1, *h1_np),
                                                                         c_from_normalized_c(self.nystrom_y.c2,
                                                                                             *h2_np)],
                                                                        *zero_br)
                    candidate_evals.append(surf-zerosurf)
                    # candidate_full_evals.append(surf)
                    all_param_combinations.append(np.copy(braid_params))
            candidate_evals = np.array(candidate_evals)
            all_param_combinations = np.array(all_param_combinations)
            # if there happens to be a nan in the candidate evaluation, do not include it!
            if np.any(np.isnan(candidate_evals)):
                nonnanrows = np.where(~np.isnan(np.sum(candidate_evals, axis=1)))[0]
                if len(nonnanrows) > 0:
                    candidate_evals = candidate_evals[nonnanrows, :]
                    all_param_combinations = all_param_combinations[nonnanrows, :]
                else:
                    print("all candidate evals contained nan!", cell, drug1, drug2)
                    candidate_evals[np.isnan(candidate_evals)] = self.max_responseval

            phic = self.nystrom_y.get_approximated_features_grid(candidate_evals)

            scores = np.dot(phic, raw_preds[ii, :])

            best_indx = np.argmax(scores)

            # evaluate with the braid that was best, this gives final prediction, not difference
            preds.append(
                braid_model_with_raw_c_input([c1_tst[ii, :], c2_tst[ii, :]], *all_param_combinations[best_indx, :]))
            pred_params.append(all_param_combinations[best_indx, :])

        return np.array(preds), pred_params

    def predict_with_gd_nadam_and_braid(self, cell_drug_drug_tst, c1_tst, c2_tst):

        # torch and nesterov accelerated adam?

        print("Predicting raw values in the approximated output feature space")
        raw_preds = self._predict_raw(cell_drug_drug_tst)
        print("Predicting raw values done")

        print("raw preds:", raw_preds.shape)
        print("Nyström approx dimension:", self.nystrom_y.Wsqinv.shape[0])

        preds = []
        pred_params = []
        tz = time.process_time()

        failed_fits = 0
        for tt in range(raw_preds.shape[0]):

            cell, drug1, drug2 = cell_drug_drug_tst[tt, :]

            # ------------------------------------------------------------------
            # form the initial guess
            if self.hills is not None:
                # find from dataframe with drug and cell ids
                h1_df = self.hills[((self.hills['drug'] == drug1) & (self.hills['cell'] == cell))]
                h2_df = self.hills[((self.hills['drug'] == drug2) & (self.hills['cell'] == cell))]
                # print(h1_df)
                # print(h2_df)
                h1 = [self.max_responseval, h1_df["Emax"].item(), h1_df["h"].item(), h1_df["C"].item()]
                h2 = [self.max_responseval, h2_df["Emax"].item(), h2_df["h"].item(), h2_df["C"].item()]

                # form the initial guess
                initial_guess_braid = [self.max_responseval, h1[1], h2[1], np.minimum(h1[1], h2[1]), h1[2], h2[2], h1[3], h2[3], 0]

                # then sample it at the concentrations of the normalised grid
                initial_guess_sampled = braid_model_with_raw_c_input([c_from_normalized_c(self.nystrom_y.c1, *h1),
                                                                      c_from_normalized_c(self.nystrom_y.c2, *h2)],
                                                                     *initial_guess_braid)
            else:
                initial_guess_sampled = np.zeros(len(self.nystrom_y.c1))+self.max_responseval
                initial_guess_braid = [self.max_responseval, h1[1], h2[1], np.minimum(h1[1], h2[1]), h1[2], h2[2], h1[3], h2[3], 0]

            # ------------------------------------------------------------------

            z = np.dot(self.nystrom_y.Wsqinv, raw_preds[tt, :])

            # intial guess zeros
            result = self._fit_with_torch_surface(z, np.zeros(len(self.nystrom_y.c1)), self.nystrom_y.nystrom_surfaces,
                                                  self.nystrom_y)

            # fit BRAID to 0-braid + deviance (result)
            braid_result = self._fit_braid_to_normalised_grid_prediction([cell, drug1, drug2], initial_guess_sampled + result)
            if braid_result is None:
                try:
                    preds.append(
                        braid_model_with_raw_c_input([c1_tst[tt, :], c2_tst[tt, :]], *initial_guess_braid))
                except:
                    preds.append(np.zeros(c1_tst.shape[1])+self.max_responseval)
                pred_params.append(initial_guess_braid)
                failed_fits+=1
            else:
                preds.append(
                    braid_model_with_raw_c_input([c1_tst[tt, :], c2_tst[tt, :]], *braid_result))
                pred_params.append(braid_result)

            # if tt%10 == 0:
            #     print("predicted with nadam %d/%d"%(tt, raw_preds.shape[0]))
            #     td = time.process_time()-tz
            #     print("this took ", timedelta(seconds=td))
            #     print("On average per surface: ", timedelta(seconds=td/(tt+1)))

        # print(preds_on_grid.shape)
        print("All predictions with gd done successfully :) ")
        # if self.verbosity > 0:
        #     random_inds = np.random.permutation(preds_on_grid.shape[0])[:6]
        #     plt.figure()
        #     for ii in range(len(random_inds)):
        #         plt.subplot(231+ii)
        #         plt.matshow(vec_to_mat(preds_on_grid[ii, :], (11, 11)), fignum=False)
        #     plt.show()
        print("--> FAILED TO FIT BRAID %d/%d times!!"%(failed_fits, cell_drug_drug_tst.shape[0]))

        return np.array(preds), np.array(pred_params)

