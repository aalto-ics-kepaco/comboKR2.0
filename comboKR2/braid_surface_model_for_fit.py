import numpy as np
from synergy.combination import BRAID
from synergy.single import Hill
import synergy.utils

from scipy.optimize import curve_fit, least_squares


def braid_model_with_raw_c_input(d, E0, E1, E2, E3, h1, h2, C1, C2, kappa, delta=1, print_params=False):
    """
From the braidrm R package (https://rdrr.io/cran/braidrm/man/evalBRAIDrsm.html)
The parameters of this equation must satisfy h1>0, h2>0, delta>0, kappa>-2, sign(E3-E0)=sign(E1-E0)=sign(E2-E0), |E3-E0|>=|E1-E0|, and |E3-E0|>=|E2-E0|.
    """

    d1 = d[0]
    d2 = d[1]

    if print_params:
        print("%7.2e - %7.2e - %7.2e - %7.2e - %7.2e - %7.2e - %7.2e - %7.2e - %7.2e - %7.2e"
              % (E0, E1, E2, E3, h1, h2, C1, C2, kappa, delta))

    delta_Es = [E1 - E0, E2 - E0, E3 - E0]
    # delta_Es = [E1 - E0, E2 - E0]  # , E3 - E0]   <- to make combination not dependent on combination info
    delta_Es = -np.abs(delta_Es)  # band-aid..
    max_delta_E_index = np.argmax(np.abs(delta_Es))
    max_delta_E = delta_Es[max_delta_E_index]

    deltaA_div_maxdelta = delta_Es[0] / max_delta_E
    deltaB_div_maxdelta = delta_Es[1] / max_delta_E

    h = np.sqrt(h1 * h2)
    power = 1 / (delta * h)

    pow1 = np.power(d1 / C1, h1)
    D1 = deltaA_div_maxdelta * pow1 / (
            1 + (1 - deltaA_div_maxdelta) * pow1)

    pow2 = np.power(d2 / C2, h2)
    D2 = deltaB_div_maxdelta * pow2 / (
            1 + (1 - deltaB_div_maxdelta) * pow2)

    pow11 = np.power(D1, power)
    pow22 = np.power(D2, power)
    D = pow11 + pow22 + kappa * np.sqrt(pow11 * pow22)

    # D = np.real(D)

    # print(h, h1, h2, h1*h2, np.sqrt(h1*h2))
    # print(power, delta)
    # print("D1:", D1, "\n ", d1,  C1, "\n ", d1/C1, "\n ", np.power(d1 / C1, h1))
    # print("D2:", D2, "\n ", d2,  C2, "\n ", d2/C2, "\n ", np.power(d2 / C2, h2))
    # print(pow11*pow22, "\n  ;", np.sqrt(pow11*pow22))
    # print(D, "\n  ", pow22)
    # # print("\n  ", np.real(pow22))
    # # print(max_delta_E / (1 + np.power(D, -delta * h)), "\n  ", np.power(D, -delta * h))

    Dpow = np.power(D, -delta * h)
    # # Dpow = np.sign(D)*np.power(np.abs(D), -delta*h)
    # print(Dpow, "\n", E0 + max_delta_E / (1 + Dpow))
    # print(-delta*h)
    res = E0 + max_delta_E / (1 + Dpow)
    # TypeError: 'numpy.float64' object does not support item assignment
    try:
        res[D == 0] = 100
    except TypeError:
        if D == 0:
            res = 100
    return res


def braid_model_with_log_c_input(d, E0, E1, E2, E3, h1_log, h2_log, C1_log, C2_log, kappa, delta=1, print_params=False):
    """
From the braidrm R package (https://rdrr.io/cran/braidrm/man/evalBRAIDrsm.html)
The parameters of this equation must satisfy h1>0, h2>0, delta>0, kappa>-2, sign(E3-E0)=sign(E1-E0)=sign(E2-E0), |E3-E0|>=|E1-E0|, and |E3-E0|>=|E2-E0|.
    """

    h1 = np.exp(h1_log)
    h2 = np.exp(h2_log)
    C1 = np.exp(C1_log)
    C2 = np.exp(C2_log)

    d1 = d[0]
    d2 = d[1]

    if print_params:
        print("%7.2e - %7.2e - %7.2e - %7.2e - %7.2e - %7.2e - %7.2e - %7.2e - %7.2e - %7.2e"
              % (E0, E1, E2, E3, h1, h2, C1, C2, kappa, delta))

    delta_Es = [E1 - E0, E2 - E0, E3 - E0]
    # delta_Es = [E1 - E0, E2 - E0]  # , E3 - E0]   <- to make combination not dependent on combination info
    delta_Es = -np.abs(delta_Es)  # band-aid..
    max_delta_E_index = np.argmax(np.abs(delta_Es))
    max_delta_E = delta_Es[max_delta_E_index]

    deltaA_div_maxdelta = delta_Es[0] / max_delta_E
    deltaB_div_maxdelta = delta_Es[1] / max_delta_E

    h = np.sqrt(h1 * h2)
    power = 1 / (delta * h)

    pow1 = np.power(d1 / C1, h1)
    D1 = deltaA_div_maxdelta * pow1 / (
            1 + (1 - deltaA_div_maxdelta) * pow1)

    pow2 = np.power(d2 / C2, h2)
    D2 = deltaB_div_maxdelta * pow2 / (
            1 + (1 - deltaB_div_maxdelta) * pow2)

    pow11 = np.power(D1, power)
    pow22 = np.power(D2, power)
    D = pow11 + pow22 + kappa * np.sqrt(pow11 * pow22)

    # D = np.real(D)

    # print(h, h1, h2, h1*h2, np.sqrt(h1*h2))
    # print(power, delta)
    # print("D1:", D1, "\n ", d1,  C1, "\n ", d1/C1, "\n ", np.power(d1 / C1, h1))
    # print("D2:", D2, "\n ", d2,  C2, "\n ", d2/C2, "\n ", np.power(d2 / C2, h2))
    # print(pow11*pow22, "\n  ;", np.sqrt(pow11*pow22))
    # print(D, "\n  ", pow22)
    # # print("\n  ", np.real(pow22))
    # # print(max_delta_E / (1 + np.power(D, -delta * h)), "\n  ", np.power(D, -delta * h))

    Dpow = np.power(D, -delta * h)
    # # Dpow = np.sign(D)*np.power(np.abs(D), -delta*h)
    # print(Dpow, "\n", E0 + max_delta_E / (1 + Dpow))
    # print(-delta*h)
    res = E0 + max_delta_E / (1 + Dpow)
    # TypeError: 'numpy.float64' object does not support item assignment
    try:
        res[D == 0] = 100
    except TypeError:
        if D == 0:
            res = 100
    return res


class MyBRAID(BRAID):

    # note! default values here assume [0, 100]; always overwrite with appropriate if possible
    # (also note: kappa values [-2, 100] not related to response values range

    def __init__(self, cell_id=None, drug1_id=None, drug2_id=None, h1_bounds=(1e-12,np.inf), h2_bounds=(1e-12,np.inf),  \
            E0_bounds=(90,110), E1_bounds=(0,120), \
            E2_bounds=(0,120), E3_bounds=(0,120), \
            C1_bounds=(1e-16,1), C2_bounds=(1e-16,1), kappa_bounds=(-2, 100),
                 delta_bounds=(0, np.inf), E0=None, E1=None, E2=None, E3=None,
                 h1=None, h2=None, C1=None, C2=None, kappa=None, delta=None, variant="kappa"):

        super().__init__(h1_bounds=h1_bounds, h2_bounds=h2_bounds,
                         E0_bounds=E0_bounds, E1_bounds=E1_bounds, E2_bounds=E2_bounds, E3_bounds=E3_bounds,
                         C1_bounds=C1_bounds, C2_bounds=C2_bounds, kappa_bounds=kappa_bounds, delta_bounds=delta_bounds,
                         E0=E0, E1=E1, E2=E2, E3=E3, h1=h1, h2=h2, C1=C1, C2=C2, kappa=kappa, delta=delta, variant=variant)

        self.cell_id = cell_id
        self.drug1_id = drug1_id
        self.drug2_id = drug2_id

    def _get_initial_guess(self, d1, d2, E, drug1_model=None, drug2_model=None, p0=None):

        # If there is no intial guess, use single-drug models to come up with intitial guess
        if p0 is None:
            # Sanitize single-drug models
            default_class, expected_superclass = self._get_single_drug_classes()

            drug1_model = synergy.utils.sanitize_single_drug_model(drug1_model, default_class,
                                                           expected_superclass=expected_superclass,
                                                           E0_bounds=self.E0_bounds, Emax_bounds=self.E1_bounds,
                                                           h_bounds=self.h1_bounds, C_bounds=self.C1_bounds)

            drug2_model = synergy.utils.sanitize_single_drug_model(drug2_model, default_class,
                                                           expected_superclass=expected_superclass,
                                                           E0_bounds=self.E0_bounds, Emax_bounds=self.E2_bounds,
                                                           h_bounds=self.h2_bounds, C_bounds=self.C2_bounds)

            # Fit the single drug models if they were not pre-fit by the user
            if not drug1_model.is_fit():
                mask = np.where(d2 == min(d2))

                concentrations = d1[mask]
                responses = E[mask]

                drug1_model.fit(concentrations, responses)
            if not drug2_model.is_fit():
                mask = np.where(d1 == min(d1))

                concentrations = d2[mask]
                responses = E[mask]
                drug2_model.fit(concentrations, responses)

            # Get initial guesses of E0, E1, E2, h1, h2, C1, and C2 from single-drug fits
            E0_1, E1, h1, C1 = drug1_model.get_parameters()
            E0_2, E2, h2, C2 = drug2_model.get_parameters()
            E0 = (E0_1 + E0_2) / 2

            E3 = E[(d1 == max(d1)) & (d2 == max(d2))]
            if len(E3) > 0:
                E3 = np.mean(E3)
            else:
                E3 = np.min(E)

            if self.variant == "kappa":
                if self.kappa_bounds[1] == 0:
                    p0 = [E0, E1, E2, E3, h1, h2, C1, C2, -0.01]
                else:
                    p0 = [E0, E1, E2, E3, h1, h2, C1, C2, 0.01]
            elif self.variant == "delta":
                p0 = [E0, E1, E2, E3, h1, h2, C1, C2, 1]
            elif self.variant == "both":
                p0 = [E0, E1, E2, E3, h1, h2, C1, C2, 0, 1]

        p0 = list(self._transform_params_to_fit(p0))
        synergy.utils.sanitize_initial_guess(p0, self.bounds)
        return p0

    def set_params(self, E0, E1, E2, E3, h1, h2, C1, C2, kappa, delta=1):
        self._set_params(E0, E1, E2, E3, h1, h2, C1, C2, kappa, delta)

    def get_params(self):
        if self.variant == "kappa":
            return [self.E0, self.E1, self.E2, self.E3, self.h1, self.h2, self.C1, self.C2, self.kappa]
        elif self.variant == "delta":
            return [self.E0, self.E1, self.E2, self.E3, self.h1, self.h2, self.C1, self.C2, self.delta]
        elif self.variant == "both":
            return [self.E0, self.E1, self.E2, self.E3, self.h1, self.h2, self.C1, self.C2, self.kappa, self.delta]

    def _model(self, d1, d2, E0, E1, E2, E3, h1, h2, C1, C2, kappa, delta=1, print_params=False):
        """
From the braidrm R package (https://rdrr.io/cran/braidrm/man/evalBRAIDrsm.html)
The parameters of this equation must satisfy h1>0, h2>0, delta>0, kappa>-2, sign(E3-E0)=sign(E1-E0)=sign(E2-E0), |E3-E0|>=|E1-E0|, and |E3-E0|>=|E2-E0|.
        """

        if print_params:
            print("or here, in _model?")
            print("%7.2e - %7.2e - %7.2e - %7.2e - %7.2e - %7.2e - %7.2e - %7.2e - %7.2e - %7.2e"
                  % (E0, E1, E2, E3, h1, h2, C1, C2, kappa, delta))

        delta_Es = [E1 - E0, E2 - E0, E3 - E0]
        # delta_Es = [E1 - E0, E2 - E0]  # , E3 - E0]   <- to make combination not dependent on combination info
        delta_Es = -np.abs(delta_Es)  # band-aid..
        max_delta_E_index = np.argmax(np.abs(delta_Es))
        max_delta_E = delta_Es[max_delta_E_index]

        deltaA_div_maxdelta = delta_Es[0] / max_delta_E
        deltaB_div_maxdelta = delta_Es[1] / max_delta_E

        h = np.sqrt(h1 * h2)
        power = 1 / (delta * h)

        pow1 = np.power(d1 / C1, h1)
        D1 = deltaA_div_maxdelta * pow1 / (
                1 + (1 - deltaA_div_maxdelta) * pow1)

        pow2 = np.power(d2 / C2, h2)
        D2 = deltaB_div_maxdelta * pow2 / (
                1 + (1 - deltaB_div_maxdelta) * pow2)

        pow11 = np.power(D1, power)
        pow22 = np.power(D2, power)
        D = pow11 + pow22 + kappa * np.sqrt(pow11 * pow22)

        # D = np.real(D)

        # print(h, h1, h2, h1*h2, np.sqrt(h1*h2))
        # print(power, delta)
        # print("D1:", D1, "\n ", d1,  C1, "\n ", d1/C1, "\n ", np.power(d1 / C1, h1))
        # print("D2:", D2, "\n ", d2,  C2, "\n ", d2/C2, "\n ", np.power(d2 / C2, h2))
        # print(pow11*pow22, "\n  ;", np.sqrt(pow11*pow22))
        # print(D, "\n  ", pow22)
        # # print("\n  ", np.real(pow22))
        # # print(max_delta_E / (1 + np.power(D, -delta * h)), "\n  ", np.power(D, -delta * h))

        Dpow = np.power(D, -delta * h)
        # # Dpow = np.sign(D)*np.power(np.abs(D), -delta*h)
        # print(Dpow, "\n", E0 + max_delta_E / (1 + Dpow))
        # print(-delta*h)
        res = E0 + max_delta_E / (1 + Dpow)
        res[D == 0] = 100
        return res

    @staticmethod
    def model(d1, d2, E0, E1, E2, E3, h1, h2, C1, C2, kappa, delta=1, print_params=False):
        """
From the braidrm R package (https://rdrr.io/cran/braidrm/man/evalBRAIDrsm.html)
The parameters of this equation must satisfy h1>0, h2>0, delta>0, kappa>-2, sign(E3-E0)=sign(E1-E0)=sign(E2-E0), |E3-E0|>=|E1-E0|, and |E3-E0|>=|E2-E0|.
        """

        if print_params:
            print("%7.2e - %7.2e - %7.2e - %7.2e - %7.2e - %7.2e - %7.2e - %7.2e - %7.2e - %7.2e"
                  % (E0, E1, E2, E3, h1, h2, C1, C2, kappa, delta))

        delta_Es = [E1 - E0, E2 - E0, E3 - E0]
        # delta_Es = [E1 - E0, E2 - E0]  # , E3 - E0]   <- to make combination not dependent on combination info
        delta_Es = -np.abs(delta_Es)  # band-aid..
        max_delta_E_index = np.argmax(np.abs(delta_Es))
        max_delta_E = delta_Es[max_delta_E_index]

        deltaA_div_maxdelta = delta_Es[0] / max_delta_E
        deltaB_div_maxdelta = delta_Es[1] / max_delta_E

        h = np.sqrt(h1 * h2)
        power = 1 / (delta * h)

        pow1 = np.power(d1 / C1, h1)
        D1 = deltaA_div_maxdelta * pow1 / (
                1 + (1 - deltaA_div_maxdelta) * pow1)

        pow2 = np.power(d2 / C2, h2)
        D2 = deltaB_div_maxdelta * pow2 / (
                1 + (1 - deltaB_div_maxdelta) * pow2)

        pow11 = np.power(D1, power)
        pow22 = np.power(D2, power)
        D = pow11 + pow22 + kappa * np.sqrt(pow11 * pow22)

        # D = np.real(D)

        # print(h, h1, h2, h1*h2, np.sqrt(h1*h2))
        # print(power, delta)
        # print("D1:", D1, "\n ", d1,  C1, "\n ", d1/C1, "\n ", np.power(d1 / C1, h1))
        # print("D2:", D2, "\n ", d2,  C2, "\n ", d2/C2, "\n ", np.power(d2 / C2, h2))
        # print(pow11*pow22, "\n  ;", np.sqrt(pow11*pow22))
        # print(D, "\n  ", pow22)
        # # print("\n  ", np.real(pow22))
        # # print(max_delta_E / (1 + np.power(D, -delta * h)), "\n  ", np.power(D, -delta * h))

        Dpow = np.power(D, -delta * h)
        # # Dpow = np.sign(D)*np.power(np.abs(D), -delta*h)
        # print(Dpow, "\n", E0 + max_delta_E / (1 + Dpow))
        # print(-delta*h)
        res = E0 + max_delta_E / (1 + Dpow)

        # TypeError: 'numpy.float64' object does not support item assignment
        try:
            res[D == 0] = 100
        except TypeError:
            if D == 0:
                res = 100
        return res

    def _model_nosynergy(self, d1, d2, E0, E1, E2, E3, h1, h2, C1, C2, kappa, delta, print_params=False):
        """
From the braidrm R package (https://rdrr.io/cran/braidrm/man/evalBRAIDrsm.html)
The parameters of this equation must satisfy h1>0, h2>0, delta>0, kappa>-2, sign(E3-E0)=sign(E1-E0)=sign(E2-E0), |E3-E0|>=|E1-E0|, and |E3-E0|>=|E2-E0|.
        """

        # does NOT depend on kappa nor E3

        if print_params:
            print("%7.2e - %7.2e - %7.2e - %7.2e - %7.2e - %7.2e - %7.2e - %7.2e - %7.2e - %7.2e"
                  % (E0, E1, E2, E3, h1, h2, C1, C2, kappa, delta))

        delta_Es = [E1 - E0, E2 - E0, E3 - E0]
        # delta_Es = [E1 - E0, E2 - E0]  # , E3 - E0]   <- to make combination not dependent on combination info
        delta_Es = -np.abs(delta_Es)  # band-aid..
        max_delta_E_index = np.argmax(np.abs(delta_Es))
        max_delta_E = delta_Es[max_delta_E_index]

        deltaA_div_maxdelta = delta_Es[0] / max_delta_E
        deltaB_div_maxdelta = delta_Es[1] / max_delta_E

        h = np.sqrt(h1 * h2)
        power = 1 / (delta * h)

        pow1 = np.power(d1 / C1, h1)
        D1 = deltaA_div_maxdelta * pow1 / (
                1 + (1 - deltaA_div_maxdelta) * pow1)

        pow2 = np.power(d2 / C2, h2)
        D2 = deltaB_div_maxdelta * pow2 / (
                1 + (1 - deltaB_div_maxdelta) * pow2)

        pow11 = np.power(D1, power)
        pow22 = np.power(D2, power)
        D = pow11 + pow22  # + kappa * np.sqrt(pow11 * pow22)

        # D = np.real(D)

        # print(h, h1, h2, h1*h2, np.sqrt(h1*h2))
        # print(power, delta)
        # print("D1:", D1, "\n ", d1,  C1, "\n ", d1/C1, "\n ", np.power(d1 / C1, h1))
        # print("D2:", D2, "\n ", d2,  C2, "\n ", d2/C2, "\n ", np.power(d2 / C2, h2))
        # print(pow11*pow22, "\n  ;", np.sqrt(pow11*pow22))
        # print(D, "\n  ", pow22)
        # # print("\n  ", np.real(pow22))
        # # print(max_delta_E / (1 + np.power(D, -delta * h)), "\n  ", np.power(D, -delta * h))

        Dpow = np.power(D, -delta * h)
        # # Dpow = np.sign(D)*np.power(np.abs(D), -delta*h)
        # print(Dpow, "\n", E0 + max_delta_E / (1 + Dpow))
        # print(-delta*h)
        res = E0 + max_delta_E / (1 + Dpow)
        res[D == 0] = 100
        return res

    def _model_onlysynergy(self, d1, d2, E0, E1, E2, E3, h1, h2, C1, C2, kappa, delta, print_params=False):
        """
From the braidrm R package (https://rdrr.io/cran/braidrm/man/evalBRAIDrsm.html)
The parameters of this equation must satisfy h1>0, h2>0, delta>0, kappa>-2, sign(E3-E0)=sign(E1-E0)=sign(E2-E0), |E3-E0|>=|E1-E0|, and |E3-E0|>=|E2-E0|.
        """

        return self._model(d1, d2, E0, E1, E2, E3, h1, h2, C1, C2, kappa, delta) - \
               self._model_nosynergy(d1, d2, E0, E1, E2, E3, h1, h2, C1, C2, kappa, delta)

    def _set_params(self, E0, E1, E2, E3, h1, h2, C1, C2, kappa, delta=1):
        self.E0 = E0
        self.E1 = E1
        self.E2 = E2
        self.E3 = E3
        self.h1 = h1
        self.h2 = h2
        self.C1 = C1
        self.C2 = C2
        self.kappa = kappa
        self.delta = delta

    def print_self_params(self):
        self.print_params(self.E0, self.E1, self.E2, self.E3, self.h1, self.h2, self.C1, self.C2, self.kappa, self.delta)

    @staticmethod
    def print_params(E0, E1, E2, E3, h1, h2, C1, C2, kappa, delta=1):
        print("%8s - %8s - %8s - %8s - %8s - %8s - %8s - %8s - %8s - %8s"
              %("E0", "E1", "E2", "E3", "h1", "h2", "C1", "C2", "kappa", "delta"))
        print("%8.2e - %8.2e - %8.2e - %8.2e - %8.2e - %8.2e - %8.2e - %8.2e - %8.2e - %8.2e"
              % (E0, E1, E2, E3, h1, h2, C1, C2, kappa, delta))

    def _get_single_drug_classes(self):
        return MyHill, MyHill


class MyHill(Hill):

    def _get_initial_guess(self, d, E, p0=None, maxval=100):
        if p0 is None:
            p0 = [maxval, np.min(E), 1, np.median(d)]  # I can't assume that I give d and E in order here
            # p0 = [max(E), min(E), 1, np.median(d)]

        p0 = list(self._transform_params_to_fit(p0))
        # note: transformed here for fit! so the value here (median(d)) will be large, not e-07 tjsp
        synergy.utils.sanitize_initial_guess(p0, self.bounds)
        return p0


class MyBraidWithOptimisers(MyBRAID):

    def fit_with_scipy_least_squares(self, concentrations, r, function, initial_guess, loss="linear", f_scale=1):

        """
        Here the function should given to fit should be be minimised
        I.e. the difference to responses should be already included in what is given there

        :param concentrations:
        :param r:
        :param function:
        :param initial_guess:
        :param loss: "linear", "soft_l1", "cauchy"
        :param f_scale: 0.1 by default ("inlier residuals should not significantly exceed 0.1")
        :return:
        """

        # new_function = lambda params, x, y: function(x, *params) - y
        def new_function(params, x, y):
            return function(x, *params) - y  # hmm todo why different result than curve_fit? cf uses ls internally..

        result = least_squares(new_function, initial_guess, args=(concentrations, r), bounds=self.bounds,
                               loss=loss, f_scale=f_scale)["x"]
        return result

    def fit_with_scipy_curve_fit(self, concentrations, r, function, initial_guess):
        if concentrations.shape[0] == len(r):
            concentrations = concentrations.T
        result = curve_fit(function, concentrations, r, p0=initial_guess, bounds=self.bounds)[0]
        return result

