import numpy as np


ZEROCONCENTRATION = 1e-16


def vec_to_mat(m, shape=(4, 4)):
    return np.reshape(m, shape, order="F")


def mat_to_vec(M):
    return np.reshape(M, (-1, 1), order="F").squeeze()


def coordinate_transformation(z):
    x = np.copy(z)
    x[x<=0] = 1e-16
    x = np.log10(x)
    x += 16
    return x


def my_log_transformation(x):
    return (np.log10(x)+16)  # => zeroconcentration will map to 0 => actually, would 1 be better?


def my_inv_log_transformation(x):
    return 10**(x-16)


def get_loggrid(n_c_in_log):
    a1 = np.logspace(-12, -4, 9)
    c_for_kernel = np.append([0], np.outer(a1, np.arange(1, 10, 9 / n_c_in_log)).flatten())
    return c_for_kernel


def hill_equation(c_query, zero_resp, max_resp, hill_slope, halfway_c):

    dh = np.power(c_query,hill_slope)
    return zero_resp + (max_resp-zero_resp)*dh/(np.power(halfway_c,hill_slope)+dh)


def normalized_c_from_c(c_query, zero_resp, max_resp, hill_slope, halfway_c):
    transfer = lambda resp: (resp - zero_resp)/(max_resp-zero_resp)  # response into concentration
    from_c_to_transfer = lambda c: transfer(hill_equation(c, zero_resp, max_resp, hill_slope, halfway_c))  # response
    res = from_c_to_transfer(c_query)
    # case bigger than max or smaller than zero? not going to happen because this is from model, not measurement.
    # res[res < 0] = 0
    # res[res > 1] = 1
    return res


def c_from_normalized_c(c_query, zero_resp, max_resp, hill_slope, halfway_c):
    inverse_transfer = lambda resp: resp*(max_resp-zero_resp) + zero_resp

    tmp_query = np.copy(c_query)
    tmp_query[tmp_query==1] = 0.99

    def inverse_hill(resp_query):
        E_ratio = (resp_query-zero_resp)/(max_resp-resp_query)
        d = np.float_power(E_ratio, 1./hill_slope)*halfway_c
        # hmm todo do I need these checks about E_ratio? no?
        # if hasattr(E,"__iter__"):  # hmm?
        #     d[E_ratio<0] = np.nan
        #     return d
        # elif d < 0: return np.nan
        return d

    # print("1:", inverse_hill(inverse_transfer(1)))
    # print("0.99:", inverse_hill(inverse_transfer(0.99)))
    # print("0.95:", inverse_hill(inverse_transfer(0.95)))
    # print("0.9:", inverse_hill(inverse_transfer(0.9)))

    return inverse_hill(inverse_transfer(tmp_query))


def transpose_fun(d):
    # d for d*d matrix
    # if the d*d matrix has been vectorized, how to rearrange columns to get vectorization of its transpose
    vals = np.arange(d**2)
    valmat = vec_to_mat(vals, (d, d))
    new_inds = mat_to_vec(valmat.T)
    return new_inds
