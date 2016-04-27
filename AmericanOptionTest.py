import math
import numpy as np
import matplotlib.pyplot as plt

def get_random_paths(n_path=100000, n_timestep=100, r=0.05, sigma=0.2, T=1.0):
    """
    gets random paths. Can be used for computation of regular paths
    and mini paths.
    :param n_path: number of paths
    :type n_path: integer
    :param n_timestep: number of timesteps
    :type n_timestep: integer
    :return: gets n_path random paths of length n_timestep
    :rtype: np.ndarray([n_time_step, n_path])
    """

    dt = T/(n_timestep - 1.0)

    dw = math.sqrt(dt) * np.random.randn(n_path, n_timestep-1)
    ds_list = (r - 0.5 * sigma**2)*dt + sigma * dw

    s_list = np.exp(np.cumsum(ds_list, axis=1).transpose())
    result = np.concatenate((np.ones((1, n_path)), s_list))

    return result
#end def get_random_paths

def main():
    r = 0.05
    T = 1.0
    k = 1.0
    sigma = 0.2
    paths = get_random_paths(r=r, T=T, sigma=sigma)
    eur_option = np.exp(-r * T) * np.maximum(k - paths[-1, :], 0.0)
    print('European put option: E[max(K-S, 0)]: {0:g}+/-{1:g}'
          .format(np.mean(eur_option), np.std(eur_option)/math.sqrt(len(eur_option)-1.0)))

#end def main

if __name__ == '__main__':
    main()