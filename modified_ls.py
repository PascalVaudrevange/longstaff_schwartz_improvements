import logging
import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import operator


def blackScholes(s, k, sigma, r, tau, call=True):
    ones = np.ones_like(s)
    d1 = 1/(sigma * math.sqrt(tau)) * (np.log(s/k) + ones * (r + 0.5 * sigma**2) * tau)
    d2 = d1 - sigma * math.sqrt(tau) * ones
    sign = 1
    if not call:
        d1 = -d1
        d2 = -d2
        sign = -1
    #end if
    result = sign * (s * scipy.stats.norm.cdf(d1)
                    - k * scipy.stats.norm.cdf(d2) * math.exp(-r * tau))
    return result
#end def blackScholes

class AmericanOption(object):
    def __init__(self, S0=1.0
                    , K=1.0
                    , r=0.05
                    , sigma=0.2
                    , T=1.0
                    , payoff=None
                    , n_timestep=64
                    , n_path=1.0e5
                    , seed = None
                    , use_in_the_money=False
                    , use_independent_paths=False
                    , debug_level=logging.INFO
                    , order=3):
        """

        :param S0: value of the underlying today
        :type S0: float
        :param K: strike
        :type K: float
        :param r: interest rate
        :type r: float
        :param sigma: variance
        :type sigma: float
        :param T: expiry
        :type T: float
        :param payoff: provide your own payoff function, default
                       max(K-S,0)
        :type payoff: function
        :param n_timestep: number of timesteps
        :type n_timestep: integer
        :param n_path: number of paths
        :type n_path: integer
        :param seed: random seed.
        :type seed: integer
        :param use_in_the_money: use in-the-money paths only?
        :type use_in_the_money: boolean
        :param use_independent_paths: use independent paths for
                                      the computation of the
                                      continuation value and
                                      for computing the value of
                                      the option
        :param debug_level: debug level
        :type debug_level: one of logging.DEBUG, logging.INFO,
                           logging.WARN, logging.CRITICAL
        :param order: order of the expansion of the continuation value
        :type order: integer

        :return: None
        :rtype: NoneType
        """
        self.S0 = S0
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T = T
        logging.basicConfig(
            format='%(asctime)s: %(levelname)s: %(message)s'
            , level=debug_level)
        logging.debug('begin setup')
        self.n_timestep = n_timestep
        self.n_path = n_path
        self.use_in_the_money = use_in_the_money
        self.order = order
        self.dt = self.T/(1.0 * (self.n_timestep))
        self.discount_factor = np.exp(-self.r*self.dt)
        self.use_independent_paths = use_independent_paths
        if seed is not None:
            np.random.seed(seed)
        #end if

        if payoff is None:
            self.__payoff = self.__default_payoff
        else:
            self.__payoff = payoff
        #end if
    #end def __init__

    def __call__(self):
        """
        :return: see the description of method __evolve__()
        """

        if self.use_independent_paths:
            logging.debug('begin computation using independent paths')
            logging.debug('getting betas')
            beta_list = self.__evolve()['beta']
            logging.debug('begin evaluation')
            result = self.__evolve(beta_list)
            logging.debug('end computation using independent paths')
        else:
            logging.debug('begin computation')
            result = self.__evolve()
            logging.debug('end computation')
        return result

    #end def __call__

    def __get_path_list(self):
        """
        get the random paths for the MC evolution
        :return: paths for the MC evolution
        :rtype: np.ndarray(self.n_timestep, self.n_paths)
        """
        result = self.S0 * self.__get_random_path()
        return result
    #end def __get_path_list

    def __get_mini_path_endpoint_list(self, s0, n_path):
        """
        gets the endpoints of the n_path mini paths, starting from s0
        :param s0: starting point
        :type s0: float
        :param n_path: number of mini paths
        :type n_path: integer
        :return: list of endpoints of the minipaths
        :rtype: [float]
        """
        path_list = self.__get_random_path(n_path=n_path*s0.size
                                           , n_timestep=2)
        result = (np.outer(s0, np.ones(n_path))
                  * path_list[1, :].reshape(s0.shape + (n_path,)))
        return result
    #end def __get_mini_path_endpoint_list

    def __get_random_path(self, n_path=None, n_timestep=None):
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
        if n_path is None:
            n_path = self.n_path
        if n_timestep is None:
            n_timestep = self.n_timestep

        dw = math.sqrt(self.dt) * np.random.randn(n_path, n_timestep-1)
        ds_list = (self.r - 0.5 * self.sigma**2)*self.dt + self.sigma * dw

        s_list = np.exp(np.cumsum(ds_list, axis=1).transpose())
        result = np.concatenate((np.ones((1, n_path)), s_list))

        return result
    #end def get_random_path

    def __evolve(self, beta_list=None):
        """
        runs the MC evolution.

        If beta_list is None, compute continuation value. That is, return
        the coefficients of the expansion of the continuation value.

        If beta_list is not None, use the provided betas as coefficients
        for the expansion of the continuation value and compute the value
        of the option.

        :param beta_list: coefficients of the expansion of the continuation
                          value
        :type beta_list: np.ndarray([self.n_timestep, self.order]) or None
        :return: dictionary, containing
                'x': the sample paths in format x[t, path_number]
                'v': the value of the option along each sample path in
                     format v[t, path_number]
                'h': the payoff along each path
                     h[t, path_number] = max(K - x[t, path_number], 0)
                'beta': the coefficients of the expansion of the
                        continuation value. If the parameter beta
                        is not None, returns the values of beta
                'c': the continuation value in format c[t, path_number]
                'npv': the present value of the option as the mean of
                       v[0, path_number] over all paths
                'stddev': stddev of v[t, path_number] over all paths
        :rtype: dictionary
        """
        compute_beta = beta_list is None
        x = self.__get_path_list()

        v = np.zeros_like(x)
        c = np.zeros_like(x)
        bs = blackScholes
        my_cont_val = np.zeros_like(x)
        my_cont_val_low = np.zeros_like(x)
        if compute_beta:
            beta_list = np.zeros([self.n_timestep, self.order])
            logmessage = 'Computing betas, timestep'
        else:
            logmessage = 'Computing lower bound, timestep'
        #end if

        h = self.__payoff(x)
        v[-1, :] = h[-1, :]
        for t in xrange(self.n_timestep-2, 0, -1):
            logging.info('%s %s' % (logmessage, t))

            v[t, :] = self.discount_factor * v[t+1, :]

            if compute_beta:
                if self.use_in_the_money:
                    ind_in_the_money = h[t, :] > 0.0
                else:
                    ind_in_the_money = slice(0, None)
                #end if

                beta_list[t, :] = np.polyfit(x[t, ind_in_the_money]
                                             , v[t, ind_in_the_money]
                                             , self.order-1)
            #end if
            c[t, :] = np.polyval(beta_list[t, :], x[t, :])
            # S - K <= Call - Put <= S - K * exp(-r *tau)
            my_cont_val_low[t, :] = bs(x[t, :], self.K, self.sigma, self.r, self.T - t*self.dt, call=True)
            my_cont_val[t, :] = (my_cont_val_low[t, :]
                              + self.K * math.exp(-self.r * (self.T - t * self.dt)) - x[t, :])

            ind = h[t, :] > np.maximum(my_cont_val[t, :], c[t,:])

            print t, sum(0 > c[t,:]), sum(my_cont_val[t,: ] > c[t, :])
            ind = h[t, :] > np.maximum(0.0, c[t,:])
            if any(ind):
                v[t, ind] = h[t, ind]
            #end if

        #end for

        # no exercise right at the start of the option, thus just discount
        v[0, :] = self.discount_factor * v[1, :]

        npv = np.mean(v[0, :])
        stddev = np.std(v[0, :])/np.sqrt(len(v[0, :]))
        result = dict(npv=npv, stddev=stddev, beta=beta_list, v=v, h=h,
                      c=c, x=x, my_cont_val=my_cont_val)
        return result
    #end def __evolve

    def get_upper_bound(self, x, v, h, c, beta_list, n_minipath=100):
        """
        compute the Rogers upper bound of the option value. Must be called
        using the return values of __call__(), i.e. after having called
        __call__()

        :param x: list of paths used for the computation of the lower
                  bound in format x[t, path_number]
        :type x: np.ndarray([self.n_timestep, self.n_paths])
        :param v: the value of the option along each sample path in format
                  v[t, path_number]
        :type v: np.ndarray([self.n_timestep, self.n_paths])
        :param h: the payoff along each path
                  h[t, path_number] = max(K - x[t, path_number], 0)
        :type h: np.ndarray([self.n_timestep, self.n_paths])
        :param c: the continuation value in format c[t, path_number]
        :type c: np.ndarray([self.n_timestep, self.n_paths])
        :param beta_list: the coefficients of the expansion of the
                          continuation value. If the parameter beta is not
                          None, returns the values of beta_list
        :type beta_list: np.ndarray([self.n_timestep, self.order])
        :type c: np.ndarray([self.n_timestep, self.n_paths])
        :param n_minipath: the number of mini paths
        :type n_minipath: integer
        :return: dictionary containing
                'upper_bound': the upper bound
                'stddev':  sttdev of the upper bound estimate
                'upper_bound_per_path': the collection of self.n_paths
                                        upper bounds
        :rtype: dictionary
        """
        m_part = np.zeros([self.n_timestep, self.n_path])

        for t in xrange(1, self.n_timestep):
            logging.info('Computing upper bound, timestep %s' % t)

            mini_path_endpoint_list = self.__get_mini_path_endpoint_list(
                                        s0=x[t - 1, :]
                                        , n_path=n_minipath)
            mini_h = self.__payoff(mini_path_endpoint_list)
            mini_c = np.polyval(beta_list[t, :], mini_path_endpoint_list)
            mini_v = np.maximum(mini_h, mini_c)
            mini_expectation_value = np.mean(mini_v, axis=1)
            m_part[t, :] = (np.maximum(h[t, :], c[t, :])
                            - mini_expectation_value)
        #end for
        m = np.cumsum(m_part, axis=0)

        upper_bound_per_path = np.max(h - m, axis=0)

        result = {'upper_bound': np.mean(upper_bound_per_path)
                    , 'stddev': np.std(upper_bound_per_path)
                    , 'upper_bound_per_path':  upper_bound_per_path}
        return result
    #end def get_upper_bound

    def __default_payoff(self, s):
        """
        get the default payoff for a put option
        :param s: value of the underlying
        :type s: float
        :return: payoff for a put option with underlying value s
        :rtype: float
        """
        return np.maximum(self.K-s, 0.0)
    #end def __payoff

#end class AmericanOption


def plot(x, c, v):
    n_timestep = len(x)
    dim = int(np.ceil(np.sqrt(n_timestep)))
    fig, axs = plt.subplots(dim
                            , dim
                            , sharex=True
                            , sharey=True
                            , figsize=(11,8))
    fig.subplots_adjust(wspace=0, hspace=0)
    plt.hold(True)
    for t in xrange(n_timestep):
        ax = axs[t/dim, t % dim]
        line_v, = ax.plot(x[t, :], v[t, :], 'g.', label=r'$V$')
        x_t = x[t, :]
        c_t = c[t, :]
        line_c, = ax.plot(x_t, c_t, 'b.', label=r'$\hat{C}$', linewidth=2)
        x_t_neg, c_t_neg = x_t[c_t<0.0], c_t[c_t<0.0]
        line_c_neg, = ax.plot(x_t_neg, c_t_neg, 'r.', label=r'$\hat{C}<0$'
                              , linewidth=2)
        ax.set_xlim([0, 1.9])
        ax.set_ylim([-0.04, 0.34])
        ax.text(0.5, 0.8, '%s' % (n_timestep-t-1)
                , horizontalalignment='center'
                , transform=ax.transAxes)
        if t % dim == 0:
            ax.set_ylabel(r'$C_{\tau}, V_{\tau}$')
        if (t > 55):
            ax.set_xlabel(r'$S_{\tau}$')
    #end for

    fig.legend([line_v, line_c, line_c_neg,]
                , [r'$V$', r'$\hat{C}$', r'$\hat{C}<0$']
                , bbox_to_anchor=(0.5, 1)
                , loc='upper center'
                , ncol=3
                , columnspacing=3
                #, borderaxespad=0.
               )

    plt.show()
#end def plot


def test(x, c, v):
    plot(x, c, v)
#end def test


def test_upper_bound(result):
    plt.hist(result['upper_bound_per_path'], bins=100)
    plt.show()
#end def test_upper_bound

def print_table_row(message, option, npv, stddev):
    """
    output the results of the calls to section*() in a format that can
    be copied directly into LaTeX tables

    :param message: description of the run
    :type message: string
    :param option: to fill the
    :type option: AmericanOption
    :param npv: value for column 'value'
    :type npw: float
    :param stddev: value for column 'stddev'
    :type stddev: float
    :return: None
    """
    print('%s&$%d$&$%2.1e$&%s&%s&$%g$&$%g$\\\\' % (message
                            , option.n_timestep
                            , option.n_path
                            , option.use_independent_paths
                            , option.use_in_the_money
                            , npv
                            , stddev))
#end def print_table_row


def section1_2_0():
    option = AmericanOption(use_independent_paths=False
                            , use_in_the_money=False
                            , n_path=1.e5
                            , debug_level=logging.WARNING
                            , seed=1)
    result = option()

    logging.info("Using all paths, the value of a single run is %(npv)g, stddev: %(stddev)g" % result)
    print_table_row('Section 1.2: single run for the unmodified original code', option, result['npv'], result['stddev'])
    return result
#end def section1_2_0


def section1_2_1():
    option = AmericanOption(use_independent_paths=True
                            , use_in_the_money=False
                            , n_path=1.e5
                            , debug_level=logging.WARNING
                            , seed=1)
    result = option()
    logging.info("Using all paths, the value of a single run is %(npv)g, stddev: %(stddev)g" % result)
    print_table_row('Section 1.2.1: single run', option, result['npv']
                    , result['stddev'])
    return result
#end def section1_2_1
section1_2_1()

def section1_2_2(n_runs=100):
    option = AmericanOption(use_independent_paths=True
                                , use_in_the_money=False
                                , n_path=1.e5
                                , debug_level=logging.WARNING
                                , seed=1)
    result_list = [option()['npv'] for _ in xrange(n_runs)]
    avg = np.mean(result_list)
    stddev = np.std(result_list)
    logging.info('Using all paths, the average value of %s runs is %s with a stddev of %s'
          %(n_runs, avg, stddev))
    print_table_row('Section 1.2.2: average value of %s runs' % n_runs
                    , option, avg, stddev)
    return result_list
#end def section1_2_2


def section1_2_3(n_runs=100):
    option = AmericanOption(use_independent_paths=True
                                , use_in_the_money=True
                                , n_path=1.e5
                                , debug_level=logging.WARNING
                                , seed=1)
    result_list = [option()['npv'] for _ in xrange(n_runs)]
    avg = np.mean(result_list)
    stddev = np.std(result_list)
    logging.info('Using in-the-money paths only, the average value of %s runs is %s with a stddev of %s'
          % (n_runs, avg, stddev))
    print_table_row('Section 1.2.3: average value of %s runs' % n_runs
                    , option, avg, stddev)
    return result_list
#end def section1_2_3


def section1_3():
    option = AmericanOption(use_independent_paths=True
                            , use_in_the_money=False
                            , n_path=1000
                            , debug_level=logging.WARNING
                            , seed=1)
    result = option()

    x = result['x']
    c = result['c']
    h = result['h']
    v = result['v']
    beta_list = result['beta']

    result = option.get_upper_bound(x, v, h, c, beta_list, n_minipath=100)
    logging.info('Upper limit: %s, stddev: %s'
                 % (result['upper_bound'], result['stddev']))
    print_table_row('Section 1.3: upper bound using Roger\'s algorithm and 100 minipaths'
                    , option, result['upper_bound'], result['stddev'])
    return result
#end def section1_3


def section1_3_2():
    option = AmericanOption(use_independent_paths=True
                            , use_in_the_money=False
                            , n_path=1.e5
                            , debug_level=logging.WARNING
                            , seed=1)
    result = option()

    x = result['x']
    c = result['c']
    h = result['h']
    v = result['v']
    beta_list = result['beta']

    result = option.get_upper_bound(x, v, h, c, beta_list, n_minipath=100)
    logging.info('Upper limit: %s, stddev: %s'
                 % (result['upper_bound'], result['stddev']))
    print_table_row('Section 1.3.2: upper bound using Roger\'s algorithm and 100 minipaths'
                    , option, result['upper_bound'], result['stddev'])
    return result
#end def section1_3_2
"""
def plot_hist():
plt.gcf()
plt.hold(True)
_ = plt.hist(result1_2_2
         , facecolor='None'
         , edgecolor='b'
         , lw=3
         , label='Using all paths')
_ = plt.hist(result1_2_3
         , facecolor='None'
         , edgecolor='r'
         , lw=3
         , ls='dashed'
         , label='Using only in-the-money paths')
plt.xlabel('Value of the American put option')
plt.ylabel('Number')
plt.legend()
plt.show()
plt.hold(False)


def plot_hist_upper():
plt.gcf()
plt.hold(True)
_ = plt.hist(result1_2_0['v'][0, :]
         , facecolor='None'
         , edgecolor='b'
         , lw=3
         , normed=True
         , label='Not using independent paths for the continuation value'
         , bins=50)
_ = plt.hist(result1_2_1['v'][0, :]
         , facecolor='None'
         , edgecolor='r'
         , lw=3
         , ls='dashed'
         , normed=True
         , label='using independent paths for the continuation value'
         , bins=50)
_ = plt.hist(result1_3['upper_bound_per_path']
         , facecolor='None'
         , edgecolor='g'
         , lw=3
         , ls='dotted'
         , normed=True
         , label='upper bound')
plt.xlabel('Value of the American put option')
plt.ylabel('Frequency')
plt.show()
plt.hold(False)
"""
#end def plot_hist



