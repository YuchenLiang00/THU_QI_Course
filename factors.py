import time
import pandas as pd
import numpy as np
import math
from scipy.stats import linregress
from collections import defaultdict


class Factor:
    """ static class for factor calculation """
    # TODO: 在使用因子之前做条件检验，看看这些因子有哪些特性，（前人没用过的 
    # 例如物理上的假设条件在股市上不存在，那么我们可能要做一些针对性的调整
    # 小波： 还可以对信号做拆解。原来的信号可能很难找到规律，但是我们可以使用小波分析对原始信号做拆解。

    @staticmethod
    def max(x):
        """
        Calculates the highest value of the time series x.

        :param x: the time series to calculate the feature of
        :type x: numpy.ndarray
        :return: the value of this feature
        :return type: float
        """
        return np.max(x)

    @staticmethod
    def index_mass_quantile(x, q=.25):
        """ 分位数索引 
        Calculates the relative index i of time series x where q% of the mass of x lies left of i.
        For example for q = 50% this feature calculator will return the mass center of the time series.

        :param x: the time series to calculate the feature of
        :type x: numpy.ndarray
        :param param: contains dictionaries {"q": x} with x float
        :type param: list
        :return: the different feature values
        :return type: pandas.Series
        """
        x = np.asarray(x)
        abs_x = np.abs(x)
        s = np.sum(abs_x)

        if not s:
            # all values in x are zero or it has length 0
            return np.nan
        else:
            # at least one value is not zero
            mass_centralized = np.cumsum(abs_x) / s

            return (np.argmax(mass_centralized >= q) + 1) / len(x)  

    @staticmethod
    def quantile(x, q=.25):
        """ x的q分位数 
        Calculates the q quantile of x. This is the value of x greater than q% of the ordered values from x.

        :param x: the time series to calculate the feature of
        :type x: numpy.ndarray
        :param q: the quantile to calculate
        :type q: float
        :return: the value of this feature
        :return type: float
        """
        if not x:
            return np.nan
        else:
            return np.quantile(x, q)
        

    @staticmethod
    def energy_ratio_by_chunks(x):
        """ 分块局部熵比率 
        Calculates the sum of squares of chunk i out of N chunks expressed as a ratio with the sum of squares over the whole
        series.

        Takes as input parameters the number num_segments of segments to divide the series into and segment_focus
        which is the segment number (starting at zero) to return a feature on.

        If the length of the time series is not a multiple of the number of segments, the remaining data points are
        distributed on the bins starting from the first. For example, if your time series consists of 8 entries, the
        first two bins will contain 3 and the last two values, e.g. `[ 0.,  1.,  2.], [ 3.,  4.,  5.]` and `[ 6.,  7.]`.

        Note that the answer for `num_segments = 1` is a trivial "1" but we handle this scenario
        in case somebody calls it. Sum of the ratios should be 1.0.

        :param x: the time series to calculate the feature of
        :type x: numpy.ndarray
        :param param: contains dictionaries {"num_segments": N, "segment_focus": i} with N, i both ints
        :return: the feature values
        :return type: list of tuples (index, data)
        """
        res_data = []
        full_series_energy = np.sum(x ** 2)
        num_segments = 10
        segment_focus = 5

        assert segment_focus < num_segments
        assert num_segments > 0

        if full_series_energy == 0:
            return np.nan
        else:
            return (
                np.sum(np.array_split(x, num_segments)[segment_focus] ** 2.0)
                / full_series_energy
            )     

    @staticmethod
    def c3(x, lag = 2):
        """
        时序数据非线性度量

        Uses c3 statistics to measure non linearity in the time series

        This function calculates the value of

        .. math::

            \\frac{1}{n-2lag} \\sum_{i=1}^{n-2lag} x_{i + 2 \\cdot lag} \\cdot x_{i + lag} \\cdot x_{i}

        which is

        .. math::

            \\mathbb{E}[L^2(X) \\cdot L(X) \\cdot X]

        where :math:`\\mathbb{E}` is the mean and :math:`L` is the lag operator. It was proposed in [1] as a measure of
        non linearity in the time series.

        .. rubric:: References

        |  [1] Schreiber, T. and Schmitz, A. (1997).
        |  Discrimination power of measures for nonlinearity in a time series
        |  PHYSICAL REVIEW E, VOLUME 55, NUMBER 5

        :param x: the time series to calculate the feature of
        :type x: numpy.ndarray
        :param lag: the lag that should be used in the calculation of the feature
        :type lag: int
        :return: the value of this feature
        :return type: float
        """
        def _roll(a, shift):
            """
            Roll 1D array elements. Improves the performance of numpy.roll() by reducing the overhead introduced from the
            flexibility of the numpy.roll() method such as the support for rolling over multiple dimensions.

            Elements that roll beyond the last position are re-introduced at the beginning. Similarly, elements that roll
            back beyond the first position are re-introduced at the end (with negative shift).

            Examples
            --------
            Benchmark
            ---------
            :param a: the input array
            :type a: array_like
            :param shift: the number of places by which elements are shifted
            :type shift: int

            :return: shifted array with the same shape as a
            :return type: ndarray
            """
            if not isinstance(a, np.ndarray):
                a = np.asarray(a)
            idx = shift % len(a)
            return np.concatenate([a[-idx:], a[:-idx]])
        if not isinstance(x, (np.ndarray, pd.Series)):
            x = np.asarray(x)
        n = x.size
        if 2 * lag >= n:
            return 0
        else:
            return np.mean(
                (_roll(x, 2 * -lag) * _roll(x, -lag) * x)[0 : (n - 2 * lag)]
            )

    @staticmethod
    def fft_aggregated(x, aggtype="centroid"):
        """ 绝对傅里叶变换的谱统计量 
        Returns the spectral centroid (mean), variance, skew, and kurtosis of the absolute fourier transform spectrum.

        :param x: the time series to calculate the feature of
        :type x: numpy.ndarray
        :param param: contains dictionaries {"aggtype": s} where s str and in ["centroid", "variance",
            "skew", "kurtosis"]
        :type param: list
        :return: the different feature values
        :return type: pandas.Series
        """

        assert aggtype in {
            "centroid",
            "variance",
            "skew",
            "kurtosis",
        }, 'Attribute must be "centroid", "variance", "skew", "kurtosis"'

        def get_moment(y, moment):
            """
            Returns the (non centered) moment of the distribution y:
            E[y**moment] = \\sum_i[index(y_i)^moment * y_i] / \\sum_i[y_i]

            :param y: the discrete distribution from which one wants to calculate the moment
            :type y: pandas.Series or np.array
            :param moment: the moment one wants to calcalate (choose 1,2,3, ... )
            :type moment: int
            :return: the moment requested
            :return type: float
            """
            return y.dot(np.arange(len(y), dtype=float) ** moment) / y.sum()

        def get_centroid(y):
            """
            :param y: the discrete distribution from which one wants to calculate the centroid
            :type y: pandas.Series or np.array
            :return: the centroid of distribution y (aka distribution mean, first moment)
            :return type: float
            """
            return get_moment(y, 1)

        def get_variance(y):
            """
            :param y: the discrete distribution from which one wants to calculate the variance
            :type y: pandas.Series or np.array
            :return: the variance of distribution y
            :return type: float
            """
            return get_moment(y, 2) - get_centroid(y) ** 2

        def get_skew(y):
            """
            Calculates the skew as the third standardized moment.
            Ref: https://en.wikipedia.org/wiki/Skewness#Definition

            :param y: the discrete distribution from which one wants to calculate the skew
            :type y: pandas.Series or np.array
            :return: the skew of distribution y
            :return type: float
            """

            variance = get_variance(y)
            # In the limit of a dirac delta, skew should be 0 and variance 0.  However, in the discrete limit,
            # the skew blows up as variance --> 0, hence return nan when variance is smaller than a resolution of 0.5:
            if variance < 0.5:
                return np.nan
            else:
                return (
                    get_moment(y, 3) - 3 * get_centroid(y) * variance - get_centroid(y) ** 3
                ) / get_variance(y) ** (1.5)

        def get_kurtosis(y):
            """
            Calculates the kurtosis as the fourth standardized moment.
            Ref: https://en.wikipedia.org/wiki/Kurtosis#Pearson_moments

            :param y: the discrete distribution from which one wants to calculate the kurtosis
            :type y: pandas.Series or np.array
            :return: the kurtosis of distribution y
            :return type: float
            """

            variance = get_variance(y)
            # In the limit of a dirac delta, kurtosis should be 3 and variance 0.  However, in the discrete limit,
            # the kurtosis blows up as variance --> 0, hence return nan when variance is smaller than a resolution of 0.5:
            if variance < 0.5:
                return np.nan
            else:
                return (
                    get_moment(y, 4)
                    - 4 * get_centroid(y) * get_moment(y, 3)
                    + 6 * get_moment(y, 2) * get_centroid(y) ** 2
                    - 3 * get_centroid(y)
                ) / get_variance(y) ** 2

        calculation = dict(
            centroid=get_centroid,
            variance=get_variance,
            skew=get_skew,
            kurtosis=get_kurtosis,
        )

        fft_abs = np.abs(np.fft.rfft(x))

        res = calculation[aggtype](fft_abs)
        return res

    @staticmethod
    def fft_coefficients(x, coeff=2, attr="abs"):
        """ - 傅里叶变换系数 
        Calculates the fourier coefficients of the one-dimensional discrete Fourier Transform for real input by fast
        fourier transformation algorithm

        .. math::
            A_k =  \\sum_{m=0}^{n-1} a_m \\exp \\left \\{ -2 \\pi i \\frac{m k}{n} \\right \\}, \\qquad k = 0,
            \\ldots , n-1.

        The resulting coefficients will be complex, this feature calculator can return the real part (attr=="real"),
        the imaginary part (attr=="imag), the absolute value (attr=""abs) and the angle in degrees (attr=="angle).

        :param x: the time series to calculate the feature of
        :type x: numpy.ndarray
        :param param: contains dictionaries {"coeff": x, "attr": s} with x int and x >= 0, s str and in ["real", "imag",
            "abs", "angle"]
        :type param: list
        :return: the different feature values
        :return type: pandas.Series
        """
 
        assert coeff >= 0, "Coefficients must be positive or zero."
        assert attr in ("imag","real", "abs", "angle"), 'Attribute must be "real", "imag", "angle" or "abs"'

        fft = np.fft.rfft(x)

        def complex_agg(x, agg):
            if agg == "real":
                return x.real
            elif agg == "imag":
                return x.imag
            elif agg == "abs":
                return np.abs(x)
            elif agg == "angle":
                return np.angle(x, deg=True)
        if coeff < len(fft):
            return complex_agg(fft[coeff], attr)
        else:
            return np.nan
        
    @staticmethod
    def mean_n_absolute_max(x):
        """ - n 个绝对最大值的算术平均值 
        Calculates the arithmetic mean of the n absolute maximum values of the time series.

        :param x: the time series to calculate the feature of
        :type x: numpy.ndarray
        :param number_of_maxima: the number of maxima, which should be considered
        :type number_of_maxima: int

        :return: the value of this feature
        :return type: float
        """

        number_of_maxima = math.ceil(len(x) * .1)

        assert number_of_maxima > 0, f" number_of_maxima={number_of_maxima} which is not greater than 1"

        n_absolute_maximum_values = np.sort(np.absolute(x))[-number_of_maxima:]

        return np.mean(n_absolute_maximum_values)
        

    @staticmethod
    def cid_ce(x, is_normalize=False):
        """ - 时序数据复杂度，用来评估时间序列的复杂度，越复杂的序列有越多的谷峰 
        This function calculator is an estimate for a time series complexity [1] (A more complex time series has more peaks,
        valleys etc.). It calculates the value of

        .. math::

            \\sqrt{ \\sum_{i=1}^{n-1} ( x_{i} - x_{i-1})^2 }

        .. rubric:: References

        |  [1] Batista, Gustavo EAPA, et al (2014).
        |  CID: an efficient complexity-invariant distance for time series.
        |  Data Mining and Knowledge Discovery 28.3 (2014): 634-669.

        :param x: the time series to calculate the feature of
        :type x: numpy.ndarray
        :param normalize: should the time series be z-transformed?
        :type normalize: bool

        :return: the value of this feature
        :return type: float
        """
        if not isinstance(x, (np.ndarray, pd.Series)):
            x = np.asarray(x)
        if is_normalize:
            s = np.std(x)
            if s:
                x = (x - np.mean(x)) / s
            else:
                return 0.0

        x = np.diff(x)
        return np.sqrt(np.dot(x, x))
        

    @staticmethod
    def linear_trend(x):
        """ - 线性回归分析
        Calculate a linear least-squares regression for the values of the time series versus the sequence from 0 to
        length of the time series minus one.
        This feature assumes the signal to be uniformly sampled. It will not use the time stamps to fit the model.
        The parameters control which of the characteristics are returned.

        Possible extracted attributes are "pvalue", "rvalue", "intercept", "slope", "stderr", see the documentation of
        linregress for more information.

        :param x: the time series to calculate the feature of
        :type x: numpy.ndarray
        :param param: contains dictionaries {"attr": x} with x an string, the attribute name of the regression model
        :type param: list
        :return: the different feature values
        :return type: pandas.Series
        """
        attr = "slop"
        # todo: we could use the index of the DataFrame here
        linReg = linregress(range(len(x)), x)

        return getattr(linReg, attr)
        

    @staticmethod
    def agg_linear_trend(x, attr="slope", chunk_len=2, f_agg="mean"):
        """ - 基于分块时序聚合值的线性回归 
        Calculates a linear least-squares regression for values of the time series that were aggregated over chunks versus
        the sequence from 0 up to the number of chunks minus one.

        This feature assumes the signal to be uniformly sampled. It will not use the time stamps to fit the model.

        The parameters attr controls which of the characteristics are returned. Possible extracted attributes are "pvalue",
        "rvalue", "intercept", "slope", "stderr", see the documentation of linregress for more information.

        The chunksize is regulated by "chunk_len". It specifies how many time series values are in each chunk.

        Further, the aggregation function is controlled by "f_agg", which can use "max", "min" or , "mean", "median"

        :param x: the time series to calculate the feature of
        :type x: numpy.ndarray
        :param param: contains dictionaries {"attr": x, "chunk_len": l, "f_agg": f} with x, f an string and l an int
        :type param: list
        :return: the different feature values
        :return type: pandas.Series
        """

        def _aggregate_on_chunks(x, f_agg, chunk_len):
            """
            Takes the time series x and constructs a lower sampled version of it by applying the aggregation function f_agg on
            consecutive chunks of length chunk_len

            :param x: the time series to calculate the aggregation of
            :type x: numpy.ndarray
            :param f_agg: The name of the aggregation function that should be an attribute of the pandas.Series
            :type f_agg: str
            :param chunk_len: The size of the chunks where to aggregate the time series
            :type chunk_len: int
            :return: A list of the aggregation function over the chunks
            :return type: list
            """
            return [getattr(x[i * chunk_len : (i + 1) * chunk_len], f_agg)()
                for i in range(int(np.ceil(len(x) / chunk_len)))
            ]
        
        # todo: we could use the index of the DataFrame here

        calculated_agg = defaultdict(dict)
        res_data = []

        if f_agg not in calculated_agg or chunk_len not in calculated_agg[f_agg]:
            if chunk_len >= len(x):
                calculated_agg[f_agg][chunk_len] = np.nan
            else:
                aggregate_result = _aggregate_on_chunks(x, f_agg, chunk_len)
                lin_reg_result = linregress(
                    range(len(aggregate_result)), aggregate_result
                )
                calculated_agg[f_agg][chunk_len] = lin_reg_result

        if chunk_len >= len(x):
            return np.nan
        else:
            return getattr(calculated_agg[f_agg][chunk_len], attr)