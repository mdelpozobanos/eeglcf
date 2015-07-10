# -*- coding: utf-8 -*-

"""
This module contains some tools, related to the numpy library, that are used
throughout the eeglcf package.
"""

import numpy as np


def pearson(x, y=None):
    """Pearson Correlation Coefficient.

    r = sum((X - Mx)(Y - My)) / sqrt(SSx*SSy)

    Parameters
    ----------
    x : array
        If vector : Single data vector.
        If matrix : Matrix of shape L x N, where L is the length of the data and N the number of vectors.
    y : array, optional
        If specified, the correlation between each vector of x and each vector of y will be returned. Otherwise, the
        auto-correlation matrix of x will be returned.
        If vector : Single data vector.
        If matrix : Matrix of shape L x N, where L is the length of the data and N the number of vectors.

    Returns
    -------
    r : array
        Result

    """
    # Check x
    if not isinstance(x, np.ndarray):
        try:
            x = np.array(x)
        except TypeError:
            raise TypeError('x must be {}; '
                            'is {} instead'
                            .format(type(np.ndarray(0)), type(x)))
    if x.ndim is 1:  # Single vector, convert to column vector
        x = x[:, None]
    elif x.ndim is not 2:
        raise ValueError('x must be 1-D or 2-D; '
                         'is {}-D instead.'
                         .format(x.ndim))

    # Remove average from each vector of x
    x = x - x.mean(axis=0)
    # Compute variance of each vector
    x_var = (x**2).sum(axis=0)[None, :]

    # Check y
    if y is None:  # Compute auto-correlation matrix
        y = x
        y_var = x_var
    else:
        if not isinstance(y, np.ndarray):
            try:
                y = np.array(y)
            except TypeError:
                raise TypeError('y must be {}; '
                                'is {} instead'
                                .format(type(np.ndarray(0)), type(y)))
        if y.ndim is 1:  # Single vector, convert to column vector
            y = y[:, None]
        elif y.ndim is not 2:
            raise ValueError('y must be 1-D or 2-D; '
                             'is {}-D instead.'
                             .format(y.ndim))

        # Remove average and compute variance for each vector of y
        y = y - y.mean(axis=0)
        y_var = (y**2).sum(axis=0)[None, :]

    # Return correlation coefficient
    return np.dot(x.T, y) / np.sqrt(np.dot(x_var.T, y_var))


def hurst(x):
    """FASTER [1] implementation of the Hurst Exponent.

    Parameters
    ----------
    x : array
        Vector with the data sequence.

    Returns
    -------
    h : float
        Compute hurst exponent

    [1] H. Nolan, R. Whelan, and R.B. Reilly. Faster: Fully automated statistical thresholding for eeg artifact
        rejection. Journal of Neuroscience Methods, 192(1):152-162, 2010.
    """

    # Get a copy of the data
    x0 = x.copy()
    x0_len = len(x)

    yvals = np.zeros(x0_len)
    xvals = np.zeros(x0_len)
    x1 = np.zeros(x0_len)

    index = 0
    binsize = 1

    while x0_len > 4:

        y = x0.std()
        index += 1
        xvals[index] = binsize
        yvals[index] = binsize*y

        x0_len = int(x0_len/2)
        binsize *= 2

        for ipoints in range(x0_len):
            x1[ipoints] = (x0[2*ipoints] + x0[2*ipoints - 1])*0.5

        x0 = x1[:x0_len]

    # First value is always 0
    xvals = xvals[1:index+1]
    yvals = yvals[1:index+1]

    logx = np.log(xvals)
    logy = np.log(yvals)

    p2 = np.polyfit(logx, logy, 1)
    return p2[0]


def zscore_outliers(x, axis=None, keep_dims=False):
    """
    Detects outliers in the data.

    This is an iterative process to identify outliers based on the data
    z-scores. Within each iteration, z-scores higher than 3 are labeled as
    outliers and excluded from the computation of the mean and standard
    deviation (std) factors -- used in the computation of z-scores -- in the
    following iteration.

    Parameters
    ----------
    x : array
        Data array.
    axis : int, None, optional
        Axis of application.
    keep_dims : bool, optional
        If True, the returned mean and std results will have the same number of
        dimensions as x.

    Returns
    -------
    outliers : array
        Boolean array signaling the detected outliers.
    m : array
        Mean of x, computed without considering the outliers.
    s : array
        Standard deviation of x, computed without considering the outliers.
    """

    # Mask values with z-score greater than 3
    mx = np.ma.masked_array(x)
    just_masked = np.ones(1, dtype=bool)

    if x.ndim == 1:

        while just_masked.any():
            m = mx.mean(axis=axis)
            s = mx.std(axis=axis)
            just_masked = np.abs((mx - m) / s) > 3
            mx[just_masked] = np.ma.masked
            if mx.mask.all():
                return mx.mask, None, None

    else:

        while just_masked.any():
            m = np.expand_dims(mx.mean(axis=axis), axis=axis)
            s = np.expand_dims(mx.std(axis=axis), axis=axis)
            just_masked = np.abs((mx - m) / s) > 3
            mx[just_masked] = np.ma.masked
            if mx.mask.all():
                return mx.mask, None, None

        if not keep_dims:
            m = np.squeeze(m, axis=axis)
            s = np.squeeze(s, axis=axis)

    # Now, all values with z-scores greater than 3 have been removed from the
    # computation of the mean (m) and std (s)

    return mx.mask, m, s