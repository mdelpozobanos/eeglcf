# -*- coding: utf-8 -*-

"""
===============================
eeglcf.py
===============================

This is the main file from the eeglcf package.

"""

import numpy as np
import scipy.signal as sp_signal

# Define here the dimensions for readability of the algorithm
_comp_dim = 0  # Components dimension
_t_dim = 1  # Time dimension
_ev_dim = 2  # Events dimension


def lcf(comp0, comp1=None,
        integrator_width=20, detection_th=1.,
        dilator_width=10, transition_width=10):
    """
    Localized Component Filtering

    Detects the location of artifacts in the time representation of source
    components and mixes them with an alternative (cleaned) version.

    Parameters
    ----------
    comp0 : array
        Array containing the original components, which will be analysed in
        search of noise. It must be a 3D array with shape CxTxE, where C, T and
        E are the number of components, time samples and recorded events
        respectively.
    comp1 : array, optional
        Array containing the alternative (cleaned) components. It must have
        the same shape as *comp0*. If not specified, an all 0s alternative
        components will be used (this is equivalent to component rejection).
    integrator_width : int > 0, optional
        Width (in number of samples) of the integration
    detection_th : float > 0, optional
        Detection threshold
    dilator_width : int > 0, optional
        Width (in number of samples) of the dilator
    transition_width : int > 0, optional
        Width (in number of samples) of the transition window

    Returns
    -------
    comp2 : array
        Array with the resulting components. This will have the same shape
        as *comp0*.
    """

    # 1. Compute features
    features = _features(comp0)
    # 2. Integrate features
    integrated_features = _integrate(integrator_width, *features)
    # 3. Classify features
    ctr_signal = _classify(detection_th, dilator_width, *integrated_features)
    # 4. Mix components
    return _mix(transition_width, comp0, ctr_signal, comp1)


def _chk_parameters(comp0=None, comp1=None,
                    integrator_width=None, detection_th=None,
                    dilator_width=None, transition_width=None):
    """
    Checks input parameters.

    Parameters
    ----------
    comp0 : array
        A 3D array with shape CxTxE, where C, T and E are the number of
        components, time samples and recorded events respectively.
    comp1 : array
        A 3D array with shape CxTxE, where C, T and E are the number of
        components, time samples and recorded events respectively. If both
        *comp0* and *comp1* are provided, the must have the same shape.
    integrator_width : int > 0, optional
        Width (in number of samples) of the integration.
    detection_th : {float, int} > 0, optional
        Detection threshold.
    dilator_width : int > 0, optional
        Width (in number of samples) of the dilator.
    transition_width : int > 0, optional
        Width (in number of samples) of the transition window.
    """

    # Check components comp0 and comp1
    if comp0 is not None:
        if not isinstance(comp0, np.ndarray):
            return TypeError('comp0 must be {}; '
                             'is {} instead'
                             .format(type(np.ndarray(0)), type(comp0)))
        if comp0.ndim != 3:
            return ValueError('comp0 must be 3D array; '
                              'is {}D instead'.format(comp0.ndim))
    if comp1 is not None:
        if not isinstance(comp1, np.ndarray):
            return TypeError('comp1 must be {}; '
                             'is {} instead'
                             .format(type(np.ndarray(0)), type(comp1)))
        if comp1.ndim != 3:
            return ValueError('comp1 must be 3D array; '
                              'is {}D instead'.format(comp1.ndim))
    if (comp0 is not None) and (comp1 is not None):
        if comp0.shape != comp1.shape:
            return ValueError('comp0 and comp1 must have equal shape')

    # Check integrator_width
    if integrator_width is not None:
        if not isinstance(integrator_width, int):
            return TypeError('integrator_width must be {}; '
                             'is {} instead'
                             .format(type(1), type(integrator_width)))
        if not integrator_width > 0:
            return ValueError('integrator_width must be > 0')

    # Check detection_th
    if detection_th is not None:
        if not isinstance(detection_th, (float, int)):
            return TypeError('detection_th must be {}; '
                             'is {} instead'
                             .format(type(0.), type(detection_th)))
        if not detection_th > 0:
            return ValueError('detection_th must be > 0')

    # Check dilator_width
    if dilator_width is not None:
        if not isinstance(dilator_width, int):
            return TypeError('dilator_width must be {}; '
                             'is {} instead'
                             .format(type(0), type(dilator_width)))
        if not dilator_width > 0:
            return ValueError('dilator_width must be int')

    # Check transition_width
    if transition_width is not None:
        if not isinstance(transition_width, int):
            return TypeError('transition_width must be {}; '
                             'is {} instead'
                             .format(type(0), type(transition_width)))
        if not transition_width > 0:
            return ValueError('transition_width must be > 0')


def _features(comp0):
    """
    Computes features characterizing the presence of noise in the components.

    Parameters
    ----------
    comp0 : array
        Array containing the original components, which will be analysed in
        search of noise. It must be a 3D array with shape CxTxE, where C, T and
        E are the number of components, time samples and recorded events
        respectively.

    Returns
    -------
    v : array
        Normalized voltages.
    dvdt : array
        Normalized first backward time derivative of the voltage.
    """

    def zscore(x):
        """
        Computes (in place) a robust zscore of the data, using the trimmed
        mean and std.

        Parameters
        ----------
        x : array
            Data with shape equal to comp0.shape

        Returns
        -------
        y : array
            Normalized data, with shape equal to comp0.shape

        """
        # Compute mean and std
        try:
            m = x.mean(axis=(_t_dim, _ev_dim), keepdims=True)
        except IndexError:
            raise _chk_parameters(comp0=comp0)
        s = x.std(axis=(_t_dim, _ev_dim), keepdims=True)
        # A masked array will be used to remove outliers
        masked_x = np.ma.array(x, mask=np.abs((x - m) / s) > 3)
        del m, s
        # Recompute mean and std. As for now, masked arrays don't support
        # a tuple of axes. Hence, the array has to be reshaped.
        # assert(_comp_dim is 0)
        masked_x = masked_x.reshape([masked_x.shape[0],
                                     np.prod(masked_x.shape[1:])])
        m = masked_x.mean(axis=1)
        s = masked_x.std(axis=1)
        del masked_x
        # Now, mean and std have to be reshaped to match x
        shape_ms = np.ones(x.ndim)  # Shape of mean and std vectors
        shape_ms[_comp_dim] = x.shape[_comp_dim]
        m = m.view(np.ndarray).reshape(shape_ms)
        s = s.view(np.ndarray).reshape(shape_ms)
        del shape_ms
        # Compute final z-scores
        return (x - m) / s

    # Absolute voltage peaks
    try:
        abs_comp0 = np.abs(comp0)
    except TypeError:
        raise _chk_parameters(comp0=comp0)

    v = zscore(abs_comp0)

    # Time derivative of voltage
    aux_shape = list(comp0.shape)
    aux_shape[_t_dim] = 1
    dvdt0 = zscore(np.abs(np.diff(comp0, axis=_t_dim)))
    # Trailing zeros are added to make dvdt.shape == v.shape
    dvdt = np.concatenate((np.zeros(aux_shape), dvdt0), _t_dim)

    # comp0 dimensionality has to be checked explicitly, as ND with N>3 runs
    # unnoticed through the function
    if comp0.ndim > 3:
        raise _chk_parameters(comp0=comp0)

    # Return features
    return v, dvdt


def _integrate(integrator_width, *args, **kwargs):
    """
    Smooth features response.

    Parameters
    ----------
    integrator_width : int > 0
        Width (in number of samples) of the integration
    feats : {keyworded: tuple of numpy.ndarray, non-keyworded: numpy.ndarray}
        Features to be integrated can be passed as non-keyworded arguments in
        the shape of arrays or as a keyworded argument *feats* in the shape of
        a tuple of arrays.
        All the arrays must have equal shape CxTxE, with C the number of
        components, T the number of time samples and E the number of recorded
        events.

    Returns
    -------
    *res : arrays
        Arrays (as many as passed to the function) containing the integrated
        features
    """

    if len(args) > 0:
        feats = args
        if 'feats' in kwargs:
            raise KeyError('_integrate() got multiple feature definitions. '
                           'Features must be passed as non-keyworded arguments'
                           ' OR as a list under the keyword "feats".')
        if len(kwargs) is not 0:
            raise KeyError('_integrate() got unexpected keyword arguments {}'
                           .format(kwargs.keys()))
    elif 'feats' in kwargs:
        if not isinstance(kwargs['feats'], tuple):
            raise TypeError('feats keyword must be {}; is {} instead'
                            .format(type(tuple()), type(kwargs['feats'])))
        feats = kwargs['feats']
        if len(kwargs) is not 1:
            keys = list(kwargs.keys()).remove('feats')
            raise KeyError('_integrate() got unexpected keyword arguments {}'
                           .format(keys))
    else:
        raise KeyError('No features parameters entered. At least one feature '
                       'has to be specified.')

    # integrator_width has to be checked explicitly, since np.hanning doesn't
    # raise an exception for non-int values.
    if not isinstance(integrator_width, int):
        # Generate the error from _chk_parameters for consistency
        raise _chk_parameters(integrator_width=integrator_width)

    # Allocate integrated features
    integrated_feats = []

    # Allocate the integrator as a Hanning window
    integrator = np.hanning(integrator_width)

    # Allocate an corrector to circumvent border effects
    win_ones = np.ones(feats[0].shape[_t_dim])
    try:
        corrector = np.convolve(win_ones, integrator, 'same')
    except ValueError:
        raise _chk_parameters(integrator_width=integrator_width)

    # Integrate each feature individually
    fcn = lambda x: np.convolve(x, integrator, 'same')/corrector
    try:
        for feat_i in feats:
            # Integrate
            integrated_feats.append(
                np.apply_along_axis(fcn, axis=_t_dim, arr=feat_i)
            )
            # features dimensionality has to be checked explicitly, as ND
            # values with N != 3 run unnoticed in the code
            if np.ndim(feat_i) != 3:
                raise ValueError('features must be 3D; '
                                 'some are {}D instead'
                                 .format(np.ndim(feat_i)))
            # Some debugging plots
            # n = np.linspace(0, 1, feat_i.shape[_t_dim])
            # plt.plot(n, integrated_feats[-1][0, :, 0], n, feat_i[0, :, 0])
            # plt.show()
    except ValueError:  # Bad feature
        if not isinstance(feat_i, np.ndarray):
            raise TypeError('features must be {}; some are {} instead'
                            .format(type(np.zeros(0)), type(feat_i)))

        if np.ndim(feat_i) != 3:
            raise ValueError('features must be 3D arrays; '
                             'some are {}D instead'.format(np.ndim(feat_i)))

    # Return a tuple instead of a list
    return tuple(integrated_feats)


def _classify(detection_th, dilator_width, *args, **kwargs):
    """
    Identifies noisy segments and builds a control signal for the mixer.

    Parameters
    ----------
    detection_th : float > 0, optional
        Detection threshold
    dilator_width : int > 0, optional
        Width (in number of samples) of the dilator
    feats : {keyworded: tuple of numpy.ndarray, non-keyworded: numpy.ndarray}
        Features to be classified can be passed as non-keyworded arguments in
        the shape of arrays or as a keyworded argument *feats* in the shape of
        a tuple of arrays.
        All the arrays must have equal shape CxTxE, with C the number of
        components, T the number of time samples and E the number of recorded
        events.

    Returns
    -------
    ctrl : array
        Control signal with shape CxTxE. Noisy segments are denoted by '1'.

    """

    if len(args) > 0:
        feats = args
        if 'feats' in kwargs:
            raise KeyError('_integrate() got multiple feature definitions. '
                           'Features must be passed as non-keyworded arguments'
                           ' OR as a list under the keyword "feats".')
        if len(kwargs) is not 0:
            raise KeyError('_integrate() got unexpected keyword arguments {}'
                           .format(kwargs.keys()))
    elif 'feats' in kwargs:
        if not isinstance(kwargs['feats'], tuple):
            raise TypeError('feats keyword must be {}; is {} instead'
                            .format(type(tuple()), type(kwargs['feats'])))
        feats = kwargs['feats']
        if len(kwargs) is not 1:
            keys = list(kwargs.keys()).remove('feats')
            raise KeyError('_integrate() got unexpected keyword arguments {}'
                           .format(keys))
    else:
        raise KeyError('No features parameters entered. At least one feature '
                       'has to be specified.')

    # detection_th has to be explicitly checked. The code doesn't raise an
    # exception with an unsupported value
    if not isinstance(detection_th, (float, int)) or detection_th <= 0:
        # Raise from _chk_parameters for uniformity
        raise _chk_parameters(detection_th=detection_th)

    # Apply the specified threshold to the features
    ctrl_signal = feats[0] > detection_th

    # An "or" operator is used to combine the result of different features
    for feat_i in feats[1:]:
        ctrl_signal |= (feat_i > detection_th)

    # Events with more than 75% of noisy time are fully labeled as noise
    try:
        tsum_ctrl_signal = ctrl_signal.sum(_t_dim, keepdims=True)
    except AttributeError:
        # Check controlled parameter errors
        for feat_i in feats:
            if not isinstance(feat_i, np.ndarray):
                raise TypeError('features must be {}; some are {} instead'
                                .format(type(np.zeros(0)), type(feat_i)))

    # At this point features are np.ndarrays. Still, they have to be checked
    # explicitly because ND arrays with N>3 are not detected by the code.
    for feat_i in feats:
        if not feat_i.ndim == 3:
            raise ValueError('features must be 3D; some are {}D instead'
                             .format(feat_i.ndim))

    rm_ev_bool = (tsum_ctrl_signal.astype(float)
                  / ctrl_signal.shape[_t_dim]) > .75

    rm_ev = np.where(rm_ev_bool)
    rm_ev_slice = [slice(None)]*ctrl_signal.ndim
    rm_ev_slice[_ev_dim] = rm_ev[_ev_dim]

    rm_ev_slice[_comp_dim] = rm_ev[_comp_dim]
    ctrl_signal[rm_ev_slice] = True
    del rm_ev_slice, rm_ev

    # Components with more than 75% of noisy time are completely labels as
    # noise
    rm_c = np.where(
        (rm_ev_bool.sum(_ev_dim, keepdims=True) /
         float(ctrl_signal.shape[_ev_dim])) > .75)
    del rm_ev_bool
    rm_ic_slice = [slice(None)]*ctrl_signal.ndim
    rm_ic_slice[_comp_dim] = rm_c[_comp_dim]
    ctrl_signal[rm_ic_slice] = True
    del rm_c, rm_ic_slice

    # Dilate the detected zones to account for the mixer transition equation
    # dilator_width has to be explicitly checked. np.ones(n) doesn't raise an
    # exception for non-int or negative values
    err = _chk_parameters(dilator_width=dilator_width)
    if err is not None:
        raise err
    dilator = np.ones(dilator_width)

    ctrl_signal = np.apply_along_axis(
        lambda x: np.convolve(x.astype(int), dilator, 'same'),
        axis=_t_dim, arr=ctrl_signal)

    # Binarize signal
    ctrl_signal[ctrl_signal > 0] = 1

    return ctrl_signal


def _mix(transition_width, comp0, ctrl_signal, comp1):
    """
    Mixes two components according to a control signal.

    Parameters
    ----------
    transition_width : int > 0, optional
        Width (in number of samples) of the transition window
    comp0 : array
        Array containing the original components, which will be analysed in
        search of noise. It must be a 3D array with shape CxTxE, where C, T and
        E are the number of components, time samples and recorded events
        respectively.
    ctrl_signal : array
        Binary control signal with values equal to 0 and 1 where *comp0* and
        comp1 are to be used respectively.
    comp1 : array, optional
        Array containing the alternative (cleaned) components. It must have
        the same shape as *comp0*. If not specified, an all 0s alternative
        components will be used (this is equivalent to component rejection).

    Returns
    -------
    comp2 : array
        Resulting mixed component with shape equal to comp0.shape.

    """

    # Check controlled parameter errors. transition_width has to be explicitly
    # checked because sp_signal.hann doesn't raise and error with non-int
    # values
    if not isinstance(transition_width, int):
        # Raise from _chk_parameters for uniformity
        raise _chk_parameters(transition_width=transition_width)
    # Allocate normalized transition window
    trans_win = sp_signal.hann(transition_width, True)
    trans_win /= trans_win.sum()

    # Pad extremes of control signal
    try:
        pad_width = [tuple([0, 0])]*ctrl_signal.ndim
    except AttributeError:
        # Check ctrl_signal parameter errors
        if not isinstance(ctrl_signal, np.ndarray):
            raise TypeError('ctrl_signal must be {}; is {} instead'
                            .format(type(np.zeros(0)), type(ctrl_signal)))

    pad_size = transition_width/2 + 1
    pad_width[_t_dim] = (pad_size, pad_size)
    # Padded control signal
    pad_ctrl = np.pad(ctrl_signal, tuple(pad_width), mode='edge')
    del pad_width

    # Combine the transition window and the control signal to build a final
    # transition-control signal, which could be applied to the components
    fcn = lambda x: np.convolve(x, trans_win, 'same')
    try:
        transition_ctrl = np.apply_along_axis(fcn, axis=_t_dim, arr=pad_ctrl)
    except ValueError:
        raise _chk_parameters(transition_width=transition_width)

    del pad_ctrl
    rm_pad_slice = [slice(None)]*ctrl_signal.ndim
    rm_pad_slice[_t_dim] = slice(pad_size, -pad_size)
    transition_ctrl = transition_ctrl[rm_pad_slice]
    # Some debugging plots
    # ctrl_signal.plot(rm_baseline=False, y_offset=2)
    # plt.figure()
    # ctrl_signal.based(transition_ctrl).plot(rm_baseline=False, y_offset=2)
    # plt.show()
    del rm_pad_slice, pad_size

    # Apply mixer
    # comp0 has to be checked explicitly, as numeric types does not rise an
    # exception
    if not isinstance(comp0, np.ndarray):
        raise _chk_parameters(comp0=comp0)
    try:
        mix_data = comp0*(1 - transition_ctrl)
    except ValueError:
        if ctrl_signal.shape != comp0.shape:
            raise ValueError('ctrl_signal and components must have equal '
                             'shape.')

    # If comp1 is not specified, return as it is
    if comp1 is None:
        return mix_data

    # ... else, mix the signal also with comp1.
    # comp1 has to be checked explicitly, as numeric types does not rise an
    # exception
    if not isinstance(comp1, np.ndarray):
        raise _chk_parameters(comp1=comp1)
    try:
        return mix_data + comp1*transition_ctrl
    except ValueError:
        raise _chk_parameters(comp0=comp0, comp1=comp1)
