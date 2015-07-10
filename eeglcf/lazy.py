# -*- coding: utf-8 -*-

"""
This module contains sample functions for some of the steps defined in
:ref:`step_by_step`

.. _FastICA: http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FastICA.html

"""

import copy
from eeglcf import lcf
import numpy as np
import scipy.signal as sp_signal
from eeglcf import mixnp
from sklearn.decomposition import FastICA

# Dimension variables are defined globally to facilitate the interpretation of
# the code
ch_dim = 0
t_dim = 1
ev_dim = 2


def noisy_ch(eeg_data, ref_ch=None):
    """
    Identify noisy channels based on the following scores:

    1. **Correlation score (r)**: Clean EEG signals are highly correlation
       across channels. Therefore, a noise descriptor for a channels might be a
       low average correlation with other channels.

    2. **Variance score (v)**: Extremely high event-average time-variances are
       also indicators of noise.

    3. **Hurst score (h)**: Clean EEG signals has very specific Hurst exponent
       values. Deviations from this point are also noise indicators.

    Outliers are labeled as noisy channels. They are detected based on the
    z-score of each of the above measurements (z-score > 3).

    Parameters
    ----------
    eeg_data : array
        EEG data in a CxTxE array. With C the number of channels, T the number
        of time samples and E the number of events.
    ref_ch : int, None
        Index of the reference channel.

    Returns
    -------
    noisy_ch : list
        List with the index of the channel identified as noisy.
    """
    # TODO: Distance to reference channel
    # When the reference channel is specified in the calling instance, the
    # first two scores are corrected for the quadratic effect of the distance
    # to this reference channel.

    # Dimension shapes
    ch_len = eeg_data.shape[ch_dim]
    t_len = eeg_data.shape[t_dim]
    ev_len = eeg_data.shape[ev_dim]

    # -------------------------------------------------------------------------
    # 1. Compute channel scores

    # 1.1 Average time correlation:
    # We need to collapse time and events dimensions
    coll_eeg = eeg_data.transpose([t_dim, ev_dim, ch_dim])\
        .reshape([t_len*ev_len, ch_len])
    # coll_eeg.shape = (time&events)x(channels)
    r_coef = np.abs(mixnp.pearson(coll_eeg).mean(axis=1))

    # 1.2 Time variance
    v_coef = coll_eeg.var(axis=0)

    # 1.3 Hurst exponent
    h_coef = np.zeros(ch_len)
    for ch_n in range(ch_len):
        h_coef[ch_n] = mixnp.hurst(coll_eeg[:, ch_n])
    del coll_eeg

    # Remove quadratic effect of distance to reference channel
    if ref_ch is not None:
        # Compute distance to the reference data to correct the scores
        dist_to_ref = np.squeeze(eeg_data.ch_dist(ch=ref_ch))

        # Correlation score
        poly_coef = np.polyfit(dist_to_ref, r_coef, 2)
        r_coef -= np.polyval(poly_coef, dist_to_ref)
        del poly_coef

        # Variance score
        poly_coef = np.polyfit(dist_to_ref, v_coef, 2)
        v_coef -= np.polyval(poly_coef, dist_to_ref)

    # -------------------------------------------------------------------------
    # Classify channels

    # Detect outliers
    r_outliers = mixnp.zscore_outliers(r_coef)[0]
    v_outliers = mixnp.zscore_outliers(v_coef)[0]
    h_outliers = mixnp.zscore_outliers(h_coef)[0]

    # Use an OR operator to compute the final result
    return np.where(r_outliers | v_outliers | h_outliers)[0]


def noisy_ev(eeg_data):
    """
    Indentify noisy events based on the scores:

    1. **Range score (r)**: Clean EEG signals have a voltage range within some
       specific limits. Ranges off these limits are a signal of noise presence.
    2. **Deviation score (d)**: High deviations of voltage from the average
       value are also noise indicators.
    3. **Variance score (v)**:  Extremely high voltage variances are also
       indicators of noise.

    Parameters
    ----------
    eeg_data : array
        EEG data in a CxTxE array. With C the number of channels, T the number
        of time samples and E the number of events.

    Returns
    -------
    coefs : dict
        Dictionary with the resulting scores arrays 'r', 'd' and 'v', each of
        length #events.
    """

    # -------------------------------------------------------------------------
    # 1. Compute trial scores

    # 1.1 Channel-average voltage range
    r_coef = (eeg_data.max(axis=t_dim) - eeg_data.min(axis=t_dim))\
        .mean(axis=ch_dim)

    # 1.2 Channel-average voltage deviation for mean voltage
    d_coef = np.squeeze((eeg_data.mean(axis=t_dim, keepdims=True) -
                         eeg_data.mean(axis=(t_dim, ev_dim), keepdims=True)
                         ).mean(axis=ch_dim))

    # 1.3 Voltage variance
    v_coef = eeg_data.var(axis=t_dim).mean(axis=ch_dim)

    # -------------------------------------------------------------------------
    # Classify trials

    # Detect outliers
    r_outliers = mixnp.zscore_outliers(r_coef)[0]
    d_outliers = mixnp.zscore_outliers(d_coef)[0]
    v_outliers = mixnp.zscore_outliers(v_coef)[0]

    # Use an OR operator to compute the final result
    return np.where(r_outliers | d_outliers | v_outliers)


def fastica(eeg_data):
    """
    Sample function to apply `FastICA`_ to the EEG data.

    Parameters
    ----------
    eeg_data : array
        EEG data in a CxTxE array. With C the number of channels, T the number
        of time samples and E the number of events.

    Returns
    -------
    ica : ICA object
        Trained `FastICA`_ object.
    ica_data : array
        EEG projected data in a CxTxE array. With C the number of components, T
        the number of time samples and E the number of events.
    """

    # Dimension shapes
    ch_len = eeg_data.shape[ch_dim]
    t_len = eeg_data.shape[t_dim]
    ev_len = eeg_data.shape[ev_dim]

    # -------------------------------------------------------------------------
    # 1. Fit the FastICA model

    # We need to collapse time and events dimensions
    coll_data = eeg_data.transpose([t_dim, ev_dim, ch_dim])\
        .reshape([t_len*ev_len, ch_len])

    # Fit model
    ica = FastICA()
    ica.fit(coll_data)

    # Normalize ICs to unit norm
    k = np.linalg.norm(ica.mixing_, axis=0)  # Frobenius norm
    ica.mixing_ /= k
    ica.components_[:] = (ica.components_.T * k).T

    # -------------------------------------------------------------------------
    # 2. Transform data

    # Project data
    bss_data = ica.transform(coll_data)

    # Adjust shape and dimensions back to "eeg_data" shape
    ic_len = bss_data.shape[1]
    bss_data = np.reshape(bss_data, [ev_len, t_len, ic_len])
    new_order = [0, 0, 0]
    # TODO: Check the following order
    new_order[ev_dim] = 0
    new_order[ch_dim] = 2
    new_order[t_dim] = 1
    bss_data = bss_data.transpose(new_order)

    # End
    return ica, bss_data


def inv_fastica(ica, ica_data):
    """
    Sample function to apply the inverse transform of `FastICA`_ to the
    projected data.

    Parameters
    ----------
    ica : ICA object
        Trained `FastICA`_ object.
    ica_data : array
        EEG projected data in a CxTxE array. With C the number of components, T
        the number of time samples and E the number of events.

    Returns
    -------
    eeg_data : array
        Back-projected EEG data in a CxTxE array. With C the number of
        channels, T the number of time samples and E the number of events.
    """

    # Dimension shapes
    ic_len = ica_data.shape[ch_dim]
    t_len = ica_data.shape[t_dim]
    ev_len = ica_data.shape[ev_dim]

    # We need to collapse time and events dimensions
    coll_data = ica_data.transpose([t_dim, ev_dim, ch_dim])\
        .reshape([t_len*ev_len, ic_len])

    # Project back to the EEG space
    eeg_data = ica.inverse_transform(coll_data)

    # Adjust shape and dimensions
    ch_len = eeg_data.shape[1]
    eeg_data = np.reshape(eeg_data, [ev_len, t_len, ch_len])
    new_order = [0, 0, 0]
    new_order[ev_dim] = 0
    new_order[ch_dim] = 2
    new_order[t_dim] = 1
    eeg_data = eeg_data.transpose(new_order)

    # End
    return eeg_data


def artifact_rejection_with_lcf(eeg_data, fs,
            freq=(0.5, None),
            ref_ch=None, eeg_ch=None):
    """
    Example function covering the steps defined in :ref:`step_by_step`.

    1. **Filter the data**: FIR filters are applied within the specified range
       (*freq*).

    2. **Check reference channel**: Re-reference the data to the specified
       channel (*ref_ch*).

    3. **Remove the baseline**: Baseline is computed using function
       :func:`eeglcf.mixnp.zscore_outliers`, which ignores outliers.

    4. **Reject noisy EEG channels**: Channel rejected using sample function
       :func:`eeglcf.lazy.noisy_ch`.

    5. **Reject noisy EEG trials**: Trial rejected using sample function
       :func:`eeglcf.lazy.noisy_ev`.

    6. **Apply the BSS algorithm**: `FastICA`_ algorithm is applied as the BSS
       technique, via the sample function :func:`eeglcf.lazy.fastica`.

    7. **Identify artifactual components**: In this case, we do not apply this
       step. Hence, the architecture corresponds to the application of the LCF
       method by itself.

    8. **Process artifactual components**: All 0s components are used as
       alternative components.

    9. **Apply LCF**: This is:

       .. code-block:: python

         c_data = eeglcf.lcf(b_data, a_data)

       where *b_data* is a numpy.ndarray containing the result of applying a BSS method
       to EEG data, and *a_data* is an alternative "cleaner" version of the previous.
       Both variable have dimensions CxTxE, where C, T and E are the number of
       channels, time samples and events respectively.

    10. **Apply the BSS^{-1} algorithm**: Back-project `FastICA`_ via the
        sample function :func:`eeglcf.lazy.inv_fastica`.

    11. **Remove baseline**. Baseline is computed using function
       :func:`eeglcf.mixnp.zscore_outliers`, which ignores outliers.

    12. **Reject channels inside events**: Not coded.

    13. **Interpolate channels**: Not coded.

    Parameters
    ----------
    eeg_data : array
        EEG data.
    fs : float
        EEG sampling frequency.
    freq : tuple
        Tuple with the lower and higher cut-off frequencies respectively. If
        freq[0] is None, no high-pass filter is applied. If freq[1] is None, no
        low-pass filter is applied.
    ref_ch : int, None
        Index of the EEG reference channel. If None, the EEG will be
        re-referenced to channel 0.
    eeg_ch : list, None
        List of the channel indices corresponding to EEG signals (i.e.
        excluding EOG, EMG, ...). If None, all channels will be considered EEG
        channels.

    Returns
    -------
    res : array
        Clean EEG data

    """

    # Set the dimensions of eeg_data
    ch_dim = 0
    t_dim = 1

    # -------------------------------------------------------------------------
    # 1. Filter the data to remove uninteresting frequencies, which can
    #    potentially carry noise.

    if freq[0] is not None:  # High-pass filter the data
        # Design the filter. We will use a FIR filter of order 100 and a
        # Hamming window
        if freq[0] > 2:  # Set a transition band of 2 Hz.
            x_freq = [0, freq[0]/fs, (freq[0]-2)/fs, 1]
            gain = [0, 0, 1, 1]
        else:  # freq[0] <= 2  # No transition band
            x_freq = [0, freq[0]/fs, 1]
            gain = [0, 1, 1]
        filt_ba = [sp_signal.firwin2(numtaps=100, freq=x_freq, gain=gain,
                                     window='hamming', antisymmetric=True),
                   np.array([1])]
        # High-pass filtered data
        hpf_data = sp_signal.filtfilt(b=filt_ba[0], a=filt_ba[1], x=eeg_data,
                                      axis=t_dim, padtype='odd',
                                      padlen=len(filt_ba[0]))
    else:  # Do not filter
        hpf_data = eeg_data

    if freq[1] is not None:  # Low-pass filter the data
        # Design the filter. We will use a FIR filter of order 101 and a
        # Hamming window
        if freq[1]+2 < fs:  # Use a transition band of 2 Hz
            x_freq = [0, freq[1]/fs, (freq[1]+2)/fs, 1]
            gain = [1, 1, 0, 0]
        else:  # freq[1]+2 >= fs  # No transition band
            x_freq = [0, freq[1]/eeg_data.fs, 1]
            gain = [1, 1, 0]
        filt_ba = [sp_signal.firwin2(numtaps=101, freq=x_freq, gain=gain,
                                     window='hamming', antisymmetric=False),
                   np.array([1])]
        # Low-pass filtered data
        f_data = sp_signal.filtfilt(b=filt_ba[0], a=filt_ba[1], x=hpf_data,
                                    axis=t_dim, padtype='odd',
                                    padlen=len(filt_ba[0]))

    else:  # Do not filter
        f_data = hpf_data

    # -------------------------------------------------------------------------
    # 2. Make sure that the data has a reference channel.
    #
    #    Avoid common average reference before cleaning the signal, as it will
    # spread the noise throughout all channels.
    #    Some systems, such as BIOSEMI, provide signal without reference.

    if ref_ch is None:  # Re-reference the data to channel 0
        ref_ch = 0
        ref_data = f_data - f_data.take([ref_ch], axis=ch_dim)
    else:
        ref_data = f_data

    # -------------------------------------------------------------------------
    # 3. Remove the baseline

    # We compute the average rejecting outliers
    rb_data = ref_data - mixnp.zscore_outliers(ref_data, axis=t_dim, keep_dims=True)[1]

    # -------------------------------------------------------------------------
    # 4. Reject noisy EEG channels

    # Apply your preferred method for channel rejection. In this case, we will
    # use *noisy_ch* function as an example.
    if eeg_ch is None:  # All channels are EEG channels
        eeg_ch = range(rb_data.shape[ch_dim])
    sel_chs = np.array([ch_n for ch_n in range(rb_data.shape[ch_dim])
                        if (ch_n != ref_ch) and (ch_n in eeg_ch)])
    rej_ch = noisy_ch(rb_data.take(sel_chs, axis=ch_dim))
    # Adapt the list of rejected channels to consider all channels
    rej_ch = sel_chs[rej_ch]
    # Remove channels
    kept_ch = [ch_n for ch_n in range(rb_data.shape[ch_dim])
               if ch_n not in rej_ch]
    ch_data = rb_data.take(kept_ch, axis=ch_dim)

    # You should keep the list of rejected channels if you wish to interpolate
    # them back into the data later on.

    # -------------------------------------------------------------------------
    # 5. Reject noisy EEG events/trials

    # Apply your preferred method for event rejection. We will use the support
    # function *noisy_ev*.
    rej_ev = noisy_ev(ch_data)
    kept_ev = [ev_n for ev_n in range(rb_data.shape[ev_dim])
               if ev_n not in rej_ev]
    ev_data = ch_data.take(kept_ev, axis=ev_dim)

    # -------------------------------------------------------------------------
    # 6. Apply the BSS algorithm

    ica, comp = fastica(ev_data)

    # -------------------------------------------------------------------------
    # 7. Apply the artifactual component detection and processing algorithm

    # In this example, we do not apply any artifactual component detection or
    # processing algorithm. We evaluate all components with LCF using all 0s
    # alternative signals.

    pp_comp = None
    rej_comps = None

    # -------------------------------------------------------------------------
    # 8. Apply LCF

    if rej_comps is None:  # All components considered
        lcf_comp = lcf(comp, pp_comp)

    else:  # Apply LCF only to the rejected components
        rej_slice = [None]*3
        rej_slice[ch_dim] = rej_comps
        lcf_comp = copy.copy(comp)
        lcf_comp[rej_slice] = lcf(comp[rej_slice], pp_comp[rej_slice])

    # -------------------------------------------------------------------------
    # 9. Back-project to the original space

    rc_data = inv_fastica(ica, lcf_comp)

    # -------------------------------------------------------------------------
    # 10. Remove baseline

    # The baseline can be offset by the BSS processing. It is recommended to
    # re-remove the baseline
    rb_data = rc_data - mixnp.zscore_outliers(rc_data, axis=t_dim,
                                              keep_dims=True)[1]

    # -------------------------------------------------------------------------
    # 11. Reject channels inside events
    #
    #   Analyze channels within each event to reject and interpolate those that
    # are still too noisy. Note that the interpolation is also done within
    # individual events.

    pass

    # -------------------------------------------------------------------------
    # 12. Interpolate channels rejected in step 4

    pass

    return rb_data