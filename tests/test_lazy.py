#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_lazy
----------------------------------

Tests for `eeglcf.lazy` module.
"""

import pytest
import numpy as np
from eeglcf import eeglcf
import eeglcf.lazy

# =============================================================================
# VARIABLES

FS = 128  # Assumed sampling frequency

@pytest.fixture(scope='module')
def eeg_data():
    """
    Randomly generate EEG data with:

    + 32 channels
    + Sampling frequency 128 Hz
    + 1 seconds trials
    + 10 trials

    """

    # Some direct accessed for readability
    urnd = np.random.uniform
    nrnd = np.random.normal

    # A controlled experiment will be created with C channels containing L
    # time samples and E events. EEG activity will be the combination of S
    # sources
    C = 32
    L = 1*128
    E = 10
    S = 128

    # Time vector
    t = np.linspace(0, 10, E*L)

    # Generator: Original sources
    src_fcn = lambda n: urnd(-1, 1, 1)*np.cos(urnd(0, 1, 1)*n) + \
                         urnd(-1, 1, 1)*np.sin(urnd(0, 1, 1)*n)
    # Generator: Added noise
    noise_fcn = lambda n: np.abs(nrnd(1, 0.1, len(n))**4) * \
                          np.cos(urnd(0, 0.5, 1)*n)
    # Generate sources
    sources = []
    for s_n in range(S):
        sources.append(src_fcn(t))
    # Merge into an array
    sources = np.array(sources)

    # Generate the EEG activity
    xeeg_data = []
    for c_n in range(C):
        xeeg_data.append(
            (urnd(-5, 5, sources.shape[0])[:, None]*sources).sum(0)
            + noise_fcn(t)
        )
    # Merge into an array
    xeeg_data = np.array(xeeg_data)

    # Reshape components
    return xeeg_data.reshape([C, L, E])


class TestArtifactRejectionWithLCF:
    """Test artifact_rejection_with_lcf function"""

    def test_main(self, eeg_data):
        """Functional test"""
        res = eeglcf.lazy.artifact_rejection_with_lcf(eeg_data, FS)
        assert(isinstance(res, np.ndarray))
        assert(res.shape[1] == eeg_data.shape[1])