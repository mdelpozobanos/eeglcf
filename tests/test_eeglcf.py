#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_eeglcf
----------------------------------

Tests for `eeglcf` module.
"""

import pytest
import numpy as np
from eeglcf import eeglcf

# =============================================================================
# VARIABLES

@pytest.fixture(scope='module')
def comps():
    """Randomly generate components"""

    # Some direct accessed for readability
    urnd = np.random.uniform
    nrnd = np.random.normal

    # A controlled experiment will be created with C components containing L
    # time samples and E events.
    C = 2
    L = 1000
    E = 10

    # Time vector
    t = np.linspace(0, 10, E*L)

    # Generator: Original components
    comp_fcn = lambda n: urnd(-1, 1, 1)*np.cos(urnd(0, 1, 1)*n) + \
                         urnd(-1, 1, 1)*np.sin(urnd(0, 1, 1)*n)
    # Generator: Added noise
    noise_fcn = lambda n: np.abs(nrnd(1, 0.1, len(n))**4) * \
                          np.cos(urnd(0, 0.5, 1)*n)

    # Generate components
    xcomps = []
    for c_n in range(C):
        xcomps.append(comp_fcn(t) + noise_fcn(t))

    # Merge into an array
    xcomps = np.array(xcomps)  # Noisy components (stat point)
    # Reshape components
    return xcomps.reshape([C, L, E])


@pytest.fixture(scope='module')
def feats(comps):
    return eeglcf._features(comps)


@pytest.fixture(scope='module')
def ifeats(feats):
    """Integrated features"""
    return eeglcf._integrate(10, *feats)


@pytest.fixture(scope='module')
def ctrl(ifeats):
    """Control signal"""
    return eeglcf._classify(1., 10, *ifeats)


class TestChkParameters:
    """Test _chk_parameters function"""

    def test_comp0(self):
        """*comp0* must be a 3D array"""
        _test_parameters(eeglcf._chk_parameters, 'comp0')

    def test_comp1(self):
        """*comp1* must be a 3D array"""
        _test_parameters(eeglcf._chk_parameters, 'comp1')

    def test_comps(self):
        """*comp0* and *comp1* must have equal shape"""
        with pytest.raises(ValueError):
            raise eeglcf._chk_parameters(comp0=np.zeros([10, 10]),
                                         comp1=np.zeros(5))

    def test_integrator_width(self):
        """*integrator_width* must be an integer > 0"""
        _test_parameters(eeglcf._chk_parameters, 'integrator_width')

    def test_detection_th(self):
        """*detection_th* must be a float > 0"""
        _test_parameters(eeglcf._chk_parameters, 'detection_th')

    def test_dilator_width(self):
        """*dilator_width* must be an integer > 0"""
        _test_parameters(eeglcf._chk_parameters, 'dilator_width')

    def transition_width(self):
        """*transition_width* must be an integer > 0"""
        _test_parameters(eeglcf._chk_parameters, 'transition_width')


class TestFeatures:
    """Test _features function"""

    def test_main(self, comps):
        """Functional test"""
        res = eeglcf._features(comps)
        # All results must have the same shape as comps
        for res_n in res:
            assert(res_n.shape == comps.shape)

    def test_comp0(self):
        """*comp0* must be 3D numpy.ndarray"""
        _test_parameters(eeglcf._features, 'comp0')


class TestIntegrate:
    """Test _integrate function"""

    def test_main(self, comps, feats):
        """Functional test"""
        eeglcf._integrate(10, *feats)
        res = eeglcf._integrate(10, feats=feats)
        # All results must have the same shape as comps
        for res_n in res:
            assert(res_n.shape == comps.shape)

    def test_integrator_width(self):
        """*integrator_width* must be an integer > 0"""
        _test_parameters(eeglcf._integrate, 'integrator_width',
                         feats=(np.zeros([10]*3), ))

    def test_feats(self):
        """Features must be arrays"""
        # Specified as non-keyworded arguments
        with pytest.raises(TypeError):
            eeglcf._integrate(10, np.zeros([10]*3), range(5))
        with pytest.raises(TypeError):
            eeglcf._integrate(10, np.zeros([10]*3), 'error')
        with pytest.raises(ValueError):
            eeglcf._integrate(10, np.zeros([10]*3), np.zeros([10]*2))
        with pytest.raises(ValueError):
            eeglcf._integrate(10, np.zeros([10]*3), np.zeros([10]*4))
        # Specified as keyworded a argument
        with pytest.raises(TypeError):
            eeglcf._integrate(10, feats=np.zeros([10]*3))
        # Multiple definition of features
        with pytest.raises(KeyError):
            eeglcf._integrate(10, np.zeros([10]*3), feats=[np.zeros([10]*3), ])

    def test_kwargs(self):
        """Unexpected keywords"""
        with pytest.raises(KeyError):
            eeglcf._integrate(10, np.zeros([10]*3), error=0)
        with pytest.raises(KeyError):
            eeglcf._integrate(10, feats=(np.zeros([10]*3), ), error=0)
        with pytest.raises(KeyError):
            eeglcf._integrate(10)


class TestClassify:
    """Checks _classify function"""

    def test_main(self, ifeats):
        """Functional test"""
        res = eeglcf._classify(1., 10, *ifeats)
        # Results must have the same shape as comps
        assert(res.shape == ifeats[0].shape)

    def test_detection_th(self):
        """*detection_th* must be a float > 0 (int allowed)"""
        _test_parameters(eeglcf._classify, 'detection_th',
                         dilator_width=10, feats=(np.zeros([10]*3), ))

    def test_dilator_width(self):
        """*dilator_width* must be an int > 0"""
        _test_parameters(eeglcf._classify, 'dilator_width',
                         detection_th=1, feats=(np.zeros([10]*3), ))

    def test_features(self):
        """features must be arrays"""
        # Specified as non-keyworded arguments
        with pytest.raises(KeyError):
            eeglcf._classify(1, 10)
        with pytest.raises(TypeError):
            eeglcf._classify(1, 10, [0, 1, 2])
        with pytest.raises(TypeError):
            eeglcf._classify(1, 10, 'error')
        with pytest.raises(ValueError):
            eeglcf._classify(1, 10, np.zeros([2]*4))
        with pytest.raises(ValueError):
            eeglcf._classify(1, 10, np.zeros([2]*2))
        # Specified as keyworded a argument
        with pytest.raises(TypeError):
            eeglcf._classify(1, 10, feats=np.zeros([10]*3))
        # Multiple definition of features
        with pytest.raises(KeyError):
            eeglcf._classify(1, 10, np.zeros([10]*3),
                             feats=[np.zeros([10]*3), ])

    def test_kwargs(self):
        """Unexpected keywords"""
        with pytest.raises(KeyError):
            eeglcf._classify(1, 10, np.zeros([10]*3), error=0)
        with pytest.raises(KeyError):
            eeglcf._classify(1, 10, feats=(np.zeros([10]*3), ), error=0)
        with pytest.raises(KeyError):
            eeglcf._classify(1, 10)


class TestMix:
    """Tests _mix function"""

    def test_core(self, comps, ctrl):
        """Functional test"""
        eeglcf._mix(10, comps, ctrl, comps)

    def test_core_no_comp1(self, comps, ctrl):
        """Mixing without *comp1*"""
        eeglcf._mix(10, comps, ctrl, None)

    def test_transition_width(self, comps, ctrl):
        """*transition_width* must be an int > 0"""
        _test_parameters(eeglcf._mix, 'transition_width',
                         comp0=comps, ctrl_signal=ctrl, comp1=comps)

    def test_comp0(self, comps, ctrl):
        """*comp0* must be a 3D array"""
        _test_parameters(eeglcf._mix, 'comp0',
                         transition_width=10, ctrl_signal=ctrl, comp1=comps)

    def test_comp1(self, comps, ctrl):
        """*comp1* must be a 3D array"""
        _test_parameters(eeglcf._mix, 'comp1',
                         transition_width=10, ctrl_signal=ctrl, comp0=comps)

    def test_comps(self, comps, ctrl):
        """comp0 and comp1 must have equal shape"""
        with pytest.raises(ValueError):
            eeglcf._mix(10, comps, ctrl, comps[:, 0:2])

    def test_ctrl(self, comps, ctrl):
        """ctrl must be an array with the same shape as the components"""
        with pytest.raises(TypeError):
            eeglcf._mix(10, comps, 'error', comps)
        with pytest.raises(TypeError):
            eeglcf._mix(10, comps, [0, 1, 2], comps)
        with pytest.raises(ValueError):
            eeglcf._mix(10, comps, np.zeros([4]*2), comps)
        with pytest.raises(ValueError):
            eeglcf._mix(10, comps, np.zeros([4]*4), comps)
        with pytest.raises(ValueError):
            eeglcf._mix(10, comps, ctrl[:, 0:2], comps)


class TestLCF:
    """Tests the lcf function"""

    def test_main(self, comps):
        """Tests the usage of lcf"""
        eeglcf.lcf(comps)

    def test_comp0(self):
        """*comp0* must be 3D numpy.ndarray"""
        _test_parameters(eeglcf.lcf, 'comp0')

    def test_comp1(self, comps):
        """*comp1* must be 3D numpy.ndarray"""
        _test_parameters(eeglcf.lcf, 'comp1', comp0=comps)

    def test_comps(self, comps):
        """*comp0* and *comp1* must have equal shape"""
        with pytest.raises(ValueError):
            eeglcf.lcf(comps, comp1=comps[:, 0:2])

    def test_integrator_width(self, comps):
        """*integrator_width* must be an integer > 0"""
        _test_parameters(eeglcf.lcf, 'integrator_width',
                         comp1=comps, comp0=comps)

    def test_detection_th(self, comps):
        """*detection_th* must be a float > 0"""
        _test_parameters(eeglcf.lcf, 'detection_th',
                         comp1=comps, comp0=comps)

    def test_dilator_width(self, comps):
        """*dilator_width* must be an integer >= 0"""
        _test_parameters(eeglcf.lcf, 'dilator_width',
                         comp1=comps, comp0=comps)

    def transition_width(self, comps):
        """*transition_width* must be an integer > 0"""
        _test_parameters(eeglcf.lcf, 'transition_width',
                         comp1=comps, comp0=comps)


def _test_parameters(fcn, key, *args, **kwargs):
    """
    Tests function parameters.

    This function asserts that a wrongly specified input parameter is deceted
    by a function

    Parameters
    ----------
    fcn : functions
        Function to be called
    key : str
        Name of the parameter to be tested
    *args : tuple
        Non-keyword arguments passed to the function, other than "key"
    **kwargs : dict
        Keyword arguments passed to the function, other than "key"
    """

    if key is 'comp0':
        with pytest.raises(TypeError):
            res = fcn(comp0=0., *args, **kwargs)
            if fcn is eeglcf._chk_parameters and res is not None:
                raise res
        with pytest.raises(TypeError):
            res = fcn(comp0='error', *args, **kwargs)
            if fcn is eeglcf._chk_parameters and res is not None:
                raise res
        with pytest.raises(TypeError):
            res = fcn(comp0=[0, 10], *args, **kwargs)
            if fcn is eeglcf._chk_parameters and res is not None:
                raise res
        with pytest.raises(ValueError):
            res = fcn(comp0=np.zeros([4, 5]), *args, **kwargs)
            if fcn is eeglcf._chk_parameters and res is not None:
                raise res
        with pytest.raises(ValueError):
            res = fcn(comp0=np.zeros([4, 5, 6, 7]), *args, **kwargs)
            if fcn is eeglcf._chk_parameters and res is not None:
                raise res

    elif key is 'comp1':
        with pytest.raises(TypeError):
            res = fcn(comp1=0., *args, **kwargs)
            if fcn is eeglcf._chk_parameters and res is not None:
                raise res
        with pytest.raises(TypeError):
            res = fcn(comp1='error', *args, **kwargs)
            if fcn is eeglcf._chk_parameters and res is not None:
                raise res
        with pytest.raises(TypeError):
            res = fcn(comp1=[0, 10], *args, **kwargs)
            if fcn is eeglcf._chk_parameters and res is not None:
                raise res
        with pytest.raises(ValueError):
            res = fcn(comp1=np.zeros([4, 5]), *args, **kwargs)
            if fcn is eeglcf._chk_parameters and res is not None:
                raise res
        with pytest.raises(ValueError):
            res = fcn(comp1=np.zeros([4, 5, 6, 7]), *args, **kwargs)
            if fcn is eeglcf._chk_parameters and res is not None:
                raise res

    elif key is 'integrator_width':
        with pytest.raises(TypeError):
            res = fcn(integrator_width=1., *args, **kwargs)
            if fcn is eeglcf._chk_parameters and res is not None:
                raise res
        with pytest.raises(TypeError):
            res = fcn(integrator_width='error', *args, **kwargs)
            if fcn is eeglcf._chk_parameters and res is not None:
                raise res
        # [NOTE] A integrator_width=0 is now supported
        # with pytest.raises(ValueError):
        #     res = fcn(integrator_width=0, *args, **kwargs)
        #     if fcn is eeglcf._chk_parameters and res is not None:
        #         raise res
        # [END NOTE]
        with pytest.raises(ValueError):
            res = fcn(integrator_width=-2, *args, **kwargs)
            if fcn is eeglcf._chk_parameters and res is not None:
                raise res

    elif key is 'detection_th':
        # int values are also allowed
        res = fcn(detection_th=1, *args, **kwargs)
        if fcn is eeglcf._chk_parameters and res is not None:
            raise res
        with pytest.raises(TypeError):
            res = fcn(detection_th='error', *args, **kwargs)
            if fcn is eeglcf._chk_parameters and res is not None:
                raise res
        with pytest.raises(TypeError):
            res = fcn(detection_th=[1], *args, **kwargs)
            if fcn is eeglcf._chk_parameters and res is not None:
                raise res
        with pytest.raises(ValueError):
            res = fcn(detection_th=0., *args, **kwargs)
            if fcn is eeglcf._chk_parameters and res is not None:
                raise res

    elif key is 'dilator_width':
        with pytest.raises(TypeError):
            res = fcn(dilator_width=10.3, *args, **kwargs)
            if fcn is eeglcf._chk_parameters and res is not None:
                raise res
        with pytest.raises(TypeError):
            res = fcn(dilator_width=[10], *args, **kwargs)
            if fcn is eeglcf._chk_parameters and res is not None:
                raise res
        with pytest.raises(TypeError):
            res = fcn(dilator_width='error', *args, **kwargs)
            if fcn is eeglcf._chk_parameters and res is not None:
                raise res
        with pytest.raises(ValueError):
            res = fcn(dilator_width=0, *args, **kwargs)
            if fcn is eeglcf._chk_parameters and res is not None:
                raise res

    elif key is 'transition_width':
        with pytest.raises(TypeError):
            res = fcn(transition_width=10., *args, **kwargs)
            if fcn is eeglcf._chk_parameters and res is not None:
                raise res
        with pytest.raises(TypeError):
            res = fcn(transition_width='error', *args, **kwargs)
            if fcn is eeglcf._chk_parameters and res is not None:
                raise res
        with pytest.raises(ValueError):
            res = fcn(transition_width=0, *args, **kwargs)
            if fcn is eeglcf._chk_parameters and res is not None:
                raise res
