=======================================
eeglcf: Readme
=======================================

This package implements the Localized Component Filtering (LCF) method for EEG
artifact rejection.

------------
Code Example
------------

Let *b_data* be a numpy.ndarray containing the result of applying a BSS method
to EEG data, and *a_data* be an alternative "cleaner" version of the previous.
Let these variable have dimensions CxTxE, where C, T and E are the number of
channels, time samples and events respectively. Then, LCF can be applied as:

.. code-block:: python

  import eeglcf
  c_data = eeglcf.lcf(b_data, a_data)

where *c_data* is a new version of the BSS data built from the mixing of
*b_data* and *a_data*.


------------
Installation
------------

To install this package, you can use the make file. From the root directory of
the package, run:

.. code-block:: bash

  make install

.. note::

  The installation of the dependencies NumPy_ and SciPy_ may fail. It
  is recommended to install these packages manually.

-----
Tests
-----

To test the package against your installed python version, from the root
directory of the package you can run:

.. code-block:: bash

  make test

-------------------
Issues and comments
-------------------

Please, `file an issue`_ if you encounter any problem with the package or if
you have any suggestions.

=======
License
=======

The eeglcf framework is open-sourced software licensed
under the `MIT license <http://opensource.org/licenses/MIT>`_.

.. _NumPy: http://www.numpy.org/
.. _SciPy: http://www.scipy.org/
.. _file an issue: https://github.com/mdelpozobanos/eeglcf/issues