.. Orbs documentation master file, created by
   sphinx-quickstart on Sat May 26 01:02:35 2012.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


ORBS Documentation
##################

.. image:: images/logo.*
   :width: 35%
   :align: center

.. topic:: Welcome to ORBS documentation !
   
  **ORBS** (*Outil de Réduction Binoculaire pour* SITELLE_) is a data reduction software created to process data obtained with SITELLE_. Is it the reduction software used by the CFHT_.


.. contents::

Installation
------------

Installation instructions can be found on Github:

https://github.com/thomasorb/orbs

  
Reduction Guide
---------------

You will find here useful examples describing the whole reduction
process and giving hints on how to handle and check the outputs.

.. toctree::
   :maxdepth: 2
	      
   calibration_laser_map.ipynb
   standard_spectrum.ipynb
   science_cube.ipynb
   

Old Reduction Guide (outdated)
------------------------------

This is the old reduction guide. Some infos are oudated but most of it
is still useful to understand the underlying core concepts.

.. toctree::
   :maxdepth: 2

   reduction_guide
   reduction_faq


Code Documentation
------------------

The code documentation can help you understand how the whole reduction
process works in details.

.. toctree::
   :maxdepth: 2

   core_module
   orbs_module
   process_module
   report_module
   phase_module
   utils_module


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. _CFHT: http://www.cfht.hawaii.edu/
.. _Python: http://www.python.org/
.. _ORB: https://github.com/thomasorb/orb
.. _ORCS: https://github.com/thomasorb/orcs
.. _SITELLE: http://www.cfht.hawaii.edu/Instruments/Sitelle
