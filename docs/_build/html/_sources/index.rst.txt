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

You may have to install Python_ or some modules before installing
ORBS. You will find here informations on how to install ORBS on
Ubuntu.

.. toctree::
   :maxdepth: 2

   installing_orbs

  

Reduction Guide
---------------

You will find here what you need to know to reduce your data. This is
also certainly the first place to look if you experience any problem
using ORBS.

.. toctree::
   :maxdepth: 2

   reduction_guide
   reduction_faq
   reduction_errors


Code Documentation
------------------

The code documentation can help you understand how the whole reduction
process works in details.

.. toctree::
   :maxdepth: 2

   orbs_module
   process_module


Changelog
---------

.. toctree::
   :maxdepth: 2

   changelog

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
