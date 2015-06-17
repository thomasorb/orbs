.. Orbs documentation master file, created by
   sphinx-quickstart on Sat May 26 01:02:35 2012.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


ORBS Documentation
##################

.. image:: orbs.png
   :width: 25%
   :align: center

.. topic:: Welcome to ORBS documentation !
   
  **ORBS** (*Outil de Réduction Binoculaire pour* SpIOMM_ *et* SITELLE_) is a data reduction software created to process data obtained with SpIOMM_ and SITELLE_.

  .. _SpIOMM: 

  **SpIOMM** (*Spectromètre Imageur de l'Observatoire du Mont Mégantic*) is an astronomical instrument operating at Mont Mégantic_ (Québec, CANADA) designed to obtain the visible spectra of all the objects in a 12 arc minutes field of view.

  .. _SITELLE: 

  **SITELLE** (Spectromètre-Imageur pour l’Étude en Long et en Large des raie d’Émissions) is a larger version of SpIOMM operating at the CFHT_ (Canada-France-Hawaii Telescope, Hawaii, USA).

Table of contents
-----------------

.. contents::

Installation
------------

You may have to install Python_ or some modules before installing
ORBS. You will find here informations on how to install ORBS on
Ubuntu.

.. toctree::
   :maxdepth: 2

   installing_orbs
   installing_python


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


.. _Mégantic: http://omm.craq-astro.ca/
.. _CFHT: http://www.cfht.hawaii.edu/
.. _Python: http://www.python.org/
.. _Scipy: http://www.scipy.org/
.. _Numpy: http://numpy.scipy.org/
.. _PyFITS: http://www.stsci.edu/resources/software_hardware/pyfits
.. _Parallel: http://www.parallelpython.com/
