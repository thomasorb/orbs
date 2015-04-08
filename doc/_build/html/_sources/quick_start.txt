Quick start Guide
#################

.. contents::

Step 1: Create your option file
-------------------------------

The option file defines all the reduction parameters (files folders,
observation parameters ...). Just run the following into your
reduction folder to open *OptCreator* a little program which will
help you create this file ::

  orbs-optcreator

.. image:: orbs-optcreator.png
   :width: 100%

The first two columns are essentiel parameters. The last column
contains optional (but useful) parameters for data calibration.

Most of the options are self-explanatory only the bad frames option needs some information. Just see below.

Step 2: Reduce your data
------------------------

Automated reduction
~~~~~~~~~~~~~~~~~~~


Running ``orbs`` from the reduction folder starts the reduction process. You just have to enter the name of the option file to be used::

  orbs option_file.opt start

.. note:: ``orbs`` command alone prints the command usage

Important options
~~~~~~~~~~~~~~~~~

Only some of the most used options are listed here.
      
:option:`-c --calib=` Calibration file path
  
:option:`-a --apod=` Apodization function that will be used during FFT
     computation (can be barthann, bartlett, blackman, blackmanharris,
     bohman, hamming, hann, nuttall, parzen)


:option:`--step=` Starting step. Use it to recover from an error
     at a certain step without having to run the whole process one
     more time. Note that the step designation is different for the
     full reduction (2 cameras) or the single camera reduction (1
     camera).

.. note:: The steps designations for the full reduction are :

            1. Compute alignment vectors (see:
               :py:meth:`~orbs.Orbs.compute_alignment_vector`)

            2. Compute cosmic ray maps (see:
               :py:meth:`~orbs.Orbs.compute_cosmic_ray_map`)

            3. Compute interferograms (see:
               :py:meth:`~orbs.Orbs.compute_interferogram`)

            4. Transform cube B (see:
               :py:meth:`~orbs.Orbs.transform_cube_B`)

            5. Merge interferograms (see:
               :py:meth:`~orbs.Orbs.merge_interferograms`)

            6. Compute calibration map (see:
               :py:meth:`~orbs.Orbs.compute_calibration_map`)

            7. Compute phase maps (see: 
               :py:meth:`~orbs.Orbs.compute_phase_maps()`)

            8. Compute spectrum (see:
               :py:meth:`~orbs.Orbs.compute_spectrum`)

            9. Calibrate spectrum (see: 
               :py:meth:`~orbs.Orbs.calibrate_spectrum`)
            


.. note:: The steps designations for the single camera reduction are :

            1. Compute alignment vectors (see:
               :py:meth:`~orbs.Orbs.compute_alignment_vector`)

            2. Compute cosmic ray maps (see:
               :py:meth:`~orbs.Orbs.compute_cosmic_ray_map`)

            3. Compute interferogram (see:
               :py:meth:`~orbs.Orbs.compute_interferogram`)

            4. Correct interferogram (see:  
               :py:meth:`~orbs.Orbs.correct_interferogram`)

            5. Compute calibration map (see:
               :py:meth:`~orbs.Orbs.compute_calibration_map`)

            6. Compute phase maps (see: 
               :py:meth:`~orbs.Orbs.compute_phase_maps()`)

            7. Compute spectrum (see:
               :py:meth:`~orbs.Orbs.compute_spectrum`)

            8. Calibrate spectrum (see: 
               :py:meth:`~orbs.Orbs.calibrate_spectrum`)


Get full control over ORBS
~~~~~~~~~~~~~~~~~~~~~~~~~~

You can also take the full control over the reduction steps by modifying and running the script (see the file ``scripts/reduction_script.py`` for more information)::

  python reduction_script.py
