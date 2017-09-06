
Changelog
#########

	
v3.1
****

v3.1.0
======

Alignment
---------

* :py:meth:`~process.InterferogramMerger.find_alignment` has been
  completly modified. There's no more need for a list of detected
  stars in the cube B (or camera 2). Stars found in cube A are
  searched in cube B. The process is now faster and must be more
  robust.

* :py:meth:`orbs.orbs.Orbs.transform_cube_B` has also been changed to
  avoid looking for stars in the cube B.


Cosmic rays detection
---------------------

* :py:meth:`~process.RawData.create_cosmic_ray_map` remove a low order
  polynomial on each tested interferogram vector before checking for
  cosmic rays. It helps a lot in avoiding too much detected cosmic
  rays on stars.


Overwriting FITS files
----------------------

* :py:meth:`~core.Tools.write_fits` has a new option to overwrite a
  FITS file.
* 'orbs' script can be passed the option -o to enable overwriting over
  the existing FITS files.
* :py:class:`orbs.orbs.Orbs`, :py:class:`~core.Cube` and all processing
  classes have been modified to use this new option and pass it to any
  self._write_fits() function.

Transmission function
---------------------

* :py:meth:`~process.InterferogramMerger.merge` has been changed to
  eliminate the stars with not enough flux (set to 0.25 times the
  maximum flux of the detected stars)

v3.1.1
======

* :py:meth:`~process.RawData.add_missing_frames` : correction of a
  minor bug when adding zeros frames for a single cube interferogram
* :py:meth:`~process.RawData.add_missing_frames` and
  :py:meth:`~process.InterferogramMerger.add_missing_frames` have been
  corrected to add the correct header to the created zeros frames.

v3.1.2
======

* :py:meth:`~process.InterferogramMerger.find_alignment` has been
  modified to use a larger box_size to detect stars during the rough
  alignment step. A warning is printed if less than 80 % of the stars
  can be fitted after the first optimization pass and an error is
  raised if less than 50 % of the stars can be fitted.

* **orbs** script can be passed the option -p or --phase to enter the
  number of points to use for space correction. If 0 is entered no
  phase correction will be done.

v3.2
****

Dark and Bias correction for SpIOMM CAM2
========================================

* :py:meth:`~process.RawData.create_interferogram` : Dark and
  Bias are now removed using their precise temperature and a
  calibrated function. The temperature of the frames (bias frames,
  dark frames and frames to be corrected) must be recorded in the
  header of the files with the keyword CCD-TEMP.

* **orbs-tempreader** script: This script has been created to read the
  temperature files created during an observing run and write the
  temperature of the files in the headers of the frames. It must be
  used in each folder where the temperature are needed (bias, dark and
  cam2 folders)

* **config.orb** : new keywords (DARK_CALIB_PARAM_A, DARK_CALIB_PARAM_B,
  DARK_CALIB_PARAM_C, BIAS_CALIB_PARAM_A, BIAS_CALIB_PARAM_B) have
  been added for the calibration coefficients of the bias and the
  dark. They have been computed from calibrated bias and dark curves :
  I(T) = f(T).

Frames transformation
=====================

* :py:meth:`~utils.transform_frame` has been completely changed and
  optimized using scipy.ndimage fast routines for image
  transformation. The time consumption of the transformation step has
  been dramatically decreased : this process is now more than 10 times
  faster.

Orbs script
===========

* **orbs** script option -r --raw has been removed and replaced by the
  option --nostar.Using the option -s (single reduction : only the
  cam1 cube is reduced) it is now possible to reduce one or both cubes
  without stars. The alignment steps are skipped and the default
  alignment parameters are used during the merging process.


v3.3
****

Cosmic ray detection and correction
===================================

* :py:meth:`~process.RawData.create_cosmic_ray_map` is now capable of
  detecting ver faint cosmic rays without overdetecting cosmic rays in
  stars. Planes and satellites are also detected.

* Cosmic rays corrected by
  :py:meth:`~process.RawData.create_interferogram` are now replaced by
  a weighted average of the neighbourhood. Weights are defined using a
  gaussian kernel. The kernel degree (i.e. neighbourhood radius) can
  be choosen.

v3.4
****

v 3.4.0:  Phase correction
==========================

* :py:meth:`~process.InterferogramMerger.merge` has been modified to
  create a small data cube containing the interferograms of choosen
  stars. Those interferograms can be used by
  :py:meth:`~process.Interferogram.compute_spectrum` to recover the
  first order coefficient of the phase. This way the phase is not
  computed for each pixel but a general correction is made.

* :py:meth:`~process.Interferogram.compute_spectrum` has been modified
  to compute the mean first order coefficient given a cube of stars
  interferogram.

* :py:meth:`~utils.transform_frame` has been modified to compute the
  phase fo each pixel given a phase map which gives the zeroth order
  of the polynomial function of the phase for each pixel and the first
  order coefficient. Both parameters (the phase map and the first
  order coefficient) must be given to avoid a pixel by pixel phase
  computation which can be unreliable.

* :py:class:`~process.Phase`: A new class has been created to manage
  phase data cubes. Those data cubes are useful to recover the phase
  maps. Three methods have been created:
  :py:meth:`~process.Phase.create_phase_maps` which create the phase
  maps from a phase cube, :py:meth:`~process.Phase.smooth_phase_map`
  which smooth the values of the phase map of 0th order (remember that
  phase values are defined modulo PI) and
  :py:meth:`~process.Phase.fit_phase_map` which fit the created
  smoothed phase map of 0th order to remove noisy data.

* :py:meth:`orbs.orbs.Orbs.full_reduction` and
  :py:meth:`orbs.orbs.Orbs.single_reduction` use computed phase maps by
  default. An external phase map of order 0 can be given if it has
  been computed (e.g. from a flat cube).

* **orbs** script option -s replaced by the options -1 or -2 in order
    to reduce only the camera 1 cube (-1) or the camera 2 cube
    (-2). --flat option added to reduce flat cubes and obtain only
    their phase map (spectrum is not computed)


v3.4.1
======

Correct for strange phase with calibration stars
------------------------------------------------

* :py:meth:`~process.Interferogram.compute_spectrum` no longer use
  stars interferogram to recover the first order coefficient but use
  the mean of the first order phase map. The precision is far better.

Better stars interferogram at merging step
------------------------------------------
* :py:meth:`~process.InterferogramMerger.merge` compute better stars
  interferograms. :py:meth:`~utils.fit_stars_in_frame` and
  :py:meth:`~utils.fit_gaussian2d` have been modified to give better
  fitted photometry points and retry a fit if it fails because the
  stars is not centered in the box.

v3.4.2
======

Bad frames vectors management
-----------------------------

The idea is to use the different processes to detect bad frames and
collect their bad frames vectors to suppress bad frames prior to the
transformation of the interferograms.

* :py:meth:`~process.InterferogramMerger.merge` creates a bad frames
  vector using a threshold of transmission (70%).

* :py:meth:`~orbs.create_bad_frames_vector` has been created to
  collect the bad frames vector created by various processes and
  create a full bad frames vector which can be passed
  :py:meth:`~process.Interferogram.compute_spectrum` and
  :py:meth:`~utils.transform_interferogram` in order to remove all the
  detected bad frames prior to transform the interferograms.

Zeros smoothing
---------------

* :py:meth:`~utils.transform_interferogram` has been modified to do
  what we call the zeros smoothing. The objective is to reduce ringing
  due to steep transition between 'normal' points and zeros. The
  interferogram is multiplied by a function which smoothes the
  transition between zeros parts and 'good parts' of the
  interferogram. The good parts symmetrical to the zeros parts (The
  ZPD is the center of symmetry) are multiplied by 2. And the same
  transition is applied from parts multpilied by 2 to parts
  multplied by 1. This way the same weight is given to each and every
  point of the interferogram (points multiplied by zero have their
  symmetrical point multplied by 2). The degree of smoothing can be
  choosen (smoothing_deg option). A higher degree means a smoother
  transition between one part to another but may reduce the SNR.

* :py:meth:`~utils.smooth` can now smooth a vector using a gaussian
  kernel convolution (much faster).

v3.4.3
======

* Minor bugs corrections

Better fit of stars
-------------------

* :py:meth:`~utils.fit_stars_in_frame` and
  :py:meth:`~utils.fit_gaussian2d` modified to give better fitting
  results. Especially for the method
  :py:meth:`~process.InterferogramMerger.merge` which depends a lot on
  a good fit of all the detected stars.


v3.4.4
======

Various phase fit degree
------------------------

* :py:meth:`~process.Phase.create_phase_maps` and
  :py:meth:`orbs.orbs.Orbs.compute_spectrum` modified to use any order of
  the polynomial fit to the phase

* **config.orb**: New keyword PHASE_FIT_DEG to configure the desired
    degree of the polynomial fit o the
    phase. :py:meth:`orbs.orbs.Orbs.__init__` modified to use this
    keyword.

v3.4.5 (stable)
===============

* Minor modifications of :py:meth:`~process.InterferogramMerger.merge`
  to make it more stable

* PHASE_FIT_DEG in **config.orb** set to 1

This version is considered as stable.

v3.4.6
======

Enhanced phase determination
----------------------------

* :py:meth:`~utils.get_lr_phase` window changed to a NORTON_BEER 2.0
  to get a phase with much less artefacts: give a much more precise
  phase and thus much more precise phase maps.

* :py:meth:`~process.Interferogram.compute_phase_coeffs_vector` use a
  cleaner way to get the median phase coefficient for each phase map:
  used points are choosen from the residual map created by
  :py:meth:`~process.Phase.create_phase_maps` and sigma-clipped before the mean
  is taken from a well defined gaussian-like distribution of phase
  coefficients.


v3.4.7
======

Reversed spectrum corection
---------------------------

.. note:: The problem comes from the 0th order phase map which is
  defined modulo PI. An addition of PI on the phase vector (thus on
  the 0th order of the polynomial) reverses the returned spectrum.

* :py:meth:`~process.Interferogram.compute_spectrum` modified to avoid the
  spectrum to be reversed (values are negative instead of positive)
  after phase correction. Spectrum polarity is checked using a mean
  interferogram over the whole cube. If the resulting spectrum is
  reversed the whole 0th order phase map is added PI.

  

Sky transmission correction in single-camera mode
-------------------------------------------------

* :py:meth:`~process.Interferogram.create_correction_vectors` created to get
  the correction vectors (sky transmission and added light) and
  correct interferograms in single camera-mode. Now **phase
  correction** and **sky transmission correction** are available in
  single camera-mode (but less precise than in binocular mode).

.. note:: The sky transmission vector is computed from star
  photometry. Its precision is good but it must be corrected for ZPD
  because with only one camera and near the ZPD stars interferograms
  are not 'flat' anymore. The 'added light' vector is computed from a
  median 'interferogram' It has also to be corrected near the ZPD.

* minor bugs correction and enhancements


Passing alignment parameters to orbs command
--------------------------------------------

* **orbs** script: new option : **--align** to pass precomputed alignement
  parameters. Useful in the case of the computation of a FLAT cube
  (with no possible alignment) if the alignment parameters are already
  knowm from the reduction of an object taken during the same mission.


v3.4.8
======

Master combination algorithms
-----------------------------

* :py:meth:`~process.RawData._create_master_frame` created to use
  better combination algorithms for the creation of master bias, master
  dark and master flat. Some pixels are rejected using a rejection
  algorithm before the images are combined using a median or an
  average function. The rejection algorithms proposed are:

    * Min-Max rejection
    * Sigma-Clipping
    * Average Sigma-Clipping (default)
    
  Master frames are also written to the disk for checking
  purpose. Note that those rejection algorithm have been inspired by
  the IRAF function combine.

* :py:meth:`~process.RawData.detect_stars` also uses
  :py:meth:`~process.RawData._create_master_frame` to combine frames.
  
Minor modifications
-------------------

* DATA_FRAMES in :py:meth:`~process.RawData.detect_stars` changed from
  10 to 30. Help in finding more stars in some cubes.

* :py:meth:`orbs.orbs.Orbs._create_list_from_dir` now check if all the FITS
  files in the directory have the same shape.

* :py:meth:`~process.RawData.correct_frame` and
  :py:meth:`~process.RawData.create_interferogram` modified to
  correct for bias, flat and dark even if one of them are not given
  (before, without biases no correction at all would have been made)

* :py:meth:`~process.Spectrum.correct_filter` modified when filter min
  or max are outside the spectrum.

* :py:class:`orbs.orbs.Orbs.__init__` prints modules versions

* :py:class:`orbs.orbs.Orbs.__init__` modified. It is now possible to
  change configuration options for a particular reduction using the
  option file. Keywords are the same.

* :py:meth:`~process.RawData.correct_frame` modified to avoid strange
  behaviour when dark level is too low.


v3.5
****

3.5.0
=====

Alignment and photometry
-------------------------

:py:class:`~astrometry.Astrometry` class created with a whole new
astrometry module. This module is used for all astrometrical processes
(star position detection for alignment and star
photometry). Astrometry and photometry precision are now a lot better.


Merging process
---------------

* Single camera reduction: A new step of reduction has been added to
  better correct single camera interferograms for variations of
  transmission and light refracted on clouds.

* 2-camera reduction without merging frames (optional):
  :py:meth:`~process.InterferogramMerger.merge` has a better way of
  correct interferograms without merging frames. Camera 2 frames are
  used to create correction vectors but are not merged to the frames
  of the camera 1.


Cosmic Ray Detection
--------------------

:py:meth:`~process.RawData.create_cosmic_ray_map`. Completly changed
and upgraded using ORUS simulated cubes. Faster and far more
efficient. 95% of good detection over CR's with an energy higher than
the median CR energy. Small number of false detections. Less problems
with stars and ZPD.

FFT
---

'Zeros smoothing' step in :py:meth:`~utils.transform_interferogram`
modified to avoid correcting very small zeros parts (CR, bad frames)
which was creating noise.


3.5.1
=====

Minor bugs correction

3.5.2
=====

Alternative Merging Process
---------------------------

Addition of an alternative merging process
(:py:meth:`~process.InterferogramMerger.alternative_merge`): in fact,
this is the basic merging process which makes no use of star
photometry. This alternative way of merging, somehow more noisy than
the regular way, is more robust and might be the best guess if there's
not enough good stars in the field or when all the fiel is covered
with intense emission lines. It is recommanded to always do the
reduction this way and the regular way to take what seems the best
cube.

The new option in the orbs launch command is::

 --alt_merge

3.5.3
=====

Aperture Photometry
-------------------

An aperture photometry function
(:py:meth:`~astrometry.aperture_photometry`) has been designed to get
a far more robust and precise photometry of the stars during the
'normal' merging process. Sky transmission vector precision is now a
lot better and do not need any more smoothing.


Cosmic Ray Detection
--------------------

New step frame check added (removed a long time ago but added once
again) to get rid of star detection in some frames due to disalignment
and the size of interferometric fringes. Avoid getting bad photometry
on stars.

Mask
----

Frames created by ORBS are coupled with a mask frame. Mask frames are
used to get the exact position of all the pixels affected by the
cosmic rays correction. Cosmic rays correction in stars creates bad
pixels that have to be taken into account during the photometry
process t avoid too deviant values.


Tuning parameters
-----------------

It is now possible to tune the parameters of some methods externally
(in the option file). To tune a 'tunable' parameter you must use the
keyword TUNE, give the full name of the method parameter
(class.method.parameter_name) and its new value::
  
  TUNE InterferogramMerger.find_alignement.BOX_SIZE_COEFF 7.

.. note:: All the parameters are not tunable: this option has to be
  implemented in the method itself with the method
  :py:meth:`~core.Tools._get_tuning_parameter`.

.. warning:: This possibility is intented be used only for the
  reduction of some particular cubes. If the default value of a
  paramater has to be changed it is better to do it in the method
  itself.

3.5.4
=====

Astrometry & Photometry
-----------------------

Astrometry and photometry processes (fit and aperture) upgraded. They
know meet the theoretical error and their returned reduced-chi-square
is far better. All dependant processes in the process module have been
updated to use this better information and filter bad fitted stars.


v3.6
****

3.6.0
=====

Flux Calibration
----------------

Flux calibration has been added. :py:meth:`orbs.orbs.Orbs.calibrate_spectrum`
replace the old function :py:meth:`orbs.orbs.Orbs.correct_spectrum`. The path
to a standard spectrum reduced by ORBS must be given (STDPATH). This
can be achieved by reducing a standard cube using the option
--standard. The standard name must also be given in the option file
(STDNAME). This must be recorded in the standard table
(orbs/data/std_table). Standard spectra form MASSEY et al. 1988 and
CALSPEC have been added to the data of ORBS so that they can be used
to do a flux calibration. To do a flux calibration the steps are thus:

1. Reduce standard cube with option --standard

2. Give the path to the standard spectrum (STDPATH) and the name of
   the standard (STDNAME) in the option file of the cube yo want to
   calibrate

3. Reduce the cube you want to calibrate (or only redo the last step)
  
.. seealso:: :py:meth:`~process.Spectrum.get_flux_calibration_vector`

.. note:: The standard cube must be reduced with the same number of
     camera as the cube you want to reduce.

.. note:: A new class has been created to manage standard spectra:
     :py:class:`~process.Standard`

.. note:: Final spectrum cube is now rescaled pixel to pixel in order
     to keep the same energy at the input and at the output of the
     reduction process. With 2 cubes we use the scale map. The scale
     map is the sum of the deep frame of both cubes ; The deep frame
     of cube A is scaled by the modulation coefficient which comes
     from the difference of gain between both cameras. For a single
     cube reduction, its own deep frame is used.

WCS correction
--------------

If the ra/dec (TARGETR/TARGETD) and x/y (TARGETX/TARGETY)
corresponding position of a target near the center of the frame is
given, WCS coords of the cube are updated at the last step (Calibration step
step): 

.. seealso:: :py:meth:`~process.Spectrum.get_corrected_wcs`

.. note:: An **internet connection** must be available to correct WCS
     because the USNO database is used to get precise astrometric
     coordinates of the stars in the field.

.. warning:: A new module is now required to launch ORBS: PyWCS (see
    http://stsdas.stsci.edu/astrolib/pywcs/)



Simplification
--------------

* :py:class:`core.Indexer` created to index reduction files and get
  their location easily in :py:class:`orbs.Orbs`.

* No more quadrants: Reduction of big data cubes has been simplified
  and do not save reduced files in quadrants any more. Big data cubes
  are thus handled as small data cubes. Reduced by quadrants but saved
  as one set of frames.

3.6.2
=====

Cython & speed optimization
---------------------------

* :py:meth:`utils.transform_frame` has been modified to do only one
  geometrical transformation instead of a set of transformations
  (tip-tilt then translations then rotation etc.). Coordinates
  transformation function (:meth:`cutils.transform_A_to_B`) written in
  Cython to optimize processing speed .

* core functions for fitting stars (:meth:`cutils.gaussian_array2d`
  and :meth:`cutils.surface_value`) have been transcripted to Cython
  for faster processing.

* lots of functions have been cythonized to improve the overall speed.

v3.7
****

3.7.0
=====

ORB: A new core module
----------------------

ORBS core classes and functions (core.py, utils.py and cutils.pyx)
have been moved to a module of shared core libraries: ORB. This way,
ORBS, ORCS, OACS, IRIS and ORUS can share the same core module without
importing ORBS entirely each time. Conceptually ORBS, like the others,
just wraps around ORB module and is not any more the central part of
the whole suite of softwares.

3.7.1
=====

Multi fit of the stars
----------------------

The functions based on star fitting have been updated to take full
advantage of the multi_fit mode of
:meth:`orb.astrometry.fit_stars_in_frame`. Stars are fitted all
together based on the idea that the position pattern is good but may
be shifted, rotated or zoomed. The stars share also the same
FWHM. This update has made ORBS far more robust and precise on the
alignment and merge processes. Even a cube like ORION which contains
only few stars with very bad SNR can be perfectly aligned.

USNO-B1 based star detection
----------------------------

It is now possible to use a star catalogue like USNO-B1 to detect
stars in the cube. It is not a default behaviour because extended
emission region contains virtually no catalogued stars. This option
can be useful for galaxies to avoid the confision of HII regions and
stars.

3.7.2
=====

Minor bugs fix. This version is considered as a nearly stable version
ready for release.

3.7.2.1
=======

* Better integration of the multi fit mode (now used most of the time)

* option file keyword added: TRYCAT that must be set to 1 to use the
  USNO-B1 catalogue for star detection.

* Better treatment of NaNs. Begin to remove the use of zeros in place
  of NaNs.

* doc update

* bug fix

3.7.2.2
=======

Wavenumber computation & better integration with ORCS
-----------------------------------------------------

The whole spectrum computation process can now be done in wavenumber
(useful to avoid the mutiple interpolation nescessary to move from a
regular wavenumber space to an iregular wavelength space back and
forth).

It is also possible to compute an uncalibrated spectrum. This way
there is absolutly no interpolation made during the spectrum
computation. The output can be used by ORCS directly and ORCS itself
does not have any interpolation to do for the extraction of the lines
parameters. **This ensure that the spectral information is not distorded
at all during the process**.

The filter correction during the calibration process takes into
account the fact that no wavelength calibration has been done.

* Important modified methods:

  * :py:meth:`~process.Interferogram.compute_spectrum`
  * :py:meth:`~process.Spectrum.calibrate`

* New keyword added in the option file: WAVENUMBER, WAVE_CALIB


Enhanced phase map fit
----------------------

* :py:meth:`~process.Phase.fit_phase_map` has been enhanced to give
  better results and use the residual map on phase fit.


Miscellaneous
-------------

* :py:meth:`orbs.orbs.Orbs.__init__` simplified by the use of
  :py:class:`orb.core.OptionFile` previously used only by
  :py:class:`orcs.orcs.SpectralCube()`

* doc updated

ORB's scripts
-------------

* move ORB's scripts (dstack, combine, rollxz, rollyz, reduce) from
  orbs/scripts to orb/scripts so that only ORBS specific scripts are
  in orbs/scripts.

* create **unstack** script to unstack a cube into a set of frames

3.7.2.3
=======

Astropy.fits.io
---------------

* PyFITS is now part of Astropy (http://www.astropy.org/). PyFITS
  library will not be used anymore and the required import have been
  changed to astropy.fits.io.

* Bugs with new version of pyfits fixed

Miscellaneous
-------------

* FITS keywords updated for standard flux calibration (standard name,
  standard file path, mean flambda calibration). Target ra, dec, x, y,
  step number, ORBS version and ORBS option file name also added.

Full precision
--------------

* :py:meth:`~process.InterferogramMerger.find_alignment` default
  behaviour changed. The alignement pass for tip and tilt angles is
  not anymore by default. It is still possible to do it by adding this
  line in the option file::

    TUNE InterferogramMerger.find_alignment.FULL_PRECISION 1


3.7.3
=====

SITELLE data
------------

Sitelle image mode
~~~~~~~~~~~~~~~~~~

Using the SITELLE's configuration file config.sitelle.orb and thus
having the keyword INSTRUMENT_NAME set to SITELLE enables the sitelle
mode. The only modification which has to be done was to pass two new
options to :py:meth:`orb.Tools._create_list_from_dir`: image_mode and
chip_index (see ORB documentation). This is done during ORBS init:
:py:meth:`orbs.orbs.Orbs.__init__`.

Prebinning
~~~~~~~~~~

Used for faster computation of big data set. It
can also be useful if the user simply wants binned data. At the user
level only one option must be passed to the option file::

  PREBINNING 2 # Data is prebinned by 2

.. warning:: The real binning of the original data must be kept to the
   same values. The user must no modify the the values of BINCAM1 and
   BINCAM2.

Modification of :py:meth:`orbs.orbs.Orbs.__init__` which gives the
prebinning option to :py:meth:`orb.Tools._create_list_from_dir`. The
final data binning is also handled at the init level : the real
binning of each camera is multiplied by the prebinning size.

3.7.4
=====

Enhanced flux calibration
-------------------------

* Flux calibration is now much more precise because the whole spectrum
  are no more rescaled only from deep frames or energy map but on the
  whole conservation of the energy from the input to the output. This
  relies on the use of both deep frame and energy maps and assert,
  just before the calibration that E(I) / M(I) = E(S) / M(S) if we
  consider that E() is the energy map and M() is the deep frame of the
  interferogram cube I or the spectral cube S. Both
  :py:meth:`~process.InterferogramMerger.extract_stars_spectrum` and
  :py:meth:`~process.Spectrum.calibrate` have been modified to take
  that into account.

* Bug fix in :py:meth:`~process.InterferogramMerger.extract_stars_spectrum`.

Astropy
-------

Astropy (http://www.astropy.org/) is definitly needed, pyfits and
pywcs standalone modules are not needed anymore by ORBS (but they
still can be used by other modules ;) even modules imported by ORBS so
be carefull before removing them)

* PYFITS: now imported from astropy.io.fits
* PYWCS: now imported from astropy.wcs

3.7.5
=====

Miscellaneous
-------------

* :py:meth:`~process.RawData.correct_frame`: Hot pixels correction
  algorithm changed for a standard median correction (hot pixel value
  is replaced by the median of the neighbouring pixels).

* :py:meth:`~process.InterferogramMerger.merge`: Merging operation
  when the frames of the camera B are not used do not rely anymore on
  the computed stray light vector because of the impossibility to
  determine correctly the base level of this vector.

v3.8 Start of CFHT integration
******************************

3.8.0
=====

* script **orbs** can take a 'SITELLE job file' as input. It creates an option file from it and launch orbs from this option file.

* script **orbs** can be lauched in SITELLE or SPIOMM mode. This way
  is ensures that the configuration file will be the good one (instead
  of relying on the default behaviour)

* :py:class:`orbs.orbs.Orbs` can be passed lists instead of directory paths for
  object, dark, flat ...

* :py:class:`orbs.orbs.Orbs` can be passed a special configuration file and
  this choice is reflected on all the processes it launches.


3.8.1
=====

**orbs** script
---------------

**orbs** script has been modified a lot to integrate new operations:

* *start*: start a reduction.

* *status*: display the status of all the reduction processes started
  with the same option file.

* *resume*: simply resume the last reduction process started with the
  given option file.

* *clean*: clean the folder from all the reduction files but the final
  products.

The new way of calling orbs is now::

  orbs [option_file_path] {start, status, resume, clean} [options]

To simply run the default reduction process (for an astrophysical object)::

  orbs [option_file_path] start

If the data is e.g. a laser cube you must run orbs with the --laser option::

  orbs [option_file_path] start --laser

All the special targets are (the default target is a common
astrophysical object):

- *laser*: for a laser cube
- *flat*: for a flat cube (or more generally a continnum source cube)
- *stars*: for a star/galaxy cluster
- *nostar*: for an astophysical object containing no star
- *standard*: for a standard star

More options are available and can be listed by adding -h to the
command line.

Those operations are all based on the concept of RoadMaps (see below).

RoadMaps
--------

The integration of the 'roadmaps' concept avoid to modify the code in
order to add or modify a reduction sequence. It is now possible to
change the steps, their order and the associated options of a given
reduction sequence by modfying a simple xml file.

All the possible reduction steps are listed in the file :file:`orbs/data/roadmap.steps.xml`

Each reduction process has its own roadmap. Each instrument has also
its own roadmaps and some reduction processes have different roadmaps
depending on the camera we want to reduce. A name of a roadmap file is thus::

  roadmap.[instrument].[target].[camera].xml

- *instrument* can be *spiomm* or *sitelle*
- *target* can be one of the special targets listed above or object for the default target
- *camera* can be *full* for a process using both cameras; *single1* or
  *single2* for a process using only the camera 1 or 2.

e.g. the roadmap file for the default reduction process of SITELLE is :file:`orbs/data/roadmap.sitelle.object.full.xml`


* classes :py:class:`orbs.orbs.RoadMap` and
  :py:class:`orbs.orbs.Step` created to manage the roadmap files.

* :py:class:`orbs.orbs.Orbs` has been changed to integrate this new
  concept. The methods :py:meth:`orbs.orbs.Orbs.full_reduction` and
  :py:meth:`orbs.orbs.Orbs.single_reduction` have been replaced by one
  simple method :py:meth:`orbs.orbs.Orbs.start_reduction` which just
  read the xml files and run the sequence.


* most of the early 'development' options of the methods of
  :py:class:`orbs.orbs.Orbs` have been removed, keeping only the
  useful ones.

Miscellaneous
-------------

* :py:meth:`~process.CalibrationLaser.create_calibration_laser_map`
  output all the parameters of the fitted laser line. this way the ILS
  (fwhm) map can be derived directly. The laser spectral cube is also
  generated by default.

* in :py:meth:`~process.Interferogram.create_correction_vectors`, the
  correction of the transmission vector, in the case when only one
  camera is reduced, has been enhanced (the transmission near ZPD
  cannot be retrieved and must be guessed).

3.8.2
=====

Aligner
-------

:py:class:`orb.astrometry.Aligner` has been created in ORB to handle
alignement between frames. The whole alignment procedure has been
updated and is now much faster and robust. More details on the
alignment procedure can be found in ORB documentation.


3.8.3
=====


Spectral calibration
--------------------

Spectral calibration is now done at the calibration step, see:
:py:meth:`orbs.orbs.Orbs.calibrate_spectrum` (before this was done
during the FFT computation).


3.8.4.0
=======

Update of the documentation
---------------------------

A far more complete **reduction guide** (see :ref:`reduction_guide`)
has been written. Some parts still need some further explanations
(coming with 3.8.4.1).

Required option file keywords
-----------------------------

* The keywords **BINCAM1** and **BINCAM2** are not used anymore. The
  binning is deduced from the detector size. CAM1_DETECTOR_SIZE_X,
  CAM1_DETECTOR_SIZE_Y, CAM2_DETECTOR_SIZE_X, CAM2_DETECTOR_SIZE_Y
  have been added to the ORB configuration file.

* **SPEDART** keyword is no more required if not path to a dark folder
  is set (**DIRDRK1** or **DIRDRK2**).


v3.9 The HDF5 Miracle
*********************

All ORBS internal cubes used for computation have been passed to an
HDF5 format which makes data loading incredibly faster. If those
changes have small effects on small data cubes like SpIOMM data, it
changes a lot the computation time on SITELLE's data cubes (passing
from ~10 hours to 6.5 hours on a 16 procs machine).

The HDF5 format is also very useful to display large data cubes with
**orb-viewer** without loading the full cube in memory.


v3.9.0
======

Nearly all classes in :file:`orbs.py` (see :ref:`orbs_module`) and
:file:`process.py` (see :ref:`process_module`) have been modified to
accept hdf5 cubes as input and output hdf5 cubes.

During the Init of ORBS, FITS cubes are exported to HDF5 cubes before
the reduction can start.

.. seealso:: ORB documentation (v1.4.0) for more info on the numerous
             core changes

Scripts
-------

* **orbs-check** has been modified to be based on GTK instead of
  Tkinter (like all the other viewer of ORB). Tkinter is no more used
  by ORBS.

* **orbs-optcreator** has not survived all the upgrades and has been
  removed. A command line tool might come instead of a gui-based tool.



Default Output Cube
-------------------

The default output cube format is now the classic format (see
:ref:`choosing-right-output`) i.e. : 

* Apodization 2.0 
* Wavelength
* Calibrated

This format **is not the best format** for data analysis but it is a
comprehensive format for eye checking. The best format would be:

* Apodization 1.0 
* Wavenumber
* Uncalibrated


v3.10: handling real SITELLE's data cubes
*****************************************

v3.10.0
=======

Phase correction
----------------

SITELLE's phase map is nearly ideal so that a **better kind of phase
correction is possible**. Now, the 'order 0 phase map' depends only on
the OPD path i.e. the incident angle of the light (if we consider that
the surfaces ot the interferometer's optics are perfect, which seems
to be a good enough assumption up to now). The order 0 phase map can
thus be modeled directly from the calibration laser map which gives
the incident angle at each pixels. As the calibration laser map can be
tilted (2 angles along X and Y axes) and rotated around its center,
the model must take into account all those 3 parameters.

There are at least two major **advantages**:

  * We have an **understood model** with physical parameters to fit
    the phase map (and the fitting approximation is really great,
    giving a gaussian shaped error distribution with no apparent bias
    or skewness).

  * **We get the real calibration laser map** which corresponds to the
    scientific cube and not a calibration laser map taken in different
    conditions (gravity vector, temperature and so on).


* :py:class:`~process.Phase`:

  * :py:meth:`~process.Phase.fit_phase_map` can be set to use a
    'sitelle' model to fit the phase. In this case the new calibration
    laser map is also
    returned. :py:meth:`orbs.orbs.Orbs.calibrate_spectrum` will use
    the new calibration laser map.

  * :py:meth:`~process.Phase.smooth_phase_map` is now parallelized to
    smooth the phase map by quadrants ('unwrap' instead of 'smooth'
    would be more precise). A special algorithm has been developped to
    reorder the quadrants (each one being individually smoothed) and
    get a perfectly unwrapped phase map.

* :py:class:`orbs.orbs.Orbs` has been adapted to all those changes.

* **n_phase** argument (which gives the number of points used to
  compute the phase) has been definitly removed. Now, the whole
  interferogram is always considered because there is no interest in
  using less points instead for a more aesthetic and smoother phase
  vector (but the fitting precision is exactly the same).



Cosmic rays detection
---------------------

* A lot of high incident angle CR are visible and they are badly
  detected by the actual process. The levels of detections have bee
  lowered but the whole detection process must be recoded to get a
  more robust detection (using both cameras for example).

* All point sources are shielded (see ORB's documentation for the
  detection of all sources).


Miscellaneous
-------------

* **orbs** script accepts a new special target: '--raw' to get a
  faster and robust reduction with no cosmic ray detection nor phase
  correction (useful for fast verification of the general quality of a
  data cube). Also, the old argument --nphase has been replaced
  by --nophase to simply avoid phase correction.

* :py:meth:`orbs.orbs.Orbs.calibrate_spectrum` now does not fail on
  bad WCS calibration.

* script **orbs-sitelle-makejob** created to help in creating a job
  file from a list of the files to reduce (object files, flats and
  calibration map)



v3.10.1
=======


Source extraction
-----------------

Source extraction has been implemented. The new target **--sources**
is available in orbs command. The keyword SOURCE_LIST_PATH can be set
to the path of a source list in the option file.

Source spectra are not background subtracted.


Source extraction is handled by
:py:class:`~process.SourceExtractor`.

Phase maps
----------

Phase computation process has completely changed. It is now based on a
binned interferogram cube from which the order 1 of the phase is
extracted by a fitting process. The order 0 maps is then computed
knowing the order 1 with precision by minimization of the imaginary
part at each binned pixel. The obtained order 0 map is much more
precise and can be fitted with an opto-mechanical model. Phase map
unwrapping process (generally called smoothing, which must be made
before fitting) has also been improved.

Smoothing and fitting is now handled with
:py:class:`~process.PhaseMaps`.

HDF5 Output
-----------

The final spectral cube generated at the output is now in HDF5
format. Parts of it can be extracted and converted into FITS format
with **orb-extract** script. 


Miscellaneous
-------------

* Flat frames are now normalized before being combined to avoid an
  intensity change problem when they are combined.

* **orbs-fit-calibration-laser-map** script has been created to fit exacly
  a calibration laser map and remove artefacts generated by small
  calibration laser cubes (not enough resolution and SNR). The
  precision obtained on a fitted calibration laser map is less than a
  few 10 m/s (yes, meters !)

v3.10.2
=======


HDFQuad
-------

All calibrations updated and stable
-----------------------------------

* photometry
* astrometry
* source extraction


Miscellaneous
-------------

* Moved Standard class from orbs.process to orb.core



Calibrated output
-----------------

Calibrated spectral cube are now projected on an axis at the center of
the field instead of an axis on the interferometer axis (0
degree). Avoids cutting the borders of the filters when the
observation parameters are so well defined that there would be some
folding on the interferometer axis (but no folding on the FOV, by
definition).


v.4. Data Release 1
*******************

This is a major version corresponding to the first Data Release of
SITELLE made in March 2016. **This version is stable for SITELLE**. It
has not been checked for SpIOMM and some problems might be experienced
since a lot has changed.

v4.0-DR1-beta
=============

Scripts
-------

* **orbs-create-phase-file** created to create a high order phase file
  from the binned phase cube obtained during phase computation. Phase
  is smoothed with a Spline.

* **orbs-check-standard-spectrum** plot the observed standard spectrum
  vs the real and flux calibrated standard spectrum for checking
  purpose.

Phase
-----

Stable and robust phase algorithm for SN1, SN2, SN3

* order 0: order 0 map fitted with the calibration laser map and
  residual fitted with a Zernike modes.
* order 1: Fitted with a simple 1 order polynomial in 2D.
* Higher orders are computed from an external source
  (e.g. Flat/Continuum cube) and used during the fitting procedure.


Calibration
-----------

* Flux calibration in SN1, SN2 and SN3 have been derived from standard
  cubes of GD71 taken in January 2016 during the Science Verification
  phase.

* Astrometry: Stable and robust registration algorithm.

* Wavelength calibration and nm axis projection are done via an super
  interpolated spectrum (pure FFT interpolation). Interpolation errors
  can now be considered as negigible. There is no more distorsion of
  the spectral information whatever the output flavor can be. In this
  respect an calibrated spectrum in cm-1 has the same spectral
  information as a non calibrated spectrum. But a nm spectrum still
  present a non-symetrical ILS (spectral information is conserved but
  it's hard to fit).

Problems
--------

* C1 filter: Strange continuum shape due to a strong refraction index
  change with respect to the wavelength. This problem prevents a
  relieable calibration and phase correction.

* Phase correction of pure emission-line cubes (extended nebulae)
  cannot be computed. Extra laser frames taken at the beginning of
  each cube are still not used.

* Precise wavelength calibration is not done. Extra laser frames taken
  at the beginning of each cube are still not used.

* Optical distorsion not taken into account.

* Filter correction question has still not been adressed. Spectral
  cube are not filter corrected. Filters curves are not good enough to
  permit a good correction.

v4.1
====

* Apodization changed for a gaussian apodization (Norton-Beer
  apodization function are not used anymore).

* Standard star position computation takes into account proper motion.

* --nocheck option added to **orbs** script to avoid checking raw data
  files when the raw data cubes have already been built.
