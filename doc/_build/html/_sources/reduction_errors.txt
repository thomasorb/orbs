==============================
 ORBS Common reduction errors
==============================

.. contents::

.. _how-can-access-orbs-commands:

Malformed sequence of files ?
=============================

ORBS will always try to sort the name of the files in a given
folder. So all the files must have a common nomenclature. Any kind of
name is accepted as long as only one number is changing. For example,
all those nomenclatures are accepted::

  0image.fits
  image000.fits
  image_M51_CAM1_0000.fits
  etc.

This error happens when some files in the folder do not respect the
nomenclature. If the files are named e.g.::

  dark_1.fits
  dark_2.fits
  dark_3.fits
  dark.fits

The last file (:file:`dark.fits`) do not respect the nomenclature.
Just try to rename (or remove if it can be removed) the file. If the
file cannot be renamed or remove (e.g. all the data files are in the
same folder) you can give the path to a :ref:`file-list` in place of
the path to a folder (see the note in :ref:`required-parameters`).


Error during cube alignment ?
=============================

An error like *poor ratio of fitted stars in both cubes* or *not
enough fitted stars in both cubes* is caused by stars detected in the
camera 1 and undetected in the camera 2 (which is very unlikely with
SITELLE). It happens when the integration time is more than 30s while
the camera 2 temperature is too high: the dark current becomes too
important and most of the non-saturating stars simply disappear in the
frames of the camera 2.

Nevertheless the alignment might be good enough, even if the checking
step gives you an error. In this case just put this line in your
option file::

  TUNE Aligner.compute_alignment_parameters.SKIP_CHECK 1

This will disable the checking step and continue to reduce the
cube. Just make sure that the frames have been aligned correctly by
looking at the deep frame of the merged cube. This frame can be found
in the temporary reduction folder
:file:`OBJECT_FILTER/MERGED/OBJECT_FILTER.merged.InterferogramMerger.deep_frame.fits`.
