#!/usr/bin/python
# *-* coding: utf-8 *-*
# Author: Thomas Martin <thomas.martin.1@ulaval.ca>
# File: orbs.py

## Copyright (c) 2010-2017 Thomas Martin <thomas.martin.1@ulaval.ca>
## 
## This file is part of ORBS
##
## ORBS is free software: you can redistribute it and/or modify it
## under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## ORBS is distributed in the hope that it will be useful, but WITHOUT
## ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
## or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
## License for more details.
##
## You should have received a copy of the GNU General Public License
## along with ORBS.  If not, see <http://www.gnu.org/licenses/>.

"""
Orbs module contains the high level classes of ORBS and above all Orbs
class which acts as a user interface to the processing module
process.py
"""

__author__ = "Thomas Martin"
__licence__ = "Thomas Martin (thomas.martin.1@ulaval.ca)"                      
__docformat__ = 'reStructuredText'
import version
__version__ = version.__version__

import os
import time
import resource
import sys
import traceback
import scipy
import numpy as np
import xml.etree.ElementTree
import logging
import warnings
import shutil
from datetime import datetime
import astropy
import pp
import bottleneck as bn


from orb.core import Tools, FDCube, Indexer, OptionFile, HDFCube, TextColor
from orb.core import FilterFile, ProgressBar
from process import RawData, InterferogramMerger, Interferogram
from process import Spectrum, CalibrationLaser
from process import SourceExtractor, CosmicRayDetector
import orb.constants
import orb.version
import orb.utils.spectrum
import orb.utils.fft
import orb.utils.stats
import orb.utils.image
import orb.utils.vector
import orb.utils.io

ORBS_DATA_PATH = os.path.join(os.path.split(__file__)[0], "data")

##################################################
#### CLASS Orbs ##################################
##################################################

class Orbs(Tools):
    """ORBS user-interface implementing the high level reduction methods

    Help managing files during the reduction process and offer
    simple high level methods to reduce SpIOMM data. You must init
    Orbs class with a path to an option file (e.g. option.opt)
    containing all the parameters needed to run a reduction.

    .. note:: The option file must contain at least the following
      parameters (each parameter is preceded by a keyword). See
      'options' attribute to get all the possible keywords.

    :OBJECT: Name of the object

    :FILTER: Filter name

    :SPESTEP: Step size of the moving mirror (in nm)

    :SPESTNB: Number of steps

    :SPEORDR: Order of the spectral folding

    :SPEEXPT: Exposition time of the frames (in s)

    :SPEDART: Exposition time of the dark frames (in s)
    
    :OBSDATE: Observation date (YYYY-MM-DD)

    :HOUR_UT: UT hour of the observation (HH:MM:SS)

    :BAD_FRAMES: List of bad frames indexes

    :TARGETR: RA of the target (hour:min:sec)
        
    :TARGETD: DEC of the target (degree:min:sec)

    :TARGETX: X position of the target in the first frame

    :TARGETY: Y position of the target in the first frame

    :DIRCAM1: Path to the directory containing the images of the
      camera 1

    :DIRCAM2: Path to the directory containing the images of the
      camera 2

    :DIRBIA1: Path to the directory containing the bias frames for the
      camera 1

    :DIRBIA2: Path to the directory containing the bias frames for the
      camera 2

    :DIRDRK1: Path to the directory containing the dark frames for the
      camera 1

    :DIRDRK2: Path to the directory containing the dark frames for the
      camera 2

    :DIRFLT1: Path to the directory containing the flat frames for the
      camera 1
          
    :DIRFLT2: Path to the directory containing the flat frames for the
      camera 2
          
    :DIRCAL1: Path to the directory containing the images of the
      calibration laser cube of the camera 1

    :DIRCAL2: Path to the directory containing the images of the
      calibration laser cube of the camera 2

    :DIRFLTS: Path to the directory containing the flat spectrum
      frames

    :STDPATH: Path to the standard spectrum file

    :PHASEMAPS: Path to the external phase map file

    :STDNAME: Name of the standard used for flux calibration

    :FRINGES: Fringes parameters

    :STARLIST1: Path to a list of star positions for the camera 1

    :STARLIST2: Path to a list of star positions for the camera 2

    :APOD: Apodization function name

    :CALIBMAP: Path to the calibration laser map

    :TRYCAT: If True (an integer > 0) a star catalogue (e.g. USNO-B1)
      is used for star detection (TARGETR, TARGETD, TARGETX, TARGETY
      must be given in the option file). If False star detection will
      use its own algorithm. You can also force ORBS to use a given
      star list, see STARLIST1 and STARLIST2 keywords. This option is
      set to False by default
        
    :WAVENUMBER: If True (an integer > 0) the output spectrum will be
      in wavenumber instead of wavelength. This option avoids the use
      of interpolation to transform the original wavenumber spectrum
      to a wavelength spectrum
          
    :WAVE_CALIB: If True (an integer > 0) the output sepctrum will be
      wavelength calibrated using the calibration laser map. This
      option is set to True by default

    :NOSKY: If True (an integer > 0) sky-dependant processes are skipped.

    :NOSTAR: If True (an integer > 0) star-dependant processes are skipped.

    :SOURCE_LIST_PATH: Path to a list of sources (X Y), one source by
       line. Used for sources extraction.


    .. note::
    
        * The order of the parameters is not important
        
        * Lines without a keyword are treated as commentaries (but the
          use of # is better)
        
        * Paths can be either relative or absolut
        
        * An example of an option file (options.opt) can be found in
          the scripts folder (Orbs/scripts) of the package
          
        * 'orbs-optcreator' is an executable script that can be used
          to create an option file

          
    .. warning:: Two parameters are needed **at least** : the object
        name (OBJECT) and the filter name (FILTER)
    """

    __version__ = None # imported from __version__ given in the core module
    
    _APODIZATION_FUNCTIONS = ['learner95']
    """Apodization functions that recognized by ORBS. Any float > 1 is
    recognized by :py:meth:`orb.utils.gaussian_window` (see
    :py:meth:`orb.utils.fft.gaussian_window`)"""

    
    project_name = None
    """Name of the project, created during class initialization"""
    overwrite = None
    """Overwriting option. If True, existing FITS files will be
    overwritten"""

    config = dict()
    """Dictionary containing all the options of the config file.  All
    those options can be changed for any particular reduction using
    the option file.

    .. note:: Keywords used (The keywords of the configuration file
       are the same)

        * INIT_ANGLE: Rough angle between images of the cameras 1 and 2
        
        * INIT_DX: Rough disalignment along x axis between cameras 1
          and 2 for a 1x1 binning
          
        * INIT_DY: Rough disalignment along y axis between cameras 1
          and 2 for a 1x1 binning
          
        * FIELD_OF_VIEW_1: Size of the field of view of the camera 1 in
          arc-minutes
          
        * FIELD_OF_VIEW_2: Size of the field of view of the camera 2 in
          arc-minutes
          
        * PIX_SIZE_CAM1: Camera 1 pixel size in um

        * PIX_SIZE_CAM2: Camera 2 pixel size in um
        
        * BALANCED_CAM: Number of the camera on the balanced port
        
        * CALIB_NM_LASER: Wavelength of the calibration laser in nm
        
        * CALIB_ORDER: Folding order of the calibration laser cube
        
        * CALIB_STEP_SIZE: Step size of the calibration laser cube
        
        * PHASE_FIT_DEG: Degree of the polynomial used to fit the phase
        
        * DETECT_STAR_NB: Number of star to use for alignment and photometry
        
        * INIT_FWHM: Rough estimate of the stars FWHM in arcseconds
        
        * PSF_PROFILE: PSF used to fit stars (can be gaussian of moffat)
        
        * MOFFAT_BETA: Default beta parameter for the Moffat PSF
        
        * DETECT_STACK: Number of frames to combine for star detection
        
        * OPTIM_DARK_CAM1: If set to 1 : run the optimization routine
          to remove camera 1 dark. Set to 0 to avoid optimization routine
          
        * OPTIM_DARK_CAM2: If set to 1 : run the optimization routine
          to remove camera 2 dark. Set to 0 to avoid optimization routine
          
        * DARK_ACTIVATION_ENERGY: Calibrated activation energy of the
          dark frames. Used to correct for varying dark level of the
          camera 2 of SpIOMM
                    
        * EXT_ILLUMINATION: If there is a chance for some light to
          enter in one of the cameras and not the other this must be
          set to 1. This way this external light can be tracked by the
          merge process.

        * PREBINNING: Prebinning of the frames, useful for a quick
          reduction.
    """
    
    options = dict()
    """Dictionary containing all the options of the option file and
    others created during initialization needed by processing classes.

    .. note:: Keywords used and related keyword in the option file:
    
        * object_name: OBJECT
        
        * filter_name: FILTER
        
        * bin_cam_1: BINCAM1
        
        * bin_cam_2: BINCAM2
        
        * step: SPESTEP
        
        * step_nb: SPESTNB
        
        * order: SPEORDR
        
        * exposure_time: SPEEXPT
        
        * dark_time: SPEDART
        
        * obs_date: OBSDATE
        
        * bad_frames: BAD_FRAMES
        
        * target_ra: TARGETR
        
        * target_dec: TARGETD
        
        * target_x: TARGETX
        
        * target_y: TARGETY
        
        * image_list_path_1: DIRCAM1
          
        * image_list_path_2: DIRCAM2
          
        * bias_path_1: DIRBIA1

        * bias_path_2: DIRBIA2
          
        * dark_path_1: DIRDRK1

        * dark_path_2: DIRDRK2
          
        * flat_path_1: DIRFLT1
          
        * flat_path_2: DIRFLT2
          
        * calib_path_1: DIRCAL1
          
        * calib_path_2: DIRCAL2
          
        * standard_path: STDPATH
        
        * phase_map_path: PHASEMAPS
        
        * fringes: FRINGES
        
        * flat_spectrum_path: DIRFLTS
          
        * star_list_path_1: STARLIST1
          
        * star_list_path_2: STARLIST2
          
        * apodization_function: APOD
        
        * calibration_laser_map_path: CALIBMAP

        * wavefront_map_path: WFMAP
          
        * try_catalogue: TRYCAT
          
        * wavenumber: WAVENUMBER

        * spectral_calibration: WAVE_CALIB

        * prebinning: PREBINNING

        * no_sky: NOSKY

        * source_list_path: SOURCE_LIST_PATH
    """
    tuning_parameters = dict()
    """Dictionary containg the tuning parameters of some methods
       called by ORBS. The dictionary must contains the full parameter
       name (class.method.parameter_name) and its value. For example :
       {'InterferogramMerger.find_alignment.BOX_SIZE': 7}. Note that
       only some parameters can be tuned. This possibility is
       implemented into the method itself with the method
       :py:meth:`orb.core.Tools._get_tuning_parameter`.

       To set a tuning parameter in the options file use the keyword
       TUNE followed by the full parameter name and its new value::

          TUNE InterferogramMerger.find_alignment.WARNING_RATIO 0.8
          TUNE InterferogramMerger.find_alignment.BOX_SIZE_COEFF 7
    """
    
    optionfile = None
    """OptionFile instance"""

    option_file_path = None
    """Path to the option file"""

    targets = ['object', 'flat', 'standard', 'laser', 'nostar', 'raw',
               'sources', 'extphase', 'nophase', 'phasecube']
    """Possible target types"""
    
    target = None
    """Choosen target to reduce"""

    cams = ['single1', 'single2', 'full']
    """Possible camera names"""
    
    def __init__(self, option_file_path, target, cam,
                 instrument=None, ncpus=None,
                 overwrite=True, silent=False, fast_init=False,
                 raw_data_files_check=True, logfile_path=None):
        """Initialize Orbs class.

        :param option_file_path: Path to the option file.

        :param target: Target type to reduce. Used to define the
          reduction road map. Target may be 'object', 'flat',
          'standard', 'laser', 'raw', 'sources' or 'extphase'.

        :param cam: Camera number. Can be 'single1', 'single2' or
          'full' for both cameras.

        :param instrument: (Optional) name of the instrument config to
          load (default None).
    
        :param ncpus: (Optional) Number of CPUs to use for parallel
          processing. set to None gives the maximum number available
          (default None).  

        :param overwrite: (Optional) If True, any existing FITS file
          created by Orbs will be overwritten during the reduction
          process (default True).

        :param silent: (Optional) If True no messages nor warnings are
          displayed by Orbs (useful for silent init).

        :param fast_init: (Optional) Fast init. Data files are not
          checked. Gives access to Orbs variables (e.g. object dependant file
          paths). This mode is faster but less safe.

        :param raw_data_files_check: (Optional) If True, correspondance
          between original data files and built raw data cubes is
          checked. If False, raw data cubes are considered ok (default
          True).
        """
        def export_images(value, fast_init, option_key,
                          mask_key, camera_index):
            if os.path.isdir(value):
                raise ValueError('A path to a directory cannot be given. Please give a path to a list of files.')
            else:
                self.options[option_key] = value

            # export fits frames as an hdf5 cube
            if not fast_init:

                # check if the hdf5 cube already
                # exists. 
                export_path = (
                    self._get_project_dir()
                    + os.path.splitext(
                        os.path.split(
                            self.options[option_key])[1])[0]
                    + '.hdf5')

                already_exported = False
                check_ok = False
                if os.path.exists(export_path):
                    already_exported = True
                    if not raw_data_files_check:
                        check_ok = True
                        warnings.warn(
                            'Exported HDF5 cube {} not checked !'.format(export_path))


                if not check_ok or not already_exported:
                    # If the list of the imported
                    # files in the hdf5 cube is the same,
                    # export is not done again.
                    cube = FDCube(
                        self.options[option_key],
                        silent_init=True, no_sort=False,
                        ncpus=self.ncpus,
                        instrument=self.instrument)

                    if already_exported:
                        check_ok = check_exported_cube(export_path, cube)
                        
                if not check_ok or not already_exported:
                    # create mask
                    if mask_key is not None:
                        mask_key += '_{}'.format(camera_index)
                        if mask_key in self.options:
                            warnings.warn('Mask applied: {}'.format(self.options[mask_key]))
                            image_mask = self.read_fits(
                                self.options[mask_key])
                        else: image_mask = None
                    else: image_mask = None
                        
                    cube.export(export_path, force_hdf5=True,
                                overwrite=True,
                                mask=image_mask)

                self.options[option_key + '.hdf5'] = export_path


        def check_exported_cube(export_path, cube):
            with self.open_hdf5(export_path, 'r') as f:
                if 'image_list' in f:
                    new_image_list = f['image_list'][:]
                    old_image_list = np.array(
                        cube.image_list)
                    if (np.size(new_image_list) == np.size(old_image_list)):
                        if (np.all(new_image_list == old_image_list)
                            and f.attrs['dimz'] == len(cube.image_list)):
                            logging.info('Exported HDF5 cube {} check: OK'.format(export_path))
                            return True
            return False
                         

        def store_folder_parameter(option_key, key, camera_index,
                                   mask_key=None):
            value = self.optionfile.get(key, str)
            if value is None : return

            if os.path.exists(value):
                export_images(value, fast_init,
                              option_key, mask_key,
                              camera_index)

            else: raise StandardError(
                'Given path does not exist {}'.format(value))            

        def store_option_parameter(option_key, key, cast, optional=True):
            value = self.optionfile.get(key, cast)
            if value is not None:
                self.options[option_key] = value
                
            elif not optional:
                raise StandardError('option {} must be set'.format(key))

        self.option_file_path = option_file_path
        Tools.__init__(self, instrument=instrument, ncpus=ncpus, silent=silent)

        if overwrite in [True, False]:
            self.overwrite = bool(overwrite)
        else:
            raise ValueError('overwrite must be True or False')
        
        if not silent:
            # First, print ORBS version
            logging.info("ORBS version: %s"%self.__version__)
            logging.info("ORB version: %s"%orb.version.__version__)

            # Print modules versions
            logging.info("Numpy version: %s"%np.__version__)
            logging.info("Scipy version: %s"%scipy.__version__)
            logging.info("Astropy version: %s"%astropy.__version__)
            logging.info("Parallel Python version: %s"%pp.version)
            logging.info("Bottleneck version: %s"%bn.__version__)

            # Print the entire config file for log
            with self.open_file(self._get_config_file_path(), 'r') as conf_file:
                logging.info("Configuration file content:")
                for line in conf_file:
                    logging.info(line[:-1])


        # check params
        if cam not in self.cams:
            raise ValueError('cams must be in {}'.format(self.cams))
        
        # Read option file to get observation parameters
        if not os.path.exists(option_file_path):
            raise StandardError("Option file does not exists !")

        # Print first the entire option file for logging
        op_file = open(option_file_path)
        if not silent:
            logging.info("Option file content :")
            for line in op_file:
                logging.info(line[:-1])

        # record some default options
        self.options["try_catalogue"] = False
        self.options['spectral_calibration'] = True
        self.options['wavenumber'] = False
        self.options['no_sky'] = False

        ##########
        ## Parse the option file to get reduction parameters
        ##########
        self.optionfile = OptionFile(option_file_path)

        # In the case of LASER cube the parameters are set
        # automatically
        if target == 'laser': optional_keys = True
        else:  optional_keys = False
            
        store_option_parameter('object_name', 'OBJECT', str,
                               optional=optional_keys)
        store_option_parameter('filter_name', 'FILTER', str,
                               optional=optional_keys)
        
            
        store_option_parameter('step', 'SPESTEP', float)
        store_option_parameter('step_nb', 'SPESTNB', int)
        store_option_parameter('order', 'SPEORDR', float)
        
        # check step and order
        if 'filter_name' in self.options and target != 'laser':
            _step, _order = FilterFile(
                self.options['filter_name']).get_observation_params()
            if int(self.options['order']) != int(_order):
                warnings.warn('Bad order. Option file tells {} while filter file tells {}. Order parameter replaced by {}'.format(int(self.options['order']), _order, _order))
                self.options['order'] = _order
            if abs(self.options['step'] - float(_step))/float(_step) > 0.1:
                raise StandardError('There is more than 10% difference between the step size in the option file ({}) and the step size recorded in the filter file ({})'.format(self.options['step'], _step))
                
        store_option_parameter('exposure_time', 'SPEEXPT', float)
        store_option_parameter('dark_time', 'SPEDART', float)
        store_option_parameter('obs_date', 'OBSDATE', str)
        store_option_parameter('target_ra', 'TARGETR', str)
        if 'target_ra' in self.options:
            self.options['target_ra'] = orb.utils.astrometry.ra2deg(
                self.options['target_ra'].split(':'))
        store_option_parameter('target_dec', 'TARGETD', str)
        if 'target_dec' in self.options:
            self.options['target_dec'] = orb.utils.astrometry.dec2deg(
                self.options['target_dec'].split(':'))
        store_option_parameter('target_x', 'TARGETX', float)
        store_option_parameter('target_y', 'TARGETY', float)
        store_option_parameter('standard_path', 'STDPATH', str)
        store_option_parameter('phase_maps_path', 'PHASEMAPS', str)
        store_option_parameter('star_list_path_1', 'STARLIST1', str)
        store_option_parameter('star_list_path_2', 'STARLIST2', str)
        store_option_parameter('apodization_function', 'APOD', str)
        store_option_parameter('calibration_laser_map_path', 'CALIBMAP', str)
        store_option_parameter('wavefront_map_path', 'WFMAP', str)
        
        store_option_parameter('try_catalogue', 'TRYCAT', bool)
        store_option_parameter('wavenumber', 'WAVENUMBER', bool)
        store_option_parameter('spectral_calibration', 'WAVE_CALIB', bool)
        store_option_parameter('no_sky', 'NOSKY', bool)
        store_option_parameter('prebinning', 'PREBINNING', int)
        store_option_parameter('source_list_path', 'SOURCE_LIST_PATH', str)
        
        # get image mask
        store_option_parameter('object_mask_path_1', 'OBJMASK1', str)
        store_option_parameter('object_mask_path_2', 'OBJMASK2', str)
        store_option_parameter('std_mask_path_1', 'STDMASK1', str)
        store_option_parameter('std_mask_path_2', 'STDMASK2', str)
        
        
        fringes = self.optionfile.get_fringes()
        if fringes is not None:
            self.options['fringes'] = fringes

        bad_frames = self.optionfile.get_bad_frames()
        if bad_frames is not None:
            self.options['bad_frames'] = bad_frames
            
        # In the case of LASER cube the parameters are set
        # automatically
        if target == 'laser':
            self.options['object_name'] = 'LASER'
            self.options['filter_name'] = 'None'
            if 'order' not in self.options:
                self.options['order'] = self.config['CALIB_ORDER']
            if 'step' not in self.options:
                self.options['step'] = self.config['CALIB_STEP_SIZE']
                
        # If a keyword is the same as a configuration keyword,
        # config option is changed
        for key in self.config.iterkeys():
           if self.optionfile[key] is not None:
               key_type = type(self.config[key])
               self.config.reset(key, self.optionfile.get(key, key_type))
               if not silent:
                   warnings.warn("Configuration option %s changed to %s"%(key, self.config[key]))

        # Calibration laser wavelength is changed if the calibration
        # laser map gives a new calibration laser wavelentgh
        if target != 'laser':
            calib_hdu = self.read_fits(
                self.options['calibration_laser_map_path'],
                return_hdu_only=True)
            if 'CALIBNM' in calib_hdu[0].header:
                self.config['CALIB_NM_LASER'] = calib_hdu[0].header['CALIBNM']
                if not silent:
                    warnings.warn('Calibration laser wavelength (CALIB_NM_LASER) read from calibration laser map header: {}'.format(self.config['CALIB_NM_LASER']))
               
        # Get tuning parameters
        self.tuning_parameters = self.optionfile.get_tuning_parameters()
        for itune in self.tuning_parameters:
            if not silent:
                warnings.warn("Tuning parameter %s changed to %s"%(
                    itune, self.tuning_parameters[itune]))

        if (("object_name" not in self.options)
            or ("filter_name" not in self.options)):
            raise StandardError("The option file needs at least an object name (use keyword : OBJECT) and a filter name (use keyword : FILTER)")
        else:
            self.options["project_name"] = (self.options["object_name"] 
                                            + "_" + self.options["filter_name"])

        
        # get folders paths
        if not silent:
            logging.info('Reading data folders')
        if cam in ['single1', 'full']:
            store_folder_parameter('image_list_path_1', 'DIRCAM1', 1,
                                   mask_key='object_mask_path')
        if cam in ['single2', 'full']:
            store_folder_parameter('image_list_path_2', 'DIRCAM2', 2,
                                   mask_key='object_mask_path')
            
        store_folder_parameter('bias_path_1', 'DIRBIA1', 1)
        store_folder_parameter('bias_path_2', 'DIRBIA2', 2)
        store_folder_parameter('dark_path_1', 'DIRDRK1', 1)
        store_folder_parameter('dark_path_2', 'DIRDRK2', 2)
        if (('dark_path_2' in self.options or 'dark_path_1' in self.options)
            and 'dark_time' not in self.options):
            raise StandardError('Dark integration time must be set (SPEDART) if the path to a dark calibration files folder is given')
            
        store_folder_parameter('flat_path_1', 'DIRFLT1', 1)
        store_folder_parameter('flat_path_2', 'DIRFLT2', 2)
        store_folder_parameter('calib_path_1', 'DIRCAL1', 1)
        store_folder_parameter('calib_path_2', 'DIRCAL2', 2)
        #store_folder_parameter('flat_spectrum_path', 'DIRFLTS', None)
        store_folder_parameter('standard_image_path_1', 'DIRSTD1', 1,
                               mask_key='std_mask_path')
        store_folder_parameter('standard_image_path_2', 'DIRSTD2', 2,
                               mask_key='std_mask_path')
        
        if target == 'laser':
            self.options['image_list_path_1'] = self.options['calib_path_1']
            self.options['image_list_path_2'] = self.options['calib_path_2']
            if 'calib_path_1.hdf5' in self.options:
                self.options['image_list_path_1.hdf5'] = self.options[
                    'calib_path_1.hdf5']
            if 'calib_path_2.hdf5' in self.options:
                self.options['image_list_path_2.hdf5'] = self.options[
                    'calib_path_2.hdf5']

        if 'image_list_path_1' in self.options:
            if fast_init:
                cube1 = FDCube(self.options['image_list_path_1'],
                               silent_init=True,
                               instrument=self.instrument,
                               no_sort=True, ncpus=self.ncpus,
                               params=self.options,
                               config=self.config,
                               camera_index=1,
                               zpd_index=0)
            else:
                cube1 = HDFCube(self.options['image_list_path_1.hdf5'],
                                silent_init=True,
                                instrument=self.instrument,
                                ncpus=self.ncpus,
                                params=self.options,
                                config=self.config,
                                camera_index=1, zpd_index=0)
            
            dimz1 = cube1.dimz
            self.options['bin_cam_1'] = int(cube1.params.binning)
            
            # prebinning
            if 'prebinning' in self.options:
                if self.options['prebinning'] is not None:
                    self.options['bin_cam_1'] = (
                        self.options['bin_cam_1']
                        * self.options['prebinning'])
                    
        if 'image_list_path_2' in self.options:
            if fast_init:
                cube2 = FDCube(self.options['image_list_path_2'],
                               silent_init=True,
                               instrument=self.instrument,
                               no_sort=True, ncpus=self.ncpus,
                               params=self.options,
                               config=self.config,
                               camera_index=2,
                               zpd_index=0)
            else:
                cube2 = HDFCube(self.options['image_list_path_2.hdf5'],
                                silent_init=True,
                                instrument=self.instrument,
                                ncpus=self.ncpus,
                                params=self.options,
                                config=self.config,
                                camera_index=2,
                                zpd_index=0)
            dimz2 = cube2.dimz
            self.options['bin_cam_2'] = int(cube2.params.binning)
            
            # prebinning
            if 'prebinning' in self.options:
                if self.options['prebinning'] is not None:
                    self.options['bin_cam_2'] = (
                        self.options['bin_cam_2']
                        * self.options['prebinning'])
                    
        # Check step number, number of raw images
        if (('image_list_path_1' in self.options)
            and ('image_list_path_2' in self.options)):
            
            if dimz1 != dimz2:
                raise StandardError('The number of images of CAM1 and CAM2 are not the same (%d != %d)'%(dimz1, dimz2))
                
            if self.options['step_nb'] < dimz1:
                if not silent:
                    warnings.warn('Step number option changed to {} because the number of steps ({}) of a full cube must be greater or equal to the number of images given for CAM1 and CAM2 ({})'.format(
                    dimz1, self.options['step_nb'], dimz1))
                self.options['step_nb'] = dimz1

        # get ZPD shift in SITELLE's case
        if self.config["INSTRUMENT_NAME"] == 'SITELLE' and not fast_init:
            
            try: # check if the zpd index has already been computed
                zpd_index = int(self.read_fits(self._get_zpd_index_file_path()))
            except IOError: 
                cube_list = self.options['image_list_path_1']

                cube1 = FDCube(cube_list,
                               silent_init=True,
                               instrument=self.instrument,
                               no_sort=True, ncpus=self.ncpus)


                zpd_found = False
                if target == 'laser':
                    for ik in range(cube1.dimz):
                        sitstep = cube1.get_frame_header(ik)['SITFRING']
                        if sitstep > 0:
                            zpd_index = ik
                            zpd_found = True
                            break
                    if not zpd_found:
                        if not silent:
                            warnings.warn('zpd index could not be found, forced to 25% of the interferogram size')
                        zpd_index = int(cube1.dimz * 0.25)
                else:
                    for ik in range(cube1.dimz):
                        sitstep = cube1.get_frame_header(ik)['SITSTEP']
                        if sitstep == 0:
                            zpd_index = ik
                            zpd_found = True
                            break
                    if not zpd_found: raise StandardError('zpd index could not be found')
                
            self.options['zpd_index'] = zpd_index
            if not silent:
                logging.info('ZPD index: {}'.format(
                    self.options['zpd_index']))
            
            self.write_fits(self._get_zpd_index_file_path(),
                            np.array(self.options['zpd_index']),
                            overwrite=True)
        
        # Init Indexer
        self.indexer = Indexer(data_prefix=self.options['object_name']
                               + '_' + self.options['filter_name'] + '.',
                               instrument=self.instrument)
        self.indexer.load_index()

        # Load roadmap
        if target in self.targets:
            self.target = target
        else:
            raise StandardError('Unknown target type: target must be in {}'.format(self.targets))

        self.roadmap = RoadMap(
            self.config["INSTRUMENT_NAME"].lower(), target, cam, self.indexer)

        # attach methods to roadmap steps
        self.roadmap.attach('compute_alignment_vector',
                            self.compute_alignment_vector)
        self.roadmap.attach('compute_cosmic_ray_map',
                            self.compute_cosmic_ray_map)
        self.roadmap.attach('compute_cosmic_ray_maps',
                            self.compute_cosmic_ray_maps)
        self.roadmap.attach('compute_interferogram',
                            self.compute_interferogram)
        self.roadmap.attach('transform_cube_B',
                            self.transform_cube_B)
        self.roadmap.attach('merge_interferograms',
                            self.merge_interferograms)
        self.roadmap.attach('merge_interferograms_alt',
                            self.merge_interferograms_alt)
        self.roadmap.attach('correct_interferogram',
                            self.correct_interferogram)
        self.roadmap.attach('compute_calibration_laser_map',
                            self.compute_calibration_laser_map)
        self.roadmap.attach('compute_phase_maps',
                            self.compute_phase_maps)
        self.roadmap.attach('compute_spectrum',
                            self.compute_spectrum)
        self.roadmap.attach('calibrate_spectrum',
                            self.calibrate_spectrum)
        self.roadmap.attach('extract_source_interferograms',
                            self.extract_source_interferograms)
        self.roadmap.attach('compute_source_spectra',
                            self.compute_source_spectra)        
        
    def _get_calibration_standard_fits_header(self):

        if 'standard_path' in self.options:
            std_path = self.options['standard_path']
            std_name = self._get_standard_name(std_path)
            hdr = list()
            hdr.append(('COMMENT','',''))
            hdr.append(('COMMENT','Calibration standard parameters',''))
            hdr.append(('COMMENT','-------------------------------',''))
            hdr.append(('COMMENT','',''))
            hdr.append(('STDNAME', std_name,
                        'Name of the standard star'))
            std_path = os.path.basename(std_path)[
                :orb.constants.FITS_CARD_MAX_STR_LENGTH]
            hdr.append(('STDPATH', std_path,
                        'Path to the standard star file'))
            return hdr
        else: return None
            
    def _get_calibration_laser_fits_header(self):
        """Return the header corresponding to the calibration laser
        that can be added to the created FITS files."""
        hdr = list()
        hdr.append(('COMMENT','',''))
        hdr.append(('COMMENT','Calibration laser parameters',''))
        hdr.append(('COMMENT','----------------------',''))
        hdr.append(('COMMENT','',''))
        hdr.append(('CALIBNM',self.config["CALIB_NM_LASER"],
                    'Wavelength of the calibration laser in nm'))
        hdr.append(('CALIBOR',self.config["CALIB_ORDER"],
                    'Folding order of the calibration laser cube'))
        hdr.append(('CALIBST',self.config["CALIB_STEP_SIZE"],
                    'Step size of the calibration laser cube'))
        return hdr

    def _get_project_fits_header(self, camera_number=None):
        """Return the header of the project that can be added to the
        created FITS files.

        :param camera_number: Number of the camera (can be 0, 1 or 2)
        """
        hdr = list()
        hdr.append(('COMMENT','',''))
        hdr.append(('COMMENT','ORBS',''))
        hdr.append(('COMMENT','----',''))
        hdr.append(('COMMENT','',''))
        hdr.append(('ORBSVER', self.__version__, 'ORBS version'))
        option_file_name = os.path.basename(self.option_file_path)[
            :orb.constants.FITS_CARD_MAX_STR_LENGTH]
        hdr.append(('OPTNAME', option_file_name,
                    'Name of the option file'))
        
        hdr.append(('COMMENT','',''))
        hdr.append(('COMMENT','Observation parameters',''))
        hdr.append(('COMMENT','----------------------',''))
        hdr.append(('COMMENT','',''))
        if "object_name" in self.options:
            hdr.append(('OBJECT', self.options["object_name"], 'Object Name'))
        if "exposure_time" in self.options:
            hdr.append(('EXPTIME', self.options["exposure_time"], 
                        'Exposure time'))
        if "filter_name" in self.options:
            hdr.append(('FILTER', self.options["filter_name"], 
                        'Name of filter used during the observation'))
        if "obs_date" in self.options:
            hdr.append(('DATE-OBS', self.options["obs_date"], 
                        'Date of the observation'))
        if "order" in self.options:
            hdr.append(('ORDER', self.options["order"], 
                        'Order of spectral folding'))
        if "step" in self.options:
            hdr.append(('STEP', self.options["step"], 
                        'Step size in nm'))
        if "step_nb" in self.options:
            hdr.append(('STEPNB', self.options["step_nb"], 
                        'Number of steps'))
        if "target_ra" in self.options:
            hdr.append(('TARGETR', orb.utils.astrometry.deg2ra(
                self.options["target_ra"],
                string=True),
                        'Target Right Ascension'))
        if "target_dec" in self.options:
            hdr.append(('TARGETD', orb.utils.astrometry.deg2dec(
                self.options["target_dec"],
                string=True), 
                        'Target Declination'))
        if "target_x" in self.options:
            hdr.append(('TARGETX', self.options["target_x"], 
                        'Target estimated X coordinate'))
        if "target_y" in self.options:
            hdr.append(('TARGETY', self.options["target_y"], 
                        'Target estimated Y coordinate'))

        hdr.append(('COMMENT','',''))
        hdr.append(('COMMENT','Image Description',''))
        hdr.append(('COMMENT','-----------------',''))
        hdr.append(('COMMENT','',''))
        if camera_number is None:
            if "camera_number" in self.options:
                camera_number = self.options["camera_number"]
        if camera_number is not None:
            if camera_number != 0:
                hdr.append(('CAMERA', "CAM%d"%camera_number, 
                            'Camera number'))
            else:
                hdr.append(('CAMERA', "MERGED_DATA", 
                            'Merged data from CAM1 and CAM2'))       
        if camera_number is not None:
            if ((camera_number == 1) 
                or (camera_number == 0)):  
                if "bin_cam_1" in self.options:
                    hdr.append(('BINNING', self.options["bin_cam_1"], 
                                'Binning of the camera'))
            if camera_number == 2:  
                if "bin_cam_2" in self.options:
                    hdr.append(('BINNING', self.options["bin_cam_2"], 
                                'Binning of the camera'))
        return hdr
            
    def _get_project_dir(self):
        """Return the path to the project directory depending on 
        the project name."""
        return os.curdir + os.sep + self.options["project_name"] + os.sep

    def _get_data_dir(self, data_kind):
        """Return the path to the data directory depending on the
        kind of data and the project name.

        :param data_kind: Integer, depends on the type of data and 
          can be:

          * 0: merged data (resulting from merging data from camera
            1 and 2)
          * 1 or 2: data relative to each camera
          * 3: spectral data ready for analysis.
        """
        if (data_kind == 1) or (data_kind == 2):
            return self._get_project_dir() + "CAM" + str(data_kind) + os.sep
        if (data_kind == 0):
            return self._get_project_dir() + "MERGED" + os.sep
        if (data_kind == 3):
            return self._get_project_dir() + "ANALYSIS" + os.sep

    def _get_data_prefix(self, data_kind):
        """Return the prefix of the data files name

        :param data_kind: Integer, depends on the type of data and 
          can be:

          * 0: merged data (resulting from merging data from camera
            1 and 2)
          * 1 or 2: data relative to each camera
          * 3: spectral data ready for analysis.
        """
        if (data_kind == 1) or (data_kind == 2):
            return self._get_data_dir(data_kind) + self.options["project_name"] + ".cam" + str(data_kind) + "."
        if (data_kind == 0):
            return self._get_data_dir(data_kind) + self.options["project_name"] + ".merged."
        if (data_kind == 3):
            return self._get_data_dir(data_kind) + self.options["project_name"] + ".analysis."

    def _get_root_data_path_hdr(self, camera_number):
        """Return the beginning of the path to a file at the root of
        the reduction folder.

        :param camera_number: Camera number (must be 1, 2 or 0 for
          merged data).
        """
        if camera_number == 1: cam = 'cam1'
        elif camera_number == 2: cam = 'cam2'
        elif camera_number == 0: cam = 'merged'
        else: raise StandardError('Camera number must be 0, 1, or 2')
        
        return ('.' + os.sep + self.options['object_name']
                + '_' +  self.options['filter_name'] + '.' + cam + '.')

    def _get_zpd_index_file_path(self):
        """Return path to the zpd index file.        
        """
        return self._get_root_data_path_hdr(1) + 'zpd_index.fits'

    def _get_calibration_laser_map_path(self, camera_number):
        """Return path to the calibration laser map from a flat cube.
        
        :param camera_number: Camera number (must be 1 or 2).
        """
        return (self._get_root_data_path_hdr(camera_number)
                + 'calibration_laser_map.fits')
    
    def _get_calibrated_spectrum_cube_path(self, camera_number, apod,
                                           wavenumber=False,
                                           spectral_calibration=True):
        """Return path to the calibrated spectrum cube resulting of the
        reduction process.
        
        :param camera_number: Camera number (must be 1, 2 or 0 for
          merged data).

        :param apod: Apodization function name to be added to the
          path.

        :param wavenumber: (Optional) If True the spectral axis of the
          cube is considered to be a wavenumber axis. If False it is
          considered to be a wavelength axis (default False).

        :param spectral_calibration: (Optional) If True the cube is
          calibrated in wavelength/wavenumber (default True).
        """
        if wavenumber: wave_type = 'cm1'
        else: wave_type = 'nm'
        if spectral_calibration: calib = ''
        else: calib = '.uncalib'
            
        return (self._get_root_data_path_hdr(camera_number)
                + wave_type + '.' + apod + calib + '.hdf5')

    def _get_standard_spectrum_path(self, camera_number):
        """Return path to the standard star spectrum
        
        :param camera_number: Camera number (must be 1, 2 or 0 for
          merged data).
        """
        return (self._get_root_data_path_hdr(camera_number)
                + 'standard_spectrum.fits')

    def _get_extracted_source_spectra_path(self, camera_number):
        """Return path to the source spectra
        
        :param camera_number: Camera number (must be 1, 2 or 0 for
          merged data).
        """
        return (self._get_root_data_path_hdr(camera_number)
                + 'extracted_source_spectra.fits')
    
    def _init_raw_data_cube(self, camera_number):
        """Return instance of :class:`orbs.process.RawData` class

        :param camera_number: Camera number (can be either 1 or 2).
        """
        if (camera_number == 1):
            if ("image_list_path_1" in self.options):
                self.indexer.set_file_group('cam1')
                cube = RawData(
                    self.options["image_list_path_1.hdf5"],
                    params=self.options,
                    config=self.config,
                    data_prefix=self._get_data_prefix(1),
                    project_header=self._get_project_fits_header(1),
                    overwrite=self.overwrite,
                    tuning_parameters=self.tuning_parameters,
                    indexer=self.indexer,
                    instrument=self.instrument,
                    ncpus=self.ncpus)
            else:
                raise StandardError("No image list file for camera 1 given, please check option file")
        elif (camera_number == 2):
            if ("image_list_path_2" in self.options):
                self.indexer.set_file_group('cam2')
                cube = RawData(
                    self.options["image_list_path_2.hdf5"],
                    params=self.options,
                    config=self.config,
                    data_prefix=self._get_data_prefix(2),
                    project_header=self._get_project_fits_header(2),
                    overwrite=self.overwrite,
                    tuning_parameters=self.tuning_parameters,
                    indexer=self.indexer,
                    instrument=self.instrument,
                    ncpus=self.ncpus)
            else:
                raise StandardError("No image list file for camera 2 given, please check option file")
        else:
            raise StandardError("Camera number must be either 1 or 2, please check 'camera_number' parameter")
            return None
        return cube

    def _init_astrometry(self, cube, camera_number, standard_star=False):
        """Init Astrometry class.

        The Astrometry class is used for star detection and star fitting
        (position and photometry)
        
        :param cube: an orbs.Cube instance

        :param camera_number: Camera number (can be 1, 2 or 0 
          for merged data).

        :param standard_star: If True target x and y are derived fom
          STD_X and STD_Y keywords in the option file.
        
        :return: :py:class:`orb.astrometry.Astrometry` instance

        .. seealso:: :py:class:`orb.astrometry.Astrometry`
        """

        cube.params['camera_index'] = camera_number

        if 'target_x' in cube.params and 'target_y' in cube.params:
            
            if camera_number == 2:
                # get binning factor for each camera
                if "bin_cam_1" in self.options: 
                    bin_cam_1 = self.options["bin_cam_1"]
                else:
                    raise StandardError("No binning for the camera 1 given")

                if "bin_cam_2" in self.options: 
                    bin_cam_2 = self.options["bin_cam_2"]
                else:
                    raise StandardError("No binning for the camera 2 given")

                # get initial shift
                init_dx = self.config["INIT_DX"] / bin_cam_2
                init_dy = self.config["INIT_DY"] / bin_cam_2
                pix_size_1 = self.config["PIX_SIZE_CAM1"]
                pix_size_2 = self.config["PIX_SIZE_CAM2"]
                zoom = (pix_size_2 * bin_cam_2) / (pix_size_1 * bin_cam_1)
                xrc = cube.dimx / 2.
                yrc = cube.dimy / 2.
                
                target_x, target_y = orb.cutils.transform_A_to_B(
                    cube.params.target_x, cube.params.target_y,
                    init_dx, init_dy,
                    self.config["INIT_ANGLE"],
                    0., 0., xrc, yrc, zoom, zoom)
                
                cube.params['target_x'] = target_x
                cube.params['target_y'] = target_y
               
                
        wcs_rotation = self._get_wcs_rotation(camera_number)
        
        cube.params['wcs_rotation'] = wcs_rotation

        # load SIP file in ORB's data/ folder
        sip = self.load_sip(self._get_sip_file_path(camera_number))

        return cube.get_astrometry(sip=sip)


    def _get_wcs_rotation(self, camera_number):
        """Return wcs rotation parameter, given the camera number"""
        if camera_number == 2:
            wcs_rotation = (self.config["WCS_ROTATION"]
                            - self.config["INIT_ANGLE"])
        else:
            wcs_rotation = self.config["WCS_ROTATION"]
        return wcs_rotation
        

    def _get_interfero_cube_path(self, camera_number, corrected=False):
        """Return the path to the interferogram cube for each camera
        or the merged interferogram (camera_number = 0)

        :param camera_number: Camera number (can be 1, 2 or 0 
          for merged data).
        """
        if camera_number == 0:
            return self.indexer.get_path(
                'merged.merged_interfero_cube', err=True)
        elif camera_number == 1:
            if corrected:
                return self.indexer.get_path('cam1.corr_interfero_cube', err=True)
            else:
                return self.indexer.get_path('cam1.interfero_cube', err=True)
        elif camera_number == 2:
            if corrected:
                return self.indexer.get_path('cam2.corr_interfero_cube', err=True)
            else:
                return self.indexer.get_path('cam2.interfero_cube', err=True)
        else: raise StandardError('Camera number must be 0, 1 or 2')


    def _get_phase_fit_order(self):
        """Return phase fit order from the filter file"""
        fit_order = FilterFile(
            self.options["filter_name"]).get_phase_fit_order()
        if fit_order is not None:
            return fit_order
        else:
            return self.config["PHASE_FIT_DEG"]

    def _get_source_list(self):
        """Return the list of sources positions (option file keyword
        SOURCE_LIST_PATH), the position of the standard target if
        target is set to 'standard', the list of autodetected stars of
        target is set to 'extphase'.
        """

        if self.target == 'standard':
            source_list = [[self.options['target_x'], self.options['target_y']]]
            logging.info('Standard target position loaded'.format(len(source_list)))
            
        elif self.target == 'extphase':
            cube = self._init_raw_data_cube(1)
            star_list_path, mean_fwhm_pix = self.detect_stars(
            cube, 0, return_fwhm_pix=True)
            source_list = orb.utils.astrometry.load_star_list(star_list_path)
            
        else: # standard target
            source_list = list()
            if not 'source_list_path' in self.options:
                raise StandardError('A list of sources must be given (option file keyword SOURCE_LIST_PATH)')
            with self.open_file(self.options['source_list_path'], 'r') as f:
                for line in f:
                    x,y = line.strip().split()[:2]
                    source_list.append([float(x),float(y)])
            
            logging.info('Loaded {} sources to extract'.format(len(source_list)))
  
            
        return source_list
        
    def _get_standard_name(self, std_path):
        """Return value associated to keyword 'OBJECT'

        :param std_path: Path to the file containing the standard.    
        """
        if 'hdf5' in std_path:
            cube = HDFCube(std_path, instrument=self.instrument, ncpus=self.ncpus)
            hdr = cube.get_frame_header(0)
        else:
            hdr = self.read_fits(std_path, return_hdu_only=True)[0].header
        return ''.join(hdr['OBJECT'].strip().split()).upper()
        
   
    def _is_balanced(self, camera_number):
        """Return True if the camera is balanced.

        :param camera_number: Camera number, can be 0, 1 or 2.
        """
        if (camera_number == 1):
            if self.config["BALANCED_CAM"] == 1:
                balanced = True
            else: balanced = False
        elif (camera_number == 2):
            if self.config["BALANCED_CAM"] == 2:
                balanced = True
            else: balanced = False
        elif (camera_number == 0):
            balanced = True
        else:
            raise StandardError(
                "Please choose a correct camera number : 0, 1 or 2")
        return balanced

    def set_init_angle(self, init_angle):
        """Change config variable :py:const:`orbs.orbs.Orbs.INIT_ANGLE`. 

        You can also change it by editing the file
        :file:`orbs/data/config.orb`.

        .. note:: This value is modified only for this instance of
           Orbs class. The initial value stored in the file
           'config.orb' will be restored at the next initialisation of
           the class.

        :param init_angle: the new value 
           of :py:const:`orbs.orbs.Orbs.INIT_ANGLE`
        """
        self.config["INIT_ANGLE"] = init_angle

    def create_bad_frames_vector(self, camera_number, bad_frames_list=None):
        """Create a bad frames vector from the collected bad frames
        vectors before computing the spectrum.

        Bad frames vectors are created by some processes.
        
        :param camera_number: Camera number.

        :param bad_frames_list: (Optional) List of bad frames indexes
          that must be added to the bad frames vector (default None).
        """
        interf_cube = HDFCube(self._get_interfero_cube_path(camera_number),
                              instrument=self.instrument, ncpus=self.ncpus)
        bad_frames_vector = np.zeros(interf_cube.dimz)
        
        if camera_number == 0:
            # Get bad frames vector created by the merge function
            merge_bad_frames_path = self.indexer['merged.bad_frames_vector']
            if merge_bad_frames_path is not None:
                if os.path.exists(merge_bad_frames_path):
                    merge_bad_frames = self.read_fits(merge_bad_frames_path)
                    bad_frames_vector[np.nonzero(merge_bad_frames)] = 1

        if bad_frames_list is not None:
            if np.all(np.array(bad_frames_list) >= 0):
                bad_frames_list = orb.utils.vector.correct_bad_frames_vector(
                    bad_frames_list, interf_cube.dimz)
                bad_frames_vector[bad_frames_list] = 1
            else:
                raise StandardError('Bad indexes in the bad frame list')
    
        return bad_frames_vector


    def start_reduction(self, apodization_function=None, start_step=0,
                        phase_correction=True, phase_cube=False,
                        alt_merge=False,
                        save_as_quads=False,
                        add_frameB=True, filter_correction=True,
                        wcs_calibration=True):
        
        """Run the whole reduction process for two cameras using
        default options 

        :param apodization_function: (Optional) Name of the apodization
          function used during the spectrum computation (default None).

        :param start_step: (Optional) Starting step number. Use it to
          cover from an error at a certain step without having to
          run the whole process one more time (default 0).  
           
        :param phase_correction: (Optional) If False, no phase
          correction will be done and the resulting spectrum will be
          the absolute value of the complex spectrum (default True).

        :param phase_cube: (Optional) If True, the phase cube will
          be computed instead of the spectrum (default False).

        :param alt_merge: (Optional) If True, alternative merging
          process will be choosen. Star photometry is not used during
          the merging process. Might be more noisy but useful if for
          some reason the correction vectors cannot be well computed
          (e.g. not enough good stars, intense emission lines
          everywhere in the field)

        :param save_as_quads: (Optional) If True, final calibrated
          spectrum is saved as quadrants instead of being saved as a
          full cube. Quadrants can be read independantly. This option
          is useful for big data cubes (default False).
    
        :param add_frameB: (Optional) If False use the images of the
          camera 2 only to correct for the variations of the sky
          transmission. Default True.

        :param filter_correction: (Optional) If True, spectral cube is
          corrected for filter during calibration step (default True).

        :param wcs_calibration: (Optional) If True, WCS calibratio is
          intented during calibration step (default True).
        """
        # save passed kwargs
        local_kwargs = locals()

        # launch the whole process    
        for istep in range(self.roadmap.get_road_len()):
            f, args, kwargs = self.roadmap.get_step_func(istep)
            
            if f is not None:
                if istep >= start_step:
                    # construct kwarg dict to be passed to the process
                    # step
                    kwargs_dict = {}
                    for kwarg in kwargs:
                        if kwargs[kwarg] == 'undef':
                            if kwarg in local_kwargs:
                                kwargs_dict[kwarg] = local_kwargs[kwarg]
                            else:
                                raise StandardError(
                                    'kwarg {} not defined'.format(kwarg))
                        else:
                            kwargs_dict[kwarg] = kwargs[kwarg]

                    # launch process step        
                    logging.info('run {}({}, {})'.format(
                        f.__name__, str(args), str(kwargs)))
                    
                    f(*args, **kwargs_dict)
                    
            else: raise StandardError("No function attached to step '{}'".format(
                self.roadmap.road[istep]['name']))


        # Once the whole process has been done, final data can be exported
        if self.roadmap.cams == 'single1': cam = 1
        elif self.roadmap.cams == 'single2': cam = 2
        elif self.roadmap.cams == 'full': cam = 0
        
        if (self.target == 'object' or self.target == 'nostar'
            or self.target == 'raw' or self.target == 'extphase'
            or self.target == 'nophase' ):
            self.export_calibrated_spectrum_cube(cam)
        if self.target == 'flat': self.export_flat_phase_map(cam)
        if self.target == 'laser': self.export_calibration_laser_map(cam)
        if self.target == 'standard': self.export_standard_spectrum(
            cam, auto_phase=True)
        if self.target == 'sources': self.export_source_spectra(cam)


    def detect_stars(self, cube, camera_number, 
                     saturation_threshold=None, return_fwhm_pix=False,
                     all_sources=False, realign=False, deep_frame=None):
        """Detect stars in a cube and save the star list in a file.

        This method is a simple wrapper around
        :py:meth:`orb.astrometry.Astrometry.detect_stars`

        If a path to a list of star is given in the option file
        (keyword STARLIST1 or STARLIST2), no automatic detection is
        done and the the given list is returned. In this case the mean
        FWHM returned is the FWHM given in the configuration file.
        
        :param cube: an orbs.Cube instance
        
        :param min_star_number: Minimum number of star to detect

        :param saturation_threshold: (Optional) Number of counts above
          which the star can be considered as saturated. Low by
          default because at the ZPD the intensity of a star can be
          twice the intensity far from it. If None the default
          configuration value SATURATION_THRESHOLD / 2 is used (default
          None).

        :param return_fwhm_pix: (Optional) If True, the returned fwhm
          will be given in pixels instead of arcseconds (default
          False).

        :param all_sources: (Optional) If True, all point sources are
          detected regardless of their FWHM (galaxies, HII regions,
          filamentary knots and stars might be detected).

        :param realign: (Optional) Realign frames with a
          cross-correlation algorithm (default False). Much better if
          used on a small number of frames.

        :param deep_frame: (Optional) If a deep frame is passed it is
          used directly instead of creating a new one.
        
        :return: Path to a star list, mean FWHM of stars in arcseconds.

        .. seealso:: :py:meth:`orb.astrometry.Astrometry.detect_stars`
        """
        # check first if a star list has been passed to the option file
        if ((camera_number == 1
            or camera_number == 0)
            and 'star_list_path_1' in self.options):
            logging.info('Using external star list: %s'%self.options['star_list_path_1'])
            star_list_path = self.options['star_list_path_1']
            mean_fwhm = self.config['INIT_FWHM']
            refit = True
            
        elif (camera_number == 2 and 'star_list_path_2' in self.options):
            logging.info('Using external star list: %s'%self.options['star_list_path_2'])
            star_list_path = self.options['star_list_path_2']
            mean_fwhm = self.config['INIT_FWHM']
            refit = True
        else: refit = False

        if refit:
            logging.info('Fitting manual star list')
            astrom = self._init_astrometry(cube, camera_number)
            astrom.load_star_list(star_list_path)
            fit_results = astrom.fit_stars_in_frame(0)
            star_list = fit_results.get_star_list()
            star_list_fit_path = star_list_path + '.fit'
            with open(star_list_fit_path, 'w') as f:
                for istar in range(np.array(star_list).shape[0]):
                    f.write('{} {}\n'.format(
                        star_list[istar, 0], star_list[istar, 1]))
            star_list_path = star_list_fit_path
            mean_fwhm = orb.utils.stats.robust_median(fit_results[:,'fwhm_arc'])
            logging.info('Mean FWHM: {} arcsec'.format(mean_fwhm))
            
        
        # else try to auto-detect stars
        else:
            if saturation_threshold is None:
                saturation_threshold = self.config['SATURATION_THRESHOLD'] / 2.
                
            logging.info('Autodetecting stars')
            astrom = self._init_astrometry(cube, camera_number)
            astrom.deep_frame = deep_frame
            if not all_sources:
                star_list_path, mean_fwhm = astrom.detect_stars(
                    min_star_number=self.config['DETECT_STAR_NB'],
                    saturation_threshold=saturation_threshold,
                    try_catalogue=self.options['try_catalogue'],
                    realign=realign)
            else:
                star_list_path, mean_fwhm = astrom.detect_all_sources()
                    
            if return_fwhm_pix: mean_fwhm = astrom.arc2pix(mean_fwhm)
            del astrom

        return star_list_path, mean_fwhm


    def compute_alignment_vector(self, camera_number):
        """Run the computation of the alignment vector.

        If no path to a star list file is given 
        use: :py:meth:`orb.astrometry.Astrometry.detect_stars` 
        method to detect stars.

        :param camera_number: Camera number (can be either 1 or 2).
              
        .. seealso:: :meth:`orb.astrometry.Astrometry.detect_stars`
        .. seealso:: :meth:`process.RawData.create_alignment_vector`
        """        
        cube = self._init_raw_data_cube(camera_number)
        
        perf = Performance(cube, "Alignment vector computation", camera_number,
                           instrument=self.instrument)
        
        star_list_path, mean_fwhm_arc = self.detect_stars(
            cube, camera_number)

        cube.create_alignment_vector(
            star_list_path, 
            profile_name='gaussian') # gaussian is better for alignment tasks
        
        perf_stats = perf.print_stats()
        del cube, perf
        return perf_stats

    def compute_cosmic_ray_maps(self):
        """Run computation of cosmic ray maps.
        
        .. seealso:: :meth:`process.CosmicRayDetector.create_cosmic_ray_maps`
        """
        cube, star_list_path = self.compute_alignment_parameters(
            no_star=False, raw=True,
            return_star_list=True)

        perf = Performance(
            cube.cube_A, "Cosmic ray map computation", 0,
            instrument=self.instrument)

        alignment_vector_path_1 = self.indexer['cam1.alignment_vector']

        cube.create_cosmic_ray_maps(alignment_vector_path_1, star_list_path,
                                    self._get_init_fwhm_pix())
        perf_stats = perf.print_stats()
        del cube, perf
        return perf_stats

    def _get_init_fwhm_pix(self):
        """Return init FWHM of the stars in pixels"""
        return (float(self.config['INIT_FWHM'])
                * float(self.config['CAM1_DETECTOR_SIZE_X'])
                / float(self.config['FIELD_OF_VIEW_1'])
                / 60.)

    def compute_cosmic_ray_map(self, camera_number, z_coeff=3.):
        """Run computation of cosmic ray map.

        :param camera_number: Camera number (can be either 1 or 2).
        
        :param z_coeff: (Optional) Threshold coefficient for cosmic
          ray detection, lower it to detect more cosmic rays (default
          : 3.).

        .. seealso:: :meth:`process.RawData.create_cosmic_ray_map`
        """
        raise NotImplementedError('Used only with SpIOMM. SHould be reimplemented based on stuff done before level2')

    def check_bad_frames(self, camera_number):
        """Check for bad frames using the number of detected cosmic
        rays. If too much cosmic rays are detected the frame is
        considered as bad

        :param camera_number: Camera number (can be either 1 or 2).
        
        .. seealso:: :meth:`process.RawData.check_bad_frames`
        """
        cube = self._init_raw_data_cube(camera_number)
        bad_frames = cube.check_bad_frames()
        del cube
        return bad_frames

    def compute_interferogram(self, camera_number, 
                              z_range=[], combine='average', reject='avsigclip',
                              flat_smooth_deg=0, no_corr=False):
        """Run the computation of the corrected interferogram from raw
           frames

        :param camera_number: Camera number (can be either 1 or 2).
          
        :param z_range: (Optional) 1d array containing the index of
          the frames to be computed.

        :param reject: (Optional) Rejection operation for master
          frames creation. Can be 'sigclip', 'avsigclip', 'minmax' or
          None (default 'avsigclip'). See
          :py:meth:`process.RawData._create_master_frame`.
        
        :param combine: (Optional) Combining operation for master
          frames creation. Can be 'average' or 'median' (default
          'average'). See
          :py:meth:`process.RawData._create_master_frame`.

        :param flat_smooth_deg: (Optional) If > 0 smooth the master
          flat (help removing possible fringe pattern) (default
          0). See :py:meth:`process.RawData._load_flat`.
         
        :param no_corr: (Optional) If True, no correction is made and
          the interferogram cube is just a copy of the raw cube
          (default False).

        .. seealso:: :py:meth:`process.RawData.correct`

        """
        cube = self._init_raw_data_cube(camera_number)

        if no_corr:
            if not os.path.exists(os.path.abspath(cube._get_interfero_cube_path())):
                os.symlink(os.path.abspath(self.options["image_list_path_1.hdf5"]),
                           os.path.abspath(cube._get_interfero_cube_path()))
                self.indexer['interfero_cube'] = cube._get_interfero_cube_path()
            else:
                warnings.warn('interferogram symlink already created')

            return
        
        perf = Performance(cube, "Interferogram computation", camera_number,
                           instrument=self.instrument)

        bias_path = None
        dark_path = None
        flat_path = None

        if camera_number == 1: 
            if "dark_path_1" in self.options:
                dark_path = self.options["dark_path_1"]
            else:
                warnings.warn("No path to dark frames given, please check the option file.")
            if "bias_path_1" in self.options:
                bias_path = self.options["bias_path_1"]
            else:
                warnings.warn("No path to bias frames given, please check the option file.")
            if "flat_path_1" in self.options:
                flat_path = self.options["flat_path_1"]
            else:
                warnings.warn("No path to flat frames given, please check the option file.")

            optimize_dark_coeff = self.config['OPTIM_DARK_CAM1']
            
        if camera_number == 2: 
            if "dark_path_2" in self.options:
                dark_path = self.options["dark_path_2"]
            else:
                warnings.warn("No path to dark frames given, please check the option file.")
            if "bias_path_2" in self.options:
                bias_path = self.options["bias_path_2"]
            else:
                warnings.warn("No path to bias frames given, please check the option file.")
            if "flat_path_2" in self.options:
                flat_path = self.options["flat_path_2"]
            else:
                warnings.warn("No path to flat frames given, please check the option file.")

            optimize_dark_coeff = self.config['OPTIM_DARK_CAM2']
        
        if "bad_frames" in self.options:
            bad_frames_vector = self.options["bad_frames"]
        else:
            bad_frames_vector = []

        if self.config["INSTRUMENT_NAME"] == 'SITELLE':
            if camera_number == 1:
                cr_map_cube_path = self.indexer.get_path(
                    'cr_map_cube_1', 0)
            elif camera_number == 2:
                cr_map_cube_path = self.indexer.get_path(
                    'cr_map_cube_2', 0)
            logging.info('Cosmic ray map: {}'.format(
                cr_map_cube_path))
        else:
            cr_map_cube_path = None
        cube.correct(
            bias_path=bias_path, dark_path=dark_path, 
            flat_path=flat_path, alignment_vector_path=None,
            cr_map_cube_path=cr_map_cube_path,
            bad_frames_vector=bad_frames_vector, 
            optimize_dark_coeff=optimize_dark_coeff,
            z_range=z_range, combine=combine, reject=reject,
            flat_smooth_deg=flat_smooth_deg)
        perf_stats = perf.print_stats()
        del cube, perf
        return perf_stats

    def transform_cube_B(self, interp_order=1, no_star=False, laser=False):
        """Calculate the alignment parameters of the camera 2
        relatively to the first one. Transform the images of the
        camera 2 using linear interpolation by default.
    
        :param interp_order: (Optional) Interpolation order (Default 1.).

        :param no_star: (Optional) If the cube does not contain any star, the
          transformation is made using the default alignment
          parameters (recorded in the configuration file :
          'data/config.orb') (default False).

        :param laser: (Optional) If the cube is a laser source, the
          frames can be aligned with a brute force algorithm (default
          False).
                         
        .. seealso:: :py:meth:`process.InterferogramMerger.transform`
        """
        if laser: no_star = True

        # compute alignment parameters
        cube = self.compute_alignment_parameters(no_star=no_star, laser=laser)

        perf = Performance(cube.cube_B, "Cube B transformation", 2,
                           instrument=self.instrument)
        
        # transform frames of cube B
        cube.transform(interp_order=interp_order)
        
        perf_stats = perf.print_stats()
        del cube, perf
        return perf_stats

    def compute_alignment_parameters(self, no_star=False, raw=False,
                                     return_star_list=False, laser=False):
        """Compute alignement parameters between cubes and return a
        :py:class:`process.InterferogramMerger instance`.

        :param no_star: (Optional) If the cube does not contain any star, the
          transformation is made using the default alignment
          parameters (recorded in the configuration file :
          'data/config.orb') (default False).

        :param return_star_list: (Optional) If True, the star list
          used is returned with the cube. This option is incompatible
          with no_star option (default False).
    
        :param laser: (Optional) If the cube is a laser source, the
          frames can be aligned with a brute force algorithm (default
          False).

        .. seealso:: :py:meth:`process.InterferogramMerger.find_alignment`
        """
        if laser: no_star = True

        if no_star and return_star_list:
            raise StandardError('return_star_list is incompatible with no_star')
        
        # get binning factor for each camera
        if "bin_cam_1" in self.options: 
            bin_cam_1 = self.options["bin_cam_1"]
        else:
            raise StandardError("No binning for the camera 1 given")

        if "bin_cam_2" in self.options: 
            bin_cam_2 = self.options["bin_cam_2"]
        else:
            raise StandardError("No binning for the camera 2 given")

        # get interferograms frames paths
        if raw:
            interf_cube_path_1 = self.options["image_list_path_1.hdf5"]
            interf_cube_path_2 = self.options["image_list_path_2.hdf5"]
        else:
            interf_cube_path_1 = self.indexer['cam1.interfero_cube']
            interf_cube_path_2 = self.indexer['cam2.interfero_cube']

        # detect stars in cube 1
        if not no_star:
            cube1 = self._init_raw_data_cube(1)
            star_list_path_1, mean_fwhm_1_arc = self.detect_stars(
                cube1, 0, saturation_threshold=self.config['SATURATION_THRESHOLD'])
            del cube1            
        else:
            star_list_path_1 = None
            

        # Init InterferogramMerger class
        self.indexer.set_file_group('merged')
        if raw:
            ComputingClass = CosmicRayDetector
        else:
            ComputingClass = InterferogramMerger

        params = dict(self.options) 
        params['wcs_rotation'] = self._get_wcs_rotation(0)

        cube = ComputingClass(
            interf_cube_path_1, interf_cube_path_2,
            bin_A=bin_cam_1, bin_B=bin_cam_2,
            data_prefix=self._get_data_prefix(0),
            project_header=self._get_project_fits_header(0),
            cube_A_project_header=self._get_project_fits_header(1),
            cube_B_project_header=self._get_project_fits_header(2),
            overwrite=self.overwrite,
            tuning_parameters=self.tuning_parameters,
            indexer=self.indexer,
            instrument=self.instrument,
            ncpus=self.ncpus,
            params=params,
            config=self.config)

        # find alignment coefficients
        
        if not no_star:
            cube.find_alignment(
                star_list_path_1,
                combine_first_frames=raw)
        else:
            if laser:
                raise NotImplementedError('init_dx, init_dy and init_angle must be defined in find_laser_alignment itself')
                cube.find_laser_alignment(
                    init_dx, init_dy, self.config["INIT_ANGLE"])

            logging.info("Alignment parameters: {} {} {} {} {}".format(
                cube.dx, cube.dy, cube.dr, cube.da, cube.db))

        if return_star_list:
            return cube, star_list_path_1
        else:
            return cube
        

    def merge_interferograms_alt(self, add_frameB=True):
        
        """Alternative merge the images of the camera 1 with the
        transformed images of the camera 2.

        Star photometry is not used during the merging process. Might
        be more noisy but useful if for some reason the correction
        vectors cannot be well computed (e.g. not enough good stars,
        intense emission lines everywhere in the field)
        
        :param add_frameB: (Optional) If False use the images of the
          camera 2 only to correct for the variations of the sky
          transmission. Default True.
          
        .. seealso::
          :meth:`process.InterferogramMerger.alternative_merge`
        """
        
        # get frame list paths
        interf_cube_path_1 = self.indexer.get_path(
            'cam1.interfero_cube', err=True)
        interf_cube_path_2 = self.indexer.get_path(
            'merged.transformed_interfero_cube', err=True)

        # Init class
        self.indexer.set_file_group('merged')

        params = dict(self.options) 
        params['wcs_rotation'] = self._get_wcs_rotation(0)
        cube = InterferogramMerger(
            interf_cube_path_A=interf_cube_path_1,
            interf_cube_path_B=interf_cube_path_2,
            data_prefix=self._get_data_prefix(0),
            project_header=self._get_project_fits_header(0),
            overwrite=self.overwrite,
            tuning_parameters=self.tuning_parameters,
            indexer=self.indexer,
            instrument=self.instrument,
            ncpus=self.ncpus,
            params=params,
            config=self.config)
        
        perf = Performance(cube.cube_A, "Alternative merging process", 1,
                           instrument=self.instrument)

        # Launch merging process
        cube.alternative_merge(add_frameB=add_frameB)
        
        perf_stats = perf.print_stats()
        del cube, perf
        return perf_stats
        
    def merge_interferograms(self, add_frameB=True, smooth_vector=True):
        
        """Merge the images of the camera 1 with the transformed
        images of the camera 2.
        
        :param add_frameB: (Optional) If False use the images of the
          camera 2 only to correct for the variations of the sky
          transmission. Default True. 
    
        :param smooth_vector: (Optional) If True smooth the obtained
          correction vector with a gaussian weighted moving average.
          Reduce the possible high frequency noise of the transmission
          function. (Default True).
         
        .. seealso:: :meth:`process.InterferogramMerger.merge`
        """
        
        # detect stars
        cube1 = self._init_raw_data_cube(1)
        star_list_path_1, mean_fwhm_arc = self.detect_stars(
            cube1, 1)
        del cube1
                    
        # get frame list paths
        interf_cube_path_1 = self.indexer.get_path(
            'cam1.interfero_cube', err=True)
        interf_cube_path_2 = self.indexer.get_path(
            'merged.transformed_interfero_cube', err=True)
        
        self.indexer.set_file_group('merged')
        params = dict(self.options) 
        params['wcs_rotation'] = self._get_wcs_rotation(0)

        cube = InterferogramMerger(
            interf_cube_path_A=interf_cube_path_1,
            interf_cube_path_B=interf_cube_path_2,
            data_prefix=self._get_data_prefix(0),
            project_header=self._get_project_fits_header(0),
            overwrite=self.overwrite,
            tuning_parameters=self.tuning_parameters,
            indexer=self.indexer,
            instrument=self.instrument,
            ncpus=self.ncpus,
            params=params,
            config=self.config)
        
        perf = Performance(cube.cube_A, "Merging process", 1,
                           instrument=self.instrument)

        cube.merge(star_list_path_1, 
                   add_frameB=add_frameB, 
                   smooth_vector=smooth_vector,
                   compute_ext_light=(not self.options['no_sky']
                                      and self.config['EXT_ILLUMINATION']))
        
        perf_stats = perf.print_stats()
        del perf, cube
        return perf_stats

    def compute_calibration_laser_map(self, camera_number,
                                      get_calibration_laser_spectrum=False,
                                      fast=True):
        """Run the computation of the calibration laser map from the
        calibration laser cube. This map is used to correct for the
        off-axis shift in wavelength.

        :param camera_number: Camera number (can be either 1 or 2).

        :param get_calibration_laser_spectrum: (Optional) If True
          output the computed calibration laser spectrum cube for
          checking purpose (Default False)

        :param fast: (Optional) If False a sinc^2 is fitted so the fit
          is better but the procedure becomes slower. If True a
          gaussian is fitted (default True).
          
        .. seealso:: :meth:`process.CalibrationLaser.create_calibration_laser_map`
        """
        if 'calibration_laser_map_path' in self.options:
            warnings.warn('A path to a calibration laser map has already been given ({}), this step is skipped.'.format(
                self.options['calibration_laser_map_path']))

            return 

        if camera_number == 1:
            if "calib_path_1" in self.options: 
                calib_path = self.options["calib_path_1.hdf5"]
            else: 
                raise StandardError("No path to the calibration laser files list given, check the option file")
        elif camera_number == 2:
            if "calib_path_2" in self.options: 
                calib_path = self.options["calib_path_2.hdf5"]
            else: 
                raise StandardError("No path to the calibration laser files list given, check the option file")
        elif camera_number == 0:
            calib_path = self._get_interfero_cube_path(
                camera_number, corrected=True)
        else:
            raise StandardError("Camera number must be either 1, 2 or 0")

        if self.target == 'laser':
            ## order = self.options['order']
            ## step = self.options['step']
            # Step and order forced to config to avoid bad headers bug
            order = self.config["CALIB_ORDER"]
            step = self.config["CALIB_STEP_SIZE"]
            if (order != self.options['order']
                or step != self.options['step']):
                warnings.warn('Recorded STEP and ORDER in the option file are not the same as defined in the configuration file')
            
        else:
            order = self.config["CALIB_ORDER"]
            step = self.config["CALIB_STEP_SIZE"]
            
        logging.info('Calibration laser observation parameters: step={}, order={}'.format(step, order))
            
        self.options["camera_number"] = camera_number
        
        
        self.indexer.set_file_group(camera_number)
        cube = CalibrationLaser(
            calib_path, 
            data_prefix=self._get_data_prefix(camera_number),
            calibration_laser_header=self._get_calibration_laser_fits_header(),
            overwrite=self.overwrite,
            tuning_parameters=self.tuning_parameters,
            indexer=self.indexer,
            instrument=self.instrument,
            ncpus=self.ncpus)
        perf = Performance(cube, "Calibration laser map processing",
                           camera_number,
                           instrument=self.instrument)

        cube.create_calibration_laser_map(
            order=order, step=step, 
            get_calibration_laser_spectrum=get_calibration_laser_spectrum,
            fast=fast)
        
        perf_stats = perf.print_stats()
        del cube, perf
        return perf_stats
        

    def correct_interferogram(self, camera_number):
        """Correct a single-camera interferogram cube for variations
        of sky transission and stray light.

        :param camera_number: Camera number (can be 1 or 2).
         
        .. note:: The sky transmission vector gives the absorption
          caused by clouds or airmass variation.

        .. note:: The stray light vector gives the counts added
          homogeneously to each frame caused by a cloud reflecting
          light coming from the ground, the moon or the sun.

        .. warning:: This method is intented to be used to correct a
          'single camera' interferogram cube. In the case of a merged
          interferogram this is already done during the
          :py:meth:`orbs.Orbs.merge_interferograms` with a far better
          precision (because both cubes are used to compute it)

        .. seealso:: :py:meth:`process.Interferogram.create_correction_vectors`
        """
        raise NotImplementedError('Must be reimplemented properly based on level 2 enhancements')
        if (camera_number != 1) and (camera_number != 2):
            raise StandardError('This method (Orbs.orbs.correct_interferogram) is intended to be used only to correct single-camera interferograms (i.e. camera_number must be 1 or 2)')

        # Load interferogram frames
        interf_cube_path = self._get_interfero_cube_path(camera_number)
        
        # init cube
        self.options["camera_number"] = camera_number
        self.indexer.set_file_group(camera_number)
        cube = Interferogram(
            interf_cube_path,
            params=self.options,
            config=self.config,        
            data_prefix=self._get_data_prefix(camera_number),
            project_header = self._get_project_fits_header(
                camera_number),
            calibration_laser_header=self._get_calibration_laser_fits_header(),
            overwrite=self.overwrite,
            tuning_parameters=self.tuning_parameters,
            indexer=self.indexer,
            instrument=self.instrument, ncpus=self.ncpus)
        perf = Performance(cube, "Interferogram correction", camera_number,
                           instrument=self.instrument)
        
        # detect stars
        raw_cube = self._init_raw_data_cube(camera_number)
        star_list_path, mean_fwhm_arc = self.detect_stars(
            raw_cube, camera_number)
        del raw_cube

        if "step_nb" in self.options: 
            step_number = self.options["step_nb"]
        else:
            warnings.warn("No step number given, check the option file")
            
        # create correction vectors
        cube.create_correction_vectors(
            star_list_path, mean_fwhm_arc,
            self.config["FIELD_OF_VIEW_1"],
            profile_name=self.config["PSF_PROFILE"],
            moffat_beta=self.config["MOFFAT_BETA"],
            step_number=step_number)

        sky_transmission_vector_path = cube._get_transmission_vector_path()
        stray_light_vector_path = cube._get_stray_light_vector_path()

        # correct interferograms
        cube.correct_interferogram(sky_transmission_vector_path,
                                   stray_light_vector_path)
        perf_stats = perf.print_stats()
        del cube, perf
        return perf_stats


    # def _get_calibration_laser_map(self, camera_number):
    #     """Return calibration laser map path.

    #     :param camera_number: Camera number (can be 1, 2 or 0)
    #     """
        
    #     if 'calibration_laser_map_path' in self.options:
    #         calibration_laser_map_path = self.options[
    #             'calibration_laser_map_path']
    #         logging.info('Using an external calibration laser map: %s'%(
    #             calibration_laser_map_path))
            
    #     else:
    #         if (camera_number == 0 or camera_number == 1):
    #             calibration_laser_map_path = self.indexer[
    #                 'cam1.calibration_laser_map']
    #         elif camera_number == 2:
    #             calibration_laser_map_path = self.indexer[
    #                 'cam2.calibration_laser_map']
    #         else:
    #             raise StandardError("Camera number must be 0,1 or 2")

    #     if calibration_laser_map_path is None:
    #         warnings.warn("No calibration laser map found")
    #         return None
            
    #     if not os.path.exists(calibration_laser_map_path):
    #         warnings.warn("Calibration laser map not found ({} does not exist)".format(calibration_laser_map_path))
    #         return None
            
    #     return calibration_laser_map_path


    def compute_spectrum(self, camera_number,
                         apodization_function=None,
                         phase_correction=True,
                         wave_calibration=False,
                         phase_cube=False,
                         no_star=False):

        """Compute a spectral cube from an interferogram cube.
     
        :param apodization_function: (Optional) Apodization
          function. Default None.
    
        :param phase_correction: (Optional) If False, no phase
          correction will be done and the resulting spectrum will be
          the absolute value of the complex spectrum (default True).

        :param wave_calibration: (Optional) If True
          wavenumber/wavelength calibration is done (default False).

        :param phase_cube: (Optional) If True, only the phase cube is
          returned (default False).   

        :param no_star: (Optional) If True, the cube is considered to
          have been computed without the star dependant processes so
          that the interferogram could not be corrected for sky
          transmission variations. The interferogram cube used will
          thus be the uncorrected one (default False).

        .. seealso:: :meth:`process.Interferogram.compute_spectrum`
        .. seealso:: :meth:`orb.utils.transform_interferogram`
        """
        if phase_cube: phase_correction = False
        
        # get cube path
        cube_path = self._get_interfero_cube_path(
            camera_number, corrected=True)
                      
        # Get final bad frames vector
        ## if 'bad_frames' in self.options:
        ##     bad_frames_list = self.options['bad_frames']
        ## else: bad_frames_list = None
        ## bad_frames_vector = self.create_bad_frames_vector(
        ##     camera_number,
        ##     bad_frames_list=bad_frames_list)
        
        # init cube
        self.options["camera_number"] = camera_number
        self.indexer.set_file_group(camera_number)

        cube = Interferogram(
            cube_path,
            params=self.options,
            config=self.config,        
            data_prefix=self._get_data_prefix(camera_number),
            project_header = self._get_project_fits_header(
                camera_number),
            calibration_laser_header=self._get_calibration_laser_fits_header(),
            overwrite=self.overwrite,
            tuning_parameters=self.tuning_parameters,
            indexer=self.indexer,
            instrument=self.instrument,
            ncpus=self.ncpus)
        
        perf = Performance(cube, "Spectrum computation", camera_number,
                           instrument=self.instrument)
        
        balanced = self._is_balanced(camera_number)
        
        if apodization_function is None:
            if 'apodization_function' in self.options:
                apodization_function = self.options['apodization_function']
            else:
                apodization_function = 1.0

        if apodization_function not in self._APODIZATION_FUNCTIONS:
            try:
                apodization_function = float(apodization_function)
            except ValueError:
                raise StandardError("Unrecognized apodization function. Please try a float or " + str(self._APODIZATION_FUNCTIONS))
            
        # wavenumber option
        wavenumber = True # output always in cm-1 (calibration step
                          # transform the spectrum in nm if needed)

        if self.target == 'extphase':
            phase_maps_path = self.options['phase_maps_path']
        else:     
            phase_maps_path = self.indexer.get_path(
                'phase_maps', file_group=camera_number)

        high_order_phase_path = self._get_phase_file_path(self.options['filter_name'])

        ## Compute spectrum
        cube.compute_spectrum(
            phase_correction=phase_correction,
            wave_calibration=wave_calibration,
            window_type=apodization_function,
            phase_cube=phase_cube,
            phase_maps_path=phase_maps_path,
            high_order_phase_path=high_order_phase_path,
            balanced=balanced,
            wavenumber=wavenumber)

        perf_stats = perf.print_stats()
        del cube, perf
        return perf_stats


    def compute_phase_maps(self, camera_number, no_star=False):
        
        """Create phase maps

        Phase maps are maps of the coefficients of a polynomial or
        model fit to the phase. The dimensions of a phase map are the
        same as the dimensions of the frames of the phase cube.
        
        :param camera_number: Camera number (must be 1, 2 or 0 for
          merged data).

        :param no_star: (Optional) If True, the cube is considered to
          have been computed without the star dependant processes so
          that the interferogram could not be corrected for sky
          transmission variations. The interferogram cube used will
          thus be the uncorrected one (default False).
        
        .. seealso:: :meth:`process.Phase.create_phase_maps`
        
        """             
        # get base phase maps
        if not no_star:
            interfero_cube_path = self._get_interfero_cube_path(
                camera_number, corrected=True)
        else:
            interfero_cube_path = self._get_interfero_cube_path(
                camera_number, corrected=False)        
            
        self.indexer.set_file_group(camera_number)
        cube = Interferogram(
            interfero_cube_path,
            params=self.options,
            config=self.config,
            silent_init=True,
            instrument=self.instrument,
            overwrite=self.overwrite,
            tuning_parameters=self.tuning_parameters,
            indexer=self.indexer,
            project_header = self._get_project_fits_header(
                camera_number),
            data_prefix=self._get_data_prefix(camera_number),
            calibration_laser_header=self._get_calibration_laser_fits_header(),
            ncpus=self.ncpus)
        
        perf = Performance(cube, "Phase map creation", camera_number,
                           instrument=self.instrument)

        if self.config["INSTRUMENT_NAME"] == 'SITELLE':
            binning = 6
        else:
            binning = 3

        high_order_phase_path = self._get_phase_file_path(
            self.options['filter_name'])
        
        if high_order_phase_path is not None and os.path.exists(high_order_phase_path):
            fit_order = 1
        else:
            high_order_phase_path = None
            fit_order = self._get_phase_fit_order()
            

        cube.create_phase_maps(binning, fit_order,
                               high_order_phase_path=high_order_phase_path)
            
        perf_stats = perf.print_stats()
        del perf
        return perf_stats                

    def _find_standard_star(self, camera_number):
        """Register cube to find standard star position

        :param camera_number: Camera number (can be 1 or 2)
        
        :return: star position as a tuple (x,y)
        """
        std_path = self.options['standard_image_path_{}.hdf5'.format(camera_number)]
        std_name = self._get_standard_name(std_path)
        if std_path is None:
            
            warnings.warn("Standard related options were not given")
            return None, None, None
            
        logging.info('Registering standard image cube to find standard star position')
        
        # standard image registration to find std star
        std_cube = HDFCube(std_path, ncpus=self.ncpus,
                           instrument=self.instrument,
                           params=self.options,
                           config=self.config)
        std_astrom = self._init_astrometry(std_cube, camera_number)
        std_hdr = std_cube.get_frame_header(0) 
        std_ra, std_dec, std_pm_ra, std_pm_dec = self._get_standard_radec(
            std_name, return_pm=True)
        std_yr_obs = float(std_hdr['DATE-OBS'].split('-')[0])
        pm_orig_yr = 2000 # radec are considered to be J2000
        # compute ra/dec with proper motion
        std_ra, std_dec = orb.utils.astrometry.compute_radec_pm(
            std_ra, std_dec, std_pm_ra, std_pm_dec,
            std_yr_obs - pm_orig_yr)
        std_ra_str = '{:.0f}:{:.0f}:{:.3f}'.format(
            *orb.utils.astrometry.deg2ra(std_ra))
        std_dec_str = '{:.0f}:{:.0f}:{:.3f}'.format(
            *orb.utils.astrometry.deg2dec(std_dec))
        logging.info('Standard star {} RA/DEC: {} ({:.3f}) {} ({:.3f}) (corrected for proper motion)'.format(
            std_name,
            std_ra_str, std_ra,
            std_dec_str, std_dec))
        
        # telescope center position is used to register std frame
        std_astrom.target_ra = std_hdr['RA_DEG']
        std_astrom.target_dec = std_hdr['DEC_DEG']
        std_astrom.target_x = std_cube.dimx / 2.
        std_astrom.target_y = std_cube.dimy / 2.
        try:
            # register std frame
            std_correct_wcs = std_astrom.register(
                full_deep_frame=True, realign=False)
            # get std_x and std_y
            std_x, std_y = std_correct_wcs.wcs_world2pix(std_ra, std_dec, 0)

            logging.info('Standard star found at X/Y: {} {}'.format(
                std_x, std_y))

        except Exception, e:
            exc_info = sys.exc_info()
            warnings.warn('Error during standard image registration')
            traceback.print_exception(*exc_info)
            del exc_info
            std_x = None ; std_y = None
            
        return std_x, std_y, std_astrom.fwhm_pix
        
    def calibrate_spectrum(self, camera_number, cam1_scale=False,
                           no_star=False, filter_correction=True,
                           wcs_calibration=True):
        
        """Calibrate spectrum cube and correct WCS.

        :param camera_number: Camera number (can be 1, 2 or
          0 for merged data).   

        :param no_star: (Optional) If True, data is considered to
          contain no star, so no WCS calibration is possible (default
          False).

        :param filter_correction: (Optional) If True, spectral cube
          will be corrected for filter (default True).

        :param wcs_calibration: (Optional) If True, WCS calibration is
          intended (default True).

        .. seealso:: :py:class:`process.Spectrum`
        """
        for iopt in ['target_ra', 'target_dec', 'target_x', 'target_y']:
            if iopt not in self.options:
                raise Exception('{} must be in the options'.format(iopt))
        
        
        self.indexer.set_file_group(camera_number)
        
        spectrum = Spectrum(
            self.indexer.get_path(
                'spectrum_cube', camera_number),
            params=self.options,
            config=self.config,
            data_prefix=self._get_data_prefix(camera_number),
            project_header = self._get_project_fits_header(
                camera_number),
            calibration_laser_header=self._get_calibration_laser_fits_header(),
            overwrite=self.overwrite,
            tuning_parameters=self.tuning_parameters,
            indexer=self.indexer,
            instrument=self.instrument,
            ncpus=self.ncpus)
        
        perf = Performance(spectrum, "Spectrum calibration", camera_number,
                           instrument=self.instrument)

        ################
        ## del self.options['standard_image_path_1.hdf5']
        ## del self.options['standard_path']
        ## target_ra = None
        ################

        # Get WCS
        if (self.options['target_ra'] is None or self.options['target_dec'] is None
            or self.options['target_x'] is None or self.options['target_y'] is None):
            warnings.warn("Some WCS options were not given. WCS correction cannot be done.")
            correct_wcs = None
        elif no_star:
            warnings.warn("No-star reduction: no WCS calibration.")
            correct_wcs = None
        elif not wcs_calibration:
            warnings.warn('No WCS calibration')
            correct_wcs = None
        else:
            astrom = self._init_astrometry(spectrum, camera_number)
            deep_frame_path = self.indexer.get_path(
                'deep_frame', camera_number)
            astrom.set_deep_frame(deep_frame_path)

            try:
                correct_wcs = astrom.register(
                    full_deep_frame=True,
                    compute_distortion=True)
                
            except Exception, e:
                exc_info = sys.exc_info()
                warnings.warn('Error during WCS computation, check WCS parameters in the option file: {}'.format(e))
                traceback.print_exception(*exc_info)
                del exc_info
                correct_wcs = None


        # Get flux calibration vector
        (flux_calibration_axis,
         flux_calibration_vector) = (None, None)

        if 'standard_path' in self.options:
            std_path = self.options['standard_path']
            std_name = self._get_standard_name(std_path)
            (flux_calibration_axis,
             flux_calibration_vector) = spectrum.get_flux_calibration_vector(
                std_path, std_name)
        else:
            warnings.warn("Standard related options were not given or the name of the filter is unknown. Flux calibration vector cannot be computed")

        # Get flux calibraton coeff
        flux_calibration_coeff = None
        if 'standard_image_path_1.hdf5' in self.options:
            
            std_path = self.options['standard_image_path_1.hdf5']
            std_name = self._get_standard_name(std_path)

            # find the real star position
            std_x1, std_y1, fwhm_pix1 = self._find_standard_star(1)
            std_x2, std_y2, fwhm_pix2 = self._find_standard_star(2)
            ## std_x1, std_y1 = (1099.417, 1034.814)
            ## std_x2, std_y2 = (1102.676, 1027.210)
            ## fwhm_pix1 = 4
            
            if std_x1 is not None and std_x2 is not None:
                flux_calibration_coeff = spectrum.get_flux_calibration_coeff(
                    self.options['standard_image_path_1.hdf5'],
                    self.options['standard_image_path_2.hdf5'],
                    std_name,
                    (std_x1, std_y1),
                    (std_x2, std_y2),
                    fwhm_pix1)
    
        else:
            warnings.warn("Standard related options were not given or the name of the filter is unknown. Flux calibration coeff cannot be computed")
            
        # Calibration
        spectrum.calibrate(
            correct_wcs=correct_wcs,
            flux_calibration_vector=(
                flux_calibration_axis,
                flux_calibration_vector),
            flux_calibration_coeff=flux_calibration_coeff,
            wavenumber=self.options['wavenumber'],
            standard_header = self._get_calibration_standard_fits_header(),
            spectral_calibration=self.options['spectral_calibration'],
            filter_correction=filter_correction)
        
        perf_stats = perf.print_stats()
        del perf, spectrum
        return perf_stats


    def extract_source_interferograms(self, camera_number,
                                      alignment_coeffs=None):
        """Extract source interferograms

        :param camera_number: Camera number, can be 0, 1 or 2.
        
        :param alignment_coeffs: (Optional) Alignement coefficients if
          different from those calculated at merge step (default
          None).
        """

        # get binning factor for each camera
        if "bin_cam_1" in self.options: 
            bin_cam_1 = self.options["bin_cam_1"]
        else:
            raise StandardError("No binning for the camera 1 given")

        if "bin_cam_2" in self.options: 
            bin_cam_2 = self.options["bin_cam_2"]
        else:
            raise StandardError("No binning for the camera 2 given")

        # get initial shift
        init_dx = self.config["INIT_DX"] / bin_cam_2
        init_dy = self.config["INIT_DY"] / bin_cam_2

        # get interferograms frames paths
        interf_cube_path_1 = self.options["image_list_path_1.hdf5"]
        interf_cube_path_2 = self.options["image_list_path_2.hdf5"]

        # get computed alignement parameters
        alignment_parameters_path = self.indexer.get_path(
            'alignment_parameters',
            file_group=camera_number)
        if alignment_parameters_path is not None:
            alignment_coeffs = self.read_fits(alignment_parameters_path)[:5]
            mean_fwhm_1_arc = self.read_fits(alignment_parameters_path)[8]
        else:
            alignment_coeffs = None

        # Init SourceExtractor class
        self.indexer.set_file_group('merged')
        
        sex = SourceExtractor(
            interf_cube_path_1, interf_cube_path_2,
            bin_A=bin_cam_1, bin_B=bin_cam_2,
            pix_size_A=self.config["PIX_SIZE_CAM1"],
            pix_size_B=self.config["PIX_SIZE_CAM2"],
            data_prefix=self._get_data_prefix(0),
            project_header=self._get_project_fits_header(0),
            cube_A_project_header=self._get_project_fits_header(1),
            cube_B_project_header=self._get_project_fits_header(2),
            alignment_coeffs=alignment_coeffs,
            overwrite=self.overwrite,
            tuning_parameters=self.tuning_parameters,
            indexer=self.indexer,
            instrument=self.instrument,
            ncpus=self.ncpus,
            params=self.options,
            config=self.config)

        perf = Performance(sex.cube_B, "Extraction of source interferograms", 0,
                           instrument=self.instrument)

        # detect stars in cube 1
        cube1 = self._init_raw_data_cube(1)
        deep_frame = self.read_fits(self.indexer['cam1.deep_frame'])
        star_list_path_1, mean_fwhm_1_arc = self.detect_stars(
                cube1, 0, saturation_threshold=self.config[
                'SATURATION_THRESHOLD'], deep_frame=deep_frame)
        del cube1

        # find alignment coefficients
        if alignment_coeffs is None:
            sex.find_alignment(
                star_list_path_1,
                self.config["INIT_ANGLE"], init_dx, init_dy,
                mean_fwhm_1_arc, self.config["FIELD_OF_VIEW_1"],
                combine_first_frames=True)
        
        logging.info("Alignment parameters: {} {} {} {} {}".format(
            sex.dx, sex.dy, sex.dr, sex.da, sex.db))
        logging.info("Mean FWHM {} arcseconds".format(mean_fwhm_1_arc))

        # get source list
        source_list = self._get_source_list()
        
        self.indexer.set_file_group('merged')
        sex.extract_source_interferograms(
            source_list,
            star_list_path_1,
            self.config["FIELD_OF_VIEW_1"],
            self.indexer['cam1.alignment_vector'],
            self.indexer['cam2.alignment_vector'],
            self.indexer['merged.modulation_ratio'],
            self.indexer['merged.transmission_vector'],
            self.indexer['merged.ext_illumination_vector'],
            mean_fwhm_1_arc, deep_frame=deep_frame)
        
        perf.print_stats()


    ## def compute_source_phase(self, camera_number):
    ##     """Compute source phase.

    ##     :param camera_number: Camera number, can be 0, 1 or 2.        
    ##     """
    ##     return self.compute_source_spectra(camera_number, return_phase=True,
    ##                                        apodization_function=2.0)


    def compute_source_spectra(self, camera_number,
                               phase_correction=True,
                               filter_correction=True,
                               optimize_phase=False,
                               return_phase=False,
                               apodization_function=None):
        """Compute source spectra

        :param camera_number: Camera number, can be 0, 1 or 2.
    
        :param phase_correction: (Optional) If True, phase correction
          is done (default True).
        
        :param filter_correction: (Optional) If True, spectral cube
          will be corrected for filter (default True).

        :param optimize_phase: (Optional) If True phase is computed
          for each source independantly (default False).

        :param return_phase: (Optional) Instead of returning spectra,
          phase is returned (default False).

        :param apodization_function: (Optional) Apodization function
          (default None).
        
        .. warning:: by definition no spectral calibration is
          made. The real axis of each spectrum is given in the output
          file.
        """
        # Force some options for standard star
        if self.target == 'standard':
            optimize_phase = True
            filter_correction = False
            apodization_function = 2.0
        elif apodization_function is None:
            apodization_function = self.options['apodization_function']

        # get source list
        source_list = self._get_source_list()
        
        # print sources
        for i in range(len(source_list)):
            logging.info('source {}: {} {}'.format(
                i, source_list[i][0], source_list[i][1]))
        self.indexer.set_file_group('merged')

        # get zpd shift
        if 'zpd_index' in self.options:
            zpd_index = self.options['zpd_index']
        else:
            zpd_index = None
            
        # get phase map and phase coeffs
        if not optimize_phase:
            phase_map_paths = self._get_phase_map_paths(camera_number)
            logging.info('Loaded phase maps:')
            for phase_map_path in phase_map_paths:
                logging.info('  {}'.format(phase_map_path))
        else:
            phase_map_paths = None
                

        # get calibration laser map path
        calibration_laser_map_path = self._get_calibration_laser_map(
            camera_number)
        
        logging.info('Calibration laser map used: {}'.format(
            calibration_laser_map_path))
            
        sex = SourceExtractor(
            self._get_interfero_cube_path(
                camera_number, corrected=True),
            data_prefix=self._get_data_prefix(0),
            project_header=self._get_project_fits_header(0),
            cube_A_project_header=self._get_project_fits_header(1),
            cube_B_project_header=self._get_project_fits_header(2),
            overwrite=self.overwrite,
            tuning_parameters=self.tuning_parameters,
            indexer=self.indexer,
            instrument=self.instrument,
            ncpus=self.ncpus,
            params=self.options,
            config=self.config)

        sex.compute_source_spectra(
            source_list,
            self.indexer.get_path('extracted_source_interferograms',
                                  file_group=camera_number),
            self.options['step'],
            self.options['order'],
            apodization_function,
            self.options["filter_name"], phase_map_paths,
            self.config['CALIB_NM_LASER'],
            calibration_laser_map_path,
            optimize_phase=optimize_phase,
            filter_correction=filter_correction,
            cube_A_is_balanced = self._is_balanced(1),
            phase_correction=phase_correction,
            zpd_index=zpd_index,
            phase_order=self._get_phase_fit_order(),
            return_phase=return_phase)


    def export_calibration_laser_map(self, camera_number):
        """Export the computed calibration laser map at the root of the
        reduction folder.

        :param camera_number: Camera number (must be 1, 2 or 0 for
          merged data).
        """
        logging.info('Writing calibration laser map to disk')
        calibration_laser_map_path = self.indexer.get_path(
            'calibration_laser_map', camera_number)
       
        map_data, map_hdr = self.read_fits(
            calibration_laser_map_path,
            return_header=True)
        
        self.write_fits(self._get_calibration_laser_map_path(camera_number),
                        map_data, fits_header=map_hdr,
                        overwrite=self.overwrite)


    def export_flat_phase_map(self, camera_number):
        """Export the computed phase map at the root of the reduction folder.

        :param camera_number: Camera number (must be 1, 2 or 0 for
          merged data).
        """
        logging.info('Writing flat phase maps to disk')

        phase_maps_path = self.indexer.get_path(
            'phase_maps', file_group=camera_number)

        exported_path = ('.' + os.sep + os.path.split(phase_maps_path)[-1])

        shutil.copyfile(phase_maps_path, exported_path)
        logging.info('Flat phase maps exported at {}'.format(exported_path))

        high_order_phase_path = self.indexer.get_path(
            'high_order_phase', file_group=camera_number)

        exported_path = ('.' + os.sep + os.path.split(high_order_phase_path)[-1])

        shutil.copyfile(high_order_phase_path, exported_path)
        logging.info('High order phase file exported at {}'.format(exported_path))


        
        

    def export_calibrated_spectrum_cube(self, camera_number):
        """Extract a calibrated spectrum cube from the 'frame-divided'
        calibrated spectrum cube resulting of the reduction
        process. Write this cube at the root of the reduction folder.

        :param camera_number: Camera number (must be 1, 2 or 0 for
          merged data).
        """
        logging.info('Writing calibrated spectrum cube to disk')            
        spectrum_cube_path = self.indexer.get_path('calibrated_spectrum_cube',
                                                   camera_number)
        spectrum = HDFCube(spectrum_cube_path,
                           instrument=self.instrument,
                           ncpus=self.ncpus,
                           params=self.options,
                           config=self.config)
        spectrum_header = spectrum.get_cube_header()

        if 'wavenumber' in self.options:
            wavenumber = self.options['wavenumber']
        else:
            wavenumber = False

        if not wavenumber:
            axis = orb.utils.spectrum.create_nm_axis(
                spectrum.dimz,
                spectrum_header['STEP'],
                spectrum_header['ORDER'],
                corr=spectrum_header['AXISCORR'])
        else:
            axis = orb.utils.spectrum.create_cm1_axis(
                spectrum.dimz,
                spectrum_header['STEP'],
                spectrum_header['ORDER'],
                corr=spectrum_header['AXISCORR'])
        
        spectrum_header.extend(
            self._get_basic_spectrum_cube_header(
                axis, wavenumber=wavenumber),
            strip=True, update=False, end=True)
        
        spectrum_header.set('FILETYPE', 'Calibrated Spectrum Cube')

        # get zpd index
        if 'zpd_index' in self.options:
            zpd_index = self.options['zpd_index']
        else:
            cube = HDFCube(self._get_interfero_cube_path(
                camera_number, corrected=True), 
                           cpus=self.ncpus,
                           instrument=self.instrument,
                           params=self.options,
                           config=self.config)
            zpd_index = orb.utils.fft.find_zpd(
                cube.get_zmedian(nozero=True))
            
        spectrum_header.set('ZPDINDEX', zpd_index)

        spectrum_header.set('WAVCALIB', self.options['spectral_calibration'])

        # set calib laser nm
        spectrum_header.set('CALIBNM', self.config['CALIB_NM_LASER'])

        # get apodization
        apod = spectrum_header['APODIZ']
        spectrum_path = self._get_calibrated_spectrum_cube_path(
            camera_number, apod, wavenumber=wavenumber,
            spectral_calibration=self.options['spectral_calibration'])

        # get deep frame path
        deep_frame_path = self.indexer.get_path('deep_frame', camera_number)

        
        spectrum.export(spectrum_path, header=spectrum_header,
                        overwrite=self.overwrite, force_hdf5=True,
                        deep_frame_path=deep_frame_path)

    def export_standard_spectrum(self, camera_number, phase_correction=True,
                                 aperture_photometry=True,
                                 apodization_function=2.0,
                                 auto_phase=True):
        """Extract spectrum of the standard stars and write it at the
        root of the reduction folder.

        .. note:: The position of the standard star is defined in the
          option file with TARGETX and TARGETY keywords.

        :param camera_number: Camera number (must be 1, 2 or 0 for
          merged data).
    
        :param phase_correction: (Optional) If False, no phase
          correction will be done and the resulting spectrum will be
          the absolute value of the complex spectrum (default True).
    
        :param apodization_function: (Optional) Apodization function to use for
          spectrum computation (default 2.0).

        :param aperture_photometry: (Optional) If True, star flux is
          computed by aperture photometry. If False, star flux is
          computed from the results of the fit.

        :param auto_phase: (Optional) If True, phase is computed for
          each star independantly. Useful for high SNR stars when no
          reliable external phase can be provided (e.g. Standard
          stars).
        """
        std_spectrum, hdr = self.read_fits(self.indexer.get_path(
            'extracted_source_spectra', file_group=camera_number),
                                           return_header=True)
        corr = hdr['AXCORR0']
        axis = orb.utils.spectrum.create_cm1_axis(
            std_spectrum.shape[0], self.options['step'], self.options['order'],
            corr=corr)
        
        std_header = (self._get_project_fits_header()
                      + self._get_basic_header('Standard Spectrum')
                      + self._get_fft_params_header(apodization_function)
                      + self._get_basic_spectrum_cube_header(
                          axis, wavenumber=True))

        hdr.extend(std_header, strip=False, update=True, end=True)

        std_spectrum_path = self._get_standard_spectrum_path(camera_number)
        
        self.write_fits(std_spectrum_path, std_spectrum,
                        fits_header=hdr,
                        overwrite=True)
        
    def export_source_spectra(self, camera_number):
        """Export computed source spectra to the root folder.

        :param camera_number: Camera number (must be 1, 2 or 0 for
          merged data).
        """
        source_spectra, hdr = self.read_fits(self.indexer.get_path(
            'extracted_source_spectra', file_group=camera_number),
                                             return_header=True)
        if len(source_spectra.shape) == 1:
            step_nb = source_spectra.shape[0]
        else:
            step_nb = source_spectra.shape[1]

        nm_axis = orb.utils.spectrum.create_nm_axis(
            step_nb, self.options['step'], self.options['order'])
        source_header = (self._get_project_fits_header()
                         + self._get_basic_header('Extracted source spectra')
                         + self._get_fft_params_header(self.options[
                             'apodization_function'])
                         + self._get_basic_spectrum_cube_header(nm_axis))

        hdr.extend(source_header, strip=False, update=True, end=True)
        
        source_spectra_path = self._get_extracted_source_spectra_path(
            camera_number)
        
        self.write_fits(source_spectra_path, source_spectra,
                        fits_header=hdr,
                        overwrite=True)
    
        
        
            
##################################################
#### CLASS Performance ###########################
##################################################

class Performance(Tools):
    """
    Give some details on the efficiency of a reduction process.

    Help user to optimize ORBS's performances (e.g. adjusting the
      number of quadrants in ORBS config file)

    :param cube: Reference to a cube in order to get details on the
      reduced data

    :param process_name: Name of the running process checked

    :param camera_number: Number of the camera which cube is processed
      (can be 1, 2 or 0 for merged data)
    """
    
    _start_time = None
    _process_name = None
    _camera_number = None
    _sx = None
    _sy = None
    _sz = None
    _quad_nb = None
    
    def __init__(self, cube, process_name, camera_number, **kwargs):
        """
        Initialize class

        :param cube: Reference to a cube in order to get details on the
          reduced data

        :param process_name: Name of the running process checked

        :param camera_number: Number of the camera which cube is
          processed (can be 1, 2 or 0 for merged data)

        :param kwargs: Kwargs are :meth:`core.Tools` properties.
        """
        Tools.__init__(self, **kwargs)
        
        self._start_time = time.time()
        self._process_name = process_name
        self._camera_number = camera_number
        self._sx = cube.dimx
        self._sy = cube.dimy
        self._sz = cube.dimz
        self._quad_nb = cube.config.QUAD_NB
        logging.info("%s started for camera %d"%(self._process_name,
                                                    self._camera_number))

    def get_max_mem(self):
        """Return max memory used by the process in bytes
        """
        return (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss + 
                resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
        
    def print_stats(self):
        """
        Print performance statistics about the whole running process
        and it's children processes.

        .. warning:: The Max memory used can be largely overestimated
          for it gives the maximum memory used during the passed
          reduction process and not for the running function. To have
          a good idea of the maximum memory used by a single function
          run this function for a single reduction step.
        """
        total_time = time.time() - self._start_time
        pix_nb = float(self._sx * self._sy * self._sz)
        max_mem = self.get_max_mem()
        logging.info(
            "%s performance stats:\n"%self._process_name +
            " > Camera number: %d\n"%(self._camera_number) +
            " > Data cube size: %d x %d x %d \n"%(self._sx,
                                                  self._sy,
                                                  self._sz) +
            " > Number of quadrants: %d\n"%(self._quad_nb) +
            " > Computation time: %d s\n"%(total_time) +
            " > Max memory used: %d Mb\n"%(int(max_mem / 1000.)) +
            " > Efficiency: %.3e s/pixel\n"%float(total_time/pix_nb))
        return {'max-mem': max_mem / 1000.,
                'total-time': total_time,
                'quad-nb': self._quad_nb}
        

##################################################
#### CLASS RoadMap ###############################
##################################################
    
class RoadMap(Tools):
    """Manage a reduction road map given a target and the camera to
    use (camera 1, 2 or both cameras).

    All steps are defined in a particular xml files in the data folder
    of ORBS (:file:`orbs/data/roadmap.steps.xml`).

    Each roadmap is defined by an xml file which can also be found in
    orbs/data.

    .. note:: The name of a roadmap file is defined as follows::

         roadmap.[instrument].[target].[camera].xml

       - *instrument* can be *spiomm* or *sitelle*
       - *target* can be one of the special targets listed above or object
         for the default target
       - *camera* can be *full* for a process using both cameras; *single1*
         or *single2* for a process using only the camera 1 or 2.


    .. note:: RoadMap file syntax:
    
        .. code-block:: xml

           <?xml version="1.0"?>
           <steps>
             <step name='compute_alignment_vector' cam='1'>
               <arg value='1' type='int'></arg>
             </step>

             <step name='compute_alignment_vector' cam='2'>
               <arg value='2' type='int'></arg>
             </step>

             <step name='compute_spectrum' cam='0'>
               <arg value='0' type='int'></arg>
               <kwarg name='phase_correction'></kwarg>
               <kwarg name='apodization_function'></kwarg>
             </step>
           </steps>

        * <step> Each step is defined by its **name** (which can be
          found in :file:`orbs/data/roadmap.steps.xml`) and the camera
          used (1, 2 or 0 for merged data).

        * <arg> Every needed arguments can be passed by giving the
          value and its type (see
          :py:data:`orbs.orbs.RoadMap.types_dict`).

        * <kwarg> optional arguments. They must be added to the step
          definition if their value has to be passed from the calling
          method (:py:meth:`orbs.orbs.Orbs.start_reduction`). Only the
          optional arguments of
          :py:meth:`orbs.orbs.Orbs.start_reduction` can thus be passed
          as optional arguments of the step function.
        

        
    """ 
    road = None # the reduction road to follow
    steps = None # all the possible reduction steps
    indexer = None # an orb.Indexer instance

    instrument = None # instrument name
    target = None # target type
    cams = None # camera used 

    ROADMAP_STEPS_FILE_NAME = 'roadmap.steps.xml'
    """Roadmap steps file name"""

    def _str2bool(s):
        """Convert a string to a boolean value.

        String must be 'True','1' or 'False','0'.

        :param s: string to convert.
        """
        if s.lower() in ("true", "1"): return True
        elif s.lower() in ("false", "0"): return False
        else: raise Exception(
            "Boolean value must be 'True','1' or 'False','0'")


    types_dict = { # map type to string definition in xml files
        'int':int,
        'str':str,
        'float':float,
        'bool':_str2bool}
    """Dictionary of the defined arguments types"""
    

    def __init__(self, instrument, target, cams, indexer, **kwargs):
        """Init class.

        Load steps definitions and roadmap.

        :param instrument: Instrument. Can be 'sitelle' or 'spiomm'
        :param target: Target of the data to reduce
        :param cams: Camera to use (cam be 'single1', 'single2' or 'full')
        :param indexer: An orb.Indexer instance.
        :param kwargs: Kwargs are :meth:`core.Tools` properties.
        """
        Tools.__init__(self, **kwargs)
        self.indexer = indexer
        self.instrument = instrument
        self.target = target
        self.cams = cams

        # load roadmap steps
        roadmap_steps_path = os.path.join(
            ORBS_DATA_PATH, self.ROADMAP_STEPS_FILE_NAME)
        
        steps =  xml.etree.ElementTree.parse(roadmap_steps_path).getroot()
        
        self.steps = dict()
        for step in steps:
            infiles = list()
            infiles_xml = step.findall('infile')
            for infile_xml in infiles_xml:
                infiles.append(infile_xml.attrib['name'])
            outfiles = list()
            outfiles_xml = step.findall('outfile')
            for outfile_xml in outfiles_xml:
                outfiles.append(outfile_xml.attrib['name'])
                
            self.steps[step.attrib['name']] = Step(infiles,
                                                   None,
                                                   outfiles)

        # load roadmap
        roadmap_path = os.path.join(
            ORBS_DATA_PATH, 'roadmap.{}.{}.{}.xml'.format(
                instrument, target, cams))

        if not os.path.exists(roadmap_path):
            raise StandardError('Roadmap {} does not exist'.format(
                roadmap_path))
            
        steps =  xml.etree.ElementTree.parse(roadmap_path).getroot()
        
        self.road = list()
        for step in steps:
            args_xml = step.findall('arg')
            args = list()
            for arg_xml in args_xml:
                args.append(self.types_dict[arg_xml.attrib['type']](arg_xml.attrib['value']))

            kwargs_xml = step.findall('kwarg')
            kwargs = dict()
            for kwarg_xml in kwargs_xml:
                if 'value' in kwarg_xml.attrib:
                    kwargs[kwarg_xml.attrib['name']] = self.types_dict[
                        kwarg_xml.attrib['type']](kwarg_xml.attrib['value'])
                else:
                    kwargs[kwarg_xml.attrib['name']] = 'undef'
                    
            if step.attrib['name'] in self.steps:
                self.road.append({'name':step.attrib['name'],
                                  'cam':int(step.attrib['cam']),
                                  'args':args, 'kwargs':kwargs,
                                  'status':False})
            else:
                raise StandardError('Step {} found in {} not recorded in {}'.format(
                    step.attrib['name'], os.path.split(roadmap_path)[1],
                    os.path.split(roadmap_steps_path)[1]))

        # check road (ony possible if an Indexer instance has been given)
        self.check_road()

    def attach(self, step_name, func):
        """Attach a reduction function to a step.

        :param step_name: Name of the step
        :param func: function to attach
        """
        if step_name in self.steps:
            self.steps[step_name].func = func
        else:
            raise StandardError('No step called {}'.format(step_name))

    def check_road(self):
        """Check the status of each step of the road."""
        if self.indexer is None: return
        
        for istep in range(len(self.road)):
            step = self.road[istep]
            for outf in self.steps[step['name']].get_outfiles(step['cam']):
                if outf in self.indexer.index:
                    if os.path.exists(self.indexer[outf]):
                        self.road[istep]['status'] = True

    def get_road_len(self):
        """Return the number of steps of the road"""    
        return len(self.road)

    def get_step_func(self, index):
        """Return the function and the arguments for a particular step.

        :param index: Index of of the step.

        :return: (func, args, kwargs)
        """
        if index < self.get_road_len():
            return (self.steps[self.road[index]['name']].func,
                    self.road[index]['args'],
                    self.road[index]['kwargs'])
        else:
            raise StandardError(
                'Bad index number. Must be < {}'.format(self.get_road_len()))

    def print_status(self):
        """Print roadmap status"""
        self.check_road()
        
        print 'Status of roadmap for {} {} {}'.format(self.instrument,
                                                      self.target,
                                                      self.cams)
        index = 0
        for step in self.road:
            if step['status'] :
                status = 'done'
                color = TextColor.OKGREEN
            else:
                status = 'not done'
                color = TextColor.KORED
            
            print color + '  {} - {} {}: {}'.format(index, step['name'], step['cam'], status) + TextColor.END
            index += 1

    def get_steps_str(self, indent=0):
        """Return a string describing the different steps and their index.
        
        :param indent: (Optional) Indentation of each line (default 0)
        """
        str = ''
        istep = 0
        for step in self.road:
            step_name = step['name'].replace('_', ' ').capitalize()
            
            if step['cam'] == 0:
                str += ' '*indent + '{}. {}\n'.format(istep, step_name)
            else:
                str += ' '*indent + '{}. {} ({})\n'.format(istep, step_name, step['cam'])
            istep += 1
        return str

    def get_resume_step(self):
        index = 0
        for step in self.road:
            if not step['status']: return index
            index += 1
        return index

    
##################################################
#### CLASS Step ##################################
##################################################
class Step(object):
    """Reduction step definition.

    This class is used by :class:`orbs.orbs.RoadMap`.
    """
    def __init__(self, infiles, func, outfiles):
        """Init class

        :param infiles: a list of strings defining the input files.
        
        :param func: a function object attached to the reduction step.
        
        :param outfiles: a list of strings defining the output files.
        """
        self.infiles = infiles
        self.func = func
        self.outfiles = outfiles

    def get_outfiles(self, cam):
        """Return the complete output name of the file as it is
        recorded in the indexer (see :py:class:`orb.core.Indexer`).

        :param cam: camera used (can be 0,1 or 2)
        """
        outfiles = list()
        if cam != 0:
            for outf in self.outfiles:
                outfiles.append('cam{}.{}'.format(cam, outf))
        else:
            for outf in self.outfiles:
                outfiles.append('merged.{}'.format(outf))
                
        return outfiles


##################################################
#### CLASS JobFile ###############################
##################################################
class JobFile(OptionFile):
    """This class is aimed to parse a SITELLE's job file (*.job) and
    convert it to a classic option file (*.opt).
    """

    # special keywords that can be used mulitple times without being
    # overriden.
    protected_keys = ['OBS', 'FLAT', 'DARK', 'COMPARISON', 'STDIM']

    # OptionFile params dict
    option_file_params = dict()

    # Header of the first file found, used to get observation
    # parameters.
    _first_hdr = None 

    # If True input file is an object file, else it is an option file
    _is_jobfile = False

    # IF True target is a laser
    _is_laser = False

    # If True Init is fast, some checking is not done
    _fast_init = False

    # convertion dict between sitelle's file header keywords and
    # optionf file keywords
    convert_key = {
        'SITSTEPS':'SPESTNB',
        'OBJNAME':'OBJECT',
        'SITORDER':'SPEORDR',
        'EXPTIME':'SPEEXPT',
        'DATE-OBS':'OBSDATE',
        'TIME-OBS':'HOUR_UT',
        'RA':'TARGETR',
        'DEC':'TARGETD',
        'FILTER': 'FILTER'}

    def __init__(self, option_file_path, protected_keys=[], is_laser=False,
                 fast_init=False, **kwargs):
        """Initialize class

        :param option_file_path: Path to the option file

        :param protected_keys: (Optional) Add other protected keys to
          the basic ones (default []).

        :param is_laser: (Optional) If True target is a Laser (default
          False)

        :param fast_init: (Optional) If True conversion is fast and
          some checking is not done but the result might be uncorrect
          (default False).    

        :param kwargs: Kwargs are :meth:`core.Tools` properties.    
        """
        OptionFile.__init__(self, option_file_path,
                            protected_keys=protected_keys,
                            **kwargs)

                
        if is_laser:
            self._is_laser = True

        if fast_init:
            self._fast_init = True

        if self.header_line is not None:
            if 'SITELLE_JOB_FILE' in self.header_line:
                self._is_jobfile = True

        if self._is_jobfile:
            # try to import orbdb
            try:
                from orbdb.core import OrbDB
                self.db = OrbDB('sitelle', **kwargs)
            except ImportError, e:
                warnings.warn('Orbdb import error: {}'.format(e))
                self.db = None

            ## get first object file
            if 'OBS' in self.options:
                hdu = orb.utils.io.read_fits(
                    self.check_file_path(
                        self.options['OBS']),
                    return_hdu_only=True)
                self._first_hdr = hdu[0].header
            elif 'COMPARISON' in self.options:
                hdu = orb.utils.io.read_fits(
                    self.check_file_path(self.options['COMPARISON']),
                    return_hdu_only=True)
                self._first_hdr = hdu[0].header
            else:
                raise StandardError('Keywords OBS or COMPARISON must be at least in the job file.')

            # check
            if self._first_hdr['CCDBIN1'] != self._first_hdr['CCDBIN2']:
                self.print_error(
                    'CCD Binning appears to be different for both axes')


    def check_file_path(self, file_path):
        """Check if the file exists. If it does not exists look in the
        database to find it's right path (orbdb must be installed and
        the database must be up-to-date)

        :param file_path: File path
        """
        # if the file path does not exist, try to find it with orbdb
        if os.path.exists(file_path): return file_path
        elif self.db is not None:
            odom = os.path.splitext(os.path.split(file_path)[1])[0]
    
            self.db.cur.execute("SELECT fitsfilepath from files WHERE fitsfilepath LIKE '%{}%'".format(
                odom))
            rows_list = list()
            for row in self.db.cur.fetchall():
                row = [str(irow) for irow in row]
                rows_list.append(row)
                
            if len(rows_list) == 0:
                warnings.warn('File does not exist and was not found in the database')
                return file_path
            
            if len(rows_list) > 1:
                warnings.warn('Multiple files in the database found matching {}'.format(odom))
                
            return rows_list[0][0]
        
        else:
            warnings.warn('File not found and database OrbDB could not be imported')
            return file_path
            

    ## generate list of files
    def _generate_file_list(self, key, ftype,
                            chip_index, prebin):
        """Generate a file list from the option file and write it in a file.

        :param key: Base key of the files
        
        :param ftype: Type of list created ('object', 'dark', 'flat',
          'calib')
        
        :param chip_index: SITELLE's chip index (1 or 2 for camera 1
          or camera 2) :

        :param prebin: Prebinning.
        """

        # list is sorted in the job file order so the job file
        # is assumed to give a sorted list of files
        l = list()
        for k in self.options:
            if [''.join(i for i in k if not i.isdigit())][0] == key:
                index_str = k[len(key):]
                if len(index_str) > 0:
                    index = int(index_str)
                else:
                    index = 1
                l.append((self.options[k], index))
        l = sorted(l, key=lambda ifile: ifile[1])
        l = [ifile[0] for ifile in l]

        fpath = '{}.{}.cam{}.list'.format(self.input_file_path, ftype, chip_index)
        with open(fpath, 'w') as flist:
            flist.write('# {} {}\n'.format('sitelle', chip_index))
            if prebin is not None:
                flist.write('# prebinning {}\n'.format(int(prebin)))
            progress = ProgressBar(len(l))
            for i in range(len(l)):
                progress.update(i, info='adding file: {}'.format(l[i]))
                flist.write('{}\n'.format(self.check_file_path(l[i])))
            progress.end()
        return fpath

    def is_jobfile(self):
        """Return True if input file is a job file. False if it is an
        option file."""
        return bool(self._is_jobfile)

    def is_laser(self):
        """Return True if target is a laser"""
        return bool(self._is_laser)


    def _get_from_hdr(self, key, cast=str, optional=False):
        """Return the value associated to a keyword in the header of
        the first file loaded. Return None if keyword does not exist
        or raise an exception.

        :param key: Keyword

        :param cast: (Optional) Cast function for the returned value
          (default str).

        :param optional: (Optional) if True return None if the keyword
          does not exist. If False raise an exception (default False).
        """
        if key in self._first_hdr:
            param = self._first_hdr[key]
        else:
            if optional: return None
            else: raise StandardError(
                'Keyword {} must be recorded in the header'.format(key))
        
        if cast is not bool:
            return cast(param)
        else:
            return bool(int(param))
        
        
    def convert2opt(self):
        """Convert the job file to an option file"""

        out_params = dict()
        if not self._is_jobfile:
            raise StandardError('File is already an option file and cannot be converted')
            
        output_file_path = os.path.split(self.input_file_path)[1] + '.opt'
        self.option_file_params = dict() # parameters to write in the option file

        # parse header for basic keywords
        for key in self.convert_key:
            if key in self._first_hdr:
                self.option_file_params[self.convert_key[key]] = self._first_hdr[key]
            elif key in self.options:
                self.option_file_params[self.convert_key[key]] = self.options[key]
            elif self.convert_key[key] in self.options:
                self.option_file_params[self.convert_key[key]] = self.options[self.convert_key[key]]
            else:
                raise StandardError('Keyword {} must be in the header of the files or in the job file')
        
        # parse option file and replace duplicated key by the one in
        # the option file (so that the option file as priority over
        # the header)
        for key in self.options:
            if key in self.convert_key:
                self.option_file_params[self.convert_key[key]] = self.options[key]
            elif (''.join([i for i in key if not i.isdigit()])
                  not in self.protected_keys):
                self.option_file_params[key] = self.options[key]


        # convert name
        self.option_file_params['OBJECT'] = ''.join(
            self.option_file_params['OBJECT'].strip().split())

        # compute step size in nm
        if not self.is_laser():
            step_fringe = self._get_from_hdr('SITSTPSZ', cast=float)
            fringe_sz = self._get_from_hdr('SITFRGNM', cast=float)
            self.option_file_params['SPESTEP'] = step_fringe * fringe_sz
        else:
            self.option_file_params.pop('SPEORDR')

        # get dark exposition time
        if 'DARK' in self.options:
            dark_hdr = to.read_fits(
                self.options['DARK'], return_hdu_only=True)[0].header
            self.option_file_params['SPEDART'] = dark_hdr['EXPTIME']
            
        # define target position in the frame
        sec_cam1 =self._get_from_hdr('DSEC1')
        sec_cam1 = sec_cam1[1:-1].split(',')
        sec_cam1x = np.array(sec_cam1[0].split(':'), dtype=int)
        sec_cam1y = np.array(sec_cam1[1].split(':'), dtype=int)

        if 'TARGETX' not in self.options:
            self.option_file_params['TARGETX'] = (
                float(sec_cam1x[1]-sec_cam1x[0]) / 2.)
            
        if 'TARGETY' not in self.options:
            self.option_file_params['TARGETY'] = (
                float(sec_cam1y[1]-sec_cam1y[0]) / 2.)

        # get calibration laser map path
        if 'CALIBMAP' in self.options:
            out_params['calibration_laser_map_path'] = self.options['CALIBMAP']
       
        elif not self.is_laser() and not self._fast_init:
            raise StandardError('CALIBMAP keyword must be set')


        # get standard spectrum params
        if 'STDPATH' in self.option_file_params:
            std_path = self.option_file_params['STDPATH']
            if os.path.exists(std_path):
                std_hdr = orb.utils.io.read_fits(
                    std_path, return_hdu_only=True)[0].header
                if 'OBJECT' in std_hdr:
                    self.option_file_params['STDNAME'] = std_hdr['OBJECT']
                else:
                    raise StandardError('OBJECT key is not in standard file header ({})'.format(std_path))
            else:
                raise StandardError('Standard star file does not exist ({})'.format(std_path))

        # get standard image list parames
        if 'STDIM' in self.options:
            std_path = self.check_file_path(self.options['STDIM'])
            if os.path.exists(std_path):
                std_hdr = orb.utils.io.read_fits(std_path, return_hdu_only=True)[0].header
                if 'OBJECT' in std_hdr:
                    self.option_file_params['STDNAME'] = ''.join(std_hdr['OBJECT'].strip().split())
                else:
                    raise StandardError('OBJECT key is not in standard file header ({})'.format(std_path))
            else:
                raise StandardError('Standard image file does not exist ({})'.format(std_path))

        if 'OBS' in self.options: # target image list
            self.option_file_params['DIRCAM1'] = self._generate_file_list(
                'OBS', 'object', 1, self['PREBINNING'])
            self.option_file_params['DIRCAM2'] = self._generate_file_list(
                'OBS', 'object', 2, self['PREBINNING'])
        if 'FLAT' in self.options: # flat image list
            self.option_file_params['DIRFLT1'] = self._generate_file_list(
                'FLAT', 'flat', 1, self['PREBINNING'])
            self.option_file_params['DIRFLT2'] = self._generate_file_list(
                'FLAT', 'flat', 2, self['PREBINNING'])
        if 'DARK' in self.options: # dark image list
            self.option_file_params['DIRDRK1'] = self._generate_file_list(
                'DARK', 'dark', 1, self['PREBINNING'])
            self.option_file_params['DIRDRK2'] = self._generate_file_list(
                'DARK', 'dark', 2, self['PREBINNING'])
        if 'COMPARISON' in self.options: # wavelength calibration file list
            self.option_file_params['DIRCAL1'] = self._generate_file_list(
                'COMPARISON', 'calib', 1, self['PREBINNING'])
            self.option_file_params['DIRCAL2'] = self._generate_file_list(
                'COMPARISON', 'calib', 2, self['PREBINNING'])
        if 'STDIM' in self.options: # standard image list
            self.option_file_params['DIRSTD1'] = self._generate_file_list(
                'STDIM', 'stdim', 1, self['PREBINNING'])
            self.option_file_params['DIRSTD2'] = self._generate_file_list(
                'STDIM', 'stdim', 2, self['PREBINNING'])

        with open(output_file_path, 'w') as f:
            # create option file header
            f.write('## ORBS Option file\n# Auto-generated from SITELLE job file: {}\n'.format(
                self.input_file_path))
            # write params in the option file           
            for key in self.option_file_params:
                f.write('{} {}\n'.format(key, self.option_file_params[key]))
                
        return output_file_path, out_params, 'sitelle_job_file'

        

##################################################
#### CLASS JobsWalker ############################
##################################################

    
class JobsWalker():

    """Construct a database of all the job files found in a given folder
    and its subfolders.
    """
    keys = ['OBJECT', 'SPESTNB', 'OBSDATE', 'STDPATH', 'TARGETR', 'DIRCAM1', 'DIRCAM2', 'TARGETD', 'SPEEXPT', 'CALIBMAP', 
            'TARGETX', 'TARGETY', 'SPEORDR', 'DIRFLT1', 'DIRFLT2', 'SPESTEP', 'STDNAME', 'HOUR_UT', 'FILTER']
    base_keys = ['OBJECT', 'SPESTNB', 'OBSDATE', 'SPEEXPT', 'FILTER', 'LASTRUN', 'PATH']
    
    def __init__(self, root_folders):
        """Init class.

        :param root_folders: A list of path to the folders where the
          job files are to be found.
        """
        if not isinstance(root_folders, list):
            raise TypeError('root_folders must be a list of folders where the job files (*.job) are to be found')
        self.root_folders = list()
        for irf in root_folders:
            if not os.path.isdir(irf):
                raise IOError('{} not found'.format(irf))
            self.root_folders.append(irf)
        self.update()
        

    def update(self):
        """update the database. """
        self.optfiles = list()
        for irootf in self.root_folders:
            for root, dirs, files in os.walk(irootf):
                for file_ in files:
                    if file_.endswith(".job"):
                        ijobpath = os.path.join(root, file_)
                        if os.path.exists(ijobpath + '.opt'):
                            self.optfiles.append(ijobpath + '.opt')
                        else:
                            warnings.warn('{} does not have any corresponding opt file.'.format(ijobpath))

        self.data = dict()
        self.data['LASTRUN'] = list()
        self.data['PATH'] = list()
        for ioptpath in self.optfiles:
            iof = orb.core.OptionFile(ioptpath)
            for key in self.keys:
                if key not in self.data:
                    self.data[key] = list()
                if key in iof.options:
                    val = iof.options[key]
                else:
                    val = None
                if key == 'OBSDATE':
                    val = datetime.strptime(val, '%Y-%m-%d')
                self.data[key].append(val)
            if os.path.exists(ioptpath + '.log'):
                self.data['LASTRUN'].append(datetime.fromtimestamp(
                    os.path.getmtime(ioptpath + '.log')))
            else:
                self.data['LASTRUN'].append(None)
            self.data['PATH'].append(ioptpath)
            
    def get_opt_files(self):
        """Return a list of the option files found"""
        return list(self.optfiles)
    
    def get_all_data(self):
        """Return the whole content of the job files as a dict, which can be
           directly passed to a pandas DataFrame.

           .. code::
             jw = JobWalker(['path1', 'path2'])
             data = pd.DataFrame(jw.get_data()))
        """
        return dict(self.data)

    def get_data(self):
        """Return the content of the job files as a dict, which can be
           directly passed to a pandas DataFrame.

           .. code::
             jw = JobWalker(['path1', 'path2'])
             data = pd.DataFrame(jw.get_data()))
        """
        _data = dict()
        for key in self.base_keys:
            _data[key] = list(self.data[key])
        return _data

