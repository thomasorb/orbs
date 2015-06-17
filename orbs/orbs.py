#!/usr/bin/python
# *-* coding: utf-8 *-*
# Author: Thomas Martin <thomas.martin.1@ulaval.ca>
# File: orbs.py

## Copyright (c) 2010-2014 Thomas Martin <thomas.martin.1@ulaval.ca>
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

import scipy
import numpy as np
import xml.etree.ElementTree
    
import astropy
import astropy.wcs as pywcs
import astropy.io.fits as pyfits

import pp
import bottleneck as bn


from orb.core import Tools, Cube, Indexer, OptionFile, HDFCube
from process import RawData, InterferogramMerger, Interferogram
from process import Phase, Spectrum, CalibrationLaser
from orb.astrometry import Astrometry
import orb.utils
import orb.constants
import orb.version


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

    :PHAPATH: Path to the external phase map file

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
    
    _APODIZATION_FUNCTIONS = [
        "barthann","bartlett", "blackman", "blackmanharris",
        "bohman","hamming","hann","nuttall","parzen",
        '1.0', '1.1', '1.2', '1.3', '1.4', '1.5',
        '1.6', '1.7', '1.8', '1.9', '2.0']
    """Apodization functions that are recognized by
    :py:class:`scipy.signal` or
    :py:meth:`orb.utils.norton_beer_window` and that can be used
    directly by ORBS. Number-like names stands for the FWHM of
    extended Norton-Beer function (see
    :py:meth:`orb.utils.norton_beer_window`)"""

    
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
          
        * FIELD_OF_VIEW: Size of the field of view of the camera 1 in
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
          
        * BIAS_CALIB_PARAMS: Bias calibration parameters a, b of the
          function : bias_level = aT + b [T in degrees C]. Used to
          correct for varying dark level of the camera 2 of SpIOMM
          
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
        
        * step_number: SPESTNB
        
        * order: SPEORDR
        
        * exp_time: SPEEXPT
        
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
        
        * phase_map_path: PHAPATH
        
        * standard_name: STDNAME
        
        * fringes: FRINGES
        
        * flat_spectrum_path: DIRFLTS
          
        * star_list_path_1: STARLIST1
          
        * star_list_path_2: STARLIST2
          
        * apodization_function: APOD
        
        * calibration_laser_map_path: CALIBMAP
          
        * try_catalogue: TRYCAT
          
        * wavenumber: WAVENUMBER

        * spectral_calibration: WAVE_CALIB

        * prebinning: PREBINNING

        * no_sky: NOSKY
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

    targets = ['object', 'flat', 'standard', 'laser', 'nostar']
    """Possible target types"""
    
    target = None
    """Choosen target to reduce"""

    def __init__(self, option_file_path, target, cams,
                 config_file_name="config.orb",
                 overwrite=False, silent=False, fast_init=False):
        """Initialize Orbs class.

        :param option_file_path: Path to the option file.

        :param target: Target type to reduce. Used to define the
          reduction road map. Target may be 'object', 'flat',
          'standard', 'laser'.

        :param config_file_name: (Optional) Name of the config file to
          use. Must be located in orbs/data/.

        :param overwrite: (Optional) If True, any existing FITS file
          created by Orbs will be overwritten during the reduction
          process (default False).

        :param silent: (Optional) If True no messages nor warnings are
          displayed by Orbs (useful for silent init).

        :param fast_init: (Optional) Fast init. Data files are not
          checked. Gives access to Orbs variables (e.g. object dependant file
          paths). This mode is faster but less safe.
        """
        def store_config_parameter(key, cast):
            if cast is not bool:
                self.config[key] = cast(self._get_config_parameter(key))
            else:
                self.config[key] = bool(int(self._get_config_parameter(key)))

        def store_option_parameter(option_key, key, cast, folder=False,
                                   camera_index=None, optional=True):
            value = self.optionfile.get(key, cast)
            if value is not None:
                if not folder:
                    self.options[option_key] = value
                else:
                    list_file_path =os.path.join(
                        self._get_project_dir(), key + ".list")
                    
                    if self.config["INSTRUMENT_NAME"] == 'SITELLE':
                        image_mode = 'sitelle'
                        chip_index = camera_index
                    elif self.config["INSTRUMENT_NAME"] == 'SpIOMM':
                        image_mode = 'spiomm'
                        chip_index = None
                    else:
                        image_mode = None
                        chip_index = None

                    if 'prebinning' in self.options:
                        prebinning = self.options['prebinning']
                    else:
                        prebinning = None

                    # check if path is a directory or a file list
                    if os.path.exists(value):
                        if os.path.isdir(value):
                            self.options[option_key] = (
                                self._create_list_from_dir(
                                    value, list_file_path,
                                    image_mode=image_mode,
                                    chip_index=chip_index,
                                    prebinning=prebinning,
                                    check=not fast_init))
                        else:
                             self.options[option_key] = value
                             

                        # export fits frames as hdf5 cubes
                        if not fast_init:
                            cube = Cube(self.options[option_key],
                                        silent_init=True)
                            export_path = os.path.splitext(
                                self.options[option_key])[0] + '.hdf5'

                            # check if the hdf5 cube already
                            # exists. If the list of the imported
                            # files in the hdf5 cube is the same,
                            # export is not done again.
                            already_exported = False
                            if os.path.exists(export_path):
                                with self.open_hdf5(export_path, 'r') as f:
                                    if 'image_list' in f:
                                        if np.all(
                                            f['image_list'][:]
                                            == np.array(cube.image_list)):
                                            already_exported = True
                                            self._print_msg(
                                                'HDF5 cube {} already created'.format(export_path))
                            if not already_exported:
                                cube.export(export_path, force_hdf5 = True,
                                            overwrite=True)


                            self.options[option_key + '.hdf5'] = export_path
                             
                             
                    else: self._print_error(
                        'Given path does not exist {}'.format(value))

            elif not optional:
                self._print_error('option {} must be set'.format(key))

        self.option_file_path = option_file_path
        self.config_file_name = config_file_name
        self._logfile_name = os.path.basename(option_file_path) + '.log'
        self._msg_class_hdr = self._get_msg_class_hdr()
        self.overwrite = overwrite
        self.__version__ = __version__
        self._silent = silent
        
        # First, print ORBS version
        self._print_msg("ORBS version: %s"%self.__version__, color=True)
        self._print_msg("ORB version: %s"%orb.version.__version__, color=True)

        # Print modules versions
        self._print_msg("Numpy version: %s"%np.__version__)
        self._print_msg("Scipy version: %s"%scipy.__version__)
        self._print_msg("Astropy version: %s"%astropy.__version__)
        self._print_msg("Parallel Python version: %s"%pp.version)
        self._print_msg("Bottleneck version: %s"%bn.__version__)
        
        # Print the entire config file for log
        with self.open_file(self._get_config_file_path(), 'r') as conf_file:
            self._print_msg("Configuration file content:", color=True)
            for line in conf_file:
                self._print_msg(line[:-1], no_hdr=True)

        # read config file to get instrumental parameters
        store_config_parameter("INSTRUMENT_NAME", str)
        store_config_parameter("INIT_ANGLE", float)
        store_config_parameter("INIT_DX", float)
        store_config_parameter("INIT_DY", float)
        store_config_parameter("FIELD_OF_VIEW", float)
        store_config_parameter("FIELD_OF_VIEW_2", float)
        store_config_parameter("PIX_SIZE_CAM1", int)
        store_config_parameter("PIX_SIZE_CAM2", int)
        store_config_parameter("BALANCED_CAM", int)
        store_config_parameter("CALIB_NM_LASER", float)
        store_config_parameter("CALIB_ORDER", float)
        store_config_parameter("CALIB_STEP_SIZE", float)
        store_config_parameter("PHASE_FIT_DEG", int)
        store_config_parameter("DETECT_STAR_NB", int)
        store_config_parameter("INIT_FWHM", float)
        store_config_parameter("PSF_PROFILE", str)
        store_config_parameter("MOFFAT_BETA", float)
        store_config_parameter("DETECT_STACK", float)
        store_config_parameter("OPTIM_DARK_CAM1", int)
        store_config_parameter("OPTIM_DARK_CAM2", int)
        store_config_parameter("WCS_ROTATION", float)
        store_config_parameter("EXT_ILLUMINATION", bool)
        store_config_parameter("SATURATION_THRESHOLD", float)
        store_config_parameter("CAM1_DETECTOR_SIZE_X", int)
        store_config_parameter("CAM1_DETECTOR_SIZE_Y", int)
        store_config_parameter("CAM2_DETECTOR_SIZE_X", int)
        store_config_parameter("CAM2_DETECTOR_SIZE_Y", int)
        
        # defining DARK_ACTIVATION_ENERGY
        self.config["DARK_ACTIVATION_ENERGY"] = float(
            self._get_config_parameter(
                "DARK_ACTIVATION_ENERGY", optional=True))

        # defining BIAS_CALIB_PARAMS
        BIAS_CALIB_PARAM_A = self._get_config_parameter(
            "BIAS_CALIB_PARAM_A", optional=True)
        BIAS_CALIB_PARAM_B = self._get_config_parameter(
            "BIAS_CALIB_PARAM_B", optional=True)

        if ((BIAS_CALIB_PARAM_A is not None)
            and (BIAS_CALIB_PARAM_B is not None)):
            self.config["BIAS_CALIB_PARAMS"] = [
                float(BIAS_CALIB_PARAM_A),
                float(BIAS_CALIB_PARAM_B)]
        else: self._print_warning("No bias calibration parameters (check BIAS_CALIB_A, BIAS_CALIB_B in the configuration file.)")

        # Read option file to get observation parameters
        if not os.path.exists(option_file_path):
            self._print_error("Option file does not exists !")

        # Print first the entire option file for logging
        op_file = open(option_file_path)
        self._print_msg("Option file content :", color=True)
        for line in op_file:
            self._print_msg(line[:-1], no_hdr=True)

        # record some default options
        self.options["try_catalogue"] = False
        self.options['spectral_calibration'] = True
        self.options['wavenumber'] = False
        self.options['no_sky'] = False

        ##########
        ## Parse the option file to get reduction parameters
        ##########
        self.optionfile = OptionFile(option_file_path,
                                     config_file_name=config_file_name)

        # In the case of LASER cube the parameters are set
        # automatically
        if target == 'laser': optional_keys = True
        else:  optional_keys = False
            
        store_option_parameter('object_name', 'OBJECT', str,
                               optional=optional_keys)
        store_option_parameter('filter_name', 'FILTER', str,
                               optional=optional_keys)
        
            
        store_option_parameter('step', 'SPESTEP', float)
        store_option_parameter('step_number', 'SPESTNB', int)
        store_option_parameter('order', 'SPEORDR', float)
        store_option_parameter('exp_time', 'SPEEXPT', float)
        store_option_parameter('dark_time', 'SPEDART', float)
        store_option_parameter('obs_date', 'OBSDATE', str)
        store_option_parameter('target_ra', 'TARGETR', str)
        if 'target_ra' in self.options:
            self.options['target_ra'] = self.options['target_ra'].split(':')
        store_option_parameter('target_dec', 'TARGETD', str)
        if 'target_dec' in self.options:
            self.options['target_dec'] = self.options['target_dec'].split(':')
        store_option_parameter('target_x', 'TARGETX', float)
        store_option_parameter('target_y', 'TARGETY', float)
        store_option_parameter('standard_path', 'STDPATH', str)
        store_option_parameter('standard_name', 'STDNAME', str)
        store_option_parameter('phase_map_path', 'PHAPATH', str)
        store_option_parameter('star_list_path_1', 'STARLIST1', str)
        store_option_parameter('star_list_path_2', 'STARLIST2', str)
        store_option_parameter('apodization_function', 'APOD', str)
        store_option_parameter('calibration_laser_map_path', 'CALIBMAP', str)
        store_option_parameter('try_catalogue', 'TRYCAT', bool)
        store_option_parameter('wavenumber', 'WAVENUMBER', bool)
        store_option_parameter('spectral_calibration', 'WAVE_CALIB', bool)
        store_option_parameter('no_sky', 'NOSKY', bool)
        store_option_parameter('prebinning', 'PREBINNING', int)
        
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
               self.config[key] = self.optionfile.get(key, key_type)
               self._print_warning("Configuration option %s changed to %s"%(key, self.config[key]))
               
        # Get tuning parameters
        self.tuning_parameters = self.optionfile.get_tuning_parameters()
        for itune in self.tuning_parameters:
            self._print_warning("Tuning parameter %s changed to %s"%(
                itune, self.tuning_parameters[itune]))

        if (("object_name" not in self.options)
            or ("filter_name" not in self.options)):
            self._print_error("The option file needs at least an object name (use keyword : OBJECT) and a filter name (use keyword : FILTER)")
        else:
            self.options["project_name"] = (self.options["object_name"] 
                                            + "_" + self.options["filter_name"])

        
        # get folders paths
        self._print_msg('Reading data folders')
        store_option_parameter('image_list_path_1', 'DIRCAM1', str, True, 1)
        store_option_parameter('image_list_path_2', 'DIRCAM2', str, True, 2)
        store_option_parameter('bias_path_1', 'DIRBIA1', str, True, 1)
        store_option_parameter('bias_path_2', 'DIRBIA2', str, True, 2)
        store_option_parameter('dark_path_1', 'DIRDRK1', str, True, 1)
        store_option_parameter('dark_path_2', 'DIRDRK2', str, True, 2)
        if (('dark_path_2' in self.options or 'dark_path_1' in self.options)
            and 'dark_time' not in self.options):
            self._print_error('Dark integration time must be set (SPEDART) if the path to a dark calibration files folder is given')
            
        store_option_parameter('flat_path_1', 'DIRFLT1', str, True, 1)
        store_option_parameter('flat_path_2', 'DIRFLT2', str, True, 2)
        store_option_parameter('calib_path_1', 'DIRCAL1', str, True, 1)
        store_option_parameter('calib_path_2', 'DIRCAL2', str, True, 2)
        store_option_parameter('flat_spectrum_path', 'DIRFLTS', str, True)

        if 'image_list_path_1' in self.options:
            cube1 = Cube(self.options['image_list_path_1'],
                         silent_init=True,
                         config_file_name=self.config_file_name)
            dimz1 = cube1.dimz
            cam1_image_shape = [cube1.dimx, cube1.dimy]
            
            # Get data binning
            cam1_detector_shape = [self.config['CAM1_DETECTOR_SIZE_X'],
                                   self.config['CAM1_DETECTOR_SIZE_Y']]
            bin_cam_1 = orb.utils.compute_binning(
                cam1_image_shape, cam1_detector_shape)
            self._print_msg('Computed binning of camera 1: {}x{}'.format(
                *bin_cam_1))
            if bin_cam_1[0] != bin_cam_1[1]:
                self._print_error('Images with different binning along X and Y axis are not handled by ORBS')
            self.options['bin_cam_1'] = bin_cam_1[0]
            
            # prebinning
            if 'prebinning' in self.options:
                if self.options['prebinning'] is not None:
                    self.options['bin_cam_1'] = (
                        self.options['bin_cam_1']
                        * self.options['prebinning'])
                    
        if 'image_list_path_2' in self.options:
            cube2 = Cube(self.options['image_list_path_2'],
                         silent_init=True,
                         config_file_name=self.config_file_name)
            dimz2 = cube2.dimz
            cam2_image_shape = [cube2.dimx, cube2.dimy]
            # Get data binning
            cam2_detector_shape = [self.config['CAM2_DETECTOR_SIZE_X'],
                                   self.config['CAM2_DETECTOR_SIZE_Y']]
            bin_cam_2 = orb.utils.compute_binning(
                cam2_image_shape, cam2_detector_shape)
            self._print_msg('Computed binning of camera 2: {}x{}'.format(
                *bin_cam_2))
            if bin_cam_2[0] != bin_cam_2[1]:
                self._print_error('Images with different binning along X and Y axis are not handled by ORBS')
            self.options['bin_cam_2'] = bin_cam_2[0]
            
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
                self._print_error('The number of images of CAM1 and CAM2 are not the same (%d != %d)'%(dimz1, dimz2))
                
            if self.options['step_number'] < dimz1:
                self._print_warning('Step number option changed to {} because the number of steps ({}) of a full cube must be greater or equal to the number of images given for CAM1 and CAM2 ({})'.format(
                    dimz1, self.options['step_number'], dimz1))
                self.options['step_number'] = dimz1


        # Init Indexer
        self.indexer = Indexer(data_prefix=self.options['object_name']
                               + '_' + self.options['filter_name'] + '.',
                               config_file_name=self.config_file_name)
        self.indexer.load_index()

        # Load roadmap
        if target in self.targets:
            self.target = target
        else:
            self._print_error('Unknown target type: target must be in {}'.format(self.targets))

        self.roadmap = RoadMap(
            self.config["INSTRUMENT_NAME"].lower(), target, cams, self.indexer)
        

        # attach methods to roadmap steps
        self.roadmap.attach('compute_alignment_vector',
                            self.compute_alignment_vector)
        self.roadmap.attach('compute_cosmic_ray_map',
                            self.compute_cosmic_ray_map)
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
        self.roadmap.attach('compute_phase',
                            self.compute_phase)
        self.roadmap.attach('compute_phase_maps',
                            self.compute_phase_maps)
        self.roadmap.attach('compute_spectrum',
                            self.compute_spectrum)
        self.roadmap.attach('calibrate_spectrum',
                            self.calibrate_spectrum)
        
        
        
        

    def _get_calibration_standard_fits_header(self):

        if ('standard_name' in self.options
            and 'standard_path' in self.options):
            hdr = list()
            hdr.append(('COMMENT','',''))
            hdr.append(('COMMENT','Calibration standard parameters',''))
            hdr.append(('COMMENT','-------------------------------',''))
            hdr.append(('COMMENT','',''))
            hdr.append(('STDNAME', self.options['standard_name'],
                        'Name of the standard star'))
            std_path = os.path.basename(self.options['standard_path'])[
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
        if "exp_time" in self.options:
            hdr.append(('EXPTIME', self.options["exp_time"], 
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
        if "step_number" in self.options:
            hdr.append(('STEPNB', self.options["step_number"], 
                        'Number of steps'))
        if "target_ra" in self.options:
            hdr.append(('TARGETR', ':'.join(self.options["target_ra"]), 
                        'Target Right Ascension'))
        if "target_dec" in self.options:
            hdr.append(('TARGETD', ':'.join(self.options["target_dec"]), 
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
        else: self._print_error('Camera number must be 0, 1, or 2')
        
        return ('.' + os.sep + self.options['object_name']
                + '_' +  self.options['filter_name'] + '.' + cam + '.')

    def _get_flat_phase_map_path(self, camera_number):
        """Return path to the order 0 phase map from a flat cube.
        
        :param camera_number: Camera number (must be 1, 2 or 0 for
          merged data).
        """
        return (self._get_root_data_path_hdr(camera_number)
                + 'flat_phase_map.fits')

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
                + wave_type + '.' + apod + calib + '.fits')

    def _get_standard_spectrum_path(self, camera_number):
        """Return path to the standard star spectrum
        
        :param camera_number: Camera number (must be 1, 2 or 0 for
          merged data).
        """
        return (self._get_root_data_path_hdr(camera_number)
                + 'standard_spectrum.fits')
    
    def _init_raw_data_cube(self, camera_number):
        """Return instance of :class:`orbs.process.RawData` class

        :param camera_number: Camera number (can be either 1 or 2).
        """
        if (camera_number == 1):
            if ("image_list_path_1" in self.options):
                self.indexer.set_file_group('cam1')
                cube = RawData(
                    self.options["image_list_path_1.hdf5"], 
                    data_prefix=self._get_data_prefix(1),
                    project_header=self._get_project_fits_header(1),
                    overwrite=self.overwrite,
                    tuning_parameters=self.tuning_parameters,
                    indexer=self.indexer,
                    logfile_name=self._logfile_name,
                    config_file_name=self.config_file_name)
            else:
                self._print_error("No image list file for camera 1 given, please check option file")
        elif (camera_number == 2):
            if ("image_list_path_2" in self.options):
                self.indexer.set_file_group('cam2')
                cube = RawData(
                    self.options["image_list_path_2.hdf5"], 
                    data_prefix=self._get_data_prefix(2),
                    project_header=self._get_project_fits_header(2),
                    overwrite=self.overwrite,
                    tuning_parameters=self.tuning_parameters,
                    indexer=self.indexer,
                    logfile_name=self._logfile_name,
                    config_file_name=self.config_file_name)
            else:
                self._print_error("No image list file for camera 2 given, please check option file")
        else:
            self._print_error("Camera number must be either 1 or 2, please check 'camera_number' parameter")
            return None
        return cube

    def _init_astrometry(self, cube, camera_number):
        """Init Astrometry class.

        The Astrometry class is used for star detection and star fitting
        (position and photometry)
        
        :param cube: an orbs.Cube instance

        :param camera_number: Camera number (can be 1, 2 or 0 
          for merged data).
        
        :return: :py:class:`orb.astrometry.Astrometry` instance

        .. seealso:: :py:class:`orb.astrometry.Astrometry`
        """

        if "target_ra" in self.options:
            target_ra = orb.utils.ra2deg(self.options["target_ra"])
        else: target_ra = None
            
        if "target_dec" in self.options:
            target_dec = orb.utils.dec2deg(self.options["target_dec"])
        else: target_dec = None
        
        if "target_x" in self.options:
            target_x = self.options["target_x"]
        else: target_x = None
        
        if "target_y" in self.options:
            target_y = self.options["target_y"]
        else: target_y = None

        if target_x is not None and target_y is not None:
            if camera_number == 2:
                # get binning factor for each camera
                if "bin_cam_1" in self.options: 
                    bin_cam_1 = self.options["bin_cam_1"]
                else:
                    self._print_error("No binning for the camera 1 given")

                if "bin_cam_2" in self.options: 
                    bin_cam_2 = self.options["bin_cam_2"]
                else:
                    self._print_error("No binning for the camera 2 given")
                # get initial shift
                init_dx = self.config["INIT_DX"] / bin_cam_2
                init_dy = self.config["INIT_DY"] / bin_cam_2
                pix_size_1 = self.config["PIX_SIZE_CAM1"]
                pix_size_2 = self.config["PIX_SIZE_CAM2"]
                zoom = (pix_size_2 * bin_cam_2) / (pix_size_1 * bin_cam_1)
                xrc = cube.dimx / 2.
                yrc = cube.dimy / 2.
                
                target_xy = orb.cutils.transform_A_to_B(
                    target_x, target_y,
                    init_dx, init_dy,
                    self.config["INIT_ANGLE"],
                    0., 0., xrc, yrc, zoom, zoom)
               
            else:
                target_xy = [target_x, target_y]
                
        else:
            target_xy = None

        if target_ra is not None and target_dec is not None:
            target_radec = [target_ra, target_dec]
        else:
            target_radec = None

        
        if camera_number == 2:
            fov = self.config["FIELD_OF_VIEW_2"]
            wcs_rotation = (self.config["WCS_ROTATION"]
                            - self.config["INIT_ANGLE"])
        else:
            fov = self.config["FIELD_OF_VIEW"]
            wcs_rotation = self.config["WCS_ROTATION"]
            
        return Astrometry(cube, self.config["INIT_FWHM"],
                          fov,
                          profile_name=self.config["PSF_PROFILE"],
                          moffat_beta=self.config["MOFFAT_BETA"],
                          detect_stack=self.config["DETECT_STACK"],
                          data_prefix=self._get_data_prefix(camera_number),
                          logfile_name=self._logfile_name,
                          tuning_parameters=self.tuning_parameters,
                          target_radec=target_radec, target_xy=target_xy,
                          wcs_rotation=wcs_rotation,
                          config_file_name=self.config_file_name)


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
        else: self._print_error('Camera number must be 0, 1 or 2')

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
                              config_file_name=self.config_file_name)
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
                bad_frames_list = orb.utils.correct_bad_frames_vector(
                    bad_frames_list, interf_cube.dimz)
                bad_frames_vector[bad_frames_list] = 1
            else:
                self._print_error('Bad indexes in the bad frame list')
    
        return bad_frames_vector


    def start_reduction(self, apodization_function=None, start_step=0,
                        n_phase=None, alt_merge=False, save_as_quads=False,
                        add_frameB=True):
        
        """Run the whole reduction process for two cameras using
        default options 

        :param apodization_function: (Optional) Name of the apodization
          function be used during the spectrum computation (default None).

        :param start_step: (Optional) Starting step number. Use it to
          cover from an error at a certain step without having to
          run the whole process one more time (default 0).  
           
        :param n_phase: (Optional) Number of points around ZPD to use
          for phase correction during spectrum computation. If 0, no
          phase correction will be done and the resulting spectrum
          will be the absolute value of the complex spectrum. If
          None, the number of points is set to 50 percent of the
          interferogram length (default None). 

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
        """
        # save passed kwargs
        local_kwargs = locals()

        for istep in range(self.roadmap.get_road_len()):
            f, args, kwargs = self.roadmap.get_step_func(istep)
            
            if f is not None:
                if istep >= start_step:
                    kwargs_dict = {}
                    for kwarg in kwargs:
                        if kwargs[kwarg] == 'undef':
                            if kwarg in local_kwargs:
                                kwargs_dict[kwarg] = local_kwargs[kwarg]
                            else:
                                self._print_error(
                                    'kwarg {} not defined'.format(kwarg))
                        else:
                            kwargs_dict[kwarg] = kwargs[kwarg]
                    self._print_msg('run {}({}, {})'.format(
                        f.__name__, str(args), str(kwargs)), color=True)
                    
                    
                    f(*args, **kwargs_dict)
                    
            else: self._print_error("No function attached to step '{}'".format(
                self.roadmap.road[istep]['name']))


        # Once the whole process has been done, final data can be exported
        if self.roadmap.cams == 'single1': cam = 1
        elif self.roadmap.cams == 'single2': cam = 2
        elif self.roadmap.cams == 'full':
            if self.target == 'laser': cam = 1
            else: cam = 0
        
        if self.target == 'object' or self.target == 'nostar':
            self.export_calibrated_spectrum_cube(cam)
        if self.target == 'flat': self.export_flat_phase_map(cam)
        if self.target == 'laser': self.export_calibration_laser_map(cam)
        if self.target == 'standard': self.export_standard_spectrum(
            cam, auto_phase=True)


    def detect_stars(self, cube, camera_number, 
                     saturation_threshold=None, return_fwhm_pix=False):
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
        
        :return: Path to a star list, mean FWHM of stars in arcseconds.

        .. seealso:: :py:meth:`orb.astrometry.Astrometry.detect_stars`
        """
        # check first if a star list has been passed to the option file
        if ((camera_number == 1
            or camera_number == 0)
            and 'star_list_path_1' in self.options):
            self._print_msg('Using external star list: %s'%self.options['star_list_path_1'], color=True)
            star_list_path = self.options['star_list_path_1']
            mean_fwhm = self.config['INIT_FWHM']
            refit = True
            
        elif (camera_number == 2 and 'star_list_path_2' in self.options):
            self._print_msg('Using external star list: %s'%self.options['star_list_path_2'], color=True)
            star_list_path = self.options['star_list_path_2']
            mean_fwhm = self.config['INIT_FWHM']
            refit = True
        else: refit = False

        if refit:
            self._print_msg('Fitting manual star list')
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
            mean_fwhm = orb.utils.robust_median(fit_results[:,'fwhm_arc'])
            self._print_msg('Mean FWHM: {} arcsec'.format(mean_fwhm))
            
        
        # else try to auto-detect stars
        else:
            if saturation_threshold is None:
                saturation_threshold = self.config['SATURATION_THRESHOLD'] / 2.
                
            self._print_msg('Autodetecting stars', color=True)
            astrom = self._init_astrometry(cube, camera_number)
            star_list_path, mean_fwhm = astrom.detect_stars(
                min_star_number=self.config['DETECT_STAR_NB'],
                saturation_threshold=saturation_threshold,
                try_catalogue=self.options['try_catalogue'])
            if return_fwhm_pix: mean_fwhm = astrom.arc2pix(mean_fwhm)
            del astrom

        return star_list_path, mean_fwhm

    def get_noise_values(self, camera_number):
        """Return readout noise and dark current level from bias and
        dark frames.

        :param camera_number: Camera number (can be either 1 or 2).

        :return: readout_noise, dark_current_level
         
        .. seealso:: :py:meth:`process.RawData.get_noise_values`
        """
        self._print_msg("Computing noise levels", color=True)
        
        bias_path = None
        dark_path = None
        
        if camera_number == 1:
            if "bias_path_1" in self.options:
                bias_path = self.options["bias_path_1"]
            if "dark_path_1" in self.options:
                dark_path = self.options["dark_path_1"]
                
        if camera_number == 2:
            if "bias_path_2" in self.options:
                bias_path = self.options["bias_path_2"]
            if "dark_path_2" in self.options:
                dark_path = self.options["dark_path_2"]
            
        exposition_time = self.options["exp_time"]
        if "dark_time" in self.options:
                dark_int_time = self.options["dark_time"]
        else: dark_int_time = None
        
        if (bias_path is not None and dark_path is not None
            and dark_int_time is not None):
               
            cube = self._init_raw_data_cube(camera_number)
            readout_noise, dark_current_level = cube.get_noise_values(
                bias_path, dark_path, exposition_time,
                dark_int_time, combine='average',
                reject='avsigclip')
            del cube

            self._print_msg(
                'Computed readout noise of camera %d from bias frames: %f ADU/pixel'%(camera_number, readout_noise))
            self._print_msg(
                'Computed dark current level  of camera %d from dark frames: %f ADU/pixel'%(camera_number, dark_current_level))
 
            return readout_noise, dark_current_level
        
        else:
            return None, None


    def compute_alignment_vector(self, camera_number):
        """Run the computation of the alignment vector.

        If no path to a star list file is given 
        use: :py:meth:`orb.astrometry.Astrometry.detect_stars` 
        method to detect stars.

        :param camera_number: Camera number (can be either 1 or 2).
              
        .. seealso:: :meth:`orb.astrometry.Astrometry.detect_stars`
        .. seealso:: :meth:`process.RawData.create_alignment_vector`
        """
        readout_noise, dark_current_level = self.get_noise_values(camera_number)
        
        cube = self._init_raw_data_cube(camera_number)
        
        perf = Performance(cube, "Alignment vector computation", camera_number,
                           config_file_name=self.config_file_name)
        
        star_list_path, mean_fwhm_arc = self.detect_stars(
            cube, camera_number)

        cube.create_alignment_vector(
            star_list_path, mean_fwhm_arc,
            self.config["FIELD_OF_VIEW"],
            profile_name='gaussian', # Better for alignment tasks
            moffat_beta=self.config["MOFFAT_BETA"],
            readout_noise=readout_noise,
            dark_current_level=dark_current_level)
        
        perf_stats = perf.print_stats()
        del cube, perf
        return perf_stats

    def compute_cosmic_ray_map(self, camera_number, z_coeff=3.):
        """Run the computation of the cosmic ray map.

        :param camera_number: Camera number (can be either 1 or 2).
        
        :param z_coeff: (Optional) Threshold coefficient for cosmic ray
          detection, lower it to detect more cosmic rays (default : 3.).

        .. seealso:: :meth:`process.RawData.create_cosmic_ray_map`
        """
        cube = self._init_raw_data_cube(camera_number)
        perf = Performance(cube, "Cosmic ray map computation", camera_number,
                           config_file_name=self.config_file_name)
        
        if "step_number" in self.options: 
            step_number = self.options["step_number"]
        else:
            self._print_warning("No step number given, check the option file.")

        if "bad_frames" in self.options:
            bad_frames_vector = self.options["bad_frames"]
        else:
            bad_frames_vector = []

        star_list_path, mean_fwhm_pix = self.detect_stars(
            cube, camera_number, return_fwhm_pix=True)
            
        cube.create_cosmic_ray_map(z_coeff=z_coeff, step_number=step_number,
                                   bad_frames_vector=bad_frames_vector,
                                   star_list_path=star_list_path)

        perf_stats = perf.print_stats()
        del cube, perf
        return perf_stats

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
                              flat_smooth_deg=0):
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
          
        .. seealso:: :py:meth:`process.RawData.correct`
        """
        cube = self._init_raw_data_cube(camera_number)
        perf = Performance(cube, "Interferogram computation", camera_number,
                           config_file_name=self.config_file_name)

        bias_path = None
        dark_path = None
        flat_path = None

        if camera_number == 1: 
            if "dark_path_1" in self.options:
                dark_path = self.options["dark_path_1"]
            else:
                self._print_warning("No path to dark frames given, please check the option file.")
            if "bias_path_1" in self.options:
                bias_path = self.options["bias_path_1"]
            else:
                self._print_warning("No path to bias frames given, please check the option file.")
            if "flat_path_1" in self.options:
                flat_path = self.options["flat_path_1"]
            else:
                self._print_warning("No path to flat frames given, please check the option file.")

            optimize_dark_coeff = self.config['OPTIM_DARK_CAM1']
            
        if camera_number == 2: 
            if "dark_path_2" in self.options:
                dark_path = self.options["dark_path_2"]
            else:
                self._print_warning("No path to dark frames given, please check the option file.")
            if "bias_path_2" in self.options:
                bias_path = self.options["bias_path_2"]
            else:
                self._print_warning("No path to bias frames given, please check the option file.")
            if "flat_path_2" in self.options:
                flat_path = self.options["flat_path_2"]
            else:
                self._print_warning("No path to flat frames given, please check the option file.")

            optimize_dark_coeff = self.config['OPTIM_DARK_CAM2']
        
        if "exp_time" in self.options:
            exposition_time = self.options["exp_time"]
        else:  exposition_time = None
        if "dark_time" in self.options:
            dark_int_time = self.options["dark_time"]
        else: dark_int_time = None

        if "bad_frames" in self.options:
            bad_frames_vector = self.options["bad_frames"]
        else:
            bad_frames_vector = []
            
        cube.correct(
            bias_path=bias_path, dark_path=dark_path, 
            flat_path=flat_path, alignment_vector_path=None,
            cr_map_cube_path=None, bad_frames_vector=bad_frames_vector, 
            dark_int_time=dark_int_time, flat_int_time=None,
            exposition_time=exposition_time,
            optimize_dark_coeff=optimize_dark_coeff,
            dark_activation_energy=self.config["DARK_ACTIVATION_ENERGY"],
            bias_calibration_params=self.config["BIAS_CALIB_PARAMS"],
            z_range=z_range, combine=combine, reject=reject,
            flat_smooth_deg=flat_smooth_deg)
        perf_stats = perf.print_stats()
        del cube, perf
        return perf_stats

    def transform_cube_B(self, interp_order=1, no_star=False):
        
        """Calculate the alignment parameters of the camera 2
        relatively to the first one. Transform the images of the
        camera 2 using linear interpolation by default.
    
        :param interp_order: (Optional) Interpolation order (Default 1.).

        :param no_star: (Optional) If the cube does not contain any star, the
          transformation is made using the default alignment
          parameters (recorded in the configuration file :
          'data/config.orb') (default False).
                         
        .. seealso:: :py:meth:`process.InterferogramMerger.find_alignment`
        .. seealso:: :py:meth:`process.InterferogramMerger.transform`
        """
        # get binning factor for each camera
        if "bin_cam_1" in self.options: 
            bin_cam_1 = self.options["bin_cam_1"]
        else:
            self._print_error("No binning for the camera 1 given")

        if "bin_cam_2" in self.options: 
            bin_cam_2 = self.options["bin_cam_2"]
        else:
            self._print_error("No binning for the camera 2 given")

        # get initial shift
        init_dx = self.config["INIT_DX"] / bin_cam_2
        init_dy = self.config["INIT_DY"] / bin_cam_2

        # get interferograms frames paths
        interf_cube_path_1 = self.indexer['cam1.interfero_cube']
        interf_cube_path_2 = self.indexer['cam2.interfero_cube']

        # detect stars in cube 1
        if not no_star:
            cube1 = self._init_raw_data_cube(1)
            star_list_path_1, mean_fwhm_1_arc = self.detect_stars(
                cube1, 0, saturation_threshold=self.config['SATURATION_THRESHOLD'])
            del cube1
            alignment_coeffs = None
            
        else:
            star_list_path_1 = None
            mean_fwhm_1_arc = None
            alignment_coeffs = [init_dx, init_dy,
                                self.config["INIT_ANGLE"], 0., 0.]

        # Init InterferogramMerger class
        self.indexer.set_file_group('merged')
        cube = InterferogramMerger(
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
            logfile_name=self._logfile_name,
            config_file_name=self.config_file_name)

        perf = Performance(cube.cube_B, "Cube B transformation", 2,
                           config_file_name=self.config_file_name)

        # find alignment coefficients
        if not no_star and alignment_coeffs is None:
            cube.find_alignment(
                star_list_path_1,
                self.config["INIT_ANGLE"], init_dx, init_dy,
                mean_fwhm_1_arc, self.config["FIELD_OF_VIEW"])
        else:
            self._print_msg("Alignment parameters: {} {} {} {} {}".format(
                cube.dx, cube.dy, cube.dr, cube.da, cube.db))
            

        # transform frames of cube B
        cube.transform(interp_order=interp_order)
        
        perf_stats = perf.print_stats()
        del cube, perf
        return perf_stats

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
        cube = InterferogramMerger(
            interf_cube_path_A=interf_cube_path_1,
            interf_cube_path_B=interf_cube_path_2,
            data_prefix=self._get_data_prefix(0),
            project_header=self._get_project_fits_header(0),
            alignment_coeffs=None,
            overwrite=self.overwrite,
            tuning_parameters=self.tuning_parameters,
            indexer=self.indexer,
            logfile_name=self._logfile_name,
            config_file_name=self.config_file_name)
        
        perf = Performance(cube.cube_A, "Alternative merging process", 1,
                           config_file_name=self.config_file_name)

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
        
        if "step_number" in self.options: 
            step_number = self.options["step_number"]
        else:
            self._print_error("No step number given, check the option file")
            
        # get frame list paths
        interf_cube_path_1 = self.indexer.get_path(
            'cam1.interfero_cube', err=True)
        interf_cube_path_2 = self.indexer.get_path(
            'merged.transformed_interfero_cube', err=True)
        
        # get noise values
        readout_noise_1, dark_current_level_1 = self.get_noise_values(1)
        readout_noise_2, dark_current_level_2 = self.get_noise_values(2)

        self.indexer.set_file_group('merged')
        cube = InterferogramMerger(
            interf_cube_path_A=interf_cube_path_1,
            interf_cube_path_B=interf_cube_path_2,
            data_prefix=self._get_data_prefix(0),
            project_header=self._get_project_fits_header(0),
            alignment_coeffs=None,
            overwrite=self.overwrite,
            tuning_parameters=self.tuning_parameters,
            indexer=self.indexer,
            logfile_name=self._logfile_name,
            config_file_name=self.config_file_name)
        
        perf = Performance(cube.cube_A, "Merging process", 1,
                           config_file_name=self.config_file_name)

        cube.merge(star_list_path_1, step_number,
                   mean_fwhm_arc, self.config["FIELD_OF_VIEW"],
                   add_frameB=add_frameB, 
                   smooth_vector=smooth_vector,
                   profile_name=self.config["PSF_PROFILE"],
                   moffat_beta=self.config["MOFFAT_BETA"],
                   compute_ext_light=(not self.options['no_sky']
                                      and self.config['EXT_ILLUMINATION']),
                   readout_noise_1=readout_noise_1,
                   dark_current_level_1=dark_current_level_1,
                   readout_noise_2=readout_noise_2,
                   dark_current_level_2=dark_current_level_2)
        
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
            self._print_warning('A path to a calibration laser map has already been given ({}), this step is skipped.'.format(
                self.options['calibration_laser_map_path']))

            return 

        if camera_number == 1:
            if "calib_path_1" in self.options: 
                calib_path = self.options["calib_path_1.hdf5"]
            else: 
                self._print_error("No path to the calibration laser files list given, check the option file")
        elif camera_number == 2:
            if "calib_path_2" in self.options: 
                calib_path = self.options["calib_path_2.hdf5"]
            else: 
                self._print_error("No path to the calibration laser files list given, check the option file")
        else:
            self._print_error("Camera number must be either 1 or 2")

        if self.target == 'laser':
            order = self.options['order']
            step = self.options['step']
        else:
            order = self.config["CALIB_ORDER"]
            step = self.config["CALIB_STEP_SIZE"]
            
        self._print_msg('Calibration laser observation parameters: step={}, order={}'.format(step, order))
            
        self.options["camera_number"] = camera_number
        
        
        self.indexer.set_file_group(camera_number)
        cube = CalibrationLaser(
            calib_path, 
            data_prefix=self._get_data_prefix(camera_number),
            calibration_laser_header=self._get_calibration_laser_fits_header(),
            overwrite=self.overwrite,
            tuning_parameters=self.tuning_parameters,
            indexer=self.indexer,
            logfile_name=self._logfile_name,
            config_file_name=self.config_file_name)
        perf = Performance(cube, "Calibration laser map processing",
                           camera_number,
                           config_file_name=self.config_file_name)

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
        if (camera_number != 1) and (camera_number != 2):
            self._print_error('This method (Orbs.orbs.correct_interferogram) is intended to be used only to correct single-camera interferograms (i.e. camera_number must be 1 or 2)')

        # Load interferogram frames
        interf_cube_path = self._get_interfero_cube_path(camera_number)
        
        # init cube
        self.options["camera_number"] = camera_number
        self.indexer.set_file_group(camera_number)
        cube = Interferogram(
            interf_cube_path, 
            data_prefix=self._get_data_prefix(camera_number),
            project_header = self._get_project_fits_header(
                camera_number),
            calibration_laser_header=self._get_calibration_laser_fits_header(),
            overwrite=self.overwrite,
            tuning_parameters=self.tuning_parameters,
            indexer=self.indexer,
            logfile_name=self._logfile_name,
            config_file_name=self.config_file_name)
        perf = Performance(cube, "Interferogram correction", camera_number,
                           config_file_name=self.config_file_name)
        
        # detect stars
        raw_cube = self._init_raw_data_cube(camera_number)
        star_list_path, mean_fwhm_arc = self.detect_stars(
            raw_cube, camera_number)
        del raw_cube

        if "step_number" in self.options: 
            step_number = self.options["step_number"]
        else:
            self._print_warning("No step number given, check the option file")
            
        # create correction vectors
        cube.create_correction_vectors(
            star_list_path, mean_fwhm_arc,
            self.config["FIELD_OF_VIEW"],
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


    def _get_calibration_laser_map(self, camera_number):
        """Return the calibration laser map.

        :param camera_number: Camera number (can be 1, 2 or 0)
        """
        if 'calibration_laser_map_path' in self.options:
            calibration_laser_map_path = self.options[
                'calibration_laser_map_path']
            self._print_msg('Using an external calibration laser map: %s'%(
                calibration_laser_map_path))
            
        else:
            if (camera_number == 0 or camera_number == 1):
                calibration_laser_map_path = self.indexer[
                    'cam1.calibration_laser_map']
            elif camera_number == 2:
                calibration_laser_map_path = self.indexer[
                    'cam2.calibration_laser_map']
            else:
                self._print_error("Camera number must be 0,1 or 2")

        if calibration_laser_map_path is None:
            self._print_warning("No calibration laser map found")
            return None
            
        if not os.path.exists(calibration_laser_map_path):
            self._print_warning("Calibration laser map not found ({} does not exist)".format(calibration_laser_map_path))
            return None
            
        return calibration_laser_map_path


    def compute_spectrum(self, camera_number, apodization_function=None,
                         polyfit_deg=1, n_phase=None,
                         phase_cube=False,
                         phase_coeffs=None,
                         smoothing_deg=2, no_star=False):

        """Run the computation of the spectrum from an interferogram
        cube.  
     
        :param apodization_function: (Optional) Apodization function. Default
          None.

        :param n_phase: (Optional) Number of points around ZPD to use
          for phase correction. If 0, no phase correction will be done
          and the resulting spectrum will be the absolute value of the
          complex spectrum. If None, the number of points is set to 50
          percent of the interferogram length (default None).

        :param polyfit_deg: (Optional) Degree of the polynomial fit to
          the computed phase. If < 0, no fit will be performed
          (Default 1).  

        :param phase_cube: (Optional) If True, only the phase cube is
          returned. The number of points of the phase can be defined
          with the option n_phase (default False).   

        :param phase_coeffs: (Optional) Polynomial coefficients of
          order higher than 0. If given those coefficients are used to
          define the phase vector. If none given default path to the
          phase maps of order > 0 are used to create it (Default
          None).   

        :param smoothing_deg: (Optional) Degree of zeros smoothing. A
          higher degree means a smoother transition from zeros parts
          (bad frames) to non-zero parts (good frames) of the
          interferogram. Good parts on the other side of the ZPD in
          symetry with zeros parts are multiplied by 2. The same
          transition is used to multiply interferogram points by zero
          and 2 (default 2).

        :param no_star: (Optional) If True, the cube is considered to
          have been computed without the star dependant processes so
          that the interferogram could not be corrected for sky
          transmission variations. The interferogram cube used will
          thus be the uncorrected one (default False).

        .. warning:: No calibration of any sort is made (e.g. no
          wavelength calibration) if the wanted cube is a spectral
          cube. This way, if an output in wavenumber is desired no
          interpolation has to be done. In the case of a phase cube
          the wavelength calibration is done.
      
        .. seealso:: :meth:`process.Interferogram.compute_spectrum`
        .. seealso:: :meth:`orb.utils.transform_interferogram`
        """
        # get calibration laser map path
        if phase_cube:
            calibration_laser_map_path = self._get_calibration_laser_map(
                camera_number)
        else:
            calibration_laser_map_path = None
      
        ## Load phase maps and create phase coefficients vector
        phase_map_correction = False

        if (phase_coeffs is None
            and not phase_cube and n_phase != 0):
            
            phase_map_correction = True
            
            # get phase map 0 path
            if 'phase_map_path' not in self.options:
                phase_map_0_path = self.indexer.get_path(
                        'phase_map_fitted_0', camera_number)
                if  phase_map_0_path is None:
                    self._print_warning("No phase map found for the zeroth order. Phase correction will not use phase maps and will be less accurate !")
                    phase_map_correction = False
            else:
                phase_map_0_path = self.options['phase_map_path']
                    
                self._print_msg("Oth order phase map taken from external source: %s"%phase_map_0_path)
                
            # get other phase map paths
            if phase_coeffs is None:
                phase_map_paths = list()
                for iorder in range(1, self.config["PHASE_FIT_DEG"] + 1):
                    phase_map_path = self.indexer.get_path(
                        'phase_map_%d'%iorder, camera_number)
                    if phase_map_path is None:
                        self._print_warning("No phase map found for the order %d. Phase correction will not use phase maps and will be less accurate !"%iorder)
                        phase_map_correction = False
                        
                    elif os.path.exists(phase_map_path):
                        phase_map_paths.append(phase_map_path)
                    else:
                        self._print_warning("No phase map found for the order %d. Phase correction will not use phase maps and will be less accurate !"%iorder)
                        phase_map_correction = False
                    
            # get residual map path
            residual_map_path = self.indexer.get_path(
                'phase_map_residual', camera_number)
            if residual_map_path is None:
                self._print_warning("No residual map path found. Phase correction will not use phase maps and will be less accurate !")
                phase_map_correction = False
            elif not os.path.exists(residual_map_path):
                self._print_warning("No residual map path found. Phase correction will not use phase maps and will be less accurate !")
                phase_map_correction = False

        else: phase_map_0_path = None
        
        # Load interferogram frames
        cube_path = self._get_interfero_cube_path(
            camera_number, corrected=True)
                      
        # Get final bad frames vector
        if 'bad_frames' in self.options:
            bad_frames_list = self.options['bad_frames']
        else: bad_frames_list = None
        bad_frames_vector = self.create_bad_frames_vector(
            camera_number,
            bad_frames_list=bad_frames_list)
        
        # init cube
        self.options["camera_number"] = camera_number
        self.indexer.set_file_group(camera_number)
        
        cube = Interferogram(
            cube_path, 
            data_prefix=self._get_data_prefix(camera_number),
            project_header = self._get_project_fits_header(
                camera_number),
            calibration_laser_header=self._get_calibration_laser_fits_header(),
            overwrite=self.overwrite,
            tuning_parameters=self.tuning_parameters,
            indexer=self.indexer,
            config_file_name=self.config_file_name)
        
        perf = Performance(cube, "Spectrum computation", camera_number,
                           config_file_name=self.config_file_name)
        
        if (camera_number == 1):
            if self.config["BALANCED_CAM"] == 1:
                balanced = True
            else: balanced = False
            
            if "bin_cam_1" in self.options: 
                bin_factor = self.options["bin_cam_1"]
            else: 
                self._print_error("No binning factor for the camera 1 given, check the option file")

        elif (camera_number == 2):
            if self.config["BALANCED_CAM"] == 2:
                balanced = True
            else: balanced = False
            
            if "bin_cam_2" in self.options: 
                bin_factor = self.options["bin_cam_2"]
            else: 
                self._print_error("No binning factor for the camera 2 given, check the option file")

        elif (camera_number == 0):
            balanced = True
            if "bin_cam_1" in self.options: 
                bin_factor = self.options["bin_cam_1"]
            else: 
                self._print_error("No binning factor for the camera 1 given, check the option file")
        else:
            self._print_error(
                "Please choose a correct camera number : 0, 1 or 2")

        if "step" in self.options:
            step = self.options["step"]
        else: 
            self._print_error("No step size given, check the option file")
        if "order" in self.options:
            order = self.options["order"]
        else: 
            self._print_error("No folding order given, check the option file")

        if 'apodization_function' in self.options and apodization_function is None:
            apodization_function = self.options['apodization_function']
            
        if apodization_function is not None:
            if apodization_function not in self._APODIZATION_FUNCTIONS:
                self._print_error("Unrecognized apodization function. Please try : " + str(self._APODIZATION_FUNCTIONS))

        if "fringes" in self.options:
            fringes=self.options['fringes']
        else: fringes = None

        # wavenumber option
        wavenumber = False
        if not phase_cube:
            if 'wavenumber' in self.options:
                wavenumber = self.options['wavenumber']
                
        ## Compute phase coeffs vector
        if (phase_coeffs is None
            and not phase_cube
            and phase_map_correction):
            phase_coeffs = cube.compute_phase_coeffs_vector(
                phase_map_paths,
                residual_map_path=residual_map_path)

        ## Compute spectrum
        cube.compute_spectrum(
            calibration_laser_map_path, bin_factor, step, order,
            self.config["CALIB_NM_LASER"],
            bad_frames_vector=bad_frames_vector,
            n_phase=n_phase,
            polyfit_deg=polyfit_deg,
            window_type=apodization_function,
            phase_cube=phase_cube,
            phase_map_0_path=phase_map_0_path,
            phase_coeffs=phase_coeffs,
            filter_file_path=self._get_filter_file_path(
                self.options["filter_name"]),
            balanced=balanced,
            smoothing_deg=smoothing_deg,
            fringes=fringes,
            wavenumber=wavenumber)

        perf_stats = perf.print_stats()
        del cube, perf
        return perf_stats


    def compute_phase(self, camera_number, n_phase=None):
        """Create a phase cube.

        :param camera_number: Camera number (must be 1, 2 or 0 for
          merged data)

        :param n_phase: (Optional) Number of points around ZPD to use
          for phase correction. If 0, no phase correction will be done
          and the resulting spectrum will be the absolute value of the
          complex spectrum. If None, the number of points is set to 50
          percent of the interferogram length (default None).
        """
        self.compute_spectrum(camera_number, apodization_function='2.0', 
                              n_phase=n_phase, phase_cube=True)


    def compute_phase_maps(self, camera_number, fit=True,
                           no_star=False, flat_cube=False):
        
        """Create a phase map.

        The phase map is a map of the zeroth order coefficient of the
        polynomial fit to the phase. The dimensions of the phase map
        are the same as the dimensions of the frames of the phase
        cube.
        
        :param camera_number: Camera number (must be 1, 2 or 0 for
          merged data).

        :param fit: (Optional) If True the computed phase map is
          fitted to remove noise. Especially useful if the phase map
          is created from the astronomical data cube itself and not
          from a flat cube (default True).

        :param no_star: (Optional) If True, the cube is considered to
          have been computed without the star dependant processes so
          that the interferogram could not be corrected for sky
          transmission variations. The interferogram cube used will
          thus be the uncorrected one (default False).

        :param flat_cube: (Optional) If True, cube is considered to be
          a flat cube with a high SNR at all wavelengths.
        
        .. seealso:: :meth:`process.Phase.create_phase_maps`
        
        """
        if "step" in self.options:
            step = self.options["step"]
        else: 
            self._print_error("No step size given, check the option file")
        if "order" in self.options:
            order = self.options["order"]
        else: 
            self._print_error("No folding order given, check the option file")

        filter_path = self._get_filter_file_path(self.options["filter_name"])
        if filter_path is None:
            self._print_warning("Unknown filter name.")
            
        # get calibration laser map path   
        calibration_laser_map_path = self._get_calibration_laser_map(
            camera_number)

        # get default phase list path
        if camera_number == 0:
            phase_cube_path = self.indexer['merged.phase_cube']
        elif camera_number == 1:
            phase_cube_path = self.indexer['cam1.phase_cube']
        elif camera_number == 2:
            phase_cube_path = self.indexer['cam2.phase_cube']
        else:
            self._print_error('Camera number must be 1, 2 or 0')
                
        # get default interferogram length
        if not no_star:
            interfero_cube_path = self._get_interfero_cube_path(
                camera_number, corrected=True)
        else:
            interfero_cube_path = self._get_interfero_cube_path(
                camera_number, corrected=False)
        cube = Interferogram(interfero_cube_path, silent_init=True,
                             logfile_name=self._logfile_name,
                             config_file_name=self.config_file_name)
        interferogram_length = cube.dimz
        del cube

        self.indexer.set_file_group(camera_number)
        phase = Phase(
            phase_cube_path, 
            data_prefix=self._get_data_prefix(camera_number),
            project_header = self._get_project_fits_header(
                camera_number),
            calibration_laser_header=self._get_calibration_laser_fits_header(),
            overwrite=self.overwrite,
            tuning_parameters=self.tuning_parameters,
            indexer=self.indexer,
            logfile_name=self._logfile_name,
            config_file_name=self.config_file_name)
        
        perf = Performance(phase, "Phase map creation", camera_number,
                           config_file_name=self.config_file_name)
        # create phase map
        phase.create_phase_maps(
            calibration_laser_map_path,
            filter_path,
            self.config["CALIB_NM_LASER"], step, order,
            interferogram_length=interferogram_length,
            fit_order=self.config["PHASE_FIT_DEG"],
            flat_cube=flat_cube)

        # smooth the 0th order phase map
        phase_map_path = phase._get_phase_map_path(0)
        phase.smooth_phase_map(phase_map_path)
        if fit:
            # fit the 0th order phase map
            phase_map_path = phase._get_phase_map_path(
                0, phase_map_type='smoothed')
            residual_map_path = phase._get_phase_map_path(
                0, phase_map_type='residual') 
            phase.fit_phase_map(phase_map_path, residual_map_path)
        perf_stats = perf.print_stats()
        del perf
        return perf_stats
        
    def calibrate_spectrum(self, camera_number, cam1_scale=False,
                           no_star=False):
        
        """Calibrate spectrum cube and correct WCS.

        :param camera_number: Camera number (can be 1, 2 or
          0 for merged data).   

        :param cam1_scale: (Optional) If True scale map used is cam 1
          deep frame. Useful for SpIOMM which cam 2 frames cannot be
          well corrected for bias. This option is used only for a
          two-camera calibration process (default False).

        :param no_star: (Optional) If True, data is considered to
          contain no star, so no WCS calibration is possible (default
          False).

        .. seealso:: :py:class:`process.Spectrum`
        """
        if "step" in self.options:
            step = self.options["step"]
        else: 
            self._print_error("No step size given, check the option file")
        if "order" in self.options:
            order = self.options["order"]
        else: 
            self._print_error("No folding order given, check the option file")
                    
        if "target_ra" in self.options:
            target_ra = self.options["target_ra"]
        else: target_ra = None
            
        if "target_dec" in self.options:
            target_dec = self.options["target_dec"]
        else: target_dec = None
        
        if "target_x" in self.options:
            target_x = self.options["target_x"]
        else: target_x = None
        
        if "target_y" in self.options:
            target_y = self.options["target_y"]
        else: target_y = None

        # Check if filter file exists
        filter_path = self._get_filter_file_path(self.options["filter_name"])
        if filter_path is None:
            self._print_warning(
                "Unknown filter. No filter correction can be made")
    
        spectrum_cube_path = self.indexer.get_path(
            'spectrum_cube', camera_number)

        self.indexer.set_file_group(camera_number)
        spectrum = Spectrum(
            spectrum_cube_path, 
            data_prefix=self._get_data_prefix(camera_number),
            project_header = self._get_project_fits_header(
                camera_number),
            calibration_laser_header=self._get_calibration_laser_fits_header(),
            overwrite=self.overwrite,
            tuning_parameters=self.tuning_parameters,
            indexer=self.indexer,
            logfile_name=self._logfile_name,
            config_file_name=self.config_file_name)
        perf = Performance(spectrum, "Spectrum calibration", camera_number,
                           config_file_name=self.config_file_name)

        # Get flux calibration vector
        if ('standard_path' in self.options
            and 'standard_name' in self.options
            and filter_path is not None):
            std_path = self.options['standard_path']
            std_name = self.options['standard_name']
            flux_calibration_vector = spectrum.get_flux_calibration_vector(
                std_path, std_name, self.options['step'],
                self.options['order'], self.options['exp_time'],
                self._get_filter_file_path(self.options["filter_name"]))
        else:
            self._print_warning("Standard related options were not given or the name of the filer is unknown. Flux calibration cannot be done")
            flux_calibration_vector = None
        
        # Get WCS
        if (target_ra is None or target_dec is None
            or target_x is None or target_y is None):
            self._print_warning("Some WCS options were not given. WCS correction cannot be done.")
            correct_wcs = None
        elif no_star:
            self._print_warning("No-star reduction: no WCS calibration.")
            correct_wcs = None
        else:
            astrom = self._init_astrometry(spectrum, camera_number)
            correct_wcs = astrom.register(full_deep_frame=True)

        # Get deep frame
        if camera_number == 0 and cam1_scale:
            self._print_warning('Flux rescaled relatively to camera 1')
            deep_frame_path = self.indexer.get_path('deep_frame', 1)
        else:
            deep_frame_path = self.indexer.get_path('deep_frame', camera_number)
        
        # check wavelength calibration
        calibration_laser_map_path = self._get_calibration_laser_map(
            camera_number)
            
        # Calibration
        spectrum.calibrate(
            filter_path, step, order,
            calibration_laser_map_path,
            self.config['CALIB_NM_LASER'],
            correct_wcs=correct_wcs,
            flux_calibration_vector=flux_calibration_vector,
            deep_frame_path=deep_frame_path,
            wavenumber=self.options['wavenumber'],
            standard_header = self._get_calibration_standard_fits_header(),
            spectral_calibration=self.options['spectral_calibration'])
        
        perf_stats = perf.print_stats()
        del perf, spectrum
        return perf_stats

    def extract_stars_spectrum(self, camera_number, apodization_function,
                               star_list_path=None,
                               aperture_photometry=True, n_phase=None,
                               auto_phase=False, filter_correct=True,
                               aper_coeff=3., saturation=None):
        
        """Extract the spectrum of the stars in a list of stars location
        list by photometry.
        
        :param camera_number: Camera number (can be 1, 2 or 0 for
          merged data).
          
        :param apodization_function: Apodization function to use during
          spectrum computation.

        :param star_list_path: (Optional) Path to a list of stars
          positions. If None, stars are autodetected.

        :param stars_fwhm_arc: (Optional) FWHM of the stars in
          arcsec. Used only when an external list of stars is given
          (using 'star_list_path=' parameter).

        :param min_star_number: (Optional) Minimum number of star to
          be detected by the automatic detection process (used if no
          path to a list of stars is given). Default 15.

        :param aperture_photometry: (Optional) If True, star flux is
          computed by aperture photometry. If False, star flux is
          computed from the results of the fit.

        :param n_phase: (Optional) Number of points around ZPD to use
          for phase correction. If 0, no phase correction will be done
          and the resulting spectrum will be the absolute value of the
          complex spectrum. If None, the number of points is set to 50
          percent of the interferogram length (default None).

        :param auto_phase: (Optional) If True, phase is computed for
          each star independantly. Useful for high SNR stars when no
          reliable external phase can be provided (e.g. Standard
          stars). Note that if auto_phase is set to True, phase will
          be corrected even if n_phase is set to 0. (default False).

        :param filter_correct: (Optional) If True returned spectra
          are corrected for filter. Points out of the filter band
          are set to NaN (default True).

        :param aper_coeff: (Optional) Aperture coefficient. The
          aperture radius is Rap = aper_coeff * FWHM. Better when
          between 1.5 to reduce the variation of the collected photons
          with varying FWHM and 3. to account for the flux in the
          wings (default 3., better for star with a high SNR).
          
        :param saturation: (Optional) If not None, all pixels above
          the saturation level are removed from the fit (default None).
        """

        self._print_msg('Extracting stars spectra', color=True)

        if 'flat_spectrum_path' in self.options:
            flat_spectrum_path = self.options['flat_spectrum_path']
        else:
            flat_spectrum_path = None
        
        # detect stars
        if star_list_path is None:
            if camera_number == 0 or camera_number == 1:
                raw_cube = self._init_raw_data_cube(1)
            else:
                raw_cube = self._init_raw_data_cube(2)
            star_list_path, stars_fwhm_arc = self.detect_stars(
                raw_cube, camera_number, self.config['DETECT_STAR_NB'])
            del raw_cube
        else: stars_fwhm_arc = self.config['INIT_FWHM']
        
        # get frame list paths
        interf_cube_path_1 = self.indexer['cam1.interfero_cube']
        if camera_number == 0:
            interf_cube_path_2 = self.indexer[
                'merged.transformed_interfero_cube']

        # load calibration laser map
        if 'calibration_laser_map_path' in self.options:
            calibration_laser_map_path = self.options[
                'calibration_laser_map_path']
            self._print_msg('Using an external calibration laser map: %s'%(
                calibration_laser_map_path))
        else:
            calibration_laser_map_path = self.indexer[
                'cam1.calibration_laser_map']

        # get phase coefficents
        phase_map_0_path = None
        phase_coeffs = None
        if n_phase != 0 and not auto_phase:
            phase_map_0_path = self.indexer.get_path(
                            'phase_map_fitted_0', 0)
            phase_map_paths = list()
            for iorder in range(1, self.config["PHASE_FIT_DEG"] + 1):
                phase_map_path = self.indexer.get_path(
                    'phase_map_%d'%iorder, 0)
                if phase_map_path is not None:
                    phase_map_paths.append(phase_map_path)
            residual_map_path = self.indexer.get_path(
                'phase_map_residual', 0)

            if (phase_map_0_path is not None and residual_map_path is not None
                and len(phase_map_paths) > 0):
                cube = Interferogram(
                    '', config_file_name=self.config_file_name)
                phase_coeffs = cube.compute_phase_coeffs_vector(
                    phase_map_paths,
                    residual_map_path=residual_map_path)
                
        # get bad frames vector
        if "bad_frames" in self.options:
            bad_frames_list = self.options["bad_frames"]
        else: bad_frames_list = []
        bad_frames_vector = self.create_bad_frames_vector(
            camera_number,
            bad_frames_list=bad_frames_list)
        
        # check parameters
        if 'apodization_function' in self.options and apodization_function is None:
            apodization_function = self.options['apodization_function']
            
        if apodization_function is not None and apodization_function not in self._APODIZATION_FUNCTIONS:
                self._print_error("Unrecognized apodization function. Please try : " + str(self._APODIZATION_FUNCTIONS))

        if camera_number == 0:
            cube = InterferogramMerger(
                interf_cube_path_A=interf_cube_path_1,
                interf_cube_path_B=interf_cube_path_2,
                data_prefix=self._get_data_prefix(0),
                project_header=self._get_project_fits_header(0),
                alignment_coeffs=None,
                overwrite=self.overwrite,
                tuning_parameters=self.tuning_parameters,
                logfile_name=self._logfile_name,
                config_file_name=self.config_file_name)

            perf = Performance(cube.cube_A, "Extract stars spectrum", 0,
                               config_file_name=self.config_file_name)

            stars_spectrum = cube.extract_stars_spectrum(
                star_list_path,
                self.config["INIT_FWHM"], self.config["FIELD_OF_VIEW"],
                cube._get_modulation_ratio_path(),
                cube._get_transmission_vector_path(),
                cube._get_ext_illumination_vector_path(),
                calibration_laser_map_path, self.options['step'],
                self.options['order'], self.config["CALIB_NM_LASER"],
                self._get_filter_file_path(self.options["filter_name"]),
                self.options['step_number'],
                window_type=apodization_function,
                bad_frames_vector=bad_frames_vector,
                phase_map_0_path=phase_map_0_path,
                phase_coeffs=phase_coeffs,
                aperture=aperture_photometry,
                profile_name=self.config["PSF_PROFILE"],
                moffat_beta=self.config["MOFFAT_BETA"],
                n_phase=n_phase, 
                auto_phase=auto_phase, filter_correct=filter_correct,
                flat_spectrum_path=flat_spectrum_path,
                aper_coeff=aper_coeff,
                saturation=saturation)

            perf.print_stats()
        else:
            cube = Interferogram(
                interf_cube_path_1,          
                data_prefix=self._get_data_prefix(camera_number),
                project_header = self._get_project_fits_header(
                    camera_number),
                calibration_laser_header=
                self._get_calibration_laser_fits_header(),
                overwrite=self.overwrite,
                tuning_parameters=self.tuning_parameters,
                indexer=self.indexer,
                logfile_name=self._logfile_name,
                config_file_name=self.config_file_name)
            
            stars_spectrum = cube.extract_stars_spectrum(
                star_list_path,
                self.config["INIT_FWHM"], self.config["FIELD_OF_VIEW"],
                cube._get_transmission_vector_path(),
                cube._get_stray_light_vector_path(),
                calibration_laser_map_path, self.options['step'],
                self.options['order'], self.config["CALIB_NM_LASER"],
                self._get_filter_file_path(self.options["filter_name"]),
                self.options['step_number'],
                window_type=apodization_function,
                bad_frames_vector=bad_frames_vector,
                aperture=aperture_photometry,
                profile_name=self.config["PSF_PROFILE"],
                moffat_beta=self.config["MOFFAT_BETA"],
                filter_correct=filter_correct,
                flat_spectrum_path=flat_spectrum_path,
                aper_coeff=aper_coeff,
                saturation=saturation)
            
        return stars_spectrum


    def export_calibration_laser_map(self, camera_number):
        """Export the computed calibration laser map at the root of the
        reduction folder.

        :param camera_number: Camera number (must be 1, 2 or 0 for
          merged data).
        """
        self._print_msg('Writing calibration laser map to disk', color=True)
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
        self._print_msg('Writing flat phase map to disk', color=True)
        phase_map_path = self.indexer.get_path('phase_map_0', camera_number)
        phase_map_data, phase_map_hdr = self.read_fits(phase_map_path,
                                                       return_header=True)
        self.write_fits(self._get_flat_phase_map_path(camera_number),
                        phase_map_data, fits_header=phase_map_hdr,
                        overwrite=self.overwrite)
        

    def export_calibrated_spectrum_cube(self, camera_number):
        """Extract a calibrated spectrum cube from the 'frame-divided'
        calibrated spectrum cube resulting of the reduction
        process. Write this cube at the root of the reduction folder.

        :param camera_number: Camera number (must be 1, 2 or 0 for
          merged data).
        """
        self._print_msg('Writing calibrated spectrum cube to disk', color=True)
        spectrum_cube_path = self.indexer.get_path('calibrated_spectrum_cube',
                                                   camera_number)
        spectrum = HDFCube(spectrum_cube_path,
                           config_file_name=self.config_file_name)
        spectrum_header = spectrum.get_cube_header()

        if 'wavenumber' in self.options:
            wavenumber = self.options['wavenumber']
        else:
            wavenumber = False

        if not wavenumber:
            axis = orb.utils.create_nm_axis(
                spectrum.dimz, self.options['step'], self.options['order'])
        else:
            axis = orb.utils.create_cm1_axis(
                spectrum.dimz,  self.options['step'], self.options['order'])
        
        spectrum_header.extend(
            self._get_basic_spectrum_cube_header(
                axis, wavenumber= wavenumber),
            strip=True, update=False, end=True)
        
        spectrum_header.set('FILETYPE', 'Calibrated Spectrum Cube')

        apod = spectrum_header['APODIZ']
        spectrum_path = self._get_calibrated_spectrum_cube_path(
            camera_number, apod, wavenumber=wavenumber,
            spectral_calibration=self.options['spectral_calibration'])
        
        spectrum.export(spectrum_path, header=spectrum_header,
                        overwrite=self.overwrite)

    def export_standard_spectrum(self, camera_number, n_phase=None,
                                 aperture_photometry=True,
                                 apodization_function='2.0',
                                 auto_phase=True):
        """Extract spectrum of the standard stars and write it at the
        root of the reduction folder.

        .. note:: The position of the standard star is defined in the
          option file with TARGETX and TARGETY keywords.

        :param camera_number: Camera number (must be 1, 2 or 0 for
          merged data).

        :param n_phase: (Optional) Number of points around ZPD to use
          for phase correction. If 0, no phase correction will be done
          and the resulting spectrum will be the absolute value of the
          complex spectrum. If None, the number of points is set to 50
          percent of the interferogram length (default None).

        :param apodization_function: (Optional) Apodization function to use for
          spectrum computation (default '2.0').

        :param aperture_photometry: (Optional) If True, star flux is
          computed by aperture photometry. If False, star flux is
          computed from the results of the fit.

        :param auto_phase: (Optional) If True, phase is computed for
          each star independantly. Useful for high SNR stars when no
          reliable external phase can be provided (e.g. Standard
          stars). Note that if auto_phase is set to True, phase will
          be corrected even if n_phase is set to 0. (default True).
        """
        std_list = [[self.options['target_x'], self.options['target_y']]]

        std_spectrum = self.extract_stars_spectrum(
            camera_number, apodization_function, star_list_path=std_list,
            aperture_photometry=aperture_photometry,
            n_phase=n_phase, auto_phase=auto_phase, filter_correct=True)[0]

        nm_axis = orb.utils.create_nm_axis(
            std_spectrum.shape[0], self.options['step'], self.options['order'])
        
        std_header = (self._get_project_fits_header()
                      + self._get_basic_header('Standard Spectrum')
                      + self._get_fft_params_header(apodization_function)
                      + self._get_basic_spectrum_cube_header(nm_axis))
        std_spectrum_path = self._get_standard_spectrum_path(camera_number)
        
        self.write_fits(std_spectrum_path, std_spectrum,
                        fits_header=std_header,
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
        self._quad_nb = cube.QUAD_NB
        self._print_msg("%s started for camera %d"%(self._process_name,
                                                    self._camera_number),
                        color='alt')

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
        self._print_msg(
            "%s performance stats:\n"%self._process_name +
            " > Camera number: %d\n"%(self._camera_number) +
            " > Data cube size: %d x %d x %d \n"%(self._sx,
                                                  self._sy,
                                                  self._sz) +
            " > Number of quadrants: %d\n"%(self._quad_nb) +
            " > Computation time: %d s\n"%(total_time) +
            " > Max memory used: %d Mb\n"%(int(max_mem / 1000.)) +
            " > Efficiency: %.3e s/pixel\n"%float(total_time/pix_nb),
            color=True)
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
               <kwarg name='n_phase'></kwarg>
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
    
    color_OKGREEN = '\033[92m'
    color_END = '\033[0m'
    color_KORED = '\033[91m'

    
    
    

    
    
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
            self._print_error('Roadmap {} does not exist'.format(
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
                self._print_error('Step {} found in {} not recorded in {}'.format(
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
            self._print_error('No step called {}'.format(step_name))

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
            self._print_error(
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
                color = self.color_OKGREEN
            else:
                status = 'not done'
                color = self.color_KORED
            
            print color + '  {} - {} {}: {}'.format(index, step['name'], step['cam'], status) + self.color_END
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
