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


try:
    import astropy
    import astropy.io.fits as pyfits
except:
    import pyfits
    import warnings
    warnings.warn('PyFITS is now a part of Astropy (http://www.astropy.org/). PyFITS support as a standalone module will be stopped soon. It is better to install Astropy. You can still keep PyFITS for other applications.', FutureWarning)
    
import pp
import bottleneck as bn
import pywcs

from orb.core import Tools, Cube, Indexer, OptionFile
from process import RawData, InterferogramMerger, Interferogram
from process import Phase, Spectrum, CalibrationLaser
from orb.astrometry import Astrometry
import orb.utils
import orb.constants
import orb.version

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

    :BINCAM1: Binning of the camera 1

    :BINCAM2: Binning of the camera 2

    :SPESTEP: Step size of the moving mirror (in nm)

    :SPESTNB: Number of steps

    :SPEORDR: Order of the spectral folding

    :SPEEXPT: Exposition time of the frames (in s)

    :SPEDART: Exposition time of the dark frames (in s)
    
    :OBSDATE: Observation date (YYYY-MM-DD)

    :HOUR_UT: UT hour of the observation (HH:MM:SS)

    :BADFRMS: List of bad frames indexes

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
    
    _WINDOW_TYPE = ["barthann","bartlett", "blackman", "blackmanharris",
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
        
        * bad_frames: BADFRMS
        
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

        * wavelength_calibration: WAVE_CALIB
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
    
    def __init__(self, option_file_path, config_file_name="config.orb",
                 overwrite=False, silent=False):
        """Initialize Orbs class.

        :param option_file_path: Path to the option file.

        :param config_file_name: (Optional) Name of the config file to
          use. Must be located in orbs/data/.

        :param overwrite: (Optional) If True, any existing FITS file
          created by Orbs will be overwritten during the reduction
          process (default False).

        :param silent: (Optional) If True no messages nor warnings are
          displayed by Orbs (useful for silent init).
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
                    else:
                        image_mode = None
                        chip_index = None

                    if 'prebinning' in self.options:
                        prebinning = self.options['prebinning']
                    else:
                        prebinning = None
                    
                    self.options[option_key] = self._create_list_from_dir(
                        value, list_file_path,
                        image_mode=image_mode, chip_index=chip_index,
                        prebinning=prebinning)
            elif not optional:
                self._print_error('option {} must be set'.format(key))

        self.option_file_path = option_file_path
        self.config_file_name = config_file_name
        self._logfile_name =  os.path.basename(option_file_path) + '.log'
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
        
        try:
            self._print_msg("Pyfits version: %s"%pyfits.__version__)
        except:
            self._print_msg("Astropy version: %s"%astropy.__version__)
        
        self._print_msg("Parallel Python version: %s"%pp.version)
        self._print_msg("Bottleneck version: %s"%bn.__version__)
        self._print_msg("PyWCS version: %s"%pywcs.__version__)
        
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

        # Print first the entire option file for log
        op_file = open(option_file_path)
        self._print_msg("Option file content :", color=True)
        for line in op_file:
            self._print_msg(line[:-1], no_hdr=True)

        # record some default options
        self.options["try_catalogue"] = False
        self.options['wavelength_calibration'] = True
        self.options['wavenumber'] = False
        
        # Parse the option file to get reduction parameters
        self.optionfile = OptionFile(option_file_path)
        store_option_parameter('object_name', 'OBJECT', str, optional=False)
        store_option_parameter('filter_name', 'FILTER', str, optional=False)
        store_option_parameter('bin_cam_1', 'BINCAM1', int, optional=False)
        store_option_parameter('bin_cam_2', 'BINCAM2', int, optional=False)
        store_option_parameter('step', 'SPESTEP', float, optional=False)
        store_option_parameter('step_number', 'SPESTNB', int, optional=False)
        store_option_parameter('order', 'SPEORDR', float, optional=False)
        store_option_parameter('exp_time', 'SPEEXPT', float, optional=False)
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
        store_option_parameter('wavelength_calibration', 'WAVE_CALIB', bool)
        store_option_parameter('prebinning', 'PREBINNING', int)
        # recompute the real data binning
        if 'prebinning' in self.options:
            if self.options['prebinning'] is not None:
                self.options['bin_cam_1'] = (self.options['bin_cam_1']
                                             * self.options['prebinning'])
                self.options['bin_cam_2'] = (self.options['bin_cam_2']
                                             * self.options['prebinning'])
        
        fringes = self.optionfile.get_fringes()
        if fringes is not None:
            self.options['fringes'] = fringes

        bad_frames = self.optionfile.get_bad_frames()
        if bad_frames is not None:
            self.options['bad_frames'] = bad_frames
            
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
        
        # compute nm min and nm max from step and order parameters
        if ("step" in self.options) and ("order" in self.options):
            self.options["nm_min"] = (1. / ((self.options["order"] + 1.) / 
                                            (2. * self.options["step"])))
            if self.options["order"] > 0:
                self.options["nm_max"] = (1. / (self.options["order"] / 
                                                (2. * self.options["step"])))
            else:
                self.options["nm_max"] = np.inf
                self._print_error('Order 0 is still not handled by ORBS! Sorry...')

        if (("object_name" not in self.options)
            or ("filter_name" not in self.options)):
            self._print_error("The option file needs at least an object name (use keyword : OBJECT) and a filter name (use keyword : FILTER)")
        else:
            self.options["project_name"] = (self.options["object_name"] 
                                            + "_" + self.options["filter_name"])

        # get folders paths
        self._print_msg('Reading data folders and checking files')
        store_option_parameter('image_list_path_1', 'DIRCAM1', str, True, 1)
        store_option_parameter('image_list_path_2', 'DIRCAM2', str, True, 2)
        store_option_parameter('bias_path_1', 'DIRBIA1', str, True, 1)
        store_option_parameter('bias_path_2', 'DIRBIA2', str, True, 2)
        store_option_parameter('dark_path_1', 'DIRDRK1', str, True, 1)
        store_option_parameter('dark_path_2', 'DIRDRK2', str, True, 2)
        store_option_parameter('flat_path_1', 'DIRFLT1', str, True, 1)
        store_option_parameter('flat_path_2', 'DIRFLT2', str, True, 2)
        store_option_parameter('calib_path_1', 'DIRCAL1', str, True, 1)
        store_option_parameter('calib_path_2', 'DIRCAL2', str, True, 2)
        store_option_parameter('flat_spectrum_path', 'DIRFLTS', str, True)
                    
        # Check step number and number of raw images
        if (('image_list_path_1' in self.options)
            and ('image_list_path_2' in self.options)):
            dimz1 = Cube(self.options['image_list_path_1'],
                         silent_init=True).dimz
            dimz2 = Cube(self.options['image_list_path_2'],
                             silent_init=True).dimz
            if dimz1 != dimz2:
                self._print_error('The number of images of CAM1 and CAM2 are not the same (%d != %d)'%(dimz1, dimz2))
            if self.options['step_number'] < dimz1:
                self._print_error('The number of steps (%d) of a full cube must be greater or equal to the number of images given for CAM1 and CAM2 (%d)'%(
                    self.options['step_number'], dimz1))

        # Init Indexer
        self.indexer = Indexer(data_prefix=self.options['object_name']
                               + '_' + self.options['filter_name'] + '.')
        self.indexer.load_index()

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
        """Return path to the order 0 phase map from a flat cube
        
        :param camera_number: Camera number (must be 1, 2 or 0 for
          merged data).
        """
        return (self._get_root_data_path_hdr(camera_number)
                + 'flat_phase_map.fits')
    
    def _get_calibrated_spectrum_cube_path(self, camera_number, apod,
                                           wavenumber=False,
                                           wavelength_calibration=True):
        """Return path to the calibrated spectrum cube resulting of the
        reduction process
        
        :param camera_number: Camera number (must be 1, 2 or 0 for
          merged data).

        :param apod: Apodization function name to be added to the
          path.

        :param wavenumber: If True the spectral axis of the cube is
          considered to be a wavenumber axis. If False it is
          considered to be a wavelength axis (default False).
        """
        if wavenumber: wave_type = 'cm1'
        else: wave_type = 'nm'
        if wavelength_calibration: calib = ''
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
                cube =  RawData(
                    self.options["image_list_path_1"], 
                    data_prefix=self._get_data_prefix(1),
                    project_header=self._get_project_fits_header(1),
                    overwrite=self.overwrite,
                    tuning_parameters=self.tuning_parameters,
                    indexer=self.indexer,
                    logfile_name=self._logfile_name)
            else:
                self._print_error("No image list file for camera 1 given, please check option file")
        elif (camera_number == 2):
            if ("image_list_path_2" in self.options):
                self.indexer.set_file_group('cam2')
                cube =  RawData(
                    self.options["image_list_path_2"], 
                    data_prefix=self._get_data_prefix(2),
                    project_header=self._get_project_fits_header(2),
                    overwrite=self.overwrite,
                    tuning_parameters=self.tuning_parameters,
                    indexer=self.indexer,
                    logfile_name=self._logfile_name)
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
                    0., 0., xrc, yrc, zoom)
               
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
                          wcs_rotation=wcs_rotation)


    def _get_interfero_list_path(self, camera_number, corrected=False):
        """Return the path to the file containing the list of 
        the interferogram frames computed for each camera or
        the merged interferogram (camera_number = 0)

        :param camera_number: Camera number (can be 1, 2 or 0 
          for merged data).
        """
        if camera_number == 0:
            return self.indexer.get_path(
                'merged.merged_interfero_frame_list', err=True)
        elif camera_number == 1:
            if corrected:
                return self.indexer.get_path('cam1.corr_interf_list', err=True)
            else:
                return self.indexer.get_path('cam1.interfero_list', err=True)
        elif camera_number == 2:
            if corrected:
                return self.indexer.get_path('cam2.corr_interf_list', err=True)
            else:
                return self.indexer.get_path('cam2.interfero_list', err=True)
        else: self._print_error('Camera number must be 0, 1 or 2')

    def set_init_angle(self, init_angle):
        """Change config variable :py:const:`~orbs.Orbs.INIT_ANGLE`. 

        You can also change it by editing the file
        :file:`orbs/data/config.orb`.

        .. note:: This value is modified only for this instance of
           Orbs class. The initial value stored in the file
           'config.orb' will be restored at the next initialisation of
           the class.

        :param init_angle: the new value 
           of :py:const:`~orbs.Orbs.INIT_ANGLE`
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
        interf_cube = Cube(self._get_interfero_list_path(camera_number))
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


    def full_reduction(self, calibration_laser_map_path=None,
                       phase_map_0_path=None, bad_frames_vector=[],
                       alignment_coeffs=None,
                       apodization_function=None, start_step=1,
                       create_stars_cube=False, n_phase=None,
                       no_star=False, phase_map_only=False,
                       no_sky=False, alt_merge=False,
                       save_as_quads=False, standard=False):
        
        """Run the whole reduction process for two cameras using
        default options

        :param calibration_laser_map_path: (Optional) Path to an
           already computed calibration laser map. If a calibration
           map is given the calibration laser map computation step is
           skipped.

        :param phase_map_0_path: (Optional) Path to an already
          computed phase map (0th order) from an other cube (e.g. a
          flat cube). If a phase map is given the phase map
          computation step is skipped.
    
        :param bad_frames_vector: (Optional) Vector containing indexes 
           of frames considered as bad (presence of plane, satellites, 
           important flux loss due to too much clouds)

        :param apodization_function: (Optional) Name of the
           apodization function to be used during the spectrum
           computation.

        :param start_step: (Optional) Starting step number. Use it to
           recover from an error at a certain step without having to
           run the whole process one more time.

        :param create_stars_cube: (Optional) Compute only the spectrum
           of the detected stars in the cube. The interferogram of
           each star is created using a 2D gaussian fit (see:
           :py:meth:`~orbs.Orbs.merge_interferograms`).
           
        :param n_phase: (Optional) Number of points around ZPD to use
           for phase correction during spectrum computation. If 0, no
           phase correction will be done and the resulting spectrum
           will be the absolute value of the complex spectrum. If
           None, the number of points is set to 50 percent of the
           interferogram length (default None).

        :param no_star: (Optional) All the star-dependant processes
          are skipped. The reduction is thus far less precise and must
          be used only on non-astronomical data. The cubes are merged
          using the default alignment parameters recorded in the
          configuration file (data/config.orb) (default False).

        :param phase_map_only: (Optional) The reduction stops to the
          phase map step (7). This option is best used to reduce flat
          cube in order to obtain a high resolution phase map. The
          phase map (zeroth order of the polynomial fit) is computed
          from the phase cube. Note that a phase map cannot be created
          from a stars cube. Those options are not compatible. To use
          a phase map for a star cube, a normal cube must have been
          computed first (default False).

        :param alignment_coeffs: (Optional) If the alignments
          coefficients are given the alignment step is skipped and the
          images of the cube B are transformed using the given
          alignment coefficients. Must a vector giving [dx, dy, dr,
          da, db] (see: :py:meth:`~orbs.Orbs.transform_cube_B`)
          (Default None)

        :param no_sky: (Optional) If intense emission lines are
          present in the whole area, no 'sky' like pixels (dominated
          by the continuum) are present. In this case sky dependant
          processes must be skipped.

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

        :param standard: (Optional) If True, the cube is considered to
          be a cube of a standard star. Return only the spectrum of
          the standard instead of returning a full cube. Target
          position MUST be defined in the option file. It must be the
          standard star position. Note that the returned spectrum will
          not be corrected for the filter (default False).
        
        .. note:: The steps are:

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

            6. Compute calibration laser map (see:
               :py:meth:`~orbs.Orbs.compute_calibration_laser_map`)

            7. Compute phase map (see:
               :py:meth:`~orbs.Orbs.compute_phase_maps`)

            8. Compute spectrum (see:
               :py:meth:`~orbs.Orbs.compute_spectrum`)

            9. Calibrate spectrum (see:
               :py:meth:`~orbs.Orbs.calibrate_spectrum`)

        """
        if no_star: alt_merge = True
        
        if bad_frames_vector is None and "bad_frames" in self.options:
            bad_frames_vector = self.options["bad_frames"]
        elif "bad_frames" in self.options:
            self._print_warning("Bad frames defined in the option file and passed also using 'bad frames_vector' argument. In this case 'bad_frames_vector' argument has priority. Bad frames are: %s"%str(bad_frames_vector))

        if bad_frames_vector is None:
            bad_frames_vector = []
        else:
            bad_frames_vector = bad_frames_vector
                
        if (start_step <= 1) and (not no_star):
            self.compute_alignment_vector(
                1, min_star_number=self.config['DETECT_STAR_NB'])
            self.compute_alignment_vector(
                2, min_star_number=self.config['DETECT_STAR_NB'])
        if no_star:
            self._print_warning(
                "Alignment step skipped (option no_star set to True)")
            
        if (start_step <= 2) and (not no_star):
            if not standard:
                self.compute_cosmic_ray_map(
                    1, bad_frames_vector=bad_frames_vector)
                self.compute_cosmic_ray_map(
                    2, bad_frames_vector=bad_frames_vector)
            else: self._print_warning(
                    'Cosmic ray map not computed for Standard Cube reduction')
            
        if (start_step <= 3):
            self.compute_interferogram(
                1, bad_frames_vector=bad_frames_vector,
                optimize_dark_coeff=self.config["OPTIM_DARK_CAM1"])
            self.compute_interferogram(
                2, bad_frames_vector=bad_frames_vector,
                optimize_dark_coeff=self.config["OPTIM_DARK_CAM2"])
            
        if (start_step <= 4):   
            self.transform_cube_B(
                no_star=no_star,
                alignment_coeffs=alignment_coeffs,
                min_star_number=self.config['DETECT_STAR_NB'])

        if (start_step <= 5):
            if alt_merge:
                self.merge_interferograms_alt()
            else:
                self.merge_interferograms(
                    create_stars_cube=create_stars_cube,
                    bad_frames_vector=bad_frames_vector,
                    compute_ext_light=(
                        (not no_sky and self.config['EXT_ILLUMINATION'])),
                    min_star_number=self.config['DETECT_STAR_NB'])
            self.add_missing_frames(0, stars_cube=create_stars_cube)
            
        if (start_step <= 6):   
            if (calibration_laser_map_path is None and
                'calibration_laser_map_path' not in self.options):
                self.compute_calibration_laser_map(1)
                
        if (start_step <= 7) and n_phase != 0 and not standard:
            self._print_msg("Phase map computation", color=True)
            if not create_stars_cube:
                self.compute_spectrum(
                    0, calibration_laser_map_path=calibration_laser_map_path,
                    window_type='2.0', 
                    stars_cube=create_stars_cube,
                    n_phase=n_phase,
                    phase_cube=True,
                    balanced=True)
                self.compute_phase_maps(
                    0, calibration_laser_map_path=calibration_laser_map_path)
                if phase_map_only:
                    self.get_flat_phase_map(0)
        if n_phase == 0:
            self._print_warning("No phase maps computed because the number of points to use for phase computation (n_phase) is equal to 0")
            
        if (start_step <= 8) and not phase_map_only and not standard:
            self.compute_spectrum(
                0, calibration_laser_map_path=calibration_laser_map_path,
                window_type=apodization_function, 
                stars_cube=create_stars_cube,
                n_phase=n_phase,
                phase_map_0_path=phase_map_0_path,
                phase_cube=False,
                balanced=True)
            
        if (start_step <= 9) and not phase_map_only:
            if not standard:
                self.calibrate_spectrum(0, stars_cube=create_stars_cube)
                self.get_calibrated_spectrum_cube(0)
            else:
                self.get_standard_spectrum(0, auto_phase=True)
                

    def single_reduction(self, camera_number=1,
                         calibration_laser_map_path=None,
                         phase_map_0_path=None,
                         bad_frames_vector=None,
                         apodization_function=None, start_step=1,
                         n_phase=None, no_star=False,
                         phase_map_only=False,
                         save_as_quads=False,
                         standard=False):
        
        """Run the whole reduction process for one camera using
        default options

        :param camera_number: (Optional) Number of the camera to be
          reduced. Can be 1 or 2 (Default 1).

        :param calibration_laser_map_path: (Optional) Path to an
           already computed calibration laser map. If a calibration
           laser map is given the calibration laser map computation
           step is skipped.

        :param phase_map_0_path: (Optional) Path to an already
          computed phase map from an other cube (e.g. a flat cube). If
          a phase map is given the phase map computation step is
          skipped.
        
        :param bad_frames_vector: (Optional) Vector containing indexes 
           of frames considered as bad (presence of plane, satellites, 
           important flux loss due to too much clouds)

        :param apodization_function: (Optional) Name of the
           apodization function to be used during the spectrum
           computation.

        :param start_step: (Optional) Starting step. Use it to recover
           from an error at a certain step without having to run the
           whole process one more time.

        :param n_phase: (Optional) Number of points around ZPD to use
           for phase correction during spectrum computation. If 0, no
           phase correction will be done and the resulting spectrum
           will be the absolute value of the complex spectrum. If
           None, the number of points is set to 50 percent of the
           interferogram length (default None).

        :param no_star: (Optional) All the star-dependant processes
          are skipped. The reduction is thus far less precise and must
          be used only on non-astronomical data (default False).

        :param phase_map_only: (Optional) The reduction stops to the
          phase map step (5). This option is best used to reduce flat
          cube in order to obtain a high resolution phase map. The
          phase map (zeroth order of the polynomial fit) is computed
          from the phase cube. Note that a phase map cannot be created
          from a stars cube. Those options are not compatible. To use
          a phase map for a star cube, a normal cube must have been
          computed first.

        :param save_as_quads: (Optional) If True, final calibrated
          spectrum is saved as quadrants instead of being saved as a
          full cube. Quadrants can be read independantly. This option
          is useful for big data cubes (default False).

        :param standard: (Optional) If True, the cube is considered to
          be a cube of a standard star. Instead of returning a full
          cube return the spectrum of the standard. The standard star
          position must be the target position defined in the option
          file (default False).

        .. note:: The step numbers are:

            1. Compute alignment vectors (see:
               :py:meth:`~orbs.Orbs.compute_alignment_vector`)

            2. Compute cosmic ray maps (see:
               :py:meth:`~orbs.Orbs.compute_cosmic_ray_map`)

            3. Compute interferograms (see:
               :py:meth:`~orbs.Orbs.compute_interferogram`)

            4. Correct interferogram (see:
               :py:meth:`~orbs.Orbs.correct_interferogram`)
               
            5. Compute calibration laser map (see:
               :py:meth:`~orbs.Orbs.compute_calibration_laser_map`)

            6. Compute phase map (see:
               :py:meth:`~orbs.Orbs.compute_phase_maps`)

            7. Compute spectrum (see:
               :py:meth:`~orbs.Orbs.compute_spectrum`)

            8. Calibrate spectrum (see:
               :py:meth:`~orbs.Orbs.calibrate_spectrum`)
        """
        if bad_frames_vector is None and "bad_frames" in self.options:
            bad_frames_vector = self.options["bad_frames"]
        elif "bad_frames" in self.options:
            self._print_warning("Bad frames defined in the option file and passed also using 'bad frames_vector' argument. In this case 'bad_frames_vector' argument has priority. Bad frames are: %s"%str(bad_frames_vector))

        if bad_frames_vector==None:
            #bad_frames = self.check_bad_frames(camera_number)
            bad_frames_vector = []
        else:
            bad_frames_vector = bad_frames_vector
                
        if camera_number == 1:
            if self.config["BALANCED_CAM"] == 1:
                balanced = True
            else: balanced = False
            optimize_dark_coeff=self.config["OPTIM_DARK_CAM1"]
        elif camera_number == 2:
            if self.config["BALANCED_CAM"] == 2:
                balanced = True
            else: balanced = False
            optimize_dark_coeff=self.config["OPTIM_DARK_CAM2"]
        else:
            self._print_error("Camera number must be 1 or 2")
        
        if (start_step <= 1) and not (no_star):
            self.compute_alignment_vector(
                camera_number,
                min_star_number=self.config['DETECT_STAR_NB'])
        if no_star:
            self._print_warning(
                "Alignment step skipped (option no_star set to True)")
            
        if (start_step <= 2) and (not no_star):
            if not standard:
                self.compute_cosmic_ray_map(camera_number,
                                            bad_frames_vector=bad_frames_vector)
            else: self._print_warning(
                'Cosmic ray map not computed for Standard Cube reduction')
            
        if (start_step <= 3):
            self.compute_interferogram(
                camera_number,
                bad_frames_vector=bad_frames_vector,
                optimize_dark_coeff=optimize_dark_coeff)
            self.add_missing_frames(camera_number)
            
        if (start_step <= 4) and not (no_star):
            self.correct_interferogram(
                camera_number,
                bad_frames_vector=bad_frames_vector,
                min_star_number=self.config['DETECT_STAR_NB'])
            
        if (start_step <= 5):
            if (calibration_laser_map_path is None and
                'calibration_laser_map_path' not in self.options):
                self.compute_calibration_laser_map(camera_number)
                
        if (start_step <= 6) and n_phase != 0 and not standard:
            self._print_msg("Phase map computation", color=True)
            self.compute_spectrum(
                camera_number,
                calibration_laser_map_path=calibration_laser_map_path,
                window_type='2.0',
                n_phase=n_phase,
                phase_cube=True,
                balanced=balanced,
                no_star=no_star)
            self.compute_phase_maps(
                camera_number,
                calibration_laser_map_path=calibration_laser_map_path,
                no_star=no_star)
            if phase_map_only:
                self.get_flat_phase_map(camera_number)
                
        if n_phase == 0:
            self._print_warning("No phase maps computed because the number of points to use for phase computation (n_phase) is equal to 0")
            
        if (start_step <= 7) and not phase_map_only and not standard:
            self.compute_spectrum(
                camera_number,
                calibration_laser_map_path=calibration_laser_map_path, 
                window_type=apodization_function,
                n_phase=n_phase,
                phase_map_0_path=phase_map_0_path,
                phase_cube=False,
                balanced=balanced,
                no_star=no_star)
            
        if (start_step <= 8) and not phase_map_only:
            if not standard:
                self.calibrate_spectrum(camera_number)
                self.get_calibrated_spectrum_cube(camera_number)
            else:
                self.get_standard_spectrum(camera_number, auto_phase=True)


    def detect_stars(self, cube, camera_number, min_star_number,
                     saturation_threshold=35000, return_fwhm_pix=False):
        """Detect stars in a cube and save the star list in a file.

        This method is a simple wrapper around
        :py:meth:`orb.astrometry.Astrometry.detect_stars`
        
        :param cube: an orbs.Cube instance
        
        :param min_star_number: Minimum number of star to detect

        :param saturation_threshold: (Optional) Number of counts above
          which the star can be considered as saturated. Very low by
          default because at the ZPD the intensity of a star can be
          twice the intensity far from it (default 35000).

        :param return_fwhm_pix: (Optional) If True, the returned fwhm
          will be given in pixels instead of arcseconds (default
          False).
        
        :return: Path to a star list, mean FWHM of stars in arcseconds.

        .. seealso:: :py:meth:`orb.astrometry.Astrometry.detect_stars`
        """
        if ((camera_number == 1
            or camera_number == 0)
            and 'star_list_path_1' in self.options):
            self._print_msg('Using external star list: %s'%self.options['star_list_path_1'], color=True)
            star_list_path = self.options['star_list_path_1']
            mean_fwhm = self.config['INIT_FWHM']
            
        elif (camera_number == 2 and 'star_list_path_2' in self.options):
            self._print_msg('Using external star list: %s'%self.options['star_list_path_2'], color=True)
            star_list_path = self.options['star_list_path_2']
            mean_fwhm = self.config['INIT_FWHM']

        else:
            self._print_msg('Autodetecting stars', color=True)
            astrom = self._init_astrometry(cube, camera_number)
            star_list_path, mean_fwhm = astrom.detect_stars(
                min_star_number=min_star_number,
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
        dark_int_time = self.options["dark_time"]

        if bias_path is not None and dark_path is not None:
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


    def compute_alignment_vector(self, camera_number, star_list_path=None,
                                 min_star_number=15, stars_fwhm_arc=2.):
        """Run the computation of the alignment vector.

        If no path to a star list file is given 
        use: :py:meth:`orb.astrometry.Astrometry.detect_stars` 
        method to detect stars.

        :param camera_number: Camera number (can be either 1 or 2).
        
        :param star_list_path: (Optional) Path to the list of star
          coordinates. You must set the stars FWHM using 'stars_fwhm='
          parameter.
          
        :param min_star_number: (Optional) Minimum number of star that
          must be detected by
          :meth:`orb.astrometry.Astrometry.detect_stars`. Stars are used
          to align images

        :param stars_fwhm_arc: (Optional) FWHM of the stars in
          arcsec. Used only when an external list of stars is given
          (using 'star_list_path=' parameter).
          
        .. seealso:: :meth:`orb.astrometry.Astrometry.detect_stars`
        .. seealso:: :meth:`process.RawData.create_alignment_vector`
        """
        readout_noise, dark_current_level = self.get_noise_values(camera_number)
        
        cube = self._init_raw_data_cube(camera_number)
        
        perf = Performance(cube, "Alignment vector computation", camera_number)
        
        if star_list_path is None:
            star_list_path, mean_fwhm_arc = self.detect_stars(
                cube, camera_number, min_star_number)
        else:
            mean_fwhm_arc = stars_fwhm_arc

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

    def compute_cosmic_ray_map(self, camera_number, z_coeff=3.,
                               bad_frames_vector=[], min_star_number=30):
        """Run the computation of the cosmic ray map.

        :param camera_number: Camera number (can be either 1 or 2).
        
        :param z_coeff: (Optional) Threshold coefficient for cosmic ray
          detection, lower it to detect more cosmic rays (default : 3.).

        :param bad_frames_vector: (Optional) 1d array containing the
          indexes of the frames considered as bad.

        :param min_star_number: (Optional) Number of the most luminous
          stars that must be protectd from an over detection of cosmic
          rays. CR detected in thos stars will be removed.

        .. seealso:: :meth:`process.RawData.create_cosmic_ray_map`
        """
        cube = self._init_raw_data_cube(camera_number)
        perf = Performance(cube, "Cosmic ray map computation", camera_number)
        
        if "step_number" in self.options: 
            step_number = self.options["step_number"]
        else:
            self._print_warning("No step number given, check the option file.")

        star_list_path, mean_fwhm_pix = self.detect_stars(
                cube, camera_number, min_star_number, return_fwhm_pix=True)
            
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

    def compute_interferogram(self, camera_number, bad_frames_vector=[], 
                              optimize_dark_coeff=False, z_range=[],
                              combine='average', reject='avsigclip',
                              flat_smooth_deg=0):
        """Run the computation of the corrected interferogram from raw
           frames

        :param camera_number: Camera number (can be either 1 or 2).

        :param bad_frames_vector: (Optional) 1d array containing the
          indexes of the frames considered as bad.
          
        :param optimize_dark_coeff: (Optional) If True use a fast
          optimization routine to calculate the best coefficient for
          dark correction. This routine is used to correct for the
          images of the camera 2 on SpIOMM, because it contains a lot
          of hot pixels and varying dark and bias levels (because of a
          varying temperature). In order to get the best results the
          temperature of the bias frames and the interferogram frames
          must be recorded in the header [keyword 'CCD-TEMP'] (Default
          False)
          
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
        perf = Performance(cube, "Interferogram computation", camera_number)

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

        if "exp_time" in self.options:
            exposition_time = self.options["exp_time"]
        else:  exposition_time = None
        if "dark_time" in self.options:
            dark_int_time = self.options["dark_time"]
        else: dark_int_time = None
     
        cube.correct(
            bias_path=bias_path, dark_path=dark_path, 
            flat_path=flat_path, alignment_vector_path=None,
            cr_map_list_path=None, bad_frames_vector=bad_frames_vector, 
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

    def transform_cube_B(self, alignment_coeffs=None, 
                         star_list_path_1=None, min_star_number=15,
                         full_precision=False, interp_order=1,
                         stars_fwhm_1_arc=2.,
                         no_star=False):
        
        """Calculate the alignment parameters of the camera 2
        relatively to the first one. Transform the images of the
        camera 2 using linear interpolation by default.

        :param star_list_path_1: (Optional) Path to the list of star
          coordinates for the camera 1. You must set the stars FWHM
          using 'stars_fwhm_1=' parameter.
          
        :param alignment_coeffs: (Optional) Array containing
          precalculated alignment coefficients [dx, dy, dr, da,
          db]. If alignment coefficients are given no further
          calculation is made and the images are transformed using the
          given coefficients.

        :param min_star_number: (Optional) Minimum number of star to
          be detected by the automatic detection process (used if no
          path to a list of stars is given). Default 15.

        :param full_precision: (Optional) If True tip and tilt angles
          (da and db) are checked. Note that this can take a lot of
          time. If False da and db are set to 0 (default False).

        :param interp_order: (Optional) Interpolation order (Default 1.).

        :param stars_fwhm_1_arc: (Optional) FWHM of the stars of the
          camera 1 in arcsec. Used only when an external list of stars
          is given (using 'star_list_path_1=' parameter).

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
        image_list_path_1 = self.indexer['cam1.interfero_list']
        image_list_path_2 = self.indexer['cam2.interfero_list']

        # detect stars in cube 1
        if not no_star:
            if star_list_path_1 is None:
                cube1 = self._init_raw_data_cube(1)
                star_list_path_1, mean_fwhm_1_arc = self.detect_stars(
                    cube1, 0, min_star_number, saturation_threshold=60000)
                del cube1
            else:
                mean_fwhm_1_arc = stars_fwhm_1_arc
        else:
            star_list_path_1 = None
            mean_fwhm_1_arc = None
            if alignment_coeffs is None:
                alignment_coeffs = [init_dx, init_dy,
                                    self.config["INIT_ANGLE"], 0., 0.]

        # Init InterferogramMerger class
        self.indexer.set_file_group('merged')
        cube = InterferogramMerger(
            image_list_path_1, image_list_path_2,
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
            logfile_name=self._logfile_name)

        perf = Performance(cube.cube_B, "Cube B transformation", 2)

        # find alignment coefficients
        if not no_star and alignment_coeffs is None:
            cube.find_alignment(
                star_list_path_1,
                self.config["INIT_ANGLE"], init_dx, init_dy,
                mean_fwhm_1_arc, self.config["FIELD_OF_VIEW"],
                full_precision=full_precision,
                profile_name='gaussian', # Better for alignement tasks
                moffat_beta=self.config["MOFFAT_BETA"])
        else:
            cube.print_alignment_coeffs()
            self._print_msg("Alignment parameters: ")

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
        image_list_path_1 = self.indexer.get_path(
            'cam1.interfero_list', err=True)
        image_list_path_2 = self.indexer.get_path(
            'merged.transformed_interfero_frame_list', err=True)

        # Init class
        self.indexer.set_file_group('merged')
        cube = InterferogramMerger(
            image_list_path_A=image_list_path_1,
            image_list_path_B=image_list_path_2,
            data_prefix=self._get_data_prefix(0),
            project_header=self._get_project_fits_header(0),
            alignment_coeffs=None,
            overwrite=self.overwrite,
            tuning_parameters=self.tuning_parameters,
            indexer=self.indexer,
            logfile_name=self._logfile_name)
        
        perf = Performance(cube.cube_A, "Alternative merging process", 1)

        # Launch merging process
        cube.alternative_merge(add_frameB=add_frameB)
        
        perf_stats = perf.print_stats()
        del cube, perf
        return perf_stats
        
    def merge_interferograms(self, add_frameB=True, star_list_path_1=None, 
                             min_star_number=15, smooth_vector=True,
                             create_stars_cube=False, stars_fwhm_1_arc=2.,
                             bad_frames_vector=[], compute_ext_light=True):
        
        """Merge the images of the camera 1 with the transformed
        images of the camera 2.
        
        :param add_frameB: (Optional) If False use the images of the
          camera 2 only to correct for the variations of the sky
          transmission. Default True.

        :param star_list_path_1: (Optional) Path to the list of star
          coordinates for the camera 1. You must set the stars FWHM
          using 'stars_fwhm_1_arc=' parameter.

        :param min_star_number: (Optional) Minimum number of star to
          be detected by the automatic detection process (used if no
          path to a list of stars is given). Default 15.

        :param smooth_vector: (Optional) If True smooth the obtained
          correction vector with a gaussian weighted moving average.
          Reduce the possible high frequency noise of the transmission
          function. (Default True).

        :param create_stars_cube: (Optional) If True only the
          interferogram of the stars in the star list are computed
          using their photometric parameters returned by a 2D fit
          (default False).

        :param stars_fwhm_1_arc: (Optional) FWHM of the stars of the
          camera 1 in arcsec. Used only when an external list of stars
          is given (using 'star_list_path_1=' parameter).

        :param bad_frames_vector: (Optional) Vector containing indexes 
          of frames considered as bad (presence of plane, satellites, 
          important flux loss due to too much clouds)

        :param compute_ext_light: (Optional) If True compute the
          external light vector. Make sure that there's enough 'sky'
          pixels in the frames. The vector will be deeply affected if
          the object covers the whole area (default True).  

        .. note:: The transmission function used to correct for the
          variations of the sky transmission is calculated by summing
          the flux of stars in each frame (which is theoretically a
          constant). The flux of stars is obtained by 2D gaussian
          fitting using the formula :
          
          .. math::

                 Flux_{star} = FWHM_x \\times FWHM_y \\times amplitude
          
        .. seealso:: :meth:`process.InterferogramMerger.merge`
        """
        if star_list_path_1 is None:
            cube1 = self._init_raw_data_cube(1)
            star_list_path_1, mean_fwhm_arc = self.detect_stars(
                cube1, 1, min_star_number)
            del cube1
        else:
            mean_fwhm_arc = stars_fwhm_1_arc
        
        if "step_number" in self.options: 
            step_number = self.options["step_number"]
        else:
            self._print_error("No step number given, check the option file")
            
        # get frame list paths
        image_list_path_1 = self.indexer.get_path(
            'cam1.interfero_list', err=True)
        image_list_path_2 = self.indexer.get_path(
            'merged.transformed_interfero_frame_list', err=True)
        
        # get noise values
        readout_noise_1, dark_current_level_1 = self.get_noise_values(1)
        readout_noise_2, dark_current_level_2 = self.get_noise_values(2)

        self.indexer.set_file_group('merged')
        cube = InterferogramMerger(
            image_list_path_A=image_list_path_1,
            image_list_path_B=image_list_path_2,
            data_prefix=self._get_data_prefix(0),
            project_header=self._get_project_fits_header(0),
            alignment_coeffs=None,
            overwrite=self.overwrite,
            tuning_parameters=self.tuning_parameters,
            indexer=self.indexer,
            logfile_name=self._logfile_name)
        
        perf = Performance(cube.cube_A, "Merging process", 1)
            
        cube.merge(star_list_path_1, step_number,
                   mean_fwhm_arc, self.config["FIELD_OF_VIEW"],
                   add_frameB=add_frameB, 
                   smooth_vector=smooth_vector,
                   create_stars_cube=create_stars_cube,
                   profile_name=self.config["PSF_PROFILE"],
                   moffat_beta=self.config["MOFFAT_BETA"],
                   bad_frames_vector=bad_frames_vector,
                   compute_ext_light=compute_ext_light,
                   readout_noise_1=readout_noise_1,
                   dark_current_level_1=dark_current_level_1,
                   readout_noise_2=readout_noise_2,
                   dark_current_level_2=dark_current_level_2)
        
        perf_stats = perf.print_stats()
        del perf, cube
        return perf_stats

    def compute_calibration_laser_map(self, camera_number,
                                      get_calibration_laser_spectrum=False):
        """Run the computation of the calibration laser map from the
        calibration laser cube. This map is used to correct for the
        off-axis shift in wavelength.

        :param camera_number: Camera number (can be either 1 or 2).

        :param get_calibration_laser_spectrum: (Optional) If True
          return the computed calibration laser spectrum cube for
          checking purpose (Default False)
          
        .. seealso:: :meth:`process.CalibrationLaser.create_calibration_laser_map`
        """
        if camera_number == 1:
            if "calib_path_1" in self.options: 
                calib_path = self.options["calib_path_1"]
            else: 
                self._print_error("No path to the calibration laser files list given, check the option file")
        elif camera_number == 2:
            if "calib_path_2" in self.options: 
                calib_path = self.options["calib_path_2"]
            else: 
                self._print_error("No path to the calibration laser files list given, check the option file")
        else:
            self._print_error("Camera number must be either 1 or 2")

        order = self.config["CALIB_ORDER"]
        step = self.config["CALIB_STEP_SIZE"]
        self.options["camera_number"] = camera_number

        self.indexer.set_file_group(camera_number)
        cube = CalibrationLaser(
            calib_path, 
            data_prefix=self._get_data_prefix(camera_number),
            calibration_laser_header=self._get_calibration_laser_fits_header(),
            overwrite=self.overwrite,
            tuning_parameters=self.tuning_parameters,
            indexer=self.indexer,
            logfile_name=self._logfile_name)
        perf = Performance(cube, "Calibration laser map processing",
                           camera_number)

        cube.create_calibration_laser_map(
            order=order, step=step, 
            get_calibration_laser_spectrum=get_calibration_laser_spectrum)
        
        perf_stats = perf.print_stats()
        del cube, perf
        return perf_stats
        
    def add_missing_frames(self, camera_number, stars_cube=False):
        """Add non taken frames at the end of a cube in order to
        complete it and have a centered ZDP. Useful when a cube could
        not be completed during the night.
        
        :param camera_number: Camera number (can be 1, 2 or 0 
          for merged data).

        :param stars_cube: (Optional) if True the missing frames are
          added to the stars interferogram cube (default False).
          
        .. seealso:: :meth:`process.RawData.add_missing_frames`
        """
        if stars_cube:
            image_list_path = self.indexer.get_path(
                'merged.stars_interfero_frame_list', err=True)
        else:
            image_list_path = self._get_interfero_list_path(camera_number)
            
        self.options["camera_number"] = camera_number
        if camera_number == 0:
            cube = InterferogramMerger(
                image_list_path, 
                data_prefix=self._get_data_prefix(camera_number),
                project_header=self._get_project_fits_header(0),
                overwrite=self.overwrite,
                tuning_parameters=self.tuning_parameters,
                logfile_name=self._logfile_name)
        else:
            cube = self._init_raw_data_cube(camera_number)
            
        if "step_number" in self.options: 
            step_number = self.options["step_number"]
        else:
            self._print_error("No step number given, check the option file")

        cube.add_missing_frames(step_number)
        del cube

    def correct_interferogram(self, camera_number, star_list_path=None,
                              stars_fwhm=None,
                              image_list_path=None,
                              min_star_number=15, bad_frames_vector=[]):
        """Correct a single-camera interferogram cube for variations
        of sky transission and added light.

        :param camera_number: Camera number (can be 1 or 2).

        :param image_list_path: (Optional) Path to the
          list of the interferogram images (default None).

        :param min_star_number: (Optional) Minimum number of star to
          be detected by the automatic detection process (used if no
          path to a list of stars is given) (default 15).

        :param bad_frames_vector: (Optional) Vector containing indexes 
          of frames considered as bad (presence of plane, satellites, 
          important flux loss due to too much clouds)
          
        .. note:: The sky transmission vector gives the absorption
          caused by clouds or airmass variation.

        .. note:: The added light vector gives the counts added
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
        if image_list_path is None:
            image_list_path = self._get_interfero_list_path(camera_number)

        # init cube
        self.options["camera_number"] = camera_number
        self.indexer.set_file_group(camera_number)
        cube = Interferogram(
            image_list_path, 
            data_prefix=self._get_data_prefix(camera_number),
            project_header = self._get_project_fits_header(
                camera_number),
            calibration_laser_header=self._get_calibration_laser_fits_header(),
            overwrite=self.overwrite,
            tuning_parameters=self.tuning_parameters,
            indexer=self.indexer,
            logfile_name=self._logfile_name)
        perf = Performance(cube, "Interferogram correction", camera_number)
        
        # detect stars
        raw_cube = self._init_raw_data_cube(camera_number)
        if star_list_path is None:
            star_list_path, mean_fwhm_arc = self.detect_stars(
                raw_cube, camera_number, min_star_number)
            del raw_cube
        else:
            if stars_fwhm is None:
                mean_fwhm_arc=self.config["INIT_FWHM"]

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
            step_number=step_number,
            bad_frames_vector=bad_frames_vector)

        sky_transmission_vector_path = cube._get_transmission_vector_path()
        sky_added_light_vector_path = cube._get_added_light_vector_path()

        # correct interferograms
        cube.correct_interferogram(sky_transmission_vector_path,
                                   sky_added_light_vector_path)
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
                
        if not os.path.exists(calibration_laser_map_path):
            self._print_error("Calibration laser map not found ({} does not exist)".format(calibration_laser_map_path))
        return calibration_laser_map_path


    def compute_spectrum(self, camera_number, zpd_shift=None,
                         calibration_laser_map_path=None,
                         image_list_path=None, window_type=None,
                         polyfit_deg=1, n_phase=None,
                         stars_cube=False, phase_cube=False,
                         phase_map_0_path=None,
                         residual_map_path=None, phase_coeffs=None,
                         balanced=True, smoothing_deg=2,
                         bad_frames_list=None, no_star=False):

        """Run the computation of the spectrum from an interferogram
        cube.

        :param zpd_shift: (Optional) Shift of the ZPD in
          frames. Automaticaly computed if none given.

        :param calibration_laser_map_path: (Optional) Path to the
          calibration laser map (defined in the option file)

        :param image_list_path: (Optional) Path to the
          list of the interferogram images

        :param window_type: (Optional) Apodization function. Default
          None.

        :param n_phase: (Optional) Number of points around ZPD to use
          for phase correction. If 0, no phase correction will be done
          and the resulting spectrum will be the absolute value of the
          complex spectrum. If None, the number of points is set to 50
          percent of the interferogram length (default None).

        :param polyfit_deg: (Optional) Degree of the polynomial fit to
          the computed phase. If < 0, no fit will be performed
          (Default 1).

        :param stars_cube: (Optional) if True interferogram used will
          be the one computed only for stars (default False).

        :param phase_cube: (Optional) If True, only the phase cube is
          returned. The number of points of the phase can be defined
          with the option n_phase (default False).

        :param phase_map_0_path: (Optional) This map contains the 0th
          order coefficient of the phase. It must have the same
          dimensions as the frames of the interferogram cube. If none
          given default path to the phase map of order 0 is used
          (Default None).

        :param residual_map_path: (Optional) This map contains the
          residual of the fit for each phase vector. It must have the
          same dimensions as the frames of the interferogram cube. If
          none given default path to the residual map is checked
          (Default None).

        :param phase_coeffs: (Optional) Polynomial coefficients of
          order higher than 0. If given those coefficients are used to
          define the phase vector. If none given default path to the
          phase maps of order > 0 are used to create it (Default
          None).

        :param balanced: (Optional) If False, the interferogram is
          considered as unbalanced. It is flipped before its
          transformation to get a positive spectrum. Note that a
          merged interferogram is balanced (default True).

        :param smoothing_deg: (Optional) Degree of zeros smoothing. A
          higher degree means a smoother transition from zeros parts
          (bad frames) to non-zero parts (good frames) of the
          interferogram. Good parts on the other side of the ZPD in
          symetry with zeros parts are multiplied by 2. The same
          transition is used to multiply interferogram points by zero
          and 2 (default 2).

        :param bad_frames_list: (Optional) List containing indexes of
          frames considered as bad (presence of planes, satellites,
          important flux loss due to too much clouds)

        :param no_star: (Optional) If True, the cube is considered to
          have been computed without the star dependant processes so
          that the interferogram could not be corrected for sky
          transmission variations. The interferogram cube used will
          thus be the uncorrected one (default False).
      
        .. seealso:: :meth:`process.Interferogram.compute_spectrum`
        .. seealso:: :meth:`orb.utils.transform_interferogram`
        """
        # get calibration laser map path   
        if calibration_laser_map_path is None:
            calibration_laser_map_path = self._get_calibration_laser_map(
                camera_number)
      
        ## Load phase maps and create phase coefficients vector
        phase_map_correction = False

        if (((phase_map_0_path is None) or (phase_coeffs is None)
             or (residual_map_path is None))
            and not phase_cube and n_phase != 0):
            
            phase_map_correction = True
            
            # get phase map 0 path
            if ((phase_map_0_path is None)
                and ('phase_map_path' not in self.options)):
                phase_map_0_path = self.indexer.get_path(
                        'phase_map_fitted_0', camera_number)
                if  phase_map_0_path is None:
                    self._print_warning("No phase map found for the zeroth order. Phase correction will not use phase maps and will be less accurate !")
                    phase_map_correction = False
            else:
                if 'phase_map_path' in self.options:
                    phase_map_0_path = self.options['phase_map_path']
                    
                self._print_msg("Oth order phase map taken from external source: %s"%phase_map_0_path)
                
            # get other phase map paths
            if phase_coeffs is None:
                phase_map_paths = list()
                for iorder in range(1, self.config["PHASE_FIT_DEG"] + 1):
                    phase_map_path = self.indexer.get_path(
                        'phase_map_%d'%iorder, camera_number)
                    if os.path.exists(phase_map_path):
                        phase_map_paths.append(phase_map_path)
                    else:
                        self._print_warning("No phase map found for the order %d. Phase correction will not use phase maps and will be less accurate !"%iorder)
                        phase_map_correction = False
                    
            # get residual map path
            if residual_map_path is None:
                residual_map_path = self.indexer.get_path(
                    'phase_map_residual', camera_number)
                if not os.path.exists(residual_map_path):
                    self._print_warning("No residual map path found. Phase correction will not use phase maps and will be less accurate !")
                    phase_map_correction = False

            
        # Load interferogram frames
        if image_list_path is None:
            if stars_cube:
                image_list_path = self.indexer[
                    'merged.stars_interfero_frame_list']
            else:
                if not no_star:
                    image_list_path = self._get_interfero_list_path(
                        camera_number, corrected=True)
                else:
                    image_list_path = self._get_interfero_list_path(
                        camera_number, corrected=False)
                
        # Get ZPD from the normal interferogram cube if the stars
        # interferogram cube is transformed.
        if stars_cube and zpd_shift is None:       
            no_stars_cube = Interferogram(
                self._get_interfero_list_path(
                    camera_number, corrected=True), 
                tuning_parameters=self.tuning_parameters,
                logfile_name=self._logfile_name)
            zpd_shift = orb.utils.find_zpd(
                no_stars_cube.get_zmedian(nozero=True),
                return_zpd_shift=True)
            
        # Get final bad frames vector
        if bad_frames_list is None:
            if 'bad_frames' in self.options:
                bad_frames_list = self.options['bad_frames']
        bad_frames_vector = self.create_bad_frames_vector(
            camera_number,
            bad_frames_list=bad_frames_list)
        
        # init cube
        self.options["camera_number"] = camera_number
        self.indexer.set_file_group(camera_number)
        cube = Interferogram(
            image_list_path, 
            data_prefix=self._get_data_prefix(camera_number),
            project_header = self._get_project_fits_header(
                camera_number),
            calibration_laser_header=self._get_calibration_laser_fits_header(),
            overwrite=self.overwrite,
            tuning_parameters=self.tuning_parameters,
            indexer=self.indexer)
        
        perf = Performance(cube, "Spectrum computation", camera_number)
        
        if (camera_number == 1): 
            if "bin_cam_1" in self.options: 
                bin_factor = self.options["bin_cam_1"]
            else: 
                self._print_error("No binning factor for the camera 1 given, check the option file")

        elif (camera_number == 2): 
            if "bin_cam_2" in self.options: 
                bin_factor = self.options["bin_cam_2"]
            else: 
                self._print_error("No binning factor for the camera 2 given, check the option file")

        elif (camera_number == 0): 
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

        if 'apodization_function' in self.options and window_type is None:
            window_type = self.options['apodization_function']
            
        if window_type is not None:
            if window_type not in self._WINDOW_TYPE:
                self._print_error("Unrecognized apodization function. Please try : " + str(self._WINDOW_TYPE))

        if "fringes" in self.options:
            fringes=self.options['fringes']
        else: fringes = None

        # wavenumber option
        wavenumber = False
        if not phase_cube:
            if 'wavenumber' in self.options:
                wavenumber = self.options['wavenumber']

        # wavelength calibration option
        if not phase_cube:
            if 'wavelength_calibration' in self.options:
                if not self.options['wavelength_calibration']:
                    calibration_laser_map_path = None
                
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
            self.config["CALIB_NM_LASER"], zpd_shift=zpd_shift,
            bad_frames_vector=bad_frames_vector,
            n_phase=n_phase,
            polyfit_deg=polyfit_deg,
            window_type=window_type,
            stars_cube=stars_cube,
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


    def compute_phase_maps(self, camera_number,
                           interferogram_length=None,
                           phase_list_path=None,
                           calibration_laser_map_path=None, fit=True,
                           no_star=False):
        
        """Create a phase map.

        The phase map is a map of the zeroth order coefficient of the
        polynomial fit to the phase. The dimensions of the phase map
        are the same as the dimensions of the frames of the phase
        cube.
        
        :param camera_number: Camera number (must be 1, 2 or 0 for
          merged data).

        :param interferogram_length: Length of the interferogram from
          which the phase has benn computed.  Useful if the phase
          vectors have a lower number of points than the
          interferogram: this parameter is used to correct the fit
          coefficients. If None given the interferogram length is
          searched upon default interferogram path (default None).

        :param phase_list_path: (Optional) Path to the list of
          phase frames. If none given the default path is used
          (default None).

        :param calibration_laser_map_path: (Optional) Path to the
          calibration laser map. If none is given the default path is used
          (default None).
        
        :param camera_number: (Optional) Camera number (can be 1, 2 or
          0 for merged data) (default 0).

        :param fit: (Optional) If True the computed phase map is
          fitted to remove noise. Especially useful if the phase map
          is created from the astronomical data cube itself and not
          from a flat cube (default True).

        :param no_star: (Optional) If True, the cube is considered to
          have been computed without the star dependant processes so
          that the interferogram could not be corrected for sky
          transmission variations. The interferogram cube used will
          thus be the uncorrected one (default False).
          
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
        if calibration_laser_map_path is None:
            calibration_laser_map_path = self._get_calibration_laser_map(
                camera_number)

        # get default phase list path
        if phase_list_path is None:
            if camera_number == 0:
                phase_list_path = self.indexer['merged.phase_list']
            elif camera_number == 1:
                phase_list_path = self.indexer['cam1.phase_list']
            elif camera_number == 2:
                phase_list_path = self.indexer['cam2.phase_list']
            else:
                self._print_error('Camera number must be 1, 2 or 0')
                
        # get default interferogram length
        if interferogram_length is None:
            if not no_star:
                interfero_list_path = self._get_interfero_list_path(
                    camera_number, corrected=True)
            else:
                interfero_list_path = self._get_interfero_list_path(
                    camera_number, corrected=False)
            cube = Interferogram(interfero_list_path, silent_init=True,
                                 logfile_name=self._logfile_name)
            interferogram_length = cube.dimz
            del cube

        self.indexer.set_file_group(camera_number)
        phase = Phase(
            phase_list_path, 
            data_prefix=self._get_data_prefix(camera_number),
            project_header = self._get_project_fits_header(
                camera_number),
            calibration_laser_header=self._get_calibration_laser_fits_header(),
            overwrite=self.overwrite,
            tuning_parameters=self.tuning_parameters,
            indexer=self.indexer,
            logfile_name=self._logfile_name)
        
        perf = Performance(phase, "Phase map creation", camera_number)
        # create phase map
        phase.create_phase_maps(
            calibration_laser_map_path,
            filter_path,
            self.config["CALIB_NM_LASER"], step, order,
            interferogram_length=interferogram_length,
            fit_order=self.config["PHASE_FIT_DEG"])

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
        
    def calibrate_spectrum(self, camera_number, spectrum_list_path=None,
                           stars_cube=False, cam1_scale=False):
        
        """Calibrate spectrum cube and correct WCS.

        :param camera_number: Camera number (can be 1, 2 or
          0 for merged data).
          
        :param spectrum_list_path: (Optional) Path to the list of
          spectrum frames. If none is given the default path is used
          (default None).

        :param stars_cube: (Optional) If True the spectrum cube is
          assumed to be a star cube and the name of the resulting
          corrected spectrum cube will be changed (default False).

        :param cam1_scale: (Optional) If True scale map used is cam 1
          deep frame. Useful for SpIOMM which cam 2 frames cannot be
          well corrected for bias. This option is used only for a
          two-camera calibration process (default False).

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
        
        if spectrum_list_path is None:
            if stars_cube:
                spectrum_list_path = self.indexer.get_path(
                    'stars_spectrum_list', camera_number)
            else:
                spectrum_list_path = self.indexer.get_path(
                    'spectrum_list', camera_number)

        self.indexer.set_file_group(camera_number)
        spectrum = Spectrum(
            spectrum_list_path, 
            data_prefix=self._get_data_prefix(camera_number),
            project_header = self._get_project_fits_header(
                camera_number),
            calibration_laser_header=self._get_calibration_laser_fits_header(),
            overwrite=self.overwrite,
            tuning_parameters=self.tuning_parameters,
            indexer=self.indexer,
            logfile_name=self._logfile_name)
        perf = Performance(spectrum, "Spectrum calibration", camera_number)

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
        else:
            astrom = self._init_astrometry(spectrum, camera_number)
            correct_wcs = astrom.register(full_deep_frame=True)

        # Get Scale Map
        if camera_number == 0 and cam1_scale:
            self._print_warning('Flux rescaled relatively to camera 1')
            energy_map_path = self.indexer.get_path('energy_map', 1)
        else:
            energy_map_path = self.indexer.get_path('energy_map', camera_number)

        # Get deep frame
        if camera_number == 0 and cam1_scale:
            self._print_warning('Flux rescaled relatively to camera 1')
            deep_frame_path = self.indexer.get_path('deep_frame', 1)
        else:
            deep_frame_path = self.indexer.get_path('deep_frame', camera_number)

        # check wavelength calibration
        if not self.options['wavelength_calibration']:
            calibration_laser_map_path = self._get_calibration_laser_map(
                camera_number)
        else:
            calibration_laser_map_path = None
            
        # Calibration
        spectrum.calibrate(
            filter_path, step, order, stars_cube=stars_cube,
            correct_wcs=correct_wcs,
            flux_calibration_vector=flux_calibration_vector,
            energy_map_path=energy_map_path,
            deep_frame_path=deep_frame_path,
            wavenumber=self.options['wavenumber'],
            calibration_laser_map_path=calibration_laser_map_path,
            nm_laser=self.config['CALIB_NM_LASER'],
            standard_header = self._get_calibration_standard_fits_header())
        
        perf_stats = perf.print_stats()
        del perf, spectrum
        return perf_stats

    def extract_stars_spectrum(self, camera_number, window_type,
                               star_list_path=None, stars_fwhm_arc=2.,
                               min_star_number=15,
                               aperture_photometry=True, n_phase=None,
                               auto_phase=False, filter_correct=True,
                               aper_coeff=3., blur=False):
        
        """Extract the spectrum of the stars in a list of stars location
        list by photometry.
        
        :param camera_number: Camera number (can be 1, 2 or 0 for
          merged data).
          
        :param window_type: Apodization function to use during
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

        :param blur: (Optional) If True, blur frame (low pass
          filtering) before fitting stars. It can be used to enhance
          the quality of the fitted flux of undersampled data (default
          False). Useful only if aperture_photometry is True.
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
                raw_cube, camera_number, min_star_number)
            del raw_cube
        
        # get frame list paths
        image_list_path_1 = self.indexer['cam1.interfero_list']
        if camera_number == 0:
            image_list_path_2 = self.indexer[
                'merged.transformed_interfero_frame_list']

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
                cube = Interferogram('')
                phase_coeffs = cube.compute_phase_coeffs_vector(
                    phase_map_paths,
                    residual_map_path=residual_map_path)
                
        # get bad frames vector
        bad_frames_vector = self.create_bad_frames_vector(camera_number)

        # check parameters
        if 'apodization_function' in self.options and window_type is None:
            window_type = self.options['apodization_function']
            
        if window_type is not None and window_type not in self._WINDOW_TYPE:
                self._print_error("Unrecognized apodization function. Please try : " + str(self._WINDOW_TYPE))

        if camera_number == 0:
            cube = InterferogramMerger(
                image_list_path_A=image_list_path_1,
                image_list_path_B=image_list_path_2,
                data_prefix=self._get_data_prefix(0),
                project_header=self._get_project_fits_header(0),
                alignment_coeffs=None,
                overwrite=self.overwrite,
                tuning_parameters=self.tuning_parameters,
                logfile_name=self._logfile_name)

            perf = Performance(cube.cube_A, "Extract stars spectrum", 1)

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
                window_type=window_type,
                bad_frames_vector=bad_frames_vector,
                phase_map_0_path=phase_map_0_path,
                phase_coeffs=phase_coeffs,
                aperture=aperture_photometry,
                profile_name=self.config["PSF_PROFILE"],
                moffat_beta=self.config["MOFFAT_BETA"],
                n_phase=n_phase, 
                auto_phase=auto_phase, filter_correct=filter_correct,
                flat_spectrum_path=flat_spectrum_path,
                aper_coeff=aper_coeff, blur=blur)

            perf.print_stats()
        else:
            cube = Interferogram(
                image_list_path_1,          
                data_prefix=self._get_data_prefix(camera_number),
                project_header = self._get_project_fits_header(
                    camera_number),
                calibration_laser_header=
                self._get_calibration_laser_fits_header(),
                overwrite=self.overwrite,
                tuning_parameters=self.tuning_parameters,
                indexer=self.indexer,
                logfile_name=self._logfile_name)
            
            stars_spectrum = cube.extract_stars_spectrum(
                star_list_path,
                self.config["INIT_FWHM"], self.config["FIELD_OF_VIEW"],
                cube._get_transmission_vector_path(),
                cube._get_added_light_vector_path(),
                calibration_laser_map_path, self.options['step'],
                self.options['order'], self.config["CALIB_NM_LASER"],
                self._get_filter_file_path(self.options["filter_name"]),
                self.options['step_number'],
                window_type=window_type,
                bad_frames_vector=bad_frames_vector,
                aperture=aperture_photometry,
                profile_name=self.config["PSF_PROFILE"],
                moffat_beta=self.config["MOFFAT_BETA"],
                filter_correct=filter_correct,
                flat_spectrum_path=flat_spectrum_path,
                aper_coeff=aper_coeff, blur=blur)
            
        return stars_spectrum


    def get_flat_phase_map(self, camera_number):
        self._print_msg('Writing flat phase map to disk', color=True)
        phase_map_path = self.indexer.get_path('phase_map_0', camera_number)
        phase_map_data, phase_map_hdr = self.read_fits(phase_map_path,
                                                       return_header=True)
        self.write_fits(self._get_flat_phase_map_path(camera_number),
                        phase_map_data, fits_header=phase_map_hdr,
                        overwrite=self.overwrite)
        

    def get_calibrated_spectrum_cube(self, camera_number):
        """Extract a calibrated spectrum cube from the 'frame-divided'
        calibrated spectrum cube resulting of the reduction
        process. Write this cube at the root of the reduction folder.

        :param camera_number: Camera number (must be 1, 2 or 0 for
          merged data).
        """
        self._print_msg('Writing calibrated spectrum cube to disk', color=True)
        spectrum_list_path = self.indexer.get_path('calibrated_spectrum_list',
                                                   camera_number)
        spectrum = Cube(spectrum_list_path)
        spectrum_header = spectrum.get_frame_header(0)

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
            wavelength_calibration=self.options['wavelength_calibration'])
        spectrum.export(spectrum_path, fits_header=spectrum_header,
                        overwrite=self.overwrite)

    def get_standard_spectrum(self, camera_number, n_phase=None,
                              aperture_photometry=True, window_type='2.0',
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

        :param window_type: (Optional) Apodization function to use for
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
            camera_number, window_type, star_list_path=std_list,
            min_star_number=self.config['DETECT_STAR_NB'],
            aperture_photometry=aperture_photometry,
            n_phase=n_phase, auto_phase=auto_phase, filter_correct=False)[0]

        nm_axis = orb.utils.create_nm_axis(
            std_spectrum.shape[0], self.options['step'], self.options['order'])
        
        std_header = (self._get_project_fits_header()
                      + self._get_basic_header('Standard Spectrum')
                      + self._get_fft_params_header(window_type)
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
    
    def __init__(self, cube, process_name, camera_number,
                 logfile_name=None):
        """
        Initialize class

        :param cube: Reference to a cube in order to get details on the
          reduced data

        :param process_name: Name of the running process checked

        :param camera_number: Number of the camera which cube is
          processed (can be 1, 2 or 0 for merged data)

        :param logfile_name: (Optional) Give a specific name to the
          logfile (default None).
        """
        self._init_logfile_name(logfile_name)
        self._msg_class_hdr = self._get_msg_class_hdr()
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
          run this function alone.
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
        
