#!/usr/bin/python
# *-* coding: utf-8 *-*
# Author: Thomas Martin <thomas.martin.1@ulaval.ca>
# File: orbs.py

## Copyright (c) 2010-2018 Thomas Martin <thomas.martin.1@ulaval.ca>
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
import logging
import warnings
import shutil
from datetime import datetime
import astropy
import pp
import bottleneck as bn


from orb.core import Tools, Indexer, TextColor
from orb.cube import FDCube, HDFCube, Cube, RWHDFCube
from orb.core import FilterFile, ProgressBar
from process import RawData, InterferogramMerger, Interferogram
from process import Spectrum, CalibrationLaser
from process import CosmicRayDetector
from core import JobFile, RoadMap

import orb.constants
import orb.version
import orb.utils.spectrum
import orb.utils.fft
import orb.utils.stats
import orb.utils.image
import orb.utils.vector
import orb.utils.io

##################################################
#### CLASS Orbs ##################################
##################################################

class Orbs(Tools):
    """ORBS user-interface implementing the high level reduction methods

    Help managing files during the reduction process and offer
    simple high level methods to reduce SpIOMM/SITELLE data. You must init
    Orbs class with a path to a job file (e.g. file.job)
    containing all the parameters needed to run a reduction.
    """

    __version__ = None # imported from __version__ given in the core module
    
    _APODIZATION_FUNCTIONS = ['learner95']
    """Apodization functions that recognized by ORBS. Any float > 1 is
    recognized by :py:meth:`orb.utils.gaussian_window` (see
    :py:meth:`orb.utils.fft.gaussian_window`)"""

    config = dict()
    """Dictionary containing all the options of the config file.  All
    those options can be changed for any particular reduction using
    the option file.
    """
    
    options = dict()
    """Dictionary containing all the reduction parameters needed by
    processing classes.
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
    
    targets = ['object', 'flat', 'standard', 'laser', 'extphase']
    """Possible target types"""

     
    def __init__(self, job_file_path, target,
                 fast_init=False, silent=False, **kwargs):
        """Initialize Orbs class.

        :param job_file_path: Path to the job file.

        :param target: Target type to reduce. Used to define the
          reduction road map. Target may be 'object', 'flat',
          'standard', 'laser' or 'extphase'.

        :param silent: (Optional) If True, nothing is printed durint
          init (default False)
    
        :param fast_init: (Optional) Fast init. Data files are not
          checked. Gives access to Orbs variables (e.g. object dependant file
          paths). This mode is faster but less safe.

        :param kwargs: orb.core.Tools kwargs.

        """
        def export_images(option_key, camera, mask_key=None):
            # export fits frames as an hdf5 cube

            # check if the hdf5 cube already
            # exists.
            if option_key not in self.options: return

            logging.info('exporting {}'.format(option_key))
            export_path = (
                self._get_project_dir()
                + os.path.splitext(
                    os.path.split(
                        self.options[option_key])[1])[0]
                + '.hdf5')

            if not os.path.exists(export_path):
                cube = FDCube(
                    self.options[option_key],
                    silent_init=True, no_sort=False,
                    instrument=self.instrument)
                
                # create mask
                if mask_key is not None:
                    if mask_key in self.options:
                        warnings.warn('Mask applied: {}'.format(self.options[mask_key]))
                        image_mask = orb.utils.io.read_fits(
                            self.options[mask_key])
                    else: image_mask = None
                else: image_mask = None

                export_params = dict(self.options)
                export_params['camera'] = int(camera)
                cube.export(export_path, mask=image_mask, params=export_params)
                self.newly_exported = True

            self.options[option_key + '.hdf5'] = export_path


        Tools.__init__(self, **kwargs)
        
        self.newly_exported = False

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
            with orb.utils.io.open_file(self._get_config_file_path(), 'r') as conf_file:
                logging.info("Configuration file content:")
                for line in conf_file:
                    logging.info(line[:-1])
 
        # Read job file to get observation parameters
        self.jobfile = JobFile(job_file_path, self.instrument)

        # Print first the entire option file for logging
        if not silent:
            logging.info("Job file content: \n{}".format(self.jobfile.as_str()))

        # record some default options
        self.options = dict()
        self.options["try_catalogue"] = False
        self.options['spectral_calibration'] = True
        self.options['wavenumber'] = False
        self.options['no_sky'] = False
        self.options['apodization_function'] = None
        
        self.options.update(self.jobfile.get_params())

        ##########
        ## get the other parameters from the jobfile
        ##########
                    
        # check step and order
        if target != 'laser':
            _step, _order = FilterFile(
                self.options['filter_name']).get_observation_params()
            if int(self.options['order']) != int(_order):
                warnings.warn('Bad order. Option file tells {} while filter file tells {}. Order parameter replaced by {}'.format(int(self.options['order']), _order, _order))
                self.options['order'] = _order
            if abs(self.options['step'] - float(_step))/float(_step) > 0.1:
                raise StandardError('There is more than 10% difference between the step size in the option file ({}) and the step size recorded in the filter file ({})'.format(self.options['step'], _step))
                
        
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
        newconf = self.jobfile.get_config()
        for key in self.config.iterkeys():
           if key in newconf:
               self.config.reset(key, newconf[key])
               if not silent:
                   warnings.warn("Configuration option %s changed to %s"%(key, self.config[key]))

        # Calibration laser wavelength is changed if the calibration
        # laser map gives a new calibration laser wavelentgh
        if target != 'laser':
            calib_hdu = orb.utils.io.read_fits(
                self.options['calibration_laser_map_path'],
                return_hdu_only=True)
            if 'CALIBNM' in calib_hdu.header:
                self.config['CALIB_NM_LASER'] = calib_hdu.header['CALIBNM']
                if not silent:
                    warnings.warn('Calibration laser wavelength (CALIB_NM_LASER) read from calibration laser map header: {}'.format(self.config['CALIB_NM_LASER']))
               
        # Get tuning parameters
        # self.tuning_parameters = self.optionfile.get_tuning_parameters()
        # for itune in self.tuning_parameters:
        #     if not silent:
        #         warnings.warn("Tuning parameter %s changed to %s"%(
        #             itune, self.tuning_parameters[itune]))
        self.tuning_parameters = dict(self.jobfile.raw_params)
        for itune in self.tuning_parameters:
            if not silent:
                warnings.warn("Tuning parameter %s changed to %s"%(
                    itune, self.tuning_parameters[itune]))
        self.config.update(self.tuning_parameters)
        
        
        self.options["project_name"] = (self.options["object_name"] 
                                        + "_" + self.options["filter_name"])

        
        # get folders paths
        if not silent:
            logging.info('Reading data folders')

        if not fast_init:
            export_images('image_list_path_1', 1, mask_key='object_mask_path_1')
            export_images('image_list_path_2', 2, mask_key='object_mask_path_2')
            export_images('bias_path_1', 1)
            export_images('bias_path_2', 2)
            export_images('dark_path_1', 1)
            export_images('dark_path_2', 2)
        
            if (('dark_path_2' in self.options or 'dark_path_1' in self.options)
                and 'dark_time' not in self.options):
                raise StandardError('Dark integration time must be set (SPEDART) if the path to a dark calibration files folder is given')
            
            export_images('flat_path_1', 1)
            export_images('flat_path_2', 2)
            export_images('calib_path_1', 1)
            export_images('calib_path_2', 2)
            export_images('standard_image_path_1', 1, mask_key='std_mask_path_1')
            export_images('standard_image_path_2', 2, mask_key='std_mask_path_2')
        
        
        if target == 'laser':
            self.options['image_list_path_1'] = self.options['calib_path_1']
            self.options['image_list_path_2'] = self.options['calib_path_2']
            if 'calib_path_1.hdf5' in self.options:
                self.options['image_list_path_1.hdf5'] = self.options[
                    'calib_path_1.hdf5']
            if 'calib_path_2.hdf5' in self.options:
                self.options['image_list_path_2.hdf5'] = self.options[
                    'calib_path_2.hdf5']

        if not fast_init:
            if 'image_list_path_1' in self.options:
                cube1 = Cube(self.options['image_list_path_1.hdf5'],
                             instrument=self.instrument,
                             params=self.options,
                             config=self.config,
                             camera=1,
                             zpd_index=0)
                self.options['bin_cam_1'] = int(cube1.params.binning)
                dimz1 = cube1.dimz
                del cube1


            if 'image_list_path_2' in self.options:
                cube2 = Cube(self.options['image_list_path_2.hdf5'],
                             instrument=self.instrument,
                             params=self.options,
                             config=self.config,
                             camera=2,
                             zpd_index=0)
                self.options['bin_cam_2'] = int(cube2.params.binning)
                dimz2 = cube2.dimz
                del cube2
                                
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

        logging.info('looking for ZPD shift')
        # get ZPD shift in SITELLE's case
        if self.config["INSTRUMENT_NAME"] == 'SITELLE' and not fast_init:
            
            try: # check if the zpd index has already been computed
                zpd_index = int(orb.utils.io.read_fits(self._get_zpd_index_file_path()))
                logging.info('ZPD index read from file')
            except IOError:
                cube1 = HDFCube(self.options['image_list_path_1.hdf5'],
                                instrument=self.instrument,
                                no_sort=True)

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
                del cube1
                
            self.options['zpd_index'] = zpd_index
            if not silent:
                logging.info('ZPD index: {}'.format(
                    self.options['zpd_index']))                
            
            orb.utils.io.write_fits(self._get_zpd_index_file_path(),
                                    np.array(self.options['zpd_index']),
                                    overwrite=True)

        if not fast_init:
            if not silent:
                logging.info('loading airmass')
            try: # check if the airmass as already been computed
                airmass = orb.utils.io.read_fits(self._get_airmass_file_path())
                logging.info('airmass read from file')
            except IOError:
                cube1 = Cube(self.options['image_list_path_1.hdf5'],
                             instrument=self.instrument,
                             zpd_index=self.options['zpd_index'])
                airmass = cube1.get_airmass()
                orb.utils.io.write_fits(self._get_airmass_file_path(),
                                        np.array(airmass), overwrite=True)
                self.options['airmass'] = airmass
                self.newly_exported = True
                del cube1


        
        # update parameters of the newly exported hdf cubes
        if not fast_init and self.newly_exported:
            logging.info('parameters update')
            for ikey in self.options:
                if '.hdf5' in ikey:
                    icube = RWHDFCube(self.options[ikey])
                    icube.update_params(self.options)
                    del icube
                            
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
            self.config["INSTRUMENT_NAME"].lower(), target, 'full', self.indexer)

        # attach methods to roadmap steps
        self.roadmap.attach('compute_alignment_vector',
                            self.compute_alignment_vector)
        self.roadmap.attach('compute_cosmic_ray_maps',
                            self.compute_cosmic_ray_maps)
        self.roadmap.attach('compute_interferogram',
                            self.compute_interferogram)
        self.roadmap.attach('transform_cube_B',
                            self.transform_cube_B)
        self.roadmap.attach('merge_interferograms',
                            self.merge_interferograms)
        self.roadmap.attach('compute_calibration_laser_map',
                            self.compute_calibration_laser_map)
        self.roadmap.attach('compute_phase_maps',
                            self.compute_phase_maps)
        self.roadmap.attach('compute_spectrum',
                            self.compute_spectrum)
        self.roadmap.attach('calibrate_spectrum',
                            self.calibrate_spectrum)
        
            
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

    def _get_file_folder_path_hdr(self, camera_number):
        """Return path to the file folder to where reduction files are stored.
        """
        return ('.' + os.sep + self.options['object_name'] + '_'
                + self.options['filter_name'] + os.sep
                + os.path.split(self._get_root_data_path_hdr(camera_number))[1])

    def _get_zpd_index_file_path(self):
        """Return path to the zpd index file.        
        """
        return self._get_file_folder_path_hdr(1) + 'zpd_index.fits'

    def _get_airmass_file_path(self):
        """Return path to the airmass file.        
        """
        return self._get_file_folder_path_hdr(1) + 'airmass.fits'

    def _get_wcs_deep_frame_path(self):
        """Return path to the registered deep frame
        """
        return self._get_file_folder_path_hdr(1) + 'wcs_deep_frame.hdf5'

    def _get_wcs_standard_image_path(self):
        """Return path to the registered standard image
        """
        return self._get_file_folder_path_hdr(1) + 'wcs_standard_image.hdf5'


    def _get_calibration_laser_map_path(self, camera_number):
        """Return path to the calibration laser map from a flat cube.
        
        :param camera_number: Camera number (must be 1 or 2).
        """
        return (self._get_root_data_path_hdr(camera_number)
                + 'calibration_laser_map.fits')
    
    def _get_calibrated_spectrum_cube_path(self, camera_number):
        """Return path to the calibrated spectrum cube resulting of the
        reduction process.
        
        :param camera_number: Camera number (must be 1, 2 or 0 for
          merged data).
        """
        return (self._get_root_data_path_hdr(camera_number)
                + 'cm1.1.0.hdf5')

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
                    indexer=self.indexer,
                    instrument=self.instrument)
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
                    indexer=self.indexer,
                    instrument=self.instrument)
            else:
                raise StandardError("No image list file for camera 2 given, please check option file")
        else:
            raise StandardError("Camera number must be either 1 or 2, please check 'camera_number' parameter")
            return None
        
        cube.params.reset('camera', camera_number)
        return cube


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
            with orb.utils.io.open_file(self.options['source_list_path'], 'r') as f:
                for line in f:
                    x,y = line.strip().split()[:2]
                    source_list.append([float(x),float(y)])
            
            logging.info('Loaded {} sources to extract'.format(len(source_list)))
  
            
        return source_list
        
    def _get_standard_name(self, std_path):
        """Return value associated to keyword 'OBJECT'

        :param std_path: Path to the file containing the standard.    
        """
        raise NotImplementedError('remove me please. at least if the file is a fits file')
        if 'hdf5' in std_path:
            cube = HDFCube(std_path, instrument=self.instrument, ncpus=self.config.NCPUS)
            hdr = cube.get_frame_header(0)
        else:
            hdr = orb.utils.io.read_fits(std_path, return_hdu_only=True).header
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
                              instrument=self.instrument, ncpus=self.config.NCPUS)
        bad_frames_vector = np.zeros(interf_cube.dimz)
        
        if camera_number == 0:
            # Get bad frames vector created by the merge function
            merge_bad_frames_path = self.indexer['merged.bad_frames_vector']
            if merge_bad_frames_path is not None:
                if os.path.exists(merge_bad_frames_path):
                    merge_bad_frames = orb.utils.io.read_fits(merge_bad_frames_path)
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
                        wcs_calibration=True, flux_calibration=True):
        
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

        :param wcs_calibration: (Optional) If True, WCS calibration is
          done at calibration step (default True).

        :param flux_calibration: (Optional) If True, flux calibration is
          done at calibration step (default True).
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
        
        if (self.target == 'object' or self.target == 'extphase'):
            self.export_calibrated_spectrum_cube(cam)
        if self.target == 'flat': self.export_flat_phase_map(cam)
        if self.target == 'laser': self.export_calibration_laser_map(cam)
        if self.target == 'standard': self.export_standard_spectrum(
            cam, auto_phase=True)

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
        
        star_list_path, _ = cube.detect_stars(min_star_number=self.config.DETECT_STAR_NB)

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
        cube = self.compute_alignment_parameters(
            no_star=False, raw=True)

        perf = Performance(
            cube.cube_A, "Cosmic ray map computation", 0,
            instrument=self.instrument)

        alignment_vector_path_1 = self.indexer['cam1.alignment_vector']

        cube.clean_cosmic_ray_maps()
        perf_stats = perf.print_stats()
        del cube, perf
        return perf_stats

    def _get_init_fwhm_pix(self):
        """Return init FWHM of the stars in pixels"""
        return (float(self.config['INIT_FWHM'])
                * float(self.config['CAM1_DETECTOR_SIZE_X'])
                / float(self.config['FIELD_OF_VIEW_1'])
                / 60.)

    def compute_interferogram(self, camera_number,
                              no_corr=False):
        """Run the computation of the corrected interferogram from raw
           frames

        :param camera_number: Camera number (can be either 1 or 2).
          
        :param reject: (Optional) Rejection operation for master
          frames creation. Can be 'sigclip', 'avsigclip', 'minmax' or
          None (default 'avsigclip'). See
          :py:meth:`process.RawData._create_master_frame`.
        
        :param combine: (Optional) Combining operation for master
          frames creation. Can be 'average' or 'median' (default
          'average'). See
          :py:meth:`process.RawData._create_master_frame`.
         
        :param no_corr: (Optional) If True, no correction is made and
          the interferogram cube is just a copy of the raw cube
          (default False).

        .. seealso:: :py:meth:`process.RawData.correct`

        """
        cube = self._init_raw_data_cube(camera_number)

        if no_corr:
            try: os.makedirs(os.path.split(cube._get_interfero_cube_path())[0])
            except OSError: pass
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
                dark_path = self.options["dark_path_1.hdf5"]
            else:
                warnings.warn("No path to dark frames given, please check the option file.")
            if "bias_path_1" in self.options:
                bias_path = self.options["bias_path_1.hdf5"]
            else:
                warnings.warn("No path to bias frames given, please check the option file.")
            if "flat_path_1" in self.options:
                flat_path = self.options["flat_path_1.hdf5"]
            else:
                warnings.warn("No path to flat frames given, please check the option file.")
    
        if camera_number == 2: 
            if "dark_path_2" in self.options:
                dark_path = self.options["dark_path_2.hdf5"]
            else:
                warnings.warn("No path to dark frames given, please check the option file.")
            if "bias_path_2" in self.options:
                bias_path = self.options["bias_path_2.hdf5"]
            else:
                warnings.warn("No path to bias frames given, please check the option file.")
            if "flat_path_2" in self.options:
                flat_path = self.options["flat_path_2.hdf5"]
            else:
                warnings.warn("No path to flat frames given, please check the option file.")
        
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
            raise NotImplementedError()
            
        cube.correct(
            bias_path=bias_path, dark_path=dark_path, 
            flat_path=flat_path, cr_map_cube_path=cr_map_cube_path)
        
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

    def compute_alignment_parameters(self, no_star=False, raw=False, laser=False):
        """Compute alignement parameters between cubes and return a
        :py:class:`process.InterferogramMerger instance`.

        :param no_star: (Optional) If the cube does not contain any star, the
          transformation is made using the default alignment
          parameters (recorded in the configuration file :
          'data/config.orb') (default False).
    
        :param laser: (Optional) If the cube is a laser source, the
          frames can be aligned with a brute force algorithm (default
          False).

        .. seealso:: :py:meth:`process.InterferogramMerger.find_alignment`
        """
        if laser: no_star = True
       
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
            

        # Init InterferogramMerger class
        self.indexer.set_file_group('merged')
        if raw:
            ComputingClass = CosmicRayDetector
        else:
            ComputingClass = InterferogramMerger

        params = dict(self.options)
        cube = ComputingClass(
            interf_cube_path_1, interf_cube_path_2,
            bin_A=bin_cam_1, bin_B=bin_cam_2,
            data_prefix=self._get_data_prefix(0),
            indexer=self.indexer,
            instrument=self.instrument,
            params=params,
            config=self.config)

        # find alignment coefficients
        
        if not no_star:
            cube.compute_alignment_parameters(combine_first_frames=raw)
        else:
            if laser:
                raise NotImplementedError('init_dx, init_dy and init_angle must be defined in find_laser_alignment itself')
                cube.find_laser_alignment(
                    init_dx, init_dy, self.config["INIT_ANGLE"])

            logging.info("Alignment parameters: {} {} {} {} {}".format(
                cube.dx, cube.dy, cube.dr, cube.da, cube.db))

        return cube
                
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
        # get cubes path
        interf_cube_path_1 = self.indexer.get_path(
            'cam1.interfero_cube', err=True)
        interf_cube_path_2 = self.indexer.get_path(
            'merged.transformed_interfero_cube', err=True)
        
        self.indexer.set_file_group('merged')
        cube = InterferogramMerger(
            interf_cube_path_A=interf_cube_path_1,
            interf_cube_path_B=interf_cube_path_2,
            data_prefix=self._get_data_prefix(0),
            indexer=self.indexer,
            instrument=self.instrument,
            params=self.options,
            config=self.config)
        
        perf = Performance(cube.cube_A, "Merging process", 1,
                           instrument=self.instrument)

        # cube.compute_correction_vectors(
        #     smooth_vector=smooth_vector,
        #     compute_ext_light=(not self.options['no_sky']
        #                        and self.config['EXT_ILLUMINATION']))
        cube.merge(add_frameB=add_frameB)
        
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
            indexer=self.indexer,
            instrument=self.instrument)
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
        

    def compute_spectrum(self, camera_number,
                         apodization_function=None,
                         phase_correction=True,
                         wave_calibration=False,
                         no_star=False):

        """Compute a spectral cube from an interferogram cube.
     
        :param apodization_function: (Optional) Apodization
          function. Default None.
    
        :param phase_correction: (Optional) If False, no phase
          correction will be done and the resulting spectrum will be
          the absolute value of the complex spectrum (default True).

        :param wave_calibration: (Optional) If True
          wavenumber/wavelength calibration is done (default False).

        :param no_star: (Optional) If True, the cube is considered to
          have been computed without the star dependant processes so
          that the interferogram could not be corrected for sky
          transmission variations. The interferogram cube used will
          thus be the uncorrected one (default False).

        .. seealso:: :meth:`process.Interferogram.compute_spectrum`
        .. seealso:: :meth:`orb.utils.transform_interferogram`
        """        
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
            indexer=self.indexer,
            instrument=self.instrument)
        
        perf = Performance(cube, "Spectrum computation", camera_number,
                           instrument=self.instrument)
        
        balanced = self._is_balanced(camera_number)

        if apodization_function is None:
            if 'apodization_function' in self.options:
                apodization_function = self.options['apodization_function']
        
        if apodization_function is None: # keep it here, apodization function may still be None
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
            instrument=self.instrument,
            indexer=self.indexer,
            data_prefix=self._get_data_prefix(camera_number))
        
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

    def _get_standard_image(self, spectrum):
        """register standard image
        """
        if 'standard_image_path_1.hdf5' in self.options:
            spectrum.params.reset('standard_image_path', self.options['standard_image_path_1.hdf5'])
        
        logging.info('registering standard image')
        try:
            std_im = spectrum.get_standard_image()
            std_im.writeto(self._get_wcs_standard_image_path())
            # image must be reopened for the correct wcs parameters to be loaded
            std_im = orb.image.Image(self._get_wcs_standard_image_path())
            std_im.register()
            std_im.writeto(self._get_wcs_standard_image_path())
            return self._get_wcs_standard_image_path()
        
        except StandardError, e:
            warnings.warn('no standard image can be created: {}'.format(e))
            return None
        
    def _compute_wcs(self, camera_number):
        """Register deep frame and compute wcs
        """
        if (self.options['target_ra'] is None or self.options['target_dec'] is None
            or self.options['target_x'] is None or self.options['target_y'] is None):
            warnings.warn("wcs options are missing. wcs correction cannot be done.")
            return
        
        logging.info('registering deep frame')
        deep_frame_path = self.indexer.get_path(
            'deep_frame', camera_number)
        deep_frame = orb.image.Image(deep_frame_path, instrument=self.instrument,
                                     params=self.options)
        try:
            deep_frame.register()
        except Exception, e:
            exc_info = sys.exc_info()
            warnings.warn('Error during WCS computation, check WCS parameters in the option file: {}'.format(e))
            traceback.print_exception(*exc_info)
            del exc_info
            correct_wcs = None
        deep_frame.writeto(self._get_wcs_deep_frame_path())
            

    def calibrate_spectrum(self, camera_number, cam1_scale=False,
                           no_star=False, filter_correction=True,
                           wcs_calibration=True, flux_calibration=True):
        
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
            indexer=self.indexer,
            instrument=self.instrument)

        perf = Performance(spectrum, "Spectrum calibration", camera_number,
                           instrument=self.instrument)

        # compute flambda
        std_im_path = None
        if flux_calibration:
            std_im_path = self._get_standard_image(spectrum)
        else:
            warnings.warn("no flux calibration.")
        
        # Get WCS
        deep_frame = None
        if not no_star and wcs_calibration:
            self._compute_wcs(camera_number)
        else:
            warnings.warn("no wcs calibration.")

        # Calibration
        spectrum.calibrate(
            self._get_wcs_deep_frame_path(),
            self.indexer.get_path(
                'phase_maps', file_group=camera_number),
            std_im_path)
        
        perf_stats = perf.print_stats()
        del perf, spectrum
        return perf_stats



    def export_calibration_laser_map(self, camera_number):
        """Export the computed calibration laser map at the root of the
        reduction folder.

        :param camera_number: Camera number (must be 1, 2 or 0 for
          merged data).
        """
        logging.info('Writing calibration laser map to disk')
        calibration_laser_map_path = self.indexer.get_path(
            'calibration_laser_map', camera_number)
       
        map_data, map_hdr = orb.utils.io.read_fits(
            calibration_laser_map_path,
            return_header=True)
        
        orb.utils.io.write_fits(self._get_calibration_laser_map_path(camera_number),
                                map_data, fits_header=map_hdr,
                                overwrite=True)


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
        logging.info('High order phase file exported to {}'.format(exported_path))

    def export_calibrated_spectrum_cube(self, camera_number):
        """Extract a calibrated spectrum cube from the 'frame-divided'
        calibrated spectrum cube resulting of the reduction
        process. Write this cube at the root of the reduction folder.

        :param camera_number: Camera number (must be 1, 2 or 0 for
          merged data).
        """
        link_path = self._get_calibrated_spectrum_cube_path(camera_number)
        real_path = self.indexer.get_path(
            'calibrated_spectrum_cube', file_group=camera_number)

        logging.info('Exporting calibrated spectrum')
        
        if os.path.exists(link_path):
            os.remove(link_path)
            
        shutil.copyfile(real_path, link_path)
        logging.info('Calibrated spectrum cube exprted to {}'.format(link_path))
        
        


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
        std_spectrum, hdr = orb.utils.io.read_fits(
            self.indexer.get_path(
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
        
        orb.utils.io.write_fits(std_spectrum_path, std_spectrum,
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
        
