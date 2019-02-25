#!/usr/bin/python
# *-* coding: utf-8 *-*
# Author: Thomas Martin <thomas.martin.1@ulaval.ca>
# File: process.py

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
The Process module contains all the processing classes of ORBS.
"""

__author__ = "Thomas Martin"
__licence__ = "Thomas Martin (thomas.martin.1@ulaval.ca)"                      
__docformat__ = 'reStructuredText'
import version
__version__ = version.__version__

import orb.core
import orb.cube
import orb.utils.fft
import orb.fft
import orb.utils.filters
import orb.utils.spectrum
import orb.utils.image
import orb.utils.misc
import orb.utils.photometry
import orb.utils.io
import orb.cutils
import orb.fit

import orb.astrometry
import orb.utils.astrometry
import orb.constants

from phase import BinnedInterferogramCube, BinnedPhaseCube


import bottleneck as bn

import os
import numpy as np
import math
from scipy import optimize, interpolate

import astropy.io.fits as pyfits
import warnings
import logging
import time


#################################################
#### CLASS CubeMask #############################
#################################################
class CubeMask(object):

    def __init__(self, shape):
        """Init CubeMask class

        :param shape: Cube shape.
        """
        if not isinstance(shape, tuple) or isinstance(shape, list):
            raise TypeError('shape must be a tuple or a list')
        
        if len(shape) != 3:
            raise TypeError('shape must be of length 3')

        for ishape in shape:
            if not isinstance(ishape, int):
                raise TypeError('shape must be a tuple of 3 int')
            
        self.shape = tuple(shape)
        self.dimx, self.dimy, self.dimz = self.shape

        # this step initializes the whole set of bad pixels list
        self.reset()

        logging.debug('Cube Mask initialized with shape {}'.format(self.shape))

    def _get_empty_frame(self):
        """Return an empty mask frame"""
        try:
            self._empty_frame.fill(False)
        except AttributeError:
            self._empty_frame = np.zeros((self.dimx, self.dimy), dtype=bool)
        return self._empty_frame

    def _get_cr_byframe_mask(self, index):
        """Return mask of a frame as a list of pixels

        :param index: frame index
        """
        if index not in range(self.dimz): raise ValueError('invalid frame index')

        if self.cr_byframe[index] is not None:
            return self.cr_byframe[index]
        else:
            return (np.array([], dtype=int), np.array([], dtype=int),)

    def _get_cr_key(self, xy):
        """Return the key corresponding to a given pixel position in the
        cr_byspectrum dict.

        :param xy: (x,y) coordinates
        """
        if np.size(xy) != 2: raise TypeError('pos must be a couple of integer values')
        return '{},{}'.format(*xy)

    def _get_cr_byspectrum_mask(self, xy):
        """Return a mask of the cosmic rays at a given position in the
        cr_byspectrum dict

        :param xy: (x,y) coordinates
        """
        key = self._get_cr_key(xy)
        if key in self.cr_byspectrum:
            return self.cr_byspectrum[key]
        else:
            return None

    def _set_cr_byspectrum_mask(self, xy, mask):
        """Set the cr_byspectrum dict at a given position.

        .. warning:: mask is not updated but overwritten.

        :param xy: (x,y) coordinates

        :param mask: mask in a numpy.nonzero format.
        """
        
        self.cr_byspectrum[self._get_cr_key(xy)] = mask

    def reset(self):
        """Reset all bad pixels lists"""
        self.cr_byframe = [None for i in range(self.dimz)]
        self.cr_byspectrum = dict()
        self.bad_frames = list()
        self.bad_region = np.copy(self._get_empty_frame())
        
    def append(self, index, bad_pix_list):
        """Append new bad pixels to a frame

        :param index: frame index

        :param bad_pix_list: list of bad pixels as returne by numpy.nonzero
        """
        if index not in range(self.dimz): raise ValueError('invalid frame index')
        maskf = self._get_empty_frame()

        # load previous bad pixels
        maskf[self._get_cr_byframe_mask(index)] = True

        # append new bad pixels to cr frame
        maskf[bad_pix_list] = True
        self.cr_byframe[index] = np.nonzero(maskf)

        # append new bad pixels to cr spectra
        masksp = np.zeros(self.dimz, dtype=bool)
        bad_pixels = np.array(self.cr_byframe[index]).T
        for ipix in range(bad_pixels.shape[0]):
            ixy = bad_pixels[ipix,:]
            imask = self._get_cr_byspectrum_mask(ixy)
            if imask is not None:
                masksp[imask] = True
                
            masksp[index] = True
            self._set_cr_byspectrum_mask(ixy, np.nonzero(masksp))
        
    def load_cr_map(self, cr_map_path, alignment_parameters_path=None):
        """Load a cosmic ray map. If the map comes from camera B, it must be
        realigned with the corresponding alignment parameters which
        can be found in the files of the reduction pipeline.

        :param cr_map_path: Path to the cosmic ray map

        :param alignment_parameters: Path to a file containing the
          alignement parameters (dx, dy, dr, da, db, crx, cry, zx, zy)
        """
        cr_map_file = orb.cube.Cube(cr_map_path)
        if cr_map_file.shape != self.shape:
            raise TypeError('Cosmic ray map must have shape {} but has shape {}'.format(
                self.shape, cr_map.shape))

        if alignment_parameters_path is not None:
            if not isinstance(alignment_parameters_path, str):
                raise TypeError('alignment_parameters must be a path to an alignment parameters file')
            dx, dy, dr, da, db, rcx, rcy, zx, zy =  orb.utils.io.read_fits(alignment_parameters_path)
            
        progress = orb.core.ProgressBar(self.dimz)
        for iframe in range(self.dimz):
            progress.update(iframe, info='loading cr frame {}'.format(iframe))
            masked_pixels = np.nonzero(cr_map_file.get_data_frame(iframe))
            if alignment_parameters_path is not None:
                masked_pixels = np.array(masked_pixels, dtype=float).T
                for ipix in range(masked_pixels.shape[0]):                        
                    masked_pixels[ipix,:] = orb.cutils.transform_B_to_A(
                        masked_pixels[ipix, 0], masked_pixels[ipix, 1],
                        dx, dy, dr, da, db, rcx, rcy, zx, zy)
                    
                masked_pixels = (masked_pixels[:,0].astype(int),
                                 masked_pixels[:,1].astype(int))
                
            self.append(iframe, masked_pixels)
        progress.end()


    def load_ds9_region_file(self, reg_path):
        """Load a ds9 region file as a mask"""
        self.bad_region[orb.utils.misc.get_mask_from_ds9_region_file(
            reg_path, (0, self.dimx), (0, self.dimy), integrate=True)] = True

    def load_bad_frames(self, bad_frames_list):
        """Load a list of bad frames indexes"""
        orb.utils.validate.is_iterable(bad_frames_list)

        if (np.any(bad_frames_list >= self.dimz)
            or np.any(bad_frames_list < 0)):
            raise ValueError('invalid bad frame index')

        self.bad_frames = list(bad_frames_list)

    def get_spectrum_mask(self, x, y):
        """Return a mask along a spectrum taken at a given position
        :param x: X position of the spectrum
        :param y: Y position of the spectrum
        """
        orb.utils.validate.index(x, 0, self.dimx)
        orb.utils.validate.index(y, 0, self.dimy)

        mask = np.zeros(self.dimz, dtype=bool)

        if self.bad_region[x,y]:
            mask.fill(True)
            return np.nonzero(mask)
        
        for ibad in self.bad_frames:
            mask[ibad] = True

        spmask = self._get_cr_byspectrum_mask([x,y])
        if spmask is not None:
            mask[spmask] = True
            

        return np.nonzero(mask)


##################################################
#### CLASS RawData ###############################
##################################################

class RawData(orb.cube.InterferogramCube):
    """ORBS raw data processing class.

    .. note:: Raw data is the output data of SpIOMM/SITELLE without
      any kind of processing.
    """

    cr_map = None
    alignment_vector = None

    def _get_alignment_vector_path(self, err=False):
        """Return the default path to the alignment vector.

        :param err: (Optional) If True, the error vector path is
          returned (default False).
        """
        if not err:
            return self._data_path_hdr + "alignment_vector.fits"
        else:
            return self._data_path_hdr + "alignment_vector_err.fits"

    def _get_cr_map_cube_path(self):
        """Return the default path to a HDF5 cube of the cosmic rays."""
        return self._data_path_hdr + "cr_map.hdf5"
    
    def _get_hp_map_path(self):
        """Return the default path to the hot pixels map."""
        return self._data_path_hdr + "hp_map.fits"

    def _get_deep_frame_path(self):
        """Return the default path to the deep frame."""
        return self._data_path_hdr + "deep_frame.fits"

    def _get_interfero_cube_path(self):
        """Return the default path to the interferogram HDF5 cube."""
        return self._data_path_hdr + "interferogram.hdf5"

    def _get_master_path(self, kind):
        """Return the default path to a master frame.

        :param kind: Kind of master frame (e.g. : 'bias', 'dark',
          'flat')
        """
        return self._data_path_hdr + "master_%s.fits"%kind

            
    def _load_alignment_vector(self, alignment_vector_path):
        """Load the alignment vector.
          
        :param alignment_vector_path: Path to the alignment vector file.
        """
        logging.info("Loading alignment vector")
        alignment_vector = orb.utils.io.read_fits(alignment_vector_path, no_error=True)
        if (alignment_vector is not None):
            if (alignment_vector.shape[0] == self.dimz):
                logging.info("Alignment vector loaded")
                return alignment_vector
            else:
                raise StandardError("Alignment vector dimensions are not compatible")
                return None
        else:
            warnings.warn("Alignment vector not loaded")
            return None


    def create_alignment_vector(self, star_list_path, 
                                profile_name='gaussian',
                                min_coeff=0.3):
        """Create the alignment vector used to compute the
          interferogram from the raw images.

        :param star_list_path: Path to a list of star coordinates that
          will be used to calculates the displacement vector. Please
          refer to :meth:`orb.utils.astrometry.load_star_list` for more
          information about a list of stars.

        :param profile_name: (Optional) PSF profile for star
          fitting. Can be 'moffat' or 'gaussian'. See:
          :py:class:`orb.astrometry.Astrometry` (default 'gaussian').
    
        :param min_coeff: (Optional) The minimum proportion of stars
            correctly fitted to assume a good enough calculated
            disalignment (default 0.3).   

        .. note:: The alignement vector contains the calculated
           disalignment for each image along x and y axes to the first
           image.
        """
        logging.info("Creating alignment vector")

        # get alignment vectors
        (alignment_vector_x,
         alignment_vector_y,
         alignment_error) = self.get_alignment_vectors(star_list_path)
        

        self.alignment_vector = np.array([alignment_vector_x,
                                          alignment_vector_y]).T
        
        alignment_vector_path = self._get_alignment_vector_path()
        alignment_err_vector_path = self._get_alignment_vector_path(err=True)
        orb.utils.io.write_fits(alignment_vector_path, self.alignment_vector, 
                                fits_header=self.get_header())
        if self.indexer is not None:
            self.indexer['alignment_vector'] = alignment_vector_path
        orb.utils.io.write_fits(alignment_err_vector_path, np.array(alignment_error), 
                        fits_header=self.get_header())
        if self.indexer is not None:
            self.indexer['alignment_err_vector'] = alignment_err_vector_path
    
    
    def correct(self, bias_path=None, dark_path=None, flat_path=None,
                cr_map_cube_path=None, alignment_vector_path=None):
        
        """Correct raw data for bias, dark, flat, cosmic rays and
        alignment using the precomputed alignment vector and the
        cosmic ray map.
        
        :param bias_path: (Optional) Path to a list of bias files. If
          none given no correction is done.
        
        :param dark_path: (Optional) Path to a list of dark files. If
          none given no dark and flat corrections are done.
  
        :param flat_path: (Optional) Path to a list of flat files. If
          none given no flat correction is done.
        
        :param cr_map_cube_path: (Optional) Path to the cosmic ray map
          HDF5 cube, if none given the default path is used.
          
        :param alignment_vector_path: (Optional) Path to the alignment
          vector file, if none given the default path is used.                    
        """
        
        master_bias = None
        master_flat = None
        master_dark = None
        
        # load bias
        if bias_path is not None:
            bias_cube = orb.cube.HDFCube(bias_path, config=self.config)
            master_bias = bias_cube.get_master_frame()
            
            orb.utils.io.write_fits(self._get_master_path('bias'),
                                    master_bias, fits_header=self.get_header())
            
        # load flat
        if flat_path is not None:
            flat_cube = orb.cube.HDFCube(flat_path, config=self.config)
            master_flat = flat_cube.get_master_frame()
            orb.utils.io.write_fits(self._get_master_path('flat'),
                                    master_flat, fits_header=self.get_header())

        # load dark
        if dark_path is not None:
            dark_cube = orb.cube.HDFCube(dark_path, config=self.config)
            master_dark = dark_cube.get_master_frame()
            master_dark /= self.params.dark_time            
            orb.utils.io.write_fits(self._get_master_path('dark'),
                                    master_dark, fits_header=self.get_header())

        # load alignment vector
        if (alignment_vector_path is None):
            alignment_vector_path = self._get_alignment_vector_path() 
        alignment_vector = self._load_alignment_vector(alignment_vector_path)
        if (alignment_vector is None):
            alignment_vector = np.zeros((self.dimz, 2), dtype = float)
            warnings.warn("No alignment vector loaded : there will be no alignment of the images")

        # load cosmic ray map cube
        if cr_map_cube_path is None:
            cr_map_cube_path = self._get_cr_map_cube_path()
            
        if os.path.exists(cr_map_cube_path):
            cr_map_cube = orb.cube.Cube(cr_map_cube_path,
                                        instrument=self.instrument,
                                        config=self.config,
                                        params=self.params)
            logging.info("Loaded cosmic ray map: {}".format(cr_map_cube_path))
        else:
            warnings.warn("No cosmic ray map loaded")


        # Multiprocessing server init
        job_server, ncpus = self._init_pp_server() 
        ncpus_max = ncpus

        # creating output hdfcube
        out_cube = orb.cube.RWHDFCube(self._get_interfero_cube_path(),
                                      shape=(self.dimx, self.dimy, self.dimz),
                                      instrument=self.instrument,
                                      config=self.config,
                                      params=self.params,
                                      reset=True)

        
        progress = orb.core.ProgressBar(self.dimz)


        def detrend(iframe, instrument, config, params, bias, dark, flat, shift, crm):
            import warnings
            warnings.simplefilter('ignore')
            
            image = orb.image.Image(iframe,
                                    instrument=instrument,
                                    config=config,
                                    params=params)
            iframe = image.detrend(bias=bias,
                                   dark=dark,
                                   flat=flat,
                                   shift=shift,
                                   cr_map=crm)
            return iframe

        
        for ik in range(0, self.dimz, ncpus):

                                    
            # No more jobs than frames to compute
            if (ik + ncpus >= self.dimz): 
                ncpus = self.dimz - ik
                
            progress.update(ik, info="loading " + str(ik))

            frames = self[:,:,ik:ik+ncpus]
            cr_frames = cr_map_cube[:,:,ik:ik+ncpus].astype(np.bool)
            

            jobs = [(ijob, job_server.submit(
                detrend,
                args=(frames[:,:,ijob],
                      str(self.instrument),
                      dict(self.config),
                      dict(self.params),
                      master_bias,
                      master_dark,
                      master_flat,
                      alignment_vector[ik+ijob,:],
                      cr_frames[:,:,ijob]),
                modules=(
                    "import orb.image",))) 
                    for ijob in range(ncpus)]

            progress.update(ik, info="detrending " + str(ik))

            for ijob, job in jobs:
                frames[:,:,ijob] = job()
                
            out_cube[:,:,ik:ik+ncpus] = frames

        self._close_pp_server(job_server)
        progress.end()

        if self.indexer is not None:
            self.indexer['interfero_cube'] = self._get_interfero_cube_path()
            
        logging.info("Interferogram computed")

        # Create deep frame        
        deep_frame = out_cube.get_deep_frame().data
        out_cube.set_deep_frame(deep_frame)
        
        if np.nanmedian(deep_frame) < 0.:
            warnings.warn('Deep frame median of the corrected cube is < 0. ({}), please check the calibration files (dark, bias, flat).')
        
        orb.utils.io.write_fits(
            self._get_deep_frame_path(), deep_frame,
            fits_header=self.get_header())
        
        if self.indexer is not None:
            self.indexer['deep_frame'] = self._get_deep_frame_path()
        
        del out_cube    


##################################################
#### CLASS CalibrationLaser ######################
##################################################

class CalibrationLaser(orb.cube.InterferogramCube):
    """ ORBS calibration laser processing class.

    CalibrationLaser class is aimed to compute the calibration laser map that
    is used to correct for the **off-axis effect**.

    .. note:: The **off-axis effect** comes from the angle between one
     pixel of a camera and the optical axis of the interferometer. For
     a given displacement of the mirror (step) the optical path
     difference 'seen' by a pixel is different and depends on the
     off-axis angle. The effect on the spectrum corresponds to a
     changing step in wavelength (between two channels) and thus an
     expanded spectrum relatively to its theoretical shape.

     The **calibration laser cube** is an interferogram cube taken with a
     monochromatic light. The real position of the emission line (its
     channel) help us to correct for the step variations using the
     formula :

     .. math::
     
        step_{real} = step_{th} * \\frac{\\lambda_{LASER}}{channel}
    """

    x_center = None
    y_center = None
    pix_size = None
    
    def _get_calibration_laser_map_path(self):
        """Return the default path to the calibration laser map."""
        return self._data_path_hdr + "calibration_laser_map.fits"
    
    def _get_calibration_laser_fitparams_path(self):
        """Return the path to the file containing the fit parameters
        of the calibration laser cube."""
        return self._data_path_hdr + "calibration_laser_fitparams.fits"
    
    def _get_calibration_laser_ils_ratio_path(self):
        """Return the path to the file containing the instrumental
        line shape ratio map (ILS / theoretical ILS) of the
        calibration laser cube."""
        return self._data_path_hdr + "calibration_laser_ils_ratio_map.fits"
    
    def _get_calibration_laser_spectrum_cube_path(self):
        """Return the default path to the reduced calibration laser
        HDF5 cube."""
        return self._data_path_hdr + "calibration_laser_cube.hdf5"

    def create_calibration_laser_map(self, order=30, step=9765,
                                     get_calibration_laser_spectrum=False,
                                     fast=True):
        """ Create the calibration laser map.

        Compute the spectral cube from the calibration laser cube and
        create the calibration laser map containing the fitted central
        position of the emission line for each pixel of the image
        plane (x/y axes).

        :param order: (Optional) Folding order
        :param step: (Optional) Step size in nm
        
        :param get_calibration_laser_spectrum: (Optional) If True return the
          calibration laser spectrum

        :param fast: (Optional) If False a sinc^2 is fitted so the fit
          is better but the procedure becomes slower. If True a
          gaussian is fitted (default True).
        """
        def _find_max_in_column(column_data, step, order, cm1_axis_min,
                                cm1_axis_step,
                                get_calibration_laser_spectrum, fast,
                                fwhm_guess, fwhm_guess_cm1):


            """Return the fitted central position of the emission line"""
            dimy = column_data.shape[0]
            dimz = column_data.shape[1]
            BORDER = int(0.3 * dimz) + 1
            max_array_column = np.empty((dimy), dtype=float)
            fitparams_column = np.empty((dimy, 10), dtype=float)
            max_array_column.fill(np.nan)
            fitparams_column.fill(np.nan)
            
                        
            # FFT of the interferogram
                
            for ij in range(column_data.shape[0]):
                if np.all(np.isnan(column_data[ij,:])): continue
                column_data[ij,np.isnan(column_data[ij,:])] = 0.

                raise NotImplementedError('new fft transform must be used')
                ## zpv = orb.utils.fft.transform_interferogram(
                ##     column_data[ij,:], 1, 1, step, order,
                ##     '2.0', 0, phase_correction=False,
                ##     wavenumber=True, return_zp_vector=True)
                spectrum_vector = np.abs(np.fft.fft(zpv)[:zpv.shape[0]/2])
                if (int(order) & 1):
                    spectrum_vector = spectrum_vector[::-1]
                    
                # defining window
                max_index = np.argmax(spectrum_vector)
                range_min = max_index - BORDER
                if (range_min < 0):
                    range_min = 0
                range_max = max_index + BORDER + 1L
                if (range_max >= len(spectrum_vector)):
                    range_max = len(spectrum_vector) - 1

                if (not np.any(np.isnan(spectrum_vector))
                    and (max_index > 3*fwhm_guess)
                    and (max_index < dimz - 3*fwhm_guess)):
                    # gaussian fit (fast)
                    if fast:
                        fitp = orb.fit.fit_lines_in_vector(
                            spectrum_vector, [max_index],
                            fmodel='gaussian',
                            fwhm_guess=fwhm_guess,
                            poly_order=0,
                            signal_range=[range_min, range_max],
                            cont_guess=[0.], no_error=True)

                        ## fitp = {'lines-params':[[0,1,max_index,1]],
                        ##         'lines-params-err':[[0,0,0,0]]}
                    # or sinc2 fit (slow)
                    else:
                        raise Exception("Very bad, please don't use it")
                        fitp = orb.fit.fit_lines_in_vector(
                            spectrum_vector, [max_index], fmodel='sinc2',
                            fwhm_guess=fwhm_guess,
                            poly_order=0,
                            signal_range=[range_min, range_max],
                            cont_guess=[0.], no_error=True)
                else:
                    fitp = []
                    
                if (fitp != []):
                    max_index_fit = fitp['lines_params'][0][2]
                    max_array_column[ij] = 1. / orb.cutils.fast_pix2w(
                        np.array([max_index_fit], dtype=float),
                        cm1_axis_min, cm1_axis_step) * 1e7
                    if 'lines_params_err' in fitp:
                        fitparams_column[ij,:] = np.array(
                            list(fitp['lines_params'][0])
                            + list(fitp['lines_params_err'][0]))
                    else:
                        fitparams_column.fill(np.nan)
                        fitparams_column[ij,:5] = fitp['lines_params'][0]
                else:
                    max_array_column[ij] = np.nan
                    fitparams_column[ij,:].fill(np.nan)

                # check if fit range is large enough
                fwhm_median = np.median(fitparams_column[:,3])
                if fwhm_median is not np.nan:
                    if (range_max - range_min) < 5. * fwhm_median:
                        import warnings
                        warnings.warn('fit range is not large enough: median fwhm ({}) > 5xrange ({})'.format(fwhm_median, range_max - range_min))
                    
            if not get_calibration_laser_spectrum:
                return max_array_column, fitparams_column
            else:
                return max_array_column, fitparams_column, column_spectrum

        logging.info("Computing calibration laser map")

        order = float(order)
        step = float(step)

        # create the fft axis in cm1
        cm1_axis_min = orb.cutils.get_cm1_axis_min(self.dimz, step, order)
        cm1_axis_step = orb.cutils.get_cm1_axis_step(self.dimz, step)
    
        # guess fwhm
        fwhm_guess = orb.utils.spectrum.compute_line_fwhm_pix(
            oversampling_ratio=2.)
        fwhm_guess_cm1 = orb.utils.spectrum.compute_line_fwhm(
            self.dimz/2, step, order, wavenumber=True)
        
        logging.info('FWHM guess: {} pixels, {} cm-1'.format(
            fwhm_guess,
            fwhm_guess_cm1))

        out_cube = OutHDFQuadCube(
            self._get_calibration_laser_spectrum_cube_path(),
            (self.dimx, self.dimy, self.dimz),
            self.config.QUAD_NB,
            reset=True)

        fitparams = np.empty((self.dimx, self.dimy, 10), dtype=float)
        fitparams.fill(np.nan)
        max_array = np.empty((self.dimx, self.dimy), dtype=float)
        max_array.fill(np.nan)
        
        for iquad in range(self.config.QUAD_NB):
            x_min, x_max, y_min, y_max = self.get_quadrant_dims(iquad)
            iquad_data = self.get_data(x_min, x_max, 
                                       y_min, y_max, 
                                       0, self.dimz)
            # init multiprocessing server
            job_server, ncpus = self._init_pp_server()

            progress = orb.core.ProgressBar(x_max - x_min)
            for ii in range(0, x_max - x_min, ncpus):
                # create no more jobs than work to do
                if (ii + ncpus >= x_max - x_min): 
                    ncpus = x_max - x_min - ii
                # create jobs
                jobs = [(ijob, job_server.submit(
                    _find_max_in_column, 
                    args=(iquad_data[ii+ijob,:,:],
                          step, order, cm1_axis_min, cm1_axis_step,
                          get_calibration_laser_spectrum, fast,
                          fwhm_guess, fwhm_guess_cm1),
                    modules=("import logging",
                             "numpy as np",
                             "math",
                             "import orb.utils.fft",
                             "import orb.cutils",
                             "import orb.fit"))) 
                        for ijob in range(ncpus)]

                # execute jobs
                for ijob, job in jobs:
                    if not get_calibration_laser_spectrum:
                        (max_array[x_min + ii + ijob, y_min:y_max],
                         fitparams[x_min + ii + ijob, y_min:y_max,:]) = job()
                    else:
                        (max_array[x_min + ii + ijob, y_min:y_max],
                         fitparams[x_min + ii + ijob, y_min:y_max,:],
                         iquad_data[ii+ijob,:,:]) = job()
                        
                progress.update(ii, info="quad %d/%d, column : %d"%(
                    iquad+1L, self.config.QUAD_NB, ii))
            self._close_pp_server(job_server)
            progress.end()

            if get_calibration_laser_spectrum:
                # save data
                logging.info('Writing quad {}/{} to disk'.format(
                    iquad+1, self.config.QUAD_NB))
                write_start_time = time.time()
                out_cube.write_quad(iquad, data=iquad_data)
                logging.info('Quad {}/{} written in {:.2f} s'.format(
                    iquad+1, self.config.QUAD_NB, time.time() - write_start_time))
            

        out_cube.close()
        del out_cube

        # Write uncorrected calibration laser map to disk (in case the
        # correction does not work)
        orb.utils.io.write_fits(self._get_calibration_laser_map_path(), max_array,
                                fits_header=self.get_header())

        # Correct non-fitted values by interpolation
        ## max_array = orb.utils.image.correct_map2d(max_array, bad_value=np.nan)
        ## max_array = orb.utils.image.correct_map2d(max_array, bad_value=0.)

        ## # Re-Write calibration laser map to disk
        ## orb.utils.io.write_fits(self._get_calibration_laser_map_path(), max_array,
        ##                 fits_header=self._get_calibration_laser_map_header(),
        ##                 overwrite=self.overwrite)

        # write fit params
        orb.utils.io.write_fits(
            self._get_calibration_laser_fitparams_path(), fitparams,
            fits_header=self.get_header())

        # write ils_ratio
        ils_ratio = fitparams[:,:,3] / fwhm_guess
        orb.utils.io.write_fits(
            self._get_calibration_laser_ils_ratio_path(), ils_ratio,
            fits_header=self.get_header())

        if self.indexer is not None:
            self.indexer['calibration_laser_map'] = (
                self._get_calibration_laser_map_path())


##################################################
#### CLASS Interferogram #########################
##################################################

class Interferogram(orb.cube.InterferogramCube):
    """ORBS interferogram processing class.

    .. note:: Interferogram data is defined as data already processed
       (corrected and aligned frames) by :class:`process.RawData` and
       ready to be transformed to a spectrum by a Fast Fourier
       Transform (FFT).
    """

    def _get_phase_cube_model_path(self):
        """Return path to the phase cube model"""
        return self._data_path_hdr + 'phase_cube_model.fits'

    def _get_high_order_phase_path(self):
        """Return path to the high order phase"""
        return self._data_path_hdr + 'high_order_phase.hdf5'

    def _get_high_order_phase_std_path(self):
        """Return path to the high order phase std"""
        return self._data_path_hdr + 'high_order_phase_std.hdf5'

    def _get_binned_phase_cube_path(self):
        """Return path to the binned phase cube."""
        return self._data_path_hdr + "binned_phase_cube.hdf5"

    def _get_binned_interferogram_cube_path(self):
        """Return path to the binned interferogram cube."""
        return self._data_path_hdr + "binned_interferogram_cube.hdf5"

    def _get_binned_calibration_laser_map_path(self):
        """Return path to the binned calibration laser map."""
        return self._data_path_hdr + "binned_calibration_laser_map.fits"

    def _get_transmission_vector_path(self):
        """Return the path to the transmission vector"""
        return self._data_path_hdr + "transmission_vector.fits"
   
    def _get_stray_light_vector_path(self):
        """Return the path to the stray light vector"""
        return self._data_path_hdr + "stray_light_vector.fits"

    def _get_extracted_star_spectra_path(self):
        """Return the path to the extracted star spectra"""
        return self._data_path_hdr + "extracted_star_spectra.fits"

    def _get_corrected_interferogram_cube_path(self):
        """Return the default path to a spectrum HDF5 cube."""
        return self._data_path_hdr + 'corrected_interferogram_cube.hdf5'

    def _get_spectrum_cube_path(self, phase=False):
        """Return the default path to a spectrum HDF5 cube.

        :param phase: (Optional) If True the path is changed for a
          phase cube (default False).
        """
        if not phase: cube_type = "spectrum"
        else: cube_type = "phase"
        return self._data_path_hdr + cube_type + '.hdf5'

    def correct_interferogram(self, transmission_vector_path,
                              stray_light_vector_path):
        """Correct an interferogram cube for for variations
        of sky transmission and stray light.

        :param sky_transmission_vector_path: Path to the transmission
          vector. All the interferograms of the cube are divided by
          this vector. The vector must have the same size as the 3rd
          axis of the cube (the OPD axis).

        :param stray_light_vector_path: Path to the stray light
          vector. This vector is substracted from the interferograms
          of all the cube. The vector must have the same size as the
          3rd axis of the cube (the OPD axis).

        .. note:: The sky transmission vector gives the absorption
          caused by clouds or airmass variation.

        .. note:: The stray light vector gives the counts added
          homogeneously to each frame caused by a cloud reflecting
          light coming from the ground, the moon or the sun.

        .. seealso:: :py:meth:`process.Interferogram.create_correction_vectors`
        """
        def _correct_frame(frame, transmission_coeff, stray_light_coeff):
            if not np.all(frame==0.):
                return (frame - stray_light_coeff) / transmission_coeff
            else:
                return frame

        logging.info('Correcting interferogram')

        # Avoid transmission correction (useful for testing purpose)
        NO_TRANSMISSION_CORRECTION = bool(int(
            self._get_tuning_parameter('NO_TRANSMISSION_CORRECTION', 0)))
        if NO_TRANSMISSION_CORRECTION:
            warnings.warn('No transmission correction')
            
        transmission_vector = orb.utils.io.read_fits(transmission_vector_path)
        if NO_TRANSMISSION_CORRECTION:
            transmission_vector.fill(1.)
            
        stray_light_vector = orb.utils.io.read_fits(stray_light_vector_path)
        
        # Multiprocessing server init
        job_server, ncpus = self._init_pp_server() 

        # creating output hdfcube
        out_cube = OutHDFCube(self._get_corrected_interferogram_cube_path(),
                              (self.dimx, self.dimy, self.dimz),
                              overwrite=self.overwrite)
        
        # Interferogram creation
        progress = orb.core.ProgressBar(self.dimz)
        for ik in range(0, self.dimz, ncpus):
            # No more jobs than frames to compute
            if (ik + ncpus >= self.dimz): 
                ncpus = self.dimz - ik

            frames = np.empty((self.dimx, self.dimy, ncpus), dtype=float)
            
            jobs = [(ijob, job_server.submit(
                _correct_frame,
                args=(np.array(self.get_data_frame(ik+ijob)),
                      transmission_vector[ik+ijob],
                      stray_light_vector[ik+ijob]),
                modules=("import logging",
                         'import numpy as np',)))
                    for ijob in range(ncpus)]

            for ijob, job in jobs:
                frames[:,:,ijob] = job()
                
            for ijob in range(ncpus):
                
                out_cube.write_frame(
                    ik+ijob,
                    data=frames[:,:,ijob],
                    header=self.get_header())
            progress.update(ik, info="Correcting frame %d"%ik)

        progress.end()

        if self.indexer is not None:
            self.indexer['corr_interfero_cube'] = (
                self._get_corrected_interferogram_cube_path())
            
        self._close_pp_server(job_server)
            

    def create_phase_maps(self, binning, poly_order,
                          high_order_phase_path=None):
        """Create phase maps

        :param binning: Interferogram cube is binned before to
          accelerate computation.

        :param poly_order: Order of the fitted polynomial (must be >= 1)

        :param high_order_phase_path: (Optional) Path to an HDF5 phase
          file.
        """
        if not isinstance(poly_order, int): raise TypeError('poly_order must be an int')
        if poly_order < 1: raise ValueError('poly_order = {} but must be >= 1'.format(poly_order))
        
        logging.info('Computing phase maps up to order {}'.format(poly_order))        

        if high_order_phase_path is not None:
            high_order_phase = orb.fft.Phase(high_order_phase_path, None)
            logging.info('High order phase file loaded: {}'.format(high_order_phase_path))
        else:
            high_order_phase = None

        self.create_binned_interferogram_cube(binning)

        interf_cube = BinnedInterferogramCube(
            self._get_binned_interferogram_cube_path(), config=self.config,
            params=self.params, instrument=self.instrument)

        phase_cube = interf_cube.compute_phase()
        out_cube = orb.cube.RWHDFCube(self._get_binned_phase_cube_path(),
                                      shape=phase_cube.shape,
                                      instrument=self.instrument,
                                      config=self.config,
                                      params=self.params,
                                      reset=True)
        out_cube[:,:,:] = phase_cube
        del out_cube

        if self.indexer is not None:
            self.indexer['binned_phase_cube'] = (
                self._get_binned_phase_cube_path())

        phase_cube = BinnedPhaseCube(
            self._get_binned_phase_cube_path(),
            params=self.params, instrument=self.instrument, config=self.config,
            data_prefix=self._data_prefix)

        # compute phase maps iteratively
        final_phase_maps_path = phase_cube.iterative_polyfit(
            poly_order, high_order_phase=high_order_phase)

        logging.info('final computed phase maps path: {}'.format(final_phase_maps_path))
        if self.indexer is not None:                 
            self.indexer['phase_maps'] = final_phase_maps_path

        # compute high order phase
        phasemaps = orb.fft.PhaseMaps(final_phase_maps_path)
        phasemaps.modelize()

        coeffs = [None] * 2 + [0] * (poly_order - 1)
        phasemaps.generate_phase_cube(self._get_phase_cube_model_path(),
                                      coeffs=coeffs)
        
        phase_cube_model = orb.utils.io.read_fits(self._get_phase_cube_model_path())
        phase_cube_residual = phase_cube[:,:,:] - phase_cube_model

        ## remove the median of the phase vector at each pixel
        fake_phase = phase_cube.get_phase(10, 10)
        zmin, zmax = fake_phase.get_filter_bandpass_pix(border_ratio=0.2)
        medframe = np.nanmean(phase_cube_residual[:,:,zmin:zmax], axis=2)        
        phase_cube_residual = np.subtract(phase_cube_residual.T, medframe.T).T

        # compute mean
        high_order_phase = np.nanmean(phase_cube_residual, axis=(0,1))
        high_order_phase = orb.fft.Phase(high_order_phase, phase_cube.get_base_axis(),
                                         params=phase_cube.params)
        
        # compute std
        high_order_phase_std = np.nanstd(phase_cube_residual, axis=(0,1))

        median_std = np.median(high_order_phase_std[zmin:zmax])
        logging.info('median std of each phase value: {:.2e} rad'.format(
            median_std))
        
        logging.info('estimated precision: {:.2e} rad'.format(
            median_std / np.sqrt(phase_cube.dimx * phase_cube.dimy)))
        
        
        high_order_phase_std = orb.fft.Phase(
            high_order_phase_std, phase_cube.get_base_axis(),
            params=phase_cube.params)

        # remove orders 0 and 1 from the phase residual
        high_order_phase = high_order_phase.cleaned(border_ratio=-0.1)
        phase_corr = high_order_phase.data - high_order_phase.polyfit(1).data

        # extrapolate values on the border to avoid phase switch off
        # filter borders
        filtmin = np.argmin(np.isnan(phase_corr))
        phase_corr[:filtmin] = phase_corr[filtmin]
        filtmax = np.argmax(np.isnan(phase_corr))
        phase_corr[filtmax:] = phase_corr[filtmax-1]
        high_order_phase_corr = orb.fft.Phase(
            phase_corr,
            high_order_phase.axis.data,
            params=phase_cube.params)

        high_order_phase_corr.writeto(self._get_high_order_phase_path())
        high_order_phase_std.writeto(self._get_high_order_phase_std_path())
    
        logging.info('high order phase path: {}'.format(
            self._get_high_order_phase_path()))
        if self.indexer is not None:                 
            self.indexer['high_order_phase'] = self._get_high_order_phase_path()


    def compute_spectrum(self, phase_correction=True,
                         bad_frames_vector=None, window_type=None,
                         phase_cube=False, phase_maps_path=None,
                         high_order_phase_path=None,
                         wave_calibration=False, balanced=True,
                         wavenumber=False):
        
        """Compute the spectrum from the corrected interferogram
        frames. Can be used to compute spectrum for camera 1, camera 2
        or merged interferogram.

        :param window_type: (Optional) Apodization window to be used
          (Default None, no apodization)

        :param phase_correction: (Optional) If False, no phase
          correction will be done and the resulting spectrum will be
          the absolute value of the complex spectrum (default True).

        :param phase_cube: (Optional) If True, only the phase cube is
          returned (default False).

        :param phase_maps_path: (Optional) Path to the HDF5 phase map
          file.

        :param high_order_phase_path: (Optional) Path to an HDF5 phase
          file.
      
        :param balanced: (Optional) If False, the interferogram is
          considered as unbalanced. It is flipped before its
          transformation to get a positive spectrum. Note that a
          merged interferogram is balanced (default True).
              
        :param wavenumber: (Optional) If True, the returned spectrum
          is projected onto its original wavenumber axis (emission
          lines and especially unapodized sinc emission lines are thus
          symetric which is not the case if the spectrum is projected
          onto a, more convenient, regular wavelength axis) (default
          False).

        :param wave_calibration: (Optional) If True wavelength
          calibration is done (default False).

        """

        def _compute_spectrum_in_column(params, calibration_coeff_map_column,
                                        data, window_type,
                                        phase_correction,
                                        wave_calibration,
                                        phase_maps_col,
                                        phase_maps_axis,
                                        phase_maps_params,
                                        high_order_phase_data,
                                        high_order_phase_axis,
                                        return_phase, # not implemented
                                        balanced,
                                        wavenumber): # nm computation not implemented
            """Compute spectrum in one column. Used to parallelize the
            process"""
            from orb.fft import Interferogram, Phase
            import logging
            import orb.utils.log
            import time
            orb.utils.log.setup_socket_logging()
            dimz = data.shape[1]
            spectrum_column = np.zeros_like(data, dtype=complex)
            ho_phase = Phase(high_order_phase_data,
                             axis=high_order_phase_axis,
                             params=params)

            times = {'loop':list(), 'probe1':list(), 'probe2':list()}
            for ij in range(data.shape[0]):
                itime = dict()
                stime = time.time()
            
                # throw out interferograms with less than half non-zero values
                # (zero values are considered as bad points : cosmic rays, bad
                # frames etc.)
                iinterf_data = np.copy(data[ij,:])
                iinterf_data[np.nonzero(np.isnan(iinterf_data))] = 0.
                
                if len(np.nonzero(iinterf_data)[0]) < dimz/2.:
                    continue

                icalib_coeff = calibration_coeff_map_column[ij]

                times['probe1'].append(time.time()-stime)
                iinterf = Interferogram(
                    iinterf_data, params=params,
                    calib_coeff=icalib_coeff)

                times['probe2'].append(time.time()-stime)
                ispectrum = iinterf.get_spectrum()

                iphase = Phase(phase_maps_col[ij,:],
                               axis=phase_maps_axis,
                               params=phase_maps_params)
                iphase.add(ho_phase)

                if phase_correction:
                    ispectrum.correct_phase(iphase)
                    spectrum_column[ij,:] = ispectrum.data
                else:
                    spectrum_column[ij,:] = ispectrum.get_amplitude()

                times['loop'].append(time.time()-stime)
            if np.nansum(spectrum_column) == 0:
                logging.debug('Whole column filled with zeroes')
                
            return spectrum_column, times

            
        if not phase_cube:
            logging.info("Computing spectrum")
        else: 
            logging.info("Computing phase")
            raise NotImplementedError('Phase computation not implemented')

        if phase_correction:
            # get phase
            phase_maps = orb.fft.PhaseMaps(phase_maps_path)
            phase_maps.modelize() # phase maps model is computed in place
            logging.info('Phase maps file: {}'.format(phase_maps_path))
            if high_order_phase_path is not None:
                high_order_phase = orb.fft.Phase(high_order_phase_path, None)
                logging.info('High order phase file loaded: {}'.format(high_order_phase_path))
                high_order_phase_data = np.copy(high_order_phase.data)
                high_order_phase_axis = np.copy(high_order_phase.axis.data)
            else:
                high_order_phase = None
                high_order_phase_data = None
                high_order_phase_axis = None
        else:
            phase_maps = None
            warnings.warn('No phase correction')

        if wave_calibration:
            warnings.warn('Wavelength/wavenumber calibration')
            raise NotImplementedError('Wavenumber calibration not implemented. Must be done through process.Spectrum.calibrate()')
        
        #############################
        ## Note: variable names are all "spectrum" related even if it
        ## is possible to get only the phase cube. Better for the
        ## clarity of the code
        #############################

        ## Searching ZPD shift 
            
        ## Check spectrum polarity
            
        # Note: The Oth order phase map is defined modulo pi. But a
        # difference of pi in the 0th order of the phase vector change
        # the polarity of the spectrum (the returned spectrum is
        # reversed). As there is no way to know the correct phase,
        # spectrum polarity must be tested. We get the mean
        # interferogram and transform it to check.
        if (phase_maps is not None and phase_correction):
            
            logging.info("Check spectrum polarity with phase correction")
            
            # get mean interferogram
            xmin, xmax, ymin, ymax = orb.utils.image.get_box_coords(
                self.dimx/2, self.dimy/2,
                int(0.02*self.dimx),
                0, self.dimx,
                0, self.dimy)
            ### DEBUG
            # xmin = 213-5
            # xmax = 213+5
            # ymin = 1416-5
            # ymax = 1416+5
            ### DEBUG

            mean_interf = bn.nanmedian(bn.nanmedian(
                self.get_data(xmin, xmax, ymin, ymax, 0, self.dimz),
                axis=0), axis=0)
            
            mean_calib = np.nanmean(self.get_calibration_coeff_map()[xmin:xmax, ymin:ymax])

            mean_interf = orb.fft.Interferogram(
                mean_interf, params=self.params, calib_coeff=mean_calib)
            mean_spectrum = mean_interf.get_spectrum()
            
            mean_phase = phase_maps.get_phase(self.dimx/2, self.dimy/2, unbin=True)
            mean_phase.add(high_order_phase)
            mean_spectrum.correct_phase(mean_phase)
                        
            if np.nanmean(mean_spectrum.data.real) < 0:
                logging.info("Negative polarity : 0th order phase map has been corrected (add PI)")
                phase_maps.reverse_polarity()


            if (orb.utils.fft.spectrum_mean_energy(mean_spectrum.data.imag)
                > .5 * orb.utils.fft.spectrum_mean_energy(mean_spectrum.data.real)):
                warnings.warn("Too much energy in the imaginary part, check the phase correction")
      
        ## Spectrum computation

        # Print some informations about the spectrum transformation
        
        logging.info("Apodization function: %s"%window_type)
        logging.info("Folding order: %f"%self.params.order)
        logging.info("Step size: %f"%self.params.step)
        logging.info("Wavenumber output: {}".format(wavenumber))
        if not wavenumber: raise NotImplementedError('Wavelength computation not implemented')
        
        out_cube = OutHDFQuadCube(
            self._get_spectrum_cube_path(phase=phase_cube),
            (self.dimx, self.dimy, self.dimz),
            self.config.QUAD_NB,
            overwrite=self.overwrite)

        out_cube.append_header(pyfits.Header(self.get_header()))
        
        def get_phase_maps_cols(phase_maps, _x, _y_min, _y_max):            
            # create phase column
            phase_maps_cols = np.empty((_y_max - _y_min, np.size(phase_maps.axis)), dtype=float)
            for ij in range(phase_maps_cols.shape[0]):
                phase_maps_cols[ij, :] = phase_maps.get_phase(_x, _y_min + ij, unbin=True).data

            return phase_maps_cols
        
        for iquad in range(0, self.config.QUAD_NB):
            x_min, x_max, y_min, y_max = self.get_quadrant_dims(iquad)
            iquad_data = self.get_data(x_min, x_max, 
                                       y_min, y_max, 
                                       0, self.dimz)
            iquad_data = iquad_data.astype(np.complex128)
            ## iquad_data = np.empty((x_max-x_min, y_max-y_min,self.dimz), dtype=float)
            # multi-processing server init
            job_server, ncpus = self._init_pp_server()
            progress = orb.core.ProgressBar(x_max - x_min)
            for ii in range(0, x_max - x_min, ncpus):
                # no more jobs than columns
                if (ii + ncpus >= x_max - x_min): 
                    ncpus = x_max - x_min - ii
              
                # jobs creation
                jobs = [(ijob, job_server.submit(
                    _compute_spectrum_in_column,
                    args=(self.params.convert(),  
                          self.get_calibration_coeff_map()[x_min + ii + ijob,
                                                           y_min:y_max], 
                          iquad_data[ii+ijob,:,:].real,
                          window_type, 
                          phase_correction, wave_calibration,
                          get_phase_maps_cols(
                              phase_maps, x_min + ii + ijob, y_min, y_max),
                          phase_maps.axis,
                          phase_maps.params.convert(),
                          high_order_phase_data,
                          high_order_phase_axis,
                          phase_cube, balanced,
                          wavenumber), 
                    modules=("import logging",
                             "import numpy as np")))
                        for ijob in range(ncpus)]

                for ijob, job in jobs:
                    # spectrum comes in place of the interferograms
                    # to avoid using too much memory
                    iquad_data[ii+ijob,:,:], times = job()
                    logging.debug({'looping time: {}'.format(np.median(times['loop']))})
                    logging.debug({'probe1 time: {}'.format(np.median(times['probe1']))})
                    logging.debug({'probe2 time: {}'.format(np.median(times['probe2']))})
                
                progress.update(ii, info="Quad %d/%d column : %d"%(
                        iquad+1L, self.config.QUAD_NB, ii))
            self._close_pp_server(job_server)
            progress.end()
            
            # save data
            logging.info('Writing quad {}/{} to disk'.format(
                iquad+1, self.config.QUAD_NB))
            write_start_time = time.time()
            out_cube.write_quad(
                iquad, data=iquad_data,
                force_float32=False, force_complex64=True)
            logging.info('Quad {}/{} written in {:.2f} s'.format(
                iquad+1, self.config.QUAD_NB, time.time() - write_start_time))
            
            
        out_cube.close()
        del out_cube
            
        # Create indexer key
        if not phase_cube:
            logging.info("Spectrum computed")
            cube_file_key = 'spectrum_cube'
            
        else:
            logging.info("Phase computed")
            cube_file_key = 'phase_cube'
            

        if self.indexer is not None:
                self.indexer[cube_file_key] = self._get_spectrum_cube_path(
                    phase=phase_cube)

    def create_binned_calibration_laser_map(self, binning,
                                            calibration_laser_map_path):
        """Create a binned calibration laser map

        :param binning: Binning
        :param calibration_laser_map_path: Calibration laser map path
        """
        # Loading calibration laser map
        logging.info("loading calibration laser map")
        calibration_laser_map = orb.utils.io.read_fits(calibration_laser_map_path)

        if (calibration_laser_map.shape[0] != self.dimx):
            calibration_laser_map = orb.utils.image.interpolate_map(
                calibration_laser_map, self.dimx, self.dimy)
            
        if binning > 1:
            calibration_laser_map = orb.utils.image.nanbin_image(
                calibration_laser_map, binning)
        
        # write binned calib map
        orb.utils.io.write_fits(self._get_binned_calibration_laser_map_path(),
                                calibration_laser_map, fits_header=self.get_header())
        
        if self.indexer is not None:
            self.indexer['binned_calibration_laser_map'] = (
                self._get_binned_calibration_laser_map_path())

    def create_binned_interferogram_cube(self, binning):
        """Create a binned interferogram cube

        :param binning: Binning
        """
        if binning > 1:
            cube_bin = self.get_binned_cube(binning)
        else:
            cube_bin = self
            self._silent_load = True
            
        # write binned interferogram cube
        out_cube = orb.cube.RWHDFCube(self._get_binned_interferogram_cube_path(),
                                      shape=cube_bin.shape,
                                      instrument=self.instrument,
                                      config=self.config,
                                      params=self.params,
                                      reset=True)

        out_cube[:,:,:] = cube_bin
        del out_cube
        
        if self.indexer is not None:
            self.indexer['binned_interferogram_cube'] = (
                self._get_binned_interferogram_cube_path())



##################################################
#### CLASS InterferogramMerger ###################
##################################################

class InterferogramMerger(orb.core.Tools):
    """ORBS interferogram merging class.

    The InterferogramMerger class is aimed to merge the interferogram
    cubes of the two cameras of SpIOMM/SITELLE.

    .. note:: In this class the letter 'A' refers to the camera 1 and the
       letter 'B' to the camera 2
    """
    
    cube_A = None
    cube_B = None
    dx = None
    dy = None
    dr = None
    da = None
    db = None
    xc = None
    yc = None
    rc = None
    zoom_factor = None
    alignment_coeffs = None
    
    pix_size_A = None
    pix_size_B = None
    bin_A = None
    bin_B = None
    
    overwrite = False
    
    _data_prefix = None
    instrument = None
    ncpus = None
    _data_path_hdr = None
    _project_header = None
    _wcs_header = None

    optional_params = ('dark_time', 'flat_time', 'camera')
    
    def __init__(self, interf_cube_path_A=None, interf_cube_path_B=None,
                 bin_A=None, bin_B=None,
                 project_header=list(),
                 cube_A_project_header = list(),
                 cube_B_project_header = list(),
                 wcs_header=list(), overwrite=False,
                 indexer=None, params=None, **kwargs):
        """
        Initialize InterferogramMerger class

        :param interf_cube_path_A: (Optional) Path to the interferogram
          cube of the camera 1

        :param interf_cube_path_B: (Optional) Path to the interferogram
          cube of the camera 2

        :param bin_A: (Optional) Binning factor of the camera A

        :param bin_B: (Optional) Binning factor of the camera B

        :param pix_size_A: (Optional) Pixel size of the camera A
        
        :param pix_size_A: (Optional) Pixel size of the camera B

        :param project_header: (Optional) header section to be added
          to each output files based on merged data (an empty list by
          default).

        :param cube_A_project_header: (Optional) header section to be
          added to each output files based on pure cube A data (an
          empty list by default).

        :param cube_B_project_header: (Optional) header section to be
          added to each output files based on pure cube B data (an
          empty list by default).

        :param wcs_header: (Optional) header section describing WCS
          that can be added to each created image files (an empty list
          by default).

        :param alignment_coeffs: (Optional) Pre-calculated alignement
          coefficients. Setting alignment_coeffs to something else
          than 'None' will avoid alignment coeffs calculation in
          :meth:`process.InterferogramMerger.find_alignment`

        :param overwrite: (Optional) If True existing FITS files will
          be overwritten (default False).

        :param indexer: (Optional) Must be a :py:class:`orb.core.Indexer`
          instance. If not None created files can be indexed by this
          instance.

        :param params: (Optional) observation parameters dictionary
          (default None).

        :param kwargs: Kwargs are :meth:`orb.core.Tools` properties.
        """
        orb.core.Tools.__init__(self, **kwargs)
        self.params = orb.core.ROParams()
        
        # manage params to pass in HDFCubes        
        self.params.update(params)
        self.params.reset('instrument', self.instrument)
        self.needed_params = orb.cube.Cube.needed_params + ('bin_cam_1', 'bin_cam_2')
        for iparam in self.needed_params:
            if iparam not in self.params:
                raise ValueError('param {} must be in params'.format(iparam))
                
        self.overwrite = overwrite
        self.indexer = indexer
        self._project_header = project_header
        self._wcs_header = wcs_header
       
        if interf_cube_path_A is not None:
            self.cube_A = orb.cube.Cube(interf_cube_path_A,
                                        instrument=self.instrument,
                                        config=self.config,
                                        params=self.params,
                                        data_prefix=self._data_prefix,
                                        camera=1)
        if interf_cube_path_B is not None:
            self.cube_B = orb.cube.Cube(interf_cube_path_B,
                                        instrument=self.instrument,
                                        config=self.config,
                                        params=self.params,
                                        data_prefix=self._data_prefix,
                                        camera=2)

        self.bin_A = self.cube_A.params.binning
        self.bin_B = self.cube_B.params.binning
        
        self.pix_size_A = self.config.PIX_SIZE_CAM1
        self.pix_size_B = self.config.PIX_SIZE_CAM2
        
        # defining zoom factor, init_dx, init_dy and init_angle
        self.zoom_factor = ((float(self.pix_size_B) * float(self.bin_B)) / 
                            (float(self.pix_size_A) * float(self.bin_A)))
        self.dx = self.config.INIT_DX / self.bin_B
        self.dy = self.config.INIT_DY / self.bin_B
            
        self.dr = self.config.INIT_ANGLE
        self.da = 0.
        self.db = 0.

        # defining rotation center
        if self.cube_B is not None:
            self.rc = [(float(self.cube_B.dimx) / 2.), 
                       (float(self.cube_B.dimy) / 2.)]

    def _get_alignment_parameters_path(self):
        """Return the path to the alignment parameters."""
        return self._data_path_hdr + "alignment_parameters.fits"

    
    def _get_modulation_ratio_path(self):
        """Return the path to the modulation ratio."""
        return self._data_path_hdr + "modulation_ratio.fits"
        
    def _get_energy_map_path(self):
        """Return the path to the energy map.

        The energy map is the mean frame from the merged cube. It is
        useful to check the alignement.
        """
        return self._data_path_hdr + "energy_map.fits"

    def _get_stray_light_vector_path(self):
        """Return the path to the stray light vector.

        The external illuminaton vector records lights coming from
        reflections over clouds, the moon or the sun.
        """
        return self._data_path_hdr + "stray_light_vector.fits"
   
    def _get_ext_illumination_vector_path(self):
        """Return the path to the external illumination vector.

        The external illuminaton vector records the external
        illumination difference between both cameras (e.g. if one
        camera get some diffused light from the sky while the other is
        well isolated). This vector is used to correct
        interferograms.
        """
        return self._data_path_hdr + "ext_illumination_vector.fits"
    
    def _get_transmission_vector_path(self, err=False):
        """Return the path to the transmission vector.

        Transmission vector is the vector used to correct
        interferograms for the variations of the sky transmission.

        :param err: (Optional) True if error vector (default False).
        """
        if not err:
            return self._data_path_hdr + "transmission_vector.fits"
        else:
            return self._data_path_hdr + "transmission_vector_err.fits"

    def _get_calibration_stars_path(self):
        """Return the path to a data file containing the merged
        interferograms of the calibrated stars"""
        
        return self._data_path_hdr + "calibration_stars.fits"

    def _get_extracted_star_spectra_path(self):
        """Return the path to a data file containing the spectra of
        the extracted stars"""
        return self._data_path_hdr + "extracted_star_spectra.fits"

    def _get_deep_frame_path(self):
        """Return the path to the deep frame.

        The energy map is the mean frame from the merged cube. It is
        useful to check the alignement.
        """
        return self._data_path_hdr + "deep_frame.fits"

    def _get_mean_image_path(self, index):
        """Return the path to the mean image

        The energy map is the mean frame from the merged cube. It is
        useful to check the alignement.
        """
        return self._data_path_hdr + "mean_image_{}.fits".format(index)

    def _get_merged_interfero_cube_path(self):
        """Return the default path to the merged interferogram frames."""
        return self._data_path_hdr + "interferogram.hdf5"

    def _get_bad_frames_vector_path(self):
        """Return the path to the bad frames vector.

        This vector is created by
        :py:meth:`process.InterferogramMerger.merge` method.
        """
        return self._data_path_hdr + "bad_frames_vector.fits"

    def _get_transformed_interfero_cube_path(self):
        """Return the default path to the transformed interferogram frames."""
        return self._data_path_hdr + "transformed_cube_B.hdf5"

    def _get_fit_results_path(self, camera):
        """Return the default path to the fit results."""
        return self._data_path_hdr + "fit_results.cam{}.hdf5".format(camera)

    def _get_star_list_path(self, camera):
        """Return the default path to the star list."""
        return self._data_path_hdr + "star_list.cam{}.hdf5".format(camera)

    def get_header(self):
        """return self.params as a fits header"""
        return orb.utils.io.dict2header(dict(self.params))
    
    def find_alignment(self, star_list_path_A, 
                       combine_first_frames=False):
        """
        Return the alignment coefficients to align the cube of the
        camera 2 on the cube of the camera 1

        :param star_list_path_A: Path to a list of star for the camera A
        
        :param combine_first_frames: If True, only the fist frames are
          combined to compute alignement parameters (default False).

        .. seealso:: py:meth:`orb.astrometry.Aligner.compute_alignment_parameters`
        """
        # High pass filtering of the frames
        HPFILTER = int(self._get_tuning_parameter('HPFILTER', 0))
        N_FRAMES = 10 # number of combined frames
        
        logging.info("Computing alignment parameters")

        # defining FOV of the camera B
        ccd_size_A = self.bin_A * self.cube_A.dimx * self.pix_size_A
        ccd_size_B = self.bin_B * self.cube_B.dimx * self.pix_size_B
        scale = self.config.FIELD_OF_VIEW_1 / ccd_size_A # absolute scale [arcsec/um]
        fov_B = scale * ccd_size_B

        logging.info("Calculated FOV of the camera B: %f arcmin"%fov_B)

        # Printing some information
        logging.info("Rotation center: %s"%str(self.rc))
        logging.info("Zoom factor: %f"%self.zoom_factor)

        # creating deep frames for cube A and B
        if not combine_first_frames:
            frameA = self.cube_A.get_deep_frame().data
            frameB = self.cube_B.get_deep_frame().data
        else:
            frameA = bn.nanmedian(self.cube_A[:,:,:N_FRAMES], axis=2)
            frameB = bn.nanmedian(self.cube_B[:,:,:N_FRAMES], axis=2)
            

        if HPFILTER: # Filter alignment frames
            frameA = orb.utils.image.high_pass_diff_image_filter(frameA, deg=1)
            frameB = orb.utils.image.high_pass_diff_image_filter(frameB, deg=1)
        
        frameA = orb.image.Image(frameA, instrument=self.instrument,
                                 config=self.config, data_prefix=self._data_prefix,
                                 params=self.params, camera=1)
        
        frameB = orb.image.Image(frameB, instrument=self.instrument,
                                 config=self.config, data_prefix=self._data_prefix,
                                 params=self.params, camera=2)
        
        result = frameA.compute_alignment_parameters(
            frameB, star_list1=star_list_path_A,
            fwhm_arc=self.config.INIT_FWHM,
            correct_distortion=False)

        [self.dx, self.dy, self.dr, self.da, self.db] = result['coeffs']
        self.rc = result['rc']
        self.zoom_factor = result['zoom_factor']

        alignment_parameters_array = np.array([self.dx, self.dy, self.dr,
                                               self.da, self.db, self.rc[0], self.rc[1],
                                               self.zoom_factor, self.config.INIT_FWHM])
        
        orb.utils.io.write_fits(
            self._get_alignment_parameters_path(),
            alignment_parameters_array,
            fits_header=self.get_header())

        if self.indexer is not None:
            self.indexer['alignment_parameters'] = self._get_alignment_parameters_path()

        return [[self.dx, self.dy, self.dr, self.da, self.db], self.rc, 
                self.zoom_factor]

    def find_laser_alignment(self, init_dx, init_dy, init_angle):
        """Brute force algorithm used to find laser frames alignment"""

        ### laser frame alignment is not precise enough to be used.
        warnings.warn('No brute force search: init alignment parameters unchanged')
        self.dx, self.dy, self.dr = init_dx, init_dy, init_angle
        self.da = 0.
        self.db = 0.
        return [[self.dx, self.dy, self.dr, self.da, self.db], self.rc, 
                self.zoom_factor]

        ### old alignment function
        CROP_COEFF = 0.5
        
        logging.info("Computing alignment parameters for laser frames")
        logging.info("Rotation center: %s"%str(self.rc))
        logging.info("Zoom factor: %f"%self.zoom_factor)

        cx = int(self.cube_A.dimx / 2)
        cy = int(self.cube_A.dimy / 2)
        dx = int(self.cube_A.dimx * CROP_COEFF / 2.)
        dy = int(self.cube_A.dimy * CROP_COEFF / 2.)
        frameA = self.cube_A[cx-dx:cx+dx+1,cy-dy:cy+dy+1,0]
        frameB = self.cube_B[cx-dx:cx+dx+1,cy-dy:cy+dy+1,0]
        deep_A = self.cube_A.get_mean_image()[cx-dx:cx+dx+1,cy-dy:cy+dy+1]
        deep_B = self.cube_B.get_mean_image()[cx-dx:cx+dx+1,cy-dy:cy+dy+1]
        frameA /= deep_A
        frameB /= deep_B

        self.dx, self.dy, self.dr = orb.utils.image.bf_laser_aligner(
            frameA, frameB, init_dx, init_dy, init_angle, self.zoom_factor)
        self.da = 0.
        self.db = 0.
        return [[self.dx, self.dy, self.dr, self.da, self.db], self.rc, 
                self.zoom_factor]

    def transform(self, interp_order=1):
        """Transform cube B given a set of alignment coefficients.

        :param interp_order: Order of interpolation. (1: linear by default)

        .. seealso:: :meth:`orb.utils.image.transform_frame`
        """
        # Init of the multiprocessing server
        job_server, ncpus = self._init_pp_server()
        ncpus_max = ncpus       

        logging.info("Transforming cube B")
        logging.info("Alignment parameters : %s"%str([self.dx, self.dy,
                                                         self.dr, self.da,
                                                         self.db]))
        logging.info("Zoom factor : %s"%str(self.zoom_factor))
        
        out_cube = orb.cube.RWHDFCube(self._get_transformed_interfero_cube_path(),
                                      shape=(self.cube_A.dimx, self.cube_A.dimy, self.cube_A.dimz),
                                      instrument=self.instrument,
                                      config=self.config,
                                      params=self.params,
                                      reset=True)
        
        progress = orb.core.ProgressBar(self.cube_A.dimz)
        
        for ik in range(0, self.cube_A.dimz, ncpus):
            # no more jobs than frames to compute
            if (ik + ncpus >= self.cube_A.dimz):
                ncpus = self.cube_A.dimz - ik

            progress.update(ik, info="loading: " + str(ik))
            framesB = self.cube_B[:,:,ik:ik+ncpus]

            # transform frames of camera B to align them with those of camera A
            jobs = [(ijob, job_server.submit(
                orb.utils.image.transform_frame, 
                args=(framesB[:,:,ijob],
                      0, self.cube_A.dimx, 
                      0, self.cube_A.dimy, 
                      [self.dx, self.dy, self.dr, self.da, self.db],
                      self.rc, self.zoom_factor,
                      interp_order),
                modules=("import logging",
                         "numpy as np", 
                         "from scipy import ndimage",
                         "import orb.cutils"))) 
                    for ijob in range(ncpus)]

            progress.update(ik, info="transforming : " + str(ik))
            for ijob, job in jobs:
                framesB[:,:,ijob] = job()
            
            out_cube[:,:,ik:ik+ncpus] = framesB
            
            
        self._close_pp_server(job_server)
        progress.end()

        del out_cube
        
        if self.indexer is not None:
            self.indexer['transformed_interfero_cube'] = self._get_transformed_interfero_cube_path()

    def merge(self, 
              add_frameB=True, smooth_vector=True,
              compute_ext_light=True,
              aperture_photometry=True):
        
        """
        Merge the cube of the camera 1 and the transformed cube of the
        camera 2.

        :param add_frameB: (Optional) Set it to False if B frame is
           too noisy to be added to the result. In this case frame B
           is used only to correct for variations of flux from the
           source (airmass, clouds ...) (Default False).
           
        :param smooth_vector: (Optional) If True smooth the obtained
           correction vector with a gaussian weighted moving average.
           Reduce the possible high frequency noise of the correction
           function. (Default True).

        :param compute_ext_light: (Optional) If True compute the
          external light vector. Make sure that there's enough 'sky'
          pixels in the frame. The vector will be deeply affected if
          the object covers the whole area (default True).

        :param aperture_photometry: (Optional) If True, flux of stars
          is computed by aperture photometry. Else, The flux is
          evaluated given the fit parameters (default True).

        .. note:: The merging process goes throught 3 steps:

           1. Compute external illumination vector: This vector
              records the external illumination difference between
              both cameras (e.g. if one camera get some diffused light
              from the sky while the other is well isolated). This
              vector is used to correct interferograms.
        
           2. Compute transmission vector: the transmission vector is
              computed from star photometry (2D gaussian or moffat
              fitting. See
              :py:meth:`orb.astrometry.Astrometry.fit_stars_in_cube`) of
              both frames from camera 1 and camera 2 (frames must
              therefore be aligned).

           3. Cube merging using for each frame (nth) of the cube the
              formula:
              
              .. math::

                 Frame_{n,M} = \\frac{Frame_{n,1}
                 -Frame_{n,2}}{transmission vector[n]}
        """
        def _get_stray_light_coeff(frameA, frameB, transmission_factor,
                                   modulation_ratio, ext_level):
            """Return the stray light coefficient. This light comes
            from reflections over clouds, the sun or the moon.
            
            :param frameA: Frame of the camera 1
            
            :param frameB: Frame of the camera 2
            
            :param transmission_factor: Correction factor for the sky
              variation of transmission

            :param modulation_ratio: The ratio of modulation between
              the two cameras. It depends on the gain and the quantum
              efficiency of the CCD.

            :param ext_level: Level of stray light (external
              illumination) in the camera B (if level is negative,
              the stray light is thus in the camera A)
            """
            ## This must be done by adding frames instead of doing all this
            ## which has exactly the same effect
            if np.any(frameB) and np.any(frameA):
                result_frame = ((((frameB / modulation_ratio) + frameA)
                                 / transmission_factor) - ext_level)
                
                stray_light_coeff = orb.utils.stats.robust_median(result_frame) / 2.
            
            else:
                stray_light_coeff = np.nan
                
            return stray_light_coeff
            
        def _create_merged_frame(frameA, frameB, transmission_factor,
                                 modulation_ratio, ext_level,
                                 add_frameB):
            """Create the merged frame given the frames of both cubes
            and the correction factor.

            :param frameA: Frame of the camera 1
            
            :param frameB: Frame of the camera 2
            
            :param transmission_factor: Correction factor for the sky
              variation of transmission

            :param modulation_ratio: The ratio of modulation between
              the two cameras. It depends on the gain and the quantum
              efficiency of the CCD.

            :param ext_level: Level of light coming from an external
              source in the camera B but not in the camera A (if level
              is negative, the stray light is thus in the camera A)
              
            :param add_frameB: If False the frame B is not added. The
              resulting frame is thus the frame A divided by the
              transmission factor (generally not recommanded)
            """
            
            # Correcting for the variation of sky transmission
            if add_frameB:
                result_frame = ((((frameB / modulation_ratio) - frameA)
                                 / transmission_factor) - ext_level)
                
                flux_frame = ((((frameB / modulation_ratio) + frameA)
                                 / transmission_factor) - ext_level)
            else: # frame B is not added
                if np.any(frameA):
                    cont_frame = ((frameB / modulation_ratio) + frameA
                                  - ext_level) / 2.
                    
                    result_frame = ((frameA - np.nanmean(cont_frame))
                                    / transmission_factor)
                else:
                    result_frame = frameA
                    cont_frame = frameA
                    
                flux_frame = cont_frame
                
            result_frame[np.nonzero(frameA == 0.)] = np.nan
            result_frame[np.nonzero(frameB == 0.)] = np.nan
            flux_frame[np.nonzero(frameA == 0.)] = np.nan
            flux_frame[np.nonzero(frameB == 0.)] = np.nan
            flux_frame[np.nonzero(np.isinf(flux_frame))] = np.nan
            
            return result_frame, flux_frame

        def get_sky_level_vector(cube):
            """Create a vector containing the sky level evaluated in
            each frame of a cube

            :param cube: Data cube
            """
            def get_sky_level(frame):
                if len(np.nonzero(frame)[0]) > 0:
                    return orb.utils.astrometry.sky_background_level(
                        frame[np.nonzero(frame)])
                else:
                    return 0.
                
            BORDER_COEFF = 0.15
            
            xmin = int(cube.dimx * BORDER_COEFF)
            xmax = cube.dimx - xmin + 1
            ymin = int(cube.dimy * BORDER_COEFF)
            ymax = cube.dimy - ymin + 1
            
            median_vector = np.empty(cube.dimz, dtype=float)
            job_server, ncpus = cube._init_pp_server()
            
            progress = orb.core.ProgressBar(cube.dimz)
            for ik in range(0, cube.dimz, ncpus):
                
                progress.update(ik, info="Creating sky level vector")
                if (ik + ncpus >= cube.dimz):
                    ncpus = cube.dimz - ik
                 
                jobs = [(ijob, job_server.submit(
                    get_sky_level, 
                    args=(cube.get_data_frame(ik+ijob)[xmin:xmax,ymin:ymax],),
                    modules=("import logging",
                             "import numpy as np",
                             'import orb.utils.astrometry')))
                        for ijob in range(ncpus)]

                for ijob, job in jobs:
                    median_vector[ik+ijob] = job()
                
            progress.end()
            cube._close_pp_server(job_server)
            
            return median_vector

        SMOOTH_DEG = 0 # number of pixels used on each side to
                       # smooth the transmission vector

        SMOOTH_RATIO_EXT = 0.05 # Ratio of the number of pixels over
                                # the vector length used to smooth the
                                # external illumination vector

        EXT_ZPD_SIZE = 0.20 # Length ratio of the ZPD over the entire
                            # cube. This is used to correct the external
                            # illumination vector

        # Minimum number of stars to compute the transmission vector
        MIN_STAR_NUMBER = float(self._get_tuning_parameter(
            'MIN_STAR_NUMBER', 5))

        # Minimum transmission coefficient for bad frames detection
        # (taken relatively to the median transmission coefficient)
        BAD_FRAME_COEFF = float(self._get_tuning_parameter(
            'BAD_FRAME_COEFF', 0.5))

        SIGMA_CUT_COEFF = 2.0 # Number of sigmas of the sigmacut
                              # (better if < 3.)

        # Coefficient on the mean reduced chi square to reject bad
        # star fit
        RED_CHISQ_COEFF = float(self._get_tuning_parameter(
            'RED_CHISQ_COEFF', 1.5))

        # FIXED MODULATION RATIO
        FIXED_MODULATION_RATIO = self._get_tuning_parameter(
            'FIXED_MODULATION_RATIO', None)
        if FIXED_MODULATION_RATIO is not None:
            FIXED_MODULATION_RATIO = float(FIXED_MODULATION_RATIO)

        # Define fit parameters depending on the type of frame
        EXTENDED_EMISSION = bool(int(
            self._get_tuning_parameter('EXTENDED_EMISSION', 0)))

        # Avoid transmission correction (useful for testing purpose)
        NO_TRANSMISSION_CORRECTION = bool(int(
            self._get_tuning_parameter('NO_TRANSMISSION_CORRECTION', 0)))
        if NO_TRANSMISSION_CORRECTION:
            warnings.warn('No transmission correction')


        local_background = True
        
        if EXTENDED_EMISSION:
            fix_fwhm = True
            optimized_modulation_ratio = False
            # Length ratio of the ZPD over the entire cube to correct
            # the transmission vector
            TRANS_ZPD_SIZE = float(
                self._get_tuning_parameter('TRANS_ZPD_SIZE', 0.1))
            
            warnings.warn(
                'Region considered as an extended emission region')
        else:
            fix_fwhm = False
            optimized_modulation_ratio = True
            # Length ratio of the ZPD over the entire cube to correct
            # the transmission vector
            TRANS_ZPD_SIZE = float(
                self._get_tuning_parameter('TRANS_ZPD_SIZE', 0.01))
        
        if aperture_photometry:
            logging.info('Star flux evaluated by aperture photometry')
            photometry_type = 'aperture_flux'
            no_fit = True
        else:
            logging.info('Star flux evaluated from fit parameters')
            photometry_type = 'flux'
            no_fit = False

        ## COMPUTING STARS PHOTOMETRY #############################
        logging.info("Computing stars photometry")
        
        star_list_A, fwhm_A = self.cube_A.detect_stars(path=self._get_star_list_path(1))
        star_list_B, fwhm_B = self.cube_B.detect_stars(path=self._get_star_list_path(2))
            
        fwhm_arc_A = np.nanmedian(star_list_A['fwhm_arc'])
        logging.info(
            'mean FWHM of the stars in camera 1: {} arc-seconds'.format(
                fwhm_arc_A))
        fwhm_arc_B = np.nanmedian(star_list_B['fwhm_arc'])
        logging.info(
            'mean FWHM of the stars in camera 2: {} arc-seconds'.format(
                fwhm_arc_B))

        astrom_A = self.cube_A.fit_stars_in_cube(
            star_list_A, fix_fwhm=False, fix_height=False, no_fit=False,
            fix_aperture_fwhm_pix=fwhm_A * 1.5, multi_fit=True)
        orb.utils.io.save_dflist(astrom_A, self._get_fit_results_path(1))

        astrom_B = self.cube_B.fit_stars_in_cube(
            star_list_A, fix_fwhm=False, fix_height=False, no_fit=False,
            fix_aperture_fwhm_pix=fwhm_B * 1.5, multi_fit=True)
        orb.utils.io.save_dflist(astrom_B, self._get_fit_results_path(2))
        
        astrom_A = orb.utils.io.load_dflist(self._get_fit_results_path(1))
        astrom_B = orb.utils.io.load_dflist(self._get_fit_results_path(2))

        def get_photometry_array(photom, key):
            _photom = list()
            _len = None
            for ik in photom:
                if not ik.empty:
                    _photom.append(ik[key].values)
                    _len = len(_photom[-1])
                else:
                    _photom.append(None)
            if _len is None: raise StandardError('photometry dataframe is empty')
            for ik in range(len(_photom)):
                if _photom[ik] is None:
                    _photom[ik] = list([np.nan]) * _len
                    
            return np.array(_photom).T
        
        photom_A = get_photometry_array(astrom_A, key=photometry_type)
        photom_B = get_photometry_array(astrom_B, key=photometry_type)
        
        ## MODULATION RATIO #######################################
        # Calculating the mean modulation ratio (to correct for
        # difference of camera gain and transmission of the optical
        # path)

        # Optimization routine
        def photom_diff(modulation_ratio, photom_A, photom_B, zpd_min, zpd_max):
            return orb.utils.stats.robust_median((photom_A * modulation_ratio
                                                  - photom_B)**2.)
        
        # use EXT_ZPD_SIZE to remove ZPD from MODULATION RATIO calculation
        ext_zpd_min = self.params.zpd_index - int(EXT_ZPD_SIZE * self.params.step_nb / 2.)
        if ext_zpd_min < 0: ext_zpd_min = 0
        ext_zpd_max = self.params.zpd_index + int(EXT_ZPD_SIZE * self.params.step_nb / 2.) + 1
        if ext_zpd_max > self.cube_A.dimz:
            ext_zpd_max = self.cube_A.dimz - 1

        photom_A_nozpd = np.copy(photom_A)
        photom_A_nozpd[:,ext_zpd_min:ext_zpd_max] = np.nan
        photom_B_nozpd = np.copy(photom_B)
        photom_B_nozpd[:,ext_zpd_min:ext_zpd_max] = np.nan
        
        if optimized_modulation_ratio and FIXED_MODULATION_RATIO is None:
            modulation_ratio = optimize.fmin_powell(
                photom_diff, [1.0],
                args=(photom_A_nozpd, photom_B_nozpd, ext_zpd_min, ext_zpd_max),
                ftol=1e-3, xtol=1e-3, disp=False)


            flux_error = np.nanmean(
                photom_A * modulation_ratio - photom_B, axis=1)
            flux_sum = np.nanmean(
                photom_A * modulation_ratio + photom_B, axis=1)
         
            flux_error_ratio = orb.utils.stats.robust_mean(
                np.abs(flux_error/flux_sum),
                weights=flux_sum/np.nansum(flux_sum))
  
            logging.info(
                "Optimized modulation ratio: %f (std: %f)"%(
                    modulation_ratio, modulation_ratio * flux_error_ratio))
        
        elif FIXED_MODULATION_RATIO is None:
            # If the optimization does not work we try a more robust
            # but sometimes less precise method
            modulation_ratios = list()
            for index in range(photom_A_nozpd.shape[1]):
                index_mod = list()
                for istar in range(photom_A_nozpd.shape[0]):
                    if (photom_A[istar,index] != 0.
                        and not np.isnan(photom_B_nozpd[istar,index])
                        and not np.isnan(photom_A_nozpd[istar,index])):
                        index_mod.append(photom_B_nozpd[istar,index]
                                         / photom_A_nozpd[istar,index])
                if len(index_mod) > 0:
                    modulation_ratios.append(orb.utils.stats.robust_mean(
                        orb.utils.stats.sigmacut(
                            index_mod, sigma=SIGMA_CUT_COEFF)))

            modulation_ratio = orb.utils.stats.robust_mean(
                orb.utils.stats.sigmacut(
                    modulation_ratios, sigma=SIGMA_CUT_COEFF))

            modulation_ratio_std = orb.utils.stats.robust_std(
                orb.utils.stats.sigmacut(
                    modulation_ratios, sigma=SIGMA_CUT_COEFF))

            logging.info(
                "Modulation ratio: %f (std: %f)"%(
                    modulation_ratio, modulation_ratio_std))
        else:
            modulation_ratio = FIXED_MODULATION_RATIO
            logging.info(
                "Fixed modulation ratio: %f"%(
                    modulation_ratio))

        orb.utils.io.write_fits(
            self._get_modulation_ratio_path(), 
            np.array([modulation_ratio]),
            fits_header=self.get_header())

        if self.indexer is not None:
            self.indexer['modulation_ratio'] = self._get_modulation_ratio_path()

        # PHOTOMETRY ON MERGED FRAMES #############################
        astrom_merged = self.cube_B.fit_stars_in_cube(
            star_list_A, local_background=local_background,
            fix_aperture_fwhm_pix=fwhm_B * 1.5, multi_fit=True,
            add_cube=[self.cube_A, modulation_ratio],
            no_fit=no_fit)
        orb.utils.io.save_dflist(astrom_merged, self._get_fit_results_path('M'))

        astrom_merged = orb.utils.io.load_dflist(self._get_fit_results_path('M'))

        photom_merged = get_photometry_array(astrom_merged, key=photometry_type)
        photom_merged_err = get_photometry_array(astrom_merged, key=photometry_type + '_err')

        ## TRANSMISSION VECTOR ####################################
        logging.info("Computing transmission vector")

        transmission_vector_list = list()
        red_chisq_list = list()
        trans_err_list = list()
        
        # normalization of the merged photometry vector
        chisq_A = get_photometry_array(astrom_A, key='reduced-chi-square')
        for istar in range(star_list_A.shape[0]):
            if not np.all(np.isnan(photom_merged)):
                trans = np.copy(photom_merged[istar,:])
                trans_err = np.copy(photom_merged_err[istar,:])
                trans_mean = orb.utils.stats.robust_mean(
                    orb.utils.stats.sigmacut(trans))
                trans /= trans_mean
                trans_err /= trans_mean
                transmission_vector_list.append(trans)
                red_chisq = orb.utils.stats.robust_mean(
                    orb.utils.stats.sigmacut(chisq_A[istar, :]))
                
                trans_err_list.append(trans_err)
                red_chisq_list.append(red_chisq)

        # reject stars with a bad reduced-chi-square
        mean_red_chisq = orb.utils.stats.robust_mean(
            orb.utils.stats.sigmacut(red_chisq_list))
        temp_list_trans = list()
        temp_list_trans_err = list()
        for istar in range(len(transmission_vector_list)):
            if red_chisq_list[istar] < mean_red_chisq * RED_CHISQ_COEFF:
                temp_list_trans.append(transmission_vector_list[istar])
                temp_list_trans_err.append(trans_err_list[istar])
        transmission_vector_list = temp_list_trans
        transmission_vector_list_err = temp_list_trans_err

        if len(transmission_vector_list) <  MIN_STAR_NUMBER:
            raise StandardError("Too much stars have been rejected. The transmission vector cannot be computed !")

        logging.info(
            "Transmission vector will be computed using %d stars"%len(
                transmission_vector_list))
        transmission_vector_list = np.array(transmission_vector_list)
        transmission_vector_list_err = np.array(transmission_vector_list_err)
        
        # Create transmission vector
        transmission_vector = np.empty((self.cube_A.dimz), dtype=float)
        transmission_vector_err = np.empty_like(transmission_vector)
        transmission_vector.fill(np.nan)
        transmission_vector_err.fill(np.nan)
        
        for ik in range(self.cube_A.dimz):
            trans_ik = transmission_vector_list[:,ik]
            trans_err_ik = transmission_vector_list_err[:,ik]
            
            if len(np.nonzero(trans_ik)[0]) > 0:
                if len(trans_ik) >= MIN_STAR_NUMBER:
                    trans_cut, trans_cut_index = orb.utils.stats.sigmacut(
                        trans_ik, sigma=SIGMA_CUT_COEFF, return_index_list=True)
                    transmission_vector[ik] = orb.utils.stats.robust_mean(
                        trans_cut)
                    trans_cut_err = trans_err_ik[trans_cut_index]
                    transmission_vector_err[ik] = np.sqrt(
                        orb.utils.stats.robust_mean(trans_cut_err**2.))
        
        # Transmission is corrected for bad values
        transmission_vector = orb.utils.vector.correct_vector(
            transmission_vector, bad_value=0., polyfit=True, deg=3)

        # correct vector for ZPD
        if TRANS_ZPD_SIZE > 0:
            trans_zpd_min = (self.params.zpd_index
                             - int((TRANS_ZPD_SIZE * self.params.step_nb)/2.))
            trans_zpd_max = (self.params.zpd_index
                             + int((TRANS_ZPD_SIZE * self.params.step_nb)/2.) + 1)

            if trans_zpd_min < 0: trans_zpd_min = 0
            if trans_zpd_max > self.cube_A.dimz:
                trans_zpd_max = self.cube_A.dimz - 1

            transmission_vector[trans_zpd_min:trans_zpd_max] = 0.
        
            transmission_vector = orb.utils.vector.correct_vector(
                transmission_vector, bad_value=0., polyfit=True, deg=3)
            
        # Transmission vector smoothing
        if smooth_vector:
            if SMOOTH_DEG > 0:
                transmission_vector = orb.utils.vector.smooth(transmission_vector,
                                                       deg=SMOOTH_DEG)

        # Normalization of the star transmission vector to 1.5% clip
        nz = np.nonzero(transmission_vector)
        max_trans = orb.cutils.part_value(transmission_vector[nz], 0.985)
        transmission_vector[nz] /= max_trans

        if NO_TRANSMISSION_CORRECTION:
            transmission_vector.fill(1.)
            transmission_vector_err.fill(0.)
        
        # Save transmission vector
        orb.utils.io.write_fits(
            self._get_transmission_vector_path(), 
            transmission_vector.reshape((transmission_vector.shape[0],1)),
            fits_header=self.get_header())
        if self.indexer is not None:
            self.indexer[
                'transmission_vector'] = self._get_transmission_vector_path()

        orb.utils.io.write_fits(
            self._get_transmission_vector_path(err=True), 
            transmission_vector_err.reshape(
                (transmission_vector_err.shape[0],1)),
            fits_header=self.get_header())
        if self.indexer is not None:
            self.indexer['transmission_vector_err'] = (
                self._get_transmission_vector_path(err=True))
        
        # Create BAD FRAMES VECTOR from transmission vector
        bad_frames_vector = np.zeros((self.cube_A.dimz), dtype=int)
        bad_frames_vector[
            np.nonzero(transmission_vector <= BAD_FRAME_COEFF)] = 1
        if np.any(bad_frames_vector):
            logging.info("Detected bad transmission frames: %s"%str(
                np.nonzero(bad_frames_vector)[0]))
        orb.utils.io.write_fits(
            self._get_bad_frames_vector_path(), 
            bad_frames_vector,
            fits_header=self.get_header())
        
        if self.indexer is not None:
            self.indexer[
                'bad_frames_vector'] = self._get_bad_frames_vector_path()

        ## EXTERNAL ILLUMINATION VECTOR ##########################
        # Computing the external illumination level (if some light
        # enters in one of the cameras).
        
        # WARNING : This vector will be correct if there's enough
        # 'sky' pixels.

        if compute_ext_light:
            logging.info("Computing external illumination vector")
            median_frame_vector_A = get_sky_level_vector(self.cube_A)
            median_frame_vector_B = get_sky_level_vector(self.cube_B)

            ext_level_vector = ((median_frame_vector_B / modulation_ratio)
                                - median_frame_vector_A)

            # correct vector for nan values and zeros
            ext_level_vector = orb.utils.vector.correct_vector(
                ext_level_vector, bad_value=0., polyfit=True, deg=3)

            
            # correct vector for ZPD
            ext_level_vector[ext_zpd_min:ext_zpd_max] = 0.
            ext_level_vector = orb.utils.vector.correct_vector(
                ext_level_vector, bad_value=0., polyfit=True, deg=3)
            
            # vector smoothing
            if SMOOTH_RATIO_EXT > 0.:
                ext_level_vector = orb.utils.vector.smooth(
                    ext_level_vector, 
                    deg=int(ext_level_vector.shape[0] * SMOOTH_RATIO_EXT))

        else:
            warnings.warn(
                "External illumination vector computation skipped")
            ext_level_vector = np.zeros(self.cube_A.dimz, dtype=float)
     
        # Save external illumination vector
        orb.utils.io.write_fits(
            self._get_ext_illumination_vector_path(), 
            ext_level_vector.reshape((ext_level_vector.shape[0],1)),
            fits_header=self.get_header())
        if self.indexer is not None:
            self.indexer['ext_illumination_vector'] = (
                self._get_ext_illumination_vector_path())
        

        ## MERGE FRAMES ###########################################
        logging.info("Merging cubes")
        
        flux_frame = np.zeros((self.cube_A.dimx, self.cube_A.dimy),
                              dtype=float)
        flux_vector = np.zeros(self.cube_A.dimz, dtype=float)

        out_cube = orb.cube.RWHDFCube(self._get_merged_interfero_cube_path(),
                                      shape=(self.cube_A.dimx,
                                             self.cube_A.dimy,
                                             self.cube_A.dimz),
                                      instrument=self.instrument,
                                      config=self.config,
                                      params=self.params,
                                      reset=True)

        
        progress = orb.core.ProgressBar(self.cube_A.dimz)
        header = self.get_header()

        NFRAMES = 30
        
        for ik in range(0, self.cube_A.dimz, NFRAMES):
            # no more jobs than frames to compute
            if (ik + NFRAMES >= self.cube_A.dimz):
                NFRAMES = self.cube_A.dimz - ik

            progress.update(int(ik), info="loading: " + str(ik))
            frames_A = self.cube_A[:,:,ik:ik+NFRAMES]
            frames_B = self.cube_B[:,:,ik:ik+NFRAMES]

            progress.update(int(ik), info="merging: " + str(ik))
            result_frames, flux_frames = _create_merged_frame(
                frames_A, frames_B,
                transmission_vector[ik:ik+NFRAMES],
                modulation_ratio,
                ext_level_vector[ik:ik+NFRAMES],
                add_frameB)                

            flux_frame += np.nansum(flux_frames, axis=2)
            
            for ijob in range(NFRAMES):
                flux_vector[ik + ijob] = orb.utils.stats.robust_median(
                    flux_frames[:,:,ijob])

            progress.update(int(ik), info="writing: " + str(ik))
            out_cube[:,:,ik:ik+NFRAMES] = result_frames

            
        progress.end()

        if self.indexer is not None:
            self.indexer['merged_interfero_cube'] = (
                self._get_merged_interfero_cube_path())
        

        # ENERGY MAP & DEEP FRAME
        # Before being added deep frames of cam1 and cam2 must be
        # scaled to keep the same amount of photons/ADU from one
        # reduction to another. It is here scaled relatively to frameB
        # because on SpIOMM cam2 has kept the same gain and cam1 has
        # not.

        if add_frameB:
            # remove pedestal of flux vector, 1.5% clip
            stray_light_vector = flux_vector - orb.cutils.part_value(
                flux_vector[np.nonzero(~np.isnan(flux_vector))], 0.015)
       
            # Save stray light vector
            orb.utils.io.write_fits(
                self._get_stray_light_vector_path(), 
                stray_light_vector,
                fits_header=self.get_header())
        
            if self.indexer is not None:
                self.indexer['stray_light_vector'] = (
                    self._get_stray_light_vector_path())
                
            logging.info('Mean flux of stray light: {} ADU'.format(
                np.nanmean(stray_light_vector)))
        else:
            stray_light_vector = np.zeros_like(flux_vector)
       
        merged_cube = orb.cube.Cube(self._get_merged_interfero_cube_path(),
                                    instrument=self.instrument,
                                    config=self.config,
                                    params=self.params)
        deep_frame = (flux_frame - np.nansum(stray_light_vector))
        out_cube.set_deep_frame(deep_frame)
    
        
        orb.utils.io.write_fits(self._get_deep_frame_path(), deep_frame,
                                fits_header=self.get_header())

        if self.indexer is not None:
            self.indexer['deep_frame'] = self._get_deep_frame_path()


        # SAVE CALIBRATION STARS INTERFEROGRAMS
        logging.info("Saving corrected calibration stars interferograms")
        calib_stars_interf_list = list()
        for istar in range(star_list_A.shape[0]):
            calib_stars_interf_list.append(
                (((photom_B[istar,:]/modulation_ratio) - photom_A[istar,:])
                 / transmission_vector) - ext_level_vector)
        calibration_stars_path = self._get_calibration_stars_path()
        orb.utils.io.write_fits(calibration_stars_path,
                                np.array(calib_stars_interf_list),
                                fits_header=self.get_header())

        if self.indexer is not None:
            self.indexer['calibration_stars'] = calibration_stars_path
            
        logging.info("Cubes merged")
        del out_cube


##################################################
#### CLASS CosmicRayDetector #####################
##################################################

class CosmicRayDetector(InterferogramMerger):
    """Class created for cosmic ray detection using both cubes (cam1
    and cam2).

    .. warning:: This class has been designed for SITELLE's
      data. Cosmic ray detection for SpIOMM must use the RawData
      class.
    """
        
    def _get_cr_map_cube_path(self, camera_number):
        """Return the default path to a HDF5 cube of the cosmic rays."""
        return self._data_path_hdr + "cr_map.cam{}.hdf5".format(camera_number)

    def create_cosmic_ray_maps(self, alignment_vector_path_1,
                               star_list_path, fwhm_pix):
        """Create cosmic ray maps for both cubes.

        :param alignment_vector_path_1: Alignement vector of the camera 1.
        """
        def detect_crs_in_frame(frameA, frameB, frameM, frameref, params,
                                star_list, fwhm_pix, dx, dy):

            warnings.simplefilter('ignore', RuntimeWarning)
            
            PREDETECT_COEFF = 15
            PREDETECT_NEI_BOX_SIZE = 3 # must be odd
            PREDETECT_NEI_COEFF = 2.
            DETECT_BOX_SIZE = 7 # must be odd
            DETECT_COEFF = 2.7
            DETECT_NEI_BOX_SIZE = 3 # must be odd
            DETECT_NEI_COEFF = 2.

            framediv = frameM / frameref

            
            PREDETECT_COEFF = np.nanstd(orb.utils.stats.sigmacut(
                framediv)) * PREDETECT_COEFF + 1.
            

            ## predetection in frameM
            fcr_mapdiv = orb.cutils.check_cosmic_rays_neighbourhood(
                framediv, (framediv > PREDETECT_COEFF).astype(np.uint8),
                PREDETECT_NEI_BOX_SIZE, PREDETECT_NEI_COEFF)
            #print len(np.nonzero(fcr_mapdiv)[0])

            ## detection in frameM
            fcr_mapM = orb.cutils.detect_cosmic_rays(
                frameM, np.nonzero(fcr_mapdiv),
                DETECT_BOX_SIZE, DETECT_COEFF)
            #print len(np.nonzero(fcr_mapM)[0])

            # neighbourhood check
            fcr_mapM = orb.cutils.check_cosmic_rays_neighbourhood(
                frameM, fcr_mapM,
                DETECT_NEI_BOX_SIZE, DETECT_NEI_COEFF)
            #print 'M', len(np.nonzero(fcr_mapM)[0])

            ## detect crs in frame A
            fcr_mapA = orb.cutils.detect_cosmic_rays(
                frameA, np.nonzero(fcr_mapM),
                             DETECT_BOX_SIZE, DETECT_COEFF)
            # neighbourhood check
            fcr_mapA = orb.cutils.check_cosmic_rays_neighbourhood(
                frameA, fcr_mapA,
                DETECT_NEI_BOX_SIZE, DETECT_NEI_COEFF)
            #print 'A', len(np.nonzero(fcr_mapA)[0])

            ## detect crs in frame B
            cr_listM = np.nonzero(fcr_mapM)

            # transforming list to frame B coordinates
            cr_listBx = list()
            cr_listBy = list()
            for i in range(len(cr_listM[0])):
                ixb, iyb = orb.cutils.transform_A_to_B(
                    cr_listM[0][i], cr_listM[1][i], *params)
                ixb = int(round(ixb)); iyb = int(round(iyb))
                if (ixb >= 0 and ixb < frameB.shape[0]
                    and iyb >= 0 and iyb < frameB.shape[1]):
                    cr_listBx.append(ixb)
                    cr_listBy.append(iyb)
            cr_listB = (cr_listBx, cr_listBy)


            fcr_mapB = orb.cutils.detect_cosmic_rays(frameB, cr_listB,
                                                    DETECT_BOX_SIZE, DETECT_COEFF)

            # remove cosmic rays detected near a star
            star_list[:,0] += dx
            star_list[:,1] += dy

            for istar in range(star_list.shape[0]):
        
                xmin, xmax, ymin, ymax = orb.utils.image.get_box_coords(
                    star_list[istar, 0], star_list[istar, 1],
                    int(3.*fwhm_pix)+1,
                    0, fcr_mapA.shape[0], 0, fcr_mapA.shape[1])
                fcr_mapA[xmin:xmax, ymin:ymax] = 0

                ixb, iyb = orb.cutils.transform_A_to_B(
                    star_list[istar, 0], star_list[istar, 1], *params)
                xmin, xmax, ymin, ymax = orb.utils.image.get_box_coords(
                    ixb, iyb, int(3.*fwhm_pix)+1,
                    0, fcr_mapB.shape[0], 0, fcr_mapB.shape[1])
                fcr_mapB[xmin:xmax, ymin:ymax] = 0

            # neighbourhood check
            fcr_mapB = orb.cutils.check_cosmic_rays_neighbourhood(
                frameB, fcr_mapB,
                DETECT_NEI_BOX_SIZE, DETECT_NEI_COEFF)
            #print 'B', len(np.nonzero(fcr_mapB)[0])

            return fcr_mapA, fcr_mapB
           

        BIAS = 100000
        MAX_CRS = 3 # Max nb of cosmic rays in one pixels
        
        alignment_vector_1 = orb.utils.io.read_fits(alignment_vector_path_1)
        star_list = orb.utils.astrometry.load_star_list(star_list_path)
        
        # job_server, ncpus = self._init_pp_server()
        # ncpus_max = ncpus

        # out_cubeA = orb.cube.RWHDFCube(self._get_cr_map_cube_path(1),
        #                                shape=(self.cube_A.dimx, self.cube_A.dimy, self.cube_A.dimz),
        #                                instrument=self.instrument,
        #                                config=self.config,
        #                                params=self.params,
        #                                reset=True)
        # out_cubeB = orb.cube.RWHDFCube(self._get_cr_map_cube_path(2),
        #                                shape=(self.cube_B.dimx, self.cube_B.dimy, self.cube_B.dimz),
        #                                instrument=self.instrument,
        #                                config=self.config,
        #                                params=self.params,
        #                                reset=True)
        
        # progress = orb.core.ProgressBar(int(self.cube_A.dimz/ncpus_max))
        # for ik in range(0, self.cube_A.dimz, ncpus):
        #     progress.update(int(ik/ncpus_max), info="starting ({})".format(ik))
            
        #     # no more jobs than frames to compute
        #     if (ik + ncpus >= self.cube_A.dimz):
        #         ncpus = self.cube_A.dimz - ik

        #     progress.update(int(ik/ncpus_max), info="loading ({})".format(ik))
        #     framesA = self.cube_A[:,:,ik:ik+ncpus].astype(np.float32)
        #     framesB_init = self.cube_B[:,:,ik:ik+ncpus].astype(np.float32)
        #     framesB = np.empty_like(framesA)
        #     cr_mapA = np.empty_like(framesA, dtype=bool)
        #     cr_mapB = np.empty_like(framesA, dtype=bool)
            
        #     # transform frames of camera B to align them with those of camera A
        #     progress.update(int(ik/ncpus_max), info="aligning ({})".format(ik))
            
        #     jobs = [(ijob, job_server.submit(
        #         orb.utils.image.transform_frame, 
        #         args=(framesB_init[:,:,ijob],
        #               0, self.cube_A.dimx, 
        #               0, self.cube_A.dimy, 
        #               [self.dx - alignment_vector_1[ik+ijob, 0],
        #                self.dy - alignment_vector_1[ik+ijob, 1],
        #                self.dr, self.da, self.db],
        #               self.rc, self.zoom_factor, 1),
        #         modules=("import logging",
        #                  "import numpy as np", 
        #                  "from scipy import ndimage",
        #                  "import orb.cutils"))) 
        #             for ijob in range(ncpus)]

        #     for ijob, job in jobs:
        #         framesB[:,:,ijob] = job()
            
                
        #     framesM = framesA + framesB # ok for SITELLE, not for SpIOMM
        #     framesM -= np.nanmean(np.nanmean(framesM, axis=0), axis=0)
        #     framesM += BIAS
        #     frameref = bn.nanmedian(framesM, axis=2)

        #     # detect CRS
        #     progress.update(int(ik/ncpus_max), info="detecting ({})".format(ik))
            
        #     jobs = [(ijob, job_server.submit(
        #         detect_crs_in_frame, 
        #         args=(framesA[:,:,ijob],
        #               framesB_init[:,:,ijob],
        #               framesM[:,:,ijob],
        #               frameref,
        #               [self.dx - alignment_vector_1[ik+ijob, 0],
        #                self.dy - alignment_vector_1[ik+ijob, 1],
        #                self.dr, self.da, self.db,
        #                self.rc[0], self.rc[1],
        #                self.zoom_factor, self.zoom_factor],
        #               star_list, fwhm_pix,
        #               alignment_vector_1[ik+ijob, 0],
        #               alignment_vector_1[ik+ijob, 1]),
        #         modules=("import logging",
        #                  "import numpy as np", 
        #                  "import orb.cutils",
        #                  "import orb.utils.image",
        #                  "import orb.utils.stats",
        #                  "import warnings"))) 
        #             for ijob in range(ncpus)]
            
        #     for ijob, job in jobs:
        #         cr_mapA[:,:,ijob], cr_mapB[:,:,ijob] = job()

        #     progress.update(int(ik/ncpus_max), info="writing ({})".format(ik))
        #     out_cubeA[:,:,ik:ik+cr_mapA.shape[2]] = cr_mapA.astype(bool)
        #     out_cubeB[:,:,ik:ik+cr_mapB.shape[2]] = cr_mapB.astype(bool)
            
                
        # self._close_pp_server(job_server)   
        # progress.end()

        # del out_cubeA
        # del out_cubeB


        if self.indexer is not None:
            self.indexer['cr_map_cube_1'] = (
                self._get_cr_map_cube_path(1))

        if self.indexer is not None:
            self.indexer['cr_map_cube_2'] = (
                self._get_cr_map_cube_path(2))


        out_cubeA = orb.cube.RWHDFCube(self._get_cr_map_cube_path(1), camera=1)
        out_cubeB = orb.cube.RWHDFCube(self._get_cr_map_cube_path(2), camera=2)
        
        # check to remove over detected pixels (stars)
        cr_mapA_deep = out_cubeA.get_deep_frame().data
        cr_mapB_deep = out_cubeB.get_deep_frame().data

        badpixA = np.nonzero(cr_mapA_deep > MAX_CRS)
        badpixB = np.nonzero(cr_mapB_deep > MAX_CRS)

        if len(badpixA[0]) > 0:
            for i in range(len(badpixA[0])):
                out_cubeA[badpixA[0][i], badpixA[1][i], :] = 0
            logging.info('{} pixels with too much detections cleaned in camera 1'.format(
                len(badpixA[0])))
        if len(badpixB[0]) > 0:
            for i in range(len(badpixB[0])):
                out_cubeB[badpixB[0][i], badpixB[1][i], :] = 0
            logging.info('{} pixels with too much detections cleaned in camera 2'.format(
                len(badpixB[0])))
        
        
        logging.info('Final number of contaminated pixels in camera 1: {}'.format(
            np.sum(out_cubeA.get_deep_frame(recompute=True).data)))
        logging.info('Final number of contaminated pixels in camera 2: {}'.format(
            np.sum(out_cubeB.get_deep_frame(recompute=True).data)))
        
        del out_cubeA
        del out_cubeB
      

 
##################################################
#### CLASS Spectrum ##############################
##################################################
class Spectrum(orb.cube.SpectralCube):
    """
    ORBS spectrum processing class.

    This class is used to correct the spectrum computed by the
    Interferogram class. 
    
    :param spectrum_cube_path: Path to the spectrum cube
    """

    def _get_stars_coords_path(self):
        """Return path to the list of stars coordinates used to correct WCS"""
        return self._data_path_hdr + "stars_coords"

    def _get_modulation_efficiency_map_path(self, imag=False):
        """Return path to the modulation efficiency map.

        :param imag: (Optional) True for imaginary part of the
          modulation efficiency (default False)."""
        if not imag:
            return self._data_path_hdr + "modulation_efficiency_map.real.fits"
        else:
            return self._data_path_hdr + "modulation_efficiency_map.imag.fits"

    def _get_calibrated_spectrum_cube_path(self):
        """Return the default path to a calibrated spectral cube."""
        return self._data_path_hdr + "calibrated_spectrum.hdf5"

    def _update_hdr_wcs(self, hdr, wcs_hdr):
        """Update a header with WCS parameters

        :param hdr: A pyfits.Header() instance

        :param wcs_header: A pyfits.Header() instance containing the
          new WCS parameters.
        """
        hdr.extend(wcs_hdr, strip=True,
                   update=True, end=True)

        # delete unused keywords created by pywcs
        if 'RESTFRQ' in hdr:
            del hdr['RESTFRQ']
        if 'RESTWAV' in hdr:
            del hdr['RESTWAV']
        if 'LONPOLE' in hdr:
            del hdr['LONPOLE']
        if 'LATPOLE' in hdr:
            del hdr['LATPOLE']
        return hdr
        
    def calibrate(self,
                  correct_wcs=None,
                  flux_calibration_vector=None,
                  flux_calibration_coeff=None,
                  wavenumber=False, standard_header=None,
                  spectral_calibration=True, filter_correction=True):
        
        """Calibrate spectrum cube: correct for filter transmission
        function, correct WCS parameters and flux calibration.

        :param spectral_calibration (Optional): If True, the ouput
          spectral cube will be calibrated in
          wavelength/wavenumber. Note that in this case the spectrum
          must be interpolated. Else no spectral calibration is done:
          the channel position of a given wavelength/wavenumber
          changes with its position in the field (default True).

        :param filter_correction: (Optional) If True spectra are
          corrected for the filter transmission.
                 
        :param correct_wcs: (Optional) Must be a pywcs.WCS
          instance. If not None header of the corrected spectrum
          cube is updated with the new WCS.

        :param flux_calibration_vector: (Optional) Tuple (cm1_axis,
          vector). Must be a vector calibrated in erg/cm^2/A as the
          one given by
          :py:meth:`process.Spectrum.get_flux_calibration_vector`. Each
          spectrum will be multiplied by this vector to be flux
          calibrated (default None).

        :param flux_calibration_coeff: (Optional) If given flux
          calibration vector is adjusted to fit the mean calibration
          coeff. If no flux calibration vector is given, this flux
          calibration coefficient is used as a flat flux calibration
          vector (default None).
    
        :param wavenumber: (Optional) If True, the spectrum is
          considered to be in wavenumber. If False it is considered to
          be in wavelength (default False).    

        :param standard_header: (Optional) Header for the standard
          star used for flux calibration. This header part will be
          appended to the FITS header.

        .. note:: The filter file used must have two colums separated
          by a space character. The first column contains the
          wavelength axis in nm. The second column contains the
          transmission coefficients. Comments are preceded with a #.
          Filter edges can be specified using the keywords :
          FILTER_MIN and FILTER_MAX::

            ## ORBS filter file 
            # Author: Thomas Martin <thomas.martin.1@ulaval.ca>
            # Filter name : SpIOMM_R
            # Wavelength in nm | Transmission percentage
            # FILTER_MIN 648
            # FILTER_MAX 678
            1000 0.001201585284
            999.7999878 0.009733387269
            999.5999756 -0.0004460749624
            999.4000244 0.01378122438
            999.2000122 0.002538740868

        """
        
        def _calibrate_spectrum_column(spectrum_col, filter_function,
                                       filter_min, filter_max,
                                       flux_calibration_function,
                                       calibration_laser_col, nm_laser,
                                       wavenumber,
                                       spectral_calibration,
                                       base_axis_correction_coeff,
                                       output_sz_coeff, params):
            """
            
            """            
            INTERP_POWER = 30 * output_sz_coeff
            ZP_LENGTH = orb.utils.fft.next_power_of_two(
                spectrum_col.shape[1] * INTERP_POWER)

            spectrum_col[np.nonzero(np.isnan(spectrum_col))] = 0.

            result_col = np.empty((spectrum_col.shape[0],
                                   spectrum_col.shape[1]
                                   * output_sz_coeff))
            result_col.fill(np.nan)

            # converting to flux (ADU/s)
            spectrum_col /= params['exposure_time'] * spectrum_col.shape[1]

            if wavenumber:
                axis_proj = orb.utils.spectrum.create_cm1_axis(
                    spectrum_col.shape[1] * output_sz_coeff,
                    params['step'],
                    params['order'],
                    corr=base_axis_correction_coeff).astype(float)
            else:
                axis_proj = orb.utils.spectrum.create_nm_axis(
                    spectrum_col.shape[1] * output_sz_coeff, params['step'],
                    params['order'], corr=base_axis_correction_coeff).astype(float)

            for icol in range(spectrum_col.shape[0]):

                corr = calibration_laser_col[icol]/nm_laser
                if np.isnan(corr):
                    result_col[icol,:] = np.nan
                    continue

                # converting to ADU/s/A
                axis_corr_cm1_lowres = orb.utils.spectrum.create_cm1_axis(
                    spectrum_col.shape[1],
                    params['step'],
                    params['order'],
                    corr=corr).astype(float)
                
                ispectrum = spectrum_col[icol,:]

                ispectrum = orb.utils.photometry.convert_cm1_flux2fluxdensity(
                    ispectrum, axis_corr_cm1_lowres)

                # pure fft interpolation of the input spectrum
                # (i.e. perfect interpolation as long as the imaginary
                # part is given)
                if ((spectral_calibration or not wavenumber)
                    or output_sz_coeff != 1):
                    interf_complex = np.fft.ifft(ispectrum)

                    zp_interf = np.zeros(ZP_LENGTH, dtype=complex)
                    center = interf_complex.shape[0]/2
                    zp_interf[:center] = interf_complex[:center]
                    zp_interf[
                        -center-int(interf_complex.shape[0]&1):] = interf_complex[
                        -center-int(interf_complex.shape[0]&1):]
                    interf_complex = np.copy(zp_interf)
                    spectrum_highres = np.fft.fft(interf_complex).real
                else:
                    spectrum_highres = np.copy(ispectrum)

                
                # remember : output from 'compute spectrum' step is
                # always in cm-1
                if wavenumber:
                    axis_corr = orb.utils.spectrum.create_cm1_axis(
                        spectrum_highres.shape[0],
                        params['step'],
                        params['order'],
                        corr=corr).astype(float)
                else:
                    axis_corr = orb.utils.spectrum.create_nm_axis_ireg(
                        spectrum_highres.shape[0],
                        params['step'],
                        params['order'],
                        corr=corr).astype(float)

                # filter function and flux calibration function are
                # projected
                if filter_function is not None:
                    filter_corr = filter_function(axis_corr)
                    # filter correction is made only between filter
                    # edges to conserve the filter shape
                    if wavenumber:
                        filter_min_pix, filter_max_pix = orb.cutils.fast_w2pix(
                            np.array([filter_min, filter_max], dtype=float),
                            axis_corr[0], axis_corr[1]-axis_corr[0])
                    else:
                        filter_min_pix, filter_max_pix = orb.utils.spectrum.nm2pix(
                            axis_corr, np.array([filter_min, filter_max], dtype=float))
                    # filter_min_pix and max_pix are not used anymore
                    # because filter function is prepared before. Can
                    # be reused to put nans.
                    spectrum_highres /= filter_corr
                    
                if flux_calibration_function is not None:
                    flux_corr = flux_calibration_function(axis_corr)
                    spectrum_highres *= flux_corr

                # replacing nans by zeros before interpolation
                nans = np.nonzero(np.isnan(spectrum_highres))
                spectrum_highres[nans] = 0.

                # spectrum projection onto its output axis (if output
                # in nm or calibrated)
                if spectral_calibration or not wavenumber:
                    result_col[icol,:] = (
                        orb.utils.vector.interpolate_axis(
                            spectrum_highres, axis_proj, 1,
                            old_axis=axis_corr))
                    
                else:
                    result_col[icol,:] = spectrum_highres

            return result_col.real

        OUTPUT_SZ_COEFF = 1

        
        if filter_correction:
            raise StandardError("Filter correction is not stable please don't use it")

        # get correction coeff at the center of the field (required to
        # project the spectral cube at the center of the field instead
        # of projecting it on the interferometer axis)
        if spectral_calibration:
            base_axis_correction_coeff = self.get_calibration_coeff_map()[
                int(self.dimx/2), int(self.dimy/2)]
        else:
            base_axis_correction_coeff = 1.
        
        
        # Get filter parameters
        FILTER_STEP_NB = 4000
        FILTER_RANGE_THRESHOLD = 0.97
        filter_vector, filter_min_pix, filter_max_pix = (
            orb.core.FilterFile(self.params.filter_name).get_filter_function(
                self.params.step, self.params.order,
                FILTER_STEP_NB, wavenumber=wavenumber,
                corr=base_axis_correction_coeff))
        filter_min_pix = np.nanmin(np.arange(FILTER_STEP_NB)[
            filter_vector >  FILTER_RANGE_THRESHOLD * np.nanmax(filter_vector)])
        filter_max_pix = np.nanmax(np.arange(FILTER_STEP_NB)[
            filter_vector >  FILTER_RANGE_THRESHOLD * np.nanmax(filter_vector)])
        
        if filter_min_pix < 0: filter_min_pix = 0
        if filter_max_pix > FILTER_STEP_NB: filter_max_pix = FILTER_STEP_NB - 1

        if not wavenumber:
            filter_axis = orb.utils.spectrum.create_nm_axis(
                FILTER_STEP_NB, self.params.step, self.params.order)
        else:
            filter_axis = orb.utils.spectrum.create_cm1_axis(
                FILTER_STEP_NB, self.params.step, self.params.order)

        # prepare filter vector (smooth edges)
        filter_vector[:filter_min_pix] = 1.
        filter_vector[filter_max_pix:] = 1.
        filter_vector[
            filter_min_pix:filter_max_pix] /= np.nanmean(filter_vector[
            filter_min_pix:filter_max_pix])
        smooth_coeff = np.zeros_like(filter_vector)
        smooth_coeff[int(filter_min_pix)] = 1.
        smooth_coeff[int(filter_max_pix)] = 1.
        smooth_coeff = orb.utils.vector.smooth(
            smooth_coeff, deg=0.1*FILTER_STEP_NB, kind='cos_conv')
        smooth_coeff /= np.nanmax(smooth_coeff)
        filter_vector_smooth = orb.utils.vector.smooth(
            filter_vector, deg=0.1*FILTER_STEP_NB, kind='gaussian')
        
        filter_vector = ((1. - smooth_coeff) * filter_vector
                         + smooth_coeff * filter_vector_smooth)
        filter_function = interpolate.UnivariateSpline(
            filter_axis, filter_vector, s=0, k=1)
        filter_min = filter_axis[filter_min_pix]
        filter_max = filter_axis[filter_max_pix]

        # Get modulation efficiency
        modulation_efficiency = orb.core.FilterFile(
            self.params.filter_name).get_modulation_efficiency()

        logging.info('Modulation efficiency: {}'.format(
            modulation_efficiency))
        
        # Get flux calibration function
        if flux_calibration_vector[0] is not None:
            (flux_calibration_axis,
             flux_calibration_vector) = flux_calibration_vector
                       
            if not wavenumber:
                flux_calibration_axis = orb.utils.spectrum.cm12nm(
                    flux_calibration_axis)[::-1]
                flux_calibration_vector = flux_calibration_vector[::-1]
                
            flux_calibration_function = interpolate.UnivariateSpline(
                flux_calibration_axis, flux_calibration_vector, s=0, k=1)
    
            # adjust flux calibration vector with flux calibration coeff
            if flux_calibration_coeff is not None:
                mean_flux_calib_vector = orb.utils.photometry.compute_mean_star_flux(
                    flux_calibration_function(filter_axis.astype(float)),
                    filter_vector)
                logging.info('Mean flux calib vector before adjustment with flux calibration coeff: {} erg/cm2/ADU'.format(mean_flux_calib_vector))
                # ME must be taken into account only when using the
                # flux calibration coeff derived from std images
                flux_calibration_coeff /= modulation_efficiency
                logging.info('Flux calibration coeff (corrected for modulation efficiency {}): {} erg/cm2/ADU'.format(modulation_efficiency, flux_calibration_coeff))
                flux_calibration_vector /=  mean_flux_calib_vector
                flux_calibration_vector *= flux_calibration_coeff
                flux_calibration_function = interpolate.UnivariateSpline(
                    flux_calibration_axis, flux_calibration_vector, s=0, k=3)

        # Calibrate with coeff derived from std images
        elif flux_calibration_coeff is not None:
            # ME must be taken into account only when using the flux
            # calibration coeff derived from std images
            flux_calibration_coeff /= modulation_efficiency
            logging.info('Flux calibration coeff (corrected for modulation efficiency {}): {} erg/cm2/ADU'.format(modulation_efficiency, flux_calibration_coeff))
            flux_calibration_function = interpolate.UnivariateSpline(
                filter_axis,
                np.ones_like(filter_axis, dtype=float)
                * flux_calibration_coeff, s=0, k=3)
        else:
            flux_calibration_function = None

        # Get FFT parameters
        header = self.get_cube_header()
        if 'APODIZ' in header:
            apodization_function = header['APODIZ']
        else:
            apodization_function = 'None'
                            
        # set filter function to None if no filter correction
        if not filter_correction:
            filter_function = None
            filter_min = None
            filter_max = None

        logging.info("Calibrating spectra")
        
        if filter_correction:
            logging.info("Filter correction")
        else:
            warnings.warn("No filter correction")

        if spectral_calibration:
            logging.info("Spectral calibration")
        else:
            warnings.warn("No spectral calibration")
        
        if flux_calibration_vector is not None:
            logging.info("Flux calibration")
        else:
            warnings.warn("No flux calibration")
            
        if correct_wcs is not None:
            logging.info("WCS correction")
        else:
            warnings.warn("No WCS correction")

        # control energy in the imaginary part ratio
        ## deep_frame_spectrum = self.get_mean_image()
        ## imag_energy = orb.utils.stats.sigmacut(
        ##     deep_frame_spectrum.imag/deep_frame_spectrum.real, sigma=2.5)
        ## logging.info("Median energy ratio imaginary/real: {:.2f} [std {:.3f}] %".format(np.nanmedian(imag_energy)*100., np.nanstd(imag_energy)*100.))
               
        out_cube = OutHDFQuadCube(
            self._get_calibrated_spectrum_cube_path(),
            (self.dimx, self.dimy, self.dimz * OUTPUT_SZ_COEFF),
            self.config.QUAD_NB,
            reset=True)
        
        # Init of the multiprocessing server    
        for iquad in range(0, self.config.QUAD_NB):
            (x_min, x_max, 
             y_min, y_max) = self.get_quadrant_dims(iquad)
            
            iquad_data = np.empty((x_max - x_min,
                                   y_max - y_min,
                                   self.dimz * OUTPUT_SZ_COEFF),
                                  dtype=float)
            iquad_data[:,:,:self.dimz] = self.get_data(x_min, x_max, 
                                                       y_min, y_max, 
                                                       0, self.dimz)
            iquad_calibration_laser_map = self.get_calibration_laser_map()[
                x_min:x_max, y_min:y_max]
            job_server, ncpus = self._init_pp_server()
            ncpus_max = int(ncpus)
            progress = orb.core.ProgressBar(int((x_max-x_min)/ncpus_max))

            # ii = 0 ## np.int((x_max-x_min)/2) #### DEBUG
            # testit = _calibrate_spectrum_column(iquad_data[ii,:,:self.dimz],
            #                                     filter_function,
            #                                     filter_min, filter_max,
            #                                     flux_calibration_function,
            #                                     exposure_time,
            #                                     iquad_calibration_laser_map[ii,:],
            #                                     nm_laser, step, order, wavenumber,
            #                                     spectral_calibration,
            #                                     base_axis_correction_coeff,
            #                                     OUTPUT_SZ_COEFF)

            for ii in range(0, x_max-x_min, ncpus):
                progress.update(int(ii/ncpus_max), 
                                info="quad : %d, column : %d"%(iquad + 1, ii))
                
                # no more jobs than frames to compute
                if (ii + ncpus >= x_max-x_min):
                    ncpus = x_max - x_min - ii

                # correct spectrum columns
                jobs = [(ijob, job_server.submit(
                    _calibrate_spectrum_column, 
                    args=(
                        iquad_data[ii+ijob,:,:self.dimz], 
                        filter_function,
                        filter_min, filter_max,
                        flux_calibration_function,
                        iquad_calibration_laser_map[ii+ijob,:],
                        self.config.CALIB_NM_LASER,
                        wavenumber,
                        spectral_calibration,
                        base_axis_correction_coeff,
                        OUTPUT_SZ_COEFF,
                        self.params.convert()),
                    modules=("import logging",
                             "import numpy as np",
                             "import orb.utils.spectrum",
                             "import orb.utils.vector",
                             "import orb.utils.fft",
                             "import orb.utils.photometry"))) 
                        for ijob in range(ncpus)]

                for ijob, job in jobs:
                    # corrected data comes in place of original data
                    iquad_data[ii+ijob,:,:] = job()

            self._close_pp_server(job_server)
            progress.end()

            # save data
            logging.info('Writing quad {}/{} to disk'.format(
                iquad+1, self.config.QUAD_NB))
            write_start_time = time.time()
            out_cube.write_quad(iquad, data=iquad_data.real)
            ### out_cube.write_quad(iquad, data=iquad_data, force_float32=False, force_complex64=True) ### DEBUG
            logging.info('Quad {}/{} written in {:.2f} s'.format(
                iquad+1, self.config.QUAD_NB, time.time() - write_start_time))
            
        
        ### update header
        if wavenumber:
            axis = orb.utils.spectrum.create_nm_axis(
                self.dimz, self.params.step, self.params.order,
                corr=base_axis_correction_coeff)
        else:
            axis = orb.utils.spectrum.create_cm1_axis(
                self.dimz, self.params.step, self.params.order,
                corr=base_axis_correction_coeff)

        # update standard header
        if standard_header is not None:
            if flux_calibration_vector is not None:
                standard_header.append((
                    'FLAMBDA',
                    np.nanmean(flux_calibration_vector),
                    'mean energy/ADU [erg/cm^2/A/ADU]'))

        hdr = self.get_header()
        
        hdr.append(('AXISCORR',
                    base_axis_correction_coeff,
                    'Spectral axis correction coeff'))
        
        new_hdr = pyfits.PrimaryHDU(
            np.empty((self.dimx, self.dimy),
                     dtype=float).transpose()).header
        new_hdr.extend(hdr, strip=True, update=True, end=True)
        if correct_wcs is not None:
            hdr = self._update_hdr_wcs(
                new_hdr, correct_wcs.to_header(relax=True))
        else:
            hdr = new_hdr

        ## hdr.set('PC1_1', after='CROTA2')
        ## hdr.set('PC1_2', after='PC1_1')
        ## hdr.set('PC2_1', after='PC1_2')
        ## hdr.set('PC2_2', after='PC2_1')
        ## hdr.set('WCSAXES', before='CTYPE1')
        
        # Create Standard header
        if standard_header is not None:
            hdr.extend(standard_header, strip=False, update=False, end=True)
                    
        # Create flux header
        flux_hdr = list()
        flux_hdr.append(('COMMENT','',''))
        flux_hdr.append(('COMMENT','Flux',''))
        flux_hdr.append(('COMMENT','----',''))
        flux_hdr.append(('COMMENT','',''))
        if flux_calibration_vector is not None:
            flux_hdr.append(('BUNIT','FLUX','Flux unit [erg/cm^2/s/A]'))
        else:
            flux_hdr.append(('BUNIT','UNCALIB','Uncalibrated Flux'))
            
        hdr.extend(flux_hdr, strip=False, update=False, end=True)

        out_cube.append_header(hdr)
    
        if self.indexer is not None:
            self.indexer['calibrated_spectrum_cube'] = (
                self._get_calibrated_spectrum_cube_path())


    def get_flux_calibration_coeff(self,
                                   std_image_cube_path_1,
                                   std_image_cube_path_2,
                                   std_name,
                                   std_pos_1, std_pos_2,
                                   fwhm_pix):
        """Return flux calibration coefficient in [erg/cm2/s/A]/ADU
        from a set of images.
    
        :param std_spectrum_path_1: Path to the standard image list
    
        :param std_spectrum_path_2: Path to the standard image list

        :param std_name: Name of the standard

        :param std_pos_1: X,Y Position of the standard star.in camera 1

        :param std_pos_2: X,Y Position of the standard star.in camera 2

        :param fwhm_pix: Rough FWHM size in pixels.

        .. note:: This calibration coefficient must be used with non
          filter-corrected data

        .. warning:: This calibration coeff cannot take Modulation
            Efficiency into account. A more representative calibration
            coefficient would be divided by the modulation
            efficiency. This coefficient must thus be used on spectral
            data already normalized to an ME of 100%
        """
        def _get_std_position(im, box_size, x, y):
            (x_min, x_max,
             y_min, y_max) = orb.utils.image.get_box_coords(
                x, y, box_size,
                0, im.shape[0],
                0, im.shape[1])
            box = im[x_min:x_max,
                     y_min:y_max]

            x, y = np.unravel_index(
                np.argmax(box), box.shape)
            x += x_min
            y += y_min
            return x, y

        def _get_photometry(im, x, y, fwhm_pix, exp_time):
            photom = orb.utils.astrometry.multi_aperture_photometry(
                im, [[x, y]], fwhm_pix)[0]

            std_flux = photom['aperture_flux'] / exp_time # ADU/s
            std_flux_err = photom['aperture_flux_err'] / exp_time # ADU/s
            return std_flux, std_flux_err
        
        
        BOX_SIZE = int(8 * fwhm_pix) + 1
        STEP_NB = 500
        ERROR_FLUX_COEFF = 1.5
        
        logging.info('Computing flux calibration coeff')
        logging.info('Standard Name: %s'%std_name) 
        logging.info('Standard image cube 1 path:{}'.format(
            std_image_cube_path_1))
        logging.info('Standard image cube 2 path:{}'.format(
            std_image_cube_path_2))

        ## Compute standard flux in erg/cm2/s/A

        # compute correction coeff from angle at center of the frame
        corr = self.get_calibration_coeff_map()[int(self.dimx/2), int(self.dimy/2)]
        
        # Get standard spectrum in erg/cm^2/s/A
        std = Standard(std_name,
                       instrument=self.instrument,
                       ncpus=self.ncpus)
        th_spectrum_axis, th_spectrum = std.get_spectrum(
            self.params.step, self.params.order, STEP_NB,
            wavenumber=False, corr=corr)

        # get filter function
        (filter_function,
         filter_min_pix, filter_max_pix) = (
            orb.core.FilterFile(self.params.filter_name).get_filter_function(
                self.params.step, self.params.order, STEP_NB,
                wavenumber=False, corr=corr))


        raise NotImplementedError('should be reconsidered, calculation must not be done here and user should use orb.photometry directly')
        # convert it to erg/cm2/s by summing all the photons
        std_th_flux = th_spectrum * filter_function / np.nanmax(filter_function)
        std_th_flux = np.diff(th_spectrum_axis) * 10. * std_th_flux[:-1]
        std_th_flux = np.nansum(std_th_flux)

        ## std_th_flux = orb.utils.photometry.compute_mean_star_flux(
        ##     th_spectrum, filter_function)


        # compute simulated flux
        std_sim_flux = std.compute_star_flux_in_frame(
            self.params.step, self.params.order, self.params.filter_name, 1, corr=corr)

        logging.info('Simulated star flux in one camera: {} ADU/s'.format(
            std_sim_flux))

        ## Compute photometry in real images
        std_x1, std_y1 = std_pos_1
        std_x2, std_y2 = std_pos_2
        
        cube1 = orb.cube.Cube(std_image_cube_path_1)
        std_hdr = cube1.get_frame_header(0)
        cube2 = orb.cube.Cube(std_image_cube_path_2)
    
        if 'EXPTIME' in std_hdr:
            std_exp_time = std_hdr['EXPTIME']
        else: raise StandardError('Integration time (EXPTIME) keyword must be present in the header of the standard image {}'.format(cube1.image_list[0]))
        logging.info('Standard integration time: {}s'.format(std_exp_time))
        
        if cube1.dimz == 1:
            warnings.warn('standard image list contains only one file')
            master_im1 = np.copy(cube1[:,:,0])
            master_im2 = np.copy(cube2[:,:,0])
        else:
            #raise StandardError('standard images must be realigned first (to be implemented)')
            cube1_r = orb.utils.astrometry.realign_images(cube1[:,:,:])
            cube2_r = orb.utils.astrometry.realign_images(cube2[:,:,:])
            
            master_im1 = orb.utils.image.pp_create_master_frame(
                cube1_r[:,:,:], ncpus=self.config.NCPUS)
            master_im2 = orb.utils.image.pp_create_master_frame(
                cube2_r[:,:,:], ncpus=self.config.NCPUS)

            #orb.utils.io.write_fits('master1.fits', master_im1, overwrite=True)
            #orb.utils.io.write_fits('master2.fits', master_im2, overwrite=True)
            

        # find star around std_x1, std_y1:
        std_x1, std_y1 =_get_std_position(
            master_im1, BOX_SIZE, std_x1, std_y1)
        # find star around std_x2, std_y2:
        std_x2, std_y2 =_get_std_position(
            master_im2, BOX_SIZE, std_x2, std_y2)
    
        # photometry
        std_flux1, std_flux_err1 = _get_photometry(
            master_im1, std_x1, std_y1, fwhm_pix, std_exp_time)
        logging.info('Aperture flux of the standard star in camera 1 is {} [+/-{}] ADU/s'.format(std_flux1, std_flux_err1))

        std_flux2, std_flux_err2 = _get_photometry(
            master_im2, std_x2, std_y2, fwhm_pix, std_exp_time)
        logging.info('Aperture flux of the standard star in camera 2 is {} [+/-{}] ADU/s'.format(std_flux2, std_flux_err2))

        logging.info('Ratio of real flux/ simulated flux for camera 1: {}'.format(
            std_flux1 / std_sim_flux))
        logging.info('Ratio of real flux/ simulated flux for camera 2: {}'.format(
            std_flux2 / std_sim_flux))


        ## New test compares sum of fluxes in both cameras to twice the simulated flux in one camera without modulation
        flux_ratio = 2*std_sim_flux/(std_flux1+std_flux2)
        if (flux_ratio > ERROR_FLUX_COEFF):
            raise StandardError('Measured flux is too low compared to simulated flux. There must be a problem. Check standard image files.')
 
        coeff = std_th_flux / (std_flux1 + std_flux2) # erg/cm2/ADU
        
        logging.info('Flux calibration coeff: {} ergs/cm2/ADU'.format(coeff))

        return coeff
        
    def get_flux_calibration_vector(self, std_spectrum_path, std_name):
        """
        Return a flux calibration vector in [erg/cm^2]/ADU on the range
        corresponding to the observation parameters of the spectrum to
        be calibrated.

        The spectrum to be calibrated can then be simply multiplied by
        the returned vector to be converted in [erg/cm^2]

        :param std_spectrum_path: Path to the standard spectrum

        :param std_name: Name of the standard

        .. note:: Standard spectrum must not be corrected for the
          filter.

        .. warning:: This flux calibration vector is computed from a
          spectrum which has a given modulation efficiency. It must be
          normalized to a modulation efficiency of 100% via the
          calibration coefficient computed from standard images.
        """
        logging.info('Computing flux calibration vector')
        logging.info('Standard Name: %s'%std_name)
        logging.info('Standard spectrum path: %s'%std_spectrum_path)

        # Get real spectrum
        re_spectrum_data, hdr = orb.utils.io.read_fits(
            std_spectrum_path, return_header=True)
        re_spectrum = re_spectrum_data[:,0]

        if len(re_spectrum.shape) > 1:
            raise StandardError(
                'Bad standard shape. Standard spectrum must be a 1D vector !')
            
        # Standard observation parameters
        std_step = hdr['STEP']
        std_order = hdr['ORDER']
        std_exp_time = hdr['EXPTIME']
        std_corr = hdr['AXCORR0']
        
        # Get standard spectrum in erg/cm^2/s/A
        std = Standard(std_name, instrument=self.instrument,
                       ncpus=self.ncpus)
        th_spectrum_axis, th_spectrum = std.get_spectrum(
            std_step, std_order,
            re_spectrum.shape[0], wavenumber=True,
            corr=std_corr)

        # get filter bandpass
        (filter_function,
         filter_min_pix, filter_max_pix) = (
            orb.core.FilterFile(self.params.filter_name).get_filter_function(
                std_step, std_order, re_spectrum.shape[0],
                corr=std_corr, wavenumber=True))
        
        return orb.utils.photometry.compute_flux_calibration_vector(
            re_spectrum, th_spectrum,
            std_step, std_order, std_exp_time,
            std_corr, 
            filter_min_pix, filter_max_pix)
        
