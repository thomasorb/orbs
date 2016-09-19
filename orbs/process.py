#!/usr/bin/python
# *-* coding: utf-8 *-*
# Author: Thomas Martin <thomas.martin.1@ulaval.ca>
# File: process.py

## Copyright (c) 2010-2015 Thomas Martin <thomas.martin.1@ulaval.ca>
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

from orb.core import Tools, Cube, ProgressBar, Standard, PhaseFile, FilterFile
from orb.core import HDFCube, OutHDFCube, OutHDFQuadCube
import orb.utils.fft
import orb.utils.filters
import orb.utils.spectrum
import orb.utils.image
import orb.utils.misc
import orb.utils.photometry

import orb.astrometry
import orb.utils.astrometry
import orb.constants
from orb.astrometry import Astrometry, Aligner
import bottleneck as bn

import os
import numpy as np
import math
from scipy import optimize, interpolate

import astropy.io.fits as pyfits
import warnings
import time

##################################################
#### CLASS RawData ###############################
##################################################

class RawData(HDFCube):
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

    def _get_alignment_vector_header(self, err=False):
        """Return the header of the alignment vector.

        :param err: (Optional) If True, the error vector header is
          returned (default False).
        """
        if not err:
            return (self._get_basic_header('Alignment vector')
                    + self._project_header)
        else:
            return (self._get_basic_header('Alignment vector error')
                    + self._project_header) 

    def _get_cr_map_cube_path(self):
        """Return the default path to a HDF5 cube of the cosmic rays."""
        return self._data_path_hdr + "cr_map.hdf5"

    def _get_cr_map_frame_header(self):
        """Return the header of the cosmic ray map."""
        return (self._get_basic_header('Cosmic ray map')
                + self._project_header)
    
    def _get_hp_map_path(self):
        """Return the default path to the hot pixels map."""
        return self._data_path_hdr + "hp_map.fits"

    def _get_deep_frame_path(self):
        """Return the default path to the deep frame."""
        return self._data_path_hdr + "deep_frame.fits"

    def _get_deep_frame_header(self):
        """Return the header of the deep frame."""
        return (self._get_basic_header('Deep frame')
                + self._project_header
                + self._get_basic_frame_header(self.dimx, self.dimy))

    def _get_energy_map_path(self):
        """Return the default path to the energy map."""
        return self._data_path_hdr + "energy_map.fits"

    def _get_energy_map_header(self):
        """Return the header of the energy map."""
        return (self._get_basic_header('Energy Map')
                + self._project_header
                + self._get_basic_frame_header(self.dimx, self.dimy))


    def _get_interfero_cube_path(self):
        """Return the default path to the interferogram HDF5 cube."""
        return self._data_path_hdr + "interferogram.hdf5"

    def _get_interfero_frame_header(self):
        """Return the header of an interferogram frame"""
        return (self._get_basic_header('Interferogram frame')
                + self._project_header
                + self._get_basic_frame_header(self.dimx, self.dimy))


    def _get_master_path(self, kind):
        """Return the default path to a master frame.

        :param kind: Kind of master frame (e.g. : 'bias', 'dark',
          'flat')
        """
        return self._data_path_hdr + "master_%s.fits"%kind

    def _get_master_header(self, kind):
        """Return the header of a master frame.
        
        :param kind: Kind of master frame (e.g. : 'bias', 'dark',
          'flat')
        """
        return (self._get_basic_header('Master %s'%kind)
                + self._project_header
                + self._get_basic_frame_header(self.dimx, self.dimy))

    def _load_bias(self, bias_list_path, return_temperature=False,
                   combine='average', reject='avsigclip'):
        """Return a master bias.

        :param bias_list_path: Path to the list of bias frames

        :param return_temperature: If True return also the mean
          temperature of the bias frames. Note that the header of the
          files must have the keyword 'CCD_TEMP'. Return None if the
          temperatures could not be read.

        :param reject: (Optional) Rejection operation. Can be
          'sigclip', 'minmax', 'avsigclip' or None (default
          'avsigclip'). See
          :py:meth:`orb.utils.image.create_master_frame`.
        
        :param combine: (Optional) Combining operation. Can be
          'average' or 'median' (default 'average'). See
          :py:meth:`orb.utils.image.create_master_frame`.

        .. note:: Bias images are resized if x and y dimensions of the
            flat images are not equal to the cube dimensions.

        .. seealso:: :py:meth:`orb.utils.image.create_master_frame`
        """
        bias_cube = Cube(bias_list_path,
                         config_file_name=self.config_file_name,
                         ncpus=self.ncpus)
        self._print_msg('Creating Master Bias')
        # try to read temperatures in the header of each bias and
        # return the mean
        if return_temperature:
            temp_list = list()
            error = False
            for ibias in range(bias_cube.dimz):
                try:
                    temp_list.append(
                        bias_cube.get_frame_header(ibias)['CCD-TEMP'])
                except:
                    error = True
                
            if len(temp_list) == 0:
                self._print_warning("Bias temperatures could not be read. Check presence of the keyword 'CCD-TEMP'")
                bias_temp = None
            else:
                bias_temp = np.mean(temp_list)
                self._print_msg(
                    "Master bias mean temperature : %f C"%bias_temp)
                if error:
                    self._print_warning("Some of the bias temperatures could not be read. Check presence of the keyword 'CCD-TEMP'")

        # Create master bias
        # Resizing if nescessary (Warning this must be avoided)
        if self.is_same_2D_size(bias_cube):
            bias_frames = bias_cube.get_all_data()   
        else:
            self._print_warning("Bad bias cube dimensions : resizing data")
            bias_frames = bias_cube.get_resized_data(self.dimx, self.dimy)

        if not self.BIG_DATA:
            master_bias = orb.utils.image.create_master_frame(
                bias_frames, combine=combine, reject=reject)
        else:
            master_bias = orb.utils.image.pp_create_master_frame(
                bias_frames, combine=combine, reject=reject)
        
            
        self.write_fits(self._get_master_path('bias'),
                        master_bias, overwrite=True,
                        fits_header=self._get_master_header('Bias'))
        
        if return_temperature:
            return master_bias, bias_temp
        else:
            return master_bias

    def _load_dark(self, dark_list_path, return_temperature=False,
                   combine='average', reject='avsigclip'):
        """Return a master dark.
            
        :param dark_list_path: Path to the list of dark frames

        :param return_temperature: If True return also the mean
          temperature of the dark frames. Note that the header of the
          files must have the keyword 'CCD_TEMP'. Return None if the
          temperatures could not be read.

        :param reject: (Optional) Rejection operation. Can be
          'sigclip', 'minmax', 'avsigclip' or None (default
          'avsigclip'). See
          :py:meth:`orb.utils.image.create_master_frame`.
        
        :param combine: (Optional) Combining operation. Can be
          'average' or 'median' (default 'average'). See
          :py:meth:`orb.utils.image.create_master_frame`.

        .. note:: Dark images are resized if x and y dimensions of the
            flat images are not equal to the cube dimensions.

        .. seealso:: :py:meth:`orb.utils.image.create_master_frame`
        """
        dark_cube = Cube(dark_list_path,
                         config_file_name=self.config_file_name,
                         ncpus=self.ncpus)
        self._print_msg('Creating Master Dark')

        # try to read temperatures in the header of each dark and
        # return the mean
        if return_temperature:
            temp_list = list()
            error = False
            for idark in range(dark_cube.dimz):
                try:
                    temp_list.append(
                        dark_cube.get_frame_header(idark)['CCD-TEMP'])
                except:
                    error = True
                
            if len(temp_list) == 0:
                self._print_warning("Dark temperatures could not be read. Check presence of the keyword 'CCD-TEMP'")
                dark_temp = None
            else:
                dark_temp = np.mean(temp_list)
                self._print_msg(
                    "Master dark mean temperature : %f C"%dark_temp)
                if error:
                    self._print_warning("Some of the dark temperatures could not be read. Check presence of the keyword 'CCD-TEMP'")

        # Resizing operation if nescessary (this must be avoided)
        if self.is_same_2D_size(dark_cube):
            dark_frames = dark_cube.get_all_data().astype(float)
            
        else:
            self._print_warning("Bad dark cube dimensions : resizing data. To avoid resizing hot pixels, the master dark frame is replaced by its median value.")
            #dark_frames = dark_cube.get_resized_data(self.dimx, self.dimy)
            dark_frames_badsize = dark_cube.get_all_data()
            dark_frames = np.empty((self.dimx, self.dimy), dtype=float)
            dark_median = list()
            for iframe in range(dark_cube.dimz):
                dark_median.append(orb.utils.stats.robust_median(
                    dark_frames_badsize[:,:,iframe]))
            dark_frames.fill(np.median(dark_median))
                
                
                
                
            
            

        # Create master dark
        if not self.BIG_DATA:
            master_dark = orb.utils.image.create_master_frame(
                dark_frames, combine=combine, reject=reject)
        else:
            master_dark = orb.utils.image.pp_create_master_frame(
                dark_frames, combine=combine, reject=reject)

        # Write master dark
        self.write_fits(self._get_master_path('dark'),
                        master_dark, overwrite=True,
                        fits_header=self._get_master_header('Dark'))
            
        if return_temperature:
            return master_dark, dark_temp
        else:
            return master_dark

    def _load_flat(self, flat_list_path, combine='average', reject='avsigclip',
                   smooth_deg=0):
        """Return a master flat.

        :param flat_list_path: Path to the list of flat frames
        
        :param reject: (Optional) Rejection operation. Can be
          'sigclip', 'minmax', 'avsigclip' or None (default
          'avsigclip'). See
          :py:meth:`orb.utils.image.create_master_frame`.
        
        :param combine: (Optional) Combining operation. Can be
          'average' or 'median' (default 'average'). See
          :py:meth:`orb.utils.image.create_master_frame`.

        :param smooth_deg: (Optional) If > 0 smooth the master flat (help in
          removing possible fringe pattern) (default 0).

        .. note:: Flat images are resized if the x and y dimensions of
            the flat images are not equal to the cube dimensions.

        .. seealso:: :py:meth:`orb.utils.image.create_master_frame`
        """
        flat_cube = Cube(flat_list_path,
                         config_file_name=self.config_file_name,
                         ncpus=self.ncpus)
        self._print_msg('Creating Master Flat')
        
        # resizing if nescessary
        if self.is_same_2D_size(flat_cube):   
            flat_frames = flat_cube.get_all_data().astype(float)
            
        else:
            self._print_warning("Bad flat cube dimensions : resizing data")
            flat_frames = flat_cube.get_resized_data(self.dimx, self.dimy)

        # flat frames are all normalized before beeing combined to
        # account for flux changes
        for i in range(flat_frames.shape[2]):
            flat_frames[:,:,i] = (
                flat_frames[:,:,i] / bn.nanmedian(flat_frames[:,:,i]))

        # create master flat
        if not self.BIG_DATA:
            master_flat = orb.utils.image.create_master_frame(
                flat_frames, combine=combine, reject=reject)
        else:
            master_flat = orb.utils.image.pp_create_master_frame(
                flat_frames, combine=combine, reject=reject)

        if smooth_deg > 0:
            master_flat = orb.utils.image.low_pass_image_filter(master_flat,
                                                                smooth_deg)
            self._print_warning('Master flat smoothed (Degree: %d)'%smooth_deg)


        # write master flat
        self.write_fits(self._get_master_path('flat'),
                        master_flat, overwrite=True,
                        fits_header=self._get_master_header('Flat'))

        return master_flat
            
    def _load_alignment_vector(self, alignment_vector_path):
        """Load the alignment vector.
          
        :param alignment_vector_path: Path to the alignment vector file.
        """
        self._print_msg("Loading alignment vector")
        alignment_vector = self.read_fits(alignment_vector_path, no_error=True)
        if (alignment_vector is not None):
            if (alignment_vector.shape[0] == self.dimz):
                self._print_msg("Alignment vector loaded")
                return alignment_vector
            else:
                self._print_error("Alignment vector dimensions are not compatible")
                return None
        else:
            self._print_warning("Alignment vector not loaded")
            return None


    def create_alignment_vector(self, star_list_path, init_fwhm_arc,
                                fov, profile_name='gaussian',
                                moffat_beta=2.1, min_coeff=0.3,
                                readout_noise=10., dark_current_level=0.):
        """Create the alignment vector used to compute the
          interferogram from the raw images.

        :param star_list_path: Path to a list of star coordinates that
          will be used to calculates the displacement vector. Please
          refer to :meth:`orb.utils.astrometry.load_star_list` for more
          information about a list of stars.

        :param init_fwhm_arc: Initial guess for the FWHM in arcsec

        :param fov: Field of View along x axis in arcmin

        :param profile_name: (Optional) PSF profile for star
          fitting. Can be 'moffat' or 'gaussian'. See:
          :py:class:`orb.astrometry.Astrometry` (default 'gaussian').

        :param moffat_beta: (Optional) Beta parameter to use for
          moffat PSF (default 3.5).

        :param min_coeff: (Optional) The minimum proportion of stars
            correctly fitted to assume a good enough calculated
            disalignment (default 0.3).
            
        :param readout_noise: (Optional) Readout noise in ADU/pixel
          (can be computed from bias frames: std(master_bias_frame))
          (default 10.)
    
        :param dark_current_level: (Optional) Dark current level in
          ADU/pixel (can be computed from dark frames:
          median(master_dark_frame)) (default 0.)

        .. note:: The alignement vector contains the calculated
           disalignment for each image along x and y axes to the first
           image.
        """
        self._print_msg("Creating alignment vector", color=True)
        # init Astrometry class
        astrom = Astrometry(self, init_fwhm_arc,
                            fov, profile_name=profile_name,
                            moffat_beta=moffat_beta,
                            data_prefix=self._data_prefix,
                            readout_noise=readout_noise,
                            dark_current_level=dark_current_level,
                            tuning_parameters=self._tuning_parameters,
                            config_file_name=self.config_file_name,
                            ncpus=self.ncpus)
        astrom.load_star_list(star_list_path)

        # get alignment vectors
        (alignment_vector_x,
         alignment_vector_y,
         alignment_error) = astrom.get_alignment_vectors(fit_cube=True)

        self.alignment_vector = np.array([alignment_vector_x,
                                          alignment_vector_y]).T
        
        alignment_vector_path = self._get_alignment_vector_path()
        alignment_err_vector_path = self._get_alignment_vector_path(err=True)
        self.write_fits(alignment_vector_path, self.alignment_vector, 
                        fits_header=
                        self._get_alignment_vector_header(),
                        overwrite=self.overwrite)
        if self.indexer is not None:
            self.indexer['alignment_vector'] = alignment_vector_path
        self.write_fits(alignment_err_vector_path, np.array(alignment_error), 
                        fits_header=
                        self._get_alignment_vector_header(err=True),
                        overwrite=self.overwrite)
        if self.indexer is not None:
            self.indexer['alignment_err_vector'] = alignment_err_vector_path
    

    def create_cosmic_ray_map(self, z_coeff=3., step_number=None,
                              bad_frames_vector=[], star_list_path=None,
                              stars_fwhm_pix=2.):
        """Find the cosmic rays in the raw images and creates 
        the cosmic ray map used to compute the interferogram.

        :param z_coeff: (Optional) Threshold coefficient for cosmic ray
          detection, lower it to detect more cosmic rays (default : 3.).

        :param step_number: (Optional) 'Full' number of steps if the
          cube was complete. Might be different from the 'real' number
          of steps obtained. Helps in finding ZPD (default None).

        :param bad_frames_vector: (Optional) Contains the index of the
          frames considered as bad(default []).

        :param star_list_path: (Optional) Path to a list of stars that
          must be protected from over detection. All cosmic rays
          detected in those stars will be removed.

        :param stars_fwhm_pix: (Optional) mean FWHM of the stars in
          pixels.

        .. note:: A cosmic ray map is a 'mask like' cube filled with
           zeros and containing a one for each pixel identified as a
           cosmic ray. This cube is stored as a virtual
           **frame-divided** cube
        """


        def predetect_crs_in_column(fdata, z_coeff):
            """predetect CR in a column of filtered data"""

            MAX_VALUES_NB = 2 # Number of max values removed before
                              # computing median and std of the
                              # interferogram. Must be > 0
                              
            MAX_CR = 5 # Maximum pre-detected number of cosmic rays
            
            FFT_CUT = 0.1 # Ratio giving the position of the cut for
                           # FFT filtering
                           
            result = np.zeros_like(fdata, dtype=bool)
            
            for ij in range(fdata.shape[0]):
                test_vector = np.copy(fdata[ij,:])
                    
                # FFT filtering: remove low frequency modes due to
                # misalignment of the frames
                test_vector = orb.utils.vector.fft_filter(
                    test_vector, FFT_CUT,
                    width_coeff=0.1)
       
                # median and std computed over the vector without
                # its too deviant values
                max_threshold = np.sort(
                    np.abs(test_vector))[-MAX_VALUES_NB]
                filter_test_vector = test_vector[
                    np.nonzero(np.abs(test_vector) < max_threshold)]
                z_median = orb.utils.stats.robust_median(
                    filter_test_vector, warn=False)
                z_std = orb.utils.stats.robust_std(
                    filter_test_vector, warn=False)
                       
                # CR detection: If too much CRs are pre detected the
                # level of detection is raised
                too_much_cr = True
                while too_much_cr:
                    cr_vector = np.nonzero(test_vector > z_median + 
                                           (z_std * z_coeff))
                    if len(cr_vector[0]) <= MAX_CR:
                        too_much_cr = False
                    else:
                        z_coeff += 0.5

                    if z_coeff > z_coeff * 2:
                        too_much_cr = False
                        
                result[ij,:][cr_vector] = 1
            return result
            

        def check_predected_crs_in_frame(frames, pre_cr_map, ik):
            """Check pre-detected cosmic rays in a frame"""

            MINI_BOX_HSZ = 3 #5 # length degree of the mini box (final
                             # length is 2 * MINI_BOX_HSZ + 1)
                             
            
            REJECT_COEFF = 3.5 # Rejection coefficient # 3.5
            
            dimx = frames.shape[0]
            dimy = frames.shape[1]
            
            cr_map = np.zeros((dimx, dimy), dtype=np.bool)
            pre_crs = np.nonzero(pre_cr_map)
            del pre_cr_map
            
            ## checking pre-detected cosmic ray the classic way.
            for icr in range(len(pre_crs[0])):
                ii = pre_crs[0][icr]
                ij = pre_crs[1][icr]

                x_min, x_max, y_min, y_max = orb.utils.image.get_box_coords(
                    ii, ij, MINI_BOX_HSZ*2+1, 0, dimx, 0, dimy)
                
                box = frames[x_min:x_max, y_min:y_max, ik]

                median_box = np.median(
                    frames[x_min:x_max, y_min:y_max, :], axis=2)
                median_box[np.nonzero(median_box == 0)] = np.nan

                box /= median_box
                tested_pixel = box[ii-x_min, ij-y_min]

                # removing tested pixel to compute mean and std
                stat_box = np.copy(box)
                stat_box[ii-x_min, ij-y_min] = np.nan
                stat_box = np.sort(stat_box.flatten())[:-MINI_BOX_HSZ]
                
                box_median = orb.utils.stats.robust_median(stat_box)
                box_std = orb.utils.stats.robust_std(stat_box)

                # pre-detected cr is finally checked
                if (tested_pixel >
                    box_median + REJECT_COEFF * box_std):
                    cr_map[ii,ij] = True

            return cr_map


        def filter_frame(frame):
            """High pass filter applied on frame to help for cosmic
            ray detection
            """
            return frame - orb.utils.image.low_pass_image_filter(frame,deg=1)



        def check_cr_frame(frame, cr_map_frame, star_list, stars_fwhm):
            """Check pixels around the detected cosmic rays.
            """
            BASE_COEFF = 1. # Detection coeff
            BOX_SIZE = 3 # Size of the box around the cosmic ray to be
                         # checked

            STAR_BOX_SIZE_COEFF = 4 # Size coefficient of the box
                                    # around a star where all cosmic
                                    # rays are removed. # 10

            star_box_size = math.ceil(STAR_BOX_SIZE_COEFF * stars_fwhm)
        
            dimx = frame.shape[0]
            dimy = frame.shape[1]
            cr_coords = np.nonzero(cr_map_frame)
            
            for icr in range(len(cr_coords[0])):
                ii = cr_coords[0][icr]
                ij = cr_coords[1][icr]
                x_min, x_max, y_min, y_max = orb.utils.image.get_box_coords(
                        ii, ij, BOX_SIZE, 0, dimx, 0, dimy)
                box = frame[x_min:x_max, y_min:y_max]
                cr_box = cr_map_frame[x_min:x_max, y_min:y_max]
                box_mask = np.zeros_like(box).astype(np.bool)
                box_mask[np.nonzero(cr_box > 0)] = True
                masked_box = np.ma.masked_array(box, mask=box_mask)
                median_box = np.ma.median(masked_box)
                std_box = np.ma.std(masked_box)
                
                # check around to detect adjacent cr
                stat_box = np.zeros_like(box, dtype=bool)
                stat_box[np.nonzero(
                    box > median_box + BASE_COEFF * std_box)] = True
                cr_map_frame[x_min:x_max, y_min:y_max] = stat_box

            # stars used for calibration are protected from cosmic-ray detection
            for istar in range(star_list.shape[0]):
                 ii = star_list[istar,0]
                 ij = star_list[istar,1]
                 x_min, x_max, y_min, y_max = orb.utils.image.get_box_coords(
                     ii, ij, star_box_size, 0, dimx, 0, dimy)
                 cr_map_frame[x_min:x_max, y_min:y_max] = False
            
            return cr_map_frame
        

        
        self._print_msg("Creating cosmic ray map", color=True)
        self._print_msg("First detection pass", color=True)

        star_list = orb.utils.astrometry.load_star_list(star_list_path)
        
        cr_map = np.empty((self.dimx, self.dimy, self.dimz), dtype=np.bool)

        for iquad in range(0, self.QUAD_NB):
            x_min, x_max, y_min, y_max = self.get_quadrant_dims(iquad)
            iquad_data = self.get_data(x_min, x_max, 
                                       y_min, y_max, 
                                       0, self.dimz)
            quad_dimx = x_max - x_min

            ## Filtering frames first
            # Init multiprocessing server
            job_server, ncpus = self._init_pp_server()
            progress = ProgressBar(self.dimz)
            for iframe in range(0, self.dimz, ncpus):
                progress.update(
                    iframe,
                    info="filtering quad %d/%d"%(iquad+1, self.QUAD_NB))
                if iframe + ncpus >= self.dimz:
                    ncpus = self.dimz - iframe

                jobs = [(ijob, job_server.submit(
                    filter_frame, 
                    args=(iquad_data[:,:,iframe+ijob],),
                    modules=("import numpy as np",
                             "import orb.utils.image")))
                        for ijob in range(ncpus)]

                for ijob, job in jobs:
                    # filtered data is written in place of non filtered data
                    iquad_data[:,:,iframe+ijob] = job() 
                    
            self._close_pp_server(job_server)
            progress.end()

            ## Predetecting CR
            iquad_pre_cr_map = np.empty_like(iquad_data, dtype=bool)
            
            # Init multiprocessing server
            job_server, ncpus = self._init_pp_server()
        
            progress = ProgressBar(quad_dimx)
            for ii in range(0, quad_dimx, ncpus):
                progress.update(ii, info="CR predetection")
                
                if (ii + ncpus >= quad_dimx):
                    ncpus = quad_dimx - ii
                
                jobs = [(ijob, job_server.submit(
                    predetect_crs_in_column, 
                    args=(iquad_data[ii+ijob,:,:], z_coeff),
                    modules=("import numpy as np",
                             "import orb.utils.vector",
                             "import orb.utils.stats")))
                        for ijob in range(ncpus)]
                    
                for ijob, job in jobs:
                    iquad_pre_cr_map[ii+ijob,:,:] = job()
                
            progress.end()
            self._close_pp_server(job_server)
            self._print_msg("%d pre-detected cosmic rays in quad %d/%d"%(
                np.sum(iquad_pre_cr_map), iquad+1, self.QUAD_NB))

            # unfiltered data is reloaded
            iquad_data = self.get_data(x_min, x_max, 
                                       y_min, y_max, 
                                       0, self.dimz)
            
            ## Checking pre-detected CRs the classic way
            Z_STACK = 6
            
            # Init multiprocessing server
            job_server, ncpus = self._init_pp_server()
            progress = ProgressBar(self.dimz)
            for iframe in range(0, self.dimz, ncpus):
                progress.update(
                    iframe,
                    info="checking CRs in frames [%d/%d]"%(
                        iquad+1, self.QUAD_NB))
                if iframe + ncpus >= self.dimz:
                    ncpus = self.dimz - iframe

                z_min_list = list()
                z_max_list = list()
                for ijob in range(ncpus):
                    z_min = iframe + ijob - int(Z_STACK/2)
                    z_max = iframe + ijob + int(Z_STACK/2) + 1
                    if z_min < 0: z_min = 0
                    if z_max > self.dimz: z_max = self.dimz
                    z_min_list.append(z_min)
                    z_max_list.append(z_max)
                
                jobs = [(ijob, job_server.submit(
                    check_predected_crs_in_frame, 
                    args=(iquad_data[:,:,z_min_list[ijob]:z_max_list[ijob]],
                          iquad_pre_cr_map[:,:,iframe+ijob],
                          iframe+ijob-z_min_list[ijob]),
                    modules=("import numpy as np",
                             "import orb.utils.image",
                             "import orb.utils.stats")))
                        for ijob in range(ncpus)]

                for ijob, job in jobs:
                    cr_map[x_min:x_max, y_min:y_max, iframe+ijob] = job()
                    
            progress.end()     
            self._close_pp_server(job_server)
            self._print_msg("%d detected cosmic rays in quad %d/%d"%(
                np.sum(cr_map[x_min:x_max, y_min:y_max,:]),
                iquad+1, self.QUAD_NB))
            
        ## Check 'strange' frames with too much CRs
        self._print_msg("Re-checking strange frames", color=True)
        
        STRANGE_DETECT_COEFF = 2. # 1.
        CR_BOX_SIZE = 3
        LARGE_BOX_SIZE = 15
        
        cr_by_frame = [np.sum(cr_map[:,:,iz]) for iz in range(cr_map.shape[2])]
        cr_med = np.median(cr_by_frame)
        cr_std = orb.utils.stats.robust_std(cr_by_frame)
        
        strange_frames = list(np.nonzero(
            cr_by_frame > cr_med + STRANGE_DETECT_COEFF * cr_std)[0])

        progress = ProgressBar(np.size(strange_frames))
        frame_nb = 0
        for iframe in strange_frames:
            cr_frame = np.copy(cr_map[:,:,iframe])
            cr_coords = np.nonzero(cr_frame)
            cr_pos = np.array([(cr_coords[0][icr], cr_coords[1][icr])
                                for icr in range(len(cr_coords[0]))])
            frame = self.get_data_frame(iframe)
            for istar in range(cr_pos.shape[0]):
                crx = cr_pos[istar,0]
                cry = cr_pos[istar,1]
                cr_level = frame[crx, cry]
                # define cr box
                x_min, x_max, y_min, y_max = orb.utils.image.get_box_coords(
                    crx, cry, CR_BOX_SIZE, 0,
                    cr_frame.shape[0], 0, cr_frame.shape[1])
                cr_box = frame[x_min:x_max, y_min:y_max]

                # define a large box around cr box
                x_min, x_max, y_min, y_max = orb.utils.image.get_box_coords(
                    crx, cry, LARGE_BOX_SIZE, 0,
                    cr_frame.shape[0], 0, cr_frame.shape[1])
                large_box = frame[x_min:x_max, y_min:y_max]

                # test
                large_mean = orb.utils.stats.robust_mean(
                    orb.utils.stats.sigmacut(large_box))
                box_mean = (np.sum(cr_box) - cr_level) / (np.size(cr_box) - 1)
                if box_mean > 2.* large_mean:
                    cr_map[crx,cry,iframe] = 0

                elif cr_level < 2. * box_mean:
                    cr_map[crx,cry,iframe] = 0
                    
            frame_nb += 1
            progress.update(frame_nb, info="checking frame: %d"%iframe)
        progress.end()  
     
        ## Second pass : check around the detected cosmic rays
        ## self._print_msg("Checking CRs neighbourhood", color=True)
        
        ## # Init multiprocessing server
        ## job_server, ncpus = self._init_pp_server()
        ## progress = ProgressBar(self.dimz)
        ## for ik in range(0, self.dimz, ncpus):
        ##     # No more jobs than frames to compute
        ##     if (ik + ncpus >= self.dimz): 
        ##         ncpus = self.dimz - ik
                
        ##     frames = np.empty((self.dimx, self.dimy, ncpus), dtype=float)
        ##     for ijob in range(ncpus): 
        ##         frames[:,:,ijob] = self.get_data_frame(ik+ijob)
                
        ##     cr_map_frames = np.copy(cr_map[:,:,ik: ik + ncpus + 1])
            
        ##     jobs = [(ijob, job_server.submit(
        ##         check_cr_frame, 
        ##         args=(frames[:,:,ijob], 
        ##               cr_map_frames[:,:,ijob],
        ##               star_list, stars_fwhm_pix),
        ##         modules=("import numpy as np",
        ##                  "import math",
        ##                  "import orb.utils",
        ##                  "import orb.astrometry")))
        ##             for ijob in range(ncpus)]
            
        ##     for ijob, job in jobs:
        ##         cr_map[:,:,ik + ijob] = job()

        ##     progress.update(ik, info="checking frame: %d"%ik)
            
        ## self._close_pp_server(job_server)
        ## progress.end()
        
        self._print_msg("Total number of detected cosmic rays: %d"%np.sum(
            cr_map), color=True)
        
        ## Writing cosmic ray map to disk
        out_cube = OutHDFCube(self._get_cr_map_cube_path(),
                              shape=cr_map.shape,
                              overwrite=self.overwrite)
        for iframe in range(self.dimz):
            out_cube.write_frame(
                iframe, cr_map[:,:,iframe].astype(np.bool_),
                header=self._get_basic_header(file_type="Cosmic ray map"),
                force_float32=False)

        out_cube.close()
        del out_cube
            
        if self.indexer is not None:
            self.indexer['cr_map_cube'] = self._get_cr_map_cube_path()

    def check_bad_frames(self, cr_map_cube_path=None, coeff=2.):
        """Check an interferogram cube for bad frames.

        If the number of detected cosmic rays is too important the
        frame is considered as bad

        :param cr_map_cube_path: (Optional) Path to the cosmic ray map
          cube. If None given, default path is used (default None).
        
        :param coeff: (Optional) Threshold coefficient (default 2.)
        """
        self._print_msg("Checking bad frames")
        MIN_CR = 30.
        
        # Instanciating cosmic ray map cube
        if (cr_map_cube_path is None):
                cr_map_cube_path = self._get_cr_map_cube_path()
        cr_map_cube = HDFCube(cr_map_cube_path,
                              config_file_name=self.config_file_name,
                              ncpus=self.ncpus)
        cr_map = cr_map_cube.get_all_data()
        cr_map_vector = np.sum(np.sum(cr_map, axis=0), axis=0)
        median_rc = np.median(cr_map_vector)
        pre_bad_frames_vector = np.nonzero(cr_map_vector > median_rc + coeff*median_rc)[0]
        bad_frames_vector = list()
        for ibad_frame in pre_bad_frames_vector:
            if cr_map_vector[ibad_frame] > MIN_CR:
                bad_frames_vector.append(ibad_frame)
        self._print_msg("Detected bad frames : " + str(bad_frames_vector))
        return np.array(bad_frames_vector)

    def create_hot_pixel_map(self, dark_image, bias_image):
        """Create a hot pixel map from a cube of dark frame

        :param bias_image: Master bias frame (can be set to None)
        :param dark_image: Master dark frame
        
        .. note:: A hot pixel map is a mask like frame (1 for a hot
           pixel, 0 elsewhere)"""
        nsigma = 5.0 # Starting sigma
        MIN_COEFF = 0.026 # Min percentage of hot pixels to find

        self._print_msg("Creating hot pixel map")

        if bias_image is not None:
            dark_image = np.copy(dark_image) - np.copy(bias_image)
        hp_map = np.zeros_like(dark_image).astype(np.uint8)
        dark_mean = np.mean(dark_image)
        dark_std = np.std(dark_image)

        nsigma_ok = False
        while not (nsigma_ok):    
            hp_map[np.nonzero(
                dark_image > (dark_mean + nsigma * dark_std))] = 1
            if (np.shape(np.nonzero(hp_map))[1]
                > MIN_COEFF * self.dimx * self.dimy):
                nsigma_ok = True
            else:
                if (nsigma > 0.0):
                    nsigma -= 0.01
                else:
                    nsigma_ok = True
                    hp_map = np.zeros_like(dark_image).astype(np.uint8)
                    self._print_warning("No hot pixel found on frame")
                    
        self._print_msg("Percentage of hot pixels : %.2f %%"%(
            float(np.shape(np.nonzero(hp_map))[1])
            / (self.dimx * self.dimy) * 100.))
        self.write_fits(self._get_hp_map_path(), hp_map,
                        fits_header=self._get_cr_map_frame_header(),
                        overwrite=self.overwrite)

    def get_noise_values(self, bias_path, dark_path, exposition_time,
                         dark_int_time, combine='average', reject='avsigclip'):
        """
        Return readout noise and dark current level from bias and dark
        frames.
        
        :param bias_path: Path to a list of bias files.
        
        :param dark_path: Path to a list of dark files.
        
        :param exposition_time: Integration time of the frames.

        :param dark_int_time: Integration time of the dark frames.

        :param reject: (Optional) Rejection operation for master
          frames creation. Can be 'sigclip', 'minmax', 'avsigclip' or
          None (default 'avsigclip'). See
          :py:meth:`orb.utils.image.create_master_frame`.
        
        :param combine: (Optional) Combining operation for master
          frames creation. Can be 'average' or 'median' (default
          'average'). See
          :py:meth:`orb.utils.image.create_master_frame`.

        :return: readout_noise, dark_current_level
        """
        BORDER_COEFF = 0.45 # Border coefficient to take only the
                            # center of the frames to compute noise
                            # levels
                           
        bias_image, master_bias_temp = self._load_bias(
            bias_path, return_temperature=True, combine=combine,
            reject=reject)
        
        bias_cube = Cube(bias_path,
                         config_file_name=self.config_file_name,
                         ncpus=self.ncpus)
        
        min_x = int(bias_cube.dimx * BORDER_COEFF)
        max_x = int(bias_cube.dimx * (1. - BORDER_COEFF))
        min_y = int(bias_cube.dimy * BORDER_COEFF)
        max_y = int(bias_cube.dimy * (1. - BORDER_COEFF))
        
        readout_noise = [orb.utils.stats.robust_std(bias_cube[min_x:max_x,
                                                              min_y:max_y, ik])
                         for ik in range(bias_cube.dimz)]
        
        readout_noise = orb.utils.stats.robust_mean(readout_noise)
        
        dark_cube = Cube(dark_path,
                         config_file_name=self.config_file_name,
                         ncpus=self.ncpus)
        dark_cube_dimz = dark_cube.dimz
        
        if not self.is_same_2D_size(dark_cube):
            dark_cube = dark_cube.get_resized_data(self.dimx, self.dimy)
            dark_cube_dimz = dark_cube.shape[2]

        dark_current_level = [orb.utils.stats.robust_median(
            (dark_cube[:,:,ik] - bias_image)[min_x:max_x, min_y:max_y])
                              for ik in range(dark_cube_dimz)]
        
        dark_current_level = orb.utils.stats.robust_mean(dark_current_level)
        dark_current_level = (dark_current_level
                              / dark_int_time * exposition_time)
        
        return readout_noise, dark_current_level


    def get_bias_coeff_from_T(self, master_bias_temp, master_bias_level,
                              frame_temp, calibrated_params):
        """
        :param master_bias_temp: Temperature of the master bias frame.

        :param master_bias_level: Median of the master bias frame.

        :param frame_temp: Temperature of the frame to correct.

        :param calibrated_params: parameters [a, b] of the
          function bias_level(T) = aT + b. T is in degrees C and
          bias_level(T) is the median of the bias frame at the
          given temperature.
        """
        calib_master_bias_level = np.polyval(
            calibrated_params, master_bias_temp)
        calib_frame_bias_level = np.polyval(
            calibrated_params, frame_temp)
        return  calib_frame_bias_level / calib_master_bias_level

    def get_dark_coeff_from_T(self, master_dark_temp, master_dark_level,
                              frame_temp, activation_energy):
        """
         Return calibrated coefficient of the calibrated dark frame
         given the temperature of the frame to correct (see Widenhorn
         et al. 2002)

        :param frame_temp: Temperature of the frame to correct.
        
        :master_dark_level: Master dark level in counts
        
        :master_dark_temp: Master dark temperature in Celsius
            
        :activation_energy: Activation energy in eV
        """
        def dark_level(frame_temp, master_dark_level,
                       master_dark_temp, activation_energy):
            """Return calibrated dark level (see Widenhorn et al. 2002)

            :master_dark_level: Master dark level in counts

            :master_dark_temp: Master dark temperature in Celsius

            :activation_energy: Activation energy in eV
            """
            k = 8.62e-5 # eV/K
            level = (master_dark_level * np.exp(
                activation_energy * (1./(k * (master_dark_temp + 273.15))
                                     - 1./(k * (frame_temp + 273.15)))))

            if level > 0.: return level
            else: return 0.

        calib_frame_dark_level = dark_level(frame_temp, master_dark_level,
                                            master_dark_temp,
                                            activation_energy)

        return  calib_frame_dark_level/ master_dark_level
    
    def correct_frame(self, index, master_bias, master_dark, master_flat,
                      dark_int_time, flat_int_time, hp_map_path,
                      optimize_dark_coeff, exposition_time,
                      negative_values, master_dark_temp,
                      master_bias_temp, dark_activation_energy,
                      bias_calibration_params, master_bias_level,
                      master_dark_level):
        
        """Correct a frame for the bias, dark and flat field.
        
        :param index: Index of the frame to be corrected
        
        :param master_bias: Master Bias (if None, no correction is done)
        
        :param master_dark: Master Dark. Must be in counts/s and bias
          must have been removed. (if None, no dark and flat
          corrections are done)
        
        :param master_flat: Master Flat (if None, no flat correction
          is done)
        
        :param dark_int_time: Dark integration time
        
        :param flat_int_time: Flat integration time
        
        :param hp_map_path: Path to the hot pixel map
        
        :param optimize_dark_coeff: If True use a fast optimization
          routine to calculate the best coefficient for dark
          correction. This routine is used to correct for the images
          of the camera 2 on SpIOMM, because it has a varying dark and
          bias level and contains a lot of hot pixels (Default False).

        :param dark_activation_energy: Activation energy in eV. This
          is a calibration parameter used to guess the master dark
          coefficient to apply in case the temperature of the frame is
          different from the master dark temperature. Useful only if
          optimize_dark_coeff is True and the temperature of the dark
          frames and the interferogram frames is given in their header
          [keyword 'CCD-TEMP'] (Default None).

        :param bias_calibration_params: A tuple of 2 parameters [a,b]
          that are used to compute the bias coefficient for a varying
          temperature of the camera. Useful only if
          optimize_dark_coeff is True and the temperature of the bias
          frames and the interferogram frames is given in their header
          [keyword 'CCD-TEMP'] (Default None).

        :param master_dark_temp: Mean temperature of the master dark
          frame.

        :param master_bias_temp: Mean temperature of the master bias
          frame.

        :param master_bias_level: Median level of the master bias frame.

        :param master_dark_level: Median level of the master dark frame.
        
        :param exposition_time: (Optional) Integration time of the 
          frames (can be defined in the option file).

        :param negative_values: if False, replace negative values in
          the calculated interferogram by zero.

        .. note:: The correction steps are:
        
          * (If optimize_dark_coeff is True) : Dark and bias levels
            are corrected using calibrated functions and the level of
            hot pixels are optimized by minimizing their standard
            deviation.
          * Bias is substracted to dark and flat
          * Dark is substracted to flat if the integration 
            time of the flat is given
          * Flat is normalized (median = 1)
          * The corrected image is calculated
            :math:`frame=\\frac{frame - dark + bias}{flat}`.
          * Negative values are set to 0 by default
          
        """
        
        def _optimize_dark_coeff(frame, dark_frame, hp_map,
                                 only_hp=False):
            """Return an optimized coefficient to apply to the dark
            integration time.
            
            Useful if the frames contain a lot of hot pixels and a
            varying bias and dark level because of a varying
            temperature.

            :param frame: The frame to correct
            
            :param hp_map: Hot pixels map
                    
            :param only_hp: if True optimize the dark coefficient for
              the hot pixels of the frame. If False optimize the dark
              coefficient for the 'normal' pixels of the frame (default
              False).
            """
            def _coeff_test(dark_coeff, frame, dark_frame, hp_map, only_hp):
                test_frame = frame - (dark_frame * dark_coeff)
                if hp_map is not None:
                    if only_hp:
                        hp_frame = test_frame[np.nonzero(hp_map)]
                        # we try to minimize the std of the hot pixels in
                        # the frame 
                        std = np.sqrt(np.mean(
                            ((orb.utils.stats.robust_median(hp_frame)
                              - orb.utils.stats.robust_median(
                                  test_frame))**2.)))
                    else:
                        non_hp_frame = test_frame[np.nonzero(hp_map==0)]
                        non_hp_frame = non_hp_frame[np.nonzero(non_hp_frame)]
                        # We try to find the best dark coefficient to
                        # apply to the non hp frame
                        std = orb.utils.stats.robust_std(non_hp_frame)
                        
                return std

            

            guess = [1.0]        
            result = optimize.fmin_powell(_coeff_test, guess,
                                          (frame, dark_frame, hp_map, only_hp),
                                          xtol=1e-5, ftol=1e-5, disp=False)
            return result
            
            
        frame = np.array(self.get_data_frame(index), dtype = float)
        
        if master_bias is not None: master_bias = np.copy(master_bias)
        if master_dark is not None: master_dark = np.copy(master_dark)
        if master_flat is not None: master_flat = np.copy(master_flat)
        
        # getting frame temperature
        frame_header = self.get_frame_header(index)
        if 'CCD-TEMP' in frame_header:
            frame_temp = frame_header["CCD-TEMP"]
        else:
            frame_temp = None

        # getting bias level
        frame_header = self.get_frame_header(index)
        if "BIAS-LVL" in frame_header:
            frame_bias_level = frame_header["BIAS-LVL"]
        else:
            frame_bias_level = None
        
        # bias substraction
        if master_bias is not None:
            bias_coeff = 1.0
            if (optimize_dark_coeff):
                if frame_bias_level is not None:
                    bias_coeff = frame_bias_level / master_bias_level
                elif ((bias_calibration_params is not None)
                      and (master_bias_temp is not None)
                      and (frame_temp is not None)):
                    bias_coeff = self.get_bias_coeff_from_T(
                        master_bias_temp,
                        master_bias_level,
                        frame_temp,
                        bias_calibration_params)
                
            frame -= master_bias * bias_coeff

        # computing dark image (bias substracted)
        if (master_dark is not None
            and exposition_time is not None):
            
            if optimize_dark_coeff:
                # load hot pixels map
                if hp_map_path is None:
                    hp_map = self.read_fits(self._get_hp_map_path())
                else: 
                    hp_map = self.read_fits(hp_map_path)
                # remove border on hp map
                hp_map_corr = np.copy(hp_map)
                hp_map_corr[0:self.dimx/5., :] = 0.
                hp_map_corr[:, 0:self.dimx/5.] = 0.
                hp_map_corr[4.*self.dimx/5.:, :] = 0.
                hp_map_corr[:, 4.*self.dimx/5.:] = 0.
                    
                # If we can use calibrated parameters, the dark frame
                # is scaled using the temperature difference between
                # the master dark and the frame.
                if ((dark_activation_energy is not None)
                    and (master_dark_temp is not None)
                    and (frame_temp is not None)):
                    dark_coeff = self.get_dark_coeff_from_T(
                        master_dark_temp, master_dark_level,
                        frame_temp, dark_activation_energy)
                    
                    dark_coeff *= exposition_time
                    
                # If no calibrated params are given the dark
                # coefficient to apply is guessed using an
                # optimization routine.
                else:
                    dark_coeff = _optimize_dark_coeff(frame, master_dark, 
                                                      hp_map_corr,
                                                      only_hp=False)
                    
                temporary_frame = frame - (master_dark * dark_coeff)

                # hot pixels correction
                if np.any(hp_map):
                    temporary_frame = orb.utils.image.correct_hot_pixels(
                        temporary_frame, hp_map)
                    
                
                ## # hot pixels only are now corrected using a special
                ## # dark coefficient that minimize their std
                ## hp_dark_coeff = _optimize_dark_coeff(
                ##     frame, master_dark, hp_map_corr, only_hp=True)
                ## hp_frame = (frame
                ##             - master_dark * hp_dark_coeff
                ##             - orb.utils.stats.robust_median(master_dark) * dark_coeff)
                ## temporary_frame[np.nonzero(hp_map)] = hp_frame[
                ##     np.nonzero(hp_map)]
                
                frame = temporary_frame

            # else: simple dark substraction
            else:
                frame -= (master_dark * exposition_time)
                

        # computing flat image
        if master_flat is not None:
            if (dark_int_time is not None) and (flat_int_time is not None):
                dark_flat_coeff = float(flat_int_time / float(dark_int_time))
                dark_master_flat = master_dark * dark_flat_coeff
                master_flat = master_flat - dark_master_flat
                
            if master_bias is not None:
                master_flat -= master_bias
                
            # flat normalization
            master_flat /= np.median(master_flat)
            # flat correction
            flat_zeros = np.nonzero(master_flat==0)
            # avoid dividing by zeros
            master_flat[flat_zeros] = 1.
            frame /= master_flat
            # zeros are replaced by NaNs in the final frame
            frame[flat_zeros] = np.nan

        return frame



    def correct(self, bias_path=None, dark_path=None, flat_path=None,
                cr_map_cube_path=None, alignment_vector_path=None,
                dark_int_time=None, flat_int_time=None,
                exposition_time=None, bad_frames_vector=[],
                optimize_dark_coeff=False,
                dark_activation_energy=None,
                bias_calibration_params=None, negative_values=False,
                z_range=[], order=1, zeros=False, combine='average',
                reject='avsigclip', flat_smooth_deg=0):
        
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
          
        :param dark_int_time: (Optional) Integration time of the dark
          frames. Used to remove the dark pattern from a flat with a
          different integration time. User must specify flat_int_time.
          
        :param flat_int_time: (Optional) Integration time of the flat
          frames. see dark_int_time.
          
        :param bad_frames_vector: (Optional) Contains the index of the
          frames to be replaced by zeros.
          
        :param optimize_dark_coeff: (Optional) If True use a fast optimization
          routine to calculate the best coefficient for dark
          correction. This routine is used to correct for the images
          of the camera 2 on SpIOMM, because it contains a lot of hot
          pixels (Default False).

        :param dark_activation_energy: Activation energy in eV. This is a
          calibration parameter used to guess the master dark
          coefficient to apply in case the temperature of the frame is
          different from the master dark temperature. Useful only if
          optimize_dark_coeff is True and the temperature of the dark
          frames and the interferogram frames is given in their header
          [keyword 'CCD-TEMP'] (Default None).

        :param bias_calibration_params: (Optional) a tuple of 2
          parameters [a,b] that are used to compute the bias
          coefficient for a varying temperature of the camera. Useful
          only if optimize_dark_coeff is True and the temperature of
          the bias frames and the interferogram frames is given in
          their header [keyword 'CCD-TEMP'] (Default None).

        :param exposition_time: (Optional) Integration time of the 
          frames (can be defined in the option file).

        :param negative_values: (Optional) If False, replace negative values in
          the calculated interferogram by zero (Default False).

        :param z_range: (Optional) 1d array containing the index of
          the frames to be computed.

        :param order: (Optional) Interpolation order (Default 1). Be
          careful in using an interpolation order greater than 1 with
          images containing stars.

        :param zeros: (Optional) If True, cosmic rays are replaced by
          zeros. If False, cosmic rays are replaced by the median of
          the neighbouring region (default False).

        :param reject: (Optional) Rejection operation for master
          frames creation. Can be 'sigclip', 'minmax', 'avsigclip' or
          None (default 'avsigclip'). See
          :py:meth:`orb.utils.image.create_master_frame`.
        
        :param combine: (Optional) Combining operation for master
          frames creation. Can be 'average' or 'median' (default
          'average'). See
          :py:meth:`orb.utils.image.create_master_frame`.

        :param flat_smooth_deg: (Optional) If > 0 smooth the master
          flat (help removing possible fringe pattern) (default
          0). See :py:meth:`process.RawData._load_flat`.

        .. note:: Frames considered as bad (which index are in
          ``bad_frames_vector`` or not in the ``z_range`` vector) are just
          replaced by frames of zeros.

        .. note:: The creation of the corrected interferogram frames
          walks through 2 steps:

          1. Correction for bias, dark, and flat field. Please refer
             to: :py:meth:`process.RawData.correct_frame`.
               
          2. Alignment of the frame using linear interpolation by
             default. A higher order interpolation can be used if the
             field contains no star like object. By default cosmic
             rays are replaced by the weighted average of the
             neighbouring region. Weights are computed from a gaussian
             kernel.
             
        .. seealso:: :py:meth:`process.RawData.correct_frame` 
        .. seealso:: :py:meth:`process.RawData.create_cosmic_ray_map`
        .. seealso:: :py:meth:`process.RawData.create_alignment_vector`
        """

        def _correct_frame(frame, ik, dimx, dimy, 
                           border_size, deg, alignment_vector_ii, 
                           cr_map_ii, bad_frames_vector, order,
                           zeros):
            """Process and return one frame of the interferogram. 
            This function is used for parallel processing.
            """
            mask_frame = np.zeros_like(frame, dtype=float)
            
            MEDIAN_DEG = 2
            x_range = range(border_size, int(dimx - border_size - 1L))
            y_range = range(border_size, int(dimy - border_size - 1L))
            x_min = np.min(x_range)
            x_max = np.max(x_range) + 1L
            y_min = np.min(y_range)
            y_max = np.max(y_range) + 1L
            plain_frame = np.empty((dimx, dimy), dtype=float)
            plain_frame.fill(np.nan)
            plain_mask_frame = np.zeros((dimx, dimy), dtype=float)
            
            if bad_frames_vector is None:
                bad_frames_vector = []
                
            if (ik not in bad_frames_vector): # we don't work on bad frames
                bad_pixels = np.nonzero(cr_map_ii)
                dx = alignment_vector_ii[0]
                dy = alignment_vector_ii[1]

                ## CR CORRECTION: CRs are replaced by a weighted
                # average of the neighbouring region. Weights are
                # calculated from a 2d gaussian kernel.
                for ibp in range(len(bad_pixels[0])):
                    ix = bad_pixels[0][ibp]
                    iy = bad_pixels[1][ibp]
                    mask_frame[ix, iy] = 1 # masking cr in mask frame
                    if (ix < x_max and iy < y_max
                        and ix >= x_min and iy >= y_min):
                        (med_x_min, med_x_max,
                         med_y_min, med_y_max) = orb.utils.image.get_box_coords(
                            ix, iy, MEDIAN_DEG*2+1,
                            x_min, x_max, y_min, y_max)
                        box = frame[med_x_min:med_x_max,
                                    med_y_min:med_y_max]
             
                        # definition of the kernel. It must be
                        # adjusted to the real box
                        ker = orb.cutils.gaussian_kernel(MEDIAN_DEG)
                        if (box.shape[0] != MEDIAN_DEG*2+1
                            or box.shape[1] != MEDIAN_DEG*2+1):
                            if ix - med_x_min != MEDIAN_DEG:
                                ker = ker[MEDIAN_DEG - (ix - med_x_min):,:]
                            if iy - med_y_min != MEDIAN_DEG:
                                ker = ker[:,MEDIAN_DEG - (iy - med_y_min):]
                            if med_x_max - ix != MEDIAN_DEG + 1:
                                ker = ker[:- (MEDIAN_DEG + 1
                                              - (med_x_max - ix)),:]
                            if med_y_max - iy != MEDIAN_DEG + 1:
                                ker = ker[:,:- (MEDIAN_DEG + 1
                                                - (med_y_max - iy))]
                        
                        # cosmic rays are removed from the
                        # weighted average (their weight is set to
                        # 0)
                        ker *=  1 - cr_map_ii[med_x_min:med_x_max,
                                              med_y_min:med_y_max]

                        # pixel is replaced by the weighted average
                        if np.sum(ker) != 0:
                            frame[ix, iy] = np.sum(box * ker)/np.sum(ker)
                        else:
                            # if no good pixel around can be found
                            # the pixel is replaced by the median
                            # of the whole box
                            frame[ix, iy] = np.median(box)

                ## SHIFT
                if (dx != 0.) and (dy != 0.):
                    
                    frame = orb.utils.image.shift_frame(frame, dx, dy, 
                                                        x_min, x_max, 
                                                        y_min, y_max, 1)
                    
                    mask_frame = orb.utils.image.shift_frame(
                        mask_frame, dx, dy, 
                        x_min, x_max, 
                        y_min, y_max, 1)
                else:
                    frame = frame[x_min:x_max, y_min:y_max]
                    mask_frame = mask_frame[x_min:x_max, y_min:y_max]

                ## FINAL CORRECTION (If zeros == True)
                if zeros:
                    # if required we eliminate the cosmic rays by
                    # setting the region around to zero
                    for ibp in range(len(bad_pixels[0])):
                        ix = bad_pixels[0][ibp] - x_min - dx
                        iy = bad_pixels[1][ibp] - y_min - dy
                        frame[ix-deg:ix+deg+1L,
                              iy-deg:iy+deg+1L] = 0.
                        
                plain_frame[x_min:x_max, y_min:y_max] = frame
                plain_mask_frame[x_min:x_max, y_min:y_max] = mask_frame

            return plain_frame, plain_mask_frame
 

        DEG = 1L # radius of the zone considered as bad around a cosmic ray
        CENTER_SIZE_COEFF = 0.1 # size ratio of the center region used
                                # for frames stats

        self._print_msg("Creating interferogram")
        
        x_min, x_max, y_min, y_max = orb.utils.image.get_box_coords(
            self.dimx/2., self.dimy/2.,
            max((self.dimx, self.dimy))*CENTER_SIZE_COEFF,
            0, self.dimx, 0, self.dimy)
         
        ### load needed data ##################################

        # check existence of dark and bias calibration parameters in
        # case of an optimization of the bias level and the dark level
        if optimize_dark_coeff:
            if dark_activation_energy is None:
                self._print_warning("No dark activation energy have been passed. The dark level will have to be guessed (less precise)")
            else:
                self._print_msg("Dark activation energy (in eV): %s"%str(
                    dark_activation_energy))
            if bias_calibration_params is None:
                self._print_warning("No bias calibration parameters have been passed. The bias level will not be optimized (less precise)")
            else:
                self._print_msg("Bias calibration parameters: %s"%str(
                    bias_calibration_params))
                
        # load master bias
        if (bias_path is not None):
            master_bias, master_bias_temp = self._load_bias(
                bias_path, return_temperature=True, combine=combine,
                reject=reject)
            master_bias_level = orb.utils.stats.robust_median(
                master_bias[x_min:x_max,
                            y_min:y_max])
            self._print_msg('Master bias median level at the center of the frame: %f'%master_bias_level)
            if optimize_dark_coeff and master_bias_temp is None:
                self._print_warning("The temperature of the master bias could not be defined. The bias level will not be optimized (less precise)")
        else:
            master_bias = None
            master_bias_temp = None
            master_bias_level = None
            self._print_warning("no bias list given, there will be no bias correction of the images")
            
        # load master dark (bias is substracted and master dark is
        # divided by the dark integration time)
        if dark_path is not None:
            master_dark, master_dark_temp = self._load_dark(
                dark_path, return_temperature=True, combine=combine,
                reject=reject)
            master_dark_uncorrected = np.copy(master_dark)
            
            if optimize_dark_coeff:
                # remove bias
                if master_dark_temp is None and master_bias is not None:
                    self._print_warning("The temperature of the master dark could not be defined. The dark level will have to be guessed (less precise)")
                    master_dark -= master_bias
                elif (bias_calibration_params is not None
                      and master_bias is not None):
                    master_bias_coeff = self.get_bias_coeff_from_T(
                        master_bias_temp, master_bias_level,
                        master_dark_temp, bias_calibration_params)
                    master_dark -= master_bias * master_bias_coeff
                
            elif master_bias is not None:
                master_dark -= master_bias

            # master dark in counts/s
            master_dark /= dark_int_time
            
            master_dark_level = orb.utils.stats.robust_median(master_dark[
                x_min:x_max, y_min:y_max])
            self._print_msg('Master dark median level at the center of the frame: %f'%master_dark_level)
                
        else:
            master_dark = None
            master_dark_temp = None
            master_dark_level = None
            self._print_warning("no dark list given, there will be no dark corrections of the images")

        # load master flat
        if (flat_path is not None):
            master_flat = self._load_flat(flat_path, combine=combine,
                                          reject=reject,
                                          smooth_deg=flat_smooth_deg)
        else:
            master_flat = None
            self._print_warning("No flat list given, there will be no flat field correction of the images")
            
        # load alignment vector
        if (alignment_vector_path is None):
            alignment_vector_path = self._get_alignment_vector_path() 
        alignment_vector = self._load_alignment_vector(alignment_vector_path)
        if (alignment_vector is None):
            alignment_vector = np.zeros((self.dimz, 2), dtype = float)
            self._print_warning("No alignment vector loaded : there will be no alignment of the images")

        # create hot pixel map
        hp_map_path = None
        if optimize_dark_coeff:
            if master_dark is not None:
                self.create_hot_pixel_map(master_dark_uncorrected, master_bias)
                hp_map_path=self._get_hp_map_path()
            else:
                self._print_warning("No dark or bias frame given : The hot pixel map cannot be created")
        
        # define the nice zone (without extrapolations due to
        # disalignment)
        border_size = int(math.ceil(np.max(alignment_vector)) + 1L)
        if (DEG > border_size):
            border_size = DEG + 1L

        if (z_range == []):
            z_min = 0
            z_max = self.dimz
        else:
            z_min = min(z_range)
            z_max = max(z_range)
            

        cr_map_cube = None
        # Instanciating cosmic ray map cube
        if cr_map_cube_path is None:
            cr_map_cube_path = self._get_cr_map_cube_path()
            
        if os.path.exists(cr_map_cube_path):
            cr_map_cube = HDFCube(cr_map_cube_path,
                                  config_file_name=self.config_file_name,
                                  ncpus=self.ncpus)
            self._print_msg("Loaded cosmic ray map: {}".format(cr_map_cube_path))
        else:
            self._print_warning("No cosmic ray map loaded")
                
        self._print_msg("Computing interferogram")

        # Multiprocessing server init
        job_server, ncpus = self._init_pp_server() 
        ncpus_max = ncpus

        # creating output hdfcube
        out_cube = OutHDFCube(self._get_interfero_cube_path(),
                              (self.dimx, self.dimy, z_max-z_min),
                              overwrite=self.overwrite)
        
        # Interferogram creation
        progress = ProgressBar(int((z_max - z_min) / ncpus_max))
        for ik in range(z_min, z_max, ncpus):
                        
            # No more jobs than frames to compute
            if (ik + ncpus >= z_max): 
                ncpus = z_max - ik
                
            frames = np.empty((self.dimx, self.dimy, ncpus), dtype=float)
            cr_maps = np.zeros((self.dimx, self.dimy, ncpus), dtype=np.bool)
            
            for icpu in range(ncpus):
                if cr_map_cube is not None:
                    cr_maps[:,:,icpu] = cr_map_cube.get_data_frame(ik+icpu)

            # 1 - frames correction for bias, dark, flat.
            jobs = [(ijob, job_server.submit(
                self.correct_frame,
                args=(ik + ijob, 
                      master_bias, 
                      master_dark, 
                      master_flat, 
                      dark_int_time, 
                      flat_int_time, 
                      hp_map_path, 
                      optimize_dark_coeff,
                      exposition_time,
                      negative_values,
                      master_dark_temp,
                      master_bias_temp,
                      dark_activation_energy,
                      bias_calibration_params,
                      master_bias_level,
                      master_dark_level),
                modules=("numpy as np", 
                         "from scipy import optimize",
                         "import orb.utils.stats",
                         "import orb.utils.image")))
                    for ijob in range(ncpus)]
            
            for ijob, job in jobs:
                frames[:,:,ijob] = job()

            # 2 - frames alignment and correction for cosmic rays
            jobs = [(ijob, job_server.submit(
                _correct_frame, 
                args=(frames[:,:,ijob], 
                      ik + ijob, 
                      self.dimx, self.dimy, 
                      border_size, DEG, 
                      alignment_vector[ik + ijob,:], 
                      cr_maps[:,:,ijob], 
                      bad_frames_vector, order, zeros),
                modules=(
                    "numpy as np",
                    "import orb.utils.image",
                    "import orb.cutils",
                    "from scipy import ndimage",))) 
                    for ijob in range(ncpus)]

            for ijob, job in jobs:
                iframe, imask_frame = job()
                out_cube.write_frame(
                    ik+ijob,
                    data=iframe,
                    header=self._get_interfero_frame_header(),
                    mask=imask_frame,
                    record_stats=True)

            progress.update(int((ik - z_min) / ncpus_max), 
                            info="frame : " + str(ik))

        self._close_pp_server(job_server)
        progress.end()
        out_cube.close()
        del out_cube

        
        # check median level of frames (Because bad bias/dark frames can
        # cause a negative median level for the frames of camera 2)
        self._print_msg('Checking frames level')
        interf_cube = HDFCube(self._get_interfero_cube_path(),
                              config_file_name=self.config_file_name,
                              ncpus=self.ncpus)
        zmedian = interf_cube.get_zmedian()
        corr_level = -np.min(zmedian) + 10. # correction level
        header = self._get_interfero_frame_header()
        # correct frames if nescessary by adding the same level to every frame
        if np.min(zmedian) < 0.:
            out_cube = OutHDFCube(self._get_interfero_cube_path(),
                                  (self.dimx, self.dimy, z_max-z_min),
                                  overwrite=self.overwrite)
            
            self._print_warning('Negative median level of some frames. Level of all frames is being added %f counts'%(corr_level))
            progress = ProgressBar(interf_cube.dimz)
            for iz in range(interf_cube.dimz):
                progress.update(iz, info='Correcting negative level of frames')
                frame = interf_cube.get_data_frame(iz) + corr_level
                mask = interf_cube.get_data_frame(iz, mask=True)
                
                out_cube.write_frame(
                    iz,
                    data=frame,
                    mask=mask,
                    header=header,
                    record_stats=True)
            progress.end()

            out_cube.close()
            del out_cube
        
        if self.indexer is not None:
            self.indexer['interfero_cube'] = self._get_interfero_cube_path()
            
        self._print_msg("Interferogram computed")
        
        energy_map = interf_cube.get_interf_energy_map()
        deep_frame = interf_cube.get_mean_image()

        del interf_cube
        
        out_cube = OutHDFCube(self._get_interfero_cube_path(),
                              (self.dimx, self.dimy, z_max-z_min),
                              overwrite=self.overwrite)
        
        # create energy map
        out_cube.append_energy_map(energy_map)
        self.write_fits(
            self._get_energy_map_path(), energy_map,
            fits_header=self._get_energy_map_header(),
            overwrite=True, silent=False)
        
        if self.indexer is not None:
            self.indexer['energy_map'] = self._get_energy_map_path()

        # Create deep frame
        out_cube.append_deep_frame(deep_frame)
        
        if bn.nanmedian(deep_frame) < 0.:
            self._print_warning('Deep frame median of the corrected cube is < 0. ({}), please check the calibration files (dark, bias, flat).')
        
        self.write_fits(
            self._get_deep_frame_path(), deep_frame,
            fits_header=self._get_deep_frame_header(),
            overwrite=True, silent=False)
        
        if self.indexer is not None:
            self.indexer['deep_frame'] = self._get_deep_frame_path()

        
        out_cube.close()
        del out_cube    

##################################################
#### CLASS CalibrationLaser ######################
##################################################

class CalibrationLaser(HDFCube):
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
    
    def _get_calibration_laser_map_header(self):
        """Return the header of the calibration laser map."""
        return (self._get_basic_header('Calibration laser map')
                + self._calibration_laser_header)

    def _get_calibration_laser_fitparams_path(self):
        """Return the path to the file containing the fit parameters
        of the calibration laser cube."""
        return self._data_path_hdr + "calibration_laser_fitparams.fits"
    
    def _get_calibration_laser_fitparams_header(self):
        """Return the header of the file containing the fit parameters
        of the calibration laser cube."""
        return (self._get_basic_header('Calibration laser fit parameters')
                + self._calibration_laser_header)

    def _get_calibration_laser_ils_ratio_path(self):
        """Return the path to the file containing the instrumental
        line shape ratio map (ILS / theoretical ILS) of the
        calibration laser cube."""
        return self._data_path_hdr + "calibration_laser_ils_ratio_map.fits"

    def _get_calibration_laser_ils_ratio_header(self):
        """Return the header of the file containing the instrumental
        line shape ratio map (ILS / theoretical ILS) of the
        calibration laser cube."""
        return (self._get_basic_header('ILS ratio map')
                + self._calibration_laser_header)
    
    def _get_calibration_laser_spectrum_cube_path(self):
        """Return the default path to the reduced calibration laser
        HDF5 cube."""
        return self._data_path_hdr + "calibration_laser_cube.hdf5"

    def _get_calibration_laser_spectrum_frame_header(self, index, axis):
        """Return the header of a calibration spectrum frame.

        :param index: Index of the frame.
        :param axis: Wavenumber axis.
        """
        return (self._get_basic_header('Calibration laser spectrum frame')
                + self._project_header
                + self._calibration_laser_header
                + self._get_basic_frame_header(self.dimx, self.dimy)
                + self._get_basic_spectrum_frame_header(index, axis,
                                                        wavenumber=True))

    def create_calibration_laser_map(self, order=30, step=9765,
                                     get_calibration_laser_spectrum=False,
                                     fast=True):
        """ Create the calibration laser map.

        Compute the spectral cube from the calibration laser cube and
        create the calibration laser map containing the fitted central
        position of the emission line for each pixel of the image
        plane (x/y axes).

        :param order: (Optional) Folding order
        :param step: (Optional) Step size in um
        
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
                        
            # FFT of the interferogram
            column_spectrum = orb.utils.fft.cube_raw_fft(column_data, apod=None)
                
            for ij in range(column_data.shape[0]):
                spectrum_vector = column_spectrum[ij,:]
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
                            cont_guess=[0.])
                        
                        ## fitp = {'lines-params':[[0,1,max_index,1]],
                        ##         'lines-params-err':[[0,0,0,0]]}
                    # or sinc2 fit (slow)
                    else:
                        fitp = orb.fit.fit_lines_in_vector(
                            spectrum_vector, [max_index], fmodel='sinc2',
                            fwhm_guess=fwhm_guess,
                            poly_order=0,
                            signal_range=[range_min, range_max],
                            cont_guess=[0.])
                else:
                    fitp = []
                    
                if (fitp != []):
                    max_index_fit = fitp['lines-params'][0][2]
                    max_array_column[ij] = 1. / orb.cutils.fast_pix2w(
                        np.array([max_index_fit], dtype=float),
                        cm1_axis_min, cm1_axis_step) * 1e7
                    if 'lines-params-err' in fitp:
                        fitparams_column[ij,:] = np.array(
                            list(fitp['lines-params'][0])
                            + list(fitp['lines-params-err'][0]))
                    else:
                        fitparams_column.fill(np.nan)
                        fitparams_column[ij,:5] = fitp['lines-params'][0]
                else:
                    max_array_column[ij] = np.nan
                    fitparams_column[ij,:].fill(np.nan)

            # check if fit range is large enough
            fwhm_median = np.median(fitparams_column[:,3])
            if fwhm_median is not np.nan:
                if (range_max - range_min) < 5. * fwhm_median:
                    import warnings
                    warnings.warn('fit range is not large enough: median fwhm ({}) > 5xrange ({})'.format(
                        fwhm_median, range_max - range_min))
                    
            if not get_calibration_laser_spectrum:
                return max_array_column, fitparams_column
            else:
                return max_array_column, fitparams_column, column_spectrum

        self._print_msg("Computing calibration laser map")

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
        
        self._print_msg('FWHM guess: {} pixels, {} cm-1'.format(
            fwhm_guess,
            fwhm_guess_cm1))
        
        out_cube = OutHDFQuadCube(
            self._get_calibration_laser_spectrum_cube_path(),
            (self.dimx, self.dimy, self.dimz),
            self.QUAD_NB,
            reset=True)

        fitparams = np.empty((self.dimx, self.dimy, 10), dtype=float)
        max_array = np.empty((self.dimx, self.dimy), dtype=float)
        
        for iquad in range(0, self.QUAD_NB):
            x_min, x_max, y_min, y_max = self.get_quadrant_dims(iquad)
            iquad_data = self.get_data(x_min, x_max, 
                                       y_min, y_max, 
                                       0, self.dimz)
            # init multiprocessing server
            job_server, ncpus = self._init_pp_server()

            progress = ProgressBar(x_max - x_min)
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
                    modules=("numpy as np",
                             "math",
                             "import orb.utils.fft",
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
                    iquad+1L, self.QUAD_NB, ii))
            self._close_pp_server(job_server)
            progress.end()

            if get_calibration_laser_spectrum:
                # save data
                self._print_msg('Writing quad {}/{} to disk'.format(
                    iquad+1, self.QUAD_NB))
                write_start_time = time.time()
                out_cube.write_quad(iquad, data=iquad_data)
                self._print_msg('Quad {}/{} written in {:.2f} s'.format(
                    iquad+1, self.QUAD_NB, time.time() - write_start_time))
            

        out_cube.close()
        del out_cube

        # Write uncorrected calibration laser map to disk (in case the
        # correction does not work)
        self.write_fits(self._get_calibration_laser_map_path(), max_array,
                        fits_header=self._get_calibration_laser_map_header(),
                        overwrite=self.overwrite)

        # Correct non-fitted values by interpolation
        max_array = orb.utils.image.correct_map2d(max_array, bad_value=np.nan)
        max_array = orb.utils.image.correct_map2d(max_array, bad_value=0.)

        # Re-Write calibration laser map to disk
        self.write_fits(self._get_calibration_laser_map_path(), max_array,
                        fits_header=self._get_calibration_laser_map_header(),
                        overwrite=self.overwrite)

        # write fit params
        self.write_fits(
            self._get_calibration_laser_fitparams_path(), fitparams,
            fits_header=self._get_calibration_laser_fitparams_header(),
            overwrite=self.overwrite)

        # write ils_ratio
        ils_ratio = fitparams[:,:,3] / fwhm_guess
        self.write_fits(
            self._get_calibration_laser_ils_ratio_path(), ils_ratio,
            fits_header=self._get_calibration_laser_ils_ratio_header(),
            overwrite=self.overwrite)

        if self.indexer is not None:
            self.indexer['calibration_laser_map'] = (
                self._get_calibration_laser_map_path())


##################################################
#### CLASS Interferogram #########################
##################################################

class Interferogram(HDFCube):
    """ORBS interferogram processing class.

    .. note:: Interferogram data is defined as data already processed
       (corrected and aligned frames) by :class:`process.RawData` and
       ready to be transformed to a spectrum by a Fast Fourier
       Transform (FFT).
    """

    def _get_binned_phase_cube_path(self):
        """Return path to the binned phase cube.
        """
        return self._data_path_hdr + "binned_phase_cube.fits"

    def _get_binned_interferogram_cube_path(self):
        """Return path to the binned interferogram cube.
        """
        return self._data_path_hdr + "binned_interferogram_cube.fits"

    def _get_binned_calibration_laser_map_path(self):
        """Return path to the binned calibration laser map
        """
        return self._data_path_hdr + "binned_calibration_laser_map.fits"

    def _get_phase_map_path(self, order, err=False):
        """Return path to phase map

        :param order: Order of the phase map.
        
        :param err: (Optional) If True, map is an error map (default
          False).
        """
        if err: map_type = '.err'
        else: map_type = ''
        return self._data_path_hdr + "phase_map_order_{}{}.fits".format(
            int(order), map_type)

    def _get_phase_map_header(self, order, err=False):
        """Return phase map header
    
        :param order: Order of the phase map.
        
        :param err: (Optional) If True, map is an error map (default
          False).

        """
        if err: map_type = '(err)'
        else: map_type = ''
        return (self._get_basic_header('Phase map order {} {}'.format(
            int(order), map_type))
                + self._project_header)

    def _get_transmission_vector_path(self):
        """Return the path to the transmission vector"""
        return self._data_path_hdr + "transmission_vector.fits"

    def _get_transmission_vector_header(self):
        """Return the header of the transmission vector"""
        return (self._get_basic_header('Transmission vector')
                + self._project_header)
    
    def _get_stray_light_vector_path(self):
        """Return the path to the stray light vector"""
        return self._data_path_hdr + "stray_light_vector.fits"

    def _get_stray_light_vector_header(self):
        """Return the header of the stray light vector"""
        return (self._get_basic_header('stray light vector')
                + self._project_header)

    def _get_extracted_star_spectra_path(self):
        """Return the path to the extracted star spectra"""
        return self._data_path_hdr + "extracted_star_spectra.fits"

    def _get_extracted_star_spectra_header(self):
        """Return the header to the extracted star spectra"""
        return (self._get_basic_header('Extracted star spectra')
                + self._project_header
                + self._calibration_laser_header
                + self._get_fft_params_header('2.0'))
    
    def _get_corrected_interferogram_cube_path(self):
        """Return the default path to a spectrum HDF5 cube."""
        return self._data_path_hdr + 'corrected_interferogram_cube.hdf5'

    def _get_corrected_interferogram_frame_header(self):
        """Return the header of a corrected interferogram frame"""
        return (self._get_basic_header('Corrected interferogram frame')
                + self._project_header
                + self._get_basic_frame_header(self.dimx, self.dimy))
       

    def _get_spectrum_cube_path(self, phase=False):
        """Return the default path to a spectrum HDF5 cube.

        :param phase: (Optional) If True the path is changed for a
          phase cube (default False).
        """
        if not phase: cube_type = "spectrum"
        else: cube_type = "phase"
        return self._data_path_hdr + cube_type + '.hdf5'

    def _get_spectrum_frame_header(self, frame_index, axis,
                                   apodization_function, phase=False,
                                   wavenumber=False):
        """Return the header of the spectral frames.
        
        :param frame_index: Index of the frame.
        
        :param axis: Spectrum axis (must be in wavelength or in
          wavenumber).
        
        :param phase: (Optional) If True the path is changed for a
          phase cube (default False).

        :param wavenumber: (Optional) If True the axis is considered
          to be in wavenumber. If False the axis is considered to be
          in wavelength (default False).
        """
        if not phase: cube_type = "Spectrum"
        else: cube_type = "Phase"
        
        return (self._get_basic_header("%s frame"%cube_type)
                + self._project_header
                + self._calibration_laser_header
                + self._get_basic_frame_header(self.dimx, self.dimy)
                + self._get_basic_spectrum_frame_header(frame_index, axis,
                                                        wavenumber=wavenumber)
                + self._get_fft_params_header(apodization_function))
 
    def _get_spectrum_header(self, axis, apodization_function,
                             phase=False, wavenumber=False):
        """Return the header of the spectal cube.
        
        :param axis: Spectrum axis (must be in wavelength or in
          wavenumber).
              
        :param phase: (Optional) If True the path is changed for a
          phase cube (default False).

        :param wavenumber: (Optional) If True the axis is considered
          to be in wavenumber. If False the axis is considered to be
          in wavelength (default False).
        """
        if not phase: cube_type = "Spectrum"
        else: cube_type = "Phase"
        
      
        header = self._get_basic_header('%s cube'%cube_type)
        
        return (header
                + self._project_header
                + self._calibration_laser_header
                + self._get_basic_frame_header(self.dimx, self.dimy)
                + self._get_basic_spectrum_cube_header(axis,
                                                       wavenumber=wavenumber)
                + self._get_fft_params_header(apodization_function))

    def create_correction_vectors(self, star_list_path,
                                  fwhm_arc, fov, profile_name='gaussian',
                                  moffat_beta=3.5, step_number=None,
                                  bad_frames_vector=[],
                                  aperture_photometry=True):
        """Create a sky transmission vector computed from star
        photometry and an stray light vector computed from the median
        of the frames.

        :param star_list_path: Path to a list of star positions.
        
        :param box_size: (Optional) The size of the box in pixel
            around each given star used to fit a 2D gaussian (default
            15 pixels). Choose it to be between 3 and 6 times the
            FWHM.

        :param step_number: (Optional) 'Full' number of steps if the
          cube was complete. Might be different from the 'real' number
          of steps obtained. Helps in finding ZPD (default None).

        :param bad_frames_vector: (Optional) Contains the index of the
          frames considered as bad (default []).

        :param aperture_photometry: If True, flux of stars is computed
          by aperture photometry. Else, The flux is evaluated given
          the fit parameters.
        
        .. note:: The sky transmission vector gives the absorption
          caused by clouds or airmass variation.

        .. note:: The stray light vector gives the counts added
          homogeneously to each frame caused by a cloud reflecting
          light coming from the ground, the moon or the sun.

        .. warning:: This method is intented to be used to correct a
          'single camera' interferogram cube. In the case of a merged
          interferogram this is already done by the
          :py:meth:`process.InterferogramMerger.merge` with a far
          better precision (because both cubes are used to compute it)
        """
        # Length ratio of the ZPD over the entire cube. This is used
        # to correct the external illumination vector
        ZPD_SIZE = float(self._get_tuning_parameter('ZPD_SIZE', 0.20))
        
        # number of pixels used on each side to smooth the
        # transmission vector
        SMOOTH_DEG = int(self._get_tuning_parameter('SMOOTH_DEG', 0))

        def _sigmean(frame):
            return orb.utils.stats.robust_mean(orb.utils.stats.sigmacut(frame))
        
        self._print_msg("Creating correction vectors", color=True)


        if aperture_photometry:
            self._print_msg('Star flux evaluated by aperture photometry')
            photometry_type = 'aperture_flux'
        else:
            self._print_msg('Star flux evaluated from fit parameters')
            photometry_type = 'flux'

        ## Computing stray light vector
        self._print_msg("Computing stray light vector")
        # Multiprocessing server init
        job_server, ncpus = self._init_pp_server() 
        
        stray_light_vector = np.empty(self.dimz, dtype=float)
        progress = ProgressBar(self.dimz)
        for ik in range(0, self.dimz, ncpus):
            # No more jobs than frames to compute
            if (ik + ncpus >= self.dimz): 
                ncpus = self.dimz - ik

            jobs = [(ijob, job_server.submit(
                _sigmean,
                args=(self.get_data_frame(ik+ijob),),
                modules=('import orb.utils.stats',)))
                    for ijob in range(ncpus)]

            for ijob, job in jobs:
                stray_light_vector[ik+ijob] = job()
                
            progress.update(ik, info='Computing frame %d'%ik)
            
        self._close_pp_server(job_server)
        progress.end()

        ## get stars photometry to compute transmission vector
        self._print_msg("Computing transmission vector")
        astrom = Astrometry(self, fwhm_arc, fov,
                            profile_name=profile_name,
                            moffat_beta=moffat_beta,
                            data_prefix=self._data_prefix,
                            star_list_path=star_list_path,
                            tuning_parameters=self._tuning_parameters,
                            config_file_name=self.config_file_name,
                            ncpus=self.ncpus)

        astrom.fit_stars_in_cube(local_background=True,
                                 fix_aperture_size=True,
                                 precise_guess=True,
                                 multi_fit=True, save=True)
        
        astrom.load_fit_results(astrom._get_fit_results_path())
        
        photom = astrom.fit_results[:,:,photometry_type]

        for iph in range(photom.shape[0]):
            photom[iph,:] /= np.median(photom[iph,:])
        
        transmission_vector = np.array(
            [orb.utils.stats.robust_mean(orb.utils.stats.sigmacut(photom[:,iz]))
             for iz in range(self.dimz)])
        
        # correct for zeros, bad frames and NaN values
        bad_frames_vector = [bad_frame
                             for bad_frame in bad_frames_vector
                             if (bad_frame < step_number and bad_frame >= 0)]
        
        transmission_vector[bad_frames_vector] = np.nan
        stray_light_vector[bad_frames_vector] = np.nan
        
        transmission_vector = orb.utils.vector.correct_vector(
            transmission_vector, bad_value=0., polyfit=False, deg=1)
        stray_light_vector = orb.utils.vector.correct_vector(
            stray_light_vector, bad_value=0., polyfit=False, deg=1)
        
        # correct for ZPD
        zmedian = self.get_zmedian(nozero=True)
        zpd_index = orb.utils.fft.find_zpd(zmedian,
                                           step_number=step_number)
        self._print_msg('ZPD index: %d'%zpd_index)
        
        zpd_min = zpd_index - int((ZPD_SIZE * step_number)/2.)
        zpd_max = zpd_index + int((ZPD_SIZE * step_number)/2.) + 1
        if zpd_min < 1: zpd_min = 1
        if zpd_max > self.dimz:
            zpd_max = self.dimz - 1
        
        transmission_vector[zpd_min:zpd_max] = 0.
        transmission_vector = orb.utils.vector.correct_vector(
            transmission_vector, bad_value=0., polyfit=False, deg=1)
        stray_light_vector[zpd_min:zpd_max] = 0.
        stray_light_vector = orb.utils.vector.correct_vector(
            stray_light_vector, bad_value=0., polyfit=False, deg=1)
        
        # smooth
        if SMOOTH_DEG > 0:
            transmission_vector = orb.utils.vector.smooth(transmission_vector,
                                                          deg=SMOOTH_DEG)
            stray_light_vector = orb.utils.vector.smooth(stray_light_vector,
                                                         deg=SMOOTH_DEG)
            
        # normalization of the transmission vector
        transmission_vector /= orb.utils.stats.robust_median(
            transmission_vector)

        # save correction vectors
        self.write_fits(self._get_transmission_vector_path(),
                        transmission_vector,
                        fits_header= self._get_transmission_vector_header(),
                        overwrite=self.overwrite)
        if self.indexer is not None:
            self.indexer['transmission_vector'] = (
                self._get_transmission_vector_path())
        
        self.write_fits(self._get_stray_light_vector_path(),
                        stray_light_vector,
                        fits_header= self._get_stray_light_vector_header(),
                        overwrite=self.overwrite)
        if self.indexer is not None:
            self.indexer['stray_light_vector'] = (
                self._get_stray_light_vector_path())

    def correct_interferogram(self, transmission_vector_path,
                              stray_light_vector_path):
        """Correct an interferogram cube for for variations
        of sky transmission and stray light.

        :param sky_transmission_vector_path: Path to the transmission
          vector.All the interferograms of the cube are divided by
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

        self._print_msg('Correcting interferogram', color=True)

        # Avoid transmission correction (useful for testing purpose)
        NO_TRANSMISSION_CORRECTION = bool(int(
            self._get_tuning_parameter('NO_TRANSMISSION_CORRECTION', 0)))
        if NO_TRANSMISSION_CORRECTION:
            self._print_warning('No transmission correction')
            
        transmission_vector = self.read_fits(transmission_vector_path)
        if NO_TRANSMISSION_CORRECTION:
            transmission_vector.fill(1.)
            
        stray_light_vector = self.read_fits(stray_light_vector_path)
        
        # Multiprocessing server init
        job_server, ncpus = self._init_pp_server() 

        # creating output hdfcube
        out_cube = OutHDFCube(self._get_corrected_interferogram_cube_path(),
                              (self.dimx, self.dimy, self.dimz),
                              overwrite=self.overwrite)
        
        # Interferogram creation
        progress = ProgressBar(self.dimz)
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
                modules=('import numpy as np',)))
                    for ijob in range(ncpus)]

            for ijob, job in jobs:
                frames[:,:,ijob] = job()
                
            for ijob in range(ncpus):
                
                out_cube.write_frame(
                    ik+ijob,
                    data=frames[:,:,ijob],
                    header=self._get_corrected_interferogram_frame_header())
            progress.update(ik, info="Correcting frame %d"%ik)

        progress.end()

        if self.indexer is not None:
            self.indexer['corr_interfero_cube'] = (
                self._get_corrected_interferogram_cube_path())
            
        self._close_pp_server(job_server)
            

    def compute_spectrum(self, calibration_laser_map_path, step,
                         order, nm_laser, zpd_shift=None,
                         phase_correction=True,
                         bad_frames_vector=None, window_type=None,
                         phase_cube=False, phase_map_paths=None,
                         filter_name=None,
                         wave_calibration=False,
                         balanced=True, fringes=None,
                         wavenumber=False):
        
        """Compute the spectrum from the corrected interferogram
        frames. Can be used to compute spectrum for camera 1, camera 2
        or merged interferogram.

        :param calibration_laser_map_path: Path to the calibration
          map.

        :param order: Folding order
          
        :param step: Step size in nm

        :param zpd_shift: (Optional) Shift of the ZPD in
          frames. Automaticaly computed if none given.

        :param window_type: (Optional) Apodization window to be used
          (Default None, no apodization)

        :param phase_correction: (Optional) If False, no phase
          correction will be done and the resulting spectrum will be
          the absolute value of the complex spectrum (default True).

        :param bad_frames_vector: (Optional) Mask-like vector
          containing ones for bad frames. Bad frames are replaced by
          zeros using a special function that smoothes transition
          between good parts and zeros (default None).

        :param phase_cube: (Optional) If True, only the phase cube is
          returned (default False).

        :param phase_map_paths: (Optional) List of path to the phase
          maps. Each path must point to a FITS file that contains a
          map or a number. If a map is given it must have the same
          dimensions as the frames of the interferogram cube (default
          None).

        :param filter_name: (Optional) Name of the filter used.
        
        :param balanced: (Optional) If False, the interferogram is
          considered as unbalanced. It is flipped before its
          transformation to get a positive spectrum. Note that a
          merged interferogram is balanced (default True).
    
        :param fringes: (Optional) If not None, must be an array
          giving for each fringe to remove its frequency and
          intensity. The array must be like [[freq1, amp1], [freq2,
          amp2], [freq3, amp3], ...]. Fringes are removed by dividing
          the interferograms by a sinusoidal function representing a
          periodically variable modulation efficiency (default None).
          
        :param wavenumber: (Optional) If True, the returned spectrum
          is projected onto its original wavenumber axis (emission
          lines and especially unapodized sinc emission lines are thus
          symetric which is not the case if the spectrum is projected
          onto a, more convenient, regular wavelength axis) (default
          False).

        :param wave_calibration: (Optional) If True wavelength
          calibration is done (default False).
    
        .. Note:: External phase computation:

          In order to achieve a better phase correction it can be
          useful to compute some of the phase coefficients from an
          external source.
        
        .. Note:: Spectrum computation walks through 8 steps:
        
           1. Mean interferogram subtraction to suppress the
              zero-frequency term in the spectrum
           
           2. Low order polynomial subtraction to suppress low
              frequency noise in the spectrum
           
           3. Apodization (the user can choose which apodization
              function to use)
           
           4. Zero-padding to have two times more points in the
              interferogram in order to keep the same resolution during
              the Fourier transform.

           5. ZPD shift to correct for a non-centered ZPD.
           
           6. Fast Fourier Transform of the interferogram
           
           7. Phase correction (if the user chooses to get the real
              part of the spectrum with phase correction instead of the
              power spectrum)

           8. Wavelength correction using the data obtained with the
              calibration cube.

        .. seealso:: :meth:`orb.utils.fft.transform_interferogram`
        .. seealso:: :class:`process.Phase`
        """

        def _compute_spectrum_in_column(nm_laser, 
                                        calibration_laser_map_column, step,
                                        order, 
                                        data, window_type, zpd_shift,
                                        phase_correction,
                                        wave_calibration,
                                        bad_frames_vector, phase_map_cols,
                                        return_phase,
                                        balanced, filter_min, filter_max,
                                        fringes, wavenumber, phf):
            """Compute spectrum in one column. Used to parallelize the
            process"""
            dimz = data.shape[1]
            spectrum_column = np.zeros_like(data, dtype=complex)
  
                
            for ij in range(data.shape[0]):
                # throw out interferograms with less than half non-zero values
                # (zero values are considered as bad points : cosmic rays, bad
                # frames etc.)
                if len(np.nonzero(data[ij,:])[0]) > dimz/2.:
                    
                    # Compute external phase vector from given coefficients
                    if phase_map_cols is not None:
                        coeffs_list = list()
                        for phase_map_col in phase_map_cols:
                            if np.size(phase_map_col) == spectrum_column.shape[0]:
                                coeffs_list.append(phase_map_col[ij])
                            else:
                                coeffs_list.append(phase_map_col)

                        ext_phase = np.polynomial.polynomial.polyval(
                            np.arange(dimz), coeffs_list)
                            
                    else:
                        ext_phase = None
                
                    interf = np.copy(data[ij,:])

                    # defringe
                    if fringes is not None and not return_phase:
                        for ifringe in range(len(fringes)):
                            fringe_vector = orb.utils.fft.variable_me(
                                dimz, [fringes[ifringe, 0],
                                       fringes[ifringe, 1], 0.])
                            
                            interf = interf / fringe_vector

                    # zeros are replaced by NaNs
                    interf[np.nonzero(interf == 0)] = np.nan
                            
                    # Spectrum computation                    
                    spectrum_column[ij,:] = (
                        orb.utils.fft.transform_interferogram(
                            interf, nm_laser, calibration_laser_map_column[ij],
                            step, order, window_type, zpd_shift,
                            bad_frames_vector=bad_frames_vector,
                            phase_correction=phase_correction,
                            wave_calibration=wave_calibration,
                            ext_phase=ext_phase,
                            return_phase=return_phase,
                            balanced=balanced,
                            wavenumber=wavenumber,
                            return_complex=True,
                            high_order_phase=phf))

                    if not phase_correction:
                        spectrum_column[ij,:] = np.abs(spectrum_column[ij,:])
                        
            return spectrum_column

        def get_phase_map_cols(phase_maps, x, ymin, ymax):
            if phase_maps is None:
                return None
            phase_map_cols = list()
            for phase_map in phase_maps:
                
                if np.size(phase_map) > 1:
                    phase_map_cols.append(phase_map[x, ymin:ymax])
                else:
                    phase_map_cols.append(phase_map)
            return phase_map_cols

            
        order = float(order)
        step = float(step)

        if not phase_cube:
            self._print_msg("Computing spectrum", color=True)
        else: 
            self._print_msg("Computing phase", color=True)

        phf = None
        if phase_correction:
            # get high order phase
            if filter_name is not None:
                phf = PhaseFile(filter_name,
                                config_file_name=self.config_file_name,
                                ncpus=self.ncpus)
                self._print_msg('Phase file: {}'.format(phf.improved_path))
        else:
            self._print_warning('No phase correction')
                
                

        ## Calibration map loading and interpolation
        self._print_msg("loading calibration laser map")
        calibration_laser_map = self.read_fits(calibration_laser_map_path)
        if (calibration_laser_map.shape[0] != self.dimx):
            calibration_laser_map = orb.utils.image.interpolate_map(
                calibration_laser_map, self.dimx, self.dimy)

        if not wave_calibration:
            self._print_warning('No wavelength/wavenumber calibration')
        
        #############################
        ## Note: variable names are all "spectrum" related even if it
        ## is possible to get only the phase cube. Better for the
        ## clarity of the code
        #############################

        ## Searching ZPD shift 
        if zpd_shift is None:
            zpd_shift = orb.utils.fft.find_zpd(self.get_zmedian(nozero=True),
                                               return_zpd_shift=True)
            
        self._print_msg("Zpd will be shifted from %d frames"%zpd_shift)

        ## Loading phase maps
        if (phase_map_paths is not None and phase_correction):
            phase_maps = list()
            for phase_map_path in phase_map_paths:
                phase_maps.append(self.read_fits(phase_map_path))
                self._print_msg('Loaded phase map: {}'.format(phase_map_path))
            
        else:
            phase_maps = None
            
        ## Check spectrum polarity
            
        # Note: The Oth order phase map is defined modulo pi. But a
        # difference of pi in the 0th order of the phase vector change
        # the polarity of the spectrum (the returned spectrum is
        # reversed). As there is no way to know the correct phase,
        # spectrum polarity must be tested. We get the mean
        # interferogram and transform it to check.
        if (phase_maps is not None and phase_correction):
            
            self._print_msg("Check spectrum polarity with phase correction")
            
            # get mean interferogram
            xmin, xmax, ymin, ymax = orb.utils.image.get_box_coords(
                self.dimx/2, self.dimy/2,
                int(0.02*self.dimx),
                0, self.dimx,
                0, self.dimy)
            mean_interf = bn.nanmedian(bn.nanmedian(
                self.get_data(xmin, xmax, ymin, ymax, 0, self.dimz),
                axis=0), axis=0)
            
            mean_calib = np.nanmean(calibration_laser_map[xmin:xmax, ymin:ymax])
            
            # create mean phase vector
            coeffs_list_mean = list()
            for phase_map in phase_maps:
                if len(phase_map.shape) == 2:
                    coeffs_list_mean.append(np.nanmean(phase_map[xmin:xmax, ymin:ymax]))
                else:
                    coeffs_list_mean.append(phase_map)
           
            mean_phase_vector = np.polynomial.polynomial.polyval(
                np.arange(self.dimz),
                coeffs_list_mean)

            # zeros are replaced by NANs
            mean_interf[np.nonzero(mean_interf == 0)] = np.nan
            
            # transform interferogram and check polarity
            if phf is not None:
                mean_high_phase = phf.get_improved_phase(mean_calib)
            else:
                mean_high_phase = None
            mean_spectrum = orb.utils.fft.transform_interferogram(
                mean_interf, nm_laser, mean_calib, step, order, '2.0', zpd_shift,
                phase_correction=phase_correction, wave_calibration=wave_calibration,
                ext_phase=mean_phase_vector,
                return_phase=False, balanced=balanced, wavenumber=wavenumber,
                return_complex=True, high_order_phase=mean_high_phase)

            if np.nanmean(mean_spectrum.real) < 0:
                self._print_msg("Negative polarity : 0th order phase map has been corrected (add PI)")
                phase_maps[0] += math.pi

            if (orb.utils.fft.spectrum_mean_energy(mean_spectrum.imag)
                > .5 * orb.utils.fft.spectrum_mean_energy(mean_spectrum.real)):
                self._print_warning("Too much energy in the imaginary part, check the phase correction")
      
        ## Spectrum computation

        # Print some informations about the spectrum transformation
        
        self._print_msg("Apodization function: %s"%window_type)
        self._print_msg("Folding order: %f"%order)
        self._print_msg("Step size: %f"%step)
        self._print_msg("Bad frames: %s"%str(np.nonzero(bad_frames_vector)[0]))
        self._print_msg("Wavenumber output: {}".format(wavenumber))
        if fringes is not None:
            if not phase_cube:
                self._print_msg("Fringes:")
                for ifringe in range(fringes.shape[0]):
                    self._print_msg("fringe %d: %s"%(ifringe, str(fringes[ifringe,:])))
            else:
                self._print_warning(
                    "Defringing will not be done before phase computation")
        
        # get filter min and filter max edges for weights definition
        # in case no external phase is provided
        if phase_correction and filter_name is not None:
            fif = FilterFile(filter_name,
                             config_file_name=self.config_file_name,
                             ncpus=self.ncpus)
            (filter_nm, filter_trans,
             filter_min, filter_max) = fif.read_filter_file()
        else:
            filter_min = None
            filter_max = None

        if not wavenumber:
            axis = orb.utils.spectrum.create_nm_axis(self.dimz, step, order)
        else:
            axis = orb.utils.spectrum.create_cm1_axis(self.dimz, step, order)

        out_cube = OutHDFQuadCube(
            self._get_spectrum_cube_path(phase=phase_cube),
            (self.dimx, self.dimy, self.dimz),
            self.QUAD_NB,
            overwrite=self.overwrite)

        out_cube.append_header(pyfits.Header(self._get_spectrum_header(
            axis, window_type, phase=phase_cube,
            wavenumber=wavenumber)))

        for iquad in range(0, self.QUAD_NB):
            x_min, x_max, y_min, y_max = self.get_quadrant_dims(iquad)
            iquad_data = self.get_data(x_min, x_max, 
                                       y_min, y_max, 
                                       0, self.dimz)
            iquad_data = iquad_data.astype(complex)
                
            # multi-processing server init
            job_server, ncpus = self._init_pp_server()
            progress = ProgressBar(x_max - x_min)
            for ii in range(0, x_max - x_min, ncpus):
                # no more jobs than columns
                if (ii + ncpus >= x_max - x_min): 
                    ncpus = x_max - x_min - ii
                    
                # jobs creation
                jobs = [(ijob, job_server.submit(
                    _compute_spectrum_in_column,
                    args=(nm_laser, 
                          calibration_laser_map[x_min + ii + ijob,
                                                y_min:y_max], 
                          step, order, iquad_data[ii+ijob,:,:].real, 
                          window_type, zpd_shift,
                          phase_correction, wave_calibration,
                          bad_frames_vector,
                          get_phase_map_cols(
                              phase_maps, x_min + ii + ijob, y_min, y_max),
                          phase_cube, balanced,
                          filter_min, filter_max,
                          fringes, wavenumber, phf), 
                    modules=("import numpy as np", "import math",  
                             "from scipy import interpolate", 
                             "from scipy import fftpack, signal", 
                             "import orb.utils.filters",
                             "import orb.utils.fft")))
                        for ijob in range(ncpus)]

                for ijob, job in jobs:
                    # spectrum comes in place of the interferograms
                    # to avoid using too much memory
                    iquad_data[ii+ijob,:,:] = job()
                
                progress.update(ii, info="Quad %d/%d column : %d"%(
                        iquad+1L, self.QUAD_NB, ii))
            self._close_pp_server(job_server)
            progress.end()
            
            # save data
            self._print_msg('Writing quad {}/{} to disk'.format(
                iquad+1, self.QUAD_NB))
            write_start_time = time.time()
            out_cube.write_quad(
                iquad, data=iquad_data,
                force_float32=False, force_complex64=True)
            self._print_msg('Quad {}/{} written in {:.2f} s'.format(
                iquad+1, self.QUAD_NB, time.time() - write_start_time))
            
            
        out_cube.close()
        del out_cube
            
        # Create indexer key
        if not phase_cube:
            self._print_msg("Spectrum computed")
            cube_file_key = 'spectrum_cube'
            
        else:
            self._print_msg("Phase computed")
            cube_file_key = 'phase_cube'
            

        if self.indexer is not None:
                self.indexer[cube_file_key] = self._get_spectrum_cube_path(
                    phase=phase_cube)


    def create_phase_maps(self, step, order, zpd_shift=None, binning=4,
                          filter_name=None, calibration_laser_map_path=None,
                          nm_laser=None, fit_order=1):
        """Create phase maps directly from the interferogram cube.

        Cube is binned before each phase vector is fitted.

        The order 0 map is then computed by minimizing the imaginary
        part of the spectrum with the order 1 fixed.

        :param step: Step size in nm.
        
        :param order: Folding order.
        
        :param zpd_shift: (Optional) ZPD shift. If None, ZPD shift is
          computed from the median interferogram vector (default None).

        :param binning: (Optional) Binning factor of the cube to
          accelerate the phase map computation (default 4).

        :param filter_name: (Optional) Filter name (default None).

        :param calibration_laser_map_path: (Optional) Path to a
          calibration laser map (default None). Must be set if
          filter_name is set.

        :param nm_laser: (Optional) Calibration laser wavelength in nm
          (default None) Must be set if calibration_laser_map_path is
          set.

        :param fit_order: (Optional) Order of the polynomial used to
          fit the phase (default 1).
        """

        def fit_phase_in_column(cube_col, step, order, zpd_shift,
                                calib_col, filter_min_cm1, filter_max_cm1,
                                nm_laser, fit_order, phf):
        
            RANGE_BORDER_COEFF = 0.1
            
            warnings.simplefilter('ignore', RuntimeWarning)

            phase_col = np.empty_like(cube_col)
            fit_coeffs_col = np.empty((cube_col.shape[0], fit_order+1),
                                      dtype=float)
            fit_err_col = np.empty(cube_col.shape[0], dtype=float)
            fit_coeffs_col.fill(np.nan)
            fit_err_col.fill(np.nan)
            
            for ij in range(cube_col.shape[0]):

                interf = cube_col[ij,:]
                # zeros are replaced by nans
                interf[np.nonzero(interf == 0)] = np.nan

                # compute phase
                ifft = orb.utils.fft.transform_interferogram(
                    interf, 1., 1., step, order, '2.0',
                    zpd_shift, wavenumber=True,
                    return_complex=True,
                    phase_correction=False)
                    
                if ifft is not None:
                
                    iphase = np.unwrap(np.angle(ifft))
                    if int(order) & 1: iphase = iphase[::-1]

                    phase_col[ij,:] = np.copy(iphase)

                    # compute weights on phase
                    if filter_min_cm1 is None:
                        weights = np.abs(ifft)
                        weights /= np.nanmean(weights)
                        weights[np.nonzero(weights < 0.5)] = 1e-35
                        weights = weights**2.
                    else:
                        (weights,
                         filter_range) = orb.utils.filters.compute_weights(
                            calib_col[ij], nm_laser, cube_col.shape[1],
                            step, order, RANGE_BORDER_COEFF,
                            filter_min_cm1, filter_max_cm1)

                    # fit phase
                    if (not np.any(np.isnan(weights))
                        and not np.any(np.isnan(iphase))):
                        try:
                            if phf is not None:
                                high_phase = phf.get_improved_phase(
                                    calib_col[ij])
                                # high order phase interpolation to real axis
                                cm1_axis = orb.utils.spectrum.create_cm1_axis(
                                    iphase.shape[0], step, order,
                                    corr=calib_col[ij]/nm_laser).astype(
                                    np.float64)
                                ihigh_phase = high_phase(cm1_axis)
                                # high order phase is subtracted
                                phase_to_fit = iphase - ihigh_phase
                            else:
                                phase_to_fit = iphase

                                
                            # polynomial fit
                            def model(x, *p):
                                return np.polynomial.polynomial.polyval(x, p)

                            x= np.arange(iphase.shape[0])
                            pfit, pcov = optimize.curve_fit(
                                model, x, iphase, [0., 0.], 1/weights)
                            perr = np.sqrt(np.diag(pcov))
                        
                        except Exception, e:
                            print 'Exception occured during phase fit: ', e
                            pfit = None
                    else:
                        pfit = None

                    if pfit is not None:
                        fit_coeffs_col[ij,:] = np.array(pfit)
                        fit_err_col[ij] = perr[1]
                    
            return fit_coeffs_col, fit_err_col, phase_col
        
        def optimize_phase_in_column(cube_col, step, order, zpd_shift,
                                     phase_maps, column_nb, calib_col, nm_laser,
                                     filter_min_cm1, filter_max_cm1, phf):

            
            RANGE_BORDER_COEFF = 0.1

            warnings.simplefilter('ignore', RuntimeWarning)
            order0_col = np.empty(cube_col.shape[0], dtype=float)
            err_col = np.empty(cube_col.shape[0], dtype=float)
            order0_col.fill(np.nan)
            err_col.fill(np.nan)
            
            for ij in range(cube_col.shape[0]):
                if np.all(cube_col[ij,:] == 0):
                    cube_col[ij,:].fill(np.nan)
                guess = list()
                fixed_params = list()
                guess.append(0.)
                fixed_params.append(0)
                
                for ik in range(len(phase_maps)):
                    if np.size(phase_maps[ik]) > 1:
                        guess.append(phase_maps[ik][column_nb,ij])
                    else:
                        guess.append(phase_maps[ik])
                    fixed_params.append(1)
                    
                if not np.all(np.isnan(cube_col[ij,:])):
                    (weights,
                     filter_range) = orb.utils.filters.compute_weights(
                        calib_col[ij], nm_laser, cube_col.shape[1],
                        step, order, RANGE_BORDER_COEFF,
                        filter_min_cm1, filter_max_cm1)
                    
                    result = orb.utils.fft.optimize_phase(
                        cube_col[ij,:],
                        step, order, zpd_shift,
                        calib_col[ij], nm_laser,
                        guess=guess, return_coeffs=True,
                        fixed_params=fixed_params,
                        weights=weights,
                        high_order_phase=phf)
                    if result is not None:
                        order0_col[ij] = result[0][0]
                        err_col[ij] = result[1]
                    
            return order0_col, err_col

        self._print_msg('Computing phase maps up to order {}'.format(fit_order))

        # get high order phase
        if filter_name is not None:
            phf = PhaseFile(filter_name,
                            config_file_name=self.config_file_name,
                            ncpus=self.ncpus)
            self._print_msg('Phase file: {}'.format(phf.improved_path))
        else:
            phf = None

        if ((filter_name is not None and calibration_laser_map_path is None)
            or (filter_name is None and calibration_laser_map_path is not None)):
            self._print_error('filter_name and calibration_laser_map_path must be set together or both must be set to None')

        if calibration_laser_map_path is not None and nm_laser is None:
            self._print_error('calibration_laser_map_path and nm_laser must be set together')
            
        if filter_name is not None:
            filter_params = FilterFile(
                filter_name,
                config_file_name=self.config_file_name,
                ncpus=self.ncpus).read_filter_file()
            filter_max_cm1 = orb.utils.spectrum.nm2cm1(filter_params[2])
            filter_min_cm1 = orb.utils.spectrum.nm2cm1(filter_params[3])
        else: filter_max_cm1 = None ; filter_min_cm1 = None

        if zpd_shift is None:
            median_interf = self.get_zmedian()
            zpd_shift = orb.utils.fft.find_zpd(
                median_interf, return_zpd_shift=True)
        
        # binning interferogram cube and calibration laser map
        self.create_binned_interferogram_cube(binning)
        cube_bin = self.read_fits(self._get_binned_interferogram_cube_path())
        
        if calibration_laser_map_path is not None:
            self.create_binned_calibration_laser_map(
                binning, calibration_laser_map_path)
            calibration_laser_map = self.read_fits(
                self._get_binned_calibration_laser_map_path())
        else:
            calibration_laser_map = np.ones((cube_bin.shape[0],
                                             cube_bin.shape[1]), dtype=float)


        # compute orders > 0
        phase_cube = np.empty_like(cube_bin)
        fit_coeffs_map = np.empty((cube_bin.shape[0],
                               cube_bin.shape[1],
                               fit_order+1), dtype=float)
        fit_err_map = np.empty((cube_bin.shape[0],
                                cube_bin.shape[1]), dtype=float)
        fit_coeffs_map.fill(np.nan)
        fit_err_map.fill(np.nan)

        job_server, ncpus = self._init_pp_server()
        progress = ProgressBar(cube_bin.shape[0])
        for ii in range(0, cube_bin.shape[0], ncpus):
            progress.update(ii, info='Computing orders > 0')
            
            # no more jobs than frames to compute
            if (ii + ncpus >= cube_bin.shape[0]):
                ncpus = cube_bin.shape[0] - ii
            jobs = [(ijob, job_server.submit(
                fit_phase_in_column, 
                args=(cube_bin[ii+ijob,:,:],
                      step, order, zpd_shift,
                      calibration_laser_map[ii+ijob,:],
                      filter_min_cm1, filter_max_cm1, nm_laser,
                      fit_order, phf),
                modules=("numpy as np", "import orb.utils.fft",
                         "import warnings", "import orb.cutils",
                         "import orb.utils.filters", "import scipy.interpolate",
                         "from scipy import optimize"))) 
                    for ijob in range(ncpus)]

            for ijob, job in jobs:
                (fit_coeffs_map[ii+ijob,:,:],
                 fit_err_map[ii+ijob,:],
                 phase_cube[ii+ijob,:,:]) = job()

        progress.end()

        # write binned phase cube
        self.write_fits(self._get_binned_phase_cube_path(), phase_cube,
                        overwrite=self.overwrite)
        
        if self.indexer is not None:
            self.indexer['binned_phase_cube'] = (
                self._get_binned_phase_cube_path())
        
        # write phase maps
        for iorder in range(fit_order+1):
            self.write_fits(self._get_phase_map_path(iorder),
                            fit_coeffs_map[:,:,iorder],
                            fits_header=self._get_phase_map_header(iorder),
                            overwrite=self.overwrite)
            if self.indexer is not None:
                self.indexer['phase_map_{}'.format(iorder)] = (
                    self._get_phase_map_path(iorder))

        # write error map
        self.write_fits(self._get_phase_map_path(1, err=True),
                        fit_err_map,
                        fits_header=self._get_phase_map_header(1, err=True),
                        overwrite=self.overwrite)
        if self.indexer is not None:
            self.indexer['phase_map_1_err'] = (
                self._get_phase_map_path(1, err=True))

        # fit order 1 map
        phase_maps = [self.read_fits(self._get_phase_map_path(iorder))
                      for iorder in range(fit_order+1)]

        if fit_order > 0:
            phase_maps[1], _err = orb.utils.image.fit_highorder_phase_map(
                phase_maps[1],
                self.read_fits(
                    self._get_phase_map_path(1, err=True)),
                calibration_laser_map, nm_laser)
            
        # compute average orders > 2:
        if fit_order > 1:
            if len(phase_maps[2:]) > 0:
                self._print_msg('computing mean phase coeffs of orders > 2')
                phase_maps[2:] = orb.utils.fft.compute_phase_coeffs_vector(
                    phase_maps[2:],
                    err_map=self.read_fits(self._get_phase_map_path(1, err=True)))
        
        # computing order 0 map
        order0_map = np.empty((cube_bin.shape[0],
                               cube_bin.shape[1]), dtype=float)
        order0_err_map = np.empty_like(order0_map)
        order0_map.fill(np.nan)
        order0_err_map.fill(np.nan)

        job_server, ncpus = self._init_pp_server()
        progress = ProgressBar(cube_bin.shape[0])
        for ii in range(0, cube_bin.shape[0], ncpus):
            progress.update(ii, info='Computing order 0')
            
            # no more jobs than frames to compute
            if (ii + ncpus >= cube_bin.shape[0]):
                ncpus = cube_bin.shape[0] - ii

            jobs = [(ijob, job_server.submit(
                optimize_phase_in_column, 
                args=(cube_bin[ii+ijob,:,:],
                      step, order, zpd_shift,
                      phase_maps[1:], ii+ijob,
                      calibration_laser_map[ii+ijob,:], nm_laser,
                      filter_min_cm1, filter_max_cm1,
                      phf),
                modules=("numpy as np", "import orb.utils.fft",
                         "import warnings", "import orb.utils.filters"))) 
                    for ijob in range(ncpus)]

            for ijob, job in jobs:
                (order0_map[ii+ijob,:],
                 order0_err_map[ii+ijob,:]) = job()

        progress.end()
        
        # write phase map order 0
        self.write_fits(self._get_phase_map_path(0),
                        order0_map,
                        fits_header=self._get_phase_map_header(0),
                        overwrite=self.overwrite)
        if self.indexer is not None:
            self.indexer['phase_map_0'] = (
                self._get_phase_map_path(0))

        # write residual map
        self.write_fits(self._get_phase_map_path(0, err=True),
                        order0_err_map,
                        fits_header=self._get_phase_map_header(0, err=True),
                        overwrite=self.overwrite)
        if self.indexer is not None:
            self.indexer['phase_map_0_err'] = (
                self._get_phase_map_path(0, err=True))

    def create_binned_calibration_laser_map(self, binning,
                                            calibration_laser_map_path):
        """Create a binned calibration laser map

        :param binning: Binning
        :param calibration_laser_map_path: Calibration laser map path
        """
        # Loading calibration laser map
        self._print_msg("loading calibration laser map")
        calibration_laser_map = self.read_fits(calibration_laser_map_path)

        if (calibration_laser_map.shape[0] != self.dimx):
            calibration_laser_map = orb.utils.image.interpolate_map(
                calibration_laser_map, self.dimx, self.dimy)
            
        if binning > 1:
            calibration_laser_map = orb.utils.image.nanbin_image(
                calibration_laser_map, binning)
        
        # write binned calib map
        self.write_fits(self._get_binned_calibration_laser_map_path(),
                        calibration_laser_map,
                        overwrite=self.overwrite)
        
        if self.indexer is not None:
            self.indexer['binned_calibration_laser_map'] = (
                self._get_binned_calibration_laser_map_path())

    def create_binned_interferogram_cube(self, binning):
        """Create a binned interferogram cube

        :param binning: Binning
        """
        if binning > 1:
            self._print_msg('Binning interferogram cube')
            image0_bin = orb.utils.image.nanbin_image(
                self.get_data_frame(0), binning)
            
            cube_bin = np.empty((image0_bin.shape[0],
                                 image0_bin.shape[1],
                                 self.dimz), dtype=float)
            cube_bin.fill(np.nan)
            cube_bin[:,:,0] = image0_bin
            progress = ProgressBar(self.dimz-1)
            for ik in range(1, self.dimz):
                progress.update(ik, info='Binning cube')
                cube_bin[:,:,ik] = orb.utils.image.nanbin_image(
                    self.get_data_frame(ik), binning)
            progress.end()
          
            
        else:
            cube_bin = self
            self._silent_load = True

        # write binned interferogram cube
        self.write_fits(self._get_binned_interferogram_cube_path(),
                        cube_bin, overwrite=True)
        
        if self.indexer is not None:
            self.indexer['binned_interferogram_cube'] = (
                self._get_binned_interferogram_cube_path())



            
    def create_phase_maps_from_external_source(self, phase_map_order_0_path,
                                               phase_map_order_1_path,
                                               step, order, zpd_shift,
                                               filter_name,
                                               calibration_laser_map_path,
                                               nm_laser, fit_order=1, binning=4):
        """Create phase maps directly from the interferogram cube.

        Cube is binned before each phase vector is fitted.

        The order 0 map is then computed by minimizing the imaginary
        part of the spectrum with the order 1 fixed.

        :param phase_map_order_0_path: Path to the zeroth order phase
          map.

        :param step: Step size in nm.
        
        :param order: Folding order.
        
        :param zpd_shift: ZPD shift.
        
        :param filter_name: Filter name

        :param calibration_laser_map_path: Path to a
          calibration laser map

        :param nm_laser: Calibration laser wavelength in nm
    
        :param binning: (Optional) Binning factor of the cube to
          accelerate the phase map computation (default 4).

        .. warning:: This function cannot be used with SpIOMM.
        """
        # get phase map order 0
        phase_map_order_0 = self.read_fits(phase_map_order_0_path)
        phase_map_order_0 = orb.utils.image.nanbin_image(
                phase_map_order_0, binning)

        # get phase map order 1
        phase_map_order_1 = self.read_fits(phase_map_order_1_path)
        phase_map_order_1 = orb.utils.image.nanbin_image(
                phase_map_order_1, binning)
        
        # get filter parameters
        filter_params = FilterFile(
            filter_name,
            config_file_name=self.config_file_name,
            ncpus=self.ncpus).read_filter_file()
        filter_max_cm1 = orb.utils.spectrum.nm2cm1(filter_params[2])
        filter_min_cm1 = orb.utils.spectrum.nm2cm1(filter_params[3])

        # get high phase 
        phf = PhaseFile(filter_name,
                        config_file_name=self.config_file_name,
                        ncpus=self.ncpus)
        
        # binning interferogram cube and calibration laser map
        self.create_binned_interferogram_cube(binning)
        self.create_binned_calibration_laser_map(
            binning, calibration_laser_map_path)
        
        calibration_laser_map = self.read_fits(
            self._get_binned_calibration_laser_map_path())
        cube_bin = self.read_fits(
            self._get_binned_interferogram_cube_path())

        # 3D optimization
        params_fit = orb.utils.fft.optimize_phase3d(
            cube_bin, step, order, zpd_shift,
            calibration_laser_map, nm_laser,
            phf,
            pm0=np.copy(phase_map_order_0),
            pm1=np.copy(phase_map_order_1))
        
        order0_map = params_fit[0] + phase_map_order_0
        order1_map = params_fit[1] + phase_map_order_1 / self.dimz
        
        # write phase map order 0
        self.write_fits(self._get_phase_map_path(0),
                        order0_map,
                        fits_header=self._get_phase_map_header(0),
                        overwrite=self.overwrite)
        if self.indexer is not None:
            self.indexer['phase_map_0'] = (
                self._get_phase_map_path(0))

        # write phase map order 1
        self.write_fits(self._get_phase_map_path(1),
                        order1_map,
                        fits_header=self._get_phase_map_header(1),
                        overwrite=self.overwrite)
        if self.indexer is not None:
            self.indexer['phase_map_1'] = (
                self._get_phase_map_path(1))




##################################################
#### CLASS InterferogramMerger ###################
##################################################

class InterferogramMerger(Tools):
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
    config_file_name = None
    ncpus = None
    _msg_class_hdr = None
    _data_path_hdr = None
    _project_header = None
    _wcs_header = None
    
    def __init__(self, interf_cube_path_A=None, interf_cube_path_B=None,
                 bin_A=None, bin_B=None, pix_size_A=None, pix_size_B=None,
                 alignment_coeffs=None, project_header=list(),
                 cube_A_project_header = list(),
                 cube_B_project_header = list(),
                 wcs_header=list(), overwrite=False,
                 indexer=None, **kwargs):
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

        :param kwargs: Kwargs are :meth:`core.Tools` properties.
        """
        Tools.__init__(self, **kwargs)

        self.overwrite = overwrite
        self.indexer = indexer
        self._project_header = project_header
        self._wcs_header = wcs_header
       
        if alignment_coeffs is not None:
            [self.dx, self.dy, self.dr, self.da, self.db] = alignment_coeffs

        if interf_cube_path_A is not None:
            self.cube_A = HDFCube(interf_cube_path_A,
                                  project_header=cube_A_project_header,
                                  config_file_name=self.config_file_name,
                                  ncpus=self.ncpus)
        if interf_cube_path_B is not None:
            self.cube_B = HDFCube(interf_cube_path_B,
                                  project_header=cube_B_project_header,
                                  config_file_name=self.config_file_name,
                                  ncpus=self.ncpus)

        self.bin_A = bin_A
        self.bin_B = bin_B
        self.pix_size_A = pix_size_A
        self.pix_size_B = pix_size_B
        
        # defining zoom factor
        if (self.pix_size_A is not None and self.pix_size_B is not None
            and self.bin_A is not None and self.bin_B is not None):
            self.zoom_factor = ((float(self.pix_size_B) * float(self.bin_B)) / 
                                (float(self.pix_size_A) * float(self.bin_A)))
        
        # defining rotation center
        if self.cube_B is not None:
            self.rc = [(float(self.cube_B.dimx) / 2.), 
                       (float(self.cube_B.dimy) / 2.)]


    
    def _get_alignment_parameters_path(self):
        """Return the path to the alignment parameters."""
        return self._data_path_hdr + "alignment_parameters.fits"

    def _get_alignment_parameters_header(self):
        """Return the header of the alignment parameters."""
        return (self._get_basic_header('Alignment parameters')
                + self._project_header)
    

    def _get_modulation_ratio_path(self):
        """Return the path to the modulation ratio."""
        return self._data_path_hdr + "modulation_ratio.fits"

    def _get_modulation_ratio_header(self):
        """Return the header of the modulation ratio."""
        return (self._get_basic_header('Modulation ratio')
                + self._project_header)
        
    def _get_energy_map_path(self):
        """Return the path to the energy map.

        The energy map is the mean frame from the merged cube. It is
        useful to check the alignement.
        """
        return self._data_path_hdr + "energy_map.fits"

    def _get_energy_map_header(self):
        """Return the header of the energy map."""
        return (self._get_basic_header('Energy map')
                + self._project_header
                + self._get_basic_frame_header(
                    self.cube_A.dimx, self.cube_A.dimy))

    def _get_stray_light_vector_path(self):
        """Return the path to the stray light vector.

        The external illuminaton vector records lights coming from
        reflections over clouds, the moon or the sun.
        """
        return self._data_path_hdr + "stray_light_vector.fits"

    def _get_stray_light_vector_header(self):
        """Return the header of the stray light vector."""
        return (self._get_basic_header('stray light vector')
                + self._project_header)
   
    def _get_ext_illumination_vector_path(self):
        """Return the path to the external illumination vector.

        The external illuminaton vector records the external
        illumination difference between both cameras (e.g. if one
        camera get some diffused light from the sky while the other is
        well isolated). This vector is used to correct
        interferograms.
        """
        return self._data_path_hdr + "ext_illumination_vector.fits"

    def _get_ext_illumination_vector_header(self):
        """Return the header of the external illumination vector."""
        return (self._get_basic_header('External illumination vector')
                + self._project_header)
    
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

    def _get_transmission_vector_header(self, err=False):
        """Return the header of the transmission vector.

        :param err: (Optional) True if error vector (default False).
        """
        if not err:
            return (self._get_basic_header('Transmission vector')
                    + self._project_header)
        else:
            return (self._get_basic_header('Transmission vector error')
                    + self._project_header)


    def _get_calibration_stars_path(self):
        """Return the path to a data file containing the merged
        interferograms of the calibrated stars"""
        
        return self._data_path_hdr + "calibration_stars.fits"

    def _get_calibration_stars_header(self):
        """Return the header of the calibration stars data file."""
        return (self._get_basic_header('Calibration stars interferograms')
                + self._project_header)

    def _get_extracted_star_spectra_path(self):
        """Return the path to a data file containing the spectra of
        the extracted stars"""
        return self._data_path_hdr + "extracted_star_spectra.fits"

    def _get_extracted_star_spectra_header(self):
        """Return the header of a data file containing the spectra of
        the extracted stars"""
        return (self._get_basic_header('Extracted star spectra')
                + self._project_header)


    def _get_deep_frame_path(self):
        """Return the path to the deep frame.

        The energy map is the mean frame from the merged cube. It is
        useful to check the alignement.
        """
        return self._data_path_hdr + "deep_frame.fits"

    def _get_deep_frame_header(self):
        """Return the header of the deep frame."""
        return (self._get_basic_header('Deep Frame')
                + self._project_header
                + self._get_basic_frame_header(
                    self.cube_A.dimx, self.cube_A.dimy))

    def _get_merged_interfero_cube_path(self):
        """Return the default path to the merged interferogram frames."""
        return self._data_path_hdr + "interferogram.hdf5"

    def _get_merged_interfero_frame_header(self):
        """Return the header of the merged interferogram frames."""
        return (self._get_basic_header('Merged interferogram frame')
                + self._project_header
                + self._get_basic_frame_header(self.cube_A.dimx,
                                               self.cube_A.dimy))

    def _get_bad_frames_vector_path(self):
        """Return the path to the bad frames vector.

        This vector is created by
        :py:meth:`process.InterferogramMerger.merge` method.
        """
        return self._data_path_hdr + "bad_frames_vector.fits"

    def _get_bad_frames_vector_header(self):
        """Return the header of the bad frames vector."""
        return (self._get_basic_header('Bad frames vector')
                + self._project_header)


    def _get_transformed_interfero_cube_path(self):
        """Return the default path to the transformed interferogram frames."""
        return self._data_path_hdr + "transformed_cube_B.hdf5"

    def _get_transformed_interfero_frame_header(self):
        """Return the header of the transformed interferogram frames."""
        return (self._get_basic_header('Transformed interferogram frame')
                + self.cube_B._project_header
                + self._get_basic_frame_header(self.cube_A.dimx,
                                               self.cube_A.dimy))


    def find_alignment(self, star_list_path_A, init_angle, init_dx, init_dy,
                       fwhm_arc_A, fov_A, combine_first_frames=False):
        """
        Return the alignment coefficients to align the cube of the
        camera 2 on the cube of the camera 1

        :param star_list_path_A: Path to a list of star for the camera A
        
        :param init_angle: Initial rotation angle between images

        :param init_dx: Initial shift along x axis between images

        :param init_dy: Initial shift along y axis between images

        :param fwhm_arc_A: rough FWHM of the stars in arcsec in the
          camera A.
        
        :param fov_A: Field of view of the camera A in arcminutes (given
          along x axis.

        :param combine_first_frames: If True, only the fist frames are
          combined to compute alignement parameters (default False).

        .. seealso:: py:meth:`orb.astrometry.Aligner.compute_alignment_parameters`
        """
        # High pass filtering of the frames
        HPFILTER = int(self._get_tuning_parameter('HPFILTER', 0))
        N_FRAMES = 10 # number of combined frames
        
        self._print_msg("Computing alignment parameters")

        # defining FOV of the camera B
        ccd_size_A = self.bin_A * self.cube_A.dimx * self.pix_size_A
        ccd_size_B = self.bin_B * self.cube_B.dimx * self.pix_size_B
        scale = fov_A / ccd_size_A # absolute scale [arcsec/um]
        fov_B = scale * ccd_size_B

        fwhm_arc_B = float(fwhm_arc_A)
        self._print_msg("Calculated FOV of the camera B: %f arcmin"%fov_B)

        # Printing some information
        self._print_msg("Rotation center: %s"%str(self.rc))
        self._print_msg("Zoom factor: %f"%self.zoom_factor)

        # creating deep frames for cube A and B
        if not combine_first_frames:
            frameA = self.cube_A.get_mean_image()
            frameB = self.cube_B.get_mean_image()
        else:
            frameA = bn.nanmedian(self.cube_A[:,:,:N_FRAMES], axis=2)
            frameB = bn.nanmedian(self.cube_B[:,:,:N_FRAMES], axis=2)
            

        if HPFILTER: # Filter alignment frames
            frameA = orb.utils.image.high_pass_diff_image_filter(frameA, deg=1)
            frameB = orb.utils.image.high_pass_diff_image_filter(frameB, deg=1)
        
        aligner = Aligner(
            frameA, frameB, fwhm_arc_A, fov_A, fov_B,
            self.bin_A, self.bin_B, self.pix_size_A, self.pix_size_B,
            init_angle, init_dx, init_dy,
            tuning_parameters=self._tuning_parameters,
            project_header=self._project_header, overwrite=self.overwrite,
            data_prefix=self._data_prefix,
            config_file_name=self.config_file_name, ncpus=self.ncpus)
        
        result = aligner.compute_alignment_parameters(
            star_list_path1=star_list_path_A,
            fwhm_arc=fwhm_arc_A,
            correct_distorsion=False)

        [self.dx, self.dy, self.dr, self.da, self.db] = result['coeffs']
        self.rc = result['rc']
        self.zoom_factor = result['zoom_factor']
        sip1 = result['sip1']
        sip2 = result['sip2']

        alignment_parameters_array = np.array([self.dx, self.dy, self.dr,
                                               self.da, self.db, self.rc[0], self.rc[1],
                                               self.zoom_factor, fwhm_arc_A])
        
        self.write_fits(
            self._get_alignment_parameters_path(),
            alignment_parameters_array,
            fits_header=self._get_alignment_parameters_header(),
            overwrite=self.overwrite)

        if self.indexer is not None:
            self.indexer['alignment_parameters'] = self._get_alignment_parameters_path()

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

        framesA = np.empty((self.cube_A.dimx, self.cube_A.dimy, ncpus), 
                           dtype=float)
        framesB = np.empty((self.cube_A.dimx, self.cube_A.dimy, ncpus), 
                           dtype=float)
        framesB_mask = np.empty((self.cube_A.dimx, self.cube_A.dimy, ncpus), 
                                dtype=float)
        framesB_init = np.empty((self.cube_B.dimx, self.cube_B.dimy, ncpus), 
                                dtype=float)
        framesB_init_mask = np.empty((self.cube_B.dimx, self.cube_B.dimy,
                                      ncpus), dtype=float)
        
       

        self._print_msg("Transforming cube B")
        self._print_msg("Alignment parameters : %s"%str([self.dx, self.dy,
                                                         self.dr, self.da,
                                                         self.db]))
        self._print_msg("Zoom factor : %s"%str(self.zoom_factor))
        
        out_cube = OutHDFCube(self._get_transformed_interfero_cube_path(),
                              shape=(self.cube_A.dimx, self.cube_A.dimy, self.cube_A.dimz),
                              overwrite=self.overwrite)
        
        progress = ProgressBar(int(self.cube_A.dimz/ncpus_max))
        for ik in range(0, self.cube_A.dimz, ncpus):
            # no more jobs than frames to compute
            if (ik + ncpus >= self.cube_A.dimz):
                ncpus = self.cube_A.dimz - ik
            
            for ijob in range(ncpus):
                framesA[:,:,ijob] = self.cube_A.get_data_frame(ik + ijob)
                framesB_init[:,:,ijob] = self.cube_B.get_data_frame(ik + ijob)
                if self.cube_B._mask_exists:
                    framesB_init_mask[:,:,ijob] = self.cube_B.get_data_frame(
                        ik + ijob, mask=True)
                else:
                    framesB_init_mask[:,:,ijob].fill(0.)

            # transform frames of camera B to align them with those of camera A
            jobs = [(ijob, job_server.submit(
                orb.utils.image.transform_frame, 
                args=(framesB_init[:,:,ijob],
                      0, self.cube_A.dimx, 
                      0, self.cube_A.dimy, 
                      [self.dx, self.dy, self.dr, self.da, self.db],
                      self.rc, self.zoom_factor,
                      interp_order,
                      framesB_init_mask[:,:,ijob]),
                modules=("numpy as np", 
                         "from scipy import ndimage",
                         "import orb.cutils"))) 
                    for ijob in range(ncpus)]

            for ijob, job in jobs:
                framesB[:,:,ijob], framesB_mask[:,:,ijob] = job()
            
            for ijob in range(ncpus):
                out_cube.write_frame(
                    ik + ijob,
                    data=framesB[:,:,ijob],
                    header=self._get_transformed_interfero_frame_header(),
                    mask=framesB_mask[:,:,ijob])

            progress.update(int(ik/ncpus_max), info="frame : " + str(ik))
        self._close_pp_server(job_server)
        progress.end()

        out_cube.close()
        del out_cube
        
        if self.indexer is not None:
            self.indexer['transformed_interfero_cube'] = self._get_transformed_interfero_cube_path()


    def alternative_merge(self, add_frameB=True):
        """Alternative merging process.

        Star photometry is not used during the merging process. Might
        be more noisy but useful if for some reason the correction
        vectors cannot be well computed (e.g. not enough good stars,
        intense emission lines everywhere in the field)

        :param add_frameB: (Optional) Set it to False if B frame is
           too noisy to be added to the result. In this case frame B
           is used only to correct for variations of flux from the
           source (airmass, clouds ...) (Default False).
           
        .. note:: Cubes are merged using the formula (for the nth frame):
              
              .. math::

                 Frame_{n,M} = \\frac{Frame_{n,1} -
                 Frame_{n,2}}{Frame_{n,1} + Frame_{n,2}}

        """

        def _create_merged_frame(frameA, frameB, modulation_ratio,
                                 add_frameB, transmission_scale):
            """Create the merged frame given the frames of both cubes
            and the correction factor.

            :param frameA: Frame of the camera 1
            
            :param frameB: Frame of the camera 2
            
            :param modulation_ratio: The ratio of modulation between
              the two cameras. It depends on the gain and the quantum
              efficiency of the CCD.
               
            :param add_frameB: If False the frame B is not added but
            used to determine the transmission of the sky in each
            pixel.

            :param transmission_scale: Scaling factor for the
              transmission frame
            """
            
            if np.all(frameA == 0) and np.all(frameB == 0):
                return np.zeros_like(frameA)
            
            # Correcting for the variation of sky transmission
            frameB = frameB / modulation_ratio

            # Sky transmission frame
            transmission_frame = (frameB + frameA) / transmission_scale
            bad_pix = np.nonzero(transmission_frame==0.)
            transmission_frame[bad_pix] = 1.
            
            if add_frameB:
                result_frame = ((frameB - frameA) / transmission_frame)
            else:
                result_frame = ((frameA) / transmission_frame)

            result_frame[bad_pix] = np.nan
            result_frame[np.nonzero(frameA == 0.)] = np.nan
            result_frame[np.nonzero(frameB == 0.)] = np.nan
            
            return result_frame
        
        def get_nostar_modulation_ratio(frameA, frameB, saturation_level,
                                        pix_ratio=0.015):
            """The most intense non saturated pixels are taken to
            evaluate precisely the modulation ratio.
            
            :param frameA: Frame of the camera 1
            
            :param frameB: Frame of the camera 2

            :param saturation_level: Level above which values are
              considered as saturated and thus incorrect.

            :param pix_ratio: (Optional) Percentage of the most
              intense pixels used to compute the modulation ratio.          
            """
            
            # get intensity threshold
            intensity_list = np.sort(frameA.flatten())
            intensity_list = intensity_list[np.nonzero(
                intensity_list < saturation_level)]
            threshold_number = int(np.size(frameA)*pix_ratio)
            threshold_intensity = intensity_list[-threshold_number]
            # get good pixel indexes
            good_pix = np.nonzero(
                (frameA > threshold_intensity)
                * (frameA < SATURATION_LEVEL))
            # setting bad pixels in frame A to 1.
            frameA[np.nonzero(np.isnan(frameA))] = 1.
            frameA[np.nonzero(frameA == 0.)] = 1.
            
            frames_ratio = frameB/frameA
            return orb.utils.stats.robust_mean((frames_ratio)[good_pix])

        SATURATION_LEVEL = 65000 # Level of image saturation
        
        self._print_warning('Alternative merging process: Merging cubes without using star photometry')

        ## MODULATION RATIO
        # creating deep frames for cube A and B
        deep_frameA = self.cube_A.get_mean_image()
        deep_frameB = self.cube_B.get_mean_image()
        energy_mapA = self.cube_A.get_interf_energy_map()
        energy_mapB = self.cube_B.get_interf_energy_map()

        modulation_ratio = get_nostar_modulation_ratio(
            np.copy(deep_frameA), np.copy(deep_frameB),
            SATURATION_LEVEL/2.)

        self._print_msg(
            "Modulation ratio: %f"%
            modulation_ratio)

        ## ENERGY MAP & DEEP FRAME
        energy_map = energy_mapA * modulation_ratio + energy_mapB
        deep_frame = deep_frameA * modulation_ratio + deep_frameB
        
        self.write_fits(self._get_energy_map_path(), energy_map, 
                        fits_header=
                        self._get_energy_map_header(),
                        overwrite=self.overwrite)

        if self.indexer is not None:
            self.indexer['energy_map'] = self._get_energy_map_path()

        self.write_fits(self._get_deep_frame_path(), deep_frame, 
                        fits_header=
                        self._get_deep_frame_header(),
                        overwrite=self.overwrite)

        if self.indexer is not None:
            self.indexer['deep_frame'] = self._get_deep_frame_path()

        ## TRANSMISSION SCALE
        # scaling factor for the transmission frame during merging
        transmission_scale = np.median(deep_frameA
                                       + deep_frameB / modulation_ratio)
        
        
        ## MERGE FRAMES
        self._print_msg("Merging cubes")
        
        # Init of the multiprocessing server
        job_server, ncpus = self._init_pp_server()
        framesA = np.empty((self.cube_A.dimx, self.cube_A.dimy, ncpus),
                           dtype=float)
        ncpus_max = ncpus
        
        result_frames = np.empty(
            (self.cube_A.dimx, self.cube_A.dimy, ncpus), dtype=float)


        out_cube = OutHDFCube(self._get_merged_interfero_cube_path(),
                              shape=(self.cube_A.dimx,
                                     self.cube_A.dimy,
                                     self.cube_A.dimz),
                              overwrite=self.overwrite)
        
        progress = ProgressBar(int(self.cube_A.dimz/ncpus_max))
        header = self._get_merged_interfero_frame_header()
        for ik in range(0, self.cube_A.dimz, ncpus):
            # no more jobs than frames to compute
            if (ik + ncpus >= self.cube_A.dimz):
                ncpus = self.cube_A.dimz - ik

            for ijob in range(ncpus):
                framesA[:,:,ijob] = self.cube_A.get_data_frame(ik + ijob)
                # result_frames are used instead of a framesB variable
                # to reduce the amount of memory used
                result_frames[:,:,ijob] = self.cube_B.get_data_frame(ik + ijob)

            # compute merged frames
            jobs = [(ijob, job_server.submit(
                _create_merged_frame, 
                args=(framesA[:,:,ijob],
                      result_frames[:,:,ijob], 
                      modulation_ratio,
                      add_frameB,
                      transmission_scale),
                modules=("numpy as np",)))
                    for ijob in range(ncpus)]
                
            for ijob, job in jobs:
                result_frames[:,:,ijob] = job()
             
            for ijob in range(ncpus):
                out_cube.write_frame(
                    ik + ijob,
                    data=result_frames[:,:,ijob],
                    header=header,
                    record_stats=True)
        
            progress.update(int(ik/ncpus_max), info="frame : " + str(ik))
        self._close_pp_server(job_server)
        progress.end()

        if self.indexer is not None:
            self.indexer['merged_interfero_cube'] = (
                self._get_merged_interfero_cube_path())
        

    def merge(self, star_list_path, step_number, fwhm_arc, fov,
              add_frameB=True, smooth_vector=True,
              profile_name='gaussian',
              moffat_beta=3.5, bad_frames_vector=[],
              compute_ext_light=True, aperture_photometry=True,
              readout_noise_1=10., dark_current_level_1=0.,
              readout_noise_2=10., dark_current_level_2=0.):
        
        """
        Merge the cube of the camera 1 and the transformed cube of the
        camera 2.

        :param star_list_path: Path to a list of star positions.

        :param step_number: Number of steps for a full cube.
        
        :param add_frameB: (Optional) Set it to False if B frame is
           too noisy to be added to the result. In this case frame B
           is used only to correct for variations of flux from the
           source (airmass, clouds ...) (Default False).
           
        :param smooth_vector: (Optional) If True smooth the obtained
           correction vector with a gaussian weighted moving average.
           Reduce the possible high frequency noise of the correction
           function. (Default True).

        :param fwhm_arc: rough FWHM of the stars in arcsec
        
        :param fov: Field of view of the frame in arcminutes (given
          along x axis.
    
        :param profile_name: (Optional) PSF profile to use to fit
          stars. Can be 'gaussian' or 'moffat' (default
          'gaussian'). See:
          :py:meth:`orb.astrometry.Astrometry.fit_stars_in_frame`.

        :param moffat_beta: (Optional) Beta parameter to use for
          moffat PSF (default 3.5).

        :param bad_frames_vector: (Optional) Contains the index of the
          frames considered as bad(default []).

        :param compute_ext_light: (Optional) If True compute the
          external light vector. Make sure that there's enough 'sky'
          pixels in the frame. The vector will be deeply affected if
          the object covers the whole area (default True).

        :param aperture_photometry: (Optional) If True, flux of stars
          is computed by aperture photometry. Else, The flux is
          evaluated given the fit parameters (default True).

        :param readout_noise_1: (Optional) Readout noise in ADU/pixel
          of camera 1 (can be computed from bias frames:
          std(master_bias_frame)) (default 10.)
    
        :param dark_current_level_1: (Optional) Dark current level of
          camera 1 in ADU/pixel (can be computed from dark frames:
          median(master_dark_frame)) (default 0.)

        :param readout_noise_2: (Optional) Readout noise in ADU/pixel
          of camera 2 (can be computed from bias frames:
          std(master_bias_frame)) (default 10.)
    
        :param dark_current_level_2: (Optional) Dark current level of
          camera 2 in ADU/pixel (can be computed from dark frames:
          median(master_dark_frame)) (default 0.)

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
                                 add_frameB, frameA_mask,
                                 frameB_mask):
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

            :param frameA_mask: Mask of the frame A.

            :param frameB_mask: Mask of the frame B.
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
                
                ## if np.any(frameA):
                ##     result_frame = ((frameA - stray_light_coeff)
                ##                     / transmission_factor) + ext_level
                ## else:
                ##     result_frame = frameA
                ## flux_frame = result_frame

            result_frame_mask = frameA_mask + frameB_mask
            result_frame[np.nonzero(frameA == 0.)] = 0.
            result_frame[np.nonzero(frameB == 0.)] = 0.
            flux_frame[np.nonzero(frameA == 0.)] = 0.
            flux_frame[np.nonzero(frameB == 0.)] = 0.
            flux_frame[np.nonzero(np.isnan(flux_frame))] = 0.
            flux_frame[np.nonzero(np.isinf(flux_frame))] = 0.
            
            return result_frame, result_frame_mask, flux_frame

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
            
            progress = ProgressBar(cube.dimz)
            for ik in range(0, cube.dimz, ncpus):
                
                progress.update(ik, info="Creating sky level vector")
                if (ik + ncpus >= cube.dimz):
                    ncpus = cube.dimz - ik
                 
                jobs = [(ijob, job_server.submit(
                    get_sky_level, 
                    args=(cube.get_data_frame(ik+ijob)[xmin:xmax,ymin:ymax],),
                    modules=("import numpy as np",
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
            self._print_warning('No transmission correction')


        local_background = True
        
        if EXTENDED_EMISSION:
            fix_fwhm = True
            optimized_modulation_ratio = False
            # Length ratio of the ZPD over the entire cube to correct
            # the transmission vector
            TRANS_ZPD_SIZE = float(
                self._get_tuning_parameter('TRANS_ZPD_SIZE', 0.1))
            
            self._print_warning(
                'Region considered as an extended emission region')
        else:
            fix_fwhm = False
            optimized_modulation_ratio = True
            # Length ratio of the ZPD over the entire cube to correct
            # the transmission vector
            TRANS_ZPD_SIZE = float(
                self._get_tuning_parameter('TRANS_ZPD_SIZE', 0.01))
        
        if aperture_photometry:
            self._print_msg('Star flux evaluated by aperture photometry')
            photometry_type = 'aperture_flux'
        else:
            self._print_msg('Star flux evaluated from fit parameters')
            photometry_type = 'flux'

        # creating deep frames for cube A and B
        frameA = self.cube_A.get_mean_image()
        frameB = self.cube_B.get_mean_image()
        
        # fit stars on deep frames to get a better guess on position
        # and FWHM
        mean_params_A = Astrometry(
            frameA, fwhm_arc, fov, profile_name=profile_name,
            star_list_path=star_list_path, readout_noise=readout_noise_1,
            dark_current_level=dark_current_level_1,
            tuning_parameters=self._tuning_parameters,
            config_file_name=self.config_file_name,
            ncpus=self.ncpus).fit_stars_in_frame(
            0, precise_guess=True, local_background=local_background,
            fix_fwhm=fix_fwhm, fix_height=False, save=False)
        mean_params_B = Astrometry(
            frameB, fwhm_arc, fov, profile_name=profile_name,
            star_list_path=star_list_path, readout_noise=readout_noise_2,
            dark_current_level=dark_current_level_2,
            tuning_parameters=self._tuning_parameters,
            config_file_name=self.config_file_name,
            ncpus=self.ncpus).fit_stars_in_frame(
            0, precise_guess=True, local_background=local_background,
            fix_fwhm=fix_fwhm, fix_height=False, save=False)

        star_list_A = mean_params_A.get_star_list()
        star_list_B = mean_params_A.get_star_list()

        fwhm_arc_A = orb.utils.stats.robust_mean(mean_params_A[:,'fwhm_arc'])
        fwhm_arc_B = orb.utils.stats.robust_mean(mean_params_B[:,'fwhm_arc'])

        self._print_msg(
            'mean FWHM of the stars in camera 1: {} arc-seconds'.format(
                fwhm_arc_A))
        self._print_msg(
            'mean FWHM of the stars in camera 2: {} arc-seconds'.format(
                fwhm_arc_B))
        

        ## COMPUTING STARS PHOTOMETRY #############################
        self._print_msg("Computing stars photometry")
        astrom_A = Astrometry(self.cube_A, fwhm_arc_A, fov,
                              profile_name=profile_name,
                              moffat_beta=moffat_beta,
                              data_prefix=self._data_prefix + 'cam1.',
                              readout_noise=readout_noise_1,
                              dark_current_level=dark_current_level_1,
                              tuning_parameters=self._tuning_parameters,
                              check_mask=True,
                              config_file_name=self.config_file_name,
                              ncpus=self.ncpus)
        astrom_A.reset_star_list(star_list_A)

        astrom_B = Astrometry(self.cube_B, fwhm_arc_B, fov,
                              profile_name=profile_name,
                              moffat_beta=moffat_beta,
                              data_prefix=self._data_prefix + 'cam2.',
                              readout_noise=readout_noise_2,
                              dark_current_level=dark_current_level_2,
                              tuning_parameters=self._tuning_parameters,
                              check_mask=True,
                              config_file_name=self.config_file_name,
                              ncpus=self.ncpus)
        astrom_B.reset_star_list(star_list_B)

        # Fit stars and get stars photometry
        astrom_A.fit_stars_in_cube(local_background=local_background,
                                   fix_fwhm=fix_fwhm,
                                   fix_height=False,
                                   fix_aperture_size=True,
                                   multi_fit=True,
                                   save=True)
        astrom_B.fit_stars_in_cube(local_background=local_background,
                                   fix_fwhm=fix_fwhm,
                                   fix_height=False,
                                   fix_aperture_size=True,
                                   multi_fit=True,
                                   save=True)
        
        astrom_A.load_fit_results(astrom_A._get_fit_results_path())
        astrom_B.load_fit_results(astrom_B._get_fit_results_path())

        photom_A = astrom_A.fit_results[:,:,photometry_type]
        photom_B = astrom_B.fit_results[:,:,photometry_type]

        # Find ZPD ################################################
        bad_frames_vector = orb.utils.misc.correct_bad_frames_vector(
            bad_frames_vector, self.cube_A.dimz)
        zmedian = self.cube_A.get_zmedian(nozero=True)
        zmedian[bad_frames_vector] = 0.
        zpd_index = orb.utils.fft.find_zpd(zmedian,
                                           step_number=step_number)
        
        self._print_msg('ZPD index: %d'%zpd_index)


        ## MODULATION RATIO #######################################
        # Calculating the mean modulation ratio (to correct for
        # difference of camera gain and transmission of the optical
        # path)

        # Optimization routine
        def photom_diff(modulation_ratio, photom_A, photom_B, zpd_min, zpd_max):
            return orb.utils.stats.robust_median((photom_A * modulation_ratio
                                                  - photom_B)**2.)
        
        # use EXT_ZPD_SIZE to remove ZPD from MODULATION RATION calculation
        ext_zpd_min = zpd_index - int(EXT_ZPD_SIZE * step_number / 2.)
        if ext_zpd_min < 0: ext_zpd_min = 0
        ext_zpd_max = zpd_index + int(EXT_ZPD_SIZE * step_number / 2.) + 1
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
  
            self._print_msg(
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

            self._print_msg(
                "Modulation ratio: %f (std: %f)"%(
                    modulation_ratio, modulation_ratio_std))
        else:
            modulation_ratio = FIXED_MODULATION_RATIO
            self._print_msg(
                "Fixed modulation ratio: %f"%(
                    modulation_ratio))

        self.write_fits(
            self._get_modulation_ratio_path(), 
            np.array([modulation_ratio]),
            fits_header=self._get_modulation_ratio_header(),
            overwrite=self.overwrite)

        if self.indexer is not None:
            self.indexer['modulation_ratio'] = self._get_modulation_ratio_path()

        # PHOTOMETRY ON MERGED FRAMES #############################
        astrom_merged = Astrometry(self.cube_B, fwhm_arc_B, fov,
                                   profile_name=profile_name,
                                   moffat_beta=moffat_beta,
                                   data_prefix=self._data_prefix + 'merged.',
                                   readout_noise=readout_noise_2,
                                   dark_current_level=dark_current_level_2,
                                   tuning_parameters=self._tuning_parameters,
                                   check_mask=False,
                                   config_file_name=self.config_file_name,
                                   ncpus=self.ncpus)
        astrom_merged.reset_star_list(star_list_B)
        
        astrom_merged.fit_stars_in_cube(
            local_background=local_background,
            fix_aperture_size=True,
            add_cube=[self.cube_A, modulation_ratio],
            no_fit=True, save=True)
        astrom_merged.load_fit_results(astrom_merged._get_fit_results_path())
        photom_merged = astrom_merged.fit_results[:,:,photometry_type]
        photom_merged_err = astrom_merged.fit_results[
            :,:,photometry_type + '_err']

        ## TRANSMISSION VECTOR ####################################
        self._print_msg("Computing transmission vector")

        transmission_vector_list = list()
        red_chisq_list = list()
        trans_err_list = list()
        
        # normalization of the merged photometry vector
        for istar in range(astrom_A.star_list.shape[0]):
            if not np.all(np.isnan(photom_merged)):
                trans = np.copy(photom_merged[istar,:])
                trans_err = np.copy(photom_merged_err[istar,:])
                trans_mean = orb.utils.stats.robust_mean(
                    orb.utils.stats.sigmacut(trans))
                trans /= trans_mean
                trans_err /= trans_mean
                transmission_vector_list.append(trans)
                red_chisq = orb.utils.stats.robust_mean(
                    orb.utils.stats.sigmacut(astrom_A.fit_results[
                        istar, :, 'reduced-chi-square']))
                
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
            self._print_error("Too much stars have been rejected. The transmission vector cannot be computed !")

        self._print_msg(
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
            trans_zpd_min = (zpd_index
                             - int((TRANS_ZPD_SIZE * step_number)/2.))
            trans_zpd_max = (zpd_index
                             + int((TRANS_ZPD_SIZE * step_number)/2.) + 1)

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
        self.write_fits(
            self._get_transmission_vector_path(), 
            transmission_vector.reshape((transmission_vector.shape[0],1)),
            fits_header=self._get_transmission_vector_header(),
            overwrite=self.overwrite)
        if self.indexer is not None:
            self.indexer[
                'transmission_vector'] = self._get_transmission_vector_path()

        self.write_fits(
            self._get_transmission_vector_path(err=True), 
            transmission_vector_err.reshape(
                (transmission_vector_err.shape[0],1)),
            fits_header=self._get_transmission_vector_header(err=True),
            overwrite=self.overwrite)
        if self.indexer is not None:
            self.indexer['transmission_vector_err'] = (
                self._get_transmission_vector_path(err=True))
        
        # Create BAD FRAMES VECTOR from transmission vector
        bad_frames_vector = np.zeros((self.cube_A.dimz), dtype=int)
        bad_frames_vector[
            np.nonzero(transmission_vector <= BAD_FRAME_COEFF)] = 1
        if np.any(bad_frames_vector):
            self._print_msg("Detected bad frames : %s"%str(
                np.nonzero(bad_frames_vector)[0]))
        self.write_fits(
            self._get_bad_frames_vector_path(), 
            bad_frames_vector,
            fits_header=self._get_bad_frames_vector_header(),
            overwrite=self.overwrite)
        
        if self.indexer is not None:
            self.indexer[
                'bad_frames_vector'] = self._get_bad_frames_vector_path()

        ## EXTERNAL ILLUMINATION VECTOR ##########################
        # Computing the external illumination level (if some light
        # enters in one of the cameras).
        
        # WARNING : This vector will be correct if there's enough
        # 'sky' pixels.

        if compute_ext_light:
            self._print_msg("Computing external illumination vector")
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
            self._print_warning(
                "External illumination vector computation skipped")
            ext_level_vector = np.zeros(self.cube_A.dimz, dtype=float)
     
        # Save external illumination vector
        self.write_fits(
            self._get_ext_illumination_vector_path(), 
            ext_level_vector.reshape((ext_level_vector.shape[0],1)),
            fits_header=self._get_ext_illumination_vector_header(),
            overwrite=self.overwrite)
        if self.indexer is not None:
            self.indexer['ext_illumination_vector'] = (
                self._get_ext_illumination_vector_path())
        

        ## MERGE FRAMES ###########################################
        self._print_msg("Merging cubes")
        
        # Init of the multiprocessing server
        job_server, ncpus = self._init_pp_server()


        flux_frame = np.zeros((self.cube_A.dimx, self.cube_A.dimy),
                              dtype=float)
        flux_frame_nb = 0
        flux_vector = np.zeros(self.cube_A.dimz, dtype=float)
        result_frames = np.empty((self.cube_A.dimx, self.cube_A.dimy, ncpus),
                                 dtype=float)
        result_mask_frames = np.empty((self.cube_A.dimx, self.cube_A.dimy,
                                       ncpus), dtype=float)
        framesA_mask = np.empty((self.cube_A.dimx, self.cube_A.dimy,
                                 ncpus), dtype=float)
        framesB_mask = np.empty((self.cube_A.dimx, self.cube_A.dimy,
                                 ncpus), dtype=float)

        out_cube = OutHDFCube(self._get_merged_interfero_cube_path(),
                              shape=(self.cube_A.dimx,
                                     self.cube_A.dimy,
                                     self.cube_A.dimz),
                              overwrite=self.overwrite)
        
        ncpus_max = ncpus
        progress = ProgressBar(int(self.cube_A.dimz/ncpus_max))
        header = self._get_merged_interfero_frame_header()
        
        for ik in range(0, self.cube_A.dimz, ncpus):
            # no more jobs than frames to compute
            if (ik + ncpus >= self.cube_A.dimz):
                ncpus = self.cube_A.dimz - ik

            if (self.cube_A._mask_exists
                and self.cube_A._mask_exists):
                for ijob in range(ncpus):
                    framesA_mask[:,:,ijob] = self.cube_A.get_data_frame(
                        ik + ijob, mask=True)
                    framesB_mask[:,:,ijob] = self.cube_B.get_data_frame(
                        ik + ijob, mask=True)
            else:
                framesA_mask.fill(0.)
                framesB_mask.fill(0.)
            
            # compute merged frames
            
            jobs = [(ijob, job_server.submit(
                _create_merged_frame, 
                args=(self.cube_A.get_data_frame(ik + ijob),
                      self.cube_B.get_data_frame(ik + ijob), 
                      transmission_vector[ik + ijob],
                      modulation_ratio,
                      ext_level_vector[ik + ijob],
                      add_frameB,
                      framesA_mask[:,:,ijob],
                      framesB_mask[:,:,ijob]),
                modules=("numpy as np",)))
                    for ijob in range(ncpus)]
                
            for ijob, job in jobs:
                (result_frames[:,:,ijob],
                 result_mask_frames[:,:,ijob],
                 flux_frame_temp) = job()
                
                if np.any(flux_frame_temp != 0.):
                    flux_frame += flux_frame_temp
                    flux_frame_nb += 1
                    flux_vector[ik + ijob] = orb.utils.stats.robust_median(
                        flux_frame_temp)
                else:
                    flux_vector[ik + ijob] = np.nan
             
            for ijob in range(ncpus):
                out_cube.write_frame(
                    ik + ijob,
                    data=result_frames[:,:,ijob],
                    header=header,
                    mask=result_mask_frames[:,:,ijob],
                    record_stats=True)

            progress.update(int(ik/ncpus_max), info="frame : " + str(ik))
        self._close_pp_server(job_server)
        progress.end()
        flux_frame /= flux_frame_nb

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
            self.write_fits(
                self._get_stray_light_vector_path(), 
                stray_light_vector,
                fits_header=self._get_stray_light_vector_header(),
                overwrite=self.overwrite)
        
            if self.indexer is not None:
                self.indexer['stray_light_vector'] = (
                    self._get_stray_light_vector_path())
                
            self._print_msg('Mean flux of stray light: {} ADU'.format(
                np.nanmean(stray_light_vector)))
        else:
            stray_light_vector = np.zeros_like(flux_vector)
       
        merged_cube = HDFCube(self._get_merged_interfero_cube_path(),
                              config_file_name=self.config_file_name,
                              ncpus=self.ncpus)
        energy_map = merged_cube.get_interf_energy_map()
        out_cube.append_energy_map(energy_map)
        deep_frame = (flux_frame - np.nanmean(stray_light_vector))
        out_cube.append_deep_frame(deep_frame)
    
        self.write_fits(self._get_energy_map_path(), energy_map, 
                        fits_header=
                        self._get_energy_map_header(),
                        overwrite=self.overwrite)

        if self.indexer is not None:
            self.indexer['energy_map'] = self._get_energy_map_path()

        
        self.write_fits(self._get_deep_frame_path(), deep_frame, 
                        fits_header=self._get_deep_frame_header(),
                        overwrite=self.overwrite)

        if self.indexer is not None:
            self.indexer['deep_frame'] = self._get_deep_frame_path()


        # SAVE CALIBRATION STARS INTERFEROGRAMS
        self._print_msg("Saving corrected calibration stars interferograms")
        calib_stars_interf_list = list()
        for istar in range(astrom_A.star_list.shape[0]):
            calib_stars_interf_list.append(
                (((photom_B[istar,:]/modulation_ratio) - photom_A[istar,:])
                 / transmission_vector) - ext_level_vector)
        calibration_stars_path = self._get_calibration_stars_path()
        self.write_fits(calibration_stars_path,
                        np.array(calib_stars_interf_list),
                        fits_header=self._get_calibration_stars_header(),
                        overwrite=self.overwrite)

        if self.indexer is not None:
            self.indexer['calibration_stars'] = calibration_stars_path
            
        self._print_msg("Cubes merged")
        out_cube.close()
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

    def _get_cr_map_frame_header(self):
        """Return the header of the cosmic ray map."""
        return (self._get_basic_header('Cosmic ray map')
                + self._project_header)

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
        
        alignment_vector_1 = self.read_fits(alignment_vector_path_1)
        star_list = orb.utils.astrometry.load_star_list(star_list_path)
        
        job_server, ncpus = self._init_pp_server()
        ncpus_max = ncpus

        framesA = np.empty((self.cube_A.dimx, self.cube_A.dimy, ncpus), 
                           dtype=float)
        framesB_init = np.empty((self.cube_B.dimx, self.cube_B.dimy, ncpus), 
                                dtype=float)
        framesB = np.empty((self.cube_A.dimx, self.cube_A.dimy, ncpus), 
                           dtype=float)

        cr_mapA = np.empty((self.cube_A.dimx, self.cube_A.dimy, self.cube_A.dimz), dtype=np.bool)
        cr_mapB = np.empty((self.cube_B.dimx, self.cube_B.dimy, self.cube_B.dimz), dtype=np.bool)
        
        progress = ProgressBar(int(self.cube_A.dimz/ncpus_max))
        for ik in range(0, self.cube_A.dimz, ncpus):
            progress.update(int(ik/ncpus_max), info="frame : " + str(ik))
            
            # no more jobs than frames to compute
            if (ik + ncpus >= self.cube_A.dimz):
                ncpus = self.cube_A.dimz - ik
            
            for ijob in range(ncpus):
                framesA[:,:,ijob] = self.cube_A.get_data_frame(ik + ijob)
                framesB_init[:,:,ijob] = self.cube_B.get_data_frame(ik + ijob)

            # transform frames of camera B to align them with those of camera A
            jobs = [(ijob, job_server.submit(
                orb.utils.image.transform_frame, 
                args=(framesB_init[:,:,ijob],
                      0, self.cube_A.dimx, 
                      0, self.cube_A.dimy, 
                      [self.dx - alignment_vector_1[ik+ijob, 0],
                       self.dy - alignment_vector_1[ik+ijob, 1],
                       self.dr, self.da, self.db],
                      self.rc, self.zoom_factor, 1),
                modules=("import numpy as np", 
                         "from scipy import ndimage",
                         "import orb.cutils"))) 
                    for ijob in range(ncpus)]

            for ijob, job in jobs:
                framesB[:,:,ijob] = job()
            
                
            framesM = framesA + framesB # ok for SITELLE, not for SpIOMM
            framesM -= np.nanmean(np.nanmean(framesM, axis=0), axis=0)
            framesM += BIAS
            
            frameref = bn.nanmedian(framesM, axis=2)

            # detect CRS
            jobs = [(ijob, job_server.submit(
                detect_crs_in_frame, 
                args=(framesA[:,:,ijob],
                      framesB_init[:,:,ijob],
                      framesM[:,:,ijob],
                      frameref,
                      [self.dx - alignment_vector_1[ik+ijob, 0],
                       self.dy - alignment_vector_1[ik+ijob, 1],
                       self.dr, self.da, self.db,
                       self.rc[0], self.rc[1],
                       self.zoom_factor, self.zoom_factor],
                      star_list, fwhm_pix,
                      alignment_vector_1[ik+ijob, 0],
                      alignment_vector_1[ik+ijob, 1]),
                modules=("import numpy as np", 
                         "import orb.cutils",
                         "import orb.utils.image",
                         "import orb.utils.stats",
                         "import warnings"))) 
                    for ijob in range(ncpus)]
            
            for ijob, job in jobs:
                cr_mapA[:,:,ik+ijob], cr_mapB[:,:,ik+ijob] = job()
                
        self._close_pp_server(job_server)   
        progress.end()  
        
        # check to remove over detected pixels (stars)
        cr_mapA_deep = np.sum(cr_mapA, axis=2)
        cr_mapB_deep = np.sum(cr_mapB, axis=2)

        badpixA = np.nonzero(cr_mapA_deep > MAX_CRS)
        badpixB = np.nonzero(cr_mapB_deep > MAX_CRS)

        if len(badpixA[0]) > 0:
            cr_mapA[badpixA[0], badpixA[1], :] = 0
            self._print_msg('{} pixels with too much detections cleaned in camera 1'.format(
                len(badpixA[0])))
        if len(badpixB[0]) > 0:
            cr_mapB[badpixB[0], badpixB[1], :] = 0
            self._print_msg('{} pixels with too much detections cleaned in camera 2'.format(
                len(badpixB[0])))
        
        
        self._print_msg('Final number of contaminated pixels in camera 1: {}'.format(
            np.sum(cr_mapA)))
        self._print_msg('Final number of contaminated pixels in camera 2: {}'.format(
            np.sum(cr_mapB)))

        out_cubeA = OutHDFCube(self._get_cr_map_cube_path(1),
                               shape=cr_mapA.shape,
                               overwrite=self.overwrite)
        out_cubeB = OutHDFCube(self._get_cr_map_cube_path(2),
                               shape=cr_mapB.shape,
                               overwrite=self.overwrite)
        for iframe in range(self.cube_A.dimz):
            out_cubeA.write_frame(
                iframe, cr_mapA[:,:,iframe].astype(np.bool_),
                header=self._get_basic_header(file_type="Cosmic ray map"),
                force_float32=False)
            out_cubeB.write_frame(
                iframe, cr_mapB[:,:,iframe].astype(np.bool_),
                header=self._get_basic_header(file_type="Cosmic ray map"),
                force_float32=False)

        out_cubeA.close()
        out_cubeB.close()
        del out_cubeA
        del out_cubeB
      
        if self.indexer is not None:
            self.indexer['cr_map_cube_1'] = (
                self._get_cr_map_cube_path(1))

        if self.indexer is not None:
            self.indexer['cr_map_cube_2'] = (
                self._get_cr_map_cube_path(2))

 
##################################################
#### CLASS Spectrum ##############################
##################################################
class Spectrum(HDFCube):
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

    def _get_modulation_efficiency_map_header(self, imag=False):
        """Return the header of the modulation efficiency map.
    
        :param imag: (Optional) True for imaginary part of the
          modulation efficiency (default False)."""

        if not imag:
            header = self._get_basic_header('Modulation efficiency map')
        else:
            header = self._get_basic_header('Modulation efficiency map (imaginary part)')
       
        return (header
                + self._project_header
                + self._calibration_laser_header
                + self._get_basic_frame_header(self.dimx, self.dimy))

    def _get_calibrated_spectrum_cube_path(self):
        """Return the default path to a calibrated spectral cube."""
        return self._data_path_hdr + "calibrated_spectrum.hdf5"

    def _get_calibrated_spectrum_frame_header(self, frame_index, axis,
                                              apodization_function,
                                              wavenumber=False):
        
        """Return the header of the calibrated spectral frames.
        
        :param frame_index: Index of the frame.  
    
        :param axis: Spectrum axis (must be in wavelength or in
          wavenumber).
    
        :param wavenumber: (Optional) If True the axis is considered
          to be in wavenumber. If False the axis is considered to be
          in wavelength (default False).
        """
        file_type = "Calibrated spectrum frame"
            
        return (self._get_basic_header(file_type)
                + self._project_header
                + self._calibration_laser_header
                + self._get_basic_frame_header(self.dimx, self.dimy)
                + self._get_basic_spectrum_frame_header(frame_index, axis,
                                                        wavenumber=wavenumber)
                + self._get_fft_params_header(apodization_function))
    
    def _get_calibrated_spectrum_header(self, axis, apodization_function,
                                        wavenumber=False):
        """Return the header of the calibrated spectral cube.
        
        :param axis: Spectrum axis (must be in wavelength or in
          wavenumber).
          
        :param wavenumber: (Optional) If True the axis is considered
          to be in wavenumber. If False the axis is considered to be
          in wavelength (default False).
        """
        header = self._get_basic_header('Calibrated spectrum cube')
        
        return (header
                + self._project_header
                + self._calibration_laser_header
                + self._get_basic_frame_header(self.dimx, self.dimy)
                + self._get_basic_spectrum_cube_header(
                    axis, wavenumber=wavenumber)
                + self._get_fft_params_header(apodization_function))

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
        
    def calibrate(self, filter_name, step, order,
                  calibration_laser_map_path, nm_laser,
                  exposure_time,
                  correct_wcs=None,
                  flux_calibration_vector=None,
                  flux_calibration_coeff=None,
                  wavenumber=False, standard_header=None,
                  spectral_calibration=True, filter_correction=True):
        
        """Calibrate spectrum cube: correct for filter transmission
        function, correct WCS parameters and flux calibration.

        :param filter_name: Name of the filter (If None, no filter
           correction will be made).

        :param step: Step size of the interferogram in nm.
        
        :param order: Folding order of the interferogram.

        :param calibration_laser_map_path: Path to the calibration
          laser map.

        :param nm_laser: Calibration laser wavelength.

        :param exposure_time: Exposure time in s (by frame)..

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
                                       exposure_time,
                                       calibration_laser_col, nm_laser,
                                       step, order, wavenumber,
                                       spectral_calibration,
                                       base_axis_correction_coeff,
                                       output_sz_coeff):
            """
            
            """
            INTERP_POWER = 30 * output_sz_coeff
            ZP_LENGTH = orb.utils.fft.next_power_of_two(
                spectrum_col.shape[1] * INTERP_POWER)

            result_col = np.empty((spectrum_col.shape[0],
                                   spectrum_col.shape[1]
                                   * output_sz_coeff))
            result_col.fill(np.nan)

            # converting to flux (ADU/s)
            spectrum_col /= exposure_time * spectrum_col.shape[1]

            # converting to ADU/s/A
            axis_step = orb.cutils.get_nm_axis_step(
                spectrum_col.shape[1], step, order)
            spectrum_col /= axis_step * 10.

            if wavenumber:
                axis_base = orb.utils.spectrum.create_cm1_axis(
                    spectrum_col.shape[1] * output_sz_coeff, step, order,
                    corr=base_axis_correction_coeff).astype(float)
            else:
                axis_base = orb.utils.spectrum.create_nm_axis(
                    spectrum_col.shape[1] * output_sz_coeff, step, order,
                    corr=base_axis_correction_coeff).astype(float)

            for icol in range(spectrum_col.shape[0]):

                # pure fft interpolation of the input spectrum
                # (i.e. perfect interpolation as long as the imaginary
                # part is given)
                if (spectral_calibration or not wavenumber) or output_sz_coeff != 1:
                    #spectrum_col[np.isnan(spectrum_col)] = 0.
                    interf_complex = np.fft.ifft(spectrum_col[icol,:])
                    zp_interf = np.zeros(ZP_LENGTH, dtype=complex)
                    center = interf_complex.shape[0]/2
                    zp_interf[:center] = interf_complex[:center]
                    zp_interf[
                        -center-int(interf_complex.shape[0]&1):] = interf_complex[
                        -center-int(interf_complex.shape[0]&1):]
                    interf_complex = np.copy(zp_interf)
                    spectrum_highres = np.fft.fft(interf_complex).real
                else:
                    spectrum_highres = np.copy(spectrum_col[icol,:])

                corr = calibration_laser_col[icol]/nm_laser
                if np.isnan(corr):
                    result_col[icol,:] = np.nan
                    continue

                # remember : output from 'compute spectrum' step is
                # always in cm-1
                if wavenumber:
                    axis_corr = orb.utils.spectrum.create_cm1_axis(
                        spectrum_highres.shape[0],
                        step, order, corr=corr).astype(float)
                else:
                    axis_corr = orb.utils.spectrum.create_nm_axis_ireg(
                        spectrum_highres.shape[0],
                        step, order, corr=corr).astype(float)

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
                            spectrum_highres, axis_base, 1,
                            old_axis=axis_corr))
                    
                else:
                    result_col[icol,:] = spectrum_highres

            return result_col.real

        
        def get_mean_scale_map(ref_scale_map, spectrum_scale_map):
            
            MEAN_SCALING_BORDER = 0.3
            
            x_min = int(self.dimx * MEAN_SCALING_BORDER)
            x_max = int(self.dimx * (1. - MEAN_SCALING_BORDER))
            y_min = int(self.dimy * MEAN_SCALING_BORDER)
            y_max = int(self.dimy * (1. - MEAN_SCALING_BORDER))
            spectrum_scale_map_box = spectrum_scale_map[x_min:x_max,
                                                        y_min:y_max]
            ref_scale_map_box = ref_scale_map[x_min:x_max, y_min:y_max]
            return (orb.utils.stats.robust_median(
                orb.utils.stats.sigmacut(
                    ref_scale_map_box
                    / spectrum_scale_map_box, sigma=2.5)),
                    orb.utils.stats.robust_std(orb.utils.stats.sigmacut(
                        ref_scale_map_box
                        / spectrum_scale_map_box, sigma=2.5)))


        OUTPUT_SZ_COEFF = 1

        if filter_correction:
            self._print_error("Filter correction is not stable please don't use it")

        # get calibration laser map
        calibration_laser_map = self.read_fits(calibration_laser_map_path)
        if (calibration_laser_map.shape[0] != self.dimx):
            calibration_laser_map = orb.utils.image.interpolate_map(
                calibration_laser_map,
                self.dimx, self.dimy)

        # get correction coeff at the center of the field (required to
        # project the spectral cube at the center of the field instead
        # of projecting it on the interferometer axis)
        if spectral_calibration:
            base_axis_correction_coeff = calibration_laser_map[
                int(self.dimx/2), int(self.dimy/2)] / nm_laser
        else:
            base_axis_correction_coeff = 1.
        
        
        # Get filter parameters
        FILTER_STEP_NB = 4000
        FILTER_RANGE_THRESHOLD = 0.97
        filter_vector, filter_min_pix, filter_max_pix = (
            FilterFile(filter_name).get_filter_function(
                step, order, FILTER_STEP_NB, wavenumber=wavenumber,
                corr=base_axis_correction_coeff))
        filter_min_pix = np.nanmin(np.arange(FILTER_STEP_NB)[
            filter_vector >  FILTER_RANGE_THRESHOLD * np.nanmax(filter_vector)])
        filter_max_pix = np.nanmax(np.arange(FILTER_STEP_NB)[
            filter_vector >  FILTER_RANGE_THRESHOLD * np.nanmax(filter_vector)])
        
        if filter_min_pix < 0: filter_min_pix = 0
        if filter_max_pix > FILTER_STEP_NB: filter_max_pix = FILTER_STEP_NB - 1

        if not wavenumber:
            filter_axis = orb.utils.spectrum.create_nm_axis(
                FILTER_STEP_NB, step, order)
        else:
            filter_axis = orb.utils.spectrum.create_cm1_axis(
                FILTER_STEP_NB, step, order)

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
        modulation_efficiency = FilterFile(
            filter_name).get_modulation_efficiency()

        self._print_msg('Modulation efficiency: {}'.format(
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
                self._print_msg('Mean flux calib vector before adjustment with flux calibration coeff: {} erg/cm2/ADU'.format(mean_flux_calib_vector))
                 # ME must be taken into account only when using the
                 # flux calibration coeff derived from std images
                flux_calibration_coeff /= modulation_efficiency
                self._print_msg('Flux calibration coeff (corrected for modulation efficiency {}): {} erg/cm2/ADU'.format(modulation_efficiency, flux_calibration_coeff))
                flux_calibration_vector /=  mean_flux_calib_vector
                flux_calibration_vector *= flux_calibration_coeff
                flux_calibration_function = interpolate.UnivariateSpline(
                    flux_calibration_axis, flux_calibration_vector, s=0, k=3)

        # Calibrate with coeff derived from std images
        elif flux_calibration_coeff is not None:
            # ME must be taken into account only when using the flux
            # calibration coeff derived from std images
            flux_calibration_coeff /= modulation_efficiency
            self._print_msg('Flux calibration coeff (corrected for modulation efficiency {}): {} erg/cm2/ADU'.format(modulation_efficiency, flux_calibration_coeff))
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

        self._print_msg("Calibrating spectra", color=True)
        
        if filter_correction:
            self._print_msg("Filter correction")
        else:
            self._print_warning("No filter correction")

        if spectral_calibration:
            self._print_msg("Spectral calibration")
        else:
            self._print_warning("No spectral calibration")
        
        if flux_calibration_vector is not None:
            self._print_msg("Flux calibration")
        else:
            self._print_warning("No flux calibration")
            
        if correct_wcs is not None:
            self._print_msg("WCS correction")
        else:
            self._print_warning("No WCS correction")


        # control energy in the imaginary part ratio
        ## deep_frame_spectrum = self.get_mean_image()
        ## imag_energy = orb.utils.stats.sigmacut(
        ##     deep_frame_spectrum.imag/deep_frame_spectrum.real, sigma=2.5)
        ## self._print_msg("Median energy ratio imaginary/real: {:.2f} [std {:.3f}] %".format(np.nanmedian(imag_energy)*100., np.nanstd(imag_energy)*100.))
            
   
        out_cube = OutHDFQuadCube(
            self._get_calibrated_spectrum_cube_path(),
            (self.dimx, self.dimy, self.dimz * OUTPUT_SZ_COEFF),
            self.QUAD_NB,
            reset=True)

        
        # Init of the multiprocessing server    
        for iquad in range(0, self.QUAD_NB):
            (x_min, x_max, 
             y_min, y_max) = self.get_quadrant_dims(iquad)
            
            iquad_data = np.empty((x_max - x_min,
                                   y_max - y_min,
                                   self.dimz * OUTPUT_SZ_COEFF),
                                  dtype=float)
            iquad_data[:,:,:self.dimz] = self.get_data(x_min, x_max, 
                                                       y_min, y_max, 
                                                       0, self.dimz)
            iquad_calibration_laser_map = calibration_laser_map[
                x_min:x_max, y_min:y_max]
            job_server, ncpus = self._init_pp_server()
            ncpus_max = int(ncpus)
            progress = ProgressBar(int((x_max-x_min)/ncpus_max))

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
                        exposure_time,
                        iquad_calibration_laser_map[ii+ijob,:],
                        nm_laser, step, order, wavenumber,
                        spectral_calibration,
                        base_axis_correction_coeff,
                        OUTPUT_SZ_COEFF),
                    modules=("import numpy as np",
                             "import orb.utils.spectrum",
                             "import orb.utils.vector",
                             "import orb.utils.fft"))) 
                        for ijob in range(ncpus)]

                for ijob, job in jobs:
                    # corrected data comes in place of original data
                    iquad_data[ii+ijob,:,:] = job()

            self._close_pp_server(job_server)
            progress.end()

            # save data
            self._print_msg('Writing quad {}/{} to disk'.format(
                iquad+1, self.QUAD_NB))
            write_start_time = time.time()
            out_cube.write_quad(iquad, data=iquad_data.real)
            self._print_msg('Quad {}/{} written in {:.2f} s'.format(
                iquad+1, self.QUAD_NB, time.time() - write_start_time))
            

        ### update header
        if wavenumber:
            axis = orb.utils.spectrum.create_nm_axis(
                self.dimz, step, order,
                corr=base_axis_correction_coeff)
        else:
            axis = orb.utils.spectrum.create_cm1_axis(
                self.dimz, step, order,
                corr=base_axis_correction_coeff)

        # update standard header
        if standard_header is not None:
            if flux_calibration_vector is not None:
                standard_header.append((
                    'FLAMBDA',
                    np.nanmean(flux_calibration_vector),
                    'mean energy/ADU [erg/cm^2/A/ADU]'))

        hdr = self._get_calibrated_spectrum_header(
            axis, apodization_function, wavenumber=wavenumber)

        hdr.append(('AXISCORR',
                    base_axis_correction_coeff,
                    'Spectral axis correction coeff'))
        
        new_hdr = pyfits.PrimaryHDU(
            np.empty((self.dimx, self.dimy),
                     dtype=float).transpose()).header
        new_hdr.extend(hdr, strip=True, update=True, end=True)
        if correct_wcs is not None:
            hdr = self._update_hdr_wcs(new_hdr, correct_wcs.to_header())
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
                                   fwhm_pix,
                                   step, order, filter_name,
                                   optics_file_path,
                                   calibration_laser_map_path,
                                   nm_laser):
        """Return flux calibration coefficient in [erg/cm2/s/A]/ADU
        from a set of images.
    
        :param std_spectrum_path_1: Path to the standard image list
    
        :param std_spectrum_path_2: Path to the standard image list

        :param std_name: Name of the standard

        :param std_pos_1: X,Y Position of the standard star.in camera 1

        :param std_pos_2: X,Y Position of the standard star.in camera 2

        :param fwhm_pix: Rough FWHM size in pixels.

        :param step: Step size of the spectrum to calibrate

        :param order: Order of the spectrum to calibrate

        :param filter_name: Name fo the filter. If given the
          filter edges can be used to give a weight to the phase
          points. See :meth:`process.Spectrum.correct_filter` for more
          information about the filter file.

        :param optics_file_path: Path to the optics fileé

        :param calibration_laser_map_path: Path to the calibration
          laser map.

        :param nm_laser: Calibration laser wavelength in nm.

        .. note:: This calibration coefficient must be used for non
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
        ERROR_STD_DIFF_COEFF = 1.25
        
        self._print_msg('Computing flux calibration coeff', color=True)
        self._print_msg('Standard Name: %s'%std_name) 
        self._print_msg('Standard image cube 1 path:{}'.format(
            std_image_cube_path_1))
        self._print_msg('Standard image cube 2 path:{}'.format(
            std_image_cube_path_2))

        ## Compute standard flux in erg/cm2/s/A

        # compute correction coeff from angle at center of the frame
        calibration_laser_map = self.read_fits(calibration_laser_map_path)
        corr = calibration_laser_map[
            calibration_laser_map.shape[0]/2,
            calibration_laser_map.shape[1]/2] / nm_laser
        
        # Get standard spectrum in erg/cm^2/s/A
        std = Standard(std_name, config_file_name=self.config_file_name,
                       ncpus=self.ncpus)
        th_spectrum_axis, th_spectrum = std.get_spectrum(
            step, order, STEP_NB,
            wavenumber=False, corr=corr)

        # get filter function
        (filter_function,
         filter_min_pix, filter_max_pix) = (
            FilterFile(filter_name).get_filter_function(
                step, order, STEP_NB,
                wavenumber=False, corr=corr))

        # convert it to erg/cm2/s by summing all the photons
        std_th_flux = th_spectrum * filter_function / np.nanmax(filter_function)
        std_th_flux = np.diff(th_spectrum_axis) * 10. * std_th_flux[:-1]
        std_th_flux = np.nansum(std_th_flux)

        ## std_th_flux = orb.utils.photometry.compute_mean_star_flux(
        ##     th_spectrum, filter_function)


        # compute simulated flux
        std_sim_flux = std.compute_star_flux_in_frame(
            step, order, filter_name,
            optics_file_path, 1, corr=corr)

        self._print_msg('Simulated star flux in one camera: {} ADU/s'.format(
            std_sim_flux))

        ## Compute photometry in real images
        std_x1, std_y1 = std_pos_1
        std_x2, std_y2 = std_pos_2
        
        cube1 = HDFCube(std_image_cube_path_1)
        std_hdr = cube1.get_frame_header(0)
        cube2 = HDFCube(std_image_cube_path_2)
    
        if 'EXPTIME' in std_hdr:
            std_exp_time = std_hdr['EXPTIME']
        else: self._print_error('Integration time (EXPTIME) keyword must be present in the header of the standard image {}'.format(cube1.image_list[0]))
        self._print_msg('Standard integration time: {}s'.format(std_exp_time))
        
        if cube1.dimz == 1:
            self._print_warning('standard image list contains only one file')
            master_im1 = np.copy(cube1[:,:,0])
            master_im2 = np.copy(cube2[:,:,0])
        else:
            master_im1 = orb.utils.image.pp_create_master_frame(
                cube1[:,:,:])
            master_im2 = orb.utils.image.pp_create_master_frame(
                cube2[:,:,:])

        # find star around std_x1, std_y1:
        std_x1, std_y1 =_get_std_position(
            master_im1, BOX_SIZE, std_x1, std_y1)
        # find star around std_x2, std_y2:
        std_x2, std_y2 =_get_std_position(
            master_im2, BOX_SIZE, std_x2, std_y2)
    
        # photometry
        std_flux1, std_flux_err1 = _get_photometry(
            master_im1, std_x1, std_y1, fwhm_pix, std_exp_time)
        self._print_msg('Aperture flux of the standard star in camera 1 is {} [+/-{}] ADU/s'.format(std_flux1, std_flux_err1))

        std_flux2, std_flux_err2 = _get_photometry(
            master_im2, std_x2, std_y2, fwhm_pix, std_exp_time)
        self._print_msg('Aperture flux of the standard star in camera 2 is {} [+/-{}] ADU/s'.format(std_flux2, std_flux_err2))

        self._print_msg('Ratio of real flux/ simulated flux for camera 1: {}'.format(
            std_flux1 / std_sim_flux))
        self._print_msg('Ratio of real flux/ simulated flux for camera 2: {}'.format(
            std_flux2 / std_sim_flux))


        if (max(std_flux1, std_sim_flux)
            / min(std_flux1, std_sim_flux) > ERROR_FLUX_COEFF
            or max(std_flux2, std_sim_flux)
            / min(std_flux2, std_sim_flux) > ERROR_FLUX_COEFF):
            self._print_error('Measured flux is too low compared to simulated flux. There must be a problem. Check standard image files.')
        if (max(std_flux1, std_flux2)
            / min(std_flux1, std_flux2)) > ERROR_STD_DIFF_COEFF:
            self._print_error('Difference between measured flux of both standards is too high')
        
        coeff = std_th_flux / (std_flux1 + std_flux2) # erg/cm2/ADU
        
        self._print_msg('Flux calibration coeff: {} ergs/cm2/ADU'.format(coeff))

        return coeff
        

    def get_flux_calibration_vector(self, std_spectrum_path, std_name,
                                    filter_name):
        """
        Return a flux calibration vector in [erg/cm^2]/ADU on the range
        corresponding to the observation parameters of the spectrum to
        be calibrated.

        The spectrum to be calibrated can then be simply multiplied by
        the returned vector to be converted in [erg/cm^2]

        :param std_spectrum_path: Path to the standard spectrum

        :param std_name: Name of the standard

        :param filter_name: Name fo the filter. If given the
          filter edges can be used to give a weight to the phase
          points. See :meth:`process.Spectrum.correct_filter` for more
          information about the filter file.

        .. note:: Standard spectrum must not be corrected for the
          filter.

        .. warning:: This flux calibration vector is computed from a
          spectrum which as a given modulation efficiency. It must be
          normalized to a modulation efficiency of 100% via the
          calibration coefficient computed from standard images.
        """
        self._print_msg('Computing flux calibration vector', color=True)
        self._print_msg('Standard Name: %s'%std_name)
        self._print_msg('Standard spectrum path: %s'%std_spectrum_path)
        
        # Get real spectrum
        re_spectrum_data, hdr = self.read_fits(std_spectrum_path, return_header=True)
        re_spectrum = re_spectrum_data[:,0]
        
        if len(re_spectrum.shape) > 1:
            self._print_error(
                'Bad standard shape. Standard spectrum must be a 1D vector !')

        # Standard observation parameters
        std_step = hdr['STEP']
        std_order = hdr['ORDER']
        std_exp_time = hdr['EXPTIME']
        std_corr = hdr['AXCORR0']
        std_step_nb = re_spectrum.shape[0]
        std_nm_axis = orb.utils.spectrum.create_nm_axis_ireg(
            std_step_nb, std_step, std_order, corr=std_corr)
        std_cm1_axis = orb.utils.spectrum.create_cm1_axis(
            std_step_nb, std_step, std_order, corr=std_corr)
        
        # Real spectrum is converted to ADU/s
        # We must divide by the total exposition time
        re_spectrum /= std_exp_time * std_step_nb # ADU -> ADU/s

        # Real spectrum is converted to ADU/A/s
        std_nm_axis_reg = orb.utils.spectrum.create_nm_axis(
            std_step_nb, std_step, std_order)
        channel_A = np.abs(np.nanmean(np.diff(std_nm_axis_reg))) * 10.
        re_spectrum /= channel_A

        # Remove portions outside the filter
        (filter_function,
         filter_min_pix, filter_max_pix) = (
            FilterFile(filter_name).get_filter_function(
                std_step, std_order, re_spectrum.shape[0],
                corr=std_corr, wavenumber=True))

        re_spectrum[:filter_min_pix] = np.nan
        re_spectrum[filter_max_pix:] = np.nan
        
        # Get standard spectrum in erg/cm^2/s/A
        std = Standard(std_name, config_file_name=self.config_file_name,
                       ncpus=self.ncpus)
        th_spectrum_axis, th_spectrum = std.get_spectrum(
            std_step, std_order,
            re_spectrum.shape[0], wavenumber=True,
            corr=std_corr)

        th_spectrum[np.nonzero(np.isnan(re_spectrum))] = np.nan

        # fit model * polynomial to adjust model and spectrum
        flux_calibf = orb.utils.photometry.fit_std_spectrum(
            re_spectrum, th_spectrum)

        ## import pylab as pl
        ## pl.axvline(filter_min_pix)
        ## pl.axvline(filter_max_pix)
        ## pl.plot(th_spectrum/np.nanmax(th_spectrum))
        ## pl.plot(re_spectrum/np.nanmax(re_spectrum))
        ## pl.plot(filter_function/np.nanmax(filter_function))
        ## pl.show()

        ## import pylab as pl
        ## pl.plot(std_cm1_axis, re_spectrum * flux_calibf)
        ## pl.plot(std_cm1_axis, th_spectrum)
        ## pl.show()

        self._print_msg('Mean theoretical flux of the star: %e ergs/cm^2/A/s'%orb.utils.stats.robust_mean(th_spectrum))
        self._print_msg('Mean flux of the star in the cube: %e ADU/A/s'%orb.utils.stats.robust_mean(re_spectrum))
        self._print_msg(
            'Mean Flambda calibration: %e ergs/cm^2/[ADU]'%np.nanmean(flux_calibf[~np.isnan(th_spectrum)]))

        return std_cm1_axis, flux_calibf
        
        

#################################################
#### CLASS SourceExtractor ######################
#################################################
class SourceExtractor(InterferogramMerger):


    def _get_extracted_source_interferograms_path(self):
        """Return path to extracted source interferograms"""
        return self._data_path_hdr + "extracted_source_interferograms.fits"

    def _get_extracted_source_interferograms_header(self):
        """Return header of extracted source interferograms data file"""
        return (self._get_basic_header('Extracted source interferograms')
                + self._project_header)

    def _get_extracted_source_spectra_path(self):
        """Return path to extracted source spectra"""
        return self._data_path_hdr + "extracted_source_spectra.fits"

    def _get_extracted_source_spectra_header(self, apodization_function):
        """Return header of extracted source spectra data file"""
        return (self._get_basic_header('Extracted source spectra')
                + self._project_header
                + self._get_fft_params_header(apodization_function))

    def extract_source_interferograms(self, source_list, fov,
                                      alignment_vector_1_path,
                                      alignment_vector_2_path,
                                      modulation_ratio_path,
                                      transmission_vector_path,
                                      ext_illumination_vector_path,
                                      fwhm_arc,
                                      profile_name='gaussian',
                                      moffat_beta=3.5):

        """Extract the interferogram of all sources.
        
        """
        def _fit_sources_in_frame(frameA, frameB, source_listA, box_size,
                                  profile_name, scale, fwhm_pix, default_beta,
                                  fit_tol, alignment_coeffs, rc, zoom_factor,
                                  modulation_ratio, dxA, dyA):
        
            source_listA[:,0] += dxA
            source_listA[:,1] += dyA
            source_listB = orb.utils.astrometry.transform_star_position_A_to_B(
                source_listA, alignment_coeffs, rc, zoom_factor)    

            fit_resA = orb.utils.astrometry.multi_aperture_photometry(
                frameA, source_listA, fwhm_pix, silent=True)
            fit_resB = orb.utils.astrometry.multi_aperture_photometry(
                frameB, source_listB, fwhm_pix, silent=True)
            photomA = [fit['aperture_flux'] for fit in fit_resA]
            photomB = [fit['aperture_flux'] for fit in fit_resB]
                
            return (photomA - modulation_ratio * photomB,
                    photomA + modulation_ratio * photomB)
        
        
        MERGE_BOX_SZ_COEFF = 7        

        fwhm_arc_A = float(fwhm_arc)
        fwhm_arc_B = float(fwhm_arc)
        
        alignment_coeffs = [self.dx, self.dy, self.dr, self.da, self.db]

        alignment_vector_1 = self.read_fits(
            alignment_vector_1_path)
        alignment_vector_2 = self.read_fits(
            alignment_vector_2_path)
        modulation_ratio = self.read_fits(
            modulation_ratio_path)
        transmission_vector = self.read_fits(
            transmission_vector_path)
        ext_illumination_vector = self.read_fits(
            ext_illumination_vector_path)
        
        astrom = Astrometry(self.cube_B, fwhm_arc_B, fov,
                            profile_name=profile_name,
                            moffat_beta=moffat_beta,
                            data_prefix=self._data_prefix + 'merged.',
                            tuning_parameters=self._tuning_parameters,
                            check_mask=False,
                            config_file_name=self.config_file_name,
                            ncpus=self.ncpus)
        
        source_list = np.array(source_list)
        astrom.reset_star_list(source_list)

        astrom.fit_results = orb.astrometry.StarsParams(
            source_list.shape[0], self.cube_A.dimz,
            silent=self._silent,
            config_file_name=self.config_file_name,
            ncpus=self.ncpus)

        job_server, ncpus = self._init_pp_server()

        frames = np.empty((self.cube_A.dimx, self.cube_A.dimy, ncpus),
                          dtype=float)
        
        photom = np.empty((source_list.shape[0],
                           self.cube_A.dimz), dtype=float)
        
        # transm is not used at the moment ...
        transm = np.empty((source_list.shape[0],
                           self.cube_A.dimz), dtype=float)
        
        progress = ProgressBar(int(self.cube_A.dimz), silent=self._silent)
        # fit all sources in each frame
        for ik in range(0, self.cube_A.dimz, ncpus):
            progress.update(ik, info="frame : " + str(ik))
            
            # no more jobs than frames to compute
            if (ik + ncpus >= self.cube_A.dimz):
                ncpus = self.cube_A.dimz - ik

            framesA = self.cube_A.get_data(0, self.cube_A.dimx,
                                           0, self.cube_A.dimy,
                                           ik, ik+ncpus, silent=True)
            framesB = self.cube_B.get_data(0, self.cube_B.dimx,
                                           0, self.cube_B.dimy,
                                           ik, ik+ncpus, silent=True)

            # get stars photometry for each frame
            jobs = [(ijob, job_server.submit(
                _fit_sources_in_frame,
                args=(framesA[:,:,ijob], framesB[:,:,ijob],
                      source_list, astrom.box_size,
                      astrom.profile_name,
                      astrom.scale,
                      astrom.fwhm_pix,
                      astrom.default_beta,
                      astrom.fit_tol,
                      alignment_coeffs, self.rc, self.zoom_factor,
                      modulation_ratio,
                      alignment_vector_1[ik+ijob,0],
                      alignment_vector_1[ik+ijob,1]),
                modules=("from orb.utils.astrometry import fit_star, sky_background_level, aperture_photometry, get_profile",
                         "from orb.astrometry import StarsParams",
                         "import orb.astrometry",
                         "import orb.utils.image",
                         "import orb.utils.astrometry",
                         "import numpy as np",
                         "import math",
                         "import orb.utils as utils",
                         "import orb.cutils as cutils",
                         "import bottleneck as bn",
                         "import warnings")))
                    for ijob in range(ncpus)]
            
            for ijob, job in jobs:
                photom[:,ik+ijob], transm[:,ik+ijob] = job()
            
        self._close_pp_server(job_server)
        progress.end()

        # interferogram correction
        for i in range(photom.shape[0]):
            photom[i,:] = photom[i,:] / transmission_vector

        self.write_fits(
            self._get_extracted_source_interferograms_path(),
            photom,
            fits_header=self._get_extracted_source_interferograms_header(),
            overwrite=self.overwrite)

        if self.indexer is not None:
            self.indexer['extracted_source_interferograms'] = (
                self._get_extracted_source_interferograms_path())

        return photom


    def compute_source_spectra(self, source_list, source_interf_path,
                               step, order, apodization_function,
                               filter_name, phase_map_paths,
                               nm_laser,
                               calibration_laser_map_path,
                               phase_correction=True,
                               optimize_phase=False,
                               wavenumber=True,
                               filter_correction=True,
                               cube_A_is_balanced=True,
                               zpd_shift=None,
                               phase_order=None,
                               return_phase=False):
        
        """Compute source spectra


        :param phase_order: (Optional) If phase_map_paths is set to None, phase_order must be given.
        """

        RANGE_BORDER_COEFF = 0.1
        
        source_interf = self.read_fits(source_interf_path)

        if phase_map_paths is None and phase_order is None and phase_correction:
            self._print_error('If phase correction is required and phase_map_paths is not given, phase must be computed for each source independantly and phase_order must be set.')

        if filter_name is None:
            self._print_error('No filter name given')

        
        if len(source_interf.shape) == 1:
            source_interf = np.array([source_interf])


        # get calibration laser map
        calibration_laser_map = self.read_fits(
            calibration_laser_map_path)
        
        # wavenumber
        if wavenumber:
            self._print_msg('Wavenumber (cm-1) output')
        else:
            self._print_msg('Wavelength (nm) output')

        
        source_list = np.array(source_list)

        # create spectra array (spec_nb, spec_size, 2) : spectra are in [:,0]
        # and spectral axes are in [:,1]
        spectra = np.empty((source_interf.shape[0],
                            source_interf.shape[1], 2), dtype=float)
        spectra.fill(np.nan)
        step_nb = source_interf.shape[1]

        # get zpd shift
        if zpd_shift is None:
            zpd_shift = orb.utils.fft.find_zpd(
                self.cube_A.get_zmedian(nozero=True),
                return_zpd_shift=True)
        
        self._print_msg("ZPD shift: {}".format(zpd_shift))

        # load phase maps
        if (phase_map_paths is not None and phase_correction):
            phase_maps = list()
            for phase_map_path in phase_map_paths:
                phase_maps.append(self.read_fits(phase_map_path))
                self._print_msg('Loaded phase map: {}'.format(phase_map_path))
            
        else:
            phase_maps = None
        
        # get high order phase
        if filter_name is not None:
            phf = PhaseFile(filter_name)
            self._print_msg('Phase file: {}'.format(phf.improved_path))
            
        self._print_msg("Apodization function: %s"%apodization_function)
        self._print_msg("Folding order: %f"%order)
        self._print_msg("Step size: %f"%step)
        self._print_msg("ZPD shift: %f"%zpd_shift)

        
        hdr = self._get_extracted_source_spectra_header(
            apodization_function)

        for isource in range(source_interf.shape[0]):
            x = source_list[isource,0]
            y = source_list[isource,1]
            interf = source_interf[isource, :]
            
            nm_laser_obs = calibration_laser_map[int(x), int(y)]
            
            if phase_correction:                
                coeffs_list = list()
                if phase_maps is not None:
                    for phase_map in phase_maps:
                        if np.size(phase_map) > 1:
                            coeffs_list.append(phase_map[int(x), int(y)])
                        else:
                            coeffs_list.append(phase_map)
                else:
                    optimize_phase = True
                    coeffs_list = np.empty(phase_order+1, dtype=float)
                            
                if optimize_phase:
                    guess = np.array(coeffs_list)
                    guess.fill(0.)
                    
                    ext_phase = orb.utils.fft.optimize_phase(
                        interf, step, order, zpd_shift,
                        nm_laser_obs, nm_laser, guess=guess,
                        high_order_phase=phf)
                else:
                    ext_phase = np.polynomial.polynomial.polyval(
                        np.arange(step_nb), coeffs_list)

            else:
                ext_phase = None

            # replace zeros by nans
            interf[np.nonzero(interf == 0)] = np.nan

            if return_phase:
                phase_correction = False
                ext_phase = None
            
            spec = orb.utils.fft.transform_interferogram(
                interf, nm_laser, nm_laser_obs, step, order,
                apodization_function, zpd_shift,
                phase_correction=phase_correction,
                ext_phase=ext_phase,
                wavenumber=wavenumber,
                wave_calibration=False,
                return_phase=return_phase)

            if wavenumber:
                spec_axis = orb.utils.spectrum.create_cm1_axis(
                    spec.shape[0], step, order, corr=nm_laser_obs/nm_laser)
            else:
                spec_axis = orb.utils.spectrum.create_nm_axis(
                    spec.shape[0], step, order, corr=nm_laser_obs/nm_laser)

            if filter_correction:
                # load filter function
                (filter_function,
                 filter_min_pix, filter_max_pix) = (
                    FilterFile(filter_name).get_filter_function(
                        step, order, step_nb,
                        wavenumber=wavenumber,
                        corr=nm_laser_obs/nm_laser))    
       
                spec /= filter_function
                spec[:filter_min_pix] = np.nan
                spec[filter_max_pix:] = np.nan

            # check polarity 
            if optimize_phase:
                if np.nanmean(spec) < 0.:
                    self._print_warning('Negative polarity of the spectrum')
                    spec = -spec

            spectra[isource, :, 0] = spec
            spectra[isource, :, 1] = spec_axis

            hdr.append(('AXCORR{}'.format(isource),
                        nm_laser_obs/nm_laser,
                        'Spectral axis correction coeff for source {}'.format(isource)))

        if wavenumber:
            wave_type = 'WAVENUMBER'
        else:
            wave_type = 'WAVELENGTH'
        hdr.append(('WAVTYPE', '{}'.format(wave_type),
                    'Spectral axis type: wavenumber or wavelength'))
        
        self.write_fits(
            self._get_extracted_source_spectra_path(),
            spectra,
            fits_header=hdr,
            overwrite=self.overwrite)
        
        if self.indexer is not None:
            self.indexer['extracted_source_spectra'] = (
                self._get_extracted_source_spectra_path())
            
            

        

#################################################
#### CLASS PhaseMaps ############################
#################################################
class PhaseMaps(Tools):

    phase_maps = None

    def __init__(self, phase_map_path_list,
                 residual_map_path_list,
                 dimx_unbinned, dimy_unbinned,
                 project_header=list(),
                 calibration_laser_header=list(),
                 overwrite=False, indexer=None, **kwargs):
        """Initialize PhaseMaps class.

        :param phase_map_path_list: List of phase map paths as a
          tuple.

        :param residual_map_path_list: List of residual map paths as a
          tuple.

        :param dimx_unbinned: Unbinned X dimension of the maps.
        
        :param dimy_unbinned: Unbinned Y dimension of the maps.
        
        :param project_header: (Optional) header section describing
          the observation parameters that can be added to each output
          files (an empty list() by default).
          
        :param calibration_laser_header: (Optional) header section
          describing the calibration laser parameters that can be
          added to the concerned output files e.g. calibration laser map,
          spectral cube (an empty list() by default).

        :param overwrite: (Optional) If True existing FITS files will
          be overwritten (default False).

        :param indexer: (Optional) Must be a :py:class:`core.Indexer`
          instance. If not None created files can be indexed by this
          instance.    

        :param kwargs: Kwargs are :meth:`core.Tools` properties.
        """
        Tools.__init__(self, **kwargs)
        self.overwrite = overwrite
        self._project_header = project_header
        self._calibration_laser_header = calibration_laser_header
        self.indexer = indexer
        

        self.phase_maps = [self.read_fits(phase_map_path)
                           for phase_map_path
                           in phase_map_path_list]

        self.residual_maps = [self.read_fits(phase_map_path)
                              for phase_map_path
                              in residual_map_path_list]

        self._print_msg('Number of maps loaded: {}'.format(len(self.phase_maps)))
        
        # detect binning
        self.dimx = self.phase_maps[0].shape[0]
        self.dimy = self.phase_maps[0].shape[1]
        self._print_msg('Shape ({}, {})'.format(self.dimx, self.dimy))
        self.dimx_unbinned = int(dimx_unbinned)
        self.dimy_unbinned = int(dimy_unbinned)
        binx = self.dimx_unbinned/self.dimx
        biny = self.dimy_unbinned/self.dimy
        if binx != biny: self._print_error('Binning along x and y axes is different ({} != {})'.format(
            binx, biny))
        else: self.binning = binx
        self._print_msg('Detected maps binning: {}'.format(
            self.binning))

    def _get_phase_map_path(self, order, phase_map_type=None):
        """Return the default path to the phase map.

        :param order: Order of the parameter of the polynomial fitted
          to the phase.

        :param phase_map_type: (Optional) Type of phase map. Must be
          None, 'unwraped', 'fitted', 'error' or 'residual'
        """
        if phase_map_type is not None:
            if phase_map_type == 'unwraped':
                pm_type = "_unwraped"
            elif phase_map_type == 'fitted':
                pm_type = "_fitted"
            elif phase_map_type == 'error':
                pm_type = "_fitted_error"
            elif phase_map_type == 'unbinned':
                pm_type = "_unbinned"
            elif phase_map_type != 'residual':
                self._print_error("Phase_map_type must be set to 'unwraped', 'fitted', 'error', 'residual', 'unbinned' or None")
        else:
            pm_type = ""
            
      
        if phase_map_type != 'residual':
            return self._data_path_hdr + "phase_map%s_order_%d.fits"%(
                pm_type, order)
        else:
            return self._data_path_hdr + "phase_map_residual.fits"
    
    def _get_phase_map_header(self, order, phase_map_type=None):
        """Return the header of the phase map.

        :param order: Order of the parameter of the polynomial fitted
          to the phase.

        :param phase_map_type: (Optional) Type of phase map. Must be
          None, 'unwraped', 'fitted', 'error', 'unbinned' or 'residual'

        .. note:: unwraped, fitted and error are incompatible. If more
          than one of those options are set to True the priority order
          is unwraped, then fitted, then error.
        """
        if phase_map_type is not None:
            if phase_map_type == 'unwraped':
                header = self._get_basic_header(
                    'Unwraped phase map order %d'%order)
            elif phase_map_type == 'fitted':
                header = self._get_basic_header(
                    'Fitted phase map order %d'%order)
            elif phase_map_type == 'error':
                header = self._get_basic_header(
                    'Fitted phase error map order %d'%order)
            elif phase_map_type == 'residual':
                header = self._get_basic_header(
                    'Residual map on phase fit')
            elif phase_map_type == 'unbinned':
                header = self._get_basic_header(
                    'Unbinned phase map order %d'%order)    
            else:
                self._print_error("Phase_map_type must be set to 'unwraped', 'fitted', 'error', 'residual', 'unbinned' or None")
        else:
            header = self._get_basic_header('Phase map order %d'%order)
       
        if self.dimx is not None and self.dimy is not None:
            return (header
                    + self._project_header
                    + self._calibration_laser_header
                    + self._get_basic_frame_header(self.dimx, self.dimy))
        else:
            self._print_warning("Basic header could not be created (frame dimensions are not available)")
            return (header + self._project_header + self._calibration_header)

    def _get_calibration_laser_map_path(self):
        """Return path to the calibration laser map created during the
        fitting procedure of the phase map (with a SITELLE's fit model
        of the phase)."""
        return self._data_path_hdr + 'calibration_laser_map.fits'

    def _get_calibration_laser_map_header(self):
        """Return a calibration laser map header."""
        return (self._get_basic_header('Calibration laser map')
                + self._project_header
                + self._calibration_laser_header
                + self._get_basic_frame_header(self.dimx, self.dimy))


    def unwrap_phase_map_0(self):
        """Unwrap order 0 phase map.

        Phase is defined modulo pi/2. The Unwrapping is a
        reconstruction of the phase so that the distance between two
        neighboor pixels is always less than pi/4 so that the real
        phase pattern can be rcovered and fitted easily.

        The idea is the same as with np.unwrap() but in 2D, on a
        possibly very noisy map, so that a naive 2d unwrapping cannot
        be done.
        """
        
        BIN_SIZE = 20
        LINE_SIZE = 30


        phase_map = self.phase_maps[0]
        
        def unwrap(val, target):
            while abs(val - target) > math.pi / 2.:
                if val  - target > 0. :
                    val -= math.pi
                else:
                    val += math.pi
            return val

        def unwrap_columns(pm0, bin_size):
            for ii in range(pm0.shape[0]):
                for ij in range(pm0.shape[1]):
                    colbin = pm0[ii,ij:ij+bin_size+1]
                    colbin_med = np.nanmedian(colbin)
                    for ik in range(colbin.shape[0]):
                        colbin[ik] = unwrap(colbin[ik], colbin_med)
                    pm0[ii,ij:ij+bin_size+1] = colbin
            return pm0

        def unwrap_all(pm0, bin_size):
            test_line = np.nanmedian(
                pm0[:, pm0.shape[1]/2-LINE_SIZE/2:pm0.shape[1]/2+LINE_SIZE/2],
                axis=1)
            test_line_init = np.copy(test_line)
            for ii in range(0, test_line.shape[0]-bin_size/2):
                linebin = test_line[ii:ii+bin_size+1]
                linebin_med = np.nanmedian(orb.utils.stats.sigmacut(
                    linebin, sigma=2))
                for ik in range(linebin.shape[0]):
                    linebin[ik] = unwrap(linebin[ik], linebin_med)
                test_line[ii:ii+bin_size+1] = linebin
            diff = test_line - test_line_init
            pm0 = (pm0.T + diff.T).T
            return pm0
        
        # unwrap pixels along columns
        phase_map = unwrap_columns(phase_map, BIN_SIZE)
        # unwrap columns along a line
        phase_map = unwrap_all(phase_map, BIN_SIZE)

        phase_map[np.nonzero(np.isnan(phase_map))] = 0.
        
        # Save unwraped map
        self.phase_map_order_0_unwraped = np.copy(phase_map)
        phase_map_path = self._get_phase_map_path(0, phase_map_type='unwraped')

        self.write_fits(phase_map_path,
                        orb.cutils.unbin_image(phase_map,
                                               self.dimx_unbinned,
                                               self.dimy_unbinned), 
                        fits_header=self._get_phase_map_header(
                            0, phase_map_type='unwraped'),
                        overwrite=self.overwrite)
        if self.indexer is not None:
            self.indexer['phase_map_unwraped_0'] = phase_map_path

    def fit_phase_map(self, order, calibration_laser_map_path, nm_laser):
        """Robust fit using Zernike modes fit of phase maps of order > 0

        :param order: Order of the phase map to fit.

        :param calibration_laser_map_path: Path to the calibration laser map.

        :param nm_laser: Calibration laser wavelength in nm
        
        .. seealso:: :py:meth:`orb.utils.image.fit_highorder_phase_map`
    
        """
        if order == 0:
            self._print_warning('This function is better used for phase maps of order > 0')
        if order not in range(len(self.phase_maps)):
            self._print_error('Order {} does not exist'.format(order))
        phase_map = np.copy(self.phase_maps[order])
        res_map = np.copy(self.residual_maps[1])

        # load calibration laser map
        calibration_laser_map = self.read_fits(calibration_laser_map_path)
        if (calibration_laser_map.shape[0] != self.dimx_unbinned):
            calibration_laser_map = orb.utils.image.interpolate_map(
                calibration_laser_map, self.dimx_unbinned, self.dimy_unbinned)
        calibration_laser_map = orb.utils.image.nanbin_image(
            calibration_laser_map, self.binning)

        
        (fitted_phase_map,
         error_map) = orb.utils.image.fit_highorder_phase_map(
            phase_map, res_map, calibration_laser_map, nm_laser)

        ## save fitted phase map and error map
        error_map_path = self._get_phase_map_path(
            order, phase_map_type='error')
        fitted_map_path = self._get_phase_map_path(
            order, phase_map_type='fitted')
        unbinned_map_path = self._get_phase_map_path(
            order, phase_map_type='unbinned')

        self.write_fits(
            error_map_path,
            orb.cutils.unbin_image(error_map,
                                   self.dimx_unbinned,
                                   self.dimy_unbinned),
            fits_header= self._get_phase_map_header(
                order, phase_map_type='error'),
            overwrite=self.overwrite)
        if self.indexer is not None:
            self.indexer[
                'phase_map_fitted_error_{}'.format(order)] = error_map_path
            
        self.write_fits(
            fitted_map_path,
            orb.cutils.unbin_image(fitted_phase_map,
                                   self.dimx_unbinned,
                                   self.dimy_unbinned), 
            fits_header= self._get_phase_map_header(
                order, phase_map_type='fitted'),
            overwrite=self.overwrite)
        if self.indexer is not None:
            self.indexer['phase_map_fitted_{}'.format(order)] = fitted_map_path

        self.write_fits(
            unbinned_map_path,
            orb.cutils.unbin_image(np.copy(self.phase_maps[order]),
                                   self.dimx_unbinned,
                                   self.dimy_unbinned), 
            fits_header= self._get_phase_map_header(
                order, phase_map_type='unbinned'),
            overwrite=self.overwrite)
        if self.indexer is not None:
            self.indexer['phase_map_unbinned_{}'.format(order)] = unbinned_map_path

            
    def reduce_phase_map(self, order):
        """Reduce a phase map to only one number.

        .. note:: The reduced phase maps are also saved as 'fitted'.

        :param order: Order of the phase map to reduce.
        """
        if order == 0:
            self._print_warning('This function is better used for phase maps of order > 0')
        if order not in range(len(self.phase_maps)):
            self._print_error('Order {} does not exist'.format(order))
        phase_map = np.copy(self.phase_maps[order])
        res_map = np.copy(self.residual_maps[1])

        # reduction
        phase_coeff = orb.utils.fft.compute_phase_coeffs_vector(
            [phase_map], res_map=res_map)[0]
        self._print_msg('Reduced coeff of order {}: {}'.format(
            order, phase_coeff))

        ## save fitted phase map and error map
        ## error_map_path = self._get_phase_map_path(order, phase_map_type='error')
        fitted_map_path = self._get_phase_map_path(order, phase_map_type='fitted')

        ## self.write_fits(
        ##     error_map_path,
        ##     orb.cutils.unbin_image(error_map,
        ##                            self.dimx_unbinned,
        ##                            self.dimy_unbinned),
        ##     fits_header= self._get_phase_map_header(order, phase_map_type='error'),
        ##     overwrite=self.overwrite)
        ## if self.indexer is not None:
        ##     self.indexer['phase_map_fitted_error_{}'.format(order)] = error_map_path
            
        self.write_fits(
            fitted_map_path,
            np.array(phase_coeff, dtype=float), 
            fits_header= self._get_phase_map_header(order, phase_map_type='fitted'),
            overwrite=self.overwrite)
        if self.indexer is not None:
            self.indexer['phase_map_fitted_{}'.format(order)] = fitted_map_path


    def fit_phase_map_0(self, 
                        phase_model='spiomm',
                        calibration_laser_map_path=None,
                        calibration_laser_nm=None):
        """Fit the order 0 phase map.

        Help remove most of the noise. This process is useful if the
        phase map has been computed from astronomical data without a
        high SNR for every pixel in all the filter bandpass.

        :param phase_model: (Optional) Model used to fit phase, can be
          'spiomm' or 'sitelle'. If 'spiomm', the model is a very
          basic one considering only linear variation of the phase map
          along the columns. If 'sitelle', the phase model is based on
          a the calibration laser map, tip tilt and rotation of the
          calibration laser map are considered in the fit, along with
          the first order of the phase (default 'spiomm').

        :param calibration_laser_map_path: (Optional) Path to the
          calibration laser map. Only useful if the phase model is
          'sitelle' (default None).

        :param calibration_laser_nm: (Optional) Wavelength of the
          calibration laser (in nm). Only useful if the phase model is
          'sitelle' (default None).
          
        """

        FIT_DEG = 1 # Degree of the polynomial used for the fit
        
        MAX_FIT_ERROR = 0.1 # Maximum fit error over which a warning
                            # is raised
                       
        BORDER = 0.02 # percentage of the image length removed on the
                      # border to fit the phase map (cannot be more
                      # than 0.5)

        phase_map = np.copy(self.phase_map_order_0_unwraped)
        # border points are removed
        mask = np.ones_like(phase_map, dtype=bool)
        border = (self.dimx + self.dimy)/2. * BORDER 
        mask[border:-border,border:-border:] = False
        mask[np.nonzero(phase_map == 0.)] = True
        phase_map[np.nonzero(mask)] = np.nan
        # error map
        err_map = self.residual_maps[0]
        err_map[np.nonzero(mask)] = np.nan
        err_map = err_map**0.5

        if phase_model == 'sitelle' and calibration_laser_map_path is None:
            self._print_error('Calibration laser map must be set for a SITELLE phase map fit')

        # load calibration laser map
        calibration_laser_map = self.read_fits(calibration_laser_map_path)
        if (calibration_laser_map.shape[0] != self.dimx_unbinned):
            calibration_laser_map = orb.utils.image.interpolate_map(
                calibration_laser_map, self.dimx_unbinned, self.dimy_unbinned)
        calibration_laser_map = orb.utils.image.nanbin_image(
            calibration_laser_map, self.binning)
        
        if phase_model == 'spiomm':
            fitted_phase_map, error_map, fit_error = orb.utils.image.fit_map(
                phase_map, err_map, FIT_DEG)
            new_calibration_laser_map = calibration_laser_map

        elif phase_model == 'sitelle':
            (fitted_phase_map, error_map,
             fit_error, new_calibration_laser_map, trans_coeffs) = (
                orb.utils.image.fit_sitelle_phase_map(
                    np.copy(phase_map), np.copy(err_map),
                    np.copy(calibration_laser_map),
                    calibration_laser_nm,
                    pixel_size=(
                        float(self._get_config_parameter('PIX_SIZE_CAM1'))
                        * self.binning),
                    binning=1, return_coeffs=True))

        self._print_msg(
            "Normalized root-mean-square deviation on the fit: %f%%"%(
                fit_error*100.))

        ## compute max flux error due to order 0 error. An error of
        ## pi/2 is equivalent to 100% loss of flux. Note that noise
        ## plus model errors are considered at the same
        ## time. Therefore this estimation is very conservative.
        res_std = orb.utils.stats.robust_std(
            orb.utils.stats.sigmacut(
                error_map, sigma=3)) # in radians
        max_flux_error = res_std / (math.pi / 2.) 
        self._print_msg(
            "Phase Map order 0 residual std: {}, Max flux error (conservative estimate): {}%".format(res_std, max_flux_error*100.))

        if fit_error > MAX_FIT_ERROR:
            self._print_warning("Normalized root-mean-square deviation on the fit is too high (%f > %f): phase correction will certainly be uncorrect !"%(fit_error, MAX_FIT_ERROR))
        
        ## save fitted phase map and error map
        error_map_path = self._get_phase_map_path(0, phase_map_type='error')
        fitted_map_path = self._get_phase_map_path(0, phase_map_type='fitted')

        if new_calibration_laser_map is not None:
            self.write_fits(
                self._get_calibration_laser_map_path(),
                orb.cutils.unbin_image(new_calibration_laser_map,
                                       self.dimx_unbinned,
                                       self.dimy_unbinned), 
                fits_header=self._get_calibration_laser_map_header(),
                overwrite=self.overwrite)
            if self.indexer is not None:
                self.indexer['phase_calibration_laser_map'] = self._get_calibration_laser_map_path()
        
        self.write_fits(
            error_map_path,
            orb.cutils.unbin_image(error_map,
                                   self.dimx_unbinned,
                                   self.dimy_unbinned),
            fits_header= self._get_phase_map_header(0, phase_map_type='error'),
            overwrite=self.overwrite)
        if self.indexer is not None:
            self.indexer['phase_map_fitted_error_0'] = error_map_path
            
        self.write_fits(
            fitted_map_path,
            orb.cutils.unbin_image(fitted_phase_map,
                                   self.dimx_unbinned,
                                   self.dimy_unbinned), 
            fits_header= self._get_phase_map_header(0, phase_map_type='fitted'),
            overwrite=self.overwrite)
        if self.indexer is not None:
            self.indexer['phase_map_fitted_0'] = fitted_map_path
       
        
