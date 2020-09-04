#!/usr/bin/python
# *-* coding: utf-8 *-*
# Author: Thomas Martin <thomas.martin.1@ulaval.ca>
# File: phase.py

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
The phase module contains phase processing related classes

Phase processing is generally made on small binned cubes
"""

__author__ = "Thomas Martin"
__licence__ = "Thomas Martin (thomas.martin.1@ulaval.ca)"                      
__docformat__ = 'reStructuredText'

import numpy as np
import logging

import orb.fft
import orb.core
import orb.cube
import warnings
import orb.utils
import orb.utils.io
import gvar

#################################################
#### CLASS BinnedInterferogramCube ##############
#################################################
class BinnedInterferogramCube(orb.cube.InterferogramCube):

    def compute_phase(self, path, calibrate=True):
        """Compute binned phase cube

        :param path: path of the output binned phase cube.
        """

        def compute_phase_in_column(col, calib_col, zpd_index,
                                    params, base_axis, calibrate):
            warnings.simplefilter('ignore', RuntimeWarning)
            warnings.simplefilter('ignore', UserWarning)
            warnings.simplefilter('ignore', FutureWarning)
            
            phase_col = np.empty_like(col)
            phase_col.fill(np.nan)
            for ij in range(col.shape[0]):
                interf = orb.fft.Interferogram(
                    col[ij,:],
                    params=params,
                    zpd_index=zpd_index,
                    calib_coeff=calib_col[ij])
                interf = interf.symmetric()
                interf = interf.subtract_mean()
                interf = interf.apodize('1.5')
                spectrum = interf.transform()
                if isinstance(spectrum, orb.fft.Spectrum): # test if transform ok
                    if calibrate:
                        spectrum = spectrum.interpolate(base_axis, quality=10)
                    iphase = np.copy(spectrum.get_phase().data)
                    # note: the [:iphase.size] as been addedwhen the phase is
                    # not interpolated (calibrate=False) because the
                    # number of samples can be smaller after the
                    # symmetric() step
                    phase_col[ij,:iphase.size] = iphase
            return phase_col

        calib_map = self.get_calibration_coeff_map()
        base_axis = np.copy(self.get_base_axis().data)
        phase_cube = np.empty(self.shape, dtype=float)
        phase_cube.fill(np.nan)

        job_server, ncpus = self._init_pp_server()
        progress = orb.core.ProgressBar(self.dimx)
        for ii in range(0, self.dimx, ncpus):
                progress.update(
                    ii, info="computing column {}/{}".format(
                        ii, self.dimx))
                if ii + ncpus >= self.dimx:
                    ncpus = self.dimx - ii

                jobs = [(ijob, job_server.submit(
                    compute_phase_in_column, 
                    args=(self[ii+ijob,:,:],
                          calib_map[ii+ijob,:],
                          self.params.zpd_index,
                          self.params.convert(),
                          base_axis, calibrate),
                    modules=("import logging",
                             "import warnings",
                             "import numpy as np",
                             "import orb.fft")))
                        for ijob in range(ncpus)]

                for ijob, job in jobs:
                    # new data is written in place of old data
                    phase_cube[ii+ijob,:,:] = job()
                    
        progress.end()
        self._close_pp_server(job_server)

        out_cube = orb.cube.RWHDFCube(
            path,
            shape=self.shape,
            instrument=self.instrument,
            config=self.config,
            params=self.params,
            reset=True)
        
        out_cube[:,:,:] = phase_cube
        
        del out_cube

        return path

 
#################################################
#### CLASS BinnedPhaseCube ######################
#################################################
class BinnedPhaseCube(orb.cube.Cube):

    def get_phase_maps_path(self, suffix=None):
        if suffix is None: suffix = ''       
        else:
            if not isinstance(suffix, str):
                raise TypeError('suffix must be a string')
            suffix = '.' + suffix
        return self._data_path_hdr + 'phase_maps{}.hdf5'.format(suffix)

    def get_phase(self, x, y):
        """Return a phase vector at position x, y

        :param x: x position
        :param y: y position
        """
        return orb.fft.Phase(
            self[x, y, :], self.get_base_axis(), params=self.params)
        
    def polyfit(self, polydeg, coeffs=None, suffix=None, high_order_phase=None):
        """Create phase maps from a polynomial fit of the binned phase cube

        :param polydeg: Degree of the fitting polynomial. Must be >= 0.

        :param coeffs: Used to fix some coefficients to a given
          value. If not None, must be a list of length = polydeg +
          1. set a coeff to a np.nan or a None to let the parameter
          free. Each coefficient can also be a map (like the one
          returned by PhaseMaps.get_map()).

        :param suffix: Phase maps hdf5 file suffix (added before the
          extension .hdf5)

        :param high_order_phase: Phase vector removed before
          fitting. Must be an orb.fft.Phase instance.

        :return: Path to the phase maps file (can then be opened with
          PhaseMaps).

        """
        def fit_phase_in_column(col, deg, incoeffs_col, params, base_axis,
                                high_order_phase_proj, calibcoeff_col):
            warnings.simplefilter('ignore', RuntimeWarning)
            outcoeffs_col = np.empty((col.shape[0], deg + 1), dtype=float)
            outcoeffs_col.fill(np.nan)
            outcoeffs_err_col = np.empty((col.shape[0], deg + 1), dtype=float)
            outcoeffs_err_col.fill(np.nan)
            if high_order_phase_proj is not None:
                _ho_phase = orb.fft.Phase(high_order_phase_proj, base_axis, params)
            else:
                _ho_phase = None
            for ij in range(col.shape[0]):
                icoeffs = tuple([icoeff[ij] for icoeff in incoeffs_col])
                _phase = orb.fft.Phase(col[ij,:], base_axis, params)
                if _ho_phase is not None:
                    _phase = _phase.subtract(_ho_phase)
                try:
                    outcoeffs_col[ij,:], outcoeffs_err_col[ij,:] = _phase.polyfit(
                        deg, coeffs=icoeffs, calib_coeff=calibcoeff_col[ij],
                        return_coeffs=True)
                except orb.utils.err.FitError:
                    logging.debug('fit error')
            return outcoeffs_col, outcoeffs_err_col                  

        if not isinstance(polydeg, int): raise TypeError('polydeg must be an integer')
        if polydeg < 0: raise ValueError('polydeg must be >= 0')

        calibcoeff_map = self.get_calibration_coeff_map()
        
        if high_order_phase is not None:
            if not isinstance(high_order_phase, orb.fft.Phase):
                raise TypeError('high_order_phase must be an orb.fft.Phase instance')
            high_order_phase_proj = high_order_phase.project(self.get_base_axis()).data
        else: 
            high_order_phase_proj = None

        logging.info('Coefficients: {}'.format(coeffs))
        
        if coeffs is None:
            coeffs = [None] * (polydeg + 1)
        else:
            coeffs = list(coeffs)

        if len(coeffs) != polydeg + 1: raise TypeError('coeffs must have length {}'.format(
                polydeg+1))
        
        for icoeff in range(len(coeffs)):
            if coeffs[icoeff] is None: coeffs[icoeff] = np.nan

            if isinstance(coeffs[icoeff], np.ndarray):
                if coeffs[icoeff].shape != (self.dimx, self.dimy):
                    raise TypeError('coefficient map must have the same shape as the cube {}'.format((self.dimx, self.dimy)))
                if np.any(np.isnan(coeffs[icoeff])):
                    raise ValueError('coefficient map contains at least one NaN')
            elif isinstance(coeffs[icoeff], float):
                # float coefficient is converted to a map
                coeffs[icoeff] = np.ones((self.dimx, self.dimy), dtype=float) * coeffs[icoeff]
            else:    
                raise TypeError('coefficient {} ({}) has type {} but must be a float, a numpy.ndarray map or None'.format(icoeff + 1, coeffs[icoeff], type(coeffs[icoeff])))


        coeffs_cube = np.empty((self.dimx, self.dimy, polydeg + 1), dtype=float)
        coeffs_cube.fill(np.nan)
        coeffs_err_cube = np.empty((self.dimx, self.dimy, polydeg + 1), dtype=float)
        coeffs_err_cube.fill(np.nan)
            
        base_axis = np.copy(self.get_base_axis().data)
        
        job_server, ncpus = self._init_pp_server()
        progress = orb.core.ProgressBar(self.dimx)
        
        for ii in range(0, self.dimx, ncpus):
            progress.update(
                ii,
                info="computing column {}/{}".format(ii, self.dimx))
            if ii + ncpus >= self.dimx:
                ncpus = self.dimx - ii

            jobs = [(ijob, job_server.submit(
                fit_phase_in_column, 
                args=(np.copy(self[ii+ijob,:,:]),
                      polydeg,
                      [icoeff[ii+ijob,:] for icoeff in coeffs],
                      self.params.convert(), np.copy(base_axis),
                      high_order_phase_proj, calibcoeff_map[ii+ijob,:]),
                modules=("import logging",
                         "import warnings",
                         "import numpy as np",
                         "import orb.fft",
                         "import orb.utils.err")))
                    for ijob in range(ncpus)]

            for ijob, job in jobs:
                coeffs_cube[ii+ijob,:,:], coeffs_err_cube[ii+ijob,:,:] = job() 
        progress.end()
        self._close_pp_server(job_server)

        coeffs_err_cube[np.nonzero(coeffs_cube == 0)] = np.nan
        coeffs_cube[np.nonzero(coeffs_cube == 0)] = np.nan
        
        coeffs_cube[:,:,0] = orb.utils.image.unwrap_phase_map0(coeffs_cube[:,:,0])
        coeffs_cube[np.nonzero(coeffs_cube == 0)] = np.nan
        
        phase_maps_path = self.get_phase_maps_path(suffix=suffix)
        with orb.utils.io.open_hdf5(phase_maps_path, 'w') as hdffile:
            for ipar in self.params:
                value = orb.utils.io.cast2hdf5(self.params[ipar])
                hdffile.attrs[ipar] = value
            hdffile.create_dataset(
                '/calibration_coeff_map',
                data=self.get_calibration_coeff_map())
            hdffile.create_dataset(
                '/cm1_axis',
                data=base_axis.astype(float))

            for iz in range(coeffs_cube.shape[2]):
                hdffile.create_dataset(
                    '/phase_map_{}'.format(iz),
                    data=coeffs_cube[:,:,iz])
                hdffile.create_dataset(
                    '/phase_map_err_{}'.format(iz),
                    data=coeffs_err_cube[:,:,iz])
        logging.info('phase maps written: {}'.format(phase_maps_path))
            
        return phase_maps_path
    
    def iterative_polyfit(self, polydeg, suffix=None, high_order_phase=None):
        """Fit the cube iteratively, starting by fitting all orders, then
        fixing the last free order to its mean in the map obtained
        from the preceding fit.

        :param polydeg: Degree of the fitting polynomial. Must be >= 0.

        :param suffix: Phase maps hdf5 file suffix (added before the
          extension .hdf5)

        :param high_order_phase: Phase vector removed before
          fitting. Must be an orb.fft.Phase instance.

        :return: Path to the last phase maps file (can then be opened
          with PhaseMaps).
        """

        if not isinstance(polydeg, int): raise TypeError('polydeg must be an integer')
        if polydeg < 0: raise ValueError('polydeg must be >= 0')

        if suffix is None: suffix = ''
        else: suffix += '.'

        coeffs = list([None]) * (polydeg + 1)
        
        for ideg in range(polydeg + 1)[::-1]:
            ipm_path = self.polyfit(
                polydeg, suffix=suffix + 'iter{}'.format(ideg),
                coeffs=coeffs, high_order_phase=high_order_phase)
            if ideg > 0:
                ipm = orb.fft.PhaseMaps(ipm_path)
                last_map = ipm.get_map(ideg)
                last_dist = orb.utils.stats.sigmacut(last_map)
                logging.info('Computed coefficient of order {}: {:.2e} ({:.2e})'.format(
                    ideg, np.nanmean(last_dist), np.nanstd(last_dist)))
                coeffs[ideg] = np.nanmean(last_dist)
        logging.info('final computed phase maps path: {}'.format(ipm_path))

        return ipm_path
    

