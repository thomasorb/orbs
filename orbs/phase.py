#!/usr/bin/python
# *-* coding: utf-8 *-*
# Author: Thomas Martin <thomas.martin.1@ulaval.ca>
# File: phase.py

## Copyright (c) 2010-2020 Thomas Martin <thomas.martin.1@ulaval.ca>
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
import os

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
            
            phase_col = np.full_like(col, np.nan)
            abs_col = np.full_like(phase_col, np.nan)
            
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
                    iabs = np.copy(spectrum.get_amplitude())
                    # note: the [:iphase.size] has been added when the phase is
                    # not interpolated (calibrate=False) because the
                    # number of samples can be smaller after the
                    # symmetric() step
                    phase_col[ij,:iphase.size] = iphase
                    abs_col[ij,:iabs.size] = iabs
            return phase_col, abs_col

        calib_map = self.get_calibration_coeff_map()
        base_axis = np.copy(self.get_base_axis().data)
        phase_cube = np.full(self.shape, np.nan, dtype=float)
        abs_cube = np.full(self.shape, np.nan, dtype=float)
        
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
                    phase_cube[ii+ijob,:,:], abs_cube[ii+ijob,:,:] = job()
                    
        progress.end()
        self._close_pp_server(job_server)

        # write phase cube
        out_cube = orb.cube.RWHDFCube(
            path,
            shape=self.shape,
            instrument=self.instrument,
            config=self.config,
            params=self.params,
            reset=True)
        
        out_cube[:,:,:] = phase_cube
        
        del out_cube

        # write amplitude cube
        abs_path = os.path.splitext(path)[0] + '.abs.hdf5'
        
        out_cube = orb.cube.RWHDFCube(
            abs_path,
            shape=self.shape,
            instrument=self.instrument,
            config=self.config,
            params=self.params,
            reset=True)
        
        out_cube[:,:,:] = abs_cube


        return path, abs_path

 
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

    def get_high_order_phase_cube_path(self):
        return self._data_path_hdr + 'high_order_phase_cube.hdf5'

    def get_phase(self, x, y):
        """Return a phase vector at position x, y

        :param x: x position
        :param y: y position
        """
        return orb.fft.Phase(
            self[x, y, :], self.get_base_axis(), params=self.params)
        
    def polyfit(self, polydeg, abs_path, high_order_phase, coeffs=None, suffix=None):
        """Create phase maps from a polynomial fit of the binned phase cube

        :param polydeg: Degree of the fitting polynomial. Must be >= 0.

        :param abs_path: Path to the binned amplitude cube, used
          to weight phase data during the fit.

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
        def fit_phase_in_column(col, abscol, deg, incoeffs_col, params, base_axis,
                                high_order_phase_col):
            warnings.simplefilter('ignore', RuntimeWarning)
            outcoeffs_col = np.empty((col.shape[0], deg + 1), dtype=float)
            outcoeffs_col.fill(np.nan)
            outcoeffs_err_col = np.empty((col.shape[0], deg + 1), dtype=float)
            outcoeffs_err_col.fill(np.nan)
            for ij in range(col.shape[0]):
                
                icoeffs = tuple([icoeff[ij] for icoeff in incoeffs_col])
                _phase = orb.fft.Phase(col[ij,:], base_axis, params)
                if high_order_phase_col is not None:
                    _phase.data -= high_order_phase_col[ij,:]
                try:
                    outcoeffs_col[ij,:], outcoeffs_err_col[ij,:] = _phase.polyfit(
                        deg, amplitude=abscol[ij,:], coeffs=icoeffs, 
                        return_coeffs=True)
                except orb.utils.err.FitError:
                    logging.debug('fit error')
            return outcoeffs_col, outcoeffs_err_col                  

        if not isinstance(polydeg, int): raise TypeError('polydeg must be an integer')
        if polydeg < 0: raise ValueError('polydeg must be >= 0')

        abs_cube = orb.cube.Cube(abs_path)
        assert abs_cube.shape == self.shape, 'amplitude cube has shape {} and phase cube has shape {}'.format(abs_cube.shape, self.shape)

        if isinstance(high_order_phase, orb.fft.Phase):
            high_order_phase_proj = np.zeros(self.shape, dtype=float)
            high_order_phase_proj += high_order_phase.project(self.get_base_axis()).data

        elif isinstance(high_order_phase, orb.fft.HighOrderPhaseCube):
            high_order_phase_proj = high_order_phase.generate_phase_cube(
                None, self.dimx, self.dimy, axis=self.get_base_axis())
        else:
            raise TypeError('high_order_phase must be an orb.fft.Phase/HighOrderPhaseMaps instance')

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

            ijob = 0
                            
            jobs = [(ijob, job_server.submit(
                fit_phase_in_column, 
                args=(np.copy(self[ii+ijob,:,:]),
                      np.copy(abs_cube[ii+ijob,:,:]),
                      polydeg,
                      [icoeff[ii+ijob,:] for icoeff in coeffs],
                      self.params.convert(), np.copy(base_axis),
                      high_order_phase_proj[ii+ijob,:,:]),
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
        
        coeffs_cube[:,:,0] = orb.utils.image.unwrap_phase_map0(
            coeffs_cube[:,:,0])
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
    
    def iterative_polyfit(self, polydeg, abs_path, high_order_phase, suffix=None):
        """Fit the cube iteratively, starting by fitting all orders, then
        fixing the last free order to the model of the map obtained
        from the preceding fit.

        :param polydeg: Degree of the fitting polynomial. Must be >= 0.

        :param abs_path: Path to the binned amplitude cube, used
          to weight phase data during the fit.

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
                polydeg, abs_path, high_order_phase,
                suffix=suffix + 'iter{}'.format(ideg),
                coeffs=coeffs)
            if ideg > 0:
                ipm = orb.fft.PhaseMaps(ipm_path)
                ipm.modelize()
                last_map = ipm.get_map(ideg)
                logging.info('Computed map of order {}: {:.2e} ({:.2e})'.format(
                    ideg, np.nanmean(last_map), np.nanstd(last_map)))
                coeffs[ideg] = last_map
        logging.info('final computed phase maps path: {}'.format(ipm_path))

        return ipm_path
    
    def compute_high_order_phase_cube(self, polydeg=None, divnb=30):
        
        smin, smax = orb.utils.spectrum.cm12pix(
            self.get_base_axis().data,
            self.filterfile.get_filter_bandpass_cm1())
        smin = int(smin)
        smax = int(smax) + 1

        if polydeg is None:
            polydeg = int(self.filterfile.get_phase_fit_order())
            
        assert isinstance(polydeg, int), 'polydeg must be an int'
        logging.info('fitting high order phase with a {} degrees polynomial'.format(polydeg))
        boxsize = int(self.shape[0]//divnb)

        high_orders_map = np.empty((divnb, divnb, polydeg + 1), dtype=float)
        axis = self.get_base_axis().data[smin:smax].astype(float)
        progress = orb.core.ProgressBar(divnb)
        for ii in range(divnb):
            progress.update(ii)
            for ij in range(divnb):
                xmin, xmax = ii*boxsize, (ii+1)*boxsize
                ymin, ymax = ij*boxsize, (ij+1)*boxsize
                ibox = self[xmin:xmax, ymin:ymax, smin:smax]
                ibox -= np.nanmedian(ibox, axis=2).reshape((ibox.shape[0], ibox.shape[1], 1))
                
                iphase = np.nanmedian(ibox, axis=(0,1))
                icoeffs = np.polyfit(axis, iphase, polydeg)
                high_orders_map[ii,ij,:] = icoeffs
        progress.end()
        
        params = dict(self.params)
        params['polyfit_phase_axis'] = axis
        params['phase_axis'] = self.get_base_axis().data
        data = orb.core.Data(high_orders_map, params=params)
        data.writeto(self.get_high_order_phase_cube_path())
        logging.info('high order phase cube written as: {}'.format(self.get_high_order_phase_cube_path()))

        
