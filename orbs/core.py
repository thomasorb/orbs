#!/usr/bin/python
# *-* coding: utf-8 *-*
# Author: Thomas Martin <thomas.martin.1@ulaval.ca>
# File: core.py

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

import os
import orb.utils.io
import orb.utils.misc
import numpy as np
import orb.utils.astrometry
import orb.core

##################################################
#### CLASS JobFile ###############################
##################################################

class JobFile(object):
    """manage a job file and convert it to a dictionary of reduction parameters."""

    header_keys = {
        'object_name': ('OBJNAME', str),
        'filter_name': ('FILTER', str),
        'step_nb': ('SITSTEPS', int),
        'order': ('SITORDER', float),
        'exposure_time': ('EXPTIME', float),
        'obs_date': ('DATE-OBS', str),
        'obs_time': ('TIME-OBS', str),
        'target_ra': ('RA', str),
        'target_dec': ('DEC', str),
    }

    params_keys = {
        'target_x': ('TARGETX', float),
        'target_y': ('TARGETY', float),
        'try_catalogue': ('TRYCAT', bool),
        'wavenumber': ('WAVENUMBER', bool),
        'spectral_calibration': ('WAVE_CALIB', bool),
        'no_sky': ('NOSKY', bool),
        'phase_maps_path': ('PHASEMAPS', str),
        'star_list_path_1': ('STARLIST1', str),
        'star_list_path_2': ('STARLIST2', str),
        'apodization_function': ('APOD', str),
        'wavefront_map_path': ('WFMAP', str),
        'source_list_path': ('SOURCE_LIST_PATH', str),
        'object_mask_path_1': ('OBJMASK1', str),
        'object_mask_path_2': ('OBJMASK2', str),
        'std_mask_path_1': ('STDMASK1', str),
        'std_mask_path_2': ('STDMASK2', str)
    }

    file_keys = {'OBS': 'image_list_path',
                 'BIAS': 'bias_path',
                 'FLAT': 'flat_path',
                 'DARK': 'dark_path',
                 'COMPARISON': 'calib_path',
                 'STDIM': 'standard_image_path'}


    def __init__(self, path, instrument, is_laser=False):
        """Init class.

        :param path: Path to the job file.

        :param instrument: Instrument (may be 'sitelle' or 'spiomm')

        :param is_laser: True if target is a laser cube (default False).
        """
        def generate_file_list(flist, ftype, chip_index):
            """Generate a file list from the option file and write it in a file.

            :param flist: list of file paths

            :param ftype: Type of list created ('object', 'dark', 'flat',
              'calib')

            :param chip_index: SITELLE's chip index (1 or 2 for camera 1
              or camera 2) :
            """

            # list is sorted in the job file order so the job file
            # is assumed to give a sorted list of files
            l = list()
            for path in flist:
                index = orb.utils.misc.get_cfht_odometer(path)
                l.append((path, index))
            l = sorted(l, key=lambda ifile: ifile[1])
            l = [ifile[0] for ifile in l]

            fpath = '{}.{}.cam{}.list'.format(self.path, ftype, chip_index)
            with open(fpath, 'w') as flist:
                flist.write('# {} {}\n'.format('sitelle', chip_index))
                for i in range(len(l)):
                    flist.write('{}\n'.format(l[i]))
            return fpath

        
        self.path = path
        self.raw_params = dict()
        for key in self.file_keys:
            self.raw_params[key] = list()
        self.params = dict()

        # parse jobfile and convert it to a dictionary of raw parameters
        with self.open() as f:
            for line in f:
                line = line.strip().split('#')[0] # comment are removed
                if len(line) <= 2: continue
                line = line.split()
                key = line[0]
                value = line[1:]
                if len(value) == 1: value = value[0]

                if key in self.file_keys:
                    self.raw_params[key].append(value)
                else:
                    ikey = str(key)
                    index = 0
                    while ikey in self.raw_params:
                        index += 1
                        ikey = '{}{}'.format(key, index)
                    
                self.raw_params[ikey] = value

        # parse raw_params
        for ikey in self.params_keys:
            par_key = self.params_keys[ikey][0]
            par_cast = self.params_keys[ikey][1]
            if par_key in self.raw_params:
                self.params[ikey] = par_cast(self.raw_params.pop(par_key))
            
        
        # get header of the first observation file
        if len(self.raw_params['OBS']):
            header_key = 'OBS'
        elif len(self.raw_params['COMPARISON']):
            header_key = 'COMPARISON'
        else:
            raise StandardError('Keywords OBS or COMPARISON must be at least in the job file.')
        
        self.header = orb.utils.io.read_fits(
            self.raw_params[header_key][0],
            return_hdu_only=True)[0].header

        # check header
        if self.header['CCDBIN1'] != self.header['CCDBIN2']:
            self.print_error(
                'CCD Binning appears to be different for both axes')

        # parse header
        for ikey in self.header_keys:
            hdr_key = self.header_keys[ikey][0]
            hdr_cast = self.header_keys[ikey][1]
            if hdr_key not in self.header:
                raise StandardError('malformed header. {} keyword should be present'.format(hdr_key))
            self.params[ikey] = hdr_cast(self.header[hdr_key])


        # convert name
        self.params['object_name'] = ''.join(
            self.params['object_name'].strip().split())

        # compute step size in nm
        if not is_laser:
            step_fringe = float(self.header['SITSTPSZ'])
            fringe_sz = float(self.header['SITFRGNM'])
            self.params['step'] = step_fringe * fringe_sz
        else:
            self.params.pop['order']

        # get dark exposition time
        if len(self.raw_params['DARK']) > 0:
            dark_hdr = to.read_fits(
                self.raw_params['DARK'][0], return_hdu_only=True)[0].header
            self.params['dark_time'] = float(dark_hdr['EXPTIME'])

        # define target position in the frame
        sec_cam1 = self.header['DSEC1']
        sec_cam1 = sec_cam1[1:-1].split(',')
        sec_cam1x = np.array(sec_cam1[0].split(':'), dtype=int)
        sec_cam1y = np.array(sec_cam1[1].split(':'), dtype=int)

        if 'target_x' not in self.params:
            self.params['target_x'] = (
                float(sec_cam1x[1]-sec_cam1x[0]) / 2.)
            
        if 'target_y' not in self.params:
            self.params['target_y'] = (
                float(sec_cam1y[1]-sec_cam1y[0]) / 2.)

        # get calibration laser map path
        if 'CALIBMAP' in self.raw_params:
            self.params['calibration_laser_map_path'] = self.raw_params.pop('CALIBMAP')
       
        elif not is_laser:
            raise StandardError('CALIBMAP keyword must be set')

        # get standard spectrum params
        if 'STDPATH' in self.raw_params:
            self.params['standard_path'] = self.raw_params.pop('STDPATH')
            if not os.path.exists(self.params['standard_path']):
                raise StandardError('Standard star file does not exist ({})'.format(std_path))

        # convert ra and dec
        if 'target_ra' in self.params:
            self.params['target_ra'] = orb.utils.astrometry.ra2deg(
                self.params['target_ra'].split(':'))
        if 'target_dec' in self.params:
            self.params['target_dec'] = orb.utils.astrometry.dec2deg(
                self.params['target_dec'].split(':'))
    
        # parse raw params and get file lists
        for ikey in self.file_keys:
            if len(self.raw_params[ikey]) > 0:
                self.params['{}_1'.format(self.file_keys[ikey])] = generate_file_list(
                    self.raw_params[ikey], ikey, 1)
                self.params['{}_2'.format(self.file_keys[ikey])] = generate_file_list(
                    self.raw_params.pop(ikey), ikey, 2)
            else:
                self.raw_params.pop(ikey)

        # parse other parameters as config parameters
        self.config = dict()
        for ikey in orb.core.Tools(instrument=instrument).config:
            if ikey in self.raw_params:
                self.config[ikey] = self.raw_params.pop(ikey)
                
        if len(self.raw_params) > 0:
            raise StandardError('Some parameters in the job file are not recognized: {}'.format(self.raw_params.keys()))
        
    def get_params(self):
        return orb.core.ROParams(self.params)

    def get_config(self):
        return orb.core.ROParams(self.config)
        
    def as_str(self):
        """Return job file as a string"""
        _str = ''
        with self.open() as f:
            for line in f:
                _str += line
        return _str
        
    def open(self):
        return open(self.path, 'r')
