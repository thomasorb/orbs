#!/usr/bin/python
# *-* coding: utf-8 *-*
# Author: Thomas Martin <thomas.martin.1@ulaval.ca>
# File: cadc.py

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


import pandas as pd
import os
import subprocess
import shlex
import warnings

try:
    import cadcdata
except ImportError:
    print('cadcdata not installed, please try pip install cadcdata')
    raise

def read_cadc_results(path):
    with open(path, 'r') as f:
        data = dict()
        for line in f:
            if len(data) == 0:
                line = line.replace('"""', '')
                header = line.strip().split('\t')
                for ikey in header: data[ikey] = list()
                continue
            
            
            line = line.strip().split('\t')
            
            for i in range(len(header)):
                ival = line[i]
                try:
                    ival = float(ival)
                except: pass
                data[header[i]].append(ival)
                
    return pd.DataFrame(data)

def get_cadc_data(to_download):
    if len(to_download) > 0:
        error_files = list()
        for ifile in to_download:
            if os.path.exists(ifile + '.fits'):
                print('file {} already downloaded'.format(ifile + '.fits'))
                continue

            command = 'cadc-data get -v -z CFHT {}'.format(ifile)

            # downloading
            print('> downloading {} (command: {})'.format(ifile, command))
            try:
                p = subprocess.run(shlex.split(command), timeout=60)
            except subprocess.TimeoutExpired:
                print('timeout expired')
                continue
            
            if p.returncode != 0:
                warnings.warn('{} could not be downloaded'.format(ifile))
                error_files.append(ifile)
                continue

            # uncompressing
            command = 'funpack {}'.format(ifile + '.fits.fz')
            print('> uncompressing {} (command: {})'.format(ifile, command))
            p = subprocess.run(shlex.split(command))
            if p.returncode != 0:
                warnings.warn("{} could not be uncompressed (do you have funpack installed, if not try 'sudo apt install libcfitsio-bin')".format(ifile))
                error_files.append(ifile)
                continue
            
            # deleting compressed file
            print('> removing compressed file {}'.format(ifile + '.fits.fz'))
            os.remove(ifile + '.fits.fz')
            
            print('----------------')
            
        if len(error_files) > 0:
            print('some files could not be downloaded (timeout ?).')
            for ifile in error_files:
                print('   {}'.format(ifile))
            print('Please repeat the same command to try again.')
