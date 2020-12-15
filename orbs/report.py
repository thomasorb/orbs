#!/usr/bin/python
# *-* coding: utf-8 *-*
# Author: Thomas Martin <thomas.martin.1@ulaval.ca>
# File: report.py

## Copyright (c) 2010-2019 Thomas Martin <thomas.martin.1@ulaval.ca>
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


from . import orbs
import pylab as pl
from fpdf import FPDF
import shutil
import xml.etree.ElementTree
from . import core
import os
import orb.utils.io
import logging
import numpy as np
import orb.fft
import warnings
import pandas as pd

class Graph(object):


    def __init__(self, params, indexer, fast):

        def imshow(data, star_list=None):
            pl.figure()
            pl.imshow(data.T, origin='lower',
                      vmin=np.nanpercentile(data, 10),
                      vmax=np.nanpercentile(data, 90))
            pl.colorbar()
            try:
                xmin, xmax, ymin, ymax = self.getp(('xmin', 'xmax', 'ymin', 'ymax'), cast=int)
                pl.xlim((xmin, xmax))
                pl.ylim((ymin, ymax))
            except Exception as e:
                logging.debug('error in getting xlim, ylim parameters: {}'.format(e))
            if star_list is not None:
                try:
                    pl.scatter(star_list.x, star_list.y, marker='+', color='red')
                    for i in range(len(star_list)):
                        pl.text(star_list.x.values[i] - 100,
                                star_list.y.values[i] - 100,
                                i, c='red')

                except Exception as e:
                    logging.warning('star list could not be read')
                
        def starsgrid(im, star_list):
            xdiv = int(np.sqrt(len(star_list)))
            ydiv = len(star_list)//xdiv + 1
            boxsz = 10
            fig, axes = pl.subplots(xdiv, ydiv)
            axes = axes.flatten()
            for i in range(len(star_list)):
                ix = int(star_list.x.values[i])
                iy = int(star_list.y.values[i])
                ibox = im[ix-boxsz:ix+boxsz+1, iy-boxsz:iy+boxsz+1]
                vmin, vmax = np.nanpercentile(ibox, (10, 95))
                axes[i].imshow(ibox.T, vmin=vmin, vmax=vmax, origin='lower')
                axes[i].text(0,0,i, c='red')

            for i in range(len(axes)):
                axes[i].axis('off')

    
        self.params = params

        try: slow = bool(self.getp('slow'))
        except Exception: slow = False
        if slow and fast:
            logging.warning('{}/{} not generated because of fast keyword'.format(self.getp('name'), self.getp('type')))
            return
            
        logging.info('generating graph for {}/{}'.format(self.getp('name'), self.getp('type')))
        path = indexer.get_path(self.getp('name'))
        if self.getp('type') == 'vector':
            pl.figure(figsize=(8, 4))
            data = orb.utils.io.read_fits(path)
        
            try:
                errpath = indexer.get_path(self.getp('err'))
                err = orb.utils.io.read_fits(errpath).T
            except Exception:
                err = None

            if len(data.shape) == 1:
                data = data.reshape((data.size, 1))
                
            for i in range(data.T.shape[0]):
                ivector = data.T[i,:]
                if err is not None:
                    ierr = err[i,:]
                else:
                    ierr = None
                pl.errorbar(np.arange(ivector.size), ivector, yerr=ierr)
                
        elif self.getp('type') == 'image':
            try:
                slname = self.getp('slname')
            except Exception as e:
                slname = self.getp('name').split('.')[0] + '.star_list'
                
            star_list_path = indexer.get_path(slname)
            
            sl = None
            if star_list_path is not None:
                if os.path.exists(star_list_path):
                    sl = pd.read_hdf(star_list_path)
                else:
                    logging.warning('no star list file attached to {} ({} does not exist)'.format(path, star_list_path))
                
            if 'fits' in path:
                data = orb.utils.io.read_fits(path)
            elif 'hdf' in path:
                data = orb.image.Image(path).data
            else:
                raise Exception('image must be of type hdf or fits')
            imshow(data, star_list=sl)

        elif self.getp('type') == 'starsgrid':
            try:
                slname = self.getp('slname')
            except Exception as e:
                slname = self.getp('name').split('.')[0] + '.star_list'

            if not os.path.exists(slname):
                star_list_path = indexer.get_path(slname)
            else:
                star_list_path = slname
            if os.path.exists(star_list_path):
                sl = pd.read_hdf(star_list_path)
            else:
                raise Exception('no star list file attached to {} ({} does not exist)'.format(path, star_list_path))

            if 'fits' in path:
                data = orb.utils.io.read_fits(path)
            elif 'hdf' in path:
                data = orb.image.Image(path).data
            else:
                raise Exception('image must be of type hdf or fits')
            starsgrid(data, sl)

            
        elif self.getp('type') == 'phase':
            order = int(self.getp('order'))
            if order == 1:
                path = path.replace('iter0', 'iter1')
                
            pm = orb.fft.PhaseMaps(path)
            pm0 = pm.get_map(order)
            if self.getp('model') == 'True':
                pm.modelize()
                pm0 = pm.get_map(order)
            if self.getp('model') == 'res':
                old_pm0 = np.copy(pm0)
                pm.modelize()
                pm0 = old_pm0 - pm.get_map(order)
            imshow(pm0)

        elif self.getp('type') == 'spectrum':
            pl.figure(figsize=(8, 4))
            spectrum = orb.fft.Spectrum(path)
            spectrum.plot()
            
        elif self.getp('type') == 'interferogram':
            pl.figure(figsize=(8, 4))
            interf = orb.fft.Interferogram(path)
            interf.plot()
            
        elif self.getp('type') == 'spectrum_cube':
            pl.figure(figsize=(8, 4))
            cube = orb.cube.SpectralCube(path)
            ix, iy = self.getp(('x', 'y'), cast=int)
            r = self.getp('r', cast=float)
            spectrum = cube.get_spectrum(ix, iy, r)
            spectrum.plot()
            pl.legend()

        elif self.getp('type') == 'calib_image':
            im = orb.image.Image(path)
            im.imshow(perc=90)
            try:
                starcat = im.get_stars_from_catalog(max_stars=200)
                starcat.to_hdf('calib_grid.hdf', 'data', mode='w')
            except Exception as e:
                logging.warn('star position could not be added: ', e)
            else:
                pl.scatter(starcat.x, starcat.y, c='red', marker='+')
            
            try:
                xmin, xmax, ymin, ymax = self.getp(('xmin', 'xmax', 'ymin', 'ymax'), cast=int)
                pl.xlim((xmin, xmax))
                pl.ylim((ymin, ymax))
            except Exception as e:
                logging.debug('error getting xlim, ylim parameters: {}'.format(e))

            
        elif self.getp('type') == 'modulation_ratio':
            logging.info('Note that this may take a while. if you want to skip it, please use the --fast option')
            cube = orb.cube.SpectralCube(path)
            im = cube.compute_modulation_ratio()
            imshow(im)
        else:
            raise TypeError('type {}  not understood'.format(self.getp('type')))

        pl.gcf().suptitle(self.getp('title'))    
        pl.grid()
        
        try:
            pl.xlabel(self.getp('xlabel'))
            pl.ylabel(self.getp('ylabel'))
            
        except Exception: pass
            
            

        


    def savefig(self, folderpath):
        index = 0
        pathok = False
        while not pathok:
            path = os.path.join(folderpath, self.getp('name') + '.' + str(index) + '.png')
            if not os.path.exists(path):
                pathok = True
            index += 1
            
        logging.debug('saving graph to {}'.format(path))
        dirname = os.path.dirname(path)
        if dirname != '':
            if not os.path.exists(dirname): 
                os.makedirs(dirname)
            
        pl.savefig(path)
        return path

    def getp(self, keys, cast=str):
        if isinstance(keys, str):
            keys = (keys, )
        allp = list()
        for ikey in keys: 
            if ikey in list(self.params.keys()):
                allp.append(cast(self.params.get(ikey)))
            else:
                raise Exception('no such parameter {}'.format(ikey))
        if len(allp) == 1:
            allp = allp[0]
        return allp
        
class Reporter(object):

    GRAPHWIDTH = 200
    
    def __init__(self, job_file_path, instrument, fast=False):


        self.orbs = orbs.Orbs(
            job_file_path, 'object', instrument=instrument,
            fast_init=True, silent=True)
        self.fast = bool(fast)
        
        graphs = xml.etree.ElementTree.parse(
            os.path.join(core.ORBS_DATA_PATH, 'report.xml')).findall('graph')

        self.pdf = FPDF()
        self.pdf.add_page()
        self.pdf.set_xy(0, 0)
        self.pdf.set_font('arial', '', 30)
        self.pdf.cell(0, 40, 'ORBS Report'.format(self.orbs.options['object_name'], self.orbs.options['filter_name']), 0, 0, 'C')
        self.pdf.ln(20)
        self.pdf.cell(0, 40, '{} {}'.format(self.orbs.options['object_name'], self.orbs.options['filter_name']), 0, 0, 'C')
        self.pdf.ln(20)

        for igraphxml in graphs:
            if igraphxml.get('type') == 'text':
                try:
                    idata = orb.utils.io.read_fits(self.orbs.indexer.get_path(igraphxml.get('name')))
                except:
                    logging.warning('{} could not be retrieved'.format(igraphxml.get('name')))
                    continue
                
                if idata.size > 1:
                    idata = ' '.join(['{:.3f}'.format(ii) for ii in idata])
                else:
                    idata = '{:.3f}'.format(idata)

                self.pdf.set_font('arial', 'B', 12)
                self.pdf.cell(0, 12, igraphxml.get('title') + ': ', 0, 0, 'L')
                self.pdf.ln(4)

                self.pdf.set_font('arial', '', 12)
                self.pdf.cell(0, 12, idata, 0, 0, 'L')
                self.pdf.ln(20)
            elif igraphxml.get('type') == 'part':
                self.pdf.set_font('arial', '', 25)
                self.pdf.add_page()
                self.pdf.cell(0, 12, igraphxml.get('title'), 1, 0, 'L')
                self.pdf.ln(20)
            else:
                try:
                    igraph = Graph(igraphxml, self.orbs.indexer, self.fast)
                    ipath = igraph.savefig(self.get_temp_folder_path())
                    self.pdf.image(ipath, None, None, self.GRAPHWIDTH)
                except Exception as e:
                    logging.warn('graph {} not generated: {}'.format(igraphxml.get('name'), e))
                    self.pdf.set_font('arial', '', 15)
                    self.pdf.cell(0, 12, igraphxml.get('name') + ' NOT GENERATED !!!', 1, 0, 'L')
                    self.pdf.ln(20)
                    
                
        self.pdf.output('{}_{}.report.pdf'.format(self.orbs.options['object_name'], self.orbs.options['filter_name']), 'F')

    def get_temp_folder_path(self):
        return './.report.temp/'

    def get_data_path(self, filename):
        return self.get_temp_folder_path() + filename
    
    def __del__(self):
        shutil.rmtree(self.get_temp_folder_path(), ignore_errors=True)

