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


import orbs
import pylab as pl
from fpdf import FPDF
import shutil
import xml.etree.ElementTree
import core
import os
import orb.utils.io
import logging
import numpy as np
import orb.fft
import warnings

class Graph(object):


    def __init__(self, params, indexer):

        def imshow(data):
            pl.figure()
            pl.imshow(data, origin='bottom-left',
                      vmin=np.nanpercentile(data, 10),
                      vmax=np.nanpercentile(data, 90))
            pl.colorbar()
            try:
                xmin, xmax, ymin, ymax = self.getp(('xmin', 'xmax', 'ymin', 'ymax'), cast=int)
                pl.xlim((xmin, xmax))
                pl.ylim((ymin, ymax))
            except Exception, e:
                logging.debug('error in getting xlim, ylim parameters: {}'.format(e))

            

        self.params = params
        logging.info('generating graph for {}'.format(self.getp('name')))
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
            data = orb.utils.io.read_fits(path)
            imshow(data)            

        elif self.getp('type') == 'phase':
            pm = orb.fft.PhaseMaps(path)
            if self.getp('model') == 'True':
                pm.modelize()
            imshow(pm.get_map(0))

        else:
            raise TypeError('type {}  not understood'.format(self.getp('type')))
        pl.xlabel(self.getp('xlabel'))
        pl.ylabel(self.getp('ylabel'))
        pl.title(self.getp('title'))
        pl.grid()


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
            if ikey in self.params.keys():
                allp.append(cast(self.params.get(ikey)))
            else:
                raise StandardError('no such parameter {}'.format(ikey))
        if len(allp) == 1:
            allp = allp[0]
        return allp
        
class Reporter(object):

    GRAPHWIDTH = 200
    
    def __init__(self, job_file_path, instrument):


        self.orbs = orbs.Orbs(
            job_file_path, 'object', instrument=instrument,
            fast_init=True, silent=True)

        graphs = xml.etree.ElementTree.parse(
            os.path.join(core.ORBS_DATA_PATH, 'report.xml')).findall('graph')

        self.pdf = FPDF()
        self.pdf.add_page()
        self.pdf.set_xy(0, 0)
        self.pdf.set_font('arial', '', 40)
        self.pdf.cell(0, 40, 'ORBS Report for {} {}'.format(self.orbs.options['object_name'], self.orbs.options['filter_name']), 1, 0, 'L')
        self.pdf.ln(20)

        for igraphxml in graphs:
            if igraphxml.get('type') == 'text':
                idata = orb.utils.io.read_fits(self.orbs.indexer.get_path(igraphxml.get('name')))
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
                    igraph = Graph(igraphxml, self.orbs.indexer)
                    ipath = igraph.savefig(self.get_temp_folder_path())
                    self.pdf.image(ipath, None, None, self.GRAPHWIDTH)
                except Exception, e:
                    warnings.warn('graph {} not generated: {}'.format(igraphxml.get('name'), e))
                    self.pdf.set_font('arial', '', 15)
                    self.pdf.cell(0, 12, igraphxml.get('name') + ' NOT GENERATED !!!', 1, 0, 'L')
                    self.pdf.ln(20)
                    
                
                

        self.pdf.output('test.pdf', 'F')

        #     infiles = list()
        #     infiles_xml = step.findall('infile')
        #     for infile_xml in infiles_xml:
        #         infiles.append(infile_xml.attrib['name'])
        #     outfiles = list()
        #     outfiles_xml = step.findall('outfile')
        #     for outfile_xml in outfiles_xml:
        #         outfiles.append(outfile_xml.attrib['name'])
                
        #     self.steps[step.attrib['name']] = Step(infiles,
        #                                            None,
        #                                            outfiles)

    def get_temp_folder_path(self):
        return './.report.temp/'

    def get_data_path(self, filename):
        return self.get_temp_folder_path() + filename
    
    def generate_pdf(self):

        pdf = FPDF()
        
        pdf.add_page()
        pdf.set_xy(0, 0)
        pdf.set_font('arial', 'B', 12)
        pdf.cell(60)
        pdf.cell(75, 10, "A Tabular and Graphical Report of Professor Criss's Ratings by Users Charles and Mike", 0, 2, 'C')
        pdf.cell(90, 10, " ", 0, 2, 'C')
        pdf.cell(-40)
        pdf.cell(50, 10, 'Question', 1, 0, 'C')
        pdf.cell(40, 10, 'Charles', 1, 0, 'C')
        pdf.cell(40, 10, 'Mike', 1, 2, 'C')
        pdf.cell(-90)
        pdf.set_font('arial', '', 12)
        for i in range(0, len(df)):
            pdf.cell(50, 10, '%s' % (df['Question'].ix[i]), 1, 0, 'C')
            pdf.cell(40, 10, '%s' % (str(df.Mike.ix[i])), 1, 0, 'C')
            pdf.cell(40, 10, '%s' % (str(df.Charles.ix[i])), 1, 2, 'C')
            pdf.cell(-90)
        pdf.cell(90, 10, " ", 0, 2, 'C')
        pdf.cell(-30)
        pdf.image('barchart.png', x = None, y = None, w = 0, h = 0, type = '', link = '')
        pdf.output('test.pdf', 'F')

    def __del__(self):
        shutil.rmtree(self.get_temp_folder_path(), ignore_errors=True)

