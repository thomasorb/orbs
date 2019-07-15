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
import fpdf
import shutil
import xml.etree.ElementTree
import core
import os


class Graph(object):


    def __init__(self, params):

        params.get()

class Reporter(object):

    def __init__(self, job_file_path, instrument):


        self.orbs = orbs.Orbs(
            job_file_path, 'object', instrument=instrument,
            fast_init=True, silent=True)

        
        #print self.orbs.indexer
        graphs = xml.etree.ElementTree.parse(
            os.path.join(core.ORBS_DATA_PATH, 'report.xml')).findall('graph')
        
        
        for igraph in graphs:
            pass
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

    def check(self):
        pass
    
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

