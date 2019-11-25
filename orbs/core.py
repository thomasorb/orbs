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
import xml.etree.ElementTree
import warnings
import fnmatch

import orb.utils.io
import orb.utils.misc
import numpy as np
import orb.utils.astrometry
import orb.core

ORBS_DATA_PATH = os.path.join(os.path.split(__file__)[0], "data")

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

            fpath = self.rootpath + os.sep + self.file_folder + os.sep + '{}.{}.cam{}.list'.format(
                self.pathprefix, ftype, chip_index)
            with orb.utils.io.open_file(fpath, 'w') as flist:
                flist.write('# {} {}\n'.format('sitelle', chip_index))
                for i in range(len(l)):
                    flist.write('{}\n'.format(l[i]))
            return fpath

        
        self.path = os.path.abspath(path)
        self.rootpath = os.path.split(self.path)[0]
        self.pathprefix = os.path.split(self.path)[1]
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
            raise Exception('Keywords OBS or COMPARISON must be at least in the job file.')

        try:
            self.header = orb.utils.io.read_fits(
                self.raw_params[header_key][0],
                return_hdu_only=True).header
            
        except IOError:
            warnings.warn('File {} could not be opened!'.format(self.raw_params[header_key][0]))
            self.header = None
            return
        
        # check header
        if self.header['CCDBIN1'] != self.header['CCDBIN2']:
            self.print_error(
                'CCD Binning appears to be different for both axes')

        # parse header
        for ikey in self.header_keys:
            hdr_key = self.header_keys[ikey][0]
            hdr_cast = self.header_keys[ikey][1]
            if hdr_key not in self.header:
                raise Exception('malformed header. {} keyword should be present'.format(hdr_key))
            self.params[ikey] = hdr_cast(self.header[hdr_key])


        # convert name
        self.params['object_name'] = ''.join(
            self.params['object_name'].strip().split())

        self.file_folder = self.params['object_name'] + '_' + self.params['filter_name']

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
                self.raw_params['DARK'][0], return_hdu_only=True).header
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
            raise Exception('CALIBMAP keyword must be set')

        # get standard spectrum params
        if 'STDPATH' in self.raw_params:
            self.params['standard_path'] = self.raw_params.pop('STDPATH')
            if not os.path.exists(self.params['standard_path']):
                raise Exception('Standard star file does not exist ({})'.format(
                    self.params['standard_path']))

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
            warnings.warn('Some parameters in the job file are not recognized: {}'.format(list(self.raw_params.keys())))
        
    def get_params(self):
        self.check_validity()
        return orb.core.ROParams(self.params)

    def get_config(self):
        self.check_validity()
        return orb.core.ROParams(self.config)
        
    def as_str(self):
        """Return job file as a string"""
        _str = ''
        with self.open() as f:
            for line in f:
                _str += line
        return _str
        
    def open(self):
        return orb.utils.io.open_file(self.path, 'r')

    def is_valid(self):
        if self.header is not None: return True
        return False

    def check_validity(self):
        if not self.is_valid(): raise Exception('JobFile invalid. One or more file could not be opened properly.')


##################################################
#### CLASS RoadMap ###############################
##################################################
    
class RoadMap(orb.core.Tools):
    """Manage a reduction road map given a target and the camera to
    use (camera 1, 2 or both cameras).

    All steps are defined in a particular xml files in the data folder
    of ORBS (:file:`orbs/data/roadmap.steps.xml`).

    Each roadmap is defined by an xml file which can also be found in
    orbs/data.

    .. note:: The name of a roadmap file is defined as follows::

         roadmap.[instrument].[target].[camera].xml

       - *instrument* can be *spiomm* or *sitelle*
       - *target* can be one of the special targets listed above or object
         for the default target
       - *camera* can be *full* for a process using both cameras; *single1*
         or *single2* for a process using only the camera 1 or 2.


    .. note:: RoadMap file syntax:
    
        .. code-block:: xml

           <?xml version="1.0"?>
           <steps>
             <step name='compute_alignment_vector' cam='1'>
               <arg value='1' type='int'></arg>
             </step>

             <step name='compute_alignment_vector' cam='2'>
               <arg value='2' type='int'></arg>
             </step>

             <step name='compute_spectrum' cam='0'>
               <arg value='0' type='int'></arg>
               <kwarg name='phase_correction'></kwarg>
               <kwarg name='apodization_function'></kwarg>
             </step>
           </steps>

        * <step> Each step is defined by its **name** (which can be
          found in :file:`orbs/data/roadmap.steps.xml`) and the camera
          used (1, 2 or 0 for merged data).

        * <arg> Every needed arguments can be passed by giving the
          value and its type (see
          :py:data:`orbs.orbs.RoadMap.types_dict`).

        * <kwarg> optional arguments. They must be added to the step
          definition if their value has to be passed from the calling
          method (:py:meth:`orbs.orbs.Orbs.start_reduction`). Only the
          optional arguments of
          :py:meth:`orbs.orbs.Orbs.start_reduction` can thus be passed
          as optional arguments of the step function.
        

        
    """ 
    road = None # the reduction road to follow
    steps = None # all the possible reduction steps
    indexer = None # an orb.Indexer instance

    instrument = None # instrument name
    target = None # target type
    cams = None # camera used 

    ROADMAP_STEPS_FILE_NAME = 'roadmap.steps.xml'
    """Roadmap steps file name"""

    def _str2bool(s):
        """Convert a string to a boolean value.

        String must be 'True','1' or 'False','0'.

        :param s: string to convert.
        """
        if s.lower() in ("true", "1"): return True
        elif s.lower() in ("false", "0"): return False
        else: raise Exception(
            "Boolean value must be 'True','1' or 'False','0'")


    types_dict = { # map type to string definition in xml files
        'int':int,
        'str':str,
        'float':float,
        'bool':_str2bool}
    """Dictionary of the defined arguments types"""
    

    def __init__(self, instrument, target, cams, indexer, **kwargs):
        """Init class.

        Load steps definitions and roadmap.

        :param instrument: Instrument. Can be 'sitelle' or 'spiomm'
        :param target: Target of the data to reduce
        :param cams: Camera to use (cam be 'single1', 'single2' or 'full')
        :param indexer: An orb.Indexer instance.
        :param kwargs: Kwargs are :meth:`core.Tools` properties.
        """
        orb.core.Tools.__init__(self, **kwargs)
        self.indexer = indexer
        self.instrument = instrument
        self.target = target

        self.cams = cams

        # load roadmap steps
        roadmap_steps_path = os.path.join(
            ORBS_DATA_PATH, self.ROADMAP_STEPS_FILE_NAME)
        
        steps =  xml.etree.ElementTree.parse(roadmap_steps_path).getroot()
        
        self.steps = dict()
        for step in steps:
            infiles = list()
            infiles_xml = step.findall('infile')
            for infile_xml in infiles_xml:
                infiles.append(infile_xml.attrib['name'])
            outfiles = list()
            outfiles_xml = step.findall('outfile')
            for outfile_xml in outfiles_xml:
                outfiles.append(outfile_xml.attrib['name'])
                
            self.steps[step.attrib['name']] = Step(infiles,
                                                   None,
                                                   outfiles)

        # load roadmap
        roadmap_path = os.path.join(
            ORBS_DATA_PATH, 'roadmap.{}.{}.{}.xml'.format(
                instrument, target, cams))

        if not os.path.exists(roadmap_path):
            raise Exception('Roadmap {} does not exist'.format(
                roadmap_path))
            
        steps =  xml.etree.ElementTree.parse(roadmap_path).getroot()
        
        self.road = list()
        for step in steps:
            args_xml = step.findall('arg')
            args = list()
            for arg_xml in args_xml:
                args.append(self.types_dict[arg_xml.attrib['type']](arg_xml.attrib['value']))

            kwargs_xml = step.findall('kwarg')
            kwargs = dict()
            for kwarg_xml in kwargs_xml:
                if 'value' in kwarg_xml.attrib:
                    kwargs[kwarg_xml.attrib['name']] = self.types_dict[
                        kwarg_xml.attrib['type']](kwarg_xml.attrib['value'])
                else:
                    kwargs[kwarg_xml.attrib['name']] = 'undef'
                    
            if step.attrib['name'] in self.steps:
                self.road.append({'name':step.attrib['name'],
                                  'cam':int(step.attrib['cam']),
                                  'args':args, 'kwargs':kwargs,
                                  'status':False})
            else:
                raise Exception('Step {} found in {} not recorded in {}'.format(
                    step.attrib['name'], os.path.split(roadmap_path)[1],
                    os.path.split(roadmap_steps_path)[1]))

        # check road (ony possible if an Indexer instance has been given)
        self.check_road()

    def attach(self, step_name, func):
        """Attach a reduction function to a step.

        :param step_name: Name of the step
        :param func: function to attach
        """
        if step_name in self.steps:
            self.steps[step_name].func = func
        else:
            raise Exception('No step called {}'.format(step_name))

    def check_road(self):
        """Check the status of each step of the road."""
        if self.indexer is None: return
        
        for istep in range(len(self.road)):
            step = self.road[istep]
            for outf in self.steps[step['name']].get_outfiles(step['cam']):
                if outf in self.indexer.index:
                    if os.path.exists(self.indexer[outf]):
                        self.road[istep]['status'] = True

    def get_road_len(self):
        """Return the number of steps of the road"""    
        return len(self.road)

    def get_step_func(self, index):
        """Return the function and the arguments for a particular step.

        :param index: Index of of the step.

        :return: (func, args, kwargs)
        """
        if index < self.get_road_len():
            return (self.steps[self.road[index]['name']].func,
                    self.road[index]['args'],
                    self.road[index]['kwargs'])
        else:
            raise Exception(
                'Bad index number. Must be < {}'.format(self.get_road_len()))

    def print_status(self):
        """Print roadmap status"""
        self.check_road()
        
        print('Status of roadmap for {} {} {}'.format(self.instrument,
                                                      self.target,
                                                      self.cams))
        index = 0
        for step in self.road:
            if step['status'] :
                status = 'done'
                color = orb.core.TextColor.OKGREEN
            else:
                status = 'not done'
                color = orb.core.TextColor.KORED
            
            print(color + '  {} - {} {}: {}'.format(index, step['name'], step['cam'], status) + orb.core.TextColor.END)
            index += 1

    def get_steps_str(self, indent=0):
        """Return a string describing the different steps and their index.
        
        :param indent: (Optional) Indentation of each line (default 0)
        """
        str = ''
        istep = 0
        for step in self.road:
            step_name = step['name'].replace('_', ' ').capitalize()
            
            if step['cam'] == 0:
                str += ' '*indent + '{}. {}\n'.format(istep, step_name)
            else:
                str += ' '*indent + '{}. {} ({})\n'.format(istep, step_name, step['cam'])
            istep += 1
        return str

    def get_resume_step(self):
        index = 0
        for step in self.road:
            if not step['status']: return index
            index += 1
        return index

    
##################################################
#### CLASS Step ##################################
##################################################
class Step(object):
    """Reduction step definition.

    This class is used by :class:`orbs.orbs.RoadMap`.
    """
    def __init__(self, infiles, func, outfiles):
        """Init class

        :param infiles: a list of strings defining the input files.
        
        :param func: a function object attached to the reduction step.
        
        :param outfiles: a list of strings defining the output files.
        """
        self.infiles = infiles
        self.func = func
        self.outfiles = outfiles

    def get_outfiles(self, cam):
        """Return the complete output name of the file as it is
        recorded in the indexer (see :py:class:`orb.core.Indexer`).

        :param cam: camera used (can be 0,1 or 2)
        """
        outfiles = list()
        if cam != 0:
            for outf in self.outfiles:
                outfiles.append('cam{}.{}'.format(cam, outf))
        else:
            for outf in self.outfiles:
                outfiles.append('merged.{}'.format(outf))
                
        return outfiles


        
##################################################
#### CLASS JobsWalker ############################
##################################################
class JobsWalker():

    """Construct a database of all the job files found in a given folder
    and its subfolders.
    """    
    def __init__(self, root_folders):
        """Init class.

        :param root_folders: A list of path to the folders where the
          job files are to be found.
        """
        if not isinstance(root_folders, list):
            raise TypeError('root_folders must be a list of folders where the job files (*.job) are to be found')
        self.root_folders = list()
        for irf in root_folders:
            if not os.path.isdir(irf):
                raise IOError('{} not found'.format(irf))
            self.root_folders.append(irf)
        self.update()
        

    def update(self):
        """update the database. """
        self.jobfiles = list()
        for irootf in self.root_folders:
            for root, dirs, files in os.walk(irootf):
                for file_ in files:
                    if file_.endswith(".job"):
                        ijobpath = os.path.join(root, file_)
                        idir = os.path.split(ijobpath)[0]
                        if os.path.isdir(idir):
                            for ifile in os.listdir(idir):
                                if fnmatch.fnmatch(ifile, os.path.split(file_)[1] + '*.log'):
                                    self.jobfiles.append(ijobpath)
                                else:
                                    warnings.warn('{} does not have any corresponding log file.'.format(ijobpath))
        self.data = dict()
        self.data['jobfile'] = list()
        
        for i in range(len(self.jobfiles)):
            ijobfile = self.jobfiles[i]
            try:
                iparams = JobFile(ijobfile, 'sitelle').get_params()
            except Exception as e:
                warnings.warn('job file {} could not be read: {}'.format(ijobfile, e))
                continue
                
            for param in iparams:
                # update keys
                if param not in self.data:
                    if i > 0:
                        self.data[param] = list([None]) * i
                    else:
                        self.data[param] = list()
            for ikey in self.data:
                if ikey == 'jobfile':
                    self.data[ikey].append(ijobfile)            
                elif ikey in iparams:
                    self.data[ikey].append(iparams[ikey])
                else:
                    self.data[ikey].append(None)

        # append indexer
        self.data['indexer'] = list()
        for i in range(len(self.data['object_name'])):
            basename = self.data['object_name'][i] + '_' + self.data['filter_name'][i]
            basedir = os.path.split(self.data['jobfile'][i])[0]
            ifiles = os.listdir(basedir)
            indexer_found = False
            for ifile in ifiles:
                if fnmatch.fnmatch(ifile, basename + '*Indexer*'):
                    self.data['indexer'].append(os.path.join(basedir, ifile))
                    indexer_found = True
            if not indexer_found:
                self.data['indexer'].append(None)
            
    def get_job_files(self):
        """Return a list of the job files found"""
        return list(self.jobfiles)
    
    def get_data(self):
        """Return the whole content of the job files as a dict, which can be
           directly passed to a pandas DataFrame.

           .. code::
             jw = JobWalker(['path1', 'path2'])
             data = pd.DataFrame(jw.get_data()))
        """
        return dict(self.data)


##############################################################
##### CLASS RECORDFILE #######################################
##############################################################
class RecordFile(object):
    """Manage a file where all the launched reductions are recorded.

    This class is used for 'status', 'clean' and 'resume' operations
    """

    file_path = None
    records = None
    last_command = None
    
    def __init__(self, job_file_path):
        """Init class

        :param job_file_path: Path to the job file.
        """
        
        self.file_path = job_file_path + '.rec'
        # parse file
        self.records = list()
        if os.path.exists(self.file_path):
            with orb.utils.io.open_file(self.file_path, 'r') as f:
                for line in f:
                    if 'last_command' in line:
                        self.last_command = line.split()[1:]
                    else:
                        rec = line.split()
                        if len(rec) != 3:
                            raise Exception(
                                'Bad formatted record file ({})'.format(
                                    self.file_path))
                    
                        self.records.append({
                            'instrument':rec[0],
                            'target':rec[1],
                            'cams':rec[2]})
        
    def update(self):
        """update file on disk"""
        with orb.utils.io.open_file(self.file_path, 'w') as f:
            for record in self.records:
                f.write('{} {} {}\n'.format(
                    record['instrument'],
                    record['target'],
                    record['cams']))
            f.write('last_command ' + ' '.join(self.last_command) + '\n')
        
    def add_record(self, mode, target, cams):
        """Add a new record and update file on disk"""
        record = {'instrument':mode,
                  'target':target,
                  'cams':cams}
        if record not in self.records:
            self.records.append(record)
        self.update()
    
