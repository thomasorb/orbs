#!/usr/bin/python2.7
# *-* coding: utf-8 *-*
# Author: Thomas Martin <thomas.martin.1@ulaval.ca>

## Copyright (c) 2010-2014 Thomas Martin <thomas.martin.1@ulaval.ca>
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

####################################################
############ ORBS reduction script #################
####################################################
# To run this script simply use the following command : 
#
# $ python reduction_script.py 
#
# Remember : for any help please refer to the file "Orbs/orbs.py"
# which describes fully the methods used below
####################################################

import sys, time

######################################
# 1 - Define the path where you have unzipped the Orbs module
######################################
sys.path.append("/path/to/Orbs")

from orbs.orbs import Orbs

start_time = time.time()
print  "Script started at : " + time.ctime(start_time)

######################################
# 2 - You have to put the option file path (you may have created this
# file using orbs-optcreator in Orbs/scripts/)
######################################
project = Orbs("option_file.opt", overwrite=False)

######################################
#### BASIC REDUCTION PIPELINE ########
######################################
# 3 - Uncomment a line to make it works ;) Don't forget to define
# bad_frames_vector variable (see below)
######################################

project.compute_alignment_vector(1, star_list_path=None,
                                 stars_fwhm_arc=2.0)
project.compute_alignment_vector(2, star_list_path=None,
                                 stars_fwhm_arc=2.0)
project.compute_cosmic_ray_map(1)
project.compute_cosmic_ray_map(2)

# Define here a vector containing bad frames index. If the vector is
# not defined you can use check_bad_frames() function to do it
# automatically. But be careful this function is not as good as your
# own eyes)
bad_frames = []
project.compute_interferogram(1, bad_frames_vector=bad_frames,
                              optimize_dark_coeff=False)
project.compute_interferogram(2, bad_frames_vector=bad_frames,
                              optimize_dark_coeff=True)
project.transform_cube_B(star_list_path_1=None,
                         stars_fwhm_1_arc=2.0)
project.merge_interferograms(star_list_path_1=None,
                             stars_fwhm_1_arc=2.0)
project.add_missing_frames(0)
project.compute_calibration_laser_map(1)
project.compute_spectrum(0, window_type='2.0', phase_cube=True)
project.compute_phase_maps(0)
project.compute_spectrum(0)
project.calibrate_spectrum(0)
project.get_calibrated_spectrum_cube(0)
            
######################################

end_time = time.time()

print "Script finished at : " + time.ctime(end_time) 
print "Total time : " + str((end_time - start_time) / 3600.) + " hours"

# END ################################
