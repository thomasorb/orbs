====================
 ORBS Reduction FAQ
====================

.. contents::

.. _how-can-access-orbs-commands:

How can I have access to ORBS commands ?
========================================

There are a lot of different ways to get ORBS commands like any other
OS commands on a UNIX-like system. Here are two possibilities:

If you have administrator rights
--------------------------------

* |linux| |mac| You can create a symbolic link in one of the folders
  :file:`/usr/local/bin` or :file:`/usr/bin` to the ORBS command (but
  you need to do this **for each command you want to add**), e.g.::

    sudo ln -s /opt/orbs/scripts/orbs /usr/local/bin/orbs


* |linux| You can also modify the file :file:`/etc/profile` by adding
  this line at the end of the file (**this way all the command
  contained in the scripts folder of ORBS will be added**)::

    export PATH=$PATH:/opt/orbs/scripts/


If you only are a simple user
-----------------------------

* |linux| |mac| You can modify the file in your home folder
  :file:`~/.profile` by adding the line at the end of the file (**this
  way all the command contained in the scripts folder of ORBS will be
  added**)::

    export PATH=$PATH:/opt/orbs/scripts/

.. note:: Either way you have to logout and login to see the changes.



.. |linux| image:: os_linux.*
           :scale: 10
           :alt: Linux


.. |mac| image:: os_apple.*
         :scale: 10
         :alt: Mac


What is the binning of my frames ?
==================================

If you are talking about SITELLE's frames, the binning is likely to
be 1. But for SpIOMM, each camera has its own binning (e.g. 3x3 for
camera 1 and 4x4 for camera 2).

SpIOMM detector size is:

* camera 1: 1340x1300
* camera 2: 2047x2047

SITELLE detector size is:

* camera 1: 2048x2064
* camera 2: 2048x2064

The binning can be deduced by dividing the size of the detector by the
actual size of the frames and truncating the result to its integer value.





