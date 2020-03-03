# ORBS

[ORBS](https://github.com/thomasorb/orbs) (*Outil de Réduction Binoculaire pour* [SITELLE](http://www.cfht.hawaii.edu/Instruments/Sitelle)) is a data
reduction software created to process data obtained with
SITELLE. Is it the reduction software used by the CFHT.



## Installation


### Install ORB
   
[ORBS](https://github.com/thomasorb/orbs) depends on
[ORB](https://github.com/thomasorb/orb) which must be installed
first.

The archive and the installation instructions for
[ORB](https://github.com/thomasorb/orb) can be found on github

https://github.com/thomasorb/orb


### Install ORBS


#### Install specific dependencies

If you have followed the installation steps for orb, you already have a conda environment named `orb3`.
```bash
conda install -n orb3 -c conda-forge clint html2text distro lxml python-magic
conda activate orb3
pip install cadcdata --no-deps
pip install cadcutils --no-deps
```

You will also need cfitsio. On Ubuntu you can install it with
``` bash
sudo apt install libcfitsio5 libcfitsio-bin
```
#### Install orbs module

ORBS can be downloaded and installed from github also
  
https://github.com/thomasorb/orbs

Once the archive has been downloaded (from github just click on the
green button `clone or download` and click on `Download ZIP`) you may
extract it in a temporary folder. Then cd into the extracted folder
and type:

```bash
python setup.py install
```

