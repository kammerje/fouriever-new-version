from __future__ import division

import os
import pdb
import sys

import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import numpy as np

import fouriever

import matplotlib
matplotlib.rcParams.update({'font.size': 14})


# =============================================================================
# MAIN
# =============================================================================

# odir = '../testdata/plots/'
# obj = 'AXCir'
odir = '/Users/jkammere/Documents_new/ESO/GRAVITY/YSES_2/plots/epoch2_standard/'
obj = 'YSES2'

# scifiles = ['../testdata/AXCir.oifits']
# calfiles = None
# scifiles = ['/Users/jkammere/Documents_new/ESO/GRAVITY/YSES_2/data_gravity/GRAVI.2022-05-21T23:50:00.064_dualscivis.oifits',
#             '/Users/jkammere/Documents_new/ESO/GRAVITY/YSES_2/data_gravity/GRAVI.2022-05-21T23:53:51.074_dualscivis.oifits',
#             '/Users/jkammere/Documents_new/ESO/GRAVITY/YSES_2/data_gravity/GRAVI.2022-05-21T23:55:42.079_dualscivis.oifits']
# calfiles = ['/Users/jkammere/Documents_new/ESO/GRAVITY/YSES_2/data_gravity/GRAVI.2022-05-22T02:08:33.413_dualscivis.oifits',
#             '/Users/jkammere/Documents_new/ESO/GRAVITY/YSES_2/data_gravity/GRAVI.2022-05-22T02:27:03.459_dualscivis.oifits']
scifiles = ['/Users/jkammere/Documents_new/ESO/GRAVITY/YSES_2/data_gravity/GRAVI.2023-05-10T04:02:06.734_dualscivis.oifits',
            '/Users/jkammere/Documents_new/ESO/GRAVITY/YSES_2/data_gravity/GRAVI.2023-05-10T04:05:18.742_dualscivis.oifits']
calfiles = ['/Users/jkammere/Documents_new/ESO/GRAVITY/YSES_2/data_gravity/GRAVI.2023-05-10T04:24:42.791_dualscivis.oifits',
            '/Users/jkammere/Documents_new/ESO/GRAVITY/YSES_2/data_gravity/GRAVI.2023-05-10T04:40:09.830_dualscivis.oifits']

fe = fouriever.Fouriever(odir=odir, obj=obj)
fe.read(scifiles=scifiles, calfiles=calfiles)
fe.add_cov()
fe.calibrate_classical(weighting='standard')

# =============================================================================
# cp
# =============================================================================

fe.set_obs(obs=['cp'])
fe.chi2map(rlim=(2, 60),
           step=1,
           model='bin',
           fixpos2grid=True,
           cov=False,
           ofile=obj + '_cp')

# =============================================================================
# cp fine
# =============================================================================

fe.set_obs(obs=['cp'])
fit = fe.chi2map(rlim=(2, 60),
                 step=1,
                 model='bin',
                 fixpos2grid=False,
                 cov=False,
                 ofile=obj + '_cp_fine')
fe.detlim(rlim=(2, 60),
          step=2,
          model='bin',
          fit_sub=fit,
          cov=False,
          sigma=5,
          ofile=obj + '_cp')

# =============================================================================
# cp cov
# =============================================================================

fe.set_obs(obs=['cp'])
fe.chi2map(rlim=(2, 60),
           step=1,
           model='bin',
           fixpos2grid=True,
           cov=True,
           ofile=obj + '_cp_cov')

# =============================================================================
# cp cov fine
# =============================================================================

fe.set_obs(obs=['cp'])
fe.chi2map(rlim=(2, 60),
           step=1,
           model='bin',
           fixpos2grid=False,
           cov=True,
           ofile=obj + '_cp_cov_fine')

# =============================================================================
# kp
# =============================================================================

fe.set_obs(obs=['kp'])
fe.chi2map(rlim=(2, 60),
           step=1,
           model='bin',
           fixpos2grid=True,
           cov=False,
           ofile=obj + '_kp')

# =============================================================================
# kp fine
# =============================================================================

fe.set_obs(obs=['kp'])
fit = fe.chi2map(rlim=(2, 60),
                 step=1,
                 model='bin',
                 fixpos2grid=False,
                 cov=False,
                 ofile=obj + '_kp_fine')
fe.detlim(rlim=(2, 60),
          step=2,
          model='bin',
          fit_sub=fit,
          cov=False,
          sigma=5,
          ofile=obj + '_kp')

# =============================================================================
# kp cov
# =============================================================================

fe.set_obs(obs=['kp'])
fe.chi2map(rlim=(2, 60),
           step=1,
           model='bin',
           fixpos2grid=True,
           cov=True,
           ofile=obj + '_kp_cov')

# =============================================================================
# vis2 cp
# =============================================================================

fe.set_obs(obs=['vis2', 'cp'])
fe.chi2map(rlim=(2, 60),
           step=1,
           model='ud_bin',
           fixpos2grid=True,
           cov=False,
           ofile=obj + '_vis2_cp')

# =============================================================================
# vis2 cp fine
# =============================================================================

fe.set_obs(obs=['vis2', 'cp'])
fit = fe.chi2map(rlim=(2, 60),
                 step=1,
                 model='ud_bin',
                 fixpos2grid=False,
                 cov=False,
                 ofile=obj + '_vis2_cp_fine')
fe.detlim(rlim=(2, 60),
          step=2,
          model='ud_bin',
          fit_sub=fit,
          cov=False,
          ofile=obj + '_vis2_cp')

# =============================================================================
# vis2 kp
# =============================================================================

fe.set_obs(obs=['vis2', 'kp'])
fe.chi2map(rlim=(2, 60),
           step=1,
           model='ud_bin',
           fixpos2grid=True,
           cov=False,
           ofile=obj + '_vis2_kp')

# =============================================================================
# vis2 kp fine
# =============================================================================

fe.set_obs(obs=['vis2', 'kp'])
fit = fe.chi2map(rlim=(2, 60),
                 step=1,
                 model='ud_bin',
                 fixpos2grid=False,
                 cov=False,
                 ofile=obj + '_vis2_kp_fine')
fe.detlim(rlim=(2, 60),
          step=2,
          model='ud_bin',
          fit_sub=fit,
          cov=False,
          ofile=obj + '_vis2_kp')

pdb.set_trace()
