from __future__ import division

import os
import pdb
import sys

import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import numpy as np

from fouriever import core

import matplotlib
matplotlib.rcParams.update({'font.size': 14})


# =============================================================================
# MAIN
# =============================================================================

# Initialize fouriever.
obj = 'axcir'
fe = core.Fouriever(odir='test/', obj=obj)

# Read data.
scifiles = ['test/axcir.oifits']
fe.read(scifiles=scifiles, calfiles=None)

# Set observables.
fe.set_obs(obs=['vis2', 'cp'])

# Search for companion.
fit = fe.chi2map(rlim=(4., 40.),  # mas
                 step=2.,  # mas
                 model='ud_bin',
                 fixpos2grid=False,
                 cov=False,
                 ofile=obj)

# Nail down best fit parameters.
fit = fe.mcmc(fit=fit)

# Compute contrast curve after subtracting off best fit companion.
fe.detlim(rlim=(4., 40.),
          step=4.,
          model='ud_bin',
          fit_sub=fit,
          cov=False,
          sigma=5,
          ofile=obj)

# Search for second companion after subtracting off first companion.
fe.chi2map(rlim=(4., 40.),
           step=2.,
           model='ud_bin',
           fixpos2grid=False,
           fit_sub=fit,
           cov=False,
           ofile=obj + '_sub')

pdb.set_trace()
