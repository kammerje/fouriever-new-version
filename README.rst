#########
fouriever
#########

Fouriever is a Python tool for companion search in interferometric data. Installation instructions and usage information can be found below.

Installation
************

Start by cloning the Git repository:

::

	git clone https://github.com/kammerje/fouriever-new-version.git

If you would like to install a specific branch (e.g., the :code:`develop` branch):

::

	git clone https://github.com/kammerje/fouriever-new-version.git@develop

It is **highly** recommended that you create a unique Conda environment to hold all of the fouriever dependencies:

::

	conda create -n fouriever python=3.9
	conda activate fouriever

With the Conda environment created, move to the cloned repository and install the dependencies and fouriever itself:

::

	cd where/you/saved/the/git/repo
	pip install -r requirements.txt
	pip install -e .

You should now be able to run fouriever:

::

	python run_fouriever_axcir.py

If you have issues with the PyPI installation of pymultinest, try the following:

::

	pip uninstall pymultinest
	conda install conda-forge::pymultinest

Usage
*****

The following functions are available in the :code:`Fouriever` class:

:code:`read`: Read in science and calibrator target data. Note that :code:`calibrate_classical` needs to be called afterwards to calibrate the data, it is not done automatically!

- :code:`scifiles` (required): List of science target files (paths).
- :code:`calfiles` (optional): List of calibrator target files (paths).
- :code:`plotoi` (optional): Bool - make plot of read in data or not?

:code:`set_ins`: Set the instrument whose data shall be fitted.

- :code:`ins` (required): Str - name of instrument from OIFITS file headers.

:code:`set_obs`: Set the observables which shall be fitted.

- :code:`obs` (required): List of observables which shall be fitted - available options are visibility amplitudes (:code:`'vis2'`), closure phases (:code:`'cp'`), and kernel phases (:code:`'kp'`).

:code:`add_cov`: Add covariances to the data according to `Kammerer et al. (2020) <https://ui.adsabs.harvard.edu/abs/2020A%26A...644A.110K/abstract>`__.

:code:`calibrate_classical`: Calibrate the science target data using the calibrator target data.

- :code:`weighting` (optional): Can be :code:`'standard'`, :code:`'error'`, or :code:`'covariance'` to use the mean, error-weighted mean, or covariance-weighted mean of the calibrator target data.
- :code:`write` (optional): Bool - save calibrated data to OIFITS file or not?

:code:`chi2map`: Compute a chi-squared grid to search for a companion.

- :code:`rlim` (required): Tuple of inner & outer radius of search region in mas.
- :code:`step` (required): Step size of grid in mas.
- :code:`model` (optional): Can be :code:`'bin'` or :code:`'ud_bin'` to fit a binary or uniform disk + binary model.
- :code:`fixpos2grid` (optional): Bool - fix best fit position to grid point or allow it to lie in between grid points?
- :code:`searchbox` (optional): Dict with :code:`'RA'` and :code:`'Dec'` entries which each contains a tuple of the inner & outer boundary of the search region in mas.
- :code:`fit_sub` (optional): Dict of fouriever fit result which shall be subtracted from the data before doing the companion search. This can be used to search for a second companion after subtracting off the first one.
- :code:`cov` (optional): Bool - use data covariance if available?
- :code:`smear` (optional): Int - width of bandwidth smearing to be used.
- :code:`use_ins` (optional): List of names of instruments from OIFITS file headers whose data shall be fitted. Can be used to fit to data from multiple instruments at once.
- :code:`ofile` (optional): Name tag for output file.
- :code:`plotoi` (optional): Bool - make plot of fitted data or not?

:code:`mcmc`: Nail down the best fit parameters with nested sampling.

- :code:`fit` (required): Dict of fouriever fit result whose parameters shall be nailed down.
- :code:`temperature` (optional): Temperature of the sampler.
- :code:`n_live_points` (optional): Number of live points of the sampler.
- :code:`fixpos` (optional): Bool - fix best fit position?
- :code:`fixpos` (optional): Bool - fix best fit position?
- :code:`fit_sub` (optional): Dict of fouriever fit result which shall be subtracted from the data before doing the companion search. This can be used to search for a second companion after subtracting off the first one.
- :code:`use_ins` (optional): List of names of instruments from OIFITS file headers whose data shall be fitted. Can be used to fit to data from multiple instruments at once.
- :code:`ofile` (optional): Name tag for output file.
- :code:`plotoi` (optional): Bool - make plot of fitted data or not?

:code:`detlim`: Compute a contrast curve (detection limit as function of angular separation).

- :code:`rlim` (required): Tuple of inner & outer radius of search region in mas.
- :code:`step` (required): Step size of grid in mas.
- :code:`model` (optional): Can be :code:`'bin'` or :code:`'ud_bin'` to fit a binary or uniform disk + binary model.
- :code:`fit_sub` (optional): Dict of fouriever fit result which shall be subtracted from the data before doing the companion search. This can be used to search for a second companion after subtracting off the first one.
- :code:`cov` (optional): Bool - use data covariance if available?
- :code:`smear` (optional): Int - width of bandwidth smearing to be used.
- :code:`sigma` (optional): Number of sigma for detection limit.
- :code:`cmin` (optional): Minimal test contrast (<1).
- :code:`use_ins` (optional): List of names of instruments from OIFITS file headers whose data shall be fitted. Can be used to fit to data from multiple instruments at once.
- :code:`ofile` (optional): Name tag for output file.
