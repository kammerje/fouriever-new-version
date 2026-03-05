from __future__ import division

import os
import pdb
import sys

import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import numpy as np

import pathlib
import pymultinest

from copy import deepcopy
from scipy.interpolate import interp1d, Rbf
from scipy.optimize import minimize
from typing import Any, Dict, List, Optional, Tuple, Union
from typeguard import typechecked

from fouriever import plot, util

import matplotlib
matplotlib.rcParams.update({'font.size': 14})


# =============================================================================
# MAIN
# =============================================================================

class Fouriever():
    """
    
    """

    @typechecked
    def __init__(self,
                 odir: str,
                 obj: Optional[str] = 'UNKNOWN'):
        """
        
        """

        self.odir = odir
        self.obj = obj

        self.scidata = []
        self.scidata_org = []
        self.caldata = []
        self.caldata_org = []

        self.obs_known = ['vis2', 'cp', 'kp']
        self.ftol = 1e-8
        self.maxiter = 10000

        pass

    @typechecked
    def read(self,
             scifiles: List[str],
             calfiles: Optional[List[str]] = None,
             plotoi: Optional[bool] = True):
        """
        
        """

        if calfiles is None:
            calfiles = []

        for i, scifile in enumerate(scifiles):
            hdul = pyfits.open(scifile)
            if 'OI_WAVELENGTH' in hdul:
                scidata = self.read_oifits(hdul=hdul, fitsfile=scifile)
                for ins in scidata.keys():
                    try:
                        scidata[ins]['OBJECT'] = hdul[0].header['OBJECT']
                    except:
                        scidata[ins]['OBJECT'] = 'UNKNOWN'
                if plotoi:
                    ofile = pathlib.Path(scifile).stem
                    ofile = os.path.join(self.odir, ofile + '.pdf')
                    plot.plot_oidata(data=scidata, ofile=ofile)
                self.scidata += [deepcopy(scidata)]
                self.scidata_org += [deepcopy(scidata)]
            else:
                raise NotImplementedError()
            hdul.close()

        for i, calfile in enumerate(calfiles):
            hdul = pyfits.open(calfile)
            if 'OI_WAVELENGTH' in hdul:
                caldata = self.read_oifits(hdul=hdul, fitsfile=calfile)
                for ins in caldata.keys():
                    try:
                        caldata[ins]['OBJECT'] = hdul[0].header['OBJECT']
                    except:
                        caldata[ins]['OBJECT'] = 'UNKNOWN'
                if plotoi:
                    ofile = pathlib.Path(calfile).stem
                    ofile = os.path.join(self.odir, ofile + '.pdf')
                    plot.plot_oidata(data=caldata, ofile=ofile)
                self.caldata += [deepcopy(caldata)]
                self.caldata_org += [deepcopy(caldata)]
            else:
                raise NotImplementedError()
            hdul.close()

        self.set_ins(ins=self.get_ins()[-1])
        self.set_obs(obs=self.get_obs())

        pass

    @typechecked
    def get_ins(self):
        """
        
        """

        inss = []
        for data in self.scidata:
            inss += [ins for ins in data.keys()]
        inss = np.unique(np.array(inss)).tolist()

        return inss

    @typechecked
    def set_ins(self,
                ins: str):
        """
        
        """

        inss_valid = self.get_ins()

        if ins in inss_valid:
            self.ins = ins
            print('Selected instrument = ' + self.ins)
            print('    Available instruments: ' + str(inss_valid))
            print('    Use self.set_ins(ins) to change the selected instrument')
            print()
        else:
            print('Available instruments: ' + str(inss_valid))
            raise UserWarning(ins + ' is an unknown instrument')

        pass

    @typechecked
    def get_obs(self):
        """
        
        """

        obss = []
        for data in self.scidata:
            if self.ins in data.keys():
                obss += [obs for obs in data[self.ins].keys() if obs in self.obs_known]
        obss = np.unique(np.array(obss)).tolist()

        return obss

    @typechecked
    def set_obs(self,
                obs: List[str]):
        """
        
        """

        obss_valid = self.get_obs()

        for obs_temp in obs:
            if obs_temp not in obss_valid:
                print('Available observables: ' + str(obss_valid))
                raise UserWarning(obs_temp + ' is not a valid observable for the currently selected instrument')
        self.obs = obs
        print('Selected observables = ' + str(self.obs))
        print('    Available observables: ' + str(obss_valid))
        print('    Use self.set_obs(obs) to change the selected observables')
        print()

        pass

    @typechecked
    def read_oifits(self,
                    hdul: pyfits.HDUList,
                    fitsfile: str,
                    flagged: Optional[str] = 'nan'):
        """
        
        """

        data = {}
        for hdu in hdul:

            try:
                extname = hdu.header['EXTNAME']
            except:
                continue

            if extname == 'OI_WAVELENGTH':
                ins = hdu.header['INSNAME']
                if ins.find('/') != -1:
                    ins = ins[:ins.find('/')]
                if ins.find('(') != -1:
                    ins = ins[:ins.find('(')]
                try:
                    data[ins]['wave'] = np.append(data[ins]['wave'], hdu.data['EFF_WAVE'], axis=0)  # m
                    data[ins]['dwave'] = np.append(data[ins]['dwave'], hdu.data['EFF_BAND'], axis=0)  # m
                except:
                    if ins not in data:
                        data[ins] = {}
                    data[ins]['wave'] = hdu.data['EFF_WAVE']  # m
                    data[ins]['dwave'] = hdu.data['EFF_BAND']  # m
                data[ins]['fitsfile'] = fitsfile

            elif extname == 'OI_FLUX':
                ins = hdu.header['INSNAME']
                if ins.find('/') != -1:
                    ins = ins[:ins.find('/')]
                if ins.find('(') != -1:
                    ins = ins[:ins.find('(')]
                try:
                    try:
                        data[ins]['flux'] = np.append(data[ins]['flux'], hdu.data['FLUX'], axis=0)
                    except:
                        data[ins]['flux'] = np.append(data[ins]['flux'], hdu.data['FLUXDATA'], axis=0)
                    data[ins]['dflux'] = np.append(data[ins]['dflux'], hdu.data['FLUXERR'], axis=0)
                    data[ins]['fluxflag'] = np.append(data[ins]['fluxflag'], hdu.data['FLAG'], axis=0)
                    data[ins]['fluxsta'] = np.append(data[ins]['fluxsta'], hdu.data['STA_INDEX'], axis=0)
                    data[ins]['fluxtint'] = np.append(data[ins]['fluxtint'], hdu.data['INT_TIME'], axis=0)  # s
                    data[ins]['fluxmjd'] = np.append(data[ins]['fluxmjd'], hdu.data['MJD'], axis=0)  # d
                except:
                    if ins not in data:
                        data[ins] = {}
                    try:
                        data[ins]['flux'] = hdu.data['FLUX']
                    except:
                        data[ins]['flux'] = hdu.data['FLUXDATA']
                    data[ins]['dflux'] = hdu.data['FLUXERR']
                    data[ins]['fluxflag'] = hdu.data['FLAG']
                    data[ins]['fluxsta'] = hdu.data['STA_INDEX']
                    data[ins]['fluxtint'] = hdu.data['INT_TIME']  # s
                    data[ins]['fluxmjd'] = hdu.data['MJD']  # d
                data[ins]['fluxflag'][data[ins]['dflux'] < 1e-9] = True

            elif extname == 'OI_VIS':
                ins = hdu.header['INSNAME']
                if ins.find('/') != -1:
                    ins = ins[:ins.find('/')]
                if ins.find('(') != -1:
                    ins = ins[:ins.find('(')]
                try:
                    data[ins]['vis'] = np.append(data[ins]['vis'], hdu.data['VISDATA'], axis=0)
                    data[ins]['dvis'] = np.append(data[ins]['dvis'], hdu.data['VISERR'], axis=0)
                    data[ins]['visflag'] = np.append(data[ins]['visflag'], hdu.data['FLAG'], axis=0)
                    data[ins]['visu'] = np.append(data[ins]['visu'], hdu.data['UCOORD'], axis=0)  # m
                    data[ins]['visv'] = np.append(data[ins]['visv'], hdu.data['VCOORD'], axis=0)  # m
                    data[ins]['vissta'] = np.append(data[ins]['vissta'], hdu.data['STA_INDEX'], axis=0)
                    data[ins]['vistint'] = np.append(data[ins]['vistint'], hdu.data['INT_TIME'], axis=0)  # s
                    data[ins]['vismjd'] = np.append(data[ins]['vismjd'], hdu.data['MJD'], axis=0)  # d
                except:
                    if ins not in data:
                        data[ins] = {}
                    data[ins]['vis'] = hdu.data['VISDATA']
                    data[ins]['dvis'] = hdu.data['VISERR']
                    data[ins]['visflag'] = hdu.data['FLAG']
                    data[ins]['visu'] = hdu.data['UCOORD']  # m
                    data[ins]['visv'] = hdu.data['VCOORD']  # m
                    data[ins]['vissta'] = hdu.data['STA_INDEX']
                    data[ins]['vistint'] = hdu.data['INT_TIME']  # s
                    data[ins]['vismjd'] = hdu.data['MJD']  # d
                data[ins]['visflag'][data[ins]['dvis'] < 1e-9] = True

            elif extname == 'OI_VIS2':
                ins = hdu.header['INSNAME']
                if ins.find('/') != -1:
                    ins = ins[:ins.find('/')]
                if ins.find('(') != -1:
                    ins = ins[:ins.find('(')]
                try:
                    data[ins]['vis2'] = np.append(data[ins]['vis2'], hdu.data['VIS2DATA'], axis=0)
                    data[ins]['dvis2'] = np.append(data[ins]['dvis2'], hdu.data['VIS2ERR'], axis=0)
                    data[ins]['vis2flag'] = np.append(data[ins]['vis2flag'], hdu.data['FLAG'], axis=0)
                    data[ins]['vis2u'] = np.append(data[ins]['vis2u'], hdu.data['UCOORD'], axis=0)  # m
                    data[ins]['vis2v'] = np.append(data[ins]['vis2v'], hdu.data['VCOORD'], axis=0)  # m
                    data[ins]['vis2sta'] = np.append(data[ins]['vis2sta'], hdu.data['STA_INDEX'], axis=0)
                    data[ins]['vis2tint'] = np.append(data[ins]['vis2tint'], hdu.data['INT_TIME'], axis=0)  # s
                    data[ins]['vis2mjd'] = np.append(data[ins]['vis2mjd'], hdu.data['MJD'], axis=0)  # d
                except:
                    if ins not in data:
                        data[ins] = {}
                    data[ins]['vis2'] = hdu.data['VIS2DATA']
                    data[ins]['dvis2'] = hdu.data['VIS2ERR']
                    data[ins]['vis2flag'] = hdu.data['FLAG']
                    data[ins]['vis2u'] = hdu.data['UCOORD']  # m
                    data[ins]['vis2v'] = hdu.data['VCOORD']  # m
                    data[ins]['vis2sta'] = hdu.data['STA_INDEX']
                    data[ins]['vis2tint'] = hdu.data['INT_TIME']  # s
                    data[ins]['vis2mjd'] = hdu.data['MJD']  # d
                data[ins]['vis2flag'][data[ins]['dvis2'] < 1e-9] = True

            elif extname == 'OI_T3':
                ins = hdu.header['INSNAME']
                if ins.find('/') != -1:
                    ins = ins[:ins.find('/')]
                if ins.find('(') != -1:
                    ins = ins[:ins.find('(')]
                try:
                    data[ins]['cp'] = np.append(data[ins]['cp'], np.deg2rad(hdu.data['T3PHI']), axis=0)  # rad
                    data[ins]['dcp'] = np.append(data[ins]['dcp'], np.deg2rad(hdu.data['T3PHIERR']), axis=0)  # rad
                    data[ins]['cpflag'] = np.append(data[ins]['cpflag'], hdu.data['FLAG'], axis=0)
                    data[ins]['cpu1'] = np.append(data[ins]['cpu1'], hdu.data['U1COORD'], axis=0)  # m
                    data[ins]['cpv1'] = np.append(data[ins]['cpv1'], hdu.data['V1COORD'], axis=0)  # m
                    data[ins]['cpu2'] = np.append(data[ins]['cpu2'], hdu.data['U2COORD'], axis=0)  # m
                    data[ins]['cpv2'] = np.append(data[ins]['cpv2'], hdu.data['V2COORD'], axis=0)  # m
                    data[ins]['cpsta'] = np.append(data[ins]['cpsta'], hdu.data['STA_INDEX'], axis=0)
                    data[ins]['cptint'] = np.append(data[ins]['cptint'], hdu.data['INT_TIME'], axis=0)  # s
                    data[ins]['cpmjd'] = np.append(data[ins]['cpmjd'], hdu.data['MJD'], axis=0)  # d
                except:
                    if ins not in data:
                        data[ins] = {}
                    data[ins]['cp'] = np.deg2rad(hdu.data['T3PHI'])  # rad
                    data[ins]['dcp'] = np.deg2rad(hdu.data['T3PHIERR'])  # rad
                    data[ins]['cpflag'] = hdu.data['FLAG']
                    data[ins]['cpu1'] = hdu.data['U1COORD']  # m
                    data[ins]['cpv1'] = hdu.data['V1COORD']  # m
                    data[ins]['cpu2'] = hdu.data['U2COORD']  # m
                    data[ins]['cpv2'] = hdu.data['V2COORD']  # m
                    data[ins]['cpsta'] = hdu.data['STA_INDEX']
                    data[ins]['cptint'] = hdu.data['INT_TIME']  # s
                    data[ins]['cpmjd'] = hdu.data['MJD']  # d
                data[ins]['cpflag'][data[ins]['dcp'] < 1e-9] = True

        data_prep = {}
        inss = data.keys()
        for ins in inss:
            print('#========================================')
            print('# ' + ins)
            print('#========================================')
            print()
            data_prep[ins] = {}
            data_prep[ins]['fitsfile'] = data[ins]['fitsfile']
            data_prep[ins]['klflag'] = False

            nwave = data[ins]['wave'].shape[-1]
            data_prep[ins]['wave'] = data[ins]['wave']
            data_prep[ins]['dwave'] = data[ins]['dwave']
            print('# OI_WAVELENGTH')
            print('Wavelength shape = ' + str(data_prep[ins]['wave'].shape))
            print()

            if 'flux' in data[ins].keys():
                nsta = len(np.unique(data[ins]['fluxsta']))
                nobs = data[ins]['flux'].shape[0] // nsta
                stas = data[ins]['fluxsta'][0:nsta]
                data_prep[ins]['flux'] = np.zeros((nobs, nsta, nwave))
                data_prep[ins]['dflux'] = np.zeros((nobs, nsta, nwave))
                data_prep[ins]['fluxflag'] = np.zeros((nobs, nsta, nwave), dtype=bool)
                data_prep[ins]['fluxsta'] = np.zeros((nobs, nsta), dtype=int)
                data_prep[ins]['fluxtint'] = np.zeros((nobs, nsta))
                data_prep[ins]['fluxmjd'] = np.zeros((nobs, nsta))
                data_prep[ins]['fluxpa'] = np.zeros((nobs, nsta))  # deg
                for i in range(nobs):
                    stas_temp = data[ins]['fluxsta'][i*nsta:(i+1)*nsta]
                    ww_temp = [np.where(stas_temp == stas[j])[0][0] for j in range(nsta)]
                    if data[ins]['flux'].ndim == 1 and nwave == 1:
                        data_prep[ins]['flux'][i] = data[ins]['flux'][:, np.newaxis][i*nsta:(i+1)*nsta][ww_temp]
                        data_prep[ins]['dflux'][i] = data[ins]['dflux'][:, np.newaxis][i*nsta:(i+1)*nsta][ww_temp]
                        data_prep[ins]['fluxflag'][i] = data[ins]['fluxflag'][:, np.newaxis][i*nsta:(i+1)*nsta][ww_temp]
                    else:
                        data_prep[ins]['flux'][i] = data[ins]['flux'][i*nsta:(i+1)*nsta][ww_temp]
                        data_prep[ins]['dflux'][i] = data[ins]['dflux'][i*nsta:(i+1)*nsta][ww_temp]
                        data_prep[ins]['fluxflag'][i] = data[ins]['fluxflag'][i*nsta:(i+1)*nsta][ww_temp]
                    data_prep[ins]['fluxsta'][i] = data[ins]['fluxsta'][i*nsta:(i+1)*nsta][ww_temp]
                    data_prep[ins]['fluxtint'][i] = data[ins]['fluxtint'][i*nsta:(i+1)*nsta][ww_temp]
                    data_prep[ins]['fluxmjd'][i] = data[ins]['fluxmjd'][i*nsta:(i+1)*nsta][ww_temp]
                print('# OI_FLUX')
                print('Flux shape = ' + str(data_prep[ins]['flux'].shape))
                print('Flux station shape = ' + str(data_prep[ins]['fluxsta'].shape))
                nelem = np.prod(data_prep[ins]['flux'].shape)
                if flagged == 'nan':
                    nflag = np.sum(data_prep[ins]['fluxflag'])
                    data_prep[ins]['flux'][data_prep[ins]['fluxflag']] = np.nan
                    data_prep[ins]['dflux'][data_prep[ins]['fluxflag']] = np.nan
                else:
                    nflag = 0
                print('Replaced %.0f flagged values with nans (%.0f%%)' % (nflag, nflag / nelem * 100.))
                print()

            if 'vis' in data[ins].keys():
                nsta = len(np.unique(data[ins]['vissta'], axis=0))
                nobs = data[ins]['vis'].shape[0] // nsta
                stas = data[ins]['vissta'][0:nsta]
                data_prep[ins]['vis'] = np.zeros((nobs, nsta, nwave), dtype=complex)
                data_prep[ins]['dvis'] = np.zeros((nobs, nsta, nwave), dtype=complex)
                data_prep[ins]['viscov'] = None
                data_prep[ins]['visicv'] = None
                data_prep[ins]['visflag'] = np.zeros((nobs, nsta, nwave), dtype=bool)
                data_prep[ins]['visu'] = np.zeros((nobs, nsta))
                data_prep[ins]['visv'] = np.zeros((nobs, nsta))
                data_prep[ins]['visb'] = np.zeros((nobs, nsta))
                data_prep[ins]['visuu'] = np.zeros((nobs, nsta, nwave))
                data_prep[ins]['visvv'] = np.zeros((nobs, nsta, nwave))
                data_prep[ins]['vissta'] = np.zeros((nobs, nsta, 2), dtype=int)
                data_prep[ins]['vistint'] = np.zeros((nobs, nsta))
                data_prep[ins]['vismjd'] = np.zeros((nobs, nsta))
                data_prep[ins]['vispa'] = np.zeros((nobs, nsta))  # deg
                for i in range(nobs):
                    stas_temp = data[ins]['vissta'][i*nsta:(i+1)*nsta]
                    ww_temp = [np.where((stas_temp == stas[j]).all(axis=1))[0][0] for j in range(nsta)]
                    if data[ins]['vis'].ndim == 1 and nwave == 1:
                        data_prep[ins]['vis'][i] = data[ins]['vis'][:, np.newaxis][i*nsta:(i+1)*nsta][ww_temp]
                        data_prep[ins]['dvis'][i] = data[ins]['dvis'][:, np.newaxis][i*nsta:(i+1)*nsta][ww_temp]
                        data_prep[ins]['visflag'][i] = data[ins]['visflag'][:, np.newaxis][i*nsta:(i+1)*nsta][ww_temp]
                    else:
                        data_prep[ins]['vis'][i] = data[ins]['vis'][i*nsta:(i+1)*nsta][ww_temp]
                        data_prep[ins]['dvis'][i] = data[ins]['dvis'][i*nsta:(i+1)*nsta][ww_temp]
                        data_prep[ins]['visflag'][i] = data[ins]['visflag'][i*nsta:(i+1)*nsta][ww_temp]
                    data_prep[ins]['visu'][i] = data[ins]['visu'][i*nsta:(i+1)*nsta][ww_temp]
                    data_prep[ins]['visv'][i] = data[ins]['visv'][i*nsta:(i+1)*nsta][ww_temp]
                    data_prep[ins]['visb'][i] = np.sqrt(data_prep[ins]['visu'][i]**2 + data_prep[ins]['visv'][i]**2)
                    data_prep[ins]['visuu'][i] = np.divide(data_prep[ins]['visu'][i][:, np.newaxis], data_prep[ins]['wave'][np.newaxis, :])
                    data_prep[ins]['visvv'][i] = np.divide(data_prep[ins]['visv'][i][:, np.newaxis], data_prep[ins]['wave'][np.newaxis, :])
                    data_prep[ins]['vissta'][i] = data[ins]['vissta'][i*nsta:(i+1)*nsta][ww_temp]
                    data_prep[ins]['vistint'][i] = data[ins]['vistint'][i*nsta:(i+1)*nsta][ww_temp]
                    data_prep[ins]['vismjd'][i] = data[ins]['vismjd'][i*nsta:(i+1)*nsta][ww_temp]
                print('# OI_VIS')
                print('Visibility shape = ' + str(data_prep[ins]['vis'].shape))
                print('Visibility station shape = ' + str(data_prep[ins]['vissta'].shape))
                nelem = np.prod(data_prep[ins]['vis'].shape)
                if flagged == 'nan':    
                    nflag = np.sum(data_prep[ins]['visflag'])
                    data_prep[ins]['vis'][data_prep[ins]['visflag']] = np.nan
                    data_prep[ins]['dvis'][data_prep[ins]['visflag']] = np.nan
                else:
                    nflag = 0
                print('Replaced %.0f flagged values with nans (%.0f%%)' % (nflag, nflag / nelem * 100.))
                print()

            if 'vis2' in data[ins].keys():
                nsta = len(np.unique(data[ins]['vis2sta'], axis=0))
                nobs = data[ins]['vis2'].shape[0] // nsta
                stas = data[ins]['vis2sta'][0:nsta]
                data_prep[ins]['vis2'] = np.zeros((nobs, nsta, nwave))
                data_prep[ins]['dvis2'] = np.zeros((nobs, nsta, nwave))
                data_prep[ins]['vis2cov'] = None
                data_prep[ins]['vis2icv'] = None
                data_prep[ins]['vis2flag'] = np.zeros((nobs, nsta, nwave), dtype=bool)
                data_prep[ins]['vis2u'] = np.zeros((nobs, nsta))
                data_prep[ins]['vis2v'] = np.zeros((nobs, nsta))
                data_prep[ins]['vis2b'] = np.zeros((nobs, nsta))
                data_prep[ins]['vis2uu'] = np.zeros((nobs, nsta, nwave))
                data_prep[ins]['vis2vv'] = np.zeros((nobs, nsta, nwave))
                data_prep[ins]['vis2sta'] = np.zeros((nobs, nsta, 2), dtype=int)
                data_prep[ins]['vis2tint'] = np.zeros((nobs, nsta))
                data_prep[ins]['vis2mjd'] = np.zeros((nobs, nsta))
                data_prep[ins]['vis2pa'] = np.zeros((nobs, nsta))  # deg
                for i in range(nobs):
                    stas_temp = data[ins]['vis2sta'][i*nsta:(i+1)*nsta]
                    ww_temp = [np.where((stas_temp == stas[j]).all(axis=1))[0][0] for j in range(nsta)]
                    if data[ins]['vis2'].ndim == 1 and nwave == 1:
                        data_prep[ins]['vis2'][i] = data[ins]['vis2'][:, np.newaxis][i*nsta:(i+1)*nsta][ww_temp]
                        data_prep[ins]['dvis2'][i] = data[ins]['dvis2'][:, np.newaxis][i*nsta:(i+1)*nsta][ww_temp]
                        data_prep[ins]['vis2flag'][i] = data[ins]['vis2flag'][:, np.newaxis][i*nsta:(i+1)*nsta][ww_temp]
                    else:
                        data_prep[ins]['vis2'][i] = data[ins]['vis2'][i*nsta:(i+1)*nsta][ww_temp]
                        data_prep[ins]['dvis2'][i] = data[ins]['dvis2'][i*nsta:(i+1)*nsta][ww_temp]
                        data_prep[ins]['vis2flag'][i] = data[ins]['vis2flag'][i*nsta:(i+1)*nsta][ww_temp]
                    data_prep[ins]['vis2u'][i] = data[ins]['vis2u'][i*nsta:(i+1)*nsta][ww_temp]
                    data_prep[ins]['vis2v'][i] = data[ins]['vis2v'][i*nsta:(i+1)*nsta][ww_temp]
                    data_prep[ins]['vis2b'][i] = np.sqrt(data_prep[ins]['vis2u'][i]**2 + data_prep[ins]['vis2v'][i]**2)
                    data_prep[ins]['vis2uu'][i] = np.divide(data_prep[ins]['vis2u'][i][:, np.newaxis], data_prep[ins]['wave'][np.newaxis, :])
                    data_prep[ins]['vis2vv'][i] = np.divide(data_prep[ins]['vis2v'][i][:, np.newaxis], data_prep[ins]['wave'][np.newaxis, :])
                    data_prep[ins]['vis2sta'][i] = data[ins]['vis2sta'][i*nsta:(i+1)*nsta][ww_temp]
                    data_prep[ins]['vis2tint'][i] = data[ins]['vis2tint'][i*nsta:(i+1)*nsta][ww_temp]
                    data_prep[ins]['vis2mjd'][i] = data[ins]['vis2mjd'][i*nsta:(i+1)*nsta][ww_temp]
                print('# OI_VIS2')
                print('Visibility amplitude shape = ' + str(data_prep[ins]['vis2'].shape))
                print('Visibility amplitude station shape = ' + str(data_prep[ins]['vis2sta'].shape))
                nelem = np.prod(data_prep[ins]['vis2'].shape)
                if flagged == 'nan':    
                    nflag = np.sum(data_prep[ins]['vis2flag'])
                    data_prep[ins]['vis2'][data_prep[ins]['vis2flag']] = np.nan
                    data_prep[ins]['dvis2'][data_prep[ins]['vis2flag']] = np.nan
                else:
                    nflag = 0
                print('Replaced %.0f flagged values with nans (%.0f%%)' % (nflag, nflag / nelem * 100.))
                print()

            if 'cp' in data[ins].keys():
                nsta = len(np.unique(data[ins]['cpsta'], axis=0))
                nobs = data[ins]['cp'].shape[0] // nsta
                stas = data[ins]['cpsta'][0:nsta]
                data_prep[ins]['cp'] = np.zeros((nobs, nsta, nwave))
                data_prep[ins]['dcp'] = np.zeros((nobs, nsta, nwave))
                data_prep[ins]['cpcov'] = None
                data_prep[ins]['cpicv'] = None
                data_prep[ins]['cpflag'] = np.zeros((nobs, nsta, nwave), dtype=bool)
                data_prep[ins]['cpu1'] = np.zeros((nobs, nsta))
                data_prep[ins]['cpv1'] = np.zeros((nobs, nsta))
                data_prep[ins]['cpu2'] = np.zeros((nobs, nsta))
                data_prep[ins]['cpv2'] = np.zeros((nobs, nsta))
                data_prep[ins]['cpsta'] = np.zeros((nobs, nsta, 3), dtype=int)
                data_prep[ins]['cptint'] = np.zeros((nobs, nsta))
                data_prep[ins]['cpmjd'] = np.zeros((nobs, nsta))
                data_prep[ins]['cppa'] = np.zeros((nobs, nsta))  # deg
                for i in range(nobs):
                    stas_temp = data[ins]['cpsta'][i*nsta:(i+1)*nsta]
                    ww_temp = [np.where((stas_temp == stas[j]).all(axis=1))[0][0] for j in range(nsta)]
                    if data[ins]['cp'].ndim == 1 and nwave == 1:
                        data_prep[ins]['cp'][i] = data[ins]['cp'][:, np.newaxis][i*nsta:(i+1)*nsta][ww_temp] / np.sqrt(3.)
                        data_prep[ins]['dcp'][i] = data[ins]['dcp'][:, np.newaxis][i*nsta:(i+1)*nsta][ww_temp] / np.sqrt(3.)
                        data_prep[ins]['cpflag'][i] = data[ins]['cpflag'][:, np.newaxis][i*nsta:(i+1)*nsta][ww_temp]
                    else:
                        data_prep[ins]['cp'][i] = data[ins]['cp'][i*nsta:(i+1)*nsta][ww_temp] / np.sqrt(3.)
                        data_prep[ins]['dcp'][i] = data[ins]['dcp'][i*nsta:(i+1)*nsta][ww_temp] / np.sqrt(3.)
                        data_prep[ins]['cpflag'][i] = data[ins]['cpflag'][i*nsta:(i+1)*nsta][ww_temp]
                    data_prep[ins]['cpu1'][i] = data[ins]['cpu1'][i*nsta:(i+1)*nsta][ww_temp]
                    data_prep[ins]['cpv1'][i] = data[ins]['cpv1'][i*nsta:(i+1)*nsta][ww_temp]
                    data_prep[ins]['cpu2'][i] = data[ins]['cpu2'][i*nsta:(i+1)*nsta][ww_temp]
                    data_prep[ins]['cpv2'][i] = data[ins]['cpv2'][i*nsta:(i+1)*nsta][ww_temp]
                    data_prep[ins]['cpsta'][i] = data[ins]['cpsta'][i*nsta:(i+1)*nsta][ww_temp]
                    data_prep[ins]['cptint'][i] = data[ins]['cptint'][i*nsta:(i+1)*nsta][ww_temp]
                    data_prep[ins]['cpmjd'][i] = data[ins]['cpmjd'][i*nsta:(i+1)*nsta][ww_temp]
                data_prep[ins]['cpmat'] = np.zeros((data_prep[ins]['cpsta'].shape[1], data_prep[ins]['vis2sta'].shape[1]))
                for i in range(data_prep[ins]['cpmat'].shape[0]):
                    base1 = data_prep[ins]['cpsta'][0, i][[0, 1]]
                    base2 = data_prep[ins]['cpsta'][0, i][[1, 2]]
                    base3 = data_prep[ins]['cpsta'][0, i][[2, 0]]
                    flag1 = False
                    flag2 = False
                    flag3 = False
                    j = 0
                    while ((flag1 & flag2 & flag3) == False):
                        base = data_prep[ins]['vis2sta'][0, j]
                        if ((flag1 == False) & np.array_equal(base1, base)):
                            data_prep[ins]['cpmat'][i, j] = 1. / np.sqrt(3.)
                            flag1 = True
                        elif ((flag2 == False) & np.array_equal(base2, base)):
                            data_prep[ins]['cpmat'][i, j] = 1. / np.sqrt(3.)
                            flag2 = True
                        elif ((flag3 == False) & np.array_equal(base3, base)):
                            data_prep[ins]['cpmat'][i, j] = 1. / np.sqrt(3.)
                            flag3 = True
                        elif ((flag1 == False) & np.array_equal(base1[::-1], base)):
                            data_prep[ins]['cpmat'][i, j] = -1. / np.sqrt(3.)
                            flag1 = True
                        elif ((flag2 == False) & np.array_equal(base2[::-1], base)):
                            data_prep[ins]['cpmat'][i, j] = -1. / np.sqrt(3.)
                            flag2 = True
                        elif ((flag3 == False) & np.array_equal(base3[::-1], base)):
                            data_prep[ins]['cpmat'][i, j] = -1. / np.sqrt(3.)
                            flag3 = True
                        j += 1

                U, S, Vh = np.linalg.svd(data_prep[ins]['cpmat'].T, full_matrices=1)
                ncp = data_prep[ins]['cpmat'].shape[0]
                nkp = np.size(np.where(abs(S) > 1e-3))
                kpcol = np.where(abs(S) > 1e-3)[0]
                kpmat = np.zeros((nkp, ncp))
                for i in range(nkp):
                    kpmat[i, :] = Vh[kpcol[i], :]
                data_prep[ins]['kpmat'] = np.dot(kpmat, data_prep[ins]['cpmat'])
                norm = np.linalg.norm(data_prep[ins]['kpmat'], axis=1)
                kpmat = np.divide(kpmat.T, norm).T
                data_prep[ins]['kp'] = np.matmul(kpmat, data_prep[ins]['cp'])
                data_prep[ins]['dkp'] = np.sqrt(np.matmul(kpmat**2, data_prep[ins]['dcp']**2))
                data_prep[ins]['kpcov'] = None
                data_prep[ins]['kpicv'] = None
                data_prep[ins]['kpmat'] = np.dot(kpmat, data_prep[ins]['cpmat'])

                print('# OI_T3')
                print('Closure phase shape = ' + str(data_prep[ins]['cp'].shape))
                print('Closure phase station shape = ' + str(data_prep[ins]['cpsta'].shape))
                nelem = np.prod(data_prep[ins]['cp'].shape)
                if flagged == 'nan':    
                    nflag = np.sum(data_prep[ins]['cpflag'])
                    data_prep[ins]['cp'][data_prep[ins]['cpflag']] = np.nan
                    data_prep[ins]['dcp'][data_prep[ins]['cpflag']] = np.nan
                else:
                    nflag = 0
                print('Replaced %.0f flagged values with nans (%.0f%%)' % (nflag, nflag / nelem * 100.))
                print()

        return data_prep

    @typechecked
    def add_cov(self):
        """
        
        """

        print('#========================================')
        print('# Adding covariances')
        print('#========================================')
        print()

        vis2flag = False
        cpflag = False
        kpflag = False
        for ind, data in enumerate(self.scidata_org + self.caldata_org):
            inss = data.keys()
            for ins in inss:

                if 'vis2' in data[ins].keys():
                    nwave = data[ins]['wave'].shape[0]
                    nbase = data[ins]['vis2'].shape[1]
                    cor = np.diag(np.ones(nwave * nbase))
                    ofile = 'vis2cov_' + ins + '.pdf'
                    ofile = os.path.join(self.odir, ofile)
                    plot.plot_cor(cor=cor, obs='vis2', ofile=ofile)
                    data[ins]['vis2cov'] = np.zeros((data[ins]['dvis2'].shape[0], nbase * nwave, nbase * nwave))
                    data[ins]['vis2icv'] = np.zeros((data[ins]['dvis2'].shape[0], nbase * nwave, nbase * nwave))
                    for i in range(data[ins]['dvis2'].shape[0]):
                        dd = data[ins]['dvis2'][i].flatten()
                        ww = ~np.isnan(dd)
                        xx = np.arange(len(dd))
                        dd_interp = interp1d(xx[ww], dd[ww], kind='linear')
                        ww = np.isnan(dd)
                        dd[ww] = dd_interp(xx[ww])
                        data[ins]['vis2cov'][i] = np.multiply(cor, dd[:, None] * dd[None, :])
                        data[ins]['vis2icv'][i] = np.linalg.pinv(data[ins]['vis2cov'][i])
                    if ind < len(self.scidata_org):
                        self.scidata[ind][ins]['vis2cov'] = data[ins]['vis2cov'].copy()
                        self.scidata[ind][ins]['vis2icv'] = data[ins]['vis2icv'].copy()
                    else:
                        self.caldata[ind - len(self.scidata_org)][ins]['vis2cov'] = data[ins]['vis2cov'].copy()
                        self.caldata[ind - len(self.scidata_org)][ins]['vis2icv'] = data[ins]['vis2icv'].copy()
                    vis2flag = True

                if 'cp' in data[ins].keys():
                    nwave = data[ins]['wave'].shape[0]
                    nbase = data[ins]['vis2'].shape[1]
                    ncp = data[ins]['cp'].shape[1]
                    trafo = np.zeros((ncp * nwave, nbase * nwave))
                    for i in range(ncp):
                        for j in range(nbase):
                            trafo[i * nwave:(i + 1) * nwave, j * nwave:(j + 1) * nwave] = np.diag(np.ones(nwave)) * data[ins]['cpmat'][i, j]
                            cor = np.dot(trafo, np.dot(np.diag(np.ones(nwave * nbase)), trafo.T))
                    ofile = 'cpcov_' + ins + '.pdf'
                    ofile = os.path.join(self.odir, ofile)
                    plot.plot_cor(cor=cor, obs='cp', ofile=ofile)
                    data[ins]['cpcov'] = np.zeros((data[ins]['dcp'].shape[0], ncp * nwave, ncp * nwave))
                    data[ins]['cpicv'] = np.zeros((data[ins]['dcp'].shape[0], ncp * nwave, ncp * nwave))
                    for i in range(data[ins]['dcp'].shape[0]):
                        dd = data[ins]['dcp'][i].flatten()
                        ww = ~np.isnan(dd)
                        xx = np.arange(len(dd))
                        dd_interp = interp1d(xx[ww], dd[ww], kind='linear')
                        ww = np.isnan(dd)
                        dd[ww] = dd_interp(xx[ww])
                        data[ins]['cpcov'][i] = np.multiply(cor, dd[:, None] * dd[None, :])
                        data[ins]['cpicv'][i] = np.linalg.pinv(data[ins]['cpcov'][i])
                    if ind < len(self.scidata_org):
                        self.scidata[ind][ins]['cpcov'] = data[ins]['cpcov'].copy()
                        self.scidata[ind][ins]['cpicv'] = data[ins]['cpicv'].copy()
                    else:
                        self.caldata[ind - len(self.scidata_org)][ins]['cpcov'] = data[ins]['cpcov'].copy()
                        self.caldata[ind - len(self.scidata_org)][ins]['cpicv'] = data[ins]['cpicv'].copy()
                    cpflag = True

                if 'kp' in data[ins].keys():
                    nwave = data[ins]['wave'].shape[0]
                    nbase = data[ins]['vis2'].shape[1]
                    nkp = data[ins]['kp'].shape[1]
                    trafo = np.zeros((nkp * nwave, nbase * nwave))
                    for i in range(nkp):
                        for j in range(nbase):
                            trafo[i * nwave:(i + 1) * nwave, j * nwave:(j + 1) * nwave] = np.diag(np.ones(nwave)) * data[ins]['kpmat'][i, j]
                            cor = np.dot(trafo, np.dot(np.diag(np.ones(nwave * nbase)), trafo.T))
                    ofile = 'kpcov_' + ins + '.pdf'
                    ofile = os.path.join(self.odir, ofile)
                    plot.plot_cor(cor=cor, obs='kp', ofile=ofile)
                    data[ins]['kpcov'] = np.zeros((data[ins]['dkp'].shape[0], nkp * nwave, nkp * nwave))
                    data[ins]['kpicv'] = np.zeros((data[ins]['dkp'].shape[0], nkp * nwave, nkp * nwave))
                    for i in range(data[ins]['dkp'].shape[0]):
                        dd = data[ins]['dkp'][i].flatten()
                        ww = ~np.isnan(dd)
                        xx = np.arange(len(dd))
                        dd_interp = interp1d(xx[ww], dd[ww], kind='linear')
                        ww = np.isnan(dd)
                        dd[ww] = dd_interp(xx[ww])
                        data[ins]['kpcov'][i] = np.multiply(cor, dd[:, None] * dd[None, :])
                        data[ins]['kpicv'][i] = np.linalg.pinv(data[ins]['kpcov'][i])
                    if ind < len(self.scidata_org):
                        self.scidata[ind][ins]['kpcov'] = data[ins]['kpcov'].copy()
                        self.scidata[ind][ins]['kpicv'] = data[ins]['kpicv'].copy()
                    else:
                        self.caldata[ind - len(self.scidata_org)][ins]['kpcov'] = data[ins]['kpcov'].copy()
                        self.caldata[ind - len(self.scidata_org)][ins]['kpicv'] = data[ins]['kpicv'].copy()
                    kpflag = True

        if vis2flag:
            print('Added visibility amplitude covariances')
        if cpflag:
            print('Added closure phase covariances')
        if kpflag:
            print('Added kernel phase covariances')
        if not vis2flag and not cpflag and not kpflag:
            print('Found no data to add covariances')
        print()

        pass

    @typechecked
    def calibrate_classical(self,
                            weighting: Optional[str] = 'standard',
                            write: bool = False):
        """
        
        """

        print('#========================================')
        print('# Calibrating data')
        print('#========================================')
        print()

        vis2_cal = []
        dvis2_cal = []
        vis2cov_cal = []
        cov = False
        for data in self.caldata_org:
            if self.ins in data.keys():
                if 'vis2' in data[self.ins].keys():
                    wave = data[self.ins]['wave']
                    vis2_cal += [data[self.ins]['vis2']]
                    dvis2_cal += [data[self.ins]['dvis2']]
                    if data[self.ins]['vis2cov'] is not None:
                        vis2cov_cal += [data[self.ins]['vis2cov']]
                        cov = True

        if len(vis2_cal) > 0:
            vis2_cal = np.vstack([vis2 for vis2 in vis2_cal])
            dvis2_cal = np.vstack([dvis2 for dvis2 in dvis2_cal])
            if cov:
                vis2cov_cal = np.vstack([vis2cov for vis2cov in vis2cov_cal])

            print('Visibility amplitude')
            print('    Found %.0f calibrator observations' % vis2_cal.shape[0])

            if weighting == 'standard':
                vis2_cal_master = np.mean(vis2_cal, axis=0)
                dvis2_cal_master = np.sqrt(np.sum(dvis2_cal**2, axis=0)) / dvis2_cal.shape[0]
                if cov:
                    vis2cov_cal_master = np.sum(vis2cov_cal, axis=0) / vis2cov_cal.shape[0]**2
                else:
                    vis2cov_cal_master = None

            elif weighting == 'error':
                ww_cal = []
                vis2_cal_master = []
                for i in range(vis2_cal.shape[0]):
                    ww_cal += [(1. / dvis2_cal[i]**2)]
                    vis2_cal_master += [ww_cal[i] * vis2_cal[i]]
                ww_cal = np.array(ww_cal)
                vis2_cal_master = np.array(vis2_cal_master)
                vis2_cal_master = np.sum(vis2_cal_master, axis=0) / np.sum(ww_cal, axis=0)
                dvis2_cal_master = np.sqrt(1. / np.sum(ww_cal, axis=0))
                vis2cov_cal_master = None

            elif weighting == 'covariance':
                if not cov:
                    raise UserWarning()
                vis2_cal_master = []
                vis2cov_cal_master = []
                for i in range(vis2_cal.shape[0]):
                    temp = np.linalg.pinv(vis2cov_cal[i])
                    vis2_cal_master += [np.dot(temp, vis2_cal[i].flatten())]
                    vis2cov_cal_master += [temp]
                vis2_cal_master = np.array(vis2_cal_master)
                vis2cov_cal_master = np.array(vis2cov_cal_master)
                vis2cov_cal_master = np.linalg.pinv(np.sum(vis2cov_cal_master, axis=0))
                vis2_cal_master = np.dot(vis2cov_cal_master, np.sum(vis2_cal_master, axis=0))
                vis2_cal_master = vis2_cal_master.reshape((vis2_cal.shape[1], vis2_cal.shape[2]))
                dvis2_cal_master = np.sqrt(np.diag(vis2cov_cal_master)).reshape((dvis2_cal.shape[1], dvis2_cal.shape[2]))

            print()

            ofile = self.obj
            ofile = os.path.join(self.odir, ofile + '_vis2_cal_avg.pdf')
            plot.plot_avg_vis2(wave=wave,
                               vis2_cal=vis2_cal,
                               dvis2_cal=dvis2_cal,
                               vis2cov_cal=vis2cov_cal,
                               vis2_cal_master=vis2_cal_master,
                               dvis2_cal_master=dvis2_cal_master,
                               vis2cov_cal_master=vis2cov_cal_master,
                               ofile=ofile)

            if write:
                odir = os.path.join(self.odir, 'calibrated_oifits/')
                if not os.path.exists(odir):
                    os.makedirs(odir)

            for ind, data in enumerate(self.scidata_org):
                if self.ins in data.keys():
                    if 'vis2' in data[self.ins].keys():
                        for i in range(data[self.ins]['vis2'].shape[0]):
                            self.scidata[ind][self.ins]['vis2'][i] = np.divide(data[self.ins]['vis2'][i], vis2_cal_master)
                            self.scidata[ind][self.ins]['dvis2'][i] = np.sqrt(np.divide(data[self.ins]['dvis2'][i], vis2_cal_master)**2 + np.divide(np.multiply(data[self.ins]['vis2'][i], dvis2_cal_master), vis2_cal_master**2)**2)
                            if vis2cov_cal_master is not None:
                                dd_cal = vis2_cal_master.flatten()
                                ww_cal = ~np.isnan(dd_cal)
                                xx_cal = np.arange(len(dd_cal))
                                dd_cal_interp = interp1d(xx_cal[ww_cal], dd_cal[ww_cal], kind='linear')
                                ww_cal = np.isnan(dd_cal)
                                dd_cal[ww_cal] = dd_cal_interp(xx_cal[ww_cal])
                                temp_cal = np.meshgrid(1. / dd_cal, 1. / dd_cal)
                                temp_cal = np.multiply(temp_cal[0], temp_cal[1])
                                dd_sci = data[self.ins]['vis2'][i].flatten()
                                ww_sci = ~np.isnan(dd_sci)
                                xx_sci = np.arange(len(dd_sci))
                                dd_sci_interp = interp1d(xx_sci[ww_sci], dd_sci[ww_sci], kind='linear')
                                ww_sci = np.isnan(dd_sci)
                                dd_sci[ww_sci] = dd_sci_interp(xx_sci[ww_sci])
                                temp_sci = np.meshgrid(dd_sci / dd_cal**2, dd_sci / dd_cal**2)
                                temp_sci = np.multiply(temp_sci[0], temp_sci[1])
                                self.scidata[ind][self.ins]['vis2cov'][i] = np.multiply(temp_cal, data[self.ins]['vis2cov'][i]) + np.multiply(temp_sci, vis2cov_cal_master)
                                self.scidata[ind][self.ins]['vis2icv'][i] = np.linalg.pinv(self.scidata[ind][self.ins]['vis2cov'][i])
                            else:
                                self.scidata[ind][self.ins]['vis2cov'] = None
                                self.scidata[ind][self.ins]['vis2icv'] = None

                        ofile = pathlib.Path(self.scidata[ind][self.ins]['fitsfile']).stem
                        ofile = os.path.join(self.odir, ofile + '_vis2_sci_cal.pdf')
                        plot.plot_cal_vis2(wave=wave,
                                           vis2_sci=self.scidata_org[ind][self.ins]['vis2'],
                                           dvis2_sci=self.scidata_org[ind][self.ins]['dvis2'],
                                           vis2cov_sci=self.scidata_org[ind][self.ins]['vis2cov'],
                                           vis2_cal=vis2_cal_master,
                                           dvis2_cal=dvis2_cal_master,
                                           vis2cov_cal=vis2cov_cal_master,
                                           vis2_sci_cal=self.scidata[ind][self.ins]['vis2'],
                                           dvis2_sci_cal=self.scidata[ind][self.ins]['dvis2'],
                                           vis2cov_sci_cal=self.scidata[ind][self.ins]['vis2cov'],
                                           ofile=ofile)

                        if write:
                            ofile = os.path.split(data[self.ins]['fitsfile'])[1]
                            ofile = os.path.join(odir,  ofile)
                            try:
                                hdul = pyfits.open(ofile)
                            except:
                                hdul = pyfits.open(data[self.ins]['fitsfile'])
                            for i, hdu in enumerate(hdul):
                                try:
                                    if hdu.header['EXTNAME'] == 'OI_VIS2' and hdu.header['INSNAME'] == self.ins:
                                        break
                                except:
                                    pass
                            temp = hdul[i].data
                            temp['VIS2DATA'] = self.scidata[ind][self.ins]['vis2'][0]
                            temp['VIS2ERR'] = self.scidata[ind][self.ins]['dvis2'][0]
                            hdul[i].data = temp
                            hdul.writeto(ofile, overwrite=True)

        cp_cal = []
        dcp_cal = []
        cpcov_cal = []
        cov = False
        for data in self.caldata_org:
            if self.ins in data.keys():
                if 'cp' in data[self.ins].keys():
                    wave = data[self.ins]['wave']
                    cp_cal += [data[self.ins]['cp']]
                    dcp_cal += [data[self.ins]['dcp']]
                    if data[self.ins]['cpcov'] is not None:
                        cpcov_cal += [data[self.ins]['cpcov']]
                        cov = True

        if len(cp_cal) > 0:
            cp_cal = np.vstack([cp for cp in cp_cal])
            dcp_cal = np.vstack([dcp for dcp in dcp_cal])
            if cov:
                cpcov_cal = np.vstack([cpcov for cpcov in cpcov_cal])

            print('Closure phase')
            print('    Found %.0f calibrator observations' % cp_cal.shape[0])

            if weighting == 'standard':
                cp_cal_master = np.mean(cp_cal, axis=0)
                dcp_cal_master = np.sqrt(np.sum(dcp_cal**2, axis=0)) / dcp_cal.shape[0]
                if cov:
                    cpcov_cal_master = np.sum(cpcov_cal, axis=0) / cpcov_cal.shape[0]**2
                else:
                    cpcov_cal_master = None

            elif weighting == 'error':
                ww_cal = []
                cp_cal_master = []
                for i in range(cp_cal.shape[0]):
                    ww_cal += [(1. / dcp_cal[i]**2)]
                    cp_cal_master += [ww_cal[i] * cp_cal[i]]
                ww_cal = np.array(ww_cal)
                cp_cal_master = np.array(cp_cal_master)
                cp_cal_master = np.sum(cp_cal_master, axis=0) / np.sum(ww_cal, axis=0)
                dcp_cal_master = np.sqrt(1. / np.sum(ww_cal, axis=0))
                cpcov_cal_master = None

            elif weighting == 'covariance':
                if not cov:
                    raise UserWarning()
                print('    WARNING: computing a covariance-weighted mean with closure phases is not recommended!')
                cp_cal_master = []
                cpcov_cal_master = []
                for i in range(cp_cal.shape[0]):
                    temp = np.linalg.pinv(cpcov_cal[i])
                    cp_cal_master += [np.dot(temp, cp_cal[i].flatten())]
                    cpcov_cal_master += [temp]
                cp_cal_master = np.array(cp_cal_master)
                cpcov_cal_master = np.array(cpcov_cal_master)
                cpcov_cal_master = np.linalg.pinv(np.sum(cpcov_cal_master, axis=0))
                cp_cal_master = np.dot(cpcov_cal_master, np.sum(cp_cal_master, axis=0))
                cp_cal_master = cp_cal_master.reshape((cp_cal.shape[1], cp_cal.shape[2]))
                dcp_cal_master = np.sqrt(np.diag(cpcov_cal_master)).reshape((dcp_cal.shape[1], dcp_cal.shape[2]))

            print()

            ofile = self.obj
            ofile = os.path.join(self.odir, ofile + '_cp_cal_avg.pdf')
            plot.plot_avg_cp(wave=wave,
                             cp_cal=cp_cal,
                             dcp_cal=dcp_cal,
                             cpcov_cal=cpcov_cal,
                             cp_cal_master=cp_cal_master,
                             dcp_cal_master=dcp_cal_master,
                             cpcov_cal_master=cpcov_cal_master,
                             ofile=ofile)

            for ind, data in enumerate(self.scidata_org):
                if self.ins in data.keys():
                    if 'cp' in data[self.ins].keys():
                        for i in range(data[self.ins]['cp'].shape[0]):
                            self.scidata[ind][self.ins]['cp'][i] = data[self.ins]['cp'][i] - cp_cal_master
                            self.scidata[ind][self.ins]['dcp'][i] = np.sqrt(data[self.ins]['dcp'][i]**2 + dcp_cal_master**2)
                            if cpcov_cal_master is not None:
                                self.scidata[ind][self.ins]['cpcov'][i] = data[self.ins]['cpcov'][i] + cpcov_cal_master
                                self.scidata[ind][self.ins]['cpicv'][i] = np.linalg.pinv(self.scidata[ind][self.ins]['cpcov'][i])
                            else:
                                self.scidata[ind][self.ins]['cpcov'] = None
                                self.scidata[ind][self.ins]['cpicv'] = None

                        ofile = pathlib.Path(self.scidata[ind][self.ins]['fitsfile']).stem
                        ofile = os.path.join(self.odir, ofile + '_cp_sci_cal.pdf')
                        plot.plot_cal_cp(wave=wave,
                                         cp_sci=self.scidata_org[ind][self.ins]['cp'],
                                         dcp_sci=self.scidata_org[ind][self.ins]['dcp'],
                                         cpcov_sci=self.scidata_org[ind][self.ins]['cpcov'],
                                         cp_cal=cp_cal_master,
                                         dcp_cal=dcp_cal_master,
                                         cpcov_cal=cpcov_cal_master,
                                         cp_sci_cal=self.scidata[ind][self.ins]['cp'],
                                         dcp_sci_cal=self.scidata[ind][self.ins]['dcp'],
                                         cpcov_sci_cal=self.scidata[ind][self.ins]['cpcov'],
                                         ofile=ofile)

                        if write:
                            ofile = os.path.split(data[self.ins]['fitsfile'])[1]
                            ofile = os.path.join(odir,  ofile)
                            try:
                                hdul = pyfits.open(ofile)
                            except:
                                hdul = pyfits.open(data[self.ins]['fitsfile'])
                            for i, hdu in enumerate(hdul):
                                try:
                                    if hdu.header['EXTNAME'] == 'OI_T3' and hdu.header['INSNAME'] == self.ins:
                                        break
                                except:
                                    pass
                            temp = hdul[i].data
                            temp['T3PHI'] = np.rad2deg(self.scidata[ind][self.ins]['cp'][0] * np.sqrt(3.))
                            temp['T3PHIERR'] = np.rad2deg(self.scidata[ind][self.ins]['dcp'][0] * np.sqrt(3.))
                            hdul[i].data = temp
                            hdul.writeto(ofile, overwrite=True)

        kp_cal = []
        dkp_cal = []
        kpcov_cal = []
        cov = False
        for data in self.caldata_org:
            if self.ins in data.keys():
                if 'kp' in data[self.ins].keys():
                    wave = data[self.ins]['wave']
                    kp_cal += [data[self.ins]['kp']]
                    dkp_cal += [data[self.ins]['dkp']]
                    if data[self.ins]['kpcov'] is not None:
                        kpcov_cal += [data[self.ins]['kpcov']]
                        cov = True

        if len(kp_cal) > 0:
            kp_cal = np.vstack([kp for kp in kp_cal])
            dkp_cal = np.vstack([dkp for dkp in dkp_cal])
            if cov:
                kpcov_cal = np.vstack([kpcov for kpcov in kpcov_cal])

            print('Kernel phase')
            print('    Found %.0f calibrator observations' % kp_cal.shape[0])

            if weighting == 'standard':
                kp_cal_master = np.mean(kp_cal, axis=0)
                dkp_cal_master = np.sqrt(np.sum(dkp_cal**2, axis=0)) / dkp_cal.shape[0]
                if cov:
                    kpcov_cal_master = np.sum(kpcov_cal, axis=0) / kpcov_cal.shape[0]**2
                else:
                    kpcov_cal_master = None

            elif weighting == 'error':
                ww_cal = []
                kp_cal_master = []
                for i in range(kp_cal.shape[0]):
                    ww_cal += [(1. / dkp_cal[i]**2)]
                    kp_cal_master += [ww_cal[i] * kp_cal[i]]
                ww_cal = np.array(ww_cal)
                kp_cal_master = np.array(kp_cal_master)
                kp_cal_master = np.sum(kp_cal_master, axis=0) / np.sum(ww_cal, axis=0)
                dkp_cal_master = np.sqrt(1. / np.sum(ww_cal, axis=0))
                kpcov_cal_master = None

            elif weighting == 'covariance':
                if not cov:
                    raise UserWarning()
                kp_cal_master = []
                kpcov_cal_master = []
                for i in range(kp_cal.shape[0]):
                    temp = np.linalg.pinv(kpcov_cal[i])
                    kp_cal_master += [np.dot(temp, kp_cal[i].flatten())]
                    kpcov_cal_master += [temp]
                kp_cal_master = np.array(kp_cal_master)
                kpcov_cal_master = np.array(kpcov_cal_master)
                kpcov_cal_master = np.linalg.pinv(np.sum(kpcov_cal_master, axis=0))
                kp_cal_master = np.dot(kpcov_cal_master, np.sum(kp_cal_master, axis=0))
                kp_cal_master = kp_cal_master.reshape((kp_cal.shape[1], kp_cal.shape[2]))
                dkp_cal_master = np.sqrt(np.diag(kpcov_cal_master)).reshape((dkp_cal.shape[1], dkp_cal.shape[2]))

            print()

            ofile = self.obj
            ofile = os.path.join(self.odir, ofile + '_kp_cal_avg.pdf')
            plot.plot_avg_kp(wave=wave,
                             kp_cal=kp_cal,
                             dkp_cal=dkp_cal,
                             kpcov_cal=kpcov_cal,
                             kp_cal_master=kp_cal_master,
                             dkp_cal_master=dkp_cal_master,
                             kpcov_cal_master=kpcov_cal_master,
                             ofile=ofile)

            for ind, data in enumerate(self.scidata_org):
                if self.ins in data.keys():
                    if 'kp' in data[self.ins].keys():
                        for i in range(data[self.ins]['kp'].shape[0]):
                            self.scidata[ind][self.ins]['kp'][i] = data[self.ins]['kp'][i] - kp_cal_master
                            self.scidata[ind][self.ins]['dkp'][i] = np.sqrt(data[self.ins]['dkp'][i]**2 + dkp_cal_master**2)
                            if kpcov_cal_master is not None:
                                self.scidata[ind][self.ins]['kpcov'][i] = data[self.ins]['kpcov'][i] + kpcov_cal_master
                                self.scidata[ind][self.ins]['kpicv'][i] = np.linalg.pinv(self.scidata[ind][self.ins]['kpcov'][i])
                            else:
                                self.scidata[ind][self.ins]['kpcov'] = None
                                self.scidata[ind][self.ins]['kpicv'] = None

                        ofile = pathlib.Path(self.scidata[ind][self.ins]['fitsfile']).stem
                        ofile = os.path.join(self.odir, ofile + '_kp_sci_cal.pdf')
                        plot.plot_cal_kp(wave=wave,
                                         kp_sci=self.scidata_org[ind][self.ins]['kp'],
                                         dkp_sci=self.scidata_org[ind][self.ins]['dkp'],
                                         kpcov_sci=self.scidata_org[ind][self.ins]['kpcov'],
                                         kp_cal=kp_cal_master,
                                         dkp_cal=dkp_cal_master,
                                         kpcov_cal=kpcov_cal_master,
                                         kp_sci_cal=self.scidata[ind][self.ins]['kp'],
                                         dkp_sci_cal=self.scidata[ind][self.ins]['dkp'],
                                         kpcov_sci_cal=self.scidata[ind][self.ins]['kpcov'],
                                         ofile=ofile)

        pass

    def chi2map(self,
                rlim: Tuple[float],
                step: float,
                model: Optional[str] = 'bin',
                fixpos2grid: Optional[bool] = False,
                searchbox: Optional[dict] = None,
                fit_sub: Optional[dict] = None,
                cov: Optional[bool] = False,
                smear: Optional[int] = None,
                use_ins: Optional[list[str]] = None,
                ofile: Optional[str] = None,
                plotoi: Optional[bool] = True):
        """
        
        """

        print('#========================================')
        print('# CHI2MAP')
        print('#========================================')
        print('')

        if rlim[0] < 0. or rlim[1] < 0. or rlim[0] >= rlim[1]:
            raise UserWarning('Radius limit needs to be 0 <= rlim[0] < rlim[1]')

        if model == 'bin':
            if 'cp' not in self.obs and 'kp' not in self.obs:
                raise UserWarning('Need closure or kernel phases to fit binary model')
        elif model == 'ud_bin':
            if 'vis2' not in self.obs or ('cp' not in self.obs and 'kp' not in self.obs):
                raise UserWarning('Need visibility amplitude and closure or kernel phases to fit binary model')

        data_list = []
        inss = []
        for data in self.scidata:
            if use_ins is None:
                if self.ins in data.keys():
                    data_list += [deepcopy(data[self.ins])]
            else:
                for ins in use_ins:
                    if ins in data.keys():
                        inss += [ins]
                        data_list += [deepcopy(data[ins])]

        if fit_sub is not None:
            fit_inj = deepcopy(fit_sub)
            fit_inj['p'][0] *= -1.
            data_list = util.injec_companion(fit_inj=fit_inj,
                                             data_list=data_list,
                                             obs=self.obs)

        nstep = np.ceil(rlim[1] / step)
        xy = np.arange(-nstep * step, (nstep + 1) * step, step)
        grid_radec = np.meshgrid(xy, xy)
        grid_radec = (np.fliplr(grid_radec[0]), grid_radec[1])
        rr = np.sqrt(grid_radec[0]**2 + grid_radec[1]**2)
        ww = (rr < rlim[0]) | (rr > rlim[1])

        if not fixpos2grid:
            fact = 4.
            xy = np.arange(-nstep * step, (nstep + 1. / fact) * step, step / fact)
            grid_radec_fine = np.meshgrid(xy, xy)
            grid_radec_fine = (np.fliplr(grid_radec_fine[0]), grid_radec_fine[1])
            rr_fine = np.sqrt(grid_radec_fine[0]**2 + grid_radec_fine[1]**2)
            ww_fine = (rr_fine < rlim[0]) | (rr_fine > rlim[1])

        if model == 'bin' and fixpos2grid:
            chi2_ps = util.chi2_bin_c(c0=np.array([0.]),
                                      p0=np.array([0., 0.]),
                                      data_list=data_list,
                                      obs=self.obs,
                                      cov=cov,
                                      smear=smear)
            c0 = 1e-4
            p0s = []
            pps = []
            pes = []
            chi2s = []
            niter = []
            ctr = 0
            nc = np.prod(grid_radec[0].shape)
            for i in range(grid_radec[0].shape[0]):
                for j in range(grid_radec[0].shape[1]):
                    ctr += 1
                    if ctr % 10 == 0:
                        sys.stdout.write('\rCell %.0f of %.0f' % (ctr, nc))
                        sys.stdout.flush()
                    p0 = np.array([c0, grid_radec[0][i, j], grid_radec[1][i, j]])
                    p0s += [p0]
                    if ww[i, j]:
                        pps += [np.ones(3) * np.nan]
                        pes += [np.ones(3) * np.nan]
                        chi2s += [np.nan]
                        niter += [np.nan]
                        continue
                    pp = minimize(util.chi2_bin_c,
                                  p0[0:1],
                                  args=(p0[1:3], data_list, self.obs, cov, smear),
                                  method='L-BFGS-B',
                                  bounds=[(0., 1.)],
                                  tol=self.ftol,
                                  options={'maxiter': self.maxiter})
                    if not pp['success']:
                        pps += [np.ones(3) * np.nan]
                        pes += [np.ones(3) * np.nan]
                        chi2s += [np.nan]
                        niter += [np.inf]
                        continue
                    temp = p0.copy()
                    temp[0] = pp['x'][0]
                    pps += [temp]
                    temp = np.zeros(3)
                    temp[0] = np.sqrt(max(1., abs(pp['fun'])) * self.ftol * np.diag(pp['hess_inv'].todense()))
                    pes += [temp]
                    chi2s += [pp['fun']]
                    niter += [pp['nit']]
                    # c0s_temp = np.logspace(-6, 0, 100)
                    # chi2s_temp = [util.chi2bin_c(c0_temp, p0[1:3], data_list, self.obs, cov, smear) for c0_temp in c0s_temp]
                    # plt.plot(c0s_temp, chi2s_temp)
                    # plt.xscale('log')
                    # plt.yscale('log')
                    # plt.axvline(pps[-1][0])
                    # plt.show()
                    # pdb.set_trace()
            sys.stdout.write('\rCell %.0f of %.0f' % (nc, nc))
            print('')
            print('')
            p0s = np.array(p0s)
            pps = np.array(pps)
            pes = np.array(pes)
            chi2s = np.array(chi2s)
            niter = np.array(niter)

            is_in_lims = False
            temp = chi2s.copy()
            while not is_in_lims:
                best = np.nanargmin(temp)
                rho = np.sqrt(np.sum(pps[best][1:3]**2))
                if rlim[0] <= rho <= rlim[1]:
                    if searchbox is not None:
                        RA = pps[best][1]
                        Dec = pps[best][2]
                        if (searchbox['RA'][0] <= RA <= searchbox['RA'][1]) and (searchbox['Dec'][0] <= Dec <= searchbox['Dec'][1]):
                            is_in_lims = True
                        else:
                            temp[best] = np.inf
                    else:
                        is_in_lims = True
                else:
                    temp[best] = np.inf
            chi2, ndof = util.chi2_bin_c_ndof(c0=pps[best][0:1],
                                              p0=pps[best][1:3],
                                              data_list=data_list,
                                              obs=self.obs,
                                              cov=cov,
                                              smear=smear)
            nsigma, lnprob = util.nsigma(chi2r_test=chi2_ps / ndof,
                                         chi2r_true=chi2 / ndof,
                                         ndof=ndof,
                                         use_mpmath=False)
            fit = {}
            fit['model'] = model
            fit['p'] = pps[best]
            fit['dp'] = pes[best]
            fit['chi2_red'] = chi2 / ndof
            fit['ndof'] = ndof
            fit['lnprob'] = lnprob
            fit['nsigma'] = nsigma
            fit['smear'] = smear
            fit['cov'] = str(cov)
            fit['obs'] = self.obs
            fit['plot_grid'] = p0s.reshape((grid_radec[0].shape[0], grid_radec[0].shape[1], 3))
            fit['plot_chi2'] = chi2s.reshape(grid_radec[0].shape)
            fit['plot_niter'] = niter.reshape(grid_radec[0].shape)
            if ofile is None:
                ofile = self.obj
            ofile = os.path.join(self.odir, ofile + '_chi2map.pdf')
            plot.plot_chi2map(fit, ofile=ofile, searchbox=searchbox)

        elif model == 'bin' and not fixpos2grid:
            chi2_ps = util.chi2_bin(p0=np.array([0., 0., 0.]),
                                    data_list=data_list,
                                    obs=self.obs,
                                    cov=cov,
                                    smear=smear)
            c0 = 1e-4
            p0s = []
            pps = []
            pes = []
            chi2s = []
            niter = []
            ctr = 0
            nc = np.prod(grid_radec[0].shape)
            for i in range(grid_radec[0].shape[0]):
                for j in range(grid_radec[0].shape[1]):
                    ctr += 1
                    if ctr % 10 == 0:
                        sys.stdout.write('\rCell %.0f of %.0f' % (ctr, nc))
                        sys.stdout.flush()
                    p0 = np.array([c0, grid_radec[0][i, j], grid_radec[1][i, j]])
                    p0s += [p0]
                    if ww[i, j]:
                        pps += [np.ones(3) * np.nan]
                        pes += [np.ones(3) * np.nan]
                        chi2s += [np.nan]
                        niter += [np.nan]
                        continue
                    pp = minimize(util.chi2_bin,
                                  p0,
                                  args=(data_list, self.obs, cov, smear),
                                  method='L-BFGS-B',
                                  bounds=[(0., 1.), (-np.inf, np.inf), (-np.inf, np.inf)],
                                  tol=self.ftol,
                                  options={'maxiter': self.maxiter})
                    if not pp['success']:
                        pps += [np.ones(3) * np.nan]
                        pes += [np.ones(3) * np.nan]
                        chi2s += [np.nan]
                        niter += [np.inf]
                        continue
                    pps += [pp['x']]
                    pes += [np.sqrt(max(1., abs(pp['fun'])) * self.ftol * np.diag(pp['hess_inv'].todense()))]
                    chi2s += [pp['fun']]
                    niter += [pp['nit']]
            sys.stdout.write('\rCell %.0f of %.0f' % (nc, nc))
            print('')
            print('')
            p0s = np.array(p0s)
            pps = np.array(pps)
            pes = np.array(pes)
            chi2s = np.array(chi2s)
            niter = np.array(niter)

            ww = np.where(~np.isnan(niter) & ~np.isinf(niter))[0]
            pps_unique = [pps[ww[0]]]
            chi2s_unique = [chi2s[ww[0]]]
            dists_unique = []
            for i in range(1, len(ww)):
                diffs = np.array(pps_unique) - pps[ww[i]]
                dists = np.sqrt(np.sum(diffs[:, 1:3]**2, axis=1))
                if np.sum(dists < 1e-1) > 0:
                    continue
                pps_unique += [pps[ww[i]]]
                chi2s_unique += [chi2s[ww[i]]]
                dists_unique += [np.min(dists)]
            pps_unique = np.array(pps_unique)
            chi2s_unique = np.array(chi2s_unique)
            rbf = Rbf(pps_unique[:, 1], pps_unique[:, 2], chi2s_unique, function='linear')
            chi2s_fine = rbf(grid_radec_fine[0].flatten(), grid_radec_fine[1].flatten())
            chi2s_fine[ww_fine.flatten()] = np.nan

            is_in_lims = False
            temp = chi2s.copy()
            while not is_in_lims:
                best = np.nanargmin(temp)
                rho = np.sqrt(np.sum(pps[best][1:3]**2))
                if rlim[0] <= rho <= rlim[1]:
                    if searchbox is not None:
                        RA = pps[best][1]
                        Dec = pps[best][2]
                        if (searchbox['RA'][0] <= RA <= searchbox['RA'][1]) and (searchbox['Dec'][0] <= Dec <= searchbox['Dec'][1]):
                            is_in_lims = True
                        else:
                            temp[best] = np.inf
                    else:
                        is_in_lims = True
                else:
                    temp[best] = np.inf
            chi2, ndof = util.chi2_bin_ndof(p0=pps[best],
                                            data_list=data_list,
                                            obs=self.obs,
                                            cov=cov,
                                            smear=smear)
            nsigma, lnprob = util.nsigma(chi2r_test=chi2_ps / ndof,
                                         chi2r_true=chi2 / ndof,
                                         ndof=ndof,
                                         use_mpmath=False)
            fit = {}
            fit['model'] = model
            fit['p'] = pps[best]
            fit['dp'] = pes[best]
            fit['chi2_red'] = chi2 / ndof
            fit['ndof'] = ndof
            fit['lnprob'] = lnprob
            fit['nsigma'] = nsigma
            fit['smear'] = smear
            fit['cov'] = str(cov)
            fit['obs'] = self.obs
            fit['plot_grid'] = p0s.reshape((grid_radec[0].shape[0], grid_radec[0].shape[1], 3))
            fit['plot_chi2'] = chi2s.reshape(grid_radec[0].shape)
            fit['plot_niter'] = niter.reshape(grid_radec[0].shape)
            p0s_fine = np.array([np.ones_like(grid_radec_fine[0].flatten()) * c0, grid_radec_fine[0].flatten(), grid_radec_fine[1].flatten()]).T
            fit['plot_grid_fine'] = p0s_fine.reshape((grid_radec_fine[0].shape[0], grid_radec_fine[0].shape[1], 3))
            fit['plot_chi2_fine'] = chi2s_fine.reshape(grid_radec_fine[0].shape)
            if ofile is None:
                ofile = self.obj
            ofile = os.path.join(self.odir, ofile + '_chi2map.pdf')
            plot.plot_chi2map(fit, ofile=ofile, searchbox=searchbox)

        elif model == 'ud_bin' and fixpos2grid:
            p0s_temp = np.linspace(0., 10., 1000)
            chi2s_temp = [util.chi2_ud(np.array([p0]), data_list, self.obs, cov, smear) for p0 in p0s_temp]
            p0_ud = np.array([p0s_temp[np.argmin(chi2s_temp)]])
            pp_ud = minimize(util.chi2_ud, p0_ud, args=(data_list, self.obs, cov, smear), method='L-BFGS-B', bounds=[(0., np.inf)], tol=self.ftol, options={'maxiter': self.maxiter})
            chi2_ud = pp_ud['fun']
            c0 = 1e-4
            p0s = []
            pps = []
            pes = []
            chi2s = []
            niter = []
            ctr = 0
            nc = np.prod(grid_radec[0].shape)
            for i in range(grid_radec[0].shape[0]):
                for j in range(grid_radec[0].shape[1]):
                    ctr += 1
                    if ctr % 10 == 0:
                        sys.stdout.write('\rCell %.0f of %.0f' % (ctr, nc))
                        sys.stdout.flush()
                    p0 = np.array([c0, grid_radec[0][i, j], grid_radec[1][i, j], pp_ud['x'][0]])
                    p0s += [p0]
                    if ww[i, j]:
                        pps += [np.ones(4) * np.nan]
                        pes += [np.ones(4) * np.nan]
                        chi2s += [np.nan]
                        niter += [np.nan]
                        continue
                    pp = minimize(util.chi2_ud_bin_c,
                                  p0[[0, 3]],
                                  args=(p0[1:3], data_list, self.obs, cov, smear),
                                  method='L-BFGS-B',
                                  bounds=[(0., 1.), (0., np.inf)],
                                  tol=self.ftol,
                                  options={'maxiter': self.maxiter})
                    if not pp['success']:
                        pps += [np.ones(4) * np.nan]
                        pes += [np.ones(4) * np.nan]
                        chi2s += [np.nan]
                        niter += [np.inf]
                        continue
                    temp = p0.copy()
                    temp[0] = pp['x'][0]
                    temp[3] = pp['x'][1]
                    pps += [temp]
                    temp = np.zeros(4)
                    temp[[0, 3]] = np.sqrt(max(1., abs(pp['fun'])) * self.ftol * np.diag(pp['hess_inv'].todense()))
                    pes += [temp]
                    chi2s += [pp['fun']]
                    niter += [pp['nit']]
            sys.stdout.write('\rCell %.0f of %.0f' % (nc, nc))
            print('')
            print('')
            p0s = np.array(p0s)
            pps = np.array(pps)
            pes = np.array(pes)
            chi2s = np.array(chi2s)
            niter = np.array(niter)

            is_in_lims = False
            temp = chi2s.copy()
            while not is_in_lims:
                best = np.nanargmin(temp)
                rho = np.sqrt(np.sum(pps[best][1:3]**2))
                if rlim[0] <= rho <= rlim[1]:
                    if searchbox is not None:
                        RA = pps[best][1]
                        Dec = pps[best][2]
                        if (searchbox['RA'][0] <= RA <= searchbox['RA'][1]) and (searchbox['Dec'][0] <= Dec <= searchbox['Dec'][1]):
                            is_in_lims = True
                        else:
                            temp[best] = np.inf
                    else:
                        is_in_lims = True
                else:
                    temp[best] = np.inf
            chi2, ndof = util.chi2_ud_bin_c_ndof(c0=pps[best][[0, 3]],
                                                 p0=pps[best][1:3],
                                                 data_list=data_list,
                                                 obs=self.obs,
                                                 cov=cov,
                                                 smear=smear)
            nsigma, lnprob = util.nsigma(chi2r_test=chi2_ud / ndof,
                                         chi2r_true=chi2 / ndof,
                                         ndof=ndof,
                                         use_mpmath=False)
            fit = {}
            fit['model'] = model
            fit['p'] = pps[best]
            fit['dp'] = pes[best]
            fit['chi2_red'] = chi2 / ndof
            fit['ndof'] = ndof
            fit['lnprob'] = lnprob
            fit['nsigma'] = nsigma
            fit['smear'] = smear
            fit['cov'] = str(cov)
            fit['obs'] = self.obs
            fit['plot_grid'] = p0s.reshape((grid_radec[0].shape[0], grid_radec[0].shape[1], 4))
            fit['plot_chi2'] = chi2s.reshape(grid_radec[0].shape)
            fit['plot_niter'] = niter.reshape(grid_radec[0].shape)
            if ofile is None:
                ofile = self.obj
            ofile = os.path.join(self.odir, ofile + '_chi2map.pdf')
            plot.plot_chi2map(fit, ofile=ofile, searchbox=searchbox)

        elif model == 'ud_bin' and not fixpos2grid:
            p0s_temp = np.linspace(0., 10., 1000)
            chi2s_temp = [util.chi2_ud(np.array([p0]), data_list, self.obs, cov, smear) for p0 in p0s_temp]
            p0_ud = np.array([p0s_temp[np.argmin(chi2s_temp)]])
            pp_ud = minimize(util.chi2_ud, p0_ud, args=(data_list, self.obs, cov, smear), method='L-BFGS-B', bounds=[(0., np.inf)], tol=self.ftol, options={'maxiter': self.maxiter})
            chi2_ud = pp_ud['fun']
            c0 = 1e-4
            p0s = []
            pps = []
            pes = []
            chi2s = []
            niter = []
            ctr = 0
            nc = np.prod(grid_radec[0].shape)
            for i in range(grid_radec[0].shape[0]):
                for j in range(grid_radec[0].shape[1]):
                    ctr += 1
                    if ctr % 10 == 0:
                        sys.stdout.write('\rCell %.0f of %.0f' % (ctr, nc))
                        sys.stdout.flush()
                    p0 = np.array([c0, grid_radec[0][i, j], grid_radec[1][i, j], pp_ud['x'][0]])
                    p0s += [p0]
                    if ww[i, j]:
                        pps += [np.ones(4) * np.nan]
                        pes += [np.ones(4) * np.nan]
                        chi2s += [np.nan]
                        niter += [np.nan]
                        continue
                    pp = minimize(util.chi2_ud_bin,
                                  p0,
                                  args=(data_list, self.obs, cov, smear),
                                  method='L-BFGS-B',
                                  bounds=[(0., 1.), (-np.inf, np.inf), (-np.inf, np.inf), (0., np.inf)],
                                  tol=self.ftol,
                                  options={'maxiter': self.maxiter})
                    if not pp['success']:
                        pps += [np.ones(4) * np.nan]
                        pes += [np.ones(4) * np.nan]
                        chi2s += [np.nan]
                        niter += [np.inf]
                        continue
                    pps += [pp['x']]
                    pes += [np.sqrt(max(1., abs(pp['fun'])) * self.ftol * np.diag(pp['hess_inv'].todense()))]
                    chi2s += [pp['fun']]
                    niter += [pp['nit']]
            sys.stdout.write('\rCell %.0f of %.0f' % (nc, nc))
            print('')
            print('')
            p0s = np.array(p0s)
            pps = np.array(pps)
            pes = np.array(pes)
            chi2s = np.array(chi2s)
            niter = np.array(niter)

            ww = np.where(~np.isnan(niter) & ~np.isinf(niter))[0]
            pps_unique = [pps[ww[0]]]
            chi2s_unique = [chi2s[ww[0]]]
            dists_unique = []
            for i in range(1, len(ww)):
                diffs = np.array(pps_unique) - pps[ww[i]]
                dists = np.sqrt(np.sum(diffs[:, 1:3]**2, axis=1))
                if np.sum(dists < 1e-1) > 0:
                    continue
                pps_unique += [pps[ww[i]]]
                chi2s_unique += [chi2s[ww[i]]]
                dists_unique += [np.min(dists)]
            pps_unique = np.array(pps_unique)
            chi2s_unique = np.array(chi2s_unique)
            rbf = Rbf(pps_unique[:, 1], pps_unique[:, 2], chi2s_unique, function='linear')
            chi2s_fine = rbf(grid_radec_fine[0].flatten(), grid_radec_fine[1].flatten())
            chi2s_fine[ww_fine.flatten()] = np.nan

            is_in_lims = False
            temp = chi2s.copy()
            while not is_in_lims:
                best = np.nanargmin(temp)
                rho = np.sqrt(np.sum(pps[best][1:3]**2))
                if rlim[0] <= rho <= rlim[1]:
                    if searchbox is not None:
                        RA = pps[best][1]
                        Dec = pps[best][2]
                        if (searchbox['RA'][0] <= RA <= searchbox['RA'][1]) and (searchbox['Dec'][0] <= Dec <= searchbox['Dec'][1]):
                            is_in_lims = True
                        else:
                            temp[best] = np.inf
                    else:
                        is_in_lims = True
                else:
                    temp[best] = np.inf
            chi2, ndof = util.chi2_ud_bin_ndof(p0=pps[best],
                                               data_list=data_list,
                                               obs=self.obs,
                                               cov=cov,
                                               smear=smear)
            nsigma, lnprob = util.nsigma(chi2r_test=chi2_ud / ndof,
                                         chi2r_true=chi2 / ndof,
                                         ndof=ndof,
                                         use_mpmath=False)
            fit = {}
            fit['model'] = model
            fit['p'] = pps[best]
            fit['dp'] = pes[best]
            fit['chi2_red'] = chi2 / ndof
            fit['ndof'] = ndof
            fit['lnprob'] = lnprob
            fit['nsigma'] = nsigma
            fit['smear'] = smear
            fit['cov'] = str(cov)
            fit['obs'] = self.obs
            fit['plot_grid'] = p0s.reshape((grid_radec[0].shape[0], grid_radec[0].shape[1], 4))
            fit['plot_chi2'] = chi2s.reshape(grid_radec[0].shape)
            fit['plot_niter'] = niter.reshape(grid_radec[0].shape)
            p0s_fine = np.array([np.ones_like(grid_radec_fine[0].flatten()) * c0, grid_radec_fine[0].flatten(), grid_radec_fine[1].flatten(), np.ones_like(grid_radec_fine[0].flatten()) * pp_ud['x'][0]]).T
            fit['plot_grid_fine'] = p0s_fine.reshape((grid_radec_fine[0].shape[0], grid_radec_fine[0].shape[1], 4))
            fit['plot_chi2_fine'] = chi2s_fine.reshape(grid_radec_fine[0].shape)
            if ofile is None:
                ofile = self.obj
            ofile = os.path.join(self.odir, ofile + '_chi2map.pdf')
            plot.plot_chi2map(fit, ofile=ofile, searchbox=searchbox)

        else:
            raise UserWarning()

        if plotoi:
            for i, data in enumerate(data_list):
                ofile_temp = deepcopy(ofile)
                temp = {}
                if use_ins is None:
                    temp[self.ins] = data
                else:
                    temp[inss[i]] = data
                ofile_temp = ofile_temp.replace('_chi2map', '_chi2map_' + pathlib.Path(data['fitsfile']).stem)
                plot.plot_oidata(data=temp, ofile=ofile_temp, fit=fit)

        return fit

    def mcmc(self,
             fit: dict,
             temperature: Optional[float] = 1.,
             n_live_points: Optional[int] = 400,
             fixpos: Optional[bool] = False,
             fit_sub: Optional[bool] = None,
             use_ins: Optional[list[str]] = None,
             ofile: Optional[str] = None,
             plotoi: Optional[bool] = True):
        """
        
        """

        print('#========================================')
        print('# MCMC')
        print('#========================================')
        print('')

        data_list = []
        inss = []
        for data in self.scidata:
            if use_ins is None:
                if self.ins in data.keys():
                    data_list += [deepcopy(data[self.ins])]
            else:
                for ins in use_ins:
                    if ins in data.keys():
                        inss += [ins]
                        data_list += [deepcopy(data[ins])]

        if fit_sub is not None:
            fit_inj = deepcopy(fit_sub)
            fit_inj['p'][0] *= -1.
            data_list = util.injec_companion(fit_inj=fit_inj,
                                             data_list=data_list,
                                             obs=self.obs)

        try:
            from mpi4py import MPI
            mpi_rank = MPI.COMM_WORLD.Get_rank()
        except ModuleNotFoundError:
            mpi_rank = 0

        odir = './multinest/'
        if mpi_rank == 0 and not os.path.exists(odir):
            os.makedirs(odir)

        prior_bounds = []
        if fit['model'] == 'bin' or fit['model'] == 'ud_bin':
            for i, item in enumerate(fit['p']):
                if i == 0:
                    prior_bounds.append((0., 1.))  # contrast
                elif i == 1:
                    if not fixpos:
                        prior_bounds.append((item - 3. * fit['dp'][i], item + 3. * fit['dp'][i]))  # RA
                elif i == 2:
                    if not fixpos:
                        prior_bounds.append((item - 3. * fit['dp'][i], item + 3. * fit['dp'][i]))  # Dec
                elif i == 3:
                    prior_bounds.append((0., 10.))  # UD diam
        elif fit['model'] == 'tri':
            for i, item in enumerate(fit['p']):
                if i == 0 or i == 3:
                    prior_bounds.append((0., 1.))  # contrast
                elif i == 1 or i == 4:
                    if not fixpos:
                        prior_bounds.append((item - 3. * fit['dp'][i], item + 3. * fit['dp'][i]))  # RA
                elif i == 2 or i == 5:
                    if not fixpos:
                        prior_bounds.append((item - 3. * fit['dp'][i], item + 3. * fit['dp'][i]))  # Dec
        else:
            raise UserWarning()

        def lnprior_multinest(cube, n_dim, n_param):
            """
            
            """

            for i in range(n_param):
                cube[i] = (prior_bounds[i][0] + (prior_bounds[i][1] - prior_bounds[i][0]) * cube[i])

            return cube

        def lnprob_multinest(cube, n_dim, n_param) -> np.float64:
            """
            
            """

            params = np.zeros(n_param)
            for i in range(n_param):
                params[i] = cube[i]

            if fit['model'] == 'bin':
                if fixpos:
                    temp = params.copy()
                    params = np.zeros(n_param + 2)
                    params[0] = temp[0]
                    params[1] = fit['p'][1]
                    params[2] = fit['p'][2]

                return util.lnprob_bin(p0=params, data_list=data_list, obs=fit['obs'], cov=fit['cov'] == 'True', smear=fit['smear'], temperature=temperature)

            elif fit['model'] == 'tri':
                if fixpos:
                    temp = params.copy()
                    params = np.zeros(n_param + 4)
                    params[0] = temp[0]
                    params[1] = fit['p'][1]
                    params[2] = fit['p'][2]
                    params[3] = temp[1]
                    params[4] = fit['p'][4]
                    params[5] = fit['p'][5]

                return util.lnprob_tri(p0=params, data_list=data_list, obs=fit['obs'], cov=fit['cov'] == 'True', smear=fit['smear'], temperature=temperature)

            elif fit['model'] == 'ud_bin':
                if fixpos:
                    temp = params.copy()
                    params = np.zeros(n_param + 2)
                    params[0] = temp[0]
                    params[1] = fit['p'][1]
                    params[2] = fit['p'][2]
                    params[3] = temp[1]

                return util.lnprob_ud_bin(p0=params, data_list=data_list, obs=fit['obs'], cov=fit['cov'] == 'True', smear=fit['smear'], temperature=temperature)

            else:
                raise UserWarning()

        pymultinest.run(lnprob_multinest,
                        lnprior_multinest,
                        len(prior_bounds),
                        outputfiles_basename=odir,
                        resume=False,
                        n_live_points=n_live_points)

        analyzer = pymultinest.analyse.Analyzer(len(prior_bounds),
                                                outputfiles_basename=odir)
        sampling_stats = analyzer.get_stats()
        samples = analyzer.get_equal_weighted_posterior()

        ln_z = sampling_stats['nested importance sampling global log-evidence']
        ln_z_error = sampling_stats['nested importance sampling global log-evidence error']
        print('')
        print(f'Log-evidence: {ln_z:.2f} +/- {ln_z_error:.2f}')
        print('')

        ln_prob = samples[:, -1]
        samples = samples[:, :-1]
        if fixpos:
            if fit['model'] == 'bin':
                temp = samples.copy()
                samples = np.zeros((temp.shape[0], temp.shape[1] + 2))
                samples[:, 0] = temp[:, 0]
                samples[:, 1] = fit['p'][1]
                samples[:, 2] = fit['p'][2]
            elif fit['model'] == 'tri':
                temp = samples.copy()
                samples = np.zeros((temp.shape[0], temp.shape[1] + 4))
                samples[:, 0] = temp[:, 0]
                samples[:, 1] = fit['p'][1]
                samples[:, 2] = fit['p'][2]
                samples[:, 3] = temp[:, 1]
                samples[:, 4] = fit['p'][4]
                samples[:, 5] = fit['p'][5]
            elif fit['model'] == 'ud_bin':
                temp = samples.copy()
                samples = np.zeros((temp.shape[0], temp.shape[1] + 2))
                samples[:, 0] = temp[:, 0]
                samples[:, 1] = fit['p'][1]
                samples[:, 2] = fit['p'][2]
                samples[:, 3] = temp[:, 1]
            else:
                raise UserWarning()
        pp = np.percentile(samples, 50., axis=0)
        pu = np.percentile(samples, 84., axis=0) - pp
        pl = pp - np.percentile(samples, 16., axis=0)
        pe = np.mean(np.vstack((pu, pl)), axis=0)

        if fit['model'] == 'bin':
            chi2_ps = util.chi2_bin(p0=np.array([0., 0., 0.]),
                                    data_list=data_list,
                                    obs=fit['obs'],
                                    cov=fit['cov'] == 'True',
                                    smear=fit['smear'])
            chi2, ndof = util.chi2_bin_ndof(p0=pp,
                                            data_list=data_list,
                                            obs=fit['obs'],
                                            cov=fit['cov'] == 'True',
                                            smear=fit['smear'])
            nsigma, lnprob = util.nsigma(chi2r_test=chi2_ps / ndof,
                                         chi2r_true=chi2 / ndof,
                                         ndof=ndof,
                                         use_mpmath=False)
        elif fit['model'] == 'tri':
            chi2_ps = util.chi2_tri(p0=np.array([0., 0., 0., 0., 0., 0.]),
                                    data_list=data_list,
                                    obs=fit['obs'],
                                    cov=fit['cov'] == 'True',
                                    smear=fit['smear'])
            chi2, ndof = util.chi2_tri_ndof(p0=pp,
                                            data_list=data_list,
                                            obs=fit['obs'],
                                            cov=fit['cov'] == 'True',
                                            smear=fit['smear'])
            nsigma, lnprob = util.nsigma(chi2r_test=chi2_ps / ndof,
                                         chi2r_true=chi2 / ndof,
                                         ndof=ndof,
                                         use_mpmath=False)
        elif fit['model'] == 'ud_bin':
            p0s_temp = np.linspace(0., 10., 1000)
            chi2s_temp = [util.chi2_ud(np.array([p0]), data_list, fit['obs'], fit['cov'] == 'True', fit['smear']) for p0 in p0s_temp]
            p0_ud = np.array([p0s_temp[np.argmin(chi2s_temp)]])
            pp_ud = minimize(util.chi2_ud, p0_ud, args=(data_list, fit['obs'], fit['cov'] == 'True', fit['smear']), method='L-BFGS-B', bounds=[(0., np.inf)], tol=self.ftol, options={'maxiter': self.maxiter})
            chi2_ud = pp_ud['fun']
            chi2, ndof = util.chi2_ud_bin_ndof(p0=pp,
                                               data_list=data_list,
                                               obs=fit['obs'],
                                               cov=fit['cov'] == 'True',
                                               smear=fit['smear'])
            nsigma, lnprob = util.nsigma(chi2r_test=chi2_ud / ndof,
                                         chi2r_true=chi2 / ndof,
                                         ndof=ndof,
                                         use_mpmath=False)
        else:
            raise UserWarning()
        fit = deepcopy(fit)
        fit['p'] = pp
        fit['dp'] = pe
        fit['chi2_red'] = chi2 / ndof
        fit['ndof'] = ndof
        fit['lnprob'] = lnprob
        fit['nsigma'] = nsigma
        if ofile is None:
            ofile = self.obj
        ofile = os.path.join(self.odir, ofile + '_chains.pdf')
        plot.plot_chains(fit, samples, ofile)
        ofile = ofile.replace('_chains.pdf', '_corner.pdf')
        plot.plot_corner(fit, samples, ofile, fixpos=fixpos)

        if plotoi:
            for data in data_list:
                ofile_temp = deepcopy(ofile)
                temp = {}
                if use_ins is None:
                    temp[self.ins] = data
                else:
                    temp[inss[i]] = data
                ofile_temp = ofile_temp.replace('_chi2map', '_chi2map_' + pathlib.Path(data['fitsfile']).stem)
                plot.plot_oidata(data=temp, ofile=ofile_temp, fit=fit)

        return fit

    def __lim_absil(self,
                    c0,
                    func,
                    p0,
                    data_list,
                    obs,
                    cov,
                    smear,
                    chi2_true,
                    ndof,
                    sigma=5.):
        """
        
        """

        if c0 <= 0.:

            return np.inf

        else:
            pp = p0.copy()
            pp[0] = c0
            chi2_test = func(p0=pp,
                             data_list=data_list,
                             obs=obs,
                             cov=cov,
                             smear=smear)
            nsigma, _ = util.nsigma(chi2r_test=chi2_test / ndof,
                                    chi2r_true=chi2_true / ndof,
                                    ndof=ndof)

            return np.abs(nsigma - sigma)**2

    def __lim_injec(self,
                    c0,
                    fit_inj,
                    data_list,
                    obs,
                    cov,
                    smear,
                    ndof,
                    sigma=5.,
                    thetap=None):
        """
        
        """

        if c0 <= 0.:

            return np.inf

        else:
            fit_inj_copy = deepcopy(fit_inj)
            fit_inj_copy['p'][0] = c0
            data_list_copy = deepcopy(data_list)
            data_list_copy = util.injec_companion(fit_inj=fit_inj_copy,
                                                  data_list=data_list_copy,
                                                  obs=obs)
            if fit_inj['model'] == 'bin':
                chi2_test = util.chi2_bin(p0=np.array([0., 0., 0.]),
                                          data_list=data_list_copy,
                                          obs=obs,
                                          cov=cov,
                                          smear=smear)
                chi2_true = util.chi2_bin(p0=fit_inj_copy['p'],
                                          data_list=data_list_copy,
                                          obs=obs,
                                          cov=cov,
                                          smear=smear)
            elif fit_inj['model'] == 'ud_bin':
                pp_test = minimize(util.chi2_ud,
                                   np.array([thetap]),
                                   args=(data_list_copy, obs, cov, smear),
                                   method='L-BFGS-B',
                                   bounds=[(0., np.inf)],
                                   tol=self.ftol,
                                   options={'maxiter': self.maxiter})
                chi2_test = pp_test['fun']
                fit_inj_copy['p'][3] = thetap
                chi2_true = util.chi2_ud_bin(p0=fit_inj_copy['p'],
                                             data_list=data_list_copy,
                                             obs=obs,
                                             cov=cov,
                                             smear=smear)
            else:
                raise UserWarning()
            nsigma, _ = util.nsigma(chi2r_test=chi2_test / ndof,
                                    chi2r_true=chi2_true / ndof,
                                    ndof=ndof)

            return np.abs(nsigma - sigma)**2

    def detlim(self,
               rlim: Tuple[float],
               step: float,
               model: Optional[str] = 'bin',
               fit_sub: Optional[dict] = None,
               cov: Optional[bool] = False,
               smear: Optional[int] = None,
               sigma: Optional[int] = 5,
               cmin: Optional[float] = 1e-6,
               use_ins: Optional[list[str]] = None,
               ofile: Optional[str] = None):
        """
        
        """

        print('#========================================')
        print('# DETLIM')
        print('#========================================')
        print('')

        if rlim[0] < 0. or rlim[1] < 0. or rlim[0] >= rlim[1]:
            raise UserWarning('Radius limit needs to be 0 <= rlim[0] < rlim[1]')

        if model == 'bin':
            if 'cp' not in self.obs and 'kp' not in self.obs:
                raise UserWarning('Need closure or kernel phases to fit binary model')
        elif model == 'ud_bin':
            if 'vis2' not in self.obs or ('cp' not in self.obs and 'kp' not in self.obs):
                raise UserWarning('Need visibility amplitude and closure or kernel phases to fit binary model')

        data_list = []
        inss = []
        for data in self.scidata:
            if use_ins is None:
                if self.ins in data.keys():
                    data_list += [deepcopy(data[self.ins])]
            else:
                for ins in use_ins:
                    if ins in data.keys():
                        inss += [ins]
                        data_list += [deepcopy(data[ins])]

        if fit_sub is not None:
            fit_inj = deepcopy(fit_sub)
            fit_inj['p'][0] *= -1.
            data_list = util.injec_companion(fit_inj=fit_inj,
                                             data_list=data_list,
                                             obs=self.obs)

        nstep = np.ceil(rlim[1] / step)
        xy = np.arange(-nstep * step, (nstep + 1) * step, step)
        grid_radec = np.meshgrid(xy, xy)
        grid_radec = (np.fliplr(grid_radec[0]), grid_radec[1])
        rr = np.sqrt(grid_radec[0]**2 + grid_radec[1]**2)
        ww = (rr < rlim[0]) | (rr > rlim[1])

        if model == 'bin':
            chi2_ps, ndof = util.chi2_bin_c_ndof(c0=np.array([0.]),
                                                 p0=np.array([0., 0.]),
                                                 data_list=data_list,
                                                 obs=self.obs,
                                                 cov=cov,
                                                 smear=smear)
            c0s = np.logspace(np.log10(cmin), 0., 200)
            c0 = 1e-4
            p0s = []
            pps_absil = []
            pes_absil = []
            chi2s_absil = []
            niter_absil = []
            pps_injec = []
            pes_injec = []
            chi2s_injec = []
            niter_injec = []
            ctr = 0
            nc = np.prod(grid_radec[0].shape)
            for i in range(grid_radec[0].shape[0]):
                for j in range(grid_radec[0].shape[1]):
                    ctr += 1
                    if ctr % 10 == 0:
                        sys.stdout.write('\rCell %.0f of %.0f' % (ctr, nc))
                        sys.stdout.flush()
                    p0 = np.array([c0, grid_radec[0][i, j], grid_radec[1][i, j]])
                    p0s += [p0]
                    if ww[i, j]:
                        pps_absil += [np.ones(3) * np.nan]
                        pes_absil += [np.ones(3) * np.nan]
                        chi2s_absil += [np.nan]
                        niter_absil += [np.nan]
                        pps_injec += [np.ones(3) * np.nan]
                        pes_injec += [np.ones(3) * np.nan]
                        chi2s_injec += [np.nan]
                        niter_injec += [np.nan]
                        continue

                    chi2s_temp = [self.__lim_absil(c0_temp, util.chi2_bin, p0, data_list, self.obs, cov, smear, chi2_ps, ndof, sigma) for c0_temp in c0s]
                    chi2s_temp = np.array(chi2s_temp)
                    c0_temp = c0s[max(np.argmin(chi2s_temp) - 1, 0)]
                    pp = minimize(self.__lim_absil,
                                  c0_temp,
                                  args=(util.chi2_bin, p0, data_list, self.obs, cov, smear, chi2_ps, ndof, sigma),
                                  method='L-BFGS-B',
                                  bounds=[(c0s[max(np.argmin(chi2s_temp) - 1, 0)], c0s[min(np.argmin(chi2s_temp) + 1, len(c0s) - 1)])],
                                  tol=1e-5,
                                  options={'maxiter': self.maxiter})
                    if not pp['success']:
                        pps_absil += [np.ones(3) * np.nan]
                        pes_absil += [np.ones(3) * np.nan]
                        chi2s_absil += [np.nan]
                        niter_absil += [np.inf]
                    else:
                        temp = p0.copy()
                        temp[0] = pp['x']
                        pps_absil += [temp]
                        temp = np.zeros(3)
                        temp[0] = np.sqrt(max(1., abs(pp['fun'])) * self.ftol * np.diag(pp['hess_inv'].todense()))
                        pes_absil += [temp]
                        chi2s_absil += [pp['fun']]
                        niter_absil += [pp['nit']]

                    fit_inj = {'p': p0,
                               'model': model,
                               'cov': cov,
                               'smear': smear}
                    chi2s_temp = [self.__lim_injec(c0_temp, fit_inj, data_list, self.obs, cov, smear, ndof, sigma) for c0_temp in c0s]
                    chi2s_temp = np.array(chi2s_temp)
                    c0_temp = c0s[max(np.argmin(chi2s_temp) - 1, 0)]
                    pp = minimize(self.__lim_injec,
                                  c0_temp,
                                  args=(fit_inj, data_list, self.obs, cov, smear, ndof, sigma),
                                  method='L-BFGS-B',
                                  bounds=[(c0s[max(np.argmin(chi2s_temp) - 1, 0)], c0s[min(np.argmin(chi2s_temp) + 1, len(c0s) - 1)])],
                                  tol=1e-5,
                                  options={'maxiter': self.maxiter})
                    if not pp['success']:
                        pps_injec += [np.ones(3) * np.nan]
                        pes_injec += [np.ones(3) * np.nan]
                        chi2s_injec += [np.nan]
                        niter_injec += [np.inf]
                    else:
                        temp = p0.copy()
                        temp[0] = pp['x']
                        pps_injec += [temp]
                        temp = np.zeros(3)
                        temp[0] = np.sqrt(max(1., abs(pp['fun'])) * self.ftol * np.diag(pp['hess_inv'].todense()))
                        pes_injec += [temp]
                        chi2s_injec += [pp['fun']]
                        niter_injec += [pp['nit']]

            sys.stdout.write('\rCell %.0f of %.0f' % (nc, nc))
            print('')
            print('')
            p0s = np.array(p0s)
            pps_absil = np.array(pps_absil)
            pes_absil = np.array(pes_absil)
            chi2s_absil = np.array(chi2s_absil)
            niter_absil = np.array(niter_absil)
            pps_injec = np.array(pps_injec)
            pes_injec = np.array(pes_injec)
            chi2s_injec = np.array(chi2s_injec)
            niter_injec = np.array(niter_injec)

            chi2s_absil_med = np.nanmedian(chi2s_absil)
            chi2s_absil_p95 = np.nanpercentile(chi2s_absil, 95.)
            ww_chi2_absil_bad = chi2s_absil > (chi2s_absil_med + 3. * (chi2s_absil_p95 - chi2s_absil_med))
            pes_absil_med = np.nanmedian(pes_absil[:, 0])
            pes_absil_p95 = np.nanpercentile(pes_absil[:, 0], 95.)
            ww_pe_absil_bad = pes_absil[:, 0] > (pes_absil_med + 3. * (pes_absil_p95 - pes_absil_med))
            ww_absil_bad = ww_chi2_absil_bad | ww_pe_absil_bad
            niter_absil[ww_absil_bad] = np.inf
            chi2s_injec_med = np.nanmedian(chi2s_injec)
            chi2s_injec_p95 = np.nanpercentile(chi2s_injec, 95.)
            ww_chi2_injec_bad = chi2s_injec > (chi2s_injec_med + 3. * (chi2s_injec_p95 - chi2s_injec_med))
            pes_injec_med = np.nanmedian(pes_injec[:, 0])
            pes_injec_p95 = np.nanpercentile(pes_injec[:, 0], 95.)
            ww_pe_injec_bad = pes_injec[:, 0] > (pes_injec_med + 3. * (pes_injec_p95 - pes_injec_med))
            ww_injec_bad = ww_chi2_injec_bad | ww_pe_injec_bad
            niter_injec[ww_injec_bad] = np.inf

            p0s = p0s.reshape((grid_radec[0].shape[0], grid_radec[0].shape[1], p0s.shape[1]))
            pps_absil = pps_absil.reshape((grid_radec[0].shape[0], grid_radec[0].shape[1], pps_absil.shape[1]))
            niter_absil = niter_absil.reshape((grid_radec[0].shape[0], grid_radec[0].shape[1]))
            pps_injec = pps_injec.reshape((grid_radec[0].shape[0], grid_radec[0].shape[1], pps_injec.shape[1]))
            niter_injec = niter_injec.reshape((grid_radec[0].shape[0], grid_radec[0].shape[1]))
            if ofile is None:
                ofile = self.obj
            ofile = os.path.join(self.odir, ofile + '_detlim.pdf')
            plot.plot_detlim(p0s=p0s,
                             pps_absil=pps_absil,
                             niter_absil=niter_absil,
                             pps_injec=pps_injec,
                             niter_injec=niter_injec,
                             obs=self.obs,
                             cov=cov,
                             sigma=sigma,
                             ofile=ofile)

        elif model == 'ud_bin':
            p0s_temp = np.linspace(0., 10., 1000)
            chi2s_temp = [util.chi2_ud(np.array([p0]), data_list, self.obs, cov, smear) for p0 in p0s_temp]
            p0_ud = np.array([p0s_temp[np.argmin(chi2s_temp)]])
            pp_ud = minimize(util.chi2_ud, p0_ud, args=(data_list, self.obs, cov, smear), method='L-BFGS-B', bounds=[(0., np.inf)], tol=self.ftol, options={'maxiter': self.maxiter})
            chi2_ud, ndof = util.chi2_ud_ndof(p0=pp_ud['x'],
                                              data_list=data_list,
                                              obs=self.obs,
                                              cov=cov,
                                              smear=smear)
            c0s = np.logspace(np.log10(cmin), 0., 200)
            c0 = 1e-4
            p0s = []
            pps_absil = []
            pes_absil = []
            chi2s_absil = []
            niter_absil = []
            pps_injec = []
            pes_injec = []
            chi2s_injec = []
            niter_injec = []
            ctr = 0
            nc = np.prod(grid_radec[0].shape)
            for i in range(grid_radec[0].shape[0]):
                for j in range(grid_radec[0].shape[1]):
                    ctr += 1
                    if ctr % 10 == 0:
                        sys.stdout.write('\rCell %.0f of %.0f' % (ctr, nc))
                        sys.stdout.flush()
                    p0 = np.array([c0, grid_radec[0][i, j], grid_radec[1][i, j], pp_ud['x'][0]])
                    p0s += [p0]
                    if ww[i, j]:
                        pps_absil += [np.ones(4) * np.nan]
                        pes_absil += [np.ones(4) * np.nan]
                        chi2s_absil += [np.nan]
                        niter_absil += [np.nan]
                        pps_injec += [np.ones(4) * np.nan]
                        pes_injec += [np.ones(4) * np.nan]
                        chi2s_injec += [np.nan]
                        niter_injec += [np.nan]
                        continue
                    chi2s_temp = [self.__lim_absil(c0_temp, util.chi2_ud_bin, p0, data_list, self.obs, cov, smear, chi2_ud, ndof, sigma) for c0_temp in c0s]
                    chi2s_temp = np.array(chi2s_temp)
                    c0_temp = c0s[max(np.argmin(chi2s_temp) - 1, 0)]
                    pp = minimize(self.__lim_absil,
                                  c0_temp,
                                  args=(util.chi2_ud_bin, p0, data_list, self.obs, cov, smear, chi2_ud, ndof, sigma),
                                  method='L-BFGS-B',
                                  bounds=[(c0s[max(np.argmin(chi2s_temp) - 1, 0)], c0s[min(np.argmin(chi2s_temp) + 1, len(c0s) - 1)])],
                                  tol=1e-5,
                                  options={'maxiter': self.maxiter})
                    if not pp['success']:
                        pps_absil += [np.ones(4) * np.nan]
                        pes_absil += [np.ones(4) * np.nan]
                        chi2s_absil += [np.nan]
                        niter_absil += [np.inf]
                    else:
                        temp = p0.copy()
                        temp[0] = pp['x']
                        pps_absil += [temp]
                        temp = np.zeros(4)
                        temp[0] = np.sqrt(max(1., abs(pp['fun'])) * self.ftol * np.diag(pp['hess_inv'].todense()))
                        pes_absil += [temp]
                        chi2s_absil += [pp['fun']]
                        niter_absil += [pp['nit']]

                    fit_inj = {'p': p0,
                               'model': model,
                               'cov': cov,
                               'smear': smear}
                    chi2s_temp = [self.__lim_injec(c0_temp, fit_inj, data_list, self.obs, cov, smear, ndof, sigma, pp_ud['x'][0]) for c0_temp in c0s]
                    chi2s_temp = np.array(chi2s_temp)
                    c0_temp = c0s[max(np.argmin(chi2s_temp) - 1, 0)]
                    pp = minimize(self.__lim_injec,
                                  c0_temp,
                                  args=(fit_inj, data_list, self.obs, cov, smear, ndof, sigma, pp_ud['x'][0]),
                                  method='L-BFGS-B',
                                  bounds=[(c0s[max(np.argmin(chi2s_temp) - 1, 0)], c0s[min(np.argmin(chi2s_temp) + 1, len(c0s) - 1)])],
                                  tol=1e-5,
                                  options={'maxiter': self.maxiter})
                    if not pp['success']:
                        pps_injec += [np.ones(4) * np.nan]
                        pes_injec += [np.ones(4) * np.nan]
                        chi2s_injec += [np.nan]
                        niter_injec += [np.inf]
                    else:
                        temp = p0.copy()
                        temp[0] = pp['x']
                        pps_injec += [temp]
                        temp = np.zeros(4)
                        temp[0] = np.sqrt(max(1., abs(pp['fun'])) * self.ftol * np.diag(pp['hess_inv'].todense()))
                        pes_injec += [temp]
                        chi2s_injec += [pp['fun']]
                        niter_injec += [pp['nit']]

            sys.stdout.write('\rCell %.0f of %.0f' % (nc, nc))
            print('')
            print('')
            p0s = np.array(p0s)
            pps_absil = np.array(pps_absil)
            pes_absil = np.array(pes_absil)
            chi2s_absil = np.array(chi2s_absil)
            niter_absil = np.array(niter_absil)
            pps_injec = np.array(pps_injec)
            pes_injec = np.array(pes_injec)
            chi2s_injec = np.array(chi2s_injec)
            niter_injec = np.array(niter_injec)

            chi2s_absil_med = np.nanmedian(chi2s_absil)
            chi2s_absil_p95 = np.nanpercentile(chi2s_absil, 95.)
            ww_chi2_absil_bad = chi2s_absil > (chi2s_absil_med + 3. * (chi2s_absil_p95 - chi2s_absil_med))
            pes_absil_med = np.nanmedian(pes_absil[:, 0])
            pes_absil_p95 = np.nanpercentile(pes_absil[:, 0], 95.)
            ww_pe_absil_bad = pes_absil[:, 0] > (pes_absil_med + 3. * (pes_absil_p95 - pes_absil_med))
            ww_absil_bad = ww_chi2_absil_bad | ww_pe_absil_bad
            niter_absil[ww_absil_bad] = np.inf
            chi2s_injec_med = np.nanmedian(chi2s_injec)
            chi2s_injec_p95 = np.nanpercentile(chi2s_injec, 95.)
            ww_chi2_injec_bad = chi2s_injec > (chi2s_injec_med + 3. * (chi2s_injec_p95 - chi2s_injec_med))
            pes_injec_med = np.nanmedian(pes_injec[:, 0])
            pes_injec_p95 = np.nanpercentile(pes_injec[:, 0], 95.)
            ww_pe_injec_bad = pes_injec[:, 0] > (pes_injec_med + 3. * (pes_injec_p95 - pes_injec_med))
            ww_injec_bad = ww_chi2_injec_bad | ww_pe_injec_bad
            niter_injec[ww_injec_bad] = np.inf

            p0s = p0s.reshape((grid_radec[0].shape[0], grid_radec[0].shape[1], p0s.shape[1]))
            pps_absil = pps_absil.reshape((grid_radec[0].shape[0], grid_radec[0].shape[1], pps_absil.shape[1]))
            niter_absil = niter_absil.reshape((grid_radec[0].shape[0], grid_radec[0].shape[1]))
            pps_injec = pps_injec.reshape((grid_radec[0].shape[0], grid_radec[0].shape[1], pps_injec.shape[1]))
            niter_injec = niter_injec.reshape((grid_radec[0].shape[0], grid_radec[0].shape[1]))
            if ofile is None:
                ofile = self.obj
            ofile = os.path.join(self.odir, ofile + '_detlim.pdf')
            plot.plot_detlim(p0s=p0s,
                             pps_absil=pps_absil,
                             niter_absil=niter_absil,
                             pps_injec=pps_injec,
                             niter_injec=niter_injec,
                             obs=self.obs,
                             cov=cov,
                             sigma=sigma,
                             ofile=ofile)

        else:
            raise UserWarning()

        pass
