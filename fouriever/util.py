from __future__ import division

import os
import pdb
import sys

import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import numpy as np

# import mpmath

from scipy import stats
from scipy.special import j1

import matplotlib
matplotlib.rcParams.update({'font.size': 14})


# =============================================================================
# MAIN
# =============================================================================

rad2mas = 180. / np.pi * 3600. * 1000.
mas2rad = np.pi / 180. / 3600. / 1000.
pa_mtoc = '-'


def nsigma(chi2r_test,
           chi2r_true,
           ndof,
           use_mpmath=False):
    """
    
    """

    if not use_mpmath:
        bin_prob = stats.chi2.cdf(ndof * chi2r_test / chi2r_true, ndof)
        log_bin_prob = np.log(bin_prob)

    else:
        mpmath.mp.dps = 50

        def chi2_cdf(x, k):
            x, k = mpmath.mpf(x), mpmath.mpf(k)
            return mpmath.gammainc(k / 2., 0., x / 2., regularized=True)

        bin_prob = chi2_cdf(ndof * chi2r_test / chi2r_true, float(ndof))
        log_bin_prob = float(mpmath.log(bin_prob))
        bin_prob = float(bin_prob)

    nsigma = np.sqrt(stats.chi2.ppf(bin_prob, 1.))
    if bin_prob > 1. - 1e-15:
        nsigma = np.sqrt(stats.chi2.ppf(1. - 1e-15, 1.))

    return nsigma, log_bin_prob


def vis2vis2(vis,
             data):
    """
    
    """

    if data['klflag']:
        return np.abs(np.matmul(data['v2mat'], vis))**2
    else:
        return np.abs(vis)**2


def vis2cp(vis,
           data):
    """
    
    """

    return np.matmul(data['cpmat'], np.angle(vis))


def vis2kp(vis,
           data):
    """
    
    """

    return np.matmul(data['kpmat'], np.angle(vis))


def vis_ud(p0,
           data,
           smear=None):
    """
    
    """

    if smear is None:
        vis = np.pi * p0[0] * mas2rad * np.sqrt(data['vis2uu']**2 + data['vis2vv']**2)
        vis += 1e-6 * (vis == 0)
        vis = 2. * j1(vis) / vis
    else:
        vis = np.pi * p0[0] * mas2rad * np.sqrt(data['vis2uu_smear']**2 + data['vis2vv_smear']**2)
        vis += 1e-6 * (vis == 0)
        vis = 2. * j1(vis) / vis

        vis = vis.reshape((vis.shape[0], vis.shape[1] // smear, smear))
        vis = np.nanmean(vis, axis=2)

    return vis


def vis_bin(p0,
            data,
            smear=None):
    """
    
    """

    if smear is None:
        v1 = 1.0 + 0.0j
        v2 = 1.0 + 0.0j
        temp1 = v2 * p0[0]
        temp2 = np.exp(-2. * np.pi * 1.0j * mas2rad * (np.multiply(data['vis2uu'].T, p0[1].T) + np.multiply(data['vis2vv'].T, p0[2].T)))
        vis = np.multiply(temp1.T, temp2)
        vis = np.divide(v1 + vis, (1. + p0[0]).T).T
    else:
        v1 = 1.0 + 0.0j
        v2 = 1.0 + 0.0j
        temp1 = v2 * p0[0]
        temp2 = np.exp(-2. * np.pi * 1.0j * mas2rad * (np.multiply(data['vis2uu_smear'].T, p0[1].T) + np.multiply(data['vis2vv_smear'].T, p0[2].T)))
        vis = np.multiply(temp1.T, temp2)
        vis = np.divide(v1 + vis, (1. + p0[0]).T).T

        vis = vis.reshape((vis.shape[0], vis.shape[1] // smear, smear))
        vis = np.nanmean(vis, axis=2)

    return vis


def vis_tri(p0,
            data,
            smear=None):
    """
    
    """

    if smear is None:
        v1 = 1.0 + 0.0j
        v2 = 1.0 + 0.0j
        v3 = 1.0 + 0.0j
        temp1 = v2 * p0[0]
        temp2 = np.exp(-2. * np.pi * 1.0j * mas2rad * (np.multiply(data['vis2uu'].T, p0[1].T) + np.multiply(data['vis2vv'].T, p0[2].T)))
        vis2 = np.multiply(temp1.T, temp2)
        temp3 = v3 * p0[3]
        temp4 = np.exp(-2. * np.pi * 1.0j * mas2rad * (np.multiply(data['vis2uu'].T, p0[4].T) + np.multiply(data['vis2vv'].T, p0[5].T)))
        vis3 = np.multiply(temp3.T, temp4)
        vis = np.divide(v1 + vis2 + vis3, (1. + p0[0] + p0[3]).T).T
    else:
        v1 = 1.0 + 0.0j
        v2 = 1.0 + 0.0j
        v3 = 1.0 + 0.0j
        temp1 = v2 * p0[0]
        temp2 = np.exp(-2. * np.pi * 1.0j * mas2rad * (np.multiply(data['vis2uu_smear'].T, p0[1].T) + np.multiply(data['vis2vv_smear'].T, p0[2].T)))
        vis2 = np.multiply(temp1.T, temp2)
        temp3 = v3 * p0[3]
        temp4 = np.exp(-2. * np.pi * 1.0j * mas2rad * (np.multiply(data['vis2uu_smear'].T, p0[4].T) + np.multiply(data['vis2vv_smear'].T, p0[5].T)))
        vis3 = np.multiply(temp3.T, temp4)
        vis = np.divide(v1 + vis2 + vis3, (1. + p0[0] + p0[3]).T).T

        vis = vis.reshape((vis.shape[0], vis.shape[1] // smear, smear))
        vis = np.nanmean(vis, axis=2)

    return vis


def vis_ud_bin(p0,
               data,
               smear=None):
    """
    
    """

    if smear is None:
        v1 = np.multiply(np.pi * p0[3].T, mas2rad * np.sqrt(data['vis2uu']**2 + data['vis2vv']**2).T)
        v1 += 1e-6 * (v1 == 0)
        v1 = 2. * np.true_divide(j1(v1), v1)
        v2 = 1.0 + 0.0j
        temp1 = v2 * p0[0]
        temp2 = np.exp(-2. * np.pi * 1.0j * mas2rad * (np.multiply(data['vis2uu'].T, p0[1].T) + np.multiply(data['vis2vv'].T, p0[2].T)))
        vis = np.multiply(temp1.T, temp2)
        vis = np.divide(v1 + vis, (1. + p0[0]).T).T
    else:
        v1 = np.multiply(np.pi * p0[3].T, mas2rad * np.sqrt(data['vis2uu_smear']**2 + data['vis2vv_smear']**2).T)
        v1 += 1e-6 * (v1 == 0)
        v1 = 2. * np.true_divide(j1(v1), v1)
        v2 = 1.0 + 0.0j
        temp1 = v2 * p0[0]
        temp2 = np.exp(-2. * np.pi * 1.0j * mas2rad * (np.multiply(data['vis2uu_smear'].T, p0[1].T) + np.multiply(data['vis2vv_smear'].T, p0[2].T)))
        vis = np.multiply(temp1.T, temp2)
        vis = np.divide(v1 + vis, (1. + p0[0]).T).T

        vis = vis.reshape((vis.shape[0], vis.shape[1] // smear, smear))
        vis = np.nanmean(vis, axis=2)

    return vis


def chi2_ud(p0,
            data_list,
            obs,
            cov=False,
            smear=None):
    """
    
    """

    chi2 = []
    for data in data_list:
        vis_mod = vis_ud(p0=p0,
                         data=data,
                         smear=smear)

        sig = []
        err = []
        mod = []
        for obs_temp in obs:
            if obs_temp == 'vis2':
                sig += [data['vis2']]
                err += [data['dvis2']]
                mod += [vis2vis2(vis_mod, data=data)]
            elif obs_temp == 'cp':
                sig += [data['cp']]
                err += [data['dcp']]
                mod += [vis2cp(vis_mod, data=data)]
            elif obs_temp == 'kp':
                sig += [data['kp']]
                err += [data['dkp']]
                mod += [vis2kp(vis_mod, data=data)]
        if cov:
            icv = []
            for obs_temp in obs:
                if obs_temp == 'vis2':
                    icv += [data['vis2icv']]
                elif obs_temp == 'cp':
                    icv += [data['cpicv']]
                elif obs_temp == 'kp':
                    icv += [data['kpicv']]

        if not cov:
            sig = np.concatenate([temp.flatten() for temp in sig])
            mod = np.concatenate([temp.flatten() for temp in mod])
            res = sig - mod
            var = np.concatenate([temp.flatten() for temp in err])**2
            res_icv = np.true_divide(res, var)
            ww = np.isnan(res)
            res[ww] = 0.
            res_icv[ww] = 0.
            chi2 += [res_icv.dot(res)]
        else:
            for i in range(len(sig)):
                for j in range(sig[i].shape[0]):
                    sig_temp = sig[i][j].flatten()
                    mod_temp = mod[i][j].flatten()
                    res_temp = sig_temp - mod_temp
                    ww = np.isnan(res_temp)
                    res_temp[ww] = 0.
                    res_icv_temp = np.dot(res_temp, icv[i][j])
                    chi2 += [res_icv_temp.dot(res_temp)]

    return np.sum(chi2)


def chi2_ud_ndof(p0,
                 data_list,
                 obs,
                 cov=False,
                 smear=None):
    """
    
    """

    chi2 = []
    ndof = 1
    for data in data_list:
        vis_mod = vis_ud(p0=p0,
                         data=data,
                         smear=smear)

        sig = []
        err = []
        mod = []
        for obs_temp in obs:
            if obs_temp == 'vis2':
                sig += [data['vis2']]
                err += [data['dvis2']]
                mod += [vis2vis2(vis_mod, data=data)]
            elif obs_temp == 'cp':
                sig += [data['cp']]
                err += [data['dcp']]
                mod += [vis2cp(vis_mod, data=data)]
            elif obs_temp == 'kp':
                sig += [data['kp']]
                err += [data['dkp']]
                mod += [vis2kp(vis_mod, data=data)]
        if cov:
            icv = []
            for obs_temp in obs:
                if obs_temp == 'vis2':
                    icv += [data['vis2icv']]
                elif obs_temp == 'cp':
                    icv += [data['cpicv']]
                elif obs_temp == 'kp':
                    icv += [data['kpicv']]

        if not cov:
            sig = np.concatenate([temp.flatten() for temp in sig])
            mod = np.concatenate([temp.flatten() for temp in mod])
            res = sig - mod
            var = np.concatenate([temp.flatten() for temp in err])**2
            res_icv = np.true_divide(res, var)
            ww = np.isnan(res)
            res[ww] = 0.
            res_icv[ww] = 0.
            chi2 += [res_icv.dot(res)]
            ndof += np.prod(res.shape) - np.sum(ww)
        else:
            for i in range(len(sig)):
                for j in range(sig[i].shape[0]):
                    sig_temp = sig[i][j].flatten()
                    mod_temp = mod[i][j].flatten()
                    res_temp = sig_temp - mod_temp
                    ww = np.isnan(res_temp)
                    res_temp[ww] = 0.
                    res_icv_temp = np.dot(res_temp, icv[i][j])
                    chi2 += [res_icv_temp.dot(res_temp)]
                    ndof += np.prod(res_temp.shape) - np.sum(ww)

    return np.sum(chi2), ndof


def chi2_bin_c(c0,
               p0,
               data_list,
               obs,
               cov=False,
               smear=None):
    """
    
    """

    chi2 = []
    for data in data_list:
        ra = p0[0].copy()
        dec = p0[1].copy()
        rho = np.sqrt(ra**2 + dec**2)
        phi = np.rad2deg(np.arctan2(ra, dec))
        if pa_mtoc == '-':
            phi -= data['vis2pa']
        elif pa_mtoc == '+':
            phi += data['vis2pa']
        else:
            raise UserWarning('Model to chip conversion for position angle not known')
        ra_temp = rho * np.sin(np.deg2rad(phi))
        dec_temp = rho * np.cos(np.deg2rad(phi))
        c0_temp = np.ones_like(phi) * c0
        p0_temp = np.array([c0_temp, ra_temp, dec_temp])
        vis_mod = vis_bin(p0=p0_temp,
                          data=data,
                          smear=smear)

        sig = []
        err = []
        mod = []
        for obs_temp in obs:
            if obs_temp == 'vis2':
                sig += [data['vis2']]
                err += [data['dvis2']]
                mod += [vis2vis2(vis_mod, data=data)]
            elif obs_temp == 'cp':
                sig += [data['cp']]
                err += [data['dcp']]
                mod += [vis2cp(vis_mod, data=data)]
            elif obs_temp == 'kp':
                sig += [data['kp']]
                err += [data['dkp']]
                mod += [vis2kp(vis_mod, data=data)]
        if cov:
            icv = []
            for obs_temp in obs:
                if obs_temp == 'vis2':
                    icv += [data['vis2icv']]
                elif obs_temp == 'cp':
                    icv += [data['cpicv']]
                elif obs_temp == 'kp':
                    icv += [data['kpicv']]

        if not cov:
            sig = np.concatenate([temp.flatten() for temp in sig])
            mod = np.concatenate([temp.flatten() for temp in mod])
            res = sig - mod
            var = np.concatenate([temp.flatten() for temp in err])**2
            res_icv = np.true_divide(res, var)
            ww = np.isnan(res)
            res[ww] = 0.
            res_icv[ww] = 0.
            chi2 += [res_icv.dot(res)]
        else:
            for i in range(len(sig)):
                for j in range(sig[i].shape[0]):
                    sig_temp = sig[i][j].flatten()
                    mod_temp = mod[i][j].flatten()
                    res_temp = sig_temp - mod_temp
                    ww = np.isnan(res_temp)
                    res_temp[ww] = 0.
                    res_icv_temp = np.dot(res_temp, icv[i][j])
                    chi2 += [res_icv_temp.dot(res_temp)]

    return np.sum(chi2)


def chi2_bin_c_ndof(c0,
                    p0,
                    data_list,
                    obs,
                    cov=False,
                    smear=None):
    """
    
    """

    chi2 = []
    ndof = 1
    for data in data_list:
        ra = p0[0].copy()
        dec = p0[1].copy()
        rho = np.sqrt(ra**2 + dec**2)
        phi = np.rad2deg(np.arctan2(ra, dec))
        if pa_mtoc == '-':
            phi -= data['vis2pa']
        elif pa_mtoc == '+':
            phi += data['vis2pa']
        else:
            raise UserWarning('Model to chip conversion for position angle not known')
        ra_temp = rho * np.sin(np.deg2rad(phi))
        dec_temp = rho * np.cos(np.deg2rad(phi))
        c0_temp = np.ones_like(phi) * c0
        p0_temp = np.array([c0_temp, ra_temp, dec_temp])
        vis_mod = vis_bin(p0=p0_temp,
                          data=data,
                          smear=smear)

        sig = []
        err = []
        mod = []
        for obs_temp in obs:
            if obs_temp == 'vis2':
                sig += [data['vis2']]
                err += [data['dvis2']]
                mod += [vis2vis2(vis_mod, data=data)]
            elif obs_temp == 'cp':
                sig += [data['cp']]
                err += [data['dcp']]
                mod += [vis2cp(vis_mod, data=data)]
            elif obs_temp == 'kp':
                sig += [data['kp']]
                err += [data['dkp']]
                mod += [vis2kp(vis_mod, data=data)]
        if cov:
            icv = []
            for obs_temp in obs:
                if obs_temp == 'vis2':
                    icv += [data['vis2icv']]
                elif obs_temp == 'cp':
                    icv += [data['cpicv']]
                elif obs_temp == 'kp':
                    icv += [data['kpicv']]

        if not cov:
            sig = np.concatenate([temp.flatten() for temp in sig])
            mod = np.concatenate([temp.flatten() for temp in mod])
            res = sig - mod
            var = np.concatenate([temp.flatten() for temp in err])**2
            res_icv = np.true_divide(res, var)
            ww = np.isnan(res)
            res[ww] = 0.
            res_icv[ww] = 0.
            chi2 += [res_icv.dot(res)]
            ndof += np.prod(res.shape) - np.sum(ww)
        else:
            for i in range(len(sig)):
                for j in range(sig[i].shape[0]):
                    sig_temp = sig[i][j].flatten()
                    mod_temp = mod[i][j].flatten()
                    res_temp = sig_temp - mod_temp
                    ww = np.isnan(res_temp)
                    res_temp[ww] = 0.
                    res_icv_temp = np.dot(res_temp, icv[i][j])
                    chi2 += [res_icv_temp.dot(res_temp)]
                    ndof += np.prod(res_temp.shape) - np.sum(ww)

    return np.sum(chi2), ndof


def chi2_bin(p0,
             data_list,
             obs,
             cov=False,
             smear=None):
    """
    
    """

    chi2 = []
    for data in data_list:
        ra = p0[1].copy()
        dec = p0[2].copy()
        rho = np.sqrt(ra**2 + dec**2)
        phi = np.rad2deg(np.arctan2(ra, dec))
        if pa_mtoc == '-':
            phi -= data['vis2pa']
        elif pa_mtoc == '+':
            phi += data['vis2pa']
        else:
            raise UserWarning('Model to chip conversion for position angle not known')
        ra_temp = rho * np.sin(np.deg2rad(phi))
        dec_temp = rho * np.cos(np.deg2rad(phi))
        c0_temp = np.ones_like(phi) * p0[0].copy()
        p0_temp = np.array([c0_temp, ra_temp, dec_temp])
        vis_mod = vis_bin(p0=p0_temp,
                          data=data,
                          smear=smear)

        sig = []
        err = []
        mod = []
        for obs_temp in obs:
            if obs_temp == 'vis2':
                sig += [data['vis2']]
                err += [data['dvis2']]
                mod += [vis2vis2(vis_mod, data=data)]
            elif obs_temp == 'cp':
                sig += [data['cp']]
                err += [data['dcp']]
                mod += [vis2cp(vis_mod, data=data)]
            elif obs_temp == 'kp':
                sig += [data['kp']]
                err += [data['dkp']]
                mod += [vis2kp(vis_mod, data=data)]
        if cov:
            icv = []
            for obs_temp in obs:
                if obs_temp == 'vis2':
                    icv += [data['vis2icv']]
                elif obs_temp == 'cp':
                    icv += [data['cpicv']]
                elif obs_temp == 'kp':
                    icv += [data['kpicv']]

        if not cov:
            sig = np.concatenate([temp.flatten() for temp in sig])
            mod = np.concatenate([temp.flatten() for temp in mod])
            res = sig - mod
            var = np.concatenate([temp.flatten() for temp in err])**2
            res_icv = np.true_divide(res, var)
            ww = np.isnan(res)
            res[ww] = 0.
            res_icv[ww] = 0.
            chi2 += [res_icv.dot(res)]
        else:
            for i in range(len(sig)):
                for j in range(sig[i].shape[0]):
                    sig_temp = sig[i][j].flatten()
                    mod_temp = mod[i][j].flatten()
                    res_temp = sig_temp - mod_temp
                    ww = np.isnan(res_temp)
                    res_temp[ww] = 0.
                    res_icv_temp = np.dot(res_temp, icv[i][j])
                    chi2 += [res_icv_temp.dot(res_temp)]

    return np.sum(chi2)


def chi2_bin_ndof(p0,
                  data_list,
                  obs,
                  cov=False,
                  smear=None):
    """
    
    """

    chi2 = []
    ndof = 3
    for data in data_list:
        ra = p0[1].copy()
        dec = p0[2].copy()
        rho = np.sqrt(ra**2 + dec**2)
        phi = np.rad2deg(np.arctan2(ra, dec))
        if pa_mtoc == '-':
            phi -= data['vis2pa']
        elif pa_mtoc == '+':
            phi += data['vis2pa']
        else:
            raise UserWarning('Model to chip conversion for position angle not known')
        ra_temp = rho * np.sin(np.deg2rad(phi))
        dec_temp = rho * np.cos(np.deg2rad(phi))
        c0_temp = np.ones_like(phi) * p0[0].copy()
        p0_temp = np.array([c0_temp, ra_temp, dec_temp])
        vis_mod = vis_bin(p0=p0_temp,
                          data=data,
                          smear=smear)

        sig = []
        err = []
        mod = []
        for obs_temp in obs:
            if obs_temp == 'vis2':
                sig += [data['vis2']]
                err += [data['dvis2']]
                mod += [vis2vis2(vis_mod, data=data)]
            elif obs_temp == 'cp':
                sig += [data['cp']]
                err += [data['dcp']]
                mod += [vis2cp(vis_mod, data=data)]
            elif obs_temp == 'kp':
                sig += [data['kp']]
                err += [data['dkp']]
                mod += [vis2kp(vis_mod, data=data)]

        sig = []
        err = []
        mod = []
        for obs_temp in obs:
            if obs_temp == 'vis2':
                sig += [data['vis2']]
                err += [data['dvis2']]
                mod += [vis2vis2(vis_mod, data=data)]
            elif obs_temp == 'cp':
                sig += [data['cp']]
                err += [data['dcp']]
                mod += [vis2cp(vis_mod, data=data)]
            elif obs_temp == 'kp':
                sig += [data['kp']]
                err += [data['dkp']]
                mod += [vis2kp(vis_mod, data=data)]
        if cov:
            icv = []
            for obs_temp in obs:
                if obs_temp == 'vis2':
                    icv += [data['vis2icv']]
                elif obs_temp == 'cp':
                    icv += [data['cpicv']]
                elif obs_temp == 'kp':
                    icv += [data['kpicv']]

        if not cov:
            sig = np.concatenate([temp.flatten() for temp in sig])
            mod = np.concatenate([temp.flatten() for temp in mod])
            res = sig - mod
            var = np.concatenate([temp.flatten() for temp in err])**2
            res_icv = np.true_divide(res, var)
            ww = np.isnan(res)
            res[ww] = 0.
            res_icv[ww] = 0.
            chi2 += [res_icv.dot(res)]
            ndof += np.prod(res.shape) - np.sum(ww)
        else:
            for i in range(len(sig)):
                for j in range(sig[i].shape[0]):
                    sig_temp = sig[i][j].flatten()
                    mod_temp = mod[i][j].flatten()
                    res_temp = sig_temp - mod_temp
                    ww = np.isnan(res_temp)
                    res_temp[ww] = 0.
                    res_icv_temp = np.dot(res_temp, icv[i][j])
                    chi2 += [res_icv_temp.dot(res_temp)]
                    ndof += np.prod(res_temp.shape) - np.sum(ww)

    return np.sum(chi2), ndof


def chi2_tri(p0,
             data_list,
             obs,
             cov=False,
             smear=None):
    """
    
    """

    chi2 = []
    for data in data_list:
        ra1 = p0[1].copy()
        dec1 = p0[2].copy()
        rho1 = np.sqrt(ra1**2 + dec1**2)
        phi1 = np.rad2deg(np.arctan2(ra1, dec1))
        if pa_mtoc == '-':
            phi1 -= data['vis2pa']
        elif pa_mtoc == '+':
            phi1 += data['vis2pa']
        else:
            raise UserWarning('Model to chip conversion for position angle not known')
        ra1_temp = rho1 * np.sin(np.deg2rad(phi1))
        dec1_temp = rho1 * np.cos(np.deg2rad(phi1))
        c01_temp = np.ones_like(phi1) * p0[0].copy()
        ra2 = p0[4].copy()
        dec2 = p0[5].copy()
        rho2 = np.sqrt(ra2**2 + dec2**2)
        phi2 = np.rad2deg(np.arctan2(ra2, dec2))
        if pa_mtoc == '-':
            phi2 -= data['vis2pa']
        elif pa_mtoc == '+':
            phi2 += data['vis2pa']
        else:
            raise UserWarning('Model to chip conversion for position angle not known')
        ra2_temp = rho2 * np.sin(np.deg2rad(phi2))
        dec2_temp = rho2 * np.cos(np.deg2rad(phi2))
        c02_temp = np.ones_like(phi2) * p0[3].copy()
        p0_temp = np.array([c01_temp, ra1_temp, dec1_temp, c02_temp, ra2_temp, dec2_temp])
        vis_mod = vis_tri(p0=p0_temp,
                          data=data,
                          smear=smear)

        sig = []
        err = []
        mod = []
        for obs_temp in obs:
            if obs_temp == 'vis2':
                sig += [data['vis2']]
                err += [data['dvis2']]
                mod += [vis2vis2(vis_mod, data=data)]
            elif obs_temp == 'cp':
                sig += [data['cp']]
                err += [data['dcp']]
                mod += [vis2cp(vis_mod, data=data)]
            elif obs_temp == 'kp':
                sig += [data['kp']]
                err += [data['dkp']]
                mod += [vis2kp(vis_mod, data=data)]
        if cov:
            icv = []
            for obs_temp in obs:
                if obs_temp == 'vis2':
                    icv += [data['vis2icv']]
                elif obs_temp == 'cp':
                    icv += [data['cpicv']]
                elif obs_temp == 'kp':
                    icv += [data['kpicv']]

        if not cov:
            sig = np.concatenate([temp.flatten() for temp in sig])
            mod = np.concatenate([temp.flatten() for temp in mod])
            res = sig - mod
            var = np.concatenate([temp.flatten() for temp in err])**2
            res_icv = np.true_divide(res, var)
            ww = np.isnan(res)
            res[ww] = 0.
            res_icv[ww] = 0.
            chi2 += [res_icv.dot(res)]
        else:
            for i in range(len(sig)):
                for j in range(sig[i].shape[0]):
                    sig_temp = sig[i][j].flatten()
                    mod_temp = mod[i][j].flatten()
                    res_temp = sig_temp - mod_temp
                    ww = np.isnan(res_temp)
                    res_temp[ww] = 0.
                    res_icv_temp = np.dot(res_temp, icv[i][j])
                    chi2 += [res_icv_temp.dot(res_temp)]

    return np.sum(chi2)


def chi2_tri_ndof(p0,
                  data_list,
                  obs,
                  cov=False,
                  smear=None):
    """
    
    """

    chi2 = []
    ndof = 6
    for data in data_list:
        ra1 = p0[1].copy()
        dec1 = p0[2].copy()
        rho1 = np.sqrt(ra1**2 + dec1**2)
        phi1 = np.rad2deg(np.arctan2(ra1, dec1))
        if pa_mtoc == '-':
            phi1 -= data['vis2pa']
        elif pa_mtoc == '+':
            phi1 += data['vis2pa']
        else:
            raise UserWarning('Model to chip conversion for position angle not known')
        ra1_temp = rho1 * np.sin(np.deg2rad(phi1))
        dec1_temp = rho1 * np.cos(np.deg2rad(phi1))
        c01_temp = np.ones_like(phi1) * p0[0].copy()
        ra2 = p0[4].copy()
        dec2 = p0[5].copy()
        rho2 = np.sqrt(ra2**2 + dec2**2)
        phi2 = np.rad2deg(np.arctan2(ra2, dec2))
        if pa_mtoc == '-':
            phi2 -= data['vis2pa']
        elif pa_mtoc == '+':
            phi2 += data['vis2pa']
        else:
            raise UserWarning('Model to chip conversion for position angle not known')
        ra2_temp = rho2 * np.sin(np.deg2rad(phi2))
        dec2_temp = rho2 * np.cos(np.deg2rad(phi2))
        c02_temp = np.ones_like(phi2) * p0[3].copy()
        p0_temp = np.array([c01_temp, ra1_temp, dec1_temp, c02_temp, ra2_temp, dec2_temp])
        vis_mod = vis_tri(p0=p0_temp,
                          data=data,
                          smear=smear)

        sig = []
        err = []
        mod = []
        for obs_temp in obs:
            if obs_temp == 'vis2':
                sig += [data['vis2']]
                err += [data['dvis2']]
                mod += [vis2vis2(vis_mod, data=data)]
            elif obs_temp == 'cp':
                sig += [data['cp']]
                err += [data['dcp']]
                mod += [vis2cp(vis_mod, data=data)]
            elif obs_temp == 'kp':
                sig += [data['kp']]
                err += [data['dkp']]
                mod += [vis2kp(vis_mod, data=data)]

        sig = []
        err = []
        mod = []
        for obs_temp in obs:
            if obs_temp == 'vis2':
                sig += [data['vis2']]
                err += [data['dvis2']]
                mod += [vis2vis2(vis_mod, data=data)]
            elif obs_temp == 'cp':
                sig += [data['cp']]
                err += [data['dcp']]
                mod += [vis2cp(vis_mod, data=data)]
            elif obs_temp == 'kp':
                sig += [data['kp']]
                err += [data['dkp']]
                mod += [vis2kp(vis_mod, data=data)]
        if cov:
            icv = []
            for obs_temp in obs:
                if obs_temp == 'vis2':
                    icv += [data['vis2icv']]
                elif obs_temp == 'cp':
                    icv += [data['cpicv']]
                elif obs_temp == 'kp':
                    icv += [data['kpicv']]

        if not cov:
            sig = np.concatenate([temp.flatten() for temp in sig])
            mod = np.concatenate([temp.flatten() for temp in mod])
            res = sig - mod
            var = np.concatenate([temp.flatten() for temp in err])**2
            res_icv = np.true_divide(res, var)
            ww = np.isnan(res)
            res[ww] = 0.
            res_icv[ww] = 0.
            chi2 += [res_icv.dot(res)]
            ndof += np.prod(res.shape) - np.sum(ww)
        else:
            for i in range(len(sig)):
                for j in range(sig[i].shape[0]):
                    sig_temp = sig[i][j].flatten()
                    mod_temp = mod[i][j].flatten()
                    res_temp = sig_temp - mod_temp
                    ww = np.isnan(res_temp)
                    res_temp[ww] = 0.
                    res_icv_temp = np.dot(res_temp, icv[i][j])
                    chi2 += [res_icv_temp.dot(res_temp)]
                    ndof += np.prod(res_temp.shape) - np.sum(ww)

    return np.sum(chi2), ndof


def chi2_ud_bin_c(c0,
                  p0,
                  data_list,
                  obs,
                  cov=False,
                  smear=None):
    """
    
    """

    chi2 = []
    for data in data_list:
        ra = p0[0].copy()
        dec = p0[1].copy()
        rho = np.sqrt(ra**2 + dec**2)
        phi = np.rad2deg(np.arctan2(ra, dec))
        if pa_mtoc == '-':
            phi -= data['vis2pa']
        elif pa_mtoc == '+':
            phi += data['vis2pa']
        else:
            raise UserWarning('Model to chip conversion for position angle not known')
        ra_temp = rho * np.sin(np.deg2rad(phi))
        dec_temp = rho * np.cos(np.deg2rad(phi))
        c00_temp = np.ones_like(phi) * c0[0]
        c01_temp = np.ones_like(phi) * c0[1]
        p0_temp = np.array([c00_temp, ra_temp, dec_temp, c01_temp])
        vis_mod = vis_ud_bin(p0=p0_temp,
                             data=data,
                             smear=smear)

        sig = []
        err = []
        mod = []
        for obs_temp in obs:
            if obs_temp == 'vis2':
                sig += [data['vis2']]
                err += [data['dvis2']]
                mod += [vis2vis2(vis_mod, data=data)]
            elif obs_temp == 'cp':
                sig += [data['cp']]
                err += [data['dcp']]
                mod += [vis2cp(vis_mod, data=data)]
            elif obs_temp == 'kp':
                sig += [data['kp']]
                err += [data['dkp']]
                mod += [vis2kp(vis_mod, data=data)]
        if cov:
            icv = []
            for obs_temp in obs:
                if obs_temp == 'vis2':
                    icv += [data['vis2icv']]
                elif obs_temp == 'cp':
                    icv += [data['cpicv']]
                elif obs_temp == 'kp':
                    icv += [data['kpicv']]

        if not cov:
            sig = np.concatenate([temp.flatten() for temp in sig])
            mod = np.concatenate([temp.flatten() for temp in mod])
            res = sig - mod
            var = np.concatenate([temp.flatten() for temp in err])**2
            res_icv = np.true_divide(res, var)
            ww = np.isnan(res)
            res[ww] = 0.
            res_icv[ww] = 0.
            chi2 += [res_icv.dot(res)]
        else:
            for i in range(len(sig)):
                for j in range(sig[i].shape[0]):
                    sig_temp = sig[i][j].flatten()
                    mod_temp = mod[i][j].flatten()
                    res_temp = sig_temp - mod_temp
                    ww = np.isnan(res_temp)
                    res_temp[ww] = 0.
                    res_icv_temp = np.dot(res_temp, icv[i][j])
                    chi2 += [res_icv_temp.dot(res_temp)]

    return np.sum(chi2)


def chi2_ud_bin_c_ndof(c0,
                       p0,
                       data_list,
                       obs,
                       cov=False,
                       smear=None):
    """
    
    """

    chi2 = []
    ndof = 2
    for data in data_list:
        ra = p0[0].copy()
        dec = p0[1].copy()
        rho = np.sqrt(ra**2 + dec**2)
        phi = np.rad2deg(np.arctan2(ra, dec))
        if pa_mtoc == '-':
            phi -= data['vis2pa']
        elif pa_mtoc == '+':
            phi += data['vis2pa']
        else:
            raise UserWarning('Model to chip conversion for position angle not known')
        ra_temp = rho * np.sin(np.deg2rad(phi))
        dec_temp = rho * np.cos(np.deg2rad(phi))
        c00_temp = np.ones_like(phi) * c0[0]
        c01_temp = np.ones_like(phi) * c0[1]
        p0_temp = np.array([c00_temp, ra_temp, dec_temp, c01_temp])
        vis_mod = vis_ud_bin(p0=p0_temp,
                             data=data,
                             smear=smear)

        sig = []
        err = []
        mod = []
        for obs_temp in obs:
            if obs_temp == 'vis2':
                sig += [data['vis2']]
                err += [data['dvis2']]
                mod += [vis2vis2(vis_mod, data=data)]
            elif obs_temp == 'cp':
                sig += [data['cp']]
                err += [data['dcp']]
                mod += [vis2cp(vis_mod, data=data)]
            elif obs_temp == 'kp':
                sig += [data['kp']]
                err += [data['dkp']]
                mod += [vis2kp(vis_mod, data=data)]
        if cov:
            icv = []
            for obs_temp in obs:
                if obs_temp == 'vis2':
                    icv += [data['vis2icv']]
                elif obs_temp == 'cp':
                    icv += [data['cpicv']]
                elif obs_temp == 'kp':
                    icv += [data['kpicv']]

        if not cov:
            sig = np.concatenate([temp.flatten() for temp in sig])
            mod = np.concatenate([temp.flatten() for temp in mod])
            res = sig - mod
            var = np.concatenate([temp.flatten() for temp in err])**2
            res_icv = np.true_divide(res, var)
            ww = np.isnan(res)
            res[ww] = 0.
            res_icv[ww] = 0.
            chi2 += [res_icv.dot(res)]
            ndof += np.prod(res.shape) - np.sum(ww)
        else:
            for i in range(len(sig)):
                for j in range(sig[i].shape[0]):
                    sig_temp = sig[i][j].flatten()
                    mod_temp = mod[i][j].flatten()
                    res_temp = sig_temp - mod_temp
                    ww = np.isnan(res_temp)
                    res_temp[ww] = 0.
                    res_icv_temp = np.dot(res_temp, icv[i][j])
                    chi2 += [res_icv_temp.dot(res_temp)]
                    ndof += np.prod(res_temp.shape) - np.sum(ww)

    return np.sum(chi2), ndof


def chi2_ud_bin(p0,
                data_list,
                obs,
                cov=False,
                smear=None):
    """
    
    """

    chi2 = []
    for data in data_list:
        ra = p0[1].copy()
        dec = p0[2].copy()
        rho = np.sqrt(ra**2 + dec**2)
        phi = np.rad2deg(np.arctan2(ra, dec))
        if pa_mtoc == '-':
            phi -= data['vis2pa']
        elif pa_mtoc == '+':
            phi += data['vis2pa']
        else:
            raise UserWarning('Model to chip conversion for position angle not known')
        ra_temp = rho * np.sin(np.deg2rad(phi))
        dec_temp = rho * np.cos(np.deg2rad(phi))
        p00_temp = np.ones_like(phi) * p0[0]
        p03_temp = np.ones_like(phi) * p0[3]
        p0_temp = np.array([p00_temp, ra_temp, dec_temp, p03_temp])
        vis_mod = vis_ud_bin(p0=p0_temp,
                             data=data,
                             smear=smear)

        sig = []
        err = []
        mod = []
        for obs_temp in obs:
            if obs_temp == 'vis2':
                sig += [data['vis2']]
                err += [data['dvis2']]
                mod += [vis2vis2(vis_mod, data=data)]
            elif obs_temp == 'cp':
                sig += [data['cp']]
                err += [data['dcp']]
                mod += [vis2cp(vis_mod, data=data)]
            elif obs_temp == 'kp':
                sig += [data['kp']]
                err += [data['dkp']]
                mod += [vis2kp(vis_mod, data=data)]
        if cov:
            icv = []
            for obs_temp in obs:
                if obs_temp == 'vis2':
                    icv += [data['vis2icv']]
                elif obs_temp == 'cp':
                    icv += [data['cpicv']]
                elif obs_temp == 'kp':
                    icv += [data['kpicv']]

        if not cov:
            sig = np.concatenate([temp.flatten() for temp in sig])
            mod = np.concatenate([temp.flatten() for temp in mod])
            res = sig - mod
            var = np.concatenate([temp.flatten() for temp in err])**2
            res_icv = np.true_divide(res, var)
            ww = np.isnan(res)
            res[ww] = 0.
            res_icv[ww] = 0.
            chi2 += [res_icv.dot(res)]
        else:
            for i in range(len(sig)):
                for j in range(sig[i].shape[0]):
                    sig_temp = sig[i][j].flatten()
                    mod_temp = mod[i][j].flatten()
                    res_temp = sig_temp - mod_temp
                    ww = np.isnan(res_temp)
                    res_temp[ww] = 0.
                    res_icv_temp = np.dot(res_temp, icv[i][j])
                    chi2 += [res_icv_temp.dot(res_temp)]

    return np.sum(chi2)


def chi2_ud_bin_ndof(p0,
                     data_list,
                     obs,
                     cov=False,
                     smear=None):
    """
    
    """

    chi2 = []
    ndof = 4
    for data in data_list:
        ra = p0[1].copy()
        dec = p0[2].copy()
        rho = np.sqrt(ra**2 + dec**2)
        phi = np.rad2deg(np.arctan2(ra, dec))
        if pa_mtoc == '-':
            phi -= data['vis2pa']
        elif pa_mtoc == '+':
            phi += data['vis2pa']
        else:
            raise UserWarning('Model to chip conversion for position angle not known')
        ra_temp = rho * np.sin(np.deg2rad(phi))
        dec_temp = rho * np.cos(np.deg2rad(phi))
        p00_temp = np.ones_like(phi) * p0[0]
        p03_temp = np.ones_like(phi) * p0[3]
        p0_temp = np.array([p00_temp, ra_temp, dec_temp, p03_temp])
        vis_mod = vis_ud_bin(p0=p0_temp,
                             data=data,
                             smear=smear)

        sig = []
        err = []
        mod = []
        for obs_temp in obs:
            if obs_temp == 'vis2':
                sig += [data['vis2']]
                err += [data['dvis2']]
                mod += [vis2vis2(vis_mod, data=data)]
            elif obs_temp == 'cp':
                sig += [data['cp']]
                err += [data['dcp']]
                mod += [vis2cp(vis_mod, data=data)]
            elif obs_temp == 'kp':
                sig += [data['kp']]
                err += [data['dkp']]
                mod += [vis2kp(vis_mod, data=data)]
        if cov:
            icv = []
            for obs_temp in obs:
                if obs_temp == 'vis2':
                    icv += [data['vis2icv']]
                elif obs_temp == 'cp':
                    icv += [data['cpicv']]
                elif obs_temp == 'kp':
                    icv += [data['kpicv']]

        if not cov:
            sig = np.concatenate([temp.flatten() for temp in sig])
            mod = np.concatenate([temp.flatten() for temp in mod])
            res = sig - mod
            var = np.concatenate([temp.flatten() for temp in err])**2
            res_icv = np.true_divide(res, var)
            ww = np.isnan(res)
            res[ww] = 0.
            res_icv[ww] = 0.
            chi2 += [res_icv.dot(res)]
            ndof += np.prod(res.shape) - np.sum(ww)
        else:
            for i in range(len(sig)):
                for j in range(sig[i].shape[0]):
                    sig_temp = sig[i][j].flatten()
                    mod_temp = mod[i][j].flatten()
                    res_temp = sig_temp - mod_temp
                    ww = np.isnan(res_temp)
                    res_temp[ww] = 0.
                    res_icv_temp = np.dot(res_temp, icv[i][j])
                    chi2 += [res_icv_temp.dot(res_temp)]
                    ndof += np.prod(res_temp.shape) - np.sum(ww)

    return np.sum(chi2), ndof


def lnprob_bin(p0,
               data_list,
               obs,
               cov=False,
               smear=None,
               temperature=1.):
    """
    
    """

    if p0[0] < 0.:

        return -np.inf

    chi2 = chi2_bin(p0=p0,
                    data_list=data_list,
                    obs=obs,
                    cov=cov,
                    smear=smear)

    return -0.5 * np.nansum(chi2) / temperature


def lnprob_tri(p0,
               data_list,
               obs,
               cov=False,
               smear=None,
               temperature=1.):
    """
    
    """

    if p0[0] < 0. or p0[3] < 0.:

        return -np.inf

    chi2 = chi2_tri(p0=p0,
                    data_list=data_list,
                    obs=obs,
                    cov=cov,
                    smear=smear)

    return -0.5 * np.nansum(chi2) / temperature


def lnprob_ud_bin(p0,
                  data_list,
                  obs,
                  cov=False,
                  smear=None,
                  temperature=1.):
    """
    
    """

    if p0[0] < 0. or p0[3] < 0.:

        return -np.inf

    chi2 = chi2_ud_bin(p0=p0,
                       data_list=data_list,
                       obs=obs,
                       cov=cov,
                       smear=smear)

    return -0.5 * np.nansum(chi2) / temperature


def injec_companion(fit_inj,
                    data_list,
                    obs):
    """
    
    """

    if 'bin' not in fit_inj['model']:

        pass

    for data in data_list:
        p0 = fit_inj['p']
        ra = p0[1].copy()
        dec = p0[2].copy()
        rho = np.sqrt(ra**2 + dec**2)
        phi = np.rad2deg(np.arctan2(ra, dec))
        if pa_mtoc == '-':
            phi -= data['vis2pa']
        elif pa_mtoc == '+':
            phi += data['vis2pa']
        else:
            raise UserWarning('Model to chip conversion for position angle not known')
        ra_temp = rho * np.sin(np.deg2rad(phi))
        dec_temp = rho * np.cos(np.deg2rad(phi))
        if fit_inj['model'] == 'bin':
            p00_temp = np.ones_like(phi) * np.abs(p0[0])
            p0_temp = np.array([p00_temp, ra_temp, dec_temp])  # w/ companion
            vis_mod = vis_bin(p0=p0_temp,
                              data=data,
                              smear=fit_inj['smear'])
            p00_temp = np.ones_like(phi) * 0.
            p0_temp = np.array([p00_temp, ra_temp, dec_temp])  # w/o companion
            vis_ref = vis_bin(p0=p0_temp,
                              data=data,
                              smear=fit_inj['smear'])
        elif fit_inj['model'] == 'ud_bin':
            p00_temp = np.ones_like(phi) * np.abs(p0[0])
            p03_temp = np.ones_like(phi) * p0[3]
            p0_temp = np.array([p00_temp, ra_temp, dec_temp, p03_temp])  # w/ companion
            vis_mod = vis_ud_bin(p0=p0_temp,
                                 data=data,
                                 smear=fit_inj['smear'])
            p00_temp = np.ones_like(phi) * 0.
            p03_temp = np.ones_like(phi) * p0[3]
            p0_temp = np.array([p00_temp, ra_temp, dec_temp, p03_temp])  # w/o companion
            vis_ref = vis_ud_bin(p0=p0_temp,
                                 data=data,
                                 smear=fit_inj['smear'])
        else:
            raise UserWarning()

        if 'vis2' in obs:
            data['vis2'] += np.sign(p0[0]) * (vis2vis2(vis_mod, data=data) - vis2vis2(vis_ref, data=data))
        if 'cp' in obs:
            data['cp'] += np.sign(p0[0]) * (vis2cp(vis_mod, data=data) - vis2cp(vis_ref, data=data))
        if 'kp' in obs:
            data['kp'] += np.sign(p0[0]) * (vis2kp(vis_mod, data=data) - vis2kp(vis_ref, data=data))

    return data_list
