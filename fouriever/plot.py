from __future__ import division

import os
import pdb
import sys

import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import numpy as np

import corner as cp
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import matplotlib.patheffects as PathEffects

from fouriever import util

from fouriever.opticstools import opticstools as ot

import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.rcParams.update({'font.size': 14})


# =============================================================================
# MAIN
# =============================================================================

colors = ['midnightblue', 'indigo', 'darkorchid', 'mediumorchid', 'orchid', 'violet']
nc = len(colors)


def plot_oidata(data,
                ofile,
                fit=None):
    """
    
    """

    inss = data.keys()
    for ins in inss:

        if 'kp' in data[ins].keys():
            nkp = data[ins]['kp'].shape[1]
        else:
            nkp = 3
        if 'cp' in data[ins].keys():
            ncp = data[ins]['cp'].shape[1]
        else:
            ncp = 4
        if 'vis2' in data[ins].keys():
            nvis2 = data[ins]['vis2'].shape[1]
        else:
            nvis2 = 6

        fig = plt.figure(figsize=(2.0 * 6.4, 1.5 * 4.8), constrained_layout=False)
        gs1 = fig.add_gridspec(nrows=2, ncols=1, left=0.04, right=0.23, hspace=0.30)
        ax01 = fig.add_subplot(gs1[0, 0])
        ax02 = fig.add_subplot(gs1[1, 0])
        gs2 = fig.add_gridspec(nrows=nkp, ncols=1, left=0.27, right=0.48, hspace=0.00)
        axs_kp = []
        for i in range(nkp):
            axs_kp += [fig.add_subplot(gs2[i, 0])]
        gs3 = fig.add_gridspec(nrows=ncp, ncols=1, left=0.52, right=0.73, hspace=0.00)
        axs_cp = []
        for i in range(ncp):
            axs_cp += [fig.add_subplot(gs3[i, 0])]
        gs4 = fig.add_gridspec(nrows=nvis2, ncols=1, left=0.77, right=0.96, hspace=0.00)
        axs_vis2 = []
        for i in range(nvis2):
            axs_vis2 += [fig.add_subplot(gs4[i, 0])]

        if 'vis2' in data[ins].keys():
            for i in range(data[ins]['vis2'].shape[0]):
                for j in range(data[ins]['vis2'].shape[1]):
                    if i == 0:
                        ax01.plot(data[ins]['vis2u'][i, j], data[ins]['vis2v'][i, j], ls='', marker='o', ms=5, color=colors[j % nc], label=data[ins]['vis2sta'][i, j])
                    else:
                        ax01.plot(data[ins]['vis2u'][i, j], data[ins]['vis2v'][i, j], ls='', marker='o', ms=5, color=colors[j % nc])
                    ax01.plot(-data[ins]['vis2u'][i, j], -data[ins]['vis2v'][i, j], ls='', marker='o', ms=5, color=colors[j % nc])
            c1 = plt.Circle((0., 0.), np.min(data[ins]['vis2b']), ls=':', fc='none', ec='grey')
            ax01.add_patch(c1)
            c2 = plt.Circle((0., 0.), np.max(data[ins]['vis2b']), ls=':', fc='none', ec='grey')
            ax01.add_patch(c2)
            xlim = ax01.get_xlim()
            ylim = ax01.get_ylim()
            xr = xlim[1] - xlim[0]
            yr = ylim[1] - ylim[0]
            rr = max(xr, yr)
            xm = np.mean(xlim)
            ym = np.mean(ylim)
            ax01.set_xlim([xm - 0.5 * rr, xm + 0.8 * rr])
            ax01.set_ylim([ym - 0.5 * rr, ym + 0.8 * rr])
            ax01.set_xticklabels(ax01.get_xticklabels(), fontsize=8)
            ax01.set_yticklabels(ax01.get_yticklabels(), fontsize=8)
            ax01.legend(loc='upper right', ncols=2, fontsize=8)
        elif 'vis' in data[ins].keys():
            for i in range(data[ins]['vis'].shape[0]):
                for j in range(data[ins]['vis'].shape[1]):
                    if i == 0:
                        ax01.plot(data[ins]['visu'][i, j], data[ins]['visv'][i, j], ls='', marker='o', ms=5, color=colors[j % nc], label=data[ins]['vissta'][i, j])
                    else:
                        ax01.plot(data[ins]['visu'][i, j], data[ins]['visv'][i, j], ls='', marker='o', ms=5, color=colors[j % nc])
                    ax01.plot(-data[ins]['visu'][i, j], -data[ins]['visv'][i, j], ls='', marker='o', ms=5, color=colors[j % nc])
            c1 = plt.Circle((0., 0.), np.min(data[ins]['visb']), ls=':', fc='none', ec='grey')
            ax01.add_patch(c1)
            c2 = plt.Circle((0., 0.), np.max(data[ins]['visb']), ls=':', fc='none', ec='grey')
            ax01.add_patch(c2)
            xlim = ax01.get_xlim()
            ylim = ax01.get_ylim()
            xr = xlim[1] - xlim[0]
            yr = ylim[1] - ylim[0]
            rr = max(xr, yr)
            xm = np.mean(xlim)
            ym = np.mean(ylim)
            ax01.set_xlim([xm - 0.5 * rr, xm + 0.8 * rr])
            ax01.set_ylim([ym - 0.5 * rr, ym + 0.8 * rr])
            ax01.set_xticklabels(ax01.get_xticklabels(), fontsize=8)
            ax01.set_yticklabels(ax01.get_yticklabels(), fontsize=8)
            ax01.legend(loc='upper right', ncols=2, fontsize=8)
        ax01.set_title(r'u ($\leftarrow$E), v ($\uparrow$N) [m]')

        if 'flux' in data[ins].keys():
            for i in range(data[ins]['flux'].shape[0]):
                for j in range(data[ins]['flux'].shape[1]):
                    ax02.errorbar(data[ins]['wave'] * 1e6, data[ins]['flux'][i, j], yerr=data[ins]['dflux'][i, j], ls='', marker='o', ms=2, elinewidth=1, color=colors[j % nc], alpha=0.5)
                    if i == 0:
                        ax02.plot(data[ins]['wave'] * 1e6, data[ins]['flux'][i, j], ls='', marker='o', ms=2, color=colors[j % nc], label=data[ins]['fluxsta'][i, j])
                    else:
                        ax02.plot(data[ins]['wave'] * 1e6, data[ins]['flux'][i, j], ls='', marker='o', ms=2, color=colors[j % nc])
            ax02.set_xticklabels(ax02.get_xticklabels(), fontsize=8)
            ax02.set_yticklabels(ax02.get_yticklabels(), fontsize=8)
            ax02.set_xlabel('Wavelength [μm]')
            ax02.legend(loc='upper right', fontsize=8)
        ax02.set_title('Flux')

        if 'kp' in data[ins].keys():
            axs = axs_kp
            if fit is None or 'kp' not in fit['obs']:
                alpha = 1.0
                mod = None
            else:
                alpha = 0.3
                p0 = fit['p']
                p0 = np.repeat(np.repeat(p0[:, np.newaxis, np.newaxis], data[ins]['vis2'].shape[0], axis=1), data[ins]['vis2'].shape[1], axis=2)
                if fit['model'] == 'bin':
                    mod = util.vis_bin(p0=p0, data=data[ins], smear=fit['smear'])
                elif fit['model'] == 'tri':
                    mod = util.vis_tri(p0=p0, data=data[ins], smear=fit['smear'])
                elif fit['model'] == 'ud_bin':
                    mod = util.vis_ud_bin(p0=p0, data=data[ins], smear=fit['smear'])
                mod = util.vis2kp(mod, data[ins])
            for i in range(data[ins]['cp'].shape[0]):
                for j in range(nkp):
                    axs[j].errorbar(data[ins]['wave'] * 1e6, np.rad2deg(data[ins]['kp'][i, j]), yerr=np.rad2deg(data[ins]['dkp'][i, j]), ls='', marker='o', ms=2, elinewidth=1, color=colors[j % nc], alpha=alpha / 2.)
                    axs[j].plot(data[ins]['wave'] * 1e6, np.rad2deg(data[ins]['kp'][i, j]), ls='', marker='o', ms=2, color=colors[j % nc], alpha=alpha)
                    if mod is not None:
                        axs[j].plot(data[ins]['wave'] * 1e6, np.rad2deg(mod[i, j]), ls='', marker='o', ms=1, color='red', zorder=9)
            for i, ax in enumerate(axs):
                if i != (nkp - 1):
                    ax.set_xticklabels([], fontsize=8)
                else:
                    ax.set_xticklabels(ax.get_xticklabels(), fontsize=8)
                ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
            axs[-1].set_xlabel('Wavelength [μm]')
        axs[0].set_title('Kernel phase [deg]')

        if 'cp' in data[ins].keys():
            axs = axs_cp
            if fit is None or 'cp' not in fit['obs']:
                alpha = 1.0
                mod = None
            else:
                alpha = 0.3
                p0 = fit['p']
                p0 = np.repeat(np.repeat(p0[:, np.newaxis, np.newaxis], data[ins]['vis2'].shape[0], axis=1), data[ins]['vis2'].shape[1], axis=2)
                if fit['model'] == 'bin':
                    mod = util.vis_bin(p0=p0, data=data[ins], smear=fit['smear'])
                elif fit['model'] == 'tri':
                    mod = util.vis_bin(p0=p0, data=data[ins], smear=fit['smear'])
                elif fit['model'] == 'ud_bin':
                    mod = util.vis_ud_bin(p0=p0, data=data[ins], smear=fit['smear'])
                mod = util.vis2cp(mod, data[ins])
            for i in range(data[ins]['cp'].shape[0]):
                for j in range(ncp):
                    axs[j].errorbar(data[ins]['wave'] * 1e6, np.rad2deg(data[ins]['cp'][i, j]), yerr=np.rad2deg(data[ins]['dcp'][i, j]), ls='', marker='o', ms=2, elinewidth=1, color=colors[j % nc], alpha=alpha / 2.)
                    if i == 0:
                        axs[j].plot(data[ins]['wave'] * 1e6, np.rad2deg(data[ins]['cp'][i, j]), ls='', marker='o', ms=2, color=colors[j % nc], alpha=alpha, label=data[ins]['cpsta'][i, j])
                    else:
                        axs[j].plot(data[ins]['wave'] * 1e6, np.rad2deg(data[ins]['cp'][i, j]), ls='', marker='o', ms=2, color=colors[j % nc], alpha=alpha)
                    if mod is not None:
                        axs[j].plot(data[ins]['wave'] * 1e6, np.rad2deg(mod[i, j]), ls='', marker='o', ms=1, color='red', zorder=9)
            for i, ax in enumerate(axs):
                if i != (ncp - 1):
                    ax.set_xticklabels([], fontsize=8)
                else:
                    ax.set_xticklabels(ax.get_xticklabels(), fontsize=8)
                ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
                ax.legend(loc='upper right', fontsize=8)
            axs[-1].set_xlabel('Wavelength [μm]')
        axs[0].set_title('Closure phase [deg]')

        if 'vis2' in data[ins].keys():
            axs = axs_vis2
            if fit is None or 'vis2' not in fit['obs']:
                alpha = 1.0
                mod = None
            else:
                alpha = 0.3
                p0 = fit['p']
                p0 = np.repeat(np.repeat(p0[:, np.newaxis, np.newaxis], data[ins]['vis2'].shape[0], axis=1), data[ins]['vis2'].shape[1], axis=2)
                if fit['model'] == 'bin':
                    mod = util.vis_bin(p0=p0, data=data[ins], smear=fit['smear'])
                elif fit['model'] == 'tri':
                    mod = util.vis_bin(p0=p0, data=data[ins], smear=fit['smear'])
                elif fit['model'] == 'ud_bin':
                    mod = util.vis_ud_bin(p0=p0, data=data[ins], smear=fit['smear'])
                mod = util.vis2vis2(mod, data[ins])
            for i in range(data[ins]['vis2'].shape[0]):
                for j in range(nvis2):
                    axs[j].errorbar(data[ins]['wave'] * 1e6, data[ins]['vis2'][i, j], yerr=data[ins]['dvis2'][i, j], ls='', marker='o', ms=2, elinewidth=1, color=colors[j % nc], alpha=alpha / 2.)
                    if i == 0:
                        axs[j].plot(data[ins]['wave'] * 1e6, data[ins]['vis2'][i, j], ls='', marker='o', ms=2, color=colors[j % nc], alpha=alpha, label=data[ins]['vis2sta'][i, j])
                    else:
                        axs[j].plot(data[ins]['wave'] * 1e6, data[ins]['vis2'][i, j], ls='', marker='o', ms=2, color=colors[j % nc], alpha=alpha)
                    if mod is not None:
                        axs[j].plot(data[ins]['wave'] * 1e6, mod[i, j], ls='', marker='o', ms=1, color='red', zorder=9)
            for i, ax in enumerate(axs):
                if i != (nvis2 - 1):
                    ax.set_xticklabels([], fontsize=8)
                else:
                    ax.set_xticklabels(ax.get_xticklabels(), fontsize=8)
                ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
                ax.legend(loc='upper right', fontsize=8)
            axs[-1].set_xlabel('Wavelength [μm]')
        axs[0].set_title(r'|Visibility|${}^2$')

        plt.suptitle(ins)
        odir = os.path.split(ofile)[0]
        if odir != '' and not os.path.exists(odir):
            os.makedirs(odir)
        plt.savefig(ofile.replace('.pdf', '_' + ins + '.pdf'))
        plt.close()

    pass


def plot_cor(cor,
             obs,
             ofile):
    """
    
    """
    
    f, ax = plt.subplots(1, 2, figsize=(2 * 6.4, 1 * 4.8))
    p0 = ax[0].imshow(cor, origin='lower', cmap='seismic', vmin=-1., vmax=1.)
    c0 = plt.colorbar(p0, ax=ax[0])
    ax[0].set_title('Correlations')
    icr = np.linalg.pinv(cor)
    eye = np.dot(cor, icr)
    temp = eye - np.eye(eye.shape[0])
    vlim = np.nanmax(np.abs(temp))
    p1 = ax[1].imshow(temp, origin='lower', cmap='viridis', vmin=-vlim, vmax=vlim)
    c1 = plt.colorbar(p1, ax=ax[1])
    ax[1].set_title(r'$C\cdot C^{-1}-1$')
    plt.suptitle(obs)
    odir = os.path.split(ofile)[0]
    if odir != '' and not os.path.exists(odir):
        os.makedirs(odir)
    plt.savefig(ofile)
    plt.close()

    pass


def plot_avg_vis2(wave,
                  vis2_cal,
                  dvis2_cal,
                  vis2cov_cal,
                  vis2_cal_master,
                  dvis2_cal_master,
                  vis2cov_cal_master,
                  ofile):
    """
    
    """

    nwave = wave.shape[0]
    nvis2 = vis2_cal_master.shape[0]

    fig = plt.figure(figsize=(1.5 * 6.4, 1.5 * 4.8), constrained_layout=False)
    gs1 = fig.add_gridspec(nrows=nvis2, ncols=1, left=0.06, right=0.47, hspace=0.00)
    ax1 = []
    for i in range(nvis2):
        ax1 += [fig.add_subplot(gs1[i, 0])]
    gs2 = fig.add_gridspec(nrows=nvis2, ncols=1, left=0.53, right=0.94, hspace=0.00)
    ax2 = []
    for i in range(nvis2):
        ax2 += [fig.add_subplot(gs2[i, 0])]

    for i in range(nvis2):
        ax1[i].plot(wave * 1e6, vis2_cal[:, i, :].T, ls='', marker='o', ms=2, alpha=0.3)
        if i == 0:
            ax1[i].plot(wave * 1e6, vis2_cal_master[i], color='black', ls='', marker='o', ms=2, label='Average')
        else:
            ax1[i].plot(wave * 1e6, vis2_cal_master[i], color='black', ls='', marker='o', ms=2)
    ax1[-1].set_xlabel('Wavelength [μm]')
    ax1[0].set_title('Visibility amplitude')
    for i, ax in enumerate(ax1):
        if i != (nvis2 - 1):
            ax.set_xticklabels([], fontsize=8)
        else:
            ax.set_xticklabels(ax.get_xticklabels(), fontsize=8)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
        if i == 0:
            ax.legend(loc='upper right', fontsize=8)

    for i in range(nvis2):
        ax2[i].plot(wave * 1e6, dvis2_cal[:, i, :].T, ls='', marker='o', ms=2, alpha=0.3)
        if i == 0:
            ax2[i].plot(wave * 1e6, dvis2_cal_master[i], color='black', ls='', marker='o', ms=2, label='Average')
            if vis2cov_cal_master is not None:
                ax2[i].plot(wave * 1e6, np.sqrt(np.diag(vis2cov_cal_master)[i * nwave:(i + 1) * nwave]), color='gray', ls='', marker='o', ms=1, label='  from cov.')
        else:
            ax2[i].plot(wave * 1e6, dvis2_cal_master[i], color='black', ls='', marker='o', ms=2)
            if vis2cov_cal_master is not None:
                ax2[i].plot(wave * 1e6, np.sqrt(np.diag(vis2cov_cal_master)[i * nwave:(i + 1) * nwave]), color='gray', ls='', marker='o', ms=1)
    ax2[-1].set_xlabel('Wavelength [μm]')
    ax2[0].set_title('Visibility amplitude error')
    for i, ax in enumerate(ax2):
        if i != (nvis2 - 1):
            ax.set_xticklabels([], fontsize=8)
        else:
            ax.set_xticklabels(ax.get_xticklabels(), fontsize=8)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
        if i == 0:
            ax.legend(loc='upper right', fontsize=8)

    odir = os.path.split(ofile)[0]
    if odir != '' and not os.path.exists(odir):
        os.makedirs(odir)
    plt.savefig(ofile)
    plt.close()

    if vis2cov_cal_master is not None:
        f, ax = plt.subplots(1, 3, figsize=(3 * 6.4, 1 * 4.8))
        dd = np.sqrt(np.diag(vis2cov_cal_master))
        cor = np.true_divide(vis2cov_cal_master, dd[:, None] * dd[None, :])
        p0 = ax[0].imshow(cor, origin='lower', cmap='seismic', vmin=-1., vmax=1.)
        c0 = plt.colorbar(p0, ax=ax[0])
        ax[0].set_title('Correlations')
        p1 = ax[1].imshow(vis2cov_cal_master, origin='lower', cmap='viridis')
        c1 = plt.colorbar(p1, ax=ax[1])
        ax[1].set_title(r'Covariances')
        icv = np.linalg.pinv(vis2cov_cal_master)
        eye = np.dot(vis2cov_cal_master, icv)
        temp = eye - np.eye(eye.shape[0])
        vlim = np.nanmax(np.abs(temp))
        p2 = ax[2].imshow(temp, origin='lower', cmap='viridis', vmin=-vlim, vmax=vlim)
        c2 = plt.colorbar(p2, ax=ax[2])
        ax[2].set_title(r'$\Sigma\cdot\Sigma^{-1}-1$')
        plt.suptitle('vis2')
        odir = os.path.split(ofile)[0]
        if odir != '' and not os.path.exists(odir):
            os.makedirs(odir)
        plt.savefig(ofile.replace('.pdf', '_cov.pdf'))
        plt.close()

    pass


def plot_cal_vis2(wave,
                  vis2_sci,
                  dvis2_sci,
                  vis2cov_sci,
                  vis2_cal,
                  dvis2_cal,
                  vis2cov_cal,
                  vis2_sci_cal,
                  dvis2_sci_cal,
                  vis2cov_sci_cal,
                  ofile):
    """
    
    """

    vis2_sci = vis2_sci.reshape(vis2_sci.shape[0] * vis2_sci.shape[1], vis2_sci.shape[2])
    dvis2_sci = dvis2_sci.reshape(dvis2_sci.shape[0] * dvis2_sci.shape[1], dvis2_sci.shape[2])
    vis2_sci_cal = vis2_sci_cal.reshape(vis2_sci_cal.shape[0] * vis2_sci_cal.shape[1], vis2_sci_cal.shape[2])
    dvis2_sci_cal = dvis2_sci_cal.reshape(dvis2_sci_cal.shape[0] * dvis2_sci_cal.shape[1], dvis2_sci_cal.shape[2])

    nwave = wave.shape[0]
    nvis2 = vis2_sci.shape[0]

    fig = plt.figure(figsize=(1.5 * 6.4, 1.5 * 4.8), constrained_layout=False)
    gs1 = fig.add_gridspec(nrows=nvis2, ncols=1, left=0.06, right=0.47, hspace=0.00)
    ax1 = []
    for i in range(nvis2):
        ax1 += [fig.add_subplot(gs1[i, 0])]
    gs2 = fig.add_gridspec(nrows=nvis2, ncols=1, left=0.53, right=0.94, hspace=0.00)
    ax2 = []
    for i in range(nvis2):
        ax2 += [fig.add_subplot(gs2[i, 0])]

    for i in range(nvis2):
        if i == 0:
            ax1[i].plot(wave * 1e6, vis2_sci[i], ls='', marker='o', ms=2, alpha=0.3, label='SCI')
            ax1[i].plot(wave * 1e6, vis2_cal[i % vis2_cal.shape[0]], ls='', marker='o', ms=2, alpha=0.3, label='CAL')
            ax1[i].plot(wave * 1e6, vis2_sci_cal[i], color='black', ls='', marker='o', ms=2, label='Calibrated')
        else:
            ax1[i].plot(wave * 1e6, vis2_sci[i], ls='', marker='o', ms=2, alpha=0.3)
            ax1[i].plot(wave * 1e6, vis2_cal[i % vis2_cal.shape[0]], ls='', marker='o', ms=2, alpha=0.3)
            ax1[i].plot(wave * 1e6, vis2_sci_cal[i], color='black', ls='', marker='o', ms=2)
        ax1[i].axhline(1., ls=':', color='black')
        rms_before = np.sqrt(np.nanmean((vis2_sci[i] - 1.)**2))
        rms_after = np.sqrt(np.nanmean((vis2_sci_cal[i] - 1.)**2))
        ti = ax1[i].text(0.01, 0.01, 'RMS before/after = %.3f/%.3f' % (rms_before, rms_after), ha='left', va='bottom', color='black', transform=ax1[i].transAxes, fontsize=8)
        ti.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
    ax1[-1].set_xlabel('Wavelength [μm]')
    ax1[0].set_title('Visibility amplitude')
    for i, ax in enumerate(ax1):
        if i != (nvis2 - 1):
            ax.set_xticklabels([], fontsize=8)
        else:
            ax.set_xticklabels(ax.get_xticklabels(), fontsize=8)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
        if i == 0:
            ax.legend(loc='upper right', fontsize=8)

    for i in range(nvis2):
        if i == 0:
            ax2[i].plot(wave * 1e6, dvis2_sci[i], ls='', marker='o', ms=2, alpha=0.3, label='SCI')
            ax2[i].plot(wave * 1e6, dvis2_cal[i % dvis2_cal.shape[0]], ls='', marker='o', ms=2, alpha=0.3, label='CAL')
            ax2[i].plot(wave * 1e6, dvis2_sci_cal[i], color='black', ls='', marker='o', ms=2, label='Calibrated')
            if vis2cov_sci_cal is not None:
                ax2[i].plot(wave * 1e6, np.sqrt(np.diag(vis2cov_sci_cal[i // dvis2_cal.shape[0]])[(i % dvis2_cal.shape[0]) * nwave:((i % dvis2_cal.shape[0]) + 1) * nwave]), color='gray', ls='', marker='o', ms=1, label='  from cov.')
        else:
            ax2[i].plot(wave * 1e6, dvis2_sci[i], ls='', marker='o', ms=2, alpha=0.3)
            ax2[i].plot(wave * 1e6, dvis2_cal[i % dvis2_cal.shape[0]], ls='', marker='o', ms=2, alpha=0.3)
            ax2[i].plot(wave * 1e6, dvis2_sci_cal[i], color='black', ls='', marker='o', ms=2)
            if vis2cov_sci_cal is not None:
                ax2[i].plot(wave * 1e6, np.sqrt(np.diag(vis2cov_sci_cal[i // dvis2_cal.shape[0]])[(i % dvis2_cal.shape[0]) * nwave:((i % dvis2_cal.shape[0]) + 1) * nwave]), color='gray', ls='', marker='o', ms=1)
    ax2[-1].set_xlabel('Wavelength [μm]')
    ax2[0].set_title('Visibility amplitude error')
    for i, ax in enumerate(ax2):
        if i != (nvis2 - 1):
            ax.set_xticklabels([], fontsize=8)
        else:
            ax.set_xticklabels(ax.get_xticklabels(), fontsize=8)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
        if i == 0:
            ax.legend(loc='upper right', fontsize=8)

    odir = os.path.split(ofile)[0]
    if odir != '' and not os.path.exists(odir):
        os.makedirs(odir)
    plt.savefig(ofile)
    plt.close()

    if vis2cov_sci_cal is not None:
        ind = 0
        f, ax = plt.subplots(1, 3, figsize=(3 * 6.4, 1 * 4.8))
        dd = np.sqrt(np.diag(vis2cov_sci_cal[ind]))
        cor = np.true_divide(vis2cov_sci_cal[ind], dd[:, None] * dd[None, :])
        p0 = ax[0].imshow(cor, origin='lower', cmap='seismic', vmin=-1., vmax=1.)
        c0 = plt.colorbar(p0, ax=ax[0])
        ax[0].set_title('Correlations')
        p1 = ax[1].imshow(vis2cov_sci_cal[ind], origin='lower', cmap='viridis')
        c1 = plt.colorbar(p1, ax=ax[1])
        ax[1].set_title(r'Covariances')
        icv = np.linalg.pinv(vis2cov_sci_cal[ind])
        eye = np.dot(vis2cov_sci_cal[ind], icv)
        temp = eye - np.eye(eye.shape[0])
        vlim = np.nanmax(np.abs(temp))
        p2 = ax[2].imshow(temp, origin='lower', cmap='viridis', vmin=-vlim, vmax=vlim)
        c2 = plt.colorbar(p2, ax=ax[2])
        ax[2].set_title(r'$\Sigma\cdot\Sigma^{-1}-1$')
        plt.suptitle('vis2')
        odir = os.path.split(ofile)[0]
        if odir != '' and not os.path.exists(odir):
            os.makedirs(odir)
        plt.savefig(ofile.replace('.pdf', '_cov.pdf'))
        plt.close()

    pass


def plot_avg_cp(wave,
                cp_cal,
                dcp_cal,
                cpcov_cal,
                cp_cal_master,
                dcp_cal_master,
                cpcov_cal_master,
                ofile):
    """
    
    """

    nwave = wave.shape[0]
    ncp = cp_cal_master.shape[0]

    fig = plt.figure(figsize=(1.5 * 6.4, 1.5 * 4.8), constrained_layout=False)
    gs1 = fig.add_gridspec(nrows=ncp, ncols=1, left=0.06, right=0.47, hspace=0.00)
    ax1 = []
    for i in range(ncp):
        ax1 += [fig.add_subplot(gs1[i, 0])]
    gs2 = fig.add_gridspec(nrows=ncp, ncols=1, left=0.53, right=0.94, hspace=0.00)
    ax2 = []
    for i in range(ncp):
        ax2 += [fig.add_subplot(gs2[i, 0])]

    for i in range(ncp):
        ax1[i].plot(wave * 1e6, np.rad2deg(cp_cal[:, i, :].T), ls='', marker='o', ms=2, alpha=0.3)
        if i == 0:
            ax1[i].plot(wave * 1e6, np.rad2deg(cp_cal_master[i]), color='black', ls='', marker='o', ms=2, label='Average')
        else:
            ax1[i].plot(wave * 1e6, np.rad2deg(cp_cal_master[i]), color='black', ls='', marker='o', ms=2)
    ax1[-1].set_xlabel('Wavelength [μm]')
    ax1[0].set_title('Closure phase [deg]')
    for i, ax in enumerate(ax1):
        if i != (ncp - 1):
            ax.set_xticklabels([], fontsize=8)
        else:
            ax.set_xticklabels(ax.get_xticklabels(), fontsize=8)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
        if i == 0:
            ax.legend(loc='upper right', fontsize=8)

    for i in range(ncp):
        ax2[i].plot(wave * 1e6, np.rad2deg(dcp_cal[:, i, :].T), ls='', marker='o', ms=2, alpha=0.3)
        if i == 0:
            ax2[i].plot(wave * 1e6, np.rad2deg(dcp_cal_master[i]), color='black', ls='', marker='o', ms=2, label='Average')
            if cpcov_cal_master is not None:
                ax2[i].plot(wave * 1e6, np.rad2deg(np.sqrt(np.diag(cpcov_cal_master)[i * nwave:(i + 1) * nwave])), color='gray', ls='', marker='o', ms=1, label='  from cov.')
        else:
            ax2[i].plot(wave * 1e6, np.rad2deg(dcp_cal_master[i]), color='black', ls='', marker='o', ms=2)
            if cpcov_cal_master is not None:
                ax2[i].plot(wave * 1e6, np.rad2deg(np.sqrt(np.diag(cpcov_cal_master)[i * nwave:(i + 1) * nwave])), color='gray', ls='', marker='o', ms=1)
    ax2[-1].set_xlabel('Wavelength [μm]')
    ax2[0].set_title('Closure phase error [deg]')
    for i, ax in enumerate(ax2):
        if i != (ncp - 1):
            ax.set_xticklabels([], fontsize=8)
        else:
            ax.set_xticklabels(ax.get_xticklabels(), fontsize=8)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
        if i == 0:
            ax.legend(loc='upper right', fontsize=8)

    odir = os.path.split(ofile)[0]
    if odir != '' and not os.path.exists(odir):
        os.makedirs(odir)
    plt.savefig(ofile)
    plt.close()

    if cpcov_cal_master is not None:
        f, ax = plt.subplots(1, 3, figsize=(3 * 6.4, 1 * 4.8))
        dd = np.sqrt(np.diag(cpcov_cal_master))
        cor = np.true_divide(cpcov_cal_master, dd[:, None] * dd[None, :])
        p0 = ax[0].imshow(cor, origin='lower', cmap='seismic', vmin=-1., vmax=1.)
        c0 = plt.colorbar(p0, ax=ax[0])
        ax[0].set_title('Correlations')
        p1 = ax[1].imshow(np.rad2deg(np.rad2deg(cpcov_cal_master)), origin='lower', cmap='viridis')
        c1 = plt.colorbar(p1, ax=ax[1])
        ax[1].set_title(r'Covariances [deg${}^2$]')
        icv = np.linalg.pinv(cpcov_cal_master)
        eye = np.dot(cpcov_cal_master, icv)
        temp = eye - np.eye(eye.shape[0])
        vlim = np.nanmax(np.abs(temp))
        p2 = ax[2].imshow(temp, origin='lower', cmap='viridis', vmin=-vlim, vmax=vlim)
        c2 = plt.colorbar(p2, ax=ax[2])
        ax[2].set_title(r'$\Sigma\cdot\Sigma^{-1}-1$')
        plt.suptitle('cp')
        odir = os.path.split(ofile)[0]
        if odir != '' and not os.path.exists(odir):
            os.makedirs(odir)
        plt.savefig(ofile.replace('.pdf', '_cov.pdf'))
        plt.close()

    pass


def plot_cal_cp(wave,
                cp_sci,
                dcp_sci,
                cpcov_sci,
                cp_cal,
                dcp_cal,
                cpcov_cal,
                cp_sci_cal,
                dcp_sci_cal,
                cpcov_sci_cal,
                ofile):
    """
    
    """

    cp_sci = cp_sci.reshape(cp_sci.shape[0] * cp_sci.shape[1], cp_sci.shape[2])
    dcp_sci = dcp_sci.reshape(dcp_sci.shape[0] * dcp_sci.shape[1], dcp_sci.shape[2])
    cp_sci_cal = cp_sci_cal.reshape(cp_sci_cal.shape[0] * cp_sci_cal.shape[1], cp_sci_cal.shape[2])
    dcp_sci_cal = dcp_sci_cal.reshape(dcp_sci_cal.shape[0] * dcp_sci_cal.shape[1], dcp_sci_cal.shape[2])

    nwave = wave.shape[0]
    ncp = cp_sci.shape[0]

    fig = plt.figure(figsize=(1.5 * 6.4, 1.5 * 4.8), constrained_layout=False)
    gs1 = fig.add_gridspec(nrows=ncp, ncols=1, left=0.06, right=0.47, hspace=0.00)
    ax1 = []
    for i in range(ncp):
        ax1 += [fig.add_subplot(gs1[i, 0])]
    gs2 = fig.add_gridspec(nrows=ncp, ncols=1, left=0.53, right=0.94, hspace=0.00)
    ax2 = []
    for i in range(ncp):
        ax2 += [fig.add_subplot(gs2[i, 0])]

    for i in range(ncp):
        if i == 0:
            ax1[i].plot(wave * 1e6, np.rad2deg(cp_sci[i]), ls='', marker='o', ms=2, alpha=0.3, label='SCI')
            ax1[i].plot(wave * 1e6, np.rad2deg(cp_cal[i % cp_cal.shape[0]]), ls='', marker='o', ms=2, alpha=0.3, label='CAL')
            ax1[i].plot(wave * 1e6, np.rad2deg(cp_sci_cal[i]), color='black', ls='', marker='o', ms=2, label='Calibrated')
        else:
            ax1[i].plot(wave * 1e6, np.rad2deg(cp_sci[i]), ls='', marker='o', ms=2, alpha=0.3)
            ax1[i].plot(wave * 1e6, np.rad2deg(cp_cal[i % cp_cal.shape[0]]), ls='', marker='o', ms=2, alpha=0.3)
            ax1[i].plot(wave * 1e6, np.rad2deg(cp_sci_cal[i]), color='black', ls='', marker='o', ms=2)
        ax1[i].axhline(0., ls=':', color='black')
        rms_before = np.sqrt(np.nanmean(np.rad2deg(cp_sci[i])**2))
        rms_after = np.sqrt(np.nanmean(np.rad2deg(cp_sci_cal[i])**2))
        ti = ax1[i].text(0.01, 0.01, 'RMS before/after = %.3f/%.3f' % (rms_before, rms_after), ha='left', va='bottom', color='black', transform=ax1[i].transAxes, fontsize=8)
        ti.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
    ax1[-1].set_xlabel('Wavelength [μm]')
    ax1[0].set_title('Closure phase [deg]')
    for i, ax in enumerate(ax1):
        if i != (ncp - 1):
            ax.set_xticklabels([], fontsize=8)
        else:
            ax.set_xticklabels(ax.get_xticklabels(), fontsize=8)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
        if i == 0:
            ax.legend(loc='upper right', fontsize=8)

    for i in range(ncp):
        if i == 0:
            ax2[i].plot(wave * 1e6, np.rad2deg(dcp_sci[i]), ls='', marker='o', ms=2, alpha=0.3, label='SCI')
            ax2[i].plot(wave * 1e6, np.rad2deg(dcp_cal[i % dcp_cal.shape[0]]), ls='', marker='o', ms=2, alpha=0.3, label='CAL')
            ax2[i].plot(wave * 1e6, np.rad2deg(dcp_sci_cal[i]), color='black', ls='', marker='o', ms=2, label='Calibrated')
            if cpcov_sci_cal is not None:
                ax2[i].plot(wave * 1e6, np.rad2deg(np.sqrt(np.diag(cpcov_sci_cal[i // dcp_cal.shape[0]])[(i % dcp_cal.shape[0]) * nwave:((i % dcp_cal.shape[0]) + 1) * nwave])), color='gray', ls='', marker='o', ms=1, label='  from cov.')
        else:
            ax2[i].plot(wave * 1e6, np.rad2deg(dcp_sci[i]), ls='', marker='o', ms=2, alpha=0.3)
            ax2[i].plot(wave * 1e6, np.rad2deg(dcp_cal[i % dcp_cal.shape[0]]), ls='', marker='o', ms=2, alpha=0.3)
            ax2[i].plot(wave * 1e6, np.rad2deg(dcp_sci_cal[i]), color='black', ls='', marker='o', ms=2)
            if cpcov_sci_cal is not None:
                ax2[i].plot(wave * 1e6, np.rad2deg(np.sqrt(np.diag(cpcov_sci_cal[i // dcp_cal.shape[0]])[(i % dcp_cal.shape[0]) * nwave:((i % dcp_cal.shape[0]) + 1) * nwave])), color='gray', ls='', marker='o', ms=1)
    ax2[-1].set_xlabel('Wavelength [μm]')
    ax2[0].set_title('Closure phase error [deg]')
    for i, ax in enumerate(ax2):
        if i != (ncp - 1):
            ax.set_xticklabels([], fontsize=8)
        else:
            ax.set_xticklabels(ax.get_xticklabels(), fontsize=8)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
        if i == 0:
            ax.legend(loc='upper right', fontsize=8)

    odir = os.path.split(ofile)[0]
    if odir != '' and not os.path.exists(odir):
        os.makedirs(odir)
    plt.savefig(ofile)
    plt.close()

    if cpcov_sci_cal is not None:
        ind = 0
        f, ax = plt.subplots(1, 3, figsize=(3 * 6.4, 1 * 4.8))
        dd = np.sqrt(np.diag(cpcov_sci_cal[ind]))
        cor = np.true_divide(cpcov_sci_cal[ind], dd[:, None] * dd[None, :])
        p0 = ax[0].imshow(cor, origin='lower', cmap='seismic', vmin=-1., vmax=1.)
        c0 = plt.colorbar(p0, ax=ax[0])
        ax[0].set_title('Correlations')
        p1 = ax[1].imshow(np.rad2deg(np.rad2deg(cpcov_sci_cal[ind])), origin='lower', cmap='viridis')
        c1 = plt.colorbar(p1, ax=ax[1])
        ax[1].set_title(r'Covariances [deg${}^2$]')
        icv = np.linalg.pinv(cpcov_sci_cal[ind])
        eye = np.dot(cpcov_sci_cal[ind], icv)
        temp = eye - np.eye(eye.shape[0])
        vlim = np.nanmax(np.abs(temp))
        p2 = ax[2].imshow(temp, origin='lower', cmap='viridis', vmin=-vlim, vmax=vlim)
        c2 = plt.colorbar(p2, ax=ax[2])
        ax[2].set_title(r'$\Sigma\cdot\Sigma^{-1}-1$')
        plt.suptitle('cp')
        odir = os.path.split(ofile)[0]
        if odir != '' and not os.path.exists(odir):
            os.makedirs(odir)
        plt.savefig(ofile.replace('.pdf', '_cov.pdf'))
        plt.close()

    pass


def plot_avg_kp(wave,
                kp_cal,
                dkp_cal,
                kpcov_cal,
                kp_cal_master,
                dkp_cal_master,
                kpcov_cal_master,
                ofile):
    """
    
    """

    nwave = wave.shape[0]
    nkp = kp_cal_master.shape[0]

    fig = plt.figure(figsize=(1.5 * 6.4, 1.5 * 4.8), constrained_layout=False)
    gs1 = fig.add_gridspec(nrows=nkp, ncols=1, left=0.06, right=0.47, hspace=0.00)
    ax1 = []
    for i in range(nkp):
        ax1 += [fig.add_subplot(gs1[i, 0])]
    gs2 = fig.add_gridspec(nrows=nkp, ncols=1, left=0.53, right=0.94, hspace=0.00)
    ax2 = []
    for i in range(nkp):
        ax2 += [fig.add_subplot(gs2[i, 0])]

    for i in range(nkp):
        ax1[i].plot(wave * 1e6, np.rad2deg(kp_cal[:, i, :].T), ls='', marker='o', ms=2, alpha=0.3)
        if i == 0:
            ax1[i].plot(wave * 1e6, np.rad2deg(kp_cal_master[i]), color='black', ls='', marker='o', ms=2, label='Average')
        else:
            ax1[i].plot(wave * 1e6, np.rad2deg(kp_cal_master[i]), color='black', ls='', marker='o', ms=2)
    ax1[-1].set_xlabel('Wavelength [μm]')
    ax1[0].set_title('Kernel phase [deg]')
    for i, ax in enumerate(ax1):
        if i != (nkp - 1):
            ax.set_xticklabels([], fontsize=8)
        else:
            ax.set_xticklabels(ax.get_xticklabels(), fontsize=8)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
        if i == 0:
            ax.legend(loc='upper right', fontsize=8)

    for i in range(nkp):
        ax2[i].plot(wave * 1e6, np.rad2deg(dkp_cal[:, i, :].T), ls='', marker='o', ms=2, alpha=0.3)
        if i == 0:
            ax2[i].plot(wave * 1e6, np.rad2deg(dkp_cal_master[i]), color='black', ls='', marker='o', ms=2, label='Average')
            if kpcov_cal_master is not None:
                ax2[i].plot(wave * 1e6, np.rad2deg(np.sqrt(np.diag(kpcov_cal_master)[i * nwave:(i + 1) * nwave])), color='gray', ls='', marker='o', ms=1, label='  from cov.')
        else:
            ax2[i].plot(wave * 1e6, np.rad2deg(dkp_cal_master[i]), color='black', ls='', marker='o', ms=2)
            if kpcov_cal_master is not None:
                ax2[i].plot(wave * 1e6, np.rad2deg(np.sqrt(np.diag(kpcov_cal_master)[i * nwave:(i + 1) * nwave])), color='gray', ls='', marker='o', ms=1)
    ax2[-1].set_xlabel('Wavelength [μm]')
    ax2[0].set_title('Kernel phase error [deg]')
    for i, ax in enumerate(ax2):
        if i != (nkp - 1):
            ax.set_xticklabels([], fontsize=8)
        else:
            ax.set_xticklabels(ax.get_xticklabels(), fontsize=8)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
        if i == 0:
            ax.legend(loc='upper right', fontsize=8)

    odir = os.path.split(ofile)[0]
    if odir != '' and not os.path.exists(odir):
        os.makedirs(odir)
    plt.savefig(ofile)
    plt.close()

    if kpcov_cal_master is not None:
        f, ax = plt.subplots(1, 3, figsize=(3 * 6.4, 1 * 4.8))
        dd = np.sqrt(np.diag(kpcov_cal_master))
        cor = np.true_divide(kpcov_cal_master, dd[:, None] * dd[None, :])
        p0 = ax[0].imshow(cor, origin='lower', cmap='seismic', vmin=-1., vmax=1.)
        c0 = plt.colorbar(p0, ax=ax[0])
        ax[0].set_title('Correlations')
        p1 = ax[1].imshow(np.rad2deg(np.rad2deg(kpcov_cal_master)), origin='lower', cmap='viridis')
        c1 = plt.colorbar(p1, ax=ax[1])
        ax[1].set_title(r'Covariances [deg${}^2$]')
        icv = np.linalg.pinv(kpcov_cal_master)
        eye = np.dot(kpcov_cal_master, icv)
        temp = eye - np.eye(eye.shape[0])
        vlim = np.nanmax(np.abs(temp))
        p2 = ax[2].imshow(temp, origin='lower', cmap='viridis', vmin=-vlim, vmax=vlim)
        c2 = plt.colorbar(p2, ax=ax[2])
        ax[2].set_title(r'$\Sigma\cdot\Sigma^{-1}-1$')
        plt.suptitle('kp')
        odir = os.path.split(ofile)[0]
        if odir != '' and not os.path.exists(odir):
            os.makedirs(odir)
        plt.savefig(ofile.replace('.pdf', '_cov.pdf'))
        plt.close()

    pass


def plot_cal_kp(wave,
                kp_sci,
                dkp_sci,
                kpcov_sci,
                kp_cal,
                dkp_cal,
                kpcov_cal,
                kp_sci_cal,
                dkp_sci_cal,
                kpcov_sci_cal,
                ofile):
    """
    
    """

    kp_sci = kp_sci.reshape(kp_sci.shape[0] * kp_sci.shape[1], kp_sci.shape[2])
    dkp_sci = dkp_sci.reshape(dkp_sci.shape[0] * dkp_sci.shape[1], dkp_sci.shape[2])
    kp_sci_cal = kp_sci_cal.reshape(kp_sci_cal.shape[0] * kp_sci_cal.shape[1], kp_sci_cal.shape[2])
    dkp_sci_cal = dkp_sci_cal.reshape(dkp_sci_cal.shape[0] * dkp_sci_cal.shape[1], dkp_sci_cal.shape[2])

    nwave = wave.shape[0]
    nkp = kp_sci.shape[0]

    fig = plt.figure(figsize=(1.5 * 6.4, 1.5 * 4.8), constrained_layout=False)
    gs1 = fig.add_gridspec(nrows=nkp, ncols=1, left=0.06, right=0.47, hspace=0.00)
    ax1 = []
    for i in range(nkp):
        ax1 += [fig.add_subplot(gs1[i, 0])]
    gs2 = fig.add_gridspec(nrows=nkp, ncols=1, left=0.53, right=0.94, hspace=0.00)
    ax2 = []
    for i in range(nkp):
        ax2 += [fig.add_subplot(gs2[i, 0])]

    for i in range(nkp):
        if i == 0:
            ax1[i].plot(wave * 1e6, np.rad2deg(kp_sci[i]), ls='', marker='o', ms=2, alpha=0.3, label='SCI')
            ax1[i].plot(wave * 1e6, np.rad2deg(kp_cal[i % kp_cal.shape[0]]), ls='', marker='o', ms=2, alpha=0.3, label='CAL')
            ax1[i].plot(wave * 1e6, np.rad2deg(kp_sci_cal[i]), color='black', ls='', marker='o', ms=2, label='Calibrated')
        else:
            ax1[i].plot(wave * 1e6, np.rad2deg(kp_sci[i]), ls='', marker='o', ms=2, alpha=0.3)
            ax1[i].plot(wave * 1e6, np.rad2deg(kp_cal[i % kp_cal.shape[0]]), ls='', marker='o', ms=2, alpha=0.3)
            ax1[i].plot(wave * 1e6, np.rad2deg(kp_sci_cal[i]), color='black', ls='', marker='o', ms=2)
        ax1[i].axhline(0., ls=':', color='black')
        rms_before = np.sqrt(np.nanmean(np.rad2deg(kp_sci[i])**2))
        rms_after = np.sqrt(np.nanmean(np.rad2deg(kp_sci_cal[i])**2))
        ti = ax1[i].text(0.01, 0.01, 'RMS before/after = %.3f/%.3f' % (rms_before, rms_after), ha='left', va='bottom', color='black', transform=ax1[i].transAxes, fontsize=8)
        ti.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
    ax1[-1].set_xlabel('Wavelength [μm]')
    ax1[0].set_title('Kernel phase [deg]')
    for i, ax in enumerate(ax1):
        if i != (nkp - 1):
            ax.set_xticklabels([], fontsize=8)
        else:
            ax.set_xticklabels(ax.get_xticklabels(), fontsize=8)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
        if i == 0:
            ax.legend(loc='upper right', fontsize=8)

    for i in range(nkp):
        if i == 0:
            ax2[i].plot(wave * 1e6, np.rad2deg(dkp_sci[i]), ls='', marker='o', ms=2, alpha=0.3, label='SCI')
            ax2[i].plot(wave * 1e6, np.rad2deg(dkp_cal[i % dkp_cal.shape[0]]), ls='', marker='o', ms=2, alpha=0.3, label='CAL')
            ax2[i].plot(wave * 1e6, np.rad2deg(dkp_sci_cal[i]), color='black', ls='', marker='o', ms=2, label='Calibrated')
            if kpcov_sci_cal is not None:
                ax2[i].plot(wave * 1e6, np.rad2deg(np.sqrt(np.diag(kpcov_sci_cal[i // dkp_cal.shape[0]])[(i % dkp_cal.shape[0]) * nwave:((i % dkp_cal.shape[0]) + 1) * nwave])), color='gray', ls='', marker='o', ms=1, label='  from cov.')
        else:
            ax2[i].plot(wave * 1e6, np.rad2deg(dkp_sci[i]), ls='', marker='o', ms=2, alpha=0.3)
            ax2[i].plot(wave * 1e6, np.rad2deg(dkp_cal[i % dkp_cal.shape[0]]), ls='', marker='o', ms=2, alpha=0.3)
            ax2[i].plot(wave * 1e6, np.rad2deg(dkp_sci_cal[i]), color='black', ls='', marker='o', ms=2)
            if kpcov_sci_cal is not None:
                ax2[i].plot(wave * 1e6, np.rad2deg(np.sqrt(np.diag(kpcov_sci_cal[i // dkp_cal.shape[0]])[(i % dkp_cal.shape[0]) * nwave:((i % dkp_cal.shape[0]) + 1) * nwave])), color='gray', ls='', marker='o', ms=1)
    ax2[-1].set_xlabel('Wavelength [μm]')
    ax2[0].set_title('Kernel phase error [deg]')
    for i, ax in enumerate(ax2):
        if i != (nkp - 1):
            ax.set_xticklabels([], fontsize=8)
        else:
            ax.set_xticklabels(ax.get_xticklabels(), fontsize=8)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
        if i == 0:
            ax.legend(loc='upper right', fontsize=8)

    odir = os.path.split(ofile)[0]
    if odir != '' and not os.path.exists(odir):
        os.makedirs(odir)
    plt.savefig(ofile)
    plt.close()

    if kpcov_sci_cal is not None:
        ind = 0
        f, ax = plt.subplots(1, 3, figsize=(3 * 6.4, 1 * 4.8))
        dd = np.sqrt(np.diag(kpcov_sci_cal[ind]))
        cor = np.true_divide(kpcov_sci_cal[ind], dd[:, None] * dd[None, :])
        p0 = ax[0].imshow(cor, origin='lower', cmap='seismic', vmin=-1., vmax=1.)
        c0 = plt.colorbar(p0, ax=ax[0])
        ax[0].set_title('Correlations')
        p1 = ax[1].imshow(np.rad2deg(np.rad2deg(kpcov_sci_cal[ind])), origin='lower', cmap='viridis')
        c1 = plt.colorbar(p1, ax=ax[1])
        ax[1].set_title(r'Covariances [deg${}^2$]')
        icv = np.linalg.pinv(kpcov_sci_cal[ind])
        eye = np.dot(kpcov_sci_cal[ind], icv)
        temp = eye - np.eye(eye.shape[0])
        vlim = np.nanmax(np.abs(temp))
        p2 = ax[2].imshow(temp, origin='lower', cmap='viridis', vmin=-vlim, vmax=vlim)
        c2 = plt.colorbar(p2, ax=ax[2])
        ax[2].set_title(r'$\Sigma\cdot\Sigma^{-1}-1$')
        plt.suptitle('kp')
        odir = os.path.split(ofile)[0]
        if odir != '' and not os.path.exists(odir):
            os.makedirs(odir)
        plt.savefig(ofile.replace('.pdf', '_cov.pdf'))
        plt.close()

    pass


def plot_chi2map(fit,
                 ofile,
                 searchbox=None):
    """
    
    """

    ra = fit['p'][1].copy()
    dra = fit['dp'][1].copy()
    dec = fit['p'][2].copy()
    ddec = fit['dp'][2].copy()
    rho = np.sqrt(ra**2 + dec**2)
    drho = np.sqrt((ra * dra / rho)**2 + (dec * ddec / rho)**2)
    phi = np.rad2deg(np.arctan2(ra, dec))
    dphi = np.rad2deg(np.sqrt((dec / rho**2 * dra)**2 + (ra / rho**2 * ddec)**2))

    plt.figure(figsize=(6.4, 4.8))
    ax = plt.gca()
    grid = fit['plot_grid']
    if 'plot_grid_fine' in fit.keys():
        grid_fine = fit['plot_grid_fine']
        grid_ra = grid_fine[:, :, 1]
        grid_dec = grid_fine[:, :, 2]
    else:
        grid_ra = grid[:, :, 1]
        grid_dec = grid[:, :, 2]
    step_ra = grid_ra[0, 0] - grid_ra[0, 1]
    step_dec = grid_dec[1, 0] - grid_dec[0, 0]
    ext = (np.max(grid_ra) + step_ra / 2., np.min(grid_ra) - step_ra / 2., np.min(grid_dec) - step_dec / 2., np.max(grid_dec) + step_dec / 2.)
    if 'plot_chi2_fine' in fit.keys():
        p0 = ax.imshow(fit['plot_chi2_fine'] / fit['ndof'], origin='lower', extent=ext, cmap='cubehelix')
    else:
        p0 = ax.imshow(fit['plot_chi2'] / fit['ndof'], origin='lower', extent=ext, cmap='cubehelix')
    c0 = plt.colorbar(p0, ax=ax)
    c0.set_label(r'$\chi^2_{\rm{red}}$', rotation=270, labelpad=20)
    ww = np.where(np.isinf(fit['plot_niter']))
    for i in range(len(ww[0])):
        xx = grid[ww[0][i], ww[1][i]][1]
        yy = grid[ww[0][i], ww[1][i]][2]
        ax.plot(xx, yy, marker='x', color='red', markersize=2.5)
    ax.plot(0., 0., marker='*', color='black', markersize=5)
    r0 = plt.Circle(fit['p'][1:3], (ext[0] - ext[1]) / 20., lw=5, fc='none', ec='white')
    ax.add_artist(r0)
    r1 = plt.Circle(fit['p'][1:3], (ext[0] - ext[1]) / 20., lw=3, fc='none', ec='black')
    ax.add_artist(r1)
    t0 = ax.text(0.01, 0.99, r'$c$ = %.3f +/- %.3f %%' % (fit['p'][0] * 100., fit['dp'][0] * 100.), ha='left', va='top', color='black', transform=ax.transAxes)
    t0.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
    if len(fit['p']) == 4:
        t1 = ax.text(0.01, 0.94, r'$\theta$ = %.3f +/- %.3f mas' % (fit['p'][3], fit['dp'][3]), ha='left', va='top', color='black', transform=ax.transAxes)
        t1.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
    if drho > 0. or dphi > 0.:
        t2 = ax.text(0.01, 0.06, r'$\rho$ = %.1f +/- %.1f mas' % (rho, drho), ha='left', va='bottom', color='black', transform=ax.transAxes)
        t2.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
        t3 = ax.text(0.01, 0.01, r'$\varphi$ = %.1f +/- %.1f deg' % (phi, dphi), ha='left', va='bottom', color='black', transform=ax.transAxes)
        t3.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
    else:
        t2 = ax.text(0.01, 0.06, r'$\rho$ = %.1f mas' % rho, ha='left', va='bottom', color='black', transform=ax.transAxes)
        t2.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
        t3 = ax.text(0.01, 0.01, r'$\varphi$ = %.1f deg' % phi, ha='left', va='bottom', color='black', transform=ax.transAxes)
        t3.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
    t4 = ax.text(0.99, 0.01, r'$\chi^2_{\rm{red}}$ = %.3f' % fit['chi2_red'], ha='right', va='bottom', color='black', transform=ax.transAxes)
    t4.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
    t5 = ax.text(0.99, 0.99, r'$N_{\sigma}$ = %.1f' % fit['nsigma'], ha='right', va='top', color='black', transform=ax.transAxes)
    t5.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
    if searchbox is not None:
        r0 = patches.Rectangle((searchbox['RA'][0], searchbox['Dec'][0]), searchbox['RA'][1] - searchbox['RA'][0], searchbox['Dec'][1] - searchbox['Dec'][0], linewidth=1, edgecolor='red', facecolor='none')
        ax.add_patch(r0)
    ax.set_xlabel(r'$\Delta\rm{RA}$ [mas]')
    ax.set_ylabel(r'$\Delta\rm{Dec}$ [mas]')
    ax.set_title(str(fit['obs']) + ', cov = ' + str(fit['cov']))
    odir = os.path.split(ofile)[0]
    if odir != '' and not os.path.exists(odir):
        os.makedirs(odir)
    plt.savefig(ofile)
    plt.close()

    pass


def plot_chains(fit,
                samples,
                ofile):
    """
    
    """

    if fit['model'] == 'bin':
        temp = samples.copy()
        temp[:, 0] *= 100.
        labels = [r'$f$ [%]', 'RA [mas]', 'Dec [mas]']
        f, ax = plt.subplots(temp.shape[1], 1, sharex='col', figsize=(6.4, 1.6 * temp.shape[1]))
        for i in range(temp.shape[1]):
            ax[i].plot(temp[:, i], color=colors[4])
            ax[i].axhline(np.percentile(temp[:, i], 50.), color=colors[2], label='Median')
            ax[i].set_ylabel(labels[i])
            if (i == 0):
                ax[i].legend(loc='upper right', fontsize=8)
        ax[-1].set_xlabel('Step')
        plt.subplots_adjust(wspace=0.25, hspace=0.00)
        f.align_ylabels()
        plt.suptitle('Chains')
        plt.tight_layout()
    elif fit['model'] == 'tri':
        temp = samples.copy()
        temp[:, 0] *= 100.
        temp[:, 3] *= 100.
        labels = [r'$f1$ [%]', 'RA1 [mas]', 'Dec1 [mas]', r'$f2$ [%]', 'RA2 [mas]', 'Dec2 [mas]']
        f, ax = plt.subplots(temp.shape[1], 1, sharex='col', figsize=(6.4, 1.6 * temp.shape[1]))
        for i in range(temp.shape[1]):
            ax[i].plot(temp[:, i], color=colors[4])
            ax[i].axhline(np.percentile(temp[:, i], 50.), color=colors[2], label='Median')
            ax[i].set_ylabel(labels[i])
            if (i == 0):
                ax[i].legend(loc='upper right', fontsize=8)
        ax[-1].set_xlabel('Step')
        plt.subplots_adjust(wspace=0.25, hspace=0.00)
        f.align_ylabels()
        plt.suptitle('Chains')
        plt.tight_layout()
    elif fit['model'] == 'ud_bin':
        temp = samples.copy()
        temp[:, 0] *= 100.
        labels = [r'$f$ [%]', 'RA [mas]', 'Dec [mas]', r'$\theta$ [mas]']
        f, ax = plt.subplots(temp.shape[1], 1, sharex='col', figsize=(6.4, 1.6 * temp.shape[1]))
        for i in range(temp.shape[1]):
            ax[i].plot(temp[:, i], color=colors[4])
            ax[i].axhline(np.percentile(temp[:, i], 50.), color=colors[2], label='Median')
            ax[i].set_ylabel(labels[i])
            if (i == 0):
                ax[i].legend(loc='upper right', fontsize=8)
        ax[-1].set_xlabel('Step')
        plt.subplots_adjust(wspace=0.25, hspace=0.00)
        f.align_ylabels()
        plt.suptitle('Chains')
        plt.tight_layout()
    else:
        raise UserWarning()
    odir = os.path.split(ofile)[0]
    if odir != '' and not os.path.exists(odir):
        os.makedirs(odir)
    plt.savefig(ofile)
    plt.close()

    pass


def plot_corner(fit,
                samples,
                ofile,
                fixpos=False):
    """
    
    """

    if fit['model'] == 'bin':
        if not fixpos:
            temp = samples.copy()
            temp[:, 0] *= 100.
            f = cp.corner(temp,
                          labels=[r'$f$ [%]', 'RA [mas]', 'Dec [mas]'],
                          titles=[r'$f$', r'$\alpha$', r'$\delta$'],
                          quantiles=[0.16, 0.5, 0.84],
                          show_titles=True,
                          title_fmt='.3f')
        else:
            temp = samples[:, [0]].copy()
            temp[:, 0] *= 100.
            f = cp.corner(temp,
                          labels=[r'$f$ [%]'],
                          titles=[r'$f$'],
                          quantiles=[0.16, 0.5, 0.84],
                          show_titles=True,
                          title_fmt='.3f')
    elif fit['model'] == 'tri':
        if not fixpos:
            temp = samples.copy()
            temp[:, 0] *= 100.
            temp[:, 3] *= 100.
            f = cp.corner(temp,
                          labels=[r'$f1$ [%]', 'RA1 [mas]', 'Dec1 [mas]', r'$f2$ [%]', 'RA2 [mas]', 'Dec2 [mas]'],
                          titles=[r'$f1$', r'$\alpha1$', r'$\delta1$', r'$f2$', r'$\alpha2$', r'$\delta2$'],
                          quantiles=[0.16, 0.5, 0.84],
                          show_titles=True,
                          title_fmt='.3f')
        else:
            temp = samples[:, [0, 3]].copy()
            temp[:, 0] *= 100.
            temp[:, 1] *= 100.
            f = cp.corner(temp,
                          labels=[r'$f1$ [%]', r'$f2$ [%]'],
                          titles=[r'$f1$', r'$f2$'],
                          quantiles=[0.16, 0.5, 0.84],
                          show_titles=True,
                          title_fmt='.3f')
    elif fit['model'] == 'ud_bin':
        if not fixpos:
            temp = samples.copy()
            temp[:, 0] *= 100.
            f = cp.corner(temp,
                          labels=[r'$f$ [%]', 'RA [mas]', 'Dec [mas]', r'$\theta$ [mas]'],
                          titles=[r'$f$', r'$\alpha$', r'$\delta$', r'$\theta$'],
                          quantiles=[0.16, 0.5, 0.84],
                          show_titles=True,
                          title_fmt='.3f')
        else:
            temp = samples[:, [0, 3]].copy()
            temp[:, 0] *= 100.
            f = cp.corner(temp,
                          labels=[r'$f$ [%]', r'$\theta$ [mas]'],
                          titles=[r'$f$', r'$\theta$'],
                          quantiles=[0.16, 0.5, 0.84],
                          show_titles=True,
                          title_fmt='.3f')
    else:
        raise UserWarning()
    odir = os.path.split(ofile)[0]
    if odir != '' and not os.path.exists(odir):
        os.makedirs(odir)
    plt.savefig(ofile)
    plt.close()

    pass


def plot_detlim(p0s,
                pps_absil,
                niter_absil,
                pps_injec,
                niter_injec,
                obs,
                cov,
                sigma,
                ofile):
    """
    
    """

    plt.figure(figsize=(1.5 * 6.4, 1.5 * 4.8))
    gs = gridspec.GridSpec(2, 2)
    grid_ra = p0s[:, :, 1]
    grid_dec = p0s[:, :, 2]
    step_ra = grid_ra[0, 0] - grid_ra[0, 1]
    step_dec = grid_dec[1, 0] - grid_dec[0, 0]
    ext = (np.max(grid_ra) + step_ra / 2., np.min(grid_ra) - step_ra / 2., np.min(grid_dec) - step_dec / 2., np.max(grid_dec) + step_dec / 2.)

    ax = plt.subplot(gs[0, 0])
    temp_absil = -2.5 * np.log10(pps_absil[:, :, 0])
    p00 = ax.imshow(temp_absil, origin='lower', extent=ext, cmap='cubehelix')
    c00 = plt.colorbar(p00, ax=ax)
    c00.set_label(r'%.0f$\sigma$ contrast [mag]' % sigma, rotation=270, labelpad=20)
    ww = np.where(np.isinf(niter_absil))
    for i in range(len(ww[0])):
        xx = p0s[ww[0][i], ww[1][i]][1]
        yy = p0s[ww[0][i], ww[1][i]][2]
        ax.plot(xx, yy, marker='x', color='red', markersize=2.5)
    ax.plot(0., 0., marker='*', color='black', markersize=5)
    ax.set_xlabel(r'$\Delta\rm{RA}$ [mas]')
    ax.set_ylabel(r'$\Delta\rm{Dec}$ [mas]')
    ax.set_title('Method "Absil"')

    ax = plt.subplot(gs[0, 1])
    temp_injec = -2.5 * np.log10(pps_injec[:, :, 0])
    p01 = ax.imshow(temp_injec, origin='lower', extent=ext, cmap='cubehelix')
    c01 = plt.colorbar(p01, ax=ax)
    c01.set_label(r'%.0f$\sigma$ contrast [mag]' % sigma, rotation=270, labelpad=20)
    ww = np.where(np.isinf(niter_injec))
    for i in range(len(ww[0])):
        xx = p0s[ww[0][i], ww[1][i]][1]
        yy = p0s[ww[0][i], ww[1][i]][2]
        ax.plot(xx, yy, marker='x', color='red', markersize=2.5)
    ax.plot(0., 0., marker='*', color='black', markersize=5)
    ax.set_xlabel(r'$\Delta\rm{RA}$ [mas]')
    ax.set_ylabel(r'$\Delta\rm{Dec}$ [mas]')
    ax.set_title('Method "Injection"')

    ax = plt.subplot(gs[1, :])
    sep, con = ot.azimuthalAverage(temp_absil, returnradii=True, binsize=1.)
    std = ot.azimuthalAverage(temp_absil, binsize=1., stddev=True)
    temp_absil[~np.isfinite(niter_absil)] = np.nan
    sep_new, con_new = ot.azimuthalAverage(temp_absil, returnradii=True, binsize=1.)
    std_new = ot.azimuthalAverage(temp_absil, binsize=1., stddev=True)
    if step_ra != step_dec:
        raise UserWarning()
    sep *= step_ra
    sep_new *= step_ra
    con_absil = con_new
    ax.plot(sep, con, ls=':', color=colors[1], alpha=0.5, label='Absil (incl. bad convergence)')
    ax.plot(sep_new, con_new, color=colors[1], alpha=0.5, label='Absil (excl. bad convergence)')
    ax.fill_between(sep_new, con_new - std_new, con_new + std_new, fc=colors[1], alpha=0.2)
    sep, con = ot.azimuthalAverage(temp_injec, returnradii=True, binsize=1.)
    std = ot.azimuthalAverage(temp_injec, binsize=1., stddev=True)
    temp_injec[~np.isfinite(niter_injec)] = np.nan
    sep_new, con_new = ot.azimuthalAverage(temp_injec, returnradii=True, binsize=1.)
    std_new = ot.azimuthalAverage(temp_injec, binsize=1., stddev=True)
    if step_ra != step_dec:
        raise UserWarning()
    sep *= step_ra
    sep_new *= step_ra
    con_injection = con_new
    ax.plot(sep, con, ls=':', color=colors[4], alpha=0.5, label='Injection (incl. bad convergence)')
    ax.plot(sep_new, con_new, color=colors[4], alpha=0.5, label='Injection (excl. bad convergence)')
    ax.fill_between(sep_new, con_new - std_new, con_new + std_new, fc=colors[4], alpha=0.2)
    ax.invert_yaxis()
    ax.grid(axis='both')
    ax.set_xlabel('Separation [mas]')
    ax.set_ylabel(r'%.0f$\sigma$ contrast [mag]' % sigma)
    ax.legend(loc='upper right', fontsize=8)
    plt.suptitle(str(obs) + ', cov = ' + str(cov))
    plt.tight_layout()
    odir = os.path.split(ofile)[0]
    if odir != '' and not os.path.exists(odir):
        os.makedirs(odir)
    plt.savefig(ofile)
    plt.close()

    data = []
    data += [sep_new] # mas
    data += [con_absil] # mag
    data += [con_injection] # mag
    data = np.array(data)
    np.save(ofile.replace('.pdf', '.npy'), data)

    pass
