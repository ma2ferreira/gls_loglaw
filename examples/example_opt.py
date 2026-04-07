#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author:     Manuel Ferreira
Created:    April 2026
Updated:    
Version:    v1.0

Example:
    Optimal generalised least squares fit of the log-law velocity profile.

    It demonstrates how to:
    1. Generate a synthetic turbulent boundary layer profile
    2. Propagate measurement uncertainties
    3. Perform a GLS fit of the log-law
    4. Visualise the confidence region of the parameters

Run:
    python example_gls_loglaw.py
'''

import numpy as np
import bl_profile as bl
import hot_wire as hw
import opt_tools as ot
import plot_tools as pt
from dataclasses import dataclass

@dataclass
class in_cov:
    e_u: float
    e_u_tau: float
    e_nu: float
    e_q: float
    e_rho: float
    e_v: float
    s_z0: float
    s_dz: float

def main():

    # Turbulent boundary layer
    Re_tau = 10000
    delta = 0.1
    nu = 1.51e-5
    z_plus = hw.z(Re_tau, 30, 10)
    u_plus = bl.composite(z_plus, Re_tau)
    rng = np.random.default_rng(seed=42)
    u_plus += rng.normal(0, 0.001* u_plus, len(z_plus))

    # Uncertainty in primitive variables (e_phi percentage, s_phi absolute)
    cov = in_cov(
        e_u = 0.001,
        e_u_tau = 0.005,
        e_nu = 0.006,
        e_q = 0.002,
        e_rho = 0.003,
        e_v = 0.001,
        s_z0 = 10e-6,
        s_dz = 1e-6,
    )
    
    # GLS fit
    stats, z_plus_crop, u_plus_crop = ot.opt_routine(
        Re_tau,
        delta,
        nu,
        z_plus,
        u_plus,
        cov)
    
    # Unpack results
    A = stats.A
    kappa = stats.kappa
    xi = stats.xi
    eta = stats.eta
    
    # Figure 1: viscous-normalised velocity profile
    fig1, ax1 = pt.make_axes(
        70,
        [15, 5, 10, 5],
        [10, Re_tau],
        [0, 40],
        r'$z^+$',
        r'$U^+$',
        yticks=[0, 20, 40],
        xscale='log'
    )
    
    ax1.plot(
        [3 * Re_tau**0.5, 3 * Re_tau**0.5],
        ax1.get_ylim(),
        color='k',
        linewidth=0.2,
        linestyle=(0, (20, 10))
    )
    
    ax1.plot(
        [0.15 * Re_tau, 0.15 * Re_tau],
        ax1.get_ylim(),
        color='k',
        linewidth=0.2,
        linestyle=(0, (20, 10))
    )
    
    ax1.plot(
        z_plus,
        u_plus,
        marker='o',
        markersize=2,
        markerfacecolor='gray',
        markeredgecolor='none',
        linestyle='none'
    )

    ax1.plot(
        z_plus_crop,
        u_plus_crop,
        marker='o',
        markersize=3,
        markerfacecolor='k',
        markeredgecolor='none',
        linestyle='none'
    )

    # Figure 2: joint uncertainty region
    fig2, ax2 = pt.make_axes(
        70,
        [15, 5, 10, 5],
        [0.8 * A, 1.2 * A],
        [0.9 * kappa, 1.1 * kappa],
        r'$A$',
        r'$\kappa$'
    )

    # Reference lines
    ax2.plot(ax2.get_xlim(), [kappa, kappa], color='k', linewidth=0.2)
    ax2.plot([A, A], ax2.get_ylim(), color='k', linewidth=0.2)
    
    # ±5% guides
    for x in [0.95 * A, 1.05 * A]:
        ax2.plot(
            [x, x],
            ax2.get_ylim(),
            color='k',
            linewidth=0.2,
            linestyle=(0, (20, 10))
        )
    for y in [0.95 * kappa, 1.05 * kappa]:
        ax2.plot(
            ax2.get_xlim(),
            [y, y],
            color='k',
            linewidth=0.2,
            linestyle=(0, (20, 10))
        )
    
    # 95% confidence region
    ax2.plot(xi, 1 / eta, color='k', linewidth=0.6)

if __name__ == "__main__":
    main()