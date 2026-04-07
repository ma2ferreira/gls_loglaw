#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author:     Manuel Ferreira
Created:    April 2026
Updated:
Version:    v1.0

Description:
    Optimisation routine for fitting the log law to the vertical profile of the
    mean streamwise velocity using Generalised Least Squares (GLS). The optimal 
    fitting window is determined by exhaustive search over all valid index 
    pairs within the log-law region, minimising a cost function.

Includes:
    opt_routin      Optimise the log-law fitting window by exhaustive search.
    _cost_fun       Evaluate the cost of a GLS fit given its p-value and statistics.
    _print_report   Print a summary report of the optimal GLS log-law fit.
"""
import numpy as np
import fit_tools as ft
import stats_tools as st

__all__ = [
    'opt_routine'
]

def opt_routine(Re_tau, delta, nu, z_plus, u_plus, cov):
    """
    Optimise the log-law fitting window by exhaustive search over all valid
    index pairs and selecting the window that minimises the cost function.
    
    PARAMETERS
    
    Re_tau : float
        Friction Reynolds number.
    delta : float
        Boundary layer / channel half-height.
    nu : float
        Kinematic viscosity.
    z_plus : ndarray
        Wall-normal coordinates in viscous units.
    u_plus : ndarray
        Mean streamwise velocity in viscous units, same length as z_plus.
    cov : ndarray
        Covariance matrix of the u_plus measurements.
    
    RETURNS
    
    opt_stats : Stats
        Statistics object from the optimal GLS fit.
    z_plus_crop : ndarray
        Wall-normal coordinates over the optimal fitting window.
    u_plus_crop : ndarray
        Mean velocity over the optimal fitting window.
    """
    n_min = 5
    z_idl = np.searchsorted(z_plus, 100)
    z_idu = np.searchsorted(z_plus, 0.5 * Re_tau) 

    opt_cost = np.inf
    opt_z_idl, opt_z_idu = None, None
    for idl in range(z_idl, z_idu):
        for idu in range(idl + n_min - 1, z_idu):
            # Crop data
            z_plus_crop = z_plus[idl : idu + 1]
            u_plus_crop = u_plus[idl : idu + 1]
            # Fit
            fit = ft.fit(Re_tau, delta, nu, z_plus_crop, u_plus_crop, cov)
            stats = st.get_stats(fit)
            p_value = stats.p_value
            # Update optimum
            cost = _cost_fun(p_value, stats)
            if cost < opt_cost:
                opt_cost = cost
                opt_z_idl, opt_z_idu = idl, idu

    if opt_z_idl is None:
        print('No valid window found within the search domain.')
    else:
        z_plus_crop = z_plus[opt_z_idl : opt_z_idu + 1]
        u_plus_crop = u_plus[opt_z_idl : opt_z_idu + 1]
        fit = ft.fit(Re_tau, delta, nu, z_plus_crop, u_plus_crop, cov)
        opt_stats = st.get_stats(fit)
        _print_report(opt_stats, opt_z_idl, opt_z_idu, opt_cost, z_plus)
    
    return opt_stats, z_plus_crop, u_plus_crop

def _cost_fun(p_value, stats):
    """
    Evaluate the cost of a GLS log-law fit given its p-value and statistics.
    Penalises fits whose p-value falls outside the acceptable range [0.1, 0.9]
    with an exponential penalty, and otherwise rewards fits with small
    parameter uncertainties, low correlation, and few data points.
    
    PARAMETERS
    
    p_value : float
        Chi-squared p-value of the fit residuals.
    stats : Stats
        Statistics object containing fitted parameter uncertainties,
        their correlation, and the number of data points used.
    
    RETURNS
    
    cost : float
        Scalar cost value.
    """
    p_value_l = 0.1
    p_value_u = 0.9
    if p_value < p_value_l:
        P = np.exp(p_value_l - p_value)
    elif p_value > p_value_u:
        P = np.exp(p_value - p_value_u)
    else:
        P = 0.0
    return (P +
            stats.u_kappa[1] * 
            stats.u_A[1] * 
            (1 - stats.rho_kA ** 2) ** (1/2) *
            stats.n
            )

def _print_report(stats, z_idl, z_idu, cost, z_plus):
    """
    Print a summary report of the optimal GLS log-law fit, including
    the fitted window, derived parameters, and statistical assessment.
    """
    p_value = stats.p_value
    if p_value > 0.9:
        msg = (f'{p_value:.4f} (residuals are much smaller than predicted; '
               f'measurement uncertainties are strongly overestimated)')
    elif p_value > 0.68:
        msg = (f'{p_value:.4f} (residuals are smaller than predicted; '
               f'measurement uncertainties are likely overestimated)')
    elif p_value >= 0.32:
        msg = (f'{p_value:.4f} (residuals are consistent with predictions; '
               f'uncertainty budget is well-specified)')
    elif p_value >= 0.1:
        msg = (f'{p_value:.4f} (residuals are larger than predicted; '
               f'measurement uncertainties are likely underestimated)')
    else:
        msg = (f'{p_value:.4f} --> (residuals are much larger than predicted; '
               f'measurement uncertainties are strongly underestimated)')

    print('\nGLS log-law fit report')
    print(f'  z_idl    : z⁺ = {z_plus[z_idl]:.1f} ({z_idl})')
    print(f'  z_idu    : z⁺ = {z_plus[z_idu]:.1f} ({z_idu})')
    print(f'  A        : {stats.A:.4f} [{stats.u_A[0]:8.4f}, {stats.u_A[1]:8.4f}]')
    print(f'  κ        : {stats.kappa:.4f} [{stats.u_kappa[0]:8.4f}, {stats.u_kappa[1]:8.4f}]')
    print(f'  ρ(κ, A)  : {stats.rho_kA:.4f}')
    print(f'  cost     : {cost:.6f}')
    print(f'  p_value  : {msg}')