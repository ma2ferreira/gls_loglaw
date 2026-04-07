#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author:     Manuel Ferreira
Created:    April 2026
Updated:   
Version:    v1.0

Description:
    Utilities for zero-pressure-gradient turbulent boundary layers. Includes
    a composite analytical velocity profile that combines the Musker (1979)
    inner-layer formulation with the Chauhan, Nagib & Monkewitz (2009) bump
    correction and a Coles-type wake function, as well as a simple log-law
    velocity profile, and a tool for extracting data from a specified
    log region.

Includes:
    composite       Composite inner + wake velocity profile.
    crop_log        Extract the logarithmic region of a velocity profile.
    loglaw          Evaluate the log law at specified viscous-scaled 
                    wall-normal coordinates.
    loglaw_n        Evaluate the log law over the region 
                    (3 Re_τ ** (1 / 2) ≤ z⁺ ≤ 0.15 Re_τ) using a specified 
                    number of points.
    _musker_inner   Musker (1979) analytical inner-layer profile.
    _chauhan_wake   Chauhan et al. (2009) outer wake function.
    _chauhan_bump   Chauhan et al. (2009) inner-layer bump correction.

References:
    Musker, A. J. (1979) Explicit expression for the smooth wall velocity 
    distribution in a turbulent boundary layer.
    Chauhan, K. A., Nagib, H. M., & Monkewitz, P. A. (2009) Criteria for 
    assessing experiments in zero-pressure-gradient boundary layers. 
    Fluid Dynamics Research, 41.
'''

import numpy as np

__all__ = [
    'composite',
    'crop_log',
    'loglaw',
    'loglaw_n'
]

def _musker_inner(z_plus, kappa=0.384, a=-10.3061):
    '''
    Musker analytical inner-layer velocity profile.

    This explicit formulation reproduces the viscous sublayer behaviour near 
    the wall and transitions smoothly toward the logarithmic region. The 
    parameter values correspond to those reported by Chauhan et al. (2007) for
    high-Reynolds-number boundary layers.

    PARAMETERS
    
    z_plus : float or ndarray
        Viscous-scaled wall-normal coordinate:
    kappa : float, optional
        von Kármán constant. Default is 0.384.
    a : float, optional
        Musker model parameter. Default corresponds to the value
        recommended by Chauhan et al. (2007) for κ = 0.384.

    RETURNS
    
    u_plus : float or ndarray
        Mean streamwise velocity in viscous units.
    '''
    alpha = (-1.0 / kappa - a) / 2.0
    beta = np.sqrt(-2.0 * a * alpha - alpha**2)
    R = np.sqrt(alpha**2 + beta**2)
    t1 = (1.0 / kappa) * np.log((z_plus - a) / -a)
    prefactor = R**2 / (a * (4.0 * alpha - a))
    numerator = (-a / R) * np.sqrt((z_plus - alpha)**2 + beta**2)
    denominator = (z_plus - a)
    t2 = (4.0 * alpha + a) * np.log(numerator / denominator)
    t3 = (alpha / beta) * (4.0 * alpha + 5.0 * a)
    t3 *= np.arctan((z_plus - alpha) / beta) + np.arctan(alpha / beta)
    u_plus = t1 + prefactor * (t2 + t3)
    return u_plus

def _chauhan_wake(eta, Pi=0.45, a1=132.8410, a2=-166.2041, a3=71.9114):
    '''
    Polynomial–exponential wake function from Chauhan, Nagib & Monkewitz (2009).

    PARAMETERS
    
    eta : float or ndarray
        Outer coordinate
    Pi : float, optional
        Wake strength parameter. Default value (0.45) corresponds to
        typical ZPG turbulent boundary layers.
    a1, a2, a3 : float, optional
        Polynomial coefficients fitted by Chauhan et al. to collapse
        high-Reynolds-number experimental and DNS datasets.

    RETURNS
    
    W : float or ndarray
        Value of the normalised wake function.
    '''
    prefactor = 1 - (1 / (2 * Pi)) * np.log(eta)
    numerator = 1 - np.exp(
        -(1 / 4) * (5 * a1 + 6 * a2 + 7 * a3) * eta**4
        + a1 * eta**5
        + a2 * eta**6
        + a3 * eta**7
    )
    denominator = 1 - np.exp(-(1 / 4) * (a1 + 2 * a2 + 3 * a3))
    return prefactor * numerator / denominator

def _chauhan_bump(z_plus, m1=30, m2=2.85):
    '''
    Inner-layer bump correction.

    PARAMETERS
    
    z_plus : float or ndarray
        Viscous wall coordinate
    m1 : float, optional
        Empirical constant controlling the location of the bump.
    m2 : float, optional
        Empirical constant controlling the bump magnitude.

    RETURNS
    
    bump : float or ndarray
        Value of the bump correction.
    '''
    return np.exp(-np.log(z_plus / m1)) / m2

def composite(z_plus, Re_tau, kappa=0.384, Pi=0.45):
    '''
    Cmposite velocity profile for a zero-pressure-gradient turbulent
    boundary layer combining:
        - Musker inner-law
        - Chauhan inner-layer bump correction
        - Coles-type wake function
    
    PARAMETERS
    
    z_plus : ndarray
        Wall-normal coordinate in viscous units.
    Re_tau : float
        Friction Reynolds number Re_tau = delta * u_tau / nu
    kappa : float, optional
        von Kármán constant.
    Pi : float, optional
        Wake strength parameter.

    RETURNS
    
    u_plus : ndarray
        Composite velocity profile in viscous units.
    '''
    u_plus = (
        _musker_inner(z_plus, kappa)
        + _chauhan_bump(z_plus)
        + (2 * Pi / kappa) * _chauhan_wake(z_plus / Re_tau)
    )
    u_inf_plus = (
        _musker_inner(Re_tau, kappa)
        + _chauhan_bump(z_plus)
        + (2 * Pi / kappa) * _chauhan_wake(1, Pi)
    )
    u_plus = np.minimum(u_plus, u_inf_plus)
    return u_plus

def crop_log(z_plus, u_plus, z_plus_min, z_plus_max):
    '''
    Extract the logarithmic region of a velocity profile.

    PARAMETERS
    
    z_plus : ndarray
        Wall-normal coordinate in viscous units.
    u_plus : ndarray
        Mean velocity profile in viscous units.
    z_plus_min : float
        Lower bound of the logarithmic region.
    z_plus_max : float
        Upper bound of the logarithmic region.

    RETURNS
    
    z_crop : ndarray
        z_plus values within the specified range.
    u_crop : ndarray
        Corresponding velocity values.
    n : int
        Number of points in the cropped region.
    '''
    mask = (z_plus >= z_plus_min) & (z_plus <= z_plus_max)
    z_crop = z_plus[mask]
    u_crop = u_plus[mask]
    n = z_crop.size
    return z_crop, u_crop, n

def loglaw(z_plus, Re_tau, kappa=0.384, A=4.17):
    '''
    Logarithmic law of the wall.

    PARAMETERS
    
    z_plus : float or ndarray
        Viscous-scaled wall-normal coordinate.
    Re_tau : float
        Friction Reynolds number (currently unused).
    kappa : float, optional
        von Kármán constant.
    B : float, optional
        Log-law intercept constant.

    RETURNS
    
    u_plus : ndarray
        Velocity profile in viscous units.
    '''
    u_plus = (1 / kappa) * np.log(z_plus) + A
    return u_plus

def loglaw_n(Re_tau, n, kappa=0.384, A=4.17):
    '''
    Generate synthetic log-law data spanning
    3 * sqrt(Re_tau) ≤ z+ ≤ 0.15 * Re_tau (Marusic et al., 2013)

    PARAMETERS
    
    Re_tau : float
        Friction Reynolds number.
    n : int
        Number of sample points.
    kappa : float, optional
        von Kármán constant.
    B : float, optional
        Log-law intercept constant.

    RETURNS
    
    z_plus : ndarray
        Viscous-scaled wall coordinate.
    u_plus : ndarray
        Velocity profile following the logarithmic law.
    '''
    z_plus = np.logspace(
        np.log10(3 * np.sqrt(Re_tau)),
        np.log10(0.15 * Re_tau),
        n,
    )
    u_plus = (1 / kappa) * np.log(z_plus) + A
    return z_plus, u_plus