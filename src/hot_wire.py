#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author:     Manuel Ferreira
Created:    April 2026
Updated:
Version:    v1.0

Description:
    Generates a wall-normal coordinate vector and constructs covariance
    matrices for both the wall-normal coordinates and velocity measurements
    obtained from hot-wire anemometry calibrated using King's law.

Includes:
    z       Generates wall-normal coordinate vectors in viscous units.
    cov_z   Constructs the covariance matrix for wall-normal coordinates
            acquired via a linear traverse.
    cov_u   Estimates the covariance of velocity measurements for synthetic
            hot-wire data, assuming calibration based on King's law.
"""

import numpy as np

def z(Re_tau, n_log, n_lin=0):
    """
    Generate wall-normal coordinates in viscous units (z⁺) with
    logarithmic spacing near the wall and linear spacing in the outer region.

    PARAMETERS
    
    Re_tau : float
        Friction Reynolds number.
    n_log : int
        Number of points in the logarithmic (near-wall + log layer) region.
    n_lin : int, optional
        Number of points in the linear (outer/wake layer) region (default=0).

    RETURNS
    
    z_plus : ndarray of shape (n_log + n_lin,)
        Wall-normal coordinates in viscous units.
    """
    z_plus_t = 0.3 * Re_tau
    z_plus_log = np.logspace(np.log10(10), np.log10(z_plus_t), n_log)
    z_plus_lin = np.linspace(z_plus_t, Re_tau, n_lin)
    z_plus = np.concatenate([z_plus_log, z_plus_lin[1:]])
    return z_plus

def cov_z(s_z0, s_dz, n):
    """
    Construct covariance matrix for a linear wall-normal traverse.

    PARAMETERS
    
    s_z0 : float
        Standard deviation of the initial position measurement.
    s_dz : float
        Standard deviation of each incremental step.
    n : int
        Number of measurement points (including initial position z_0).

    RETURNS
    
    S_z : ndarray of shape (n, n)
        Covariance matrix representing correlated position uncertainties
        along the linear traverse.

    NOTES
    
    z_i = z_0 + sum_{k=1}^i dz_k
    Covariance: S_ij = s_z0^2 + min(i, j) * s_dz^2
    """
    i, j = np.indices((n, n))
    S_z = s_z0 ** 2 + np.minimum(i, j) * s_dz ** 2
    return S_z

def cov_u(u, e_u, e_q, e_rho, e_v, n=10, A=1.0, B=1.0):
    """
    Compute the covariance matrix of velocity measurements from a 
    simulated hot-wire calibration based on King's law.

    PARAMETERS
    
    u : ndarray of shape (m,)
        Streamwise velocities at which covariance is evaluated.
    e_u : float
        Relative error in the mean velocity estimates
    e_q : float
        Relative standard deviation of dynamic pressure measurements.
    e_rho : float
        Relative standard deviation of air density.
    e_v : float
        Relative standard deviation of hot-wire voltage measurements.
    n : int, optional
        Number of calibration points (default=10).
    A, B : float, optional
        King's law calibration coefficients (default 1.0).

    RETURNS
    
    S_u : ndarray of shape (m, m)
        Covariance matrix of the velocity measurements.
    
    NOTES
    -----
    - Simulates calibration data using v^2 = A + B * u^(1/2).
    - Constructs covariance matrix of fitted parameters beta = [beta_0, beta_1].
    - Propagates uncertainty from calibration to velocity measurements.
    """
    rho = 1.225
    u0 = np.linspace(1, 10, n)
    q = rho * u0 ** 2 / 2
    v = np.sqrt(A + B * u0 ** 0.5)
    s_v = e_v * v
    s_q = e_q * q
    s_rho = e_rho * rho
    # Covariance of primitive variables
    S_theta = np.block([
        [np.diag(s_v ** 2), np.zeros((n, n)), np.zeros((n, 1))],
        [np.zeros((n, n)), np.diag(s_q ** 2), np.zeros((n, 1))],
        [np.zeros((1, n)), np.zeros((1, n)), s_rho ** 2]
    ])
    # Jacobian of transformed measurements
    J_w = np.zeros((2 * n, 2 * n + 1))
    for i in range(n):
        J_w[i, i] = 2 * v[i]  # dy_i/dv_i
        J_w[n + i, n + i] = (1/4) * (2 * q[i] / rho) ** 0.25 / q[i]
        J_w[n + i, 2 * n] = (-1/4) * (2 * q[i] / rho) ** 0.25 / rho
    S_w = J_w @ S_theta @ J_w.T
    # Jacobian of residuals
    J_e = np.zeros((n, 2 * n))
    for i in range(n):
        J_e[i, i] = -1/B
        J_e[i, n + i] = 1
    W = np.linalg.inv(J_e @ S_w @ J_e.T)
    X = np.stack([np.ones(n), v ** 2], axis=1)
    S_beta = np.linalg.inv(X.T @ W @ X)
    # Propagate uncertainty to velocity measurements
    v = np.sqrt(A + B * np.sqrt(u))
    X = np.stack([np.ones(u.size), v ** 2], axis=1)
    S_y = X @ S_beta @ X.T
    S_u_B = np.diag(2 * u ** 0.5) @ S_y @ np.diag(2 * u ** 0.5)
    # Statistical uncertainty component
    S_u_A = np.diag(e_u * u) ** 2

    return S_u_A + S_u_B