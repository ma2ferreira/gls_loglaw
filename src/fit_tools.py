#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author:     Manuel Ferreira
Created:    April 2026
Updated:   
Version:    v1.0

Description:
    Least-squares fitting tools for the log law. Implements ordinary, weighted,
    and generalised least-squares (GLS) regression with full uncertainty 
    propagation from primitive variables to the fitted log-law parameters kappa
    and A.

Includes:
    fit             Propagate uncertainties and perform GLS fit of the log law.
    lsq             Weighted or ordinary least-squares fit of the log law.
    cov_gls         Build error covariance matrix via GLS (Sprent, 1966).
    cov_aitken      Build error covariance matrix via Aitken (1933) GLS.
    cov_wls         Build diagonal covariance matrix for WLS.
    cov_phi         Transform regression coefficient covariance to [A, kappa].
    out_fit         Dataclass holding the output of a least-squares fit.

References:
    Aitken, A. C. (1933) On least squares and linear combinations of
    observations. Proceedings of the Royal Society of Edinburgh, 55.
    Sprent, P. (1966) A generalised least-squares approach to linear
    functional relationships. Journal of the Royal Statistical Society, 28.
'''

import numpy as np
import hot_wire as hw
from dataclasses import dataclass

@dataclass
class out_fit:
    n: int
    X: np.ndarray
    Y: np.ndarray
    W: np.ndarray
    S_e: np.ndarray
    b: np.ndarray
    S_b: np.ndarray

# ------------------------------
# Fitting functions
# ------------------------------

def fit(Re_tau, delta, nu, z_plus, u_plus, cov):
    '''
    Perform least-squares fit of the log law
    
    PARAMETERS
    
    Re_tau : float
        Friction Reynolds number (-)
    delta : float
        Boundary layer thickness (m)
    nu : float
        Kinematic viscosity (m^2/s)
    z_plus : ndarray of shape (n,)
        Viscous-scaled wall-normal coordinate (dimensionless)
    u_plus : ndarray of shape (n,)
        Viscous-scaled mean streamwise velocity (dimensionless)
    cov : Cov dataclass
        Uncertainty parameters for primitive variables (e_u, e_u_tau, e_nu, e_q,
        e_rho, e_v, s_z0, s_dz)
    
    RETURNS
    
    out_fit : dataclass with fields
        n : int
            Number of observations
        X : ndarray of shape (n, 2)
            Design matrix [1, ln(z^+)]
        Y : ndarray of shape (n,)
            Dependent variable u^+
        W : ndarray of shape (n, n)
            Weighting matrix (inverse of S_e)
        S_e : ndarray of shape (n, n)
            Covariance matrix of measurement errors
        b : ndarray of shape (2,)
            Regression coefficients [A, 1/kappa]
        S_b : ndarray of shape (2, 2)
            Covariance matrix of regression coefficients
    '''
    # Convert to physical units
    n = u_plus.size
    u_tau = Re_tau * nu / delta
    z = z_plus * nu / u_tau
    u = u_plus * u_tau
    # Unpack uncertainty in the primitive variables
    e_u = cov.e_u
    e_u_tau = cov.e_u_tau
    e_nu = cov.e_nu
    e_q = cov.e_q
    e_rho = cov.e_rho
    e_v = cov.e_v
    s_z0 = cov.s_z0
    s_dz = cov.s_dz
    # Uncertainty propagation
    S_z = hw.cov_z(s_z0, s_dz, n)
    S_u = hw.cov_u(u, e_u, e_q, e_rho, e_v)
    s_u_tau = e_u_tau * u_tau
    s_nu = e_nu * nu
    S_e = cov_gls(z, u, u_tau, nu, S_z, S_u, s_u_tau, s_nu)
    # Least-squares fit
    X, Y, W, b, S_b = lsq(u_plus, z_plus, S_e)

    return out_fit(n=n, X=X, Y=Y, W=W, S_e=S_e, b=b, S_b=S_b)

def lsq(u_plus, z_plus, S_e=None):
    '''
    Perform a (weighted) least-squares fit of the logarithmic law of the wall
    u^+ = (1 / kappa) * ln(z^+) + A, recast in linear form as
    y = beta_0 + beta_1 * x, where y = u^+, x = ln(z^+), beta_0 = A, 
    and beta_1 = 1 / kappa.

    PARAMETERS
    
    u_plus : ndarray of shape (n,)
        Viscous-scaled mean streamwise velocity (dimensionless)
    z_plus : ndarray of shape (n,)
        Viscous-scaled wall-normal coordinate (dimensionless)
    S_e : ndarray of shape (n, n), optional
        Covariance matrix of measurement uncertainties. Default is None.

    RETURNS

    X : ndarray of shape (n, 2)
        Design matrix [1, ln(z^+)]
    Y : ndarray of shape (n,)
        Dependent variable vector u^+
    W : ndarray of shape (n, n)
        Weighting matrix (inverse of S_e)
    b : ndarray of shape (2,)
        Regression coefficients [A, 1/kappa]
    S_b : ndarray of shape (2, 2)
        Covariance matrix of regression coefficients
    '''
    n = u_plus.size
    X = np.column_stack([np.ones(n), np.log(z_plus)])
    Y = u_plus
    W = np.eye(n) if S_e is None else np.linalg.inv(S_e)
    b = np.linalg.inv(X.T @ W @ X) @ (X.T @ W @ Y)
    if S_e is None:
        SSE = (Y - X @ b).T @ (Y - X @ b)
        MSE = SSE / (n - 2)
        S_b = MSE * np.linalg.inv(X.T @ X)
    else:
        S_b = np.linalg.inv(X.T @ W @ X)

    return X, Y, W, b, S_b

# ------------------------------
# Covariance matrix functions
# ------------------------------
def cov_wls(u, u_tau, s_u, s_u_tau):
    '''
    Build variance-covariance matrix for weighted least-squares (WLS).

    Assumes independent, heteroscedastic errors in u^+, while errors in the 
    viscous-scaled wall-normal coordinate (z^plus) are **neglected**. 

    PARAMETERS
    
    u : ndarray of shape (n,)
        Mean streamwise velocity (m/s)
    u_tau : float
        Friction velocity (m/s)
    s_u : ndarray of shape (n,)
        Standard deviation of u measurements (m/s)
    s_u_tau : float
        Standard deviation of u_tau (m/s)

    RETURNS
    
    S_e : ndarray of shape (n, n)
        Variance-covariance matrix of u^+
    rho : ndarray of shape (n, n)
        Correlation matrix corresponding to S_e
    '''
    n = u.size
    S_e = np.zeros((n, n))
    for i in range(n):
        S_e[i, i] = (s_u[i] / u_tau) ** 2 + (u[i] * s_u_tau / u_tau**2) ** 2
    std = np.sqrt(np.diag(S_e))
    rho = S_e / np.outer(std, std)
    return S_e, rho

def cov_aitken(u, u_tau, s_u, s_u_tau):
    '''
    Build variance–covariance matrix according to the generalised least-squares 
    principle formulated by Aitken (1933).

    This function computes the variance–covariance matrix of the measurements
    under the assumption that errors in the viscous-normalised streamwise 
    velocity (u^plus) are **heteroscedastic** and **correlated** with each 
    other, while errors in the viscous-scaled wall-normal coordinate (z^plus) 
    are **neglected**. 
    
    PARAMETERS
    
    u : ndarray of shape (n,)
        Mean streamwise velocity (m/s)
    u_tau : float
        Friction velocity (m/s)
    s_u : ndarray of shape (n,)
        Standard deviation of u measurements (m/s)
    s_u_tau : float
        Standard deviation of u_tau (m/s)
    
    RETURNS
    
    S_e : ndarray of shape (n, n)
        Variance-covariance matrix of u^+
    rho : ndarray of shape (n, n)
        Correlation matrix corresponding to S_e
    '''
    n = u.size
    S_e = np.zeros((n, n))
    for i in range(n):
        S_e[i, i] = (s_u[i] / u_tau) ** 2 + (u[i] * s_u_tau / u_tau**2) ** 2
        for j in range(n):
            if i != j:
                S_e[i, j] = (u[i]*u[j] / u_tau**4) * s_u_tau**2
    std = np.sqrt(np.diag(S_e))
    rho = S_e / np.outer(std, std)
    return S_e, rho

def cov_gls(z, u, u_tau, nu, S_z, S_u, s_u_tau, s_nu):
    '''
    Build variance-covariance matrix according to the generalised least-squares 
    principle formulated by Sprent (1966).

    PARAMETERS
    
    z : ndarray of shape (n,)
        Wall-normal coordinate (m)
    u : ndarray of shape (n,)
        Mean streamwise velocity (m/s)
    u_tau : float
        Friction velocity (m/s)
    nu : float
        Kinematic viscosity (m^2/s)
    S_z : ndarray of shape (n, n)
        Covariance matrix of z
    S_u : ndarray of shape (n, n)
        Covariance matrix of u
    s_u_tau : float
        Standard deviation of u_tau (m/s)
    s_nu : float
        Standard deviation of nu (m^2/s)

    RETURNS

    S_e : ndarray of shape (n, n)
        Variance-covariance matrix of the residuals after propagating errors
    '''
    n = u.size
    S_t = np.block([
        [S_z, np.zeros((n, n + 2))],
        [np.zeros((n, n)), S_u, np.zeros((n, 2))],
        [np.zeros((1, 2 * n)), s_u_tau ** 2, 0],
        [np.zeros((1, 2 * n + 1)), s_nu ** 2]
    ])
    J_w = np.zeros((2*n, 2*n + 2))
    for i in range(n):
        J_w[i, i] = 1.0 / z[i]
        J_w[i, 2*n] = 1.0 / u_tau
        J_w[i, 2*n + 1] = -1.0 / nu
        J_w[n+i, n+i] = 1.0 / u_tau
        J_w[n+i, 2*n] = -u[i] / (u_tau ** 2)
    S_w = J_w @ S_t @ J_w.T
    J_e = np.zeros((n, 2*n))
    for i in range(n):
        J_e[i, i] = -1 / 0.39
        J_e[i, n+i] = 1
    S_e = J_e @ S_w @ J_e.T
    return S_e

def cov_phi(b, S_b):
    '''
    Compute the covariance matrix of the log-law parameters [A, kappa]
    from the covariance of the regression coefficients [A, 1/kappa].

    PARAMETERS
    
    b : ndarray of shape (2,)
        Best-fit regression coefficients [A, 1/kappa]
    S_b : ndarray of shape (2, 2)
        Covariance matrix of regression coefficients

    RETURNS
    
    S_p : ndarray of shape (2, 2)
        Covariance matrix of log-law parameters [A, kappa]
    
    NOTES
    
    Transformation is applied using the Jacobian:
        J = [[1, 0],
             [0, -1 / b[1]^2]]
    where kappa = 1 / beta_1.
    '''
    J_p = np.array([[1, 0], [0, -1 / b[1] ** 2]])
    S_p = J_p @ S_b @ J_p.T
    return S_p