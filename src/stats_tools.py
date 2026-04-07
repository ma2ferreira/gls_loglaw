#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author:     Manuel Ferreira
Created:    April 2026
Updated:    
Version:    v1.0

Description:
    Statistical tools for model fitting and uncertainty quantification.
    
Includes:
    Chi-square goodness-of-fit evaluation
    Confidence regions (polar and grid-based methods)
    Confidence intervals for fitted parameters
    Covariance-based uncertainty estimation
'''

import numpy as np
from scipy import stats
from skimage import measure
from scipy.optimize import brentq
from dataclasses import dataclass

@dataclass
class out_stats:
    A: float
    kappa: np.ndarray
    u_A: np.ndarray      
    u_kappa: np.ndarray     
    rho_kA: float
    xi: np.ndarray      
    eta: np.ndarray
    p_value: float
    n: int

# ------------------------------
# Report statistics
# ------------------------------
def get_stats(out_fit):
    '''
    Compute uncertainty statistics from GLS fit output.

    PARAMETERS
    
    out_fit : dataclass with fields
        n : int
            Number of observations
        X : ndarray of shape (n, 2)
            Design matrix [1, ln(z^+)]
        Y : ndarray of shape (n,)
            Dependent variable u^+
        W : ndarray of shape (n, n)
            Weighting matrix (inverse of S_e)
        b : ndarray of shape (2,)
            Regression coefficients [A, 1/kappa]
        S_b : ndarray of shape (2, 2)
            Covariance matrix of regression coefficients

    RETURNS

    out_stats : dataclass with fields
        A : float
            Intercept of log-law (dimensionless)
        kappa : float
            Von Kármán constant (dimensionless)
        u_A : ndarray of shape (2,)
            Uncertainty bounds on A [lower, upper]
        u_kappa : ndarray of shape (2,)
            Uncertainty bounds on κ [lower, upper]
        rho_kA : float
            Correlation coefficient between κ and A
        xi : ndarray
            First parameter (A) coordinates along confidence contour
        eta : ndarray
            Second parameter (1/κ) coordinates along confidence contour
        p_value : float
            p-value for chi-square goodness-of-fit test
    '''
    # Unpack fit output
    n = out_fit.n
    X = out_fit.X
    Y = out_fit.Y
    W = out_fit.W
    b = out_fit.b
    S_b = out_fit.S_b
    # Comput statistics
    A = b[0]
    kappa = 1 / b[1]
    xi, eta = cr_S_b(b, S_b)
    u_A, u_kappa = ci_chi2(b, xi, eta)
    rho_kA = - S_b[0,1] / (S_b[0,0] * S_b[1,1]) ** (1 / 2)
    p_value = chi2_eval(X, Y, W, b, n)[1]
    return out_stats(A,
                     kappa,
                     u_A=u_A,
                     u_kappa=u_kappa,
                     rho_kA=rho_kA,
                     xi=xi,
                     eta=eta,
                     p_value=p_value,
                     n=n)

def report(out_stats):
    '''
    Print a formatted report of fitted parameters and uncertainty statistics.

    PARAMETERS
    
    out_stats : dataclass
        Output from get_stats() containing fitted parameters and uncertainties

    RETURNS
    
    None (prints to console)
    '''
    
    print('\nGLS log-law fit report')
    print(f'  A        : {out_stats.A:.4f} [{out_stats.u_A[0]:8.4f}, {out_stats.u_A[1]:8.4f}]')
    print(f'  κ        : {out_stats.kappa:.4f} [{out_stats.u_kappa[0]:8.4f}, {out_stats.u_kappa[1]:8.4f}]')
    print(f'  ρ(κ, A)  : {out_stats.rho_kA:.4f}')
    if out_stats.p_value > 0.05:
        msg = f'{out_stats.p_value:.4f} (good fit, p > 0.1 and p < 0.9)'
    else:
        msg = f'{out_stats.p_value:.4f} (poor fit, p < 0.1 or p > 0.9)'
    print(f'  p-value  : {msg}')
    
# ------------------------------
# Chi-square functions
# ------------------------------
def chi2_eval(X, Y, W, b, n):
    '''
    Compute the chi-square statistic and associated probabilities for a model fit.

    PARAMETERS
    
    X : ndarray of shape (n, p)
        Design matrix (partial derivatives of the model w.r.t. parameters)
    Y : ndarray of shape (n,)
        Observed dependent variable values
    W : ndarray of shape (n, n)
        Weighting matrix (inverse of variance-covariance of Y)
    b : ndarray of shape (p,)
        Estimated parameter vector
    n : int
        Number of observations

    RETURNS

    chi2 : float
        Chi-square statistic: χ² = (Y - Xb)^T W (Y - Xb)
    P : float
        p-value for chi-square goodness-of-fit test
    Q : float
        Complementary probability (1 - P)
        
    NOTES
    The chi-square statistic measures goodness-of-fit. A large χ² corresponds to 
    a small p-value (P), indicating that the residuals are unlikely under the 
    assumed uncertainty model. Q = 1 - P gives the cumulative probability of 
    observing a chi-square value less than or equal to the observed χ².
    '''
    chi2 = (Y - X @ b).T @ W @ (Y - X @ b)
    P = stats.chi2.sf(chi2, n - 2)
    Q = 1 - P
    return chi2, P, Q

def chi2_map(X, Y, W, XI, ETA, n):
    '''
    Compute chi-square values on a 2D grid of parameter values.

    PARAMETERS
    
    X : ndarray of shape (n, p)
        Design matrix
    Y : ndarray of shape (n,)
        Observed data
    W : ndarray of shape (n, n)
        Weighting matrix
    XI : ndarray of shape (q, q)
        Grid of candidate values for the first parameter
    ETA : ndarray of shape (q, q)
        Grid of candidate values for the second parameter
    n : int
        Number of observations

    RETURNS

    chi2_map : ndarray of shape (q, q)
        Chi-square values evaluated at each grid point
    '''
    chi2_map = np.zeros((XI.shape[0], ETA.shape[1]))
    for i in range(XI.shape[0]):
        for j in range(ETA.shape[1]):
            chi2_map[i, j] = chi2_eval(
                X, Y, W, np.array([XI[i, j], ETA[i, j]]), n)[0]
    return chi2_map

# ------------------------------
# Confidence regions
# ------------------------------
def cr_chi2_polar(X, Y, W, b, S_b, n, n_angles=360, pval=0.05):
    '''
    Compute the chi-square confidence region using polar parametrisation.

    PARAMETERS
    
    X : ndarray of shape (n, p)
        Design matrix
    Y : ndarray of shape (n,)
        Observed data
    W : ndarray of shape (n, n)
        Weighting matrix
    b : ndarray of shape (p,)
        Best-fit parameter vector (center of confidence region)
    S_b : ndarray of shape (p, p)
        Covariance matrix of b
    n : int
        Number of observations
    n_angles : int, optional
        Number of rays to shoot for contour (default 360)
    pval : float, optional
        Significance level for confidence region (default 0.05)

    RETURNS

    xi_out : ndarray of shape (n_angles,)
        First parameter coordinates along the confidence-region contour
    eta_out : ndarray of shape (n_angles,)
        Second parameter coordinates along the confidence-region contour
    '''
    chi2_b = chi2_eval(X, Y, W, b, n)[0]
    chi2_crit = stats.chi2.ppf(1 - pval, df=2)
    thres = chi2_b + chi2_crit
    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    xi_out = np.empty(n_angles)
    eta_out = np.empty(n_angles)
    for i, angle in enumerate(angles):
        d = np.array([np.cos(angle), np.sin(angle)])
        obj = lambda r: chi2_eval(X, Y, W, b + r*d, n)[0] - thres
        r_hi = np.sqrt(chi2_crit * np.max(np.diag(S_b))) * 3
        while obj(r_hi) < 0:
            r_hi *= 2
        r_star = brentq(obj, 1e-3, r_hi, xtol=1e-3, rtol=1e-3)
        xi_out[i] = b[0] + r_star*d[0]
        eta_out[i] = b[1] + r_star*d[1]
    return xi_out, eta_out

def cr_chi2_grid(X, Y, W, XI, ETA, n, pval=0.05):
    '''
    Compute the chi-square confidence region contour using a grid approach.

    PARAMETERS
    
    X : ndarray of shape (n, p)
        Design matrix
    Y : ndarray of shape (n,)
        Observed data
    W : ndarray of shape (n, n)
        Weighting matrix
    XI : ndarray of shape (q, q)
        Grid values for first parameter
    ETA : ndarray of shape (q, q)
        Grid values for second parameter
    n : int
        Number of observations
    pval : float, optional
        Significance level (default 0.05)

    RETURNS

    xi : ndarray
        First parameter coordinates along the confidence contour
    eta : ndarray
        Second parameter coordinates along the confidence contour
    '''
    chi2 = chi2_map(X, Y, W, XI, ETA, n)
    chi2_red = chi2 / stats.chi2.ppf(1 - pval, df=2)
    cs = measure.find_contours(chi2_red, level=1)
    c = cs[0]
    xi = np.interp(c[:, 1], np.arange(chi2_red.shape[1]), XI[0,:])
    eta = np.interp(c[:, 0], np.arange(chi2_red.shape[0]), ETA[:,0])
    return xi, eta

def cr_S_b(b, S_b, pval=0.05):
    '''
    Compute covariance-based confidence ellipse for two regression parameters.

    PARAMETERS
    
    b : ndarray of shape (2,)
        Best-fit parameter vector
    S_b : ndarray of shape (2, 2)
        Covariance matrix of b
    pval : float, optional
        Significance level (default 0.05)

    RETURNS

    x : ndarray
        X-coordinates of confidence ellipse
    y : ndarray
        Y-coordinates of confidence ellipse
    '''
    v, w = np.linalg.eig(S_b)
    width, height = 2 * (v * stats.chi2.ppf(1 - pval, df=2))**0.5
    alpha = np.arctan2(*w[0, ::-1])
    t = np.linspace(0, 2*np.pi, 100)
    cr = np.stack((width*np.cos(t)/2, height*np.sin(t)/2), axis=0)
    rot = np.array([[np.cos(alpha), np.sin(alpha)], [-np.sin(alpha), np.cos(alpha)]])
    cr = rot @ cr + b[:, None]
    return cr[0,:], cr[1,:]

# ------------------------------
# Confidence intervals
# ------------------------------
def ci_chi2(b, xi, eta):
    '''
    Extract marginal confidence intervals from chi-square confidence contour.

    PARAMETERS
    
    b : ndarray of shape (2,)
        Best-fit parameter vector [A, 1/kappa]
    xi : ndarray
        First parameter values along contour (A)
    eta : ndarray
        Second parameter values along contour (1/kappa)

    RETURNS

    u_A : ndarray of shape (2,)
        Uncertainty bounds on A: [lower, upper]
    u_kappa : ndarray of shape (2,)
        Uncertainty bounds on kappa: [lower, upper]
    '''
    u_A = [xi.min(), xi.max()] - b[0]
    u_kappa = [1/eta.max(), 1/eta.min()] - 1 / b[1]
    return u_A, u_kappa

def ci_S_b(b, S_b, pval=0.05):
    '''
    Compute approximate confidence interval half-widths from covariance matrix.
    Transforms parameter uncertainties from regression space [A, 1/κ] to 
    parameter space [A, κ] using the Jacobian of the transformation.

     PARAMETERS
    
    b : ndarray of shape (2,)
        Regression coefficients [A, 1/kappa]
    S_b : ndarray of shape (2, 2)
        Covariance matrix of regression coefficients
    pval : float, optional
        Significance level (default 0.05 for 95% confidence)

    RETURNS

    u_A : float
        Symmetric confidence interval half-width for A
    u_kappa : float
        Symmetric confidence interval half-width for κ
    '''
    var = np.diag(S_b)
    u_b = (var * stats.chi2.ppf(1 - pval, df=2)) ** (1 / 2)
    u_A = u_b[0]
    u_kappa = (1 / b[1] ** 2) * u_b[1]
    return u_A, u_kappa