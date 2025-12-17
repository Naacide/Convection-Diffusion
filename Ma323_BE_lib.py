# -*- coding: utf-8 -*-
"""
MA323 – Numerical Study of 1D Convection and Convection–Diffusion Equations

This module implements several finite difference schemes for the numerical
resolution of:
    - the 1D linear convection equation
    - the 1D convection–diffusion equation


@author: Naacide
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import solve


# =============================================================================
# Utility functions
# =============================================================================

def gaussian_initial_condition(x: np.ndarray, sigma: float, mean: float) -> np.ndarray:
    """
    Compute a Gaussian initial condition.

    Parameters
    ----------
    x : np.ndarray
        Spatial grid.
    sigma : float
        Standard deviation of the Gaussian.
    mean : float
        Mean of the Gaussian.

    Returns
    -------
    np.ndarray
        Initial condition sampled on the spatial grid.
    """
    return (1.0 / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(-((x - mean) ** 2) / (2.0 * sigma ** 2))


# =============================================================================
# Exact solution
# =============================================================================

def sol_exact(dx: float, u0: np.ndarray, velocity: float, dt: float) -> np.ndarray:
    """
    Compute the exact solution of the 1D linear convection equation:
        u(t, x) = u0(x - v t)

    Homogeneous Dirichlet boundary conditions are assumed.
    """

    M_plus_2 = u0.shape[0]
    M = M_plus_2 - 2

    # Number of time steps until the signal exits the domain
    max_shift = int(np.floor(M * dx / (velocity * dt)))
    N_plus_2 = max_shift + 2

    solution = np.zeros((M_plus_2, N_plus_2))

    for n in range(N_plus_2):
        shift = int(round(velocity * n * dt / dx))

        if shift < M:
            solution[1 + shift : M + 1, n] = u0[1 : M + 1 - shift, 0]

    return solution



# =============================================================================
# Convection equation solvers
# =============================================================================

def Conv1D_schemaC(velocity: float, u0: np.ndarray, length: float, final_time: float,
                   M: int, N: int) -> np.ndarray:
    """
    Solve the 1D convection equation using the explicit centered scheme.

    Parameters
    ----------
    velocity : float
        Constant convection velocity (v > 0).
    u0 : np.ndarray
        Initial condition including boundary values.
    length : float
        Length of the spatial domain.
    final_time : float
        Final simulation time.
    M : int
        Number of interior spatial points.
    N : int
        Number of time steps.

    Returns
    -------
    np.ndarray
        Numerical solution matrix of shape (M + 2, N + 2).
    """
    dx = length / (M + 1)
    dt = final_time / (N + 1)
    beta = velocity * dt / (2.0 * dx)

    I = np.eye(M)
    Ac = -beta * np.diag(np.ones(M - 1), -1) + beta * np.diag(np.ones(M - 1), 1)

    U = np.zeros((M + 2, N + 2))
    U[1:M + 1, 0] = u0[1:M + 1, 0]

    for n in range(1, N + 2):
        U[1:M + 1, n] = (I - Ac) @ U[1:M + 1, n - 1]

    return U


def Conv1D_schemaA(velocity: float, u0: np.ndarray, length: float, final_time: float,
                   M: int, N: int) -> np.ndarray:
    """
    Solve the 1D convection equation using the explicit upwind scheme.

    Parameters
    ----------
    velocity : float
        Constant convection velocity (v > 0).
    u0 : np.ndarray
        Initial condition including boundary values.
    length : float
        Length of the spatial domain.
    final_time : float
        Final simulation time.
    M : int
        Number of interior spatial points.
    N : int
        Number of time steps.

    Returns
    -------
    np.ndarray
        Numerical solution matrix of shape (M + 2, N + 2).
    """
    dx = length / (M + 1)
    dt = final_time / (N + 1)
    beta = velocity * dt / dx

    I = np.eye(M)
    Ad = -beta * np.diag(np.ones(M - 1), -1) + beta * np.eye(M)

    U = np.zeros((M + 2, N + 2))
    U[1:M + 1, 0] = u0[1:M + 1, 0]

    for n in range(1, N + 2):
        U[1:M + 1, n] = (I - Ad) @ U[1:M + 1, n - 1]

    return U


def Conv1D_schemaCN(velocity: float, u0: np.ndarray, length: float, final_time: float,
                    M: int, N: int) -> np.ndarray:
    """
    Solve the 1D convection equation using the Crank–Nicolson scheme.

    Parameters
    ----------
    velocity : float
        Constant convection velocity (v > 0).
    u0 : np.ndarray
        Initial condition including boundary values.
    length : float
        Length of the spatial domain.
    final_time : float
        Final simulation time.
    M : int
        Number of interior spatial points.
    N : int
        Number of time steps.

    Returns
    -------
    np.ndarray
        Numerical solution matrix of shape (M + 2, N + 2).
    """
    dx = length / (M + 1)
    dt = final_time / (N + 1)
    beta = velocity * dt / (4.0 * dx)

    Acn = -beta * np.diag(np.ones(M - 1), -1) + beta * np.diag(np.ones(M - 1), 1)
    A = np.eye(M) + Acn
    B = np.eye(M) - Acn

    U = np.zeros((M + 2, N + 2))
    U[1:M + 1, 0] = u0[1:M + 1, 0]

    for n in range(1, N + 2):
        U[1:M + 1, n] = solve(A, B @ U[1:M + 1, n - 1])

    return U


# =============================================================================
# Convection–diffusion equation solvers
# =============================================================================

def Conv_diff1D_schemaCN(velocity: float, nu: float, u0: np.ndarray, length: float,
                         final_time: float, M: int, N: int) -> np.ndarray:
    """
    Solve the 1D convection–diffusion equation using the Crank–Nicolson scheme
    with homogeneous Dirichlet boundary conditions.

    Parameters
    ----------
    velocity : float
        Constant convection velocity.
    nu : float
        Diffusion coefficient.
    u0 : np.ndarray
        Initial condition.
    length : float
        Length of the spatial domain.
    final_time : float
        Final simulation time.
    M : int
        Number of interior spatial points.
    N : int
        Number of time steps.

    Returns
    -------
    np.ndarray
        Numerical solution matrix of shape (M + 2, N + 2).
    """
    dx = length / (M + 1)
    dt = final_time / (N + 1)

    a = nu * dt / dx ** 2
    b = velocity * dt / (4.0 * dx)

    A = (
        b * np.diag(np.ones(M - 1), 1)
        - b * np.diag(np.ones(M - 1), -1)
        + a * np.eye(M)
    )

    B = a * (
        0.5 * np.diag(np.ones(M - 1), 1)
        + 0.5 * np.diag(np.ones(M - 1), -1)
    )

    C = np.eye(M) + A - B
    D = np.eye(M) - A + B

    U = np.zeros((M + 2, N + 2))
    U[1:M + 1, 0] = u0[:M, 0]

    for n in range(1, N + 2):
        U[1:M + 1, n] = solve(C, D @ U[1:M + 1, n - 1])

    return U
