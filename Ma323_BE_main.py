# -*- coding: utf-8 -*-
"""
MA323 – Numerical Experiments for 1D Convection and Convection–Diffusion

This script runs numerical experiments associated with the MA323 practical work.
It compares different finite difference schemes with the exact solution and
illustrates the influence of diffusion and boundary conditions.

@author: Naacide
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

from Ma323_BE_lib_2 import (
    gaussian_initial_condition,
    sol_exact,
    Conv1D_schemaC,
    Conv1D_schemaA,
    Conv1D_schemaCN,
    Conv_diff1D_schemaCN,
)


def animate_solutions_subplots(
    x: np.ndarray,
    solutions: List[Tuple[str, np.ndarray]],
    y_limits=(-0.5, 0.5),
    step: int = 1,
) -> None:
    """
    Animate multiple solutions simultaneously using subplots.

    Parameters
    ----------
    x : np.ndarray
        Spatial grid.
    solutions : list of (str, np.ndarray)
        List of (title, solution matrix) pairs.
    y_limits : tuple, optional
        Limits for the y-axis.
    step : int, optional
        Time step increment for the animation.
    """
    n_plots = len(solutions)
    fig, axes = plt.subplots(n_plots, 1, figsize=(15, 4 * n_plots), sharex=True)

    if n_plots == 1:
        axes = [axes]

    lines = []
    for ax, (title, U) in zip(axes, solutions):
        line, = ax.plot(x, U[:, 0], "r-")
        ax.set_title(title)
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(*y_limits)
        lines.append((line, U))

    plt.show(block=False)
    fig.canvas.draw()

    max_time_steps = min(U.shape[1] for _, U in solutions)

    for n in range(1, max_time_steps, step):
        for (line, U), ax in zip(lines, axes):
            line.set_ydata(U[:, n])
            ax.draw_artist(ax.patch)
            ax.draw_artist(line)
        fig.canvas.draw()
        plt.pause(0.01)


# =============================================================================
# Main execution
# =============================================================================

def main() -> None:
    """
    Run all numerical experiments for the MA323 practical work.
    """
    # -------------------------------------------------------------------------
    # Global parameters
    # -------------------------------------------------------------------------
    length = 50.0
    final_time = 25.0
    velocity = 1.0
    diffusion = 1.0

    dx = 0.1
    dt = 0.025

    M = int(length / dx) - 1
    N = int(final_time / dt) - 1

    x = np.linspace(0.0, length, M + 2)

    # Gaussian initial condition parameters
    mean = 20.0
    sigma = 1.0

    u0_values = gaussian_initial_condition(x, sigma, mean).reshape(-1, 1)

    # -------------------------------------------------------------------------
    # Exact solution
    # -------------------------------------------------------------------------
    U_exact = sol_exact(dx, u0_values, velocity, dt)

    # -------------------------------------------------------------------------
    # Convection equation
    # -------------------------------------------------------------------------
    U_centered = Conv1D_schemaC(velocity, u0_values, length, final_time, M, N)
    U_upwind = Conv1D_schemaA(velocity, u0_values, length, final_time, M, N)
    U_cn = Conv1D_schemaCN(velocity, u0_values, length, final_time, M, N)

    animate_solutions_subplots(
        x,
        [
            ("Exact solution", U_exact),
            ("Explicit centered scheme", U_centered),
            ("Upwind scheme", U_upwind),
            ("Crank–Nicolson scheme", U_cn),
        ],
    )

    # -------------------------------------------------------------------------
    # Convection–diffusion equation
    # -------------------------------------------------------------------------
    U_cd = Conv_diff1D_schemaCN(
        velocity, diffusion, u0_values, length, final_time, M, N
    )

    animate_solutions_subplots(
        x,
        [
            ("Convection–Diffusion (Crank–Nicolson)", U_cd),
        ],
        y_limits=(-0.1, 0.6),
        step=5,
    )


if __name__ == "__main__":
    main()
