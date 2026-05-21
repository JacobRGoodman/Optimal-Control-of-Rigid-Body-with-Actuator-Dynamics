#!/usr/bin/env python3
"""
Simulate SO(3) attitude tracking with first-order torque actuation.

VERSION: ieee-two-figure-2026-05-20

Additions relative to the earlier tracking script
-------------------------------------------------
1) Metric-weighted diagnostics:
       ||e_R||_J, ||grad_J Psi||_J, ||e_Omega||_J,
       ||tau||_J, ||tau_d||_J, ||M||_J, ||u||_J.

2) Nominal optimal-control diagnostics for the desired trajectory:
       u_d^{bi}   = ddotOmega_d + 1/2 Omega_d x dotOmega_d,
       u_d^{left} = D_t^J A_d, where
                    A_d = dotOmega_d + Gamma_J(Omega_d, Omega_d).

   The left-invariant physical norm is ||u||_J = sqrt(u^T J u).
   Torque, torque-rate, and actuator-command diagnostics are plotted with the physical J-norm by convention.

3) Optional two-trajectory comparison mode. The curves for both nominal
   trajectories are placed on the same comparison plots.

Example
--------

Compare Riemannian cubics and Riemannian quintic in tension:

    python simulate_so3_actuated_tracking.py `
  desired_rigid_quintic_natural.npz `
  --trajectory2 desired_rigid_cubic.npz `
  --label1 "Quintic in tension" `
  --label2 "Cubic" `
  --traj1-metric left `
  --traj2-metric left `
  --out compare_quintic_cubic_ieee `
  --R0-rotvec-deg 0 25 0 `
  --Omega0-offset 0.3 -0.2 0.1 `
  --tau0-mode zero
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, Slerp

# ============================================================
# 1) Constants
# ============================================================

J = np.diag([0.082, 0.0845, 0.1377])  # inertia tensor
Jinv = np.diag([1 / 0.082, 1 / 0.0845, 1 / 0.1377])


def set_inertia(diagonal_entries: np.ndarray | None) -> None:
    """Set module-level inertia tensor used by all geometric computations."""
    global J, Jinv
    if diagonal_entries is None:
        return
    vals = np.asarray(diagonal_entries, dtype=float).reshape(3)
    if np.any(vals <= 0):
        raise ValueError("All inertia entries must be positive.")
    J = np.diag(vals)
    Jinv = np.diag(1.0 / vals)


# ============================================================
# 2) Linear algebra helpers
# ============================================================


def vee(A: np.ndarray) -> np.ndarray:
    """Inverse of hat for a skew-symmetric 3x3 matrix."""
    A = np.asarray(A, dtype=float)
    return np.array([A[2, 1], A[0, 2], A[1, 0]], dtype=float)


def hat(x: np.ndarray) -> np.ndarray:
    """Hat map: R^3 -> so(3), with hat(x) y = x cross y."""
    x = np.asarray(x, dtype=float).reshape(3)
    return np.array(
        [[0.0, -x[2], x[1]], [x[2], 0.0, -x[0]], [-x[1], x[0], 0.0]],
        dtype=float,
    )


def so3_exp(phi: np.ndarray) -> np.ndarray:
    """SO(3) exponential from a rotation vector."""
    return Rotation.from_rotvec(np.asarray(phi, dtype=float).reshape(3)).as_matrix()


def project_to_so3(A: np.ndarray) -> np.ndarray:
    """Closest rotation matrix to A in Frobenius norm."""
    A = np.asarray(A, dtype=float)
    if not np.isfinite(A).all():
        raise ValueError("Cannot project to SO(3): matrix contains NaN or Inf.")
    U, _, Vt = np.linalg.svd(A)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1.0
        R = U @ Vt
    return R


def saturate_vector(u: np.ndarray, u_max: float | None = None) -> np.ndarray:
    u = np.asarray(u, dtype=float).reshape(3)
    if u_max is None:
        return u
    n = np.linalg.norm(u)
    if n <= u_max or n <= 1e-15:
        return u
    return (u_max / n) * u


def Gamma_J(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Left-trivialized Levi-Civita connection coefficient for the left-invariant
    rigid-body metric determined by J.

    This returns Gamma_J(x,y) = nabla_x^so(3) y for constant Lie algebra
    vectors x,y.
    """
    x = np.asarray(x, dtype=float).reshape(3)
    y = np.asarray(y, dtype=float).reshape(3)
    return 0.5 * np.cross(x, y) + 0.5 * Jinv @ (
        np.cross(x, J @ y) + np.cross(y, J @ x)
    )


def LC_Lie(x: np.ndarray, y: np.ndarray, ydot: np.ndarray) -> np.ndarray:
    """
    Left-trivialized covariant derivative for the left-invariant metric.

    Returns D_t y = ydot + Gamma_J(x,y), where x is the body velocity.
    """
    ydot = np.asarray(ydot, dtype=float).reshape(3)
    return ydot + Gamma_J(x, y)


def Gamma_bi(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Bi-invariant connection coefficient Gamma(x,y)=1/2 x cross y."""
    x = np.asarray(x, dtype=float).reshape(3)
    y = np.asarray(y, dtype=float).reshape(3)
    return 0.5 * np.cross(x, y)


def weighted_norm(v: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Return sqrt(v^T W v) for shape (3,) or row-wise for shape (N,3)."""
    v = np.asarray(v, dtype=float)
    if v.ndim == 1:
        val = float(v.T @ W @ v)
        return np.sqrt(max(val, 0.0))
    vals = np.einsum("ni,ij,nj->n", v, W, v)
    return np.sqrt(np.maximum(vals, 0.0))


def attitude_distance_angle(R: np.ndarray, Rd: np.ndarray) -> float:
    """Geodesic angle of Rd^T R under the usual SO(3) logarithm."""
    C = Rd.T @ R
    c = 0.5 * (np.trace(C) - 1.0)
    return float(np.arccos(np.clip(c, -1.0, 1.0)))


def attitude_errors(
    R: np.ndarray, Rd: np.ndarray, Omega: np.ndarray, Omegad: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Return (e_R, e_Omega, Psi)."""
    Q = R.T @ Rd
    e_R = 0.5 * vee(Q.T - Q)  # equivalent to 0.5 vee(Rd^T R - R^T Rd)
    e_Omega = Omega - Q @ Omegad
    Psi = 0.5 * np.trace(np.eye(3) - Rd.T @ R)
    return e_R, e_Omega, float(Psi)


# ============================================================
# 3) Desired trajectory interpolation and nominal u diagnostics
# ============================================================


def _first_key(data: np.lib.npyio.NpzFile, keys: Tuple[str, ...]) -> str:
    for key in keys:
        if key in data:
            return key
    raise KeyError(
        f"None of these keys were found in the .npz file: {keys}. "
        f"Available keys: {list(data.keys())}"
    )


@dataclass
class DesiredTrajectory:
    t: np.ndarray
    slerp_R: Slerp
    Omega_spline: CubicSpline
    dOmega_spline: CubicSpline
    ddOmega_spline: CubicSpline
    source_path: str = ""

    @property
    def t0(self) -> float:
        return float(self.t[0])

    @property
    def tf(self) -> float:
        return float(self.t[-1])

    def eval(self, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return R_d, Omega_d, dotOmega_d, ddotOmega_d at time t."""
        t_clip = float(np.clip(t, self.t0, self.tf))
        Rd = self.slerp_R([t_clip]).as_matrix()[0]
        Omegad = np.asarray(self.Omega_spline(t_clip), dtype=float).reshape(3)
        dOmegad = np.asarray(self.dOmega_spline(t_clip), dtype=float).reshape(3)
        ddOmegad = np.asarray(self.ddOmega_spline(t_clip), dtype=float).reshape(3)
        return Rd, Omegad, dOmegad, ddOmegad


def load_desired_npz(path: str | Path) -> DesiredTrajectory:
    data = np.load(path)
    t_key = _first_key(data, ("t", "time"))
    R_key = _first_key(data, ("R_d", "Rd"))
    Om_key = _first_key(data, ("Omega_d", "Omegad", "Om_d", "Omd"))
    dOm_key = _first_key(data, ("dotOmega_d", "Omega_dot_d", "dOmega_d", "dOmegad", "alpha_d"))
    ddOm_key = _first_key(data, ("ddotOmega_d", "Omega_ddot_d", "ddOmega_d", "ddOmegad", "jerk_Omega_d"))

    t = np.asarray(data[t_key], dtype=float).reshape(-1)
    if not np.isfinite(t).all():
        raise ValueError("The time vector contains NaN or Inf.")
    if np.any(np.diff(t) <= 0):
        raise ValueError("The time vector must be strictly increasing.")

    Rd_raw = np.asarray(data[R_key], dtype=float)
    if Rd_raw.shape != (len(t), 3, 3):
        raise ValueError(f"{R_key} must have shape (N,3,3); got {Rd_raw.shape}.")
    if not np.isfinite(Rd_raw).all():
        raise ValueError(f"{R_key} contains NaN or Inf. The nominal trajectory solve likely failed.")
    Rd = np.stack([project_to_so3(Rd_raw[i]) for i in range(len(t))], axis=0)

    Omd = np.asarray(data[Om_key], dtype=float)
    dOmd = np.asarray(data[dOm_key], dtype=float)
    ddOmd = np.asarray(data[ddOm_key], dtype=float)
    for name, arr in [(Om_key, Omd), (dOm_key, dOmd), (ddOm_key, ddOmd)]:
        if arr.shape != (len(t), 3):
            raise ValueError(f"{name} must have shape (N,3); got {arr.shape}.")
        if not np.isfinite(arr).all():
            raise ValueError(f"{name} contains NaN or Inf. The nominal trajectory solve likely failed.")

    rotations = Rotation.from_matrix(Rd)
    slerp_R = Slerp(t, rotations)

    return DesiredTrajectory(
        t=t,
        slerp_R=slerp_R,
        Omega_spline=CubicSpline(t, Omd, axis=0),
        dOmega_spline=CubicSpline(t, dOmd, axis=0),
        ddOmega_spline=CubicSpline(t, ddOmd, axis=0),
        source_path=str(path),
    )


def nominal_u_biinvariant(Omega: np.ndarray, dOmega: np.ndarray, ddOmega: np.ndarray) -> np.ndarray:
    """
    u_d = (R_d^T nabla^2_{Rdot_d} Rdot_d)^vee for the bi-invariant metric.

    With nabla_x y = 1/2 x cross y, the acceleration is dOmega because
    Omega x Omega = 0, and u = ddOmega + 1/2 Omega x dOmega.
    """
    return np.asarray(ddOmega, dtype=float).reshape(3) + Gamma_bi(Omega, dOmega)


def nominal_u_left_invariant(Omega: np.ndarray, dOmega: np.ndarray, ddOmega: np.ndarray) -> np.ndarray:
    """
    u_d = (R_d^T nabla^2_{Rdot_d} Rdot_d)^vee for the left-invariant
    rigid-body metric determined by J.

    A = nabla_{Rdot} Rdot in body coordinates
      = dotOmega + Gamma_J(Omega,Omega)

    u = D_t A = dotA + Gamma_J(Omega,A)
    dotA = ddotOmega + Gamma_J(dotOmega,Omega) + Gamma_J(Omega,dotOmega)
    because Gamma_J is bilinear with constant coefficients.
    """
    Omega = np.asarray(Omega, dtype=float).reshape(3)
    dOmega = np.asarray(dOmega, dtype=float).reshape(3)
    ddOmega = np.asarray(ddOmega, dtype=float).reshape(3)
    A = dOmega + Gamma_J(Omega, Omega)
    A_dot = ddOmega + Gamma_J(dOmega, Omega) + Gamma_J(Omega, dOmega)
    return A_dot + Gamma_J(Omega, A)


def nominal_cov_accel_left(Omega: np.ndarray, dOmega: np.ndarray) -> np.ndarray:
    return np.asarray(dOmega, dtype=float).reshape(3) + Gamma_J(Omega, Omega)


# ============================================================
# 4) Control design
# ============================================================


@dataclass
class ControlParams:
    k_R: float = 8.81
    k_Omega: float = 2.54
    k_tau: float = 15.0
    C: float = 0.05
    K: float = 1.0
    M_max: float | None = None


def lee_tau_d(t: float, R: np.ndarray, Omega: np.ndarray, traj: DesiredTrajectory, params: ControlParams) -> np.ndarray:
    """Lee-Leok-McClamroch attitude tracking torque, specialized to SO(3)."""
    Rd, Omegad, dOmegad, _ = traj.eval(t)
    Q = R.T @ Rd
    e_R, e_Omega, _ = attitude_errors(R, Rd, Omega, Omegad)
    feedforward = hat(Omega) @ Q @ Omegad - Q @ dOmegad
    return (
        -params.k_R * e_R
        - params.k_Omega * e_Omega
        + np.cross(Omega, J @ Omega)
        - J @ feedforward
    )


def lee_tau_d_and_dot(
    t: float,
    R: np.ndarray,
    Omega: np.ndarray,
    tau: np.ndarray,
    traj: DesiredTrajectory,
    params: ControlParams,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray | float]]:
    """
    Return tau_d and its total derivative along the current closed-loop state.

    The derivative uses the supplied desired ddotOmega_d and the actual
    Omegadot determined by the current applied torque tau.
    """
    Rd, Omegad, dOmegad, ddOmegad = traj.eval(t)
    Q = R.T @ Rd
    Qdot = -hat(Omega) @ Q + Q @ hat(Omegad)

    e_R, e_Omega, Psi = attitude_errors(R, Rd, Omega, Omegad)

    Omega_dot = Jinv @ (np.cross(J @ Omega, Omega) + tau)

    e_R_dot = 0.5 * vee(Qdot.T - Qdot)
    e_Omega_dot = Omega_dot - Qdot @ Omegad - Q @ dOmegad

    cross_term = np.cross(Omega, J @ Omega)
    cross_term_dot = np.cross(Omega_dot, J @ Omega) + np.cross(Omega, J @ Omega_dot)

    ff = hat(Omega) @ Q @ Omegad - Q @ dOmegad
    ff_dot = (
        hat(Omega_dot) @ Q @ Omegad
        + hat(Omega) @ Qdot @ Omegad
        + hat(Omega) @ Q @ dOmegad
        - Qdot @ dOmegad
        - Q @ ddOmegad
    )

    tau_d = -params.k_R * e_R - params.k_Omega * e_Omega + cross_term - J @ ff
    tau_d_dot = -params.k_R * e_R_dot - params.k_Omega * e_Omega_dot + cross_term_dot - J @ ff_dot

    u_d_bi = nominal_u_biinvariant(Omegad, dOmegad, ddOmegad)
    u_d_left = nominal_u_left_invariant(Omegad, dOmegad, ddOmegad)
    A_d_left = nominal_cov_accel_left(Omegad, dOmegad)

    aux = {
        "Rd": Rd,
        "Omegad": Omegad,
        "dOmegad": dOmegad,
        "ddOmegad": ddOmegad,
        "Q": Q,
        "e_R": e_R,
        "e_Omega": e_Omega,
        "Psi": Psi,
        "Omega_dot": Omega_dot,
        "tau_d": tau_d,
        "tau_d_dot": tau_d_dot,
        "u_d_bi": u_d_bi,
        "u_d_left": u_d_left,
        "A_d_left": A_d_left,
    }
    return tau_d, tau_d_dot, aux


def rhs(
    t: float,
    R: np.ndarray,
    Omega: np.ndarray,
    tau: np.ndarray,
    traj: DesiredTrajectory,
    params: ControlParams,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray | float]]:
    """Closed-loop right-hand side for Omega and tau."""
    tau_d, tau_d_dot, aux = lee_tau_d_and_dot(t, R, Omega, tau, traj, params)
    e_tau = tau - tau_d

    # First-order actuator feedback. Without saturation, this implies
    # tau_dot = tau_d_dot - k_tau e_tau.
    M_unsat = (tau + params.C * tau_d_dot - params.C * params.k_tau * e_tau) / params.K
    M = saturate_vector(M_unsat, params.M_max)

    tau_dot = (params.K * M - tau) / params.C
    Omega_dot = np.asarray(aux["Omega_dot"], dtype=float)

    # Induced triple-integrator input in the left-trivialized covariant model.
    u = LC_Lie(Omega, Jinv @ tau, Jinv @ tau_dot)

    aux.update({"M": M, "M_unsat": M_unsat, "e_tau": e_tau, "tau_dot": tau_dot, "u": u})
    return Omega_dot, tau_dot, aux


# ============================================================
# 5) Lie-group integration
# ============================================================


def lie_rk4_step(
    t: float,
    h: float,
    R: np.ndarray,
    Omega: np.ndarray,
    tau: np.ndarray,
    traj: DesiredTrajectory,
    params: ControlParams,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    One fixed-step Lie-group RK4/Crouch-Grossman style update.

    R is reconstructed by products of exponentials, so it remains in SO(3)
    up to roundoff. Omega and tau use the classical RK4 tableau.
    """
    Om1 = Omega
    k1_Om, k1_tau, _ = rhs(t, R, Omega, tau, traj, params)

    R2 = R @ so3_exp(0.5 * h * Om1)
    Om2 = Omega + 0.5 * h * k1_Om
    tau2 = tau + 0.5 * h * k1_tau
    k2_Om, k2_tau, _ = rhs(t + 0.5 * h, R2, Om2, tau2, traj, params)

    R3 = R @ so3_exp(0.5 * h * Om2)
    Om3 = Omega + 0.5 * h * k2_Om
    tau3 = tau + 0.5 * h * k2_tau
    k3_Om, k3_tau, _ = rhs(t + 0.5 * h, R3, Om3, tau3, traj, params)

    R4 = R @ so3_exp(h * Om3)
    Om4 = Omega + h * k3_Om
    tau4 = tau + h * k3_tau
    k4_Om, k4_tau, _ = rhs(t + h, R4, Om4, tau4, traj, params)

    Omega_next = Omega + (h / 6.0) * (k1_Om + 2.0 * k2_Om + 2.0 * k3_Om + k4_Om)
    tau_next = tau + (h / 6.0) * (k1_tau + 2.0 * k2_tau + 2.0 * k3_tau + k4_tau)

    # Product-of-exponentials reconstruction. This is group preserving; the
    # final projection only removes accumulated floating-point drift.
    R_next = (
        R
        @ so3_exp((h / 6.0) * Om1)
        @ so3_exp((h / 3.0) * Om2)
        @ so3_exp((h / 3.0) * Om3)
        @ so3_exp((h / 6.0) * Om4)
    )
    R_next = project_to_so3(R_next)
    return R_next, Omega_next, tau_next


def collect_aux(
    t: float,
    R: np.ndarray,
    Omega: np.ndarray,
    tau: np.ndarray,
    traj: DesiredTrajectory,
    params: ControlParams,
) -> Dict[str, np.ndarray | float]:
    _, _, aux = rhs(t, R, Omega, tau, traj, params)
    return aux


def simulate(
    traj: DesiredTrajectory,
    params: ControlParams,
    dt: float,
    R0: np.ndarray | None = None,
    Omega0: np.ndarray | None = None,
    tau0: np.ndarray | None = None,
) -> Dict[str, np.ndarray]:
    if dt <= 0:
        raise ValueError("dt must be positive.")

    t_values = [traj.t0]
    while t_values[-1] < traj.tf - 1e-14:
        t_values.append(min(t_values[-1] + dt, traj.tf))
    t = np.asarray(t_values)
    N = len(t)

    Rd0, Omd0, _, _ = traj.eval(traj.t0)
    R = project_to_so3(Rd0 if R0 is None else R0)

    # If R0 is perturbed but Omega0 is not specified, initialize the actual
    # angular velocity as the transported desired angular velocity. This gives
    # e_Omega(0)=0 while preserving the chosen attitude error.
    if Omega0 is None:
        Omega = R.T @ Rd0 @ Omd0
    else:
        Omega = np.asarray(Omega0, dtype=float).reshape(3)

    if tau0 is None:
        tau = lee_tau_d(traj.t0, R, Omega, traj, params)
    else:
        tau = np.asarray(tau0, dtype=float).reshape(3)

    R_hist = np.zeros((N, 3, 3))
    Omega_hist = np.zeros((N, 3))
    tau_hist = np.zeros((N, 3))
    tau_d_hist = np.zeros((N, 3))
    tau_d_dot_hist = np.zeros((N, 3))
    tau_dot_hist = np.zeros((N, 3))
    M_hist = np.zeros((N, 3))
    u_hist = np.zeros((N, 3))
    eR_hist = np.zeros((N, 3))
    eOmega_hist = np.zeros((N, 3))
    eTau_hist = np.zeros((N, 3))
    psi_hist = np.zeros(N)
    theta_hist = np.zeros(N)
    Rd_hist = np.zeros((N, 3, 3))
    Omegad_hist = np.zeros((N, 3))
    dOmegad_hist = np.zeros((N, 3))
    ddOmegad_hist = np.zeros((N, 3))
    u_d_bi_hist = np.zeros((N, 3))
    u_d_left_hist = np.zeros((N, 3))
    A_d_left_hist = np.zeros((N, 3))

    for k in range(N):
        tk = t[k]
        aux = collect_aux(tk, R, Omega, tau, traj, params)
        R_hist[k] = R
        Omega_hist[k] = Omega
        tau_hist[k] = tau
        tau_d_hist[k] = np.asarray(aux["tau_d"])
        tau_d_dot_hist[k] = np.asarray(aux["tau_d_dot"])
        tau_dot_hist[k] = np.asarray(aux["tau_dot"])
        M_hist[k] = np.asarray(aux["M"])
        u_hist[k] = np.asarray(aux["u"])
        eR_hist[k] = np.asarray(aux["e_R"])
        eOmega_hist[k] = np.asarray(aux["e_Omega"])
        eTau_hist[k] = np.asarray(aux["e_tau"])
        psi_hist[k] = float(aux["Psi"])
        Rd_hist[k] = np.asarray(aux["Rd"])
        Omegad_hist[k] = np.asarray(aux["Omegad"])
        dOmegad_hist[k] = np.asarray(aux["dOmegad"])
        ddOmegad_hist[k] = np.asarray(aux["ddOmegad"])
        u_d_bi_hist[k] = np.asarray(aux["u_d_bi"])
        u_d_left_hist[k] = np.asarray(aux["u_d_left"])
        A_d_left_hist[k] = np.asarray(aux["A_d_left"])
        theta_hist[k] = attitude_distance_angle(R, Rd_hist[k])

        if k < N - 1:
            h = float(t[k + 1] - t[k])
            R, Omega, tau = lie_rk4_step(tk, h, R, Omega, tau, traj, params)

    results = {
        "t": t,
        "R": R_hist,
        "Omega": Omega_hist,
        "tau": tau_hist,
        "tau_d": tau_d_hist,
        "tau_d_dot": tau_d_dot_hist,
        "tau_dot": tau_dot_hist,
        "M": M_hist,
        "u": u_hist,
        "e_R": eR_hist,
        "e_Omega": eOmega_hist,
        "e_tau": eTau_hist,
        "Psi": psi_hist,
        "theta": theta_hist,
        "R_d": Rd_hist,
        "Omega_d": Omegad_hist,
        "dotOmega_d": dOmegad_hist,
        "ddotOmega_d": ddOmegad_hist,
        "u_d_bi": u_d_bi_hist,
        "u_d_left": u_d_left_hist,
        "A_d_left": A_d_left_hist,
        "J_diag": np.diag(J),
    }
    add_metric_diagnostics(results)
    return results


# ============================================================
# 6) Diagnostics, plotting, and saving
# ============================================================


def add_metric_diagnostics(results: Dict[str, np.ndarray]) -> None:
    """Add Euclidean and physical J-weighted norm series.

    Convention used in this version:
        all body-coordinate vectors in R^3, including tau, tau_d, tau_dot,
        tau_d_dot, M, and e_tau, are measured with the same physical norm
            ||v||_J = sqrt(v^T J v).

    This matches the convention in which these quantities are treated as
    left-trivialized Lie-algebra vectors after the R^3 identification.
    """
    # Tracking errors.
    results["e_R_norm"] = np.linalg.norm(results["e_R"], axis=1)
    results["e_R_J_norm"] = weighted_norm(results["e_R"], J)
    # If e_R is interpreted as dPsi under the Euclidean pairing, then the
    # J-gradient is J^{-1}e_R and its J-norm equals sqrt(e_R^T J^{-1} e_R).
    # Kept as an optional diagnostic, but not used in the main comparison plots.
    results["gradPsi_J_norm"] = weighted_norm(results["e_R"], Jinv)

    results["e_Omega_norm"] = np.linalg.norm(results["e_Omega"], axis=1)
    results["e_Omega_J_norm"] = weighted_norm(results["e_Omega"], J)

    results["e_tau_norm"] = np.linalg.norm(results["e_tau"], axis=1)
    results["e_tau_J_norm"] = weighted_norm(results["e_tau"], J)

    # Torque/actuator variables, all plotted in the physical J-norm by convention.
    results["tau_norm"] = np.linalg.norm(results["tau"], axis=1)
    results["tau_J_norm"] = weighted_norm(results["tau"], J)
    results["tau_d_norm"] = np.linalg.norm(results["tau_d"], axis=1)
    results["tau_d_J_norm"] = weighted_norm(results["tau_d"], J)
    results["tau_d_dot_norm"] = np.linalg.norm(results["tau_d_dot"], axis=1)
    results["tau_d_dot_J_norm"] = weighted_norm(results["tau_d_dot"], J)
    results["tau_dot_norm"] = np.linalg.norm(results["tau_dot"], axis=1)
    results["tau_dot_J_norm"] = weighted_norm(results["tau_dot"], J)

    results["M_norm"] = np.linalg.norm(results["M"], axis=1)
    results["M_J_norm"] = weighted_norm(results["M"], J)

    # Induced higher-order input and nominal trajectory diagnostics.
    results["u_norm"] = np.linalg.norm(results["u"], axis=1)
    results["u_J_norm"] = weighted_norm(results["u"], J)
    results["u_d_bi_norm"] = np.linalg.norm(results["u_d_bi"], axis=1)
    results["u_d_left_J_norm"] = weighted_norm(results["u_d_left"], J)
    results["u_d_left_norm"] = np.linalg.norm(results["u_d_left"], axis=1)

    results["A_d_left_J_norm"] = weighted_norm(results["A_d_left"], J)
    results["Omega_d_J_norm"] = weighted_norm(results["Omega_d"], J)

def _set_axes_equal_3d(ax) -> None:
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    center = limits.mean(axis=1)
    radius = 0.5 * np.max(limits[:, 1] - limits[:, 0])
    ax.set_xlim3d([center[0] - radius, center[0] + radius])
    ax.set_ylim3d([center[1] - radius, center[1] + radius])
    ax.set_zlim3d([center[2] - radius, center[2] + radius])


def plot_body_axes_on_sphere(results: Dict[str, np.ndarray], outdir: Path, every: int = 50) -> None:
    R = results["R"]
    t = results["t"]
    b = [R[:, :, 0], R[:, :, 1], R[:, :, 2]]

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    # Unit sphere wireframe.
    u_grid = np.linspace(0, 2 * np.pi, 60)
    v_grid = np.linspace(0, np.pi, 30)
    xs = np.outer(np.cos(u_grid), np.sin(v_grid))
    ys = np.outer(np.sin(u_grid), np.sin(v_grid))
    zs = np.outer(np.ones_like(u_grid), np.cos(v_grid))
    ax.plot_wireframe(xs, ys, zs, linewidth=0.25, alpha=0.25)

    labels = [r"$Re_1$", r"$Re_2$", r"$Re_3$"]
    for bi, label in zip(b, labels):
        ax.plot(bi[:, 0], bi[:, 1], bi[:, 2], label=label)
        idx = np.arange(0, len(t), max(1, every))
        ax.quiver(
            np.zeros_like(idx, dtype=float),
            np.zeros_like(idx, dtype=float),
            np.zeros_like(idx, dtype=float),
            bi[idx, 0],
            bi[idx, 1],
            bi[idx, 2],
            length=1.0,
            normalize=False,
            linewidth=0.5,
            alpha=0.35,
        )

    ax.set_xlabel("inertial e1")
    ax.set_ylabel("inertial e2")
    ax.set_zlabel("inertial e3")
    ax.set_title("Body axes as curves/vectors on the unit sphere")
    ax.legend()
    _set_axes_equal_3d(ax)
    fig.tight_layout()
    fig.savefig(outdir / "body_axes_unit_sphere.png", dpi=200)
    plt.close(fig)


def plot_norms_and_errors(results: Dict[str, np.ndarray], outdir: Path, traj_metric: str = "left") -> None:
    t = results["t"]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(t, results["tau_J_norm"], label=r"$\|\tau\|_J$")
    ax.plot(t, results["tau_d_J_norm"], linestyle="--", label=r"$\|\tau_d\|_J$")
    ax.plot(t, results["u_J_norm"], label=r"$\|u\|_J$")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("physical norm")
    ax.set_title("Applied torque and induced higher-order input")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "tau_u_norms.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(t, results["tau_J_norm"], label=r"$\|\tau\|_J$")
    ax.plot(t, results["tau_d_J_norm"], linestyle="--", label=r"$\|\tau_d\|_J$")
    ax.plot(t, results["u_J_norm"], label=r"$\|u\|_J$")
    ax.plot(t, results["u_d_left_J_norm"], linestyle="--", label=r"$\|u_d^{\mathrm{left}}\|_J$")
    ax.set_xlabel("time [s]")
    ax.set_ylabel(r"$J$-norm")
    ax.set_title("Metric-weighted control norms")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "control_norms_weighted.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    if traj_metric == "bi":
        ax.plot(t, results["u_d_bi_norm"], label=r"$\|u_d^{\mathrm{bi}}\|$")
        title = "Nominal higher-order control in the bi-invariant objective norm"
    elif traj_metric == "left":
        ax.plot(t, results["u_d_left_J_norm"], label=r"$\|u_d^{\mathrm{left}}\|_J$")
        title = "Nominal higher-order control in the left-invariant objective norm"
    else:
        raise ValueError("traj_metric must be 'bi' or 'left'.")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("objective norm")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "nominal_objective_u_norm.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(t, results["Psi"], label=r"$\Psi(R,R_d)$")
    ax.plot(t, results["e_R_norm"], label=r"$\|e_R\|$")
    ax.plot(t, results["e_Omega_norm"], label=r"$\|e_\Omega\|$")
    ax.plot(t, results["e_tau_norm"], label=r"$\|e_\tau\|$")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("error")
    ax.set_title("Tracking errors, Euclidean diagnostic norms")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "tracking_errors.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(t, results["Psi"], label=r"$\Psi(R,R_d)$")
    ax.plot(t, results["theta"], label=r"$d_{\angle}(R,R_d)$")
    ax.plot(t, results["e_R_J_norm"], label=r"$\|e_R\|_J$")
    ax.plot(t, results["gradPsi_J_norm"], label=r"$\|\mathrm{grad}_J\Psi\|_J$")
    ax.plot(t, results["e_Omega_J_norm"], label=r"$\|e_\Omega\|_J$")
    ax.plot(t, results["e_tau_J_norm"], label=r"$\|e_\tau\|_J$")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("metric-weighted error")
    ax.set_title("Tracking errors, metric-weighted diagnostics")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "tracking_errors_weighted.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(t, results["M_J_norm"], label=r"$\|M\|_J$")
    ax.set_xlabel("time [s]")
    ax.set_ylabel(r"$\|M\|_J$")
    ax.set_title("Actuator command norm")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "actuator_command_norm.png", dpi=200)
    plt.close(fig)


def objective_u_norm(results: Dict[str, np.ndarray], metric: str) -> np.ndarray:
    if metric == "bi":
        return results["u_d_bi_norm"]
    if metric == "left":
        return results["u_d_left_J_norm"]
    raise ValueError("metric must be 'bi' or 'left'.")


def integrated_half_square(t: np.ndarray, y: np.ndarray) -> float:
    return float(0.5 * np.trapz(y**2, t))


def integrated_summary(results: Dict[str, np.ndarray], traj_metric: str) -> Dict[str, float]:
    t = results["t"]
    return {
        "objective_u_cost_own_metric": integrated_half_square(t, objective_u_norm(results, traj_metric)),
        "nominal_u_cost_physical_J": integrated_half_square(t, results["u_d_left_J_norm"]),
        "actual_u_cost_physical_J": integrated_half_square(t, results["u_J_norm"]),
        "tau_d_cost_J": integrated_half_square(t, results["tau_d_J_norm"]),
        "tau_cost_J": integrated_half_square(t, results["tau_J_norm"]),
        "M_cost_J": integrated_half_square(t, results["M_J_norm"]),
        "max_e_R_J": float(np.max(results["e_R_J_norm"])),
        "max_e_Omega_J": float(np.max(results["e_Omega_J_norm"])),
        "max_u_d_left_J": float(np.max(results["u_d_left_J_norm"])),
        "max_u_actual_J": float(np.max(results["u_J_norm"])),
    }


def save_summary_text(results: Dict[str, np.ndarray], outdir: Path, traj_metric: str, label: str = "trajectory") -> None:
    summary = integrated_summary(results, traj_metric)
    lines = [
        f"Summary for {label}",
        f"trajectory metric: {traj_metric}",
        f"J diagonal: {np.diag(J)}",
        "",
    ]
    for key, val in summary.items():
        lines.append(f"{key}: {val:.12e}")
    (outdir / "metric_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_results(results: Dict[str, np.ndarray], outdir: Path, traj_metric: str = "left", label: str = "trajectory") -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    np.savez(outdir / "simulation_results.npz", **results)
    plot_body_axes_on_sphere(results, outdir)
    plot_norms_and_errors(results, outdir, traj_metric=traj_metric)
    save_summary_text(results, outdir, traj_metric=traj_metric, label=label)


def plot_control_comparison(
    results1: Dict[str, np.ndarray],
    results2: Dict[str, np.ndarray],
    outdir: Path,
    label1: str,
    label2: str,
    metric1: str,
    metric2: str,
) -> None:
    """Plot same-figure control comparisons for two nominal trajectories."""
    t1 = results1["t"]
    t2 = results2["t"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False)
    ax = axes[0, 0]
    ax.plot(t1, objective_u_norm(results1, metric1), label=label1)
    ax.plot(t2, objective_u_norm(results2, metric2), label=label2)
    ax.set_title(r"Nominal $u_d$, each method's own objective norm")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("norm")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[0, 1]
    ax.plot(t1, results1["u_d_left_J_norm"], label=label1)
    ax.plot(t2, results2["u_d_left_J_norm"], label=label2)
    ax.set_title(r"Nominal $u_d$, physical left-invariant norm $\|\cdot\|_J$")
    ax.set_xlabel("time [s]")
    ax.set_ylabel(r"$\|u_d\|_J$")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 0]
    ax.plot(t1, results1["u_J_norm"], label=label1)
    ax.plot(t2, results2["u_J_norm"], label=label2)
    ax.set_title(r"Actual induced input $u$, physical norm $\|\cdot\|_J$")
    ax.set_xlabel("time [s]")
    ax.set_ylabel(r"$\|u\|_J$")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 1]
    ax.plot(t1, results1["tau_d_J_norm"], label=rf"{label1}: $\tau_d$")
    ax.plot(t2, results2["tau_d_J_norm"], label=rf"{label2}: $\tau_d$")
    ax.plot(t1, results1["tau_J_norm"], linestyle="--", label=rf"{label1}: $\tau$")
    ax.plot(t2, results2["tau_J_norm"], linestyle="--", label=rf"{label2}: $\tau$")
    ax.set_title(r"Torque norms in the physical metric $\|\cdot\|_J$")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("norm")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(outdir / "control_comparison_weighted.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=False)
    axes[0].plot(t1, results1["M_J_norm"], label=label1)
    axes[0].plot(t2, results2["M_J_norm"], label=label2)
    axes[0].set_title(r"Actuator command $\|M\|_J$")
    axes[0].set_xlabel("time [s]")
    axes[0].set_ylabel(r"$\|M\|_J$")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(t1, results1["tau_d_dot_J_norm"], label=label1)
    axes[1].plot(t2, results2["tau_d_dot_J_norm"], label=label2)
    axes[1].set_title(r"Desired torque rate $\|\dot\tau_d\|_J$")
    axes[1].set_xlabel("time [s]")
    axes[1].set_ylabel("physical norm")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(t1, results1["e_R_J_norm"], label=rf"{label1}: $\|e_R\|_J$")
    axes[2].plot(t2, results2["e_R_J_norm"], label=rf"{label2}: $\|e_R\|_J$")
    axes[2].plot(t1, results1["e_Omega_J_norm"], linestyle="--", label=rf"{label1}: $\|e_\Omega\|_J$")
    axes[2].plot(t2, results2["e_Omega_J_norm"], linestyle="--", label=rf"{label2}: $\|e_\Omega\|_J$")
    axes[2].set_title("Metric-weighted tracking errors")
    axes[2].set_xlabel("time [s]")
    axes[2].set_ylabel("norm")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(outdir / "control_and_error_comparison_extra.png", dpi=220)
    plt.close(fig)

    write_comparison_summary(results1, results2, outdir, label1, label2, metric1, metric2)


def write_comparison_summary(
    results1: Dict[str, np.ndarray],
    results2: Dict[str, np.ndarray],
    outdir: Path,
    label1: str,
    label2: str,
    metric1: str,
    metric2: str,
) -> None:
    s1 = integrated_summary(results1, metric1)
    s2 = integrated_summary(results2, metric2)
    keys = list(s1.keys())
    lines = [
        "Metric comparison summary",
        f"J diagonal: {np.diag(J)}",
        f"{label1} metric: {metric1}",
        f"{label2} metric: {metric2}",
        "",
        f"{'quantity':40s} {label1:>20s} {label2:>20s} ratio_2_over_1",
    ]
    for key in keys:
        v1 = s1[key]
        v2 = s2[key]
        ratio = np.nan if abs(v1) < 1e-15 else v2 / v1
        lines.append(f"{key:40s} {v1:20.12e} {v2:20.12e} {ratio:20.12e}")
    (outdir / "comparison_metric_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")




def cumulative_l2_norm(t: np.ndarray, norm_values: np.ndarray) -> np.ndarray:
    """Return sqrt(int_0^t norm_values(s)^2 ds) using trapezoidal quadrature."""
    t = np.asarray(t, dtype=float).reshape(-1)
    y = np.asarray(norm_values, dtype=float).reshape(-1)
    if len(t) != len(y):
        raise ValueError("t and norm_values must have the same length")
    if len(t) == 0:
        return np.array([])
    if len(t) == 1:
        return np.array([0.0])
    increments = 0.5 * (y[:-1] ** 2 + y[1:] ** 2) * np.diff(t)
    integral = np.concatenate([[0.0], np.cumsum(increments)])
    return np.sqrt(np.maximum(integral, 0.0))


def save_figure_all_formats(fig, outdir: Path, stem: str, dpi: int = 240) -> None:
    """Save PNG and PDF copies of a figure."""
    fig.tight_layout()
    fig.savefig(outdir / f"{stem}.png", dpi=dpi)
    fig.savefig(outdir / f"{stem}.pdf")
    plt.close(fig)


def plot_paper_single(results: Dict[str, np.ndarray], outdir: Path, label: str = "trajectory") -> None:
    """Save the two paper-focused figures for a single trajectory.

    Plot 1: all tracking errors in the physical J-norm.
    Plot 2: cumulative weighted L2 norms of u and M, with final costs in the legend.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    t = results["t"]

    # Plot 1: Tracking errors.
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    ax.plot(t, results["e_R_J_norm"], label=rf"{label}: $\|e_R\|_J$")
    ax.plot(t, results["e_Omega_J_norm"], label=rf"{label}: $\|e_\Omega\|_J$")
    ax.plot(t, results["e_tau_J_norm"], label=rf"{label}: $\|e_\tau\|_J$")
    ax.set_xlabel("Time")
    ax.set_ylabel("Tracking Errors")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    save_figure_all_formats(fig, outdir, "paper_plot1_tracking_errors")

    # Plot 2: cumulative weighted L2 norms of actual induced input u and actuator command M.
    L2u = cumulative_l2_norm(t, results["u_J_norm"])
    L2M = cumulative_l2_norm(t, results["M_J_norm"])
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    ax.plot(t, L2u, label=rf"{label}: $\|u\|_{{L^2,J}}={L2u[-1]:.4g}$")
    ax.plot(t, L2M, label=rf"{label}: $\|M\|_{{L^2,J}}={L2M[-1]:.4g}$")
    ax.set_xlabel("Time")
    ax.set_ylabel(r"Weighted $L^2$ Norm")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    save_figure_all_formats(fig, outdir, "paper_plot2_weighted_L2_u_M")

    lines = [
        "Paper-focused metric summary",
        f"J diagonal: {np.diag(J)}",
        "All norms are physical J-norms for body-coordinate quantities.",
        "",
        f"{label} u L2_J: {L2u[-1]:.12e}",
        f"{label} M L2_J: {L2M[-1]:.12e}",
        f"{label} max e_R_J: {np.max(results['e_R_J_norm']):.12e}",
        f"{label} max e_Omega_J: {np.max(results['e_Omega_J_norm']):.12e}",
        f"{label} max e_tau_J: {np.max(results['e_tau_J_norm']):.12e}",
    ]
    (outdir / "paper_metric_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_paper_comparison(
    results1: Dict[str, np.ndarray],
    results2: Dict[str, np.ndarray],
    outdir: Path,
    label1: str,
    label2: str,
) -> None:
    """Save the two paper-focused same-axis comparison figures.

    Plot 1: tracking errors for both trajectories, all measured in the physical J-norm.
    Plot 2: cumulative weighted L2 norms of u and M for both trajectories,
            with final costs included in the legend.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    t1 = results1["t"]
    t2 = results2["t"]

    # Plot 1: all tracking errors, all with physical J-norm, on the same plot.
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.plot(t1, results1["e_R_J_norm"], label=rf"{label1}: $\|e_R\|_J$")
    ax.plot(t1, results1["e_Omega_J_norm"], label=rf"{label1}: $\|e_\Omega\|_J$")
    ax.plot(t1, results1["e_tau_J_norm"], label=rf"{label1}: $\|e_\tau\|_J$")
    ax.plot(t2, results2["e_R_J_norm"], "--", label=rf"{label2}: $\|e_R\|_J$")
    ax.plot(t2, results2["e_Omega_J_norm"], "--", label=rf"{label2}: $\|e_\Omega\|_J$")
    ax.plot(t2, results2["e_tau_J_norm"], "--", label=rf"{label2}: $\|e_\tau\|_J$")
    ax.set_xlabel("Time")
    ax.set_ylabel("Tracking Errors")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    save_figure_all_formats(fig, outdir, "paper_plot1_tracking_errors")

    # Plot 2: cumulative weighted L2 norms of actual induced input u and actuator command M.
    L2u1 = cumulative_l2_norm(t1, results1["u_J_norm"])
    L2M1 = cumulative_l2_norm(t1, results1["M_J_norm"])
    L2u2 = cumulative_l2_norm(t2, results2["u_J_norm"])
    L2M2 = cumulative_l2_norm(t2, results2["M_J_norm"])

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.plot(t1, L2u1, label=rf"{label1}: $\|u\|_{{L^2,J}}={L2u1[-1]:.4g}$")
    ax.plot(t1, L2M1, linestyle="--", label=rf"{label1}: $\|M\|_{{L^2,J}}={L2M1[-1]:.4g}$")
    ax.plot(t2, L2u2, label=rf"{label2}: $\|u\|_{{L^2,J}}={L2u2[-1]:.4g}$")
    ax.plot(t2, L2M2, linestyle="--", label=rf"{label2}: $\|M\|_{{L^2,J}}={L2M2[-1]:.4g}$")
    ax.set_xlabel("Time")
    ax.set_ylabel(r"Weighted $L^2$ Norm")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    save_figure_all_formats(fig, outdir, "paper_plot2_weighted_L2_u_M")

    lines = [
        "Paper-focused metric summary",
        f"J diagonal: {np.diag(J)}",
        "All norms are physical J-norms for body-coordinate quantities.",
        "Weighted L2 norm means sqrt(int_0^T ||.||_J^2 dt).",
        "",
        f"{label1} u L2_J: {L2u1[-1]:.12e}",
        f"{label1} M L2_J: {L2M1[-1]:.12e}",
        f"{label2} u L2_J: {L2u2[-1]:.12e}",
        f"{label2} M L2_J: {L2M2[-1]:.12e}",
        f"{label1} max e_R_J: {np.max(results1['e_R_J_norm']):.12e}",
        f"{label2} max e_R_J: {np.max(results2['e_R_J_norm']):.12e}",
        f"{label1} max e_Omega_J: {np.max(results1['e_Omega_J_norm']):.12e}",
        f"{label2} max e_Omega_J: {np.max(results2['e_Omega_J_norm']):.12e}",
        f"{label1} max e_tau_J: {np.max(results1['e_tau_J_norm']):.12e}",
        f"{label2} max e_tau_J: {np.max(results2['e_tau_J_norm']):.12e}",
    ]
    (outdir / "paper_metric_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ============================================================
# 7) CLI
# ============================================================


def parse_vec3(values) -> np.ndarray | None:
    if values is None:
        return None
    if len(values) != 3:
        raise argparse.ArgumentTypeError("Expected exactly 3 values.")
    return np.array([float(v) for v in values], dtype=float)


def build_initial_conditions(args, traj: DesiredTrajectory, params: ControlParams) -> Tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Construct R0, Omega0, tau0 from CLI arguments for a given desired trajectory."""
    Rd0, Omd0, _, _ = traj.eval(traj.t0)

    n_R_specs = sum(x is not None for x in (args.R0, args.R0_rotvec, args.R0_rotvec_deg))
    if n_R_specs > 1:
        raise ValueError("Specify at most one of --R0, --R0-rotvec, or --R0-rotvec-deg.")

    R0 = None
    if args.R0 is not None:
        R0 = project_to_so3(np.array([float(x) for x in args.R0], dtype=float).reshape(3, 3))
    elif args.R0_rotvec is not None:
        R0 = Rd0 @ so3_exp(parse_vec3(args.R0_rotvec))
    elif args.R0_rotvec_deg is not None:
        R0 = Rd0 @ so3_exp(np.deg2rad(parse_vec3(args.R0_rotvec_deg)))

    R_for_init = Rd0 if R0 is None else R0

    if args.Omega0 is not None and args.Omega0_offset is not None:
        raise ValueError("Specify either --Omega0 or --Omega0-offset, not both.")

    Omega0 = None
    if args.Omega0 is not None:
        Omega0 = parse_vec3(args.Omega0)
    elif args.Omega0_offset is not None:
        # The offset is exactly e_Omega(0).
        Omega0 = R_for_init.T @ Rd0 @ Omd0 + parse_vec3(args.Omega0_offset)

    Omega_for_tau = R_for_init.T @ Rd0 @ Omd0 if Omega0 is None else Omega0
    tau_d0 = lee_tau_d(traj.t0, R_for_init, Omega_for_tau, traj, params)

    tau0 = None
    if args.tau0 is not None:
        tau0 = parse_vec3(args.tau0)
    else:
        tau_offset = np.zeros(3) if args.tau0_offset is None else parse_vec3(args.tau0_offset)
        if args.tau0_mode == "desired":
            tau0 = tau_d0 + tau_offset
        elif args.tau0_mode == "zero":
            tau0 = tau_offset
        else:
            raise ValueError(f"Unsupported --tau0-mode: {args.tau0_mode}")

    return R0, Omega0, tau0


def print_initial_and_final_summary(results: Dict[str, np.ndarray], label: str = "trajectory") -> None:
    print(f"\nSummary: {label}")
    for name, idx in [("Initial", 0), ("Final", -1)]:
        print(f"{name} Psi: {results['Psi'][idx]:.6e}")
        print(f"{name} ||e_R||: {results['e_R_norm'][idx]:.6e}")
        print(f"{name} ||e_R||_J: {results['e_R_J_norm'][idx]:.6e}")
        print(f"{name} ||grad_J Psi||_J: {results['gradPsi_J_norm'][idx]:.6e}")
        print(f"{name} ||e_Omega||: {results['e_Omega_norm'][idx]:.6e}")
        print(f"{name} ||e_Omega||_J: {results['e_Omega_J_norm'][idx]:.6e}")
        print(f"{name} ||e_tau||: {results['e_tau_norm'][idx]:.6e}")
        print(f"{name} ||e_tau||_J: {results['e_tau_J_norm'][idx]:.6e}")


def safe_label(label: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", label.strip())
    return s.strip("_") or "trajectory"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SO(3) tracking simulation with torque actuation, metric-weighted diagnostics, and optional two-trajectory comparison."
    )
    parser.add_argument("trajectory", type=str, help="Path to first desired trajectory .npz file.")
    parser.add_argument("--trajectory2", "--compare-trajectory", dest="trajectory2", type=str, default=None, help="Optional second desired trajectory .npz file for same-figure comparison plots.")
    parser.add_argument("--out", type=str, default="so3_sim_out", help="Output directory.")
    parser.add_argument("--dt", type=float, default=0.002, help="Fixed integration step size.")
    parser.add_argument("--J", nargs=3, type=float, default=None, help="Diagonal inertia tensor entries. Example: --J 0.04 0.11 0.20")

    parser.add_argument("--label1", type=str, default="trajectory 1", help="Legend label for first trajectory.")
    parser.add_argument("--label2", type=str, default="trajectory 2", help="Legend label for second trajectory.")
    parser.add_argument("--traj1-metric", choices=("bi", "left"), default="left", help="Metric used to compute the first trajectory's own objective u-norm.")
    parser.add_argument("--traj2-metric", choices=("bi", "left"), default="left", help="Metric used to compute the second trajectory's own objective u-norm.")

    parser.add_argument("--k_R", type=float, default=2.3)
    parser.add_argument("--k_Omega", type=float, default=2.1)
    parser.add_argument("--k_tau", type=float, default=1.6)
    parser.add_argument("--C", type=float, default=0.75, help="Torque actuator time constant parameter.")
    parser.add_argument("--K", type=float, default=0.15, help="Torque actuator gain parameter.")
    parser.add_argument("--M_max", type=float, default=8.0, help="Optional saturation on actuator command norm. Use a large value to effectively disable.")

    parser.add_argument(
        "--R0",
        nargs=9,
        default=None,
        help="Initial rotation as 9 row-major entries. Default: R_d(t0).",
    )
    parser.add_argument(
        "--R0-rotvec",
        dest="R0_rotvec",
        nargs=3,
        default=None,
        help="Initial relative attitude error in radians, applied as R0 = R_d(t0) exp(hat(r)).",
    )
    parser.add_argument(
        "--R0-rotvec-deg",
        dest="R0_rotvec_deg",
        nargs=3,
        default=None,
        help="Initial relative attitude error in degrees, applied as R0 = R_d(t0) exp(hat(r)).",
    )
    parser.add_argument(
        "--Omega0",
        nargs=3,
        default=None,
        help="Initial body angular velocity. Default: transported Omega_d(t0), so e_Omega(0)=0 for attitude-only offsets.",
    )
    parser.add_argument(
        "--Omega0-offset",
        dest="Omega0_offset",
        nargs=3,
        default=None,
        help="Initial angular-velocity tracking error e_Omega(0). Sets Omega0 = R0^T R_d Omega_d + offset.",
    )
    parser.add_argument("--tau0", nargs=3, default=None, help="Initial applied torque. Overrides --tau0-mode and --tau0-offset.")
    parser.add_argument(
        "--tau0-mode",
        choices=("desired", "zero"),
        default="desired",
        help="Base initial applied torque when --tau0 is not provided. 'desired' gives e_tau(0)=0; 'zero' starts the actuator from rest.",
    )
    parser.add_argument(
        "--tau0-offset",
        dest="tau0_offset",
        nargs=3,
        default=None,
        help="Offset added to the base initial torque. With --tau0-mode desired this is e_tau(0).",
    )

    args = parser.parse_args()
    set_inertia(np.array(args.J, dtype=float) if args.J is not None else None)

    params = ControlParams(k_R=args.k_R, k_Omega=args.k_Omega, k_tau=args.k_tau, C=args.C, K=args.K, M_max=args.M_max)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    traj1 = load_desired_npz(args.trajectory)
    R0_1, Omega0_1, tau0_1 = build_initial_conditions(args, traj1, params)
    results1 = simulate(traj=traj1, params=params, dt=args.dt, R0=R0_1, Omega0=Omega0_1, tau0=tau0_1)

    if args.trajectory2 is None:
        save_results(results1, outdir, traj_metric=args.traj1_metric, label=args.label1)
        plot_paper_single(results1, outdir, label=args.label1)
        print(f"Saved results to: {outdir.resolve()}")
        print(f"  - {outdir / 'simulation_results.npz'}")
        print(f"  - {outdir / 'body_axes_unit_sphere.png'}")
        print(f"  - {outdir / 'tau_u_norms.png'}")
        print(f"  - {outdir / 'control_norms_weighted.png'}")
        print(f"  - {outdir / 'nominal_objective_u_norm.png'}")
        print(f"  - {outdir / 'tracking_errors.png'}")
        print(f"  - {outdir / 'tracking_errors_weighted.png'}")
        print(f"  - {outdir / 'actuator_command_norm.png'}")
        print(f"  - {outdir / 'metric_summary.txt'}")
        print(f"  - {outdir / 'paper_plot1_tracking_errors.png'}")
        print(f"  - {outdir / 'paper_plot1_tracking_errors.pdf'}")
        print(f"  - {outdir / 'paper_plot2_weighted_L2_u_M.png'}")
        print(f"  - {outdir / 'paper_plot2_weighted_L2_u_M.pdf'}")
        print(f"  - {outdir / 'paper_metric_summary.txt'}")
        print_initial_and_final_summary(results1, args.label1)
        return

    traj2 = load_desired_npz(args.trajectory2)
    R0_2, Omega0_2, tau0_2 = build_initial_conditions(args, traj2, params)
    results2 = simulate(traj=traj2, params=params, dt=args.dt, R0=R0_2, Omega0=Omega0_2, tau0=tau0_2)

    sub1 = outdir / safe_label(args.label1)
    sub2 = outdir / safe_label(args.label2)
    save_results(results1, sub1, traj_metric=args.traj1_metric, label=args.label1)
    save_results(results2, sub2, traj_metric=args.traj2_metric, label=args.label2)
    plot_control_comparison(results1, results2, outdir, args.label1, args.label2, args.traj1_metric, args.traj2_metric)
    plot_paper_comparison(results1, results2, outdir, args.label1, args.label2)

    print(f"Saved comparison results to: {outdir.resolve()}")
    print(f"  - {sub1 / 'simulation_results.npz'}")
    print(f"  - {sub2 / 'simulation_results.npz'}")
    print(f"  - {outdir / 'control_comparison_weighted.png'}")
    print(f"  - {outdir / 'control_and_error_comparison_extra.png'}")
    print(f"  - {outdir / 'comparison_metric_summary.txt'}")
    print(f"  - {outdir / 'paper_plot1_tracking_errors.png'}")
    print(f"  - {outdir / 'paper_plot1_tracking_errors.pdf'}")
    print(f"  - {outdir / 'paper_plot2_weighted_L2_u_M.png'}")
    print(f"  - {outdir / 'paper_plot2_weighted_L2_u_M.pdf'}")
    print(f"  - {outdir / 'paper_metric_summary.txt'}")
    print_initial_and_final_summary(results1, args.label1)
    print_initial_and_final_summary(results2, args.label2)


if __name__ == "__main__":
    main()
