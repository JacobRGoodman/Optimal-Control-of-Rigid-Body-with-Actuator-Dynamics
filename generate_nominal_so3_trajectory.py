#!/usr/bin/env python3
"""
Generate nominal SO(3) desired trajectories for the attitude-tracking simulation.

This script solves one of two boundary-value problems and writes an .npz file
with keys expected by simulate_so3_actuated_tracking.py:

    t, R_d, Omega_d, dotOmega_d, ddotOmega_d

Methods
-------
1. sasaki
   Geodesics of the generalized Sasaki metric on T^(2)SO(3), written in
   left-trivialized coordinates (R, eta, zeta, Omega, Lambda, Upsilon).

2. quintic
   The earlier bi-invariant/SO(3)-cross-product quintic-in-tension model,
   retained for backward compatibility.

3. rigid-quintic
   Riemannian quintics in tension for the left-invariant rigid-body metric
   with inertia tensor J.  The state is

       (rho, Omega0, Omega1, Omega2, Omega3, Omega4),

   where Omega_alpha = (R^T nabla_{Rdot}^alpha Rdot)^vee.

4. rigid-cubic
   Riemannian cubics for the left-invariant rigid-body metric with inertia
   tensor J. The state is

       (rho, Omega0, Omega1, Omega2),

   and only endpoint attitude and angular velocity are enforced.

5. rigid-quintic-natural
   Riemannian quintics in tension for the left-invariant rigid-body metric
   with attitude and angular-velocity endpoints fixed, endpoint covariant
   accelerations free, and natural boundary conditions

       Omega2(0) = 0, Omega2(T) = 0.

Implementation note
-------------------
SciPy's solve_bvp works in Euclidean coordinates, so we solve for
rho(t) = Log(R0^T R(t)) in a local SO(3) chart. The reconstruction is
R(t) = R0 Exp(rho(t)). This is robust for rotations that remain away from
the log-chart cut locus (rotation angle close to pi). For larger maneuvers,
use an intermediate waypoint or a better initial guess.

Boundary JSON format
--------------------
{
  "T": 5.0,
  "N": 1001,
  "R0": [[1,0,0],[0,1,0],[0,0,1]],
  "R1_rotvec": [0, 0, 1.5707963267948966],
  "Omega0": [0, 0, 0],
  "Omega1": [0, 0, 0],
  "dotOmega0": [0, 0, 0],
  "dotOmega1": [0, 0, 0]
}

You may specify R1 as a matrix using "R1"/"Rf"/"RT", or as a rotation
vector using "R1_rotvec". The same is supported for R0.

For --method sasaki, --sasaki-bc fiber enforces endpoint data in
T^(2)SO(3): R, eta, zeta. When eta/zeta are not explicitly supplied,
Omega/dotOmega are used as their endpoint values. Use --sasaki-bc projected
to instead enforce projected boundary data R, Omega, dotOmega at both
endpoints; this nonlinear BVP can be less well conditioned.


python generate_nominal_so3_trajectory.py `
  --bc bc_nonzero_spin_and_acceleration.json `
  --method rigid-quintic-natural `
  --eps1 0.01 `
  --eps2 0.5 `
  --out desired_rigid_quintic_natural.npz `
  --nodes 100 `
  --tol 1e-3 `
  --max-nodes 50000 `
  --plot


  python generate_nominal_so3_trajectory.py `
  --bc bc_nonzero_spin_and_acceleration.json `
  --method rigid-cubic `
  --out desired_rigid_cubic.npz `
  --nodes 60 `
  --tol 1e-4 `
  --plot
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
from scipy.integrate import solve_bvp

SCRIPT_VERSION = "2026-05-19-rigid-cubic-natural-quintic"


# ============================================================
# 1) SO(3) linear algebra helpers
# ============================================================


def hat(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(3)
    return np.array(
        [[0.0, -x[2], x[1]],
         [x[2], 0.0, -x[0]],
         [-x[1], x[0], 0.0]],
        dtype=float,
    )


def vee(A: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=float)
    return np.array([A[2, 1], A[0, 2], A[1, 0]], dtype=float)


def so3_exp(phi: np.ndarray) -> np.ndarray:
    phi = np.asarray(phi, dtype=float).reshape(3)
    th = np.linalg.norm(phi)
    K = hat(phi)
    if th < 1e-10:
        return np.eye(3) + K + 0.5 * (K @ K)
    A = np.sin(th) / th
    B = (1.0 - np.cos(th)) / (th * th)
    return np.eye(3) + A * K + B * (K @ K)


def so3_log(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=float).reshape(3, 3)
    c = 0.5 * (np.trace(R) - 1.0)
    c = np.clip(c, -1.0, 1.0)
    th = float(np.arccos(c))
    if th < 1e-10:
        return 0.5 * vee(R - R.T)
    if np.pi - th < 1e-7:
        # Numerically stable enough for a warning-level near-pi case.
        # Prefer not to use this chart close to pi for BVP solving.
        eigvals, eigvecs = np.linalg.eig(R)
        idx = int(np.argmin(np.abs(eigvals - 1.0)))
        axis = np.real(eigvecs[:, idx])
        axis = axis / np.linalg.norm(axis)
        return th * axis
    return (th / (2.0 * np.sin(th))) * vee(R - R.T)


def right_jacobian(phi: np.ndarray) -> np.ndarray:
    phi = np.asarray(phi, dtype=float).reshape(3)
    th = np.linalg.norm(phi)
    K = hat(phi)
    if th < 1e-8:
        return np.eye(3) - 0.5 * K + (1.0 / 6.0) * (K @ K)
    A = (1.0 - np.cos(th)) / (th * th)
    B = (th - np.sin(th)) / (th ** 3)
    return np.eye(3) - A * K + B * (K @ K)


def right_jacobian_inv(phi: np.ndarray) -> np.ndarray:
    phi = np.asarray(phi, dtype=float).reshape(3)
    th = np.linalg.norm(phi)
    K = hat(phi)
    if th < 1e-8:
        return np.eye(3) + 0.5 * K + (1.0 / 12.0) * (K @ K)
    half = 0.5
    C = (1.0 / (th * th)) - ((1.0 + np.cos(th)) / (2.0 * th * np.sin(th)))
    return np.eye(3) + half * K + C * (K @ K)


def apply_right_jacobian_inv(rho: np.ndarray, v: np.ndarray) -> np.ndarray:
    out = np.empty_like(v)
    for k in range(v.shape[1]):
        out[:, k] = right_jacobian_inv(rho[:, k]) @ v[:, k]
    return out


def apply_right_jacobian(rho: np.ndarray, v: np.ndarray) -> np.ndarray:
    out = np.empty_like(v)
    for k in range(v.shape[1]):
        out[:, k] = right_jacobian(rho[:, k]) @ v[:, k]
    return out


def cross_cols(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Column-wise cross product for 3 x N arrays."""
    return np.cross(a.T, b.T).T


def normalize_rotation(R: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(R)
    Rn = U @ Vt
    if np.linalg.det(Rn) < 0:
        U[:, -1] *= -1.0
        Rn = U @ Vt
    return Rn


# ============================================================
# 2) Boundary data and initial guesses
# ============================================================


@dataclass
class BoundaryData:
    T: float
    N: int
    R0: np.ndarray
    R1: np.ndarray
    Omega0: np.ndarray
    Omega1: np.ndarray
    dotOmega0: np.ndarray
    dotOmega1: np.ndarray
    eta0: np.ndarray | None = None
    eta1: np.ndarray | None = None
    zeta0: np.ndarray | None = None
    zeta1: np.ndarray | None = None
    covAccel0: np.ndarray | None = None
    covAccel1: np.ndarray | None = None


def scaled_boundary_data(bc: BoundaryData, lam: float) -> BoundaryData:
    """Homotopy in endpoint velocity/acceleration data, keeping R0, R1 fixed.

    lam=0 gives rest-to-rest projected/fiber endpoint derivatives; lam=1 gives
    the requested boundary data. This is useful for difficult Sasaki BVPs.
    """
    lam = float(lam)

    def scale_optional(x):
        return None if x is None else lam * np.asarray(x, dtype=float)

    return BoundaryData(
        T=bc.T,
        N=bc.N,
        R0=bc.R0,
        R1=bc.R1,
        Omega0=lam * bc.Omega0,
        Omega1=lam * bc.Omega1,
        dotOmega0=lam * bc.dotOmega0,
        dotOmega1=lam * bc.dotOmega1,
        eta0=scale_optional(bc.eta0),
        eta1=scale_optional(bc.eta1),
        zeta0=scale_optional(bc.zeta0),
        zeta1=scale_optional(bc.zeta1),
        covAccel0=scale_optional(bc.covAccel0),
        covAccel1=scale_optional(bc.covAccel1),
    )


def vec3(x: Iterable[float], name: str) -> np.ndarray:
    arr = np.asarray(list(x), dtype=float)
    if arr.shape != (3,):
        raise ValueError(f"{name} must be length 3, got shape {arr.shape}")
    return arr


def rotation_from_bc(data: Dict, prefix: str, default: np.ndarray | None = None) -> np.ndarray:
    # prefix examples: "R0", "R1". Also supports Rf/RT as aliases for R1.
    aliases = [prefix]
    if prefix == "R1":
        aliases += ["Rf", "RT"]

    for key in aliases:
        if key in data:
            return normalize_rotation(np.asarray(data[key], dtype=float).reshape(3, 3))
        rv_key = f"{key}_rotvec"
        if rv_key in data:
            return so3_exp(vec3(data[rv_key], rv_key))

    if default is None:
        raise ValueError(f"Missing rotation boundary {prefix}; use {prefix} or {prefix}_rotvec")
    return default


def load_boundary_data(path: str | None) -> BoundaryData:
    if path is None:
        # Demo: rest-to-rest yaw of 90 degrees in 5 seconds.
        return BoundaryData(
            T=5.0,
            N=1001,
            R0=np.eye(3),
            R1=so3_exp(np.array([0.0, 0.0, 0.5 * np.pi])),
            Omega0=np.zeros(3),
            Omega1=np.zeros(3),
            dotOmega0=np.zeros(3),
            dotOmega1=np.zeros(3),
        )

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    R0 = rotation_from_bc(data, "R0", default=np.eye(3))
    R1 = rotation_from_bc(data, "R1")

    def opt_vec(name: str) -> np.ndarray | None:
        return vec3(data[name], name) if name in data else None

    return BoundaryData(
        T=float(data.get("T", data.get("tf", 5.0))),
        N=int(data.get("N", 1001)),
        R0=R0,
        R1=R1,
        Omega0=vec3(data.get("Omega0", [0, 0, 0]), "Omega0"),
        Omega1=vec3(data.get("Omega1", data.get("OmegaT", [0, 0, 0])), "Omega1"),
        dotOmega0=vec3(data.get("dotOmega0", [0, 0, 0]), "dotOmega0"),
        dotOmega1=vec3(data.get("dotOmega1", data.get("dotOmegaT", [0, 0, 0])), "dotOmega1"),
        eta0=opt_vec("eta0"),
        eta1=opt_vec("eta1"),
        zeta0=opt_vec("zeta0"),
        zeta1=opt_vec("zeta1"),
        covAccel0=(opt_vec("covAccel0") if "covAccel0" in data else (opt_vec("A0") if "A0" in data else None)),
        covAccel1=(
            opt_vec("covAccel1") if "covAccel1" in data else
            (opt_vec("covAccelT") if "covAccelT" in data else
             (opt_vec("covAccelF") if "covAccelF" in data else
              (opt_vec("A1") if "A1" in data else (opt_vec("AT") if "AT" in data else None))))
        ),
    )


def quintic_coefficients(q0, qT, qd0, qdT, qdd0, qddT, T):
    """Coefficients a[0:6, dim] for q(t)=sum_i a[i] t^i."""
    q0 = np.asarray(q0, dtype=float)
    qT = np.asarray(qT, dtype=float)
    qd0 = np.asarray(qd0, dtype=float)
    qdT = np.asarray(qdT, dtype=float)
    qdd0 = np.asarray(qdd0, dtype=float)
    qddT = np.asarray(qddT, dtype=float)

    dim = q0.size
    coeff = np.zeros((6, dim))
    coeff[0] = q0
    coeff[1] = qd0
    coeff[2] = 0.5 * qdd0

    M = np.array(
        [[T**3, T**4, T**5],
         [3*T**2, 4*T**3, 5*T**4],
         [6*T, 12*T**2, 20*T**3]],
        dtype=float,
    )
    rhs = np.vstack([
        qT - (coeff[0] + coeff[1]*T + coeff[2]*T**2),
        qdT - (coeff[1] + 2*coeff[2]*T),
        qddT - (2*coeff[2]),
    ])
    coeff[3:6] = np.linalg.solve(M, rhs)
    return coeff


def eval_poly(coeff: np.ndarray, t: np.ndarray, deriv: int = 0) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    dim = coeff.shape[1]
    out = np.zeros((dim, t.size), dtype=float)
    for power in range(deriv, coeff.shape[0]):
        c = coeff[power].copy()
        factor = 1.0
        for j in range(deriv):
            factor *= power - j
        out += (factor * c)[:, None] * t[None, :] ** (power - deriv)
    return out


def initial_guess_common(bc: BoundaryData, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rho_f = so3_log(bc.R0.T @ bc.R1)
    rho_dot0 = bc.Omega0
    rho_dotT = right_jacobian_inv(rho_f) @ bc.Omega1
    # This ignores d/dt J_r^{-1}; it is only an initial guess for solve_bvp.
    rho_ddot0 = bc.dotOmega0
    rho_ddotT = bc.dotOmega1
    coeff = quintic_coefficients(
        np.zeros(3), rho_f, rho_dot0, rho_dotT, rho_ddot0, rho_ddotT, bc.T
    )
    rho = eval_poly(coeff, t, 0)
    rho_dot = eval_poly(coeff, t, 1)
    Omega = apply_right_jacobian(rho, rho_dot)
    A = np.gradient(Omega, t, axis=1, edge_order=2)
    return rho, Omega, A


def make_quintic_guess(bc: BoundaryData, t: np.ndarray) -> np.ndarray:
    rho, Omega, A = initial_guess_common(bc, t)
    Adot = np.gradient(A, t, axis=1, edge_order=2)
    eta = Adot + 0.5 * cross_cols(Omega, A)
    etadot = np.gradient(eta, t, axis=1, edge_order=2)
    zeta = etadot + 0.5 * cross_cols(Omega, eta)
    chi = np.gradient(zeta, t, axis=1, edge_order=2)
    return np.vstack([rho, Omega, A, eta, zeta, chi])


def make_sasaki_guess(bc: BoundaryData, t: np.ndarray) -> np.ndarray:
    rho, Omega, A = initial_guess_common(bc, t)

    eta0 = bc.Omega0 if bc.eta0 is None else bc.eta0
    eta1 = bc.Omega1 if bc.eta1 is None else bc.eta1
    zeta0 = bc.dotOmega0 if bc.zeta0 is None else bc.zeta0
    zeta1 = bc.dotOmega1 if bc.zeta1 is None else bc.zeta1

    coeff_eta = quintic_coefficients(eta0, eta1, np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3), bc.T)
    coeff_zeta = quintic_coefficients(zeta0, zeta1, np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3), bc.T)
    eta = eval_poly(coeff_eta, t, 0)
    etadot = eval_poly(coeff_eta, t, 1)
    zeta = eval_poly(coeff_zeta, t, 0)
    zetadot = eval_poly(coeff_zeta, t, 1)

    Lambda = etadot - 0.5 * cross_cols(eta, Omega)
    Upsilon = zetadot - 0.5 * cross_cols(zeta, Omega) - 0.125 * cross_cols(cross_cols(eta, Omega), eta)
    return np.vstack([rho, eta, zeta, Omega, Lambda, Upsilon])


# ============================================================
# 3) Differential equations for the two BVPs
# ============================================================


def sasaki_omega_dot_cols(eta, zeta, Omega, Lambda, Upsilon):
    return (
        0.25 * cross_cols(cross_cols(eta, Lambda), Omega)
        + 0.25 * cross_cols(cross_cols(zeta, Upsilon), Omega)
        + 0.125 * cross_cols(cross_cols(Lambda, Upsilon), Omega)
        + 0.125 * cross_cols(cross_cols(eta, Upsilon), Lambda)
    )


def sasaki_omega_dot_single(y: np.ndarray) -> np.ndarray:
    eta = y[3:6, None]
    zeta = y[6:9, None]
    Omega = y[9:12, None]
    Lambda = y[12:15, None]
    Upsilon = y[15:18, None]
    return sasaki_omega_dot_cols(eta, zeta, Omega, Lambda, Upsilon)[:, 0]


def rhs_sasaki(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    rho = y[0:3]
    eta = y[3:6]
    zeta = y[6:9]
    Omega = y[9:12]
    Lambda = y[12:15]
    Upsilon = y[15:18]

    dy = np.zeros_like(y)
    dy[0:3] = apply_right_jacobian_inv(rho, Omega)
    dy[3:6] = Lambda + 0.5 * cross_cols(eta, Omega)
    dy[6:9] = Upsilon + 0.5 * cross_cols(zeta, Omega) + 0.125 * cross_cols(cross_cols(eta, Omega), eta)
    dy[9:12] = sasaki_omega_dot_cols(eta, zeta, Omega, Lambda, Upsilon)
    dy[12:15] = (
        0.5 * cross_cols(Lambda, Omega)
        + 0.125 * cross_cols(cross_cols(eta, Upsilon), eta)
        + 0.125 * cross_cols(cross_cols(eta, Omega), Upsilon)
    )
    dy[15:18] = 0.5 * cross_cols(Upsilon, Omega)
    return dy


def rhs_quintic(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    rho = y[0:3]
    Omega = y[3:6]
    A = y[6:9]
    eta = y[9:12]
    zeta = y[12:15]
    chi = y[15:18]

    dy = np.zeros_like(y)
    dy[0:3] = apply_right_jacobian_inv(rho, Omega)
    dy[3:6] = A
    dy[6:9] = eta - 0.5 * cross_cols(Omega, A)
    dy[9:12] = zeta - 0.5 * cross_cols(Omega, eta)
    dy[12:15] = chi
    dy[15:18] = (
        4.0 * zeta
        + 4.0 * cross_cols(Omega, cross_cols(Omega, A))
        - 4.0 * A
        - cross_cols(Omega, chi)
        - 0.5 * cross_cols(A, zeta)
        - 0.5 * cross_cols(Omega, cross_cols(Omega, zeta))
        - 0.25 * cross_cols(cross_cols(A, eta), Omega)
    )
    return dy


# ============================================================
# 3b) Left-invariant rigid-body metric quintics in tension
# ============================================================


def make_inertia(J_diag: Iterable[float]) -> Tuple[np.ndarray, np.ndarray]:
    J_diag = np.asarray(list(J_diag), dtype=float)
    if J_diag.shape != (3,):
        raise ValueError(f"--J must contain three positive diagonal entries, got {J_diag}")
    if np.any(J_diag <= 0):
        raise ValueError(f"--J entries must be positive, got {J_diag}")
    J = np.diag(J_diag)
    Jinv = np.diag(1.0 / J_diag)
    return J, Jinv


def rigid_connection_cols(xi: np.ndarray, eta: np.ndarray, J: np.ndarray, Jinv: np.ndarray) -> np.ndarray:
    """Left-trivialized Levi-Civita connection for the rigid-body metric.

    Implements the convention supplied in the prompt:

        nabla_xi eta = 1/2 * (xi x eta - J^{-1}(J xi x eta + J eta x xi)).

    xi and eta are 3 x N arrays.
    """
    Jxi = J @ xi
    Jeta = J @ eta
    return 0.5 * (
        cross_cols(xi, eta)
        - Jinv @ (cross_cols(Jxi, eta) + cross_cols(Jeta, xi))
    )


def rigid_curvature_cols(xi: np.ndarray, eta: np.ndarray, sigma: np.ndarray, J: np.ndarray, Jinv: np.ndarray) -> np.ndarray:
    """Curvature R(xi, eta) sigma for the left-invariant metric.

    Since the Christoffel map is bilinear with constant coefficients in body
    coordinates, this is just

        nabla_xi nabla_eta sigma - nabla_eta nabla_xi sigma
        - nabla_{xi x eta} sigma.
    """
    return (
        rigid_connection_cols(xi, rigid_connection_cols(eta, sigma, J, Jinv), J, Jinv)
        - rigid_connection_cols(eta, rigid_connection_cols(xi, sigma, J, Jinv), J, Jinv)
        - rigid_connection_cols(cross_cols(xi, eta), sigma, J, Jinv)
    )


def rigid_covaccel_endpoint(Omega: np.ndarray, dotOmega: np.ndarray, covAccel: np.ndarray | None, J: np.ndarray, Jinv: np.ndarray) -> np.ndarray:
    """Endpoint Omega^(1).

    If covAccel is supplied in the JSON through covAccel0/covAccel1/A0/A1, use
    it directly. Otherwise convert ordinary body angular acceleration dotOmega
    to covariant acceleration via

        Omega^(1) = dotOmega + nabla_Omega Omega.
    """
    Omega = np.asarray(Omega, dtype=float).reshape(3, 1)
    dotOmega = np.asarray(dotOmega, dtype=float).reshape(3, 1)
    if covAccel is not None:
        return np.asarray(covAccel, dtype=float).reshape(3)
    return (dotOmega + rigid_connection_cols(Omega, Omega, J, Jinv))[:, 0]


def make_rigid_quintic_guess(bc: BoundaryData, t: np.ndarray, J: np.ndarray, Jinv: np.ndarray) -> np.ndarray:
    """Initial guess for the rigid-body quintic BVP.

    The guess starts from a log-chart quintic and recursively converts ordinary
    derivatives into covariant derivatives using

        Omega^(alpha+1) = d/dt Omega^(alpha) + nabla_Omega Omega^(alpha).
    """
    rho, Omega, dotOmega = initial_guess_common(bc, t)

    Gamma00 = rigid_connection_cols(Omega, Omega, J, Jinv)
    W1 = dotOmega + Gamma00

    W1_dot = np.gradient(W1, t, axis=1, edge_order=2)
    W2 = W1_dot + rigid_connection_cols(Omega, W1, J, Jinv)

    W2_dot = np.gradient(W2, t, axis=1, edge_order=2)
    W3 = W2_dot + rigid_connection_cols(Omega, W2, J, Jinv)

    W3_dot = np.gradient(W3, t, axis=1, edge_order=2)
    W4 = W3_dot + rigid_connection_cols(Omega, W3, J, Jinv)

    return np.vstack([rho, Omega, W1, W2, W3, W4])


def rhs_rigid_quintic_factory(J: np.ndarray, Jinv: np.ndarray, eps1: float, eps2: float):
    eps1 = float(eps1)
    eps2 = float(eps2)

    def rhs_rigid_quintic(t: np.ndarray, y: np.ndarray) -> np.ndarray:
        rho = y[0:3]
        W0 = y[3:6]
        W1 = y[6:9]
        W2 = y[9:12]
        W3 = y[12:15]
        W4 = y[15:18]

        dy = np.zeros_like(y)
        dy[0:3] = apply_right_jacobian_inv(rho, W0)

        dy[3:6] = W1 - rigid_connection_cols(W0, W0, J, Jinv)
        dy[6:9] = W2 - rigid_connection_cols(W0, W1, J, Jinv)
        dy[9:12] = W3 - rigid_connection_cols(W0, W2, J, Jinv)
        dy[12:15] = W4 - rigid_connection_cols(W0, W3, J, Jinv)

        curv_12_0 = rigid_curvature_cols(W1, W2, W0, J, Jinv)
        curv_30_0 = rigid_curvature_cols(W3, W0, W0, J, Jinv)
        curv_10_0 = rigid_curvature_cols(W1, W0, W0, J, Jinv)
        dy[15:18] = (
            - rigid_connection_cols(W0, W4, J, Jinv)
            - curv_12_0
            - curv_30_0
            + (eps2 ** 2) * W3
            + (eps2 ** 2) * curv_10_0
            - (eps1 ** 2) * W1
        )
        return dy

    return rhs_rigid_quintic


def solve_rigid_quintic(
    bc: BoundaryData,
    nodes: int,
    tol: float,
    max_nodes: int,
    verbose: int,
    J: np.ndarray,
    Jinv: np.ndarray,
    eps1: float,
    eps2: float,
):
    t = np.linspace(0.0, bc.T, nodes)
    y_guess = make_rigid_quintic_guess(bc, t, J, Jinv)
    rho_f = so3_log(bc.R0.T @ bc.R1)
    A0 = rigid_covaccel_endpoint(bc.Omega0, bc.dotOmega0, bc.covAccel0, J, Jinv)
    A1 = rigid_covaccel_endpoint(bc.Omega1, bc.dotOmega1, bc.covAccel1, J, Jinv)

    def bc_fun(ya, yb):
        return np.r_[
            ya[0:3],
            yb[0:3] - rho_f,
            ya[3:6] - bc.Omega0,
            yb[3:6] - bc.Omega1,
            ya[6:9] - A0,
            yb[6:9] - A1,
        ]

    sol = solve_bvp(
        rhs_rigid_quintic_factory(J, Jinv, eps1, eps2),
        bc_fun,
        t,
        y_guess,
        tol=tol,
        max_nodes=max_nodes,
        verbose=verbose,
    )
    return sol



def solve_rigid_quintic_natural(
    bc: BoundaryData,
    nodes: int,
    tol: float,
    max_nodes: int,
    verbose: int,
    J: np.ndarray,
    Jinv: np.ndarray,
    eps1: float,
    eps2: float,
):
    """Solve the left-invariant quintic-in-tension BVP with free acceleration endpoints.

    This implements the natural-boundary version appropriate for comparing
    against Riemannian cubics under the same prescribed attitude and angular
    velocity boundary data. The fixed endpoint conditions are

        R(0)=R0, R(T)=R1, Omega0(0)=Omega0, Omega0(T)=Omega1,

    while the covariant accelerations Omega1(0), Omega1(T) are free. The
    corresponding natural boundary conditions are

        Omega2(0)=0, Omega2(T)=0,

    where Omega2 = (R^T nabla^2_{Rdot} Rdot)^vee is the higher-order input u.
    """
    t = np.linspace(0.0, bc.T, nodes)
    y_guess = make_rigid_quintic_natural_guess(bc, t, J, Jinv)

    rho_f = so3_log(bc.R0.T @ bc.R1)

    def bc_fun(ya, yb):
        return np.r_[
            ya[0:3],
            yb[0:3] - rho_f,
            ya[3:6] - bc.Omega0,
            yb[3:6] - bc.Omega1,
            ya[9:12],      # Omega^(2)(0) = u(0) = 0
            yb[9:12],      # Omega^(2)(T) = u(T) = 0
        ]

    sol = solve_bvp(
        rhs_rigid_quintic_factory(J, Jinv, eps1, eps2),
        bc_fun,
        t,
        y_guess,
        tol=tol,
        max_nodes=max_nodes,
        verbose=verbose,
    )
    return sol


def package_rigid_quintic_solution(sol, bc: BoundaryData, N: int, J: np.ndarray, Jinv: np.ndarray, eps1: float, eps2: float) -> Dict[str, np.ndarray]:
    t = np.linspace(0.0, bc.T, N)
    y = sol.sol(t)
    rho = y[0:3]
    R = reconstruct_R(bc.R0, rho)

    W0 = y[3:6]
    W1 = y[6:9]
    W2 = y[9:12]
    W3 = y[12:15]
    W4 = y[15:18]

    # Ordinary angular derivatives needed by the closed-loop tracking simulator.
    dotOmega = W1 - rigid_connection_cols(W0, W0, J, Jinv)
    W1dot = W2 - rigid_connection_cols(W0, W1, J, Jinv)
    ddotOmega = (
        W1dot
        - rigid_connection_cols(dotOmega, W0, J, Jinv)
        - rigid_connection_cols(W0, dotOmega, J, Jinv)
    )

    return {
        "t": t,
        "R_d": R,
        "Omega_d": W0.T,
        "dotOmega_d": dotOmega.T,
        "ddotOmega_d": ddotOmega.T,
        "rho": rho.T,
        "Omega_cov_0": W0.T,
        "Omega_cov_1": W1.T,
        "Omega_cov_2": W2.T,
        "Omega_cov_3": W3.T,
        "Omega_cov_4": W4.T,
        "J_diag": np.diag(J),
        "eps1": np.array(float(eps1)),
        "eps2": np.array(float(eps2)),
    }




# ============================================================
# 3c) Left-invariant rigid-body metric Riemannian cubics
# ============================================================


def cubic_coefficients(q0, qT, qd0, qdT, T):
    """Coefficients a[0:4, dim] for q(t)=sum_i a[i] t^i."""
    q0 = np.asarray(q0, dtype=float)
    qT = np.asarray(qT, dtype=float)
    qd0 = np.asarray(qd0, dtype=float)
    qdT = np.asarray(qdT, dtype=float)

    dim = q0.size
    coeff = np.zeros((4, dim))
    coeff[0] = q0
    coeff[1] = qd0

    M = np.array([[T**2, T**3], [2*T, 3*T**2]], dtype=float)
    rhs = np.vstack([
        qT - (coeff[0] + coeff[1] * T),
        qdT - coeff[1],
    ])
    coeff[2:4] = np.linalg.solve(M, rhs)
    return coeff




def septic_coefficients(q0, qT, qd0, qdT, qdd0, qddT, qddd0, qdddT, T):
    """Coefficients a[0:8, dim] for q(t)=sum_i a[i] t^i.

    This is used only for initial guesses. In the natural quintic BVP, setting
    the endpoint third derivatives to zero gives a local-chart analogue of
    u(0)=u(T)=0.
    """
    q0 = np.asarray(q0, dtype=float)
    qT = np.asarray(qT, dtype=float)
    qd0 = np.asarray(qd0, dtype=float)
    qdT = np.asarray(qdT, dtype=float)
    qdd0 = np.asarray(qdd0, dtype=float)
    qddT = np.asarray(qddT, dtype=float)
    qddd0 = np.asarray(qddd0, dtype=float)
    qdddT = np.asarray(qdddT, dtype=float)

    dim = q0.size
    coeff = np.zeros((8, dim))
    coeff[0] = q0
    coeff[1] = qd0
    coeff[2] = 0.5 * qdd0
    coeff[3] = (1.0 / 6.0) * qddd0

    powers = np.arange(4, 8)
    M = np.zeros((4, 4))
    for row, deriv in enumerate([0, 1, 2, 3]):
        for col, pwr in enumerate(powers):
            factor = 1.0
            for j in range(deriv):
                factor *= pwr - j
            M[row, col] = factor * T ** (pwr - deriv)

    known_T = np.zeros((4, dim))
    for deriv in [0, 1, 2, 3]:
        known_T[deriv] = eval_poly(coeff[:4], np.array([T]), deriv=deriv)[:, 0]

    rhs = np.vstack([
        qT - known_T[0],
        qdT - known_T[1],
        qddT - known_T[2],
        qdddT - known_T[3],
    ])
    coeff[4:8] = np.linalg.solve(M, rhs)
    return coeff


def make_rigid_quintic_natural_guess(bc: BoundaryData, t: np.ndarray, J: np.ndarray, Jinv: np.ndarray) -> np.ndarray:
    """Initial guess for the natural left-invariant quintic BVP.

    The guess uses a septic Hermite curve in the log chart with fixed position
    and velocity endpoints, zero chart acceleration endpoints, and zero chart
    jerk endpoints. The zero chart jerk condition is a useful Euclidean proxy
    for the natural boundary conditions Omega^(2)(0)=Omega^(2)(T)=0.
    """
    rho_f = so3_log(bc.R0.T @ bc.R1)
    rho_dot0 = bc.Omega0
    rho_dotT = right_jacobian_inv(rho_f) @ bc.Omega1
    coeff = septic_coefficients(
        np.zeros(3), rho_f,
        rho_dot0, rho_dotT,
        np.zeros(3), np.zeros(3),
        np.zeros(3), np.zeros(3),
        bc.T,
    )
    rho = eval_poly(coeff, t, 0)
    rho_dot = eval_poly(coeff, t, 1)
    Omega = apply_right_jacobian(rho, rho_dot)

    dotOmega = np.gradient(Omega, t, axis=1, edge_order=2)
    W1 = dotOmega + rigid_connection_cols(Omega, Omega, J, Jinv)

    W1_dot = np.gradient(W1, t, axis=1, edge_order=2)
    W2 = W1_dot + rigid_connection_cols(Omega, W1, J, Jinv)
    W2[:, 0] = 0.0
    W2[:, -1] = 0.0

    W2_dot = np.gradient(W2, t, axis=1, edge_order=2)
    W3 = W2_dot + rigid_connection_cols(Omega, W2, J, Jinv)

    W3_dot = np.gradient(W3, t, axis=1, edge_order=2)
    W4 = W3_dot + rigid_connection_cols(Omega, W3, J, Jinv)

    return np.vstack([rho, Omega, W1, W2, W3, W4])

def make_rigid_cubic_guess(bc: BoundaryData, t: np.ndarray, J: np.ndarray, Jinv: np.ndarray) -> np.ndarray:
    """Initial guess for the left-invariant Riemannian cubic BVP.

    The guess is a cubic Hermite curve in the local log chart satisfying
    attitude and angular-velocity endpoint data approximately in the chart,
    then converted to covariant derivative variables.
    """
    rho_f = so3_log(bc.R0.T @ bc.R1)
    rho_dot0 = bc.Omega0
    rho_dotT = right_jacobian_inv(rho_f) @ bc.Omega1
    coeff = cubic_coefficients(np.zeros(3), rho_f, rho_dot0, rho_dotT, bc.T)
    rho = eval_poly(coeff, t, 0)
    rho_dot = eval_poly(coeff, t, 1)
    Omega = apply_right_jacobian(rho, rho_dot)

    dotOmega = np.gradient(Omega, t, axis=1, edge_order=2)
    W1 = dotOmega + rigid_connection_cols(Omega, Omega, J, Jinv)
    W1_dot = np.gradient(W1, t, axis=1, edge_order=2)
    W2 = W1_dot + rigid_connection_cols(Omega, W1, J, Jinv)
    return np.vstack([rho, Omega, W1, W2])


def rhs_rigid_cubic_factory(J: np.ndarray, Jinv: np.ndarray):
    def rhs_rigid_cubic(t: np.ndarray, y: np.ndarray) -> np.ndarray:
        rho = y[0:3]
        W0 = y[3:6]
        W1 = y[6:9]
        W2 = y[9:12]

        dy = np.zeros_like(y)
        dy[0:3] = apply_right_jacobian_inv(rho, W0)
        dy[3:6] = W1 - rigid_connection_cols(W0, W0, J, Jinv)
        dy[6:9] = W2 - rigid_connection_cols(W0, W1, J, Jinv)
        dy[9:12] = (
            - rigid_connection_cols(W0, W2, J, Jinv)
            - rigid_curvature_cols(W1, W0, W0, J, Jinv)
        )
        return dy

    return rhs_rigid_cubic


def solve_rigid_cubic(
    bc: BoundaryData,
    nodes: int,
    tol: float,
    max_nodes: int,
    verbose: int,
    J: np.ndarray,
    Jinv: np.ndarray,
):
    t = np.linspace(0.0, bc.T, nodes)
    y_guess = make_rigid_cubic_guess(bc, t, J, Jinv)
    rho_f = so3_log(bc.R0.T @ bc.R1)

    def bc_fun(ya, yb):
        return np.r_[
            ya[0:3],
            yb[0:3] - rho_f,
            ya[3:6] - bc.Omega0,
            yb[3:6] - bc.Omega1,
        ]

    sol = solve_bvp(
        rhs_rigid_cubic_factory(J, Jinv),
        bc_fun,
        t,
        y_guess,
        tol=tol,
        max_nodes=max_nodes,
        verbose=verbose,
    )
    return sol


def package_rigid_cubic_solution(sol, bc: BoundaryData, N: int, J: np.ndarray, Jinv: np.ndarray) -> Dict[str, np.ndarray]:
    t = np.linspace(0.0, bc.T, N)
    y = sol.sol(t)
    rho = y[0:3]
    R = reconstruct_R(bc.R0, rho)

    W0 = y[3:6]
    W1 = y[6:9]
    W2 = y[9:12]

    # Convert covariant derivatives to ordinary body angular derivatives for
    # compatibility with the tracking simulator.
    dotOmega = W1 - rigid_connection_cols(W0, W0, J, Jinv)
    W1dot = W2 - rigid_connection_cols(W0, W1, J, Jinv)
    ddotOmega = (
        W1dot
        - rigid_connection_cols(dotOmega, W0, J, Jinv)
        - rigid_connection_cols(W0, dotOmega, J, Jinv)
    )

    return {
        "t": t,
        "R_d": R,
        "Omega_d": W0.T,
        "dotOmega_d": dotOmega.T,
        "ddotOmega_d": ddotOmega.T,
        "rho": rho.T,
        "Omega_cov_0": W0.T,
        "Omega_cov_1": W1.T,
        "Omega_cov_2": W2.T,
        "J_diag": np.diag(J),
        "method_family": np.array("rigid-cubic"),
    }

# ============================================================
# 4) Boundary conditions and solution packaging
# ============================================================


def solve_sasaki(bc: BoundaryData, nodes: int, tol: float, max_nodes: int, bc_mode: str, verbose: int, initial_sol=None):
    if initial_sol is None:
        t = np.linspace(0.0, bc.T, nodes)
        y_guess = make_sasaki_guess(bc, t)
    else:
        # Reuse the previous continuation solution as the initial guess.
        # Keeping the adaptive mesh is usually better than interpolating onto
        # the coarse initial mesh.
        t = np.asarray(initial_sol.x, dtype=float)
        y_guess = np.asarray(initial_sol.y, dtype=float)
    rho_f = so3_log(bc.R0.T @ bc.R1)
    eta0 = bc.Omega0 if bc.eta0 is None else bc.eta0
    eta1 = bc.Omega1 if bc.eta1 is None else bc.eta1
    zeta0 = bc.dotOmega0 if bc.zeta0 is None else bc.zeta0
    zeta1 = bc.dotOmega1 if bc.zeta1 is None else bc.zeta1

    if bc_mode == "fiber":
        def bc_fun(ya, yb):
            return np.r_[
                ya[0:3],
                yb[0:3] - rho_f,
                ya[3:6] - eta0,
                yb[3:6] - eta1,
                ya[6:9] - zeta0,
                yb[6:9] - zeta1,
            ]
    elif bc_mode == "projected":
        def bc_fun(ya, yb):
            A0 = sasaki_omega_dot_single(ya)
            A1 = sasaki_omega_dot_single(yb)
            return np.r_[
                ya[0:3],
                yb[0:3] - rho_f,
                ya[9:12] - bc.Omega0,
                yb[9:12] - bc.Omega1,
                A0 - bc.dotOmega0,
                A1 - bc.dotOmega1,
            ]
    else:
        raise ValueError("bc_mode must be 'fiber' or 'projected'")

    sol = solve_bvp(rhs_sasaki, bc_fun, t, y_guess, tol=tol, max_nodes=max_nodes, verbose=verbose)
    return sol


def solve_sasaki_continuation(
    bc: BoundaryData,
    nodes: int,
    tol: float,
    max_nodes: int,
    bc_mode: str,
    verbose: int,
    steps: int,
    start_lambda: float = 0.0,
):
    """Solve the Sasaki BVP by homotopy in endpoint derivatives.

    The attitude endpoints are fixed for every subproblem. Angular velocities,
    angular accelerations, and eta/zeta endpoint values are scaled from
    start_lambda to 1. This often avoids the large residuals and overflows that
    occur when jumping directly to aggressive derivative boundary conditions.
    """
    steps = int(steps)
    if steps <= 1:
        return solve_sasaki(bc, nodes, tol, max_nodes, bc_mode, verbose)

    start_lambda = float(start_lambda)
    if not (0.0 <= start_lambda <= 1.0):
        raise ValueError("--sasaki-continuation-start must be in [0, 1]")

    lambdas = np.linspace(start_lambda, 1.0, steps)
    prev = None
    sol = None
    for j, lam in enumerate(lambdas, start=1):
        print(f"  continuation {j}/{steps}: lambda={lam:.4f}")
        bc_lam = scaled_boundary_data(bc, lam)
        sol = solve_sasaki(bc_lam, nodes, tol, max_nodes, bc_mode, verbose, initial_sol=prev)
        print(f"    success={sol.success}, status={sol.status}, nodes={sol.x.size}")
        print(f"    message: {sol.message}")
        if (not sol.success) or (not np.all(np.isfinite(sol.y))):
            print("    stopping continuation because this subproblem did not produce a finite successful solution")
            return sol
        prev = sol
    return sol


def solve_quintic(bc: BoundaryData, nodes: int, tol: float, max_nodes: int, verbose: int):
    t = np.linspace(0.0, bc.T, nodes)
    y_guess = make_quintic_guess(bc, t)
    rho_f = so3_log(bc.R0.T @ bc.R1)

    def bc_fun(ya, yb):
        return np.r_[
            ya[0:3],
            yb[0:3] - rho_f,
            ya[3:6] - bc.Omega0,
            yb[3:6] - bc.Omega1,
            ya[6:9] - bc.dotOmega0,
            yb[6:9] - bc.dotOmega1,
        ]

    sol = solve_bvp(rhs_quintic, bc_fun, t, y_guess, tol=tol, max_nodes=max_nodes, verbose=verbose)
    return sol


def reconstruct_R(R0: np.ndarray, rho: np.ndarray) -> np.ndarray:
    R = np.empty((rho.shape[1], 3, 3), dtype=float)
    for k in range(rho.shape[1]):
        R[k] = R0 @ so3_exp(rho[:, k])
    return R


def package_solution(method: str, sol, bc: BoundaryData, N: int) -> Dict[str, np.ndarray]:
    t = np.linspace(0.0, bc.T, N)
    y = sol.sol(t)
    rho = y[0:3]
    R = reconstruct_R(bc.R0, rho)

    if method == "sasaki":
        eta = y[3:6]
        zeta = y[6:9]
        Omega = y[9:12]
        Lambda = y[12:15]
        Upsilon = y[15:18]
        A = sasaki_omega_dot_cols(eta, zeta, Omega, Lambda, Upsilon)
        ddotOmega = np.gradient(A, t, axis=1, edge_order=2)
        extra = {
            "eta": eta.T,
            "zeta": zeta.T,
            "Lambda": Lambda.T,
            "Upsilon": Upsilon.T,
        }
    elif method == "quintic":
        Omega = y[3:6]
        A = y[6:9]
        eta = y[9:12]
        zeta = y[12:15]
        chi = y[15:18]
        ddotOmega = eta - 0.5 * cross_cols(Omega, A)
        extra = {
            "eta": eta.T,
            "zeta": zeta.T,
            "dotzeta": chi.T,
        }
    else:
        raise ValueError(method)

    return {
        "t": t,
        "R_d": R,
        "Omega_d": Omega.T,
        "dotOmega_d": A.T,
        "ddotOmega_d": ddotOmega.T,
        "rho": rho.T,
        **extra,
    }


def validate_payload(payload: Dict[str, np.ndarray], label: str = "trajectory") -> Tuple[bool, list[str]]:
    """Return whether the payload is finite and contains valid SO(3) samples."""
    messages: list[str] = []
    for key, val in payload.items():
        arr = np.asarray(val)
        if np.issubdtype(arr.dtype, np.number) and not np.all(np.isfinite(arr)):
            messages.append(f"{label}: key '{key}' contains nan or inf")

    R = np.asarray(payload.get("R_d"))
    if R.ndim == 3 and R.shape[1:] == (3, 3) and np.all(np.isfinite(R)):
        I = np.eye(3)
        ortho_err = np.max([np.linalg.norm(Rk.T @ Rk - I, ord="fro") for Rk in R])
        det_err = np.max(np.abs(np.linalg.det(R) - 1.0))
        if ortho_err > 1e-6:
            messages.append(f"{label}: max ||R^T R - I||_F = {ortho_err:.3e}")
        if det_err > 1e-6:
            messages.append(f"{label}: max |det(R)-1| = {det_err:.3e}")
    elif "R_d" in payload:
        messages.append(f"{label}: R_d has invalid shape or non-finite entries")

    return (len(messages) == 0), messages


def save_npz(path: Path, payload: Dict[str, np.ndarray], method: str, bc_mode: str | None, sol) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        **payload,
        method=np.array(method),
        sasaki_bc_mode=np.array("" if bc_mode is None else bc_mode),
        bvp_success=np.array(bool(sol.success)),
        bvp_status=np.array(int(sol.status)),
        bvp_message=np.array(sol.message),
        bvp_nodes=np.array(int(sol.x.size)),
        bvp_rms_residuals=sol.rms_residuals,
    )


def plot_payload(path_prefix: Path, payload: Dict[str, np.ndarray], title: str) -> None:
    import matplotlib.pyplot as plt

    t = payload["t"]
    R = payload["R_d"]
    Omega = payload["Omega_d"]
    A = payload["dotOmega_d"]
    J = payload["ddotOmega_d"]

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    u = np.linspace(0, 2*np.pi, 40)
    v = np.linspace(0, np.pi, 20)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(xs, ys, zs, linewidth=0.25, alpha=0.25)
    labels = ["Re1", "Re2", "Re3"]
    for j, label in enumerate(labels):
        pts = R[:, :, j]
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], label=label)
        ax.scatter(pts[0, 0], pts[0, 1], pts[0, 2], marker="o")
        ax.scatter(pts[-1, 0], pts[-1, 1], pts[-1, 2], marker="x")
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend()
    ax.set_title(f"{title}: projected body axes")
    fig.tight_layout()
    fig.savefig(path_prefix.with_suffix(".sphere.png"), dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(t, np.linalg.norm(Omega, axis=1), label="||Omega||")
    ax.plot(t, np.linalg.norm(A, axis=1), label="||dotOmega||")
    ax.plot(t, np.linalg.norm(J, axis=1), label="||ddotOmega||")
    ax.set_xlabel("t")
    ax.set_ylabel("norm")
    ax.set_title(f"{title}: derivative norms")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path_prefix.with_suffix(".norms.png"), dpi=200)
    plt.close(fig)


# ============================================================
# 5) CLI
# ============================================================


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="version", version=f"%(prog)s {SCRIPT_VERSION}")
    parser.add_argument("--bc", type=str, default=None, help="Boundary-condition JSON file. If omitted, a rest-to-rest 90 deg yaw demo is used.")
    parser.add_argument("--method", choices=["sasaki", "quintic", "rigid-quintic", "rigid-quintic-natural", "rigid-cubic", "both", "all"], default="rigid-quintic")
    parser.add_argument("--sasaki-bc", choices=["projected", "fiber"], default="fiber")
    parser.add_argument("--out", type=str, default="desired_traj.npz", help="Output .npz path. With --method both, suffixes are added.")
    parser.add_argument("--nodes", type=int, default=80, help="Initial BVP mesh nodes.")
    parser.add_argument("--tol", type=float, default=1e-4, help="solve_bvp tolerance.")
    parser.add_argument("--max-nodes", type=int, default=20000)
    parser.add_argument("--plot", action="store_true", help="Save quick diagnostic plots next to the .npz file.")
    parser.add_argument("--save-failed", action="store_true", help="Write failed/non-finite BVP iterates for debugging. Default: do not write them.")
    parser.add_argument("--sasaki-continuation", type=int, default=1, help="Number of homotopy steps for the Sasaki BVP. Use e.g. 8 or 12 for aggressive derivative boundary data.")
    parser.add_argument("--sasaki-continuation-start", type=float, default=0.0, help="Starting homotopy scale for endpoint velocities/accelerations. Default 0.")
    parser.add_argument("--J", nargs=3, type=float, default=[0.082, 0.0845, 0.1377], help="Diagonal inertia tensor for --method rigid-quintic or --method rigid-cubic.")
    parser.add_argument("--eps1", type=float, default=1.0, help="epsilon_1 tension parameter for --method rigid-quintic.")
    parser.add_argument("--eps2", type=float, default=1.0, help="epsilon_2 tension parameter for --method rigid-quintic.")
    parser.add_argument("--verbose", type=int, default=1, choices=[0, 1, 2])
    args = parser.parse_args()

    bc = load_boundary_data(args.bc)
    rho_f = so3_log(bc.R0.T @ bc.R1)
    if np.linalg.norm(rho_f) > 0.95 * np.pi:
        print("WARNING: final rotation is close to pi in the local log chart; BVP convergence may be poor.")

    if args.method == "both":
        methods = ["sasaki", "quintic"]
    elif args.method == "all":
        methods = ["sasaki", "quintic", "rigid-quintic", "rigid-quintic-natural", "rigid-cubic"]
    else:
        methods = [args.method]

    J, Jinv = make_inertia(args.J)
    out = Path(args.out)

    for method in methods:
        print(f"Solving {method} BVP...")
        if method == "sasaki":
            if args.sasaki_continuation > 1:
                sol = solve_sasaki_continuation(
                    bc, args.nodes, args.tol, args.max_nodes, args.sasaki_bc,
                    args.verbose, args.sasaki_continuation, args.sasaki_continuation_start
                )
            else:
                sol = solve_sasaki(bc, args.nodes, args.tol, args.max_nodes, args.sasaki_bc, args.verbose)
            bc_mode = args.sasaki_bc
        elif method == "quintic":
            sol = solve_quintic(bc, args.nodes, args.tol, args.max_nodes, args.verbose)
            bc_mode = None
        elif method == "rigid-quintic":
            sol = solve_rigid_quintic(bc, args.nodes, args.tol, args.max_nodes, args.verbose, J, Jinv, args.eps1, args.eps2)
            bc_mode = None
        elif method == "rigid-quintic-natural":
            sol = solve_rigid_quintic_natural(bc, args.nodes, args.tol, args.max_nodes, args.verbose, J, Jinv, args.eps1, args.eps2)
            bc_mode = "natural-u0-uT-zero"
        elif method == "rigid-cubic":
            sol = solve_rigid_cubic(bc, args.nodes, args.tol, args.max_nodes, args.verbose, J, Jinv)
            bc_mode = None
        else:
            raise ValueError(method)

        print(f"  success={sol.success}, status={sol.status}, nodes={sol.x.size}")
        print(f"  message: {sol.message}")
        if not sol.success and not args.save_failed:
            print("  NOT writing .npz: BVP did not converge. Use --save-failed only for debugging.")
            continue

        if not np.all(np.isfinite(sol.y)) and not args.save_failed:
            print("  NOT writing .npz: BVP iterate contains nan or inf.")
            continue

        if method == "rigid-quintic":
            payload = package_rigid_quintic_solution(sol, bc, bc.N, J, Jinv, args.eps1, args.eps2)
        elif method == "rigid-quintic-natural":
            payload = package_rigid_quintic_solution(sol, bc, bc.N, J, Jinv, args.eps1, args.eps2)
            payload["method_family"] = np.array("rigid-quintic-natural")
            payload["natural_bc"] = np.array("Omega_cov_2(0)=Omega_cov_2(T)=0")
        elif method == "rigid-cubic":
            payload = package_rigid_cubic_solution(sol, bc, bc.N, J, Jinv)
        else:
            payload = package_solution(method, sol, bc, bc.N)
        valid, messages = validate_payload(payload, method)
        if not valid:
            print("  trajectory validation failed:")
            for msg in messages:
                print(f"    - {msg}")
            if not args.save_failed:
                print("  NOT writing .npz. Use --save-failed only for debugging.")
                continue

        if len(methods) > 1:
            out_path = out.with_name(f"{out.stem}_{method}{out.suffix or '.npz'}")
        else:
            out_path = out
        save_npz(out_path, payload, method, bc_mode, sol)
        print(f"  wrote {out_path}")

        if args.plot:
            plot_payload(out_path.with_suffix(""), payload, method)
            print(f"  wrote {out_path.with_suffix('.sphere.png')} and {out_path.with_suffix('.norms.png')}")


if __name__ == "__main__":
    main()
