#!/usr/bin/env python3
"""
Animate SO(3) attitude trajectories as a rotating rectangular prism.

The body is drawn as a rectangular prism centered at the origin. A fixed
body-frame reference vector b_body can be attached to the body; by default
b_body = e3, so the inertial-space vector is b(t) = R(t) e3.

Accepted input .npz files:
- desired trajectory files with keys: t, R_d, Omega_d, dotOmega_d, ddotOmega_d
- simulation result files with keys that include: t, R, ... and optionally R_d

Outputs:
- MP4 if ffmpeg is available
- otherwise GIF via pillow

Example:
    python animate_so3_rigidbody.py so3_quintic_120deg/simulation_results.npz --out attitude_video_quintic_120deg.mp4 --show-desired --fps 30
    python animate_so3_rigidbody.py compare_cubic_quintic/Quintic_in_tension/simulation_results.npz --out attitude_video_rigid_quintic.mp4 --show-desired --fps 30
    python animate_so3_rigidbody.py so3_rigid_quintic_120deg/simulation_results.npz --out attitude_video_rigid_quintic_120deg.mp4 --show-desired --fps 30
    python animate_so3_rigidbody.py so3_sasaki_120deg/simulation_results.npz --out attitude_video_sasaki_120deg.mp4 --show-desired --fps 30

    python animate_so3_rigidbody.py so3_quintic_endpoint_spin/simulation_results.npz --out attitude_video_quintic_endpoint_spin.mp4 --show-desired --fps 30
    python animate_so3_rigidbody.py so3_rigid_quintic_endpoint_spin/simulation_results.npz --out attitude_video_rigid_quintic_endpoint_spin.mp4 --show-desired --fps 30
    python animate_so3_rigidbody.py so3_sasaki_endpoint_spin/simulation_results.npz --out attitude_video_sasaki_endpoint_spin.mp4 --show-desired --fps 30

    python animate_so3_rigidbody.py so3_quintic_nonzero_spin_and_acceleration/simulation_results.npz --out attitude_video_quintic_nonzero_spin_and_acceleration.mp4 --show-desired --fps 30
    python animate_so3_rigidbody.py so3_rigid_quintic_nonzero_spin_and_acceleration/simulation_results.npz --out attitude_video_rigid_quintic_nonzero_spin_and_acceleration.mp4 --show-desired --fps 30
"""

import argparse
import os
from pathlib import Path
import shutil

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def hat(x):
    x = np.asarray(x, dtype=float).reshape(3)
    return np.array([[0.0, -x[2], x[1]], [x[2], 0.0, -x[0]], [-x[1], x[0], 0.0]])


def rot_from_axis_angle(rotvec):
    rotvec = np.asarray(rotvec, dtype=float).reshape(3)
    theta = np.linalg.norm(rotvec)
    if theta < 1e-14:
        return np.eye(3)
    k = rotvec / theta
    K = hat(k)
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def load_attitude_data(path):
    data = np.load(path)
    keys = set(data.files)

    if 'R' in keys:
        R = data['R']
        source = 'actual'
    elif 'R_d' in keys:
        R = data['R_d']
        source = 'desired'
    else:
        raise ValueError(f"Could not find 'R' or 'R_d' in {path}.")

    if 't' not in keys:
        raise ValueError(f"Could not find 't' in {path}.")

    t = data['t']
    R_d = data['R_d'] if 'R_d' in keys else None

    return {
        't': np.asarray(t),
        'R': np.asarray(R),
        'R_d': None if R_d is None else np.asarray(R_d),
        'keys': sorted(data.files),
        'source': source,
    }


def prism_geometry(length=1.2, width=0.6, height=0.3):
    lx = length / 2.0
    ly = width / 2.0
    lz = height / 2.0
    vertices = np.array([
        [-lx, -ly, -lz],
        [ lx, -ly, -lz],
        [ lx,  ly, -lz],
        [-lx,  ly, -lz],
        [-lx, -ly,  lz],
        [ lx, -ly,  lz],
        [ lx,  ly,  lz],
        [-lx,  ly,  lz],
    ])
    faces = [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [0, 1, 5, 4],
        [2, 3, 7, 6],
        [1, 2, 6, 5],
        [0, 3, 7, 4],
    ]
    return vertices, faces


def rotated_faces(vertices, faces, R, center=None):
    if center is None:
        center = np.zeros(3)
    V = (R @ vertices.T).T + center[None, :]
    return [[V[idx] for idx in face] for face in faces], V


def set_axes_equal(ax, radius=1.4):
    ax.set_xlim([-radius, radius])
    ax.set_ylim([-radius, radius])
    ax.set_zlim([-radius, radius])
    try:
        ax.set_box_aspect([1, 1, 1])
    except Exception:
        pass


def subsample_indices(n, max_frames):
    if n <= max_frames:
        return np.arange(n)
    return np.unique(np.round(np.linspace(0, n - 1, max_frames)).astype(int))


def maybe_get_writer(desired_out):
    out = Path(desired_out)
    ffmpeg_ok = shutil.which('ffmpeg') is not None
    if out.suffix.lower() == '.mp4' and ffmpeg_ok:
        return out, 'ffmpeg'
    if out.suffix.lower() == '.gif':
        return out, 'pillow'
    if ffmpeg_ok:
        return out.with_suffix('.mp4'), 'ffmpeg'
    return out.with_suffix('.gif'), 'pillow'


def main():
    p = argparse.ArgumentParser(description='Animate SO(3) rigid body motion as a rotating rectangular prism.')
    p.add_argument('input', help='Input .npz trajectory or simulation result file')
    p.add_argument('--out', default='attitude_animation.mp4', help='Output animation filename (.mp4 preferred, .gif fallback)')
    p.add_argument('--fps', type=int, default=30, help='Frames per second for output video')
    p.add_argument('--max-frames', type=int, default=600, help='Maximum number of animation frames after subsampling')
    p.add_argument('--show-desired', action='store_true', help='Overlay desired attitude if R_d is present')
    p.add_argument('--elev', type=float, default=24.0)
    p.add_argument('--azim', type=float, default=40.0)
    p.add_argument('--body-length', type=float, default=1.2)
    p.add_argument('--body-width', type=float, default=0.6)
    p.add_argument('--body-height', type=float, default=0.3)
    p.add_argument('--body-alpha', type=float, default=0.55)
    p.add_argument('--desired-alpha', type=float, default=0.12)
    p.add_argument('--trace', action='store_true', help='Trace the tip of b(t)=R b_body')
    p.add_argument('--vector-scale', type=float, default=0.9, help='Length of body-fixed reference vector in animation')
    p.add_argument('--b-body', nargs=3, type=float, default=[0.0, 0.0, 1.0], help='Body-fixed reference vector b_body. Default is e3')
    p.add_argument('--title', default=None, help='Optional custom title')
    args = p.parse_args()

    data = load_attitude_data(args.input)
    t = data['t']
    R_seq = data['R']
    R_d_seq = data['R_d']

    if R_seq.ndim != 3 or R_seq.shape[1:] != (3, 3):
        raise ValueError(f"Expected R array of shape (N,3,3); got {R_seq.shape}")

    if args.show_desired and R_d_seq is None:
        print('Warning: --show-desired requested but no R_d found in file. Continuing without desired overlay.')
        args.show_desired = False

    idx = subsample_indices(len(t), args.max_frames)
    t_plot = t[idx]
    R_plot = R_seq[idx]
    R_d_plot = R_d_seq[idx] if (args.show_desired and R_d_seq is not None) else None

    vertices, faces = prism_geometry(args.body_length, args.body_width, args.body_height)
    b_body = np.asarray(args.b_body, dtype=float)
    if np.linalg.norm(b_body) < 1e-14:
        raise ValueError('b_body must be nonzero')
    b_body = b_body / np.linalg.norm(b_body)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    set_axes_equal(ax, radius=1.4)
    ax.view_init(elev=args.elev, azim=args.azim)
    ax.set_xlabel('inertial $e_1$')
    ax.set_ylabel('inertial $e_2$')
    ax.set_zlabel('inertial $e_3$')

    # Light unit sphere wireframe for orientation context.
    uu = np.linspace(0, 2*np.pi, 40)
    vv = np.linspace(0, np.pi, 20)
    xs = np.outer(np.cos(uu), np.sin(vv))
    ys = np.outer(np.sin(uu), np.sin(vv))
    zs = np.outer(np.ones_like(uu), np.cos(vv))
    ax.plot_wireframe(xs, ys, zs, rstride=2, cstride=2, linewidth=0.4, alpha=0.12)

    # Inertial frame arrows.
    axis_len = 1.1
    ax.quiver(0, 0, 0, axis_len, 0, 0, arrow_length_ratio=0.08, linewidth=1.2)
    ax.quiver(0, 0, 0, 0, axis_len, 0, arrow_length_ratio=0.08, linewidth=1.2)
    ax.quiver(0, 0, 0, 0, 0, axis_len, arrow_length_ratio=0.08, linewidth=1.2)
    ax.text(axis_len + 0.05, 0, 0, '$e_1$')
    ax.text(0, axis_len + 0.05, 0, '$e_2$')
    ax.text(0, 0, axis_len + 0.05, '$e_3$')

    # Initial actual prism.
    faces_now, _ = rotated_faces(vertices, faces, R_plot[0])
    prism = Poly3DCollection(faces_now, alpha=args.body_alpha, facecolor='tab:blue', edgecolor='k', linewidths=0.8)
    ax.add_collection3d(prism)

    desired_prism = None
    if R_d_plot is not None:
        faces_des, _ = rotated_faces(vertices, faces, R_d_plot[0])
        desired_prism = Poly3DCollection(faces_des, alpha=args.desired_alpha, facecolor='tab:orange', edgecolor='tab:orange', linewidths=0.7)
        ax.add_collection3d(desired_prism)

    # Body-frame reference vector b(t) = R b_body.
    b0 = args.vector_scale * (R_plot[0] @ b_body)
    b_line, = ax.plot([0, b0[0]], [0, b0[1]], [0, b0[2]], linewidth=3.0, color='tab:red', label='$b(t)=R b_{body}$')
    b_tip, = ax.plot([b0[0]], [b0[1]], [b0[2]], marker='o', color='tab:red')

    trace_line = None
    trace_pts = []
    if args.trace:
        trace_line, = ax.plot([], [], [], color='tab:red', linewidth=1.2, alpha=0.8)

    desired_b_line = None
    desired_b_tip = None
    if R_d_plot is not None:
        bd0 = args.vector_scale * (R_d_plot[0] @ b_body)
        desired_b_line, = ax.plot([0, bd0[0]], [0, bd0[1]], [0, bd0[2]], linewidth=2.0, linestyle='--', color='tab:orange', label='$b_d(t)=R_d b_{body}$')
        desired_b_tip, = ax.plot([bd0[0]], [bd0[1]], [bd0[2]], marker='o', color='tab:orange')

    if args.show_desired and R_d_plot is not None:
        ax.legend(loc='upper left')

    title = args.title or 'Rigid body attitude motion'
    title_text = ax.set_title(f'{title}\n$ t = {t_plot[0]:.2f}$ s')

    def update(frame_idx):
        R = R_plot[frame_idx]
        faces_now, _ = rotated_faces(vertices, faces, R)
        prism.set_verts(faces_now)

        b = args.vector_scale * (R @ b_body)
        b_line.set_data_3d([0, b[0]], [0, b[1]], [0, b[2]])
        b_tip.set_data_3d([b[0]], [b[1]], [b[2]])

        artists = [prism, b_line, b_tip]

        if args.trace:
            trace_pts.append(b.copy())
            pts = np.array(trace_pts)
            trace_line.set_data_3d(pts[:, 0], pts[:, 1], pts[:, 2])
            artists.append(trace_line)

        if R_d_plot is not None:
            Rd = R_d_plot[frame_idx]
            faces_des, _ = rotated_faces(vertices, faces, Rd)
            desired_prism.set_verts(faces_des)
            bd = args.vector_scale * (Rd @ b_body)
            desired_b_line.set_data_3d([0, bd[0]], [0, bd[1]], [0, bd[2]])
            desired_b_tip.set_data_3d([bd[0]], [bd[1]], [bd[2]])
            artists.extend([desired_prism, desired_b_line, desired_b_tip])

        title_text.set_text(f'{title}\n$ t = {t_plot[frame_idx]:.2f}$ s')
        artists.append(title_text)
        return artists

    anim = FuncAnimation(fig, update, frames=len(t_plot), interval=1000.0/args.fps, blit=False)

    out_path, writer = maybe_get_writer(args.out)
    print(f'Saving animation to: {out_path}')
    print(f'Using writer: {writer}')
    if writer == 'ffmpeg':
        anim.save(out_path, writer='ffmpeg', fps=args.fps, dpi=160)
    else:
        anim.save(out_path, writer='pillow', fps=args.fps, dpi=120)
    plt.close(fig)
    print('Done.')


if __name__ == '__main__':
    main()
