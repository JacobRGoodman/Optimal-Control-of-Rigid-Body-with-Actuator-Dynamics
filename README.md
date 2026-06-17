This repository contains the Python code used to generate the SO(3) attitude trajectories and tracking simulations for the conference paper “A Generalized Sasaki Metric on the Second-Order Tangent Bundle.”
The main files are generate_nominal_so3_trajectory.py and simulate_so3_actuated_tracking.py. The first script generates nominal desired trajectories on SO(3). 
The second script simulates rigid-body attitude tracking with first-order torque actuator dynamics and produces the comparison plots. 
The scripts animate_so3_rigidbody.py and split_saved_comparison_figures.py are optional helper scripts for making animations and splitting saved comparison figures.

The code requires Python 3 and the packages numpy, scipy, matplotlib, and pillow. 

To reproduce the simulations, first create a boundary-condition JSON file, or use those provided. For example, save the following as bc_example.json:
{
  "T": 2.0,
  "N": 1001,
  "R0": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
  "R1_rotvec": [0.0, 0.0, 1.5707963267948966],
  "Omega0": [0.0, 0.0, 0.0],
  "Omega1": [0.0, 0.0, 0.0],
  "dotOmega0": [0.0, 0.0, 0.0],
  "dotOmega1": [0.0, 0.0, 0.0]
}

Where [0, T] is the intergation time interval for the simulations, N is the number of time-steps, R0 is the initial orientation in SO(3), R1_rotvec specifies the final orientation (the direction is the rotation axis, and the magnitude is the
rotation angle along that axis), Omega0 and Omega1 are the initial and final body angular velocities, and dotOmega0 and dotOmega1 are the initial and final body angular accelerations. 

Then generate the Riemannian quintic-in-tension trajectory:
python generate_nominal_so3_trajectory.py \
  --bc bc_example.json \
  --method rigid-quintic-natural \
  --eps1 0.01 \
  --eps2 0.5 \
  --out desired_rigid_quintic_natural.npz \
  --nodes 100 \
  --tol 1e-3 \
  --max-nodes 50000 \
  --plot

Here, eps1 and eps2 are the tension parameters, and nodes, tol, and max-nodes are parameters used for a BVP solver.
Generate the Riemannian cubic comparison trajectory:
python generate_nominal_so3_trajectory.py \
  --bc bc_example.json \
  --method rigid-cubic \
  --out desired_rigid_cubic.npz \
  --nodes 60 \
  --tol 1e-4 \
  --plot

Finally, run the tracking simulation comparing the two nominal trajectories:
python simulate_so3_actuated_tracking.py \
  desired_rigid_quintic_natural.npz \
  --trajectory2 desired_rigid_cubic.npz \
  --label1 "Quintic in tension" \
  --label2 "Cubic" \
  --traj1-metric left \
  --traj2-metric left \
  --out compare_quintic_cubic \
  --R0-rotvec-deg 0 25 0 \
  --Omega0-offset 0.3 -0.2 0.1 \
  --tau0-mode zero

The output folder compare_quintic_cubic contains the simulation results, numerical summaries, and plots. The most relevant files are paper_plot1_tracking_errors.png, paper_plot2_weighted_L2_u_M.png, paper_metric_summary.txt, 
and comparison_metric_summary.txt. The subfolders Quintic_in_tension and Cubic contain the individual simulation results for each nominal trajectory.
The default inertia tensor used in the code is
J = diag(0.082, 0.0845, 0.1377)
and the actuator parameters used by default are
C = 0.75
K = 0.15
These values match the simulation setup used for the comparison in the paper. The quintic-in-tension trajectory uses the tension parameters
eps1 = 0.01
eps2 = 0.5
The file simulation_results.npz stores the simulated attitude, angular velocity, actuator torque, desired trajectory, tracking errors, actuator command, and diagnostic norms. These files can be loaded in Python with numpy.load.
To make an animation of one of the simulated trajectories, run for example:
python animate_so3_rigidbody.py \
  compare_quintic_cubic/Quintic_in_tension/simulation_results.npz \
  --out attitude_tracking.mp4 \
  --show-desired \
  --trace \
  --fps 30
If ffmpeg is not installed, the animation script will save a GIF instead of an MP4.
The script split_saved_comparison_figures.py can be used after the comparison simulation if separate subplot images are needed:
python split_saved_comparison_figures.py compare_quintic_cubic
The numerical results depend on the boundary conditions in the JSON file. To reproduce a different maneuver, edit bc_example.json and rerun the same three main commands above.
