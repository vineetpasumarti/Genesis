import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from TrackModel import SplineCenterline
import numpy as np


def plot_track_with_gates(traj_num=3):
    # Initialize track model
    track = SplineCenterline(Traj=traj_num)

    # Create 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot centerline trajectory
    s_vec = np.linspace(0, track.s_tag[-1], 1000)
    centerline = np.array([track.f_xc(s) for s in s_vec])
    ax.plot(centerline[:, 0], centerline[:, 1], centerline[:, 2], 'b--', linewidth=1, label='Centerline')

    # Plot gates with orientation
    for i in range(track.NumOfGates):
        # Get gate position and orientation
        pos = track.x_gates[i]
        T, N, B = track.get_gate_orientation(i)

        # Plot gate frame
        ax.quiver(pos[0], pos[1], pos[2],
                  T[0], T[1], T[2],
                  length=0.5, color='r', label='Tangent' if i == 0 else "")
        ax.quiver(pos[0], pos[1], pos[2],
                  N[0], N[1], N[2],
                  length=0.5, color='g', label='Normal' if i == 0 else "")
        ax.quiver(pos[0], pos[1], pos[2],
                  B[0], B[1], B[2],
                  length=0.5, color='b', label='Binormal' if i == 0 else "")

        # Plot gate rectangle
        if track.gate_type == 'square':
            gate = track.gate_size * np.array([
                [0, 1, 1], [0, 1, -1], [0, -1, -1], [0, -1, 1], [0, 1, 1]
            ])
            R = np.column_stack([T, N, B])
            gate_world = pos + (R @ gate.T).T
            ax.plot(gate_world[:, 0], gate_world[:, 1], gate_world[:, 2], 'm-', linewidth=2)

    # Plot settings
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'Racetrack Visualization (Trajectory {traj_num})')
    ax.legend()
    ax.view_init(elev=30, azim=-60)
    ax.set_box_aspect([1, 1, 0.5])  # Better aspect ratio for 3D racing tracks
    plt.show()


# Usage
plot_track_with_gates(traj_num=1)  # Use trajectory number from TrackModel cases


# import matplotlib.pyplot as plt
# from TrackModel import SplineCenterline
#
# # Instantiate the SplineCenterline class with the desired track configuration
# track = SplineCenterline(Traj=4)
#
# # Call the Plot3D_Track method to generate the plot
# fig, ax, ax1 = track.Plot3D_Track()
#
# # Display the plot
# plt.show()
