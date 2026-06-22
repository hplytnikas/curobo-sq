"""Visualization helpers for the object-size scaling ("sofa") benchmark.

Kept in its own module (rather than reusing ``benchmark_sq_vs_mesh``'s
viewer code) so the scaling experiment can evolve its own viewer behaviour
without touching the tabletop benchmark.

Used by ``benchmark_sofa_scaling.py`` when invoked with ``--visualize``.
"""

from __future__ import annotations

import time

from curobo.types import ContentPath, JointState
from curobo.viewer import ViserVisualizer


def make_visualizer(port: int) -> ViserVisualizer:
    """Spin up a Viser viewer with the Franka robot (no control frames/spheres)."""
    viz = ViserVisualizer(
        content_path=ContentPath(robot_config_file="franka.yml"),
        connect_ip="0.0.0.0", connect_port=port,
        add_control_frames=False, visualize_robot_spheres=False,
    )
    print(f"Viser running at http://localhost:{port}")
    return viz


def animate(viser_viz, pos_per_step, joint_names, playback_hz: float) -> None:
    """Play back an interpolated trajectory on the viewer at ``playback_hz``."""
    for t in range(pos_per_step.shape[0]):
        viser_viz.set_joint_state(
            JointState.from_position(
                pos_per_step[t:t + 1], joint_names=joint_names
            ).squeeze(0)
        )
        time.sleep(1.0 / playback_hz)
