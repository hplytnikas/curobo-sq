from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker, MarkerArray

import numpy as np


def _signed_power(val, exponent):
    """Helper to calculate the signed power function used in superquadric equations."""
    return np.sign(val) * (np.abs(val) ** exponent)


def sample_superquadric_mesh(radii, shape, grid_res=30):
    """
    Generates vertices for a Triangle List marker representing a superellipsoid surface.

    radii: [a1, a2, a3] dimensions
    shape: [e1, e2] exponents controlling squareness/roundness
    """
    a1, a2, a3 = radii
    e1, e2 = shape

    # Sample parameter space: eta (latitude, -pi/2 to pi/2), omega (longitude, -pi to pi)
    eta = np.linspace(-np.pi / 2.0, np.pi / 2.0, grid_res)
    omega = np.linspace(-np.pi, np.pi, grid_res)

    # Create grid matrix of surface points
    grid = np.zeros((grid_res, grid_res, 3))
    for i in range(grid_res):
        for j in range(grid_res):
            c_eta, s_eta = np.cos(eta[i]), np.sin(eta[i])
            c_w, s_w = np.cos(omega[j]), np.sin(omega[j])

            # Parametric definitions of a superellipsoid
            x = a1 * _signed_power(c_eta, e1) * _signed_power(c_w, e2)
            y = a2 * _signed_power(c_eta, e1) * _signed_power(s_w, e2)
            z = a3 * _signed_power(s_eta, e1)
            grid[i, j] = [x, y, z]

    # Structure triangles out of grid quads
    vertices = []
    for i in range(grid_res - 1):
        for j in range(grid_res):
            next_j = (j + 1) % grid_res

            p0 = grid[i, j]
            p1 = grid[i + 1, j]
            p2 = grid[i + 1, next_j]
            p3 = grid[i, next_j]

            for pt in [p0, p2, p1]:
                ros_pt = Point()
                ros_pt.x, ros_pt.y, ros_pt.z = float(pt[0]), float(pt[1]), float(pt[2])
                vertices.append(ros_pt)

            for pt in [p0, p3, p2]:
                ros_pt = Point()
                ros_pt.x, ros_pt.y, ros_pt.z = float(pt[0]), float(pt[1]), float(pt[2])
                vertices.append(ros_pt)

    return vertices
