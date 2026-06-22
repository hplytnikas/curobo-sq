"""
synthetic_scene.py
Generates fake tabletop point clouds for testing the perception pipeline
without needing a real camera. Simulates a depth sensor with noise.
"""

import numpy as np


def _sample_box(center, size, n_points, noise_std=0.002):
    """Sample points on the surface of a box."""
    cx, cy, cz = center
    sx, sy, sz = size
    points = []
    # distribute points across 6 faces proportionally to face area
    faces = [
        (np.array([sx/2, 0, 0]),  np.array([0, sy, 0]), np.array([0, 0, sz])),
        (np.array([-sx/2, 0, 0]), np.array([0, sy, 0]), np.array([0, 0, sz])),
        (np.array([0, sy/2, 0]),  np.array([sx, 0, 0]), np.array([0, 0, sz])),
        (np.array([0, -sy/2, 0]), np.array([sx, 0, 0]), np.array([0, 0, sz])),
        (np.array([0, 0, sz/2]),  np.array([sx, 0, 0]), np.array([0, sy, 0])),
        (np.array([0, 0, -sz/2]), np.array([sx, 0, 0]), np.array([0, sy, 0])),
    ]
    per_face = n_points // len(faces)
    for normal_offset, u_axis, v_axis in faces:
        u = np.random.uniform(-0.5, 0.5, per_face)
        v = np.random.uniform(-0.5, 0.5, per_face)
        pts = (normal_offset[None, :]
               + u[:, None] * u_axis[None, :]
               + v[:, None] * v_axis[None, :])
        points.append(pts)
    pts = np.vstack(points) + np.array([cx, cy, cz])
    pts += np.random.randn(*pts.shape) * noise_std
    return pts


def _sample_sphere(center, radius, n_points, noise_std=0.002):
    """Sample points on the surface of a sphere."""
    phi = np.random.uniform(0, 2 * np.pi, n_points)
    costheta = np.random.uniform(-1, 1, n_points)
    theta = np.arccos(costheta)
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)
    pts = np.stack([x, y, z], axis=1) + np.array(center)
    pts += np.random.randn(*pts.shape) * noise_std
    return pts


def _sample_cylinder(center, radius, height, n_points, noise_std=0.002):
    """Sample points on the surface of a cylinder."""
    n_side = int(n_points * 0.7)
    n_cap = (n_points - n_side) // 2
    # side
    theta = np.random.uniform(0, 2 * np.pi, n_side)
    z_s = np.random.uniform(-height/2, height/2, n_side)
    side = np.stack([radius * np.cos(theta), radius * np.sin(theta), z_s], axis=1)
    # caps
    r_cap = np.sqrt(np.random.uniform(0, radius**2, n_cap * 2))
    t_cap = np.random.uniform(0, 2 * np.pi, n_cap * 2)
    x_cap = r_cap * np.cos(t_cap)
    y_cap = r_cap * np.sin(t_cap)
    top = np.stack([x_cap[:n_cap], y_cap[:n_cap], np.full(n_cap, height/2)], axis=1)
    bot = np.stack([x_cap[n_cap:], y_cap[n_cap:], np.full(n_cap, -height/2)], axis=1)
    pts = np.vstack([side, top, bot]) + np.array(center)
    pts += np.random.randn(*pts.shape) * noise_std
    return pts


def generate_tabletop_scene(
    n_objects=4,
    table_size=(0.8, 0.6),
    table_height=0.75,
    table_points=8000,
    object_points=600,
    noise_std=0.002,
    seed=42
):
    """
    Generate a synthetic tabletop point cloud with random objects on it.
    Simulates what you'd get from a single RGB-D camera above the table.

    Returns
    -------
    dict with keys:
        'full_cloud'  : np.ndarray (N, 3) — full scene points
        'table_cloud' : np.ndarray (M, 3) — table plane points only
        'objects'     : list of dicts, each with:
                          'points'   : np.ndarray (K, 3)
                          'label'    : str  e.g. 'box', 'sphere', 'cylinder'
                          'center'   : np.ndarray (3,)
        'table_height': float
        'table_normal': np.ndarray (3,)  — always [0,0,1] in synthetic
    """
    rng = np.random.default_rng(seed)

    # --- Table plane ---
    tx = rng.uniform(-table_size[0]/2, table_size[0]/2, table_points)
    ty = rng.uniform(-table_size[1]/2, table_size[1]/2, table_points)
    tz = np.full(table_points, table_height) + rng.normal(0, noise_std/2, table_points)
    table_cloud = np.stack([tx, ty, tz], axis=1)

    # --- Objects ---
    shape_types = ['box', 'sphere', 'cylinder']
    object_data = []
    all_centers = []

    for i in range(n_objects):
        # pick a position that doesn't overlap with existing objects
        for _ in range(50):
            cx = rng.uniform(-table_size[0]/2 + 0.1, table_size[0]/2 - 0.1)
            cy = rng.uniform(-table_size[1]/2 + 0.1, table_size[1]/2 - 0.1)
            if all(np.linalg.norm(np.array([cx, cy]) - np.array(c[:2])) > 0.18
                   for c in all_centers):
                break
        shape = shape_types[i % len(shape_types)]
        if shape == 'box':
            sx = rng.uniform(0.04, 0.10)
            sy = rng.uniform(0.04, 0.10)
            sz = rng.uniform(0.06, 0.14)
            center = [cx, cy, table_height + sz/2]
            pts = _sample_box(center, [sx, sy, sz], object_points, noise_std)
        elif shape == 'sphere':
            r = rng.uniform(0.03, 0.07)
            center = [cx, cy, table_height + r]
            pts = _sample_sphere(center, r, object_points, noise_std)
        else:
            r = rng.uniform(0.025, 0.055)
            h = rng.uniform(0.07, 0.15)
            center = [cx, cy, table_height + h/2]
            pts = _sample_cylinder(center, r, h, object_points, noise_std)

        # simulate single-view: only keep points visible from above
        pts = pts[pts[:, 2] >= table_height - 0.005]

        all_centers.append(center)
        object_data.append({
            'points': pts,
            'label': shape,
            'center': np.array(center),
        })

    full_cloud = np.vstack([table_cloud] + [o['points'] for o in object_data])

    return {
        'full_cloud': full_cloud,
        'table_cloud': table_cloud,
        'objects': object_data,
        'table_height': table_height,
        'table_normal': np.array([0.0, 0.0, 1.0]),
    }