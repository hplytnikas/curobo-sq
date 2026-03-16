import open3d as o3d
import numpy as np
from superdec_planner.rrt_superdec import PathPlanner
from superdec_planner.superdec import Scene
from superdec_planner.vis_utils import create_camera, create_interactive_vis, vis_sq

def from_z_to_y_up(pc): # for scannet only
    angle = -np.pi / 2  # -90 degrees in radians
    rotation_matrix = np.array([[1, 0, 0],
                                [0, np.cos(angle), -np.sin(angle)],
                                [0, np.sin(angle), np.cos(angle)]])
    pc.rotate(rotation_matrix, center=(0, 0, 0))
    return pc

def main():
    # params
    bound = {"low_x": -10, "high_x": 10, "low_y": -1.5, "high_y": -1.0, "low_z": -10, "high_z": 10}
    superdec_scene = Scene("examples/room0.npz")
    collision_radius = 0.2
    waypoints = [
        {"pos": np.array([1.14, 2, -1.1]), "quat": np.array([0, 0, 0, 1])},  # start
        {"pos": np.array([4.36, 0.4, -1.1]), "quat": np.array([0, 0, 0, 1])}  # goal
    ]
    for waypoint in waypoints:
        waypoint["pos"] = from_z_to_y_up(o3d.geometry.PointCloud(o3d.utility.Vector3dVector([waypoint["pos"]]))).points[0]
        waypoint["pos"] = np.array(waypoint["pos"])

    # setup planner
    planner = PathPlanner()
    planner.update_collision_radius(collision_radius) 
    planner.update_sp(bound, superdec_scene)

    # plan path
    full_solution = []
    for i in range(len(waypoints) - 1):
        start, goal = waypoints[i], waypoints[i + 1]
        planner.update_start_goal(start, goal)
        try:
            planner.solve(time_limit=2.0, method="rrtstar")
            solution = planner.get_solution()
            full_solution.extend(solution)
        except:
            print(f"No solution found for segment {i}")

    print("avg validity check time: ", planner.get_avg_valid_time())

    # visualize
    H_1, W_1 = 960, 1280
    focal_1 = 700
    camera_intrinsic_1 = create_camera(H_1, W_1, focal_1)
    vis = create_interactive_vis(H_1, W_1, camera_intrinsic_1, show_back_face=True, light_on=False, z_near=0.02, z_far=50.0)

    # add a coodinate frame
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    vis.add_geometry(coord_frame)

    # Visualize superquadrics directly in Open3D
    sq_geometries = vis_sq(superdec_scene.superquadrics, resolution=20)
    for geom in sq_geometries:
        vis.add_geometry(geom)

    for waypoint in waypoints:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.12)
        sphere.compute_vertex_normals()
        sphere.translate(waypoint["pos"])
        sphere.paint_uniform_color([0, 0, 0])
        vis.add_geometry(sphere)

    # Visualize the solution path
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector([sol["pos"] for sol in full_solution])
    line_set.lines = o3d.utility.Vector2iVector([[i, i+1] for i in range(len(full_solution)-1)])

    for sol in full_solution:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.08)
        sphere.compute_vertex_normals()
        sphere.translate(sol["pos"])
        sphere.paint_uniform_color([0.5, 0.5, 0.5])
        vis.add_geometry(sphere)

    vis.add_geometry(line_set)
    vis.run()

if __name__ == "__main__":
    main()
