import open3d as o3d
import numpy as np

## CREATE ##
def create_camera(H,W,focal):

    assert isinstance(H, int)
    fx, fy = focal, focal
    cx, cy = W/2.0-0.5, H/2.0-0.5
    return o3d.camera.PinholeCameraIntrinsic(width=W, height=H, 
                                            fx=fx, fy=fy, cx=cx, cy=cy)


def set_viewpoint_ctr(vis, z_near=0.02, z_far=15.0):

    ctr = vis.get_view_control()
    ctr.set_constant_z_far(z_far)
    ctr.set_constant_z_near(z_near)
    return ctr


def create_interactive_vis(H, W, camera_intrinsic,
                           camera_extrinsic=np.eye(4),
                           show_back_face=False,light_on=False,
                           z_near=0.02, z_far=15.0):

    vis = o3d.visualization.VisualizerWithKeyCallback()
    assert H == camera_intrinsic.height
    assert W == camera_intrinsic.width
    vis.create_window(width=W, height=H)
    vis.get_render_option().mesh_show_back_face = show_back_face
    vis.get_render_option().light_on = light_on
    ctr = set_viewpoint_ctr(vis, z_near, z_far)
    param = ctr.convert_to_pinhole_camera_parameters()
    param.intrinsic = camera_intrinsic
    param.extrinsic = camera_extrinsic
    success = ctr.convert_from_pinhole_camera_parameters(param,allow_arbitrary=True)
    assert success
    return vis

def return_cam_position_callback(vis):
    cam_pose = vis.get_view_control().convert_to_pinhole_camera_parameters()
    cam_extrinsic = cam_pose.extrinsic
    cam_in_world = np.linalg.inv(cam_extrinsic)[:3, 3]
    print("Camera position in world frame: ", cam_in_world)
    

def vis_sq(superquadrics, resolution=10):
    vertices = superquadrics.get_vertices(resolution)
    geometries = []

    for idx in range(superquadrics.num_primitives):
        mesh = o3d.geometry.TriangleMesh()
        vertex_points = vertices[idx]

        # Assign vertices
        mesh.vertices = o3d.utility.Vector3dVector(vertex_points)

        # Create triangles for the mesh (simple grid connectivity for superquadric surface)
        triangles = []
        for i in range(resolution - 1):
            for j in range(resolution - 1):
                triangles.append([i * resolution + j, (i + 1) * resolution + j, i * resolution + (j + 1)])
                triangles.append([(i + 1) * resolution + j, (i + 1) * resolution + (j + 1), i * resolution + (j + 1)])

        if len(triangles) > 0:
            mesh.triangles = o3d.utility.Vector3iVector(triangles)
            mesh.paint_uniform_color(superquadrics.colors[idx] / 255.0)  # Normalize color values
            mesh.compute_vertex_normals()
            geometries.append(mesh)
        else:
            print(f"[Warning] Superquadric {idx} generated empty triangles.")

    return geometries