import open3d as o3d
mesh = o3d.io.read_triangle_mesh(
    'src/project_3dv/perception/outputs/sq_scene005.ply')
mesh.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh],
    window_name='SQ Scene 005', width=1200, height=800)