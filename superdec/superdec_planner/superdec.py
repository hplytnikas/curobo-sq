import numpy as np
from plyfile import (PlyData, PlyElement)
import random
import colorsys

def generate_ncolors(num):
    def get_n_hls_colors(num):
        hls_colors = []
        i = 0
        step = 360.0 / num
        while i < 360:
            h = i
            s = 90 + random.random() * 10
            l = 50 + random.random() * 10
            _hlsc = [h / 360.0, l / 100.0, s / 100.0]
            hls_colors.append(_hlsc)
            i += step
        return hls_colors
    rgb_colors = np.zeros((0,3), dtype=np.uint8)
    if num < 1:
        return rgb_colors
    hls_colors = get_n_hls_colors(num)
    for hlsc in hls_colors:
        _r, _g, _b = colorsys.hls_to_rgb(hlsc[0], hlsc[1], hlsc[2])
        r, g, b = [int(x * 255.0) for x in (_r, _g, _b)]
        rgb_colors = np.concatenate((rgb_colors,np.array([r,g,b])[np.newaxis,:]))
    return rgb_colors

class Superquadrics:
    def __init__(self, path_to_sq_parameters):
        print(f"Loading superquadrics from {path_to_sq_parameters}")
        parameters = np.load(path_to_sq_parameters)
        exists = parameters['exist']            # (N, K, 1)
        scales = parameters['scale']            # (N, K, ...)
        shapes = parameters['exponents']        # (N, K, ...)
        rots   = parameters['rotation']         # (N, K, ...)
        trans  = parameters['translation']      # (N, K, ...)

        N, K, _ = exists.shape
        total = N * K
        mask = (exists[..., 0] >= 0.5).reshape(total)  # bool mask

        def flatten(x):
            return x.reshape(total, -1) if x.ndim > 2 else x.reshape(total, 1)

        self.scales      = flatten(scales)[mask]
        self.shapes      = flatten(shapes)[mask]
        self.rotations   = flatten(rots)[mask].reshape(-1, 3, 3)
        self.translations = flatten(trans)[mask]

        self.num_primitives = self.scales.shape[0]
        self.colors = generate_ncolors(self.num_primitives)

    def move_to_sq_frame(self, points):
        pc_inver = points[None,...] - self.translations[:,None,:] #out_dict['trans'].unsqueeze(2).repeat(1,1,num_points,1)   

        pc_inver = np.einsum('abc,acd->abd', self.rotations.transpose(0,2,1), pc_inver.transpose(0,2,1)).transpose(0,2,1) #B * N * num_points * 3
        return pc_inver
    
    def get_radial_distance_and_closest_points(self, points):
        def get_directions_to_centers(indices):
            vec = self.translations[indices] - points
            norm = np.linalg.norm(vec, axis=1)[:,None]
            return vec/norm

        pc_inver = self.move_to_sq_frame(points)

        r_norm = np.sqrt(np.sum(pc_inver ** 2, -1))
        e = (
            np.pow(np.pow((pc_inver[...,0] / self.scales[...,None,0]) ** 2, (1 / self.shapes[...,None,1])) +
            np.pow((pc_inver[...,1] / self.scales[...,None,1]) ** 2, 1 / self.shapes[...,None,1]), self.shapes[...,None,1] / self.shapes[...,None,0]) +
            np.pow((pc_inver[...,2] / self.scales[...,None,2]) ** 2, (1 / self.shapes[...,None,0]))) ** (-self.shapes[...,None,0] / 2) - 1
        
        rad_res = r_norm * np.abs(e)
        rad_res = np.min(rad_res, axis=0)
        vec = get_directions_to_centers(np.argmin(rad_res, axis=0))
        # go of step rad_dist in direction vec
        closest_points = points + vec * rad_res[:,None]
        return rad_res, closest_points
    
    def get_vertices(self, N=10):
        def f(o, m): # angle and epsilon (shape par)
            return np.sign(np.sin(o)) * np.abs(np.sin(o))**m[...,None]
        def g(o, m):
            return np.sign(np.cos(o)) * np.abs(np.cos(o))**m[...,None]
        
        u = np.linspace(-np.pi, np.pi, N, endpoint=True)
        v = np.linspace(-np.pi/2.0, np.pi/2.0, N, endpoint=True)
        u = np.tile(u, N)
        v = (np.repeat(v, N))
        u = u[::-1]

        x = self.scales[...,0,None] *  g(u, self.shapes[...,0]) * g(v, self.shapes[...,1])
        y = self.scales[...,1,None] * g(u, self.shapes[...,0]) * f(v, self.shapes[...,1])
        z = self.scales[...,2,None] * f(u, self.shapes[...,0])
            
        v = np.stack([x, y, z], axis=-1)

        v = np.einsum('...jk,...ik->...ij', self.rotations, v) + self.translations[:,None,:] # BACKWARD
        return v 
    
    
    def save_ply(self, output_path, color = (253, 205, 80), resolution=10):
        vertices = self.get_vertices(resolution)
        tmp_vertex = np.zeros(vertices.shape[0]*vertices.shape[1], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),('red', 'u1'), ('green', 'u1'),('blue', 'u1')])
        for v1 in range(self.num_primitives):
            for v2 in range(vertices.shape[1]): # number of sampled points per primitive
                tmp_vertex[v1 * vertices.shape[1] + v2] = (vertices[v1, v2,0], vertices[v1, v2,1], vertices[v1, v2,2], self.colors[v1,0], self.colors[v1,1], self.colors[v1,2])
        
        
        triangles = []
        triangles_colors = []
        for k in range(self.num_primitives):
            os = k * ((resolution)**2)
            for i in range(resolution-1):
                for j in range(resolution-1):
                    triangles.append([os + i*resolution+j, os+ i*resolution+j+1, os +(i+1)*resolution+j])
                    triangles.append([os + (i+1)*resolution+j, os+i*resolution+j+1,os+ (i+1)*resolution+(j+1)])
                    triangles_colors.append(self.colors[k])
            triangles.append([os +(resolution-1)*resolution+(resolution-1), os +(resolution-1)*resolution, os +(resolution-1)])
            triangles.append([os +(resolution-1), os +(resolution-1)*resolution, os +0])
            triangles_colors.append(self.colors[k])

        #mesh = trimesh.Trimesh(vertices, triangles)

        tmp_triangles = np.zeros(len(triangles), dtype=[('vertex_indices', 'i4', (3,)),('red', 'u1'), ('green', 'u1'),('blue', 'u1')])
        for i in range(len(tmp_triangles)):
            tmp_triangles[i] = ([triangles[i][0], triangles[i][1], triangles[i][2]], triangles_colors[i//2][0],triangles_colors[i//2][1],triangles_colors[i//2][2])
        

        ply_out = PlyData([PlyElement.describe(tmp_vertex, 'vertex', comments=['vertices']),
                        PlyElement.describe(tmp_triangles, 'face')],text=True)
        
        ply_out.write(output_path)
        #mesh.export(output_path)
        

class Scene:
    def __init__(self, path_to_sq_parameters):
        self.superquadrics = Superquadrics(path_to_sq_parameters)

    def get_distances_and_closest_points(self, point):
        return self.superquadrics.get_radial_distance_and_closest_points(point)
    
    def save_superquadrics_vis(self, path):
        self.superquadrics.save_ply(path)


    
  