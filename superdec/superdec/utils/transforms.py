import torch
import torch.nn.functional as F

def transform_to_primitive_frame(pc_or_normals, trans, rotate):
    B, N, _ = pc_or_normals.shape
    P = trans.shape[1]

    centered = pc_or_normals.unsqueeze(1).repeat(1, P, 1, 1) - trans.unsqueeze(2).repeat(1,1,N,1)
    centered = centered.permute(0,1,3,2)
    rotate_T = rotate.permute(0, 1, 3, 2)
    transformed = torch.einsum('abcd,abde->abce', rotate_T, centered.float()).permute(0,1,3,2)
    return transformed

def quat2mat(quat):
    """Normalize the quaternion and convert it to rotation matrix"""
    B = quat.shape[0]
    N = quat.shape[1]
    quat = F.normalize(quat, dim=2)
    quat = quat.contiguous().view(-1,4)
    w, x, y, z = quat[:,0], quat[:,1], quat[:,2], quat[:,3]
    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z
    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                        2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                        2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).view(B, N, 3, 3)
    rotMat = rotMat.view(B,N,3,3)
    return rotMat

def mat2quat(rotMat):
    """Convert rotation matrix to quaternion"""
    B, N = rotMat.shape[0], rotMat.shape[1]
    rotMat = rotMat.view(-1, 3, 3)
    
    # Extract diagonal and off-diagonal elements
    m00, m01, m02 = rotMat[:, 0, 0], rotMat[:, 0, 1], rotMat[:, 0, 2]
    m10, m11, m12 = rotMat[:, 1, 0], rotMat[:, 1, 1], rotMat[:, 1, 2]
    m20, m21, m22 = rotMat[:, 2, 0], rotMat[:, 2, 1], rotMat[:, 2, 2]
    
    # Compute trace
    trace = m00 + m11 + m22
    
    # Initialize quaternion tensor
    quat = torch.zeros(rotMat.shape[0], 4, device=rotMat.device, dtype=rotMat.dtype)
    
    # Case 1: trace > 0
    mask1 = trace > 0
    s1 = torch.sqrt(trace[mask1] + 1.0) * 2  # s = 4 * w
    quat[mask1, 0] = 0.25 * s1  # w
    quat[mask1, 1] = (m21[mask1] - m12[mask1]) / s1  # x
    quat[mask1, 2] = (m02[mask1] - m20[mask1]) / s1  # y
    quat[mask1, 3] = (m10[mask1] - m01[mask1]) / s1  # z
    
    # Case 2: m00 > m11 and m00 > m22
    mask2 = (~mask1) & (m00 > m11) & (m00 > m22)
    s2 = torch.sqrt(1.0 + m00[mask2] - m11[mask2] - m22[mask2]) * 2  # s = 4 * x
    quat[mask2, 0] = (m21[mask2] - m12[mask2]) / s2  # w
    quat[mask2, 1] = 0.25 * s2  # x
    quat[mask2, 2] = (m01[mask2] + m10[mask2]) / s2  # y
    quat[mask2, 3] = (m02[mask2] + m20[mask2]) / s2  # z
    
    # Case 3: m11 > m22
    mask3 = (~mask1) & (~mask2) & (m11 > m22)
    s3 = torch.sqrt(1.0 + m11[mask3] - m00[mask3] - m22[mask3]) * 2  # s = 4 * y
    quat[mask3, 0] = (m02[mask3] - m20[mask3]) / s3  # w
    quat[mask3, 1] = (m01[mask3] + m10[mask3]) / s3  # x
    quat[mask3, 2] = 0.25 * s3  # y
    quat[mask3, 3] = (m12[mask3] + m21[mask3]) / s3  # z
    
    # Case 4: else (m22 is largest)
    mask4 = (~mask1) & (~mask2) & (~mask3)
    s4 = torch.sqrt(1.0 + m22[mask4] - m00[mask4] - m11[mask4]) * 2  # s = 4 * z
    quat[mask4, 0] = (m10[mask4] - m01[mask4]) / s4  # w
    quat[mask4, 1] = (m02[mask4] + m20[mask4]) / s4  # x
    quat[mask4, 2] = (m12[mask4] + m21[mask4]) / s4  # y
    quat[mask4, 3] = 0.25 * s4  # z
    
    # Reshape back to original dimensions
    quat = quat.view(B, N, 4)
    
    # Normalize quaternion
    quat = F.normalize(quat, dim=2)
    
    return quat