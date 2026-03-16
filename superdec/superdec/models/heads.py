import torch.nn as nn
import torch.nn.functional as F
import torch

class SuperDecHead(nn.Module):
    """Head for Superquadrics Prediction"""
    
    def __init__(self, emb_dims):
        super(SuperDecHead, self).__init__()
        self.emb_dims = emb_dims
        self.scale_head = nn.Linear(emb_dims, 3)
        self.shape_head = nn.Linear(emb_dims, 2)
        self.rot_head = nn.Linear(emb_dims, 4)
        self.t_head = nn.Linear(emb_dims, 3)
        self.exist_head = nn.Linear(emb_dims, 1)


    def forward(self, x):
        scale_pre_activation = self.scale_head(x)
        scale = self.scale_activation(scale_pre_activation)
        
        shape_before_activation = self.shape_head(x)
        shape = self.shape_activation(shape_before_activation)

        q = F.normalize(self.rot_head(x), dim=-1, p=2)
        rotation = self.quat2mat(q)

        translation = self.t_head(x)

        exist = self.exist_activation(self.exist_head(x))

        return {"scale": scale, "shape": shape, "rotate": rotation, "trans": translation, "exist": exist}
        
    @staticmethod  
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
    
    def scale_activation(self, x):
        return  torch.sigmoid(x) 
    
    @staticmethod 
    def shape_activation(x):
        return 0.1 + 1.8 * torch.sigmoid(x)
    
    @staticmethod 
    def exist_activation(x):
        return torch.sigmoid(x)