import torch
import torch.nn as nn
import time
from omegaconf import DictConfig
from typing import Any, Callable, Dict, Tuple
import functorch
from torch.func import jacfwd, vmap
import torch.nn.functional as F
from superdec.utils.transforms import quat2mat, mat2quat
from superdec.utils.safe_operations import safe_pow, safe_mul


def get_uniformly_sampled_points(scale, shape, R, t, N=100):
    def points_from_angles(scale, shape, eta, omega):
        def f(o, m): # angle and epsilon (shape par)
            return safe_mul(torch.sign(torch.sin(o)), safe_pow(torch.abs(torch.sin(o)),m[...,None]))
        def g(o, m):
            return safe_mul(torch.sign(torch.cos(o)), torch.abs(torch.cos(o))**m[...,None])
        x = safe_mul(safe_mul(scale[...,0,None], g(eta, shape[...,0])), g(omega, shape[...,1]))
        y = safe_mul(safe_mul(scale[...,1,None], g(eta, shape[...,0])), f(omega, shape[...,1]))
        z = safe_mul(scale[...,2,None], f(eta, shape[...,0]))
        
        v =  torch.stack((x, y, z)).transpose(0,1)
        return v 
    end_u = torch.pi  - 2 * torch.pi/ N 
    u = torch.linspace(-torch.pi, end_u, N).to(scale.device).to(scale.dtype)
    end_v = torch.pi/2.0 - torch.pi / N 
    v = torch.linspace(-torch.pi/2.0, end_v, N).to(scale.device).to(scale.dtype)
    # u = torch.linspace(-torch.pi, torch.pi, N).to(scale.device).to(scale.dtype)
    # v = torch.linspace(-torch.pi/2.0, torch.pi/2.0, N).to(scale.device).to(scale.dtype)
    u = torch.tile(u, (N,))
    v = v.repeat_interleave(N)

    v = points_from_angles(scale, shape, u, v)
    #v = torch.matmul(v, R.transpose(-1,-2)) + t[None,:]
    v = torch.einsum('...jk,...ik->...ij', R, v) + t[None,:] # BACKWARD
    return v
        
        
    def forward(self):
        return self.softmax(self.weights/self.temperature) 

def R_from_q_multi(q):
    #norm = torch.sqrt(q[...,0]*q[...,0] + q[...,1]*q[...,1] + q[...,2]*q[...,2] + q[...,3]*q[...,3])
    #q_ = q / norm[...,None]
    q = F.normalize(q, dim=-1)
    #R = torch.zeros((q.shape[0], 3, 3), device=q.device)

    R11 = 1 - 2 * (q[...,2]**2 + q[...,3]**2)
    R12 = 2 * (q[...,1]*q[...,2] - q[...,3]*q[...,0])
    R13 = 2 * (q[...,1]*q[...,3] + q[...,2]*q[...,0])
    R21 = 2 * (q[...,1]*q[...,2] + q[...,3]*q[...,0])
    R22 = 1 - 2 * (q[...,1]**2 + q[...,3]**2)
    R23 = 2 * (q[...,2]*q[...,3] - q[...,1]*q[...,0])
    R31 = 2 * (q[...,1]*q[...,3] - q[...,2]*q[...,0])
    R32 = 2 * (q[...,2]*q[...,3] + q[...,1]*q[...,0])
    R33 = 1 - 2 * (q[...,1]**2 + q[...,2]**2)

    row1 = torch.stack((R11, R12, R13), dim=-1)
    row2 = torch.stack((R21, R22, R23), dim=-1)
    row3 = torch.stack((R31, R32, R33), dim=-1)
    return torch.stack((row1, row2, row3), dim=-2)

class LRootDecay(torch.nn.Module):
        def __init__(self):
            super(LRootDecay, self).__init__()
        
        def forward(self, residuals):
            epsilon = 0.001
            residuals = torch.abs(residuals)+epsilon
            return torch.mean(torch.where(torch.abs(residuals) < 1.0, 0.5 * torch.pow(residuals, 2), torch.pow(torch.abs(residuals), 0.5) - 0.75))




class LMOptimizer(nn.Module):
    """Levenberg-Marquardt optimizer for superquadrics fitting."""
    
    def __init__(self):
        """Initialize the LM optimizer."""
        super().__init__()
        self.lamb = 0.1
        self.num_steps = 5
        self.atol = 1e-8
        self.rtol = 1e-8
        self.early_stop = True

    @staticmethod
    def get_pred_from_params(params):
        scale_par = params[:,:3]
        shape_par = params[:,3:5]
        q = params[:,5:9]
        t = params[:,9:12]
        R = R_from_q_multi(q).to(params.dtype)
        shape = 0.1 + 1.8 * torch.sigmoid(shape_par)
        scale = torch.exp(scale_par)
        return scale, shape, R, t, q
    @staticmethod
    def compute_residuals_points_unweighted(params, weights, points):

        scale_par = params[:3]
        shape_par = params[3:5]
        q = params[5:9]
        t = params[9:12]
        R = R_from_q_multi(q).to(params.dtype)
        shape = 0.1 + 1.8 * torch.sigmoid(shape_par)
        scale = torch.exp(scale_par)
        
        out = points - t #out_dict['trans'].unsqueeze(2).repeat(1,1,num_points,1)   
        out = torch.sign(out) * torch.clamp(abs(out), 1e-4, 1e6)

        # out = out @ R  #B * N * num_points * 3
        out = torch.einsum('...kj,...ik->...ij', R, out) # this is FORWARD, so I assume R.T to be the rotation of my SQ
        # same as doing  torch.einsum('cd,de->ce', R.permute(-1,-2), out.permute(-1,-2)).permute(-1,-2)
        # out = torch.sign(out) * torch.max(abs(out), out.new_tensor(1e-6) )
        # out = torch.sign(out) * torch.clamp(abs(out), 1e-4, 1e6)
        scale = scale.clamp(1e-6, 1e2)
        r_norm = torch.sqrt(torch.sum(out ** 2, -1))
        r_norm = torch.clamp(r_norm, 1e-4, 1e6)
        
        # e = safe_pow(
        #     safe_pow(
        #         safe_pow((out[...,0] / scale[None,0]) ** 2, 1 / shape[None,1]) +
        #         safe_pow((out[...,1] / scale[None,1]) ** 2, 1 / shape[None,1]), (shape[None,1] / shape[None,0]))  +
        #         safe_pow((out[...,2] / scale[None,2]) ** 2, 1 / shape[None,0]), (-shape[None,0] / 2)) - 1
        # e = (safe_pow(
        #         safe_pow((out[...,0] / scale[None,0]) ** 2, 1 / shape[None,1]) +
        #         safe_pow((out[...,1] / scale[None,1]) ** 2, 1 / shape[None,1]), (shape[None,1] / shape[None,0]))  +
        #         safe_pow((out[...,2] / scale[None,2]) ** 2, 1 / shape[None,0])) ** (-shape[None,0] / 2) - 1
        e = safe_pow(
            safe_pow(safe_pow((out[...,0] / scale[...,None,0]) ** 2, (1 / shape[...,None,1])) +
            safe_pow((out[...,1] / scale[...,None,1]) ** 2, 1 / shape[...,None,1]), shape[...,None,1] / shape[...,None,0]) +
            safe_pow((out[...,2] / scale[...,None,2]) ** 2, (1 / shape[...,None,0])), -shape[...,None,0] / 2) - 1
        rad_res = r_norm * torch.abs(e)
    
        return rad_res   

    @staticmethod
    def compute_residuals_points(params, weights, points):

        scale_par = params[:3]
        shape_par = params[3:5]
        q = params[5:9]
        t = params[9:12]
        R = R_from_q_multi(q).to(params.dtype)
        shape = 0.1 + 1.8 * torch.sigmoid(shape_par)
        scale = torch.exp(scale_par)
        
        out = points - t #out_dict['trans'].unsqueeze(2).repeat(1,1,num_points,1)   
        out = torch.sign(out) * torch.clamp(abs(out), 1e-4, 1e6)

        # out = out @ R  #B * N * num_points * 3
        out = torch.einsum('...kj,...ik->...ij', R, out) # this is FORWARD, so I assume R.T to be the rotation of my SQ
        # same as doing  torch.einsum('cd,de->ce', R.permute(-1,-2), out.permute(-1,-2)).permute(-1,-2)
        # out = torch.sign(out) * torch.max(abs(out), out.new_tensor(1e-6) )
        # out = torch.sign(out) * torch.clamp(abs(out), 1e-4, 1e6)
        scale = scale.clamp(1e-6, 1e2)
        r_norm = torch.sqrt(torch.sum(out ** 2, -1))
        r_norm = torch.clamp(r_norm, 1e-4, 1e6)
        e = safe_pow(
            safe_pow(safe_pow((out[...,0] / scale[...,None,0]) ** 2, (1 / shape[...,None,1])) +
            safe_pow((out[...,1] / scale[...,None,1]) ** 2, 1 / shape[...,None,1]), shape[...,None,1] / shape[...,None,0]) +
            safe_pow((out[...,2] / scale[...,None,2]) ** 2, (1 / shape[...,None,0])), -shape[...,None,0] / 2) - 1
        rad_res = r_norm * torch.abs(e)/len(points) #+ scale[0]*scale[1]*scale[2]
        weighted_rad_res = (rad_res * weights)    # t#     (sum(weights)+0.01)  #/sum(weights) #/len(points)    # the division stays for NORMALIZATION

        sampled_points =  get_uniformly_sampled_points(scale, shape, torch.eye(3).cuda().to(R.dtype), torch.zeros(3).cuda().to(t.dtype), N=4)
        diff = sampled_points[None,...] - out[:, None,:]
        diff = (weights < 0.5)[...,None,None] * diff
        diff += 0.0001
        distances = torch.sqrt(torch.sum(diff ** 2, -1)).min(-2).values / diff.shape[1] # the division stays for NORMALIZATION

        res = torch.hstack((weighted_rad_res, distances))
        return res 
    
    @staticmethod
    def compute_residuals_points_primitives(params, points):
        scale_par = params[:3]
        shape_par = params[3:5]
        q = params[5:9]
        t = params[9:12]
        R = R_from_q_multi(q).to(params.dtype)
        shape = 0.1 + 1.8 * torch.sigmoid(shape_par)
        scale = torch.exp(scale_par)
        
        out = points - t #out_dict['trans'].unsqueeze(2).repeat(1,1,num_points,1)   
        out = out @ R  #B * N * num_points * 3
        scale = scale.clamp(1e-6, 1e2)
        r_norm = torch.sqrt(torch.sum(out ** 2, -1))
        rad_res = r_norm * torch.abs((
            (((out[...,0] / scale[None,0]) ** 2) ** (1 / shape[None,1]) +
            ((out[...,1] / scale[None,1]) ** 2) ** (1 / shape[None,1])) ** (shape[None,1] / shape[None,0]) +
            ((out[...,2] / scale[None,2]) ** 2) ** (1 / shape[None,0])) ** (-shape[None,0] / 2) - 1
        )
        
        return rad_res 


    
    def update_deltas(self, Js, Hs, lamb, eps=1e-4):
        device = Js.device

        lamb = lamb.to(Js.device)
        diag =  torch.diagonal(Hs, dim1=-2, dim2=-1) # get diagonal
        diag = diag * lamb[...,None]  # (B, 3) # damp the diagonal by lambda
        # diag = torch.ones_like(diag).to(device) * lamb

        H =  Hs + diag.clamp(min=eps).diag_embed()
        # Now we have to solve the system H delta = J
        # and we try to decompose H = U^T U
        H_, J_ = H.cpu(), Js.cpu()
        # J_ = torch.clamp(J_, -1e6, 1e6)
        # H_ = torch.clamp(H_, -1e6, 1e6)
        if J_.isnan().any() or H_.isnan().any():
            print("NaN detected. Stopping.")
            delta = torch.zeros_like(Js)
        else:
            try:
                U = torch.linalg.cholesky(H_)
            except:
                # If it cannot be decomposed using the Cholesky dec. we use LU dec.
                print("Cholesky decomposition failed, fallback to LU.")
                try:
                    delta = torch.linalg.solve(H_, J_[..., None])[..., 0]
                except:
                    print("LU decomposition failed. Stopping.")
                    delta = torch.zeros_like(Js)
            else:
                # If it can be decomposed we solve the system
                delta = torch.cholesky_solve(J_[...,None], U)[...,0]
                #print("Cholesky successful.")

        return delta.to(device)
    
    @staticmethod   
    def get_JH_param_points(f, params, weights, points): 
        points = torch.sign(points) * torch.clamp(abs(points), 1e-4, 1e6)
        Js = vmap(vmap(jacfwd(f)))(params, weights.transpose(-1,-2), points.to(params.dtype))
        if(Js.isnan().any()):
            
            
            dtype = torch.float64
            if Js.shape[0] >= 2: # ifwe want operate with float64, we split the batch in half (if we can)
                B_half = Js.shape[0]//2
                Js_1 = vmap(vmap(jacfwd(f)))(params[:B_half,...].to(dtype), weights.transpose(-1,-2)[:B_half,...].to(dtype), points[:B_half,...].to(dtype)).to(torch.float32).cpu()
                Js_2 = vmap(vmap(jacfwd(f)))(params[B_half:,...].to(dtype), weights.transpose(-1,-2)[B_half:,...].to(dtype), points[B_half:,...].to(dtype)).to(torch.float32)
                Js = torch.vstack((Js_1.cuda(), Js_2))
                if Js.isnan().any():
                    print(f"NaN detected, also with float64. Turned {Js.isnan().sum()} to zeros.")
                    torch.nan_to_num(Js, nan=0.0)
            else:
                Js = vmap(vmap(jacfwd(f)))(params.to(dtype), weights.transpose(-1,-2).to(dtype), points.to(dtype)).to(torch.float32)
                if Js.isnan().any():
                    print(f"NaN detected, also with float64. Turned {Js.isnan().sum()} to zeros.")
                    torch.nan_to_num(Js, nan=0.0)
            
            
        Hs = Js.transpose(-1,-2) @ Js

        return Js, Hs
    @torch.no_grad()
    def optimize(self, outdict, points):
        """Run the LM optimization."""
        P = outdict["assign_matrix"].shape[2] 
        B = outdict["assign_matrix"].shape[0]

        outdict["q"] = mat2quat(outdict["rotate"].clone())
        if 'shape' in outdict:
            shape_param = torch.special.logit((outdict["shape"].clone()-0.1)/1.8)
        else:
            shape_param = 0.5 * torch.ones((B,P,2)).to(outdict["scale"].device)
        params = torch.cat((torch.log(outdict["scale"].clone().detach()), shape_param, outdict["q"].clone().detach(), outdict["trans"].clone().detach()), dim =-1).detach() #[B, P, 12]
        
        lamb = self.lamb * torch.ones((B, P)).to(params.device)
        
        # Step 1: initialize update masks
        update = [True, True, True, True] #whether to update scale, shape, q, t
        update_mask = torch.zeros(params.shape[-1], dtype=torch.bool, device = params.device)
        update_mask[:3] = update[0]
        update_mask[3:5] = update[1]
        update_mask[5:9] = update[2]
        update_mask[9:12] = update[3]

        weights =  outdict["assign_matrix"].clone().detach()

        recompute_jacobian = torch.ones(B, dtype=torch.bool, device=params.device)

        dtype = torch.float32
        params = params.to(dtype)
        new_params = params.clone().detach()
        points = points.to(dtype)
        weights = weights.to(dtype)
        
        # deltas_cum = torch.zeros_like(params)
        expanded_points = points.expand(params.shape[1], -1, -1, -1).transpose(0, 1)
        lamb = lamb.to(dtype)
        Js, Hs = LMOptimizer.get_JH_param_points(LMOptimizer.compute_residuals_points, params, weights, expanded_points)
        lamb = lamb * Hs.diagonal(dim1=-2,dim2=-1).mean(-1).abs() # here I am doing a double mean, but then I should do only one
        torch.cuda.empty_cache()
        for i in range (self.num_steps):
            torch.cuda.empty_cache()
            with torch.no_grad():
                residuals = vmap(vmap(LMOptimizer.compute_residuals_points))(params, weights.transpose(-1,-2), expanded_points)  
                cost = (residuals ** 2).sum(-1)
 
            
                
            if residuals.isnan().any():
                print("NaN detected. Stopping.")
    

                      
            # 1. Compute Cost
            # 2. Compute J, H and deltas
            if i>0:
                Js, Hs = LMOptimizer.get_JH_param_points(LMOptimizer.compute_residuals_points, params, weights, expanded_points)

            deltas = self.update_deltas(torch.matmul(Js.transpose(-1,-2), residuals[...,None]).squeeze(-1) , Hs, lamb)
            

            new_params[...,update_mask] = params[...,update_mask] - deltas 
            with torch.no_grad():
                new_residuals = vmap(vmap(LMOptimizer.compute_residuals_points))(new_params, weights.transpose(-1,-2), expanded_points)            
                new_cost = (new_residuals ** 2).sum(-1)
                mask = new_cost < cost
                lamb = torch.where(mask, lamb / 10, lamb * 10)
            #print(f"cost: {cost.mean(-1)[0].data}, new_cost {new_cost.mean(-1)[0].data}")

            
            params = torch.where(mask[..., None], new_params, params).clone()
            # deltas_cum= torch.where(mask[..., None], deltas_cum - deltas, deltas_cum)
            
            recompute_jacobian[mask.max(1).values] = True



        scale_par = params[...,:3]
        shape_par = params[...,3:5]
        q = params[...,5:9]
        t = params[...,9:]

        R = quat2mat(q)
        with torch.no_grad():
            mask = (weights > 0.5).transpose(-1, -2)
            selected = torch.where(mask, residuals[..., :weights.shape[1]], torch.tensor(0.0, device=residuals.device))
            outdict["exist_post"] = ((residuals[..., weights.shape[1]:].mean(-1) < 0.2).unsqueeze(-1) * (outdict['exist']>0.5)).float().cpu()
        outdict["exist"] = outdict["exist"]
        outdict["scale"] = torch.exp(scale_par).to(outdict["scale"].dtype)
        outdict["shape"] = 0.1 + 1.8 * torch.sigmoid(shape_par).to(outdict["scale"].dtype)
        outdict["rotate"] = R.to(outdict["scale"].dtype)
        outdict["trans"] = t.to(outdict["scale"].dtype)
        outdict["assign_matrix"] = weights.to(outdict["scale"].dtype)
        return outdict
    
    
    def forward(self, outdict: Dict[str, torch.Tensor], points) -> Dict[str, torch.Tensor]:
        """Run the LM optimization."""
        start = time.time()
        out = self.optimize(outdict, points)
        # print(f"Final shape: {out['shape'][0,0]}")
        # print(f"Optimization took {time.time() - start:.2f} seconds.")
        return out