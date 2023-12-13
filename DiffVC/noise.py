import numpy as np 
import torch

def generate_pink_noise(x,end_scale):
    xshape = x.shape
    lin = np.linspace(1,end_scale,xshape[1])
    tiled_lin = np.tile(lin,(xshape[2],1)).T
    batch_tiled_lin = np.tile(tiled_lin,(xshape[0],1,1))
    noise_scale = torch.from_numpy(batch_tiled_lin.astype(np.float32)).clone()
    noise_scale = noise_scale.to(x.device)
    z = torch.randn(xshape, dtype=x.dtype, device=x.device, requires_grad=False)
    pink_noise = torch.mul(noise_scale,z)
    return pink_noise