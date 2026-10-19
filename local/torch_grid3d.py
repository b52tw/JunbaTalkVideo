"""Lower exact 3D trilinear grid sampling to eight ONNX GatherElements operations."""
def sample3d(image,grid,mode='bilinear',padding_mode='zeros',align_corners=None):
    import torch
    if mode!='bilinear' or padding_mode not in ('zeros','border'):raise ValueError('unsupported 3D sampler mode')
    b,c,d,h,w=image.shape;out_shape=grid.shape[1:4];x,y,z=grid[...,0],grid[...,1],grid[...,2]
    if align_corners:x=(x+1)*(w-1)/2;y=(y+1)*(h-1)/2;z=(z+1)*(d-1)/2
    else:x=((x+1)*w-1)/2;y=((y+1)*h-1)/2;z=((z+1)*d-1)/2
    if padding_mode=='border':x=x.clamp(0,w-1);y=y.clamp(0,h-1);z=z.clamp(0,d-1)
    x0=torch.floor(x);y0=torch.floor(y);z0=torch.floor(z);flat=image.reshape(b,c,-1);result=0
    for dz in (0,1):
        for dy in (0,1):
            for dx in (0,1):
                xi=x0+dx;yi=y0+dy;zi=z0+dz
                weight=(1-torch.abs(x-xi))*(1-torch.abs(y-yi))*(1-torch.abs(z-zi))
                if padding_mode=='zeros':weight=weight*((xi>=0)&(xi<w)&(yi>=0)&(yi<h)&(zi>=0)&(zi<d)).to(image.dtype)
                index=zi.clamp(0,d-1).to(torch.int64)*(h*w)+yi.clamp(0,h-1).to(torch.int64)*w+xi.clamp(0,w-1).to(torch.int64)
                result=result+torch.gather(flat,2,index.reshape(b,1,-1).expand(b,c,-1))*weight.reshape(b,1,-1)
    return result.reshape(b,c,*out_shape)
def install_export_sampler():
    import torch.nn.functional as f
    original=f.grid_sample
    def wrapped(input,grid,mode='bilinear',padding_mode='zeros',align_corners=None):
        if input.ndim==5:return sample3d(input,grid,mode,padding_mode,align_corners)
        return original(input,grid,mode,padding_mode,align_corners)
    f.grid_sample=wrapped
    return original
