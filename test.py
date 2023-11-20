import torch
import torchvision
import numpy as np
from networks import *
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from unet import UNet_
from unet_pose import UNet_Pose
from scipy.spatial.transform import Rotation as R
import cv2
import torchvision.transforms as transforms
import copy
# invTrans = transforms.Compose([ 
#                                 transforms.Normalize(mean = [ 0., 0., 0. ],
#                                                      std = [ 1/0.229, 1/0.224, 1/0.225 ]),
#                                 transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
#                                                      std = [ 1., 1., 1. ]),
#                                ])

invTrans = transforms.Compose([ 
                                transforms.Normalize(mean = [ 0. ],
                                                     std = [ 1/0.5]),
                                transforms.Normalize(mean = [ -0.5 ],
                                                     std = [ 1.]),
                               ])
device_id = 0
n_steps = 1000
image_channels = 1
trans_step = 500
net = UNet_(
            image_channels= image_channels,
            n_channels= 64,
            ch_mults=[1, 2, 2, 4],
            is_attn=[False, False, False, True],
        )
net_Pose = UNet_Pose(
            image_channels= image_channels,
            n_channels= 64,
            ch_mults=[1, 2, 2, 4],
            is_attn=[False, False, False, True],
        )

net.to(f"cuda:{device_id}")
net_Pose.to(f"cuda:{device_id}")



state_dict_ = torch.load("snapshots/diffusion_DDP_headpose.pt", map_location=torch.device(f"cuda:{device_id}"))
net.load_state_dict(state_dict_['state'], strict=True)
state_dict_ = torch.load("snapshots/diffusion_DDP_pose.pt", map_location=torch.device(f"cuda:{device_id}"))
net_Pose.load_state_dict(state_dict_['state'], strict=True)
DDPM_Net = DenoiseDiffusion(eps_model = net, n_steps=n_steps, device=device_id)
DDPM_Net_Pose = DenoiseDiffusion_Pose(eps_model = net_Pose, n_steps=n_steps, device=device_id)

with torch.no_grad():
    
    x = torch.randn([1, image_channels, 256, 256]).to(f"cuda:{device_id}")
    y = torch.from_numpy(R.from_euler('yzx', [-45,0,-30], degrees=True).as_matrix().reshape((1,9))).float().to(f"cuda:{device_id}")
    for t_ in tqdm(list(range(0, n_steps))):
    # for t_ in tqdm(range(0, n_steps)):

        t = n_steps - t_ - 1
        # x = DDPM_Net.p_sample(x, x.new_full((1,), t, dtype=torch.long))

        x = DDPM_Net_Pose.p_sample(x, x.new_full((1,), t, dtype=torch.long),y)

        # if t_ == trans_step:
        #     x_pose = copy.deepcopy(x)
        # if t_ > trans_step:    
        #     x_pose = DDPM_Net_Pose.p_sample(x_pose, x_pose.new_full((1,), t, dtype=torch.long), y)

        if t_ % 100 == 99:
            x_ = invTrans(x.squeeze(0)).permute(1,2,0).cpu().numpy()*255
            cv2.imwrite(f"outputs/{t_}.png", x_)
   
            # if t_ > trans_step:
            #     x_pose_ = invTrans(x_pose.squeeze(0)).permute(1,2,0).cpu().numpy()*255
            #     cv2.imwrite(f"outputs/{t_}_pose.png", x_pose_)

