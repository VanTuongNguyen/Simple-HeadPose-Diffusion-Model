import torch
import torchvision
from dataset import *
import torchvision.transforms as transforms
import numpy as np
from networks import *
import time , sys, os
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from distributed_utls import *
import albumentations as A
from albumentations.pytorch import ToTensorV2
from unet import UNet_
from unet_pose import UNet_Pose

device_id = int(os.environ["LOCAL_RANK"])
global_rank = int(os.environ["RANK"])
dist.init_process_group("nccl")
world_size = dist.get_world_size()
print(f"Start running DDP on CUDA: {device_id}/{torch.cuda.device_count()} ON {os.uname()[1]} RANK: {global_rank}/{world_size}.", flush=True)

# create model and move it to GPU with id rank



train_transform = A.Compose([
    # A.Sequential([
    #     A.HorizontalFlip(p=0.5),
    #     A.RandomBrightnessContrast(p=0.2),
    #     A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5), 
    #     A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.2,p=0.5),
    #     A.MotionBlur(blur_limit=33, p=0.3),
    #     A.GaussNoise(var_limit=(0, 255), p=0.3),
    #     A.CoarseDropout(max_holes=6, max_height=72, max_width=72, p=0.5),
    #     A.CoarseDropout(max_holes=6, max_height=72, max_width=72, fill_value=255, p=0.5),
    #     A.ChannelDropout(p=0.3),
    #     A.GridDistortion(num_steps=5, distort_limit=0.1, p=0.3),
    #     A.OpticalDistortion(distort_limit=0.2, shift_limit=0.05, p=0.3)
    # ], p=0.7),
    A.Sequential([
        # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        A.Normalize(mean=(0.5), std=(0.5)),
        ToTensorV2()
    ], p=1)
])


# train_dataset = CustomDataset("../emotion/Manually_Annotated_Images/refined_training.csv", test = False, transform = train_transform)
train_dataset = HeadposeDataset_3x3_glob("../face_landmark_qat/data/cmore_train", transform = train_transform)

batch_size = 8

train_sampler = DistributedSampler(dataset=train_dataset,
                                    shuffle=True,
                                    num_replicas=world_size,
                                    rank=global_rank) 
train_loader = DataLoader(train_dataset, 
                          batch_size=batch_size,
                          sampler=train_sampler,
                          num_workers=16,
                          pin_memory=True,
                          )



net = UNet_Pose(
            image_channels= 1,
            n_channels= 64,
            ch_mults=[1, 2, 2, 4],
            is_attn=[False, False, False, True],
        )

# net = UNet()

learning_rate = 1e-5

optimizer = torch.optim.AdamW(net.parameters(),
                       lr=learning_rate)


# **7. define learning rate decay scheduler:-**

# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 100, eta_min=1e-12, verbose=True)



# **8. Place Model in GPU:-**


net.to(f"cuda:{device_id}")

# state_dict_ = torch.load("snapshots/best.pt", map_location=torch.device(f"cuda:{device_id}"))
# net.load_state_dict(state_dict_['state'], strict=True)
net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[device_id], find_unused_parameters=True)

DDPM_Net = DenoiseDiffusion_Pose(eps_model = net, n_steps=1000, device=device_id)


def train_net(n_epochs):


    for epoch in range(n_epochs):  # loop over the dataset multiple times
        # prepare the model for training
        net.train()
        train_running_loss=[]

        with tqdm(iter(train_loader), total=len(train_loader), file=sys.stdout) as tepoch:
            tepoch.set_description(f"Epoch {epoch}")
            for batch_i, data in enumerate(tepoch):
                # get the input images and their corresponding poses
                images, pose = data
                # put data inside gpu
                images = images.float().to(f"cuda:{device_id}")
                pose = pose.float().to(f"cuda:{device_id}")
              
                # calculate the softmax loss between predicted poses and ground truth poses
                loss = DDPM_Net.loss(images, pose)
                # loss = focal_loss(torch.softmax(pred_class,1), torch.argmax(gt_class,1))

                # zero the parameter (weight) gradients
                optimizer.zero_grad()
                
                # backward pass to calculate the weight gradients
                loss.backward()

                # update the weights
                optimizer.step()
                
                # get loss
                loss_scalar = loss.item()

                #update loss logs
                train_running_loss.append(loss_scalar)
            
                
                if batch_i % 10 == 9:    # print every 100 batches
                    tepoch.set_postfix(train_Loss=np.mean(train_running_loss))
                    tepoch.update(1)
        
        # update lr schedular
        print(f"Mean epoch {epoch}: {np.mean(train_running_loss)}", flush=True)
        scheduler.step()         
        torch.save({
                    "state" : net.module.state_dict()
                    }, 'snapshots/diffusion_DDP_pose.pt')
            
                                           

    print('Finished Training', flush=True)




try:
    # train your network
    n_epochs = 1000 
    train_net(n_epochs)

except KeyboardInterrupt:
    print('Stopping Training...', flush=True)
