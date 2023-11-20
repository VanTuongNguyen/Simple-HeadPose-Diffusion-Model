import torch
import torchvision
from dataset import CustomDataset
import torchvision.transforms as transforms
import numpy as np
from networks import DenoiseDiffusion
import time , sys, os
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from distributed_utls import *
import albumentations as A
from albumentations.pytorch import ToTensorV2
from unet import UNet

device_id = 0
# global_rank = int(os.environ["RANK"])
# dist.init_process_group("nccl")
# world_size = dist.get_world_size()
# print(f"Start running DDP on CUDA: {device_id}/{torch.cuda.device_count()} ON {os.uname()[1]} RANK: {global_rank}/{world_size}.", flush=True)

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
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ], p=1)
])


train_dataset = CustomDataset("../emotion/Manually_Annotated_Images/refined_training.csv", test = False, transform = train_transform)

batch_size = 16

# train_sampler = DistributedSampler(dataset=train_dataset,
#                                     shuffle=True,
#                                     num_replicas=world_size,
#                                     rank=global_rank) 
train_loader = DataLoader(train_dataset, 
                          batch_size=batch_size,
                        #   sampler=train_sampler,
                          num_workers=32,
                          pin_memory=True,
                          )



net = UNet(
            image_channels= 3,
            n_channels= 64,
            ch_mults=[1, 2, 2, 4],
            is_attn=[False, False, False, True],
        )

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
###########  DDP
# net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[device_id], find_unused_parameters=True)

DDPM_Net = DenoiseDiffusion(eps_model = net, n_steps=1000, device=device_id)







def train_net(n_epochs):


    for epoch in range(n_epochs):  # loop over the dataset multiple times
        # prepare the model for training
        net.train()

        with tqdm(iter(train_loader), total=len(train_loader), file=sys.stdout) as tepoch:
            tepoch.set_description(f"Epoch {epoch}")
            train_running_loss=[]
            for batch_i, data in enumerate(tepoch):
                # get the input images and their corresponding poses
                images = data
                # put data inside gpu
                images = images.float().to(f"cuda:{device_id}")

        
                # calculate the softmax loss between predicted poses and ground truth poses
                loss = DDPM_Net.loss(images)
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
                    train_running_loss=[]
        
        # update lr schedular
        scheduler.step()         
        torch.save({
                    "state" : net.state_dict()
                    }, 'snapshots/diffusion.pt')
            
                                           

    print('Finished Training', flush=True)




try:
    # train your network
    n_epochs = 1000 # start small, and increase when you've decided on your model structure and hyperparams
    train_net(n_epochs)

except KeyboardInterrupt:
    print('Stopping Training...', flush=True)
