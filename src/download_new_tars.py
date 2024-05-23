#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# # Code to convert this notebook to .py if you want to run it via command line or with Slurm
# from subprocess import call
# command = "jupyter nbconvert Train_MindEye.ipynb --to python"
# call(command,shell=True)


# # Import packages & functions

# In[2]:


import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm

# tf32 data type is faster than standard float32
torch.backends.cuda.matmul.allow_tf32 = True

# custom models and functions #
import utils
from models import Clipper, BrainNetwork, BrainDiffusionPrior, BrainDiffusionPriorOld, VersatileDiffusionPriorNetwork

# Multi-GPU config #
from accelerate import Accelerator
accelerator = Accelerator(split_batches=False,mixed_precision='fp16')  
print("PID of this process =",os.getpid())
print = accelerator.print # only print if local_rank=0
device = accelerator.device
print("device:",device)
num_devices = torch.cuda.device_count()
if num_devices==0: num_devices = 1
num_workers = num_devices
print(accelerator.state)
local_rank = accelerator.state.local_process_index
world_size = accelerator.state.num_processes
distributed = not accelerator.state.distributed_type == 'NO'
print("distributed =",distributed, "num_devices =", num_devices, "local rank =", local_rank, "world size =", world_size)


# # Configurations

# In[3]:


# if running this interactively, can specify jupyter_args here for argparser to use
if utils.is_interactive():
    # Example use
    jupyter_args = "--data_path=/fsx/proj-medarc/fmri/natural-scenes-dataset \
                    --model_name=test \
                    --subj=1 --hidden --clip_variant=ViT-L/14 --batch_size=32 --n_samples_save=0 \
                    --max_lr=3e-4 --mixup_pct=.33 --num_epochs=240 --ckpt_interval=5 --use_image_aug"
    
    jupyter_args = jupyter_args.split()
    print(jupyter_args)
    
    from IPython.display import clear_output # function to clear print outputs in cell
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2 # this allows you to change functions in models.py or utils.py and have this notebook automatically update with your revisions')


# In[4]:


parser = argparse.ArgumentParser(description="Model Training Configuration")
parser.add_argument(
    "--model_name", type=str, default="testing",
    help="name of model, used for ckpt saving and wandb logging (if enabled)",
)
parser.add_argument(
    "--data_path", type=str, default="/fsx/proj-medarc/fmri/natural-scenes-dataset",
    help="Path to where NSD data is stored / where to download it to",
)
parser.add_argument(
    "--subj",type=int, default=1, choices=[1,2,5,7],
)
parser.add_argument(
    "--batch_size", type=int, default=32,
    help="Batch size can be increased by 10x if only training v2c and not diffusion prior",
)
parser.add_argument(
    "--hidden",action=argparse.BooleanOptionalAction,default=True,
    help="if True, CLIP embeddings will come from last hidden layer (e.g., 257x768 - Versatile Diffusion), rather than final layer",
)
parser.add_argument(
    "--clip_variant",type=str,default="ViT-L/14",choices=["RN50", "ViT-L/14", "ViT-B/32", "RN50x64"],
    help='OpenAI clip variant',
)
parser.add_argument(
    "--wandb_log",action=argparse.BooleanOptionalAction,default=False,
    help="whether to log to wandb",
)
parser.add_argument(
    "--resume_from_ckpt",action=argparse.BooleanOptionalAction,default=False,
    help="if not using wandb and want to resume from a ckpt",
)
parser.add_argument(
    "--wandb_project",type=str,default="stability",
    help="wandb project name",
)
parser.add_argument(
    "--mixup_pct",type=float,default=.33,
    help="proportion of way through training when to switch from BiMixCo to SoftCLIP",
)
parser.add_argument(
    "--norm_embs",action=argparse.BooleanOptionalAction,default=True,
    help="Do l2-norming of CLIP embeddings",
)
parser.add_argument(
    "--use_image_aug",action=argparse.BooleanOptionalAction,default=True,
    help="whether to use image augmentation",
)
parser.add_argument(
    "--num_epochs",type=int,default=240,
    help="number of epochs of training",
)
parser.add_argument(
    "--prior",action=argparse.BooleanOptionalAction,default=True,
    help="if False, will only use CLIP loss and ignore diffusion prior",
)
parser.add_argument(
    "--v2c",action=argparse.BooleanOptionalAction,default=True,
    help="if False, will only use diffusion prior loss",
)
parser.add_argument(
    "--plot_umap",action=argparse.BooleanOptionalAction,default=False,
    help="Plot UMAP plots alongside reconstructions",
)
parser.add_argument(
    "--lr_scheduler_type",type=str,default='cycle',choices=['cycle','linear'],
)
parser.add_argument(
    "--ckpt_saving",action=argparse.BooleanOptionalAction,default=True,
)
parser.add_argument(
    "--ckpt_interval",type=int,default=5,
    help="save backup ckpt and reconstruct every x epochs",
)
parser.add_argument(
    "--save_at_end",action=argparse.BooleanOptionalAction,default=False,
    help="if True, saves best.ckpt at end of training. if False and ckpt_saving==True, will save best.ckpt whenever epoch shows best validation score",
)
parser.add_argument(
    "--seed",type=int,default=42,
)
parser.add_argument(
    "--max_lr",type=float,default=3e-4,
)
parser.add_argument(
    "--n_samples_save",type=int,default=0,choices=[0,1],
    help="Number of reconstructions for monitoring progress, 0 will speed up training",
)
parser.add_argument(
    "--use_projector",action=argparse.BooleanOptionalAction,default=True,
    help="Additional MLP after the main MLP so model can separately learn a way to minimize NCE from prior loss (BYOL)",
)
parser.add_argument(
    "--vd_cache_dir", type=str, default='/fsx/proj-medarc/fmri/cache/models--shi-labs--versatile-diffusion/snapshots/2926f8e11ea526b562cd592b099fcf9c2985d0b7',
    help="Where is cached Versatile Diffusion model; if not cached will download to this path",
)

if utils.is_interactive():
    args = parser.parse_args(jupyter_args)
else:
    args = parser.parse_args()

# create global variables without the args prefix
for attribute_name in vars(args).keys():
    globals()[attribute_name] = getattr(args, attribute_name)
    
# need non-deterministic CuDNN for conv3D to work
utils.seed_everything(seed, cudnn_deterministic=False)

# change learning rate based on number of devices
max_lr *= accelerator.num_processes
    
# change batch size based on number of devices if using multi-gpu
# batch_size *= accelerator.num_processes

# change num_epochs based on number of devices if using multi-gpu
num_epochs *= accelerator.num_processes


# In[5]:


outdir = os.path.abspath(f'../train_logs/{model_name}')
if not os.path.exists(outdir):
    os.makedirs(outdir,exist_ok=True)
if use_image_aug:
    import kornia
    from kornia.augmentation.container import AugmentationSequential
    img_augment = AugmentationSequential(
        kornia.augmentation.RandomResizedCrop((224,224), (0.6,1), p=0.3),
        kornia.augmentation.Resize((224, 224)),
        kornia.augmentation.RandomHorizontalFlip(p=0.5),
        kornia.augmentation.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.3),
        kornia.augmentation.RandomGrayscale(p=0.3),
        data_keys=["input"],
    )


# # Prep models and data loaders

# In[6]:


print('Pulling NSD webdataset data...')

train_url = "{" + f"{data_path}/webdataset_avg_split/train/train_subj0{subj}_" + "{0..17}.tar," + f"{data_path}/webdataset_avg_split/val/val_subj0{subj}_0.tar" + "}"
val_url = f"{data_path}/webdataset_avg_split/test/test_subj0{subj}_" + "{0..1}.tar"
print(train_url,"\n",val_url)
meta_url = f"{data_path}/webdataset_avg_split/metadata_subj0{subj}.json"
num_train = 8559 + 300
num_val = 982

print('Prepping train and validation dataloaders...')
train_dl, val_dl, num_train, num_val = utils.get_dataloaders(
    batch_size,'images',
    num_devices=num_devices,
    num_workers=num_workers,
    train_url=train_url,
    val_url=val_url,
    meta_url=meta_url,
    num_train=num_train,
    num_val=num_val,
    val_batch_size=300,
    cache_dir=data_path, #"/tmp/wds-cache",
    seed=seed,
    voxels_key='nsdgeneral.npy',
    to_tuple=["voxels", "images", "coco"],
    local_rank=local_rank,
    world_size=world_size,
)
