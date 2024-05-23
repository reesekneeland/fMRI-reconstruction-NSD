#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# # Code to convert this notebook to .py if you want to run it via command line or with Slurm
# from subprocess import call
# command = "jupyter nbconvert Reconstructions.ipynb --to python"
# call(command,shell=True)


# In[2]:


import os
os.environ['CUDA_VISIBLE_DEVICES'] = "3"
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm
from datetime import datetime
import webdataset as wds
import PIL
import argparse
from random import randrange
import time


import utils
from models import Clipper, OpenClipper, BrainNetwork, BrainDiffusionPrior, BrainDiffusionPriorOld, Voxel2StableDiffusionModel, VersatileDiffusionPriorNetwork

if utils.is_interactive():
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')

seed=randrange(100000)
utils.seed_everything(seed=seed)


# # Configurations

# In[2]:


# if running this interactively, can specify jupyter_args here for argparser to use
if utils.is_interactive():
    # Example use
    jupyter_args = "--data_path=/fsx/proj-medarc/fmri/natural-scenes-dataset \
                    --subj=1 \
                    --model_name=prior_257_final_subj01_bimixco_softclip_byol"
    
    jupyter_args = jupyter_args.split()
    print(jupyter_args)


# In[3]:


parser = argparse.ArgumentParser(description="Model Training Configuration")
parser.add_argument(
    "--model_name", type=str, default="testing",
    help="name of trained model",
)
parser.add_argument(
    "--autoencoder_name", type=str, default="None",
    help="name of trained autoencoder model",
)
parser.add_argument(
    "--data_path", type=str, default="/fsx/proj-medarc/fmri/natural-scenes-dataset",
    help="Path to where NSD data is stored (see README)",
)
parser.add_argument(
    "--subj",type=int, default=1, choices=[1,2,5,7],
)
parser.add_argument(
    "--img2img_strength",type=float, default=.85,
    help="How much img2img (1=no img2img; 0=outputting the low-level image itself)",
)
parser.add_argument(
    "--recons_per_sample", type=int, default=1,
    help="How many recons to output, to then automatically pick the best one (MindEye uses 16)",
)
parser.add_argument(
    "--vd_cache_dir", type=str, default='/fsx/proj-medarc/fmri/cache/models--shi-labs--versatile-diffusion/snapshots/2926f8e11ea526b562cd592b099fcf9c2985d0b7',
    help="Where is cached Versatile Diffusion model; if not cached will download to this path",
)

parser.add_argument(
    "--stimtype", type=str, default="all",
    help="stimulus type of reconstruction, either simple, complex, or concepts",
)
parser.add_argument(
    "--epoch", type=int, default=0,
    help="epoch of attention reconstruction, either 0 (cue period) or 1 (barrage period)",
)
parser.add_argument(
    "--average", type=str, default="True",
    help="Whether to average betas for a stimulus, if disabled, will reconstruct each beta individually",
)


if utils.is_interactive():
    args = parser.parse_args(jupyter_args)
else:
    args = parser.parse_args()

# create global variables without the args prefix
for attribute_name in vars(args).keys():
    globals()[attribute_name] = getattr(args, attribute_name)
    
if args.autoencoder_name=="None":
    autoencoder_name = None

subj = args.subj
data_path = args.data_path
vd_cache_dir = args.vd_cache_dir
recons_per_sample = args.recons_per_sample
stimtype = args.stimtype
epoch = args.epoch
average = args.average == "True"
print(average)
# In[4]:

device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
local_rank = 0
print("device:",device)
if subj == 1:
    num_voxels = 15724
elif subj == 2:
    num_voxels = 14278
elif subj == 3:
    num_voxels = 15226
elif subj == 4:
    num_voxels = 13153
elif subj == 5:
    num_voxels = 13039
elif subj == 6:
    num_voxels = 17907
elif subj == 7:
    num_voxels = 12682
elif subj == 8:
    num_voxels = 14386
print("subj",subj,"num_voxels",num_voxels)

batch_size = val_batch_size = 1


from models import Voxel2StableDiffusionModel

outdir = f'../train_logs/{autoencoder_name}'
ckpt_path = os.path.join(outdir, f'epoch120.pth')

if os.path.exists(ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint['model_state_dict']

    voxel2sd = Voxel2StableDiffusionModel(in_dim=num_voxels)

    voxel2sd.load_state_dict(state_dict,strict=False)
    voxel2sd.eval()
    voxel2sd.to(device)
    print("Loaded low-level model!")
else:
    print("No valid path for low-level model specified; not using img2img!") 
    img2img_strength = 1


print('Creating versatile diffusion reconstruction pipeline...')
from diffusers import VersatileDiffusionDualGuidedPipeline, UniPCMultistepScheduler
from diffusers.models import DualTransformer2DModel
try:
    vd_pipe =  VersatileDiffusionDualGuidedPipeline.from_pretrained(vd_cache_dir).to(device)
except:
    print("Downloading Versatile Diffusion to", vd_cache_dir)
    vd_pipe =  VersatileDiffusionDualGuidedPipeline.from_pretrained(
            "shi-labs/versatile-diffusion",
            cache_dir = vd_cache_dir).to(device).to(torch.float16)
vd_pipe.image_unet.eval()
vd_pipe.vae.eval()
vd_pipe.image_unet.requires_grad_(False)
vd_pipe.vae.requires_grad_(False)

vd_pipe.scheduler = UniPCMultistepScheduler.from_pretrained(vd_cache_dir + "/models--shi-labs--versatile-diffusion/snapshots/2926f8e11ea526b562cd592b099fcf9c2985d0b7", subfolder="scheduler")
num_inference_steps = 20

# Set weighting of Dual-Guidance 
text_image_ratio = .0 # .5 means equally weight text and image, 0 means use only image
for name, module in vd_pipe.image_unet.named_modules():
    if isinstance(module, DualTransformer2DModel):
        module.mix_ratio = text_image_ratio
        for i, type in enumerate(("text", "image")):
            if type == "text":
                module.condition_lengths[i] = 77
                module.transformer_index_for_condition[i] = 1  # use the second (text) transformer
            else:
                module.condition_lengths[i] = 257
                module.transformer_index_for_condition[i] = 0  # use the first (image) transformer

unet = vd_pipe.image_unet
vae = vd_pipe.vae
noise_scheduler = vd_pipe.scheduler


img_variations = False

out_dim = 257 * 768
clip_extractor = Clipper("ViT-L/14", hidden_state=True, norm_embs=True, device=device)
voxel2clip_kwargs = dict(in_dim=num_voxels,out_dim=out_dim)
voxel2clip = BrainNetwork(**voxel2clip_kwargs)
voxel2clip.requires_grad_(False)
voxel2clip.eval()

out_dim = 768
depth = 6
dim_head = 64
heads = 12 # heads * dim_head = 12 * 64 = 768
timesteps = 100 #100

prior_network = VersatileDiffusionPriorNetwork(
        dim=out_dim,
        depth=depth,
        dim_head=dim_head,
        heads=heads,
        causal=False,
        learned_query_mode="pos_emb"
    )

diffusion_prior = BrainDiffusionPrior(
    net=prior_network,
    image_embed_dim=out_dim,
    condition_on_text_encodings=False,
    timesteps=timesteps,
    cond_drop_prob=0.2,
    image_embed_scale=None,
    voxel2clip=voxel2clip,
)

outdir = f'/home/naxos2-raid25/kneel027/home/kneel027/fMRI-reconstruction-NSD/train_logs/{model_name}'
ckpt_path = os.path.join(outdir, f'last.pth')

print("ckpt_path",ckpt_path)
checkpoint = torch.load(ckpt_path, map_location=device)
state_dict = checkpoint['model_state_dict']
print("EPOCH: ",checkpoint['epoch'])
diffusion_prior.load_state_dict(state_dict,strict=False)
diffusion_prior.eval().to(device)
diffusion_priors = [diffusion_prior]
pass


print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

retrieve = False
plotting = False
saving = True
verbose = False
imsize = 512

if img_variations:
    guidance_scale = 7.5
else:
    guidance_scale = 3.5
    
# ind_include = np.arange(num_val)
all_brain_recons = None
    
only_lowlevel = False
if img2img_strength == 1:
    img2img = False
elif img2img_strength == 0:
    img2img = True
    only_lowlevel = True
else:
    img2img = True

# experiment_type = "mi_imagery"
# experiment_type = "mi_vision"
# experiment_type = "nsd_vision"
images = torch.load(f"{data_path}/nsddata_stimuli/stimuli/imagery_stimuli_18.pt").requires_grad_(False).to("cpu")
print("images shape", images.shape)
# "sti_A/batch_i.npy", "sti_A/batch_i2v.npy", "sti_A/batch_v.npy", "sti_A/batch_v2i.npy","sti_B/batch_i.npy",  "sti_B/batch_i2v.npy", "sti_B/batch_v.npy", "sti_B/batch_v2i.npy",
             
for file in ["sti_full/batch_i2v.npy", "sti_full/batch_v2i.npy"]:
    voxels = torch.from_numpy(np.load(f"{data_path}/batch_to_recon/{file}")).unsqueeze(1)
    print(f"voxel size {voxels.shape}")
    if "sti_A" in file:
        images = images[:6]
    elif "sti_B" in file:
        images = images[6:12]
    all_images = None
    imsize=512
    all_brain_recons = None
    for val_i, (voxel, img) in enumerate(tqdm(zip(voxels.to(device), images.to(device)))):
        # if val_i<np.min(ind_include):
        #     continue
        # voxel = torch.mean(voxel,axis=1).to(device)
        # voxel = voxel[:,0].to(device)
        
        with torch.no_grad():
            if img2img:
                ae_preds = voxel2sd(voxel.float())
                blurry_recons = vd_pipe.vae.decode(ae_preds.to(device).half()/0.18215).sample / 2 + 0.5

                if val_i==0:
                    plt.imshow(utils.torch_to_Image(blurry_recons))
                    plt.show()
            else:
                blurry_recons = None

            if only_lowlevel:
                brain_recons = blurry_recons
            else:
                # os.makedirs("../reconstructions/{}/subject{}/{}/".format(mode, subj, val_i), exist_ok=True)
                # os.makedirs("../seeds/{}/subject{}/{}/".format(mode, subj, val_i), exist_ok=True)
                grid, brain_recons, laion_best_picks, recon_img, extracted_clips = utils.reconstruction(
                    img, voxel,
                    clip_extractor, unet, vae, noise_scheduler,
                    voxel2clip_cls = None, #diffusion_prior_cls.voxel2clip,
                    diffusion_priors = diffusion_priors,
                    text_token = None,
                    img_lowlevel = blurry_recons,
                    num_inference_steps = num_inference_steps,
                    n_samples_save = batch_size,
                    recons_per_sample = recons_per_sample,
                    guidance_scale = guidance_scale,
                    img2img_strength = img2img_strength, # 0=fully rely on img_lowlevel, 1=not doing img2img
                    timesteps_prior = 100,
                    seed = seed,
                    retrieve = retrieve,
                    plotting = plotting,
                    img_variations = img_variations,
                    verbose = verbose,
                )

                if plotting:
                    plt.show()
                brain_recons = brain_recons[:,laion_best_picks.astype(np.int8)]

            print(f"mid loop brain recons {all_brain_recons}")
            if all_brain_recons is None:
                all_brain_recons = brain_recons
                # all_images = img
            else:
                all_brain_recons = torch.vstack((all_brain_recons,brain_recons))
                # all_images = torch.vstack((all_images,img))

    print(all_brain_recons.shape)
    all_brain_recons = all_brain_recons.view(-1,3,imsize,imsize)
    print(all_brain_recons.shape)
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    all_images = images
    if saving:
        torch.save(all_images,f'all_images.pt')
        torch.save(all_brain_recons,f"{file.replace('/','_')}_{model_name}_recons_img2img{img2img_strength}_{recons_per_sample}samples.pt")
    print(f"recon_path: {file[:-4].replace('/','_')}_{model_name}_recons_img2img{img2img_strength}_{recons_per_sample}samples")

    # create full grid of recon comparisons
    from PIL import Image
    all_recons = all_brain_recons
    print(all_images.shape, all_recons.shape)
    imsize = 150
    if all_images.shape[-1] != imsize:
        all_images = transforms.Resize((imsize,imsize))(transforms.CenterCrop(all_images.shape[2])(all_images)).float()
    if all_recons.shape[-1] != imsize:
        all_recons = transforms.Resize((imsize,imsize))(transforms.CenterCrop(all_images.shape[2])(all_recons)).float()

    num_images = all_recons.shape[0]
    num_rows = (2 * num_images + 11) // 12
    print(all_images.shape, all_recons.shape)
    # Interleave tensors
    merged = torch.stack([val for pair in zip(all_images, all_recons) for val in pair], dim=0)

    # Calculate grid size
    grid = torch.zeros((num_rows * 12, 3, all_recons.shape[-1], all_recons.shape[-1]))

    # Populate the grid
    grid[:2*num_images] = merged
    grid_images = [transforms.functional.to_pil_image(grid[i]) for i in range(num_rows * 12)]

    # Create the grid image
    grid_image = Image.new('RGB', (all_recons.shape[-1]*12, all_recons.shape[-1] * num_rows))  # 10 images wide

    # Paste images into the grid
    for i, img in enumerate(grid_images):
        grid_image.paste(img, (all_recons.shape[-1] * (i % 12), all_recons.shape[-1] * (i // 12)))
    os.makedirs("../figs/", exist_ok=True)
    grid_image.save(f"../figs/{file[:-4].replace('/','_')}_{model_name}_{len(all_recons)}recons.png")
