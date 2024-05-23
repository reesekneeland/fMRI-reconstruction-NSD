import numpy as np
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import PIL
from PIL import Image
import random
import os, sys
import matplotlib.pyplot as plt
import pandas as pd
import math
import webdataset as wds
import json
import requests
import io
from urllib.request import Request, urlopen
import socket
from clip_retrieval.clip_client import ClipClient
import time 
import braceexpand
from models import Clipper,OpenClipper
import pickle

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def is_interactive():
    import __main__ as main
    return not hasattr(main, '__file__')

def seed_everything(seed=0, cudnn_deterministic=True):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
    else:
        ## needs to be False to use conv3D
        print('Note: not using cudnn.deterministic')

def np_to_Image(x):
    if x.ndim==4:
        x=x[0]
    return PIL.Image.fromarray((x.transpose(1, 2, 0)*127.5+128).clip(0,255).astype('uint8'))

def torch_to_Image(x):
    if x.ndim==4:
        x=x[0]
    return transforms.ToPILImage()(x)

def Image_to_torch(x):
    try:
        x = (transforms.ToTensor()(x)[:3].unsqueeze(0)-.5)/.5
    except:
        x = (transforms.ToTensor()(x[0])[:3].unsqueeze(0)-.5)/.5
    return x

def torch_to_matplotlib(x,device=device):
    if torch.mean(x)>10:
        x = (x.permute(0, 2, 3, 1)).clamp(0, 255).to(torch.uint8)
    else:
        x = (x.permute(0, 2, 3, 1) * 255).clamp(0, 255).to(torch.uint8)
    if device=='cpu':
        return x[0]
    else:
        return x.cpu().numpy()[0]

def pairwise_cosine_similarity(A, B, dim=1, eps=1e-8):
    #https://stackoverflow.com/questions/67199317/pytorch-cosine-similarity-nxn-elements
    numerator = A @ B.T
    A_l2 = torch.mul(A, A).sum(axis=dim)
    B_l2 = torch.mul(B, B).sum(axis=dim)
    denominator = torch.max(torch.sqrt(torch.outer(A_l2, B_l2)), torch.tensor(eps))
    return torch.div(numerator, denominator)

def batchwise_pearson_correlation(Z, B):
    # Calculate means
    Z_mean = torch.mean(Z, dim=1, keepdim=True)
    B_mean = torch.mean(B, dim=1, keepdim=True)

    # Subtract means
    Z_centered = Z - Z_mean
    B_centered = B - B_mean

    # Calculate Pearson correlation coefficient
    numerator = Z_centered @ B_centered.T
    Z_centered_norm = torch.linalg.norm(Z_centered, dim=1, keepdim=True)
    B_centered_norm = torch.linalg.norm(B_centered, dim=1, keepdim=True)
    denominator = Z_centered_norm @ B_centered_norm.T

    pearson_correlation = (numerator / denominator)
    return pearson_correlation

def batchwise_cosine_similarity(Z,B):
    # https://www.h4pz.co/blog/2021/4/2/batch-cosine-similarity-in-pytorch-or-numpy-jax-cupy-etc
    B = B.T
    Z_norm = torch.linalg.norm(Z, dim=1, keepdim=True)  # Size (n, 1).
    B_norm = torch.linalg.norm(B, dim=0, keepdim=True)  # Size (1, b).
    cosine_similarity = ((Z @ B) / (Z_norm @ B_norm)).T
    return cosine_similarity

def topk(similarities,labels,k=5):
    if k > similarities.shape[0]:
        k = similarities.shape[0]
    topsum=0
    for i in range(k):
        topsum += torch.sum(torch.argsort(similarities,axis=1)[:,-(i+1)] == labels)/len(labels)
    return topsum

def soft_clip_loss(preds, targs, temp=0.125):
    clip_clip = (targs @ targs.T)/temp
    brain_clip = (preds @ targs.T)/temp
    
    loss1 = -(brain_clip.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
    loss2 = -(brain_clip.T.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
    
    loss = (loss1 + loss2)/2
    return loss

def gather_features(image_features, voxel_features):  
    all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
    if voxel_features is not None:
        all_voxel_features = torch.cat(torch.distributed.nn.all_gather(voxel_features), dim=0)
        return all_image_features, all_voxel_features
    return all_image_features

def soft_cont_loss(student_preds, teacher_preds, teacher_aug_preds, temp=0.125, distributed=True):
    if not distributed:
        teacher_teacher_aug = (teacher_preds @ teacher_aug_preds.T)/temp
        teacher_teacher_aug_t = (teacher_aug_preds @ teacher_preds.T)/temp
        student_teacher_aug = (student_preds @ teacher_aug_preds.T)/temp
        student_teacher_aug_t = (teacher_aug_preds @ student_preds.T)/temp
    else:
        all_student_preds, all_teacher_preds = gather_features(student_preds, teacher_preds)
        all_teacher_aug_preds = gather_features(teacher_aug_preds, None)

        teacher_teacher_aug = (teacher_preds @ all_teacher_aug_preds.T)/temp
        teacher_teacher_aug_t = (teacher_aug_preds @ all_teacher_preds.T)/temp
        student_teacher_aug = (student_preds @ all_teacher_aug_preds.T)/temp
        student_teacher_aug_t = (teacher_aug_preds @ all_student_preds.T)/temp
    
    loss1 = -(student_teacher_aug.log_softmax(-1) * teacher_teacher_aug.softmax(-1)).sum(-1).mean()
    loss2 = -(student_teacher_aug_t.log_softmax(-1) * teacher_teacher_aug_t.softmax(-1)).sum(-1).mean()
    
    loss = (loss1 + loss2)/2
    return loss

def mixco(voxels, beta=0.15, s_thresh=0.5):
    perm = torch.randperm(voxels.shape[0])
    voxels_shuffle = voxels[perm].to(voxels.device,dtype=voxels.dtype)
    betas = torch.distributions.Beta(beta, beta).sample([voxels.shape[0]]).to(voxels.device,dtype=voxels.dtype)
    select = (torch.rand(voxels.shape[0]) <= s_thresh).to(voxels.device)
    betas_shape = [-1] + [1]*(len(voxels.shape)-1)
    voxels[select] = voxels[select] * betas[select].reshape(*betas_shape) + \
        voxels_shuffle[select] * (1 - betas[select]).reshape(*betas_shape)
    betas[~select] = 1
    return voxels, perm, betas, select

def mixco_clip_target(clip_target, perm, select, betas):
    clip_target_shuffle = clip_target[perm]
    clip_target[select] = clip_target[select] * betas[select].reshape(-1, 1) + \
        clip_target_shuffle[select] * (1 - betas[select]).reshape(-1, 1)
    return clip_target

def mixco_nce(preds, targs, temp=0.1, perm=None, betas=None, select=None, distributed=False, 
              accelerator=None, local_rank=None, bidirectional=True):
    brain_clip = (preds @ targs.T)/temp
    
    if perm is not None and betas is not None and select is not None:
        probs = torch.diag(betas)
        probs[torch.arange(preds.shape[0]).to(preds.device), perm] = 1 - betas

        loss = -(brain_clip.log_softmax(-1) * probs).sum(-1).mean()
        if bidirectional:
            loss2 = -(brain_clip.T.log_softmax(-1) * probs.T).sum(-1).mean()
            loss = (loss + loss2)/2
        return loss
    else:
        loss =  F.cross_entropy(brain_clip, torch.arange(brain_clip.shape[0]).to(brain_clip.device))
        if bidirectional:
            loss2 = F.cross_entropy(brain_clip.T, torch.arange(brain_clip.shape[0]).to(brain_clip.device))
            loss = (loss + loss2)/2
        return loss

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('param counts:\n{:,} total\n{:,} trainable'.format(total, trainable))

def image_grid(imgs, rows, cols):
    w, h = imgs[0].size
    grid = PIL.Image.new('RGB', size=(cols*w, rows*h))
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

def get_huggingface_urls(commit='main',subj=1):
    base_url = "https://huggingface.co/datasets/pscotti/naturalscenesdataset/resolve/"
    train_url = base_url + commit + f"/webdataset_avg_new/train/train_subj0{subj}_" + "{0..17}.tar"
    val_url = base_url + commit + f"/webdataset_avg_new/val/val_subj0{subj}_0.tar"
    test_url = base_url + commit + f"/webdataset_avg_new/test/test_subj0{subj}_" + "{0..1}.tar"
    return train_url, val_url, test_url
    
def check_loss(loss):
    if loss.isnan().any():
        raise ValueError('NaN loss')
        
def _check_whether_images_are_identical(image1, image2):
    pil_image1 = transforms.ToPILImage()(image1)
    pil_image2 = transforms.ToPILImage()(image2)

    SIMILARITY_THRESHOLD = 90

    image_hash1 = phash(pil_image1, hash_size=16)
    image_hash2 = phash(pil_image2, hash_size=16)

    return (image_hash1 - image_hash2) < SIMILARITY_THRESHOLD

def get_dataloaders(
    batch_size,
    image_var='images',
    num_devices=None,
    num_workers=None,
    train_url=None,
    val_url=None,
    meta_url=None,
    num_train=None,
    num_val=None,
    cache_dir="/tmp/wds-cache",
    seed=0,
    voxels_key="nsdgeneral.npy",
    val_batch_size=None,
    to_tuple=["voxels", "images", "trial"],
    local_rank=0,
    world_size=1,
    subj=1,
):
    print("Getting dataloaders...")
    assert image_var == 'images'
    
    def my_split_by_node(urls):
        return urls
    
    train_url = list(braceexpand.braceexpand(train_url))
    val_url = list(braceexpand.braceexpand(val_url))
    print(train_url[0])
    if not os.path.exists(train_url[0]):
        # we will default to downloading from huggingface urls if data_path does not exist
        print("downloading NSD from huggingface...")
        os.makedirs(cache_dir,exist_ok=True)
        
        train_url, val_url, test_url = get_huggingface_urls("main",subj)
        train_url = list(braceexpand.braceexpand(train_url))
        val_url = list(braceexpand.braceexpand(val_url))
        test_url = list(braceexpand.braceexpand(test_url))
        
        from tqdm import tqdm
        for url in tqdm(train_url):
            destination = cache_dir + "/" + url.rsplit('/', 1)[-1]
            print(f"\nDownloading {url} to {destination}...")
            response = requests.get(url)
            response.raise_for_status()
            with open(destination, 'wb') as file:
                file.write(response.content)
                
        for url in tqdm(val_url):
            destination = cache_dir + "/" + url.rsplit('/', 1)[-1]
            print(f"\nDownloading {url} to {destination}...")
            response = requests.get(url)
            response.raise_for_status()
            with open(destination, 'wb') as file:
                file.write(response.content)
                
        for url in tqdm(test_url):
            destination = cache_dir + "/" + url.rsplit('/', 1)[-1]
            print(f"\nDownloading {url} to {destination}...")
            response = requests.get(url)
            response.raise_for_status()
            with open(destination, 'wb') as file:
                file.write(response.content)

    if num_devices is None:
        num_devices = torch.cuda.device_count()
    
    if num_workers is None:
        num_workers = num_devices
    
    if num_train is None:
        metadata = json.load(open(meta_url))
        num_train = metadata['totals']['train']
    if num_val is None:
        metadata = json.load(open(meta_url))
        num_val = metadata['totals']['val']

    if val_batch_size is None:
        val_batch_size = batch_size
        
    global_batch_size = batch_size * num_devices
    num_batches = math.floor(num_train / global_batch_size)
    num_worker_batches = math.floor(num_batches / num_workers)
    if num_worker_batches == 0: num_worker_batches = 1
    
    print("\nnum_train",num_train)
    print("global_batch_size",global_batch_size)
    print("batch_size",batch_size)
    print("num_workers",num_workers)
    print("num_batches",num_batches)
    print("num_worker_batches", num_worker_batches)
    
    # train_url = train_url[local_rank:world_size]
    train_data = wds.WebDataset(train_url, resampled=True, cache_dir=cache_dir, nodesplitter=my_split_by_node)\
        .shuffle(500, initial=500, rng=random.Random(42))\
        .decode("torch")\
        .rename(images="jpg;png", voxels=voxels_key, trial="trial.npy", coco="coco73k.npy", reps="num_uniques.npy")\
        .to_tuple(*to_tuple)\
        .batched(batch_size, partial=True)\
        .with_epoch(num_worker_batches)
    
    train_dl = torch.utils.data.DataLoader(train_data, batch_size=None, num_workers=1, shuffle=False)

    # Validation 
    # should be deterministic, no shuffling!    
    num_batches = math.floor(num_val / global_batch_size)
    num_worker_batches = math.floor(num_batches / num_workers)
    if num_worker_batches == 0: num_worker_batches = 1
    
    print("\nnum_val",num_val)
    print("val_num_batches",num_batches)
    print("val_batch_size",val_batch_size)
    
    val_data = wds.WebDataset(val_url, resampled=False, cache_dir=cache_dir, nodesplitter=my_split_by_node)\
        .decode("torch")\
        .rename(images="jpg;png", voxels=voxels_key, trial="trial.npy", coco="coco73k.npy", reps="num_uniques.npy")\
        .to_tuple(*to_tuple)\
        .batched(val_batch_size, partial=False)
    
    val_dl = torch.utils.data.DataLoader(val_data, batch_size=None, num_workers=1, shuffle=False)

    return train_dl, val_dl, num_train, num_val

def download_test_data(data_path, subj):
    # we will default to downloading from huggingface urls if data_path does not exist
    print("downloading NSD from huggingface...")
    os.makedirs(data_path,exist_ok=True)
    train_url, val_url, test_url = get_huggingface_urls("main",subj)
    test_url = list(braceexpand.braceexpand(test_url))
    
    from tqdm import tqdm
            
    for url in tqdm(test_url):
        destination = data_path + "/webdataset_avg_split/test/" + url.rsplit('/', 1)[-1]
        print(f"\nDownloading {url} to {destination}...")
        response = requests.get(url)
        response.raise_for_status()
        with open(destination, 'wb') as file:
            file.write(response.content)


def cosine_anneal(start, end, steps):
    return end + (start - end)/2 * (1 + torch.cos(torch.pi*torch.arange(steps)/(steps-1)))

laion_transform=transforms.Compose([
    transforms.Resize((512)),
    transforms.CenterCrop((512,512)),
])
laion_transform2=transforms.Compose([
    transforms.Resize((768)),
    transforms.CenterCrop((768,768)),
])
def query_laion(text=None, emb=None, num=8, indice_name="laion5B-L-14", groundtruth=None, clip_extractor=None, device=None, verbose=False):
    if isinstance(clip_extractor, OpenClipper):
        indice_name = "laion5B-H-14"
    if verbose: print("indice_name", indice_name)
    
    emb = nn.functional.normalize(emb,dim=-1)
    emb = emb.detach().cpu().numpy()
    
    assert len(emb) == 768 or len(emb) == 1024
    if groundtruth is not None:
        if indice_name == "laion5B-L-14":
            groundtruth = transforms.Resize((512,512))(groundtruth)
        elif indice_name == "laion5B-H-14":
            groundtruth = transforms.Resize((768,768))(groundtruth)
        if groundtruth.ndim==4:
            groundtruth=groundtruth[0]
        groundtruth=groundtruth[:3].cpu()
    
    result = None; res_length = 0
    try:
        client = ClipClient(
            url="https://knn.laion.ai/knn-service",
            indice_name=indice_name, 
            num_images=300,
            # aesthetic_score=0,
            # aesthetic_weight=np.random.randint(11),
            use_violence_detector=False,
            use_safety_model=False
        )
        result = client.query(text=text, embedding_input=emb.tolist() if emb is not None else None)
        res_length = len(result)
        if verbose: print(result)
    except:
        try:
            time.sleep(2)
            client = ClipClient(
                url="https://knn.laion.ai/knn-service",
                indice_name=indice_name, 
                num_images=300,
                aesthetic_score=5,
                aesthetic_weight=1,
                use_violence_detector=False,
                use_safety_model=False
            )
            result = client.query(text=text, embedding_input=emb.tolist() if emb is not None else None)
            res_length = len(result)
            if verbose: print(result)
        except:
            print("Query failed! Outputting blank retrieved images to prevent crashing.")
            retrieved_images = np.ones((16, 3, 512, 512))
            return retrieved_images
    
    retrieved_images = None
    tries = 0
    for res in result:
        if tries>=num:
            break
        try:
            if verbose: print(tries, "requesting...")
            # Define a User-Agent header to mimic a web browser, then load the image
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"}
            req = Request(res["url"], headers=headers) # Create a Request object with the URL and headers
            socket.setdefaulttimeout(5) # Set a global timeout for socket operations (in seconds)
            try:
                if verbose: print('socket')
                image_data = urlopen(req).read()
            except socket.timeout:
                raise Exception("Request timed out after 5 seconds")
            img = Image.open(io.BytesIO(image_data))
            # plt.imshow(img); plt.show()
            img = transforms.ToTensor()(img)
            if img.shape[0] == 1: #ensure not grayscale
                img = img.repeat(3,1,1)
            img = img[:3]
            if groundtruth is not None:
                if _check_whether_images_are_identical(img, groundtruth):
                    print("matched exact neighbor!")
                    continue
            if indice_name == "laion5B-L-14":
                best_cropped = laion_transform(img)
            else:
                best_cropped = laion_transform2(img)
            if verbose: print(best_cropped.shape)
            if retrieved_images is None:
                retrieved_images = best_cropped[None]
            else:
                retrieved_images = np.vstack((retrieved_images, best_cropped[None]))
            tries += 1
            if verbose: print('retrieved_images',retrieved_images.shape)
        except:
            time.sleep(np.random.rand())
    if verbose: print('final retrieved_images',retrieved_images.shape)
    return retrieved_images

def decode_latents(latents,vae):
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    return image

@torch.no_grad()
def reconstruction(
    image, voxel,
    clip_extractor,
    unet=None, 
    vae=None, 
    noise_scheduler=None,
    voxel2clip_cls=None,
    diffusion_priors=None,
    text_token = None,
    img_lowlevel = None,
    num_inference_steps = 50,
    recons_per_sample = 1,
    guidance_scale = 7.5,
    img2img_strength = .85,
    timesteps_prior = 100,
    seed = 0,
    retrieve=False,
    plotting=True,
    verbose=False,
    img_variations=False,
    n_samples_save=1,
    num_retrieved=16,
):
    assert n_samples_save==1, "n_samples_save must = 1. Function must be called one image at a time"
    
    brain_recons = None
    
    voxel=voxel[:n_samples_save]
    image=image[:n_samples_save]

    if unet is not None:
        do_classifier_free_guidance = guidance_scale > 1.0
        vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
        height = unet.config.sample_size * vae_scale_factor
        width = unet.config.sample_size * vae_scale_factor
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    if diffusion_priors is not None:
        if not isinstance(diffusion_priors, list):
            diffusion_priors = [diffusion_priors]
        brain_clip_embeddings_sum = None
        for diffusion_prior in diffusion_priors:
            brain_clip_embeddings0, proj_embeddings = diffusion_prior.voxel2clip(voxel.to(device).float())
            if retrieve:
                continue
            brain_clip_embeddings0 = brain_clip_embeddings0.view(len(voxel),-1,768) if isinstance(clip_extractor,Clipper) else brain_clip_embeddings0.view(len(voxel),-1,1024)
            
            if recons_per_sample>0:
                if not img_variations:
                    brain_clip_embeddings0 = brain_clip_embeddings0.repeat(recons_per_sample, 1, 1)
                    try:
                        brain_clip_embeddings = diffusion_prior.p_sample_loop(brain_clip_embeddings0.shape, 
                                                text_cond = dict(text_embed = brain_clip_embeddings0), 
                                                cond_scale = 1., timesteps = timesteps_prior,
                                                generator=generator) 
                    except:
                        brain_clip_embeddings = diffusion_prior.p_sample_loop(brain_clip_embeddings0.shape, 
                                                text_cond = dict(text_embed = brain_clip_embeddings0), 
                                                cond_scale = 1., timesteps = timesteps_prior)
                else:
                    brain_clip_embeddings0 = brain_clip_embeddings0.view(-1,768)
                    brain_clip_embeddings0 = brain_clip_embeddings0.repeat(recons_per_sample, 1)
                    brain_clip_embeddings = diffusion_prior.p_sample_loop(brain_clip_embeddings0.shape, 
                                                text_cond = dict(text_embed = brain_clip_embeddings0), 
                                                cond_scale = 1., timesteps = 1000, #1000 timesteps used from nousr pretraining
                                                generator=generator)
                if brain_clip_embeddings_sum is None:
                    brain_clip_embeddings_sum = brain_clip_embeddings
                else:
                    brain_clip_embeddings_sum += brain_clip_embeddings

        # average embeddings for all diffusion priors
        if recons_per_sample>0:
            brain_clip_embeddings = brain_clip_embeddings_sum / len(diffusion_priors)
    
    if voxel2clip_cls is not None:
        _, cls_embeddings = voxel2clip_cls(voxel.to(device).float())
    else:
        cls_embeddings = proj_embeddings
    if verbose: print("cls_embeddings.",cls_embeddings.shape)
    
    if retrieve:
        
        image_retrieved = query_laion(emb=cls_embeddings.flatten(),groundtruth=None,num=num_retrieved,
                                   clip_extractor=clip_extractor,device=device,verbose=verbose)          

    if retrieve and recons_per_sample==0:
        extracted_clip = cls_embeddings
        brain_recons = torch.Tensor(image_retrieved)
        brain_recons.to(device)
    elif recons_per_sample > 0:
        if not img_variations:
            for samp in range(len(brain_clip_embeddings)):
                brain_clip_embeddings[samp] = brain_clip_embeddings[samp]/(brain_clip_embeddings[samp,0].norm(dim=-1).reshape(-1, 1, 1) + 1e-6)
        else:
            brain_clip_embeddings = brain_clip_embeddings.unsqueeze(1)
        
        input_embedding = brain_clip_embeddings#.repeat(recons_per_sample, 1, 1)
        extracted_clip = input_embedding
        if verbose: print("input_embedding",input_embedding.shape)

        if text_token is not None:
            prompt_embeds = text_token.repeat(recons_per_sample, 1, 1)
        else:
            prompt_embeds = torch.zeros(len(input_embedding),77,768)
        if verbose: print("prompt!",prompt_embeds.shape)

        if do_classifier_free_guidance:
            input_embedding = torch.cat([torch.zeros_like(input_embedding), input_embedding]).to(device).to(unet.dtype)
            prompt_embeds = torch.cat([torch.zeros_like(prompt_embeds), prompt_embeds]).to(device).to(unet.dtype)

        # dual_prompt_embeddings
        if not img_variations:
            input_embedding = torch.cat([prompt_embeds, input_embedding], dim=1)

        # 4. Prepare timesteps
        noise_scheduler.set_timesteps(num_inference_steps=num_inference_steps, device=device)

        # 5b. Prepare latent variables
        batch_size = input_embedding.shape[0] // 2 # divide by 2 bc we doubled it for classifier-free guidance
        shape = (batch_size, unet.in_channels, height // vae_scale_factor, width // vae_scale_factor)
        if img_lowlevel is not None: # use img_lowlevel for img2img initialization
            init_timestep = min(int(num_inference_steps * img2img_strength), num_inference_steps)
            t_start = max(num_inference_steps - init_timestep, 0)
            timesteps = noise_scheduler.timesteps[t_start:]
            latent_timestep = timesteps[:1].repeat(batch_size)
            
            if verbose: print("img_lowlevel", img_lowlevel.shape)
            img_lowlevel_embeddings = clip_extractor.normalize(img_lowlevel)
            if verbose: print("img_lowlevel_embeddings", img_lowlevel_embeddings.shape)
            init_latents = vae.encode(img_lowlevel_embeddings.to(device).to(vae.dtype)).latent_dist.sample(generator)
            init_latents = vae.config.scaling_factor * init_latents
            init_latents = init_latents.repeat(recons_per_sample, 1, 1, 1)

            noise = torch.randn([recons_per_sample, 4, 64, 64], device=device, 
                                generator=generator, dtype=input_embedding.dtype)
            init_latents = noise_scheduler.add_noise(init_latents, noise, latent_timestep)
            latents = init_latents
        else:
            timesteps = noise_scheduler.timesteps
            latents = torch.randn([recons_per_sample, 4, 64, 64], device=device,
                                  generator=generator, dtype=input_embedding.dtype)
            latents = latents * noise_scheduler.init_noise_sigma

        # 7. Denoising loop
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t).to(device)

            if verbose: print("latent_model_input", latent_model_input.shape)
            if verbose: print("input_embedding", input_embedding.shape)
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=input_embedding).sample

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                
                # TODO:
                # noise_pred = dynamic_cfg(noise_pred_uncond, noise_pred_text, guidance_scale)

            # compute the previous noisy sample x_t -> x_t-1
            latents = noise_scheduler.step(noise_pred, t, latents).prev_sample
        recons = decode_latents(latents,vae).detach().cpu()

        brain_recons = recons.unsqueeze(0)

    if verbose: print("brain_recons",brain_recons.shape)
                    
    # pick best reconstruction out of several
    best_picks = np.zeros(n_samples_save).astype(np.int16)
    
    if retrieve==False:
        v2c_reference_out = nn.functional.normalize(proj_embeddings.view(len(proj_embeddings),-1),dim=-1)
        sims=[]
        for im in range(recons_per_sample): 
            currecon = clip_extractor.embed_image(brain_recons[0,[im]].float()).to(proj_embeddings.device).to(proj_embeddings.dtype)
            currecon = nn.functional.normalize(currecon.view(len(currecon),-1),dim=-1)
            cursim = batchwise_cosine_similarity(v2c_reference_out,currecon)
            sims.append(cursim.item())
        if verbose: print(sims)
        best_picks[0] = int(np.nanargmax(sims))   
        if verbose: print(best_picks)
    else: 
        v2c_reference_out = nn.functional.normalize(proj_embeddings.view(len(proj_embeddings),-1),dim=-1)
        retrieved_clips = clip_extractor.embed_image(torch.Tensor(image_retrieved).to(device)).float()
        sims=[]
        for ii,im in enumerate(retrieved_clips):
            currecon = nn.functional.normalize(im.flatten()[None],dim=-1)
            if verbose: print(v2c_reference_out.shape, currecon.shape)
            cursim = batchwise_cosine_similarity(v2c_reference_out,currecon)
            sims.append(cursim.item())
        if verbose: print(sims)
        best_picks[0] = int(np.nanargmax(sims)) 
        if verbose: print(best_picks)
        recon_img = image_retrieved[best_picks[0]]
    if recons_per_sample==0 and retrieve:
        recon_is_laion = True
        recons_per_sample = 1 # brain reconstruction will simply be the LAION nearest neighbor
    else:
        recon_is_laion = False
                    
    img2img_samples = 0 if img_lowlevel is None else 1
    laion_samples = 1 if retrieve else 0
    num_xaxis_subplots = 1+img2img_samples+laion_samples+recons_per_sample
    if plotting:
        fig, ax = plt.subplots(n_samples_save, num_xaxis_subplots, 
                           figsize=(num_xaxis_subplots*5,6*n_samples_save),facecolor=(1, 1, 1))
    else:
        fig = None
        recon_img = None
    
    im = 0
    if plotting:
        ax[0].set_title(f"Original Image")
        ax[0].imshow(torch_to_Image(image[im]))
        if img2img_samples == 1:
            ax[1].set_title(f"Img2img ({img2img_strength})")
            ax[1].imshow(torch_to_Image(img_lowlevel[im].clamp(0,1)))
    for ii,i in enumerate(range(num_xaxis_subplots-laion_samples-recons_per_sample,num_xaxis_subplots-laion_samples)):
        recon = brain_recons[im][ii]
        if recon_is_laion:
            recon = brain_recons[best_picks[0]]
        if plotting:
            if ii == best_picks[im]:
                ax[i].set_title(f"Reconstruction",fontweight='bold')
                recon_img = recon
            else:
                ax[i].set_title(f"Recon {ii+1} from brain")
            ax[i].imshow(torch_to_Image(recon))
    if plotting:
        if retrieve and not recon_is_laion:
            ax[-1].set_title(f"LAION5b top neighbor")
            ax[-1].imshow(torch_to_Image(image_retrieved0))
        for i in range(num_xaxis_subplots):
            ax[i].axis('off')
    return fig, brain_recons, best_picks, recon_img, extracted_clip

@torch.no_grad()
def reconstruction_mcd(
    image, voxel,
    clip_extractor,
    unet=None, 
    vae=None, 
    noise_scheduler=None,
    voxel2clip_cls=None,
    diffusion_priors=None,
    text_token = None,
    img_lowlevel = None,
    num_inference_steps = 50,
    recons_per_sample = 1,
    guidance_scale = 7.5,
    img2img_strength = .85,
    timesteps_prior = 100,
    seed = 0,
    retrieve=False,
    plotting=True,
    verbose=False,
    img_variations=False,
    n_samples_save=1,
    num_retrieved=16,
    mcd_forward_passes=10,
):
    assert n_samples_save==1, "n_samples_save must = 1. Function must be called one image at a time"
    
    brain_recons = None
    mc_variance = None
    mc_entropy = None
    mc_mutual_info = None
    voxel=voxel[:n_samples_save]
    image=image[:n_samples_save]

    if unet is not None:
        do_classifier_free_guidance = guidance_scale > 1.0
        vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
        height = unet.config.sample_size * vae_scale_factor
        width = unet.config.sample_size * vae_scale_factor
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    if diffusion_priors is not None:
        if not isinstance(diffusion_priors, list):
            diffusion_priors = [diffusion_priors]
        brain_clip_embeddings_sum = None
        for diffusion_prior in diffusion_priors:
            brain_clip_embeddings0, proj_embeddings = diffusion_prior.voxel2clip(voxel.to(device).float())
            outputs, mc_variance, mc_entropy, mc_mutual_info = monte_carlo_inference(data=voxel.to(device).float(),
                                                                                                forward_passes=mcd_forward_passes, 
                                                                                                model=diffusion_prior.voxel2clip,
                                                                                                use_projector=True)
            # Average across the multiple forward passes
            brain_clip_embeddings0 = torch.mean(outputs[0], axis=0)
            proj_embeddings = torch.mean(outputs[1], axis=0).reshape((1, 257, 768))
            if retrieve:
                continue
            brain_clip_embeddings0 = brain_clip_embeddings0.view(len(voxel),-1,768) if isinstance(clip_extractor,Clipper) else brain_clip_embeddings0.view(len(voxel),-1,1024)
            
            if recons_per_sample>0:
                if not img_variations:
                    brain_clip_embeddings0 = brain_clip_embeddings0.repeat(recons_per_sample, 1, 1)
                    try:
                        brain_clip_embeddings = diffusion_prior.p_sample_loop(brain_clip_embeddings0.shape, 
                                                text_cond = dict(text_embed = brain_clip_embeddings0), 
                                                cond_scale = 1., timesteps = timesteps_prior,
                                                generator=generator) 
                    except:
                        brain_clip_embeddings = diffusion_prior.p_sample_loop(brain_clip_embeddings0.shape, 
                                                text_cond = dict(text_embed = brain_clip_embeddings0), 
                                                cond_scale = 1., timesteps = timesteps_prior)
                else:
                    brain_clip_embeddings0 = brain_clip_embeddings0.view(-1,768)
                    brain_clip_embeddings0 = brain_clip_embeddings0.repeat(recons_per_sample, 1)
                    brain_clip_embeddings = diffusion_prior.p_sample_loop(brain_clip_embeddings0.shape, 
                                                text_cond = dict(text_embed = brain_clip_embeddings0), 
                                                cond_scale = 1., timesteps = 1000, #1000 timesteps used from nousr pretraining
                                                generator=generator)
                if brain_clip_embeddings_sum is None:
                    brain_clip_embeddings_sum = brain_clip_embeddings
                else:
                    brain_clip_embeddings_sum += brain_clip_embeddings

        # average embeddings for all diffusion priors
        if recons_per_sample>0:
            brain_clip_embeddings = brain_clip_embeddings_sum / len(diffusion_priors)
    
    if voxel2clip_cls is not None:
        _, cls_embeddings = voxel2clip_cls(voxel.to(device).float())
    else:
        cls_embeddings = proj_embeddings
    if verbose: print("cls_embeddings.",cls_embeddings.shape)
    
    if retrieve:
        
        image_retrieved = query_laion(emb=cls_embeddings.flatten(),groundtruth=None,num=num_retrieved,
                                   clip_extractor=clip_extractor,device=device,verbose=verbose)          

    if retrieve and recons_per_sample==0:
        extracted_clip = cls_embeddings
        brain_recons = torch.Tensor(image_retrieved)
        brain_recons.to(device)
    elif recons_per_sample > 0:
        if not img_variations:
            for samp in range(len(brain_clip_embeddings)):
                brain_clip_embeddings[samp] = brain_clip_embeddings[samp]/(brain_clip_embeddings[samp,0].norm(dim=-1).reshape(-1, 1, 1) + 1e-6)
        else:
            brain_clip_embeddings = brain_clip_embeddings.unsqueeze(1)
        
        input_embedding = brain_clip_embeddings#.repeat(recons_per_sample, 1, 1)
        extracted_clip = input_embedding
        if verbose: print("input_embedding",input_embedding.shape)

        if text_token is not None:
            prompt_embeds = text_token.repeat(recons_per_sample, 1, 1)
        else:
            prompt_embeds = torch.zeros(len(input_embedding),77,768)
        if verbose: print("prompt!",prompt_embeds.shape)

        if do_classifier_free_guidance:
            input_embedding = torch.cat([torch.zeros_like(input_embedding), input_embedding]).to(device).to(unet.dtype)
            prompt_embeds = torch.cat([torch.zeros_like(prompt_embeds), prompt_embeds]).to(device).to(unet.dtype)

        # dual_prompt_embeddings
        if not img_variations:
            input_embedding = torch.cat([prompt_embeds, input_embedding], dim=1)

        # 4. Prepare timesteps
        noise_scheduler.set_timesteps(num_inference_steps=num_inference_steps, device=device)

        # 5b. Prepare latent variables
        batch_size = input_embedding.shape[0] // 2 # divide by 2 bc we doubled it for classifier-free guidance
        shape = (batch_size, unet.in_channels, height // vae_scale_factor, width // vae_scale_factor)
        if img_lowlevel is not None: # use img_lowlevel for img2img initialization
            init_timestep = min(int(num_inference_steps * img2img_strength), num_inference_steps)
            t_start = max(num_inference_steps - init_timestep, 0)
            timesteps = noise_scheduler.timesteps[t_start:]
            latent_timestep = timesteps[:1].repeat(batch_size)
            
            if verbose: print("img_lowlevel", img_lowlevel.shape)
            img_lowlevel_embeddings = clip_extractor.normalize(img_lowlevel)
            if verbose: print("img_lowlevel_embeddings", img_lowlevel_embeddings.shape)
            init_latents = vae.encode(img_lowlevel_embeddings.to(device).to(vae.dtype)).latent_dist.sample(generator)
            init_latents = vae.config.scaling_factor * init_latents
            init_latents = init_latents.repeat(recons_per_sample, 1, 1, 1)

            noise = torch.randn([recons_per_sample, 4, 64, 64], device=device, 
                                generator=generator, dtype=input_embedding.dtype)
            init_latents = noise_scheduler.add_noise(init_latents, noise, latent_timestep)
            latents = init_latents
        else:
            timesteps = noise_scheduler.timesteps
            latents = torch.randn([recons_per_sample, 4, 64, 64], device=device,
                                  generator=generator, dtype=input_embedding.dtype)
            latents = latents * noise_scheduler.init_noise_sigma

        # 7. Denoising loop
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t).to(device)

            if verbose: print("latent_model_input", latent_model_input.shape)
            if verbose: print("input_embedding", input_embedding.shape)
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=input_embedding).sample

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                
                # TODO:
                # noise_pred = dynamic_cfg(noise_pred_uncond, noise_pred_text, guidance_scale)

            # compute the previous noisy sample x_t -> x_t-1
            latents = noise_scheduler.step(noise_pred, t, latents).prev_sample
        recons = decode_latents(latents,vae).detach().cpu()

        brain_recons = recons.unsqueeze(0)

    if verbose: print("brain_recons",brain_recons.shape)
                    
    # pick best reconstruction out of several
    best_picks = np.zeros(n_samples_save).astype(np.int16)
    
    if retrieve==False:
        v2c_reference_out = nn.functional.normalize(proj_embeddings.view(len(proj_embeddings),-1),dim=-1)
        sims=[]
        for im in range(recons_per_sample): 
            currecon = clip_extractor.embed_image(brain_recons[0,[im]].float()).to(proj_embeddings.device).to(proj_embeddings.dtype)
            currecon = nn.functional.normalize(currecon.view(len(currecon),-1),dim=-1)
            cursim = batchwise_cosine_similarity(v2c_reference_out,currecon)
            sims.append(cursim.item())
        if verbose: print(sims)
        best_picks[0] = int(np.nanargmax(sims))   
        if verbose: print(best_picks)
    else: 
        v2c_reference_out = nn.functional.normalize(proj_embeddings.view(len(proj_embeddings),-1),dim=-1)
        retrieved_clips = clip_extractor.embed_image(torch.Tensor(image_retrieved).to(device)).float()
        sims=[]
        for ii,im in enumerate(retrieved_clips):
            currecon = nn.functional.normalize(im.flatten()[None],dim=-1)
            if verbose: print(v2c_reference_out.shape, currecon.shape)
            cursim = batchwise_cosine_similarity(v2c_reference_out,currecon)
            sims.append(cursim.item())
        if verbose: print(sims)
        best_picks[0] = int(np.nanargmax(sims)) 
        if verbose: print(best_picks)
        recon_img = image_retrieved[best_picks[0]]
    if recons_per_sample==0 and retrieve:
        recon_is_laion = True
        recons_per_sample = 1 # brain reconstruction will simply be the LAION nearest neighbor
    else:
        recon_is_laion = False
                    
    img2img_samples = 0 if img_lowlevel is None else 1
    laion_samples = 1 if retrieve else 0
    num_xaxis_subplots = 1+img2img_samples+laion_samples+recons_per_sample
    if plotting:
        fig, ax = plt.subplots(n_samples_save, num_xaxis_subplots, 
                           figsize=(num_xaxis_subplots*5,6*n_samples_save),facecolor=(1, 1, 1))
    else:
        fig = None
        recon_img = None
    
    im = 0
    if plotting:
        ax[0].set_title(f"Original Image")
        ax[0].imshow(torch_to_Image(image[im]))
        if img2img_samples == 1:
            ax[1].set_title(f"Img2img ({img2img_strength})")
            ax[1].imshow(torch_to_Image(img_lowlevel[im].clamp(0,1)))
    for ii,i in enumerate(range(num_xaxis_subplots-laion_samples-recons_per_sample,num_xaxis_subplots-laion_samples)):
        recon = brain_recons[im][ii]
        if recon_is_laion:
            recon = brain_recons[best_picks[0]]
        if plotting:
            if ii == best_picks[im]:
                ax[i].set_title(f"Reconstruction",fontweight='bold')
                recon_img = recon
            else:
                ax[i].set_title(f"Recon {ii+1} from brain")
            ax[i].imshow(torch_to_Image(recon))
    if plotting:
        if retrieve and not recon_is_laion:
            ax[-1].set_title(f"LAION5b top neighbor")
            ax[-1].imshow(torch_to_Image(image_retrieved0))
        for i in range(num_xaxis_subplots):
            ax[i].axis('off')
    return fig, brain_recons, best_picks, recon_img, extracted_clip, mc_variance, mc_entropy, mc_mutual_info

def monte_carlo_inference(data,
                        model,
                        forward_passes=10,
                        use_projector=False,
                        ):
    """ Function to get the monte-carlo samples and uncertainty estimates
    through multiple forward passes

    Parameters
    ----------
    data : object
        data to pass through the model and get a prediction on
    forward_passes : int
        number of monte-carlo samples/forward passes
    model : object
        model object
    """
    outputs1 = None
    if use_projector:
        outputs2 = None
    for i in range(forward_passes):
        model.eval()
        enable_dropout(model)
        with torch.no_grad():
            if use_projector:
                output2, output1 = model(data)
            else:
                output1 = model(data)
            
        if outputs1 is None:
            outputs1 = output1.flatten().view(1, -1)
        else:
            outputs1 = torch.vstack((outputs1,output1.flatten()))
        if use_projector: 
            if outputs2 is None:
                outputs2 = output2.flatten().view(1, -1)
            else:
                outputs2 = torch.vstack((outputs2,output2.flatten()))

    # Calculating mean across multiple MCD forward passes 
    mean = torch.mean(outputs1, axis=0)

    # Calculating variance across multiple MCD forward passes 
    variance = torch.var(outputs1, axis=0)

    epsilon = torch.finfo(torch.float32).eps  # Smallest positive float32 number
    # Calculating entropy across multiple MCD forward passes 
    entropy = -torch.sum(mean * torch.log(mean + epsilon), axis=-1) 

    # Calculating mutual information across multiple MCD forward passes 
    mutual_info = entropy - torch.mean(torch.sum(-outputs1 * torch.log(outputs1 + epsilon),
                                           axis=-1), axis=0)  # shape (n_samples,)
    if use_projector:
        outputs = [outputs2, outputs1]
    else:
        outputs = [outputs1]
    return outputs, torch.mean(variance), entropy, mutual_info

def dynamic_cfg(noise_pred_uncond,noise_pred_text,guidance_scale):
    # DYNAMIC CFG: https://twitter.com/Birchlabs/status/1583984004864172032
    # THIS CURRENTLY DOES NOT WORK!
    
    dynamic_thresholding_percentile = 0.9999999999  # Set the desired percentile (.9995)
    dynamic_thresholding_mimic_scale = 7.5  # Set the desired mimic scale
    scale_factor = 27.712812921102035
    
    ut = noise_pred_uncond + (noise_pred_text - noise_pred_uncond) * guidance_scale
    ut_unscaled = ut / scale_factor
    ut_flattened = ut_unscaled.flatten(2)
    ut_means = ut_flattened.mean(dim=2).unsqueeze(2)
    ut_centered = ut_flattened - ut_means
    
    dt = noise_pred_uncond + (noise_pred_text - noise_pred_uncond) * dynamic_thresholding_mimic_scale
    dt_unscaled = dt / scale_factor
    dt_flattened = dt_unscaled.flatten(2)
    dt_means = dt_flattened.mean(dim=2).unsqueeze(2)
    dt_centered = dt_flattened - dt_means
    dt_q = torch.quantile(dt_centered.abs().float(), dynamic_thresholding_percentile, dim=2)

    a = ut_centered.abs().float()
    ut_q = torch.quantile(a, dynamic_thresholding_percentile, dim=2)
    ut_q = torch.maximum(ut_q, dt_q)
    q_ratio = ut_q / dt_q
    # print(q_ratio)
    q_ratio = q_ratio.unsqueeze(2).expand(*ut_centered.shape)

    t = ut_centered / q_ratio
    uncentered = t + ut_means
    unflattened = uncentered.unflatten(2, dt.shape[2:])
    noise_pred = (unflattened * scale_factor).half()
    return noise_pred

def select_annotations(annots, random=False):
    """
    There are 5 annotations per image. Select one of them for each image.
    """
    for i, b in enumerate(annots):
        t = ''
        if random:
            # select random non-empty annotation
            while t == '':
                rand = torch.randint(5, (1,1))[0][0]
                t = b[0, rand]
        else:
            # select first non-empty annotation
            for j in range(5):
                if b[0, j] != '':
                    t = b[0, j]
                    break
        if i == 0:
            txt = np.array(t)
        else:
            txt = np.vstack((txt, t))
    txt = txt.flatten()
    return txt

def voxel_select(voxels):
    if voxels.ndim == 2:
        return voxels
    choice = torch.rand(1)
    # random combine
    if choice <= 0.5:
        weights = torch.rand(voxels.shape[0], voxels.shape[1])[:,:,None].to(voxels.device)
        return (weights * voxels).sum(1)/weights.sum(1)
    # mean
    if choice <= 0.8:
        return voxels.mean(1)
    # random select
    randints = torch.randint(0, voxels.shape[1], (voxels.shape[0],))
    return voxels[torch.arange(voxels.shape[0]), randints]

#subject: nsd subject index between 1-8
#mode: vision, imagery
#stimtype: all, simple, complex, concepts
#average: whether to average across trials, will produce x that is (stimuli, 1, voxels)
#nest: whether to nest the data according to stimuli, will produce x that is (stimuli, trials, voxels)
#data_root: path to where the dataset is saved.
def load_nsd_mental_imagery(subject, mode, stimtype="all", average=False, nest=False, data_root="/export/raid1/home/kneel027/data/umn-neuroimaging/"):
    # This file has a bunch of information about the stimuli and cue associations that will make loading it easier
    img_stim_file = f"{data_root}/nsddata_stimuli/stimuli/nsdimagery_stimuli.pkl3"
    ex_file = open(img_stim_file, 'rb')
    imagery_dict = pickle.load(ex_file)
    ex_file.close()
    # Indicates what experiments trials belong to
    exps = imagery_dict['exps']
    # Indicates the cues for different stimuli
    cues = imagery_dict['cues']
    # Maps the cues to the stimulus image information
    image_map  = imagery_dict['image_map']
    # Organize the indices of the trials according to the modality and the type of stimuli
    cond_idx = {
    'visionsimple': np.arange(len(exps))[exps=='visA'],
    'visioncomplex': np.arange(len(exps))[exps=='visB'],
    'visionconcepts': np.arange(len(exps))[exps=='visC'],
    'visionall': np.arange(len(exps))[np.logical_or(np.logical_or(exps=='visA', exps=='visB'), exps=='visC')],
    'imagerysimple': np.arange(len(exps))[np.logical_or(exps=='imgA_1', exps=='imgA_2')],
    'imagerycomplex': np.arange(len(exps))[np.logical_or(exps=='imgB_1', exps=='imgB_2')],
    'imageryconcepts': np.arange(len(exps))[np.logical_or(exps=='imgC_1', exps=='imgC_2')],
    'imageryall': np.arange(len(exps))[np.logical_or(
                                        np.logical_or(
                                            np.logical_or(exps=='imgA_1', exps=='imgA_2'),
                                            np.logical_or(exps=='imgB_1', exps=='imgB_2')),
                                        np.logical_or(exps=='imgC_1', exps=='imgC_2'))]}
    # Load normalized betas
    x = torch.load(f"{data_root}/preprocessed_data/subject{subject}/nsd_imagery.pt").requires_grad_(False).to("cpu")
    # Find the trial indices conditioned on the type of trials we want to load
    cond_im_idx = {n: [image_map[c] for c in cues[idx]] for n,idx in cond_idx.items()}
    conditionals = cond_im_idx[mode+stimtype]
    # Stimuli file is of shape (18,3,425,425), these can be converted back into PIL images using transforms.ToPILImage()
    y = torch.load(f"{data_root}/nsddata_stimuli/stimuli/imagery_stimuli_18.pt").requires_grad_(False).to("cpu")
    # Prune the beta file down to specific experimental mode/stimuli type
    x = x[cond_idx[mode+stimtype]]
    # # If stimtype is not all, then prune the image data down to the specific stimuli type
    if stimtype == "simple":
        y = y[:6]
    elif stimtype == "complex":
        y = y[6:12]
    elif stimtype == "concepts":
        y = y[12:]

    # Average or nest the betas across trials
    if average or nest:
        x, y, sample_count = condition_average(x, y, conditionals, nest=nest)
    else:
        x = x.reshape((x.shape[0], 1, x.shape[1]))
        y = y[conditionals]

    print(x.shape, y.shape)
    return x, y

def condition_average(x, y, cond, nest=False):
    idx, idx_count = np.unique(cond, return_counts=True)
    idx_list = [np.array(cond)==i for i in np.sort(idx)]
    if nest:
        avg_x = torch.zeros((len(idx), idx_count.max(), x.shape[1]), dtype=torch.float32)
    else:
        avg_x = torch.zeros((len(idx), 1, x.shape[1]), dtype=torch.float32)
    arranged_y = torch.zeros((len(idx)), y.shape[1], y.shape[2], y.shape[3])
    for i, m in enumerate(idx_list):
        if nest:
            if np.sum(m) == idx_count.max():
                avg_x[i] = x[m]
            else:
                avg_x[i,:np.sum(m)] = x[m]
        else:
            avg_x[i] = torch.mean(x[m], axis=0)
        arranged_y[i] = y[m[0]]

    return avg_x, y, len(idx_count)

def search_and_open_image(directory, img_identifier):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if img_identifier in file:
                image_path = os.path.join(root, file)
                image = Image.open(image_path)
                return image
    print(f"Image {img_identifier} not found in {directory}")
    return None

def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()