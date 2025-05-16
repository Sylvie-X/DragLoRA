# *************************************************************************
# Copyright (2023) Bytedance Inc.
#
# Copyright (2023) DragDiffusion Authors 
#
# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License. 
# You may obtain a copy of the License at 
#
#     http://www.apache.org/licenses/LICENSE-2.0 
#
# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
# See the License for the specific language governing permissions and 
# limitations under the License. 
# *************************************************************************
import pickle
import os
import cv2
import numpy as np
import gradio as gr
from copy import deepcopy
from einops import rearrange
from types import SimpleNamespace
import datetime
import PIL
from PIL import Image
from torchvision import transforms
from PIL.ImageOps import exif_transpose
import torch
import torch.nn.functional as F
import safetensors
from diffusers import DDIMScheduler, AutoencoderKL, DPMSolverMultistepScheduler
from diffusers.models.embeddings import ImageProjection
from diffusers.models.lora import LoRALinearLayer
from diffusers.models.attention_processor import (
    AttnAddedKVProcessor,
    AttnAddedKVProcessor2_0,
    SlicedAttnAddedKVProcessor,
)
from torchvision.utils import save_image
from pytorch_lightning import seed_everything
from .pipelines.drag_pipeline_lora import DragPipelineDup
from .lora_utils import train_lora
from .models.unet_2d_condition import UNet2DConditionModelSAKV
from .drag_utils_lora import drag_diffusion_update
from .attn_utils_lora import register_attention_editor_diffusers, MutualSelfAttentionControl

# -------------- general UI functionality --------------
def clear_all(length=480):
    return gr.Image.update(value=None, height=length, width=length, interactive=True), \
        gr.Image.update(value=None, height=length, width=length, interactive=False), \
        gr.Image.update(value=None, height=length, width=length, interactive=False), \
        [], None, None

def clear_all_gen(length=480):
    return gr.Image.update(value=None, height=length, width=length, interactive=False), \
        gr.Image.update(value=None, height=length, width=length, interactive=False), \
        gr.Image.update(value=None, height=length, width=length, interactive=False), \
        [], None, None, None

def mask_image(image,
               mask,
               color=[255,0,0],
               alpha=0.5):
    """ Overlay mask on image for visualization purpose. 
    Args:
        image (H, W, 3) or (H, W): input image
        mask (H, W): mask to be overlaid
        color: the color of overlaid mask
        alpha: the transparency of the mask
    """
    out = deepcopy(image)
    img = deepcopy(image)
    img[mask == 1] = color
    out = cv2.addWeighted(img, alpha, out, 1-alpha, 0, out)
    return out

def store_img(img, length=512):
    image, mask = img["image"], np.float32(img["mask"][:, :, 0]) / 255.
    height,width,_ = image.shape
    image = Image.fromarray(image)
    image = exif_transpose(image)
    image = image.resize((length,int(length*height/width)), PIL.Image.BILINEAR)
    mask  = cv2.resize(mask, (length,int(length*height/width)), interpolation=cv2.INTER_NEAREST)
    image = np.array(image)

    if mask.sum() > 0:
        mask = np.uint8(mask > 0)
        masked_img = mask_image(image, 1 - mask, color=[0, 0, 0], alpha=0.3)
    else:
        masked_img = image.copy()
    # when new image is uploaded, `selected_points` should be empty
    return image, [], gr.Image.update(value=masked_img, interactive=True), mask

# once user upload an image, the original image is stored in `original_image`
# the same image is displayed in `input_image` for point clicking purpose
def store_img_gen(img):
    image, mask = img["image"], np.float32(img["mask"][:, :, 0]) / 255.
    image = Image.fromarray(image)
    image = exif_transpose(image)
    image = np.array(image)
    if mask.sum() > 0:
        mask = np.uint8(mask > 0)
        masked_img = mask_image(image, 1 - mask, color=[0, 0, 0], alpha=0.3)
    else:
        masked_img = image.copy()
    # when new image is uploaded, `selected_points` should be empty
    return image, [], masked_img, mask

# user click the image to get points, and show the points on the image
def get_points(img,
               sel_pix,
               evt: gr.SelectData):
    # collect the selected point
    sel_pix.append(evt.index)
    # draw points
    points = []
    for idx, point in enumerate(sel_pix):
        if idx % 2 == 0:
            # draw a red circle at the handle point
            cv2.circle(img, tuple(point), 10, (255, 0, 0), -1)
        else:
            # draw a blue circle at the handle point
            cv2.circle(img, tuple(point), 10, (0, 0, 255), -1)
        points.append(tuple(point))
        # draw an arrow from handle point to target point
        if len(points) == 2:
            cv2.arrowedLine(img, points[0], points[1], (255, 255, 255), 4, tipLength=0.5)
            points = []
    return img if isinstance(img, np.ndarray) else np.array(img)

# clear all handle/target points
def undo_points(original_image,
                mask):
    if mask.sum() > 0:
        mask = np.uint8(mask > 0)
        masked_img = mask_image(original_image, 1 - mask, color=[0, 0, 0], alpha=0.3)
    else:
        masked_img = original_image.copy()
    return masked_img, []
# ------------------------------------------------------

# ----------- dragging user-input image utils -----------
def train_lora_interface(original_image,
                         prompt,
                         model_path,
                         vae_path,
                         lora_path,
                         lora_step,
                         lora_lr,
                         lora_batch_size,
                         lora_rank,
                         progress=gr.Progress()):
    train_lora(
        original_image,
        prompt,
        model_path,
        vae_path,
        lora_path,
        lora_step,
        lora_lr,
        lora_batch_size,
        lora_rank,
        progress)
    return "Training RecLoRA Done!"

def preprocess_image(image,
                     device,
                     dtype=torch.float32):
    image = torch.from_numpy(image).float() / 127.5 - 1 # [-1, 1]
    image = rearrange(image, "h w c -> 1 c h w")
    image = image.to(device, dtype)
    return image

def run_drag(source_image,
             image_with_clicks,
             mask,
             prompt,
             points,
             inversion_strength,
             lam,
             lam_dds,
             lora_lr,
             n_pix_step,
             model_path,
             vae_path,
             lora_path,
             start_step,
             start_layer,
             save_dir
    ):
    # initialize model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012,
                          beta_schedule="scaled_linear", clip_sample=False,
                          set_alpha_to_one=False, steps_offset=1)
    model = DragPipelineDup.from_pretrained(model_path, scheduler=scheduler).to(device)
    model.modify_unet_forward()
    ori_unet = UNet2DConditionModelSAKV.from_pretrained(model_path, subfolder='unet').to(device)
    # set vae
    if vae_path != "default":
        model.vae = AutoencoderKL.from_pretrained(
            vae_path
        ).to(model.vae.device, model.vae.dtype)

    model.enable_model_cpu_offload()
    model.vae.requires_grad_(False)
    model.text_encoder.requires_grad_(False)
    model.unet.requires_grad_(False)
    seed = 42 
    seed_everything(seed)

    args = SimpleNamespace()
    args.prompt = prompt
    args.points = points
    args.n_inference_step = 50
    args.n_actual_inference_step = round(inversion_strength * args.n_inference_step)
    args.guidance_scale = 1.0

    args.unet_feature_idx = [3]

    args.r_m = 1
    args.r_p = 3
    args.lam_mask = lam
    args.lam_dds = lam_dds
    args.lora_lr = lora_lr
    args.n_pix_step = n_pix_step

    full_h, full_w = source_image.shape[:2]
    args.sup_res_h = int(0.5*full_h)
    args.sup_res_w = int(0.5*full_w)

    print(args)
    
    save_prefix = datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S")
    inter_path = os.path.join(lora_path, save_prefix)
    os.makedirs(inter_path, exist_ok=True)

    source_image = preprocess_image(source_image, device)
    image_with_clicks = preprocess_image(image_with_clicks, device)
    tensor_transforms = transforms.Compose(
            [
                transforms.Normalize([-1], [2]),
                transforms.ToPILImage(),
            ]
    )

    tensor_transforms(source_image[0]).save(os.path.join(inter_path, 'original_image.png'))
    tensor_transforms(image_with_clicks[0]).save(os.path.join(inter_path, 'user_drag.png'))
    mydict = {}
    mydict['prompt']=prompt
    mydict['points']=points
    mydict['mask']=mask 
    with open(os.path.join(inter_path, 'meta_data.pkl'), 'wb') as file:
        pickle.dump(mydict, file)
    
    # preparing editing meta data (handle, target, mask)
    mask = torch.from_numpy(mask).float()
    mask[mask > 0.0] = 1.0
    mask = rearrange(mask, "h w -> 1 1 h w").cuda()
    mask = F.interpolate(mask, (args.sup_res_h, args.sup_res_w), mode="nearest")

    handle_points = []
    target_points = []
    # here, the point is in x,y coordinate
    for idx, point in enumerate(points):
        cur_point = torch.tensor([point[1]/full_h*args.sup_res_h, point[0]/full_w*args.sup_res_w])
        cur_point = torch.round(cur_point)
        if idx % 2 == 0:
            handle_points.append(cur_point)
        else:
            target_points.append(cur_point)
    print('handle points:', handle_points)
    print('target points:', target_points)

    ### Set correct lora layers
    unet_lora_parameters = []
    lora_rank=16
    
    for attn_processor_name, attn_processor in model.unet.attn_processors.items():
        attn_module = model.unet        
        for n in attn_processor_name.split(".")[:-1]:
            attn_module = getattr(attn_module, n)

        # Set the `lora_layer` attribute of the attention-related matrices.
        attn_module.to_q.set_lora_layer(
            LoRALinearLayer(
                in_features=attn_module.to_q.in_features,
                out_features=attn_module.to_q.out_features,
                rank=lora_rank,
                device=device
            )
        )
        attn_module.to_k.set_lora_layer(
            LoRALinearLayer(
                in_features=attn_module.to_k.in_features,
                out_features=attn_module.to_k.out_features,
                rank=lora_rank,
                device=device
            )
        )
        attn_module.to_v.set_lora_layer(
            LoRALinearLayer(
                in_features=attn_module.to_v.in_features,
                out_features=attn_module.to_v.out_features,
                rank=lora_rank,
                device=device
            )
        )
        attn_module.to_out[0].set_lora_layer(
            LoRALinearLayer(
                in_features=attn_module.to_out[0].in_features,
                out_features=attn_module.to_out[0].out_features,
                rank=lora_rank,
                device=device
            )
        )

        # Accumulate the LoRA params to optimize.
        unet_lora_parameters.extend(attn_module.to_q.lora_layer.parameters())
        unet_lora_parameters.extend(attn_module.to_k.lora_layer.parameters())
        unet_lora_parameters.extend(attn_module.to_v.lora_layer.parameters())
        unet_lora_parameters.extend(attn_module.to_out[0].lora_layer.parameters())

        if isinstance(attn_processor, (AttnAddedKVProcessor, SlicedAttnAddedKVProcessor, AttnAddedKVProcessor2_0)):
            attn_module.add_k_proj.set_lora_layer(
                LoRALinearLayer(
                    in_features=attn_module.add_k_proj.in_features,
                    out_features=attn_module.add_k_proj.out_features,
                    rank=args.rank,
                    device=device
                )
            )
            attn_module.add_v_proj.set_lora_layer(
                LoRALinearLayer(
                    in_features=attn_module.add_v_proj.in_features,
                    out_features=attn_module.add_v_proj.out_features,
                    rank=args.rank,
                    device=device
                )
            )
            unet_lora_parameters.extend(attn_module.add_k_proj.lora_layer.parameters())
            unet_lora_parameters.extend(attn_module.add_v_proj.lora_layer.parameters())

    # Optimizer creation
    params_to_optimize = (unet_lora_parameters)

    ## post-set lora to init drag lora from inversion lora
    if lora_path == "":
        print("applying default parameters")
        model.unet.set_default_attn_processor()
    else:
        print("applying lora: " + lora_path)
        state_dict = safetensors.torch.load_file(os.path.join(lora_path,'pytorch_lora_weights.safetensors'), device="cpu")
        state_dict = {k.replace("unet.", "").replace("lora", "lora_layer"): v for k, v in state_dict.items()}
        model.unet.load_state_dict(state_dict,strict=False)
        ori_unet.load_attn_procs(lora_path)
        ori_unet.fuse_lora(lora_scale=1.0)

    # obtain text embeddings
    print(prompt)
    text_embeddings = model.get_text_embeddings(prompt)

    # invert the source image
    invert_code = model.invert(source_image,
                               prompt,
                               guidance_scale=args.guidance_scale,
                               num_inference_steps=args.n_inference_step,
                               num_actual_inference_steps=args.n_actual_inference_step)

    init_code = invert_code
    init_code_orig = deepcopy(invert_code)
    model.scheduler.set_timesteps(args.n_inference_step)
    t = model.scheduler.timesteps[args.n_inference_step - args.n_actual_inference_step]
    
    latents_update = drag_diffusion_update(
        model,
        init_code,
        text_embeddings,
        t,
        handle_points,
        target_points,
        mask,
        params_to_optimize,
        save_dir,
        args,
        ori_unet=ori_unet,
        save_interlora=False
    )
    
    with torch.no_grad():
        editor = MutualSelfAttentionControl(start_step=start_step,
                                            start_layer=start_layer,
                                            total_steps=args.n_inference_step,
                                            guidance_scale=args.guidance_scale)
        gen_image = model.dup_generation(
            ori_unet,
            editor,
            prompt=args.prompt,
            batch_size=1,
            latents=latents_update,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.n_inference_step,
            num_actual_inference_steps=args.n_actual_inference_step,
            latents_ini=init_code_orig)

    # resize gen_image into the size of source_image
    # we do this because shape of gen_image will be rounded to multipliers of 8
    gen_image = F.interpolate(gen_image, (full_h, full_w), mode='bilinear')

    # save the original image, user editing instructions, synthesized image
    save_result = torch.cat([
        source_image.float() * 0.5 + 0.5,
        torch.ones((1,3,full_h,25)).cuda(),
        image_with_clicks.float() * 0.5 + 0.5,
        torch.ones((1,3,full_h,25)).cuda(),
        gen_image[0:1].float()
    ], dim=-1)

    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    save_image(save_result, os.path.join(save_dir, save_prefix + '.png'))

    out_image = gen_image.cpu().permute(0, 2, 3, 1).numpy()[0]
    out_image = (out_image * 255).astype(np.uint8)
    return out_image