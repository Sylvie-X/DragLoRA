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
# Codes Modified by DragloRA Authors (2025)

import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
import pickle
from PIL import Image
import time
from copy import deepcopy
from einops import rearrange
from types import SimpleNamespace
import random
from diffusers import DDIMScheduler, AutoencoderKL
from diffusers.models.attention_processor import (
    AttnAddedKVProcessor,
    AttnAddedKVProcessor2_0,
    SlicedAttnAddedKVProcessor,
)
from diffusers.models.lora import LoRALinearLayer
from progress.bar import Bar
import safetensors

import sys
sys.path.insert(0, '../')
from utils.pipelines.drag_pipeline_lora import DragPipelineDup
from utils.models.unet_2d_condition import UNet2DConditionModelSAKV
from utils.drag_utils_lora import drag_diffusion_update
from utils.attn_utils_lora import MutualSelfAttentionControl


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def preprocess_image(image, device):
    image = torch.from_numpy(image).float() / 127.5 - 1 # [-1, 1]
    image = rearrange(image, "h w c -> 1 c h w")
    image = image.to(device)
    return image


def run_drag(source_image,
             mask,
             prompt,
             points,
             inversion_strength,
             lam_mask,
             lam_dds,
             lora_lr,
             unet_feature_idx,
             n_pix_step,
             model_path,
             vae_path,
             lora_path,
             start_step,
             start_layer,
             save_dir,
    ):
    seed = 42
    seed_everything(seed)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012,
                          beta_schedule="scaled_linear", clip_sample=False,
                          set_alpha_to_one=False, steps_offset=1)
    model = DragPipelineDup.from_pretrained(model_path, scheduler=scheduler).to(device)
    model.modify_unet_forward()
    ori_unet = UNet2DConditionModelSAKV.from_pretrained(model_path, subfolder='unet').to(device)
    if vae_path != "default":
        model.vae = AutoencoderKL.from_pretrained(
            vae_path
        ).to(model.vae.device, model.vae.dtype)
    model.vae.requires_grad_(False)
    model.text_encoder.requires_grad_(False)
    model.unet.requires_grad_(False)
    args = SimpleNamespace()
    args.prompt = prompt
    args.points = points
    args.n_inference_step = 50
    args.n_actual_inference_step = round(inversion_strength * args.n_inference_step)
    args.guidance_scale = 1.0
    args.unet_feature_idx = [unet_feature_idx]
    args.r_m = 1
    args.r_p = 3
    args.lam_mask = lam_mask
    args.lam_dds = lam_dds
    args.lora_lr = lora_lr
    args.n_pix_step = n_pix_step
    full_h, full_w = source_image.shape[:2]
    args.sup_res_h = int(0.5*full_h)
    args.sup_res_w = int(0.5*full_w)
    print(args)

    source_image = preprocess_image(source_image, device)
    
    # Set correct lora layers to be optimized
    unet_lora_parameters = []
    lora_rank=16
    for attn_processor_name, attn_processor in model.unet.attn_processors.items():
        attn_module = model.unet        
        for n in attn_processor_name.split(".")[:-1]:
            attn_module = getattr(attn_module, n)
        attn_module.to_q.set_lora_layer(
            LoRALinearLayer(
                in_features=attn_module.to_q.in_features,
                out_features=attn_module.to_q.out_features,
                rank=lora_rank,
                device=device,
            )
        )
        attn_module.to_k.set_lora_layer(
            LoRALinearLayer(
                in_features=attn_module.to_k.in_features,
                out_features=attn_module.to_k.out_features,
                rank=lora_rank,
                device=device,
            )
        )
        attn_module.to_v.set_lora_layer(
            LoRALinearLayer(
                in_features=attn_module.to_v.in_features,
                out_features=attn_module.to_v.out_features,
                rank=lora_rank,
                device=device,
            )
        )
        attn_module.to_out[0].set_lora_layer(
            LoRALinearLayer(
                in_features=attn_module.to_out[0].in_features,
                out_features=attn_module.to_out[0].out_features,
                rank=lora_rank,
                device=device,
            )
        )
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
    params_to_optimize = (unet_lora_parameters)
    
    # post-set lora to init drag-lora from reconstruction-lora
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

    # invert the source image
    invert_code = model.invert(source_image,
                               prompt,
                               guidance_scale=args.guidance_scale,
                               num_inference_steps=args.n_inference_step,
                               num_actual_inference_steps=args.n_actual_inference_step,
                               )
    
    mask = torch.from_numpy(mask).float() / 255.
    mask[mask > 0.0] = 1.0
    mask = rearrange(mask, "h w -> 1 1 h w").cuda()
    mask = F.interpolate(mask, (args.sup_res_h, args.sup_res_w), mode="nearest")
    
    handle_points = []
    target_points = []
    for idx, point in enumerate(points):
        cur_point = torch.tensor([point[1]/full_h*args.sup_res_h, point[0]/full_w*args.sup_res_w])
        cur_point = torch.round(cur_point)
        if idx % 2 == 0:
            handle_points.append(cur_point)
        else:
            target_points.append(cur_point)
    print('handle points:', handle_points)
    print('target points:', target_points)

    init_code = invert_code
    init_code_orig = deepcopy(invert_code)
    model.scheduler.set_timesteps(args.n_inference_step)
    t = model.scheduler.timesteps[args.n_inference_step - args.n_actual_inference_step]
    text_embeddings = model.get_text_embeddings(args.prompt)
    
    # drag at t=35
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
        save_interlora=False,
        save_finallora=False
    )

    # denose z_{35} to z_{0} with MasaCtrl
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
            latents_ini=init_code_orig,
            )

    gen_image = F.interpolate(gen_image, (full_h, full_w), mode='bilinear')
    out_image = gen_image.cpu().permute(0, 2, 3, 1).numpy()[0]
    out_image = (out_image * 255).astype(np.uint8)
    return out_image


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="setting arguments")
    parser.add_argument('--recon_steps', type=int, default=80, help='number of steps to finetune reconstruction-lora')
    parser.add_argument('--drag_steps', type=int, default=80, help='number of steps to finetune drag-lora')
    parser.add_argument('--inv_strength', type=float, default=0.7, help='inversion strength')
    parser.add_argument('--lora_lr', type=float, default=0.0001, help='learning rate for draglora')
    parser.add_argument('--unet_feature_idx', type=int, default=3, help='feature idx of unet features')
    parser.add_argument('--root_dir', type=str, default='drag_bench_data', help='path to dragbench')
    parser.add_argument('--img_dir', type=str, default='', help='path to drag-once outputs, change it if running dragback')
    parser.add_argument('--lora_dir', type=str, default='drag_bench_lora', help='path to pre-saved reconstruction-lora')
    parser.add_argument('--save_dir', type=str, default='drag_results', help='path to save dragged outputs')
    parser.add_argument('--model_path', type=str, default='runwayml/stable-diffusion-v1-5', help='path to sd1.5')
    args = parser.parse_args()
    
    all_category = [
        'animals',
        'art_work',
        'human_head',
        'building_city_view',
        'land_scape',
        'building_countryside_view',
        'human_upper_body',
        'human_full_body',
        'interior_design',
        'other_objects',
    ]

    # mkdir if necessary
    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)
        for cat in all_category:
            os.mkdir(os.path.join(args.save_dir,cat))

    bar = Bar('Processing', max=205)
    times = []
    for cat in all_category:
        file_dir = os.path.join(args.root_dir, cat)
        for sample_name in os.listdir(file_dir):
            if sample_name == '.DS_Store':
                continue
            sample_path = os.path.join(file_dir, sample_name)

            if args.img_dir!='': 
                # drag the previously dragged image back to original
                img_path = os.path.join(args.img_dir, cat, sample_name, 'dragged_image.png')
            else: 
                # drag the original image according to benchmark requirements
                img_path = os.path.join(sample_path, 'original_image.png')
            print(f'img_path:{img_path}')

            source_image = Image.open(img_path)
            source_image = np.array(source_image)

            with open(os.path.join(sample_path, 'meta_data.pkl'), 'rb') as f:
                meta_data = pickle.load(f)
            prompt = meta_data['prompt']
            mask = meta_data['mask']
            points = meta_data['points']
            
            if 'dragged_image' in img_path: # exchange the handle-target points if drag-back
                for i in range(0,len(points),2):
                    points[i], points[i+1] = points[i+1], points[i]

            # load lora
            lora_path = os.path.join(args.lora_dir, cat, sample_name, str(args.recon_steps))
            print("applying lora: " + lora_path)

            sample_save_dir = os.path.join(args.save_dir, cat, sample_name)
            if not os.path.isdir(sample_save_dir):
                os.mkdir(sample_save_dir)
            
            tic = time.time()
            out_image = run_drag(
                source_image,
                mask,
                prompt,
                points,
                inversion_strength=args.inv_strength,
                lam_mask=0.1,
                lam_dds=50,
                lora_lr=args.lora_lr,
                unet_feature_idx=args.unet_feature_idx,
                n_pix_step=args.drag_steps,
                model_path=args.model_path,
                vae_path="default",
                lora_path=lora_path,
                start_step=0,
                start_layer=10,
                save_dir=sample_save_dir,
            )
            toc = time.time()
            times.append(toc-tic)
            Image.fromarray(out_image).save(os.path.join(sample_save_dir, 'dragged_image.png'))
            bar.next()
    bar.finish()
    print(f'avg time for {args.save_dir}: {np.array(times).mean():.2f}s')
