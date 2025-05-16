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

from tqdm import tqdm
import os
import torch
import torch.nn.functional as F
from diffusers.training_utils import unet_lora_state_dict
from diffusers.loaders import LoraLoaderMixin
from copy import deepcopy
from diffusers.utils.torch_utils import randn_tensor
import csv
import math 
from utils.attn_utils_lora import register_attention_editor_diffusers


def point_tracking(F0,
                   F1,
                   handle_points,
                   handle_points_init,
                   target_points,
                   args,
                   draglora_fast = False):
    with torch.no_grad():
        _, _, max_r, max_c = F0.shape
        minD = []
        for i in range(len(handle_points)):
            pi0, pi, ti = handle_points_init[i], handle_points[i], target_points[i]
            f0 = F0[:, :, int(pi0[0]), int(pi0[1])]

            # Neighbourhood Region
            r1, r2 = max(0,int(pi[0])-args.r_p), min(max_r,int(pi[0])+args.r_p+1)
            c1, c2 = max(0,int(pi[1])-args.r_p), min(max_c,int(pi[1])+args.r_p+1)
            x_coords = torch.arange(r1, r2)
            y_coords = torch.arange(c1, c2)
            x_grid, y_grid = torch.meshgrid(x_coords, y_coords)
            coordinates = torch.stack((x_grid.flatten(), y_grid.flatten()), dim=1)

            if draglora_fast:
                # Angle-closer Region
                diretcion = ti-pi 
                distance = diretcion.norm() 
                if distance<1: 
                    continue 
                diretcion  = diretcion / distance 
                cos_angle_threshold = math.cos(math.radians(30)) 
                vectors = coordinates - pi 
                vectors = vectors / (vectors.norm(dim=1,keepdim=True)+1e-8) 
                cos_angles = torch.sum(vectors * diretcion, dim=1) 
                validate_mask = cos_angles >= cos_angle_threshold 
                coordinates = coordinates[validate_mask] 
            else:
                # Distance-closer Region
                threshold = torch.norm(pi - ti) 
                point_distances = torch.norm(coordinates-ti,dim=1) 
                coordinates = coordinates[point_distances<=threshold] 

            F1_neighbor = F1[:, :, coordinates[:,0], coordinates[:,1]]

            all_dist = (f0.unsqueeze(dim=-1) - F1_neighbor).abs().mean(dim=1) ### pt_
            all_dist = all_dist.squeeze(dim=0)
            minD.append(round(all_dist.min().item(),2))

            new_point = coordinates[all_dist.argmin().item()]
            handle_points[i][0] = new_point[0]
            handle_points[i][1] = new_point[1]

        return handle_points,tuple(minD)


def check_handle_reach_target(handle_points,
                              target_points):
    all_dist = list(map(lambda p,q: (p-q).norm(), handle_points, target_points))
    return (torch.tensor(all_dist) < 2.0).all()


def interpolate_feature_patch(feat,
                              y1,
                              y2,
                              x1,
                              x2):
    x1_floor = torch.floor(x1).long()
    x1_cell = x1_floor + 1
    dx = torch.floor(x2).long() - torch.floor(x1).long()

    y1_floor = torch.floor(y1).long()
    y1_cell = y1_floor + 1
    dy = torch.floor(y2).long() - torch.floor(y1).long()

    wa = (x1_cell.float() - x1) * (y1_cell.float() - y1)
    wb = (x1_cell.float() - x1) * (y1 - y1_floor.float())
    wc = (x1 - x1_floor.float()) * (y1_cell.float() - y1)
    wd = (x1 - x1_floor.float()) * (y1 - y1_floor.float())

    Ia = feat[:, :, y1_floor : y1_floor+dy, x1_floor : x1_floor+dx]
    Ib = feat[:, :, y1_cell : y1_cell+dy, x1_floor : x1_floor+dx]
    Ic = feat[:, :, y1_floor : y1_floor+dy, x1_cell : x1_cell+dx]
    Id = feat[:, :, y1_cell : y1_cell+dy, x1_cell : x1_cell+dx]

    return Ia * wa + Ib * wb + Ic * wc + Id * wd


def drag_diffusion_update(model,
                          init_code,
                          text_embeddings,
                          t,
                          handle_points,
                          target_points,
                          mask,
                          params_to_optimize,
                          save_lora_path,
                          args,
                          ori_unet=None,
                          save_interlora=False):

    assert len(handle_points) == len(target_points), \
        "number of handle point must equals target points"
    init_code_ori = deepcopy(init_code)
    f0_patches=[]
    with torch.no_grad():
        unet_output, F0 = model.forward_unet_features(
            init_code,
            t,
            encoder_hidden_states=text_embeddings,
            layer_idx=args.unet_feature_idx,
            interp_res_h=args.sup_res_h,
            interp_res_w=args.sup_res_w
        )
        x_prev_0, _ = model.step(unet_output, t, init_code)
        _, _, max_r, max_c = F0.shape
        for i in range(len(handle_points)):
            pi, ti = handle_points[i], target_points[i]
            r1, r2 = max(0,int(pi[0])-args.r_m), min(max_r,int(pi[0])+args.r_m+1)
            c1, c2 = max(0,int(pi[1])-args.r_m), min(max_c,int(pi[1])+args.r_m+1)
            f0_patches.append(F0[:,:,r1:r2, c1:c2].detach())
        
    init_code.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=args.lora_lr,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-08,
    )
    scaler = torch.cuda.amp.GradScaler()

    handle_points_init = deepcopy(handle_points)
    interp_mask = F.interpolate(mask, (init_code.shape[2],init_code.shape[3]), mode='nearest')
    using_mask = interp_mask.sum() != 0.0
    # prevPTs = set()
    steps=[0]
    ada_num = 0
    adanums=[ada_num]
    minD = tuple()
    minDs = [0]
    PTs = [[tuple(p.tolist()) for p in handle_points]]
    step_idx = 0
    actual_step_idx = 0
    retain = 0
    ada = False # whether to activate only-ILFA mode
    nomove=0

    while actual_step_idx<args.n_pix_step:
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            model.unet.train()
            unet_output, F1 = model.forward_unet_features(
                init_code,
                t,
                encoder_hidden_states=text_embeddings,
                layer_idx=args.unet_feature_idx,
                interp_res_h=args.sup_res_h,
                interp_res_w=args.sup_res_w
            )
            x_prev_updated,pred_x0 = model.step(unet_output, t, init_code)

            if step_idx != 0: 
                handle_points_pre = deepcopy(handle_points)
                handle_points, minD = point_tracking(F0, F1, handle_points, handle_points_init, target_points, args)
                if not ada:
                    if step_idx > 10 and min(minD) < 1.0:
                        if (torch.stack(handle_points) - torch.stack(curTs)).norm() < 1.4:
                            ada = True
                            print('good enough, activate only-ILFA!')
                    elif max(minD) > 1.3:
                        # Point tracking confidence is low, revert to previous handle points.
                        # If retain too many times, keep the tracked handle points.
                        retain += 1
                        if retain <= 3:
                            for i in range(len(minD)):
                                if minD[i] > 1.3:
                                    handle_points[i] = handle_points_pre[i]
                            step_idx -= 1
                            print('minD too large, trace back the points!')
                        else:
                            retain = 0
                            print('too many trace back, update the points!')
                    elif retain > 0:
                        retain = 0
                else:
                    if torch.stack(handle_points).equal(torch.stack(handle_points_pre)):
                        nomove += 1
                    if min(minD) > 1.3 or nomove>5:
                        nomove = 0
                        ada = False
                        print('bad enough, deactivate only-ILFA!')
                minDs.append(minD)
                PTs.append([tuple(p.tolist()) for p in handle_points])
                steps.append(tuple([step_idx, actual_step_idx]))
                adanums.append(ada_num)
                print('new handle points', handle_points)      

            # break if all handle points have reached the targets
            if check_handle_reach_target(handle_points, target_points):
                break
            
            if not ada:
                loss = 0.0
                curTs = []

                for i in range(len(handle_points)):
                    pi, ti = handle_points[i], target_points[i]

                    if (ti - pi).norm() < 2.:
                        curTs.append(ti)
                        continue

                    di = (ti - pi) / (ti - pi).norm()
                    r1, r2 = max(0,int(pi[0])-args.r_m), min(max_r,int(pi[0])+args.r_m+1)
                    c1, c2 = max(0,int(pi[1])-args.r_m), min(max_c,int(pi[1])+args.r_m+1)
                    f0_patch = f0_patches[i].detach() # use fixed F0
                    f1_patch = interpolate_feature_patch(F1,r1+di[0],r2+di[0],c1+di[1],c2+di[1])
                    curTs.append(pi+di)
                    drag_loss = ((2*args.r_m+1)**2)*F.l1_loss(f0_patch, f1_patch)
                    loss += drag_loss
                    
                if using_mask:
                    loss_mask = args.lam_mask * ((x_prev_updated-x_prev_0)*(1.0-interp_mask)).abs().sum()
                    loss += loss_mask
                    
                if ori_unet is not None: # apply dds loss
                    with torch.no_grad():
                        noise = torch.randn_like(pred_x0)
                        bsz = pred_x0.shape[0]
                        timesteps = torch.randint(
                            0, model.scheduler.config.num_train_timesteps, (bsz,), device=pred_x0.device
                        )
                        timesteps = timesteps.long()
                        alpha_prod_t = model.scheduler.alphas_cumprod[timesteps.item()] if timesteps >= 0 else model.scheduler.final_alpha_cumprod
                        scaler_t = 1 - alpha_prod_t
                        noisy_model_input = model.scheduler.add_noise(pred_x0, noise, timesteps)
                        model_pred_src, _ = ori_unet(noisy_model_input,
                                        timesteps,
                                        text_embeddings)
                        model_pred_tgt = model.unet(noisy_model_input,
                                        timesteps,
                                        text_embeddings)
                        grad = model_pred_src - model_pred_tgt
                        tgt = (pred_x0 - torch.nan_to_num(scaler_t * grad)).detach()
                    dds_loss = 0.5 * args.lam_dds * F.mse_loss(pred_x0.float(), tgt.float(), reduction="mean")
                    loss += dds_loss

                print('loss=%.2f'%(loss.item()))

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                step_idx+=1
                actual_step_idx+=1

        with torch.no_grad():   
            # Input latent Feature Adaptation (ILFA)
            ada_num += 1
            model_pred_tgt = model.unet(init_code,
                            t,
                            text_embeddings)
            x_next, _ = model.step(model_pred_tgt, t, init_code)
            noise = randn_tensor(init_code.shape,device=init_code.device,dtype=init_code.dtype)
            
            alphas_cumprod = model.scheduler.alphas_cumprod.to(device=init_code.device, dtype=init_code.dtype)
            alpha_t = alphas_cumprod[t] / alphas_cumprod[t-20]
            sqrt_alpha_t = alpha_t ** 0.5
            sqrt_beta_t = (1-alpha_t)**0.5
            sqrt_alpha_t = sqrt_alpha_t.flatten()
            sqrt_beta_t = sqrt_beta_t.flatten()
            while len(sqrt_alpha_t.shape) < len(init_code.shape):
                sqrt_alpha_t = sqrt_alpha_t.unsqueeze(-1)
                sqrt_beta_t = sqrt_beta_t.unsqueeze(-1)

            init_code_new = sqrt_alpha_t * x_next + sqrt_beta_t * noise  # noise z34 to z35 
            init_code = init_code_new*interp_mask + init_code_ori*(1-interp_mask)
            init_code = init_code.detach()

        if save_interlora:
            if not ada:
                lora_path_k = os.path.join(save_lora_path,str(actual_step_idx))
                if not os.path.isdir(lora_path_k):
                    os.mkdir(lora_path_k)
                unet_lora_layers = unet_lora_state_dict(model.unet)
                LoraLoaderMixin.save_lora_weights(
                    save_directory=lora_path_k,
                    unet_lora_layers=unet_lora_layers,
                    text_encoder_lora_layers=None,
                )
                path = os.path.join(lora_path_k,f'z35_{ada_num}.pth')
                torch.save(init_code.detach(),path)
            else:
                path = os.path.join(lora_path_k,f'z35_{ada_num}.pth')
                torch.save(init_code.detach(),path)

    with open(os.path.join(save_lora_path,'training_metrics.csv'), mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Steps', 'ada_num', 'minD', 'PTs'])
        for epoch, ada, mind, pt in zip(steps, adanums, minDs, PTs):
            writer.writerow([epoch, ada, mind, pt]) 

    unet_lora_layers = unet_lora_state_dict(model.unet)
    LoraLoaderMixin.save_lora_weights(
        save_directory=save_lora_path,
        unet_lora_layers=unet_lora_layers,
        text_encoder_lora_layers=None,
    )
    path = os.path.join(save_lora_path,'z35.pth')
    torch.save(init_code.detach(),path)
   
    return init_code







