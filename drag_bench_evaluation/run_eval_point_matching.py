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
# run evaluation of mean distance between the desired target points and the position of final handle points
# Codes modified by DragLoRA Authors to compute m-MD 

import argparse
import os
import pickle
import numpy as np
import PIL
from PIL import Image
from torchvision.transforms import PILToTensor
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import seed_everything
from progress.bar import Bar
from typing import Tuple,List
import math
from dift_sd import SDFeaturizer


def get_ellipse_coords(
    point: Tuple[int, int], radius: int = 5
) -> Tuple[int, int, int, int]:
    """
    Returns the coordinates of an ellipse centered at the given point.

    Args:
        point (Tuple[int, int]): The center point of the ellipse.
        radius (int): The radius of the ellipse.

    Returns:
        A tuple containing the coordinates of the ellipse in the format (x_min, y_min, x_max, y_max).
    """
    center = point
    return (
        center[0] - radius,
        center[1] - radius,
        center[0] + radius,
        center[1] + radius,
    )


def draw_handle_target_points(
        img: PIL.Image.Image,
        handle_points: List[Tuple[int, int]],
        target_points: List[Tuple[int, int]],
        radius: int = 5,
        color = "red"):
    """
    Draws handle and target points with arrow pointing towards the target point.

    Args:
        img (PIL.Image.Image): The image to draw on.
        handle_points (List[Tuple[int, int]]): A list of handle [x,y] points.
        target_points (List[Tuple[int, int]]): A list of target [x,y] points.
        radius (int): The radius of the handle and target points.
    """
    if not isinstance(img, PIL.Image.Image):
        img = PIL.Image.fromarray(img)

    draw = PIL.ImageDraw.Draw(img)
    for handle_point, target_point in zip(handle_points, target_points):
        handle_point = [handle_point[1], handle_point[0]]
        # Draw the handle point
        handle_coords = get_ellipse_coords(handle_point, radius)
        draw.ellipse(handle_coords, fill=color)

        if target_point is not None:
            target_point = [target_point[1], target_point[0]]
            # Draw the target point
            target_coords = get_ellipse_coords(target_point, radius)
            draw.ellipse(target_coords, fill="blue")

            arrow_head_length = radius*1.5

            # Compute the direction vector of the line
            dx = target_point[0] - handle_point[0]
            dy = target_point[1] - handle_point[1]
            angle = math.atan2(dy, dx)

            # Shorten the target point by the length of the arrowhead
            shortened_target_point = (
                target_point[0] - arrow_head_length * math.cos(angle),
                target_point[1] - arrow_head_length * math.sin(angle),
            )

            # Draw the arrow (main line)
            draw.line([tuple(handle_point), shortened_target_point], fill='white', width=int(0.8*radius))

            arrow_point1 = (
                target_point[0] - arrow_head_length * math.cos(angle - math.pi / 6),
                target_point[1] - arrow_head_length * math.sin(angle - math.pi / 6),
            )

            arrow_point2 = (
                target_point[0] - arrow_head_length * math.cos(angle + math.pi / 6),
                target_point[1] - arrow_head_length * math.sin(angle + math.pi / 6),
            )

            draw.polygon([tuple(target_point), arrow_point1, arrow_point2], fill='white')
    return img


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="setting arguments")
    parser.add_argument('--eval_root',
        action='append',
        help='root of dragging results for evaluation',
        required=True)
    parser.add_argument('--model_path',
        default='stabilityai/stable-diffusion-2-1',
        help='root of SD2.1',
        )
    parser.add_argument('--root_path',
        default='drag_bench_data',
        help='root of DragBench',
        )
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # using SD-2.1
    dift = SDFeaturizer(args.model_path)

    all_category = [
        'art_work',
        'land_scape',
        'building_city_view',
        'building_countryside_view',
        'animals',
        'human_head',
        'human_upper_body',
        'human_full_body',
        'interior_design',
        'other_objects',
    ]

    original_img_root = args.root_path

    for target_root in args.eval_root:
        seed_everything(42)

        all_dist = []
        all_dist_m = []
        for cat in all_category:
            for file_name in os.listdir(os.path.join(original_img_root, cat)):
                if file_name == '.DS_Store':
                    continue
                with open(os.path.join(original_img_root, cat, file_name, 'meta_data.pkl'), 'rb') as f:
                    meta_data = pickle.load(f)
                prompt = meta_data['prompt']
                points = meta_data['points']
                mask = torch.from_numpy(meta_data['mask'])
                mask_flat = mask.flatten()
                handle_points = []
                target_points = []
                for idx, point in enumerate(points):
                    cur_point = torch.tensor([point[1], point[0]])
                    if idx % 2 == 0:
                        handle_points.append(cur_point)
                    else:
                        target_points.append(cur_point)

                source_image_path = os.path.join(original_img_root, cat, file_name, 'original_image.png')
                
                dragged_image_path = os.path.join(target_root, cat, file_name, 'dragged_image.png')
                
                if not os.path.exists(dragged_image_path):
                    continue
                
                drawed_image_path = os.path.join(target_root, cat, file_name, 'drawed_image.png')
                
                source_image_PIL = Image.open(source_image_path)
                dragged_image_PIL = Image.open(dragged_image_path)
                dragged_image_PIL = dragged_image_PIL.resize(source_image_PIL.size,PIL.Image.BILINEAR)

                source_image_tensor = (PILToTensor()(source_image_PIL) / 255.0 - 0.5) * 2
                dragged_image_tensor = (PILToTensor()(dragged_image_PIL) / 255.0 - 0.5) * 2
                _, H, W = source_image_tensor.shape
                
                with torch.no_grad():
                    ft_source = dift.forward(source_image_tensor,
                        prompt=prompt,
                        t=261,
                        up_ft_index=1,
                        ensemble_size=8)
                    ft_source = F.interpolate(ft_source, (H, W), mode='bilinear')

                    ft_dragged = dift.forward(dragged_image_tensor,
                        prompt=prompt,
                        t=261,
                        up_ft_index=1,
                        ensemble_size=8)
                    ft_dragged = F.interpolate(ft_dragged, (H, W), mode='bilinear')

                now_points=[]
                cos = nn.CosineSimilarity(dim=1)

                now_points_m=[]
                x_coords = torch.arange(W)
                y_coords = torch.arange(H)
                x_coords = x_coords.repeat(H)
                y_coords = y_coords.repeat_interleave(W)
                candidate_coords = torch.stack((y_coords, x_coords), dim=1)
                candidate_coords = candidate_coords[mask_flat==1]
                ft_dragged_mask = ft_dragged[:,:,candidate_coords[:,0],candidate_coords[:,1]]

                for pt_idx in range(len(handle_points)):
                    hp = handle_points[pt_idx]
                    tp = target_points[pt_idx]

                    num_channel = ft_source.size(1)
                    src_vec = ft_source[0, :, hp[0], hp[1]].view(1, num_channel, 1, 1)

                    cos_map = cos(src_vec, ft_dragged).cpu().numpy()[0]  # H, W
                    max_rc = np.unravel_index(cos_map.argmax(), cos_map.shape) # the matched row,col
                    cos_map = cos(src_vec.squeeze(-1),ft_dragged_mask)[0]
                    max_rc_m = candidate_coords[cos_map.argmax()]
                    now_points.append((max_rc_m[0].item(),max_rc_m[1].item()))

                    dist = (tp - torch.tensor(max_rc)).float().norm()
                    dist_m = (tp - torch.tensor(max_rc_m)).float().norm()
                    all_dist.append(dist)
                    all_dist_m.append(dist_m)

                drawd_image = draw_handle_target_points(dragged_image_PIL,now_points,target_points)
                drawd_image.save(drawed_image_path)
        print(f'results of {target_root}:')
        print(' m-MD: ', torch.tensor(all_dist_m).mean().item())
        print('MD: ', torch.tensor(all_dist).mean().item())