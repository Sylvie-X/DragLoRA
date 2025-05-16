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
# Codes modified by DragLoRA Authors to add dragback option

import os 
import numpy as np 
import pickle
import tqdm  
import argparse
from PIL import Image
import sys  
sys.path.insert(0, '../')  
from utils.lora_utils import train_lora  


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="setting arguments")
    parser.add_argument('--root_path',
        default='drag_bench_data',
        help='root of DragBench',
        )
    parser.add_argument('--img_path',
        default='',
        help='path to drag-once outputs, change it if running dragback',
        )
    parser.add_argument('--model_path',
        default='runwayml/stable-diffusion-v1-5',
        help='root of SD1.5',
        )
    args = parser.parse_args()
    
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

    img_dir = args.img_path
    if img_dir != '':
        root_dir = img_dir
        lora_dir = 'drag_back_bench_lora/' + img_dir 
    else:
        root_dir  = args.root_path
        lora_dir = 'drag_bench_lora' 

    if not os.path.isdir(lora_dir):  
        os.mkdir(lora_dir)  
        for cat in all_category:  
            os.mkdir(os.path.join(lora_dir, cat)) 

    for cat in all_category:
        file_dir = os.path.join(root_dir, cat)  

        for sample_name in os.listdir(file_dir):
            if sample_name == '.DS_Store':  
                continue
            sample_path = os.path.join(file_dir, sample_name)  

            if img_dir != '':  
                source_image = Image.open(os.path.join(sample_path, 'dragged_image.png'))
            else:  
                source_image = Image.open(os.path.join(sample_path, 'original_image.png'))
            source_image = np.array(source_image)  

            with open(os.path.join(args.root_path, cat, sample_name, 'meta_data.pkl'), 'rb') as f:
                meta_data = pickle.load(f)
            prompt = meta_data['prompt']

            save_lora_path = os.path.join(lora_dir, cat, sample_name)
            if not os.path.isdir(save_lora_path): 
                os.mkdir(save_lora_path) 

            train_lora(
                source_image,  
                prompt,  
                model_path=args.model_path,
                vae_path="default",  
                save_lora_path=save_lora_path, 
                lora_step=80, 
                lora_lr=0.0005, 
                lora_batch_size=4,  
                lora_rank=16, 
                progress=tqdm,  
                save_interval=-1
            )
