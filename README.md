<p align="center">
  <h1 align="center">DragLoRA: Online Optimization of LoRA Adapters for Drag-based Image Editing in Diffusion Model</h1>
   <h3 align="center">Accepted by ICML 2025</h3>
  </div>
  <p align="center">
    <strong>Siwei Xia</strong>
    &nbsp;&nbsp;
    <strong>Li Sun</strong>
    &nbsp;&nbsp;
    <strong>Tiantian Sun</strong>
    &nbsp;&nbsp;
    <strong>Qingli Li</strong>
    &nbsp;&nbsp;
    <br>
    <b>East China Normal University</b>
    <div align="center">
    <a href="https://arxiv.org/abs/2505.12427"><img alt='arXiv' src="https://img.shields.io/badge/arXiv-2505.12427-blue"></a>
    </div>
  </p>
  
  <div align="center">
    <img src="cmp.png", width="700">
  </div>
</p>

## Project Status
- [x] Release benchmark evaluation code
- [x] Release Gradio user interface
- [x] Update readme for detailed usage guide 
- [x] Release paper on arXiv

## Installation
To set up the environment, run:
```bash
conda env create -f environment.yaml
conda activate draglora
```

## User Interface
To use DragLoRA with your own images:

1. Launch the interface:
```bash
python drag_ui.py
```

2. Follow these steps:
   - Upload your image to the left-most box
   - (Optional) Add a descriptive prompt below the image
   - Click "Train RecLoRA" to optimize for identity preservation
   - (Optional) Draw a mask to specify editable regions
   - Set handle-target points in the middle box:
     1. Click to place a handle point
     2. Click to place its target point
     3. Repeat for additional point pairs as needed
   - Click "Run" to process the image

3. Output and Storage:
   - Results appear in the right-most box
   - Temporary files are saved in "lora_tmp" (overwritten for each new image)
   - Input images, prompts, masks, points, and outputs are saved with unique names (e.g. 2025-05-17-0105-05) in "lora_tmp"

## Benchmark Evaluation
To evaluate the algorithm using benchmark data:

1. Navigate to the evaluation directory:
```bash
cd drag_bench_evaluation
```

2. Download and extract [DragBench](https://github.com/Yujun-Shi/DragDiffusion/releases/download/v0.1.1/DragBench.zip) to "drag_bench_data"

3. Train reconstruction LoRA:
```bash
python run_lora_training.py
```
Results will be saved in "drag_bench_lora"

4. Run DragLoRA:
```bash
python run_dragbench_draglora.py
```
Results will be saved in "drag_results"

5. Evaluate performance:
   - Point matching accuracy (MD and m-MD):
   ```bash
   python run_eval_point_matching.py --eval_root drag_results
   ```
   - Image consistency (LPIPS, CLIP, MSE):
   ```bash
   python run_eval_similarity.py --eval_root drag_results
   ```
### Drag-Back pipeline
To simultaneously measure editability and consistency through two symmetric editing operations, first drag once following above instructions.
After that, drag twice with:
1. Train reconstruction LoRA:
```bash
python run_lora_training.py --img_path drag_results
```
Results will be saved in "drag_bench_lora_for_drag_results"

2. Run DragLoRA:
```bash
python run_dragbench_draglora.py \
--img_dir drag_results \
--lora_dir drag_bench_lora_for_drag_results \
--save_dir drag_back_results
```
Results will be saved in "drag_back_results"

3. Evaluate performance by compare the similarity between the drag-back images and original images:
```bash
   python run_eval_similarity.py --eval_root drag_back_results
```

## Citation
If you find our work useful, please cite our paper:
```bibtex
@inproceedings{xia2025draglora,
  title={DragLoRA: Online Optimization of LoRA Adapters for Drag-based Image Editing in Diffusion Model},
  author={Xia, Siwei and Sun, Li and Sun, Tiantian and Li, Qingli},
  booktitle={The International Conference on Machine Learning (ICML)},
  year={2025}
}
```

## Acknowledgements
This work builds upon [DragDiffusion](https://github.com/Yujun-Shi/DragDiffusion). We thank the authors and all contributors to the open-source diffusion models and libraries.
