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
  </p>
</p>

## Project Status
- [x] Release benchmark evaluation code
- [x] Release Gradio user interface
- [x] Update readme for detailed usage guide 
- [ ] Release paper on arXiv

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

4. Run DragLoRA evaluation:
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
